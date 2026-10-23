//! **GPT3 (Global Pressure and Temperature 3)** — the empirical/blind troposphere model that is the
//! offline companion to `VMF3`. From `(latitude, longitude, day-of-year, height)` it returns the
//! `VMF3` mapping `a`-coefficients `a_h`/`a_w` plus surface pressure, temperature, water-vapour
//! pressure, weighted mean temperature `Tm`, the water-vapour decrease factor `lambda`, and the four
//! azimuthal tropospheric gradient coefficients. It lets [`crate::vmf3_mapping`] run with no external
//! grid files: a 5°×5° climatology of seasonal Fourier coefficients is embedded in the binary.
//!
//! Ported from Orekit's `GlobalPressureTemperature3` / `AbstractGlobalPressureTemperature`
//! (`org.orekit.models.earth.weather`), which in turn implements
//! "Landskron, D. & Böhm, J. J Geod (2018) 92: 349, VMF3/GPT3: refined discrete and empirical
//! troposphere mapping functions, <https://doi.org/10.1007/s00190-017-1066-2>".
//!
//! Each grid node carries, for twelve parameters, five seasonal coefficients `(a0, A1, B1, A2, B2)`
//! reconstructed as `a0 + A1·cos(θ) + B1·sin(θ) + A2·cos(2θ) + B2·sin(2θ)` with
//! `θ = day_of_year/365.25·2π`. The four cell corners are each evaluated and pressure-reduced to the
//! requested height (per Orekit's `EvaluatedGridEntry`) **before** a bilinear interpolation in
//! latitude/longitude; `a_h`/`a_w` and the gradients are interpolated as-is (no height reduction).

// Grid-index/key arithmetic uses f64<->i64 casts on values that are always small and exact; the grid
// geometry uses standard short names (lat/lon, si/wi, t/u) and the oracle literals stay byte-exact.
// The embedded grid is a trusted compile-time constant, so its parse is infallible by construction and
// panics (via `expect`) only if the vendored asset is corrupted at build time — the public API has no
// error channel by design (an empirical blind model must "just work" offline).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::expect_used,
    clippy::module_name_repetitions
)]

use core::f64::consts::PI;
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// The vendored TU Wien 5°×5° GPT3 grid, byte-identical to Orekit's `gpt3_5.grd`.
const GPT3_GRID_TEXT: &str = include_str!("../assets/gpt3_5.grd");

/// Conversion factor from degrees to milli-arcseconds (Orekit `GridEntry.DEG_TO_MAS`). Grid nodes are
/// located in this integer key space so that a query coordinate is snapped to the exact node before
/// cell selection, immune to degree/radian round-trip error.
const DEG_TO_MAS: f64 = 3_600_000.0;

/// Standard gravity `g0` [m/s²] (Orekit `Constants.G0_STANDARD_GRAVITY`).
const G0_STANDARD_GRAVITY: f64 = 9.80665;
/// Molar mass of dry air [kg/mol] (Orekit `EvaluatedGridEntry.DMTR`).
const DRY_AIR_MOLAR_MASS: f64 = 28.965e-3;
/// Universal gas constant [J/K/mol] (Orekit `EvaluatedGridEntry.RG`).
const UNIVERSAL_GAS_CONSTANT: f64 = 8.3143;
/// Virtual-temperature coefficient in `Tv = T·(1 + 0.6077·qv)` (Orekit `EvaluatedGridEntry`).
const VIRTUAL_TEMPERATURE_COEFF: f64 = 0.6077;
/// hectopascal → pascal (Orekit `TroposphericModelUtils.HECTO_PASCAL.toSI`).
const HECTO_PASCAL_TO_PASCAL: f64 = 100.0;

/// Panic message when the embedded grid header lacks a column we require.
const MISSING_COLUMN: &str = "gpt3 grid: missing expected column in header";

/// GPT3 output at a site and date, in SI units.
///
/// `pressure` and `water_vapor_pressure` are in pascals, `temperature` and `tm` in kelvin, `lambda`
/// is dimensionless, and the gradient coefficients `gnh`/`geh`/`gnw`/`gew` are dimensionless
/// (already scaled by Orekit's `1e-5`). `ah`/`aw` are the `VMF3` mapping `a`-coefficients
/// (already scaled by Orekit's `1e-3`), ready to feed [`crate::vmf3_mapping`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gpt3Result {
    /// Hydrostatic mapping coefficient `a_h`.
    pub ah: f64,
    /// Wet mapping coefficient `a_w`.
    pub aw: f64,
    /// Total surface pressure \[Pa\].
    pub pressure: f64,
    /// Surface temperature \[K\].
    pub temperature: f64,
    /// Water-vapour partial pressure \[Pa\].
    pub water_vapor_pressure: f64,
    /// Weighted mean temperature `Tm` \[K\].
    pub tm: f64,
    /// Water-vapour decrease factor `lambda`.
    pub lambda: f64,
    /// Hydrostatic north gradient coefficient `Gn_h`.
    pub gnh: f64,
    /// Hydrostatic east gradient coefficient `Ge_h`.
    pub geh: f64,
    /// Wet north gradient coefficient `Gn_w`.
    pub gnw: f64,
    /// Wet east gradient coefficient `Ge_w`.
    pub gew: f64,
}

/// The empirical GPT3 troposphere model.
///
/// Returns the [`Gpt3Result`] weather parameters, `VMF3` mapping coefficients and gradients at the
/// given geodetic `latitude`/`longitude` (radians), `day_of_year` (Orekit's continuous seasonal
/// argument: physical days since 31 December of the previous year, 00:00 UTC), and `height` (metres
/// above the ellipsoid). Uses only the embedded 5°×5° climatology — no external files.
#[must_use]
pub fn gpt3(latitude: f64, longitude: f64, day_of_year: f64, height: f64) -> Gpt3Result {
    grid().evaluate(latitude, longitude, day_of_year, height)
}

/// The five seasonal Fourier coefficients of one parameter at one grid node.
#[derive(Clone, Copy)]
struct SeasonalModel {
    a0: f64,
    a1: f64,
    b1: f64,
    a2: f64,
    b2: f64,
}

impl SeasonalModel {
    #[inline]
    fn evaluate(self, h: Harmonics) -> f64 {
        self.a0 + self.a1 * h.cos1 + self.b1 * h.sin1 + self.a2 * h.cos2 + self.b2 * h.sin2
    }
}

/// Sines and cosines of the annual and semi-annual harmonics for a given day of year, shared across
/// all corners and all parameters of one evaluation.
#[derive(Clone, Copy)]
struct Harmonics {
    cos1: f64,
    sin1: f64,
    cos2: f64,
    sin2: f64,
}

impl Harmonics {
    fn from_day_of_year(day_of_year: f64) -> Self {
        // Orekit: coef = (deltaRef / JULIAN_YEAR) · 2π, and day_of_year is deltaRef expressed in days,
        // so coef = (day_of_year / 365.25) · 2π.
        let coef = (day_of_year / 365.25) * 2.0 * PI;
        Self {
            cos1: coef.cos(),
            sin1: coef.sin(),
            cos2: (2.0 * coef).cos(),
            sin2: (2.0 * coef).sin(),
        }
    }
}

/// One grid node: geodetic position, orography, and the twelve seasonal models GPT3 carries. Unlike
/// Orekit's generic `SeasonalModelType`-keyed map (which also serves GPT2/GPT2w), the GPT3 column
/// layout is fixed, so a flat struct makes "every node has all twelve models" unrepresentable-as-bug.
#[derive(Clone)]
struct GridEntry {
    latitude: f64,
    longitude: f64,
    undulation: f64,
    hs: f64,
    pressure: SeasonalModel,
    temperature: SeasonalModel,
    qv: SeasonalModel,
    dt: SeasonalModel,
    ah: SeasonalModel,
    aw: SeasonalModel,
    lambda: SeasonalModel,
    tm: SeasonalModel,
    gn_h: SeasonalModel,
    ge_h: SeasonalModel,
    gn_w: SeasonalModel,
    ge_w: SeasonalModel,
}

/// A single grid node evaluated at a date and height-reduced to the requested altitude. Pressure,
/// temperature and water-vapour pressure are already reduced (Orekit `EvaluatedGridEntry`); the
/// remaining fields are the plain seasonal reconstructions (no height dependence).
#[derive(Clone, Copy)]
struct CornerEval {
    ah: f64,
    aw: f64,
    temperature: f64,
    pressure: f64,
    water_vapor: f64,
    tm: f64,
    lambda: f64,
    gn_h: f64,
    ge_h: f64,
    gn_w: f64,
    ge_w: f64,
}

impl GridEntry {
    /// Evaluate every model at `h` and reduce pressure/temperature/water-vapour to `height`, mirroring
    /// Orekit `EvaluatedGridEntry`. `pressure`/`water_vapor` are returned in hectopascals.
    fn evaluate(&self, h: Harmonics, height: f64) -> CornerEval {
        let corrected_height = height - self.undulation - self.hs;

        let t0 = self.temperature.evaluate(h);
        let qv = self.qv.evaluate(h) * 0.001;
        let tv = t0 * (1.0 + VIRTUAL_TEMPERATURE_COEFF * qv);
        let factor = -corrected_height * G0_STANDARD_GRAVITY * DRY_AIR_MOLAR_MASS
            / (UNIVERSAL_GAS_CONSTANT * tv);

        let dtdh = self.dt.evaluate(h) * 0.001;
        let temperature = t0 + dtdh * corrected_height;

        let p0 = self.pressure.evaluate(h) * 0.01;
        let pressure = p0 * factor.exp();

        let e0 = qv * p0 / (0.622 + 0.378 * qv);
        let lambda = self.lambda.evaluate(h);
        let water_vapor = e0 * (factor * (1.0 + lambda)).exp();

        CornerEval {
            ah: self.ah.evaluate(h),
            aw: self.aw.evaluate(h),
            temperature,
            pressure,
            water_vapor,
            tm: self.tm.evaluate(h),
            lambda,
            gn_h: self.gn_h.evaluate(h),
            ge_h: self.ge_h.evaluate(h),
            gn_w: self.gn_w.evaluate(h),
            ge_w: self.ge_w.evaluate(h),
        }
    }
}

/// The parsed grid: ascending latitude/longitude key axes and a `[nLat][nLon]` node array whose last
/// longitude column wraps the westmost column 360° east (Orekit `Grid`).
struct Gpt3Grid {
    lat_keys: Vec<i64>,
    lon_keys: Vec<i64>,
    lon0: f64,
    entries: Vec<Vec<GridEntry>>,
}

impl Gpt3Grid {
    /// Index of the southern row of the surrounding cell (Orekit `Grid.getSouthIndex`): the query
    /// latitude is snapped to a milli-arcsecond key, then located among the ascending keys and clipped
    /// so the northern neighbour stays in range.
    fn south_index(&self, latitude: f64) -> usize {
        let lat_key = (latitude.to_degrees() * DEG_TO_MAS).round() as i64;
        let count = self.lat_keys.partition_point(|&k| k <= lat_key);
        count.saturating_sub(1).min(self.lat_keys.len() - 2)
    }

    /// Index of the western column of the surrounding cell (Orekit `Grid.getWestIndex`). No clipping is
    /// needed against the eastern edge because the grid carries an extra wrap column, but we clamp
    /// defensively so the eastern neighbour index is always valid.
    fn west_index(&self, longitude: f64) -> usize {
        let lon_key = (longitude.to_degrees() * DEG_TO_MAS).round() as i64;
        let count = self.lon_keys.partition_point(|&k| k <= lon_key);
        count.saturating_sub(1).min(self.entries[0].len() - 2)
    }

    /// Normalize longitude into `[lon0, lon0 + 2π)`, matching Orekit's
    /// `MathUtils.normalizeAngle(longitude, lon0 + π)` (`lon0` is the westmost column of the
    /// shipped `gpt3_5.grd`, −177.5°).
    fn normalize_longitude(&self, longitude: f64) -> f64 {
        let two_pi = 2.0 * PI;
        longitude - two_pi * ((longitude - self.lon0) / two_pi).floor()
    }

    fn evaluate(&self, latitude: f64, longitude: f64, day_of_year: f64, height: f64) -> Gpt3Result {
        let h = Harmonics::from_day_of_year(day_of_year);
        let norm_lon = self.normalize_longitude(longitude);
        let si = self.south_index(latitude);
        let wi = self.west_index(norm_lon);

        let sw_entry = &self.entries[si][wi];
        let se_entry = &self.entries[si][wi + 1];
        let nw_entry = &self.entries[si + 1][wi];
        let ne_entry = &self.entries[si + 1][wi + 1];

        let sw = sw_entry.evaluate(h, height);
        let se = se_entry.evaluate(h, height);
        let nw = nw_entry.evaluate(h, height);
        let ne = ne_entry.evaluate(h, height);

        // Bilinear weights over the cell (Orekit CellInterpolator + BilinearInterpolatingFunction):
        // t is the longitude fraction, u the latitude fraction, using the corners' own coordinates so
        // the wrap column's +2π longitude is handled naturally.
        let t = (norm_lon - sw_entry.longitude) / (se_entry.longitude - sw_entry.longitude);
        let u = (latitude - sw_entry.latitude) / (nw_entry.latitude - sw_entry.latitude);
        let bilinear = |f_sw: f64, f_se: f64, f_nw: f64, f_ne: f64| {
            (1.0 - t) * (1.0 - u) * f_sw
                + (1.0 - t) * u * f_nw
                + t * (1.0 - u) * f_se
                + t * u * f_ne
        };

        Gpt3Result {
            ah: bilinear(sw.ah, se.ah, nw.ah, ne.ah) * 0.001,
            aw: bilinear(sw.aw, se.aw, nw.aw, ne.aw) * 0.001,
            pressure: bilinear(sw.pressure, se.pressure, nw.pressure, ne.pressure)
                * HECTO_PASCAL_TO_PASCAL,
            temperature: bilinear(
                sw.temperature,
                se.temperature,
                nw.temperature,
                ne.temperature,
            ),
            water_vapor_pressure: bilinear(
                sw.water_vapor,
                se.water_vapor,
                nw.water_vapor,
                ne.water_vapor,
            ) * HECTO_PASCAL_TO_PASCAL,
            tm: bilinear(sw.tm, se.tm, nw.tm, ne.tm),
            lambda: bilinear(sw.lambda, se.lambda, nw.lambda, ne.lambda),
            gnh: bilinear(sw.gn_h, se.gn_h, nw.gn_h, ne.gn_h) * 0.00001,
            geh: bilinear(sw.ge_h, se.ge_h, nw.ge_h, ne.ge_h) * 0.00001,
            gnw: bilinear(sw.gn_w, se.gn_w, nw.gn_w, ne.gn_w) * 0.00001,
            gew: bilinear(sw.ge_w, se.ge_w, nw.ge_w, ne.ge_w) * 0.00001,
        }
    }
}

/// Column indices of the fields we consume, discovered from the header row (Orekit `GptNParser`
/// locates columns by label rather than assuming a fixed offset). Each seasonal block occupies its
/// index plus the following four columns `A1 B1 A2 B2`.
struct ColumnLayout {
    lat: usize,
    lon: usize,
    undu: usize,
    hs: usize,
    pressure: usize,
    temperature: usize,
    qv: usize,
    dt: usize,
    ah: usize,
    aw: usize,
    lambda: usize,
    tm: usize,
    gn_h: usize,
    ge_h: usize,
    gn_w: usize,
    ge_w: usize,
}

fn parse_f64(s: &str) -> f64 {
    s.parse::<f64>()
        .expect("gpt3 grid: unparsable number in embedded asset")
}

fn read_model(fields: &[&str], first: usize) -> SeasonalModel {
    SeasonalModel {
        a0: parse_f64(fields[first]),
        a1: parse_f64(fields[first + 1]),
        b1: parse_f64(fields[first + 2]),
        a2: parse_f64(fields[first + 3]),
        b2: parse_f64(fields[first + 4]),
    }
}

fn parse_header(header: &str) -> ColumnLayout {
    let stripped = header.trim().strip_prefix('%').unwrap_or(header);
    let mut lat = None;
    let mut lon = None;
    let mut undu = None;
    let mut hs = None;
    let mut pressure = None;
    let mut temperature = None;
    let mut qv = None;
    let mut dt = None;
    let mut ah = None;
    let mut aw = None;
    let mut lambda = None;
    let mut tm = None;
    let mut gn_h = None;
    let mut ge_h = None;
    let mut gn_w = None;
    let mut ge_w = None;
    for (i, token) in stripped.split_whitespace().enumerate() {
        match token {
            "lat" => lat = Some(i),
            "lon" => lon = Some(i),
            "undu" => undu = Some(i),
            "Hs" => hs = Some(i),
            "p:a0" => pressure = Some(i),
            "T:a0" => temperature = Some(i),
            "Q:a0" => qv = Some(i),
            "dT:a0" => dt = Some(i),
            "a_h:a0" => ah = Some(i),
            "a_w:a0" => aw = Some(i),
            "lambda:a0" => lambda = Some(i),
            "Tm:a0" => tm = Some(i),
            "Gn_h:a0" => gn_h = Some(i),
            "Ge_h:a0" => ge_h = Some(i),
            "Gn_w:a0" => gn_w = Some(i),
            "Ge_w:a0" => ge_w = Some(i),
            _ => {}
        }
    }
    ColumnLayout {
        lat: lat.expect(MISSING_COLUMN),
        lon: lon.expect(MISSING_COLUMN),
        undu: undu.expect(MISSING_COLUMN),
        hs: hs.expect(MISSING_COLUMN),
        pressure: pressure.expect(MISSING_COLUMN),
        temperature: temperature.expect(MISSING_COLUMN),
        qv: qv.expect(MISSING_COLUMN),
        dt: dt.expect(MISSING_COLUMN),
        ah: ah.expect(MISSING_COLUMN),
        aw: aw.expect(MISSING_COLUMN),
        lambda: lambda.expect(MISSING_COLUMN),
        tm: tm.expect(MISSING_COLUMN),
        gn_h: gn_h.expect(MISSING_COLUMN),
        ge_h: ge_h.expect(MISSING_COLUMN),
        gn_w: gn_w.expect(MISSING_COLUMN),
        ge_w: ge_w.expect(MISSING_COLUMN),
    }
}

fn parse_grid(text: &str) -> Gpt3Grid {
    let mut lines = text.lines();
    let header = lines.next().expect("gpt3 grid: empty embedded asset");
    let cols = parse_header(header);

    let mut lat_set: BTreeSet<i64> = BTreeSet::new();
    let mut lon_set: BTreeSet<i64> = BTreeSet::new();
    let mut raw: Vec<(i64, i64, GridEntry)> = Vec::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let lat_deg = parse_f64(f[cols.lat]);
        let lon_deg = parse_f64(f[cols.lon]);
        let lat_key = (lat_deg * DEG_TO_MAS).round() as i64;
        let lon_key = (lon_deg * DEG_TO_MAS).round() as i64;
        let entry = GridEntry {
            latitude: lat_deg.to_radians(),
            longitude: lon_deg.to_radians(),
            undulation: parse_f64(f[cols.undu]),
            hs: parse_f64(f[cols.hs]),
            pressure: read_model(&f, cols.pressure),
            temperature: read_model(&f, cols.temperature),
            qv: read_model(&f, cols.qv),
            dt: read_model(&f, cols.dt),
            ah: read_model(&f, cols.ah),
            aw: read_model(&f, cols.aw),
            lambda: read_model(&f, cols.lambda),
            tm: read_model(&f, cols.tm),
            gn_h: read_model(&f, cols.gn_h),
            ge_h: read_model(&f, cols.ge_h),
            gn_w: read_model(&f, cols.gn_w),
            ge_w: read_model(&f, cols.ge_w),
        };
        lat_set.insert(lat_key);
        lon_set.insert(lon_key);
        raw.push((lat_key, lon_key, entry));
    }

    let lat_keys: Vec<i64> = lat_set.into_iter().collect();
    let lon_keys: Vec<i64> = lon_set.into_iter().collect();
    let n_lat = lat_keys.len();
    let n_real = lon_keys.len();
    let n_lon = n_real + 1;

    let mut entries: Vec<Vec<Option<GridEntry>>> = vec![vec![None; n_lon]; n_lat];
    for (lat_key, lon_key, entry) in raw {
        let i = lat_keys.partition_point(|&k| k < lat_key);
        let j = lon_keys.partition_point(|&k| k < lon_key);
        entries[i][j] = Some(entry);
    }

    // Wrap the grid around the Earth: the extra eastern column is the westmost node shifted +360°.
    for row in &mut entries {
        let mut wrapped = row[0]
            .clone()
            .expect("gpt3 grid: missing westmost column while wrapping");
        wrapped.longitude += 2.0 * PI;
        row[n_real] = Some(wrapped);
    }

    let entries: Vec<Vec<GridEntry>> = entries
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|e| e.expect("gpt3 grid: incomplete grid, a node is missing"))
                .collect()
        })
        .collect();

    let lon0 = entries[0][0].longitude;
    Gpt3Grid {
        lat_keys,
        lon_keys,
        lon0,
        entries,
    }
}

fn grid() -> &'static Gpt3Grid {
    static GRID: OnceLock<Gpt3Grid> = OnceLock::new();
    GRID.get_or_init(|| parse_grid(GPT3_GRID_TEXT))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Orekit builds the seasonal argument from `deltaRef = date.durationFrom(31 December of the prior
    // year, 00:00:00 UTC)`, a TAI-continuous (leap-second-aware) duration, then `coef =
    // deltaRef/JULIAN_YEAR·2π`. The `day_of_year` we feed `gpt3` is therefore `deltaRef` expressed in
    // days; `durationFrom` counts physical seconds, so a leap second inside the interval adds one.
    //
    // Case 1: createMJDDate(56140, JULIAN_DAY-1, UTC) = 2012-08-01T23:59:59 UTC. From 2011-12-31T00:00
    //   that is 214 days + 86399 s, plus the 2012-06-30 leap second (+1 s) = 18_576_000 s = 215.0 days.
    const DOY_CASE1: f64 = 18_576_000.0 / 86_400.0;
    // Case 2: createMJDDate(58940, 0, UTC) = 2020-04-01T00:00:00 UTC. From 2019-12-31T00:00 that is
    //   92 days with no intervening leap second = 7_948_800 s = 92.0 days.
    const DOY_CASE2: f64 = 7_948_800.0 / 86_400.0;

    const TOL: f64 = 1.0e-9;

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < TOL
    }

    #[test]
    fn day_of_year_constants_are_the_expected_whole_days() {
        assert!(close(DOY_CASE1, 215.0));
        assert!(close(DOY_CASE2, 92.0));
    }

    #[test]
    fn matches_orekit_matlab_ref_case1() {
        let r = gpt3(
            48.20_f64.to_radians(),
            16.37_f64.to_radians(),
            DOY_CASE1,
            156.0,
        );
        assert!(close(r.ah, 0.001258154), "ah = {}", r.ah);
        assert!(close(r.aw, 0.000568212), "aw = {}", r.aw);
        assert!(
            close(r.temperature, 273.15 + 22.060239452),
            "temperature = {}",
            r.temperature
        );
        assert!(
            close(r.pressure / HECTO_PASCAL_TO_PASCAL, 1002.577297099),
            "pressure hPa = {}",
            r.pressure / HECTO_PASCAL_TO_PASCAL
        );
        assert!(
            close(
                r.water_vapor_pressure / HECTO_PASCAL_TO_PASCAL,
                16.676583415
            ),
            "water vapor hPa = {}",
            r.water_vapor_pressure / HECTO_PASCAL_TO_PASCAL
        );
        assert!(close(r.tm, 281.064766962), "tm = {}", r.tm);
        assert!(close(r.lambda, 2.697732327), "lambda = {}", r.lambda);
        assert!(close(r.gnh, -0.000177344), "gnh = {}", r.gnh);
        assert!(close(r.geh, -0.000048308), "geh = {}", r.geh);
        assert!(close(r.gnw, -0.000085318), "gnw = {}", r.gnw);
        assert!(close(r.gew, -0.000035866), "gew = {}", r.gew);
    }

    #[test]
    fn matches_orekit_matlab_ref_case2() {
        let r = gpt3(
            77.5_f64.to_radians(),
            137.5_f64.to_radians(),
            DOY_CASE2,
            0.0,
        );
        assert!(close(r.ah, 0.001171577), "ah = {}", r.ah);
        assert!(close(r.aw, 0.000574057), "aw = {}", r.aw);
        assert!(
            close(r.temperature, 273.15 - 18.589601195),
            "temperature = {}",
            r.temperature
        );
        assert!(
            close(r.pressure / HECTO_PASCAL_TO_PASCAL, 1019.900868185),
            "pressure hPa = {}",
            r.pressure / HECTO_PASCAL_TO_PASCAL
        );
        assert!(
            close(r.water_vapor_pressure / HECTO_PASCAL_TO_PASCAL, 1.251369020),
            "water vapor hPa = {}",
            r.water_vapor_pressure / HECTO_PASCAL_TO_PASCAL
        );
        assert!(close(r.tm, 250.211896532), "tm = {}", r.tm);
        assert!(close(r.lambda, 1.813332081), "lambda = {}", r.lambda);
        assert!(close(r.gnh, 0.000004818), "gnh = {}", r.gnh);
        assert!(close(r.geh, 0.000154168), "geh = {}", r.geh);
        assert!(close(r.gnw, 0.000001018), "gnw = {}", r.gnw);
        assert!(close(r.gew, 0.000003392), "gew = {}", r.gew);
    }

    /// Every interpolated field of one evaluation, so the identities below can be stated on all of
    /// them at once rather than on whichever one happens to move.
    fn fields(r: &Gpt3Result) -> [f64; 11] {
        [
            r.ah,
            r.aw,
            r.pressure,
            r.temperature,
            r.water_vapor_pressure,
            r.tm,
            r.lambda,
            r.gnh,
            r.geh,
            r.gnw,
            r.gew,
        ]
    }

    fn assert_close(got: &Gpt3Result, want: &Gpt3Result, rel: f64, what: &str) {
        for (k, (g, w)) in fields(got).into_iter().zip(fields(want)).enumerate() {
            assert!(
                (g - w).abs() <= rel * w.abs().max(1e-12),
                "{what}: field {k} is {g}, expected {w}",
            );
        }
    }

    /// Beyond the northmost row the cell index is clamped to the last *full* cell and the bilinear
    /// weight extrapolates, rather than the index running off the end of the array.
    ///
    /// Both reference cases sit at 48.2°N and 77.5°N, well inside the grid, so the clamp at
    /// `.min(lat_keys.len() - 2)` never bound and could be moved without any test noticing — even
    /// though moving it upward indexes `entries[36]` on a 36-row grid. Asserting the linear
    /// continuation pins which cell is chosen, not merely that the call survives.
    #[test]
    fn north_of_the_last_grid_row_extrapolates_from_the_last_full_cell() {
        let lon = 16.37_f64.to_radians();
        let at = |lat_deg: f64| gpt3(lat_deg.to_radians(), lon, DOY_CASE1, 156.0);
        let (south, north, beyond) = (at(82.5), at(87.5), at(89.0));
        // u = (89 − 82.5)/5 = 1.3 in the same cell, so every field is the same linear continuation.
        for (k, ((s, n), b)) in fields(&south)
            .into_iter()
            .zip(fields(&north))
            .zip(fields(&beyond))
            .enumerate()
        {
            let want = s + 1.3 * (n - s);
            assert!(
                (b - want).abs() <= 1e-12 * want.abs().max(1e-12),
                "field {k}: {b} at 89°, linear continuation says {want}",
            );
        }
    }

    /// A longitude outside the grid's span is folded back into it, and the fold is a subtraction.
    ///
    /// Both reference cases pass longitudes already inside `[−177.5°, 182.5°)`, where the floor is
    /// zero and `longitude − 0` and `longitude + 0` are the same number — so the reduction itself
    /// went unchecked. 200° is one full turn east of −160°, and the two must give the same weather.
    #[test]
    fn a_longitude_outside_the_grid_span_folds_back_by_a_full_turn() {
        let lat = 48.2_f64.to_radians();
        let at = |lon_deg: f64| gpt3(lat, lon_deg.to_radians(), DOY_CASE1, 156.0);
        assert_close(&at(200.0), &at(-160.0), 1e-12, "200° vs −160°");
        assert_close(&at(-200.0), &at(160.0), 1e-12, "−200° vs 160°");
    }

    /// The wrap column carries the westmost node's weather at a longitude exactly one turn east.
    ///
    /// It is the only column no grid line supplies, and it is reached only by a query in the seam
    /// cell `(177.5°, 182.5°)` — which nothing tested. There the longitude fraction is
    /// `(180 − 177.5)/(182.5 − 177.5) = 1/2` exactly, so the seam value must be the mean of the two
    /// nodes it joins. Any other shift of the wrap column's longitude moves that denominator and
    /// with it the fraction.
    #[test]
    fn the_seam_between_the_last_column_and_the_wrap_is_the_mean_of_its_two_nodes() {
        let lat = 48.2_f64.to_radians();
        let at = |lon_deg: f64| gpt3(lat, lon_deg.to_radians(), DOY_CASE1, 156.0);
        let (east, west, seam) = (at(177.5), at(-177.5), at(180.0));
        for (k, ((e, w), s)) in fields(&east)
            .into_iter()
            .zip(fields(&west))
            .zip(fields(&seam))
            .enumerate()
        {
            let want = f64::midpoint(e, w);
            assert!(
                (s - want).abs() <= 1e-12 * want.abs().max(1e-12),
                "field {k}: {s} at the seam, the mean of its nodes is {want}",
            );
        }
    }

    #[test]
    fn feeds_the_vmf3_mapping_function() {
        // The offline chain: GPT3 blind a-coefficients -> VMF3 slant mapping factor.
        let latitude = 48.20_f64.to_radians();
        let longitude = 16.37_f64.to_radians();
        let r = gpt3(latitude, longitude, DOY_CASE1, 156.0);
        let (mh, mw) = crate::vmf3_mapping(
            r.ah,
            r.aw,
            latitude,
            longitude,
            156.0,
            DOY_CASE1,
            20.0_f64.to_radians(),
        );
        assert!(
            mh.is_finite() && mw.is_finite() && mh > 1.0 && mw > 1.0,
            "mapping factors = ({mh}, {mw})"
        );
    }
}
