//! Ocean tide loading: the station moving because the water around it does.
//!
//! The tide raises and lowers a column of water some hundreds of kilometres across, and the crust
//! under it flexes. The result is a station displacement of several centimetres at a coastal site —
//! Reykjavik reaches 7 cm peak to peak vertically — at exactly the diurnal and semi-diurnal
//! frequencies the orbit is most sensitive to. It is the largest of the station-displacement terms
//! after the solid Earth tide, and unlike that one it cannot be computed from first principles: it
//! depends on the ocean model and on the Green's function of the crust beneath the particular
//! station, so it arrives as a per-station table.
//!
//! # What the table is, and why it is only eleven lines
//!
//! The loading provider ([`blq`](crate::blq)) publishes, per station, the amplitude and phase of
//! the response at eleven tidal constituents — M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Mm, Ssa — in
//! three directions. Eleven is not the number of tides; it is the number at which the response is
//! *specified*. The rest are recovered by admittance.
//!
//! The **admittance** `Z(f)` is the ratio of the station's response to the equilibrium tide that
//! drove it. It is a property of the ocean basin and the crust, not of any one constituent, and it
//! varies smoothly with frequency within a tidal species — smoothly enough that a cubic spline
//! through the specified points recovers the unspecified ones. So: divide each of the eleven by its
//! Cartwright-Edden equilibrium amplitude to get the admittance there, spline the real and
//! imaginary parts separately within each species (long-period, diurnal, semi-diurnal), evaluate at
//! all 342 constituents of the harmonic development, and multiply back by each one's equilibrium
//! amplitude. The reference program quotes about 0.1% for the result.
//!
//! Ported from the IERS Conventions Centre's `HARDISP` and its `ADMINT`, `TDFRPH`, `SPLINE` and
//! `EVAL`, renamed and re-expressed in Rust as the IERS software licence requires. Three
//! differences, none of which changes a number by more than the last digit the reference prints:
//!
//! - **No recursion over an equispaced series.** `HARDISP` computes one epoch and then steps the
//!   phases forward by a fixed sample interval, resetting every 600 points to bound the drift.
//!   Here each call evaluates its own epoch, so there is nothing to drift and no block structure in
//!   the API. The reference's approach is an optimisation for printing a time series; this crate's
//!   callers ask for scattered normal-point epochs.
//! - **`f64` throughout.** `SPLINE` and `EVAL` declare their working variables `REAL`, so the
//!   reference interpolates the admittance in single precision. At the six decimals its output
//!   format carries, the two agree exactly — a `-fdefault-real-8` build of the reference reproduces
//!   its published test values to the last printed digit — so this is a free improvement rather
//!   than a disagreement.
//! - **The epoch is the caller's.** `HARDISP` takes a UTC calendar date and carries its own
//!   leap-second table to reach TT. Here the split arrives already made, as a UT day fraction and a
//!   TT Julian century, matching [`solid_earth_tide_displacement`](crate::solid_earth_tide_displacement)
//!   and leaving time scales to the caller.
//!
//! The published series pin the transcription at one epoch. `derivations/ocean_loading.mac` checks
//! the parts behind it: the Delaunay polynomials against SOFA's, the rates as their derivatives,
//! Darwin's relations and the textbook periods of the eleven specified constituents, and the end
//! condition of `SPLINE`.

// The Delaunay arguments are l, l', F, D and Omega and the spline's variables x, u and n, in
// every source that defines either. Renaming them to pass a lint would make the equations
// unreadable against the references they are transcribed from.
#![allow(clippy::many_single_char_names)]

use crate::displacement::Site;
use crate::loading_tables::CARTWRIGHT_EDDEN;

/// Doodson multipliers of the eleven constituents a loading table specifies, in the order the
/// provider writes them: M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Mm, Ssa.
#[rustfmt::skip]
const SPECIFIED: [[i8; 6]; 11] = [
    [2,  0,  0, 0, 0, 0], // M2
    [2,  2, -2, 0, 0, 0], // S2
    [2, -1,  0, 1, 0, 0], // N2
    [2,  2,  0, 0, 0, 0], // K2
    [1,  1,  0, 0, 0, 0], // K1
    [1, -1,  0, 0, 0, 0], // O1
    [1,  1, -2, 0, 0, 0], // P1
    [1, -2,  0, 1, 0, 0], // Q1
    [0,  2,  0, 0, 0, 0], // Mf
    [0,  1,  0, -1, 0, 0], // Mm
    [0,  0,  2, 0, 0, 0], // Ssa
];

/// The number of constituents a loading table specifies.
pub const SPECIFIED_CONSTITUENTS: usize = SPECIFIED.len();

/// One station's ocean loading response, as the loading provider publishes it.
///
/// Amplitudes are metres and phases degrees, both indexed by the constituent order in
/// [`SPECIFIED_CONSTITUENTS`]: M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Mm, Ssa. The three rows are the
/// radial, west and south directions, which is the order the `BLQ` format writes them in.
///
/// The phase sign is the provider's: a *positive* phase is a lag, so the displacement at argument
/// `θ` goes as `cos(θ − φ)`.
#[derive(Clone, Copy, Debug)]
pub struct OceanLoading {
    /// Response amplitude, metres, `[radial, west, south][constituent]`.
    pub amplitude: [[f64; SPECIFIED_CONSTITUENTS]; 3],
    /// Response phase lag, degrees, same indexing.
    pub phase: [[f64; SPECIFIED_CONSTITUENTS]; 3],
}

/// Displacement of a station by ocean tide loading, metres, in the local topocentric triad the
/// loading table is written in: `[radial, west, south]`.
///
/// `day_fraction` is the UT fraction of the day and `julian_centuries_tt` the TT epoch in Julian
/// centuries from J2000 — the same split
/// [`solid_earth_tide_displacement`](crate::solid_earth_tide_displacement) takes, and for the same
/// reason: the Earth's rotation phase and the orbits are on different clocks.
///
/// West and south rather than east and north because that is the convention the `BLQ` tables and
/// the reference program use, and silently flipping two of the three signs on the way in is how a
/// centimetre-level correction ends up with the wrong sign in one component.
#[must_use]
pub fn ocean_loading_topocentric(
    loading: &OceanLoading,
    day_fraction: f64,
    julian_centuries_tt: f64,
) -> [f64; 3] {
    let arguments = DoodsonArguments::new(day_fraction, julian_centuries_tt);
    let mut displacement = [0.0; 3];
    for (component, out) in displacement.iter_mut().enumerate() {
        let admittance = Admittance::new(loading, component, &arguments);
        *out = admittance.sum_over_development(&arguments);
    }
    displacement
}

/// Displacement of a station by ocean tide loading, metres, as a Cartesian offset in the same
/// Earth-fixed frame `station` is given in.
///
/// This is [`ocean_loading_topocentric`] turned into the frame the range model works in. `station`
/// is the geocentric Cartesian station position in metres.
#[must_use]
pub fn ocean_loading_displacement(
    loading: &OceanLoading,
    station: [f64; 3],
    day_fraction: f64,
    julian_centuries_tt: f64,
) -> [f64; 3] {
    let [radial, west, south] =
        ocean_loading_topocentric(loading, day_fraction, julian_centuries_tt);
    Site::new(station).to_cartesian(radial, -south, -west)
}

/// The six Doodson (Darwin) variables and their rates, at one epoch.
///
/// `TDFRPH`, split from the per-constituent evaluation it is called inside: the variables depend
/// only on the epoch, and the reference recomputes them lazily behind a `SAVE`d comparison against
/// the last date it saw. Computing them once and passing them down says the same thing without the
/// hidden state.
struct DoodsonArguments {
    /// Degrees.
    value: [f64; 6],
    /// Cycles per day.
    rate: [f64; 6],
}

impl DoodsonArguments {
    fn new(day_fraction: f64, t: f64) -> Self {
        // Delaunay arguments, degrees, in the IERS expressions the reference carries.
        let l = 134.963_402_51
            + t * (477_198.867_560_5
                + t * (0.008_855_333_3 + t * (0.000_014_343_1 + t * -0.000_000_068_0)));
        let l_prime = 357.529_109_180_6
            + t * (35_999.050_291_138_9
                + t * (-0.000_153_666_7 + t * (0.000_000_037_8 + t * -0.000_000_003_2)));
        let f = 93.272_090_62
            + t * (483_202.017_457_722_2
                + t * (-0.003_542_000_0 + t * (-0.000_000_288_1 + t * 0.000_000_001_2)));
        let d = 297.850_195_469_4
            + t * (445_267.111_446_944_5
                + t * (-0.001_769_611_1 + t * (0.000_001_831_4 + t * -0.000_000_008_8)));
        let omega = 125.044_555_01
            + t * (-1_934.136_261_972_2
                + t * (0.002_075_611_1 + t * (0.000_002_139_4 + t * -0.000_000_016_5)));

        let s = f + omega;
        let h = s - d;
        let value = [360.0 * day_fraction - d, s, h, s - l, -omega, h - l_prime];

        // Rates of the same five, cycles per day.
        let dl = 0.036_291_647_1 + 0.000_000_001_3 * t;
        let dl_prime = 0.002_737_778_6;
        let df = 0.036_748_195_1 - 0.000_000_000_5 * t;
        let dd = 0.033_863_192_0 - 0.000_000_000_3 * t;
        let domega = -0.000_147_093_8 + 0.000_000_000_3 * t;

        let ds = df + domega;
        let dh = ds - dd;
        let rate = [1.0 - dd, ds, dh, ds - dl, -domega, dh - dl_prime];

        Self { value, rate }
    }

    /// Frequency, cycles per day, of the constituent with these Doodson multipliers.
    fn frequency(&self, doodson: [i8; 6]) -> f64 {
        let mut f = 0.0;
        for (multiplier, rate) in doodson.iter().zip(self.rate) {
            f += f64::from(*multiplier) * rate;
        }
        f
    }

    /// Argument, degrees in `[0, 360)`, of the constituent with these Doodson multipliers.
    fn argument(&self, doodson: [i8; 6]) -> f64 {
        let mut theta = 0.0;
        for (multiplier, value) in doodson.iter().zip(self.value) {
            theta += f64::from(*multiplier) * value;
        }
        let wrapped = theta - 360.0 * libm::trunc(theta / 360.0);
        if wrapped < 0.0 {
            wrapped + 360.0
        } else {
            wrapped
        }
    }
}

/// The station's admittance, specified at the eleven tabulated frequencies and splined within each
/// tidal species.
///
/// Held as three consecutive runs of one array — long-period, then diurnal, then semi-diurnal —
/// because that is what sorting the eleven by frequency produces, and the species boundaries fall
/// at whole cycles per day.
struct Admittance {
    frequency: [f64; SPECIFIED_CONSTITUENTS],
    real: [f64; SPECIFIED_CONSTITUENTS],
    imaginary: [f64; SPECIFIED_CONSTITUENTS],
    /// Second derivatives of the two splines, one entry per specified constituent.
    real_curvature: [f64; SPECIFIED_CONSTITUENTS],
    imaginary_curvature: [f64; SPECIFIED_CONSTITUENTS],
    /// `[start, length]` of each species run: long-period, diurnal, semi-diurnal.
    species: [(usize, usize); 3],
}

impl Admittance {
    fn new(loading: &OceanLoading, component: usize, arguments: &DoodsonArguments) -> Self {
        let mut frequency = [0.0; SPECIFIED_CONSTITUENTS];
        let mut real = [0.0; SPECIFIED_CONSTITUENTS];
        let mut imaginary = [0.0; SPECIFIED_CONSTITUENTS];

        for (i, doodson) in SPECIFIED.iter().enumerate() {
            // The equilibrium amplitude the response is divided by is the catalogue's, so a
            // constituent the catalogue does not carry cannot be used to constrain the admittance.
            // All eleven are in it; the fallback keeps the loop total rather than panicking.
            let equilibrium = CARTWRIGHT_EDDEN
                .iter()
                .find(|c| [c.0, c.1, c.2, c.3, c.4, c.5] == *doodson)
                .map_or(0.0, |c| libm::fabs(c.6));
            frequency[i] = arguments.frequency(*doodson);
            if equilibrium == 0.0 {
                continue;
            }
            // The provider's phase is a lag, so the admittance carries its negative.
            let lag = -loading.phase[component][i].to_radians();
            let scale = loading.amplitude[component][i] / equilibrium;
            real[i] = scale * libm::cos(lag);
            imaginary[i] = scale * libm::sin(lag);
        }

        sort_by_frequency(&mut frequency, &mut real, &mut imaginary);

        // Species boundaries: long-period below 0.5 cycles/day, diurnal below 1.5, semi-diurnal
        // below 2.5. The eleven specified constituents contain nothing above that.
        let long_period = frequency.iter().filter(|f| **f < 0.5).count();
        let diurnal = frequency.iter().filter(|f| (0.5..1.5).contains(*f)).count();
        let semi_diurnal = SPECIFIED_CONSTITUENTS - long_period - diurnal;
        let species = [
            (0, long_period),
            (long_period, diurnal),
            (long_period + diurnal, semi_diurnal),
        ];

        let mut real_curvature = [0.0; SPECIFIED_CONSTITUENTS];
        let mut imaginary_curvature = [0.0; SPECIFIED_CONSTITUENTS];
        for (start, length) in species {
            let end = start + length;
            spline_curvature(
                &frequency[start..end],
                &real[start..end],
                &mut real_curvature[start..end],
            );
            spline_curvature(
                &frequency[start..end],
                &imaginary[start..end],
                &mut imaginary_curvature[start..end],
            );
        }

        Self {
            frequency,
            real,
            imaginary,
            real_curvature,
            imaginary_curvature,
            species,
        }
    }

    /// Sum the whole harmonic development at this epoch, metres.
    fn sum_over_development(&self, arguments: &DoodsonArguments) -> f64 {
        let mut total = 0.0;
        for constituent in &CARTWRIGHT_EDDEN {
            let doodson = [
                constituent.0,
                constituent.1,
                constituent.2,
                constituent.3,
                constituent.4,
                constituent.5,
            ];
            let (start, length) = self.species[usize::from(constituent.0.unsigned_abs())];
            // A species with no specified constituent has no admittance to interpolate; the
            // reference drops those terms rather than extrapolating from a neighbouring species.
            if length == 0 {
                continue;
            }
            let end = start + length;
            let frequency = arguments.frequency(doodson);
            let real = spline_evaluate(
                frequency,
                &self.frequency[start..end],
                &self.real[start..end],
                &self.real_curvature[start..end],
            );
            let imaginary = spline_evaluate(
                frequency,
                &self.frequency[start..end],
                &self.imaginary[start..end],
                &self.imaginary_curvature[start..end],
            );

            // The phase correction turns the equilibrium argument into the constituent's own: the
            // long-period species is defined with the opposite sign and the diurnal one in
            // quadrature, both conventions of the Cartwright-Edden development.
            let quadrature = match constituent.0 {
                0 => 180.0,
                1 => 90.0,
                _ => 0.0,
            };
            let phase = arguments.argument(doodson)
                + quadrature
                + libm::atan2(imaginary, real).to_degrees();
            let amplitude = constituent.6 * libm::sqrt(real * real + imaginary * imaginary);
            total += amplitude * libm::cos(phase.to_radians());
        }
        total
    }
}

/// Sort the three parallel arrays by ascending frequency.
///
/// Insertion sort: eleven elements, already nearly grouped by species, and it keeps the three
/// arrays in step without an index permutation the caller has to apply. The reference uses a Shell
/// sort and a key array for the same eleven.
fn sort_by_frequency(
    frequency: &mut [f64; SPECIFIED_CONSTITUENTS],
    real: &mut [f64; SPECIFIED_CONSTITUENTS],
    imaginary: &mut [f64; SPECIFIED_CONSTITUENTS],
) {
    for i in 1..SPECIFIED_CONSTITUENTS {
        let mut j = i;
        while j > 0 && frequency[j - 1] > frequency[j] {
            frequency.swap(j - 1, j);
            real.swap(j - 1, j);
            imaginary.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Second derivatives of the interpolating cubic through `(x, u)`, written into `curvature`.
///
/// The end conditions are the reference's: the slope at each end is that of the parabola through
/// the three nearest points, so this is a clamped spline with estimated end slopes. It reproduces
/// every quadratic but not a cubic, so it is not the not-a-knot spline. Four points are needed for
/// that; with three or fewer the reference falls back to straight lines, and so does this — a
/// station's long-period species is specified at exactly three constituents, so that branch is the
/// one Mf, Mm and Ssa always take.
fn spline_curvature(x: &[f64], u: &[f64], curvature: &mut [f64]) {
    let n = x.len();
    if n <= 3 {
        curvature.fill(0.0);
        return;
    }

    // The reference's Q(u1, x1, u2, x2): the slope at the anchor of the parabola through it and
    // the two points offset from it by (x1, u1) and (x2, u2).
    let q = |u1: f64, x1: f64, u2: f64, x2: f64| {
        (u1 / (x1 * x1) - u2 / (x2 * x2)) / (1.0 / x1 - 1.0 / x2)
    };
    let q_first = q(u[1] - u[0], x[1] - x[0], u[2] - u[0], x[2] - x[0]);
    let q_last = q(
        u[n - 2] - u[n - 1],
        x[n - 2] - x[n - 1],
        u[n - 3] - u[n - 1],
        x[n - 3] - x[n - 1],
    );

    curvature[0] = 6.0 * ((u[1] - u[0]) / (x[1] - x[0]) - q_first);
    for i in 1..n - 1 {
        let left = x[i] - x[i - 1];
        let right = x[i + 1] - x[i];
        curvature[i] =
            6.0 * (u[i - 1] / left - u[i] * (1.0 / left + 1.0 / right) + u[i + 1] / right);
    }
    curvature[n - 1] = 6.0 * (q_last + (u[n - 2] - u[n - 1]) / (x[n - 1] - x[n - 2]));

    // Thomas algorithm on the tridiagonal system, with the reference's first two pivots.
    let mut pivot = [0.0; SPECIFIED_CONSTITUENTS];
    pivot[0] = 2.0 * (x[1] - x[0]);
    pivot[1] = 1.5 * (x[1] - x[0]) + 2.0 * (x[2] - x[1]);
    curvature[1] -= 0.5 * curvature[0];
    for i in 2..n - 1 {
        let c = (x[i] - x[i - 1]) / pivot[i - 1];
        pivot[i] = 2.0 * (x[i + 1] - x[i - 1]) - c * (x[i] - x[i - 1]);
        curvature[i] -= c * curvature[i - 1];
    }
    let c = (x[n - 1] - x[n - 2]) / pivot[n - 2];
    pivot[n - 1] = (2.0 - c) * (x[n - 1] - x[n - 2]);
    curvature[n - 1] -= c * curvature[n - 2];

    curvature[n - 1] /= pivot[n - 1];
    for i in (0..n - 1).rev() {
        curvature[i] = (curvature[i] - (x[i + 1] - x[i]) * curvature[i + 1]) / pivot[i];
    }
}

/// The spline through `(x, u)` with second derivatives `curvature`, at `y`.
///
/// Outside the knots the value is held at the nearest endpoint rather than extrapolated. That is
/// the reference's choice and it is the right one here: a cubic run past the last specified
/// constituent would swing, and the admittance is a bounded physical ratio.
fn spline_evaluate(y: f64, x: &[f64], u: &[f64], curvature: &[f64]) -> f64 {
    let n = x.len();
    if y <= x[0] {
        return u[0];
    }
    if y >= x[n - 1] {
        return u[n - 1];
    }
    // The two returns above leave `x[0] < y < x[n-1]`, so the count of knots below `y` is at
    // least one and at most `n - 1` — the bracket exists and needs no fallback for the case
    // where it does not.
    let upper = x.partition_point(|k| *k < y);
    let lower = upper - 1;

    let above = x[upper] - y;
    let below = y - x[lower];
    let width = x[upper] - x[lower];
    (curvature[lower] * above * above * above + curvature[upper] * below * below * below)
        / (6.0 * width)
        + below * (u[upper] / width - curvature[upper] * width / 6.0)
        + above * (u[lower] / width - curvature[lower] * width / 6.0)
}

#[cfg(test)]
mod tests {
    // The reference routines are printed at seventeen digits; they are kept whole so a future
    // comparison against a higher-precision build has something to compare to.
    #![allow(clippy::excessive_precision)]

    use super::*;
    use crate::blq::parse_blq;

    /// Onsala and Reykjavik, the two stations `HARDISP`'s header tabulates, generated by the Bos-
    /// Scherneck loading service on 25 June 2009 from CSR4.0 with no geocentre correction.
    const PUBLISHED_STATIONS: &str = "  ONSALA
  .00352 .00123 .00080 .00032 .00187 .00112 .00063 .00003 .00082 .00044 .00037
  .00144 .00035 .00035 .00008 .00053 .00049 .00018 .00009 .00012 .00005 .00006
  .00086 .00023 .00023 .00006 .00029 .00028 .00010 .00007 .00004 .00002 .00001
   -64.7  -52.0  -96.2  -55.2  -58.8 -151.4  -65.6 -138.1    8.4    5.2    2.1
    85.5  114.5   56.5  113.6   99.4   19.1   94.1  -10.4 -167.4 -170.0 -177.7
   109.5  147.0   92.7  148.8   50.5  -55.1   36.4 -170.4  -15.0    2.3    5.2
  REYKJAVIK
  .02359 .01481 .00448 .00419 .00273 .00033 .00088 .00005 .00081 .00034 .00034
  .00514 .00280 .00089 .00078 .00106 .00074 .00035 .00018 .00004 .00001 .00003
  .00209 .00077 .00051 .00021 .00151 .00066 .00047 .00019 .00014 .00008 .00006
    78.5  102.3   76.3  104.1  -52.0 -160.4  -52.6 -128.0 -174.8 -175.7 -178.1
    54.2   93.8   38.9   96.9    2.3  -12.5    3.1  -31.6 -144.4 -122.6 -173.5
   156.2 -167.1  141.9 -164.9  155.9  178.7  155.6 -168.5  177.7  178.7  179.7
";

    /// The reference's own output for Onsala, `dU  dS  dW`, hourly from the epoch below.
    #[expect(
        clippy::unreadable_literal,
        reason = "transcribed character for character from the reference program's printed output"
    )]
    #[rustfmt::skip]
    const ONSALA_SERIES: [[f64; 3]; 24] = [
        [ 0.003094, -0.001538, -0.000895], [ 0.001812, -0.000950, -0.000193],
        [ 0.000218, -0.000248,  0.000421], [-0.001104,  0.000404,  0.000741],
        [-0.001668,  0.000863,  0.000646], [-0.001209,  0.001042,  0.000137],
        [ 0.000235,  0.000926, -0.000667], [ 0.002337,  0.000580, -0.001555],
        [ 0.004554,  0.000125, -0.002278], [ 0.006271, -0.000291, -0.002615],
        [ 0.006955, -0.000537, -0.002430], [ 0.006299, -0.000526, -0.001706],
        [ 0.004305, -0.000244, -0.000559], [ 0.001294,  0.000245,  0.000793],
        [-0.002163,  0.000819,  0.002075], [-0.005375,  0.001326,  0.003024],
        [-0.007695,  0.001622,  0.003448], [-0.008669,  0.001610,  0.003272],
        [-0.008143,  0.001262,  0.002557], [-0.006290,  0.000633,  0.001477],
        [-0.003566, -0.000155,  0.000282], [-0.000593, -0.000941, -0.000766],
        [ 0.001992, -0.001561, -0.001457], [ 0.003689, -0.001889, -0.001680],
    ];

    /// The same for Reykjavik, whose radial signal is seven centimetres peak to peak.
    #[expect(
        clippy::unreadable_literal,
        reason = "transcribed character for character from the reference program's printed output"
    )]
    #[rustfmt::skip]
    const REYKJAVIK_SERIES: [[f64; 3]; 24] = [
        [-0.005940, -0.001245, -0.000278], [ 0.013516, -0.001086,  0.003212],
        [ 0.029599, -0.000353,  0.005483], [ 0.038468,  0.000699,  0.005997],
        [ 0.038098,  0.001721,  0.004690], [ 0.028780,  0.002363,  0.001974],
        [ 0.013016,  0.002371, -0.001369], [-0.005124,  0.001653, -0.004390],
        [-0.021047,  0.000310, -0.006225], [-0.030799, -0.001383, -0.006313],
        [-0.032056, -0.003048, -0.004549], [-0.024698, -0.004288, -0.001314],
        [-0.010814, -0.004794,  0.002623], [ 0.005849, -0.004416,  0.006291],
        [ 0.020857, -0.003208,  0.008766], [ 0.030226, -0.001413,  0.009402],
        [ 0.031437,  0.000594,  0.007996], [ 0.024079,  0.002389,  0.004844],
        [ 0.009945,  0.003606,  0.000663], [-0.007426,  0.004022, -0.003581],
        [-0.023652,  0.003601, -0.006911], [-0.034618,  0.002505, -0.008585],
        [-0.037515,  0.001044, -0.008270], [-0.031544, -0.000402, -0.006125],
    ];

    /// Julian date of 2009 June 25, 0h UTC — the day the reference's test case runs on.
    const TEST_DAY_JD: f64 = 2_455_007.5;
    /// Seconds past midnight of the first output epoch, 01h 10m 45s.
    const TEST_FIRST_EPOCH: f64 = 4245.0;
    /// TT - UTC in 2009: 32.184 s plus the 34 s of TAI - UTC then in force.
    const TT_MINUS_UTC: f64 = 66.184;

    /// `(day fraction, TT Julian centuries)` of the reference's `k`-th hourly output.
    fn epoch(k: usize) -> (f64, f64) {
        #[expect(clippy::cast_precision_loss, reason = "k < 24")]
        let day_fraction = (TEST_FIRST_EPOCH + 3600.0 * k as f64) / 86400.0;
        let t = (TEST_DAY_JD + day_fraction - 2_451_545.0 + TT_MINUS_UTC / 86400.0) / 36525.0;
        (day_fraction, t)
    }

    /// Worst absolute disagreement with a published series, metres.
    fn worst_against(station: usize, series: &[[f64; 3]; 24]) -> f64 {
        let site = &parse_blq(PUBLISHED_STATIONS)[station];
        let mut worst: f64 = 0.0;
        for (k, want) in series.iter().enumerate() {
            let (day_fraction, t) = epoch(k);
            let [radial, west, south] = ocean_loading_topocentric(&site.loading, day_fraction, t);
            // The reference prints dU, dS, dW; this returns radial, west, south.
            worst = worst
                .max((radial - want[0]).abs())
                .max((south - want[1]).abs())
                .max((west - want[2]).abs());
        }
        worst
    }

    /// The whole model, against the reference program's own published output.
    ///
    /// The bound is the reference's print resolution and nothing looser: it writes `3F14.6`, so a
    /// value it prints is only defined to +/- 5e-7 m, and a port that lands inside that has
    /// reproduced every digit the reference committed to. Measured worst over the 72 published
    /// Onsala values: 4.915e-7 m, which is that rounding and not a modelling difference.
    #[test]
    fn the_series_reproduces_the_reference_program_at_onsala() {
        let worst = worst_against(0, &ONSALA_SERIES);
        assert!(
            worst < 1e-6,
            "worst disagreement with HARDISP at Onsala: {worst:.4e} m"
        );
    }

    /// The same at the station with the large signal, where a proportional error would show up an
    /// order of magnitude more clearly: Reykjavik's radial term reaches 3.8 cm against Onsala's
    /// 0.9 cm. Measured worst over its 72 published values: 4.980e-7 m.
    #[test]
    fn the_series_reproduces_the_reference_program_at_reykjavik() {
        let worst = worst_against(1, &REYKJAVIK_SERIES);
        assert!(
            worst < 1e-6,
            "worst disagreement with HARDISP at Reykjavik: {worst:.4e} m"
        );
    }

    /// The Cartesian form is the topocentric one in the station's own triad, and the two sign
    /// flips on the way (west to east, south to north) are the place that silently goes wrong.
    ///
    /// A station on the equator at longitude zero makes the triad the coordinate axes: up is `+x`,
    /// east is `+y`, north is `+z`. So a radial displacement must land wholly in `x`, a *west* one
    /// as negative `y`, and a *south* one as negative `z`.
    #[test]
    fn the_cartesian_form_carries_west_and_south_into_the_right_axes() {
        let station = [6_378_137.0, 0.0, 0.0];
        let (day_fraction, t) = epoch(0);
        let site = &parse_blq(PUBLISHED_STATIONS)[1];
        let [radial, west, south] = ocean_loading_topocentric(&site.loading, day_fraction, t);
        let cartesian = ocean_loading_displacement(&site.loading, station, day_fraction, t);

        assert_within_ulps(cartesian[0], radial, 1, "radial into +x");
        assert_within_ulps(cartesian[1], -west, 1, "west into -y");
        assert_within_ulps(cartesian[2], -south, 1, "south into -z");
    }

    /// A station with no response has no displacement — and the 342-term sum has to come out at
    /// exactly zero, not at the accumulated rounding of 342 nonzero terms.
    #[test]
    fn a_station_with_no_response_does_not_move() {
        let still = OceanLoading {
            amplitude: [[0.0; SPECIFIED_CONSTITUENTS]; 3],
            phase: [[0.0; SPECIFIED_CONSTITUENTS]; 3],
        };
        let (day_fraction, t) = epoch(7);
        let moved = ocean_loading_topocentric(&still, day_fraction, t);
        for (axis, value) in ["radial", "west", "south"].iter().zip(moved) {
            assert_within_ulps(value, 0.0, 0, axis);
        }
    }

    /// The response scales with the table: doubling every amplitude doubles the displacement.
    ///
    /// The admittance is linear in the tabulated amplitude and the spline is linear in its knot
    /// values, so this holds exactly in exact arithmetic and to rounding in `f64`. It is the one
    /// property that pins the whole chain — divide by equilibrium amplitude, spline, multiply
    /// back — without a second implementation of it.
    #[test]
    fn the_displacement_is_linear_in_the_tabulated_amplitude() {
        let site = &parse_blq(PUBLISHED_STATIONS)[0];
        let mut doubled = site.loading;
        for row in &mut doubled.amplitude {
            for a in row {
                *a *= 2.0;
            }
        }
        let (day_fraction, t) = epoch(11);
        let single = ocean_loading_topocentric(&site.loading, day_fraction, t);
        let double = ocean_loading_topocentric(&doubled, day_fraction, t);
        for (axis, (one, two)) in ["radial", "west", "south"]
            .iter()
            .zip(single.iter().zip(double))
        {
            assert_within_ulps(two, 2.0 * one, 4, axis);
        }
    }

    /// Shifting every tabulated phase by a whole turn is the same table.
    ///
    /// Not exactly the same bits: 360 is not representable as a multiple of the phase's ulp, so the
    /// cosine argument moves in the last place. Measured worst over the three components at this
    /// epoch: 6.9e-18 m, thirteen orders below the bound.
    #[test]
    fn the_displacement_is_invariant_under_a_whole_turn_of_phase() {
        let site = &parse_blq(PUBLISHED_STATIONS)[1];
        let mut turned = site.loading;
        for row in &mut turned.phase {
            for p in row {
                *p += 360.0;
            }
        }
        let (day_fraction, t) = epoch(3);
        let before = ocean_loading_topocentric(&site.loading, day_fraction, t);
        let after = ocean_loading_topocentric(&turned, day_fraction, t);
        for (axis, (a, b)) in ["radial", "west", "south"]
            .iter()
            .zip(before.iter().zip(after))
        {
            assert!(
                (b - a).abs() < 1e-15,
                "{axis}: a whole turn of phase moved the station by {:.3e} m",
                b - a
            );
        }
    }

    /// The tidal arguments underneath, against `TDFRPH` itself at full precision.
    ///
    /// The published displacement series pins the whole chain, but only to the half micron its
    /// print format carries. These are the reference routine's own frequency and phase at the
    /// same epoch, printed at seventeen digits by `tools/iers-oracle`, so a future disagreement
    /// separates into "the epoch chain moved" and "the admittance spline moved" instead of being
    /// one number that got worse. S2 is in the list because its phase is `720 x` the day fraction
    /// exactly, so a wrong day fraction cannot hide in it.
    ///
    /// The tolerance is zero ULP, and that is the measured agreement rather than an aspiration:
    /// the Delaunay polynomials are evaluated in the same Horner order as the reference, so the
    /// two produce the same bits on all eight values.
    #[test]
    fn the_tidal_arguments_match_the_reference_routine() {
        let (day_fraction, t) = epoch(0);
        let arguments = DoodsonArguments::new(day_fraction, t);
        let cases = [
            (
                "M2",
                [2, 0, 0, 0, 0, 0],
                1.932_273_616_056_879_6,
                337.520_428_307_922,
            ),
            (
                "K1",
                [1, 1, 0, 0, 0, 0],
                1.002_737_909_309_48,
                111.007_111_552_593_41,
            ),
            (
                "Ssa",
                [0, 0, 2, 0, 0, 0],
                0.005_475_818_618_959_891,
                186.639_223_105_186_83,
            ),
            ("S2", [2, 2, -2, 0, 0, 0], 2.0, 35.375),
        ];
        for (name, doodson, frequency, argument) in cases {
            assert_within_ulps(
                arguments.frequency(doodson),
                frequency,
                0,
                &format!("{name} freq"),
            );
            assert_within_ulps(
                arguments.argument(doodson),
                argument,
                0,
                &format!("{name} phase"),
            );
        }
    }

    /// The documented range is half-open, and the constituent that sits on the open end is the
    /// permanent tide: all six multipliers zero, so the argument is exactly zero and must stay
    /// zero rather than wrapping to a full turn. This catalogue does not carry that row, but the
    /// wrap is written for arguments in general and the boundary is where it can be written
    /// wrongly without any real constituent noticing.
    #[test]
    fn a_zero_argument_stays_at_zero_rather_than_a_whole_turn() {
        let (day_fraction, t) = epoch(0);
        let arguments = DoodsonArguments::new(day_fraction, t);
        assert_within_ulps(arguments.argument([0; 6]), 0.0, 0, "permanent tide");
        assert_within_ulps(arguments.frequency([0; 6]), 0.0, 0, "permanent tide rate");
    }

    /// The eleven specified frequencies are pairwise distinct, and by a wide margin.
    ///
    /// The insertion sort that orders them is stable, and with distinct keys stability is
    /// unobservable — which is what makes the strictness of its comparison a free choice. This
    /// pins the premise. The closest pairs are P1 against K1 and S2 against K2, both separated by
    /// 0.005 475 8 cycles per day, which is twice the rate of the Sun's mean longitude and so
    /// varies with epoch only in the ninth decimal: the margin is not an artefact of the epoch
    /// this test picks.
    #[test]
    fn the_eleven_specified_frequencies_are_distinct() {
        let (day_fraction, t) = epoch(0);
        let arguments = DoodsonArguments::new(day_fraction, t);
        let mut frequency = [0.0; SPECIFIED_CONSTITUENTS];
        let mut real = [0.0; SPECIFIED_CONSTITUENTS];
        let mut imaginary = [0.0; SPECIFIED_CONSTITUENTS];
        for (i, doodson) in SPECIFIED.iter().enumerate() {
            frequency[i] = arguments.frequency(*doodson);
        }
        sort_by_frequency(&mut frequency, &mut real, &mut imaginary);

        let mut narrowest = f64::INFINITY;
        for pair in frequency.windows(2) {
            let gap = pair[1] - pair[0];
            assert!(gap > 0.0, "frequencies {pair:?} are not strictly ordered");
            narrowest = narrowest.min(gap);
        }
        // Measured: 0.005_475_818_618_959_849 cycles per day.
        assert!(
            narrowest > 5e-3,
            "narrowest gap {narrowest} is too close to call"
        );
    }

    /// The spline, against `SPLINE` and `EVAL` themselves at six unequally spaced knots.
    ///
    /// Six and not four on purpose. Every band this model splines has exactly four knots — three
    /// for the long-period species, which takes the straight-line branch, and four each for the
    /// diurnal and semi-diurnal — and at four the index arithmetic `n - 2` and `n / 2` agree, as
    /// do `n - 3` and `n / 3`. So the published displacement series cannot distinguish them and
    /// the general case would go untested if this test used the shape the model uses.
    ///
    /// The values are the `-fdefault-real-8` build of the reference, printed at seventeen digits
    /// by `tools/iers-oracle`. The default build differs in the seventh digit because `SPLINE` and
    /// `EVAL` declare their working variables `REAL`; the tolerance below is against the double
    /// build, which is what this port computes.
    #[test]
    fn the_spline_matches_the_reference_routine_away_from_four_knots() {
        let x = [0.31, 0.62, 1.07, 1.55, 2.03, 2.40];
        let u = [0.70, -1.30, 0.45, -0.11, 0.93, -0.28];
        let want_curvature = [
            17.072_361_645_505_36,
            47.490_817_173_052_626,
            -34.301_029_912_129_962,
            25.199_405_365_322_32,
            -24.829_924_882_492_62,
            -6.774_226_747_942_885,
        ];
        let want_midpoint = [
            -0.687_782_592_778_963_7,
            -0.591_933_245_021_052_6,
            0.301_063_393_474_029_99,
            0.404_679_481_047_252_71,
            0.595_413_022_387_913_94,
        ];

        let mut curvature = [0.0; SPECIFIED_CONSTITUENTS];
        spline_curvature(&x, &u, &mut curvature[..6]);
        for (i, want) in want_curvature.iter().enumerate() {
            assert_within_ulps(curvature[i], *want, 8, &format!("second derivative {i}"));
        }
        for (i, want) in want_midpoint.iter().enumerate() {
            let y = f64::midpoint(x[i], x[i + 1]);
            assert_within_ulps(
                spline_evaluate(y, &x, &u, &curvature[..6]),
                *want,
                8,
                &format!("between knots {i} and {}", i + 1),
            );
        }
    }

    /// At a knot the spline returns the value it was built from — independently of the curvature,
    /// which is why the reference comparison above is the test that pins the arithmetic — and
    /// outside the knots it holds the endpoint rather than running a cubic off the end.
    #[test]
    fn the_spline_interpolates_its_knots_and_does_not_extrapolate() {
        let x = [0.5, 1.0, 1.5, 2.0, 2.5];
        let u = [1.0, -2.0, 3.0, -4.0, 5.0];
        let mut curvature = [0.0; SPECIFIED_CONSTITUENTS];
        spline_curvature(&x, &u, &mut curvature[..5]);
        for (knot, value) in x.iter().zip(u) {
            assert_within_ulps(
                spline_evaluate(*knot, &x, &u, &curvature[..5]),
                value,
                8,
                "at a knot",
            );
        }
        assert_within_ulps(
            spline_evaluate(-3.0, &x, &u, &curvature[..5]),
            u[0],
            0,
            "below",
        );
        assert_within_ulps(
            spline_evaluate(9.0, &x, &u, &curvature[..5]),
            u[4],
            0,
            "above",
        );
    }

    /// Three knots is not enough for the reference's end conditions, so it falls back to straight
    /// lines — zero curvature. That branch is not an edge case here: the long-period species is
    /// specified at exactly three constituents, Mf, Mm and Ssa, so every station takes it.
    #[test]
    fn three_knots_fall_back_to_straight_lines() {
        let x = [0.0, 1.0, 3.0];
        let u = [2.0, -5.0, 11.0];
        let mut curvature = [7.0; SPECIFIED_CONSTITUENTS];
        spline_curvature(&x, &u, &mut curvature[..3]);
        for (i, c) in curvature[..3].iter().enumerate() {
            assert_within_ulps(*c, 0.0, 0, &format!("curvature {i}"));
        }
        // With zero curvature the evaluation is the straight line between the bracketing knots.
        assert_within_ulps(
            spline_evaluate(2.0, &x, &u, &curvature[..3]),
            3.0,
            2,
            "midpoint",
        );
    }

    use spacelab_testkit::assert_within_ulps;
}
