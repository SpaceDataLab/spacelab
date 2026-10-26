//! NRLMSISE-00 empirical thermospheric density and temperature model (Picone et al. 2002).
//!
//! # Source of record
//!
//! The model's own FORTRAN, `nrlmsise00_sub.for`, by M. Picone, A. E. Hedin and D. Drob at the
//! **U.S. Naval Research Laboratory**, distributed through NASA GSFC's modelweb. A work of the
//! United States Government is not subject to domestic copyright (17 U.S.C. §105), which is what
//! makes it the right thing to descend from: the widely-circulated C translation by Dominik
//! Brodowski (release 20041227) carries **no licence grant at all** — no `COPYING`, only a warranty
//! disclaimer — and silence grants nothing. `docs/provenance.md` records the reading.
//!
//! The coefficient tables in `tables` are machine-extracted from `BLOCK DATA GTD7BK` of that
//! FORTRAN by `derivations/nrlmsise00_tables.py`, and the extraction is checked rather than
//! asserted: regenerating from the FORTRAN reproduces every one of the 3 290 published values
//! **byte for byte**, digits and all.
//!
//! # What this implementation is, and what it is not
//!
//! It is a port of the FORTRAN's algorithm, routine for routine — `glatf`, `ccor`, `ccor2`, `dnet`,
//! `scalh`, `spline`, `splint`, `splini`, `densm`, `densu`, `globe7`, `glob7s`,
//! [`gts7`], [`gtd7`], [`gtd7d`] each carry the name of the `SUBROUTINE` or `FUNCTION` they
//! implement, and `g0` / `sumex` / `sg0` / `zeta` are its statement functions promoted to real ones.
//!
//! It is *not* the FORTRAN's data flow. NRLMSISE-00 shuttles values between routines through eight
//! `COMMON` blocks — `/PARMB/` (`GSURF`, `RE`), `/MESO7/` (the temperature nodes), `/LPOLY/` (the
//! Legendre and activity scratch), `/DMIX/` (`DM28`), and four more — which is global mutable state
//! by another name, and which a straight translation reproduces as file-scope `static mut`. Here
//! every one of them is threaded explicitly, as locals, as small structs (`Lpoly`) and as a return
//! record (`Gts7Result`). That is a departure from both the FORTRAN and its C translation, and it
//! is the reason the whole model is `#![forbid(unsafe_code)]`-clean and reentrant.
//!
//! The public surface is [`gtd7`] (surface-to-exosphere neutral atmosphere), [`gtd7d`] (effective
//! total mass density for drag, including anomalous oxygen) and [`gts7`] (thermospheric part). The
//! tables index 1:1 with the FORTRAN arrays, zero-based rather than one-based because Rust is.
//!
//! Gated by its data and its structure: `derivations/verify.sh` regenerates the tables from the
//! pinned FORTRAN and compares them byte for byte, and `derivations/nrlmsise00_structure.mac`
//! proves the Bates–Walker profile, the diffusive closed form and the turbopause blend against the
//! equations they solve. The authors' 17 printed outputs and the 496-point sweep of the C are
//! comparators (see the module tests).

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::if_not_else,
    clippy::assign_op_pattern
)]

mod atmosphere;
mod tables;

pub use atmosphere::Nrlmsise00Atmosphere;
use tables::{PAVGM, PD, PDL, PDM, PMA, PS, PT, PTL, PTM};

// `SAM` ships with the published coefficient set but is unused by these routines — as in the
// FORTRAN, where it is the tail of `COMMON/PARM7/` and no routine reads it. Reference it here so
// the table is not flagged as dead code, without exposing the private `tables` module publicly.
const _: usize = tables::SAM.len();

/// Inputs to the NRLMSISE-00 model for a single space-time point.
///
/// Longitude, UT (`sec`) and local solar time (`lst`) are used independently; for the most
/// physically realistic result they should be consistent (`lst = sec/3600 + g_long/15`).
#[derive(Debug, Clone, Copy)]
pub struct Nrlmsise00Input {
    /// Year (currently ignored by the model, kept for API parity with the reference).
    pub year: i32,
    /// Day of year, 1..=366.
    pub doy: i32,
    /// Seconds into the day (UT).
    pub sec: f64,
    /// Altitude in kilometres.
    pub alt: f64,
    /// Geodetic latitude in degrees.
    pub g_lat: f64,
    /// Geodetic longitude in degrees.
    pub g_long: f64,
    /// Local apparent solar time in hours.
    pub lst: f64,
    /// 81-day average of F10.7 flux, centred on `doy`.
    pub f107a: f64,
    /// Daily F10.7 flux for the previous day.
    pub f107: f64,
    /// Daily magnetic index Ap.
    pub ap: f64,
    /// Optional array of 7 magnetic Ap values. `Some(..)` selects the C mode used when
    /// `switches[9] == -1`: `[daily Ap, 3h Ap now, 3h Ap −3h, −6h, −9h, avg 12–33h, avg 36–57h]`.
    pub ap_array: Option<[f64; 7]>,
}

/// Model outputs: number densities `d` and temperatures `t`.
///
/// `d[0]` He, `d[1]` O, `d[2]` N2, `d[3]` O2, `d[4]` Ar, `d[5]` total mass density, `d[6]` H,
/// `d[7]` N, `d[8]` anomalous O (all number densities in cm⁻³ except `d[5]` in g/cm³, unless the
/// metric switch is set). `t[0]` exospheric temperature, `t[1]` temperature at altitude (K).
#[derive(Debug, Clone, Copy, Default)]
pub struct Nrlmsise00Output {
    /// Species number densities plus total mass density; see the struct docs for the layout.
    pub d: [f64; 9],
    /// `[exospheric temperature, temperature at altitude]` in kelvin.
    pub t: [f64; 2],
    /// Whether a temperature this evaluation depends on was held to a bound instead of taken as the
    /// model computed it.
    ///
    /// Two bounds set it, and both are the same statement — *nothing below the thermopause is hotter
    /// than the exosphere*. `temperature_node` applies it to the spline's nodes, where the
    /// unbounded `T₀/(1 − G)` otherwise runs through `±∞` into negative temperatures;
    /// `interpolated_temperature` applies it between them, where the cubic through `1/T`
    /// overshoots. Where either engages, the reference implementation returns a negative temperature
    /// and NaN or negative densities.
    ///
    /// The numbers alongside a `true` are finite, ordered and non-negative, but they are the model's
    /// admissible envelope rather than its fitted answer, and they can be far from the reference's:
    /// worst measured `3.09×` on argon at `107 km`, `90°`, `ap 400`, `F10.7 250` — a point where the
    /// reference looks perfectly healthy from outside, every output finite and positive and its
    /// temperature below its own `T∞`, while the `110 km` node it was built on had passed `T∞`.
    /// Reading this flag is the only way to tell that case from a fitted one.
    ///
    /// Measured over an 11.0-million-point grid — altitude `0..1000 km` at `0.5 km` through
    /// `60..200`, every 5° of latitude, `ap` to `400` in steps of `10`, four `F10.7` levels and four
    /// seasons — it is set on `3.83 %` of it, none of it below `ap = 160` or inside `|lat| < 50°`.
    ///
    /// The exact claim, measured against the reference C at `%.17e` over a `93 366`-point grid: every
    /// **significant** divergence of the ten fields other than `d[8]` carries this flag — `1e-10`
    /// relative being the threshold, three and a half orders above the `2.9e-14` the two
    /// implementations differ by from evaluation order alone — and `252` further points carry it
    /// with nothing differing at all. It errs toward saying so.
    ///
    /// The one uncovered difference is `d[8]`, anomalous oxygen, below `120 km` (`271` points on that
    /// grid). Its profile is anchored on `tho`, a fitted shape parameter near `4000 K` and not a
    /// temperature, so an interpolant above it is not a physical claim and flagging it would fire on
    /// `0.7 %` of the grid down to sea level. It is still bounded — otherwise `1/T` crosses zero and
    /// `d[8]` comes back negative, as it does from the reference — and the whole band is subnormal
    /// noise: `1.27e-29 cm⁻³` at its largest, against the `1e4 cm⁻³` the same field reaches where the
    /// model actually claims anomalous oxygen. Pinned by
    /// `the_anomalous_oxygen_tail_below_the_thermopause_is_negligible`.
    pub out_of_range: bool,
    /// Whether an *input* was outside the domain the model is defined on — a separate statement
    /// from [`Self::out_of_range`], which is about a temperature the model computed.
    ///
    /// The model is a fit over a stated domain and the struct that carries its inputs has public
    /// fields, so nothing stops a caller from asking for a latitude of `1e7°` or a NaN longitude.
    /// The reference C answers those without comment, and two of the answers are traps: `NaN` fails
    /// every comparison, so a NaN longitude takes the *same* branch as the `g_long <= -1000`
    /// sentinel and silently switches the longitude terms off, and a latitude far outside `±90°`
    /// is the one input that can drive the `exp1` saturation in `globe7`'s magnetic kernel, which
    /// no physical evaluation reaches. Flagging is what makes the rest of this module's
    /// unreachability claims true statements rather than statements about a grid.
    ///
    /// It is a report, not a refusal: the outputs are still whatever the reference would produce.
    /// Set when `doy` is outside `1..=366`, when `alt` is negative, when `|g_lat| > 90`, when
    /// `f107a`, `f107`, `ap` or any [`Nrlmsise00Input::ap_array`] entry is negative, or when any
    /// `f64` input is not finite. A longitude at or below `-1000` is *not* flagged — that is the
    /// model's own way of asking for the longitude terms to be dropped, pinned by
    /// `the_longitude_sentinel_includes_its_own_boundary`.
    pub input_out_of_domain: bool,
}

/// Model switches. Standard values are 0 for `switches[0]` and 1 for `switches[1..24]`
/// ([`Default`]). A value of 2 means "main effects off but cross terms on"; `switches[9] == -1`
/// selects the array-Ap magnetic-activity mode (requires [`Nrlmsise00Input::ap_array`]).
#[derive(Debug, Clone, Copy)]
pub struct Nrlmsise00Flags {
    /// The 24 raw switch values, interpreted as in the reference model.
    pub switches: [i32; 24],
}

impl Default for Nrlmsise00Flags {
    fn default() -> Self {
        let mut switches = [1i32; 24];
        switches[0] = 0;
        Self { switches }
    }
}

/// Internal effective switches derived from [`Nrlmsise00Flags`] by the `tselec` logic:
/// `sw` gates main effects, `swc` gates cross terms.
struct Sw {
    sw: [f64; 24],
    swc: [f64; 24],
}

impl Sw {
    fn from_flags(flags: &Nrlmsise00Flags) -> Self {
        let mut sw = [0.0f64; 24];
        let mut swc = [0.0f64; 24];
        for i in 0..24 {
            if i != 9 {
                sw[i] = if flags.switches[i] == 1 { 1.0 } else { 0.0 };
                swc[i] = if flags.switches[i] > 0 { 1.0 } else { 0.0 };
            } else {
                sw[i] = flags.switches[i] as f64;
                swc[i] = flags.switches[i] as f64;
            }
        }
        Self { sw, swc }
    }
}

/// The `LPOLY` scratch shared by `globe7` (writer) and `glob7s` (reader): Legendre polynomials,
/// F10.7 departure, diurnal harmonics of local time, and the magnetic-activity terms.
#[derive(Clone, Default)]
struct Lpoly {
    dfa: f64,
    plg: [[f64; 9]; 4],
    ctloc: f64,
    stloc: f64,
    c2tloc: f64,
    s2tloc: f64,
    s3tloc: f64,
    c3tloc: f64,
    apdf: f64,
    apt: [f64; 4],
}

/// The stratospheric/mesospheric spline nodes, in kilometres, descending.
///
/// Named because a mutation-testing equivalence rests on the last entry: in `densm`'s lower block
/// `z2` is `ZN3[ZN3.len() - 1]`, and because that is exactly `0.0`, `re + z2` and `re - z2` are the
/// same double. [`the_node_tables_carry_the_invariants_the_equivalences_rest_on`] pins it.
const ZN3: [f64; 5] = [32.5, 20.0, 15.0, 10.0, 0.0];
/// The upper mesospheric spline nodes, in kilometres, descending. `ZN2[0]` is the turbopause blend
/// ceiling: above it [`gtd7`] is the thermosphere alone.
const ZN2: [f64; 4] = [72.5, 55.0, 45.0, 32.5];

/// Whether the inputs lie outside the domain the model is defined on; see
/// [`Nrlmsise00Output::input_out_of_domain`] for what each clause is for.
///
/// Longitude is checked for finiteness only. Every use of it is through `cos`/`sin` of a multiple
/// of it, so it is periodic and no finite value is out of domain — including the `<= -1000`
/// sentinel, which is a request rather than a mistake. `lst` and `sec` are periodic the same way.
fn input_out_of_domain(input: &Nrlmsise00Input) -> bool {
    let finite = input.sec.is_finite()
        && input.alt.is_finite()
        && input.g_lat.is_finite()
        && input.g_long.is_finite()
        && input.lst.is_finite()
        && input.f107a.is_finite()
        && input.f107.is_finite()
        && input.ap.is_finite()
        && input
            .ap_array
            .is_none_or(|a| a.iter().all(|v| v.is_finite() && *v >= 0.0));
    !finite
        || !(1..=366).contains(&input.doy)
        || input.alt < 0.0
        || libm::fabs(input.g_lat) > 90.0
        || input.f107a < 0.0
        || input.f107 < 0.0
        || input.ap < 0.0
}

/// Latitude variation of gravity: returns `(gsurf, re)` (surface gravity, effective Earth radius).
fn glatf(lat: f64) -> (f64, f64) {
    let dgtr = 1.74533E-2;
    let c2 = libm::cos(2.0 * dgtr * lat);
    let gv = 980.616 * (1.0 - 0.0026373 * c2);
    let reff = 2.0 * gv / (3.085462E-6 + 2.27E-9 * c2) * 1.0E-5;
    (gv, reff)
}

/// Chemistry/dissociation correction (single transition scale length).
fn ccor(alt: f64, r: f64, h1: f64, zh: f64) -> f64 {
    let e = (alt - zh) / h1;
    if e > 70.0 {
        return libm::exp(0.0);
    }
    if e < -70.0 {
        return libm::exp(r);
    }
    let ex = libm::exp(e);
    let e = r / (1.0 + ex);
    libm::exp(e)
}

/// Chemistry/dissociation correction (two transition scale lengths).
fn ccor2(alt: f64, r: f64, h1: f64, zh: f64, h2: f64) -> f64 {
    let e1 = (alt - zh) / h1;
    let e2 = (alt - zh) / h2;
    if (e1 > 70.0) || (e2 > 70.0) {
        return libm::exp(0.0);
    }
    if (e1 < -70.0) && (e2 < -70.0) {
        return libm::exp(r);
    }
    let ex1 = libm::exp(e1);
    let ex2 = libm::exp(e2);
    let ccor2v = r / (1.0 + 0.5 * (ex1 + ex2));
    libm::exp(ccor2v)
}

/// Turbopause correction: root-mean of diffusive and fully-mixed densities.
fn dnet(dd: f64, dm: f64, zhm: f64, xmm: f64, xm: f64) -> f64 {
    let a = zhm / (xmm - xm);
    // The reference guards these three with `!((dm > 0) && (dd > 0))`, whose only other job is to
    // gate a diagnostic `printf`. Dropping it changes no result on any input: entering with both
    // densities positive leaves all three exact comparisons false, so the block falls through to
    // exactly the same power mean, and a negative or NaN density falls through the same way.
    if (dd == 0.0) && (dm == 0.0) {
        return 1.0;
    }
    if dm == 0.0 {
        return dd;
    }
    if dd == 0.0 {
        return dm;
    }
    let ylog = a * libm::log(dm / dd);
    if ylog < -10.0 {
        return dd;
    }
    if ylog > 10.0 {
        return dm;
    }
    dd * libm::pow(1.0 + libm::exp(ylog), 1.0 / a)
}

/// A lower-atmosphere temperature node, `T₀ / (1 − G)`, with the fractional variation `G` held below
/// the point where the node stops being a temperature.
///
/// Every mesospheric and lower-thermospheric node in the model is this Padé-like quotient of an
/// unperturbed value and `1 − G`, with `G` a sum of unbounded spherical-harmonic, seasonal and
/// geomagnetic terms from `glob7s`. Nothing in the fit constrains `G < 1`. Past about `ap = 300`
/// at `|φ| ≥ 60°` the geomagnetic terms alone carry it over: the denominator crosses zero, the node
/// runs to `+∞` and comes back **negative**, and everything built on it inherits the sign —
/// `densu`'s `(T₀/T)^k` and `dnet`'s `ln(dm/dd)` return NaN. The reference C prints
/// `dnet log error` and hands the NaN back; swept over `8.6e6` points of the `60..140 km` band it
/// returns a NaN or negative density at `0.59 %` of them, every one inside `109..120 km`,
/// `|φ| ≥ 60°`, `ap ≥ 300`. A NaN density reaching a drag force is worse than a coarse one, so this
/// diverges from the reference deliberately.
///
/// The bound is one the model already carries: **no node below the thermopause can be hotter than
/// the exosphere**, i.e. `1 − G ≥ T₀/T∞`. No invented constant, and inert wherever the model is
/// inside its fitted range — the denominator is passed through untouched, so every input with
/// `1 − G > T₀/T∞` stays bit-identical to the reference. A NaN `G` lands on the bound rather than
/// propagating.
///
/// Sets `held` when the bound bit, which is what [`Nrlmsise00Output::out_of_range`] reports. It has
/// to: a clamped node reaches the answer far outside the band the node sits in, through the
/// mesospheric spline *and* through the fully-mixed `dm28` that [`gtd7`] sizes its turbopause blend
/// with. At `60 km`, `90°`, `ap 400`, `F10.7 250` the reported temperature is bit-identical to the
/// reference at `265.1736316687663 K` — it comes from the stratospheric nodes, which are untouched —
/// while molecular oxygen differs by `93 %`, entirely through `dm28`. Without this, that divergence
/// would be silent.
fn temperature_node(t0: f64, variation: f64, t_exospheric: f64, held: &mut bool) -> f64 {
    let floor = t0 / t_exospheric;
    let denominator = 1.0 - variation;
    // `>= floor` and not `< floor`, so that a NaN — which no comparison accepts — lands on the
    // bound, exactly as `f64::max` would.
    let bounded = if denominator >= floor {
        denominator
    } else {
        floor
    };
    *held |= bounded != denominator;
    t0 / bounded
}

/// Scale height at `alt` for molecular weight `xm` and temperature `temp`.
fn scalh(gsurf: f64, re: f64, alt: f64, xm: f64, temp: f64) -> f64 {
    let rgas = 831.4;
    let g = gsurf / libm::pow(1.0 + alt / re, 2.0);
    rgas * temp / (g * xm)
}

/// Geopotential-like altitude difference used throughout `densu`/`densm`.
fn zeta(re: f64, zz: f64, zl: f64) -> f64 {
    (zz - zl) * (re + zl) / (re + zz)
}

/// Second derivatives of a natural/clamped cubic spline (Numerical Recipes).
fn spline(x: &[f64], y: &[f64], n: usize, yp1: f64, ypn: f64, y2: &mut [f64]) {
    let mut u = [0.0f64; 10];
    if yp1 > 0.99E30 {
        y2[0] = 0.0;
        u[0] = 0.0;
    } else {
        y2[0] = -0.5;
        u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
    }
    for i in 1..(n - 1) {
        let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        let p = sig * y2[i - 1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        u[i] = (6.0
            * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
            / (x[i + 1] - x[i - 1])
            - sig * u[i - 1])
            / p;
    }
    let (qn, un);
    if ypn > 0.99E30 {
        qn = 0.0;
        un = 0.0;
    } else {
        qn = 0.5;
        un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
    }
    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
    for k in (0..=(n - 2)).rev() {
        y2[k] = y2[k] * y2[k + 1] + u[k];
    }
}

/// Cubic-spline interpolation value at `x`.
fn splint(xa: &[f64], ya: &[f64], y2a: &[f64], n: usize, x: f64) -> f64 {
    let mut klo = 0usize;
    let mut khi = n - 1;
    while (khi - klo) > 1 {
        let k = usize::midpoint(khi, klo);
        if xa[k] > x {
            khi = k;
        } else {
            klo = k;
        }
    }
    let h = xa[khi] - xa[klo];
    let a = (xa[khi] - x) / h;
    let b = (x - xa[klo]) / h;
    a * ya[klo]
        + b * ya[khi]
        + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * h * h / 6.0
}

/// The temperature the `1/T` spline gives at `x`, held below `t_max`.
///
/// The model does not interpolate temperature — it interpolates its reciprocal, with an ordinary
/// clamped cubic, and nothing makes that interpolant bounded. In `1/T` the two directions of
/// overshoot fail differently: downward toward zero sends `T` to `+∞`, and crossing zero makes `T`
/// negative, after which the reference carries the sign into `(T₁/T)^{1+α}` and hands back a
/// negative number density. Both happen in the same place — the polar storm corner above
/// `ap ≈ 300`, where [`temperature_node`] has pinned neighbouring nodes to the exospheric bound and
/// left the node sequence with a kink the cubic can only bridge by overshooting.
///
/// The bound is the one [`temperature_node`] already applies to the nodes, extended to the value
/// between them: **nothing below the thermopause is hotter than the exosphere**, so `1/T ≥ 1/T∞`.
/// It is one-sided on purpose. Holding the interpolant between the two nodes that bracket it would
/// be tighter and is *wrong*: the profile has a genuine minimum at the mesopause, between nodes,
/// and clipping to the bracketing pair would flatten it — measured at `2.4 %` on argon in the
/// `72.5..120 km` band before the sweep rejected it. A ceiling on temperature constrains no shape;
/// it removes only the two ways the interpolation stops being a temperature at all.
///
/// Returns the temperature and whether the bound changed the interpolated value, which is the one
/// honest signal that the evaluation has left the domain where the model is a fit rather than an
/// envelope. A NaN counts as held rather than passing through.
fn interpolated_temperature(
    xa: &[f64],
    ya: &[f64],
    y2a: &[f64],
    n: usize,
    x: f64,
    t_max: f64,
) -> (f64, bool) {
    let raw = splint(xa, ya, y2a, n, x);
    let floor = 1.0 / t_max;
    // `raw >= floor` and not `raw < floor`, so that a NaN — which no comparison accepts — is held.
    let held = if raw >= floor { raw } else { floor };
    (1.0 / held, held != raw)
}

/// Integral of the cubic spline from `xa[0]` to `x`.
fn splini(xa: &[f64], ya: &[f64], y2a: &[f64], n: usize, x: f64) -> f64 {
    let mut yi = 0.0;
    let mut klo = 0usize;
    let mut khi = 1usize;
    while (x > xa[klo]) && (khi < n) {
        let mut xx = x;
        if khi < (n - 1) {
            xx = if x < xa[khi] { x } else { xa[khi] };
        }
        let h = xa[khi] - xa[klo];
        let a = (xa[khi] - xx) / h;
        let b = (xx - xa[klo]) / h;
        let a2 = a * a;
        let b2 = b * b;
        yi += ((1.0 - a2) * ya[klo] / 2.0
            + b2 * ya[khi] / 2.0
            + ((-(1.0 + a2 * a2) / 4.0 + a2 / 2.0) * y2a[klo]
                + (b2 * b2 / 4.0 - b2 / 2.0) * y2a[khi])
                * h
                * h
                / 6.0)
            * h;
        klo += 1;
        khi += 1;
    }
    yi
}

/// Temperature and density for the lower atmosphere. Returns `(density, tz)`.
fn densm(
    gsurf: f64,
    re: f64,
    alt: f64,
    d0: f64,
    xm: f64,
    mn3: usize,
    zn3: &[f64],
    tn3: &[f64],
    tgn3: &[f64],
    mn2: usize,
    zn2: &[f64],
    tn2: &[f64],
    tgn2: &[f64],
    t_max: f64,
    held: &mut bool,
) -> (f64, f64) {
    let rgas = 831.4;
    let mut xs = [0.0f64; 10];
    let mut ys = [0.0f64; 10];
    let mut y2out = [0.0f64; 10];
    let mut tz = 0.0;
    let mut densm_tmp = d0;
    if alt > zn2[0] {
        return (if xm == 0.0 { tz } else { d0 }, tz);
    }

    // STRATOSPHERE / MESOSPHERE temperature
    let z = if alt > zn2[mn2 - 1] {
        alt
    } else {
        zn2[mn2 - 1]
    };
    let mut mn = mn2;
    let mut z1 = zn2[0];
    let mut z2 = zn2[mn - 1];
    let mut t1 = tn2[0];
    let mut t2 = tn2[mn - 1];
    let mut zg = zeta(re, z, z1);
    let mut zgdif = zeta(re, z2, z1);
    for k in 0..mn {
        xs[k] = zeta(re, zn2[k], z1) / zgdif;
        ys[k] = 1.0 / tn2[k];
    }
    let mut yd1 = -tgn2[0] / (t1 * t1) * zgdif;
    let mut yd2 = -tgn2[1] / (t2 * t2) * zgdif * libm::pow((re + z2) / (re + z1), 2.0);
    spline(&xs, &ys, mn, yd1, yd2, &mut y2out);
    let mut x = zg / zgdif;
    let (tzv, h) = interpolated_temperature(&xs, &ys, &y2out, mn, x, t_max);
    (tz, *held) = (tzv, *held | h);
    if xm != 0.0 {
        let glb = gsurf / libm::pow(1.0 + z1 / re, 2.0);
        let gamm = xm * glb * zgdif / rgas;
        let yi = splini(&xs, &ys, &y2out, mn, x);
        let mut expl = gamm * yi;
        if expl > 50.0 {
            expl = 50.0;
        }
        densm_tmp = densm_tmp * (t1 / tz) * libm::exp(-expl);
    }

    if alt > zn3[0] {
        return (if xm == 0.0 { tz } else { densm_tmp }, tz);
    }

    // TROPOSPHERE / STRATOSPHERE temperature
    let z = alt;
    mn = mn3;
    z1 = zn3[0];
    z2 = zn3[mn - 1];
    t1 = tn3[0];
    t2 = tn3[mn - 1];
    zg = zeta(re, z, z1);
    zgdif = zeta(re, z2, z1);
    for k in 0..mn {
        xs[k] = zeta(re, zn3[k], z1) / zgdif;
        ys[k] = 1.0 / tn3[k];
    }
    yd1 = -tgn3[0] / (t1 * t1) * zgdif;
    yd2 = -tgn3[1] / (t2 * t2) * zgdif * libm::pow((re + z2) / (re + z1), 2.0);
    spline(&xs, &ys, mn, yd1, yd2, &mut y2out);
    x = zg / zgdif;
    let (tzv, h) = interpolated_temperature(&xs, &ys, &y2out, mn, x, t_max);
    (tz, *held) = (tzv, *held | h);
    if xm != 0.0 {
        let glb = gsurf / libm::pow(1.0 + z1 / re, 2.0);
        let gamm = xm * glb * zgdif / rgas;
        let yi = splini(&xs, &ys, &y2out, mn, x);
        let mut expl = gamm * yi;
        if expl > 50.0 {
            expl = 50.0;
        }
        densm_tmp = densm_tmp * (t1 / tz) * libm::exp(-expl);
    }
    (if xm == 0.0 { tz } else { densm_tmp }, tz)
}

/// Temperature and density for the MSIS thermosphere. Returns `(density, tz)`. May write the
/// end-node temperature/gradient into `tn1[0]`/`tgn1[0]` (the Bates-profile join) as the C does.
fn densu(
    gsurf: f64,
    re: f64,
    alt: f64,
    dlb: f64,
    tinf: f64,
    tlb: f64,
    xm: f64,
    alpha: f64,
    zlb: f64,
    s2: f64,
    mn1: usize,
    zn1: &[f64],
    tn1: &mut [f64],
    tgn1: &mut [f64],
    held: &mut bool,
) -> (f64, f64) {
    let rgas = 831.4;
    let mut x = 0.0;
    let mut xs = [0.0f64; 5];
    let mut ys = [0.0f64; 5];
    let mut y2out = [0.0f64; 5];
    let za = zn1[0];
    let z = if alt > za { alt } else { za };
    let zg2 = zeta(re, z, zlb);
    let tt = tinf - (tinf - tlb) * libm::exp(-s2 * zg2);
    let ta = tt;
    let mut tz = tt;
    let mut densu_temp = tz;
    let mut z1 = 0.0;
    let mut t1 = 0.0;
    let mut zgdif = 0.0;
    let mut mn = 0usize;
    if alt < za {
        let dta = (tinf - ta) * s2 * libm::pow((re + zlb) / (re + za), 2.0);
        tgn1[0] = dta;
        tn1[0] = ta;
        let z = if alt > zn1[mn1 - 1] {
            alt
        } else {
            zn1[mn1 - 1]
        };
        mn = mn1;
        z1 = zn1[0];
        let z2 = zn1[mn - 1];
        t1 = tn1[0];
        let t2 = tn1[mn - 1];
        let zg = zeta(re, z, z1);
        zgdif = zeta(re, z2, z1);
        for k in 0..mn {
            xs[k] = zeta(re, zn1[k], z1) / zgdif;
            ys[k] = 1.0 / tn1[k];
        }
        let yd1 = -tgn1[0] / (t1 * t1) * zgdif;
        let yd2 = -tgn1[1] / (t2 * t2) * zgdif * libm::pow((re + z2) / (re + z1), 2.0);
        spline(&xs, &ys, mn, yd1, yd2, &mut y2out);
        x = zg / zgdif;
        let (tzv, h) = interpolated_temperature(&xs, &ys, &y2out, mn, x, tinf);
        (tz, *held) = (tzv, *held | h);
        densu_temp = tz;
    }
    if xm == 0.0 {
        return (densu_temp, tz);
    }

    let mut glb = gsurf / libm::pow(1.0 + zlb / re, 2.0);
    let gamma = xm * glb / (s2 * rgas * tinf);
    let mut expl = libm::exp(-s2 * gamma * zg2);
    if expl > 50.0 {
        expl = 50.0;
    }
    if tt <= 0.0 {
        expl = 50.0;
    }
    let densa = dlb * libm::pow(tlb / tt, 1.0 + alpha + gamma) * expl;
    densu_temp = densa;
    if alt >= za {
        return (densu_temp, tz);
    }

    glb = gsurf / libm::pow(1.0 + z1 / re, 2.0);
    let gamm = xm * glb * zgdif / rgas;
    let yi = splini(&xs, &ys, &y2out, mn, x);
    let mut expl = gamm * yi;
    if expl > 50.0 {
        expl = 50.0;
    }
    densu_temp = densu_temp * libm::pow(t1 / tz, 1.0 + alpha) * libm::exp(-expl);
    (densu_temp, tz)
}

/// Eq. A24d: 3-hour magnetic activity kernel.
fn g0(a: f64, p: &[f64; 150], p24: f64) -> f64 {
    let sp24 = libm::sqrt(p24 * p24);
    a - 4.0 + (p[25] - 1.0) * (a - 4.0 + (libm::exp(-sp24 * (a - 4.0)) - 1.0) / sp24)
}

/// Eq. A24c.
fn sumex(ex: f64) -> f64 {
    1.0 + (1.0 - libm::pow(ex, 19.0)) / (1.0 - ex) * libm::pow(ex, 0.5)
}

/// Eq. A24a: weighted sum of `g0` over the Ap history.
fn sg0(ex: f64, p: &[f64; 150], p24: f64, ap: &[f64; 7]) -> f64 {
    (g0(ap[1], p, p24)
        + (g0(ap[2], p, p24) * ex
            + g0(ap[3], p, p24) * ex * ex
            + g0(ap[4], p, p24) * libm::pow(ex, 3.0)
            + (g0(ap[5], p, p24) * libm::pow(ex, 4.0) + g0(ap[6], p, p24) * libm::pow(ex, 12.0))
                * (1.0 - libm::pow(ex, 8.0))
                / (1.0 - ex)))
        / sumex(ex)
}

/// G(L): upper-thermosphere spatial/temporal variation function.
fn globe7(l: &mut Lpoly, p: &[f64; 150], input: &Nrlmsise00Input, sw: &Sw) -> f64 {
    let mut t = [0.0f64; 15];
    let sr = 7.2722E-5;
    let dgtr = 1.74533E-2;
    let dr = 1.72142E-2;
    let hr = 0.2618;
    let tloc = input.lst;

    let c = libm::sin(input.g_lat * dgtr);
    let s = libm::cos(input.g_lat * dgtr);
    let c2 = c * c;
    let c4 = c2 * c2;
    let s2 = s * s;

    l.plg[0][1] = c;
    l.plg[0][2] = 0.5 * (3.0 * c2 - 1.0);
    l.plg[0][3] = 0.5 * (5.0 * c * c2 - 3.0 * c);
    l.plg[0][4] = (35.0 * c4 - 30.0 * c2 + 3.0) / 8.0;
    l.plg[0][5] = (63.0 * c2 * c2 * c - 70.0 * c2 * c + 15.0 * c) / 8.0;
    l.plg[0][6] = (11.0 * c * l.plg[0][5] - 5.0 * l.plg[0][4]) / 6.0;
    l.plg[1][1] = s;
    l.plg[1][2] = 3.0 * c * s;
    l.plg[1][3] = 1.5 * (5.0 * c2 - 1.0) * s;
    l.plg[1][4] = 2.5 * (7.0 * c2 * c - 3.0 * c) * s;
    l.plg[1][5] = 1.875 * (21.0 * c4 - 14.0 * c2 + 1.0) * s;
    l.plg[1][6] = (11.0 * c * l.plg[1][5] - 6.0 * l.plg[1][4]) / 5.0;
    l.plg[2][2] = 3.0 * s2;
    l.plg[2][3] = 15.0 * s2 * c;
    l.plg[2][4] = 7.5 * (7.0 * c2 - 1.0) * s2;
    l.plg[2][5] = 3.0 * c * l.plg[2][4] - 2.0 * l.plg[2][3];
    l.plg[2][6] = (11.0 * c * l.plg[2][5] - 7.0 * l.plg[2][4]) / 4.0;
    l.plg[2][7] = (13.0 * c * l.plg[2][6] - 8.0 * l.plg[2][5]) / 5.0;
    l.plg[3][3] = 15.0 * s2 * s;
    l.plg[3][4] = 105.0 * s2 * s * c;
    l.plg[3][5] = (9.0 * c * l.plg[3][4] - 7.0 * l.plg[3][3]) / 2.0;
    l.plg[3][6] = (11.0 * c * l.plg[3][5] - 8.0 * l.plg[3][4]) / 3.0;

    if !((sw.sw[7] == 0.0 && sw.sw[8] == 0.0) && sw.sw[14] == 0.0) {
        l.stloc = libm::sin(hr * tloc);
        l.ctloc = libm::cos(hr * tloc);
        l.s2tloc = libm::sin(2.0 * hr * tloc);
        l.c2tloc = libm::cos(2.0 * hr * tloc);
        l.s3tloc = libm::sin(3.0 * hr * tloc);
        l.c3tloc = libm::cos(3.0 * hr * tloc);
    }

    let cd32 = libm::cos(dr * (input.doy as f64 - p[31]));
    let cd18 = libm::cos(2.0 * dr * (input.doy as f64 - p[17]));
    let cd14 = libm::cos(dr * (input.doy as f64 - p[13]));
    let cd39 = libm::cos(2.0 * dr * (input.doy as f64 - p[38]));

    // F10.7 effect
    let df = input.f107 - input.f107a;
    l.dfa = input.f107a - 150.0;
    t[0] = p[19] * df * (1.0 + p[59] * l.dfa)
        + p[20] * df * df
        + p[21] * l.dfa
        + p[29] * libm::pow(l.dfa, 2.0);
    let f1 = 1.0 + (p[47] * l.dfa + p[19] * df + p[20] * df * df) * sw.swc[1];
    let f2 = 1.0 + (p[49] * l.dfa + p[19] * df + p[20] * df * df) * sw.swc[1];

    // time independent
    t[1] = (p[1] * l.plg[0][2] + p[2] * l.plg[0][4] + p[22] * l.plg[0][6])
        + (p[14] * l.plg[0][2]) * l.dfa * sw.swc[1]
        + p[26] * l.plg[0][1];

    // symmetrical annual
    t[2] = p[18] * cd32;
    // symmetrical semiannual
    t[3] = (p[15] + p[16] * l.plg[0][2]) * cd18;
    // asymmetrical annual
    t[4] = f1 * (p[9] * l.plg[0][1] + p[10] * l.plg[0][3]) * cd14;
    // asymmetrical semiannual
    t[5] = p[37] * l.plg[0][1] * cd39;

    // diurnal
    if sw.sw[7] != 0.0 {
        let t71 = (p[11] * l.plg[1][2]) * cd14 * sw.swc[5];
        let t72 = (p[12] * l.plg[1][2]) * cd14 * sw.swc[5];
        t[6] = f2
            * ((p[3] * l.plg[1][1] + p[4] * l.plg[1][3] + p[27] * l.plg[1][5] + t71) * l.ctloc
                + (p[6] * l.plg[1][1] + p[7] * l.plg[1][3] + p[28] * l.plg[1][5] + t72) * l.stloc);
    }

    // semidiurnal
    if sw.sw[8] != 0.0 {
        let t81 = (p[23] * l.plg[2][3] + p[35] * l.plg[2][5]) * cd14 * sw.swc[5];
        let t82 = (p[33] * l.plg[2][3] + p[36] * l.plg[2][5]) * cd14 * sw.swc[5];
        t[7] = f2
            * ((p[5] * l.plg[2][2] + p[41] * l.plg[2][4] + t81) * l.c2tloc
                + (p[8] * l.plg[2][2] + p[42] * l.plg[2][4] + t82) * l.s2tloc);
    }

    // terdiurnal
    if sw.sw[14] != 0.0 {
        t[13] = f2
            * ((p[39] * l.plg[3][3]
                + (p[93] * l.plg[3][4] + p[46] * l.plg[3][6]) * cd14 * sw.swc[5])
                * l.s3tloc
                + (p[40] * l.plg[3][3]
                    + (p[94] * l.plg[3][4] + p[48] * l.plg[3][6]) * cd14 * sw.swc[5])
                    * l.c3tloc);
    }

    // magnetic activity based on daily ap
    if sw.sw[9] == -1.0 {
        if p[51] != 0.0 {
            let mut exp1 = libm::exp(
                -10800.0 * libm::sqrt(p[51] * p[51])
                    / (1.0 + p[138] * (45.0 - libm::sqrt(input.g_lat * input.g_lat))),
            );
            if exp1 > 0.99999 {
                exp1 = 0.99999;
            }
            let p24 = if p[24] < 1.0E-4 { 1.0E-4 } else { p[24] };
            let apa = input.ap_array.unwrap_or([0.0; 7]);
            l.apt[0] = sg0(exp1, p, p24, &apa);
            if sw.sw[9] != 0.0 {
                t[8] = l.apt[0]
                    * (p[50]
                        + p[96] * l.plg[0][2]
                        + p[54] * l.plg[0][4]
                        + (p[125] * l.plg[0][1] + p[126] * l.plg[0][3] + p[127] * l.plg[0][5])
                            * cd14
                            * sw.swc[5]
                        + (p[128] * l.plg[1][1] + p[129] * l.plg[1][3] + p[130] * l.plg[1][5])
                            * sw.swc[7]
                            * libm::cos(hr * (tloc - p[131])));
            }
        }
    } else {
        let apd = input.ap - 4.0;
        let mut p44 = p[43];
        let p45 = p[44];
        if p44 < 0.0 {
            p44 = 1.0E-5;
        }
        l.apdf = apd + (p45 - 1.0) * (apd + (libm::exp(-p44 * apd) - 1.0) / p44);
        if sw.sw[9] != 0.0 {
            t[8] = l.apdf
                * (p[32]
                    + p[45] * l.plg[0][2]
                    + p[34] * l.plg[0][4]
                    + (p[100] * l.plg[0][1] + p[101] * l.plg[0][3] + p[102] * l.plg[0][5])
                        * cd14
                        * sw.swc[5]
                    + (p[121] * l.plg[1][1] + p[122] * l.plg[1][3] + p[123] * l.plg[1][5])
                        * sw.swc[7]
                        * libm::cos(hr * (tloc - p[124])));
        }
    }

    if (sw.sw[10] != 0.0) && (input.g_long > -1000.0) {
        // longitudinal
        if sw.sw[11] != 0.0 {
            t[10] = (1.0 + p[80] * l.dfa * sw.swc[1])
                * ((p[64] * l.plg[1][2]
                    + p[65] * l.plg[1][4]
                    + p[66] * l.plg[1][6]
                    + p[103] * l.plg[1][1]
                    + p[104] * l.plg[1][3]
                    + p[105] * l.plg[1][5]
                    + sw.swc[5]
                        * (p[109] * l.plg[1][1] + p[110] * l.plg[1][3] + p[111] * l.plg[1][5])
                        * cd14)
                    * libm::cos(dgtr * input.g_long)
                    + (p[90] * l.plg[1][2]
                        + p[91] * l.plg[1][4]
                        + p[92] * l.plg[1][6]
                        + p[106] * l.plg[1][1]
                        + p[107] * l.plg[1][3]
                        + p[108] * l.plg[1][5]
                        + sw.swc[5]
                            * (p[112] * l.plg[1][1]
                                + p[113] * l.plg[1][3]
                                + p[114] * l.plg[1][5])
                            * cd14)
                        * libm::sin(dgtr * input.g_long));
        }

        // ut and mixed ut, longitude
        if sw.sw[12] != 0.0 {
            t[11] = (1.0 + p[95] * l.plg[0][1])
                * (1.0 + p[81] * l.dfa * sw.swc[1])
                * (1.0 + p[119] * l.plg[0][1] * sw.swc[5] * cd14)
                * ((p[68] * l.plg[0][1] + p[69] * l.plg[0][3] + p[70] * l.plg[0][5])
                    * libm::cos(sr * (input.sec - p[71])));
            t[11] += sw.swc[11]
                * (p[76] * l.plg[2][3] + p[77] * l.plg[2][5] + p[78] * l.plg[2][7])
                * libm::cos(sr * (input.sec - p[79]) + 2.0 * dgtr * input.g_long)
                * (1.0 + p[137] * l.dfa * sw.swc[1]);
        }

        // ut, longitude magnetic activity
        if sw.sw[13] != 0.0 {
            if sw.sw[9] == -1.0 {
                if p[51] != 0.0 {
                    t[12] = l.apt[0]
                        * sw.swc[11]
                        * (1.0 + p[132] * l.plg[0][1])
                        * ((p[52] * l.plg[1][2] + p[98] * l.plg[1][4] + p[67] * l.plg[1][6])
                            * libm::cos(dgtr * (input.g_long - p[97])))
                        + l.apt[0]
                            * sw.swc[11]
                            * sw.swc[5]
                            * (p[133] * l.plg[1][1] + p[134] * l.plg[1][3] + p[135] * l.plg[1][5])
                            * cd14
                            * libm::cos(dgtr * (input.g_long - p[136]))
                        + l.apt[0]
                            * sw.swc[12]
                            * (p[55] * l.plg[0][1] + p[56] * l.plg[0][3] + p[57] * l.plg[0][5])
                            * libm::cos(sr * (input.sec - p[58]));
                }
            } else {
                t[12] = l.apdf
                    * sw.swc[11]
                    * (1.0 + p[120] * l.plg[0][1])
                    * ((p[60] * l.plg[1][2] + p[61] * l.plg[1][4] + p[62] * l.plg[1][6])
                        * libm::cos(dgtr * (input.g_long - p[63])))
                    + l.apdf
                        * sw.swc[11]
                        * sw.swc[5]
                        * (p[115] * l.plg[1][1] + p[116] * l.plg[1][3] + p[117] * l.plg[1][5])
                        * cd14
                        * libm::cos(dgtr * (input.g_long - p[118]))
                    + l.apdf
                        * sw.swc[12]
                        * (p[83] * l.plg[0][1] + p[84] * l.plg[0][3] + p[85] * l.plg[0][5])
                        * libm::cos(sr * (input.sec - p[75]));
            }
        }
    }

    let mut tinf = p[30];
    for i in 0..14 {
        tinf += libm::fabs(sw.sw[i + 1]) * t[i];
    }
    tinf
}

/// G(L) variant for the lower atmosphere (glob7s). Reads the `LPOLY` scratch filled by `globe7`.
fn glob7s(l: &Lpoly, p: &[f64; 100], input: &Nrlmsise00Input, sw: &Sw) -> f64 {
    let pset = 2.0;
    let mut t = [0.0f64; 14];
    let dr = 1.72142E-2;
    let dgtr = 1.74533E-2;
    let p99 = if p[99] == 0.0 { pset } else { p[99] };
    if p99 != pset {
        // Wrong parameter set: the reference prints and returns −1; verified tables never hit this.
        return -1.0;
    }
    let cd32 = libm::cos(dr * (input.doy as f64 - p[31]));
    let cd18 = libm::cos(2.0 * dr * (input.doy as f64 - p[17]));
    let cd14 = libm::cos(dr * (input.doy as f64 - p[13]));
    let cd39 = libm::cos(2.0 * dr * (input.doy as f64 - p[38]));

    // F10.7
    t[0] = p[21] * l.dfa;
    // time independent
    t[1] = p[1] * l.plg[0][2]
        + p[2] * l.plg[0][4]
        + p[22] * l.plg[0][6]
        + p[26] * l.plg[0][1]
        + p[14] * l.plg[0][3]
        + p[59] * l.plg[0][5];
    // symmetrical annual
    t[2] = (p[18] + p[47] * l.plg[0][2] + p[29] * l.plg[0][4]) * cd32;
    // symmetrical semiannual
    t[3] = (p[15] + p[16] * l.plg[0][2] + p[30] * l.plg[0][4]) * cd18;
    // asymmetrical annual
    t[4] = (p[9] * l.plg[0][1] + p[10] * l.plg[0][3] + p[20] * l.plg[0][5]) * cd14;
    // asymmetrical semiannual
    t[5] = (p[37] * l.plg[0][1]) * cd39;

    // diurnal
    if sw.sw[7] != 0.0 {
        let t71 = p[11] * l.plg[1][2] * cd14 * sw.swc[5];
        let t72 = p[12] * l.plg[1][2] * cd14 * sw.swc[5];
        t[6] = (p[3] * l.plg[1][1] + p[4] * l.plg[1][3] + t71) * l.ctloc
            + (p[6] * l.plg[1][1] + p[7] * l.plg[1][3] + t72) * l.stloc;
    }

    // semidiurnal
    if sw.sw[8] != 0.0 {
        let t81 = (p[23] * l.plg[2][3] + p[35] * l.plg[2][5]) * cd14 * sw.swc[5];
        let t82 = (p[33] * l.plg[2][3] + p[36] * l.plg[2][5]) * cd14 * sw.swc[5];
        t[7] = (p[5] * l.plg[2][2] + p[41] * l.plg[2][4] + t81) * l.c2tloc
            + (p[8] * l.plg[2][2] + p[42] * l.plg[2][4] + t82) * l.s2tloc;
    }

    // terdiurnal
    if sw.sw[14] != 0.0 {
        t[13] = p[39] * l.plg[3][3] * l.s3tloc + p[40] * l.plg[3][3] * l.c3tloc;
    }

    // magnetic activity
    if sw.sw[9] != 0.0 {
        if sw.sw[9] == 1.0 {
            t[8] = l.apdf * (p[32] + p[45] * l.plg[0][2] * sw.swc[2]);
        }
        if sw.sw[9] == -1.0 {
            t[8] = p[50] * l.apt[0] + p[96] * l.plg[0][2] * l.apt[0] * sw.swc[2];
        }
    }

    // longitudinal
    if !((sw.sw[10] == 0.0) || (sw.sw[11] == 0.0) || (input.g_long <= -1000.0)) {
        t[10] = (1.0
            + l.plg[0][1]
                * (p[80] * sw.swc[5] * libm::cos(dr * (input.doy as f64 - p[81]))
                    + p[85] * sw.swc[6] * libm::cos(2.0 * dr * (input.doy as f64 - p[86])))
            + p[83] * sw.swc[3] * libm::cos(dr * (input.doy as f64 - p[84]))
            + p[87] * sw.swc[4] * libm::cos(2.0 * dr * (input.doy as f64 - p[88])))
            * ((p[64] * l.plg[1][2]
                + p[65] * l.plg[1][4]
                + p[66] * l.plg[1][6]
                + p[74] * l.plg[1][1]
                + p[75] * l.plg[1][3]
                + p[76] * l.plg[1][5])
                * libm::cos(dgtr * input.g_long)
                + (p[90] * l.plg[1][2]
                    + p[91] * l.plg[1][4]
                    + p[92] * l.plg[1][6]
                    + p[77] * l.plg[1][1]
                    + p[78] * l.plg[1][3]
                    + p[79] * l.plg[1][5])
                    * libm::sin(dgtr * input.g_long));
    }
    let mut tt = 0.0;
    for i in 0..14 {
        tt += libm::fabs(sw.sw[i + 1]) * t[i];
    }
    tt
}

/// Values that the thermospheric routine (`gts7`) must hand back to `gtd7`: the full output, the
/// N2 mixed density `dm28`, the top lower-thermosphere temperature node and its gradient, and the
/// final `LPOLY` scratch (set by the last `globe7` call) that `gtd7`'s `glob7s` calls consume.
struct Gts7Result {
    output: Nrlmsise00Output,
    dm28: f64,
    tn1_4: f64,
    tgn1_1: f64,
    lpoly: Lpoly,
}

/// Thermospheric portion of NRLMSISE-00 (`alt > 72.5 km`). `gsurf`/`re` come from `glatf`.
fn gts7_impl(input: &Nrlmsise00Input, sw: &Sw, gsurf: f64, re: f64) -> Gts7Result {
    let dgtr = 1.74533E-2;
    let dr = 1.72142E-2;
    let alpha = [-0.38, 0.0, 0.0, 0.0, 0.17, 0.0, -0.38, 0.0, 0.0];
    let altl = [200.0, 300.0, 160.0, 250.0, 240.0, 450.0, 320.0, 450.0];
    let mn1 = 5usize;
    let mut zn1 = [120.0, 110.0, 100.0, 90.0, 72.5];
    let za = PDL[1][15];
    zn1[0] = za;

    let mut l = Lpoly::default();
    let mut held = false;
    let mut meso_tn1 = [0.0f64; 5];
    let mut meso_tgn1 = [0.0f64; 2];
    let mut output = Nrlmsise00Output {
        d: [0.0; 9],
        t: [0.0; 2],
        out_of_range: false,
        // Set by the public entry points, which see the caller's own inputs: `gtd7` hands this
        // one an altitude it has already raised to the lowest thermospheric node.
        input_out_of_domain: false,
    };

    // TINF: variations unimportant below za / zn1[0]
    let tinf = if input.alt > zn1[0] {
        PTM[0] * PT[0] * (1.0 + sw.sw[16] * globe7(&mut l, &PT, input, sw))
    } else {
        PTM[0] * PT[0]
    };
    output.t[0] = tinf;

    // gradient variations unimportant below zn1[4]
    let g0v = if input.alt > zn1[4] {
        PTM[3] * PS[0] * (1.0 + sw.sw[19] * globe7(&mut l, &PS, input, sw))
    } else {
        PTM[3] * PS[0]
    };
    let tlb = PTM[1] * (1.0 + sw.sw[17] * globe7(&mut l, &PD[3], input, sw)) * PD[3][0];
    let s = g0v / (tinf - tlb);

    // Lower thermosphere temperatures (unimportant for density above 300 km)
    if input.alt < 300.0 {
        meso_tn1[1] = temperature_node(
            PTM[6] * PTL[0][0],
            sw.sw[18] * glob7s(&l, &PTL[0], input, sw),
            tinf,
            &mut held,
        );
        meso_tn1[2] = temperature_node(
            PTM[2] * PTL[1][0],
            sw.sw[18] * glob7s(&l, &PTL[1], input, sw),
            tinf,
            &mut held,
        );
        meso_tn1[3] = temperature_node(
            PTM[7] * PTL[2][0],
            sw.sw[18] * glob7s(&l, &PTL[2], input, sw),
            tinf,
            &mut held,
        );
        meso_tn1[4] = temperature_node(
            PTM[4] * PTL[3][0],
            sw.sw[18] * sw.sw[20] * glob7s(&l, &PTL[3], input, sw),
            tinf,
            &mut held,
        );
        meso_tgn1[1] = PTM[8]
            * PMA[8][0]
            * (1.0 + sw.sw[18] * sw.sw[20] * glob7s(&l, &PMA[8], input, sw))
            * meso_tn1[4]
            * meso_tn1[4]
            / (libm::pow(PTM[4] * PTL[3][0], 2.0));
    } else {
        meso_tn1[1] = PTM[6] * PTL[0][0];
        meso_tn1[2] = PTM[2] * PTL[1][0];
        meso_tn1[3] = PTM[7] * PTL[2][0];
        meso_tn1[4] = PTM[4] * PTL[3][0];
        meso_tgn1[1] =
            PTM[8] * PMA[8][0] * meso_tn1[4] * meso_tn1[4] / (libm::pow(PTM[4] * PTL[3][0], 2.0));
    }

    // N2 variation factor at Zlb
    let g28 = sw.sw[21] * globe7(&mut l, &PD[2], input, sw);

    // Variation of turbopause height
    let zhf = PDL[1][24]
        * (1.0
            + sw.sw[5]
                * PDL[0][24]
                * libm::sin(dgtr * input.g_lat)
                * libm::cos(dr * (input.doy as f64 - PT[13])));
    output.t[0] = tinf;
    let xmm = PDM[2][4];
    let z = input.alt;

    // N2 density
    let db28 = PDM[2][0] * libm::exp(g28) * PD[2][0];
    let (d2, tzn) = densu(
        gsurf,
        re,
        z,
        db28,
        tinf,
        tlb,
        28.0,
        alpha[2],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        &mut held,
    );
    output.d[2] = d2;
    output.t[1] = tzn;
    let zh28 = PDM[2][2] * zhf;
    let zhm28 = PDM[2][3] * PDL[1][5];
    let xmd = 28.0 - xmm;
    let (b28, _tz) = densu(
        gsurf,
        re,
        zh28,
        db28,
        tinf,
        tlb,
        xmd,
        alpha[2] - 1.0,
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        &mut held,
    );
    // The fully-mixed N2 density, which [`gtd7`] divides by unconditionally to size its own
    // turbopause blend — but which is produced here only when mixing is switched on. The reference
    // hides the gap in a file-scope `static`: with `sw[15]` off it hands back whatever the
    // *previous* call to `gts7` left there, so its answer below `72.5 km` is a function of the
    // call history rather than of the inputs, and on the very first call it is `0` and the blend
    // divides by it. Threading the value out instead makes the gap visible, and the only defensible
    // filling is the one that says what "no mixing" means: with no mixed profile the diffusive
    // density *is* the reference density, and `gtd7`'s correction becomes exactly the identity.
    // Inert in every default configuration — `switches[15]` is `1` and `z` is `72.5 km` whenever
    // `gtd7` reads this — so the value below is overwritten before it can be used.
    let mut dm28 = output.d[2];
    if (sw.sw[15] != 0.0) && (z <= altl[2]) {
        let (dm28v, _tz) = densu(
            gsurf,
            re,
            z,
            b28,
            tinf,
            tlb,
            xmm,
            alpha[2],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        dm28 = dm28v;
        output.d[2] = dnet(output.d[2], dm28, zhm28, xmm, 28.0);
    }

    // He density
    let g4 = sw.sw[21] * globe7(&mut l, &PD[0], input, sw);
    let db04 = PDM[0][0] * libm::exp(g4) * PD[0][0];
    let (d0, tzn) = densu(
        gsurf,
        re,
        z,
        db04,
        tinf,
        tlb,
        4.0,
        alpha[0],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        &mut held,
    );
    output.d[0] = d0;
    output.t[1] = tzn;
    if (sw.sw[15] != 0.0) && (z < altl[0]) {
        let zh04 = PDM[0][2];
        let (b04, tzn) = densu(
            gsurf,
            re,
            zh04,
            db04,
            tinf,
            tlb,
            4.0 - xmm,
            alpha[0] - 1.0,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        output.t[1] = tzn;
        let (dm04, tzn) = densu(
            gsurf,
            re,
            z,
            b04,
            tinf,
            tlb,
            xmm,
            0.0,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        output.t[1] = tzn;
        let zhm04 = zhm28;
        output.d[0] = dnet(output.d[0], dm04, zhm04, xmm, 4.0);
        let rl = libm::log(b28 * PDM[0][1] / b04);
        let zc04 = PDM[0][4] * PDL[1][0];
        let hc04 = PDM[0][5] * PDL[1][1];
        output.d[0] = output.d[0] * ccor(z, rl, hc04, zc04);
    }

    // O density
    let g16 = sw.sw[21] * globe7(&mut l, &PD[1], input, sw);
    let db16 = PDM[1][0] * libm::exp(g16) * PD[1][0];
    let (d1, tzn) = densu(
        gsurf,
        re,
        z,
        db16,
        tinf,
        tlb,
        16.0,
        alpha[1],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        &mut held,
    );
    output.d[1] = d1;
    output.t[1] = tzn;
    if (sw.sw[15] != 0.0) && (z <= altl[1]) {
        let zh16 = PDM[1][2];
        let (b16, tzn) = densu(
            gsurf,
            re,
            zh16,
            db16,
            tinf,
            tlb,
            16.0 - xmm,
            alpha[1] - 1.0,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        output.t[1] = tzn;
        let (dm16, tzn) = densu(
            gsurf,
            re,
            z,
            b16,
            tinf,
            tlb,
            xmm,
            0.0,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        output.t[1] = tzn;
        let zhm16 = zhm28;
        output.d[1] = dnet(output.d[1], dm16, zhm16, xmm, 16.0);
        let rl = PDM[1][1] * PDL[1][16] * (1.0 + sw.sw[1] * PDL[0][23] * (input.f107a - 150.0));
        let hc16 = PDM[1][5] * PDL[1][3];
        let zc16 = PDM[1][4] * PDL[1][2];
        let hc216 = PDM[1][5] * PDL[1][4];
        output.d[1] = output.d[1] * ccor2(z, rl, hc16, zc16, hc216);
        let hcc16 = PDM[1][7] * PDL[1][13];
        let zcc16 = PDM[1][6] * PDL[1][12];
        let rc16 = PDM[1][3] * PDL[1][14];
        output.d[1] = output.d[1] * ccor(z, rc16, hcc16, zcc16);
    }

    // O2 density
    let g32 = sw.sw[21] * globe7(&mut l, &PD[4], input, sw);
    let db32 = PDM[3][0] * libm::exp(g32) * PD[4][0];
    let (d3, tzn) = densu(
        gsurf,
        re,
        z,
        db32,
        tinf,
        tlb,
        32.0,
        alpha[3],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        &mut held,
    );
    output.d[3] = d3;
    output.t[1] = tzn;
    if sw.sw[15] != 0.0 {
        if z <= altl[3] {
            let zh32 = PDM[3][2];
            let (b32, tzn) = densu(
                gsurf,
                re,
                zh32,
                db32,
                tinf,
                tlb,
                32.0 - xmm,
                alpha[3] - 1.0,
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut meso_tn1,
                &mut meso_tgn1,
                &mut held,
            );
            output.t[1] = tzn;
            let (dm32, tzn) = densu(
                gsurf,
                re,
                z,
                b32,
                tinf,
                tlb,
                xmm,
                0.0,
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut meso_tn1,
                &mut meso_tgn1,
                &mut held,
            );
            output.t[1] = tzn;
            let zhm32 = zhm28;
            output.d[3] = dnet(output.d[3], dm32, zhm32, xmm, 32.0);
            let rl = libm::log(b28 * PDM[3][1] / b32);
            let hc32 = PDM[3][5] * PDL[1][7];
            let zc32 = PDM[3][4] * PDL[1][6];
            output.d[3] = output.d[3] * ccor(z, rl, hc32, zc32);
        }
        let hcc32 = PDM[3][7] * PDL[1][22];
        let hcc232 = PDM[3][7] * PDL[0][22];
        let zcc32 = PDM[3][6] * PDL[1][21];
        let rc32 = PDM[3][3] * PDL[1][23] * (1.0 + sw.sw[1] * PDL[0][23] * (input.f107a - 150.0));
        output.d[3] = output.d[3] * ccor2(z, rc32, hcc32, zcc32, hcc232);
    }

    // Ar density
    let g40 = sw.sw[21] * globe7(&mut l, &PD[5], input, sw);
    let db40 = PDM[4][0] * libm::exp(g40) * PD[5][0];
    let (d4, tzn) = densu(
        gsurf,
        re,
        z,
        db40,
        tinf,
        tlb,
        40.0,
        alpha[4],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        &mut held,
    );
    output.d[4] = d4;
    output.t[1] = tzn;
    if (sw.sw[15] != 0.0) && (z <= altl[4]) {
        let zh40 = PDM[4][2];
        let (b40, tzn) = densu(
            gsurf,
            re,
            zh40,
            db40,
            tinf,
            tlb,
            40.0 - xmm,
            alpha[4] - 1.0,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        output.t[1] = tzn;
        let (dm40, tzn) = densu(
            gsurf,
            re,
            z,
            b40,
            tinf,
            tlb,
            xmm,
            0.0,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        output.t[1] = tzn;
        let zhm40 = zhm28;
        output.d[4] = dnet(output.d[4], dm40, zhm40, xmm, 40.0);
        let rl = libm::log(b28 * PDM[4][1] / b40);
        let hc40 = PDM[4][5] * PDL[1][9];
        let zc40 = PDM[4][4] * PDL[1][8];
        output.d[4] = output.d[4] * ccor(z, rl, hc40, zc40);
    }

    // Hydrogen density
    let g1 = sw.sw[21] * globe7(&mut l, &PD[6], input, sw);
    let db01 = PDM[5][0] * libm::exp(g1) * PD[6][0];
    let (d6, tzn) = densu(
        gsurf,
        re,
        z,
        db01,
        tinf,
        tlb,
        1.0,
        alpha[6],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        &mut held,
    );
    output.d[6] = d6;
    output.t[1] = tzn;
    if (sw.sw[15] != 0.0) && (z <= altl[6]) {
        let zh01 = PDM[5][2];
        let (b01, tzn) = densu(
            gsurf,
            re,
            zh01,
            db01,
            tinf,
            tlb,
            1.0 - xmm,
            alpha[6] - 1.0,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        output.t[1] = tzn;
        let (dm01, tzn) = densu(
            gsurf,
            re,
            z,
            b01,
            tinf,
            tlb,
            xmm,
            0.0,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        output.t[1] = tzn;
        let zhm01 = zhm28;
        output.d[6] = dnet(output.d[6], dm01, zhm01, xmm, 1.0);
        let rl = libm::log(b28 * PDM[5][1] * libm::sqrt(PDL[1][17] * PDL[1][17]) / b01);
        let hc01 = PDM[5][5] * PDL[1][11];
        let zc01 = PDM[5][4] * PDL[1][10];
        output.d[6] = output.d[6] * ccor(z, rl, hc01, zc01);
        let hcc01 = PDM[5][7] * PDL[1][19];
        let zcc01 = PDM[5][6] * PDL[1][18];
        let rc01 = PDM[5][3] * PDL[1][20];
        output.d[6] = output.d[6] * ccor(z, rc01, hcc01, zcc01);
    }

    // Atomic nitrogen density
    let g14 = sw.sw[21] * globe7(&mut l, &PD[7], input, sw);
    let db14 = PDM[6][0] * libm::exp(g14) * PD[7][0];
    let (d7, tzn) = densu(
        gsurf,
        re,
        z,
        db14,
        tinf,
        tlb,
        14.0,
        alpha[7],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        &mut held,
    );
    output.d[7] = d7;
    output.t[1] = tzn;
    if (sw.sw[15] != 0.0) && (z <= altl[7]) {
        let zh14 = PDM[6][2];
        let (b14, tzn) = densu(
            gsurf,
            re,
            zh14,
            db14,
            tinf,
            tlb,
            14.0 - xmm,
            alpha[7] - 1.0,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        output.t[1] = tzn;
        let (dm14, tzn) = densu(
            gsurf,
            re,
            z,
            b14,
            tinf,
            tlb,
            xmm,
            0.0,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            &mut held,
        );
        output.t[1] = tzn;
        let zhm14 = zhm28;
        output.d[7] = dnet(output.d[7], dm14, zhm14, xmm, 14.0);
        let rl = libm::log(b28 * PDM[6][1] * libm::sqrt(PDL[0][2] * PDL[0][2]) / b14);
        let hc14 = PDM[6][5] * PDL[0][1];
        let zc14 = PDM[6][4] * PDL[0][0];
        output.d[7] = output.d[7] * ccor(z, rl, hc14, zc14);
        let hcc14 = PDM[6][7] * PDL[0][4];
        let zcc14 = PDM[6][6] * PDL[0][3];
        let rc14 = PDM[6][3] * PDL[0][5];
        output.d[7] = output.d[7] * ccor(z, rc14, hcc14, zcc14);
    }

    // Anomalous oxygen density
    let g16h = sw.sw[21] * globe7(&mut l, &PD[8], input, sw);
    let db16h = PDM[7][0] * libm::exp(g16h) * PD[8][0];
    let tho = PDM[7][9] * PDL[0][6];
    // Held on its own flag, not the reported one: `tho` is a fitted scale parameter, so an
    // interpolant above it says nothing about the atmosphere — see [`Nrlmsise00Output::out_of_range`].
    let mut anomalous_held = false;
    let (dd, tzn) = densu(
        gsurf,
        re,
        z,
        db16h,
        tho,
        tho,
        16.0,
        alpha[8],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        &mut anomalous_held,
    );
    output.t[1] = tzn;
    let zsht = PDM[7][5];
    let zmho = PDM[7][4];
    let zsho = scalh(gsurf, re, zmho, 16.0, tho);
    output.d[8] = dd * libm::exp(-zsht / zsho * (libm::exp(-(z - zmho) / zsht) - 1.0));

    // total mass density
    output.d[5] = 1.66E-24
        * (4.0 * output.d[0]
            + 16.0 * output.d[1]
            + 28.0 * output.d[2]
            + 32.0 * output.d[3]
            + 40.0 * output.d[4]
            + output.d[6]
            + 14.0 * output.d[7]);

    // temperature at altitude
    let z = libm::sqrt(input.alt * input.alt);
    let (_ddum, tzn) = densu(
        gsurf,
        re,
        z,
        1.0,
        tinf,
        tlb,
        0.0,
        0.0,
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        &mut held,
    );
    output.t[1] = tzn;
    if sw.sw[0] != 0.0 {
        for i in 0..9 {
            output.d[i] *= 1.0E6;
        }
        output.d[5] /= 1000.0;
    }

    output.out_of_range = held;
    Gts7Result {
        output,
        dm28,
        tn1_4: meso_tn1[4],
        tgn1_1: meso_tgn1[1],
        lpoly: l,
    }
}

/// Neutral atmosphere from the surface to the lower exosphere.
///
/// `output.d[5]` is the mass density of He, O, N2, O2, Ar, H and N (it excludes anomalous oxygen;
/// use [`gtd7d`] for the drag-effective total).
pub fn gtd7(input: &Nrlmsise00Input, flags: &Nrlmsise00Flags) -> Nrlmsise00Output {
    let sw = Sw::from_flags(flags);
    let mn3 = ZN3.len();
    let zn3 = ZN3;
    let mn2 = ZN2.len();
    let zn2 = ZN2;
    let zmix = 62.5;

    // Latitude variation of gravity (none for sw[2] == 0)
    let xlat = if sw.sw[2] == 0.0 { 45.0 } else { input.g_lat };
    let (gsurf, re) = glatf(xlat);

    let xmm = PDM[2][4];

    // Thermosphere / mesosphere (above zn2[0])
    let altt = if input.alt > zn2[0] {
        input.alt
    } else {
        zn2[0]
    };
    let mut sinput = *input;
    sinput.alt = altt;
    let gts = gts7_impl(&sinput, &sw, gsurf, re);
    let soutput = gts.output;

    let dm28m = if sw.sw[0] != 0.0 {
        gts.dm28 * 1.0E6
    } else {
        gts.dm28
    };

    // Above the top mesospheric node the thermosphere's own evaluation is the whole answer, this
    // one included; below it, the two `densm` calls that follow can add to it.
    let mut held = soutput.out_of_range;
    let mut output = Nrlmsise00Output {
        d: [0.0; 9],
        t: [soutput.t[0], soutput.t[1]],
        out_of_range: held,
        input_out_of_domain: input_out_of_domain(input),
    };
    if input.alt >= zn2[0] {
        output.d = soutput.d;
        return output;
    }

    // Lower mesosphere / upper stratosphere (between zn3[0] and zn2[0])
    let mut meso_tn2 = [0.0f64; 4];
    let mut meso_tn3 = [0.0f64; 5];
    let mut meso_tgn2 = [0.0f64; 2];
    let mut meso_tgn3 = [0.0f64; 2];
    meso_tgn2[0] = gts.tgn1_1;
    meso_tn2[0] = gts.tn1_4;
    meso_tn2[1] = temperature_node(
        PMA[0][0] * PAVGM[0],
        sw.sw[20] * glob7s(&gts.lpoly, &PMA[0], input, &sw),
        soutput.t[0],
        &mut held,
    );
    meso_tn2[2] = temperature_node(
        PMA[1][0] * PAVGM[1],
        sw.sw[20] * glob7s(&gts.lpoly, &PMA[1], input, &sw),
        soutput.t[0],
        &mut held,
    );
    meso_tn2[3] = temperature_node(
        PMA[2][0] * PAVGM[2],
        sw.sw[20] * sw.sw[22] * glob7s(&gts.lpoly, &PMA[2], input, &sw),
        soutput.t[0],
        &mut held,
    );
    meso_tgn2[1] = PAVGM[8]
        * PMA[9][0]
        * (1.0 + sw.sw[20] * sw.sw[22] * glob7s(&gts.lpoly, &PMA[9], input, &sw))
        * meso_tn2[3]
        * meso_tn2[3]
        / (libm::pow(PMA[2][0] * PAVGM[2], 2.0));
    meso_tn3[0] = meso_tn2[3];

    if input.alt <= zn3[0] {
        // Lower stratosphere and troposphere (below zn3[0])
        meso_tgn3[0] = meso_tgn2[1];
        meso_tn3[1] = temperature_node(
            PMA[3][0] * PAVGM[3],
            sw.sw[22] * glob7s(&gts.lpoly, &PMA[3], input, &sw),
            soutput.t[0],
            &mut held,
        );
        meso_tn3[2] = temperature_node(
            PMA[4][0] * PAVGM[4],
            sw.sw[22] * glob7s(&gts.lpoly, &PMA[4], input, &sw),
            soutput.t[0],
            &mut held,
        );
        meso_tn3[3] = temperature_node(
            PMA[5][0] * PAVGM[5],
            sw.sw[22] * glob7s(&gts.lpoly, &PMA[5], input, &sw),
            soutput.t[0],
            &mut held,
        );
        meso_tn3[4] = temperature_node(
            PMA[6][0] * PAVGM[6],
            sw.sw[22] * glob7s(&gts.lpoly, &PMA[6], input, &sw),
            soutput.t[0],
            &mut held,
        );
        meso_tgn3[1] = PMA[7][0]
            * PAVGM[7]
            * (1.0 + sw.sw[22] * glob7s(&gts.lpoly, &PMA[7], input, &sw))
            * meso_tn3[4]
            * meso_tn3[4]
            / (libm::pow(PMA[6][0] * PAVGM[6], 2.0));
    }

    // Linear transition to full mixing below zn2[0]
    let mut dmc = 0.0;
    if input.alt > zmix {
        dmc = 1.0 - (zn2[0] - input.alt) / (zn2[0] - zmix);
    }
    let dz28 = soutput.d[2];

    // N2 density
    let mut dmr = soutput.d[2] / dm28m - 1.0;
    let (d2, _tz) = densm(
        gsurf,
        re,
        input.alt,
        dm28m,
        xmm,
        mn3,
        &zn3,
        &meso_tn3,
        &meso_tgn3,
        mn2,
        &zn2,
        &meso_tn2,
        &meso_tgn2,
        soutput.t[0],
        &mut held,
    );
    output.d[2] = d2 * (1.0 + dmr * dmc);

    // He density
    dmr = soutput.d[0] / (dz28 * PDM[0][1]) - 1.0;
    output.d[0] = output.d[2] * PDM[0][1] * (1.0 + dmr * dmc);

    // O density
    output.d[1] = 0.0;
    output.d[8] = 0.0;

    // O2 density
    dmr = soutput.d[3] / (dz28 * PDM[3][1]) - 1.0;
    output.d[3] = output.d[2] * PDM[3][1] * (1.0 + dmr * dmc);

    // Ar density
    dmr = soutput.d[4] / (dz28 * PDM[4][1]) - 1.0;
    output.d[4] = output.d[2] * PDM[4][1] * (1.0 + dmr * dmc);

    // H density
    output.d[6] = 0.0;
    // Atomic nitrogen density
    output.d[7] = 0.0;

    // Total mass density
    output.d[5] = 1.66E-24
        * (4.0 * output.d[0]
            + 16.0 * output.d[1]
            + 28.0 * output.d[2]
            + 32.0 * output.d[3]
            + 40.0 * output.d[4]
            + output.d[6]
            + 14.0 * output.d[7]);
    if sw.sw[0] != 0.0 {
        output.d[5] /= 1000.0;
    }

    // Temperature at altitude
    let (_dd, tz) = densm(
        gsurf,
        re,
        input.alt,
        1.0,
        0.0,
        mn3,
        &zn3,
        &meso_tn3,
        &meso_tgn3,
        mn2,
        &zn2,
        &meso_tn2,
        &meso_tgn2,
        soutput.t[0],
        &mut held,
    );
    output.t[1] = tz;
    output.out_of_range = held;

    output
}

/// Like [`gtd7`] but `output.d[5]` is the effective total mass density for drag, which includes
/// anomalous oxygen (`d[8]`) and so can differ above ~500 km.
pub fn gtd7d(input: &Nrlmsise00Input, flags: &Nrlmsise00Flags) -> Nrlmsise00Output {
    let sw = Sw::from_flags(flags);
    let mut output = gtd7(input, flags);
    output.d[5] = 1.66E-24
        * (4.0 * output.d[0]
            + 16.0 * output.d[1]
            + 28.0 * output.d[2]
            + 32.0 * output.d[3]
            + 40.0 * output.d[4]
            + output.d[6]
            + 14.0 * output.d[7]
            + 16.0 * output.d[8]);
    if sw.sw[0] != 0.0 {
        output.d[5] /= 1000.0;
    }
    output
}

/// Thermospheric portion of NRLMSISE-00 (intended for `alt > 72.5 km`). Sets up the latitude
/// gravity variation itself so it is safe to call standalone.
pub fn gts7(input: &Nrlmsise00Input, flags: &Nrlmsise00Flags) -> Nrlmsise00Output {
    let sw = Sw::from_flags(flags);
    let xlat = if sw.sw[2] == 0.0 { 45.0 } else { input.g_lat };
    let (gsurf, re) = glatf(xlat);
    let mut output = gts7_impl(input, &sw, gsurf, re).output;
    output.input_out_of_domain = input_out_of_domain(input);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use spacelab_testkit::nan_max;

    /// The parameter tables *are* the model: `gtd7` is a fixed set of expansions whose entire
    /// content is these 3340 numbers, transcribed from the authors' `nrlmsise-00_data.c`. The
    /// 17-vector golden below exercises the code path, not the table — most of these coefficients
    /// move the answer far below the agreement it asserts, and `cargo mutants` finds 506 sign
    /// deletions in the table that it cannot see. The seal is what makes them visible; its constant
    /// comes from `derivations/table_seals.py`, which reads the same file with Python's float
    /// parser, so the two sides are independent transcriptions rather than one checked against
    /// itself.
    #[test]
    fn the_parameter_tables_are_the_ones_the_generator_wrote() {
        let flat: Vec<f64> = PT
            .iter()
            .chain(PD.iter().flatten())
            .chain(PS.iter())
            .chain(PDL.iter().flatten())
            .chain(PTM.iter())
            .chain(PDM.iter().flatten())
            .chain(PTL.iter().flatten())
            .chain(PMA.iter().flatten())
            .chain(super::tables::SAM.iter())
            .chain(PAVGM.iter())
            .copied()
            .collect();
        assert_eq!(flat.len(), 3340, "NRLMSISE-00 parameters");
        let seal = spacelab_testkit::fnv1a_f64(flat);
        assert_eq!(seal, 0xacf8_5dec_015a_5d7c, "NRLMSISE-00 parameters moved");
    }

    /// The 17 canonical reference outputs (He, O, N2, O2, Ar, total-mass-density, H, N,
    /// anomalous-O, exospheric-T, T-at-alt) produced by the authors' C `test_gtd7`.
    const GOLDEN: [[f64; 11]; 17] = [
        [
            6.665177E+05,
            1.138806E+08,
            1.998211E+07,
            4.022764E+05,
            3.557465E+03,
            4.074714E-15,
            3.475312E+04,
            4.095913E+06,
            2.667273E+04,
            1.250540E+03,
            1.241416E+03,
        ],
        [
            3.407293E+06,
            1.586333E+08,
            1.391117E+07,
            3.262560E+05,
            1.559618E+03,
            5.001846E-15,
            4.854208E+04,
            4.380967E+06,
            6.956682E+03,
            1.166754E+03,
            1.161710E+03,
        ],
        [
            1.123767E+05,
            6.934130E+04,
            4.247105E+01,
            1.322750E-01,
            2.618848E-05,
            2.756772E-18,
            2.016750E+04,
            5.741256E+03,
            2.374394E+04,
            1.239892E+03,
            1.239891E+03,
        ],
        [
            5.411554E+07,
            1.918893E+11,
            6.115826E+12,
            1.225201E+12,
            6.023212E+10,
            3.584426E-10,
            1.059880E+07,
            2.615737E+05,
            2.819879E-42,
            1.027318E+03,
            2.068878E+02,
        ],
        [
            1.851122E+06,
            1.476555E+08,
            1.579356E+07,
            2.633795E+05,
            1.588781E+03,
            4.809630E-15,
            5.816167E+04,
            5.478984E+06,
            1.264446E+03,
            1.212396E+03,
            1.208135E+03,
        ],
        [
            8.673095E+05,
            1.278862E+08,
            1.822577E+07,
            2.922214E+05,
            2.402962E+03,
            4.355866E-15,
            3.686389E+04,
            3.897276E+06,
            2.667273E+04,
            1.220146E+03,
            1.212712E+03,
        ],
        [
            5.776251E+05,
            6.979139E+07,
            1.236814E+07,
            2.492868E+05,
            1.405739E+03,
            2.470651E-15,
            5.291986E+04,
            1.069814E+06,
            2.667273E+04,
            1.116385E+03,
            1.112999E+03,
        ],
        [
            3.740304E+05,
            4.782720E+07,
            5.240380E+06,
            1.759875E+05,
            5.501649E+02,
            1.571889E-15,
            8.896776E+04,
            1.979741E+06,
            9.121815E+03,
            1.031247E+03,
            1.024848E+03,
        ],
        [
            6.748339E+05,
            1.245315E+08,
            2.369010E+07,
            4.911583E+05,
            4.578781E+03,
            4.564420E-15,
            3.244595E+04,
            5.370833E+06,
            2.667273E+04,
            1.306052E+03,
            1.293374E+03,
        ],
        [
            5.528601E+05,
            1.198041E+08,
            3.495798E+07,
            9.339618E+05,
            1.096255E+04,
            4.974543E-15,
            2.686428E+04,
            4.889974E+06,
            2.805445E+04,
            1.361868E+03,
            1.347389E+03,
        ],
        [
            1.375488E+14,
            0.000000E+00,
            2.049687E+19,
            5.498695E+18,
            2.451733E+17,
            1.261066E-03,
            0.000000E+00,
            0.000000E+00,
            0.000000E+00,
            1.027318E+03,
            2.814648E+02,
        ],
        [
            4.427443E+13,
            0.000000E+00,
            6.597567E+18,
            1.769929E+18,
            7.891680E+16,
            4.059139E-04,
            0.000000E+00,
            0.000000E+00,
            0.000000E+00,
            1.027318E+03,
            2.274180E+02,
        ],
        [
            2.127829E+12,
            0.000000E+00,
            3.170791E+17,
            8.506280E+16,
            3.792741E+15,
            1.950822E-05,
            0.000000E+00,
            0.000000E+00,
            0.000000E+00,
            1.027318E+03,
            2.374389E+02,
        ],
        [
            1.412184E+11,
            0.000000E+00,
            2.104370E+16,
            5.645392E+15,
            2.517142E+14,
            1.294709E-06,
            0.000000E+00,
            0.000000E+00,
            0.000000E+00,
            1.027318E+03,
            2.795551E+02,
        ],
        [
            1.254884E+10,
            0.000000E+00,
            1.874533E+15,
            4.923051E+14,
            2.239685E+13,
            1.147668E-07,
            0.000000E+00,
            0.000000E+00,
            0.000000E+00,
            1.027318E+03,
            2.190732E+02,
        ],
        [
            5.196477E+05,
            1.274494E+08,
            4.850450E+07,
            1.720838E+06,
            2.354487E+04,
            5.881940E-15,
            2.500078E+04,
            6.279210E+06,
            2.667273E+04,
            1.426412E+03,
            1.408608E+03,
        ],
        [
            4.260860E+07,
            1.241342E+11,
            4.929562E+12,
            1.048407E+12,
            4.993465E+10,
            2.914304E-10,
            8.831229E+06,
            2.252516E+05,
            2.415246E-42,
            1.027318E+03,
            1.934071E+02,
        ],
    ];

    fn base_input() -> Nrlmsise00Input {
        Nrlmsise00Input {
            year: 0,
            doy: 172,
            sec: 29000.0,
            alt: 400.0,
            g_lat: 60.0,
            g_long: -70.0,
            lst: 16.0,
            f107a: 150.0,
            f107: 150.0,
            ap: 4.0,
            ap_array: None,
        }
    }

    /// One sweep point: every input the model reads, verbatim, and the twelve numbers the reference
    /// C returned for it.
    ///
    /// The switch array is carried in full rather than as a named preset. A preset would have to be
    /// expanded back into 24 values by code living in this file, and that expansion is exactly the
    /// kind of untested logic a sweep exists to avoid — a case is its inputs, not a recipe for them.
    struct Case {
        doy: i32,
        sec: f64,
        alt: f64,
        g_lat: f64,
        g_long: f64,
        lst: f64,
        f107a: f64,
        f107: f64,
        ap: f64,
        /// Always supplied, whatever `switches[9]` says: the reference C takes it unconditionally,
        /// and a case where it is present but ignored is worth sweeping too.
        ap_array: [f64; 7],
        switches: [i32; 24],
        /// `gtd7`'s nine densities and two temperatures, then `gtd7d`'s total mass density.
        out: [f64; 12],
    }

    impl Case {
        fn input(&self) -> Nrlmsise00Input {
            Nrlmsise00Input {
                year: 0,
                doy: self.doy,
                sec: self.sec,
                alt: self.alt,
                g_lat: self.g_lat,
                g_long: self.g_long,
                lst: self.lst,
                f107a: self.f107a,
                f107: self.f107,
                ap: self.ap,
                ap_array: Some(self.ap_array),
            }
        }

        /// The twelve outputs, and whether either entry point held a temperature to a bound.
        fn evaluate(&self) -> ([f64; 12], bool) {
            let (input, flags) = (
                self.input(),
                Nrlmsise00Flags {
                    switches: self.switches,
                },
            );
            let a = gtd7(&input, &flags);
            let d = gtd7d(&input, &flags);
            (
                [
                    a.d[0], a.d[1], a.d[2], a.d[3], a.d[4], a.d[5], a.d[6], a.d[7], a.d[8], a.t[0],
                    a.t[1], d.d[5],
                ],
                a.out_of_range || d.out_of_range,
            )
        }
    }

    /// How far one output is from what the reference C returned for it.
    ///
    /// The near-zero slots are exact zeros below `72.5 km` and denormal-scale anomalous oxygen
    /// elsewhere; those are held absolutely, everything else relatively.
    fn relative_miss(got: f64, want: f64) -> f64 {
        if libm::fabs(want) < 1.0e-30 {
            libm::fabs(got)
        } else {
            libm::fabs((got - want) / want)
        }
    }

    /// 496 sweep points against the reference C at full `f64` precision, through both entry points.
    ///
    /// The published verification set — the seventeen cases below — is printed by the model's own
    /// `nrlmsise-00_test.c` to **seven** significant digits, so no assertion against it can be
    /// tighter than `1e-5` relative. Two IEEE-754 implementations of the same algorithm agree four
    /// orders closer than that, and everything in between is invisible: a `globe7` term worth `1e-6`
    /// of the answer can be negated without one reference digit moving.
    ///
    /// `derivations/nrlmsise00_vectors.py` closes that gap by asking the same C — Brodowski's
    /// port, a comparator (it carries no licence grant; the coefficient tables come from the NRL
    /// FORTRAN, not from it) — for `%.17e` over a sweep built to move every term:
    ///
    /// * **Altitude is stratified over the model's four regimes** rather than drawn uniformly. This
    ///   is three models stitched together — `densm`/`glob7s` below `72.5 km`, the mesospheric spline
    ///   blend to `za`, `gts7`'s diffusive equilibrium above — and a uniform draw over `0..1000 km`
    ///   would leave the spline machinery cold.
    /// * **The other eight inputs are a Halton sequence** on the first eight primes: deterministic,
    ///   and — unlike the published set, which varies one coordinate at a time off a single base
    ///   point — it moves them *together*, which is the only way a cross term separates from the main
    ///   effects it multiplies.
    /// * **`lst` breaks away from `sec`/`g_long` on every fourth case.** It is an independent input
    ///   to the model, and holding it consistent hides which terms read which.
    /// * **The switch array is swept, including switches turned _off_.** A `0` is the setting that
    ///   matters most here: the internal `swc[i]` weights multiply whole groups of terms, and while
    ///   every one of them is `1`, multiplying by it and dividing by it are the same operation.
    ///   Every switch is driven to `0` in some case, half of those also on the `ap_array` branch.
    /// * **The last twenty-five cases turn on exactly one switch each**, everything else off — the
    ///   twenty-four `switches[i]`, then the `ap_array` branch alone. Drawing switches independently
    ///   almost never produces the combination a guarded group needs to be *observable*: `sw[i]` on
    ///   to reach the term at all, and the `swc[j]` weighting it off so that multiplying by that
    ///   weight and dividing by it stop agreeing. One-hot is that combination for every group at
    ///   once. `switches[0]`, the output-units flag, is only ever exercised here.
    /// * **Then every one of the 253 unordered switch pairs, turned off together**, everything else
    ///   on. One-hot cannot reach a group that is guarded over *several* switches at once —
    ///   `if !(sw[7] == 0 && sw[8] == 0 && sw[14] == 0)` needs one of the three on to enter and a
    ///   downstream consumer gate still on to be read, which is two switches off, not twenty-three.
    ///   Leave-two-out is the smallest draw that separates those, and it is exhaustive, so no
    ///   pairwise guard can hide behind a sampling accident.
    /// * **Then every triple drawn from the six switches the pairs showed to interact** — `5`, `10`,
    ///   `15`, `16`, `19` and `21`, the ones whose pairs contradicted the temperature profile.
    ///   Exhaustive pairs still cannot separate a guard written over three switches, and leave-three-out
    ///   over all twenty-four is `2024` cases for a return nothing suggests is there. Restricting the
    ///   triples to the switches that have already earned the suspicion is strength three where it is
    ///   warranted and strength two elsewhere.
    /// * **Finally four families a stratified draw cannot produce.** Every altitude at which the
    ///   model changes its mind, landing *on* the node rather than near it — including `za`, which
    ///   is the trap: `zn1` is written `{120, 110, 100, 90, 72.5}` and its first entry is then
    ///   overwritten with the fitted `PDL[1][15] = 123.435`, so a case at `120 km` tests a boundary
    ///   that is not there and the one that is stays untested. Then the cgs output flag, which the
    ///   port fixes at `0`, at one altitude per regime. Then the `ap_array` branch with one switch
    ///   off at a time — the mixed arms reach that branch often but only ever with every `swc` at
    ///   `1`, and `sw[9] = -1` has to stay on for the branch to be entered at all, which is what
    ///   one-hot cannot do. Then switch state `2`, "main effect off, cross terms on", the only
    ///   setting under which `sw[i]` and `swc[i]` differ.
    ///
    /// One combination is deliberately *not* asked: mixing off (`switches[15] = 0`) below `72.5 km`,
    /// where the reference reads a file-scope `static` its previous call wrote — see
    /// [`switching_off_the_turbopause_still_leaves_a_density_below_it`], which pins what this port
    /// does there instead.
    ///
    /// This is a comparator kept as a tripwire (`plans/17` D-22): `globe7`'s and `glob7s`'s
    /// harmonic expansions have no math gate of their own yet, and this is what would see a change
    /// in them. A failure means "check the mathematics", not "match the C again".
    ///
    /// `gtd7d` is here because nothing else calls it: it is what
    /// [`super::atmosphere::Nrlmsise00Atmosphere`] uses for drag, and the published cases exercise
    /// `gtd7` alone.
    ///
    /// A case that reports [`Nrlmsise00Output::out_of_range`] is checked for the invariants the
    /// bound exists to enforce rather than against the reference, because on those the reference is
    /// what this port deliberately does not reproduce. Switching a group off pins one part of the
    /// temperature profile while leaving the rest perturbed, and the profile then contradicts
    /// itself: with `switches[16] = 0` the exospheric temperature stays at its unperturbed
    /// `1027.3 K` while the mesospheric nodes keep their variation, so a node outruns `T∞` and is
    /// held. Three of the 496 land there — the pairs `(5, 15)`, `(10, 16)` and `(19, 21)`, the first
    /// of which is below `72.5 km` and so has its `15` restored by the generator. Those three are
    /// pinned as a *set*, with the switches each one turns off and the fact that each really does
    /// depart from the reference by more than `BAR`: a count alone stays green when one expected
    /// case vanishes and an unexpected one appears, and without the departure the bound would be
    /// credited for cases where it changed nothing. Every triple that contains one of those pairs
    /// stays inside `BAR`, so the contradiction is the pair's and not either switch's. The other
    /// `493` agree with the reference to `BAR`.
    #[test]
    fn the_sweep_matches_the_reference_c_at_full_precision() {
        let mut worst = 0.0f64;
        let (mut worst_case, mut worst_field) = (0usize, 0usize);
        let mut bounded = std::vec::Vec::new();
        for (case, spec) in SWEEP.iter().enumerate() {
            let (got, held) = spec.evaluate();
            if held {
                let off: std::vec::Vec<usize> =
                    (1..24).filter(|&i| spec.switches[i] == 0).collect();
                let departure = (0..12)
                    .map(|field| relative_miss(got[field], spec.out[field]))
                    .fold(0.0f64, nan_max);
                bounded.push((case, off, departure > BAR));
                assert!(
                    got[10] > 0.0 && got[10] <= got[9],
                    "case {case}: bounded to t = {:?}",
                    &got[9..11]
                );
                for (field, g) in got.iter().enumerate() {
                    assert!(
                        g.is_finite() && *g >= 0.0,
                        "case {case} field {field}: bounded to {g:e}"
                    );
                }
                continue;
            }
            for field in 0..12 {
                let miss = relative_miss(got[field], spec.out[field]);
                // `miss > worst` alone would drop a NaN, because `NaN > x` is false — and a NaN is
                // precisely what an off switch produces once a `swc[i]` weight is divided by
                // instead of multiplied: `1/0` is an infinity, `|sw| · ∞` with the main effect off
                // is a NaN, and the reduction would then report perfect agreement. Admitting it
                // makes the running worst NaN, which no later value can lower.
                if miss.is_nan() || miss > worst {
                    worst = miss;
                    worst_case = case;
                    worst_field = field;
                }
            }
        }
        assert!(
            worst < BAR,
            "case {worst_case} field {worst_field}: {worst:.3e} from the reference C"
        );
        // Pinned as the exact set, not as a count: a count stays green if one expected case
        // disappears and an unexpected one takes its place. The `true` is the departure from the
        // reference — without it the bound would be credited for cases where it changed nothing.
        assert_eq!(
            bounded,
            std::vec![
                (244usize, std::vec![5usize], true),
                (320, std::vec![10, 16], true),
                (397, std::vec![19, 21], true),
            ],
            "cases answered by the bound rather than the fit"
        );
    }

    /// The agreement floor, measured rather than allowed for. It is the difference between two
    /// `exp`/`pow`/`log` implementations compounded through the whole model, and the worst over
    /// the `493 x 12` compared slots is `6.960e-15` — about thirty ulps, at case 234 field 8
    /// (anomalous oxygen, the most cancellation-prone slot). `libm` is pure Rust and bit-identical
    /// across platforms, so
    /// that floor is a constant, not a sample. The published vectors' `1e-5` is nine orders looser.
    const BAR: f64 = 1.0e-14;

    /// Above the top mesospheric node the two public entry points are the same function.
    ///
    /// [`gtd7`] adds a mixing layer to [`gts7`], and above `zn2[0]` there is no layer to add: it
    /// copies the thermospheric answer out and returns. That makes one the oracle for the other,
    /// bit for bit, and it is the only oracle `gts7` has — the reference sweep drives `gtd7` and
    /// `gtd7d`, and nothing else compares what `gts7` returns against a number. The gap that showed
    /// is the latitude switch each one applies before computing surface gravity: with `switches[2]`
    /// off the model uses `45°` in place of the caller's latitude, and until this test the two
    /// copies of that line were free to disagree.
    #[test]
    fn above_the_mixing_layer_gts7_is_gtd7() {
        for alt in [72.5, 90.0, 120.0, 200.0, 400.0, 800.0] {
            for lat in [-70.0, -12.5, 0.0, 45.0, 70.0] {
                for latitude_gravity in [0, 1] {
                    let mut flags = Nrlmsise00Flags::default();
                    flags.switches[2] = latitude_gravity;
                    let mut input = base_input();
                    (input.alt, input.g_lat) = (alt, lat);
                    let (a, b) = (gts7(&input, &flags), gtd7(&input, &flags));
                    assert_eq!(a.d, b.d, "{alt} km, {lat}°, sw2 {latitude_gravity}");
                    assert_eq!(a.t, b.t, "{alt} km, {lat}°, sw2 {latitude_gravity}");
                }
            }
        }
    }

    /// The reported total mass density is the molar sum of the species reported beside it.
    ///
    /// That is the definition of `d[5]`, and the reference comparison cannot enforce it. Below the
    /// turbopause [`gtd7`] zeroes atomic oxygen, hydrogen, nitrogen and anomalous oxygen, so the
    /// terms carrying them contribute nothing and their arithmetic is free; above it every species
    /// is present, but the sum is of order `10^17` and the molecular masses are of order `10`, so
    /// replacing `16 · n(O)` by `16 + n(O)` moves the total by a few parts in `10^16` — below the
    /// `1e-14` the sweep compares at, and therefore invisible to it. Written out with the masses in
    /// amu, the identity is exact in both regimes and catches both.
    #[test]
    fn the_total_mass_density_is_the_molar_sum_of_the_species() {
        const AMU: f64 = 1.66E-24;
        let flags = Nrlmsise00Flags::default();
        // The two altitudes just under the mixing boundary are where the identity has teeth: the
        // sum is smallest there, so its unit in the last place is smallest, and a wrong term that
        // is lost in rounding lower down still moves a bit here.
        for alt in [
            0.0, 30.0, 62.5, 70.0, 72.4, 72.5, 95.0, 120.0, 250.0, 400.0, 700.0, 1000.0,
        ] {
            let mut input = base_input();
            input.alt = alt;
            for (name, o) in [
                ("gtd7", gtd7(&input, &flags)),
                ("gts7", gts7(&input, &flags)),
            ] {
                let want = AMU
                    * (4.0 * o.d[0]
                        + 16.0 * o.d[1]
                        + 28.0 * o.d[2]
                        + 32.0 * o.d[3]
                        + 40.0 * o.d[4]
                        + o.d[6]
                        + 14.0 * o.d[7]);
                assert_eq!(o.d[5], want, "{name} at {alt} km");
            }
            // …and `gtd7d` carries one more species into the same sum, which is the only thing
            // separating the two entry points.
            let d = gtd7d(&input, &flags);
            let want = AMU
                * (4.0 * d.d[0]
                    + 16.0 * d.d[1]
                    + 28.0 * d.d[2]
                    + 32.0 * d.d[3]
                    + 40.0 * d.d[4]
                    + d.d[6]
                    + 14.0 * d.d[7]
                    + 16.0 * d.d[8]);
            assert_eq!(d.d[5], want, "gtd7d at {alt} km");
        }
    }

    /// `dnet`'s blend, at the two densities the reference only prints a diagnostic about.
    ///
    /// The blend is a power mean: `n^a = n_diff^a + n_mix^a` with `a = zhm/(xmm − xm)`. That
    /// identity is the oracle for the ordinary case — it is the definition the turbopause join is
    /// derived from, not a second run of the code's own steps.
    ///
    /// What no test reached is the degenerate half. With either input zero the power mean is
    /// undefined (`0^a` is `0` or `∞` depending on the sign of `a`), and the model answers with the
    /// other input — `1.0` when both vanish. Every mutant of the three comparisons that select that
    /// half parts from the baseline **only where `a < 0`**: there the `log(0/dd)` and `log(dm/0)`
    /// limits the fall-through would take run to the opposite infinity and the `ylog` saturation
    /// returns the *other* density. `a < 0` is exactly `xm > xmm`, which is O₂ and argon — two of
    /// the seven species the model actually joins — so each case is run for those as well as for N₂
    /// and helium.
    #[test]
    fn the_turbopause_blend_answers_at_a_vanishing_density() {
        let xmm = PDM[2][4];
        let zhm = 28.0;
        for xm in [4.0, 28.0, 32.0, 40.0] {
            let a = zhm / (xmm - xm);
            let blend = |dd, dm| dnet(dd, dm, zhm, xmm, xm);
            assert_eq!(blend(0.0, 0.0), 1.0, "xm {xm}: both densities vanish");
            assert_eq!(
                blend(5.0e6, 0.0),
                5.0e6,
                "xm {xm}: the mixed density vanishes"
            );
            assert_eq!(
                blend(0.0, 7.0e6),
                7.0e6,
                "xm {xm}: the diffusive density vanishes"
            );
            // Ratios stay near one so that `|a·ln(dm/dd)| < 10` and neither saturation branch —
            // which is separately covered — takes over from the mean.
            for (dd, dm) in [(1.0e6, 1.2e6), (1.2e6, 1.0e6), (2.0e6, 2.0e6)] {
                let got = blend(dd, dm);
                let want = libm::pow(libm::pow(dd, a) + libm::pow(dm, a), 1.0 / a);
                assert!(
                    libm::fabs(got / want - 1.0) < 1.0e-13,
                    "xm {xm}: {got:e} is not the power mean {want:e}"
                );
            }
        }
    }
    /// One edit to a set of inputs, named.
    type Adjust = fn(&mut Nrlmsise00Input);

    /// Every way a caller can leave the model's domain is reported, and the two edges that look
    /// like mistakes but are not stay silent.
    ///
    /// [`Nrlmsise00Input`] has public fields and no constructor, so this flag is the only place the
    /// contract is written as code. Two cases are why it exists: a NaN longitude fails every
    /// comparison and so takes the *same* branch as the `-1000` sentinel, silently dropping the
    /// longitude terms; and a latitude far outside `±90°` is the one input that can drive `globe7`'s
    /// `exp1` saturation, the branch `.cargo/mutants.toml` calls unreachable — a claim that is only
    /// true while this flag is false.
    #[test]
    fn the_input_domain_is_reported_at_every_edge() {
        let flags = Nrlmsise00Flags::default();
        let clean = base_input();
        for (name, o) in [
            ("gtd7", gtd7(&clean, &flags)),
            ("gtd7d", gtd7d(&clean, &flags)),
            ("gts7", gts7(&clean, &flags)),
        ] {
            assert!(!o.input_out_of_domain, "{name}: the base case is in domain");
        }

        let breakages: [(&str, Adjust); 18] = [
            ("day zero", |i| i.doy = 0),
            ("day 367", |i| i.doy = 367),
            ("altitude below the surface", |i| i.alt = -0.5),
            ("latitude past the north pole", |i| i.g_lat = 90.5),
            ("latitude past the south pole", |i| i.g_lat = -90.5),
            ("latitude that is not a number", |i| i.g_lat = f64::NAN),
            ("longitude that is not a number", |i| i.g_long = f64::NAN),
            ("longitude at negative infinity", |i| {
                i.g_long = f64::NEG_INFINITY;
            }),
            ("infinite altitude", |i| i.alt = f64::INFINITY),
            ("seconds that are not a number", |i| i.sec = f64::NAN),
            ("local time that is not a number", |i| i.lst = f64::NAN),
            ("negative F10.7 average", |i| i.f107a = -1.0),
            ("negative F10.7", |i| i.f107 = -1.0),
            ("infinite F10.7", |i| i.f107 = f64::INFINITY),
            ("negative Ap", |i| i.ap = -1.0),
            ("Ap that is not a number", |i| i.ap = f64::NAN),
            ("negative Ap history", |i| {
                i.ap_array = Some([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, -1.0]);
            }),
            ("infinite Ap history", |i| {
                i.ap_array = Some([4.0, 4.0, 4.0, f64::INFINITY, 4.0, 4.0, 4.0]);
            }),
        ];
        for (name, break_it) in breakages {
            let mut input = clean;
            break_it(&mut input);
            for entry in [gtd7, gtd7d, gts7] {
                assert!(entry(&input, &flags).input_out_of_domain, "{name}");
            }
        }

        // The closed ends of the domain, and the sentinel, are not breakages. A valid input has to
        // be asserted for each clause as well as an invalid one: a rejection can be right for the
        // wrong reason. The negative `ap` history above is rejected either by `v >= 0.0` or by its
        // negation, because under the negation the *other six* entries fail it — only a history
        // that must pass separates them, and only one containing an exact `0.0` separates `>=`
        // from `>`. Both are here.
        let accepted: [(&str, Adjust); 7] = [
            ("the longitude sentinel", |i| i.g_long = -1000.0),
            ("a longitude past the sentinel", |i| i.g_long = -1234.5),
            ("the north pole", |i| i.g_lat = 90.0),
            ("the surface", |i| i.alt = 0.0),
            ("the first day", |i| i.doy = 1),
            ("a quiet, dark sun on the last day", |i| {
                (i.doy, i.ap, i.f107, i.f107a) = (366, 0.0, 0.0, 0.0);
            }),
            ("a three-hourly Ap history through a quiet spell", |i| {
                i.ap_array = Some([12.0, 9.0, 7.0, 4.0, 0.0, 3.0, 6.0]);
            }),
        ];
        for (name, adjust) in accepted {
            let mut input = clean;
            adjust(&mut input);
            for entry in [gtd7, gtd7d, gts7] {
                assert!(!entry(&input, &flags).input_out_of_domain, "{name}");
            }
        }
    }

    /// The three facts about the shipped tables that `.cargo/mutants.toml` calls equivalences.
    ///
    /// Those entries are equivalences *for these coefficients*, not for the code: a term multiplied
    /// by a zero coefficient can be mutated freely, and `re + 0.0` and `re - 0.0` are the same
    /// double. If a future table changes any of them the mutants stop being equivalent, and this
    /// test is what says so instead of the exclusion quietly hiding a live one.
    ///
    /// The fourth fact is not about a zero but about a range, and it is the one that makes
    /// `globe7`'s `exp1 > 0.99999` saturation unreachable. `exp1` is
    /// `exp(-10800·|p[51]| / (1 + p[138]·(45 − |φ|)))`, so over `|φ| ≤ 90°` — which
    /// [`Nrlmsise00Output::input_out_of_domain`] is what enforces — it is bounded by the extreme
    /// combinations of the two coefficients, and the bound is computed here rather than asserted
    /// from memory.
    #[test]
    fn the_node_tables_carry_the_invariants_the_equivalences_rest_on() {
        let arrays: [&[f64; 150]; 11] = [
            &PT, &PS, &PD[0], &PD[1], &PD[2], &PD[3], &PD[4], &PD[5], &PD[6], &PD[7], &PD[8],
        ];
        for (i, p) in arrays.iter().enumerate() {
            assert_eq!(p[22], 0.0, "array {i}: p[22] is no longer a dead term");
            assert_eq!(p[78], 0.0, "array {i}: p[78] is no longer a dead term");
            assert!(
                p[24] > 1.0e-4,
                "array {i}: p[24] = {} reaches its floor",
                p[24]
            );
            assert!(
                p[43] > 0.0,
                "array {i}: p[43] = {} reaches its floor",
                p[43]
            );
        }

        assert_eq!(
            ZN3[ZN3.len() - 1],
            0.0,
            "densm's lower block no longer joins at the surface"
        );

        // `exp1` grows as the numerator shrinks and the denominator grows, so the extremes over the
        // shipped coefficients and `|φ| ≤ 90°` bound it. The denominator's sign is checked too: a
        // coefficient that let it reach zero would send `exp1` to 0 or ∞ instead.
        let mut worst = 0.0f64;
        for p in arrays {
            for lat in [-90.0f64, -45.0, 0.0, 45.0, 90.0] {
                if p[51] == 0.0 {
                    continue; // the block this guards is skipped for such an array
                }
                let denominator = 1.0 + p[138] * (45.0 - libm::fabs(lat));
                assert!(
                    denominator > 0.5,
                    "the exp1 denominator reached {denominator}"
                );
                let exp1 = libm::exp(-10800.0 * libm::sqrt(p[51] * p[51]) / denominator);
                worst = nan_max(worst, exp1);
            }
        }
        assert!(
            worst < 0.99999,
            "the exp1 saturation is reachable: {worst} over the shipped coefficients"
        );
    }

    /// The barometric exponent's `50` clamps keep a density finite and positive where the profile
    /// they are handed would otherwise overflow or underflow it.
    ///
    /// In the atmosphere the model actually describes, none of them fires: the node arrays descend,
    /// which makes `zgdif` and so the exponent negative, and over `51 230 400` evaluations spanning
    /// `0..1000 km`, every latitude and `ap` to `400` the largest value any of them saw was `4.39`.
    /// That is a measurement over a grid, not a statement about every `f64`, and it is not evidence
    /// that a mutation of the comparison would be invisible — so all four are driven directly here,
    /// through the private helpers' own node arguments, rather than written off as unreachable.
    /// Ascending nodes flip the sign of `zgdif` and a molecular mass far above any species scales
    /// the exponent past `50`; the one block that cannot be reached that way — `densm`'s lower one,
    /// which only runs below its own first node — is reached instead by end gradients steep enough
    /// to drive the `1/T` interpolant through zero, which makes its integral negative. The
    /// assertion is the invariant the clamp exists for: without it `exp(-expl)` underflows to
    /// exactly `0` or the factor it multiplies overflows to `∞`.
    #[test]
    fn the_barometric_exponent_clamps_keep_the_density_finite() {
        let (gsurf, re) = glatf(45.0);
        let mut held = false;

        // `densm`, upper block: ascending nodes, `alt` above `zn3[0]` so the lower block returns
        // first. Without the clamp `exp(-expl)` underflows and the density is exactly zero.
        let zn2_up = [35.0, 45.0, 55.0, 72.5];
        let tn2 = [230.0, 240.0, 250.0, 260.0];
        let tgn2 = [1.0, 1.0];
        let tn3 = [280.0, 270.0, 260.0, 250.0, 240.0];
        let tgn3 = [-1.0, -1.0];
        let (rho, tz) = densm(
            gsurf,
            re,
            34.0,
            3.5e17,
            5.0e4,
            ZN3.len(),
            &ZN3,
            &tn3,
            &tgn3,
            zn2_up.len(),
            &zn2_up,
            &tn2,
            &tgn2,
            1.0e4,
            &mut held,
        );
        assert!(
            rho.is_finite() && rho > 0.0,
            "densm's upper block saturated to {rho:e} at {tz} K"
        );

        // `densm`, lower block: reached only for `alt <= zn3[0]`, where the geopotential difference
        // is negative and the exponent with it — so the clamp needs the spline integral to go
        // negative instead, which end gradients steep enough to drive the `1/T` interpolant through
        // zero do. Without the clamp `exp(-expl)` underflows to exactly zero.
        let isothermal = [300.0; 5];
        let plunging = [-1.0e3, -1.0e3];
        let mut lower_held = false;
        let (rho_lower, tz_lower) = densm(
            gsurf,
            re,
            5.0,
            3.5e17,
            5.0e4,
            ZN3.len(),
            &ZN3,
            &isothermal,
            &plunging,
            zn2_up.len(),
            &zn2_up,
            &tn2,
            &tgn2,
            1.0e4,
            &mut lower_held,
        );
        assert!(
            rho_lower.is_finite() && rho_lower > 0.0,
            "densm's lower block saturated to {rho_lower:e} at {tz_lower} K"
        );

        // `densu`, Bates branch: a reference height far above the node table makes the geopotential
        // difference negative, so the exponent is positive and `exp` of it overflows. `tlb == tinf`
        // makes the profile isothermal, which keeps `(tlb/tt)^(1+alpha+gamma)` exactly `1` and
        // leaves the clamped exponential as the only thing the returned density depends on.
        let zn1 = [120.0, 110.0, 100.0, 90.0, 72.5];
        let mut tn1 = [300.0, 290.0, 280.0, 260.0, 230.0];
        let mut tgn1 = [1.0, 1.0];
        let (dens_bates, _) = densu(
            gsurf,
            re,
            400.0,
            3.0e10,
            1000.0,
            1000.0,
            2.0e3,
            0.0,
            800.0,
            0.02,
            zn1.len(),
            &zn1,
            &mut tn1,
            &mut tgn1,
            &mut held,
        );
        assert!(
            dens_bates.is_finite() && dens_bates > 0.0,
            "densu's Bates branch saturated to {dens_bates:e}"
        );

        // `densu`, spline branch: ascending nodes below the requested altitude, as above.
        let zn1_up = [72.5, 90.0, 100.0, 110.0, 120.0];
        let mut tn1_up = [230.0, 260.0, 280.0, 290.0, 300.0];
        let (dens_spline, _) = densu(
            gsurf,
            re,
            50.0,
            3.0e10,
            1000.0,
            400.0,
            5.0e4,
            0.0,
            72.5,
            0.02,
            zn1_up.len(),
            &zn1_up,
            &mut tn1_up,
            &mut tgn1,
            &mut held,
        );
        assert!(
            dens_spline.is_finite() && dens_spline > 0.0,
            "densu's spline branch saturated to {dens_spline:e}"
        );
    }

    /// (G2) `densu` solves the diffusive-equilibrium law it integrates,
    /// `d ln n/dz = −(1+α) d ln T/dz − m g(z)/(R T)` with `g(z) = g_s/(1 + z/R_e)²`, on both of its
    /// branches: the Bates–Walker closed form above `za` and the `1/T` spline below it.
    ///
    /// `derivations/nrlmsise00_structure.mac` proves the closed forms satisfy that equation; this
    /// measures that the code is those closed forms. The residual is taken by central differences
    /// at `h = 1e-3 km`, whose truncation (`h² ∂³/6`, scale heights of tens of km) and rounding
    /// (`ε ln n / h`) are both below `1e-9` of the gravity term, so the `1e-7` bound fails on any
    /// wrong exponent, sign or gravity law rather than on the stencil. The probe heights avoid the
    /// spline nodes, where the cubic's third derivative jumps.
    #[test]
    fn densu_solves_the_diffusive_equilibrium_law_on_both_branches() {
        let (gsurf, re) = glatf(45.0);
        let rgas = 831.4;
        let zn1 = [120.0, 110.0, 100.0, 90.0, 72.5];
        let (tinf, tlb, zlb, s2) = (1000.0, 360.0, 120.0, 0.02);
        for (xm, alpha) in [(4.0, -0.38), (16.0, 0.0), (28.0, 0.0)] {
            let profile = |z: f64| {
                let mut tn1 = [0.0, 250.0, 190.0, 185.0, 210.0];
                let mut tgn1 = [0.0, -1.0];
                let mut held = false;
                let (n, t) = densu(
                    gsurf,
                    re,
                    z,
                    1.0e10,
                    tinf,
                    tlb,
                    xm,
                    alpha,
                    zlb,
                    s2,
                    zn1.len(),
                    &zn1,
                    &mut tn1,
                    &mut tgn1,
                    &mut held,
                );
                assert!(!held, "the bounds bit at {z} km");
                (libm::log(n), libm::log(t), t)
            };
            for z in [80.0, 95.0, 105.0, 115.0, 150.0, 300.0, 600.0] {
                let h = 1.0e-3;
                let ((n_hi, t_hi, _), (n_lo, t_lo, _), (_, _, t)) =
                    (profile(z + h), profile(z - h), profile(z));
                let g = gsurf / libm::pow(1.0 + z / re, 2.0);
                let gravity = xm * g / (rgas * t);
                let residual =
                    (n_hi - n_lo) / (2.0 * h) + (1.0 + alpha) * (t_hi - t_lo) / (2.0 * h) + gravity;
                assert!(
                    libm::fabs(residual / gravity) < 1.0e-7,
                    "m = {xm} at {z} km: residual {:e} of the gravity term",
                    residual / gravity
                );
            }
        }
    }
    /// The longitude sentinel is a closed boundary: `-1000` itself asks for longitude off.
    ///
    /// `g_long <= -1000` is how the reference is told to drop every longitude and UT term, and both
    /// guards that read it — `globe7`'s `g_long > -1000.0` and `glob7s`'s `g_long <= -1000.0` —
    /// differ from their off-by-one neighbours on exactly one input out of the continuum. No sweep
    /// over longitudes can land on it, so it is asserted directly: at the sentinel the output must
    /// be *bit-identical* to one taken far past it, which it can only be if every term reading
    /// `g_long` was skipped. Checked above and below `72.5 km`, because the two guards live in
    /// different functions and only one of them runs at each altitude.
    ///
    /// The third longitude is the non-vacuity half. One degree inside the valid side the same
    /// comparison must fail, or the equality above would hold for a model that had simply stopped
    /// reading longitude at all.
    #[test]
    fn the_longitude_sentinel_includes_its_own_boundary() {
        let flags = Nrlmsise00Flags::default();
        let at = |alt: f64, g_long: f64| {
            let input = Nrlmsise00Input {
                alt,
                g_long,
                ..base_input()
            };
            let o = gtd7(&input, &flags);
            (o.d, o.t)
        };
        for alt in [50.0, 400.0] {
            assert_eq!(
                at(alt, -1000.0),
                at(alt, -1234.5),
                "{alt} km: the sentinel itself still reads longitude"
            );
            assert_ne!(
                at(alt, -999.0),
                at(alt, -1234.5),
                "{alt} km: longitude is ignored on the valid side too"
            );
        }
    }

    #[rustfmt::skip]
    static SWEEP: [Case; 496] = [
    // case 0: alt 14.500 km, lat -64.286, lon -147.273, doy 183, ap 15.789
    Case { doy: 183, sec: 2.88000000000000000e+04, alt: 1.45000000000000000e+01,
        g_lat: -6.42857142857142918e+01, g_long: -1.47272727272727280e+02, lst: 8.00000000000000000e+00,
        f107a: 8.30769230769230802e+01, f107: 7.88235294117647101e+01, ap: 1.57894736842105257e+01,
        ap_array: [4.34782608695652151e+00, 9.34782608695652151e+00, 1.43478260869565215e+01, 1.93478260869565233e+01, 2.43478260869565233e+01, 2.93478260869565233e+01, 3.43478260869565233e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.07311834115447422e+13, 0.00000000000000000e+00, 3.08926365748306330e+18, 8.28756761392981376e+17, 3.69522272389836640e+16, 1.90066299508734348e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.07697917772741476e+02, 1.90066299508734348e-04] },
    // case 1: alt 91.500 km, lat -38.571, lon -114.545, doy 92, ap 31.579
    Case { doy: 92, sec: 5.76000000000000000e+04, alt: 9.15000000000000000e+01,
        g_lat: -3.85714285714285765e+01, g_long: -1.14545454545454547e+02, lst: 8.36363636363636331e+00,
        f107a: 1.01153846153846160e+02, f107: 9.26470588235294201e+01, ap: 3.15789473684210513e+01,
        ap_array: [8.69565217391304301e+00, 1.36956521739130430e+01, 1.86956521739130430e+01, 2.36956521739130430e+01, 2.86956521739130430e+01, 3.36956521739130466e+01, 3.86956521739130466e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.43587948547468185e+08, 4.47365326264930176e+11, 4.09491195553469141e+13, 1.02329611347696133e+13, 4.67771350285257019e+11, 2.48983439179038838e-09, 5.59502213388880715e+07, 1.87307193444816541e+05, 1.18747953771273992e-48, 1.02731846489999998e+03, 1.73414938209379756e+02, 2.48983439179038838e-09] },
    // case 2: alt 288.000 km, lat -12.857, lon -81.818, doy 274, ap 47.368
    Case { doy: 274, sec: 9.60000000000000000e+03, alt: 2.88000000000000000e+02,
        g_lat: -1.28571428571428612e+01, g_long: -8.18181818181818272e+01, lst: -2.78787878787878851e+00,
        f107a: 1.19230769230769226e+02, f107: 1.06470588235294116e+02, ap: 4.73684210526315752e+01,
        ap_array: [1.30434782608695645e+01, 1.80434782608695627e+01, 2.30434782608695627e+01, 2.80434782608695627e+01, 3.30434782608695627e+01, 3.80434782608695627e+01, 4.30434782608695627e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.51336311127076857e+06, 8.59238020720770717e+08, 1.41319864445200592e+08, 4.11391616845148336e+06, 2.85215317763522980e+04, 2.99605311093155463e-14, 1.57604885823647055e+05, 1.23393531760566644e+07, 8.37911165165346183e+00, 9.11957886221952094e+02, 9.02268321641753801e+02, 2.99605313318647566e-14] },
    // case 3: alt 880.000 km, lat +12.857, lon -49.091, doy 46, ap 63.158
    Case { doy: 46, sec: 3.84000000000000000e+04, alt: 8.80000000000000000e+02,
        g_lat: 1.28571428571428470e+01, g_long: -4.90909090909090935e+01, lst: 7.39393939393939270e+00,
        f107a: 1.37307692307692321e+02, f107: 1.20294117647058826e+02, ap: 6.31578947368421026e+01,
        ap_array: [1.73913043478260860e+01, 2.23913043478260860e+01, 2.73913043478260860e+01, 3.23913043478260860e+01, 3.73913043478260860e+01, 4.23913043478260860e+01, 4.73913043478260860e+01],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, -1, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [4.42505194802206242e+05, 2.47161026075761147e+04, 2.23213453813479346e+00, 8.87041042432365652e-03, 1.98407203717853626e-07, 3.70819336164858079e-18, 5.19076422843478431e+04, 1.17161255980997635e+03, 3.70482803708918527e+03, 9.15149015795976993e+02, 9.15149012791023665e+02, 3.80659359431366945e-18] },
    // case 4: alt 2.900 km, lat +38.571, lon -16.364, doy 229, ap 78.947
    Case { doy: 229, sec: 6.71999999999999854e+04, alt: 2.89999999999999991e+00,
        g_lat: 3.85714285714285552e+01, g_long: -1.63636363636363455e+01, lst: 1.86666666666666643e+01,
        f107a: 1.55384615384615387e+02, f107: 1.34117647058823536e+02, ap: 7.89473684210526301e+01,
        ap_array: [2.17391304347826093e+01, 2.67391304347826093e+01, 3.17391304347826093e+01, 3.67391304347826093e+01, 4.17391304347826093e+01, 4.67391304347826093e+01, 5.17391304347826093e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.85508177272739219e+13, 0.00000000000000000e+00, 1.46855803441767690e+19, 3.93970063893230182e+18, 1.75661569286870432e+17, 9.03527254915497996e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.82353886820476362e+02, 9.03527254915497996e-04] },
    // case 5: alt 83.900 km, lat +64.286, lon +16.364, doy 137, ap 94.737
    Case { doy: 137, sec: 1.92000000000000000e+04, alt: 8.39000000000000057e+01,
        g_lat: 6.42857142857142776e+01, g_long: 1.63636363636363455e+01, lst: 6.42424242424242298e+00,
        f107a: 1.73461538461538453e+02, f107: 1.47941176470588232e+02, ap: 9.47368421052631504e+01,
        ap_array: [2.60869565217391290e+01, 3.10869565217391290e+01, 3.60869565217391255e+01, 4.10869565217391255e+01, 4.60869565217391255e+01, 5.10869565217391255e+01, 5.60869565217391255e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.44271732015388393e+09, 2.09714886714956322e+10, 2.14885509253491219e+14, 5.30813458965139922e+13, 2.54821642772006348e+12, 1.29773279496158148e-08, 1.38005492874436200e+08, 1.38505421631014382e+05, 6.14704694728557584e-53, 1.02731846489999998e+03, 1.63229248921988074e+02, 1.29773279496158148e-08] },
    // case 6: alt 243.200 km, lat -86.327, lon +49.091, doy 320, ap 110.526
    Case { doy: 320, sec: 4.80000000000000000e+04, alt: 2.43199999999999989e+02,
        g_lat: -8.63265306122448948e+01, g_long: 4.90909090909090935e+01, lst: 1.66060606060606055e+01,
        f107a: 1.91538461538461547e+02, f107: 1.61764705882352928e+02, ap: 1.10526315789473685e+02,
        ap_array: [3.04347826086956559e+01, 3.54347826086956559e+01, 4.04347826086956559e+01, 4.54347826086956559e+01, 5.04347826086956559e+01, 5.54347826086956559e+01, 6.04347826086956559e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.82751100484750420e+06, 1.25486825627551603e+09, 2.24346805321090412e+09, 1.03118721639139906e+08, 4.69533777372170240e+06, 1.43627182042056313e-13, 6.36814611033435067e+04, 8.88677866574481688e+06, 6.48060821935571730e-03, 1.45449481776105313e+03, 1.35238384821956970e+03, 1.43627182042228449e-13] },
    // case 7: alt 784.000 km, lat -60.612, lon +81.818, doy 23, ap 126.316
    Case { doy: 23, sec: 7.68000000000000000e+04, alt: 7.84000000000000000e+02,
        g_lat: -6.06122448979591866e+01, g_long: 8.18181818181818130e+01, lst: 2.67878787878787854e+01,
        f107a: 2.09615384615384613e+02, f107: 1.75588235294117652e+02, ap: 1.26315789473684205e+02,
        ap_array: [3.47826086956521721e+01, 3.97826086956521721e+01, 4.47826086956521721e+01, 4.97826086956521721e+01, 5.47826086956521721e+01, 5.97826086956521721e+01, 6.47826086956521721e+01],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [3.69419542232575535e+05, 1.57720616988962953e+05, 2.52371721978058815e+02, 1.39153647248180756e+00, 5.58102352822857075e-04, 6.78375146837635189e-18, 2.55699567927693097e+04, 3.76487948951127782e+03, 5.55021325742046065e+03, 1.02731846489999998e+03, 1.02731813067422672e+03, 6.93116513249343907e-18] },
    // case 8: alt 60.900 km, lat -34.898, lon +114.545, doy 206, ap 142.105
    Case { doy: 206, sec: 3.20000000000000000e+03, alt: 6.09000000000000057e+01,
        g_lat: -3.48979591836734713e+01, g_long: 1.14545454545454561e+02, lst: 8.88888888888888840e-01,
        f107a: 2.27692307692307679e+02, f107: 1.89411764705882348e+02, ap: 1.42105263157894740e+02,
        ap_array: [3.91304347826086882e+01, 4.41304347826086882e+01, 4.91304347826086882e+01, 5.41304347826086882e+01, 5.91304347826086882e+01, 6.41304347826086882e+01, 6.91304347826086882e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.68085263428354530e+10, 0.00000000000000000e+00, 3.99488077923612600e+15, 1.07170666664567550e+15, 4.77847664408329219e+13, 2.45784203251399611e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.35723571552211126e+02, 2.45784203251399611e-07] },
    // case 9: alt 76.300 km, lat -9.184, lon +147.273, doy 115, ap 157.895
    Case { doy: 115, sec: 3.20000000000000000e+04, alt: 7.62999999999999972e+01,
        g_lat: -9.18367346938775597e+00, g_long: 1.47272727272727309e+02, lst: 1.87070707070707094e+01,
        f107a: 2.45769230769230774e+02, f107: 2.03235294117647072e+02, ap: 1.57894736842105260e+02,
        ap_array: [4.34782608695652186e+01, 4.84782608695652186e+01, 5.34782608695652186e+01, 5.84782608695652186e+01, 6.34782608695652186e+01, 6.84782608695652186e+01, 7.34782608695652186e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.67155204842721844e+09, 2.10398781826202571e+08, 5.38026655630374562e+14, 1.32669732293950031e+14, 6.40689179503824609e+12, 3.24803427176290572e-08, 1.33592380681889978e+02, 8.54825410912136867e+04, 2.63041448734679905e-60, 1.02731846489999998e+03, 1.96941305101251601e+02, 3.24803427176290572e-08] },
    // case 10: alt 198.400 km, lat +16.531, lon -177.025, doy 297, ap 173.684
    Case { doy: 297, sec: 6.08000000000000000e+04, alt: 1.98400000000000006e+02,
        g_lat: 1.65306122448979522e+01, g_long: -1.77024793388429742e+02, lst: 5.08723599632690693e+00,
        f107a: 2.63846153846153868e+02, f107: 2.17058823529411768e+02, ap: 1.73684210526315809e+02,
        ap_array: [4.78260869565217419e+01, 5.28260869565217419e+01, 5.78260869565217419e+01, 6.28260869565217419e+01, 6.78260869565217490e+01, 7.28260869565217490e+01, 7.78260869565217490e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.48251841680074986e+07, 1.11090574207502060e+10, 7.09361252547280407e+09, 1.16670257290078759e+08, 8.15080195133190881e+06, 6.33862353289978843e-13, 2.98353638142509371e+04, 9.71364864234286249e+07, 5.64058773566123383e-08, 1.36193357111118371e+03, 1.18871504163386226e+03, 6.33862353289978843e-13] },
    // case 11: alt 688.000 km, lat +42.245, lon -144.298, doy 69, ap 189.474
    Case { doy: 69, sec: 1.28000000000000000e+04, alt: 6.88000000000000000e+02,
        g_lat: 4.22448979591836746e+01, g_long: -1.44297520661157023e+02, lst: -6.06427915518824534e+00,
        f107a: 2.81923076923076906e+02, f107: 2.30882352941176464e+02, ap: 1.89473684210526301e+02,
        ap_array: [5.21739130434782581e+01, 5.71739130434782581e+01, 6.21739130434782581e+01, 6.71739130434782510e+01, 7.21739130434782510e+01, 7.71739130434782510e+01, 8.21739130434782510e+01],
        switches: [0, 0, 1, 2, 0, 1, 2, 0, 1, -1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], out: [1.31878884296132671e+06, 1.89291384457603376e+06, 4.35972415441254270e+03, 5.09970382504802444e+01, 9.91660347482190008e-03, 6.05676310208345745e-17, 6.88400611418197368e+04, 5.23002276379111718e+04, 7.26269188852487969e+03, 1.02731846489999998e+03, 1.02731775623818953e+03, 6.07605281173937946e-17] },
    // case 12: alt 49.300 km, lat +67.959, lon -111.570, doy 251, ap 205.263
    Case { doy: 251, sec: 4.16000000000000000e+04, alt: 4.93000000000000043e+01,
        g_lat: 6.79591836734693970e+01, g_long: -1.11570247933884289e+02, lst: 1.15555555555555554e+01,
        f107a: 6.63905325443787007e+01, f107: 2.44705882352941160e+02, ap: 2.05263157894736821e+02,
        ap_array: [5.65217391304347814e+01, 6.15217391304347814e+01, 6.65217391304347814e+01, 7.15217391304347814e+01, 7.65217391304347814e+01, 8.15217391304347814e+01, 8.65217391304347814e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.31571340280842102e+11, 0.00000000000000000e+00, 1.96061436449206800e+16, 5.25974015562287100e+15, 2.34518887208718719e+14, 1.20626388142695876e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.63417688466125810e+02, 1.20626388142695876e-06] },
    // case 13: alt 114.300 km, lat -82.653, lon -78.843, doy 160, ap 221.053
    Case { doy: 160, sec: 7.03999999999999854e+04, alt: 1.14299999999999997e+02,
        g_lat: -8.26530612244897895e+01, g_long: -7.88429752066115839e+01, lst: 1.42993572084481109e+01,
        f107a: 8.44674556213017809e+01, f107: 2.58529411764705856e+02, ap: 2.21052631578947370e+02,
        ap_array: [6.08695652173913118e+01, 6.58695652173913118e+01, 7.08695652173913118e+01, 7.58695652173913118e+01, 8.08695652173913118e+01, 8.58695652173913118e+01, 9.08695652173913118e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.70561152787213102e+07, 8.00864895929820404e+10, 4.19334256616636414e+11, 9.50612201749701538e+10, 3.24381424534977865e+09, 2.68830526505846016e-11, 2.88004668179345271e+06, 3.06557020327298029e+05, 3.37761658223340913e-36, 1.02731846489999998e+03, 3.74256286853402685e+02, 2.68830526505846016e-11] },
    // case 14: alt 153.600 km, lat -56.939, lon -46.116, doy 343, ap 236.842
    Case { doy: 343, sec: 2.24000000000000000e+04, alt: 1.53599999999999994e+02,
        g_lat: -5.69387755102040813e+01, g_long: -4.61157024793388359e+01, lst: 3.14784205693296659e+00,
        f107a: 1.02544378698224847e+02, f107: 2.72352941176470608e+02, ap: 2.36842105263157890e+02,
        ap_array: [6.52173913043478279e+01, 7.02173913043478279e+01, 7.52173913043478279e+01, 8.02173913043478279e+01, 8.52173913043478279e+01, 9.02173913043478279e+01, 9.52173913043478279e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.86329058933379408e+06, 7.51354677347366810e+09, 3.34459135833122787e+10, 4.67952924828826618e+09, 3.47311667269497037e+08, 2.02601242370043903e-12, 1.66169046161921608e+05, 9.00437085744460672e+06, 7.20959814382073346e-20, 1.50817692132070397e+03, 8.48079645555624779e+02, 2.02601242370043903e-12] },
    // case 15: alt 592.000 km, lat -31.224, lon -13.388, doy 12, ap 252.632
    Case { doy: 12, sec: 5.12000000000000000e+04, alt: 5.92000000000000000e+02,
        g_lat: -3.12244897959183731e+01, g_long: -1.33884297520661164e+01, lst: 1.33296602387511474e+01,
        f107a: 1.20621301775147941e+02, f107: 2.86176470588235304e+02, ap: 2.52631578947368411e+02,
        ap_array: [6.95652173913043441e+01, 7.45652173913043441e+01, 7.95652173913043441e+01, 8.45652173913043441e+01, 8.95652173913043441e+01, 9.45652173913043441e+01, 9.95652173913043441e+01],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [1.89269692557733343e+06, 1.24236923626562431e+07, 1.46368429184330307e+05, 2.95000583068817514e+03, 1.61977020825327256e+00, 3.55687203783402568e-16, 6.56946374073368206e+04, 2.61504195196533605e+05, 9.30588214206771954e+03, 1.20495183644648660e+03, 1.20456315999214144e+03, 3.55934368013095890e-16] },
    // case 16: alt 37.700 km, lat -5.510, lon +19.339, doy 194, ap 268.421
    Case { doy: 194, sec: 7.99999999999999854e+04, alt: 3.77000000000000028e+01,
        g_lat: -5.51020408163266495e+00, g_long: 1.93388429752065747e+01, lst: 2.22222222222222214e+01,
        f107a: 1.38698224852071007e+02, f107: 6.58131487889273359e+01, ap: 2.68421052631578902e+02,
        ap_array: [7.39130434782608603e+01, 7.89130434782608603e+01, 8.39130434782608603e+01, 8.89130434782608603e+01, 9.39130434782608603e+01, 9.89130434782608603e+01, 1.03913043478260860e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.15787633801385010e+11, 0.00000000000000000e+00, 9.17617832067773760e+16, 2.46169335808821680e+16, 1.09760856982786762e+15, 5.64562449313393159e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.46895750591514485e+02, 5.64562449313393159e-06] },
    // case 17: alt 106.700 km, lat +20.204, lon +52.066, doy 103, ap 284.211
    Case { doy: 103, sec: 6.40000000000000000e+03, alt: 1.06700000000000003e+02,
        g_lat: 2.02040816326530575e+01, g_long: 5.20661157024793226e+01, lst: 5.24885215794306603e+00,
        f107a: 1.56775147928994102e+02, f107: 7.96366782006920459e+01, ap: 2.84210526315789480e+02,
        ap_array: [7.82608695652173765e+01, 8.32608695652173765e+01, 8.82608695652173765e+01, 9.32608695652173765e+01, 9.82608695652173765e+01, 1.03260869565217376e+02, 1.08260869565217376e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.55825970242168009e+07, 3.54334813072916321e+11, 2.24857379516729639e+12, 3.91224182997236023e+11, 1.85822179277716675e+10, 1.35940990893770785e-10, 8.88298740757630020e+06, 4.37769262408591982e+05, 2.38242641272608011e-39, 1.02731846489999998e+03, 2.44115844357851557e+02, 1.35940990893770785e-10] },
    // case 18: alt 377.600 km, lat +45.918, lon +84.793, doy 286, ap 0.831
    Case { doy: 286, sec: 3.52000000000000000e+04, alt: 3.77600000000000023e+02,
        g_lat: 4.59183673469387656e+01, g_long: 8.47933884297520422e+01, lst: 1.54306703397612477e+01,
        f107a: 1.74852071005917139e+02, f107: 9.34602076124567418e+01, ap: 8.31024930747922386e-01,
        ap_array: [8.26086956521739069e+01, 8.76086956521739069e+01, 9.26086956521739069e+01, 9.76086956521739069e+01, 1.02608695652173907e+02, 1.07608695652173907e+02, 1.12608695652173907e+02],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.16898176338596363e+06, 1.88718053414527804e+08, 1.21586813997464236e+07, 1.46330013351933245e+05, 7.19352339843410050e+02, 5.66541616878197940e-15, 1.30134603663308473e+05, 1.96085546369720669e+06, 2.35166849021323696e+03, 9.70221903829196549e+02, 9.69307326491641106e+02, 5.66547862909707943e-15] },
    // case 19: alt 496.000 km, lat +71.633, lon +117.521, doy 58, ap 16.620
    Case { doy: 58, sec: 6.40000000000000000e+04, alt: 4.96000000000000000e+02,
        g_lat: 7.16326530612244881e+01, g_long: 1.17520661157024790e+02, lst: 2.56124885215794293e+01,
        f107a: 1.92928994082840262e+02, f107: 1.07283737024221452e+02, ap: 1.66204986149584499e+01,
        ap_array: [8.69565217391304373e+01, 9.19565217391304373e+01, 9.69565217391304373e+01, 1.01956521739130437e+02, 1.06956521739130437e+02, 1.11956521739130437e+02, 1.16956521739130437e+02],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, -1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [9.51996218474587775e+05, 1.02864803554280289e+07, 5.12378631980285514e+05, 9.80812696313334891e+03, 2.40918200689088273e+01, 3.06492983017950916e-16, 4.09033307620596024e+04, 1.10023414047689541e+05, 8.77283165942007690e+03, 1.02731846489999998e+03, 1.02719097057625004e+03, 3.06725989426825094e-16] },
    // case 20: alt 26.100 km, lat -78.980, lon +150.248, doy 240, ap 32.410
    Case { doy: 240, sec: 1.60000000000000000e+04, alt: 2.60999999999999979e+01,
        g_lat: -7.89795918367346985e+01, g_long: 1.50247933884297538e+02, lst: 4.44444444444444464e+00,
        f107a: 2.11005917159763328e+02, f107: 1.21107266435986162e+02, ap: 3.24099722991689703e+01,
        ap_array: [9.13043478260869534e+01, 9.63043478260869534e+01, 1.01304347826086953e+02, 1.06304347826086953e+02, 1.11304347826086953e+02, 1.16304347826086953e+02, 1.21304347826086953e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.54083193574129004e+12, 0.00000000000000000e+00, 3.78622850564662272e+17, 1.01573152120981952e+17, 4.52889722702920800e+15, 2.32946915819108567e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.04600112951475552e+02, 2.32946915819108567e-05] },
    // case 21: alt 99.100 km, lat -53.265, lon -174.050, doy 149, ap 48.199
    Case { doy: 149, sec: 4.48000000000000000e+04, alt: 9.90999999999999943e+01,
        g_lat: -5.32653061224489761e+01, g_long: -1.74049586776859513e+02, lst: 8.41138659320476378e-01,
        f107a: 2.29082840236686394e+02, f107: 1.34930795847750858e+02, ap: 4.81994459833794977e+01,
        ap_array: [9.56521739130434838e+01, 1.00652173913043484e+02, 1.05652173913043484e+02, 1.10652173913043484e+02, 1.15652173913043484e+02, 1.20652173913043484e+02, 1.25652173913043484e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.96953820191380680e+07, 2.75111421562805542e+11, 7.99096724905970020e+12, 1.65829605256134351e+12, 8.57012288044692535e+10, 4.72507057291269170e-10, 1.63902676627651509e+07, 1.33752573734844598e+05, 1.66062018852026163e-43, 1.02731846489999998e+03, 1.76402365475825974e+02, 4.72507057291269170e-10] },
    // case 22: alt 332.800 km, lat -27.551, lon -141.322, doy 331, ap 63.989
    Case { doy: 331, sec: 7.35999999999999854e+04, alt: 3.32800000000000068e+02,
        g_lat: -2.75510204081632679e+01, g_long: -1.41322314049586765e+02, lst: 1.10229568411386545e+01,
        f107a: 2.47159763313609488e+02, f107: 1.48754325259515554e+02, ap: 6.39889196675900251e+01,
        ap_array: [1.89035916824196598e-01, 5.18903591682419663e+00, 1.01890359168241957e+01, 1.51890359168241957e+01, 2.01890359168241957e+01, 2.51890359168241957e+01, 3.01890359168241957e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.29027311588220671e+06, 9.15828052198763490e+08, 1.29596039859201163e+08, 1.35897087609204394e+06, 2.91943628689961479e+04, 3.08271520864114146e-14, 5.70575929917929461e+04, 1.56259001562070325e+07, 8.15436301059043330e+02, 1.11482593008965569e+03, 1.11348019480316680e+03, 3.08271737443995661e-14] },
    // case 23: alt 976.000 km, lat -1.837, lon -108.595, doy 35, ap 79.778
    Case { doy: 35, sec: 2.56000000000000000e+04, alt: 9.76000000000000000e+02,
        g_lat: -1.83673469387755972e+00, g_long: -1.08595041322314046e+02, lst: -1.28558310376492457e-01,
        f107a: 2.65236686390532554e+02, f107: 1.62577854671280278e+02, ap: 7.97783933518005455e+01,
        ap_array: [4.53686200378071813e+00, 9.53686200378071902e+00, 1.45368620037807190e+01, 1.95368620037807190e+01, 2.45368620037807190e+01, 2.95368620037807190e+01, 3.45368620037807190e+01],
        switches: [0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], out: [4.28053817142050015e+05, 2.48034256194439258e+04, 2.40661553784121063e+00, 9.79523713578676902e-03, 2.26254443416938946e-07, 3.61003955562567164e-18, 4.93199509072287619e+04, 1.16177827090405003e+03, 2.65760269932622396e+03, 1.02731846489999998e+03, 1.02731845846906981e+03, 3.68062548331977667e-18] },
    // case 24: alt 0.580 km, lat +23.878, lon -75.868, doy 217, ap 95.568
    Case { doy: 217, sec: 5.44000000000000000e+04, alt: 5.79999999999999960e-01,
        g_lat: 2.38775510204081485e+01, g_long: -7.58677685950413263e+01, lst: 1.51111111111111107e+01,
        f107a: 2.83313609467455649e+02, f107: 1.76401384083044974e+02, ap: 9.55678670360110658e+01,
        ap_array: [8.88468809073724053e+00, 1.38846880907372405e+01, 1.88846880907372423e+01, 2.38846880907372423e+01, 2.88846880907372423e+01, 3.38846880907372423e+01, 3.88846880907372423e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.19935856546163188e+14, 0.00000000000000000e+00, 1.78722784658223247e+19, 4.79459614402615501e+18, 2.13779258868933728e+17, 1.09958818942507575e-03, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.95924775445314538e+02, 1.09958818942507575e-03] },
    // case 25: alt 82.380 km, lat +49.592, lon -43.140, doy 126, ap 111.357
    Case { doy: 126, sec: 8.32000000000000000e+04, alt: 8.23799999999999955e+01,
        g_lat: 4.95918367346938567e+01, g_long: -4.31404958677685784e+01, lst: 2.02350780532598726e+01,
        f107a: 6.77810650887574013e+01, f107: 1.90224913494809698e+02, ap: 1.11357340720221586e+02,
        ap_array: [1.32325141776937603e+01, 1.82325141776937585e+01, 2.32325141776937585e+01, 2.82325141776937585e+01, 3.32325141776937585e+01, 3.82325141776937585e+01, 4.32325141776937585e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.71472878322149825e+09, 1.51865293376904678e+10, 2.54318846626682219e+14, 6.79598639913629688e+13, 3.02418501330035498e+12, 1.56319887976150170e-08, 1.22381898397985786e+08, 1.34690414560830483e+05, 6.34401446447683922e-55, 1.02731846489999998e+03, 1.77526636843176846e+02, 1.56319887976150170e-08] },
    // case 26: alt 234.240 km, lat +75.306, lon -10.413, doy 308, ap 127.147
    Case { doy: 308, sec: 1.06666666666666652e+03, alt: 2.34240000000000009e+02,
        g_lat: 7.53061224489795791e+01, g_long: -1.04132231404958588e+01, lst: -3.97918579736760925e-01,
        f107a: 8.58579881656804673e+01, f107: 2.04048442906574422e+02, ap: 1.27146814404432121e+02,
        ap_array: [1.75803402646502818e+01, 2.25803402646502818e+01, 2.75803402646502818e+01, 3.25803402646502818e+01, 3.75803402646502818e+01, 4.25803402646502818e+01, 4.75803402646502818e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.20651260758347530e+06, 6.75736227300238371e+08, 2.18948418613130283e+09, 2.47958834754679948e+08, 4.90345029362249654e+06, 1.33271966146278751e-13, 6.77366745130795462e+04, 2.23325686498466693e+06, 1.27713999070993417e-04, 1.60963061184174808e+03, 1.33049247888145828e+03, 1.33271966146282134e-13] },
    // case 27: alt 764.800 km, lat -75.306, lon +22.314, doy 80, ap 142.936
    Case { doy: 80, sec: 2.98666666666666642e+04, alt: 7.64800000000000068e+02,
        g_lat: -7.53061224489795933e+01, g_long: 2.23140495867768323e+01, lst: 9.78389960208141751e+00,
        f107a: 1.03934911242603562e+02, f107: 2.17871972318339118e+02, ap: 1.42936288088642641e+02,
        ap_array: [2.19281663516068051e+01, 2.69281663516068051e+01, 3.19281663516068051e+01, 3.69281663516068051e+01, 4.19281663516068051e+01, 4.69281663516068051e+01, 5.19281663516068051e+01],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, -1, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [1.09444878027374740e+06, 1.42086327940604207e+06, 3.33015003669073940e+03, 3.91811812620899360e+01, 7.31229723115478687e-03, 4.61662559014043590e-17, 5.68807858871675271e+04, 3.91436330120020139e+04, 5.59421777710865717e+03, 1.22321677993073649e+03, 1.22315367381068768e+03, 4.63148383255643587e-17] },
    // case 28: alt 58.580 km, lat -49.592, lon +55.041, doy 263, ap 158.726
    Case { doy: 263, sec: 5.86666666666666570e+04, alt: 5.85800000000000054e+01,
        g_lat: -4.95918367346938851e+01, g_long: 5.50413223140496086e+01, lst: 1.62962962962962941e+01,
        f107a: 1.22011834319526628e+02, f107: 2.31695501730103814e+02, ap: 1.58725761772853161e+02,
        ap_array: [2.62759924385633283e+01, 3.12759924385633283e+01, 3.62759924385633283e+01, 4.12759924385633283e+01, 4.62759924385633283e+01, 5.12759924385633283e+01, 5.62759924385633283e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.76707076768483353e+10, 0.00000000000000000e+00, 5.61351206380959700e+15, 1.50593688135820075e+15, 6.71460245512584922e+13, 3.45370154027320258e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.39718869183365086e+02, 3.45370154027320258e-07] },
    // case 29: alt 74.780 km, lat -23.878, lon +87.769, doy 172, ap 174.515
    Case { doy: 172, sec: 1.06666666666666661e+04, alt: 7.47800000000000011e+01,
        g_lat: -2.38775510204081627e+01, g_long: 8.77685950413223281e+01, lst: 8.81420263238445045e+00,
        f107a: 1.40088757396449694e+02, f107: 2.45519031141868510e+02, ap: 1.74515235457063710e+02,
        ap_array: [3.06238185255198516e+01, 3.56238185255198516e+01, 4.06238185255198516e+01, 4.56238185255198516e+01, 5.06238185255198516e+01, 5.56238185255198516e+01, 6.06238185255198516e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.59605888095860863e+09, 7.21790079635921717e+07, 6.74414728379213750e+14, 1.76404604129463719e+14, 8.04652050795536914e+12, 4.12517305443784909e-08, 1.74048481924737338e-05, 5.68784560780936299e+04, 1.02580573535277963e-61, 1.02731846489999998e+03, 2.26837059652619331e+02, 4.12517305443784909e-08] },
    // case 30: alt 189.440 km, lat +1.837, lon +120.496, doy 354, ap 190.305
    Case { doy: 354, sec: 3.94666666666666642e+04, alt: 1.89439999999999998e+02,
        g_lat: 1.83673469387755972e+00, g_long: 1.20495867768595076e+02, lst: 1.89960208142026339e+01,
        f107a: 1.58165680473372788e+02, f107: 2.59342560553633234e+02, ap: 1.90304709141274230e+02,
        ap_array: [3.49716446124763678e+01, 3.99716446124763678e+01, 4.49716446124763678e+01, 4.99716446124763678e+01, 5.49716446124763678e+01, 5.99716446124763678e+01, 6.49716446124763678e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.20899533046901952e+07, 7.59135095931025028e+09, 6.24525483956107330e+09, 2.63512362419040293e+08, 8.02776245926210005e+06, 5.08265672518824438e-13, 7.26377833758863708e+04, 7.52464762946418673e+07, 2.01016766273356497e-09, 1.25133054617906259e+03, 1.02042615887760849e+03, 5.08265672518824438e-13] },
    // case 31: alt 668.800 km, lat +27.551, lon +153.223, doy 6, ap 206.094
    Case { doy: 6, sec: 6.82666666666666570e+04, alt: 6.68799999999999955e+02,
        g_lat: 2.75510204081632537e+01, g_long: 1.53223140495867767e+02, lst: 2.91778389960208138e+01,
        f107a: 1.76242603550295883e+02, f107: 2.73166089965397930e+02, ap: 2.06094182825484722e+02,
        ap_array: [3.93194706994328911e+01, 4.43194706994328911e+01, 4.93194706994328911e+01, 5.43194706994328911e+01, 5.93194706994328911e+01, 6.43194706994328840e+01, 6.93194706994328840e+01],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [1.09333748277272005e+06, 2.16736697073551174e+06, 5.03373485423442889e+03, 3.24151592465302301e+01, 3.45462035611327906e-02, 6.68001354763436205e-17, 1.92955909885849542e+04, 7.34675122119026491e+04, 2.56437441545834794e+05, 1.02731846489999998e+03, 1.02731506691370191e+03, 7.36111139238009951e-17] },
    // case 32: alt 46.980 km, lat +53.265, lon -171.074, doy 189, ap 221.884
    Case { doy: 189, sec: 2.02666666666666642e+04, alt: 4.69800000000000111e+01,
        g_lat: 5.32653061224489477e+01, g_long: -1.71074380165289256e+02, lst: 5.62962962962962976e+00,
        f107a: 1.94319526627218949e+02, f107: 2.86989619377162626e+02, ap: 2.21883656509695271e+02,
        ap_array: [4.36672967863894144e+01, 4.86672967863894144e+01, 5.36672967863894144e+01, 5.86672967863894144e+01, 6.36672967863894144e+01, 6.86672967863894144e+01, 7.36672967863894144e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.03881245665006012e+11, 0.00000000000000000e+00, 3.03814263842040840e+16, 8.15042525609043000e+15, 3.63407431694657125e+14, 1.86921089517732599e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.74937224918640425e+02, 1.86921089517732599e-06] },
    // case 33: alt 112.780 km, lat +78.980, lon -138.347, doy 97, ap 237.673
    Case { doy: 97, sec: 4.90666666666666715e+04, alt: 1.12780000000000001e+02,
        g_lat: 7.89795918367346701e+01, g_long: -1.38347107438016536e+02, lst: 4.40648913376186258e+00,
        f107a: 2.12396449704142015e+02, f107: 6.66262975778546718e+01, ap: 2.37673130193905820e+02,
        ap_array: [4.80151228733459376e+01, 5.30151228733459376e+01, 5.80151228733459376e+01, 6.30151228733459376e+01, 6.80151228733459448e+01, 7.30151228733459448e+01, 7.80151228733459448e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.66786000342194568e+07, 5.03208253908628235e+10, 6.33191405244784546e+11, 1.02769898750874298e+11, 4.81875400263298416e+09, 3.65464853839545826e-11, 4.05838604530325485e+06, 3.43181985074043623e+05, 2.88222278599217315e-35, 1.02731846489999998e+03, 4.25127202517924786e+02, 3.65464853839545826e-11] },
    // case 34: alt 144.640 km, lat -71.633, lon -105.620, doy 280, ap 253.463
    Case { doy: 280, sec: 7.78666666666666715e+04, alt: 1.44639999999999986e+02,
        g_lat: -7.16326530612244881e+01, g_long: -1.05619834710743802e+02, lst: 1.45883073155800425e+01,
        f107a: 2.30473372781065081e+02, f107: 8.04498269896193818e+01, ap: 2.53462603878116312e+02,
        ap_array: [5.23629489603024609e+01, 5.73629489603024609e+01, 6.23629489603024609e+01, 6.73629489603024609e+01, 7.23629489603024609e+01, 7.73629489603024609e+01, 8.23629489603024609e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.39627679142878894e+07, 2.65019113946461716e+10, 7.27805612179625092e+10, 8.33770479098090076e+09, 4.98265488768230021e+08, 4.56321517456133948e-12, 3.68191718894248130e+05, 1.75082435593819655e+07, 1.61949305953766781e-22, 1.05273713955986295e+03, 8.23730540924475690e+02, 4.56321517456133948e-12] },
    // case 35: alt 572.800 km, lat -45.918, lon -72.893, doy 52, ap 269.252
    Case { doy: 52, sec: 4.26666666666666606e+03, alt: 5.72799999999999955e+02,
        g_lat: -4.59183673469387799e+01, g_long: -7.28925619834710687e+01, lst: -3.67431894704621964e+00,
        f107a: 2.48550295857988175e+02, f107: 9.42733564013840777e+01, ap: 2.69252077562326804e+02,
        ap_array: [5.67107750472589771e+01, 6.17107750472589771e+01, 6.67107750472589771e+01, 7.17107750472589771e+01, 7.67107750472589771e+01, 8.17107750472589771e+01, 8.67107750472589771e+01],
        switches: [0, 0, 1, 2, 0, 1, 2, 0, 1, -1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], out: [1.97380847065575188e+06, 1.00433083468223363e+07, 8.31843455628972297e+04, 1.49075552658457036e+03, 6.76081972988693436e-01, 2.89136301489131130e-16, 7.48189881607941061e+04, 2.24186825509898859e+05, 9.54727155027858498e+03, 1.02731846489999998e+03, 1.02730572781670844e+03, 2.89389877021506535e-16] },
    // case 36: alt 35.380 km, lat -20.204, lon -40.165, doy 234, ap 285.042
    Case { doy: 234, sec: 3.30666666666666642e+04, alt: 3.53800000000000026e+01,
        g_lat: -2.02040816326530575e+01, g_long: -4.01652892561983208e+01, lst: 9.18518518518518512e+00,
        f107a: 2.66627218934911241e+02, f107: 1.08096885813148788e+02, ap: 2.85041551246537381e+02,
        ap_array: [6.10586011342155075e+01, 6.60586011342155075e+01, 7.10586011342155075e+01, 7.60586011342155075e+01, 8.10586011342155075e+01, 8.60586011342155075e+01, 9.10586011342155075e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.61584567279742554e+11, 0.00000000000000000e+00, 1.28389288672414080e+17, 3.44429944721485280e+16, 1.53572847645508100e+15, 7.89912409561248206e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.41262957704730667e+02, 7.89912409561248206e-06] },
    // case 37: alt 105.180 km, lat +5.510, lon -7.438, doy 143, ap 1.662
    Case { doy: 143, sec: 6.18666666666666715e+04, alt: 1.05180000000000007e+02,
        g_lat: 5.51020408163263653e+00, g_long: -7.43801652892560128e+00, lst: 1.66893174165901463e+01,
        f107a: 2.84704142011834278e+02, f107: 1.21920415224913484e+02, ap: 1.66204986149584477e+00,
        ap_array: [6.54064272211720237e+01, 7.04064272211720237e+01, 7.54064272211720237e+01, 8.04064272211720237e+01, 8.54064272211720237e+01, 9.04064272211720237e+01, 9.54064272211720237e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.70823092724251598e+07, 3.47476513097706848e+11, 2.24815611309691699e+12, 3.06755835499099609e+11, 1.90339849718822517e+10, 1.31282333381353198e-10, 6.45530235219486989e+06, 4.78939447175496782e+05, 2.30764192280872956e-40, 1.02731846489999998e+03, 2.34981264577899651e+02, 1.31282333381353198e-10] },
    // case 38: alt 368.640 km, lat +31.224, lon +25.289, doy 326, ap 17.452
    Case { doy: 326, sec: 1.38666666666666661e+04, alt: 3.68639999999999986e+02,
        g_lat: 3.12244897959183589e+01, g_long: 2.52892561983470898e+01, lst: 5.53780226507499052e+00,
        f107a: 6.91715976331360878e+01, f107: 1.35743944636678208e+02, ap: 1.74515235457063724e+01,
        ap_array: [6.97542533081285399e+01, 7.47542533081285399e+01, 7.97542533081285399e+01, 8.47542533081285399e+01, 8.97542533081285399e+01, 9.47542533081285399e+01, 9.97542533081285399e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.48908709332776815e+06, 1.07870979073635459e+08, 5.68185248395355418e+06, 1.69540305550313846e+05, 3.02846535285657865e+02, 3.18522473965680611e-15, 1.74578201726337080e+05, 1.01530137054858194e+06, 1.16476294704934389e+03, 8.53947855604123220e+02, 8.52512355375855350e+02, 3.18525567576067964e-15] },
    // case 39: alt 476.800 km, lat +56.939, lon +58.017, doy 29, ap 33.241
    Case { doy: 29, sec: 4.26666666666666642e+04, alt: 4.76800000000000011e+02,
        g_lat: 5.69387755102040671e+01, g_long: 5.80165289256198378e+01, lst: 1.57196204468931739e+01,
        f107a: 8.72485207100591822e+01, f107: 1.49567474048442904e+02, ap: 3.32409972299168999e+01,
        ap_array: [7.41020793950850702e+01, 7.91020793950850702e+01, 8.41020793950850702e+01, 8.91020793950850702e+01, 9.41020793950850702e+01, 9.91020793950850702e+01, 1.04102079395085070e+02],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [2.38404149149531825e+06, 2.13704507246910594e+07, 3.11803747139027924e+05, 6.74860123918005138e+03, 4.46429839903221737e+00, 6.08498705300104642e-16, 7.84413251087848475e+04, 4.34073393065394077e+05, 7.94208586035213830e+03, 9.31525444867123042e+02, 9.31310465470053714e+02, 6.08709647100555695e-16] },
    // case 40: alt 23.780 km, lat +82.653, lon +90.744, doy 212, ap 49.030
    Case { doy: 212, sec: 7.14666666666666570e+04, alt: 2.37800000000000011e+01,
        g_lat: 8.26530612244897895e+01, g_long: 9.07438016528925573e+01, lst: 1.98518518518518476e+01,
        f107a: 1.05325443786982248e+02, f107: 1.63391003460207600e+02, ap: 4.90304709141274202e+01,
        ap_array: [7.84499054820415864e+01, 8.34499054820415864e+01, 8.84499054820415864e+01, 9.34499054820415864e+01, 9.84499054820415864e+01, 1.03449905482041586e+02, 1.08449905482041586e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.51557295386848047e+12, 0.00000000000000000e+00, 8.21904796187068032e+17, 2.20492399673104736e+17, 9.83121421959161400e+15, 5.05675204449955432e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.30835016556915747e+02, 5.05675204449955432e-05] },
    // case 41: alt 97.580 km, lat -67.959, lon +123.471, doy 120, ap 64.820
    Case { doy: 120, sec: 2.34666666666666642e+04, alt: 9.75799999999999983e+01,
        g_lat: -6.79591836734693970e+01, g_long: 1.23471074380165305e+02, lst: 1.47499234771962051e+01,
        f107a: 1.23402366863905328e+02, f107: 1.77214532871972324e+02, ap: 6.48199445983379405e+01,
        ap_array: [8.27977315689981168e+01, 8.77977315689981168e+01, 9.27977315689981168e+01, 9.77977315689981168e+01, 1.02797731568998117e+02, 1.07797731568998117e+02, 1.12797731568998117e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.25701805304972485e+08, 2.49983496564688354e+11, 1.31629514151363574e+13, 3.31852281835609863e+12, 1.44881407545620422e+11, 8.04354472254545566e-10, 1.83416538631160185e+07, 2.63812896222497744e+05, 1.10189180363563073e-44, 1.02731846489999998e+03, 1.63149701057112537e+02, 8.04354472254545566e-10] },
    // case 42: alt 323.840 km, lat -42.245, lon +156.198, doy 303, ap 80.609
    Case { doy: 303, sec: 5.22666666666666715e+04, alt: 3.23840000000000032e+02,
        g_lat: -4.22448979591836817e+01, g_long: 1.56198347107438053e+02, lst: 2.49317416590143885e+01,
        f107a: 1.41479289940828409e+02, f107: 1.91038062283737020e+02, ap: 8.06094182825484751e+01,
        ap_array: [8.71455576559546330e+01, 9.21455576559546330e+01, 9.71455576559546330e+01, 1.02145557655954633e+02, 1.07145557655954633e+02, 1.12145557655954633e+02, 1.17145557655954633e+02],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.47816447923228797e+06, 5.58652183621929884e+08, 1.73605477216585129e+08, 5.25348866539985407e+06, 1.08875007257355042e+05, 2.34656210278071161e-14, 4.04746389802764315e+04, 1.04363008326610476e+07, 2.99417591971187676e+02, 1.21734367183852373e+03, 1.20670845305793387e+03, 2.34656289803383594e-14] },
    // case 43: alt 956.800 km, lat -16.531, lon -168.099, doy 75, ap 96.399
    Case { doy: 75, sec: 8.10666666666666570e+04, alt: 9.56800000000000068e+02,
        g_lat: -1.65306122448979522e+01, g_long: -1.68099173553718998e+02, lst: 1.13119069482705825e+01,
        f107a: 1.59556213017751475e+02, f107: 2.04861591695501744e+02, ap: 9.63988919667589954e+01,
        ap_array: [9.14933837429111492e+01, 9.64933837429111492e+01, 1.01493383742911149e+02, 1.06493383742911149e+02, 1.11493383742911149e+02, 1.16493383742911149e+02, 1.21493383742911149e+02],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, -1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [5.55416092957898160e+05, 3.45215670602522005e+04, 3.27752179106957930e+00, 6.84487363468644534e-03, 4.62195930185597627e-07, 4.69112462064719354e-18, 3.62797308703529707e+04, 1.11410870044916510e+03, 2.81656478639685429e+03, 1.02731846489999998e+03, 1.02731845307603317e+03, 4.76593258137389408e-18] },
    // case 44: alt 12.180 km, lat +9.184, lon -135.372, doy 257, ap 112.188
    Case { doy: 257, sec: 7.46666666666666606e+03, alt: 1.21800000000000015e+01,
        g_lat: 9.18367346938774176e+00, g_long: -1.35371900826446279e+02, lst: 2.07407407407407396e+00,
        f107a: 1.77633136094674569e+02, f107: 2.18685121107266440e+02, ap: 1.12188365650969530e+02,
        ap_array: [9.58412098298676796e+01, 1.00841209829867680e+02, 1.05841209829867680e+02, 1.10841209829867680e+02, 1.15841209829867680e+02, 1.20841209829867680e+02, 1.25841209829867680e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.62954197555314688e+13, 0.00000000000000000e+00, 5.40857311220427469e+18, 1.45095790881104077e+18, 6.46946472816314320e+16, 3.32761327952384698e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.19295242788570903e+02, 3.32761327952384698e-04] },
    // case 45: alt 89.980 km, lat +34.898, lon -102.645, doy 166, ap 127.978
    Case { doy: 166, sec: 3.62666666666666642e+04, alt: 8.99800000000000040e+01,
        g_lat: 3.48979591836734642e+01, g_long: -1.02644628099173545e+02, lst: 3.23109886746250297e+00,
        f107a: 1.95710059171597635e+02, f107: 2.32508650519031136e+02, ap: 1.27977839335180050e+02,
        ap_array: [3.78071833648393196e-01, 5.37807183364839325e+00, 1.03780718336483933e+01, 1.53780718336483933e+01, 2.03780718336483915e+01, 2.53780718336483915e+01, 3.03780718336483915e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.33710790341889024e+08, 1.78342506284307343e+11, 6.06753906842431562e+13, 1.38368768820291348e+13, 7.00754796737530518e+11, 3.60647693284063574e-09, 5.72810000049168617e+07, 1.49192545049810346e+05, 5.53901040090296990e-49, 1.02731846489999998e+03, 1.66696608036191947e+02, 3.60647693284063574e-09] },
    // case 46: alt 279.040 km, lat +60.612, lon -69.917, doy 348, ap 143.767
    Case { doy: 348, sec: 6.50666666666666642e+04, alt: 2.79040000000000020e+02,
        g_lat: 6.06122448979591582e+01, g_long: -6.99173553719008396e+01, lst: 1.34129170492806828e+01,
        f107a: 2.13786982248520701e+02, f107: 2.46332179930795832e+02, ap: 1.43767313019390571e+02,
        ap_array: [4.72589792060491476e+00, 9.72589792060491476e+00, 1.47258979206049148e+01, 1.97258979206049148e+01, 2.47258979206049148e+01, 2.97258979206049148e+01, 3.47258979206049148e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.55344842375901388e+06, 1.05181897262632239e+09, 1.13446699077831578e+09, 3.66522541158077866e+07, 8.16166537462868262e+05, 8.32970434852632970e-14, 1.32208809883964805e+04, 2.66440533569972999e+07, 5.98776504594413961e+00, 1.54547010358922125e+03, 1.48354634295155461e+03, 8.32970436442983371e-14] },
    // case 47: alt 860.800 km, lat +86.327, lon -37.190, doy 18, ap 159.557
    Case { doy: 18, sec: 1.70666666666666642e+04, alt: 8.60800000000000068e+02,
        g_lat: 8.63265306122448806e+01, g_long: -3.71900826446280917e+01, lst: 2.26140189776553457e+00,
        f107a: 2.31863905325443767e+02, f107: 2.60155709342560556e+02, ap: 1.59556786703601091e+02,
        ap_array: [9.07372400756143627e+00, 1.40737240075614363e+01, 1.90737240075614380e+01, 2.40737240075614380e+01, 2.90737240075614380e+01, 3.40737240075614380e+01, 3.90737240075614380e+01],
        switches: [0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], out: [8.82145789679540438e+05, 2.52852247753573320e+05, 1.04820787152725572e+02, 6.92283802781735513e-01, 4.49560711794105928e-05, 1.29115404020816237e-17, 7.06957392615214922e+04, 9.29745627422091820e+03, 3.93623199797258121e+03, 1.02731846489999998e+03, 1.02731846478856460e+03, 1.30160867239477757e-17] },
    // case 48: alt 70.180 km, lat -89.475, lon -4.463, doy 200, ap 175.346
    Case { doy: 200, sec: 4.58666666666666570e+04, alt: 7.01800000000000068e+01,
        g_lat: -8.94752186588921319e+01, g_long: -4.46280991735534371e+00, lst: 1.27407407407407387e+01,
        f107a: 2.49940828402366861e+02, f107: 2.73979238754325252e+02, ap: 1.75346260387811640e+02,
        ap_array: [1.34215500945179596e+01, 1.84215500945179613e+01, 2.34215500945179613e+01, 2.84215500945179613e+01, 3.34215500945179613e+01, 3.84215500945179613e+01, 4.34215500945179613e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.62975091560602188e+09, 0.00000000000000000e+00, 6.88353302166851750e+14, 1.79008010105098875e+14, 8.23429131512373730e+12, 4.20503546663684194e-08, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.27235769939034356e+02, 4.20503546663684194e-08] },
    // case 49: alt 73.260 km, lat -63.761, lon +28.264, doy 109, ap 191.136
    Case { doy: 109, sec: 7.46666666666666570e+04, alt: 7.32600000000000051e+01,
        g_lat: -6.37609329446064166e+01, g_long: 2.82644628099173474e+01, lst: 2.26250382614018939e+01,
        f107a: 2.68017751479289927e+02, f107: 2.87802768166089947e+02, ap: 1.91135734072022132e+02,
        ap_array: [1.77693761814744811e+01, 2.27693761814744811e+01, 2.77693761814744811e+01, 3.27693761814744846e+01, 3.77693761814744846e+01, 4.27693761814744846e+01, 4.77693761814744846e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.03413135506813192e+09, 7.33917322918840777e+06, 5.92360204039540625e+14, 1.49206284716193812e+14, 7.06594437999647559e+12, 3.59279458171230573e-08, 1.17630822217860764e-19, 3.65138790011545643e+04, 4.66784299449576093e-62, 1.02731846489999998e+03, 2.26381933468305959e+02, 3.59279458171230573e-08] },
    // case 50: alt 180.480 km, lat -38.047, lon +60.992, doy 291, ap 206.925
    Case { doy: 291, sec: 2.66666666666666642e+04, alt: 1.80480000000000018e+02,
        g_lat: -3.80466472303207013e+01, g_long: 6.09917355371900669e+01, lst: 1.14735231098867452e+01,
        f107a: 2.86094674556213022e+02, f107: 6.74394463667820077e+01, ap: 2.06925207756232652e+02,
        ap_array: [2.21172022684310008e+01, 2.71172022684310008e+01, 3.21172022684310008e+01, 3.71172022684310008e+01, 4.21172022684310008e+01, 4.71172022684310008e+01, 5.21172022684310008e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.02774706278508231e+07, 1.87511936451623535e+10, 1.43969450843907452e+10, 3.54578894167726755e+08, 7.05515324544520527e+07, 1.19280409001885368e-12, 2.61525817550124659e+05, 8.66544806311608106e+07, 1.76423516936641104e-11, 1.03092686338958106e+03, 9.72878890526072155e+02, 1.19280409001885368e-12] },
    // case 51: alt 649.600 km, lat -12.332, lon +93.719, doy 63, ap 222.715
    Case { doy: 63, sec: 5.54666666666666642e+04, alt: 6.49600000000000023e+02,
        g_lat: -1.23323615160349931e+01, g_long: 9.37190082644628433e+01, lst: 2.16553412917049286e+01,
        f107a: 7.05621301775147884e+01, f107: 8.12629757785467177e+01, ap: 2.22714681440443172e+02,
        ap_array: [2.64650283553875205e+01, 3.14650283553875205e+01, 3.64650283553875170e+01, 4.14650283553875170e+01, 4.64650283553875170e+01, 5.14650283553875170e+01, 5.64650283553875170e+01],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, -1, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [8.66702145363175194e+05, 2.94435556870364642e+05, 1.53186406422241532e+02, 1.09128982784416584e+00, 8.03694498980001774e-05, 1.39334842182816204e-17, 6.56236530538862135e+04, 1.04242788167430590e+04, 8.14956571975811130e+03, 7.93355485311066673e+02, 7.93355206241250585e+02, 1.41499366837983975e-17] },
    // case 52: alt 44.660 km, lat +13.382, lon +126.446, doy 246, ap 238.504
    Case { doy: 246, sec: 8.42666666666666570e+04, alt: 4.46600000000000108e+01,
        g_lat: 1.33819241982507151e+01, g_long: 1.26446280991735534e+02, lst: 2.34074074074074048e+01,
        f107a: 8.86390532544378686e+01, f107: 9.50865051903114136e+01, ap: 2.38504155124653721e+02,
        ap_array: [3.08128544423440438e+01, 3.58128544423440474e+01, 4.08128544423440474e+01, 4.58128544423440474e+01, 5.08128544423440474e+01, 5.58128544423440474e+01, 6.08128544423440474e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.39106340657809967e+11, 0.00000000000000000e+00, 3.56305047234588800e+16, 9.55859550216231400e+15, 4.26194282249653375e+14, 2.19215934062870962e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.64668348215697108e+02, 2.19215934062870962e-06] },
    // case 53: alt 111.260 km, lat +39.096, lon +159.174, doy 154, ap 254.294
    Case { doy: 154, sec: 2.13333333333333303e+03, alt: 1.11260000000000005e+02,
        g_lat: 3.90962099125364375e+01, g_long: 1.59173553719008282e+02, lst: 1.12041628405264788e+01,
        f107a: 1.06715976331360949e+02, f107: 1.08910034602076109e+02, ap: 2.54293628808864241e+02,
        ap_array: [3.51606805293005635e+01, 4.01606805293005635e+01, 4.51606805293005635e+01, 5.01606805293005635e+01, 5.51606805293005635e+01, 6.01606805293005635e+01, 6.51606805293005635e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.08327390662880279e+07, 1.09827293520246216e+11, 1.00476295575565344e+12, 1.68270740604788177e+11, 7.06702782225836468e+09, 5.90263536959425762e-11, 4.91743813550566323e+06, 8.48358995711794239e+05, 1.25736580305359376e-36, 1.02731846489999998e+03, 2.95278830978919018e+02, 5.90263536959425762e-11] },
    // case 54: alt 135.680 km, lat +64.810, lon -165.124, doy 337, ap 270.083
    Case { doy: 337, sec: 3.09333333333333321e+04, alt: 1.35680000000000007e+02,
        g_lat: 6.48104956268221599e+01, g_long: -1.65123966942148769e+02, lst: -2.41567187021732543e+00,
        f107a: 1.24792899408284029e+02, f107: 1.22733564013840834e+02, ap: 2.70083102493074762e+02,
        ap_array: [3.95085066162570797e+01, 4.45085066162570797e+01, 4.95085066162570797e+01, 5.45085066162570797e+01, 5.95085066162570797e+01, 6.45085066162570797e+01, 6.95085066162570797e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.38774051905694325e+06, 9.82047424788674355e+09, 9.33895912830038910e+10, 1.71195666975653324e+10, 6.30882035869560361e+08, 5.55290550107258329e-12, 2.41148800613426865e+05, 6.03147772191434284e+05, 1.78820014001088448e-26, 1.56893913016163333e+03, 7.11574425322521279e+02, 5.55290550107258329e-12] },
    // case 55: alt 553.600 km, lat -85.802, lon -132.397, doy 40, ap 285.873
    Case { doy: 40, sec: 5.97333333333333285e+04, alt: 5.53600000000000023e+02,
        g_lat: -8.58017492711370267e+01, g_long: -1.32396694214876050e+02, lst: 7.76614631160085445e+00,
        f107a: 1.42869822485207123e+02, f107: 1.36557093425605558e+02, ap: 2.85872576177285282e+02,
        ap_array: [4.38563327032136101e+01, 4.88563327032136101e+01, 5.38563327032136101e+01, 5.88563327032136101e+01, 6.38563327032136101e+01, 6.88563327032136101e+01, 7.38563327032136101e+01],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [2.51730182120196667e+05, 2.12568102633898752e+06, 1.07246331963666540e+05, 2.61950504055231067e+03, 5.74595732371069179e+00, 6.40608770549821058e-17, 2.61811479795424057e+04, 3.28528714071770737e+04, 1.57149814186694766e+03, 1.02731846489999998e+03, 1.02728114109394937e+03, 6.41026160456300948e-17] },
    // case 56: alt 33.060 km, lat -60.087, lon -99.669, doy 223, ap 2.493
    Case { doy: 223, sec: 1.17333333333333321e+04, alt: 3.30600000000000023e+01,
        g_lat: -6.00874635568513185e+01, g_long: -9.96694214876033016e+01, lst: 3.25925925925925908e+00,
        f107a: 1.60946745562130189e+02, f107: 1.50380622837370254e+02, ap: 2.49307479224376749e+00,
        ap_array: [4.82041587901701334e+01, 5.32041587901701334e+01, 5.82041587901701334e+01, 6.32041587901701334e+01, 6.82041587901701405e+01, 7.32041587901701405e+01, 7.82041587901701405e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.05905578520308838e+11, 0.00000000000000000e+00, 1.34993797762432176e+17, 3.62147861257276800e+16, 1.61472831193533250e+15, 8.30546513412204928e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.24379117830659339e+02, 8.30546513412204928e-06] },
    // case 57: alt 103.660 km, lat -34.373, lon -66.942, doy 132, ap 18.283
    Case { doy: 132, sec: 4.05333333333333285e+04, alt: 1.03660000000000011e+02,
        g_lat: -3.43731778425655961e+01, g_long: -6.69421487603305962e+01, lst: 6.79644934190388472e+00,
        f107a: 1.79023668639053255e+02, f107: 1.64204152249134950e+02, ap: 1.82825484764542949e+01,
        ap_array: [5.25519848771266567e+01, 5.75519848771266567e+01, 6.25519848771266567e+01, 6.75519848771266567e+01, 7.25519848771266567e+01, 7.75519848771266567e+01, 8.25519848771266567e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.78969461176665127e+07, 4.82956206827792969e+11, 3.34318206226728760e+12, 6.36815359844141357e+11, 2.92437895194566345e+10, 2.03988445444321551e-10, 9.53242054164853692e+06, 3.15758985407508968e+05, 3.27609621199649316e-41, 1.02731846489999998e+03, 2.17306191281012104e+02, 2.03988445444321551e-10] },
    // case 58: alt 359.680 km, lat -8.659, lon -34.215, doy 314, ap 34.072
    Case { doy: 314, sec: 6.93333333333333285e+04, alt: 3.59680000000000064e+02,
        g_lat: -8.65889212827988786e+00, g_long: -3.42148760330578625e+01, lst: 1.69782675237220673e+01,
        f107a: 1.97100591715976350e+02, f107: 1.78027681660899646e+02, ap: 3.40720221606648224e+01,
        ap_array: [5.68998109640831728e+01, 6.18998109640831728e+01, 6.68998109640831728e+01, 7.18998109640831728e+01, 7.68998109640831728e+01, 8.18998109640831728e+01, 8.68998109640831728e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.27571520029025618e+06, 5.14254472774092853e+08, 5.96804832940942198e+07, 7.61476461249349755e+05, 7.82824112914497073e+03, 1.67482445464449101e-14, 4.95276004312323639e+04, 1.00247201581893563e+07, 1.51736820462013634e+03, 1.20439713523059982e+03, 1.19848304435141927e+03, 1.67482848477444250e-14] },
    // case 59: alt 457.600 km, lat +17.055, lon -1.488, doy 86, ap 49.861
    Case { doy: 86, sec: 2.13333333333333321e+04, alt: 4.57600000000000023e+02,
        g_lat: 1.70553935860058203e+01, g_long: -1.48760330578511457e+00, lst: 5.82675237220691766e+00,
        f107a: 2.15177514792899416e+02, f107: 1.91851211072664370e+02, ap: 4.98614958448753427e+01,
        ap_array: [6.12476370510397032e+01, 6.62476370510397032e+01, 7.12476370510397032e+01, 7.62476370510397032e+01, 8.12476370510397032e+01, 8.62476370510397032e+01, 9.12476370510397032e+01],
        switches: [0, 0, 1, 2, 0, 1, 2, 0, 1, -1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], out: [3.21554573678386444e+06, 6.71720883142358214e+07, 2.25394037268522056e+06, 6.43927295422419775e+04, 7.46572293337918040e+01, 1.94137216214174378e-15, 8.59140757116332097e+04, 1.18756488390676631e+06, 6.87232606158443377e+03, 1.02731846489999998e+03, 1.02717825497717149e+03, 1.94155469112193978e-15] },
    // case 60: alt 21.460 km, lat +42.770, lon +31.240, doy 269, ap 65.651
    Case { doy: 269, sec: 5.01333333333333358e+04, alt: 2.14600000000000044e+01,
        g_lat: 4.27696793002915285e+01, g_long: 3.12396694214875765e+01, lst: 1.39259259259259274e+01,
        f107a: 2.33254437869822482e+02, f107: 2.05674740484429094e+02, ap: 6.56509695290858701e+01,
        ap_array: [6.55954631379962194e+01, 7.05954631379962194e+01, 7.55954631379962194e+01, 8.05954631379962194e+01, 8.55954631379962194e+01, 9.05954631379962194e+01, 9.55954631379962194e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.13679268716271777e+12, 0.00000000000000000e+00, 1.21250665907126477e+18, 3.25279161429048192e+17, 1.45033984024809340e+16, 7.45992182509716117e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.15715336525437721e+02, 7.45992182509716117e-05] },
    // case 61: alt 96.060 km, lat +68.484, lon +63.967, doy 177, ap 81.440
    Case { doy: 177, sec: 7.89333333333333285e+04, alt: 9.60600000000000023e+01,
        g_lat: 6.84839650145772509e+01, g_long: 6.39669421487603245e+01, lst: 2.61903887358432783e+01,
        f107a: 2.51331360946745576e+02, f107: 2.19498269896193790e+02, ap: 8.14404432132963905e+01,
        ap_array: [6.99432892249527356e+01, 7.49432892249527356e+01, 7.99432892249527356e+01, 8.49432892249527356e+01, 8.99432892249527356e+01, 9.49432892249527356e+01, 9.99432892249527356e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.48543559115739316e+07, 8.19620485827919464e+10, 1.13190777623077129e+13, 2.16190452536124927e+12, 1.21634323741564987e+11, 6.51205122821052789e-10, 1.35866499419251550e+07, 1.26860572798116977e+05, 6.14737435064400941e-44, 1.02731846489999998e+03, 1.60104620796653109e+02, 6.51205122821052789e-10] },
    // case 62: alt 314.880 km, lat -82.128, lon +96.694, doy 360, ap 97.230
    Case { doy: 360, sec: 5.33333333333333303e+03, alt: 3.14879999999999995e+02,
        g_lat: -8.21282798833819214e+01, g_long: 9.66942148760330724e+01, lst: 7.92776247321701888e+00,
        f107a: 2.69408284023668671e+02, f107: 2.33321799307958486e+02, ap: 9.72299168975069108e+01,
        ap_array: [7.42911153119092660e+01, 7.92911153119092660e+01, 8.42911153119092660e+01, 8.92911153119092660e+01, 9.42911153119092660e+01, 9.92911153119092660e+01, 1.04291115311909266e+02],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.93586940429453831e+06, 7.62084878798347354e+08, 5.05606468967328191e+08, 9.82387899762988277e+06, 4.09809430043757660e+05, 4.48209357928437510e-14, 2.55083919164049803e+04, 2.19785034025009535e+07, 7.74195953146848296e+02, 1.43041518129939300e+03, 1.41798433662907769e+03, 4.48209563554882678e-14] },
    // case 63: alt 937.600 km, lat -56.414, lon +129.421, doy 3, ap 113.019
    Case { doy: 3, sec: 3.41333333333333285e+04, alt: 9.37600000000000023e+02,
        g_lat: -5.64139941690962132e+01, g_long: 1.29421487603305820e+02, lst: 1.81095806550352023e+01,
        f107a: 2.87485207100591765e+02, f107: 2.47145328719723182e+02, ap: 1.13019390581717431e+02,
        ap_array: [7.86389413988657822e+01, 8.36389413988657822e+01, 8.86389413988657822e+01, 9.36389413988657822e+01, 9.86389413988657822e+01, 1.03638941398865782e+02, 1.08638941398865782e+02],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [8.80613902510482818e+05, 7.08498934323668014e+05, 1.07600377946720528e+03, 1.09550715029849570e+01, 1.50073171282992502e-03, 2.52879110561867614e-17, 5.10111263112755478e+04, 2.09823532481649672e+04, 3.01535142331752104e+03, 1.37545041365071461e+03, 1.37545027263603492e+03, 2.53679987899900768e-17] },
    // case 64: alt 9.860 km, lat -30.700, lon +162.149, doy 186, ap 128.809
    Case { doy: 186, sec: 6.29333333333333358e+04, alt: 9.86000000000000121e+00,
        g_lat: -3.06997084548105050e+01, g_long: 1.62148760330578511e+02, lst: 1.74814814814814810e+01,
        f107a: 7.19526627218934891e+01, f107: 2.60968858131487877e+02, ap: 1.28808864265927951e+02,
        ap_array: [8.29867674858223126e+01, 8.79867674858223126e+01, 9.29867674858223126e+01, 9.79867674858223126e+01, 1.02986767485822313e+02, 1.07986767485822313e+02, 1.12986767485822313e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.69058395407764609e+13, 0.00000000000000000e+00, 6.98968807233448346e+18, 1.87512361916517197e+18, 8.36071538772289280e+16, 4.30039094724383292e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.29521579711928439e+02, 4.30039094724383292e-04] },
    // case 65: alt 88.460 km, lat -4.985, lon -162.149, doy 95, ap 144.598
    Case { doy: 95, sec: 1.49333333333333321e+04, alt: 8.84600000000000080e+01,
        g_lat: -4.98542274052479684e+00, g_long: -1.62148760330578511e+02, lst: -6.66176920722375332e+00,
        f107a: 9.00295857988165693e+01, f107: 2.74792387543252630e+02, ap: 1.44598337950138500e+02,
        ap_array: [8.73345935727788287e+01, 9.23345935727788287e+01, 9.73345935727788287e+01, 1.02334593572778829e+02, 1.07334593572778829e+02, 1.12334593572778829e+02, 1.17334593572778829e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.92759330037326813e+08, 2.60856446465118683e+11, 6.53118620845876484e+13, 1.64564785583877910e+13, 7.60006768923819946e+11, 3.96725968120512815e-09, 7.01462665560424924e+07, 2.34618144297023799e+05, 8.94856623008910338e-51, 1.02731846489999998e+03, 1.71800007360245303e+02, 3.96725968120512815e-09] },
    // case 66: alt 270.080 km, lat +20.729, lon -129.421, doy 277, ap 160.388
    Case { doy: 277, sec: 4.37333333333333358e+04, alt: 2.70080000000000041e+02,
        g_lat: 2.07288629737609398e+01, g_long: -1.29421487603305792e+02, lst: 3.52004897459442923e+00,
        f107a: 1.08106508875739650e+02, f107: 2.88615916955017269e+02, ap: 1.60387811634349021e+02,
        ap_array: [9.16824196597353449e+01, 9.66824196597353449e+01, 1.01682419659735345e+02, 1.06682419659735345e+02, 1.11682419659735345e+02, 1.16682419659735345e+02, 1.21682419659735345e+02],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.74838861346323416e+06, 2.01325358183108139e+09, 5.28542761596834362e+08, 2.17294864991568066e+07, 2.90957081297033001e+05, 7.97514142899807111e-14, 5.55893357015726360e+04, 2.12667876851262823e+07, 3.99656342092759287e+00, 1.23333348635429707e+03, 1.16195621134498333e+03, 7.97514143961294306e-14] },
    // case 67: alt 841.600 km, lat +46.443, lon -96.694, doy 49, ap 176.177
    Case { doy: 49, sec: 7.25333333333333139e+04, alt: 8.41600000000000136e+02,
        g_lat: 4.64431486880466480e+01, g_long: -9.66942148760330582e+01, lst: 1.37018671564126038e+01,
        f107a: 1.26183431952662716e+02, f107: 6.82525951557093435e+01, ap: 1.76177285318559569e+02,
        ap_array: [9.60302457466918753e+01, 1.01030245746691875e+02, 1.06030245746691875e+02, 1.11030245746691875e+02, 1.16030245746691875e+02, 1.21030245746691875e+02, 1.26030245746691875e+02],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, -1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [5.08286492995681183e+05, 1.26524863053259149e+05, 5.76277292045173084e+01, 2.39496642836626500e-01, 2.88021600615243642e-05, 7.01988703411584224e-18, 3.41728383391886600e+04, 9.67926787690814854e+03, 4.25276252980688423e+03, 1.02731846489999998e+03, 1.02731835711987264e+03, 7.13284040690751274e-18] },
    // case 68: alt 67.860 km, lat +72.157, lon -63.967, doy 231, ap 191.967
    Case { doy: 231, sec: 2.45333333333333321e+04, alt: 6.78599999999999994e+01,
        g_lat: 7.21574344023323704e+01, g_long: -6.39669421487603387e+01, lst: 6.81481481481481488e+00,
        f107a: 1.44260355029585810e+02, f107: 8.20761245674740536e+01, ap: 1.91966759002770090e+02,
        ap_array: [5.67107750472589878e-01, 5.56710775047258988e+00, 1.05671077504725908e+01, 1.55671077504725908e+01, 2.05671077504725908e+01, 2.55671077504725908e+01, 3.05671077504725908e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.44868092143377190e+10, 0.00000000000000000e+00, 2.16563809124980000e+15, 5.77806967820526000e+14, 2.59100809569378359e+13, 1.33072490179870933e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.31503626003725600e+02, 1.33072490179870933e-07] },
    // case 69: alt 80.860 km, lat -78.455, lon -31.240, doy 140, ap 207.756
    Case { doy: 140, sec: 5.33333333333333285e+04, alt: 8.08599999999999994e+01,
        g_lat: -7.84548104956268162e+01, g_long: -3.12396694214876050e+01, lst: 1.27321701867156385e+01,
        f107a: 1.62337278106508876e+02, f107: 9.58996539792387495e+01, ap: 2.07756232686980582e+02,
        ap_array: [4.91493383742911139e+00, 9.91493383742911050e+00, 1.49149338374291105e+01, 1.99149338374291105e+01, 2.49149338374291105e+01, 2.99149338374291105e+01, 3.49149338374291105e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.05178308922941148e+09, 3.38621202317847109e+09, 1.47558445380184000e+14, 3.77069372291642188e+13, 1.75209525751292847e+12, 8.97794514788312167e-09, 3.21087849556690119e+07, 4.16727757592008056e+04, 1.79363435811192922e-56, 1.02731846489999998e+03, 2.25364028491213816e+02, 8.97794514788312167e-09] },
    // case 70: alt 225.280 km, lat -52.741, lon +1.488, doy 323, ap 223.546
    Case { doy: 323, sec: 8.21333333333333285e+04, alt: 2.25280000000000001e+02,
        g_lat: -5.27405247813411080e+01, g_long: 1.48760330578517141e+00, lst: 2.29139883685338255e+01,
        f107a: 1.80414201183431942e+02, f107: 1.09723183391003460e+02, ap: 2.23545706371191130e+02,
        ap_array: [9.26275992438563378e+00, 1.42627599243856338e+01, 1.92627599243856338e+01, 2.42627599243856338e+01, 2.92627599243856338e+01, 3.42627599243856338e+01, 3.92627599243856338e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.08645080867880955e+06, 2.53521901184001446e+09, 3.62966149193074560e+09, 1.92353476980378330e+08, 1.05157203501258492e+07, 2.47231632988609177e-13, 3.45610414981579015e+04, 9.45516162016960792e+06, 1.25536225269459240e-04, 1.34617816011537957e+03, 1.26237147006519717e+03, 2.47231632988612509e-13] },
    // case 71: alt 745.600 km, lat -27.026, lon +34.215, doy 26, ap 239.335
    Case { doy: 26, sec: 8.53333333333333212e+03, alt: 7.45600000000000023e+02,
        g_lat: -2.70262390670553998e+01, g_long: 3.42148760330578625e+01, lst: 4.65136210590756072e+00,
        f107a: 1.98491124260355065e+02, f107: 1.23546712802768155e+02, ap: 2.39335180055401679e+02,
        ap_array: [1.36105860113421553e+01, 1.86105860113421571e+01, 2.36105860113421571e+01, 2.86105860113421571e+01, 3.36105860113421571e+01, 3.86105860113421571e+01, 4.36105860113421571e+01],
        switches: [0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], out: [1.29318551302120788e+06, 1.27012875323023507e+06, 1.84357237639591244e+03, 1.84887504876973807e+01, 2.74182992057996544e-03, 4.34146063371343914e-17, 7.57627523801216157e+04, 3.79000485107239801e+04, 6.00777749686404604e+03, 1.02731846489999998e+03, 1.02731845626474660e+03, 4.35741729074511064e-17] },
    // case 72: alt 56.260 km, lat -1.312, lon +66.942, doy 209, ap 255.125
    Case { doy: 209, sec: 3.73333333333333285e+04, alt: 5.62600000000000122e+01,
        g_lat: -1.31195335276967739e+00, g_long: 6.69421487603305536e+01, lst: 1.03703703703703702e+01,
        f107a: 2.16568047337278131e+02, f107: 1.37370242214532880e+02, ap: 2.55124653739612171e+02,
        ap_array: [1.79584120982986768e+01, 2.29584120982986768e+01, 2.79584120982986768e+01, 3.29584120982986803e+01, 3.79584120982986803e+01, 4.29584120982986803e+01, 4.79584120982986803e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.37863220737580719e+10, 0.00000000000000000e+00, 8.01498528827876500e+15, 2.15018010328654425e+15, 9.58712465257464531e+13, 4.93120291196266436e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.54746328340412902e+02, 4.93120291196266436e-07] },
    // case 73: alt 118.860 km, lat +24.402, lon +99.669, doy 117, ap 270.914
    Case { doy: 117, sec: 6.61333333333333285e+04, alt: 1.18860000000000014e+02,
        g_lat: 2.44023323615160166e+01, g_long: 9.96694214876033584e+01, lst: 2.50149984695439258e+01,
        f107a: 2.34644970414201197e+02, f107: 1.51193771626297575e+02, ap: 2.70914127423822720e+02,
        ap_array: [2.23062381852551965e+01, 2.73062381852551965e+01, 3.23062381852551965e+01, 3.73062381852551965e+01, 4.23062381852551965e+01, 4.73062381852551965e+01, 5.23062381852551965e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.90584199135176018e+07, 1.27002827571476212e+11, 4.47920185288286621e+11, 4.39465612608455658e+10, 1.95990515087892938e+09, 2.66574158524162698e-11, 3.03611124189214269e+06, 2.02736105564025720e+06, 1.99736613997456865e-33, 1.02731846489999998e+03, 3.62392084904919614e+02, 2.66574158524162698e-11] },
    // case 74: alt 126.720 km, lat +50.117, lon +132.397, doy 300, ap 286.704
    Case { doy: 300, sec: 1.81333333333333321e+04, alt: 1.26719999999999999e+02,
        g_lat: 5.01166180758017106e+01, g_long: 1.32396694214876050e+02, lst: 1.38634833180287735e+01,
        f107a: 2.52721893491124291e+02, f107: 1.65017301038062271e+02, ap: 2.86703601108033240e+02,
        ap_array: [2.66540642722117198e+01, 3.16540642722117198e+01, 3.66540642722117198e+01, 4.16540642722117198e+01, 4.66540642722117198e+01, 5.16540642722117198e+01, 5.66540642722117198e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.18753712436580509e+07, 6.00553462987133408e+10, 2.10225451232400787e+11, 2.30369277614275589e+10, 3.11204390711643934e+08, 1.26110259911771316e-11, 8.37651187388618593e+05, 6.23076613608962297e+06, 4.55192860941387167e-29, 1.31874722476339275e+03, 6.17331976396652635e+02, 1.26110259911771316e-11] },
    // case 75: alt 534.400 km, lat +75.831, lon +165.124, doy 72, ap 3.324
    Case { doy: 72, sec: 4.69333333333333285e+04, alt: 5.34399999999999977e+02,
        g_lat: 7.58309037900874330e+01, g_long: 1.65123966942148797e+02, lst: 2.40453014998469570e+01,
        f107a: 2.70798816568047357e+02, f107: 1.78840830449826967e+02, ap: 3.32409972299168954e+00,
        ap_array: [3.10018903591682431e+01, 3.60018903591682431e+01, 4.10018903591682431e+01, 4.60018903591682431e+01, 5.10018903591682431e+01, 5.60018903591682431e+01, 6.10018903591682431e+01],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, -1, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [2.63169078453786019e+06, 5.46366603208055794e+07, 2.12271785827470105e+06, 6.36802663195209461e+04, 7.60322967111522843e+01, 1.59269515446633433e-15, 6.77845709776294243e+04, 9.42634974379568128e+05, 9.60045262630376419e+03, 1.34466082996784303e+03, 1.34445001607352515e+03, 1.59295014248808894e-15] },
    // case 76: alt 30.740 km, lat -74.781, lon -159.174, doy 254, ap 19.114
    Case { doy: 254, sec: 7.57333333333333285e+04, alt: 3.07400000000000020e+01,
        g_lat: -7.47813411078717252e+01, g_long: -1.59173553719008254e+02, lst: 2.10370370370370345e+01,
        f107a: 2.88875739644970395e+02, f107: 1.92664359861591691e+02, ap: 1.91135734072022174e+01,
        ap_array: [3.53497164461247593e+01, 4.03497164461247593e+01, 4.53497164461247593e+01, 5.03497164461247593e+01, 5.53497164461247593e+01, 6.03497164461247593e+01, 6.53497164461247593e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.23563510826976685e+12, 0.00000000000000000e+00, 1.84128544816318784e+17, 4.93961647178738400e+16, 2.20245358882039700e+15, 1.13284701557907496e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.28493114138742754e+02, 1.13284701557907496e-05] },
    // case 77: alt 102.140 km, lat -49.067, lon -126.446, doy 163, ap 34.903
    Case { doy: 163, sec: 2.77333333333333321e+04, alt: 1.02140000000000001e+02,
        g_lat: -4.90670553935860099e+01, g_long: -1.26446280991735534e+02, lst: -7.26048362411998660e-01,
        f107a: 7.33431952662721898e+01, f107: 2.06487889273356416e+02, ap: 3.49030470914127449e+01,
        ap_array: [3.96975425330812826e+01, 4.46975425330812826e+01, 4.96975425330812826e+01, 5.46975425330812826e+01, 5.96975425330812826e+01, 6.46975425330812755e+01, 6.96975425330812755e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [7.88455289813699424e+07, 3.77608344308057007e+11, 4.35222808320055811e+12, 1.05693034249302014e+12, 4.25100228138211517e+10, 2.71288194415369448e-10, 1.32120065101047996e+07, 2.02628025201437034e+05, 1.88215642512911834e-42, 1.02731846489999998e+03, 2.02741225197010607e+02, 2.71288194415369448e-10] },
    // case 78: alt 350.720 km, lat -23.353, lon -93.719, doy 346, ap 50.693
    Case { doy: 346, sec: 5.65333333333333358e+04, alt: 3.50720000000000027e+02,
        g_lat: -2.33527696793002946e+01, g_long: -9.37190082644628006e+01, lst: 9.45576981940618388e+00,
        f107a: 9.14201183431952700e+01, f107: 2.20311418685121112e+02, ap: 5.06925207756232652e+01,
        ap_array: [4.40453686200378058e+01, 4.90453686200378058e+01, 5.40453686200378058e+01, 5.90453686200378058e+01, 6.40453686200378058e+01, 6.90453686200378058e+01, 7.40453686200378058e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.41989647701527597e+06, 2.73312481428258538e+08, 2.59691129513359144e+07, 9.23765630786943133e+05, 3.34683352378576092e+03, 8.71697673056330949e-15, 1.07951639797796452e+05, 7.68385892718762066e+06, 1.23491948720007917e+03, 1.03271004017237351e+03, 1.02626314393466987e+03, 8.71700953002488867e-15] },
    // case 79: alt 438.400 km, lat +2.362, lon -60.992, doy 15, ap 66.482
    Case { doy: 15, sec: 8.53333333333333285e+04, alt: 4.38399999999999977e+02,
        g_lat: 2.36151603498542784e+00, g_long: -6.09917355371900811e+01, lst: 1.96375880012243655e+01,
        f107a: 1.09497041420118350e+02, f107: 2.34134948096885807e+02, ap: 6.64819944598337997e+01,
        ap_array: [4.83931947069943291e+01, 5.33931947069943291e+01, 5.83931947069943291e+01, 6.33931947069943291e+01, 6.83931947069943362e+01, 7.33931947069943362e+01, 7.83931947069943362e+01],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [4.39631415749475546e+06, 8.33699821274409294e+07, 2.64763052305630967e+06, 3.58135080292223138e+04, 9.94833834511295123e+01, 2.40586008667271762e-15, 6.59378901592146576e+04, 1.60419503053467022e+06, 5.12689843389314337e+04, 1.02731846489999998e+03, 1.02687379171498515e+03, 2.40722179089675949e-15] },
    // case 80: alt 19.140 km, lat +28.076, lon -28.264, doy 197, ap 82.271
    Case { doy: 197, sec: 3.55555555555555486e+02, alt: 1.91400000000000006e+01,
        g_lat: 2.80758017492711218e+01, g_long: -2.82644628099173474e+01, lst: 9.87654320987654266e-02,
        f107a: 1.27573964497041430e+02, f107: 2.47958477508650503e+02, ap: 8.22714681440443201e+01,
        ap_array: [5.27410207939508524e+01, 5.77410207939508524e+01, 6.27410207939508524e+01, 6.77410207939508524e+01, 7.27410207939508524e+01, 7.77410207939508524e+01, 8.27410207939508524e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.23913008340953164e+13, 0.00000000000000000e+00, 1.84649349609212083e+18, 4.95358810196633280e+17, 2.20868319535059040e+16, 1.13605125615957273e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.09161532155301586e+02, 1.13605125615957273e-04] },
    // case 81: alt 94.540 km, lat +53.790, lon +4.463, doy 106, ap 98.061
    Case { doy: 106, sec: 2.91555555555555511e+04, alt: 9.45400000000000063e+01,
        g_lat: 5.37900874635568300e+01, g_long: 4.46280991735540056e+00, lst: 8.39628609325579056e+00,
        f107a: 1.45650887573964496e+02, f107: 2.61782006920415199e+02, ap: 9.80609418282548404e+01,
        ap_array: [5.70888468809073757e+01, 6.20888468809073757e+01, 6.70888468809073686e+01, 7.20888468809073686e+01, 7.70888468809073686e+01, 8.20888468809073686e+01, 8.70888468809073686e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.87986665463072032e+08, 3.19196552168721680e+11, 2.26047144306881289e+13, 5.31927975615945605e+12, 2.56646886573447937e+11, 1.35874777878399686e-09, 2.57788690868138261e+07, 2.87433847329405369e+05, 4.29744136081851220e-46, 1.02731846489999998e+03, 1.63252672851754198e+02, 1.35874777878399686e-09] },
    // case 82: alt 305.920 km, lat +79.504, lon +37.190, doy 289, ap 113.850
    Case { doy: 289, sec: 5.79555555555555475e+04, alt: 3.05920000000000073e+02,
        g_lat: 7.95043731778425524e+01, g_long: 3.71900826446280917e+01, lst: 1.85781042750739687e+01,
        f107a: 1.63727810650887591e+02, f107: 2.75605536332179895e+02, ap: 1.13850415512465375e+02,
        ap_array: [6.14366729678639061e+01, 6.64366729678638990e+01, 7.14366729678638990e+01, 7.64366729678638990e+01, 8.14366729678638990e+01, 8.64366729678638990e+01, 9.14366729678638990e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.69336343127778778e+06, 5.93415647424546123e+08, 5.72023114452227712e+08, 2.70321793887228407e+07, 5.35056476621950627e+05, 4.40198773070444188e-14, 1.92693010406748472e+04, 8.10543512516614422e+06, 4.96090661492344083e+01, 1.56885271737379253e+03, 1.50279256001857402e+03, 4.40198786246612193e-14] },
    // case 83: alt 918.400 km, lat -71.108, lon +69.917, doy 60, ap 129.640
    Case { doy: 60, sec: 9.95555555555555475e+03, alt: 9.18400000000000091e+02,
        g_lat: -7.11078717201166199e+01, g_long: 6.99173553719008396e+01, lst: 7.42658912355882084e+00,
        f107a: 1.81804733727810657e+02, f107: 2.89429065743944648e+02, ap: 1.29639889196675881e+02,
        ap_array: [6.57844990548204294e+01, 7.07844990548204294e+01, 7.57844990548204294e+01, 8.07844990548204294e+01, 8.57844990548204294e+01, 9.07844990548204294e+01, 9.57844990548204294e+01],
        switches: [0, 0, 1, 2, 0, 1, 2, 0, 1, -1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], out: [5.86115932718215161e+05, 7.00215460136723123e+04, 1.32407655198014336e+01, 6.73157645019715201e-02, 2.48786805447725085e-06, 5.91527946188989203e-18, 5.71559176012878670e+04, 2.93457730377378994e+03, 3.20571207949065547e+03, 1.02731846489999998e+03, 1.02731846184028973e+03, 6.00042317472116304e-18] },
    // case 84: alt 7.540 km, lat -45.394, lon +102.645, doy 243, ap 145.429
    Case { doy: 243, sec: 3.87555555555555547e+04, alt: 7.54000000000000092e+00,
        g_lat: -4.53935860058309046e+01, g_long: 1.02644628099173588e+02, lst: 1.07654320987654302e+01,
        f107a: 1.99881656804733751e+02, f107: 6.90657439446366794e+01, ap: 1.45429362880886430e+02,
        ap_array: [7.01323251417769455e+01, 7.51323251417769455e+01, 8.01323251417769455e+01, 8.51323251417769455e+01, 9.01323251417769455e+01, 9.51323251417769455e+01, 1.00132325141776946e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.87192543564220703e+13, 0.00000000000000000e+00, 8.75006770606165606e+18, 2.34738066350516070e+18, 1.04663934866056496e+17, 5.38346082994095088e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.32114421861694296e+02, 5.38346082994095088e-04] },
    // case 85: alt 86.940 km, lat -19.679, lon +135.372, doy 152, ap 161.219
    Case { doy: 152, sec: 6.75555555555555475e+04, alt: 8.69399999999999977e+01,
        g_lat: -1.96793002915451893e+01, g_long: 1.35371900826446335e+02, lst: 2.77902254871951868e+01,
        f107a: 2.17958579881656817e+02, f107: 8.28892733564013895e+01, ap: 1.61218836565096950e+02,
        ap_array: [7.44801512287334617e+01, 7.94801512287334617e+01, 8.44801512287334617e+01, 8.94801512287334617e+01, 9.44801512287334617e+01, 9.94801512287334617e+01, 1.04480151228733462e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.25244930596180081e+08, 1.82910013188647949e+11, 8.31612555549162969e+13, 1.93967285321077422e+13, 9.74981844772792236e+11, 4.96529057451357530e-09, 9.51846106135930866e+07, 1.12715396369556242e+05, 6.13336254512819847e-52, 1.02731846489999998e+03, 1.61419773067196928e+02, 4.96529057451357530e-09] },
    // case 86: alt 261.120 km, lat +6.035, lon +168.099, doy 334, ap 177.008
    Case { doy: 334, sec: 1.95555555555555547e+04, alt: 2.61120000000000005e+02,
        g_lat: 6.03498542274050465e+00, g_long: 1.68099173553719027e+02, lst: 1.66387103356800345e+01,
        f107a: 2.36035502958579855e+02, f107: 9.67128027681660853e+01, ap: 1.77008310249307499e+02,
        ap_array: [7.88279773156899779e+01, 8.38279773156899779e+01, 8.88279773156899779e+01, 9.38279773156899779e+01, 9.88279773156899779e+01, 1.03827977315689978e+02, 1.08827977315689978e+02],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.29347796313373558e+07, 3.08050090751305389e+09, 7.26122013168118000e+08, 9.39254223412296735e+06, 2.23440104571328760e+05, 1.16889601373306938e-13, 8.64287533418922831e+04, 3.10476515836189687e+07, 1.78148254509827164e+00, 1.02863607171068656e+03, 1.02200924490935608e+03, 1.16889601420623118e-13] },
    // case 87: alt 822.400 km, lat +31.749, lon -156.198, doy 38, ap 192.798
    Case { doy: 38, sec: 4.83555555555555547e+04, alt: 8.22400000000000091e+02,
        g_lat: 3.17492711370262271e+01, g_long: -1.56198347107438025e+02, lst: 3.01887562493623030e+00,
        f107a: 2.54112426035502949e+02, f107: 1.10536332179930781e+02, ap: 1.92797783933517991e+02,
        ap_array: [8.31758034026465083e+01, 8.81758034026465083e+01, 9.31758034026465083e+01, 9.81758034026465083e+01, 1.03175803402646508e+02, 1.08175803402646508e+02, 1.13175803402646508e+02],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [4.24147721461594687e+05, 1.73525950215193370e+04, 1.09483133661478926e+00, 3.86056945385905396e-03, 6.94411824454564183e-08, 3.38791019793024785e-18, 5.44213693551338947e+04, 8.73228816075578152e+02, 4.55631938483784506e+03, 8.08035771886098019e+02, 8.08035771886016391e+02, 3.50892604079154085e-18] },
    // case 88: alt 65.540 km, lat +57.464, lon -123.471, doy 220, ap 208.587
    Case { doy: 220, sec: 7.71555555555555475e+04, alt: 6.55400000000000063e+01,
        g_lat: 5.74635568513119210e+01, g_long: -1.23471074380165291e+02, lst: 2.14320987654320980e+01,
        f107a: 2.72189349112426044e+02, f107: 1.24359861591695505e+02, ap: 2.08587257617728511e+02,
        ap_array: [8.75236294896030245e+01, 9.25236294896030245e+01, 9.75236294896030245e+01, 1.02523629489603024e+02, 1.07523629489603024e+02, 1.12523629489603024e+02, 1.17523629489603024e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.99180813823274651e+10, 0.00000000000000000e+00, 2.97118322421051750e+15, 7.83963528633029250e+14, 3.55411566074882969e+13, 1.82104803957088999e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.37653114115580792e+02, 1.82104803957088999e-07] },
    // case 89: alt 79.340 km, lat +83.178, lon -90.744, doy 129, ap 224.377
    Case { doy: 129, sec: 3.55555555555555566e+03, alt: 7.93400000000000034e+01,
        g_lat: 8.31778425655976434e+01, g_long: -9.07438016528925573e+01, lst: -5.06193245587184926e+00,
        f107a: 2.90266272189349138e+02, f107: 1.38183391003460201e+02, ap: 2.24376731301939060e+02,
        ap_array: [9.18714555765595406e+01, 9.68714555765595406e+01, 1.01871455576559541e+02, 1.06871455576559541e+02, 1.11871455576559541e+02, 1.16871455576559541e+02, 1.21871455576559541e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.06747417592230415e+09, 1.24040204533383775e+09, 4.58290463763658000e+14, 1.10975286207581734e+14, 5.46990558158216309e+12, 2.75596030265792412e-08, 1.30151753818782419e+07, 9.33120853960286913e+04, 1.52451182832270610e-55, 1.02731846489999998e+03, 1.87818340574511836e+02, 2.75596030265792412e-08] },
    // case 90: alt 216.320 km, lat -67.434, lon -58.017, doy 311, ap 240.166
    Case { doy: 311, sec: 3.23555555555555511e+04, alt: 2.16319999999999993e+02,
        g_lat: -6.74344023323615147e+01, g_long: -5.80165289256198378e+01, lst: 5.11988572594633062e+00,
        f107a: 7.47337278106508904e+01, f107: 1.52006920415224897e+02, ap: 2.40166204986149580e+02,
        ap_array: [9.62192816635160710e+01, 1.01219281663516071e+02, 1.06219281663516071e+02, 1.11219281663516071e+02, 1.16219281663516071e+02, 1.21219281663516071e+02, 1.26219281663516071e+02],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.99492328407641826e+06, 1.02941192301289868e+09, 3.83196671497408962e+09, 6.09606786306198597e+08, 3.97947797730088905e+07, 2.40765887583723201e-13, 1.35297091550312005e+05, 1.16220846477491856e+07, 7.03948644274633497e-07, 1.49498437178518088e+03, 1.16759052139135974e+03, 2.40765887583723201e-13] },
    // case 91: alt 726.400 km, lat -41.720, lon -25.289, doy 83, ap 255.956
    Case { doy: 83, sec: 6.11555555555555547e+04, alt: 7.26400000000000091e+02,
        g_lat: -4.17201166180758136e+01, g_long: -2.52892561983470898e+01, lst: 1.53017039077645158e+01,
        f107a: 9.28106508875739706e+01, f107: 1.65830449826989593e+02, ap: 2.55955678670360101e+02,
        ap_array: [7.56143667296786393e-01, 5.75614366729678650e+00, 1.07561436672967865e+01, 1.57561436672967865e+01, 2.07561436672967865e+01, 2.57561436672967865e+01, 3.07561436672967865e+01],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, -1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [1.05674848525057337e+06, 8.02251238832445000e+05, 1.05057708986405669e+03, 5.26208018351977014e+00, 1.40445073449916831e-03, 2.91256840332534553e-17, 5.62236737153422291e+04, 2.83407681299777214e+04, 6.39190387375864975e+03, 1.02731846489999998e+03, 1.02731740921252549e+03, 2.92954530001404843e-17] },
    // case 92: alt 53.940 km, lat -16.006, lon +7.438, doy 266, ap 271.745
    Case { doy: 266, sec: 1.31555555555555547e+04, alt: 5.39400000000000048e+01,
        g_lat: -1.60058309037900841e+01, g_long: 7.43801652892562970e+00, lst: 3.65432098765432123e+00,
        f107a: 1.10887573964497051e+02, f107: 1.79653979238754317e+02, ap: 2.71745152354570621e+02,
        ap_array: [5.10396975425330801e+00, 1.01039697542533080e+01, 1.51039697542533080e+01, 2.01039697542533062e+01, 2.51039697542533062e+01, 3.01039697542533062e+01, 3.51039697542533062e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [7.59026917852975311e+10, 0.00000000000000000e+00, 1.13106629073031720e+16, 3.03431153814222200e+15, 1.35292494365706891e+14, 6.95886166457322811e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.57753268258607136e+02, 6.95886166457322811e-07] },
    // case 93: alt 117.340 km, lat +9.708, lon +40.165, doy 174, ap 287.535
    Case { doy: 174, sec: 4.19555555555555547e+04, alt: 1.17340000000000003e+02,
        g_lat: 9.70845481049560988e+00, g_long: 4.01652892561983208e+01, lst: 1.43320069380675434e+01,
        f107a: 1.28964497041420117e+02, f107: 1.93477508650519042e+02, ap: 2.87534626038781141e+02,
        ap_array: [9.45179584120982952e+00, 1.44517958412098295e+01, 1.94517958412098295e+01, 2.44517958412098295e+01, 2.94517958412098295e+01, 3.44517958412098295e+01, 3.94517958412098295e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.08149271408109777e+07, 9.02180949742346039e+10, 3.67302256907755310e+11, 4.49474284065706711e+10, 1.87858238343445492e+09, 2.19809248853632156e-11, 3.25018134030159609e+06, 1.48485975704801758e+06, 3.10825820542602104e-34, 1.02731846489999998e+03, 3.52841478217501844e+02, 2.19809248853632156e-11] },
    // case 94: alt 171.520 km, lat +35.423, lon +72.893, doy 357, ap 4.155
    Case { doy: 357, sec: 7.07555555555555475e+04, alt: 1.71519999999999982e+02,
        g_lat: 3.54227405247813323e+01, g_long: 7.28925619834710687e+01, lst: 2.45138251198857233e+01,
        f107a: 1.47041420118343183e+02, f107: 2.07301038062283766e+02, ap: 4.15512465373961248e+00,
        ap_array: [1.37996219281663492e+01, 1.87996219281663492e+01, 2.37996219281663492e+01, 2.87996219281663492e+01, 3.37996219281663457e+01, 3.87996219281663457e+01, 4.37996219281663457e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.41100610086808726e+06, 8.00668181729430485e+09, 9.44143486895186996e+09, 4.78285101646596551e+08, 1.07062355513089560e+07, 6.78604514606197867e-13, 1.59177741486897343e+05, 3.99742200664122179e+07, 4.34161860732167522e-15, 1.11794693198056416e+03, 8.70883981339734419e+02, 6.78604514606197867e-13] },
    // case 95: alt 630.400 km, lat +61.137, lon +105.620, doy 9, ap 19.945
    Case { doy: 9, sec: 2.27555555555555511e+04, alt: 6.30399999999999977e+02,
        g_lat: 6.11370262390670405e+01, g_long: 1.05619834710743817e+02, lst: 1.33623099683705746e+01,
        f107a: 1.65118343195266277e+02, f107: 2.21124567474048462e+02, ap: 1.99445983379501399e+01,
        ap_array: [1.81474480151228725e+01, 2.31474480151228725e+01, 2.81474480151228725e+01, 3.31474480151228761e+01, 3.81474480151228761e+01, 4.31474480151228761e+01, 4.81474480151228761e+01],
        switches: [0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], out: [1.59820061428378755e+06, 4.22487842467607465e+06, 1.80794383002012510e+04, 2.59987056556490927e+02, 7.61070023822080083e-02, 1.26244298131684311e-16, 7.14556605058912101e+04, 1.05280156577496135e+05, 8.55127022082829171e+03, 1.02731846489999998e+03, 1.02731532348060955e+03, 1.26471419868749523e-16] },
    // case 96: alt 42.340 km, lat +86.851, lon +138.347, doy 192, ap 35.734
    Case { doy: 192, sec: 5.15555555555555475e+04, alt: 4.23400000000000034e+01,
        g_lat: 8.68513119533527629e+01, g_long: 1.38347107438016565e+02, lst: 1.43209876543209873e+01,
        f107a: 1.83195266272189372e+02, f107: 2.34948096885813158e+02, ap: 3.57340720221606674e+01,
        ap_array: [2.24952741020793923e+01, 2.74952741020793923e+01, 3.24952741020793923e+01, 3.74952741020793923e+01, 4.24952741020793923e+01, 4.74952741020793923e+01, 5.24952741020793923e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.77728732550555054e+11, 0.00000000000000000e+00, 5.62873629879588320e+16, 1.51002108687797160e+16, 6.73281292380469625e+14, 3.46306821896275252e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.79018522195414164e+02, 3.46306821896275252e-06] },
    // case 97: alt 109.740 km, lat -88.950, lon +171.074, doy 100, ap 51.524
    Case { doy: 100, sec: 8.03555555555555475e+04, alt: 1.09740000000000009e+02,
        g_lat: -8.89504373177842638e+01, g_long: 1.71074380165289313e+02, lst: 3.37259463320069415e+01,
        f107a: 2.01272189349112438e+02, f107: 2.48771626297577853e+02, ap: 5.15235457063711877e+01,
        ap_array: [2.68431001890359191e+01, 3.18431001890359191e+01, 3.68431001890359227e+01, 4.18431001890359227e+01, 4.68431001890359227e+01, 5.18431001890359227e+01, 5.68431001890359227e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.08906283688658327e+07, 1.21943304030280502e+11, 1.13104414727634375e+12, 1.93995949029879669e+11, 8.78521827072089386e+09, 6.66984389836557998e-11, 3.97356128059980879e+06, 4.92585194331499166e+05, 4.89094618181747361e-37, 1.02731846489999998e+03, 3.03050668489792201e+02, 6.66984389836557998e-11] },
    // case 98: alt 395.520 km, lat -63.236, lon -153.223, doy 283, ap 67.313
    Case { doy: 283, sec: 6.75555555555555475e+03, alt: 3.95520000000000039e+02,
        g_lat: -6.32361516034985414e+01, g_long: -1.53223140495867767e+02, lst: -8.33833282318130919e+00,
        f107a: 2.19349112426035504e+02, f107: 2.62595155709342521e+02, ap: 6.73130193905817151e+01,
        ap_array: [3.11909262759924424e+01, 3.61909262759924388e+01, 4.11909262759924388e+01, 4.61909262759924388e+01, 5.11909262759924388e+01, 5.61909262759924388e+01, 6.11909262759924388e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.47280947684006672e+06, 3.75927255851207435e+08, 5.73355905015662238e+07, 8.90168461145937908e+05, 8.91777774981981020e+03, 1.29228420654717408e-14, 2.04209143436714403e+04, 8.41844495732286572e+06, 1.81491065131236064e+04, 1.32474170288924734e+03, 1.32082762195501437e+03, 1.29233241057407297e-14] },
    // case 99: alt 419.200 km, lat -37.522, lon -120.496, doy 55, ap 83.102
    Case { doy: 55, sec: 3.55555555555555547e+04, alt: 4.19199999999999989e+02,
        g_lat: -3.75218658892128332e+01, g_long: -1.20495867768595048e+02, lst: 1.84348535863687246e+00,
        f107a: 2.37426035502958570e+02, f107: 2.76418685121107274e+02, ap: 8.31024930747922355e+01,
        ap_array: [3.55387523629489621e+01, 4.05387523629489621e+01, 4.55387523629489621e+01, 5.05387523629489621e+01, 5.55387523629489621e+01, 6.05387523629489621e+01, 6.55387523629489692e+01],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, -1, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [3.58923573843732383e+06, 1.52969935080312997e+08, 1.15547651909745373e+07, 4.32636038267082011e+05, 8.24470845109322795e+02, 4.70184642548651766e-15, 7.82908930466700840e+04, 2.36233236980909482e+06, 4.08418853062488233e+03, 1.16593559963894972e+03, 1.16524864929761884e+03, 4.70195490153389084e-15] },
    // case 100: alt 16.820 km, lat -11.808, lon -87.769, doy 237, ap 98.892
    Case { doy: 237, sec: 6.43555555555555475e+04, alt: 1.68200000000000003e+01,
        g_lat: -1.18075801749271250e+01, g_long: -8.77685950413222997e+01, lst: 1.78765432098765409e+01,
        f107a: 2.55502958579881664e+02, f107: 2.90242214532871969e+02, ap: 9.88919667590027558e+01,
        ap_array: [3.98865784499054783e+01, 4.48865784499054783e+01, 4.98865784499054783e+01, 5.48865784499054783e+01, 5.98865784499054783e+01, 6.48865784499054712e+01, 6.98865784499054712e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.79633651945578711e+13, 0.00000000000000000e+00, 2.67681637656732416e+18, 7.18109529341716096e+17, 3.20187390883100480e+16, 1.64690566933689851e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.02022411291993961e+02, 1.64690566933689851e-04] },
    // case 101: alt 93.020 km, lat +13.907, lon -55.041, doy 146, ap 114.681
    Case { doy: 146, sec: 1.63555555555555547e+04, alt: 9.30200000000000102e+01,
        g_lat: 1.39067055393585974e+01, g_long: -5.50413223140496086e+01, lst: 8.73788388939902294e-01,
        f107a: 2.73579881656804787e+02, f107: 6.98788927335640153e+01, ap: 1.14681440443213290e+02,
        ap_array: [4.42344045368620016e+01, 4.92344045368620016e+01, 5.42344045368620016e+01, 5.92344045368620016e+01, 6.42344045368620016e+01, 6.92344045368620016e+01, 7.42344045368620016e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.72421765388284266e+08, 6.52518715164510498e+11, 3.25670772999339922e+13, 6.48528108163688965e+12, 3.67781663853376343e+11, 1.89996937806246852e-09, 4.82045996419146508e+07, 2.43123647088822414e+05, 1.71744670183398231e-47, 1.02731846489999998e+03, 1.78584578134135768e+02, 1.89996937806246852e-09] },
    // case 102: alt 296.960 km, lat +39.621, lon -22.314, doy 328, ap 130.471
    Case { doy: 328, sec: 4.51555555555555547e+04, alt: 2.96960000000000036e+02,
        g_lat: 3.96209912536442914e+01, g_long: -2.23140495867768607e+01, lst: 1.10556065707580871e+01,
        f107a: 2.91656804733727824e+02, f107: 8.37024221453287112e+01, ap: 1.30470914127423811e+02,
        ap_array: [4.85822306238185320e+01, 5.35822306238185320e+01, 5.85822306238185320e+01, 6.35822306238185320e+01, 6.85822306238185320e+01, 7.35822306238185320e+01, 7.85822306238185320e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.60003184649213403e+06, 2.04660890019628143e+09, 2.59157553102993697e+08, 2.35196613209956465e+06, 9.98426009054004098e+04, 6.70206071637173137e-14, 1.21215354537254942e+05, 1.89948488679189049e+07, 1.23856225625904614e+02, 9.53162448827483217e+02, 9.52771634071572862e+02, 6.70206104533386520e-14] },
    // case 103: alt 899.200 km, lat +65.335, lon +10.413, doy 32, ap 146.260
    Case { doy: 32, sec: 7.39555555555555475e+04, alt: 8.99200000000000045e+02,
        g_lat: 6.53352769679300138e+01, g_long: 1.04132231404958588e+01, lst: 2.12374247525762669e+01,
        f107a: 7.61242603550295911e+01, f107: 9.75259515570934354e+01, ap: 1.46260387811634331e+02,
        ap_array: [5.29300567107750410e+01, 5.79300567107750410e+01, 6.29300567107750410e+01, 6.79300567107750339e+01, 7.29300567107750339e+01, 7.79300567107750339e+01, 8.29300567107750339e+01],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [2.66333959329927922e+05, 3.34476597007417004e+04, 1.39576932295154563e+01, 5.58598343676313877e-02, 5.12338822497462109e-06, 2.70792725741548957e-18, 2.39526954993192994e+04, 4.59842086415494066e+02, 2.68031552682099573e+03, 1.02731846489999998e+03, 1.02731842951320391e+03, 2.77911643780785497e-18] },
    // case 104: alt 5.220 km, lat -85.277, lon +43.140, doy 214, ap 162.050
    Case { doy: 214, sec: 2.59555555555555511e+04, alt: 5.22000000000000064e+00,
        g_lat: -8.52769679300291585e+01, g_long: 4.31404958677685784e+01, lst: 7.20987654320987570e+00,
        f107a: 9.42011834319526713e+01, f107: 1.11349480968858131e+02, ap: 1.62049861495844851e+02,
        ap_array: [5.72778827977315643e+01, 6.22778827977315643e+01, 6.72778827977315643e+01, 7.22778827977315643e+01, 7.72778827977315643e+01, 8.22778827977315643e+01, 8.72778827977315643e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.33769522050662656e+13, 0.00000000000000000e+00, 1.24244421172491571e+19, 3.33310508679443149e+18, 1.48614964385475776e+17, 7.64411199078419219e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.25886759563386221e+02, 7.64411199078419219e-04] },
    // case 105: alt 85.420 km, lat -59.563, lon +75.868, doy 123, ap 177.839
    Case { doy: 123, sec: 5.47555555555555547e+04, alt: 8.54200000000000017e+01,
        g_lat: -5.95626822157434432e+01, g_long: 7.58677685950412979e+01, lst: 2.02677277828792981e+01,
        f107a: 1.12278106508875737e+02, f107: 1.25173010380622827e+02, ap: 1.77839335180055400e+02,
        ap_array: [6.16257088846880876e+01, 6.66257088846880947e+01, 7.16257088846880947e+01, 7.66257088846880947e+01, 8.16257088846880947e+01, 8.66257088846880947e+01, 9.16257088846880947e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [7.06471079986911535e+08, 3.86589799066629257e+10, 9.79553616772568281e+13, 2.59958191833041172e+13, 1.15719980609718506e+12, 6.01173281716109466e-09, 8.97902639006078988e+07, 7.46170481128322863e+04, 3.69929401965890262e-53, 1.02731846489999998e+03, 1.92115221043232140e+02, 6.01173281716109466e-09] },
    // case 106: alt 252.160 km, lat -33.848, lon +108.595, doy 306, ap 193.629
    Case { doy: 306, sec: 8.35555555555555475e+04, alt: 2.52159999999999997e+02,
        g_lat: -3.38483965014577279e+01, g_long: 1.08595041322314046e+02, lst: 3.04495459646974780e+01,
        f107a: 1.30355029585798832e+02, f107: 1.38996539792387523e+02, ap: 1.93628808864265920e+02,
        ap_array: [6.59735349716446109e+01, 7.09735349716446109e+01, 7.59735349716446109e+01, 8.09735349716446109e+01, 8.59735349716446109e+01, 9.09735349716446109e+01, 9.59735349716446109e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.37067139066814445e+06, 1.61577122195749259e+09, 8.99802051906126499e+08, 3.88656258074647933e+07, 5.48706159258972947e+06, 8.75642854212402247e-14, 3.37987066264478271e+04, 1.58623965974576138e+07, 1.41963826003115773e-01, 1.26286997283007486e+03, 1.12821708246382650e+03, 8.75642854250107778e-14] },
    // case 107: alt 803.200 km, lat -8.134, lon +141.322, doy 77, ap 209.418
    Case { doy: 77, sec: 1.42222222222222194e+03, alt: 8.03200000000000045e+02,
        g_lat: -8.13411078717201974e+00, g_long: 1.41322314049586794e+02, lst: 9.81654933170084831e+00,
        f107a: 1.48431952662721898e+02, f107: 1.52820069204152219e+02, ap: 2.09418282548476412e+02,
        ap_array: [7.03213610586011271e+01, 7.53213610586011271e+01, 8.03213610586011271e+01, 8.53213610586011271e+01, 9.03213610586011271e+01, 9.53213610586011271e+01, 1.00321361058601127e+02],
        switches: [0, 0, 1, 2, 0, 1, 2, 0, 1, -1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], out: [7.47679331723947893e+05, 2.50718786221044633e+05, 1.43791720061088853e+02, 1.05817972032399665e+00, 7.91522457460195120e-05, 1.19251024203159837e-17, 5.52492833460616603e+04, 8.73352970622500106e+03, 4.91704118350525096e+03, 1.02731846489999998e+03, 1.02731816320010807e+03, 1.20556990341498843e-17] },
    // case 108: alt 63.220 km, lat +17.580, lon +174.050, doy 260, ap 225.208
    Case { doy: 260, sec: 3.02222222222222190e+04, alt: 6.32200000000000060e+01,
        g_lat: 1.75801749271137027e+01, g_long: 1.74049586776859542e+02, lst: 8.39506172839505993e+00,
        f107a: 1.66508875739644964e+02, f107: 1.66643598615916943e+02, ap: 2.25207756232686961e+02,
        ap_array: [7.46691871455576432e+01, 7.96691871455576432e+01, 8.46691871455576432e+01, 8.96691871455576432e+01, 9.46691871455576432e+01, 9.96691871455576432e+01, 1.04669187145557643e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.36114473344453735e+10, 0.00000000000000000e+00, 3.51588054602506450e+15, 9.40915363611077875e+14, 4.20497858217817344e+13, 2.16191814452842073e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.33230167450049720e+02, 2.16191814452842073e-07] },
    // case 109: alt 77.820 km, lat +43.294, lon -150.248, doy 169, ap 240.997
    Case { doy: 169, sec: 5.90222222222222117e+04, alt: 7.78199999999999932e+01,
        g_lat: 4.32944606413994109e+01, g_long: -1.50247933884297510e+02, lst: 6.37853280277522572e+00,
        f107a: 1.84585798816568058e+02, f107: 1.80467128027681667e+02, ap: 2.40997229916897510e+02,
        ap_array: [7.90170132325141594e+01, 8.40170132325141594e+01, 8.90170132325141594e+01, 9.40170132325141594e+01, 9.90170132325141594e+01, 1.04017013232514159e+02, 1.09017013232514159e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.59061526577917528e+09, 4.27611338057204783e+08, 5.36168727193140250e+14, 1.36405241315036953e+14, 6.40819351064552637e+12, 3.25925081092356864e-08, 3.30673641249138745e+05, 8.37124907894791395e+04, 4.62799363833895814e-58, 1.02731846489999998e+03, 1.85102342829150388e+02, 3.25925081092356864e-08] },
    // case 110: alt 207.360 km, lat +69.009, lon -117.521, doy 351, ap 256.787
    Case { doy: 351, sec: 1.10222222222222226e+04, alt: 2.07360000000000014e+02,
        g_lat: 6.90087463556851333e+01, g_long: -1.17520661157024790e+02, lst: -4.77298234873992389e+00,
        f107a: 2.02662721893491153e+02, f107: 1.94290657439446363e+02, ap: 2.56786703601108002e+02,
        ap_array: [8.33648393194706898e+01, 8.83648393194706898e+01, 9.33648393194706898e+01, 9.83648393194706898e+01, 1.03364839319470690e+02, 1.08364839319470690e+02, 1.13364839319470690e+02],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [7.67870388044057763e+05, 1.18348035319752073e+09, 6.62431092874725914e+09, 5.08646030855284512e+08, 1.25173038729267493e+07, 3.67215500643713479e-13, 7.75643095294845534e+03, 1.23721270945359720e+06, 2.16341764772084226e-07, 1.97977505233910938e+03, 1.48767256441584800e+03, 3.67215500643713479e-13] },
    // case 111: alt 707.200 km, lat -81.603, lon -84.793, doy 20, ap 272.576
    Case { doy: 20, sec: 3.98222222222222190e+04, alt: 7.07200000000000045e+02,
        g_lat: -8.16034985422740533e+01, g_long: -8.47933884297520706e+01, lst: 5.40883583307825599e+00,
        f107a: 2.20739644970414219e+02, f107: 2.08114186851211088e+02, ap: 2.72576177285318522e+02,
        ap_array: [8.77126654064272202e+01, 9.27126654064272202e+01, 9.77126654064272202e+01, 1.02712665406427220e+02, 1.07712665406427220e+02, 1.12712665406427220e+02, 1.17712665406427220e+02],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [1.22859689595898823e+06, 1.91684849028305081e+06, 5.17770386333277202e+03, 6.38699979485780460e+01, 1.33505736886942519e-02, 6.06142931880974255e-17, 6.16268660353047962e+04, 5.15731481521714886e+04, 6.81834695638038920e+03, 1.09554353489798336e+03, 1.09554294847547703e+03, 6.07953884832588889e-17] },
    // case 112: alt 51.620 km, lat -55.889, lon -52.066, doy 203, ap 288.366
    Case { doy: 203, sec: 6.86222222222222190e+04, alt: 5.16200000000000045e+01,
        g_lat: -5.58892128279883380e+01, g_long: -5.20661157024793511e+01, lst: 1.90617283950617278e+01,
        f107a: 2.38816568047337284e+02, f107: 2.21937716262975783e+02, ap: 2.88365650969529042e+02,
        ap_array: [9.20604914933837364e+01, 9.70604914933837364e+01, 1.02060491493383736e+02, 1.07060491493383736e+02, 1.12060491493383736e+02, 1.17060491493383736e+02, 1.22060491493383736e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.51165462074886703e+10, 0.00000000000000000e+00, 9.70336211427219000e+15, 2.60312095439580050e+15, 1.16066765929866797e+14, 5.96997321800486351e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.59475298257293503e+02, 5.96997321800486351e-07] },
    // case 113: alt 115.820 km, lat -30.175, lon -19.339, doy 112, ap 4.986
    Case { doy: 112, sec: 2.06222222222222226e+04, alt: 1.15819999999999993e+02,
        g_lat: -3.01749271137026298e+01, g_long: -1.93388429752066031e+01, lst: 4.43913886338128805e+00,
        f107a: 2.56893491124260379e+02, f107: 2.35761245674740479e+02, ap: 4.98614958448753498e+00,
        ap_array: [9.64083175803402668e+01, 1.01408317580340267e+02, 1.06408317580340267e+02, 1.11408317580340267e+02, 1.16408317580340267e+02, 1.21408317580340267e+02, 1.26408317580340267e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.48789920126646310e+07, 1.71088327663043121e+11, 5.30169372404474915e+11, 6.07603498695930862e+10, 2.61021323201297998e+09, 3.25877416080048489e-11, 2.10416595854923781e+06, 9.02014351502749836e+05, 7.09915626223789878e-35, 1.02731846489999998e+03, 3.71319787707674550e+02, 3.25877416080048489e-11] },
    // case 114: alt 162.560 km, lat -4.461, lon +13.388, doy 294, ap 20.776
    Case { doy: 294, sec: 4.94222222222222263e+04, alt: 1.62560000000000002e+02,
        g_lat: -4.46064139941691451e+00, g_long: 1.33884297520661164e+01, lst: 1.46209570451994697e+01,
        f107a: 2.74970414201183473e+02, f107: 2.49584775086505175e+02, ap: 2.07756232686980624e+01,
        ap_array: [9.45179584120983018e-01, 5.94517958412098313e+00, 1.09451795841209822e+01, 1.59451795841209822e+01, 2.09451795841209822e+01, 2.59451795841209822e+01, 3.09451795841209822e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.91839194133041799e+07, 2.05402734230284004e+10, 2.08751438664408951e+10, 4.67519983357826412e+08, 2.70847395022924505e+07, 1.54416248998271493e-12, 1.10137923821678938e+05, 6.77921119677257687e+07, 2.66067150872365768e-17, 1.29550785870039226e+03, 9.71797371355775113e+02, 1.54416248998271493e-12] },
    // case 115: alt 611.200 km, lat +21.254, lon +46.116, doy 66, ap 36.565
    Case { doy: 66, sec: 7.82222222222222190e+04, alt: 6.11200000000000045e+02,
        g_lat: 2.12536443148688079e+01, g_long: 4.61157024793388075e+01, lst: 2.48027752270176478e+01,
        f107a: 2.93047337278106511e+02, f107: 2.63408304498269899e+02, ap: 3.65650969529085899e+01,
        ap_array: [5.29300567107750464e+00, 1.02930056710775055e+01, 1.52930056710775055e+01, 2.02930056710775055e+01, 2.52930056710775055e+01, 3.02930056710775055e+01, 3.52930056710775020e+01],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, -1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [1.71774259600642673e+06, 4.77608219849357102e+06, 2.21378598665207355e+04, 1.61705989679025237e+02, 1.20392227623208337e-01, 1.42406456742632372e-16, 5.43407358607886708e+04, 1.29954025336488834e+05, 8.97162935663564713e+03, 1.02731846489999998e+03, 1.02730731420753682e+03, 1.42644743218344625e-16] },
    // case 116: alt 40.020 km, lat +46.968, lon +78.843, doy 249, ap 52.355
    Case { doy: 249, sec: 4.62222222222222172e+03, alt: 4.00200000000000031e+01,
        g_lat: 4.69679300291545019e+01, g_long: 7.88429752066115839e+01, lst: 1.28395061728395055e+00,
        f107a: 7.75147928994082918e+01, f107: 2.77231833910034595e+02, ap: 5.23545706371191102e+01,
        ap_array: [9.64083175803402703e+00, 1.46408317580340270e+01, 1.96408317580340253e+01, 2.46408317580340253e+01, 2.96408317580340253e+01, 3.46408317580340253e+01, 3.96408317580340253e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.84871895432404785e+11, 0.00000000000000000e+00, 7.22533342819257600e+16, 1.93834019878122240e+16, 8.64258258013255000e+14, 4.44537125889796119e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.52844034853826599e+02, 4.44537125889796119e-06] },
    // case 117: alt 108.220 km, lat +72.682, lon +111.570, doy 157, ap 68.144
    Case { doy: 157, sec: 3.34222222222222190e+04, alt: 1.08219999999999999e+02,
        g_lat: 7.26822157434402243e+01, g_long: 1.11570247933884275e+02, lst: 1.67219671462095683e+01,
        f107a: 9.55917159763313578e+01, f107: 2.91055363321799291e+02, ap: 6.81440443213296447e+01,
        ap_array: [1.39886578449905485e+01, 1.89886578449905485e+01, 2.39886578449905485e+01, 2.89886578449905485e+01, 3.39886578449905485e+01, 3.89886578449905485e+01, 4.39886578449905485e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.68798963504334204e+07, 1.13311927997080795e+11, 1.22390230928359839e+12, 2.16703902496683350e+11, 9.47702711253518105e+09, 7.20372643713687604e-11, 5.06504383575783297e+06, 5.95390876178492093e+05, 6.03754400423877691e-38, 1.02731846489999998e+03, 3.10079905559273811e+02, 7.20372643713687604e-11] },
    // case 118: alt 386.560 km, lat -77.930, lon +144.298, doy 340, ap 83.934
    Case { doy: 340, sec: 6.22222222222222263e+04, alt: 3.86560000000000002e+02,
        g_lat: -7.79300291545189481e+01, g_long: 1.44297520661157080e+02, lst: 2.69037853280277588e+01,
        f107a: 1.13668639053254438e+02, f107: 7.06920415224913512e+01, ap: 8.39335180055401651e+01,
        ap_array: [1.83364839319470718e+01, 2.33364839319470718e+01, 2.83364839319470718e+01, 3.33364839319470718e+01, 3.83364839319470718e+01, 4.33364839319470718e+01, 4.83364839319470718e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.12376349771198025e+06, 8.14716277120361477e+07, 2.02300566026730277e+07, 8.66259792053637211e+05, 5.04260223832815427e+03, 3.17713184768169644e-15, 1.70026183803547348e+05, 5.25723008835557383e+05, 2.40167492169769321e+03, 1.09092484899084093e+03, 1.08919979944643660e+03, 3.17719563616761672e-15] },
    // case 119: alt 515.200 km, lat -52.216, lon +177.025, doy 43, ap 99.723
    Case { doy: 43, sec: 1.42222222222222226e+04, alt: 5.15200000000000045e+02,
        g_lat: -5.22157434402332328e+01, g_long: 1.77024793388429771e+02, lst: 1.57522701765126012e+01,
        f107a: 1.31745562130177518e+02, f107: 8.45155709342560613e+01, ap: 9.97229916897506854e+01,
        ap_array: [2.26843100189035916e+01, 2.76843100189035916e+01, 3.26843100189035880e+01, 3.76843100189035880e+01, 4.26843100189035880e+01, 4.76843100189035880e+01, 5.26843100189035880e+01],
        switches: [0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], out: [2.97287603499186877e+06, 3.78373875515536517e+07, 7.23573490164494142e+05, 1.71387720621267727e+04, 1.40732601593956872e+01, 1.07646928096371020e-15, 9.14185751764981978e+04, 7.34663254218840972e+05, 9.30485042153807626e+03, 1.02731846489999998e+03, 1.02731118357769606e+03, 1.07671641779090619e-15] },
    // case 120: alt 28.420 km, lat -26.501, lon -179.730, doy 226, ap 115.512
    Case { doy: 226, sec: 4.30222222222222190e+04, alt: 2.84200000000000017e+01,
        g_lat: -2.65014577259475246e+01, g_long: -1.79729526671675444e+02, lst: 1.19506172839506171e+01,
        f107a: 1.49822485207100613e+02, f107: 9.83391003460207571e+01, ap: 1.15512465373961220e+02,
        ap_array: [2.70321361058601113e+01, 3.20321361058601113e+01, 3.70321361058601113e+01, 4.20321361058601113e+01, 4.70321361058601113e+01, 5.20321361058601113e+01, 5.70321361058601113e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.55376477931152002e+12, 0.00000000000000000e+00, 3.80550042217753088e+17, 1.02090159825756624e+17, 4.55194932998765400e+15, 2.34132616447346708e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.25250857122053361e+02, 2.34132616447346708e-05] },
    // case 121: alt 100.620 km, lat -0.787, lon -147.002, doy 135, ap 131.302
    Case { doy: 135, sec: 7.18222222222222190e+04, alt: 1.00620000000000005e+02,
        g_lat: -7.87172011661809279e-01, g_long: -1.47002253944402696e+02, lst: 1.01504670209904369e+01,
        f107a: 1.67899408284023679e+02, f107: 1.12162629757785467e+02, ap: 1.31301939058171740e+02,
        ap_array: [3.13799621928166346e+01, 3.63799621928166346e+01, 4.13799621928166346e+01, 4.63799621928166346e+01, 5.13799621928166346e+01, 5.63799621928166346e+01, 6.13799621928166346e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.11213781842725188e+08, 5.54622056629132202e+11, 7.69996813049873926e+12, 1.52269785763265698e+12, 7.62289601657117462e+10, 4.58573367421708914e-10, 1.62775254743382260e+07, 3.55396042891685735e+05, 5.81785306911251186e-43, 1.02731846489999998e+03, 2.03865423811711764e+02, 4.58573367421708914e-10] },
    // case 122: alt 341.760 km, lat +24.927, lon -114.275, doy 317, ap 147.091
    Case { doy: 317, sec: 2.38222222222222190e+04, alt: 3.41760000000000048e+02,
        g_lat: 2.49271137026238847e+01, g_long: -1.14274981217129977e+02, lst: -1.00104813052471542e+00,
        f107a: 1.85976331360946745e+02, f107: 1.25986159169550177e+02, ap: 1.47091412742382261e+02,
        ap_array: [3.57277882797731507e+01, 4.07277882797731507e+01, 4.57277882797731507e+01, 5.07277882797731507e+01, 5.57277882797731507e+01, 6.07277882797731507e+01, 6.57277882797731507e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.33517569110906310e+06, 6.89160608859887958e+08, 8.76633998645460606e+07, 1.40776448062269832e+06, 1.31757594976080436e+04, 2.27302111429089226e-14, 5.57741604382106743e+04, 9.48438358647084795e+06, 3.82844136392261134e+03, 1.13181579975425348e+03, 1.12700790110373669e+03, 2.27303128263115486e-14] },
    // case 123: alt 995.200 km, lat +50.641, lon -81.548, doy 89, ap 162.881
    Case { doy: 89, sec: 5.26222222222222190e+04, alt: 9.95200000000000045e+02,
        g_lat: 5.06413994169095929e+01, g_long: -8.15477084898572571e+01, lst: 9.18077005129346624e+00,
        f107a: 2.04053254437869867e+02, f107: 1.39809688581314873e+02, ap: 1.62880886426592781e+02,
        ap_array: [4.00756143667296740e+01, 4.50756143667296740e+01, 5.00756143667296740e+01, 5.50756143667296740e+01, 6.00756143667296740e+01, 6.50756143667296669e+01, 7.00756143667296669e+01],
        switches: [0, 1, 2, 0, 1, 2, 0, 1, 2, -1, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], out: [6.04319919363719644e+05, 1.34670057663996442e+05, 5.44496438782995611e+01, 3.56625941113458467e-01, 2.05792206991368874e-05, 7.78853125211018632e-18, 4.87342104318979618e+04, 4.97253450423739014e+03, 2.45945572469237368e+03, 1.23930712955990930e+03, 1.23930704497511510e+03, 7.85385439615801676e-18] },
    // case 124: alt 0.116 km, lat +76.356, lon -48.820, doy 271, ap 178.670
    Case { doy: 271, sec: 8.14222222222222190e+04, alt: 1.16000000000000006e-01,
        g_lat: 7.63556851311953153e+01, g_long: -4.88204357625845091e+01, lst: 2.26172839506172814e+01,
        f107a: 2.22130177514792933e+02, f107: 1.53633217993079569e+02, ap: 1.78670360110803330e+02,
        ap_array: [4.44234404536861973e+01, 4.94234404536861973e+01, 5.44234404536861973e+01, 5.94234404536861973e+01, 6.44234404536861973e+01, 6.94234404536861973e+01, 7.44234404536861973e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.42476211288524125e+14, 0.00000000000000000e+00, 2.12311363443153838e+19, 5.69567694708948787e+18, 2.53956237382528448e+17, 1.30624121691741576e-03, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.69907823628320727e+02, 1.30624121691741576e-03] },
    // case 125: alt 82.076 km, lat -74.257, lon -16.093, doy 180, ap 194.460
    Case { doy: 180, sec: 7.82222222222222172e+03, alt: 8.20759999999999934e+01,
        g_lat: -7.42565597667638428e+01, g_long: -1.60931630353117896e+01, lst: 1.09996197048538669e+00,
        f107a: 2.40207100591715971e+02, f107: 1.67456747404844293e+02, ap: 1.94459833795013822e+02,
        ap_array: [4.87712665406427206e+01, 5.37712665406427206e+01, 5.87712665406427206e+01, 6.37712665406427206e+01, 6.87712665406427277e+01, 7.37712665406427277e+01, 7.87712665406427277e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.07404003645817280e+08, 5.52607385954097652e+09, 1.28824103822438000e+14, 3.18241776525186250e+13, 1.53254944456146753e+12, 7.78015883381884608e-09, 5.39776912336785793e+07, 3.63165079563046820e+04, 4.94174172608843048e-55, 1.02731846489999998e+03, 2.10307692563849940e+02, 7.78015883381884608e-09] },
    // case 126: alt 232.448 km, lat -48.542, lon +16.634, doy 363, ap 210.249
    Case { doy: 363, sec: 3.66222222222222190e+04, alt: 2.32448000000000008e+02,
        g_lat: -4.85422740524781418e+01, g_long: 1.66341096919609299e+01, lst: 1.12817801523035666e+01,
        f107a: 2.58284023668639065e+02, f107: 1.81280276816608989e+02, ap: 2.10249307479224342e+02,
        ap_array: [5.31190926275992439e+01, 5.81190926275992439e+01, 6.31190926275992439e+01, 6.81190926275992439e+01, 7.31190926275992439e+01, 7.81190926275992439e+01, 8.31190926275992439e+01],
        switches: [0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.59734478081252659e+06, 2.53681504188034678e+09, 3.37091556046806717e+09, 9.46682992568463534e+07, 7.30550259368149843e+06, 2.33001418915866806e-13, 2.95130437135004104e+04, 1.46828544028734773e+08, 2.29191826119457247e-03, 1.49453649378308796e+03, 1.41597259197945709e+03, 2.33001418915927693e-13] },
    // case 127: alt 760.960 km, lat -22.828, lon +49.361, doy 2, ap 226.039
    Case { doy: 2, sec: 6.54222222222222190e+04, alt: 7.60960000000000036e+02,
        g_lat: -2.28279883381924265e+01, g_long: 4.93613824192336779e+01, lst: 2.14635983341217518e+01,
        f107a: 2.76360946745562160e+02, f107: 1.95103806228373713e+02, ap: 2.26038781163434862e+02,
        ap_array: [5.74669187145557601e+01, 6.24669187145557601e+01, 6.74669187145557601e+01, 7.24669187145557601e+01, 7.74669187145557601e+01, 8.24669187145557601e+01, 8.74669187145557601e+01],
        switches: [0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], out: [1.35990341122545372e+06, 5.56988079578056582e+05, 4.60221740980162053e+02, 2.01050546671085595e+00, 5.33322268822481989e-04, 2.45830169170210390e-17, 2.85370806357852307e+04, 2.97239836080760433e+04, 3.79023102147065569e+05, 1.02731846489999998e+03, 1.02731793661541542e+03, 3.46498705100471010e-17] },
    // case 128: alt 58.116 km, lat +2.886, lon +82.089, doy 184, ap 241.828
    Case { doy: 184, sec: 1.74222222222222226e+04, alt: 5.81160000000000068e+01,
        g_lat: 2.88629737609329595e+00, g_long: 8.20886551465064258e+01, lst: 4.83950617283950635e+00,
        f107a: 2.94437869822485254e+02, f107: 2.08927335640138409e+02, ap: 2.41828254847645411e+02,
        ap_array: [6.18147448015122905e+01, 6.68147448015122905e+01, 7.18147448015122905e+01, 7.68147448015122905e+01, 8.18147448015122905e+01, 8.68147448015122905e+01, 9.18147448015122905e+01],
        switches: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], out: [4.32027422435883120e+16, 0.00000000000000000e+00, 6.43786989228999469e+21, 1.72708735600463682e+21, 7.70065807166267720e+19, 3.96088596770397862e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.48494419337316828e+02, 3.96088596770397862e-04] },
    // case 129: alt 74.476 km, lat +28.601, lon +114.816, doy 93, ap 257.618
    Case { doy: 93, sec: 4.62222222222222117e+04, alt: 7.44759999999999991e+01,
        g_lat: 2.86005830903789899e+01, g_long: 1.14815927873779174e+02, lst: 2.04939013644247794e+01,
        f107a: 7.89053254437869782e+01, f107: 2.22750865051903133e+02, ap: 2.57617728531855903e+02,
        ap_array: [6.61625708884688066e+01, 7.11625708884688066e+01, 7.61625708884688066e+01, 8.11625708884688066e+01, 8.61625708884688066e+01, 9.11625708884688066e+01, 9.61625708884688066e+01],
        switches: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [1.02636564998616219e+08, 8.11156484754894336e+12, 5.92714378953911250e+14, 2.19483708985170094e+14, 5.47365470848376172e+13, 4.30592152426394974e-08, 4.01434716692650283e+05, 3.98329006956949997e+10, 1.12117169650474687e-61, 1.02731846489999998e+03, 2.10233093843564291e+02, 4.30592152426394974e-08] },
    // case 130: alt 187.648 km, lat +54.315, lon +147.543, doy 276, ap 273.407
    Case { doy: 276, sec: 7.50222222222222044e+04, alt: 1.87647999999999996e+02,
        g_lat: 5.43148688046646839e+01, g_long: 1.47543200601051865e+02, lst: 3.06757195462429593e+01,
        f107a: 9.69822485207100726e+01, f107: 2.36574394463667801e+02, ap: 2.73407202216066423e+02,
        ap_array: [7.05103969754253228e+01, 7.55103969754253228e+01, 8.05103969754253228e+01, 8.55103969754253228e+01, 9.05103969754253228e+01, 9.55103969754253228e+01, 1.00510396975425323e+02],
        switches: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [9.75727982054146193e+06, 5.65454709681749916e+09, 5.25404296319532013e+09, 4.54118058427687407e+08, 4.82302940083756484e+06, 4.20236374914345470e-13, 1.13645654851300365e+05, 5.74745993056952208e+07, 1.75863028595705095e-11, 1.02731846489999998e+03, 8.95119876565432833e+02, 4.20236374914345470e-13] },
    // case 131: alt 664.960 km, lat +80.029, lon -176.754, doy 48, ap 289.197
    Case { doy: 48, sec: 2.70222222222222190e+04, alt: 6.64960000000000036e+02,
        g_lat: 8.00291545189504063e+01, g_long: -1.76754320060105186e+02, lst: -4.27744849783417447e+00,
        f107a: 1.15059171597633139e+02, f107: 2.50397923875432525e+02, ap: 2.89196675900277000e+02,
        ap_array: [7.48582230623818390e+01, 7.98582230623818390e+01, 8.48582230623818390e+01, 8.98582230623818390e+01, 9.48582230623818390e+01, 9.98582230623818390e+01, 1.04858223062381839e+02],
        switches: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [1.27148253254077467e+06, 1.99922392819033074e+06, 5.31094374404437440e+03, 6.51505884179461390e+01, 1.36157841863389986e-02, 6.31522045990797717e-17, 6.40443551724349236e+04, 5.39391158399129999e+04, 7.79460645084033695e+03, 1.02731846489999998e+03, 1.02731478900125148e+03, 6.33592293464140832e-17] },
    // case 132: alt 46.516 km, lat -70.583, lon -144.027, doy 230, ap 5.817
    Case { doy: 230, sec: 5.58222222222222190e+04, alt: 4.65160000000000124e+01,
        g_lat: -7.05830903790087518e+01, g_long: -1.44027047332832467e+02, lst: 1.55061728395061706e+01,
        f107a: 1.33136094674556205e+02, f107: 2.64221453287197221e+02, ap: 5.81717451523545659e+00,
        ap_array: [7.92060491493383694e+01, 8.42060491493383694e+01, 8.92060491493383694e+01, 9.42060491493383694e+01, 9.92060491493383694e+01, 1.04206049149338369e+02, 1.09206049149338369e+02],
        switches: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], out: [1.80394191666152649e+11, 0.00000000000000000e+00, 2.68814958255079440e+16, 7.21149888510901600e+15, 3.21543012316813250e+14, 1.65387840058186945e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.66195042063859319e+02, 1.65387840058186945e-06] },
    // case 133: alt 112.476 km, lat -44.869, lon -111.300, doy 139, ap 21.607
    Case { doy: 139, sec: 8.46222222222222044e+04, alt: 1.12475999999999999e+02,
        g_lat: -4.48688046647230365e+01, g_long: -1.11299774605559733e+02, lst: 1.60861878658021844e+01,
        f107a: 1.51213017751479299e+02, f107: 2.78044982698961917e+02, ap: 2.16066481994459814e+01,
        ap_array: [8.35538752362948998e+01, 8.85538752362948998e+01, 9.35538752362948998e+01, 9.85538752362948998e+01, 1.03553875236294900e+02, 1.08553875236294900e+02, 1.13553875236294900e+02],
        switches: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [3.65824779050185531e+07, 1.86502784299475739e+11, 9.62716583676996460e+11, 1.47371233520810333e+11, 6.03043108622789001e+09, 5.79627086423632270e-11, 2.77535063814866997e+05, 1.42443911144073129e+09, 2.64244031454666747e-36, 1.02731846489999998e+03, 2.67023914967389601e+02, 5.79627086423632270e-11] },
    // case 134: alt 142.848 km, lat -19.155, lon -78.573, doy 321, ap 37.396
    Case { doy: 321, sec: 2.48888888888888869e+03, alt: 1.42848000000000013e+02,
        g_lat: -1.91545189504373212e+01, g_long: -7.85725018782870279e+01, lst: -4.54680876719444349e+00,
        f107a: 1.69289940828402393e+02, f107: 2.91868512110726670e+02, ap: 3.73961218836565052e+01,
        ap_array: [8.79017013232514159e+01, 9.29017013232514159e+01, 9.79017013232514159e+01, 1.02901701323251416e+02, 1.07901701323251416e+02, 1.12901701323251416e+02, 1.17901701323251416e+02],
        switches: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [1.54577785561699551e+07, 2.18956900176258011e+10, 4.38875172422783661e+10, 4.90130460030738449e+09, 9.19022754918557853e+07, 2.89255396981226984e-12, 1.48559186046684976e+05, 1.95789320317579776e+08, 7.67321860243286703e-24, 1.02731846489999998e+03, 6.44321318824193895e+02, 2.89255396981226984e-12] },
    // case 135: alt 568.960 km, lat +6.560, lon -45.845, doy 25, ap 53.186
    Case { doy: 25, sec: 3.12888888888888869e+04, alt: 5.68960000000000036e+02,
        g_lat: 6.55976676384837276e+00, g_long: -4.58452291510142800e+01, lst: 5.63500941462373905e+00,
        f107a: 1.87366863905325431e+02, f107: 7.15051903114186871e+01, ap: 5.31855955678670327e+01,
        ap_array: [9.22495274102079321e+01, 9.72495274102079321e+01, 1.02249527410207932e+02, 1.07249527410207932e+02, 1.12249527410207932e+02, 1.17249527410207932e+02, 1.22249527410207932e+02],
        switches: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [1.83406104966604104e+06, 8.65486726109360531e+06, 6.90021316022990650e+04, 1.22097403061322507e+03, 5.30917421008292645e-01, 2.49958522246613433e-16, 7.01877447388955770e+04, 1.94425916334784619e+05, 9.58331899367773076e+03, 1.02731846489999998e+03, 1.02729147403692014e+03, 2.50213055199085536e-16] },
    // case 136: alt 34.916 km, lat +32.274, lon -13.118, doy 207, ap 68.975
    Case { doy: 207, sec: 6.00888888888888832e+04, alt: 3.49160000000000039e+01,
        g_lat: 3.22740524781340952e+01, g_long: -1.31179564237415320e+01, lst: 1.66913580246913575e+01,
        f107a: 2.05443786982248525e+02, f107: 8.53287197231833829e+01, ap: 6.89750692520775601e+01,
        ap_array: [9.65973534971644625e+01, 1.01597353497164463e+02, 1.06597353497164463e+02, 1.11597353497164463e+02, 1.16597353497164463e+02, 1.21597353497164463e+02, 1.26597353497164463e+02],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], out: [9.24814704889835327e+11, 0.00000000000000000e+00, 1.37811547030696464e+17, 3.69707037219249440e+16, 1.64843281980767575e+15, 8.47882656770029418e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.40106910856671902e+02, 8.47882656770029418e-06] },
    // case 137: alt 104.876 km, lat +57.988, lon +19.609, doy 116, ap 84.765
    Case { doy: 116, sec: 1.20888888888888887e+04, alt: 1.04876000000000005e+02,
        g_lat: 5.79883381924198034e+01, g_long: 1.96093163035311591e+01, lst: 4.66531244492676844e+00,
        f107a: 2.23520710059171591e+02, f107: 9.91522491349481072e+01, ap: 8.47645429362880805e+01,
        ap_array: [1.13421550094517976e+00, 6.13421550094517976e+00, 1.11342155009451798e+01, 1.61342155009451815e+01, 2.11342155009451815e+01, 2.61342155009451815e+01, 3.11342155009451815e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [5.04047190318823308e+07, 4.47919474627579407e+11, 3.63135858298684424e+12, 6.46153083028643433e+11, 3.74302917918576050e+10, 2.17565389214310476e-10, 3.41587464437632007e+05, 3.17309084092130375e+09, 3.08025884720505822e-40, 1.02731846489999998e+03, 2.02975495638074420e+02, 2.17565389214310476e-10] },
    // case 138: alt 366.848 km, lat +83.703, lon +52.337, doy 298, ap 100.554
    Case { doy: 298, sec: 4.08888888888888832e+04, alt: 3.66848000000000013e+02,
        g_lat: 8.37026239067055258e+01, g_long: 5.23365890308039070e+01, lst: 1.48471306267449510e+01,
        f107a: 2.41597633136094657e+02, f107: 1.12975778546712803e+02, ap: 1.00554016620498601e+02,
        ap_array: [5.48204158790170126e+00, 1.04820415879017013e+01, 1.54820415879017013e+01, 2.04820415879017013e+01, 2.54820415879017013e+01, 3.04820415879017013e+01, 3.54820415879017048e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [4.11093897513727983e+06, 2.17786946420217186e+08, 1.94775672402586676e+07, 7.71517099034349783e+05, 1.68171960853384871e+03, 6.83426642626345735e-15, 8.59633511358599790e+04, 3.26995764091039030e+06, 9.44432478259528693e+02, 1.02731846489999998e+03, 1.02515596771094602e+03, 6.83429151039007968e-15] },
    // case 139: alt 472.960 km, lat -66.910, lon +85.064, doy 70, ap 116.343
    Case { doy: 70, sec: 6.96888888888888760e+04, alt: 4.72959999999999980e+02,
        g_lat: -6.69096209912536466e+01, g_long: 8.50638617580766550e+01, lst: 2.50289488085631326e+01,
        f107a: 2.59674556213017752e+02, f107: 1.26799307958477499e+02, ap: 1.16343490304709135e+02,
        ap_array: [9.82986767485822277e+00, 1.48298676748582228e+01, 1.98298676748582210e+01, 2.48298676748582210e+01, 2.98298676748582210e+01, 3.48298676748582210e+01, 3.98298676748582210e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [2.67327804416506691e+06, 3.90541792274582162e+07, 9.63866819685694179e+05, 2.48566972837257817e+04, 2.29584225500467198e+01, 1.11816912472028703e-15, 7.71267048043085233e+04, 7.26724867375459173e+05, 7.74170683120980812e+03, 1.02731846489999998e+03, 1.02710885458536791e+03, 1.11837474445372408e-15] },
    // case 140: alt 23.316 km, lat -41.195, lon +117.791, doy 253, ap 132.133
    Case { doy: 253, sec: 2.16888888888888869e+04, alt: 2.33159999999999989e+01,
        g_lat: -4.11953352769679384e+01, g_long: 1.17791134485349403e+02, lst: 6.02469135802469058e+00,
        f107a: 2.77751479289940846e+02, f107: 1.40622837370242223e+02, ap: 1.32132963988919641e+02,
        ap_array: [1.41776937618147443e+01, 1.91776937618147443e+01, 2.41776937618147443e+01, 2.91776937618147443e+01, 3.41776937618147443e+01, 3.91776937618147443e+01, 4.41776937618147443e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], out: [5.78863209640395410e+12, 0.00000000000000000e+00, 8.62594788100822912e+17, 2.31408303803807776e+17, 1.03179275578679940e+16, 5.30709636753440777e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.16841497562172464e+02, 5.30709636753440777e-05] },
    // case 141: alt 97.276 km, lat -15.481, lon +150.518, doy 162, ap 147.922
    Case { doy: 162, sec: 5.04888888888888905e+04, alt: 9.72760000000000105e+01,
        g_lat: -1.54810495626822160e+01, g_long: 1.50518407212622151e+02, lst: 2.40592518388661674e+01,
        f107a: 2.95828402366863884e+02, f107: 1.54446366782006919e+02, ap: 1.47922437673130190e+02,
        ap_array: [1.85255198487712676e+01, 2.35255198487712676e+01, 2.85255198487712676e+01, 3.35255198487712676e+01, 3.85255198487712676e+01, 4.35255198487712676e+01, 4.85255198487712676e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [6.54620410321175829e+07, 1.06242497970201270e+12, 1.50130894305333379e+13, 3.21492287643986084e+12, 2.75421737649741150e+11, 9.15250987021045285e-10, 3.86095390294389101e+05, 6.86062580341598511e+09, 9.87125204427708957e-45, 1.02731846489999998e+03, 1.79507290787641779e+02, 9.15250987021045285e-10] },
    // case 142: alt 322.048 km, lat +10.233, lon -173.779, doy 344, ap 163.712
    Case { doy: 344, sec: 7.92888888888888760e+04, alt: 3.22048000000000002e+02,
        g_lat: 1.02332361516034780e+01, g_long: -1.73779113448534929e+02, lst: 1.04394171281223578e+01,
        f107a: 8.02958579881656789e+01, f107: 1.68269896193771615e+02, ap: 1.63711911357340711e+02,
        ap_array: [2.28733459357277873e+01, 2.78733459357277873e+01, 3.28733459357277837e+01, 3.78733459357277837e+01, 4.28733459357277837e+01, 4.78733459357277837e+01, 5.28733459357277837e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], out: [4.96166078661523573e+06, 4.59635882967730582e+08, 7.17804792338951975e+07, 3.42384201736113802e+06, 1.08289440551581902e+04, 1.59061289684615222e-14, 9.02561879531369923e+04, 6.28891821448085643e+06, 7.55996682386999908e+01, 1.02731846489999998e+03, 1.02139551215713811e+03, 1.59061309763887125e-14] },
    // case 143: alt 952.960 km, lat +35.948, lon -141.052, doy 13, ap 179.501
    Case { doy: 13, sec: 5.68888888888888778e+03, alt: 9.52960000000000036e+02,
        g_lat: 3.59475218658892004e+01, g_long: -1.41051840721262209e+02, lst: -7.82320913450389988e+00,
        f107a: 9.83727810650887591e+01, f107: 1.82093425605536311e+02, ap: 1.79501385041551259e+02,
        ap_array: [2.72211720226843106e+01, 3.22211720226843141e+01, 3.72211720226843141e+01, 4.22211720226843141e+01, 4.72211720226843141e+01, 5.22211720226843141e+01, 5.72211720226843141e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], out: [4.48845935148600489e+05, 3.10464055189322753e+04, 3.62814938945067933e+00, 7.44965377509346216e-03, 4.09184741514141436e-07, 3.91980985231715836e-18, 4.93658775675050128e+04, 1.40980737733009596e+03, 2.85518592078649681e+03, 1.02731846489999998e+03, 1.02731845218640706e+03, 3.99564359037324738e-18] },
    // case 144: alt 11.716 km, lat +61.662, lon -108.325, doy 196, ap 195.291
    Case { doy: 196, sec: 3.44888888888888832e+04, alt: 1.17159999999999993e+01,
        g_lat: 6.16618075801749228e+01, g_long: -1.08324567993989476e+02, lst: 9.58024691358024683e+00,
        f107a: 1.16449704142011839e+02, f107: 1.95916955017301035e+02, ap: 1.95290858725761780e+02,
        ap_array: [3.15689981096408339e+01, 3.65689981096408303e+01, 4.15689981096408303e+01, 4.65689981096408303e+01, 5.15689981096408303e+01, 5.65689981096408303e+01, 6.15689981096408303e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], out: [3.66570465546418594e+13, 0.00000000000000000e+00, 5.46246104063973171e+18, 1.46541442337242086e+18, 6.53392277376121520e+16, 3.36076771463044311e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.20286258487287455e+02, 3.36076771463044311e-04] },
    // case 145: alt 89.676 km, lat +87.376, lon -75.597, doy 105, ap 211.080
    Case { doy: 105, sec: 6.32888888888888905e+04, alt: 8.96760000000000019e+01,
        g_lat: 8.73760932944606452e+01, g_long: -7.55972952667167704e+01, lst: 1.25404272291324617e+01,
        f107a: 1.34526627218934919e+02, f107: 2.09740484429065731e+02, ap: 2.11080332409972272e+02,
        ap_array: [3.59168241965973536e+01, 4.09168241965973536e+01, 4.59168241965973536e+01, 5.09168241965973536e+01, 5.59168241965973536e+01, 6.09168241965973536e+01, 6.59168241965973607e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], out: [7.79335338800008595e+07, 2.22859843145477197e+12, 5.61084001574297734e+13, 1.45658710858527266e+13, 1.82482615633132178e+12, 3.56232181979981857e-09, 3.97853678167815495e+05, 1.30705311121085510e+10, 9.07869582533231814e-50, 1.02731846489999998e+03, 1.84833590846023839e+02, 3.56232181979981857e-09] },
    // case 146: alt 277.248 km, lat -88.426, lon -42.870, doy 287, ap 226.870
    Case { doy: 287, sec: 1.52888888888888887e+04, alt: 2.77248000000000047e+02,
        g_lat: -8.84256559766763814e+01, g_long: -4.28700225394440224e+01, lst: 1.38891207761731250e+00,
        f107a: 1.52603550295857985e+02, f107: 2.23564013840830455e+02, ap: 2.26869806094182820e+02,
        ap_array: [4.02646502835538698e+01, 4.52646502835538698e+01, 5.02646502835538698e+01, 5.52646502835538698e+01, 6.02646502835538698e+01, 6.52646502835538627e+01, 7.02646502835538627e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], out: [6.03574472518593539e+06, 9.91227588810334086e+08, 2.73345903042092979e+08, 1.57592490705758687e+07, 7.29463547107888153e+04, 4.02010235173216220e-14, 9.52454176809265336e+04, 1.23360922621467542e+07, 6.85488125006020854e-01, 1.02731846489999998e+03, 1.01087355871998329e+03, 4.02010235355281909e-14] },
    // case 147: alt 856.960 km, lat -62.711, lon -10.143, doy 59, ap 242.659
    Case { doy: 59, sec: 4.40888888888888905e+04, alt: 8.56960000000000036e+02,
        g_lat: -6.27113702623906732e+01, g_long: -1.01427498121712745e+01, lst: 1.15707302594354964e+01,
        f107a: 1.70680473372781080e+02, f107: 2.37387543252595123e+02, ap: 2.42659279778393369e+02,
        ap_array: [4.46124763705103931e+01, 4.96124763705103931e+01, 5.46124763705103931e+01, 5.96124763705103931e+01, 6.46124763705103931e+01, 6.96124763705103931e+01, 7.46124763705103931e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], out: [6.29248760897074942e+05, 1.19925509975289766e+05, 3.86154031957214841e+01, 2.34433746625901385e-01, 1.19996896250813868e-05, 7.56129924450858743e-18, 5.37165538655121636e+04, 4.59936101263446835e+03, 4.02439983421079978e+03, 1.02731846489999998e+03, 1.02731838495388092e+03, 7.66818730410522641e-18] },
    // case 148: alt 69.716 km, lat -36.997, lon +22.585, doy 241, ap 258.449
    Case { doy: 241, sec: 7.28888888888888760e+04, alt: 6.97160000000000082e+01,
        g_lat: -3.69970845481049579e+01, g_long: 2.25845229151014166e+01, lst: 2.02469135802469111e+01,
        f107a: 1.88757396449704146e+02, f107: 2.51211072664359847e+02, ap: 2.58448753462603861e+02,
        ap_array: [4.89603024574669234e+01, 5.39603024574669234e+01, 5.89603024574669234e+01, 6.39603024574669234e+01, 6.89603024574669234e+01, 7.39603024574669234e+01, 7.89603024574669234e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], out: [9.31769422863549042e+09, 0.00000000000000000e+00, 1.37951578072942550e+15, 3.62349244376892812e+14, 1.64742971615221680e+13, 8.44618405506189893e-08, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.19011867658309171e+02, 8.44618405506189893e-08] },
    // case 149: alt 72.956 km, lat -11.283, lon +55.312, doy 150, ap 274.238
    Case { doy: 150, sec: 2.48888888888888869e+04, alt: 7.29560000000000031e+01,
        g_lat: -1.12827988338192426e+01, g_long: 5.53117956423741930e+01, lst: 1.06010332897385258e+01,
        f107a: 2.06834319526627240e+02, f107: 2.65034602076124543e+02, ap: 2.74238227146814381e+02,
        ap_array: [5.33081285444234396e+01, 5.83081285444234396e+01, 6.33081285444234396e+01, 6.83081285444234396e+01, 7.33081285444234396e+01, 7.83081285444234396e+01, 8.33081285444234396e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], out: [1.05268989012684882e+08, 9.14498320253625000e+12, 7.38039330890788500e+14, 2.82502042720094750e+14, 7.51170538571780312e+13, 5.45422669499321691e-08, 4.01628319031249557e+05, 4.41700175605682678e+10, 5.14845462854302560e-63, 1.02731846489999998e+03, 2.12894208978994783e+02, 5.45422669499321691e-08] },
    // case 150: alt 178.688 km, lat +14.431, lon +88.039, doy 333, ap 290.028
    Case { doy: 333, sec: 5.36888888888888832e+04, alt: 1.78687999999999988e+02,
        g_lat: 1.44314868804664655e+01, g_long: 8.80390683696468841e+01, lst: 2.07828514715567039e+01,
        f107a: 2.24911242603550306e+02, f107: 2.78858131487889295e+02, ap: 2.90027700831024902e+02,
        ap_array: [5.76559546313799558e+01, 6.26559546313799558e+01, 6.76559546313799558e+01, 7.26559546313799558e+01, 7.76559546313799558e+01, 8.26559546313799558e+01, 8.76559546313799558e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], out: [1.04413027568960618e+07, 7.03613248427793312e+09, 7.50057120299954414e+09, 6.78668464451037765e+08, 7.94694766900524311e+06, 5.73778741614752385e-13, 1.17506700125090822e+05, 6.98985304915034324e+07, 2.16919965178515536e-13, 1.02731846489999998e+03, 8.63968150397365434e+02, 5.73778741614752385e-13] },
    // case 151: alt 645.760 km, lat +40.146, lon +120.766, doy 36, ap 6.648
    Case { doy: 36, sec: 8.24888888888888760e+04, alt: 6.45759999999999991e+02,
        g_lat: 4.01457725947521737e+01, g_long: 1.20766341096919689e+02, lst: 3.09646696533748873e+01,
        f107a: 2.42988165680473372e+02, f107: 2.92681660899653934e+02, ap: 6.64819944598337909e+00,
        ap_array: [6.20037807183364862e+01, 6.70037807183364862e+01, 7.20037807183364862e+01, 7.70037807183364862e+01, 8.20037807183364862e+01, 8.70037807183364862e+01, 9.20037807183364862e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], out: [1.36703961670497130e+06, 2.67142593818902923e+06, 8.81990907683891055e+03, 1.16327122624596981e+02, 2.81025612643719773e-02, 8.21700277372106167e-17, 6.52152077708388679e+04, 6.95104983835646563e+04, 8.23703839827458796e+03, 1.02731846489999998e+03, 1.02731301190833688e+03, 8.23888034770687943e-17] },
    // case 152: alt 44.196 km, lat +65.860, lon +153.494, doy 219, ap 22.438
    Case { doy: 219, sec: 8.88888888888888869e+03, alt: 4.41960000000000122e+01,
        g_lat: 6.58600583090378962e+01, g_long: 1.53493613824192380e+02, lst: 2.46913580246913567e+00,
        f107a: 2.61065088757396438e+02, f107: 7.23183391003460230e+01, ap: 2.24376731301939039e+01,
        ap_array: [6.63516068052930024e+01, 7.13516068052930024e+01, 7.63516068052930024e+01, 8.13516068052930024e+01, 8.63516068052930024e+01, 9.13516068052930024e+01, 9.63516068052930024e+01],
        switches: [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], out: [2.44049005978308167e+11, 0.00000000000000000e+00, 3.63670375128612400e+16, 9.75618515357528600e+15, 4.35004319210089750e+14, 2.23747436623665946e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.64098940576330165e+02, 2.23747436623665946e-06] },
    // case 153: alt 110.956 km, lat -84.752, lon -170.804, doy 127, ap 38.227
    Case { doy: 127, sec: 3.76888888888888832e+04, alt: 1.10956000000000003e+02,
        g_lat: -8.47521865889212904e+01, g_long: -1.70803906836964700e+02, lst: -9.17791319995178156e-01,
        f107a: 2.79142011834319533e+02, f107: 8.61418685121107330e+01, ap: 3.82271468144044348e+01,
        ap_array: [7.06994328922495185e+01, 7.56994328922495185e+01, 8.06994328922495185e+01, 8.56994328922495185e+01, 9.06994328922495185e+01, 9.56994328922495185e+01, 1.00699432892249519e+02],
        switches: [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.37606785251932368e+07, 1.08649005437471741e+11, 1.24483941588715625e+12, 2.21876001108540588e+11, 9.04608810659775543e+09, 7.31328806521807365e-11, 7.37595757741307933e+06, 4.63561433222049731e+05, 1.10176497205737162e-37, 1.02731846489999998e+03, 2.47343505902174599e+02, 7.31328806521807365e-11] },
    // case 154: alt 133.888 km, lat -59.038, lon -138.077, doy 310, ap 54.017
    Case { doy: 310, sec: 6.64888888888888905e+04, alt: 1.33888000000000005e+02,
        g_lat: -5.90379008746355680e+01, g_long: -1.38076634109691952e+02, lst: 9.26402686182300528e+00,
        f107a: 2.97218934911242627e+02, f107: 9.99653979238754289e+01, ap: 5.40166204986149623e+01,
        ap_array: [7.50472589792060489e+01, 8.00472589792060489e+01, 8.50472589792060489e+01, 9.00472589792060489e+01, 9.50472589792060489e+01, 1.00047258979206049e+02, 1.05047258979206049e+02],
        switches: [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.70833040420919284e+06, 2.39003009804310760e+10, 1.10135671505672272e+11, 9.55400263800676537e+09, 5.16271894921987116e+08, 6.29592841840537498e-12, 7.14441402550249244e+05, 7.84550521201071795e+06, 1.58472653721294173e-26, 1.24378277838592089e+03, 6.17770874454395084e+02, 6.29592841840537498e-12] },
    // case 155: alt 549.760 km, lat -33.324, lon -105.349, doy 82, ap 69.806
    Case { doy: 82, sec: 1.84888888888888869e+04, alt: 5.49759999999999991e+02,
        g_lat: -3.33236151603498527e+01, g_long: -1.05349361382419232e+02, lst: -1.88748828969214699e+00,
        f107a: 8.16863905325443795e+01, f107: 1.13788927335640139e+02, ap: 6.98060941828254897e+01,
        ap_array: [7.93950850661625651e+01, 8.43950850661625651e+01, 8.93950850661625651e+01, 9.43950850661625651e+01, 9.93950850661625651e+01, 1.04395085066162565e+02, 1.09395085066162565e+02],
        switches: [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.34483659318053955e+06, 1.28274914613411687e+07, 1.00323803084971340e+05, 9.16487992254252731e+02, 7.91537629986788671e-01, 3.70682064922711366e-16, 5.50482210090700464e+04, 4.13554643556773895e+05, 3.64508230687719988e+04, 9.90469234615850496e+02, 9.90460307118663536e+02, 3.71650198783417921e-16] },
    // case 156: alt 32.596 km, lat -7.609, lon -72.622, doy 264, ap 85.596
    Case { doy: 264, sec: 4.72888888888888832e+04, alt: 3.25959999999999965e+01,
        g_lat: -7.60932944606413741e+00, g_long: -7.26220886551465128e+01, lst: 1.31358024691358004e+01,
        f107a: 9.97633136094674597e+01, f107: 1.27612456747404835e+02, ap: 8.55955678670360101e+01,
        ap_array: [8.37429111531190955e+01, 8.87429111531190955e+01, 9.37429111531190955e+01, 9.87429111531190955e+01, 1.03742911153119096e+02, 1.08742911153119096e+02, 1.13742911153119096e+02],
        switches: [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.33175519274823608e+12, 0.00000000000000000e+00, 1.98451908699548800e+17, 5.32386935468279600e+16, 2.37378250590965300e+15, 1.22097121188094123e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.37419311908885788e+02, 1.22097121188094123e-05] },
    // case 157: alt 103.356 km, lat +18.105, lon -39.895, doy 173, ap 101.385
    Case { doy: 173, sec: 7.60888888888888760e+04, alt: 1.03356000000000009e+02,
        g_lat: 1.81049562682215708e+01, g_long: -3.98948159278737648e+01, lst: 1.84761480739442163e+01,
        f107a: 1.17840236686390540e+02, f107: 1.41435986159169545e+02, ap: 1.01385041551246530e+02,
        ap_array: [8.80907372400756117e+01, 9.30907372400756117e+01, 9.80907372400756117e+01, 1.03090737240075612e+02, 1.08090737240075612e+02, 1.13090737240075612e+02, 1.18090737240075612e+02],
        switches: [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.31023745366965905e+07, 2.26136447169665192e+11, 3.23563708140088379e+12, 6.08068390890515747e+11, 3.04903627856788788e+10, 1.90724056491796902e-10, 7.65472418707663845e+06, 3.86909430042374006e+05, 5.03341802557080731e-40, 1.02731846489999998e+03, 2.32202642860116526e+02, 1.90724056491796902e-10] },
    // case 158: alt 357.888 km, lat +43.819, lon -7.168, doy 356, ap 117.175
    Case { doy: 356, sec: 2.80888888888888869e+04, alt: 3.57888000000000034e+02,
        g_lat: 4.38192419825072648e+01, g_long: -7.16754320060101691e+00, lst: 7.32463292242906672e+00,
        f107a: 1.35917159763313606e+02, f107: 1.55259515570934241e+02, ap: 1.17174515235457065e+02,
        ap_array: [9.24385633270321279e+01, 9.74385633270321279e+01, 1.02438563327032128e+02, 1.07438563327032128e+02, 1.12438563327032128e+02, 1.17438563327032128e+02, 1.22438563327032128e+02],
        switches: [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.07638690224678721e+06, 3.04354638527910113e+08, 2.93343563438812084e+07, 9.42804046989236725e+05, 4.93214194024179142e+03, 9.62008885701178454e-15, 5.68437487254659281e+04, 2.96204647126465663e+06, 1.05607451690690937e+03, 1.03710723221848980e+03, 1.03475521985869977e+03, 9.62011690635095229e-15] },
    // case 159: alt 453.760 km, lat +69.534, lon +25.560, doy 8, ap 132.964
    Case { doy: 8, sec: 5.68888888888888905e+04, alt: 4.53759999999999991e+02,
        g_lat: 6.95335276967929872e+01, g_long: 2.55597295266716742e+01, lst: 1.75064511042472475e+01,
        f107a: 1.53994082840236700e+02, f107: 1.69083044982698937e+02, ap: 1.32963988919667599e+02,
        ap_array: [9.67863894139886582e+01, 1.01786389413988658e+02, 1.06786389413988658e+02, 1.11786389413988658e+02, 1.16786389413988658e+02, 1.21786389413988658e+02, 1.26786389413988658e+02],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.53515833347480418e+06, 7.69330001861198694e+07, 7.17882335656551644e+06, 2.57622355990790937e+05, 7.74150634292349423e+02, 2.42399577272601054e-15, 2.86755526286350832e+04, 7.04226431390593178e+05, 3.84476033867459091e+03, 1.23478002650980443e+03, 1.23389326028677306e+03, 2.42409788956060573e-15] },
    // case 160: alt 20.996 km, lat -81.079, lon +58.287, doy 190, ap 148.753
    Case { doy: 190, sec: 8.56888888888888760e+04, alt: 2.09960000000000022e+01,
        g_lat: -8.10787172011661852e+01, g_long: 5.82870022539444221e+01, lst: 2.38024691358024683e+01,
        f107a: 1.72071005917159766e+02, f107: 1.82906574394463689e+02, ap: 1.48753462603878120e+02,
        ap_array: [1.32325141776937616e+00, 6.32325141776937638e+00, 1.13232514177693755e+01, 1.63232514177693773e+01, 2.13232514177693773e+01, 2.63232514177693773e+01, 3.13232514177693773e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.48903890848734961e+12, 0.00000000000000000e+00, 9.66966124124885248e+17, 2.59408002118982976e+17, 1.15663652937198140e+16, 5.94923882646063981e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 1.80845948937124604e+02, 5.94923882646063981e-05] },
    // case 161: alt 95.756 km, lat -55.364, lon +91.014, doy 99, ap 164.543
    Case { doy: 99, sec: 7.11111111111110972e+02, alt: 9.57560000000000002e+01,
        g_lat: -5.53644314868804699e+01, g_long: 9.10142749812171701e+01, lst: 6.26514919627867517e+00,
        f107a: 1.90147928994082861e+02, f107: 1.96730103806228385e+02, ap: 1.64542936288088640e+02,
        ap_array: [5.67107750472589789e+00, 1.06710775047258970e+01, 1.56710775047258970e+01, 2.06710775047258970e+01, 2.56710775047258970e+01, 3.06710775047258970e+01, 3.56710775047259006e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.63496316289846897e+08, 2.83615265993342590e+11, 1.76259430988115547e+13, 4.20578844769987939e+12, 2.00923748595800049e+11, 1.06354060419385659e-09, 2.30481481535815746e+07, 1.88135434002056427e+05, 2.11341435519584147e-45, 1.02731846489999998e+03, 1.64618240314288357e+02, 1.06354060419385659e-09] },
    // case 162: alt 313.088 km, lat -29.650, lon +123.742, doy 281, ap 180.332
    Case { doy: 281, sec: 2.95111111111111095e+04, alt: 3.13088000000000022e+02,
        g_lat: -2.96501457725947546e+01, g_long: 1.23741547708489918e+02, lst: 1.64469673780968577e+01,
        f107a: 2.08224852071005955e+02, f107: 2.10553633217993081e+02, ap: 1.80332409972299189e+02,
        ap_array: [1.00189035916824203e+01, 1.50189035916824203e+01, 2.00189035916824203e+01, 2.50189035916824203e+01, 3.00189035916824203e+01, 3.50189035916824167e+01, 4.00189035916824167e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.97086631565322354e+06, 9.52134848771080971e+08, 2.47085887278575301e+08, 6.20645241065039579e+06, 7.15346410044303484e+04, 3.79294769647649046e-14, 4.54985024869537738e+04, 3.36516613007874563e+07, 7.20998451605138939e+02, 1.33548519173925524e+03, 1.28217939261242327e+03, 3.79294961144837843e-14] },
    // case 163: alt 933.760 km, lat -3.936, lon +156.469, doy 53, ap 196.122
    Case { doy: 53, sec: 5.83111111111111022e+04, alt: 9.33759999999999991e+02,
        g_lat: -3.93586005830904639e+00, g_long: 1.56468820435762666e+02, lst: 2.66287855599150376e+01,
        f107a: 2.26301775147929021e+02, f107: 2.24377162629757805e+02, ap: 1.96121883656509681e+02,
        ap_array: [1.43667296786389400e+01, 1.93667296786389400e+01, 2.43667296786389400e+01, 2.93667296786389400e+01, 3.43667296786389400e+01, 3.93667296786389400e+01, 4.43667296786389400e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [7.85282734583128826e+05, 1.18723715412062258e+05, 2.19710446504080821e+01, 5.97345867399093830e-02, 7.46068195772474434e-06, 8.55356736032960230e-18, 5.42839854477569679e+04, 4.08142503261567072e+03, 1.55336491331353522e+05, 1.08888851311742110e+03, 1.08888848718987515e+03, 1.26793045700903532e-17] },
    // case 164: alt 9.396 km, lat +21.778, lon -167.829, doy 236, ap 211.911
    Case { doy: 236, sec: 1.03111111111111113e+04, alt: 9.39599999999999902e+00,
        g_lat: 2.17784256559766760e+01, g_long: -1.67828700225394442e+02, lst: 2.86419753086419737e+00,
        f107a: 2.44378698224852087e+02, f107: 2.38200692041522473e+02, ap: 2.11911357340720201e+02,
        ap_array: [1.87145557655954597e+01, 2.37145557655954597e+01, 2.87145557655954597e+01, 3.37145557655954633e+01, 3.87145557655954633e+01, 4.37145557655954633e+01, 4.87145557655954633e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.89967505712809531e+13, 0.00000000000000000e+00, 7.30126582114601062e+18, 1.95871058183884032e+18, 8.73340911196380000e+16, 4.49208850462913512e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.43436045156172412e+02, 4.49208850462913512e-04] },
    // case 165: alt 88.156 km, lat +47.493, lon -135.101, doy 145, ap 227.701
    Case { doy: 145, sec: 3.91111111111111095e+04, alt: 8.81560000000000059e+01,
        g_lat: 4.74927113702623842e+01, g_long: -1.35101427498121723e+02, lst: 1.85743569765608285e+00,
        f107a: 2.62455621301775182e+02, f107: 2.52024221453287197e+02, ap: 2.27700831024930750e+02,
        ap_array: [2.30623818525519830e+01, 2.80623818525519830e+01, 3.30623818525519795e+01, 3.80623818525519795e+01, 4.30623818525519795e+01, 4.80623818525519795e+01, 5.30623818525519795e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.17379611045800269e+08, 5.47071455818255310e+10, 7.65303748385403750e+13, 1.90107461763803789e+13, 9.15628300705915283e+11, 4.62923694295639154e-09, 6.32894767109166309e+07, 9.28156204664852849e+04, 1.45242637259338182e-49, 1.02731846489999998e+03, 1.53705255244434994e+02, 4.62923694295639154e-09] },
    // case 166: alt 268.288 km, lat +73.207, lon -102.374, doy 327, ap 243.490
    Case { doy: 327, sec: 6.79111111111110949e+04, alt: 2.68288000000000011e+02,
        g_lat: 7.32069970845481066e+01, g_long: -1.02374154770848989e+02, lst: 1.20392538794742592e+01,
        f107a: 2.80532544378698276e+02, f107: 2.65847750865051921e+02, ap: 2.43490304709141270e+02,
        ap_array: [2.74102079395085063e+01, 3.24102079395085099e+01, 3.74102079395085099e+01, 4.24102079395085099e+01, 4.74102079395085099e+01, 5.24102079395085099e+01, 5.74102079395085099e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.23060965965117258e+06, 5.78714197873001099e+08, 1.43213829349458456e+09, 3.60030407199500740e+08, 1.89869143476914894e+06, 1.01323963797985240e-13, 1.19293269225424228e+04, 5.52699777629621699e+06, 7.06853381337638198e-02, 1.76816801230299984e+03, 1.59661495057705133e+03, 1.01323963799862653e-13] },
    // case 167: alt 837.760 km, lat -77.405, lon -69.647, doy 30, ap 259.280
    Case { doy: 30, sec: 1.99111111111111095e+04, alt: 8.37760000000000105e+02,
        g_lat: -7.74052478134110800e+01, g_long: -6.96468820435762694e+01, lst: 8.87738727959112239e-01,
        f107a: 2.98609467455621314e+02, f107: 2.79671280276816617e+02, ap: 2.59279778393351762e+02,
        ap_array: [3.17580340264650332e+01, 3.67580340264650332e+01, 4.17580340264650332e+01, 4.67580340264650332e+01, 5.17580340264650332e+01, 5.67580340264650332e+01, 6.17580340264650332e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [1.05978996908684509e+05, 5.41470318820887987e+04, 1.85567753858612832e+02, 1.34079971150882682e+00, 1.25317809689569478e-03, 2.24462065668423515e-18, 3.91846080276517096e+04, 1.24922570590964347e+03, 1.54906054583369460e+04, 1.02731846489999998e+03, 1.02731846480570243e+03, 2.65605113765766442e-18] },
    // case 168: alt 67.396 km, lat -51.691, lon -36.920, doy 213, ap 275.069
    Case { doy: 213, sec: 4.87111111111111095e+04, alt: 6.73960000000000008e+01,
        g_lat: -5.16909620991253647e+01, g_long: -3.69196093163035357e+01, lst: 1.35308641975308639e+01,
        f107a: 6.51069640418752869e+01, f107: 2.93494809688581313e+02, ap: 2.75069252077562282e+02,
        ap_array: [3.61058601134215493e+01, 4.11058601134215493e+01, 4.61058601134215493e+01, 5.11058601134215493e+01, 5.61058601134215493e+01, 6.11058601134215493e+01, 6.61058601134215564e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [7.49281371512046051e+09, 0.00000000000000000e+00, 1.11064496198159888e+15, 2.95695599094863312e+14, 1.32634913719473633e+13, 6.82108736362042401e-08, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.29686841855961774e+02, 6.82108736362042401e-08] },
    // case 169: alt 80.556 km, lat -25.977, lon -4.192, doy 122, ap 290.859
    Case { doy: 122, sec: 7.75111111111111095e+04, alt: 8.05559999999999974e+01,
        g_lat: -2.59766763848396494e+01, g_long: -4.19233658903078776e+00, lst: 2.12513750915954773e+01,
        f107a: 8.31838871187983671e+01, f107: 7.31314878892733589e+01, ap: 2.90858725761772860e+02,
        ap_array: [4.04536862003780726e+01, 4.54536862003780726e+01, 5.04536862003780726e+01, 5.54536862003780726e+01, 6.04536862003780726e+01, 6.54536862003780726e+01, 7.04536862003780726e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [2.43803410200312042e+09, 7.63092719524982071e+09, 3.48776050572700000e+14, 8.91215504815495156e+13, 4.13445680376449023e+12, 2.12199944730728894e-08, 4.83066845737134740e+07, 1.26595580945057314e+05, 1.04520736294770937e-54, 1.02731846489999998e+03, 1.99402711624562613e+02, 2.12199944730728894e-08] },
    // case 170: alt 223.488 km, lat -0.262, lon +28.535, doy 304, ap 7.479
    Case { doy: 304, sec: 3.91111111111111086e+03, alt: 2.23488000000000000e+02,
        g_lat: -2.62390670553941163e-01, g_long: 2.85349361382419318e+01, lst: 2.98874882896921523e+00,
        f107a: 1.01260810195721433e+02, f107: 8.69550173010380547e+01, ap: 7.47922437673130247e+00,
        ap_array: [4.48015122873345959e+01, 4.98015122873345959e+01, 5.48015122873345959e+01, 5.98015122873345959e+01, 6.48015122873345888e+01, 6.98015122873345888e+01, 7.48015122873345888e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [1.11289235694782808e+07, 3.22699957382136154e+09, 1.46645228039696097e+09, 4.47704988444956243e+07, 6.15167434466054081e+05, 1.56812666793421735e-13, 1.52973586360236543e+05, 1.93481116588959508e+07, 7.43376136292474410e-06, 9.33291492034214571e+02, 9.04975226493693640e+02, 1.56812666793421937e-13] },
    // case 171: alt 741.760 km, lat +25.452, lon +61.262, doy 76, ap 23.269
    Case { doy: 76, sec: 3.27111111111111095e+04, alt: 7.41759999999999991e+02,
        g_lat: 2.54518950437317528e+01, g_long: 6.12622088655146513e+01, lst: 1.31705670107873960e+01,
        f107a: 1.19337733272644527e+02, f107: 1.00778546712802779e+02, ap: 2.32686980609418264e+01,
        ap_array: [4.91493383742911192e+01, 5.41493383742911192e+01, 5.91493383742911192e+01, 6.41493383742911192e+01, 6.91493383742911192e+01, 7.41493383742911192e+01, 7.91493383742911192e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [1.67709148495582305e+06, 2.38202618768025097e+06, 5.58338209364224076e+03, 3.09140462523929820e+01, 1.35106381434319297e-02, 7.80091168696442219e-17, 5.07538155133099426e+04, 1.40327189330559864e+05, 1.00662605197102093e+04, 1.17726405556583177e+03, 1.17725853975559721e+03, 7.82764767490477367e-17] },
    // case 172: alt 55.796 km, lat +51.166, lon +93.989, doy 259, ap 39.058
    Case { doy: 259, sec: 6.15111111111111095e+04, alt: 5.57960000000000136e+01,
        g_lat: 5.11661807580174468e+01, g_long: 9.39894815927873992e+01, lst: 1.70864197530864175e+01,
        f107a: 1.37414656349567593e+02, f107: 1.14602076124567475e+02, ap: 3.90581717451523573e+01,
        ap_array: [5.34971644612476354e+01, 5.84971644612476354e+01, 6.34971644612476354e+01, 6.84971644612476354e+01, 7.34971644612476354e+01, 7.84971644612476354e+01, 8.34971644612476354e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [5.58673867720241165e+10, 0.00000000000000000e+00, 8.32509578287041000e+15, 2.23337344567064500e+15, 9.95806332068044062e+13, 5.12199774426221336e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.49666421798314076e+02, 5.12199774426221336e-07] },
    // case 173: alt 118.556 km, lat +76.880, lon +126.717, doy 167, ap 54.848
    Case { doy: 167, sec: 1.35111111111111095e+04, alt: 1.18556000000000012e+02,
        g_lat: 7.68804664723031976e+01, g_long: 1.26716754320060147e+02, lst: 1.22008700410904289e+01,
        f107a: 1.55491579426490659e+02, f107: 1.28425605536332171e+02, ap: 5.48476454293628848e+01,
        ap_array: [5.78449905482041586e+01, 6.28449905482041586e+01, 6.78449905482041515e+01, 7.28449905482041515e+01, 7.78449905482041515e+01, 8.28449905482041515e+01, 8.78449905482041515e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [5.54297314299041498e+06, 2.97928775684967880e+10, 3.44728928258161682e+11, 4.50674974290505142e+10, 2.56703515558962679e+09, 1.93788028783001661e-11, 1.70529811357832956e+06, 1.17171870401765476e+06, 5.93629309611163459e-33, 1.02731846489999998e+03, 4.64859406771378417e+02, 1.93788028783001661e-11] },
    // case 174: alt 124.928 km, lat -73.732, lon +159.444, doy 350, ap 70.637
    Case { doy: 350, sec: 4.23111111111111095e+04, alt: 1.24927999999999997e+02,
        g_lat: -7.37317784256559747e+01, g_long: 1.59444027047332895e+02, lst: 2.23826882229086124e+01,
        f107a: 1.73568502503413754e+02, f107: 1.42249134948096895e+02, ap: 7.06371191135734051e+01,
        ap_array: [6.21928166351606890e+01, 6.71928166351606819e+01, 7.21928166351606819e+01, 7.71928166351606819e+01, 8.21928166351606819e+01, 8.71928166351606819e+01, 9.21928166351606819e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [2.49874816920347698e+06, 1.42452981187196331e+10, 2.30525704333220123e+11, 2.83146392456959991e+10, 1.85655048144265652e+09, 1.27205823506620846e-11, 1.30868777649891772e+06, 1.08185938475591922e+06, 2.67595023546423327e-30, 1.52132880418984701e+03, 5.16660629872507911e+02, 1.27205823506620846e-11] },
    // case 175: alt 530.560 km, lat -48.017, lon -164.853, doy 19, ap 86.427
    Case { doy: 19, sec: 7.11111111111110949e+04, alt: 5.30559999999999945e+02,
        g_lat: -4.80174927113702665e+01, g_long: -1.64853493613824185e+02, lst: 8.76285351216480279e+00,
        f107a: 1.91645425580336848e+02, f107: 1.56072664359861562e+02, ap: 8.64265927977839254e+01,
        ap_array: [6.65406427221171981e+01, 7.15406427221171981e+01, 7.65406427221171981e+01, 8.15406427221171981e+01, 8.65406427221171981e+01, 9.15406427221171981e+01, 9.65406427221171981e+01],
        switches: [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [7.93463167740120203e+05, 3.92497535201085433e+07, 3.00076648964119609e+06, 3.54904230736307873e+04, 3.16843288688987286e+02, 1.24921843721823437e-15, 2.14425005655761597e+04, 2.58428905477002403e+06, 6.87395869728571706e+04, 1.34576556439922228e+03, 1.34532726119505605e+03, 1.25104416064823339e-15] },
    // case 176: alt 30.276 km, lat -22.303, lon -132.126, doy 202, ap 102.216
    Case { doy: 202, sec: 2.31111111111111131e+04, alt: 3.02760000000000034e+01,
        g_lat: -2.23032069970845583e+01, g_long: -1.32126220886551465e+02, lst: 6.41975308641975317e+00,
        f107a: 2.09722348657259914e+02, f107: 1.69896193771626287e+02, ap: 1.02216066481994446e+02,
        ap_array: [7.08884688090737285e+01, 7.58884688090737285e+01, 8.08884688090737285e+01, 8.58884688090737285e+01, 9.08884688090737285e+01, 9.58884688090737285e+01, 1.00888468809073728e+02],
        switches: [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.87561097807046484e+12, 0.00000000000000000e+00, 2.79494745432750144e+17, 7.49800553572438880e+16, 3.34317639749384100e+15, 1.71958556751415081e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.27450840767472243e+02, 1.71958556751415081e-05] },
    // case 177: alt 101.836 km, lat +3.411, lon -99.399, doy 110, ap 118.006
    Case { doy: 110, sec: 5.19111111111111022e+04, alt: 1.01836000000000013e+02,
        g_lat: 3.41107871720116407e+00, g_long: -9.93989481592787314e+01, lst: 7.79315654246783573e+00,
        f107a: 2.27799271734182980e+02, f107: 1.83719723183391011e+02, ap: 1.18005540166204980e+02,
        ap_array: [7.52362948960302447e+01, 8.02362948960302447e+01, 8.52362948960302447e+01, 9.02362948960302447e+01, 9.52362948960302447e+01, 1.00236294896030245e+02, 1.05236294896030245e+02],
        switches: [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.97749319490067363e+07, 5.21906387083141846e+11, 4.67652617009971680e+12, 8.20873990184035156e+11, 4.41214773911819153e+10, 2.77761882613294938e-10, 9.77393090357192233e+06, 3.35564026547504182e+05, 1.37678775333633219e-40, 1.02731846489999998e+03, 2.23123444538765142e+02, 2.77761882613294938e-10] },
    // case 178: alt 348.928 km, lat +29.125, lon -66.672, doy 293, ap 133.795
    Case { doy: 293, sec: 8.07111111111110949e+04, alt: 3.48927999999999997e+02,
        g_lat: 2.91253644314868581e+01, g_long: -6.66716754320060261e+01, lst: 1.79749747242860138e+01,
        f107a: 2.45876194811106075e+02, f107: 1.97543252595155707e+02, ap: 1.33795013850415501e+02,
        ap_array: [7.95841209829867609e+01, 8.45841209829867609e+01, 8.95841209829867609e+01, 9.45841209829867609e+01, 9.95841209829867609e+01, 1.04584120982986761e+02, 1.09584120982986761e+02],
        switches: [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.73675111752455868e+06, 9.60078315776136637e+08, 1.84239963858594954e+08, 2.09517061372055579e+06, 3.87465549920187259e+04, 3.50098045661687008e-14, 2.78148698999089611e+04, 3.41929666372767463e+07, 5.66150747010814030e+03, 1.43374988074575708e+03, 1.41636657005053189e+03, 3.50099549358071025e-14] },
    // case 179: alt 434.560 km, lat +54.840, lon -33.944, doy 65, ap 149.584
    Case { doy: 65, sec: 7.11111111111111131e+03, alt: 4.34560000000000002e+02,
        g_lat: 5.48396501457725662e+01, g_long: -3.39444027047332781e+01, lst: -2.87651538340243151e-01,
        f107a: 2.63953117888029169e+02, f107: 2.11366782006920403e+02, ap: 1.49584487534626021e+02,
        ap_array: [8.39319470699432912e+01, 8.89319470699432912e+01, 9.39319470699432912e+01, 9.89319470699432912e+01, 1.03931947069943291e+02, 1.08931947069943291e+02, 1.13931947069943291e+02],
        switches: [0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.36469085748581355e+06, 2.57377289614567071e+08, 5.15651844053577557e+07, 7.74443544330622302e+05, 1.84116822825359013e+04, 9.43953455066943737e-15, 2.02522145337429865e+04, 6.40050880366920307e+06, 1.13156866256912035e+04, 1.49223738178357280e+03, 1.49068901321954945e+03, 9.43983509530621615e-15] },
    // case 180: alt 18.676 km, lat +80.554, lon -1.217, doy 247, ap 165.374
    Case { doy: 247, sec: 3.59111111111111095e+04, alt: 1.86759999999999984e+01,
        g_lat: 8.05539358600582887e+01, g_long: -1.21712997746055862e+00, lst: 9.97530864197530853e+00,
        f107a: 2.82030040964952263e+02, f107: 2.25190311418685127e+02, ap: 1.65373961218836541e+02,
        ap_array: [8.82797731568998074e+01, 9.32797731568998074e+01, 9.82797731568998074e+01, 1.03279773156899807e+02, 1.08279773156899807e+02, 1.13279773156899807e+02, 1.18279773156899807e+02],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.42263514451889395e+13, 0.00000000000000000e+00, 2.11994412599377408e+18, 5.68717410680349760e+17, 2.53577116630745280e+16, 1.30429118348932563e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.14626868412682455e+02, 1.30429118348932563e-04] },
    // case 181: alt 94.236 km, lat -70.058, lon +31.510, doy 156, ap 181.163
    Case { doy: 156, sec: 6.47111111111111022e+04, alt: 9.42360000000000042e+01,
        g_lat: -7.00583090379008695e+01, g_long: 3.15101427498121609e+01, lst: 2.00759848252961177e+01,
        f107a: 6.64974965862539875e+01, f107: 2.39013840830449794e+02, ap: 1.81163434903047090e+02,
        ap_array: [9.26275992438563236e+01, 9.76275992438563236e+01, 1.02627599243856324e+02, 1.07627599243856324e+02, 1.12627599243856324e+02, 1.17627599243856324e+02, 1.22627599243856324e+02],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.18654012611047834e+08, 7.93437118873317749e+11, 2.12247178288285586e+13, 5.57850848017359473e+12, 2.37005064953042084e+11, 1.31966759412626297e-09, 3.38011776736028790e+07, 2.07088398464751575e+05, 1.82070936475672534e-47, 1.02731846489999998e+03, 1.92333621298228906e+02, 1.31966759412626297e-09] },
    // case 182: alt 304.128 km, lat -44.344, lon +64.237, doy 338, ap 196.953
    Case { doy: 338, sec: 1.67111111111111095e+04, alt: 3.04128000000000043e+02,
        g_lat: -4.43440233236151613e+01, g_long: 6.42374154770849088e+01, lst: 8.92446967378096900e+00,
        f107a: 8.45744196631770535e+01, f107: 2.52837370242214519e+02, ap: 1.96952908587257610e+02,
        ap_array: [9.69754253308128540e+01, 1.01975425330812854e+02, 1.06975425330812854e+02, 1.11975425330812854e+02, 1.16975425330812854e+02, 1.21975425330812854e+02, 1.26975425330812854e+02],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.68674148526190547e+05, 5.77222491394149423e+08, 3.77679314286548972e+08, 2.41683814618905671e+07, 6.78606783950135345e+05, 3.55894530299100528e-14, 3.75196282208624762e+04, 5.89145842428429872e+07, 1.71480522539163672e+02, 1.36668463894323168e+03, 1.30500241185740515e+03, 3.55894575844327383e-14] },
    // case 183: alt 914.560 km, lat -18.630, lon +96.965, doy 42, ap 212.742
    Case { doy: 42, sec: 4.55111111111111022e+04, alt: 9.14560000000000059e+02,
        g_lat: -1.86297376093294389e+01, g_long: 9.69646882043576284e+01, lst: 1.91062878555991489e+01,
        f107a: 1.02651342740100148e+02, f107: 2.66660899653979186e+02, ap: 2.12742382271468102e+02,
        ap_array: [1.51228733459357279e+00, 6.51228733459357301e+00, 1.15122873345935730e+01, 1.65122873345935730e+01, 2.15122873345935730e+01, 2.65122873345935730e+01, 3.15122873345935730e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.30206926791506354e+05, 2.85203692951011879e+05, 1.64776909732028059e+02, 9.72595751688645538e-01, 1.75512434004045629e-04, 1.22519960689266858e-17, 3.88379911565472357e+04, 4.66536324538570261e+04, 2.57185408607526013e+05, 1.22296129849306612e+03, 1.22294354427438270e+03, 1.90828405215425750e-17] },
    // case 184: alt 7.076 km, lat +7.085, lon +129.692, doy 224, ap 228.532
    Case { doy: 224, sec: 7.43111111111110949e+04, alt: 7.07600000000000051e+00,
        g_lat: 7.08454810495624088e+00, g_long: 1.29691960931630376e+02, lst: 2.06419753086419711e+01,
        f107a: 1.20728265817023214e+02, f107: 2.80484429065743939e+02, ap: 2.28531855955678651e+02,
        ap_array: [5.86011342155009451e+00, 1.08601134215500945e+01, 1.58601134215500945e+01, 2.08601134215500963e+01, 2.58601134215500963e+01, 3.08601134215500963e+01, 3.58601134215500963e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.21548697278003359e+13, 0.00000000000000000e+00, 9.26202698485413478e+18, 2.48472397922681907e+18, 1.10787735779332736e+17, 5.69844270396631532e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.48670329821668759e+02, 5.69844270396631532e-04] },
    // case 185: alt 86.636 km, lat +32.799, lon +162.419, doy 133, ap 244.321
    Case { doy: 133, sec: 2.63111111111111095e+04, alt: 8.66359999999999957e+01,
        g_lat: 3.27988338192419633e+01, g_long: 1.62419233658903124e+02, lst: 1.81365908859021836e+01,
        f107a: 1.38805188893946308e+02, f107: 2.94307958477508635e+02, ap: 2.44321329639889200e+02,
        ap_array: [1.02079395085066160e+01, 1.52079395085066160e+01, 2.02079395085066160e+01, 2.52079395085066160e+01, 3.02079395085066160e+01, 3.52079395085066125e+01, 4.02079395085066125e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.88594026163013697e+08, 1.02800797028461044e+11, 9.74644091725870156e+13, 2.41602530036068008e+13, 1.15013420845139893e+12, 5.89264239578138422e-09, 8.40386261750815511e+07, 2.37058720601395820e+05, 3.25796800495242313e-50, 1.02731846489999998e+03, 1.55955269328883531e+02, 5.89264239578138422e-09] },
    // case 186: alt 259.328 km, lat +58.513, lon -161.878, doy 316, ap 260.111
    Case { doy: 316, sec: 5.51111111111111095e+04, alt: 2.59327999999999975e+02,
        g_lat: 5.85131195335276857e+01, g_long: -1.61878287002253956e+02, lst: 4.51675617515837757e+00,
        f107a: 1.56882111970869346e+02, f107: 7.39446366782006947e+01, ap: 2.60110803324099720e+02,
        ap_array: [1.45557655954631375e+01, 1.95557655954631358e+01, 2.45557655954631358e+01, 2.95557655954631358e+01, 3.45557655954631358e+01, 3.95557655954631358e+01, 4.45557655954631358e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.20334502763425838e+06, 1.07775671612899375e+09, 1.23005911193080926e+09, 8.64990608346670270e+07, 2.27772835318171326e+06, 9.06476453946615054e-14, 6.96354681309392181e+04, 2.66361496050113067e+06, 2.17797010216685995e-02, 1.33201979103067879e+03, 1.25032430828581369e+03, 9.06476453952399613e-14] },
    // case 187: alt 818.560 km, lat +84.227, lon -129.151, doy 87, ap 275.900
    Case { doy: 87, sec: 8.39111111111111095e+04, alt: 8.18560000000000059e+02,
        g_lat: 8.42274052478134081e+01, g_long: -1.29151014274981208e+02, lst: 1.46985743569765628e+01,
        f107a: 1.74959035047792440e+02, f107: 8.77681660899654048e+01, ap: 2.75900277008310240e+02,
        ap_array: [1.89035916824196590e+01, 2.39035916824196590e+01, 2.89035916824196590e+01, 3.39035916824196590e+01, 3.89035916824196590e+01, 4.39035916824196590e+01, 4.89035916824196590e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.66045522954757151e+05, 3.65299203806967428e+06, 3.14773208242052817e+05, 2.86474350185610019e+04, 2.58610946098820911e+01, 1.15410987865851581e-16, 3.93860594926259582e+04, 1.72748949907490096e+04, 1.05721341589666781e+03, 1.82878689255512631e+03, 1.82840001871418804e+03, 1.15439067454177785e-16] },
    // case 188: alt 65.076 km, lat -66.385, lon -96.424, doy 270, ap 291.690
    Case { doy: 270, sec: 1.77777777777777737e+03, alt: 6.50760000000000076e+01,
        g_lat: -6.63848396501457785e+01, g_long: -9.64237415477084880e+01, lst: 4.93827160493827078e-01,
        f107a: 1.93035958124715563e+02, f107: 1.01591695501730101e+02, ap: 2.91689750692520761e+02,
        ap_array: [2.32514177693761823e+01, 2.82514177693761823e+01, 3.32514177693761823e+01, 3.82514177693761823e+01, 4.32514177693761823e+01, 4.82514177693761823e+01, 5.32514177693761823e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [1.89028529282883224e+10, 0.00000000000000000e+00, 2.81253079572607200e+15, 7.48509519559331375e+14, 3.36432528223875586e+13, 1.72721294566689509e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.31830016883193025e+02, 1.72721294566689509e-07] },
    // case 189: alt 79.036 km, lat -40.671, lon -63.696, doy 179, ap 8.310
    Case { doy: 179, sec: 3.05777777777777774e+04, alt: 7.90360000000000014e+01,
        g_lat: -4.06705539358600632e+01, g_long: -6.36964688204357685e+01, lst: 4.24739590579810944e+00,
        f107a: 2.11112881201638629e+02, f107: 1.15415224913494797e+02, ap: 8.31024930747922497e+00,
        ap_array: [2.75992438563326985e+01, 3.25992438563326985e+01, 3.75992438563326985e+01, 4.25992438563326985e+01, 4.75992438563326985e+01, 5.25992438563326985e+01, 5.75992438563326985e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [1.82051357073171234e+09, 1.94942118304481769e+09, 2.60811120167551594e+14, 6.49678030137507031e+13, 3.10077573566730859e+12, 1.57795459452545936e-08, 5.40738956408428401e+06, 4.79797262260267380e+04, 4.99050250647125820e-58, 1.02731846489999998e+03, 2.05514615402298659e+02, 1.57795459452545936e-08] },
    // case 190: alt 214.528 km, lat -14.956, lon -30.969, doy 361, ap 24.100
    Case { doy: 361, sec: 5.93777777777777665e+04, alt: 2.14528000000000020e+02,
        g_lat: -1.49562682215743337e+01, g_long: -3.09691960931630206e+01, lst: 1.44292140876162875e+01,
        f107a: 2.29189804278561695e+02, f107: 1.29238754325259492e+02, ap: 2.40997229916897489e+01,
        ap_array: [3.19470699432892253e+01, 3.69470699432892218e+01, 4.19470699432892218e+01, 4.69470699432892218e+01, 5.19470699432892218e+01, 5.69470699432892218e+01, 6.19470699432892218e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [6.47024396621883102e+06, 5.16141940121434116e+09, 3.46351188245289278e+09, 6.02281429869833216e+07, 3.26580617635662714e+06, 3.02901216053027442e-13, 5.17634711352988525e+04, 5.89788135751993358e+07, 4.88006535730988244e-06, 1.25831618390445715e+03, 1.16692404128691146e+03, 3.02901216053027594e-13] },
    // case 191: alt 722.560 km, lat +10.758, lon +1.758, doy 5, ap 39.889
    Case { doy: 5, sec: 1.13777777777777756e+04, alt: 7.22560000000000059e+02,
        g_lat: 1.07580174927113603e+01, g_long: 1.75807663410975579e+00, lst: 3.27769893610114371e+00,
        f107a: 2.47266727355484790e+02, f107: 1.43062283737024217e+02, ap: 3.98891966759002798e+01,
        ap_array: [3.62948960302457451e+01, 4.12948960302457451e+01, 4.62948960302457451e+01, 5.12948960302457451e+01, 5.62948960302457451e+01, 6.12948960302457451e+01, 6.62948960302457522e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [1.45300931873411965e+06, 8.31630049249053700e+05, 4.61596249207192102e+02, 9.05750469914824441e-01, 3.63710534164919490e-04, 3.22029363025537466e-17, 7.07147769518612913e+04, 1.41123346422436589e+04, 4.23693250309468640e+04, 9.58880602761113778e+02, 9.58880540653256844e+02, 3.33282655753756957e-17] },
    // case 192: alt 53.476 km, lat +36.472, lon +34.485, doy 187, ap 55.679
    Case { doy: 187, sec: 4.01777777777777737e+04, alt: 5.34760000000000133e+01,
        g_lat: 3.64723032069970827e+01, g_long: 3.44853493613824469e+01, lst: 1.11604938271604937e+01,
        f107a: 2.65343650432407912e+02, f107: 1.56885813148788941e+02, ap: 5.56786703601108073e+01,
        ap_array: [4.06427221172022612e+01, 4.56427221172022612e+01, 5.06427221172022612e+01, 5.56427221172022612e+01, 6.06427221172022612e+01, 6.56427221172022541e+01, 7.06427221172022541e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [7.28963611732474365e+10, 0.00000000000000000e+00, 1.08626736286490040e+16, 2.91412945435766850e+15, 1.29933870609085062e+14, 6.68323719915366466e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.59482349041859379e+02, 6.68323719915366466e-07] },
    // case 193: alt 117.036 km, lat +62.187, lon +67.213, doy 96, ap 71.468
    Case { doy: 96, sec: 6.89777777777777665e+04, alt: 1.17036000000000001e+02,
        g_lat: 6.21865889212827767e+01, g_long: 6.72126220886551380e+01, lst: 2.36413352997374986e+01,
        f107a: 2.83420573509330950e+02, f107: 1.70709342560553637e+02, ap: 7.14681440443213347e+01,
        ap_array: [4.49905482041587845e+01, 4.99905482041587845e+01, 5.49905482041587845e+01, 5.99905482041587845e+01, 6.49905482041587845e+01, 6.99905482041587845e+01, 7.49905482041587845e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [2.98920520853589363e+07, 8.39317159476356964e+10, 4.47924983576598267e+11, 1.86646574063523521e+10, 2.26352254025484848e+09, 2.41907736619757577e-11, 3.43494600570352282e+06, 1.09149878185799858e+06, 2.37137575151020415e-34, 1.02731846489999998e+03, 3.69639377775865853e+02, 2.41907736619757577e-11] },
    // case 194: alt 169.728 km, lat +87.901, lon +99.940, doy 279, ap 87.258
    Case { doy: 279, sec: 2.09777777777777774e+04, alt: 1.69728000000000009e+02,
        g_lat: 8.79008746355684991e+01, g_long: 9.99398948159279144e+01, lst: 1.24898201482223552e+01,
        f107a: 6.78880291306326740e+01, f107: 1.84532871972318333e+02, ap: 8.72576177285318551e+01,
        ap_array: [4.93383742911153078e+01, 5.43383742911153078e+01, 5.93383742911153078e+01, 6.43383742911153149e+01, 6.93383742911153149e+01, 7.43383742911153149e+01, 7.93383742911153149e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [7.70399880365469307e+06, 5.69034621354710484e+09, 1.41596350433986530e+10, 1.63371946401929498e+09, 3.33507953206816055e+07, 8.98695250081082362e-13, 2.69203082297964604e+05, 1.59443057199078333e+07, 3.04984280251160144e-16, 1.22076152242847684e+03, 8.46029573120574923e+02, 8.98695250081082362e-13] },
    // case 195: alt 626.560 km, lat -87.901, lon +132.667, doy 50, ap 103.047
    Case { doy: 50, sec: 4.97777777777777810e+04, alt: 6.26559999999999945e+02,
        g_lat: -8.79008746355685133e+01, g_long: 1.32667167543200605e+02, lst: 2.26716383300405369e+01,
        f107a: 8.59649522075557684e+01, f107: 1.98356401384083057e+02, ap: 1.03047091412742375e+02,
        ap_array: [5.36862003780718382e+01, 5.86862003780718382e+01, 6.36862003780718382e+01, 6.86862003780718453e+01, 7.36862003780718453e+01, 7.86862003780718453e+01, 8.36862003780718453e+01],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [3.30305621490431833e+05, 6.37328697953160107e+06, 3.41872389947236865e+05, 9.48892718329758463e+03, 1.41713357617099440e+01, 1.91837158649327318e-16, 9.59872084205693391e+04, 1.64150899607336090e+05, 5.79354142066830536e+03, 1.38057004736260183e+03, 1.37929769448167008e+03, 1.91991035109460269e-16] },
    // case 196: alt 41.876 km, lat -62.187, lon +165.394, doy 233, ap 118.837
    Case { doy: 233, sec: 7.85777777777777810e+04, alt: 4.18760000000000048e+01,
        g_lat: -6.21865889212827980e+01, g_long: 1.65394440270473410e+02, lst: 2.18271604938271579e+01,
        f107a: 1.04041875284478834e+02, f107: 2.12179930795847753e+02, ap: 1.18836565096952910e+02,
        ap_array: [5.80340264650283544e+01, 6.30340264650283544e+01, 6.80340264650283473e+01, 7.30340264650283473e+01, 7.80340264650283473e+01, 8.30340264650283473e+01, 8.80340264650283473e+01],
        switches: [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.62694579802563507e+11, 0.00000000000000000e+00, 3.91455134177202320e+16, 1.05015668845718060e+16, 4.68239058746060562e+14, 2.40841951435683926e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.51320812621738412e+02, 2.40841951435683926e-06] },
    // case 197: alt 109.436 km, lat -36.472, lon -158.903, doy 142, ap 134.626
    Case { doy: 142, sec: 4.97777777777777737e+03, alt: 1.09436000000000007e+02,
        g_lat: -3.64723032069970898e+01, g_long: -1.58903080390683698e+02, lst: -9.21082264332953216e+00,
        f107a: 1.22118798361401915e+02, f107: 2.26003460207612477e+02, ap: 1.34626038781163430e+02,
        ap_array: [6.23818525519848848e+01, 6.73818525519848777e+01, 7.23818525519848777e+01, 7.73818525519848777e+01, 8.23818525519848777e+01, 8.73818525519848777e+01, 9.23818525519848777e+01],
        switches: [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.25653098048145175e+07, 2.15831003510764709e+11, 1.38315449254252856e+12, 2.35335420713485229e+11, 9.85618350034997749e+09, 8.31772733227505851e-11, 5.11441612480577268e+06, 9.38029969767186092e+05, 4.64698789285867781e-37, 1.02731846489999998e+03, 2.64265472937188292e+02, 8.31772733227505851e-11] },
    // case 198: alt 393.728 km, lat -10.758, lon -126.176, doy 324, ap 150.416
    Case { doy: 324, sec: 3.37777777777777810e+04, alt: 3.93728000000000065e+02,
        g_lat: -1.07580174927113745e+01, g_long: -1.26175807663410978e+02, lst: 9.70995538488651277e-01,
        f107a: 1.40195721438324995e+02, f107: 2.39826989619377144e+02, ap: 1.50415512465373951e+02,
        ap_array: [6.67296786389414081e+01, 7.17296786389414081e+01, 7.67296786389414081e+01, 8.17296786389414081e+01, 8.67296786389414081e+01, 9.17296786389414081e+01, 9.67296786389414081e+01],
        switches: [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.84114819163624942e+06, 2.78809432727896929e+08, 1.89243947908733189e+07, 3.39482684276515094e+05, 2.10608748252188934e+03, 8.48957007669404843e-15, 6.26128853306419478e+04, 6.64214061022340693e+06, 5.61021779676910082e+04, 1.12605873802303427e+03, 1.12387506129397957e+03, 8.49106015054087019e-15] },
    // case 199: alt 415.360 km, lat +14.956, lon -93.449, doy 28, ap 166.205
    Case { doy: 28, sec: 6.25777777777777810e+04, alt: 4.15360000000000014e+02,
        g_lat: 1.49562682215743337e+01, g_long: -9.34485349361382447e+01, lst: 1.11528137203068347e+01,
        f107a: 1.58272644515248089e+02, f107: 2.53650519031141869e+02, ap: 1.66204986149584471e+02,
        ap_array: [7.10775047258979242e+01, 7.60775047258979242e+01, 8.10775047258979242e+01, 8.60775047258979242e+01, 9.10775047258979242e+01, 9.60775047258979242e+01, 1.01077504725897924e+02],
        switches: [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.59655887819030136e+06, 2.37099060744204223e+08, 1.47033109702294040e+07, 2.85421904833026405e+05, 1.13792469869161573e+03, 7.15099187364504178e-15, 3.56066529367417606e+04, 5.35342076604700740e+06, 6.37955930529906473e+04, 1.26210596089081628e+03, 1.25696003434923659e+03, 7.15268628459652896e-15] },
    // case 200: alt 16.356 km, lat +40.671, lon -60.721, doy 210, ap 181.994
    Case { doy: 210, sec: 1.45777777777777756e+04, alt: 1.63559999999999981e+01,
        g_lat: 4.06705539358600277e+01, g_long: -6.07212622088655110e+01, lst: 4.04938271604938205e+00,
        f107a: 1.76349567592171155e+02, f107: 2.67474048442906565e+02, ap: 1.81994459833795020e+02,
        ap_array: [7.54253308128544404e+01, 8.04253308128544404e+01, 8.54253308128544404e+01, 9.04253308128544404e+01, 9.54253308128544404e+01, 1.00425330812854440e+02, 1.05425330812854440e+02],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.91968120452916797e+13, 0.00000000000000000e+00, 2.86061883751544986e+18, 7.67418215540269696e+17, 3.42172922249410520e+16, 1.75998974847785305e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.11614424055592536e+02, 1.75998974847785305e-04] },
    // case 201: alt 92.716 km, lat +66.385, lon -27.994, doy 119, ap 197.784
    Case { doy: 119, sec: 4.33777777777777737e+04, alt: 9.27160000000000082e+01,
        g_lat: 6.63848396501457501e+01, g_long: -2.79939894815927630e+01, lst: 1.01831167506098641e+01,
        f107a: 1.94426490669094250e+02, f107: 2.81297577854671317e+02, ap: 1.97783933518005512e+02,
        ap_array: [7.97731568998109566e+01, 8.47731568998109566e+01, 8.97731568998109566e+01, 9.47731568998109566e+01, 9.97731568998109566e+01, 1.04773156899810957e+02, 1.09773156899810957e+02],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.72472097754692256e+08, 3.31617678644939880e+11, 3.49281292019780898e+13, 7.69410795418817480e+12, 3.93501070607321716e+11, 2.06710856732795841e-09, 3.27101315436171070e+07, 3.16129882332155074e+05, 1.10069244410392063e-46, 1.02731846489999998e+03, 1.70885949096522296e+02, 2.06710856732795841e-09] },
    // case 202: alt 295.168 km, lat -84.227, lon +4.733, doy 301, ap 213.573
    Case { doy: 301, sec: 7.21777777777777665e+04, alt: 2.95168000000000006e+02,
        g_lat: -8.42274052478134081e+01, g_long: 4.73328324567998493e+00, lst: 2.03649349324280458e+01,
        f107a: 2.12503413746017316e+02, f107: 2.95121107266435956e+02, ap: 2.13573407202216032e+02,
        ap_array: [8.41209829867674870e+01, 8.91209829867674870e+01, 9.41209829867674870e+01, 9.91209829867674870e+01, 1.04120982986767487e+02, 1.09120982986767487e+02, 1.14120982986767487e+02],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.30074067072413117e+06, 6.64561417191242337e+08, 1.55020700625816965e+09, 6.55982695986716822e+07, 4.12579553760883166e+06, 9.37003250029924262e-14, 1.71818934392867741e+04, 9.84310528132116422e+06, 5.43021880568543551e+01, 1.86658474692439518e+03, 1.75620859150449087e+03, 9.37003264452585394e-14] },
    // case 203: alt 895.360 km, lat -58.513, lon +37.461, doy 73, ap 229.363
    Case { doy: 73, sec: 2.41777777777777774e+04, alt: 8.95360000000000127e+02,
        g_lat: -5.85131195335276999e+01, g_long: 3.74605559729526760e+01, lst: 9.21341978091289349e+00,
        f107a: 2.30580336822940382e+02, f107: 7.47577854671280306e+01, ap: 2.29362880886426581e+02,
        ap_array: [8.84688090737240032e+01, 9.34688090737240032e+01, 9.84688090737240032e+01, 1.03468809073724003e+02, 1.08468809073724003e+02, 1.13468809073724003e+02, 1.18468809073724003e+02],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.63236575292576686e+05, 7.43121364193628659e+05, 2.39880599666815169e+03, 1.18508314114214119e+01, 1.54918587887084760e-02, 2.40734551335414581e-17, 2.72117341407378226e+04, 7.60310806537528697e+04, 4.24650088245890220e+04, 1.31157954958033679e+03, 1.31157951930816625e+03, 2.52013257679225464e-17] },
    // case 204: alt 4.756 km, lat -32.799, lon +70.188, doy 256, ap 245.152
    Case { doy: 256, sec: 5.29777777777777737e+04, alt: 4.75600000000000023e+00,
        g_lat: -3.27988338192419775e+01, g_long: 7.01878287002254240e+01, lst: 1.47160493827160490e+01,
        f107a: 2.48657259899863448e+02, f107: 8.85813148788927265e+01, ap: 2.45152354570637101e+02,
        ap_array: [9.28166351606805335e+01, 9.78166351606805335e+01, 1.02816635160680534e+02, 1.07816635160680534e+02, 1.12816635160680534e+02, 1.17816635160680534e+02, 1.22816635160680534e+02],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [7.89810812628892969e+13, 0.00000000000000000e+00, 1.17693900599174615e+19, 3.15737427137405747e+18, 1.40779559201702720e+17, 7.24109258445726207e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.62121508676356143e+02, 7.24109258445726207e-04] },
    // case 205: alt 85.116 km, lat -7.085, lon +102.915, doy 164, ap 260.942
    Case { doy: 164, sec: 8.17777777777777665e+04, alt: 8.51159999999999997e+01,
        g_lat: -7.08454810495626930e+00, g_long: 1.02915101427498143e+02, lst: 2.95770561445492568e+01,
        f107a: 2.66734182976786542e+02, f107: 1.02404844290657437e+02, ap: 2.60941828254847621e+02,
        ap_array: [9.71644612476370639e+01, 1.02164461247637064e+02, 1.07164461247637064e+02, 1.12164461247637064e+02, 1.17164461247637064e+02, 1.22164461247637064e+02, 1.27164461247637064e+02],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.65863109231006145e+08, 1.30228052305444855e+11, 1.32234651992167906e+14, 3.00586581944599570e+13, 1.55929237105366113e+12, 7.84998503863235646e-09, 1.22810107871098906e+08, 1.30966447921320680e+05, 8.93461931306199742e-51, 1.02731846489999998e+03, 1.51313815237963951e+02, 7.84998503863235646e-09] },
    // case 206: alt 250.368 km, lat +18.630, lon +135.642, doy 347, ap 276.731
    Case { doy: 347, sec: 8.17777777777777737e+03, alt: 2.50367999999999995e+02,
        g_lat: 1.86297376093294389e+01, g_long: 1.35642374154770891e+02, lst: 1.13144298819229974e+01,
        f107a: 2.84811106053709636e+02, f107: 1.16228373702422147e+02, ap: 2.76731301939058142e+02,
        ap_array: [1.70132325141776941e+00, 6.70132325141776963e+00, 1.17013232514177687e+01, 1.67013232514177687e+01, 2.17013232514177687e+01, 2.67013232514177687e+01, 3.17013232514177687e+01],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.32584438372970484e+07, 6.76200882772312641e+09, 1.34693854042731094e+09, 1.56614283930238206e+07, 3.96515834655497514e+05, 2.44552131894836484e-13, 9.17848378919117968e+04, 5.74277931962010190e+07, 6.68543535171839176e-01, 1.12963886710878933e+03, 1.11337723049106035e+03, 2.44552131912592998e-13] },
    // case 207: alt 799.360 km, lat +44.344, lon +168.370, doy 16, ap 292.521
    Case { doy: 16, sec: 3.69777777777777737e+04, alt: 7.99360000000000127e+02,
        g_lat: 4.43440233236151471e+01, g_long: 1.68369646882043639e+02, lst: 2.14962480637411808e+01,
        f107a: 6.92785616750113746e+01, f107: 1.30051903114186871e+02, ap: 2.92520775623268662e+02,
        ap_array: [6.04914933837429025e+00, 1.10491493383742903e+01, 1.60491493383742920e+01, 2.10491493383742920e+01, 2.60491493383742920e+01, 3.10491493383742920e+01, 3.60491493383742920e+01],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.39121829056873103e+06, 1.06772126638978414e+05, 3.31304520204521040e+01, 1.35989692881765600e-01, 7.28063110095590205e-06, 1.22073941281664455e-17, 6.01982731005429596e+04, 1.39245844432322906e+03, 1.46567528258907987e+04, 9.29716495555649431e+02, 9.29716468205165825e+02, 1.25966774832221053e-17] },
    // case 208: alt 62.756 km, lat +70.058, lon -155.928, doy 199, ap 9.141
    Case { doy: 199, sec: 6.57777777777777810e+04, alt: 6.27560000000000073e+01,
        g_lat: 7.00583090379008695e+01, g_long: -1.55927873779113440e+02, lst: 1.82716049382716044e+01,
        f107a: 8.73554847519344548e+01, f107: 1.43875432525951567e+02, ap: 9.14127423822714746e+00,
        ap_array: [1.03969754253308118e+01, 1.53969754253308118e+01, 2.03969754253308118e+01, 2.53969754253308118e+01, 3.03969754253308118e+01, 3.53969754253308082e+01, 4.03969754253308082e+01],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [3.41809254242373734e+10, 0.00000000000000000e+00, 5.09426851658266400e+15, 1.36642365958452875e+15, 6.09331185694688828e+13, 3.13412211482249975e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.53459577793609583e+02, 3.13412211482249975e-07] },
    // case 209: alt 77.516 km, lat -80.554, lon -123.201, doy 107, ap 24.931
    Case { doy: 107, sec: 1.77777777777777774e+04, alt: 7.75160000000000053e+01,
        g_lat: -8.05539358600583171e+01, g_long: -1.23200601051840721e+02, lst: -3.27510179851777661e+00,
        f107a: 1.05432407828857535e+02, f107: 1.57698961937716263e+02, ap: 2.49307479224376713e+01,
        ap_array: [1.47448015122873333e+01, 1.97448015122873315e+01, 2.47448015122873315e+01, 2.97448015122873315e+01, 3.47448015122873315e+01, 3.97448015122873315e+01, 4.47448015122873315e+01],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [2.03018024541167593e+09, 2.28704313232389987e+08, 2.95010273002167000e+14, 7.78197952008733125e+13, 3.51451195717470850e+12, 1.80792481598994608e-08, 6.69338886400683987e+04, 3.60346498661314690e+04, 1.72277142327106444e-59, 1.02731846489999998e+03, 2.25765440145273516e+02, 1.80792481598994608e-08] },
    // case 210: alt 205.568 km, lat -54.840, lon -90.473, doy 290, ap 40.720
    Case { doy: 290, sec: 4.65777777777777665e+04, alt: 2.05568000000000012e+02,
        g_lat: -5.48396501457725947e+01, g_long: -9.04733283245680013e+01, lst: 6.90671638330040150e+00,
        f107a: 1.23509330905780615e+02, f107: 1.71522491349480958e+02, ap: 4.07202216066481952e+01,
        ap_array: [1.90926275992438548e+01, 2.40926275992438548e+01, 2.90926275992438548e+01, 3.40926275992438548e+01, 3.90926275992438548e+01, 4.40926275992438548e+01, 4.90926275992438548e+01],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [7.48572552563199960e+06, 3.67626954674720812e+09, 3.77634665498871517e+09, 1.80940381812734902e+08, 3.40376173151370138e+06, 2.83724420215693467e-13, 1.08522096792678334e+05, 2.88580165738485642e+07, 1.00097185681518205e-07, 1.10746541234264600e+03, 9.62717873897655068e+02, 2.83724420215693467e-13] },
    // case 211: alt 703.360 km, lat -29.125, lon -57.746, doy 62, ap 56.510
    Case { doy: 62, sec: 7.53777777777777665e+04, alt: 7.03360000000000014e+02,
        g_lat: -2.91253644314868865e+01, g_long: -5.77460555972952818e+01, lst: 1.70885345651185823e+01,
        f107a: 1.41586253982703681e+02, f107: 1.85346020761245654e+02, ap: 5.65096952908587156e+01,
        ap_array: [2.34404536862003781e+01, 2.84404536862003781e+01, 3.34404536862003781e+01, 3.84404536862003781e+01, 4.34404536862003781e+01, 4.84404536862003781e+01, 5.34404536862003781e+01],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [1.21409197589092096e+06, 6.03607494593225233e+06, 3.99620690423862470e+04, 3.18319628697726330e+02, 2.12707526617940484e-01, 1.78570095307853775e-16, 4.09581269905632362e+04, 3.54906338279982388e+05, 2.82608653902353435e+04, 1.28807049208082617e+03, 1.28802502080723434e+03, 1.79320703892618429e-16] },
    // case 212: alt 51.156 km, lat -3.411, lon -25.019, doy 244, ap 72.299
    Case { doy: 244, sec: 2.73777777777777774e+04, alt: 5.11560000000000059e+01,
        g_lat: -3.41107871720117828e+00, g_long: -2.50187828700225339e+01, lst: 7.60493827160493829e+00,
        f107a: 1.59663177059626776e+02, f107: 1.99169550173010379e+02, ap: 7.22991689750692501e+01,
        ap_array: [2.77882797731568978e+01, 3.27882797731569013e+01, 3.77882797731569013e+01, 4.27882797731569013e+01, 4.77882797731569013e+01, 5.27882797731569013e+01, 5.77882797731569013e+01],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [1.15435445339008041e+11, 0.00000000000000000e+00, 1.72016483088264820e+16, 4.61468619180888000e+15, 2.05757516246027969e+14, 1.05832780947323611e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.63480400445500663e+02, 1.05832780947323611e-06] },
    // case 213: alt 115.516 km, lat +22.303, lon +7.708, doy 153, ap 88.089
    Case { doy: 153, sec: 5.61777777777777737e+04, alt: 1.15516000000000005e+02,
        g_lat: 2.23032069970845441e+01, g_long: 7.70848985725021407e+00, lst: 1.61188375954216170e+01,
        f107a: 1.77740100136549842e+02, f107: 2.12993079584775074e+02, ap: 8.80886426592797704e+01,
        ap_array: [3.21361058601134246e+01, 3.71361058601134246e+01, 4.21361058601134246e+01, 4.71361058601134246e+01, 5.21361058601134246e+01, 5.71361058601134246e+01, 6.21361058601134246e+01],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [3.37716929651501775e+07, 1.19363132229007568e+11, 5.84840403492514893e+11, 6.33001482733165741e+10, 3.12303789605722713e+09, 3.39237953429937603e-11, 4.31538306127523724e+06, 1.01534015826440055e+06, 6.76988917918178569e-35, 1.02731846489999998e+03, 3.55851615242150388e+02, 3.39237953429937603e-11] },
    // case 214: alt 160.768 km, lat +48.017, lon +40.436, doy 336, ap 103.878
    Case { doy: 336, sec: 8.49777777777777665e+04, alt: 1.60768000000000001e+02,
        g_lat: 4.80174927113702381e+01, g_long: 4.04357625845229052e+01, lst: 2.63006557772397969e+01,
        f107a: 1.95817023213472936e+02, f107: 2.26816608996539799e+02, ap: 1.03878116343490291e+02,
        ap_array: [3.64839319470699408e+01, 4.14839319470699408e+01, 4.64839319470699408e+01, 5.14839319470699408e+01, 5.64839319470699408e+01, 6.14839319470699408e+01, 6.64839319470699479e+01],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [2.23943442468182966e+07, 1.32426783257445698e+10, 1.97675557699304771e+10, 1.19465460448265290e+09, 3.77517690482513160e+07, 1.33688215992210343e-12, 1.27987737053112069e+05, 1.05400362992260307e+07, 1.09286856282324474e-17, 1.12548042721657794e+03, 8.77993869329934114e+02, 1.33688215992210343e-12] },
    // case 215: alt 607.360 km, lat +73.732, lon +73.163, doy 39, ap 119.668
    Case { doy: 39, sec: 2.84444444444444389e+03, alt: 6.07360000000000014e+02,
        g_lat: 7.37317784256559605e+01, g_long: 7.31630353117956531e+01, lst: 5.66765914424316719e+00,
        f107a: 2.13893946290396002e+02, f107: 2.40640138408304466e+02, ap: 1.19667590027700825e+02,
        ap_array: [4.08317580340264641e+01, 4.58317580340264641e+01, 5.08317580340264641e+01, 5.58317580340264641e+01, 6.08317580340264641e+01, 6.58317580340264641e+01, 7.08317580340264641e+01],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [1.12589264886495587e+06, 1.84417778029859252e+07, 1.71161372292121989e+06, 2.80923805620257299e+04, 2.84374020377426689e+02, 5.81687418379992940e-16, 2.07777704080998155e+04, 1.41842775970731833e+05, 2.12467458442605530e+04, 1.55688071024920418e+03, 1.55647023820168670e+03, 5.82251731949616409e-16] },
    // case 216: alt 39.556 km, lat -76.880, lon +105.890, doy 221, ap 135.457
    Case { doy: 221, sec: 3.16444444444444453e+04, alt: 3.95560000000000045e+01,
        g_lat: -7.68804664723032118e+01, g_long: 1.05890308039068373e+02, lst: 8.79012345679012341e+00,
        f107a: 2.31970869367319068e+02, f107: 2.54463667820069190e+02, ap: 1.35457063711911360e+02,
        ap_array: [4.51795841209829874e+01, 5.01795841209829874e+01, 5.51795841209829874e+01, 6.01795841209829874e+01, 6.51795841209829803e+01, 7.01795841209829803e+01, 7.51795841209829803e+01],
        switches: [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.40732152411914062e+11, 0.00000000000000000e+00, 5.07743062461128000e+16, 1.36212231366446820e+16, 6.07336864162878250e+14, 3.12387855757979564e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.63673756012824299e+02, 3.12387855757979564e-06] },
    // case 217: alt 107.916 km, lat -51.166, lon +138.618, doy 130, ap 151.247
    Case { doy: 130, sec: 6.04444444444444380e+04, alt: 1.07915999999999997e+02,
        g_lat: -5.11661807580174894e+01, g_long: 1.38617580766341121e+02, lst: 2.60312955078795341e+01,
        f107a: 2.50047792444242162e+02, f107: 2.68287197231833886e+02, ap: 1.51246537396121880e+02,
        ap_array: [4.95274102079395107e+01, 5.45274102079395107e+01, 5.95274102079395107e+01, 6.45274102079395107e+01, 6.95274102079395107e+01, 7.45274102079395107e+01, 7.95274102079395107e+01],
        switches: [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.68723015072923377e+07, 1.02081963976601151e+11, 1.37116861483339722e+12, 2.36713182918774048e+11, 1.22110050255658913e+10, 7.98284858619454837e-11, 4.05589049735394958e+06, 2.19710307603827736e+05, 4.67190090409945771e-38, 1.02731846489999998e+03, 2.76170342000346864e+02, 7.98284858619454837e-11] },
    // case 218: alt 384.768 km, lat -25.452, lon +171.345, doy 313, ap 167.036
    Case { doy: 313, sec: 1.24444444444444434e+04, alt: 3.84768000000000029e+02,
        g_lat: -2.54518950437317812e+01, g_long: 1.71344853493613869e+02, lst: 1.48797803563643800e+01,
        f107a: 2.68124715521165285e+02, f107: 2.82110726643598582e+02, ap: 1.67036011080332401e+02,
        ap_array: [5.38752362948960339e+01, 5.88752362948960339e+01, 6.38752362948960339e+01, 6.88752362948960410e+01, 7.38752362948960410e+01, 7.88752362948960410e+01, 8.38752362948960410e+01],
        switches: [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.24140831071936991e+06, 7.30984764182519436e+08, 1.32883868728699550e+08, 1.01779797500504449e+06, 5.89856104686471153e+04, 2.61645011558351422e-14, 1.22342010975998801e+04, 2.09525932750897892e+07, 6.38763333220386776e+04, 1.48948406271115618e+03, 1.48266093772515728e+03, 2.61661977112481754e-14] },
    // case 219: alt 511.360 km, lat +0.262, lon -152.953, doy 85, ap 182.825
    Case { doy: 85, sec: 4.12444444444444453e+04, alt: 5.11360000000000014e+02,
        g_lat: 2.62390670553926952e-01, g_long: -1.52952667167543211e+02, lst: 1.25994564562057576e+00,
        f107a: 2.86201638598088323e+02, f107: 2.95934256055363335e+02, ap: 1.82825484764542949e+02,
        ap_array: [5.82230623818525501e+01, 6.32230623818525501e+01, 6.82230623818525430e+01, 7.32230623818525430e+01, 7.82230623818525430e+01, 8.32230623818525430e+01, 8.82230623818525430e+01],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.75451665843896009e+06, 1.43831706013884902e+08, 4.03618085139163444e+06, 1.78456573368635582e+04, 2.19443716397122159e+02, 4.16328223068565154e-15, 1.80334688119770981e+04, 5.29036034567212686e+06, 3.15948805435624206e+05, 1.29711309618223368e+03, 1.29698920855034157e+03, 4.17167383095802217e-15] },
    // case 220: alt 27.956 km, lat +25.977, lon -120.225, doy 267, ap 198.615
    Case { doy: 267, sec: 7.00444444444444380e+04, alt: 2.79559999999999995e+01,
        g_lat: 2.59766763848396209e+01, g_long: -1.20225394440270478e+02, lst: 1.94567901234567877e+01,
        f107a: 7.06690942193900753e+01, f107: 7.55709342560553665e+01, ap: 1.98614958448753470e+02,
        ap_array: [6.25708884688090805e+01, 6.75708884688090734e+01, 7.25708884688090734e+01, 7.75708884688090734e+01, 8.25708884688090734e+01, 8.75708884688090734e+01, 9.25708884688090734e+01],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.85565326944162402e+12, 0.00000000000000000e+00, 4.25536048209674112e+17, 1.14158555653209280e+17, 5.09004944066001700e+15, 2.61810162414816658e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.23640083374791487e+02, 2.61810162414816658e-05] },
    // case 221: alt 100.316 km, lat +51.691, lon -87.498, doy 176, ap 214.404
    Case { doy: 176, sec: 2.20444444444444453e+04, alt: 1.00316000000000003e+02,
        g_lat: 5.16909620991253291e+01, g_long: -8.74981217129977296e+01, lst: 2.90248675923608701e-01,
        f107a: 8.87460172963131555e+01, f107: 8.93944636678200766e+01, ap: 2.14404432132963962e+02,
        ap_array: [6.69187145557656038e+01, 7.19187145557656038e+01, 7.69187145557656038e+01, 8.19187145557656038e+01, 8.69187145557656038e+01, 9.19187145557656038e+01, 9.69187145557656038e+01],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.10735868642165512e+07, 8.28227946494532623e+10, 4.80700235587086914e+12, 1.10775989814758252e+12, 5.01781603732899704e+10, 2.87805569989279404e-10, 9.78864580217168853e+06, 1.05264933292487767e+05, 3.56211900469225024e-42, 1.02731846489999998e+03, 2.09617483007936983e+02, 2.87805569989279404e-10] },
    // case 222: alt 339.968 km, lat +77.405, lon -54.771, doy 358, ap 230.194
    Case { doy: 358, sec: 5.08444444444444453e+04, alt: 3.39968000000000075e+02,
        g_lat: 7.74052478134110515e+01, g_long: -5.47708489857250385e+01, lst: 1.04720668577417886e+01,
        f107a: 1.06822940373236236e+02, f107: 1.03217993079584772e+02, ap: 2.30193905817174510e+02,
        ap_array: [7.12665406427221200e+01, 7.62665406427221200e+01, 8.12665406427221200e+01, 8.62665406427221200e+01, 9.12665406427221200e+01, 9.62665406427221200e+01, 1.01266540642722120e+02],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.05916620011094725e+06, 2.24754338351102680e+08, 1.96115691196302384e+08, 2.50824080065199770e+07, 1.91813468483165198e+05, 1.64964587780911705e-14, 2.10236636006276676e+04, 2.26783710059429333e+06, 3.47524416974595951e+01, 1.46444047597999270e+03, 1.43111676528840144e+03, 1.64964597011160196e-14] },
    // case 223: alt 991.360 km, lat -73.207, lon -22.044, doy 10, ap 245.983
    Case { doy: 10, sec: 7.96444444444444380e+04, alt: 9.91360000000000127e+02,
        g_lat: -7.32069970845481066e+01, g_long: -2.20435762584523047e+01, lst: 2.06538850395599702e+01,
        f107a: 1.24899863450159316e+02, f107: 1.17041522491349468e+02, ap: 2.45983379501385059e+02,
        ap_array: [7.56143667296786361e+01, 8.06143667296786361e+01, 8.56143667296786361e+01, 9.06143667296786361e+01, 9.56143667296786361e+01, 1.00614366729678636e+02, 1.05614366729678636e+02],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.76641607673923834e+05, 2.46053998685477418e+05, 1.41201946783805374e+03, 1.55742594355039987e+01, 1.13702226161495144e-02, 7.88664201280805235e-18, 2.77734340364031341e+04, 2.83928174515404726e+03, 9.16651984883284058e+03, 1.43461561791345412e+03, 1.43461403315554162e+03, 8.13010477999305396e-18] },
    // case 224: alt 2.436 km, lat -47.493, lon +10.684, doy 193, ap 261.773
    Case { doy: 193, sec: 6.04444444444444434e+03, alt: 2.43599999999999994e+00,
        g_lat: -4.74927113702623984e+01, g_long: 1.06836964688204716e+01, lst: 1.67901234567901225e+00,
        f107a: 1.42976786527082396e+02, f107: 1.30865051903114193e+02, ap: 2.61772853185595579e+02,
        ap_array: [7.99621928166351523e+01, 8.49621928166351523e+01, 8.99621928166351523e+01, 9.49621928166351523e+01, 9.99621928166351523e+01, 1.04962192816635152e+02, 1.09962192816635152e+02],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.04439452825425906e+14, 0.00000000000000000e+00, 1.55630771102692270e+19, 4.17510669637192550e+18, 1.86157746854485344e+17, 9.57515059665694948e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.67669433465549503e+02, 9.57515059665694948e-04] },
    // case 225: alt 83.596 km, lat -21.778, lon +43.411, doy 102, ap 277.562
    Case { doy: 102, sec: 3.48444444444444453e+04, alt: 8.35960000000000036e+01,
        g_lat: -2.17784256559766760e+01, g_long: 4.34109691960931627e+01, lst: 1.25730769587518907e+01,
        f107a: 1.61053709604005462e+02, f107: 1.44688581314878888e+02, ap: 2.77562326869806100e+02,
        ap_array: [8.43100189035916827e+01, 8.93100189035916827e+01, 9.43100189035916827e+01, 9.93100189035916827e+01, 1.04310018903591683e+02, 1.09310018903591683e+02, 1.14310018903591683e+02],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.17576067199664044e+09, 3.83117107984520874e+10, 1.65351316599262750e+14, 4.11000871973773047e+13, 1.95026162197794165e+12, 9.99928874754330928e-09, 1.07836357253681034e+08, 1.41460319447922753e+05, 3.95782534621218262e-52, 1.02731846489999998e+03, 1.61820945914544581e+02, 9.99928874754330928e-09] },
    // case 226: alt 241.408 km, lat +3.936, lon +76.138, doy 284, ap 293.352
    Case { doy: 284, sec: 6.36444444444444453e+04, alt: 2.41408000000000015e+02,
        g_lat: 3.93586005830903218e+00, g_long: 7.61382419233659107e+01, lst: 2.27548951405700741e+01,
        f107a: 1.79130632680928557e+02, f107: 1.58512110726643584e+02, ap: 2.93351800554016620e+02,
        ap_array: [8.86578449905482131e+01, 9.36578449905482131e+01, 9.86578449905482131e+01, 1.03657844990548213e+02, 1.08657844990548213e+02, 1.13657844990548213e+02, 1.18657844990548213e+02],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.24087209616202507e+07, 3.81169887735960007e+09, 1.45700940429219866e+09, 1.35995853984727234e+08, 1.22428772175717843e+06, 1.78026560010909644e-13, 5.91960921682068292e+04, 7.22098139262360185e+07, 3.10017320364681714e-01, 1.21311473737906886e+03, 1.13536361586779685e+03, 1.78026560019143711e-13] },
    // case 227: alt 780.160 km, lat +29.650, lon +108.866, doy 56, ap 9.972
    Case { doy: 56, sec: 1.56444444444444434e+04, alt: 7.80160000000000082e+02,
        g_lat: 2.96501457725947404e+01, g_long: 1.08865514650638659e+02, lst: 1.16033799890549219e+01,
        f107a: 1.97207555757851651e+02, f107: 1.72335640138408309e+02, ap: 9.97229916897506996e+00,
        ap_array: [9.30056710775047293e+01, 9.80056710775047293e+01, 1.03005671077504729e+02, 1.08005671077504729e+02, 1.13005671077504729e+02, 1.18005671077504729e+02, 1.23005671077504729e+02],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [1.44835509143124451e+06, 5.82620871679730481e+05, 3.43344239164525732e+02, 9.25258063735655112e-01, 2.24634295914907843e-04, 2.61049604120435769e-17, 4.13405008367008486e+04, 3.99672640652804985e+04, 5.32235886316960205e+03, 1.02731846489999998e+03, 1.02731844362619859e+03, 2.62463222634493595e-17] },
    // case 228: alt 60.436 km, lat +55.364, lon +141.593, doy 239, ap 25.762
    Case { doy: 239, sec: 4.44444444444444453e+04, alt: 6.04360000000000070e+01,
        g_lat: 5.53644314868804486e+01, g_long: 1.41592787377911350e+02, lst: 1.23456790123456788e+01,
        f107a: 2.15284478834774717e+02, f107: 1.86159169550173033e+02, ap: 2.57617728531855938e+01,
        ap_array: [9.73534971644612597e+01, 1.02353497164461260e+02, 1.07353497164461260e+02, 1.12353497164461260e+02, 1.17353497164461260e+02, 1.22353497164461260e+02, 1.27353497164461260e+02],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [3.31304807503165512e+10, 0.00000000000000000e+00, 4.93694875517329800e+15, 1.32443524255034075e+15, 5.90533125350054062e+13, 3.03744738163445622e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.43175629226456010e+02, 3.03744738163445622e-07] },
    // case 229: alt 75.996 km, lat +81.079, lon +174.320, doy 147, ap 41.551
    Case { doy: 147, sec: 7.32444444444444234e+04, alt: 7.59959999999999951e+01,
        g_lat: 8.10787172011661710e+01, g_long: 1.74320060105184098e+02, lst: 3.19670163526912816e+01,
        f107a: 2.33361401911697783e+02, f107: 1.99982698961937729e+02, ap: 4.15512465373961248e+01,
        ap_array: [1.89035916824196604e+00, 6.89035916824196626e+00, 1.18903591682419663e+01, 1.68903591682419645e+01, 2.18903591682419645e+01, 2.68903591682419645e+01, 3.18903591682419645e+01],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [4.64746834089479923e+09, 5.08240662202662006e+07, 6.94079360338855750e+14, 1.74057234304243219e+14, 8.26718802627137500e+12, 4.20557024505757159e-08, 1.49870750640223740e+01, 7.58549781275051500e+04, 7.12079519442975430e-59, 1.02731846489999998e+03, 2.07487436225528796e+02, 4.20557024505757159e-08] },
    // case 230: alt 196.608 km, lat -69.534, lon -149.977, doy 330, ap 57.341
    Case { doy: 330, sec: 2.52444444444444453e+04, alt: 1.96608000000000004e+02,
        g_lat: -6.95335276967930014e+01, g_long: -1.49977460555972954e+02, lst: -2.98615169138585035e+00,
        f107a: 2.51438324988620877e+02, f107: 2.13806228373702425e+02, ap: 5.73407202216066523e+01,
        ap_array: [6.23818525519848777e+00, 1.12381852551984878e+01, 1.62381852551984878e+01, 2.12381852551984878e+01, 2.62381852551984878e+01, 3.12381852551984878e+01, 3.62381852551984878e+01],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [2.39346659287405480e+06, 3.21356513578596830e+09, 8.06702436627927971e+09, 1.80607881699414372e+08, 1.74842900224246755e+07, 4.72001252986565008e-13, 2.77829861150915676e+04, 3.97110225530629456e+07, 4.24619800085210259e-08, 1.60233820887787056e+03, 1.18336506194520871e+03, 4.72001252986565008e-13] },
    // case 231: alt 684.160 km, lat -43.819, lon -117.250, doy 33, ap 73.130
    Case { doy: 33, sec: 5.40444444444444380e+04, alt: 6.84159999999999968e+02,
        g_lat: -4.38192419825072861e+01, g_long: -1.17250187828700234e+02, lst: 7.19566649043232776e+00,
        f107a: 2.69515248065543972e+02, f107: 2.27629757785467149e+02, ap: 7.31301939058171797e+01,
        ap_array: [1.05860113421550093e+01, 1.55860113421550093e+01, 2.05860113421550111e+01, 2.55860113421550111e+01, 3.05860113421550111e+01, 3.55860113421550111e+01, 4.05860113421550111e+01],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [1.61925279225476133e+06, 1.50778459855589755e+07, 2.07586563199290307e+05, 6.86086525861072914e+02, 2.56427311261389512e+00, 4.39509778813330796e-16, 1.32051768246693555e+04, 7.99620983704494778e+05, 1.48611609437944368e+05, 1.37887135073724380e+03, 1.37883668305851984e+03, 4.43456903160002623e-16] },
    // case 232: alt 48.836 km, lat -18.105, lon -84.523, doy 216, ap 88.920
    Case { doy: 216, sec: 8.28444444444444380e+04, alt: 4.88360000000000127e+01,
        g_lat: -1.81049562682215708e+01, g_long: -8.45229151014275004e+01, lst: 2.30123456790123413e+01,
        f107a: 2.87592171142467009e+02, f107: 2.41453287197231816e+02, ap: 8.89196675900277000e+01,
        ap_array: [1.49338374291115290e+01, 1.99338374291115272e+01, 2.49338374291115272e+01, 2.99338374291115272e+01, 3.49338374291115272e+01, 3.99338374291115272e+01, 4.49338374291115272e+01],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [1.04255521016831345e+11, 0.00000000000000000e+00, 1.55356684553716060e+16, 4.16775377652254100e+15, 1.85829898228927469e+14, 9.55828747913610171e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.65417967122081734e+02, 9.55828747913610171e-07] },
    // case 233: alt 113.996 km, lat +7.609, lon -51.796, doy 125, ap 104.709
    Case { doy: 125, sec: 9.24444444444444343e+03, alt: 1.13996000000000009e+02,
        g_lat: 7.60932944606412320e+00, g_long: -5.17956423741547667e+01, lst: -8.85141590375750020e-01,
        f107a: 7.20596267637687760e+01, f107: 2.55276816608996540e+02, ap: 1.04709141274238220e+02,
        ap_array: [1.92816635160680541e+01, 2.42816635160680541e+01, 2.92816635160680541e+01, 3.42816635160680505e+01, 3.92816635160680505e+01, 4.42816635160680505e+01, 4.92816635160680505e+01],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [4.20297308657754064e+07, 3.23267098036053894e+11, 1.30971026257703809e+12, 2.12925074840860077e+11, 8.70423760897434425e+09, 8.13501877550593232e-11, 8.39143686932854354e+06, 1.99082015719648078e+06, 2.77016894340688558e-34, 1.02731846489999998e+03, 2.15019839380991783e+02, 8.13501877550593232e-11] },
    // case 234: alt 151.808 km, lat +33.324, lon -19.068, doy 307, ap 120.499
    Case { doy: 307, sec: 3.80444444444444453e+04, alt: 1.51807999999999993e+02,
        g_lat: 3.33236151603498456e+01, g_long: -1.90683696468820472e+01, lst: 9.29667659144243252e+00,
        f107a: 9.01365498406918562e+01, f107: 2.69100346020761208e+02, ap: 1.20498614958448755e+02,
        ap_array: [2.36294896030245738e+01, 2.86294896030245738e+01, 3.36294896030245738e+01, 3.86294896030245738e+01, 4.36294896030245738e+01, 4.86294896030245738e+01, 5.36294896030245738e+01],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [2.13244452979711443e+07, 2.53954551905028915e+10, 2.87790767718200035e+10, 3.04183109862477112e+09, 7.75436658854630440e+07, 2.17960628965511258e-12, 4.02354603039350943e+05, 2.48830396729730703e+07, 2.94007229645609216e-20, 1.05609305090428802e+03, 7.09049069752441255e+02, 2.17960628965511258e-12] },
    // case 235: alt 588.160 km, lat +59.038, lon +13.659, doy 79, ap 136.288
    Case { doy: 79, sec: 6.68444444444444380e+04, alt: 5.88160000000000082e+02,
        g_lat: 5.90379008746355396e+01, g_long: 1.36589030803907008e+01, lst: 1.94784947732606106e+01,
        f107a: 1.08213472917614936e+02, f107: 2.82923875432525961e+02, ap: 1.36288088642659289e+02,
        ap_array: [2.79773156899810971e+01, 3.29773156899810971e+01, 3.79773156899810971e+01, 4.29773156899810971e+01, 4.79773156899810971e+01, 5.29773156899810971e+01, 5.79773156899810971e+01],
        switches: [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.06608694066287507e+06, 1.93967816671837121e+07, 4.80535548831814027e+05, 1.21824057558808181e+04, 1.45381822722101948e+01, 5.61369243917773682e-16, 3.52844533126229362e+04, 6.91477896734756185e+05, 1.65197565077167274e+04, 1.30406635710642945e+03, 1.30317763587259265e+03, 5.61808008650618628e-16] },
    // case 236: alt 37.236 km, lat +84.752, lon +46.386, doy 261, ap 152.078
    Case { doy: 261, sec: 1.88444444444444416e+04, alt: 3.72360000000000042e+01,
        g_lat: 8.47521865889212620e+01, g_long: 4.63861758076633919e+01, lst: 5.23456790123456717e+00,
        f107a: 1.26290395994538002e+02, f107: 2.96747404844290656e+02, ap: 1.52077562326869781e+02,
        ap_array: [3.23251417769376204e+01, 3.73251417769376204e+01, 4.23251417769376204e+01, 4.73251417769376204e+01, 5.23251417769376204e+01, 5.73251417769376204e+01, 6.23251417769376204e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.76756265722215332e+11, 0.00000000000000000e+00, 8.59455119155940640e+16, 2.30566024815964200e+16, 1.02803724077837838e+15, 5.28777962010850215e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.41815562779575515e+02, 5.28777962010850215e-06] },
    // case 237: alt 106.396 km, lat -65.860, lon +79.113, doy 170, ap 167.867
    Case { doy: 170, sec: 4.76444444444444380e+04, alt: 1.06396000000000015e+02,
        g_lat: -6.58600583090378962e+01, g_long: 7.91134485349361967e+01, lst: 1.85087978035636453e+01,
        f107a: 1.44367319071461083e+02, f107: 7.63840830449827024e+01, ap: 1.67867036011080330e+02,
        ap_array: [3.66729678638941436e+01, 4.16729678638941436e+01, 4.66729678638941436e+01, 5.16729678638941436e+01, 5.66729678638941436e+01, 6.16729678638941436e+01, 6.66729678638941436e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.73312387614405975e+07, 4.84085971022216568e+10, 1.13979876418061182e+12, 2.52790806612398071e+11, 9.82597892338862991e+09, 6.83443968133334520e-11, 4.91349277503412217e+06, 8.74180986250447750e+04, 1.91834767686762542e-39, 1.02731846489999998e+03, 2.95420198422556723e+02, 6.83443968133334520e-11] },
    // case 238: alt 375.808 km, lat -40.146, lon +111.841, doy 353, ap 183.657
    Case { doy: 353, sec: 7.64444444444444380e+04, alt: 3.75807999999999993e+02,
        g_lat: -4.01457725947521951e+01, g_long: 1.11840721262208888e+02, lst: 2.86906159853818252e+01,
        f107a: 1.62444242148384149e+02, f107: 9.02076124567473983e+01, ap: 1.83656509695290879e+02,
        ap_array: [4.10207939508506598e+01, 4.60207939508506598e+01, 5.10207939508506598e+01, 5.60207939508506598e+01, 6.10207939508506598e+01, 6.60207939508506598e+01, 7.10207939508506598e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.65931812903969456e+06, 1.13313759983652398e+08, 9.69042830617575161e+06, 1.62804281419933366e+05, 1.53274844659977293e+03, 3.50263417493005675e-15, 1.79761244877508201e+05, 3.98596735111790302e+05, 1.73720703597374722e+03, 9.47990479936893166e+02, 9.46732924500607282e+02, 3.50268031514893210e-15] },
    // case 239: alt 492.160 km, lat -14.431, lon +144.568, doy 22, ap 199.446
    Case { doy: 22, sec: 2.84444444444444453e+04, alt: 4.92159999999999968e+02,
        g_lat: -1.44314868804664655e+01, g_long: 1.44567993989481636e+02, lst: 1.75391008338666765e+01,
        f107a: 1.80521165225307243e+02, f107: 1.04031141868512108e+02, ap: 1.99445983379501371e+02,
        ap_array: [4.53686200378071831e+01, 5.03686200378071831e+01, 5.53686200378071831e+01, 6.03686200378071831e+01, 6.53686200378071760e+01, 7.03686200378071760e+01, 7.53686200378071760e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.65113835449655820e+06, 7.69338836676159650e+07, 2.66597636220151884e+06, 2.88760832157608020e+04, 1.23095219430020038e+02, 2.23662625324666444e-15, 4.35517565905036972e+04, 2.15704802353137312e+06, 3.57830484945433738e+05, 1.22336806709269558e+03, 1.22274011271726681e+03, 2.24613023092681496e-15] },
    // case 240: alt 25.636 km, lat +11.283, lon +177.295, doy 204, ap 215.235
    Case { doy: 204, sec: 5.72444444444444453e+04, alt: 2.56359999999999992e+01,
        g_lat: 1.12827988338192284e+01, g_long: 1.77295266716754327e+02, lst: 1.59012345679012341e+01,
        f107a: 1.98598088302230366e+02, f107: 1.17854671280276818e+02, ap: 2.15235457063711891e+02,
        ap_array: [4.97164461247637064e+01, 5.47164461247637064e+01, 5.97164461247637064e+01, 6.47164461247637064e+01, 6.97164461247637064e+01, 7.47164461247637064e+01, 7.97164461247637064e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.93798013580698145e+12, 0.00000000000000000e+00, 5.86819318315617664e+17, 1.57426018524530752e+17, 7.01923927603226100e+15, 3.61039356554496266e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.21413807025569639e+02, 3.61039356554496266e-05] },
    // case 241: alt 98.796 km, lat +36.997, lon -179.459, doy 113, ap 231.025
    Case { doy: 113, sec: 8.60444444444444380e+04, alt: 9.87960000000000065e+01,
        g_lat: 3.69970845481049508e+01, g_long: -1.79459053343350860e+02, lst: 1.19372976783445104e+01,
        f107a: 2.16675011379153432e+02, f107: 1.31678200692041514e+02, ap: 2.31024930747922440e+02,
        ap_array: [5.40642722117202226e+01, 5.90642722117202226e+01, 6.40642722117202226e+01, 6.90642722117202226e+01, 7.40642722117202226e+01, 7.90642722117202226e+01, 8.40642722117202226e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.55526301092770100e+07, 4.00295126769287415e+11, 7.75229017838601172e+12, 1.53575228476807275e+12, 7.81075003579645844e+10, 4.57724448990793712e-10, 1.23777259634010643e+07, 3.66836559275633132e+05, 1.61981721052339322e-42, 1.02731846489999998e+03, 2.10807765044841403e+02, 4.57724448990793712e-10] },
    // case 242: alt 331.008 km, lat +62.711, lon -146.732, doy 296, ap 246.814
    Case { doy: 296, sec: 1.18518518518518505e+02, alt: 3.31008000000000038e+02,
        g_lat: 6.27113702623906590e+01, g_long: -1.46731780616078140e+02, lst: -9.74919689703895287e+00,
        f107a: 2.34751934456076498e+02, f107: 1.45501730103806239e+02, ap: 2.46814404432132960e+02,
        ap_array: [5.84120982986767388e+01, 6.34120982986767388e+01, 6.84120982986767388e+01, 7.34120982986767388e+01, 7.84120982986767388e+01, 8.34120982986767388e+01, 8.84120982986767388e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.29089805501956679e+06, 8.97072644439586401e+08, 5.69379984932672143e+08, 1.69614697660604753e+07, 3.90297078240641451e+05, 5.19946670678023868e-14, 1.43388516178599421e+04, 3.27664185141581148e+07, 1.22754263274003620e+03, 1.54287957610967624e+03, 1.52308401060401570e+03, 5.19946996713347107e-14] },
    // case 243: alt 972.160 km, lat +88.426, lon -114.005, doy 68, ap 262.604
    Case { doy: 68, sec: 2.89185185185185182e+04, alt: 9.72160000000000082e+02,
        g_lat: 8.84256559766763814e+01, g_long: -1.14004507888805406e+02, lst: 4.32621284779228787e-01,
        f107a: 2.52828857532999564e+02, f107: 1.59325259515570934e+02, ap: 2.62603878116343481e+02,
        ap_array: [6.27599243856332691e+01, 6.77599243856332691e+01, 7.27599243856332691e+01, 7.77599243856332691e+01, 8.27599243856332691e+01, 8.77599243856332691e+01, 9.27599243856332691e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.58181803318458144e+05, 1.66084678811471304e+06, 4.48379144092900096e+04, 4.48254693605383920e+02, 1.37438011735897714e+00, 4.85951953875423343e-17, 8.24449751431489130e+03, 2.78452170997985122e+04, 1.50025010879427537e+04, 1.79629690444883749e+03, 1.79629370814695517e+03, 4.89936618164380932e-17] },
    // case 244: alt 14.036 km, lat -87.376, lon -81.277, doy 250, ap 278.393
    Case { doy: 250, sec: 5.77185185185185110e+04, alt: 1.40359999999999996e+01,
        g_lat: -8.73760932944606452e+01, g_long: -8.12772351615327011e+01, lst: 1.60329218106995874e+01,
        f107a: 2.70905780609922658e+02, f107: 1.73148788927335659e+02, ap: 2.78393351800554001e+02,
        ap_array: [6.71077504725897853e+01, 7.21077504725897853e+01, 7.71077504725897853e+01, 8.21077504725897853e+01, 8.71077504725897853e+01, 9.21077504725897853e+01, 9.71077504725897853e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.27572248284755039e+13, 0.00000000000000000e+00, 3.39117484092250982e+18, 9.09750474574281728e+17, 4.05635378596946000e+16, 2.08641321837320707e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.02162036482747794e+02, 2.08641321837320707e-04] },
    // case 245: alt 91.196 km, lat -61.662, lon -48.550, doy 159, ap 294.183
    Case { doy: 159, sec: 9.71851851851851825e+03, alt: 9.11959999999999980e+01,
        g_lat: -6.16618075801749299e+01, g_long: -4.85499624342599532e+01, lst: -5.37075684917741381e-01,
        f107a: 2.88982703686845753e+02, f107: 1.86972318339100354e+02, ap: 2.94182825484764521e+02,
        ap_array: [7.14555765595463015e+01, 7.64555765595463015e+01, 8.14555765595463015e+01, 8.64555765595463015e+01, 9.14555765595463015e+01, 9.64555765595463015e+01, 1.01455576559546301e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [2.83247898557182670e+08, 1.59322842388104218e+11, 3.75143122563617344e+13, 8.20334630756075195e+12, 4.43545790570617371e+11, 2.21311198203383297e-09, 3.27541459485647567e+07, 9.35828935056495538e+04, 1.32207278864932431e-47, 1.02731846489999998e+03, 1.43891732105082269e+02, 2.21311198203383297e-09] },
    // case 246: alt 286.208 km, lat -35.948, lon -15.823, doy 341, ap 10.803
    Case { doy: 341, sec: 3.85185185185185182e+04, alt: 2.86208000000000027e+02,
        g_lat: -3.59475218658892146e+01, g_long: -1.58226897069872052e+01, lst: 9.64474249690044161e+00,
        f107a: 7.34501593081474766e+01, f107: 2.00795847750865050e+02, ap: 1.08033240997229925e+01,
        ap_array: [7.58034026465028319e+01, 8.08034026465028319e+01, 8.58034026465028319e+01, 9.08034026465028319e+01, 9.58034026465028319e+01, 1.00803402646502832e+02, 1.05803402646502832e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [6.09870903084778972e+06, 8.21450728911270142e+08, 1.52633791807691276e+08, 7.88111984855620470e+06, 3.03516060686529199e+04, 2.97132036419225529e-14, 1.69766101680712396e+05, 1.46134236715079378e+07, 2.06244477460401221e+00, 9.40312832710677867e+02, 9.29461970673685983e+02, 2.97132036967010794e-14] },
    // case 247: alt 876.160 km, lat -10.233, lon +16.905, doy 45, ap 26.593
    Case { doy: 45, sec: 6.73185185185185110e+04, alt: 8.76160000000000082e+02,
        g_lat: -1.02332361516035064e+01, g_long: 1.69045830202854859e+01, lst: 1.98265606787186179e+01,
        f107a: 9.15270823850705568e+01, f107: 2.14619377162629746e+02, ap: 2.65927977839335163e+01,
        ap_array: [8.01512287334593481e+01, 8.51512287334593481e+01, 9.01512287334593481e+01, 9.51512287334593481e+01, 1.00151228733459348e+02, 1.05151228733459348e+02, 1.10151228733459348e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [3.94341212862003071e+05, 8.75907316639547062e+04, 2.21118947245516111e+01, 9.99246511178959462e-02, 7.67772556912184494e-06, 5.17626601064604730e-18, 6.27196287076628578e+04, 5.43385848618431737e+03, 6.84943509534630084e+03, 1.04031516801202747e+03, 1.04031277368097267e+03, 5.35818700677844517e-18] },
    // case 248: alt 72.036 km, lat +15.481, lon +49.632, doy 227, ap 42.382
    Case { doy: 227, sec: 1.93185185185185182e+04, alt: 7.20360000000000156e+01,
        g_lat: 1.54810495626822160e+01, g_long: 4.96318557475582054e+01, lst: 5.36625514403292136e+00,
        f107a: 1.09604005461993637e+02, f107: 2.28442906574394470e+02, ap: 4.23822714681440402e+01,
        ap_array: [8.44990548204158785e+01, 8.94990548204158785e+01, 9.44990548204158785e+01, 9.94990548204158785e+01, 1.04499054820415878e+02, 1.09499054820415878e+02, 1.14499054820415878e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [6.85761347669943142e+09, 0.00000000000000000e+00, 1.01424445830925925e+15, 2.68558329016773844e+14, 1.21112777859761074e+13, 6.22121352391276992e-08, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.14635270187969297e+02, 6.22121352391276992e-08] },
    // case 249: alt 72.652 km, lat +41.195, lon +82.359, doy 136, ap 58.172
    Case { doy: 136, sec: 4.81185185185185182e+04, alt: 7.26520000000000010e+01,
        g_lat: 4.11953352769679100e+01, g_long: 8.23591284748309818e+01, lst: 1.88568637090216527e+01,
        f107a: 1.27680928538916703e+02, f107: 2.42266435986159138e+02, ap: 5.81717451523545677e+01,
        ap_array: [8.88468809073723946e+01, 9.38468809073723946e+01, 9.88468809073723946e+01, 1.03846880907372395e+02, 1.08846880907372395e+02, 1.13846880907372395e+02, 1.18846880907372395e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [6.32815668192124081e+09, 7.86606166240530927e+06, 9.35635904343910750e+14, 2.45394334916128156e+14, 1.11661405155606426e+13, 5.72651778642250927e-08, 2.67174276679858021e-28, 6.27882397275096882e+04, 7.74034285005872602e-63, 1.02731846489999998e+03, 2.13412590793859920e+02, 5.72651778642250927e-08] },
    // case 250: alt 176.896 km, lat +66.910, lon +115.086, doy 318, ap 73.961
    Case { doy: 318, sec: 7.69185185185185110e+04, alt: 1.76896000000000015e+02,
        g_lat: 6.69096209912536324e+01, g_long: 1.15086401202103673e+02, lst: 2.90386818908398325e+01,
        f107a: 1.45757851615839797e+02, f107: 2.56089965397923834e+02, ap: 7.39612188365650951e+01,
        ap_array: [9.31947069943289108e+01, 9.81947069943289108e+01, 1.03194706994328911e+02, 1.08194706994328911e+02, 1.13194706994328911e+02, 1.18194706994328911e+02, 1.23194706994328911e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [1.12824775953299440e+07, 9.15477095008055878e+09, 1.15182619448024960e+10, 6.22104953758048773e+08, 1.45089304941232577e+07, 8.13509132916793913e-13, 1.83624107500191370e+05, 3.89316993350020200e+07, 7.65464063261072124e-14, 1.38227382265919732e+03, 9.96336402820333205e+02, 8.13509132916793913e-13] },
    // case 251: alt 641.920 km, lat -83.703, lon +147.814, doy 90, ap 89.751
    Case { doy: 90, sec: 3.31851851851851825e+03, alt: 6.41920000000000073e+02,
        g_lat: -8.37026239067055400e+01, g_long: 1.47813673929376421e+02, lst: 1.07760556282135713e+01,
        f107a: 1.63834774692762863e+02, f107: 2.69913494809688586e+02, ap: 8.97506925207756154e+01,
        ap_array: [9.75425330812854412e+01, 1.02542533081285441e+02, 1.07542533081285441e+02, 1.12542533081285441e+02, 1.17542533081285441e+02, 1.22542533081285441e+02, 1.27542533081285441e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [9.99493864746097126e+05, 1.45228591543139648e+07, 5.82813550919711124e+05, 1.06772014273071836e+04, 2.56357136660124354e+01, 4.34437677151367596e-16, 1.64627979350906389e+04, 6.19127431512400042e+05, 2.42377841939633072e+04, 1.44843583129226431e+03, 1.44825441572744421e+03, 4.35081432699559228e-16] },
    // case 252: alt 43.732 km, lat -57.988, lon -176.484, doy 273, ap 105.540
    Case { doy: 273, sec: 3.21185185185185182e+04, alt: 4.37320000000000064e+01,
        g_lat: -5.79883381924198318e+01, g_long: -1.76483846731780602e+02, lst: 8.92181069958847672e+00,
        f107a: 1.81911697769685929e+02, f107: 2.83737024221453282e+02, ap: 1.05540166204986136e+02,
        ap_array: [2.07939508506616244e+00, 7.07939508506616200e+00, 1.20793950850661620e+01, 1.70793950850661638e+01, 2.20793950850661638e+01, 2.70793950850661638e+01, 3.20793950850661602e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [2.49069065764691559e+11, 0.00000000000000000e+00, 3.71151032623461520e+16, 9.95686875218960200e+15, 4.43952307672553500e+14, 2.28349895479821468e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.57830182762194283e+02, 2.28349895479821468e-06] },
    // case 253: alt 110.652 km, lat -32.274, lon -143.757, doy 182, ap 121.330
    Case { doy: 182, sec: 6.09185185185185182e+04, alt: 1.10652000000000001e+02,
        g_lat: -3.22740524781341094e+01, g_long: -1.43756574004507883e+02, lst: 7.33803909928795051e+00,
        f107a: 1.99988620846609024e+02, f107: 2.97560553633217978e+02, ap: 1.21329639889196670e+02,
        ap_array: [6.42722117202268439e+00, 1.14272211720226835e+01, 1.64272211720226835e+01, 2.14272211720226835e+01, 2.64272211720226835e+01, 3.14272211720226835e+01, 3.64272211720226835e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.68299907440524623e+07, 2.11968100342412811e+11, 9.61915901992249146e+11, 1.56052740309768005e+11, 6.41238742211733818e+09, 5.90553648475589824e-11, 3.90397744624123164e+06, 8.37179009462826652e+05, 1.27413392710197459e-36, 1.02731846489999998e+03, 2.85642175309247023e+02, 5.90553648475589824e-11] },
    // case 254: alt 132.096 km, lat -6.560, lon -111.029, doy 364, ap 137.119
    Case { doy: 364, sec: 1.29185185185185182e+04, alt: 1.32096000000000004e+02,
        g_lat: -6.55976676384840118e+00, g_long: -1.11029301277235149e+02, lst: -3.81347605222719954e+00,
        f107a: 2.18065543923532090e+02, f107: 7.71972318339100383e+01, ap: 1.37119113573407191e+02,
        ap_array: [1.07750472589792068e+01, 1.57750472589792068e+01, 2.07750472589792068e+01, 2.57750472589792068e+01, 3.07750472589792068e+01, 3.57750472589792068e+01, 4.07750472589792068e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.58251130714961700e+07, 4.99458957854333801e+10, 1.02435251789531998e+11, 6.65478585605243492e+09, 3.50326600373392344e+08, 6.46476231920403970e-12, 1.20775890297957580e+06, 5.93069692810801789e+06, 1.01507151210971010e-26, 1.00742745273424623e+03, 5.77046562621574139e+02, 6.46476231920403970e-12] },
    // case 255: alt 545.920 km, lat +19.155, lon -78.302, doy 1, ap 152.909
    Case { doy: 1, sec: 4.17185185185185182e+04, alt: 5.45920000000000073e+02,
        g_lat: 1.91545189504373212e+01, g_long: -7.83020285499624578e+01, lst: 6.36834212959098078e+00,
        f107a: 2.36142467000455156e+02, f107: 9.10207612456747484e+01, ap: 1.52908587257617739e+02,
        ap_array: [1.51228733459357283e+01, 2.01228733459357301e+01, 2.51228733459357301e+01, 3.01228733459357301e+01, 3.51228733459357301e+01, 4.01228733459357301e+01, 4.51228733459357301e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.55073381045766547e+06, 3.72239940966865793e+06, 3.85081903974079023e+03, 1.03904534911529787e+01, 5.81057547740819295e-03, 1.23543788363517977e-16, 1.85059770879210089e+05, 2.63876904449738468e+04, 3.89037217156815677e+03, 7.44354348953426779e+02, 7.44354348850251085e+02, 1.23647116648394798e-16] },
    // case 256: alt 32.132 km, lat +44.869, lon -45.575, doy 184, ap 168.698
    Case { doy: 184, sec: 7.05185185185184964e+04, alt: 3.21319999999999979e+01,
        g_lat: 4.48688046647230294e+01, g_long: -4.55747558226897240e+01, lst: 1.95884773662551410e+01,
        f107a: 2.54219390077378250e+02, f107: 1.04844290657439444e+02, ap: 1.68698060941828231e+02,
        ap_array: [1.94706994328922498e+01, 2.44706994328922498e+01, 2.94706994328922498e+01, 3.44706994328922463e+01, 3.94706994328922463e+01, 4.44706994328922463e+01, 4.94706994328922463e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.50844640410756714e+12, 0.00000000000000000e+00, 2.24781603778367616e+17, 6.03021608456226800e+16, 2.68872515359494400e+15, 1.38296410940208344e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.38854634031707604e+02, 1.38296410940208344e-05] },
    // case 257: alt 103.052 km, lat +70.583, lon -12.847, doy 92, ap 184.488
    Case { doy: 92, sec: 2.25185185185185182e+04, alt: 1.03052000000000007e+02,
        g_lat: 7.05830903790087518e+01, g_long: -1.28474830954169761e+01, lst: 5.39864515989401195e+00,
        f107a: 2.72296313154301345e+02, f107: 1.18667820069204154e+02, ap: 1.84487534626038780e+02,
        ap_array: [2.38185255198487695e+01, 2.88185255198487695e+01, 3.38185255198487695e+01, 3.88185255198487695e+01, 4.38185255198487695e+01, 4.88185255198487695e+01, 5.38185255198487695e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.83168754924216866e+07, 1.15964108113125916e+11, 3.56525842804752051e+12, 6.43264710756735840e+11, 3.69362991276884308e+10, 2.05416352988445096e-10, 1.00872595524548963e+07, 2.26976388906596636e+05, 2.44656231655367873e-40, 1.02731846489999998e+03, 1.90073444864914592e+02, 2.05416352988445096e-10] },
    // case 258: alt 356.096 km, lat -80.029, lon +19.880, doy 275, ap 200.277
    Case { doy: 275, sec: 5.13185185185185110e+04, alt: 3.56096000000000004e+02,
        g_lat: -8.00291545189504347e+01, g_long: 1.98797896318557150e+01, lst: 1.55804633417121892e+01,
        f107a: 2.90373236231224382e+02, f107: 1.32491349480968864e+02, ap: 2.00277008310249300e+02,
        ap_array: [2.81663516068052893e+01, 3.31663516068052928e+01, 3.81663516068052928e+01, 4.31663516068052928e+01, 4.81663516068052928e+01, 5.31663516068052928e+01, 5.81663516068052928e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.67459599313413887e+06, 6.53570729429014564e+08, 4.08254495171416581e+08, 7.62193368199473992e+06, 4.21280459988757619e+05, 3.68742798639703252e-14, 1.78746700583358979e+04, 4.12105101939439029e+06, 1.21676028801042557e+04, 1.53728003036019618e+03, 1.53036817025041478e+03, 3.68746030355028203e-14] },
    // case 259: alt 449.920 km, lat -54.315, lon +52.607, doy 47, ap 216.066
    Case { doy: 47, sec: 8.01185185185185110e+04, alt: 4.49920000000000016e+02,
        g_lat: -5.43148688046647266e+01, g_long: 5.26070623591284630e+01, lst: 2.57622815235303726e+01,
        f107a: 7.48406918525261773e+01, f107: 1.46314878892733589e+02, ap: 2.16066481994459792e+02,
        ap_array: [3.25141776937618161e+01, 3.75141776937618161e+01, 4.25141776937618161e+01, 4.75141776937618161e+01, 5.25141776937618161e+01, 5.75141776937618161e+01, 6.25141776937618161e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.26966411412672838e+05, 3.42374107085829824e+07, 1.13825532592510823e+07, 7.72338246266762959e+05, 4.92475994390689448e+03, 1.48856143881870053e-15, 5.17490023137475728e+04, 2.24447628707377909e+05, 2.38040611085510500e+04, 1.30024043251037619e+03, 1.29580651652238544e+03, 1.48919367468174371e-15] },
    // case 260: alt 20.532 km, lat -28.601, lon +85.334, doy 229, ap 231.856
    Case { doy: 229, sec: 6.51851851851851825e+03, alt: 2.05320000000000000e+01,
        g_lat: -2.86005830903790184e+01, g_long: 8.53343350864012109e+01, lst: 1.81069958847736601e+00,
        f107a: 9.29176149294492575e+01, f107: 1.60138408304498284e+02, ap: 2.31855955678670341e+02,
        ap_array: [3.68620037807183323e+01, 4.18620037807183323e+01, 4.68620037807183323e+01, 5.18620037807183323e+01, 5.68620037807183323e+01, 6.18620037807183323e+01, 6.68620037807183394e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.21543534296909766e+12, 0.00000000000000000e+00, 1.37324092541025382e+18, 3.68399343059808832e+17, 1.64260213292947500e+16, 8.44883603231614936e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.13553906538653081e+02, 8.44883603231614936e-05] },
    // case 261: alt 95.452 km, lat -2.886, lon +118.062, doy 138, ap 247.645
    Case { doy: 138, sec: 3.53185185185185182e+04, alt: 9.54519999999999982e+01,
        g_lat: -2.88629737609329595e+00, g_long: 1.18061607813673959e+02, lst: 1.76814734427222966e+01,
        f107a: 1.10994538006372323e+02, f107: 1.73961937716262980e+02, ap: 2.47645429362880890e+02,
        ap_array: [4.12098298676748556e+01, 4.62098298676748556e+01, 5.12098298676748556e+01, 5.62098298676748556e+01, 6.12098298676748556e+01, 6.62098298676748556e+01, 7.12098298676748556e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.83592465685216039e+07, 9.85982626150472778e+11, 1.10305081506464980e+13, 1.13233132308959473e+12, 1.81336628296526306e+11, 6.11469485839603356e-10, 2.54387354665119114e+05, 1.69100394969349346e+10, 8.36509174345486551e-44, 1.02731846489999998e+03, 2.13254581098195985e+02, 6.11469485839603356e-10] },
    // case 262: alt 311.296 km, lat +22.828, lon +150.789, doy 321, ap 263.435
    Case { doy: 321, sec: 6.41185185185185110e+04, alt: 3.11296000000000049e+02,
        g_lat: 2.28279883381924265e+01, g_long: 1.50788880540946650e+02, lst: 2.78632916245404729e+01,
        f107a: 1.29071461083295389e+02, f107: 1.87785467128027676e+02, ap: 2.63434903047091382e+02,
        ap_array: [4.55576559546313788e+01, 5.05576559546313788e+01, 5.55576559546313788e+01, 6.05576559546313788e+01, 6.55576559546313717e+01, 7.05576559546313717e+01, 7.55576559546313717e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [1.48855946224561743e+07, 1.08356284127326298e+09, 1.34918482146060705e+08, 4.29383701279133186e+06, 2.17472998386897852e+04, 3.54836825115355764e-14, 4.55605343513590633e+04, 4.50919860412712675e+06, 1.43562917073326707e+03, 1.02731846489999998e+03, 1.02096678230678094e+03, 3.54837206418463539e-14] },
    // case 263: alt 929.920 km, lat +48.542, lon -173.509, doy 24, ap 279.224
    Case { doy: 24, sec: 1.61185185185185182e+04, alt: 9.29919999999999959e+02,
        g_lat: 4.85422740524781204e+01, g_long: -1.73508640120210373e+02, lst: -7.08987641953665904e+00,
        f107a: 1.47148384160218484e+02, f107: 2.01608996539792400e+02, ap: 2.79224376731301902e+02,
        ap_array: [4.99054820415879021e+01, 5.49054820415879021e+01, 5.99054820415879021e+01, 6.49054820415879021e+01, 6.99054820415879021e+01, 7.49054820415879021e+01, 7.99054820415879021e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [7.92425932960046921e+05, 3.45475964393787435e+05, 4.24671492892373180e+02, 4.17894298421614252e+00, 1.92454558712727285e-04, 1.48876718295625920e-17, 2.01550725339771525e+04, 1.70698739579407556e+04, 1.06905620853697965e+04, 1.37446185719742448e+03, 1.37441900340229108e+03, 1.51716131585500149e-17] },
    // case 264: alt 8.932 km, lat +74.257, lon -140.781, doy 207, ap 295.014
    Case { doy: 207, sec: 4.49185185185185110e+04, alt: 8.93199999999999861e+00,
        g_lat: 7.42565597667638428e+01, g_long: -1.40781367392937625e+02, lst: 1.24773662551440303e+01,
        f107a: 1.65225307237141578e+02, f107: 2.15432525951557096e+02, ap: 2.95013850415512422e+02,
        ap_array: [5.42533081285444183e+01, 5.92533081285444183e+01, 6.42533081285444183e+01, 6.92533081285444183e+01, 7.42533081285444183e+01, 7.92533081285444183e+01, 8.42533081285444183e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [4.48099945003873438e+13, 0.00000000000000000e+00, 6.67737508052598579e+18, 1.79133941285270630e+18, 7.98714220257115840e+16, 4.10824103314454331e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.31921650046625473e+02, 4.10824103314454331e-04] },
    // case 265: alt 87.852 km, lat -76.356, lon -108.054, doy 115, ap 11.634
    Case { doy: 115, sec: 7.37185185185185110e+04, alt: 8.78520000000000039e+01,
        g_lat: -7.63556851311953295e+01, g_long: -1.08054094665664906e+02, lst: 1.32737599440997034e+01,
        f107a: 1.83302230314064644e+02, f107: 2.29256055363321821e+02, ap: 1.16343490304709132e+01,
        ap_array: [5.86011342155009416e+01, 6.36011342155009416e+01, 6.86011342155009345e+01, 7.36011342155009345e+01, 7.86011342155009345e+01, 8.36011342155009345e+01, 8.86011342155009345e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [4.87313267457232535e+08, 1.42889420003761719e+11, 6.23068147870399844e+13, 1.48524816367744297e+13, 7.27928308829226929e+11, 3.73711749538029310e-09, 5.90700236486247405e+07, 1.29747033283161378e+05, 7.44932026126228493e-51, 1.02731846489999998e+03, 2.18422393803867806e+02, 3.73711749538029310e-09] },
    // case 266: alt 266.496 km, lat -50.641, lon -75.327, doy 298, ap 27.424
    Case { doy: 298, sec: 2.57185185185185182e+04, alt: 2.66495999999999981e+02,
        g_lat: -5.06413994169096213e+01, g_long: -7.53268219383922002e+01, lst: 2.12224479258455290e+00,
        f107a: 2.01379153390987739e+02, f107: 2.43079584775086488e+02, ap: 2.74238227146814353e+01,
        ap_array: [6.29489603024574649e+01, 6.79489603024574649e+01, 7.29489603024574649e+01, 7.79489603024574649e+01, 8.29489603024574649e+01, 8.79489603024574649e+01, 9.29489603024574649e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [5.80915092332070135e+06, 1.69350924199306273e+09, 9.35446234289309740e+08, 1.59375144386374690e+07, 3.67273738301435369e+05, 8.96663695533960242e-14, 3.03282279787863772e+04, 1.28060355464808736e+07, 1.08359293882003849e+00, 1.20385115707689351e+03, 1.17581811682844454e+03, 8.96663695821762575e-14] },
    // case 267: alt 833.920 km, lat -24.927, lon -42.600, doy 70, ap 43.213
    Case { doy: 70, sec: 5.45185185185185182e+04, alt: 8.33920000000000073e+02,
        g_lat: -2.49271137026239131e+01, g_long: -4.25995492111194665e+01, lst: 1.23040629744027363e+01,
        f107a: 2.19456076467910805e+02, f107: 2.56903114186851212e+02, ap: 4.32132963988919627e+01,
        ap_array: [6.72967863894139811e+01, 7.22967863894139811e+01, 7.72967863894139811e+01, 8.22967863894139811e+01, 8.72967863894139811e+01, 9.22967863894139811e+01, 9.72967863894139811e+01],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [1.24020967338207830e+06, 2.69132954265441839e+06, 1.09258225416391360e+04, 1.46565915805921474e+01, 4.08981102018482756e-02, 8.18909356630893374e-17, 5.61851212708301755e+04, 6.76570555983970844e+04, 4.39459506042837347e+03, 1.41264066579540440e+03, 1.41263819914983969e+03, 8.20076561078943163e-17] },
    // case 268: alt 66.932 km, lat +0.787, lon -9.872, doy 252, ap 59.003
    Case { doy: 252, sec: 8.33185185185185110e+04, alt: 6.69320000000000022e+01,
        g_lat: 7.87172011661809279e-01, g_long: -9.87227648384671852e+00, lst: 2.31440329218106982e+01,
        f107a: 2.37532999544833871e+02, f107: 2.70726643598615908e+02, ap: 5.90027700831024902e+01,
        ap_array: [7.16446124763705114e+01, 7.66446124763705114e+01, 8.16446124763705114e+01, 8.66446124763705114e+01, 9.16446124763705114e+01, 9.66446124763705114e+01, 1.01644612476370511e+02],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [1.46532480189718342e+10, 0.00000000000000000e+00, 2.17349487947725025e+15, 5.67887512794531750e+14, 2.59642104124206836e+13, 1.32914347546699717e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.20674775662445597e+02, 1.32914347546699717e-07] },
    // case 269: alt 80.252 km, lat +26.501, lon +22.855, doy 161, ap 74.792
    Case { doy: 161, sec: 1.18518518518518522e+03, alt: 8.02519999999999953e+01,
        g_lat: 2.65014577259475175e+01, g_long: 2.28549962434259442e+01, lst: 1.85288452322428121e+00,
        f107a: 2.55609922621756965e+02, f107: 2.84550173010380604e+02, ap: 7.47922437673130105e+01,
        ap_array: [7.59924385633270276e+01, 8.09924385633270276e+01, 8.59924385633270276e+01, 9.09924385633270276e+01, 9.59924385633270276e+01, 1.00992438563327028e+02, 1.05992438563327028e+02],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [2.20364667133098698e+09, 3.79538687986041546e+09, 3.22671786386412188e+14, 7.73337813454578125e+13, 3.83439692451974121e+12, 1.93604745428449184e-08, 3.02981918892564923e+07, 1.18813327681269744e+05, 1.21543037672212845e-55, 1.02731846489999998e+03, 1.92183521968244634e+02, 1.93604745428449184e-08] },
    // case 270: alt 221.696 km, lat +52.216, lon +55.582, doy 343, ap 90.582
    Case { doy: 343, sec: 2.99851851851851861e+04, alt: 2.21695999999999998e+02,
        g_lat: 5.22157434402332115e+01, g_long: 5.55822689706987205e+01, lst: 1.20347027050424664e+01,
        f107a: 2.73686845698680031e+02, f107: 2.98373702422145357e+02, ap: 9.05817174515235450e+01,
        ap_array: [8.03402646502835438e+01, 8.53402646502835438e+01, 9.03402646502835438e+01, 9.53402646502835438e+01, 1.00340264650283544e+02, 1.05340264650283544e+02, 1.10340264650283544e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.40192705871951561e+07, 5.89675261562364483e+09, 3.12218113402661896e+09, 5.96973283241366744e+07, 2.17000106452629017e+06, 3.06954999580275338e-13, 2.35120209523311751e+04, 7.78801145368993580e+07, 2.50397216578482343e-05, 1.21076974196342280e+03, 1.16056058489440966e+03, 3.06954999580275994e-13] },
    // case 271: alt 737.920 km, lat +77.930, lon +88.310, doy 13, ap 106.371
    Case { doy: 13, sec: 5.87851851851851752e+04, alt: 7.37920000000000073e+02,
        g_lat: 7.79300291545189339e+01, g_long: 8.83095416979714400e+01, lst: 2.22165208868606427e+01,
        f107a: 2.91763768775603126e+02, f107: 7.80103806228373742e+01, ap: 1.06371191135734065e+02,
        ap_array: [8.46880907372400742e+01, 8.96880907372400742e+01, 9.46880907372400742e+01, 9.96880907372400742e+01, 1.04688090737240074e+02, 1.09688090737240074e+02, 1.14688090737240074e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [7.38856003216760117e+05, 4.36154905405561749e+03, 1.00318693581391404e-02, 3.12985675698802299e-06, 1.22130269192466720e-10, 5.88311273378566537e-18, 5.16899213361351693e+05, 1.38251674839244771e+02, 7.98766467212868592e+04, 5.95867131777444570e+02, 5.95867131777444570e+02, 8.00463647070304455e-18] },
    // case 272: alt 55.332 km, lat -72.682, lon +121.037, doy 195, ap 122.161
    Case { doy: 195, sec: 1.07851851851851843e+04, alt: 5.53320000000000078e+01,
        g_lat: -7.26822157434402385e+01, g_long: 1.21036814425244188e+02, lst: 2.99588477366255113e+00,
        f107a: 7.62312243969048779e+01, f107: 9.18339100346020700e+01, ap: 1.22160664819944600e+02,
        ap_array: [8.90359168241965904e+01, 9.40359168241965904e+01, 9.90359168241965904e+01, 1.04035916824196590e+02, 1.09035916824196590e+02, 1.14035916824196590e+02, 1.19035916824196590e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.12668992177186241e+10, 0.00000000000000000e+00, 4.65924658124890100e+15, 1.24993608035164275e+15, 5.57315779816087266e+13, 2.86659170074917795e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.65588733029277932e+02, 2.86659170074917795e-07] },
    // case 273: alt 118.252 km, lat -46.968, lon +153.764, doy 104, ap 137.950
    Case { doy: 104, sec: 3.95851851851851825e+04, alt: 1.18252000000000010e+02,
        g_lat: -4.69679300291545232e+01, g_long: 1.53764087152516936e+02, lst: 2.12468239171636810e+01,
        f107a: 9.43081474738279439e+01, f107: 1.05657439446366780e+02, ap: 1.37950138504155120e+02,
        ap_array: [9.33837429111531065e+01, 9.83837429111531065e+01, 1.03383742911153107e+02, 1.08383742911153107e+02, 1.13383742911153107e+02, 1.18383742911153107e+02, 1.23383742911153107e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.53861131504426450e+07, 8.67790909230677338e+10, 4.21641128770530029e+11, 7.37036977946582794e+10, 2.25282676034433031e+09, 2.59677310939214478e-11, 3.52134004989645584e+06, 1.28402735017619934e+06, 1.33980246371520363e-33, 1.02731846489999998e+03, 3.72793250752276947e+02, 2.59677310939214478e-11] },
    // case 274: alt 123.136 km, lat -21.254, lon -170.533, doy 286, ap 153.740
    Case { doy: 286, sec: 6.83851851851851825e+04, alt: 1.23135999999999996e+02,
        g_lat: -2.12536443148688079e+01, g_long: -1.70533433508640115e+02, lst: 7.62698920641987499e+00,
        f107a: 1.12385070550751038e+02, f107: 1.19480968858131490e+02, ap: 1.53739612188365641e+02,
        ap_array: [9.77315689981096369e+01, 1.02731568998109637e+02, 1.07731568998109637e+02, 1.12731568998109637e+02, 1.17731568998109637e+02, 1.22731568998109637e+02, 1.27731568998109637e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.32445892380946502e+07, 9.43244787327653503e+10, 2.93687080362786316e+11, 3.44458791833569412e+10, 1.59485023182423258e+09, 1.80917865506596713e-11, 2.28257808263676288e+06, 2.80591608896411676e+06, 9.40064078031985688e-31, 1.02731846489999998e+03, 4.19719844776119487e+02, 1.80917865506596713e-11] },
    // case 275: alt 526.720 km, lat +4.461, lon -137.806, doy 58, ap 169.529
    Case { doy: 58, sec: 2.03851851851851861e+04, alt: 5.26720000000000027e+02,
        g_lat: 4.46064139941691451e+00, g_long: -1.37806160781367396e+02, lst: -3.52452594509527461e+00,
        f107a: 1.30461993627674104e+02, f107: 1.33304498269896214e+02, ap: 1.69529085872576161e+02,
        ap_array: [2.26843100189035951e+00, 7.26843100189035951e+00, 1.22684310018903595e+01, 1.72684310018903595e+01, 2.22684310018903595e+01, 2.72684310018903595e+01, 3.22684310018903631e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.40201685967772966e+06, 2.51990281773399338e+07, 2.79473325023362937e+05, 3.38307837282965738e+03, 5.18111323696505455e+00, 7.16140384729503803e-16, 7.79170908353665582e+04, 4.71836864205621008e+05, 2.68986630285797466e+05, 1.04395093184544021e+03, 1.04376946636350021e+03, 7.23284669629894604e-16] },
    // case 276: alt 29.812 km, lat +30.175, lon -105.079, doy 241, ap 185.319
    Case { doy: 241, sec: 4.91851851851851898e+04, alt: 2.98120000000000012e+01,
        g_lat: 3.01749271137026227e+01, g_long: -1.05078888054094662e+02, lst: 1.36625514403292190e+01,
        f107a: 1.48538916704597170e+02, f107: 1.47128027681660910e+02, ap: 1.85318559556786710e+02,
        ap_array: [6.61625708884688013e+00, 1.16162570888468792e+01, 1.66162570888468792e+01, 2.16162570888468792e+01, 2.66162570888468792e+01, 3.16162570888468792e+01, 3.66162570888468792e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.18216293498990015e+12, 0.00000000000000000e+00, 3.25175679359278912e+17, 8.72348795017137600e+16, 3.88958888865601450e+15, 2.00063655675189692e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.27746511032993709e+02, 2.00063655675189692e-05] },
    // case 277: alt 101.532 km, lat +55.889, lon -72.352, doy 149, ap 201.108
    Case { doy: 149, sec: 7.79851851851851825e+04, alt: 1.01532000000000011e+02,
        g_lat: 5.58892128279883309e+01, g_long: -7.23516153268219426e+01, lst: 1.68391104185410896e+01,
        f107a: 1.66615839781520265e+02, f107: 1.60951557093425606e+02, ap: 2.01108033240997202e+02,
        ap_array: [1.09640831758034025e+01, 1.59640831758034025e+01, 2.09640831758034025e+01, 2.59640831758034025e+01, 3.09640831758034025e+01, 3.59640831758034025e+01, 4.09640831758034025e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.84665488203552738e+06, 1.24179363638091278e+11, 3.87705437661218018e+12, 1.49195689939189258e+12, 3.27899761526342468e+11, 2.84561910495761008e-10, 8.74011802470525290e+04, 1.41532636309302258e+09, 2.99307826109632767e-41, 1.02731846489999998e+03, 1.96629125768593155e+02, 2.84561910495761008e-10] },
    // case 278: alt 347.136 km, lat +81.603, lon -39.624, doy 332, ap 216.898
    Case { doy: 332, sec: 4.38518518518518522e+03, alt: 3.47135999999999967e+02,
        g_lat: 8.16034985422740533e+01, g_long: -3.96243425995492089e+01, lst: -1.42351584408517362e+00,
        f107a: 1.84692762858443359e+02, f107: 1.74775086505190302e+02, ap: 2.16897506925207722e+02,
        ap_array: [1.53119092627599240e+01, 2.03119092627599258e+01, 2.53119092627599258e+01, 3.03119092627599258e+01, 3.53119092627599258e+01, 4.03119092627599258e+01, 4.53119092627599258e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [9.65699361147090909e+05, 9.91303838315099329e+07, 7.82032309831155092e+07, 3.90670755931395013e+06, 5.09191202821859988e+04, 6.48951030623556203e-15, 1.98000044656131940e+04, 1.88067498817780783e+05, 2.15524930791071085e+02, 1.02731846489999998e+03, 1.02705771971771242e+03, 6.48951603057772437e-15] },
    // case 279: alt 430.720 km, lat -69.009, lon -6.897, doy 35, ap 232.687
    Case { doy: 35, sec: 3.31851851851851825e+04, alt: 4.30720000000000027e+02,
        g_lat: -6.90087463556851333e+01, g_long: -6.89706987227646096e+00, lst: 8.75830233773300826e+00,
        f107a: 2.02769685935366454e+02, f107: 1.88598615916954998e+02, ap: 2.32686980609418271e+02,
        ap_array: [1.96597353497164455e+01, 2.46597353497164455e+01, 2.96597353497164455e+01, 3.46597353497164420e+01, 3.96597353497164420e+01, 4.46597353497164420e+01, 4.96597353497164420e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [4.76036230935480038e+05, 6.92623529472325146e+07, 5.07106558917633668e+07, 1.52989742802922125e+06, 4.57318856507649325e+04, 4.32243601172523250e-15, 1.29877844290622834e+04, 1.64842696720692795e+06, 5.75194997622326191e+04, 1.81098002589648286e+03, 1.78434563504637708e+03, 4.32396372963891734e-15] },
    // case 280: alt 18.212 km, lat -43.294, lon +25.830, doy 218, ap 248.476
    Case { doy: 218, sec: 6.19851851851851898e+04, alt: 1.82120000000000033e+01,
        g_lat: -4.32944606413994180e+01, g_long: 2.58302028549962301e+01, lst: 1.72181069958847743e+01,
        f107a: 2.20846609012289520e+02, f107: 2.02422145328719722e+02, ap: 2.48476454293628791e+02,
        ap_array: [2.40075614366729653e+01, 2.90075614366729653e+01, 3.40075614366729653e+01, 3.90075614366729653e+01, 4.40075614366729653e+01, 4.90075614366729653e+01, 5.40075614366729653e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [1.52783147279634570e+13, 0.00000000000000000e+00, 2.27670275737379251e+18, 6.10771048720667392e+17, 2.72327800323266200e+16, 1.40073660312932017e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.11737856699109159e+02, 1.40073660312932017e-04] },
    // case 281: alt 93.932 km, lat -17.580, lon +58.557, doy 127, ap 264.266
    Case { doy: 127, sec: 1.39851851851851861e+04, alt: 9.39320000000000022e+01,
        g_lat: -1.75801749271137027e+01, g_long: 5.85574755822689781e+01, lst: 7.78860536803603942e+00,
        f107a: 2.38923532089212586e+02, f107: 2.16245674740484418e+02, ap: 2.64265927977839283e+02,
        ap_array: [2.83553875236294886e+01, 3.33553875236294886e+01, 3.83553875236294886e+01, 4.33553875236294886e+01, 4.83553875236294886e+01, 5.33553875236294886e+01, 5.83553875236294886e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [1.91959304913917154e+08, 4.72621486237653503e+11, 1.96577339123840117e+13, 4.04035426847234229e+12, 2.13419579785782593e+11, 1.15504029201022035e-09, 2.09478097730198801e+07, 2.09378822781246185e+05, 5.48487529258858448e-45, 1.02731846489999998e+03, 1.91169584986982244e+02, 1.15504029201022035e-09] },
    // case 282: alt 302.336 km, lat +8.134, lon +91.285, doy 309, ap 280.055
    Case { doy: 309, sec: 4.27851851851851825e+04, alt: 3.02336000000000013e+02,
        g_lat: 8.13411078717200553e+00, g_long: 9.12847483095417260e+01, lst: 1.79704235498542211e+01,
        f107a: 2.57000455166135680e+02, f107: 2.30069204152249142e+02, ap: 2.80055401662049803e+02,
        ap_array: [3.27032136105860118e+01, 3.77032136105860118e+01, 4.27032136105860118e+01, 4.77032136105860118e+01, 5.27032136105860118e+01, 5.77032136105860118e+01, 6.27032136105860118e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [1.25050771428824477e+07, 2.61139307300463724e+09, 6.08567562288797021e+08, 6.51217823612247873e+06, 2.77824078812990629e+05, 9.92449574456474164e-14, 2.35412562659785108e+04, 4.95993938903106451e+07, 2.37658567014801247e+03, 1.45372949500005075e+03, 1.42339304105829115e+03, 9.92450205677628144e-14] },
    // case 283: alt 910.720 km, lat +33.848, lon +124.012, doy 81, ap 295.845
    Case { doy: 81, sec: 7.15851851851851679e+04, alt: 9.10720000000000027e+02,
        g_lat: 3.38483965014577279e+01, g_long: 1.24012021036814474e+02, lst: 2.81522417316723974e+01,
        f107a: 2.75077378243058774e+02, f107: 2.43892733564013810e+02, ap: 2.95844875346260380e+02,
        ap_array: [3.70510396975425351e+01, 4.20510396975425351e+01, 4.70510396975425351e+01, 5.20510396975425351e+01, 5.70510396975425351e+01, 6.20510396975425351e+01, 6.70510396975425351e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [1.25023934200747614e+06, 2.02076821860906598e+06, 5.63001149053752852e+03, 1.83755829341564092e+00, 1.50025844597459542e-02, 6.35953114376333708e-17, 6.22290241336994077e+04, 5.40893545196491687e+04, 3.33018728314794180e+03, 1.43632658363466430e+03, 1.43632591512783233e+03, 6.36837612118737802e-17] },
    // case 284: alt 6.612 km, lat +59.563, lon +156.739, doy 264, ap 12.465
    Case { doy: 264, sec: 2.35851851851851861e+04, alt: 6.61199999999999921e+00,
        g_lat: 5.95626822157434219e+01, g_long: 1.56739293764087165e+02, lst: 6.55144032921810648e+00,
        f107a: 2.93154301319981812e+02, f107: 2.57716262975778534e+02, ap: 1.24653739612188357e+01,
        ap_array: [4.13988657844990513e+01, 4.63988657844990513e+01, 5.13988657844990513e+01, 5.63988657844990513e+01, 6.13988657844990513e+01, 6.63988657844990513e+01, 7.13988657844990513e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [6.77588749744861094e+13, 0.00000000000000000e+00, 1.00971095462910259e+19, 2.70875157798349363e+18, 1.20776575837960112e+17, 6.21222550088660545e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.50839743793697636e+02, 6.21222550088660545e-04] },
    // case 285: alt 86.332 km, lat +85.277, lon -167.558, doy 172, ap 28.255
    Case { doy: 172, sec: 5.23851851851851825e+04, alt: 8.63319999999999936e+01,
        g_lat: 8.52769679300291443e+01, g_long: -1.67558226897069858e+02, lst: 3.38089186941344977e+00,
        f107a: 7.76217569412835644e+01, f107: 2.71539792387543230e+02, ap: 2.82548476454293578e+01,
        ap_array: [4.57466918714555746e+01, 5.07466918714555746e+01, 5.57466918714555746e+01, 6.07466918714555746e+01, 6.57466918714555675e+01, 7.07466918714555675e+01, 7.57466918714555675e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [1.04045007274933207e+09, 7.55680689547169647e+10, 1.62731863949622281e+14, 4.24083899216264062e+13, 1.91098633266050708e+12, 9.94541445162586946e-09, 1.49241895441608369e+08, 2.53033963358511857e+05, 1.38646608315327166e-51, 1.02731846489999998e+03, 1.31566329990541021e+02, 9.94541445162586946e-09] },
    // case 286: alt 257.536 km, lat -65.335, lon -134.831, doy 355, ap 44.044
    Case { doy: 355, sec: 8.11851851851851825e+04, alt: 2.57536000000000001e+02,
        g_lat: -6.53352769679300280e+01, g_long: -1.34830954169797138e+02, lst: 1.35627100512316314e+01,
        f107a: 9.56986800182066446e+01, f107: 2.85363321799307982e+02, ap: 4.40443213296398852e+01,
        ap_array: [5.00945179584120979e+01, 5.50945179584120979e+01, 6.00945179584120979e+01, 6.50945179584120979e+01, 7.00945179584120979e+01, 7.50945179584120979e+01, 8.00945179584120979e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.62088587664522976e+06, 1.35280907574282551e+09, 6.97439186674043894e+08, 2.53196526295385398e+07, 3.43300984718134161e+05, 7.12931555692216181e-14, 7.04926357881303265e+04, 6.74233342435418516e+07, 1.58279820961851753e-01, 1.15168007337834979e+03, 1.09320923791773998e+03, 7.12931555734255273e-14] },
    // case 287: alt 814.720 km, lat -39.621, lon -102.104, doy 7, ap 59.834
    Case { doy: 7, sec: 7.58518518518518431e+03, alt: 8.14720000000000027e+02,
        g_lat: -3.96209912536443198e+01, g_long: -1.02103681442524405e+02, lst: -4.69991621139463156e+00,
        f107a: 1.13775603095129725e+02, f107: 2.99186851211072621e+02, ap: 5.98337950138504127e+01,
        ap_array: [5.44423440453686212e+01, 5.94423440453686212e+01, 6.44423440453686283e+01, 6.94423440453686283e+01, 7.44423440453686283e+01, 7.94423440453686283e+01, 8.44423440453686283e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.08489711457197147e+05, 1.24561065017330553e+06, 4.08347070192326009e+03, 3.15191129166361392e+01, 1.58312233402372708e-02, 3.80986006557815173e-17, 3.63927493967506962e+04, 1.16821130568303328e+05, 2.94487115653264191e+04, 1.30238304422853366e+03, 1.30234441923971144e+03, 3.88807584349565847e-17] },
    // case 288: alt 64.612 km, lat -13.907, lon -69.376, doy 189, ap 75.623
    Case { doy: 189, sec: 3.63851851851851825e+04, alt: 6.46119999999999948e+01,
        g_lat: -1.39067055393585974e+01, g_long: -6.93764087152516993e+01, lst: 1.01069958847736618e+01,
        f107a: 1.31852526172052819e+02, f107: 6.50478322817016021e+01, ap: 7.56232686980609401e+01,
        ap_array: [5.87901701323251373e+01, 6.37901701323251373e+01, 6.87901701323251302e+01, 7.37901701323251302e+01, 7.87901701323251302e+01, 8.37901701323251302e+01, 8.87901701323251302e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.88709349586213188e+10, 0.00000000000000000e+00, 2.80625624203301650e+15, 7.49569398424193125e+14, 3.35542401366292500e+13, 1.72480043422068064e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.29684039913239587e+02, 1.72480043422068064e-07] },
    // case 289: alt 78.732 km, lat +11.808, lon -36.649, doy 98, ap 91.413
    Case { doy: 98, sec: 6.51851851851851825e+04, alt: 7.87319999999999993e+01,
        g_lat: 1.18075801749271108e+01, g_long: -3.66491359879789513e+01, lst: 1.56637201522417300e+01,
        f107a: 1.49929449248975885e+02, f107: 7.88713616934663122e+01, ap: 9.14127423822714604e+01,
        ap_array: [6.31379962192816677e+01, 6.81379962192816606e+01, 7.31379962192816606e+01, 7.81379962192816606e+01, 8.31379962192816606e+01, 8.81379962192816606e+01, 9.31379962192816606e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.55557015966956091e+09, 1.63416052290478492e+09, 3.71038416520059188e+14, 9.42417533911784844e+13, 4.41216409507985449e+12, 2.25450156161604053e-08, 3.56179206621389603e+06, 8.83573457333720580e+04, 4.75485403302108333e-57, 1.02731846489999998e+03, 1.91130770692629227e+02, 2.25450156161604053e-08] },
    // case 290: alt 212.736 km, lat +37.522, lon -3.922, doy 281, ap 107.202
    Case { doy: 281, sec: 1.71851851851851861e+04, alt: 2.12735999999999990e+02,
        g_lat: 3.75218658892128332e+01, g_long: -3.92186326070623181e+00, lst: 4.51220500072658126e+00,
        f107a: 1.68006372325898951e+02, f107: 9.26948911052310223e+01, ap: 1.07202216066481981e+02,
        ap_array: [6.74858223062381910e+01, 7.24858223062381910e+01, 7.74858223062381910e+01, 8.24858223062381910e+01, 8.74858223062381910e+01, 9.24858223062381910e+01, 9.74858223062381910e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.57829923561439961e+07, 4.52528840099948788e+09, 2.84987799915840244e+09, 1.09604305254777402e+08, 2.47606750919986842e+06, 2.59244751457028242e-13, 1.53632085981105105e+05, 2.14766114176926315e+07, 1.72580058052186149e-06, 9.22411654447584169e+02, 8.91681097705944353e+02, 2.59244751457028293e-13] },
    // case 291: alt 718.720 km, lat +63.236, lon +28.805, doy 53, ap 122.992
    Case { doy: 53, sec: 4.59851851851851752e+04, alt: 7.18720000000000027e+02,
        g_lat: 6.32361516034985414e+01, g_long: 2.88054094665664593e+01, lst: 1.46940231825447576e+01,
        f107a: 1.86083295402822046e+02, f107: 1.06518420516995718e+02, ap: 1.22991689750692515e+02,
        ap_array: [7.18336483931947072e+01, 7.68336483931947072e+01, 8.18336483931947072e+01, 8.68336483931947072e+01, 9.18336483931947072e+01, 9.68336483931947072e+01, 1.01833648393194707e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.29328060623022844e+06, 2.19286793087864295e+06, 6.21092503067671259e+03, 4.75162742381949101e+01, 2.50538456921242970e-02, 6.94638384803647273e-17, 2.82968621264040703e+04, 9.87823019036439655e+04, 1.53250806638644281e+04, 1.10856061772474573e+03, 1.10856037097520016e+03, 6.98708726227969778e-17] },
    // case 292: alt 53.012 km, lat +88.950, lon +61.533, doy 235, ap 138.781
    Case { doy: 235, sec: 7.47851851851851679e+04, alt: 5.30120000000000076e+01,
        g_lat: 8.89504373177842638e+01, g_long: 6.15326821938392072e+01, lst: 2.07736625514403244e+01,
        f107a: 2.04160218479745140e+02, f107: 1.20341949928760442e+02, ap: 1.38781163434903050e+02,
        ap_array: [7.61814744801512234e+01, 8.11814744801512234e+01, 8.61814744801512234e+01, 9.11814744801512234e+01, 9.61814744801512234e+01, 1.01181474480151223e+02, 1.06181474480151223e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.70003476361495667e+10, 0.00000000000000000e+00, 1.29643834992593300e+16, 3.47795516134630050e+15, 1.55073473226390469e+14, 7.97630979520829793e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.69753516584404508e+02, 7.97630979520829793e-07] },
    // case 293: alt 116.732 km, lat -86.851, lon +94.260, doy 144, ap 154.571
    Case { doy: 144, sec: 2.67851851851851861e+04, alt: 1.16731999999999999e+02,
        g_lat: -8.68513119533527629e+01, g_long: 9.42599549211119552e+01, lst: 1.37243262128477923e+01,
        f107a: 2.22237141556668206e+02, f107: 1.34165479340525138e+02, ap: 1.54570637119113570e+02,
        ap_array: [8.05293005671077395e+01, 8.55293005671077395e+01, 9.05293005671077395e+01, 9.55293005671077395e+01, 1.00529300567107740e+02, 1.05529300567107740e+02, 1.10529300567107740e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [7.66779360452671535e+06, 1.77028560082212448e+10, 2.77958979198491699e+11, 4.96792351060415115e+10, 2.25898091124978781e+09, 1.61787375794358772e-11, 1.89273760154597904e+06, 2.15717631685193744e+05, 8.75652825400898687e-35, 1.02731846489999998e+03, 5.68422219764693068e+02, 1.61787375794358772e-11] },
    // case 294: alt 167.936 km, lat -61.137, lon +126.987, doy 326, ap 170.360
    Case { doy: 326, sec: 5.55851851851851752e+04, alt: 1.67936000000000007e+02,
        g_lat: -6.11370262390670547e+01, g_long: 1.26987227648384703e+02, lst: 2.39061443946659722e+01,
        f107a: 2.40314064633591272e+02, f107: 1.47989008752289834e+02, ap: 1.70360110803324119e+02,
        ap_array: [8.48771266540642699e+01, 8.98771266540642699e+01, 9.48771266540642699e+01, 9.98771266540642699e+01, 1.04877126654064270e+02, 1.09877126654064270e+02, 1.14877126654064270e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [1.26920901307521970e+06, 2.57129644454979849e+09, 1.69848668585416069e+10, 8.81380121467900395e+08, 1.56645991684197694e+08, 9.15088992044928843e-13, 4.80642968839353343e+04, 4.73466686260823254e+06, 1.46056418205554615e-14, 1.69526249863060389e+03, 1.05597671549160941e+03, 9.15088992044928843e-13] },
    // case 295: alt 622.720 km, lat -35.423, lon +159.715, doy 30, ap 186.150
    Case { doy: 30, sec: 8.43851851851851679e+04, alt: 6.22720000000000027e+02,
        g_lat: -3.54227405247813465e+01, g_long: 1.59714500375657451e+02, lst: 3.40879625764841592e+01,
        f107a: 2.58390987710514366e+02, f107: 1.61812538164054558e+02, ap: 1.86149584487534639e+02,
        ap_array: [8.92249527410207861e+01, 9.42249527410207861e+01, 9.92249527410207861e+01, 1.04224952741020786e+02, 1.09224952741020786e+02, 1.14224952741020786e+02, 1.19224952741020786e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [1.18179376461367728e+06, 3.80118758572628871e+07, 1.22034423557286919e+06, 7.35506114007080305e+03, 1.72538568241702308e+02, 1.10791151917721454e-15, 1.15814456061954934e+04, 1.43399325014854060e+06, 3.97591028967933380e+05, 1.46999651686921493e+03, 1.46990773406682433e+03, 1.11847153690660278e-15] },
    // case 296: alt 41.412 km, lat -9.708, lon -164.583, doy 212, ap 201.939
    Case { doy: 212, sec: 2.25185185185185173e+03, alt: 4.14120000000000061e+01,
        g_lat: -9.70845481049563830e+00, g_long: -1.64583020285499629e+02, lst: 6.25514403292180998e-01,
        f107a: 2.76467910787437461e+02, f107: 1.75636067575819254e+02, ap: 2.01939058171745160e+02,
        ap_array: [9.35727788279773165e+01, 9.85727788279773165e+01, 1.03572778827977316e+02, 1.08572778827977316e+02, 1.13572778827977316e+02, 1.18572778827977316e+02, 1.23572778827977316e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [3.59126360602257080e+11, 0.00000000000000000e+00, 5.35153248239022160e+16, 1.43565561905082480e+16, 6.40123557881106375e+14, 3.29251915149711298e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.56624103364663767e+02, 3.29251915149711298e-06] },
    // case 297: alt 109.132 km, lat +16.006, lon -131.856, doy 121, ap 217.729
    Case { doy: 121, sec: 3.10518518518518504e+04, alt: 1.09132000000000005e+02,
        g_lat: 1.60058309037900841e+01, g_long: -1.31855747558226881e+02, lst: -1.64868767256278304e-01,
        f107a: 2.94544833864360498e+02, f107: 1.89459596987583950e+02, ap: 2.17728531855955652e+02,
        ap_array: [9.79206049149338469e+01, 1.02920604914933847e+02, 1.07920604914933847e+02, 1.12920604914933847e+02, 1.17920604914933847e+02, 1.22920604914933847e+02, 1.27920604914933847e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [5.98335289637645558e+07, 4.11809305923591431e+11, 2.20366148302955762e+12, 2.67719847236153503e+11, 1.90817818622822189e+10, 1.28852584071407444e-10, 4.59591419279875886e+06, 1.27582500574914599e+06, 5.37813696842337968e-36, 1.02731846489999998e+03, 2.52729844976670620e+02, 1.28852584071407444e-10] },
    // case 298: alt 391.936 km, lat +41.720, lon -99.128, doy 303, ap 233.518
    Case { doy: 303, sec: 5.98518518518518467e+04, alt: 3.91936000000000035e+02,
        g_lat: 4.17201166180757923e+01, g_long: -9.91284748309541612e+01, lst: 1.00169494145619016e+01,
        f107a: 7.90122894856622651e+01, f107: 2.03283126399348674e+02, ap: 2.33518005540166200e+02,
        ap_array: [2.45746691871455614e+00, 7.45746691871455614e+00, 1.24574669187145552e+01, 1.74574669187145552e+01, 2.24574669187145552e+01, 2.74574669187145552e+01, 3.24574669187145588e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [4.34339672165997699e+06, 2.63497905240411490e+08, 2.67823204661158137e+07, 2.71574646934071695e+06, 2.63697795922787691e+03, 8.50716768049714031e-15, 8.79649711241965124e+04, 3.88981661682387395e+06, 2.20174389134732064e+03, 1.22297333283022158e+03, 1.21211746152117462e+03, 8.50722615881489494e-15] },
    // case 299: alt 411.520 km, lat +67.434, lon -66.401, doy 75, ap 249.307
    Case { doy: 75, sec: 1.18518518518518522e+04, alt: 4.11519999999999982e+02,
        g_lat: 6.74344023323615147e+01, g_long: -6.64012021036814701e+01, lst: -1.13456573695324980e+00,
        f107a: 9.70892125625853453e+01, f107: 2.17106655811113370e+02, ap: 2.49307479224376749e+02,
        ap_array: [6.80529300567107764e+00, 1.18052930056710785e+01, 1.68052930056710785e+01, 2.18052930056710785e+01, 2.68052930056710785e+01, 3.18052930056710785e+01, 3.68052930056710750e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [5.91740759630148532e+05, 7.84947608941476345e+07, 8.38837660949182212e+07, 1.03663321930187587e+07, 7.07126632537987171e+04, 6.54501034197693056e-15, 3.09653496054787356e+04, 8.33303132130921294e+04, 2.47978130907118384e+03, 1.80321759175279612e+03, 1.74258076358819335e+03, 6.54507620496849934e-15] },
    // case 300: alt 15.892 km, lat -83.178, lon -33.674, doy 258, ap 265.097
    Case { doy: 258, sec: 4.06518518518518540e+04, alt: 1.58920000000000030e+01,
        g_lat: -8.31778425655976719e+01, g_long: -3.36739293764087222e+01, lst: 1.12921810699588470e+01,
        f107a: 1.15166135639508425e+02, f107: 2.30930185222878066e+02, ap: 2.65096952908587241e+02,
        ap_array: [1.11531190926275983e+01, 1.61531190926275983e+01, 2.11531190926275983e+01, 2.61531190926275983e+01, 3.11531190926275983e+01, 3.61531190926275983e+01, 4.11531190926275983e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [1.70277706981825859e+13, 0.00000000000000000e+00, 2.53739847560061901e+18, 6.80707889049378048e+17, 3.03510918658968040e+16, 1.56112909776512750e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 1.91829796560653335e+02, 1.56112909776512750e-04] },
    // case 301: alt 92.412 km, lat -57.464, lon -0.947, doy 167, ap 280.886
    Case { doy: 167, sec: 6.94518518518518395e+04, alt: 9.24120000000000061e+01,
        g_lat: -5.74635568513119566e+01, g_long: -9.46656649136002670e-01, lst: 1.92290706266831108e+01,
        f107a: 1.33243058716431506e+02, f107: 2.44753714634642762e+02, ap: 2.80886426592797761e+02,
        ap_array: [1.55009451795841215e+01, 2.05009451795841215e+01, 2.55009451795841215e+01, 3.05009451795841215e+01, 3.55009451795841215e+01, 4.05009451795841215e+01, 4.55009451795841215e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.20184356192192316e+08, 3.68577337778970825e+11, 2.48624273153759062e+13, 6.10610621062802441e+12, 2.83379646004616577e+11, 1.50856932536777596e-09, 3.18410412317330278e+07, 1.88217455089530849e+05, 2.99191756590214900e-48, 1.02731846489999998e+03, 1.81751445070084941e+02, 1.50856932536777596e-09] },
    // case 302: alt 293.376 km, lat -31.749, lon +31.781, doy 349, ap 296.676
    Case { doy: 349, sec: 2.14518518518518504e+04, alt: 2.93376000000000033e+02,
        g_lat: -3.17492711370262413e+01, g_long: 3.17806160781367169e+01, lst: 8.07755547516796213e+00,
        f107a: 1.51319981793354600e+02, f107: 2.58577244046407486e+02, ap: 2.96675900277008282e+02,
        ap_array: [1.98487712665406448e+01, 2.48487712665406448e+01, 2.98487712665406448e+01, 3.48487712665406448e+01, 3.98487712665406448e+01, 4.48487712665406448e+01, 4.98487712665406448e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.63375994746290194e+06, 8.72169356741377831e+08, 2.66999373343883961e+08, 6.70070989033963904e+06, 6.41425921280785551e+04, 3.63389488026179323e-14, 5.60751933837072575e+04, 1.63329512125169989e+07, 1.01653313271452799e+01, 1.21521282650057469e+03, 1.17131701493193123e+03, 3.63389490726091302e-14] },
    // case 303: alt 891.520 km, lat -6.035, lon +64.508, doy 18, ap 13.296
    Case { doy: 18, sec: 5.02518518518518467e+04, alt: 8.91519999999999982e+02,
        g_lat: -6.03498542274053307e+00, g_long: 6.45078888054094364e+01, lst: 1.82593736569861420e+01,
        f107a: 1.69396904870277666e+02, f107: 2.72400773458172182e+02, ap: 1.32963988919667582e+01,
        ap_array: [2.41965973534971646e+01, 2.91965973534971646e+01, 3.41965973534971681e+01, 3.91965973534971681e+01, 4.41965973534971681e+01, 4.91965973534971681e+01, 5.41965973534971681e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.08732793205228227e+05, 8.46961723540855804e+05, 1.76866525376064669e+03, 7.28243696259533380e+00, 4.13981371465732984e-03, 2.72025791388282348e-17, 2.61564391022374111e+04, 8.03474891076072963e+04, 1.93185327936873318e+03, 1.38353348347214524e+03, 1.38351500340586153e+03, 2.72538891619282672e-17] },
    // case 304: alt 4.292 km, lat +19.679, lon +97.235, doy 201, ap 29.086
    Case { doy: 201, sec: 7.90518518518518395e+04, alt: 4.29199999999999982e+00,
        g_lat: 1.96793002915451893e+01, g_long: 9.72351615326821843e+01, lst: 2.19588477366255113e+01,
        f107a: 1.87473827947200732e+02, f107: 2.86224302869936878e+02, ap: 2.90858725761772838e+01,
        ap_array: [2.85444234404536878e+01, 3.35444234404536843e+01, 3.85444234404536843e+01, 4.35444234404536843e+01, 4.85444234404536843e+01, 5.35444234404536843e+01, 5.85444234404536843e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.48759177571165312e+13, 0.00000000000000000e+00, 1.26478109289489859e+19, 3.39302823790914458e+18, 1.51286790426623296e+17, 7.78153918435704403e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.77008583490974104e+02, 7.78153918435704403e-04] },
    // case 305: alt 84.812 km, lat +45.394, lon +129.962, doy 110, ap 44.875
    Case { doy: 110, sec: 5.45185185185185128e+03, alt: 8.48119999999999976e+01,
        g_lat: 4.53935860058308833e+01, g_long: 1.29962434259954932e+02, lst: 1.01785655761780660e+01,
        f107a: 2.05550751024123826e+02, f107: 6.58609810706289380e+01, ap: 4.48753462603878077e+01,
        ap_array: [3.28922495274102147e+01, 3.78922495274102147e+01, 4.28922495274102147e+01, 4.78922495274102147e+01, 5.28922495274102147e+01, 5.78922495274102147e+01, 6.28922495274102147e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.61667599638726711e+08, 6.70930819530279922e+10, 1.34099441046497016e+14, 3.17960926160197188e+13, 1.57024492708784546e+12, 8.02800329890723176e-09, 1.17682562328990966e+08, 1.31749708996676956e+05, 9.59481602790003727e-53, 1.02731846489999998e+03, 1.85471363890565385e+02, 8.02800329890723176e-09] },
    // case 306: alt 248.576 km, lat +71.108, lon +162.690, doy 292, ap 60.665
    Case { doy: 292, sec: 3.42518518518518540e+04, alt: 2.48575999999999993e+02,
        g_lat: 7.11078717201166342e+01, g_long: 1.62689706987227680e+02, lst: 2.03603837579962494e+01,
        f107a: 2.23627674101046892e+02, f107: 7.96845104823936481e+01, ap: 6.06648199445983352e+01,
        ap_array: [3.72400756143667309e+01, 4.22400756143667309e+01, 4.72400756143667309e+01, 5.22400756143667309e+01, 5.72400756143667309e+01, 6.22400756143667309e+01, 6.72400756143667309e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.77232602797081620e+07, 2.97771923490921021e+09, 5.22886368678418458e+08, 9.24689556186676770e+07, 1.88094027690160176e+05, 1.08938100167482127e-13, 1.71781626386672433e+05, 2.16743072487653829e+07, 3.20797328490647349e-02, 8.13536286595780894e+02, 8.11483201317389671e+02, 1.08938100168334172e-13] },
    // case 307: alt 795.520 km, lat -79.504, lon -161.608, doy 64, ap 76.454
    Case { doy: 64, sec: 6.30518518518518467e+04, alt: 7.95520000000000095e+02,
        g_lat: -7.95043731778425666e+01, g_long: -1.61607813673929371e+02, lst: 6.74054904725244342e+00,
        f107a: 2.41704597177969958e+02, f107: 9.35080398941583582e+01, ap: 7.64542936288088697e+01,
        ap_array: [4.15879017013232470e+01, 4.65879017013232470e+01, 5.15879017013232470e+01, 5.65879017013232470e+01, 6.15879017013232470e+01, 6.65879017013232470e+01, 7.15879017013232470e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [7.45960964434270863e+05, 4.52151893510087917e+05, 2.86604435678110690e+02, 4.82252319778764815e-01, 2.55179354604437791e-04, 1.72821543969185474e-17, 4.49884301130131134e+04, 9.97381665671247720e+03, 1.80945147431089776e+05, 1.02731846489999998e+03, 1.02731846276066358e+03, 2.20880575126882918e-17] },
    // case 308: alt 62.292 km, lat -53.790, lon -128.881, doy 246, ap 92.244
    Case { doy: 246, sec: 1.50518518518518522e+04, alt: 6.22920000000000087e+01,
        g_lat: -5.37900874635568513e+01, g_long: -1.28880540946656652e+02, lst: 4.18106995884773625e+00,
        f107a: 2.59781520254893053e+02, f107: 1.07331569305923068e+02, ap: 9.22437673130193900e+01,
        ap_array: [4.59357277882797703e+01, 5.09357277882797703e+01, 5.59357277882797703e+01, 6.09357277882797703e+01, 6.59357277882797632e+01, 7.09357277882797632e+01, 7.59357277882797632e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [1.89192608727326088e+10, 0.00000000000000000e+00, 2.81925946436933900e+15, 7.56322736506362625e+14, 3.37225720830538477e+13, 1.73454348077311825e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.38667046550094341e+02, 1.73454348077311825e-07] },
    // case 309: alt 77.212 km, lat -28.076, lon -96.153, doy 155, ap 108.033
    Case { doy: 155, sec: 4.38518518518518540e+04, alt: 7.72120000000000033e+01,
        g_lat: -2.80758017492711431e+01, g_long: -9.61532682193839179e+01, lst: 5.77085207755547724e+00,
        f107a: 2.77858443331816147e+02, f107: 1.21155098717687764e+02, ap: 1.08033240997229925e+02,
        ap_array: [5.02835538752363007e+01, 5.52835538752363007e+01, 6.02835538752363007e+01, 6.52835538752362936e+01, 7.02835538752362936e+01, 7.52835538752362936e+01, 8.02835538752362936e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [2.95158342342238903e+09, 6.84489865148456335e+08, 4.25955856581114750e+14, 1.03700965667196328e+14, 5.06444870338019727e+12, 2.56433406841846056e-08, 2.83611256695266493e+04, 6.61208315170566784e+04, 1.90848207755270209e-59, 1.02731846489999998e+03, 2.05328384067217115e+02, 2.56433406841846056e-08] },
    // case 310: alt 203.776 km, lat -2.362, lon -63.426, doy 338, ap 123.823
    Case { doy: 338, sec: 7.26518518518518395e+04, alt: 2.03776000000000010e+02,
        g_lat: -2.36151603498542784e+00, g_long: -6.34259954921112126e+01, lst: 1.59526702593736509e+01,
        f107a: 2.95935366408739242e+02, f107: 1.34978628129452488e+02, ap: 1.23822714681440445e+02,
        ap_array: [5.46313799621928169e+01, 5.96313799621928169e+01, 6.46313799621928240e+01, 6.96313799621928240e+01, 7.46313799621928240e+01, 7.96313799621928240e+01, 8.46313799621928240e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [6.31489162268045824e+06, 7.52537529134942722e+09, 3.90488083488857269e+09, 3.94188484857060611e+07, 2.93301470196444308e+06, 3.85690892589196965e-13, 6.15254085987268991e+04, 8.55141663085729033e+07, 7.56343877962639481e-09, 1.25432141601941453e+03, 1.07951352943153779e+03, 3.85690892589196965e-13] },
    // case 311: alt 699.520 km, lat +23.353, lon -30.699, doy 41, ap 139.612
    Case { doy: 41, sec: 2.46518518518518504e+04, alt: 6.99519999999999982e+02,
        g_lat: 2.33527696793002946e+01, g_long: -3.06987227648384646e+01, lst: 4.80115510785850574e+00,
        f107a: 8.04028220300409657e+01, f107: 1.48802157541217184e+02, ap: 1.39612188365650979e+02,
        ap_array: [5.89792060491493331e+01, 6.39792060491493331e+01, 6.89792060491493260e+01, 7.39792060491493260e+01, 7.89792060491493260e+01, 8.39792060491493260e+01, 8.89792060491493260e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [9.45311629084638087e+05, 7.64073646459042502e+04, 1.28834134721704334e+01, 6.81687192593065094e-02, 2.50480711974991450e-06, 8.63988734866651138e-18, 1.91967604036484030e+05, 6.18324505243556018e+02, 2.68350155646836174e+03, 7.62557888727330919e+02, 7.62557872896970366e+02, 8.71116115000631159e-18] },
    // case 312: alt 50.692 km, lat +49.067, lon +2.029, doy 224, ap 155.402
    Case { doy: 224, sec: 5.34518518518518467e+04, alt: 5.06920000000000002e+01,
        g_lat: 4.90670553935860028e+01, g_long: 2.02854996243428332e+00, lst: 1.48477366255144005e+01,
        f107a: 9.84797451069640317e+01, f107: 1.62625686952981880e+02, ap: 1.55401662049861471e+02,
        ap_array: [6.33270321361058635e+01, 6.83270321361058564e+01, 7.33270321361058564e+01, 7.83270321361058564e+01, 8.33270321361058564e+01, 8.83270321361058564e+01, 9.33270321361058564e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [1.35979510401019852e+11, 0.00000000000000000e+00, 2.02630284680361960e+16, 5.43596264712007000e+15, 2.42376215020414969e+14, 1.24667858259064548e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.64994639139409685e+02, 1.24667858259064548e-06] },
    // case 313: alt 115.212 km, lat +74.781, lon +34.756, doy 132, ap 171.191
    Case { doy: 132, sec: 8.22518518518518395e+04, alt: 1.15212000000000003e+02,
        g_lat: 7.47813411078717252e+01, g_long: 3.47558226897069744e+01, lst: 2.51647914714948655e+01,
        f107a: 1.16556668183887126e+02, f107: 1.76449216364746576e+02, ap: 1.71191135734072020e+02,
        ap_array: [6.76748582230623867e+01, 7.26748582230623867e+01, 7.76748582230623867e+01, 8.26748582230623867e+01, 8.76748582230623867e+01, 9.26748582230623867e+01, 9.76748582230623867e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [1.23994297417684868e+07, 6.31475546947464981e+10, 6.71024069198126465e+11, 9.12313369859101715e+10, 4.41655884188225365e+09, 3.80059736380203571e-11, 4.63268685730704013e+06, 7.61563372827266809e+05, 2.37745185557238311e-34, 1.02731846489999998e+03, 3.15809433936087942e+02, 3.80059736380203571e-11] },
    // case 314: alt 158.976 km, lat -75.831, lon +67.483, doy 315, ap 186.981
    Case { doy: 315, sec: 8.65185185185185037e+03, alt: 1.58975999999999999e+02,
        g_lat: -7.58309037900874614e+01, g_long: 6.74830954169796939e+01, lst: 6.90216520886860518e+00,
        f107a: 1.34633591260810192e+02, f107: 1.90272745776511329e+02, ap: 1.86980609418282569e+02,
        ap_array: [7.20226843100189029e+01, 7.70226843100189029e+01, 8.20226843100189029e+01, 8.70226843100189029e+01, 9.20226843100189029e+01, 9.70226843100189029e+01, 1.02022684310018903e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [2.91606653640714008e+06, 7.93045311946185303e+09, 2.42420968192802315e+10, 1.53067933012790060e+09, 5.00912034507985115e+07, 1.42259440411290600e-12, 1.24839646231619336e+05, 2.29602951253219098e+07, 1.37594601168362174e-17, 1.22241997334690427e+03, 8.17810394413161589e+02, 1.42259440411290600e-12] },
    // case 315: alt 603.520 km, lat -50.117, lon +100.210, doy 87, ap 202.770
    Case { doy: 87, sec: 3.74518518518518540e+04, alt: 6.03519999999999982e+02,
        g_lat: -5.01166180758017461e+01, g_long: 1.00210368144252470e+02, lst: 1.70839833906867895e+01,
        f107a: 1.52710514337733287e+02, f107: 2.04096275188276024e+02, ap: 2.02770083102493061e+02,
        ap_array: [7.63705103969754191e+01, 8.13705103969754191e+01, 8.63705103969754191e+01, 9.13705103969754191e+01, 9.63705103969754191e+01, 1.01370510396975419e+02, 1.06370510396975419e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.99104503238322563e+06, 2.84415298033410683e+07, 8.07265336780431797e+05, 1.40085272864555955e+04, 1.58371541037279506e+01, 8.36557232528527710e-16, 2.19887302977185755e+04, 1.27479704106827243e+06, 4.85311803471802414e+04, 1.42630099879121235e+03, 1.42460301439035675e+03, 8.37846220678548851e-16] },
    // case 316: alt 39.092 km, lat -24.402, lon +132.938, doy 269, ap 218.560
    Case { doy: 269, sec: 6.62518518518518395e+04, alt: 3.90919999999999987e+01,
        g_lat: -2.44023323615160450e+01, g_long: 1.32937640871525161e+02, lst: 1.84032921810699577e+01,
        f107a: 1.70787437414656353e+02, f107: 2.17919804600040720e+02, ap: 2.18559556786703581e+02,
        ap_array: [8.07183364839319353e+01, 8.57183364839319353e+01, 9.07183364839319353e+01, 9.57183364839319353e+01, 1.00718336483931935e+02, 1.05718336483931935e+02, 1.10718336483931935e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.11956319000358582e+11, 0.00000000000000000e+00, 7.62893279708225920e+16, 2.04661380147325760e+16, 9.12534796522994375e+14, 4.69368492530551282e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.53860712107758644e+02, 4.69368492530551282e-06] },
    // case 317: alt 107.612 km, lat +1.312, lon +165.665, doy 178, ap 234.349
    Case { doy: 178, sec: 1.82518518518518504e+04, alt: 1.07611999999999995e+02,
        g_lat: 1.31195335276967739e+00, g_long: 1.65664913598797966e+02, lst: 1.61142864209898207e+01,
        f107a: 1.88864360491579447e+02, f107: 2.31743334011805416e+02, ap: 2.34349030470914130e+02,
        ap_array: [8.50661625708884657e+01, 9.00661625708884657e+01, 9.50661625708884657e+01, 1.00066162570888466e+02, 1.05066162570888466e+02, 1.10066162570888466e+02, 1.15066162570888466e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.93108067359172329e+07, 2.35383945777868713e+11, 1.55436614630109521e+12, 2.38996713527059509e+11, 1.25005056663796902e+10, 9.20245609393195154e-11, 4.11617015984042361e+06, 7.74697943386003259e+05, 1.13231123870248605e-36, 1.02731846489999998e+03, 2.49933614162700451e+02, 9.20245609393195154e-11] },
    // case 318: alt 382.976 km, lat +27.026, lon -158.633, doy 361, ap 250.139
    Case { doy: 361, sec: 4.70518518518518467e+04, alt: 3.82975999999999999e+02,
        g_lat: 2.70262390670553856e+01, g_long: -1.58632607062359114e+02, lst: 2.49445171024601642e+00,
        f107a: 2.06941283568502541e+02, f107: 2.45566863423570112e+02, ap: 2.50138504155124650e+02,
        ap_array: [8.94139886578449961e+01, 9.44139886578449961e+01, 9.94139886578449961e+01, 1.04413988657844996e+02, 1.09413988657844996e+02, 1.14413988657844996e+02, 1.19413988657844996e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.09257121736326255e+06, 5.06306255464371204e+08, 3.73890392145775780e+07, 6.01146423523342586e+05, 4.73706552962145179e+03, 1.53517541309752879e-14, 1.67635946616596084e+04, 3.17416108127640188e+06, 4.79243277831903979e+04, 1.17819366337685960e+03, 1.17660448429077155e+03, 1.53530270011212110e-14] },
    // case 319: alt 507.520 km, lat +52.741, lon -125.905, doy 4, ap 265.928
    Case { doy: 4, sec: 7.58518518518518395e+04, alt: 5.07519999999999982e+02,
        g_lat: 5.27405247813410938e+01, g_long: -1.25905334335086394e+02, lst: 1.26762698920641963e+01,
        f107a: 2.25018206645425607e+02, f107: 2.59390392835334865e+02, ap: 2.65927977839335199e+02,
        ap_array: [9.37618147448015122e+01, 9.87618147448015122e+01, 1.03761814744801512e+02, 1.08761814744801512e+02, 1.13761814744801512e+02, 1.18761814744801512e+02, 1.23761814744801512e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.64684152411950519e+06, 1.39143151916774094e+08, 1.02502403689710163e+07, 2.55392892527009686e+06, 6.85861945529681520e+02, 4.63252547548907442e-15, 1.14662635613495768e+04, 1.32163459883212112e+07, 3.85980388367105843e+04, 1.44962317115627889e+03, 1.44926990774601882e+03, 4.63355063940057734e-15] },
    // case 320: alt 27.492 km, lat +78.455, lon -93.178, doy 187, ap 281.717
    Case { doy: 187, sec: 2.78518518518518504e+04, alt: 2.74919999999999973e+01,
        g_lat: 7.84548104956268162e+01, g_long: -9.31780616078136603e+01, lst: 7.73662551440329160e+00,
        f107a: 2.43095129722348673e+02, f107: 2.73213922247099504e+02, ap: 2.81717451523545662e+02,
        ap_array: [9.81096408317580426e+01, 1.03109640831758043e+02, 1.08109640831758043e+02, 1.13109640831758043e+02, 1.18109640831758043e+02, 1.23109640831758043e+02, 1.28109640831758043e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [3.11388411485161523e+12, 0.00000000000000000e+00, 4.64016396877177920e+17, 1.24481678790240528e+17, 5.55033213124636400e+15, 2.85485116338940904e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.37104901911791217e+02, 2.85485116338940904e-05] },
    // case 321: alt 100.012 km, lat -72.157, lon -60.451, doy 95, ap 297.507
    Case { doy: 95, sec: 5.66518518518518467e+04, alt: 1.00012000000000000e+02,
        g_lat: -7.21574344023323704e+01, g_long: -6.04507888805409550e+01, lst: 1.17065729223672292e+01,
        f107a: 2.61172052799271796e+02, f107: 2.87037451658864256e+02, ap: 2.97506925207756240e+02,
        ap_array: [2.64650283553875232e+00, 7.64650283553875276e+00, 1.26465028355387528e+01, 1.76465028355387510e+01, 2.26465028355387510e+01, 2.76465028355387510e+01, 3.26465028355387545e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [5.11175892023016736e+07, 1.08644057866342163e+11, 5.05196688585627441e+12, 1.04770588632174341e+12, 5.54170335646521912e+10, 2.97035194283003917e-10, 7.17477028743096720e+06, 3.53289989132818067e+05, 1.45317772109598605e-42, 1.02731846489999998e+03, 1.45954548139519403e+02, 2.97035194283003917e-10] },
    // case 322: alt 338.176 km, lat -46.443, lon -27.724, doy 278, ap 14.127
    Case { doy: 278, sec: 8.54518518518518395e+04, alt: 3.38176000000000045e+02,
        g_lat: -4.64431486880466480e+01, g_long: -2.77235161532682071e+01, lst: 2.18883911041854091e+01,
        f107a: 2.79248975876194834e+02, f107: 6.66741298595562739e+01, ap: 1.41274238227146824e+01,
        ap_array: [6.99432892249527427e+00, 1.19943289224952743e+01, 1.69943289224952743e+01, 2.19943289224952743e+01, 2.69943289224952743e+01, 3.19943289224952743e+01, 3.69943289224952707e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [5.24750451100421231e+06, 5.78503407944180965e+08, 2.30588122277517132e+07, 1.35365759865244676e+05, 2.07593535785150516e+03, 1.65672524012779877e-14, 1.89389813161608268e+05, 3.78407941565928468e+06, 1.88858224912834771e+03, 7.92987786678081420e+02, 7.92981555341322519e+02, 1.65673025620225277e-14] },
    // case 323: alt 987.520 km, lat -20.729, lon +5.004, doy 50, ap 29.917
    Case { doy: 50, sec: 4.74074074074074019e+02, alt: 9.87520000000000095e+02,
        g_lat: -2.07288629737609398e+01, g_long: 5.00375657400454088e+00, lst: 4.65271014398656590e-01,
        f107a: 2.97325898953117871e+02, f107: 8.04976592713209840e+01, ap: 2.99168975069252063e+01,
        ap_array: [1.13421550094517958e+01, 1.63421550094517940e+01, 2.13421550094517940e+01, 2.63421550094517940e+01, 3.13421550094517940e+01, 3.63421550094517940e+01, 4.13421550094517940e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [1.77104377952109178e+05, 6.24060042308155516e+02, 6.98740794601687069e-04, 1.02899371931362432e-07, 1.71763077727322870e-12, 1.40560978041677467e-18, 1.27818624978451437e+05, 3.79830526633944032e+01, 8.70605172895211581e+03, 7.46353705973125443e+02, 7.46353705973119872e+02, 1.63684251433774290e-18] },
    // case 324: alt 1.972 km, lat +4.985, lon +37.731, doy 232, ap 45.706
    Case { doy: 232, sec: 2.92740740740740730e+04, alt: 1.97200000000000020e+00,
        g_lat: 4.98542274052479684e+00, g_long: 3.77310293012772320e+01, lst: 8.13168724279835331e+00,
        f107a: 8.17933545744196664e+01, f107: 9.43211886830856940e+01, ap: 4.57063711911357302e+01,
        ap_array: [1.56899810964083173e+01, 2.06899810964083173e+01, 2.56899810964083173e+01, 3.06899810964083173e+01, 3.56899810964083173e+01, 4.06899810964083173e+01, 4.56899810964083173e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [1.05946060887094766e+14, 0.00000000000000000e+00, 1.57875847728849920e+19, 4.23533536702185677e+18, 1.88843195260863840e+17, 9.71327846587061216e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.87121506717690579e+02, 9.71327846587061216e-04] },
    // case 325: alt 83.292 km, lat +30.700, lon +70.458, doy 141, ap 61.496
    Case { doy: 141, sec: 5.80740740740740657e+04, alt: 8.32920000000000016e+01,
        g_lat: 3.06997084548104908e+01, g_long: 7.04583020285499515e+01, lst: 2.08289073780350158e+01,
        f107a: 9.98702776513427466e+01, f107: 1.08144718094850390e+02, ap: 6.14958448753462577e+01,
        ap_array: [2.00378071833648406e+01, 2.50378071833648406e+01, 3.00378071833648406e+01, 3.50378071833648406e+01, 4.00378071833648406e+01, 4.50378071833648406e+01, 5.00378071833648406e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [1.01540176696646941e+09, 2.29146019112397881e+10, 1.44344203044557156e+14, 3.86817774025862969e+13, 1.69988391675327051e+12, 8.87738237826362595e-09, 9.45155172906905115e+07, 8.88362098028601031e+04, 2.23193689472232823e-54, 1.02731846489999998e+03, 1.75292105868633200e+02, 8.87738237826362595e-09] },
    // case 326: alt 239.616 km, lat +56.414, lon +103.186, doy 323, ap 77.285
    Case { doy: 323, sec: 1.00740740740740730e+04, alt: 2.39616000000000014e+02,
        g_lat: 5.64139941690961848e+01, g_long: 1.03185574755822699e+02, lst: 9.67739222651986708e+00,
        f107a: 1.17947200728265813e+02, f107: 1.21968247506615100e+02, ap: 7.72853185595567851e+01,
        ap_array: [2.43856332703213603e+01, 2.93856332703213603e+01, 3.43856332703213639e+01, 3.93856332703213639e+01, 4.43856332703213639e+01, 4.93856332703213639e+01, 5.43856332703213639e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [1.46416426572066415e+07, 1.83549451085224271e+09, 9.81681815954406619e+08, 6.76898023066708595e+07, 5.98961797273335047e+05, 9.85919855175052259e-14, 1.00736078854418287e+05, 2.06471347979673967e+07, 4.82748758350397744e-04, 1.03099938691820921e+03, 9.97637864322660789e+02, 9.85919855175180370e-14] },
    // case 327: alt 776.320 km, lat +82.128, lon +135.913, doy 27, ap 93.075
    Case { doy: 27, sec: 3.88740740740740730e+04, alt: 7.76320000000000050e+02,
        g_lat: 8.21282798833819072e+01, g_long: 1.35912847483095447e+02, lst: 1.98592104083380505e+01,
        f107a: 1.36024123805188879e+02, f107: 1.35791776918379810e+02, ap: 9.30747922437673054e+01,
        ap_array: [2.87334593572778800e+01, 3.37334593572778800e+01, 3.87334593572778800e+01, 4.37334593572778800e+01, 4.87334593572778800e+01, 5.37334593572778800e+01, 5.87334593572778800e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [8.12998946662295726e+05, 1.10426496408522199e+06, 9.48219182916534191e+03, 1.69147628027862737e+02, 9.61037662794519826e-02, 3.54239999333722645e-17, 2.98592852289651382e+04, 8.48189709296560068e+03, 2.76753125051498228e+03, 1.30815248054279664e+03, 1.30814168418868917e+03, 3.54975055633859418e-17] },
    // case 328: alt 59.972 km, lat -68.484, lon +168.640, doy 209, ap 108.864
    Case { doy: 209, sec: 6.76740740740740584e+04, alt: 5.99720000000000013e+01,
        g_lat: -6.84839650145772652e+01, g_long: 1.68640120210368195e+02, lst: 1.87983539094650176e+01,
        f107a: 1.54101046882111973e+02, f107: 1.49615306330144506e+02, ap: 1.08864265927977826e+02,
        ap_array: [3.30812854442344033e+01, 3.80812854442344033e+01, 4.30812854442344033e+01, 4.80812854442344033e+01, 5.30812854442344033e+01, 5.80812854442344033e+01, 6.30812854442344033e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.98856539608106689e+10, 0.00000000000000000e+00, 2.96326682692925200e+15, 7.94955591660310500e+14, 3.54451161503142500e+13, 1.82314370897790516e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.49343123087913881e+02, 1.82314370897790516e-07] },
    // case 329: alt 75.692 km, lat -42.770, lon -155.657, doy 118, ap 124.654
    Case { doy: 118, sec: 1.96740740740740766e+04, alt: 7.56919999999999931e+01,
        g_lat: -4.27696793002915499e+01, g_long: -1.55657400450788884e+02, lst: -4.91213945392090334e+00,
        f107a: 1.72177969959035067e+02, f107: 1.63438835741909202e+02, ap: 1.24653739612188360e+02,
        ap_array: [3.74291115311909195e+01, 4.24291115311909195e+01, 4.74291115311909195e+01, 5.24291115311909195e+01, 5.74291115311909195e+01, 6.24291115311909195e+01, 6.74291115311909266e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.34756527587211752e+09, 1.08499218373630211e+08, 4.89675499783817688e+14, 1.26177518422374234e+14, 5.83355672894846680e+12, 2.98500402862731798e-08, 7.52461067353808355e-01, 5.80877533457839672e+04, 3.73978244934486495e-60, 1.02731846489999998e+03, 2.16875366068330607e+02, 2.98500402862731798e-08] },
    // case 330: alt 194.816 km, lat -17.055, lon -122.930, doy 301, ap 140.443
    Case { doy: 301, sec: 4.84740740740740730e+04, alt: 1.94816000000000003e+02,
        g_lat: -1.70553935860058345e+01, g_long: -1.22930127723516151e+02, lst: 5.26967872789727743e+00,
        f107a: 1.90254893035958162e+02, f107: 1.77262365153673926e+02, ap: 1.40443213296398881e+02,
        ap_array: [4.17769376181474428e+01, 4.67769376181474428e+01, 5.17769376181474428e+01, 5.67769376181474428e+01, 6.17769376181474428e+01, 6.67769376181474428e+01, 7.17769376181474428e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.60451927719457820e+07, 9.17925218943815422e+09, 6.25133503882031536e+09, 1.81642095124989629e+08, 9.35204731360758841e+06, 5.45887769542552354e-13, 6.94149918768907228e+04, 4.94113357638268918e+07, 8.96163841153245944e-09, 1.09968335333581240e+03, 9.78364049579553125e+02, 5.45887769542552354e-13] },
    // case 331: alt 680.320 km, lat +8.659, lon -90.203, doy 73, ap 156.233
    Case { doy: 73, sec: 7.72740740740740584e+04, alt: 6.80319999999999936e+02,
        g_lat: 8.65889212827987365e+00, g_long: -9.02028549962434170e+01, lst: 1.54514969097154538e+01,
        f107a: 2.08331816112881228e+02, f107: 1.91085894565438650e+02, ap: 1.56232686980609429e+02,
        ap_array: [4.61247637051039661e+01, 5.11247637051039661e+01, 5.61247637051039661e+01, 6.11247637051039661e+01, 6.61247637051039590e+01, 7.11247637051039590e+01, 7.61247637051039590e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.92441345637161285e+06, 2.33647398703864105e+07, 2.31420459387786541e+05, 9.56133640460436982e+03, 3.38291529779375777e+00, 6.89267316972003313e-16, 2.50125745431723699e+04, 1.63406428669356229e+06, 1.52986068481612921e+05, 1.44159110438762696e+03, 1.44149583819264649e+03, 6.93330626950874962e-16] },
    // case 332: alt 48.372 km, lat +34.373, lon -57.476, doy 255, ap 172.022
    Case { doy: 255, sec: 3.67407407407407391e+03, alt: 4.83720000000000070e+01,
        g_lat: 3.43731778425655961e+01, g_long: -5.74755822689707117e+01, lst: 1.02057613168724282e+00,
        f107a: 2.26408739189804294e+02, f107: 2.04909423977203346e+02, ap: 1.72022160664819921e+02,
        ap_array: [5.04725897920604964e+01, 5.54725897920604964e+01, 6.04725897920604964e+01, 6.54725897920604893e+01, 7.04725897920604893e+01, 7.54725897920604893e+01, 8.04725897920604893e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [1.52497887791580109e+11, 0.00000000000000000e+00, 2.27245195435929560e+16, 6.09630685795968200e+15, 2.71819340520687156e+14, 1.39812130547756744e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.62082684331457301e+02, 1.39812130547756744e-06] },
    // case 333: alt 113.692 km, lat +60.087, lon -24.748, doy 164, ap 187.812
    Case { doy: 164, sec: 3.24740740740740730e+04, alt: 1.13692000000000007e+02,
        g_lat: 6.00874635568513042e+01, g_long: -2.47483095416979779e+01, lst: 7.37068882890737775e+00,
        f107a: 2.44485662266727360e+02, f107: 2.18732953388968042e+02, ap: 1.87811634349030470e+02,
        ap_array: [5.48204158790170126e+01, 5.98204158790170126e+01, 6.48204158790170197e+01, 6.98204158790170197e+01, 7.48204158790170197e+01, 7.98204158790170197e+01, 8.48204158790170197e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [4.95140210458281171e+06, 1.50772080123713207e+10, 2.77165094066188660e+11, 3.89460268487968979e+10, 2.70896672917427635e+09, 1.55318176846468358e-11, 8.77292869576045778e+05, 4.64586936532981577e+05, 7.52495748288427010e-35, 1.02731846489999998e+03, 6.81878848676276334e+02, 1.55318176846468358e-11] },
    // case 334: alt 150.016 km, lat +85.802, lon +7.979, doy 346, ap 203.601
    Case { doy: 346, sec: 6.12740740740740730e+04, alt: 1.50015999999999991e+02,
        g_lat: 8.58017492711370267e+01, g_long: 7.97896318557477002e+00, lst: 1.75525070107255594e+01,
        f107a: 2.62562585343650426e+02, f107: 2.32556482800732738e+02, ap: 2.03601108033240990e+02,
        ap_array: [5.91682419659735359e+01, 6.41682419659735359e+01, 6.91682419659735359e+01, 7.41682419659735359e+01, 7.91682419659735359e+01, 8.41682419659735359e+01, 8.91682419659735359e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [2.23853939391526626e+06, 5.78901248987577152e+09, 3.93951547135450363e+10, 4.45494968711436844e+09, 3.02745754691289246e+08, 2.24169991740620247e-12, 7.04747187317430071e+04, 3.99002620064713666e+06, 4.36739480590148306e-21, 1.83574786044118332e+03, 1.01223513100766479e+03, 2.24169991740620247e-12] },
    // case 335: alt 584.320 km, lat -64.810, lon +40.706, doy 15, ap 219.391
    Case { doy: 15, sec: 1.32740740740740748e+04, alt: 5.84320000000000050e+02,
        g_lat: -6.48104956268221599e+01, g_long: 4.07062359128474611e+01, lst: 6.40099185921040714e+00,
        f107a: 2.80639508420573520e+02, f107: 2.46380012212497434e+02, ap: 2.19390581717451482e+02,
        ap_array: [6.35160680529300663e+01, 6.85160680529300663e+01, 7.35160680529300663e+01, 7.85160680529300663e+01, 8.35160680529300663e+01, 8.85160680529300663e+01, 9.35160680529300663e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [4.10907223776385537e+05, 3.43771327760711759e+07, 1.19454202522288747e+07, 1.23790525634357691e+05, 7.38390903699496266e+03, 1.56148704417067980e-15, 6.32954567032308114e+03, 3.58873877027495764e+06, 5.26166155476904823e+05, 1.91221348009549229e+03, 1.90273041835841104e+03, 1.57546201726014644e-15] },
    // case 336: alt 36.772 km, lat -39.096, lon +73.434, doy 198, ap 235.180
    Case { doy: 198, sec: 4.20740740740740730e+04, alt: 3.67719999999999985e+01,
        g_lat: -3.90962099125364517e+01, g_long: 7.34335086401202091e+01, lst: 1.16872427983539087e+01,
        f107a: 2.98716431497496615e+02, f107: 2.60203541624262130e+02, ap: 2.35180055401662031e+02,
        ap_array: [6.78638941398865825e+01, 7.28638941398865825e+01, 7.78638941398865825e+01, 8.28638941398865825e+01, 8.78638941398865825e+01, 9.28638941398865825e+01, 9.78638941398865825e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [6.16238564891309692e+11, 0.00000000000000000e+00, 9.18289788415117440e+16, 2.46349601538123560e+16, 1.09841233041274275e+15, 5.64975869048745599e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.45623278204312754e+02, 5.64975869048745599e-06] },
    // case 337: alt 106.092 km, lat -13.382, lon +106.161, doy 107, ap 250.970
    Case { doy: 107, sec: 7.08740740740740584e+04, alt: 1.06092000000000013e+02,
        g_lat: -1.33819241982507293e+01, g_long: 1.06160781367392929e+02, lst: 2.67646282228467669e+01,
        f107a: 6.52139280837505737e+01, f107: 2.74027071036026882e+02, ap: 2.50969529085872580e+02,
        ap_array: [7.22117202268430987e+01, 7.72117202268430987e+01, 8.22117202268430987e+01, 8.72117202268430987e+01, 9.22117202268430987e+01, 9.72117202268430987e+01, 1.02211720226843099e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [6.07836418577362671e+07, 3.70256887755775330e+11, 3.21056065094736475e+12, 9.17204544323264404e+11, 2.70299105907227173e+10, 2.09578004445209854e-10, 8.71248353168769181e+06, 5.56222139424274093e+05, 1.80600634955995645e-39, 1.02731846489999998e+03, 2.40611130960496325e+02, 2.09578004445209854e-10] },
    // case 338: alt 374.016 km, lat +12.332, lon +138.888, doy 289, ap 266.759
    Case { doy: 289, sec: 2.28740740740740730e+04, alt: 3.74015999999999963e+02,
        g_lat: 1.23323615160349789e+01, g_long: 1.38888054094665677e+02, lst: 1.56131130713316217e+01,
        f107a: 8.32908511606736397e+01, f107: 2.87850600447791578e+02, ap: 2.66759002770083100e+02,
        ap_array: [7.65595463137996148e+01, 8.15595463137996148e+01, 8.65595463137996148e+01, 9.15595463137996148e+01, 9.65595463137996148e+01, 1.01559546313799615e+02, 1.06559546313799615e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [7.30370733345520124e+06, 5.03998868455237627e+08, 3.81356274567336887e+07, 1.08904790117534483e+06, 3.60761262050855976e+03, 1.58761552495305651e-14, 7.14562638815468963e+04, 2.62778113205615431e+07, 1.56006116433403658e+05, 1.22753467079005577e+03, 1.21217463221663547e+03, 1.58802987719830345e-14] },
    // case 339: alt 488.320 km, lat +38.047, lon +171.615, doy 61, ap 282.548
    Case { doy: 61, sec: 5.16740740740740657e+04, alt: 4.88319999999999993e+02,
        g_lat: 3.80466472303207013e+01, g_long: 1.71615326821938424e+02, lst: 2.57949312531498016e+01,
        f107a: 1.01367774237596734e+02, f107: 6.74872786484836098e+01, ap: 2.82548476454293620e+02,
        ap_array: [8.09073724007561452e+01, 8.59073724007561452e+01, 9.09073724007561452e+01, 9.59073724007561452e+01, 1.00907372400756145e+02, 1.05907372400756145e+02, 1.10907372400756145e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [3.31217529604381835e+06, 2.23573139743092544e+07, 4.75130928556724451e+05, 1.32229223439053458e+04, 3.15540530770822514e+01, 6.41817909625649140e-16, 6.29247356426280821e+04, 1.34327412123426795e+05, 1.18733670662347969e+05, 9.52934518504227867e+02, 9.52901682438185503e+02, 6.44971475918441108e-16] },
    // case 340: alt 25.172 km, lat +63.761, lon -152.682, doy 244, ap 298.338
    Case { doy: 244, sec: 8.04740740740740584e+04, alt: 2.51720000000000006e+01,
        g_lat: 6.37609329446063953e+01, g_long: -1.52682193839218627e+02, lst: 2.23539094650205712e+01,
        f107a: 1.19444697314519800e+02, f107: 8.13108080602483199e+01, ap: 2.98337950138504141e+02,
        ap_array: [8.52551984877126756e+01, 9.02551984877126756e+01, 9.52551984877126756e+01, 1.00255198487712676e+02, 1.05255198487712676e+02, 1.10255198487712676e+02, 1.15255198487712676e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.51362570554096191e+12, 0.00000000000000000e+00, 6.72599319527705216e+17, 1.80438219449697472e+17, 8.04529676053064600e+15, 4.13815322641895806e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.25277225906039973e+02, 4.13815322641895806e-05] },
    // case 341: alt 98.492 km, lat +89.475, lon -119.955, doy 152, ap 14.958
    Case { doy: 152, sec: 6.87407407407407300e+03, alt: 9.84920000000000044e+01,
        g_lat: 8.94752186588921177e+01, g_long: -1.19954921111945907e+02, lst: -6.08752972022026206e+00,
        f107a: 1.37521620391442866e+02, f107: 9.51343374720130299e+01, ap: 1.49584487534626049e+01,
        ap_array: [8.96030245746691918e+01, 9.46030245746691918e+01, 9.96030245746691918e+01, 1.04603024574669192e+02, 1.09603024574669192e+02, 1.14603024574669192e+02, 1.19603024574669192e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.57784383875629529e+07, 1.51662037256705475e+11, 8.27079273644438867e+12, 1.72404100757979810e+12, 8.43273348631421051e+10, 4.85635447693766569e-10, 1.40161179936094061e+07, 1.80772591417634365e+05, 3.59915997748395328e-43, 1.02731846489999998e+03, 1.81608047335187962e+02, 4.85635447693766569e-10] },
    // case 342: alt 329.216 km, lat -89.925, lon -87.228, doy 335, ap 30.748
    Case { doy: 335, sec: 3.56740740740740730e+04, alt: 3.29216000000000008e+02,
        g_lat: -8.99250312369845943e+01, g_long: -8.72276483846731736e+01, lst: 4.09428846159792048e+00,
        f107a: 1.55598543468365961e+02, f107: 1.08957866883777726e+02, ap: 3.07479224376731288e+01,
        ap_array: [9.39508506616257080e+01, 9.89508506616257080e+01, 1.03950850661625708e+02, 1.08950850661625708e+02, 1.13950850661625708e+02, 1.18950850661625708e+02, 1.23950850661625708e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.23432860370940762e+05, 2.90518864373802662e+08, 2.11291870754933745e+08, 1.36182754946939535e+07, 1.24951264617690802e+05, 1.83841655348223185e-14, 4.19760912392188038e+04, 4.72898070618426893e+06, 1.81866496339642549e+03, 1.26792146716645198e+03, 1.25494617624315219e+03, 1.83842138385637445e-14] },
    // case 343: alt 968.320 km, lat -64.211, lon -54.500, doy 38, ap 46.537
    Case { doy: 38, sec: 6.44740740740740657e+04, alt: 9.68320000000000050e+02,
        g_lat: -6.42107455226988861e+01, g_long: -5.45003756574004683e+01, lst: 1.42761066434160977e+01,
        f107a: 1.73675466545289055e+02, f107: 1.22781396295542436e+02, ap: 4.65373961218836527e+01,
        ap_array: [9.82986767485822384e+01, 1.03298676748582238e+02, 1.08298676748582238e+02, 1.13298676748582238e+02, 1.18298676748582238e+02, 1.23298676748582238e+02, 1.28298676748582238e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [2.12489831520064705e+05, 2.60190142239036250e+04, 4.33160568337114071e+00, 7.12305684190444894e-03, 6.89957222402054459e-07, 2.19047222481121801e-18, 2.17047414832129616e+04, 2.24798379579255106e+03, 2.76471082488179563e+04, 1.02731846489999998e+03, 1.02731846475255043e+03, 2.92477941989982291e-18] },
    // case 344: alt 13.572 km, lat -38.496, lon -21.773, doy 221, ap 62.327
    Case { doy: 221, sec: 1.64740740740740766e+04, alt: 1.35720000000000010e+01,
        g_lat: -3.84964598084131637e+01, g_long: -2.17731029301277204e+01, lst: 4.57613168724279884e+00,
        f107a: 1.91752389622212121e+02, f107: 1.36604925707307132e+02, ap: 6.23268698060941801e+01,
        ap_array: [2.83553875236294894e+00, 7.83553875236294850e+00, 1.28355387523629485e+01, 1.78355387523629503e+01, 2.28355387523629503e+01, 2.78355387523629503e+01, 3.28355387523629503e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [2.35188717989487031e+13, 0.00000000000000000e+00, 3.50467189794071347e+18, 9.40198330060555264e+17, 4.19211329072178400e+16, 2.15624204499449066e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.13383749336096116e+02, 2.15624204499449066e-04] },
    // case 345: alt 90.892 km, lat -12.782, lon +10.954, doy 130, ap 78.116
    Case { doy: 130, sec: 4.52740740740740657e+04, alt: 9.08919999999999959e+01,
        g_lat: -1.27821740941274555e+01, g_long: 1.09541697971449992e+01, lst: 1.33064096737191289e+01,
        f107a: 2.09829312699135187e+02, f107: 1.50428455119071828e+02, ap: 7.81163434903047147e+01,
        ap_array: [7.18336483931947001e+00, 1.21833648393194700e+01, 1.71833648393194700e+01, 2.21833648393194700e+01, 2.71833648393194700e+01, 3.21833648393194665e+01, 3.71833648393194665e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [4.05836836869683266e+08, 4.44752850373436340e+11, 4.98601836610670078e+13, 1.09708359323936875e+13, 5.71084572626271484e+11, 2.95000757942290608e-09, 5.18196578482109904e+07, 2.59131974390350340e+05, 3.68360331361772665e-48, 1.02731846489999998e+03, 1.83182282948966474e+02, 2.95000757942290608e-09] },
    // case 346: alt 284.416 km, lat +12.932, lon +43.681, doy 312, ap 93.906
    Case { doy: 312, sec: 7.40740740740740584e+04, alt: 2.84416000000000054e+02,
        g_lat: 1.29321116201582669e+01, g_long: 4.36814425244176903e+01, lst: 2.34882278555373034e+01,
        f107a: 2.27906235776058253e+02, f107: 1.64251984530836552e+02, ap: 9.39058171745152350e+01,
        ap_array: [1.15311909262759915e+01, 1.65311909262759897e+01, 2.15311909262759897e+01, 2.65311909262759897e+01, 3.15311909262759897e+01, 3.65311909262759897e+01, 4.15311909262759897e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [1.08000819264396708e+07, 1.77333552474744129e+09, 3.03576267675811708e+08, 3.59477244912811927e+06, 7.58860408239366952e+04, 6.20776116277560519e-14, 6.56197014667661570e+04, 2.58081131922498196e+07, 9.86791654598637713e+00, 1.11096881445794224e+03, 1.08281641973836986e+03, 6.20776118898479185e-14] },
    // case 347: alt 872.320 km, lat +38.646, lon +76.409, doy 84, ap 109.695
    Case { doy: 84, sec: 2.60740740740740730e+04, alt: 8.72320000000000050e+02,
        g_lat: 3.86463973344439751e+01, g_long: 7.64087152516904666e+01, lst: 1.23367127040221618e+01,
        f107a: 2.45983158852981347e+02, f107: 1.78075513942601276e+02, ap: 1.09695290858725770e+02,
        ap_array: [1.58790170132325130e+01, 2.08790170132325130e+01, 2.58790170132325130e+01, 3.08790170132325130e+01, 3.58790170132325130e+01, 4.08790170132325130e+01, 4.58790170132325130e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [1.63208654078262299e+06, 2.16453898694010125e+06, 3.27954956635987946e+03, 8.11617161604593385e+00, 4.39783503839765496e-03, 7.44365150191396208e-17, 1.97777384755443600e+04, 2.54888509571755334e+05, 3.74802429477579644e+04, 1.32075711987323757e+03, 1.32075705743466619e+03, 7.54319902718320747e-17] },
    // case 348: alt 71.572 km, lat +64.361, lon +109.136, doy 266, ap 125.485
    Case { doy: 266, sec: 5.48740740740740730e+04, alt: 7.15720000000000027e+01,
        g_lat: 6.43606830487296975e+01, g_long: 1.09135987978963158e+02, lst: 1.52427983539094640e+01,
        f107a: 2.64060081929904413e+02, f107: 1.91899043354365972e+02, ap: 1.25484764542936290e+02,
        ap_array: [2.02268431001890363e+01, 2.52268431001890363e+01, 3.02268431001890363e+01, 3.52268431001890363e+01, 4.02268431001890363e+01, 4.52268431001890363e+01, 5.02268431001890363e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [5.42726300644472504e+09, 0.00000000000000000e+00, 8.01459416528916250e+14, 1.95825130685537094e+14, 9.56729415161532617e+12, 4.82893689909733800e-08, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.17249177965983051e+02, 4.82893689909733800e-08] },
    // case 349: alt 74.172 km, lat -86.252, lon +141.863, doy 175, ap 141.274
    Case { doy: 175, sec: 8.36740740740740584e+04, alt: 7.41719999999999970e+01,
        g_lat: -8.62515618492294891e+01, g_long: 1.41863260706235906e+02, lst: 3.27003490676585216e+01,
        f107a: 2.82137005006827508e+02, f107: 2.05722572766130696e+02, ap: 1.41274238227146810e+02,
        ap_array: [2.45746691871455560e+01, 2.95746691871455560e+01, 3.45746691871455596e+01, 3.95746691871455596e+01, 4.45746691871455596e+01, 4.95746691871455596e+01, 5.45746691871455596e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [2.40093654852984285e+09, 5.05827770579602662e+06, 3.54719415065051938e+14, 8.96101350399211406e+13, 4.23794676741236035e+12, 2.15288645278770970e-08, 2.72605606080007351e-10, 1.76485879548510857e+04, 9.97733346566499531e-62, 1.02731846489999998e+03, 2.27009694339718749e+02, 2.15288645278770970e-08] },
    // case 350: alt 185.856 km, lat -60.537, lon +174.591, doy 358, ap 157.064
    Case { doy: 358, sec: 1.54074074074074042e+03, alt: 1.85855999999999995e+02,
        g_lat: -6.05372761349437809e+01, g_long: 1.74590533433508654e+02, lst: 1.20673524346618937e+01,
        f107a: 6.66044606281292744e+01, f107: 2.19546102177895392e+02, ap: 1.57063711911357331e+02,
        ap_array: [2.89224952741020793e+01, 3.39224952741020758e+01, 3.89224952741020758e+01, 4.39224952741020758e+01, 4.89224952741020758e+01, 5.39224952741020758e+01, 5.89224952741020758e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [6.76613817106319359e+05, 2.48253136702764034e+09, 9.82130893678283119e+09, 1.17004472276158595e+09, 4.57770103044214547e+07, 5.91063941095866768e-13, 1.07791615285524575e+05, 1.47866944871368915e+08, 2.60436020032780067e-11, 1.34425984854619423e+03, 1.06002375713496053e+03, 5.91063941095866768e-13] },
    // case 351: alt 661.120 km, lat -34.823, lon -149.707, doy 10, ap 172.853
    Case { doy: 10, sec: 3.03407407407407372e+04, alt: 6.61120000000000005e+02,
        g_lat: -3.48229904206580585e+01, g_long: -1.49706987227648369e+02, lst: -1.55248227608190881e+00,
        f107a: 8.46813837050523404e+01, f107: 2.33369631589660088e+02, ap: 1.72853185595567851e+02,
        ap_array: [3.32703213610585991e+01, 3.82703213610585991e+01, 4.32703213610585991e+01, 4.82703213610585991e+01, 5.32703213610585991e+01, 5.82703213610585991e+01, 6.32703213610585991e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.64365105593348446e+05, 3.72769987726031570e+06, 2.78354770792066993e+04, 4.06454139081502717e+02, 4.00654678168079925e-01, 1.07849975133622015e-16, 4.19588971012866750e+04, 1.88202233273661142e+05, 1.41075455103291140e+05, 1.13522409673372727e+03, 1.13521033944222040e+03, 1.11596939221165406e-16] },
    // case 352: alt 46.052 km, lat -9.109, lon -116.980, doy 192, ap 188.643
    Case { doy: 192, sec: 5.91407407407407300e+04, alt: 4.60520000000000067e+01,
        g_lat: -9.10870470637233609e+00, g_long: -1.16979714500375650e+02, lst: 1.64279835390946474e+01,
        f107a: 1.02758306781975421e+02, f107: 2.47193161001424784e+02, ap: 1.88642659279778400e+02,
        ap_array: [3.76181474480151223e+01, 4.26181474480151223e+01, 4.76181474480151223e+01, 5.26181474480151223e+01, 5.76181474480151223e+01, 6.26181474480151223e+01, 6.76181474480151223e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.96402828098840546e+11, 0.00000000000000000e+00, 2.92670276958121560e+16, 7.85146551995552700e+15, 3.50077551783457125e+14, 1.80064774927458675e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.64048420112206145e+02, 1.80064774927458675e-06] },
    // case 353: alt 112.172 km, lat +16.606, lon -84.252, doy 101, ap 204.432
    Case { doy: 101, sec: 1.11407407407407427e+04, alt: 1.12171999999999997e+02,
        g_lat: 1.66055810079133721e+01, g_long: -8.42524417731029303e+01, lst: -2.52217924577887809e+00,
        f107a: 1.20835229858898501e+02, f107: 2.61016690413189508e+02, ap: 2.04432132963988892e+02,
        ap_array: [4.19659735349716385e+01, 4.69659735349716385e+01, 5.19659735349716385e+01, 5.69659735349716385e+01, 6.19659735349716385e+01, 6.69659735349716385e+01, 7.19659735349716385e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [7.00265929234701693e+07, 4.60996485930953979e+11, 2.34280421470722510e+12, 3.68086426542717651e+11, 1.81639687177528305e+10, 1.41896995131855148e-10, 6.30492469792212732e+06, 3.21094566287330259e+06, 1.71888162552603071e-34, 1.02731846489999998e+03, 2.00695028283073469e+02, 1.41896995131855148e-10] },
    // case 354: alt 141.056 km, lat +42.320, lon -51.525, doy 284, ap 220.222
    Case { doy: 284, sec: 3.99407407407407372e+04, alt: 1.41056000000000012e+02,
        g_lat: 4.23198667221990661e+01, g_long: -5.15251690458302107e+01, lst: 7.65963893603930135e+00,
        f107a: 1.38912152935821553e+02, f107: 2.74840219824954204e+02, ap: 2.20221606648199412e+02,
        ap_array: [4.63137996219281618e+01, 5.13137996219281618e+01, 5.63137996219281618e+01, 6.13137996219281618e+01, 6.63137996219281547e+01, 7.13137996219281547e+01, 7.63137996219281547e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [1.50496189024615809e+07, 2.46690739433913879e+10, 5.60419896713467026e+10, 6.46722881386210918e+09, 2.45757995426082999e+08, 3.62025592160733632e-12, 2.67834910899154318e+05, 1.10042247226722948e+07, 1.21896510022485092e-23, 1.29749231900508380e+03, 6.26799266059087699e+02, 3.62025592160733632e-12] },
    // case 355: alt 565.120 km, lat +68.034, lon -18.798, doy 55, ap 236.011
    Case { doy: 55, sec: 6.87407407407407300e+04, alt: 5.65120000000000005e+02,
        g_lat: 6.80341524364847885e+01, g_long: -1.87978963185574628e+01, lst: 1.78414571178574839e+01,
        f107a: 1.56989076012744647e+02, f107: 2.88663749236718900e+02, ap: 2.36011080332409961e+02,
        ap_array: [5.06616257088846922e+01, 5.56616257088846922e+01, 6.06616257088846922e+01, 6.56616257088846851e+01, 7.06616257088846851e+01, 7.56616257088846851e+01, 8.06616257088846851e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [8.47482791381067480e+05, 4.54749019619333893e+07, 5.48147099540411122e+06, 2.48991867370153108e+05, 6.86904724797015660e+02, 1.49510118625482838e-15, 9.08794406436890858e+03, 5.84965064076940762e+05, 1.06943447663772859e+04, 1.65809114010037138e+03, 1.65445190289745551e+03, 1.49538522805182349e-15] },
    // case 356: alt 34.452 km, lat -82.578, lon +13.929, doy 238, ap 251.801
    Case { doy: 238, sec: 2.07407407407407409e+04, alt: 3.44519999999999982e+01,
        g_lat: -8.25780924614743839e+01, g_long: 1.39293764087152283e+01, lst: 5.76131687242798396e+00,
        f107a: 1.75065999089667741e+02, f107: 6.83004274374109457e+01, ap: 2.51800554016620481e+02,
        ap_array: [5.50094517958412155e+01, 6.00094517958412155e+01, 6.50094517958412155e+01, 7.00094517958412155e+01, 7.50094517958412155e+01, 8.00094517958412155e+01, 8.50094517958412155e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [5.46692313490001526e+11, 0.00000000000000000e+00, 8.14655228485172320e+16, 2.18547558145717200e+16, 9.74449851552538875e+14, 5.01214923104918150e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.49402534408898958e+02, 5.01214923104918150e-06] },
    // case 357: alt 104.572 km, lat -56.864, lon +46.657, doy 147, ap 267.590
    Case { doy: 147, sec: 4.95407407407407445e+04, alt: 1.04572000000000003e+02,
        g_lat: -5.68638067471886757e+01, g_long: 4.66566491359879478e+01, lst: 1.68717601481605151e+01,
        f107a: 1.93142922166590836e+02, f107: 8.21239568491756557e+01, ap: 2.67590027700831001e+02,
        ap_array: [5.93572778827977316e+01, 6.43572778827977316e+01, 6.93572778827977316e+01, 7.43572778827977316e+01, 7.93572778827977316e+01, 8.43572778827977316e+01, 8.93572778827977316e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [4.02889114227139205e+07, 1.74470459235969513e+11, 1.66848049949530444e+12, 3.59126235088362183e+11, 1.39372496793191490e+10, 1.02187409365838625e-10, 6.05867564676283952e+06, 1.63064350969854015e+05, 2.32980013934837138e-40, 1.02731846489999998e+03, 2.36801707641383217e+02, 1.02187409365838625e-10] },
    // case 358: alt 365.056 km, lat -31.150, lon +79.384, doy 329, ap 283.380
    Case { doy: 329, sec: 7.83407407407407300e+04, alt: 3.65056000000000040e+02,
        g_lat: -3.11495210329029604e+01, g_long: 7.93839218632606958e+01, lst: 2.70535783299786914e+01,
        f107a: 2.11219845243513902e+02, f107: 9.59474862609403658e+01, ap: 2.83379501385041522e+02,
        ap_array: [6.37051039697542620e+01, 6.87051039697542620e+01, 7.37051039697542620e+01, 7.87051039697542620e+01, 8.37051039697542620e+01, 8.87051039697542620e+01, 9.37051039697542620e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [5.31046235157047585e+06, 5.01294081020495534e+08, 7.60728933286748230e+07, 4.13348084831407235e+05, 1.13915583949743486e+04, 1.70697665849970437e-14, 9.72962313982153719e+04, 6.94454731814367976e+06, 8.83213306793447259e+02, 1.15833037663935579e+03, 1.15655418292552849e+03, 1.70697900431424748e-14] },
    // case 359: alt 469.120 km, lat -5.435, lon +112.111, doy 33, ap 299.169
    Case { doy: 33, sec: 4.74074074074074088e+03, alt: 4.69120000000000005e+02,
        g_lat: -5.43523531861724507e+00, g_long: 1.12111194590533444e+02, lst: 8.79095206735243551e+00,
        f107a: 2.29296768320436968e+02, f107: 1.09771015672705062e+02, ap: 2.99168975069252042e+02,
        ap_array: [6.80529300567107782e+01, 7.30529300567107782e+01, 7.80529300567107782e+01, 8.30529300567107782e+01, 8.80529300567107782e+01, 9.30529300567107782e+01, 9.80529300567107782e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [6.47786815471663326e+06, 1.86118032285986871e+08, 4.03511043246886693e+06, 3.04347384914804570e+04, 2.89313501976814962e+02, 5.21396885315999693e-15, 6.22333647630997875e+04, 1.65102095890740165e+06, 2.52740126829391578e+06, 1.13134124306209992e+03, 1.13119384379786879e+03, 5.28109663084588330e-15] },
    // case 360: alt 22.852 km, lat +20.279, lon +144.838, doy 215, ap 0.044
    Case { doy: 215, sec: 3.35407407407407372e+04, alt: 2.28520000000000039e+01,
        g_lat: 2.02790503956684773e+01, g_long: 1.44838467317806192e+02, lst: 9.31687242798353843e+00,
        f107a: 2.47373691397360062e+02, f107: 1.23594545084469772e+02, ap: 4.37381542498906511e-02,
        ap_array: [7.24007561436672944e+01, 7.74007561436672944e+01, 8.24007561436672944e+01, 8.74007561436672944e+01, 9.24007561436672944e+01, 9.74007561436672944e+01, 1.02400756143667294e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [6.37821403852225977e+12, 0.00000000000000000e+00, 9.50451522120168960e+17, 2.54977629839177728e+17, 1.13688258818404000e+16, 5.84763308350979920e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.16057825033643326e+02, 5.84763308350979920e-05] },
    // case 361: alt 96.972 km, lat +45.993, lon +177.566, doy 124, ap 15.833
    Case { doy: 124, sec: 6.23407407407407445e+04, alt: 9.69720000000000084e+01,
        g_lat: 4.59933361099541855e+01, g_long: 1.77565740045078883e+02, lst: 2.91545884309887988e+01,
        f107a: 2.65450614474283157e+02, f107: 1.37418074496234482e+02, ap: 1.58332118384604161e+01,
        ap_array: [7.67485822306238248e+01, 8.17485822306238248e+01, 8.67485822306238248e+01, 9.17485822306238248e+01, 9.67485822306238248e+01, 1.01748582230623825e+02, 1.06748582230623825e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.11436145714467093e+07, 1.19643543333125830e+12, 1.52666794451945586e+13, 1.61001665794514316e+13, 2.62242835788108582e+11, 1.61418736117642248e-09, 2.40524617613442620e+05, 6.91032981234678555e+09, 6.44463723220607902e-44, 1.02731846489999998e+03, 1.71334813473273613e+02, 1.61418736117642248e-09] },
    // case 362: alt 320.256 km, lat +71.708, lon -179.189, doy 306, ap 31.623
    Case { doy: 306, sec: 1.43407407407407427e+04, alt: 3.20256000000000029e+02,
        g_lat: 7.17076218242399079e+01, g_long: -1.79188580015026304e+02, lst: -7.96236623968488111e+00,
        f107a: 2.83527537551206251e+02, f107: 1.51241603907999178e+02, ap: 3.16226855226709418e+01,
        ap_array: [8.10964083175803410e+01, 8.60964083175803410e+01, 9.10964083175803410e+01, 9.60964083175803410e+01, 1.01096408317580341e+02, 1.06096408317580341e+02, 1.11096408317580341e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [7.28836185148255620e+06, 1.00889015921189475e+09, 1.51911416827975661e+08, 1.52195933757888572e+06, 3.54477599976996135e+04, 3.42229786922658957e-14, 5.97584053115998031e+04, 1.00825850478040054e+07, 6.82144054557658933e+02, 1.02731846489999998e+03, 1.02667949045524756e+03, 3.42229968100119843e-14] },
    // case 363: alt 949.120 km, lat -78.905, lon -146.461, doy 78, ap 47.412
    Case { doy: 78, sec: 4.31407407407407372e+04, alt: 9.49120000000000005e+02,
        g_lat: -7.89046230737192786e+01, g_long: -1.46461307287753556e+02, lst: 2.21945194213330055e+00,
        f107a: 6.79949931725079608e+01, f107: 1.65065133319763902e+02, ap: 4.74121592068814692e+01,
        ap_array: [8.54442344045368714e+01, 9.04442344045368714e+01, 9.54442344045368714e+01, 1.00444234404536871e+02, 1.05444234404536871e+02, 1.10444234404536871e+02, 1.15444234404536871e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [1.83787649653565139e+05, 3.89045693036110970e+04, 4.46517565301927988e+01, 3.76278318342017637e-01, 4.42719598372555309e-05, 2.41798716972515551e-18, 8.60346945833442151e+04, 8.35577359112267231e+02, 2.20812063386999716e+03, 1.19441855091962793e+03, 1.19441419850931675e+03, 2.47663485376074298e-18] },
    // case 364: alt 11.252 km, lat -53.190, lon -113.734, doy 261, ap 63.202
    Case { doy: 261, sec: 7.19407407407407300e+04, alt: 1.12520000000000007e+01,
        g_lat: -5.31903373594335704e+01, g_long: -1.13734034560480836e+02, lst: 1.99835390946502010e+01,
        f107a: 8.60719162494310410e+01, f107: 1.78888662731528598e+02, ap: 6.32016328910919896e+01,
        ap_array: [8.97920604914933875e+01, 9.47920604914933875e+01, 9.97920604914933875e+01, 1.04792060491493388e+02, 1.09792060491493388e+02, 1.14792060491493388e+02, 1.19792060491493388e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [4.71507930831124922e+13, 0.00000000000000000e+00, 7.02618989961024922e+18, 1.88491596436844160e+18, 8.40437704841880000e+16, 4.32284862002552139e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.14123442209294097e+02, 4.32284862002552139e-04] },
    // case 365: alt 89.372 km, lat -27.476, lon -81.007, doy 169, ap 78.991
    Case { doy: 169, sec: 2.39407407407407372e+04, alt: 8.93719999999999999e+01,
        g_lat: -2.74760516451478551e+01, g_long: -8.10067618332081310e+01, lst: 1.24975497243632905e+00,
        f107a: 1.04148839326354121e+02, f107: 1.92712192143293294e+02, ap: 7.89911065753025099e+01,
        ap_array: [9.41398865784499037e+01, 9.91398865784499037e+01, 1.04139886578449904e+02, 1.09139886578449904e+02, 1.14139886578449904e+02, 1.19139886578449904e+02, 1.24139886578449904e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [3.88639724124779344e+08, 2.60794816168892700e+11, 4.93995218758599609e+13, 1.25785008130410352e+13, 5.71516522794432495e+11, 3.00913782865382049e-09, 5.90516254580400512e+07, 1.14055374195016644e+05, 1.40104145748216915e-49, 1.02731846489999998e+03, 1.65876413396924619e+02, 3.00913782865382049e-09] },
    // case 366: alt 275.456 km, lat -1.762, lon -48.279, doy 352, ap 94.781
    Case { doy: 352, sec: 5.27407407407407372e+04, alt: 2.75456000000000017e+02,
        g_lat: -1.76176593086213984e+00, g_long: -4.82794891059353972e+01, lst: 1.14315731542545116e+01,
        f107a: 1.22225762403277201e+02, f107: 2.06535721555058018e+02, ap: 9.47805802595130444e+01,
        ap_array: [9.84877126654064341e+01, 1.03487712665406434e+02, 1.08487712665406434e+02, 1.13487712665406434e+02, 1.18487712665406434e+02, 1.23487712665406434e+02, 1.28487712665406434e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [7.61146212438357621e+06, 1.54495802931477380e+09, 3.97352190689846694e+08, 1.28007519289069660e+07, 1.42559660657134809e+05, 6.09808433349701461e-14, 7.13986483612904412e+04, 3.17438777500996254e+07, 4.33215307163093666e+00, 1.18891073100624317e+03, 1.13759163781189636e+03, 6.09808434500321229e-14] },
    // case 367: alt 853.120 km, lat +23.953, lon -15.552, doy 21, ap 110.570
    Case { doy: 21, sec: 8.15407407407407300e+04, alt: 8.53120000000000118e+02,
        g_lat: 2.39525197834235684e+01, g_long: -1.55522163786626493e+01, lst: 2.16133913360726915e+01,
        f107a: 1.40302685480200296e+02, f107: 2.20359250966822714e+02, ap: 1.10570053943723565e+02,
        ap_array: [3.02457466918714557e+00, 8.02457466918714601e+00, 1.30245746691871460e+01, 1.80245746691871460e+01, 2.30245746691871460e+01, 2.80245746691871460e+01, 3.30245746691871460e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [7.22351092505380628e+05, 2.39637645008471765e+05, 1.39241342976574998e+02, 6.10827762295014343e-01, 7.68319725451051554e-05, 1.14495850950274272e-17, 5.32038509297218552e+04, 8.32939788852940183e+03, 4.10315788655672168e+03, 1.10557272698993211e+03, 1.10557206661561986e+03, 1.15585649684943727e-17] },
    // case 368: alt 69.252 km, lat +49.667, lon +17.175, doy 204, ap 126.360
    Case { doy: 204, sec: 7.94074074074073997e+03, alt: 6.92520000000000095e+01,
        g_lat: 4.96668054977092766e+01, g_long: 1.71750563486100702e+01, lst: 2.20576131687242771e+00,
        f107a: 1.58379608557123362e+02, f107: 2.34182780378587410e+02, ap: 1.26359527627934099e+02,
        ap_array: [7.37240075614366663e+00, 1.23724007561436657e+01, 1.73724007561436657e+01, 2.23724007561436657e+01, 2.73724007561436657e+01, 3.23724007561436693e+01, 3.73724007561436693e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [1.26223172248935490e+10, 0.00000000000000000e+00, 1.88420206187038575e+15, 4.95973447222218000e+14, 2.25300992205357461e+13, 1.15419903752609701e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.12144438992018735e+02, 1.15419903752609701e-07] },
    // case 369: alt 81.772 km, lat +75.381, lon +49.902, doy 112, ap 142.149
    Case { doy: 112, sec: 3.67407407407407372e+04, alt: 8.17720000000000056e+01,
        g_lat: 7.53810912119949990e+01, g_long: 4.99023290758827898e+01, lst: 1.35325832552646137e+01,
        f107a: 1.76456531634046428e+02, f107: 2.48006309790352105e+02, ap: 1.42149001312144605e+02,
        ap_array: [1.17202268431001890e+01, 1.67202268431001890e+01, 2.17202268431001890e+01, 2.67202268431001890e+01, 3.17202268431001890e+01, 3.67202268431001926e+01, 4.17202268431001926e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [1.73128977910557723e+09, 8.45396126248253250e+09, 2.54959897308778688e+14, 6.43188225223597422e+13, 3.03596565974903613e+12, 1.54689761653876034e-08, 7.79970808330302835e+07, 1.64805013060631609e+05, 4.68366330659556094e-55, 1.02731846489999998e+03, 1.90010221510340415e+02, 1.54689761653876034e-08] },
    // case 370: alt 230.656 km, lat -75.231, lon +82.630, doy 295, ap 157.938
    Case { doy: 295, sec: 6.55407407407407300e+04, alt: 2.30656000000000006e+02,
        g_lat: -7.52311536859641876e+01, g_long: 8.26296018031555377e+01, lst: 2.37144014370827954e+01,
        f107a: 1.94533454710969522e+02, f107: 2.61829839202116830e+02, ap: 1.57938474996355154e+02,
        ap_array: [1.60680529300567123e+01, 2.10680529300567123e+01, 2.60680529300567123e+01, 3.10680529300567123e+01, 3.60680529300567088e+01, 4.10680529300567088e+01, 4.60680529300567088e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], out: [1.93508720666681090e+06, 1.35039392565566373e+09, 3.37430067213518238e+09, 9.83721926144301534e+08, 8.31862418848543521e+06, 2.45693859373111674e-13, 1.93815277099685300e+04, 7.28721822568464372e+06, 5.82291795761987851e-04, 1.02731846489999998e+03, 1.01493576407867113e+03, 2.45693859373127123e-13] },
    // case 371: alt 757.120 km, lat -49.517, lon +115.357, doy 67, ap 173.728
    Case { doy: 67, sec: 1.75407407407407409e+04, alt: 7.57120000000000118e+02,
        g_lat: -4.95168679716784723e+01, g_long: 1.15356874530428286e+02, lst: 1.25628862855676466e+01,
        f107a: 2.12610377787892588e+02, f107: 2.75653368613881526e+02, ap: 1.73727948680565703e+02,
        ap_array: [2.04158790170132320e+01, 2.54158790170132320e+01, 3.04158790170132320e+01, 3.54158790170132320e+01, 4.04158790170132320e+01, 4.54158790170132320e+01, 5.04158790170132320e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], out: [6.52489125882531167e+05, 9.54426656160752848e+06, 2.41267160215050651e+05, 2.14464999270403096e+04, 1.21784318141498389e+01, 3.15643491914907093e-16, 7.81139308418914607e+03, 1.95559952665400133e+06, 7.55029985693895287e+04, 1.68842702848198928e+03, 1.68831901674293113e+03, 3.17648851556910086e-16] },
    // case 372: alt 57.652 km, lat -23.803, lon +148.084, doy 249, ap 189.517
    Case { doy: 249, sec: 4.63407407407407300e+04, alt: 5.76520000000000081e+01,
        g_lat: -2.38025822573927570e+01, g_long: 1.48084147257701034e+02, lst: 1.28724279835390902e+01,
        f107a: 2.30687300864815654e+02, f107: 2.89476898025646278e+02, ap: 1.89517422364776223e+02,
        ap_array: [2.47637051039697553e+01, 2.97637051039697553e+01, 3.47637051039697553e+01, 3.97637051039697553e+01, 4.47637051039697553e+01, 4.97637051039697553e+01, 5.47637051039697553e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [5.47980653169486542e+10, 0.00000000000000000e+00, 8.16575051811857100e+15, 2.19062589149566900e+15, 9.76746248224752812e+13, 5.02396090385767236e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.46893747757492122e+02, 5.02396090385767236e-07] },
    // case 373: alt 119.772 km, lat +1.912, lon -176.213, doy 158, ap 205.307
    Case { doy: 158, sec: 7.51407407407407154e+04, alt: 1.19772000000000006e+02,
        g_lat: 1.91170345689296539e+00, g_long: -1.76213373403456046e+02, lst: 9.12486975664201694e+00,
        f107a: 2.48764223941738720e+02, f107: 6.91135762263382816e+01, ap: 2.05306896048986715e+02,
        ap_array: [2.91115311909262751e+01, 3.41115311909262715e+01, 3.91115311909262715e+01, 4.41115311909262715e+01, 4.91115311909262715e+01, 5.41115311909262715e+01, 5.91115311909262715e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], out: [6.15502722030440420e+07, 2.38537355639711517e+11, 4.05775922080783752e+11, 2.28386833146443146e+11, 2.54210254853092909e+09, 3.75138280727131061e-11, 3.97440487379414029e+05, 7.18481458149371505e+08, 2.32660206122065175e-31, 1.02731846489999998e+03, 3.68155855858427401e+02, 3.75138280727131061e-11] },
    // case 374: alt 121.344 km, lat +27.626, lon -143.486, doy 341, ap 221.096
    Case { doy: 341, sec: 2.71407407407407372e+04, alt: 1.21343999999999994e+02,
        g_lat: 2.76259891711786736e+01, g_long: -1.43486100676183327e+02, lst: -2.02664539487312734e+00,
        f107a: 2.66841147018661786e+02, f107: 8.29371056381029916e+01, ap: 2.21096369733197264e+02,
        ap_array: [3.34593572778828019e+01, 3.84593572778828019e+01, 4.34593572778828019e+01, 4.84593572778828019e+01, 5.34593572778828019e+01, 5.84593572778828019e+01, 6.34593572778828019e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], out: [5.79436549468856379e+07, 2.27405126037123413e+11, 3.52609806850220032e+11, 3.00644083350477112e+11, 1.05621593312610185e+09, 3.84868651831170183e-11, 3.64447020130552875e+05, 7.29320859938886881e+08, 2.84486769259744539e-31, 1.02731846489999998e+03, 3.99187050541234953e+02, 3.84868651831170183e-11] },
    // case 375: alt 522.880 km, lat +53.340, lon -110.759, doy 44, ap 236.886
    Case { doy: 44, sec: 5.59407407407407300e+04, alt: 5.22879999999999995e+02,
        g_lat: 5.33402748854643676e+01, g_long: -1.10758827948910593e+02, lst: 8.15517278694505166e+00,
        f107a: 2.84918070095584881e+02, f107: 9.67606350498677017e+01, ap: 2.36885843417407813e+02,
        ap_array: [3.78071833648393181e+01, 4.28071833648393181e+01, 4.78071833648393181e+01, 5.28071833648393181e+01, 5.78071833648393181e+01, 6.28071833648393181e+01, 6.78071833648393181e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1], out: [3.15124689034975367e+06, 7.29667594498482496e+07, 2.83024032887222525e+06, 8.48592693834985112e+04, 1.06350402567807635e+02, 2.12438490996576021e-15, 8.11999740067865787e+04, 1.25922337033793284e+06, 9.44528933059269912e+03, 1.24231479700384284e+03, 1.24229681977180439e+03, 2.12463577685038026e-15] },
    // case 376: alt 29.348 km, lat +79.055, lon -78.032, doy 226, ap 252.675
    Case { doy: 226, sec: 8.47407407407407300e+04, alt: 2.93480000000000025e+01,
        g_lat: 7.90545605997500900e+01, g_long: -7.80315552216378876e+01, lst: 2.35390946502057581e+01,
        f107a: 6.93855257168866615e+01, f107: 1.10584164461632398e+02, ap: 2.52675317101618305e+02,
        ap_array: [4.21550094517958414e+01, 4.71550094517958414e+01, 5.21550094517958414e+01, 5.71550094517958414e+01, 6.21550094517958414e+01, 6.71550094517958485e+01, 7.21550094517958485e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [2.57326243879206494e+12, 0.00000000000000000e+00, 3.83455491928148544e+17, 1.02869604819564416e+17, 4.58670286669854900e+15, 2.35920188296469858e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.26300641125029955e+02, 2.35920188296469858e-05] },
    // case 377: alt 101.228 km, lat -71.558, lon -45.304, doy 135, ap 268.465
    Case { doy: 135, sec: 2.60740740740740694e+03, alt: 1.01228000000000009e+02,
        g_lat: -7.15576842982090824e+01, g_long: -4.53042824943651397e+01, lst: -2.29600566423339636e+00,
        f107a: 8.74624487938097417e+01, f107: 1.24407693873397108e+02, ap: 2.68464790785828825e+02,
        ap_array: [4.65028355387523646e+01, 5.15028355387523646e+01, 5.65028355387523646e+01, 6.15028355387523646e+01, 6.65028355387523646e+01, 7.15028355387523646e+01, 7.65028355387523646e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], out: [6.00686339921092615e+07, 2.29395002461063141e+11, 4.20338243193416504e+12, 8.09464509636486328e+11, 2.22439794715273834e+11, 2.59238465716830441e-10, 1.30274557237173867e+05, 1.44699750205412388e+08, 2.42165852147695018e-43, 1.02731846489999998e+03, 1.60948352694574510e+02, 2.59238465716830441e-10] },
    // case 378: alt 345.344 km, lat -45.843, lon -12.577, doy 318, ap 284.254
    Case { doy: 318, sec: 3.14074074074074051e+04, alt: 3.45344000000000051e+02,
        g_lat: -4.58433985839233671e+01, g_long: -1.25770097670924201e+01, lst: 7.88581251758478352e+00,
        f107a: 1.05539371870732822e+02, f107: 1.38231223285161832e+02, ap: 2.84254264470039345e+02,
        ap_array: [5.08506616257088808e+01, 5.58506616257088808e+01, 6.08506616257088808e+01, 6.58506616257088808e+01, 7.08506616257088808e+01, 7.58506616257088808e+01, 8.08506616257088808e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], out: [8.55911238583521335e+05, 1.23555427833988056e+08, 3.21992911315178648e+07, 1.37297713075076207e+06, 1.25254340395395229e+04, 5.03403734092943803e-15, 2.78772701932317141e+04, 7.58555787294936925e+06, 6.73454145543617051e+03, 1.02731846489999998e+03, 1.01741801962031866e+03, 5.03421621035049472e-15] },
    // case 379: alt 426.880 km, lat -20.129, lon +20.150, doy 90, ap 0.875
    Case { doy: 90, sec: 6.02074074074074015e+04, alt: 4.26879999999999995e+02,
        g_lat: -2.01291128696376518e+01, g_long: 2.01502629601802994e+01, lst: 1.80676306994029652e+01,
        f107a: 1.23616294947655888e+02, f107: 1.52054752696926528e+02, ap: 8.74763084997813078e-01,
        ap_array: [5.51984877126653970e+01, 6.01984877126653970e+01, 6.51984877126653970e+01, 7.01984877126653970e+01, 7.51984877126653970e+01, 8.01984877126653970e+01, 8.51984877126653970e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1], out: [2.73156140135463653e+06, 7.35674987541040480e+07, 2.79668500262781186e+06, 4.53373197822536240e+04, 1.18879077746841872e+02, 2.16178755107617511e-15, 8.41278454462767695e+04, 2.45918387775384309e+06, 1.83911471894423835e+03, 1.02731846489999998e+03, 1.02551889863655060e+03, 2.16183639796311016e-15] },
    // case 380: alt 17.748 km, lat +5.585, lon +52.878, doy 272, ap 16.664
    Case { doy: 272, sec: 1.22074074074074088e+04, alt: 1.77480000000000011e+01,
        g_lat: 5.58517284464805641e+00, g_long: 5.28775356874530473e+01, lst: 3.39094650205761372e+00,
        f107a: 1.41693218024578982e+02, f107: 1.65878282108691224e+02, ap: 1.66642367692083404e+01,
        ap_array: [5.95463137996219203e+01, 6.45463137996219132e+01, 6.95463137996219132e+01, 7.45463137996219132e+01, 7.95463137996219132e+01, 8.45463137996219132e+01, 8.95463137996219132e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], out: [1.56096624835778438e+13, 0.00000000000000000e+00, 2.32607864485149798e+18, 6.24017118054311424e+17, 2.78233897103911920e+16, 1.43111501448671532e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.00828612005566441e+02, 1.43111501448671532e-04] },
    // case 381: alt 93.628 km, lat +31.299, lon +85.605, doy 181, ap 32.454
    Case { doy: 181, sec: 4.10074074074074015e+04, alt: 9.36280000000000001e+01,
        g_lat: 3.12994585589337788e+01, g_long: 8.56048084147257669e+01, lst: 1.70979337297059963e+01,
        f107a: 1.59770141101502048e+02, f107: 1.79701811520455919e+02, ap: 3.24537104534188643e+01,
        ap_array: [6.38941398865784507e+01, 6.88941398865784436e+01, 7.38941398865784436e+01, 7.88941398865784436e+01, 8.38941398865784436e+01, 8.88941398865784436e+01, 9.38941398865784436e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1], out: [1.93107677721530199e+08, 3.45034286602722595e+11, 2.48547835521180352e+13, 5.51653335428485254e+12, 2.76562017254991699e+11, 1.47581775630855920e-09, 2.90236256865001358e+07, 2.58795314059626748e+05, 2.74826757594943276e-46, 1.02731846489999998e+03, 1.62180872639631872e+02, 1.47581775630855920e-09] },
    // case 382: alt 300.544 km, lat +57.014, lon +118.332, doy 363, ap 48.243
    Case { doy: 363, sec: 6.98074074074073869e+04, alt: 3.00544000000000040e+02,
        g_lat: 5.70137442732194870e+01, g_long: 1.18332081141998515e+02, lst: 2.72797519115241727e+01,
        f107a: 1.77847064178425114e+02, f107: 1.93525340932220644e+02, ap: 4.82431841376293917e+01,
        ap_array: [6.82419659735349740e+01, 7.32419659735349740e+01, 7.82419659735349740e+01, 8.32419659735349740e+01, 8.82419659735349740e+01, 9.32419659735349740e+01, 9.82419659735349740e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], out: [6.16915463062737696e+06, 8.99351383217296362e+08, 2.09957790365136534e+08, 2.86187970332037657e+06, 4.84508318821984867e+04, 3.41160259176576131e-14, 1.01684516636615532e+05, 1.17918638172263782e+07, 1.11656946388480875e+01, 1.02731846489999998e+03, 1.02245862412725762e+03, 3.41160262142184666e-14] },
    // case 383: alt 906.880 km, lat +82.728, lon +151.059, doy 3, ap 64.033
    Case { doy: 3, sec: 2.18074074074074051e+04, alt: 9.06880000000000109e+02,
        g_lat: 8.27280299875052094e+01, g_long: 1.51059353869271263e+02, lst: 1.61282367600090311e+01,
        f107a: 1.95923987255348209e+02, f107: 2.07348870343985368e+02, ap: 6.40326578218399192e+01,
        ap_array: [7.25897920604914901e+01, 7.75897920604914901e+01, 8.25897920604914901e+01, 8.75897920604914901e+01, 9.25897920604914901e+01, 9.75897920604914901e+01, 1.02589792060491490e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], out: [5.87524880922853714e+05, 5.08407538234553722e+04, 1.53798393848294292e+01, 5.26247339641446946e-02, 7.42429622588357689e-06, 5.30174626671037452e-18, 1.33388786232179191e+04, 1.17859063957234753e+03, 4.56850841373943604e+03, 1.02731846489999998e+03, 1.02731846477992917e+03, 5.42308585017929417e-18] },
    // case 384: alt 6.148 km, lat -67.884, lon -173.238, doy 185, ap 79.822
    Case { doy: 185, sec: 5.06074074074074015e+04, alt: 6.14799999999999969e+00,
        g_lat: -6.78842149104539772e+01, g_long: -1.73238166791885789e+02, lst: 1.40576131687242789e+01,
        f107a: 2.14000910332271275e+02, f107: 2.21172399755750064e+02, ap: 7.98221315060504253e+01,
        ap_array: [7.69376181474480063e+01, 8.19376181474480063e+01, 8.69376181474480063e+01, 9.19376181474480063e+01, 9.69376181474480063e+01, 1.01937618147448006e+02, 1.06937618147448006e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], out: [7.19285628374872812e+13, 0.00000000000000000e+00, 1.07184568626745385e+19, 2.87544042254969856e+18, 1.28208821762881488e+17, 6.59450813593665214e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.27131500053795833e+02, 6.59450813593665214e-04] },
    // case 385: alt 86.028 km, lat -42.170, lon -140.511, doy 94, ap 95.612
    Case { doy: 94, sec: 7.94074074074074015e+04, alt: 8.60280000000000058e+01,
        g_lat: -4.21699291961682690e+01, g_long: -1.40510894064613069e+02, lst: 1.26902202310834067e+01,
        f107a: 2.32077833409194341e+02, f107: 2.34995929167514760e+02, ap: 9.56116051902609598e+01,
        ap_array: [8.12854442344045225e+01, 8.62854442344045225e+01, 9.12854442344045225e+01, 9.62854442344045225e+01, 1.01285444234404522e+02, 1.06285444234404522e+02, 1.11285444234404522e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1], out: [9.31745198137462139e+08, 1.11411062402918289e+11, 1.27592580087000969e+14, 2.96854020079813438e+13, 1.49582243670844336e+12, 7.60967972135343801e-09, 9.86083457338223755e+07, 2.64751769171098713e+05, 1.44570782659688126e-51, 1.02731846489999998e+03, 1.90292589398897036e+02, 7.60967972135343801e-09] },
    // case 386: alt 255.744 km, lat -16.456, lon -107.784, doy 276, ap 111.401
    Case { doy: 276, sec: 5.80740740740740694e+03, alt: 2.55744000000000028e+02,
        g_lat: -1.64556434818825466e+01, g_long: -1.07783621337340335e+02, lst: -5.57240603154285363e+00,
        f107a: 2.50154756486117435e+02, f107: 2.48819458579279456e+02, ap: 1.11401078874471480e+02,
        ap_array: [8.56332703213610529e+01, 9.06332703213610529e+01, 9.56332703213610529e+01, 1.00633270321361053e+02, 1.05633270321361053e+02, 1.10633270321361053e+02, 1.15633270321361053e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], out: [6.72495594807687216e+06, 2.45149602166085148e+09, 9.11333058611757636e+08, 1.15632547155479398e+07, 4.36731240252768097e+05, 1.10597799584405705e-13, 2.60041974440761478e+04, 1.04964232595237449e+08, 2.18074949219018432e-01, 1.51784407487624117e+03, 1.33457574352631605e+03, 1.10597799590197774e-13] },
    // case 387: alt 810.880 km, lat +9.259, lon -75.056, doy 48, ap 127.191
    Case { doy: 48, sec: 3.46074074074074015e+04, alt: 8.10880000000000109e+02,
        g_lat: 9.25864223240316164e+00, g_long: -7.50563486100676300e+01, lst: 4.60941215027532536e+00,
        f107a: 2.68231679563040530e+02, f107: 2.62642987991044151e+02, ap: 1.27190552558682000e+02,
        ap_array: [8.99810964083175691e+01, 9.49810964083175691e+01, 9.99810964083175691e+01, 1.04981096408317569e+02, 1.09981096408317569e+02, 1.14981096408317569e+02, 1.19981096408317569e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1], out: [1.57703360301886685e+06, 1.21941424936383800e+06, 9.19455961746037701e+02, 1.57668251379860069e+00, 1.12938523516099907e-03, 4.34743426813122849e-17, 1.57454883002761999e+04, 2.35042789206857633e+04, 5.11883447032908152e+04, 1.15278216678435888e+03, 1.15278210452760482e+03, 4.48339051166316849e-17] },
    // case 388: alt 64.148 km, lat +34.973, lon -42.329, doy 231, ap 142.980
    Case { doy: 231, sec: 6.34074074074074015e+04, alt: 6.41479999999999961e+01,
        g_lat: 3.49729279466888840e+01, g_long: -4.23290758827948821e+01, lst: 1.76131687242798343e+01,
        f107a: 2.86308602639963624e+02, f107: 2.76466517402808904e+02, ap: 1.42980026242892535e+02,
        ap_array: [9.43289224952740994e+01, 9.93289224952740994e+01, 1.04328922495274099e+02, 1.09328922495274099e+02, 1.14328922495274099e+02, 1.19328922495274099e+02, 1.24328922495274099e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1], out: [1.78381974935815468e+10, 0.00000000000000000e+00, 2.65468406667613050e+15, 6.99241921329935750e+14, 3.17432225453558398e+13, 1.62641314702795711e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.29172849089481730e+02, 1.62641314702795711e-07] },
    // case 389: alt 78.428 km, lat +60.687, lon -9.602, doy 140, ap 158.769
    Case { doy: 140, sec: 1.54074074074074088e+04, alt: 7.84279999999999973e+01,
        g_lat: 6.06872136609745780e+01, g_long: -9.60180315552216257e+00, lst: 3.63971518057835786e+00,
        f107a: 7.07760582612653621e+01, f107: 2.90290046814573600e+02, ap: 1.58769499927103084e+02,
        ap_array: [9.86767485822306156e+01, 1.03676748582230616e+02, 1.08676748582230616e+02, 1.13676748582230616e+02, 1.18676748582230616e+02, 1.23676748582230616e+02, 1.28676748582230630e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], out: [2.79433134719489813e+09, 3.80403963382071555e+08, 4.21814936748688062e+14, 1.13692842381115203e+14, 5.04839042309150098e+12, 2.59805638338043200e-08, 1.96611095966886077e+06, 5.13781837953830109e+04, 6.51824608380252508e-58, 1.02731846489999998e+03, 1.85893369434773462e+02, 2.59805638338043200e-08] },
    // case 390: alt 210.944 km, lat +86.401, lon +23.125, doy 322, ap 174.559
    Case { doy: 322, sec: 4.42074074074074015e+04, alt: 2.10944000000000017e+02,
        g_lat: 8.64014993752603004e+01, g_long: 2.31254695717505285e+01, lst: 1.38215333623965364e+01,
        f107a: 8.88529813381884424e+01, f107: 6.99267250152656175e+01, ap: 1.74558973611313604e+02,
        ap_array: [3.21361058601134220e+00, 8.21361058601134175e+00, 1.32136105860113418e+01, 1.82136105860113418e+01, 2.32136105860113418e+01, 2.82136105860113418e+01, 3.32136105860113418e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], out: [4.17443339289968647e+06, 9.40174490676311851e+08, 2.08318172725947523e+09, 3.14989969043502212e+08, 4.02354026326504210e+06, 1.38907664946782177e-13, 5.58969380032773261e+04, 3.57584060590858618e+06, 2.57699761068305324e-08, 1.29878902256016886e+03, 1.07463824950933599e+03, 1.38907664946782177e-13] },
    // case 391: alt 714.880 km, lat -89.400, lon +55.853, doy 25, ap 190.348
    Case { doy: 25, sec: 7.30074074074073869e+04, alt: 7.14880000000000109e+02,
        g_lat: -8.94002498958767120e+01, g_long: 5.58527422990233049e+01, lst: 2.40033515442147163e+01,
        f107a: 1.06929904415111523e+02, f107: 8.37502544270303275e+01, ap: 1.90348447295524124e+02,
        ap_array: [7.56143667296786415e+00, 1.25614366729678650e+01, 1.75614366729678650e+01, 2.25614366729678650e+01, 2.75614366729678650e+01, 3.25614366729678650e+01, 3.75614366729678650e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], out: [1.69969129452172056e+05, 1.17718948349216953e+06, 2.30072155287779351e+04, 4.86436027922224127e+02, 4.71141879258525131e-01, 3.43203876595874735e-17, 5.46967306247624729e+04, 3.18243219550050271e+04, 1.39256935215298308e+04, 1.19209812280265692e+03, 1.19209534590499902e+03, 3.46902540795193043e-17] },
    // case 392: alt 52.548 km, lat -63.686, lon +88.580, doy 208, ap 206.138
    Case { doy: 208, sec: 2.50074074074074051e+04, alt: 5.25480000000000089e+01,
        g_lat: -6.36859641815910109e+01, g_long: 8.85800150262959960e+01, lst: 6.94650205761316819e+00,
        f107a: 1.25006827492034603e+02, f107: 9.75737838387950376e+01, ap: 2.06137920979734645e+02,
        ap_array: [1.19092627599243848e+01, 1.69092627599243848e+01, 2.19092627599243848e+01, 2.69092627599243848e+01, 3.19092627599243848e+01, 3.69092627599243883e+01, 4.19092627599243883e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1], out: [7.86966303778050995e+10, 0.00000000000000000e+00, 1.17270025239922260e+16, 3.14600296711139450e+15, 1.40272540690733016e+14, 7.21501374175606344e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.61244590434016686e+02, 7.21501374175606344e-07] },
    // case 393: alt 116.428 km, lat -37.972, lon +121.307, doy 117, ap 221.927
    Case { doy: 117, sec: 5.38074074074074015e+04, alt: 1.16427999999999997e+02,
        g_lat: -3.79716784673052956e+01, g_long: 1.21307287753568801e+02, lst: 2.30336545745177546e+01,
        f107a: 1.43083750568957669e+02, f107: 1.11397313250559733e+02, ap: 2.21927394663945165e+02,
        ap_array: [1.62570888468809081e+01, 2.12570888468809081e+01, 2.62570888468809081e+01, 3.12570888468809081e+01, 3.62570888468809045e+01, 4.12570888468809045e+01, 4.62570888468809045e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1], out: [3.39943894309932515e+07, 1.16515662559146423e+11, 5.38671020117716858e+11, 7.66162402087827148e+10, 2.78514679197773695e+09, 3.23871341146228479e-11, 4.43727275056926906e+06, 1.18702598923801887e+06, 1.67780842581968862e-34, 1.02731846489999998e+03, 3.21984426735127499e+02, 3.23871341146228479e-11] },
    // case 394: alt 166.144 km, lat -12.257, lon +154.035, doy 299, ap 237.717
    Case { doy: 299, sec: 8.26074074074073869e+04, alt: 1.66144000000000005e+02,
        g_lat: -1.22573927530195732e+01, g_long: 1.54034560480841492e+02, lst: 3.32154727563359273e+01,
        f107a: 1.61160673645880763e+02, f107: 1.25220842662324443e+02, ap: 2.37716868348155714e+02,
        ap_array: [2.06049149338374278e+01, 2.56049149338374278e+01, 3.06049149338374278e+01, 3.56049149338374278e+01, 4.06049149338374278e+01, 4.56049149338374278e+01, 5.06049149338374278e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], out: [2.24403316227259561e+07, 1.98437042643167191e+10, 1.94614462662444649e+10, 8.78000775464935899e+08, 5.12554938195310012e+07, 1.48255272060799319e-12, 1.91173799827192270e+05, 3.20062719406791814e+07, 1.96998014288192299e-14, 1.16867421691983213e+03, 8.58293567193722993e+02, 1.48255272060799319e-12] },
    // case 395: alt 618.880 km, lat +13.457, lon -170.263, doy 71, ap 253.506
    Case { doy: 71, sec: 9.00740740740740694e+03, alt: 6.18879999999999995e+02,
        g_lat: 1.34568929612661350e+01, g_long: -1.70262960180315559e+02, lst: -8.84880639885231446e+00,
        f107a: 1.79237596722803829e+02, f107: 1.39044372074089154e+02, ap: 2.53506342032366234e+02,
        ap_array: [2.49527410207939511e+01, 2.99527410207939511e+01, 3.49527410207939511e+01, 3.99527410207939511e+01, 4.49527410207939511e+01, 4.99527410207939511e+01, 5.49527410207939511e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0], out: [3.98540543021041574e+06, 3.02987213512353897e+07, 3.40788329351208813e+05, 2.41521810595291345e+03, 5.60833352869663848e+00, 8.78883372671784993e-16, 3.04379646555378349e+04, 1.36261635876372783e+06, 8.94196708048623521e+05, 1.31680168665392853e+03, 1.31667948152446797e+03, 9.02633237237556474e-16] },
    // case 396: alt 40.948 km, lat +39.171, lon -137.536, doy 254, ap 269.296
    Case { doy: 254, sec: 3.78074074074074015e+04, alt: 4.09480000000000075e+01,
        g_lat: 3.91711786755518290e+01, g_long: -1.37535687453042812e+02, lst: 1.05020576131687235e+01,
        f107a: 1.97314519799726924e+02, f107: 1.52867901485853849e+02, ap: 2.69295815716576726e+02,
        ap_array: [2.93005671077504743e+01, 3.43005671077504743e+01, 3.93005671077504743e+01, 4.43005671077504743e+01, 4.93005671077504743e+01, 5.43005671077504743e+01, 5.93005671077504743e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], out: [3.56039434227406067e+11, 0.00000000000000000e+00, 5.30553255429232800e+16, 1.42331521834000280e+16, 6.34621276481676750e+14, 3.26421779207751090e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.57470927329900974e+02, 3.26421779207751090e-06] },
    // case 397: alt 108.828 km, lat +64.885, lon -104.808, doy 162, ap 285.085
    Case { doy: 162, sec: 6.66074074074074015e+04, alt: 1.08828000000000003e+02,
        g_lat: 6.48854643898375514e+01, g_long: -1.04808414725770092e+02, lst: 1.15148299647840506e+01,
        f107a: 2.15391442876649990e+02, f107: 1.66691430897618545e+02, ap: 2.85085289400787303e+02,
        ap_array: [3.36483931947069905e+01, 3.86483931947069905e+01, 4.36483931947069905e+01, 4.86483931947069905e+01, 5.36483931947069905e+01, 5.86483931947069905e+01, 6.36483931947069905e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1], out: [1.33470800889444593e+07, 5.05118700128945389e+10, 2.71609398114597656e+11, 2.83764279940996284e+10, 1.72117561802396107e+09, 1.55877366983993281e-11, 2.12969842176612420e+06, 1.08885362123578263e+05, 5.92195858910118853e-39, 1.02731846489999998e+03, 8.42980971728484747e+02, 1.55877366983993281e-11] },
    // case 398: alt 390.144 km, lat -85.727, lon -72.081, doy 345, ap 1.706
    Case { doy: 345, sec: 1.86074074074074088e+04, alt: 3.90144000000000005e+02,
        g_lat: -8.57267805081216210e+01, g_long: -7.20811419984973725e+01, lst: 3.63314813268900139e-01,
        f107a: 2.33468365953573056e+02, f107: 1.80514960309383270e+02, ap: 1.70578801574573546e+00,
        ap_array: [3.79962192816635138e+01, 4.29962192816635138e+01, 4.79962192816635138e+01, 5.29962192816635138e+01, 5.79962192816635138e+01, 6.29962192816635138e+01, 6.79962192816635138e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], out: [9.07802302646279801e+05, 1.75539036580034494e+08, 5.13430209610505253e+07, 4.16499616269856691e+05, 9.67166714829756529e+03, 7.18588156464676667e-15, 1.62856687150011840e+04, 4.66091389831679221e+06, 2.16088985836322245e+05, 1.43122233784175069e+03, 1.41097204473953911e+03, 7.19162088811057926e-15] },
    // case 399: alt 407.680 km, lat -60.012, lon -39.354, doy 14, ap 17.495
    Case { doy: 14, sec: 4.74074074074074015e+04, alt: 4.07680000000000007e+02,
        g_lat: -6.00124947938359128e+01, g_long: -3.93538692712246245e+01, lst: 1.05451329950870818e+01,
        f107a: 2.51545289030496150e+02, f107: 1.94338489721147965e+02, ap: 1.74952616999562629e+01,
        ap_array: [4.23440453686200300e+01, 4.73440453686200300e+01, 5.23440453686200300e+01, 5.73440453686200300e+01, 6.23440453686200300e+01, 6.73440453686200300e+01, 7.23440453686200300e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0], out: [1.54042021461779950e+06, 2.05640397393512011e+08, 3.88558727684371695e+07, 3.10393537377863133e+05, 5.67347536079011115e+03, 7.52111818785726737e-15, 1.92066945492449668e+04, 9.73163382104748301e+06, 1.16371835163389449e+05, 1.43884333200499418e+03, 1.42382726466153736e+03, 7.52420902379920640e-15] },
    // case 400: alt 15.428 km, lat -34.298, lon -6.627, doy 197, ap 33.285
    Case { doy: 197, sec: 7.62074074074073869e+04, alt: 1.54280000000000008e+01,
        g_lat: -3.42982090795501833e+01, g_long: -6.62659654395190501e+00, lst: 2.11687242798353878e+01,
        f107a: 2.69622212107419273e+02, f107: 2.08162019132912690e+02, ap: 3.32847353841667868e+01,
        ap_array: [4.66918714555765604e+01, 5.16918714555765604e+01, 5.66918714555765604e+01, 6.16918714555765604e+01, 6.66918714555765604e+01, 7.16918714555765604e+01, 7.66918714555765604e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1], out: [2.12794389631060117e+13, 0.00000000000000000e+00, 3.17096212673245389e+18, 8.50674009738515456e+17, 3.79294634789102480e+16, 1.95092780718308776e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.11056861724958935e+02, 1.95092780718308776e-04] },
    // case 401: alt 92.108 km, lat -8.584, lon +26.101, doy 105, ap 49.074
    Case { doy: 105, sec: 2.82074074074074051e+04, alt: 9.21080000000000041e+01,
        g_lat: -8.58392336526446798e+00, g_long: 2.61006761833208145e+01, lst: 9.57543602539011118e+00,
        f107a: 2.87699135184342310e+02, f107: 2.21985548544677386e+02, ap: 4.90742090683773142e+01,
        ap_array: [5.10396975425330766e+01, 5.60396975425330766e+01, 6.10396975425330766e+01, 6.60396975425330766e+01, 7.10396975425330766e+01, 7.60396975425330766e+01, 8.10396975425330766e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1], out: [3.55922029764367282e+08, 6.06502244149109009e+11, 4.07504206287558438e+13, 8.22538594953732715e+12, 4.60014911119798767e+11, 2.37766818288246416e-09, 4.18868596172384024e+07, 3.38255174677807430e+05, 1.39865966463280677e-47, 1.02731846489999998e+03, 1.78493179844822265e+02, 2.37766818288246416e-09] },
    // case 402: alt 291.584 km, lat +17.130, lon +58.828, doy 288, ap 64.864
    Case { doy: 288, sec: 5.70074074074074015e+04, alt: 2.91584000000000060e+02,
        g_lat: 1.71303623490212402e+01, g_long: 5.88279489105935625e+01, lst: 1.97572542072082946e+01,
        f107a: 7.21665908056440628e+01, f107: 2.35809077956442081e+02, ap: 6.48636827525878346e+01,
        ap_array: [5.53875236294895998e+01, 6.03875236294895998e+01, 6.53875236294896069e+01, 7.03875236294896069e+01, 7.53875236294896069e+01, 8.03875236294896069e+01, 8.53875236294896069e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0], out: [6.04929790359771624e+06, 8.43385731222242951e+08, 1.40543956091344029e+08, 6.25851176047829073e+06, 2.96366340305763188e+04, 2.98242847060579416e-14, 1.46106384255027340e+05, 2.22309257218378186e+07, 1.37113148017272124e+01, 1.02469629186127736e+03, 9.91440665306745586e+02, 2.98242850702304667e-14] },
    // case 403: alt 887.680 km, lat +42.845, lon +91.555, doy 60, ap 80.653
    Case { doy: 60, sec: 8.58074074074074015e+04, alt: 8.87680000000000064e+02,
        g_lat: 4.28446480633069484e+01, g_long: 9.15552216378662820e+01, lst: 2.99390723890264745e+01,
        f107a: 9.02435138825671430e+01, f107: 2.49632607368206777e+02, ap: 8.06531564367983691e+01,
        ap_array: [5.97353497164461160e+01, 6.47353497164461089e+01, 6.97353497164461089e+01, 7.47353497164461089e+01, 7.97353497164461089e+01, 8.47353497164461089e+01, 8.97353497164461089e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1], out: [4.81666670720713271e+05, 2.88880768336103974e+04, 2.67262607221818183e+00, 2.02537224877697326e-02, 2.48539114903527282e-07, 4.09058419347174671e-18, 5.61616330463669292e+04, 1.36387989563128394e+03, 3.60653593248326570e+03, 9.29961489832601956e+02, 9.29961307495240590e+02, 4.18637378783850259e-18] },
    // case 404: alt 3.828 km, lat +68.559, lon +124.282, doy 242, ap 96.443
    Case { doy: 242, sec: 8.29629629629629562e+02, alt: 3.82799999999999985e+00,
        g_lat: 6.85589337775926708e+01, g_long: 1.24282494365139030e+02, lst: 2.30452674897119320e-01,
        f107a: 1.08320436959490223e+02, f107: 2.63456136779971530e+02, ap: 9.64426301210088894e+01,
        ap_array: [6.40831758034026535e+01, 6.90831758034026535e+01, 7.40831758034026535e+01, 7.90831758034026535e+01, 8.40831758034026535e+01, 8.90831758034026535e+01, 9.40831758034026535e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0], out: [9.02717526509503438e+13, 0.00000000000000000e+00, 1.34518729213780849e+19, 3.60873394861809869e+18, 1.60904577949063968e+17, 8.27623664116502338e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.57477231229721781e+02, 8.27623664116502338e-04] },
    // case 405: alt 84.508 km, lat -82.053, lon +157.010, doy 151, ap 112.232
    Case { doy: 151, sec: 2.96296296296296314e+04, alt: 8.45079999999999956e+01,
        g_lat: -8.20533111203665158e+01, g_long: 1.57009767092411778e+02, lst: 1.86977704810579048e+01,
        f107a: 1.26397360036413289e+02, f107: 2.77279666191736169e+02, ap: 1.12232103805219410e+02,
        ap_array: [6.84310018903591697e+01, 7.34310018903591697e+01, 7.84310018903591697e+01, 8.34310018903591697e+01, 8.84310018903591697e+01, 9.34310018903591697e+01, 9.84310018903591697e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], out: [6.13541727381495237e+08, 1.20586401866866589e+10, 8.79628296524427188e+13, 2.32520745919406562e+13, 1.04505992158677808e+12, 5.39337896763750807e-09, 6.73863993804557770e+07, 4.35830179898446222e+04, 2.62943694688251158e-54, 1.02731846489999998e+03, 2.15435677084111006e+02, 5.39337896763750807e-09] },
    // case 406: alt 246.784 km, lat -56.339, lon -167.288, doy 333, ap 128.022
    Case { doy: 333, sec: 5.84296296296296205e+04, alt: 2.46783999999999992e+02,
        g_lat: -5.63390254060808076e+01, g_long: -1.67287753568745302e+02, lst: 5.07793577031409527e+00,
        f107a: 1.44474283113336355e+02, f107: 2.91103195603500922e+02, ap: 1.28021577489429944e+02,
        ap_array: [7.27788279773156859e+01, 7.77788279773156859e+01, 8.27788279773156859e+01, 8.77788279773156859e+01, 9.27788279773156859e+01, 9.77788279773156859e+01, 1.02778827977315686e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.38061500258776173e+06, 1.28315269745464921e+09, 1.51231457979161406e+09, 1.51013371204633206e+08, 2.64639917857506266e+06, 1.12715667807259441e-13, 3.44931010372227829e+04, 5.27946506615015958e+06, 1.04739737489434968e-02, 1.41828513078209994e+03, 1.25581776549671531e+03, 1.12715667807537637e-13] },
    // case 407: alt 791.680 km, lat -30.625, lon -134.560, doy 37, ap 143.811
    Case { doy: 37, sec: 1.04296296296296296e+04, alt: 7.91680000000000064e+02,
        g_lat: -3.06247396917950923e+01, g_long: -1.34560480841472582e+02, lst: -6.07357938120105256e+00,
        f107a: 1.62551206190259450e+02, f107: 7.07398738041929533e+01, ap: 1.43811051173640465e+02,
        ap_array: [7.71266540642722020e+01, 8.21266540642722020e+01, 8.71266540642722020e+01, 9.21266540642722020e+01, 9.71266540642722020e+01, 1.02126654064272202e+02, 1.07126654064272202e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [7.08478932467719656e+05, 3.62241613973544387e+05, 2.44133472967505213e+02, 9.52455509000260769e-01, 1.80307987220082558e-04, 1.47676856046377296e-17, 4.97460165564010531e+04, 1.49858824526027020e+04, 5.17202233328559087e+04, 1.02731846489999998e+03, 1.02731801204038993e+03, 1.61413747363583815e-17] },
    // case 408: alt 61.828 km, lat -4.910, lon -101.833, doy 219, ap 159.601
    Case { doy: 219, sec: 3.92296296296296277e+04, alt: 6.18280000000000101e+01,
        g_lat: -4.91045397750937695e+00, g_long: -1.01833208114199849e+02, lst: 1.08971193415637870e+01,
        f107a: 1.80628129267182544e+02, f107: 8.45634032159576634e+01, ap: 1.59600524857850985e+02,
        ap_array: [8.14744801512287182e+01, 8.64744801512287182e+01, 9.14744801512287182e+01, 9.64744801512287182e+01, 1.01474480151228718e+02, 1.06474480151228718e+02, 1.11474480151228718e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [2.71484427965310707e+10, 0.00000000000000000e+00, 4.04553353388773050e+15, 1.08529528113606150e+15, 4.83906493655980859e+13, 2.48900603372585205e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.38825771168054530e+02, 2.48900603372585205e-07] },
    // case 409: alt 76.908 km, lat +20.804, lon -69.106, doy 128, ap 175.390
    Case { doy: 128, sec: 6.80296296296296205e+04, alt: 7.69080000000000013e+01,
        g_lat: 2.08038317367763455e+01, g_long: -6.91059353869271433e+01, lst: 1.42900569824353063e+01,
        f107a: 1.98705052344105638e+02, f107: 9.83869326277223735e+01, ap: 1.75389998542061534e+02,
        ap_array: [8.58223062381852486e+01, 9.08223062381852486e+01, 9.58223062381852486e+01, 1.00822306238185249e+02, 1.05822306238185249e+02, 1.10822306238185249e+02, 1.15822306238185249e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [3.46482080235721254e+09, 3.01401632603730917e+08, 5.08228465697034500e+14, 1.25347938254080953e+14, 6.05351294829792969e+12, 3.06829258386593815e-08, 6.51740048187704906e+03, 6.83799431735768012e+04, 1.95975117976731297e-59, 1.02731846489999998e+03, 1.92833956187031418e+02, 3.06829258386593815e-08] },
    // case 410: alt 201.984 km, lat +46.518, lon -36.379, doy 311, ap 191.179
    Case { doy: 311, sec: 2.00296296296296277e+04, alt: 2.01984000000000037e+02,
        g_lat: 4.65181174510620394e+01, g_long: -3.63786626596543954e+01, lst: 3.13854183092015937e+00,
        f107a: 2.16781975421028704e+02, f107: 1.12210462039487084e+02, ap: 1.91179472226272054e+02,
        ap_array: [9.01701323251417648e+01, 9.51701323251417648e+01, 1.00170132325141765e+02, 1.05170132325141765e+02, 1.10170132325141765e+02, 1.15170132325141765e+02, 1.20170132325141765e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], out: [5.93830306096686050e+06, 3.83500771227134752e+09, 6.37302558522990799e+09, 2.06514237215149546e+09, 4.42705527561192960e+07, 5.11146970267512882e-13, 4.75332570052101655e+04, 1.68459328259297870e+07, 9.78100937375666909e-08, 1.02731846489999998e+03, 9.80342475366321196e+02, 5.11146970267512882e-13] },
    // case 411: alt 695.680 km, lat +72.232, lon -3.651, doy 82, ap 206.969
    Case { doy: 82, sec: 4.88296296296296277e+04, alt: 6.95680000000000064e+02,
        g_lat: 7.22324031653477618e+01, g_long: -3.65138993238164744e+00, lst: 1.33203600127383410e+01,
        f107a: 2.34858898497951770e+02, f107: 1.26033991451251779e+02, ap: 2.06968945910482546e+02,
        ap_array: [9.45179584120982952e+01, 9.95179584120982952e+01, 1.04517958412098295e+02, 1.09517958412098295e+02, 1.14517958412098295e+02, 1.19517958412098295e+02, 1.24517958412098295e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], out: [6.79983236335520982e+05, 9.07826728275065869e+06, 1.94728932006772520e+05, 2.94380582652726589e+04, 4.77598599697748494e+00, 2.71700531172701946e-16, 1.47683846507337239e+04, 6.63815851758178906e+05, 4.78929435743277281e+04, 1.41021160146873967e+03, 1.41010099338458281e+03, 2.72972567754036096e-16] },
    // case 412: alt 50.228 km, lat -78.380, lon +29.076, doy 265, ap 222.758
    Case { doy: 265, sec: 7.76296296296296205e+04, alt: 5.02280000000000086e+01,
        g_lat: -7.83798417326114105e+01, g_long: 2.90758827948910437e+01, lst: 2.15637860082304513e+01,
        f107a: 2.52935821574874836e+02, f107: 1.39857520863016475e+02, ap: 2.22758419594693095e+02,
        ap_array: [9.88657844990548256e+01, 1.03865784499054826e+02, 1.08865784499054826e+02, 1.13865784499054826e+02, 1.18865784499054826e+02, 1.23865784499054826e+02, 1.28865784499054826e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [7.43346826396228333e+10, 0.00000000000000000e+00, 1.10770055433132120e+16, 2.97162827710463550e+15, 1.32497601806340984e+14, 6.81510446075230127e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.69692216240050186e+02, 6.81510446075230127e-07] },
    // case 413: alt 114.908 km, lat -52.666, lon +61.803, doy 174, ap 238.548
    Case { doy: 174, sec: 4.02962962962962911e+03, alt: 1.14908000000000001e+02,
        g_lat: -5.26655560183257023e+01, g_long: 6.18031555221637916e+01, lst: 5.23955193193026147e+00,
        f107a: 2.71012744651797902e+02, f107: 1.53681050274781171e+02, ap: 2.38547893278903643e+02,
        ap_array: [3.40264650283553882e+00, 8.40264650283553927e+00, 1.34026465028355393e+01, 1.84026465028355375e+01, 2.34026465028355375e+01, 2.84026465028355375e+01, 3.34026465028355375e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], out: [8.27496899416154623e+06, 2.48619219260753479e+10, 3.22336644809762756e+11, 4.99283992809823074e+10, 3.92849656484628105e+09, 1.85556521138001183e-11, 1.28768508650906198e+06, 2.75026800645643321e+05, 2.41738839502830882e-34, 1.02731846489999998e+03, 6.04856436959527286e+02, 1.85556521138001183e-11] },
    // case 414: alt 157.184 km, lat -26.951, lon +94.530, doy 356, ap 254.337
    Case { doy: 356, sec: 3.28296296296296277e+04, alt: 1.57183999999999997e+02,
        g_lat: -2.69512703040399870e+01, g_long: 9.45304282494365111e+01, lst: 1.54213701137484414e+01,
        f107a: 2.89089667728720997e+02, f107: 1.67504579686545895e+02, ap: 2.54337366963114135e+02,
        ap_array: [7.75047258979205989e+00, 1.27504725897920608e+01, 1.77504725897920608e+01, 2.27504725897920608e+01, 2.77504725897920608e+01, 3.27504725897920608e+01, 3.77504725897920608e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], out: [1.45811786821502000e+07, 1.75525651544525375e+10, 3.30974148923450775e+10, 1.52649888226139575e+08, 6.18019110672364086e+07, 2.01750868385307337e-12, 3.31983197130347544e+05, 2.73205713692236803e+07, 5.13663959632934150e-19, 1.02731846489999998e+03, 8.89254811907848193e+02, 2.01750868385307337e-12] },
    // case 415: alt 599.680 km, lat -1.237, lon +127.258, doy 8, ap 270.127
    Case { doy: 8, sec: 6.16296296296296205e+04, alt: 5.99680000000000064e+02,
        g_lat: -1.23698458975427172e+00, g_long: 1.27257700976709259e+02, lst: 2.56031882955666248e+01,
        f107a: 7.35571233500227635e+01, f107: 1.81328109098310620e+02, ap: 2.70126840647324627e+02,
        ap_array: [1.20982986767485805e+01, 1.70982986767485805e+01, 2.20982986767485805e+01, 2.70982986767485805e+01, 3.20982986767485841e+01, 3.70982986767485841e+01, 4.20982986767485841e+01],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1], out: [1.90153923305341578e+06, 9.05412045790665597e+06, 7.09996032033772499e+04, 3.47821971152579545e+03, 5.43485336304370237e-01, 2.61449782204596665e-16, 7.30719843155468698e+04, 2.03957063931234152e+05, 9.23275910370611928e+03, 1.07494857444841819e+03, 1.07493327250317111e+03, 2.61695004286391105e-16] },
    // case 416: alt 38.628 km, lat +24.477, lon +159.985, doy 191, ap 285.916
    Case { doy: 191, sec: 1.36296296296296296e+04, alt: 3.86280000000000072e+01,
        g_lat: 2.44773011245314365e+01, g_long: 1.59984973703982007e+02, lst: 3.78600823045267454e+00,
        f107a: 9.16340464269458437e+01, f107: 1.95151638510075315e+02, ap: 2.85916314331535204e+02,
        ap_array: [1.64461247637051038e+01, 2.14461247637051038e+01, 2.64461247637051038e+01, 3.14461247637051038e+01, 3.64461247637051002e+01, 4.14461247637051002e+01, 4.64461247637051002e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [5.69333851881625610e+11, 0.00000000000000000e+00, 8.48394586395536800e+16, 2.27598815692330680e+16, 1.01480718451702138e+15, 5.21972992395282270e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.51538878868937473e+02, 5.21972992395282270e-06] },
    // case 417: alt 107.308 km, lat +50.192, lon -164.313, doy 100, ap 2.537
    Case { doy: 100, sec: 4.24296296296296277e+04, alt: 1.07308000000000007e+02,
        g_lat: 5.01915868388171305e+01, g_long: -1.64312546957175044e+02, lst: 8.31838433307671821e-01,
        f107a: 1.09710969503868910e+02, f107: 2.08975167921840040e+02, ap: 2.53681294649365796e+00,
        ap_array: [2.07939508506616235e+01, 2.57939508506616235e+01, 3.07939508506616235e+01, 3.57939508506616235e+01, 4.07939508506616235e+01, 4.57939508506616235e+01, 5.07939508506616235e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], out: [4.06006169117024839e+07, 3.43452434533841248e+11, 2.87471405997180127e+12, 2.75340184790938965e+11, 2.52783575922358704e+10, 1.59091746433272658e-10, 4.48193709801731748e+05, 2.07041188672583413e+09, 1.08406728098356712e-38, 1.02731846489999998e+03, 2.10324164430712955e+02, 1.59091746433272658e-10] },
    // case 418: alt 381.184 km, lat +75.906, lon -131.585, doy 282, ap 18.326
    Case { doy: 282, sec: 7.12296296296296059e+04, alt: 3.81184000000000026e+02,
        g_lat: 7.59058725531028529e+01, g_long: -1.31585274229902325e+02, lst: 1.10136566151258481e+01,
        f107a: 1.27787892580792004e+02, f107: 2.22798697333604736e+02, ap: 1.83262866307041854e+01,
        ap_array: [2.51417769376181504e+01, 3.01417769376181504e+01, 3.51417769376181468e+01, 4.01417769376181468e+01, 4.51417769376181468e+01, 5.01417769376181468e+01, 5.51417769376181468e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1], out: [4.14634188603895716e+06, 2.70753518237783372e+08, 3.12856220338523127e+07, 1.34972381454611872e+06, 3.41729890951719335e+03, 8.83548663341092741e-15, 8.13243239399985032e+04, 3.89528961723859981e+06, 1.57441710081260953e+03, 1.21472688085957407e+03, 1.20951360244811326e+03, 8.83552844992912450e-15] },
    // case 419: alt 503.680 km, lat -74.706, lon -98.858, doy 54, ap 34.116
    Case { doy: 54, sec: 2.32296296296296314e+04, alt: 5.03680000000000007e+02,
        g_lat: -7.47063723448563053e+01, g_long: -9.88580015026295911e+01, lst: -1.37858536389297015e-01,
        f107a: 1.45864815657715070e+02, f107: 2.36622226745369431e+02, ap: 3.41157603149147093e+01,
        ap_array: [2.94896030245746665e+01, 3.44896030245746630e+01, 3.94896030245746630e+01, 4.44896030245746630e+01, 4.94896030245746630e+01, 5.44896030245746630e+01, 5.94896030245746630e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], out: [7.45385999846385675e+05, 1.40246993910521474e+07, 6.02504738392437925e+05, 9.15495342313964284e+03, 2.47501562323370621e+01, 4.14112828061753345e-16, 3.67178485399820565e+04, 3.49144698414820596e+05, 5.06613410540557961e+04, 1.02731846489999998e+03, 1.02726227779354326e+03, 4.15458393280149064e-16] },
    // case 420: alt 27.028 km, lat -48.992, lon -66.131, doy 236, ap 49.905
    Case { doy: 236, sec: 5.20296296296296205e+04, alt: 2.70280000000000022e+01,
        g_lat: -4.89920866305706042e+01, g_long: -6.61307287753569000e+01, lst: 1.44526748971193388e+01,
        f107a: 1.63941738734638136e+02, f107: 2.50445756157134127e+02, ap: 4.99052339991252367e+01,
        ap_array: [3.38374291115311934e+01, 3.88374291115311934e+01, 4.38374291115311934e+01, 4.88374291115311934e+01, 5.38374291115311934e+01, 5.88374291115311934e+01, 6.38374291115311934e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], out: [3.29069086490157129e+12, 0.00000000000000000e+00, 4.90363308989433536e+17, 1.31549764902595344e+17, 5.86548072047710900e+15, 3.01694998834817646e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.19718899174716256e+02, 3.01694998834817646e-05] },
    // case 421: alt 99.708 km, lat -23.278, lon -33.403, doy 145, ap 65.695
    Case { doy: 145, sec: 8.08296296296296205e+04, alt: 9.97079999999999984e+01,
        g_lat: -2.32778009162848889e+01, g_long: -3.34034560480841662e+01, lst: 2.02257778272470645e+01,
        f107a: 1.82018661811561230e+02, f107: 2.64269285568898852e+02, ap: 6.56947076833357642e+01,
        ap_array: [3.81852551984877095e+01, 4.31852551984877095e+01, 4.81852551984877095e+01, 5.31852551984877095e+01, 5.81852551984877095e+01, 6.31852551984877095e+01, 6.81852551984877095e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1], out: [1.10664903300586641e+08, 4.01389308006977417e+11, 9.25979533594995508e+12, 1.72735161192194849e+12, 9.38953899836039429e+10, 5.39048529468829776e-10, 1.71778422831082605e+07, 3.17677672327703214e+05, 3.57095048013880918e-43, 1.02731846489999998e+03, 2.08082977321855765e+02, 5.39048529468829776e-10] },
    // case 422: alt 336.384 km, lat +2.436, lon -0.676, doy 328, ap 81.484
    Case { doy: 328, sec: 7.22962962962962956e+03, alt: 3.36384000000000015e+02,
        g_lat: 2.43648479800083351e+00, g_long: -6.76183320811418298e-01, lst: 1.96315156462080265e+00,
        f107a: 2.00095584888484296e+02, f107: 2.78092814980663547e+02, ap: 8.14841813675462703e+01,
        ap_array: [4.25330812854442328e+01, 4.75330812854442328e+01, 5.25330812854442328e+01, 5.75330812854442328e+01, 6.25330812854442328e+01, 6.75330812854442399e+01, 7.25330812854442399e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1], out: [6.76958529736342002e+06, 6.14141945837264776e+08, 5.84669322721573487e+07, 4.80658208208901063e+06, 7.65400752944462056e+03, 1.95261331526267872e-14, 3.93426267306125810e+04, 8.43936479104015417e+06, 1.07251045303845717e+03, 1.02731846489999998e+03, 1.02410841678799966e+03, 1.95261616385044174e-14] },
    // case 423: alt 983.680 km, lat +28.151, lon +32.051, doy 31, ap 97.274
    Case { doy: 31, sec: 3.60296296296296277e+04, alt: 9.83680000000000064e+02,
        g_lat: 2.81507705122865417e+01, g_long: 3.20510894064613012e+01, lst: 1.21449697464389832e+01,
        f107a: 2.18172507965407362e+02, f107: 2.91916344392428300e+02, ap: 9.72736550517568048e+01,
        ap_array: [4.68809073724007561e+01, 5.18809073724007561e+01, 5.68809073724007561e+01, 6.18809073724007561e+01, 6.68809073724007561e+01, 7.18809073724007561e+01, 7.68809073724007561e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1], out: [4.48075171055872634e+05, 2.76695392950155765e+04, 2.80762514003598973e+00, 1.15992010251439147e-02, 2.78380061073186021e-07, 3.82489563189957968e-18, 5.10522167856292581e+04, 1.28639147866932217e+03, 2.57542836624974188e+03, 1.02731846489999998e+03, 1.02731846486561949e+03, 3.89329900930717319e-18] },
    // case 424: alt 1.508 km, lat +53.865, lon +64.778, doy 214, ap 113.063
    Case { doy: 214, sec: 6.48296296296296205e+04, alt: 1.50800000000000001e+00,
        g_lat: 5.38650562265722499e+01, g_long: 6.47783621337340207e+01, lst: 1.80082304526748942e+01,
        f107a: 2.36249431042330428e+02, f107: 7.15530225931202892e+01, ap: 1.13063128735967325e+02,
        ap_array: [5.12287334593572794e+01, 5.62287334593572794e+01, 6.12287334593572794e+01, 6.62287334593572723e+01, 7.12287334593572723e+01, 7.62287334593572723e+01, 8.12287334593572723e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1], out: [1.11552789287761703e+14, 0.00000000000000000e+00, 1.66230731259479921e+19, 4.45947182749806797e+18, 1.98836889196026912e+17, 1.02273109252389699e-03, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.82615660193940130e+02, 1.02273109252389699e-03] },
    // case 425: alt 82.988 km, lat +79.579, lon +97.506, doy 122, ap 128.853
    Case { doy: 122, sec: 1.68296296296296277e+04, alt: 8.29879999999999995e+01,
        g_lat: 7.95793419408579723e+01, g_long: 9.75056348610067971e+01, lst: 1.11752727767420161e+01,
        f107a: 2.54326354119253523e+02, f107: 8.53765520048849993e+01, ap: 1.28852602420177845e+02,
        ap_array: [5.55765595463137956e+01, 6.05765595463137956e+01, 6.55765595463138027e+01, 7.05765595463138027e+01, 7.55765595463138027e+01, 8.05765595463138027e+01, 8.55765595463138027e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1], out: [1.26027579975627685e+09, 2.57632541580034828e+10, 1.81290481083825812e+14, 3.94474874384245781e+13, 2.14379549129352344e+12, 1.06648729390421861e-08, 1.09593612002167836e+08, 1.17375919132937008e+05, 6.92067641100004341e-55, 1.02731846489999998e+03, 1.76206336082092605e+02, 1.06648729390421861e-08] },
    // case 426: alt 0.000 km, lat -71.033, lon +130.233, doy 305, ap 144.642
    Case { doy: 305, sec: 4.56296296296296205e+04, alt: 0.00000000000000000e+00,
        g_lat: -7.10329029571012143e+01, g_long: 1.30232907588279488e+02, lst: 2.13570909585601925e+01,
        f107a: 2.72403277196176646e+02, f107: 9.92000814166497094e+01, ap: 1.44642076104388366e+02,
        ap_array: [5.99243856332703189e+01, 6.49243856332703189e+01, 6.99243856332703189e+01, 7.49243856332703189e+01, 7.99243856332703189e+01, 8.49243856332703189e+01, 8.99243856332703189e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.44053115471521719e+14, 0.00000000000000000e+00, 2.14661192050214748e+19, 5.75871579913111142e+18, 2.56766984870864384e+17, 1.32069848820735511e-03, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.62889141710242768e+02, 1.32069848820735511e-03] },
    // case 427: alt 10.000 km, lat -45.319, lon +162.960, doy 77, ap 160.432
    Case { doy: 77, sec: 7.44296296296296205e+04, alt: 1.00000000000000000e+01,
        g_lat: -4.53186172428154990e+01, g_long: 1.62960180315552236e+02, lst: 3.15389091403783794e+01,
        f107a: 2.90480200273099683e+02, f107: 1.13023610828414405e+02, ap: 1.60431549788598943e+02,
        ap_array: [6.42722117202268493e+01, 6.92722117202268493e+01, 7.42722117202268493e+01, 7.92722117202268493e+01, 8.42722117202268493e+01, 8.92722117202268493e+01, 9.42722117202268493e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.42880575300996797e+13, 0.00000000000000000e+00, 6.59959848274100019e+18, 1.77047428496492826e+18, 7.89410972513064800e+16, 4.06038914425327455e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.28633057131648883e+02, 4.06038914425327455e-04] },
    // case 428: alt 15.000 km, lat -19.604, lon -161.337, doy 259, ap 176.221
    Case { doy: 259, sec: 2.64296296296296314e+04, alt: 1.50000000000000000e+01,
        g_lat: -1.96043315285297837e+01, g_long: -1.61337340345604815e+02, lst: 7.34156378600823079e+00,
        f107a: 7.49476558944014641e+01, f107: 1.26847140240179115e+02, ap: 1.76221023472809492e+02,
        ap_array: [6.86200378071833654e+01, 7.36200378071833654e+01, 7.86200378071833654e+01, 8.36200378071833654e+01, 8.86200378071833654e+01, 9.36200378071833654e+01, 9.86200378071833654e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.38835773876044922e+13, 0.00000000000000000e+00, 3.55901861314501171e+18, 9.54777923348412288e+17, 4.25712011411340560e+16, 2.18967874769972990e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.05020702200486397e+02, 2.18967874769972990e-04] },
    // case 429: alt 20.000 km, lat +6.110, lon -128.610, doy 168, ap 192.010
    Case { doy: 168, sec: 5.52296296296296277e+04, alt: 2.00000000000000000e+01,
        g_lat: 6.10995418575592453e+00, g_long: -1.28610067618332096e+02, lst: 6.76755927811942293e+00,
        f107a: 9.30245789713245301e+01, f107: 1.40670669651943825e+02, ap: 1.92010497157019984e+02,
        ap_array: [7.29678638941398816e+01, 7.79678638941398816e+01, 8.29678638941398816e+01, 8.79678638941398816e+01, 9.29678638941398816e+01, 9.79678638941398816e+01, 1.02967863894139882e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.03150733013131035e+13, 0.00000000000000000e+00, 1.53710381319934438e+18, 4.12358839966988160e+17, 1.83860672615839560e+16, 9.45699901748078511e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.07255932180241359e+02, 9.45699901748078511e-05] },
    // case 430: alt 32.500 km, lat +31.824, lon -95.883, doy 351, ap 207.800
    Case { doy: 351, sec: 8.40296296296296205e+04, alt: 3.25000000000000000e+01,
        g_lat: 3.18242399000416469e+01, g_long: -9.58827948910593477e+01, lst: 1.69493774599376010e+01,
        f107a: 1.11101502048247610e+02, f107: 1.54494199063708521e+02, ap: 2.07799970841230504e+02,
        ap_array: [7.73156899810963978e+01, 8.23156899810963978e+01, 8.73156899810963978e+01, 9.23156899810963978e+01, 9.73156899810963978e+01, 1.02315689981096398e+02, 1.07315689981096398e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.34531119585480127e+12, 0.00000000000000000e+00, 2.00471960662164608e+17, 5.37806128868389040e+16, 2.39794535746048200e+15, 1.23339953927280123e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.29141570147703220e+02, 1.23339953927280123e-05] },
    // case 431: alt 45.000 km, lat +57.539, lon -63.156, doy 20, ap 223.589
    Case { doy: 20, sec: 1.89629629629629585e+03, alt: 4.50000000000000000e+01,
        g_lat: 5.75385256143273409e+01, g_long: -6.31555221637866424e+01, lst: -3.68361917305902731e+00,
        f107a: 1.29178425125170691e+02, f107: 1.68317728475473245e+02, ap: 2.23589444525441053e+02,
        ap_array: [8.16635160680529282e+01, 8.66635160680529282e+01, 9.16635160680529282e+01, 9.66635160680529282e+01, 1.01663516068052928e+02, 1.06663516068052928e+02, 1.11663516068052928e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.70040779936502838e+11, 0.00000000000000000e+00, 2.53386789996457680e+16, 6.79760741523497000e+15, 3.03088608854262875e+14, 1.55895691849951886e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.49384364590927049e+02, 1.55895691849951886e-06] },
    // case 432: alt 55.000 km, lat +83.253, lon -30.428, doy 202, ap 239.379
    Case { doy: 202, sec: 3.06962962962962956e+04, alt: 5.50000000000000000e+01,
        g_lat: 8.32528113286130633e+01, g_long: -3.04282494365139087e+01, lst: 8.52674897119341679e+00,
        f107a: 1.47255348202093785e+02, f107: 1.82141257887237941e+02, ap: 2.39378918209651573e+02,
        ap_array: [8.60113421550094586e+01, 9.10113421550094586e+01, 9.60113421550094586e+01, 1.01011342155009459e+02, 1.06011342155009459e+02, 1.11011342155009459e+02, 1.16011342155009459e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.27188831131576385e+10, 0.00000000000000000e+00, 1.23263797495883660e+16, 3.30679789542207100e+15, 1.47441991374701250e+14, 7.58377932446351643e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.76300796166283590e+02, 7.58377932446351643e-07] },
    // case 433: alt 62.500 km, lat -67.359, lon +2.299, doy 111, ap 255.168
    Case { doy: 111, sec: 5.94962962962962847e+04, alt: 6.25000000000000000e+01,
        g_lat: -6.73594335693461090e+01, g_long: 2.29902329075886769e+00, lst: 1.66800171905773382e+01,
        f107a: 1.65332271279016823e+02, f107: 1.95964787299002637e+02, ap: 2.55168391893862093e+02,
        ap_array: [9.03591682419659747e+01, 9.53591682419659747e+01, 1.00359168241965975e+02, 1.05359168241965975e+02, 1.10359168241965975e+02, 1.15359168241965975e+02, 1.20359168241965975e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.65742875284818611e+10, 0.00000000000000000e+00, 2.46982254191530300e+15, 6.62579293319618375e+14, 2.95427823351198984e+13, 1.51955314609682577e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.44876014627965191e+02, 1.51955314609682577e-07] },
    // case 434: alt 72.500 km, lat -41.645, lon +35.026, doy 293, ap 270.958
    Case { doy: 293, sec: 1.14962962962962956e+04, alt: 7.25000000000000000e+01,
        g_lat: -4.16451478550604008e+01, g_long: 3.50262960180315872e+01, lst: 5.52850203906218773e+00,
        f107a: 1.83409194355939917e+02, f107: 2.09788316710767361e+02, ap: 2.70957865578072585e+02,
        ap_array: [9.47069943289224909e+01, 9.97069943289224909e+01, 1.04706994328922491e+02, 1.09706994328922491e+02, 1.14706994328922491e+02, 1.19706994328922491e+02, 1.24706994328922491e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.34193574443205833e+09, 3.42630650886830082e+06, 9.41790637237903500e+14, 2.44217702572850281e+14, 1.12802868192188223e+13, 5.74963264266116632e-08, 5.89616820260365539e-31, 3.75185619059364253e+04, 6.04769419246877672e-62, 1.02731846489999998e+03, 2.18561809844548833e+02, 5.74963264266116632e-08] },
    // case 435: alt 90.000 km, lat -15.931, lon +67.754, doy 65, ap 286.747
    Case { doy: 65, sec: 4.02962962962962993e+04, alt: 9.00000000000000000e+01,
        g_lat: -1.59308621407746784e+01, g_long: 6.77535687453042783e+01, lst: 1.57103202208803694e+01,
        f107a: 2.01486117432863011e+02, f107: 2.23611846122532057e+02, ap: 2.86747339262283162e+02,
        ap_array: [9.90548204158790213e+01, 1.04054820415879021e+02, 1.09054820415879021e+02, 1.14054820415879021e+02, 1.19054820415879021e+02, 1.24054820415879021e+02, 1.29054820415879021e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.52790983315208316e+08, 2.90221446057977295e+11, 4.45007253870237812e+13, 9.94768553399266211e+12, 5.10803032575652527e+11, 2.63844279369309721e-09, 4.25562900189493448e+07, 2.57859173252829001e+05, 3.83112819356227409e-47, 1.02731846489999998e+03, 1.63100174221816928e+02, 2.63844279369309721e-09] },
    // case 436: alt 100.000 km, lat +9.783, lon +100.481, doy 248, ap 3.368
    Case { doy: 248, sec: 6.90962962962962920e+04, alt: 1.00000000000000000e+02,
        g_lat: 9.78342357351102976e+00, g_long: 1.00480841472577026e+02, lst: 1.91934156378600775e+01,
        f107a: 2.19563040509786077e+02, f107: 2.37435375534296753e+02, ap: 3.36783787724158001e+00,
        ap_array: [3.59168241965973545e+00, 8.59168241965973500e+00, 1.35916824196597350e+01, 1.85916824196597368e+01, 2.35916824196597368e+01, 2.85916824196597368e+01, 3.35916824196597332e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.04459269298693910e+08, 4.31674941936265808e+11, 9.68911892986142578e+12, 1.80817136659424023e+12, 9.79749531801491852e+10, 5.64371864279342591e-10, 1.48754690696776584e+07, 5.06630016335887369e+05, 2.93687643093510199e-43, 1.02731846489999998e+03, 1.94876017816897189e+02, 5.64371864279342591e-10] },
    // case 437: alt 110.000 km, lat +35.498, lon +133.208, doy 157, ap 19.157
    Case { doy: 157, sec: 2.10962962962962956e+04, alt: 1.10000000000000000e+02,
        g_lat: 3.54977092877967522e+01, g_long: 1.33208114199849774e+02, lst: 1.47406232511834006e+01,
        f107a: 2.37639963586709143e+02, f107: 2.51258904946061449e+02, ap: 1.91573115614521043e+01,
        ap_array: [7.93950850661625829e+00, 1.29395085066162583e+01, 1.79395085066162565e+01, 2.29395085066162565e+01, 2.79395085066162565e+01, 3.29395085066162565e+01, 3.79395085066162565e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.16615925113450103e+07, 1.60113953963195557e+11, 1.27348104836956641e+12, 1.54676377353715698e+11, 8.39683814711669445e+09, 7.22182291592648405e-11, 4.51303213685308956e+06, 1.13903525694765197e+06, 1.03302800632426194e-36, 1.02731846489999998e+03, 2.72366144895703826e+02, 7.22182291592648405e-11] },
    // case 438: alt 120.000 km, lat +61.212, lon +165.935, doy 339, ap 34.947
    Case { doy: 339, sec: 4.98962962962962993e+04, alt: 1.20000000000000000e+02,
        g_lat: 6.12119950020824604e+01, g_long: 1.65935386927122522e+02, lst: 2.49224414330015840e+01,
        f107a: 2.55716886663632238e+02, f107: 2.65082434357826173e+02, ap: 3.49467852456626318e+01,
        ap_array: [1.22873345935727798e+01, 1.72873345935727798e+01, 2.22873345935727798e+01, 2.72873345935727798e+01, 3.22873345935727798e+01, 3.72873345935727798e+01, 4.22873345935727798e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.30400564098405540e+07, 8.60641006996364288e+10, 3.11563175335672607e+11, 3.62240712032729340e+10, 1.51834918906051135e+09, 1.87926730975238049e-11, 1.79338805138957035e+06, 1.04913378631088627e+06, 9.26629803896316056e-33, 1.02731846489999998e+03, 4.16750204335454782e+02, 1.87926730975238049e-11] },
    // case 439: alt 123.435 km, lat +86.926, lon -158.362, doy 43, ap 50.736
    Case { doy: 43, sec: 7.86962962962962920e+04, alt: 1.23435000000000002e+02,
        g_lat: 8.69262807163681828e+01, g_long: -1.58362133734034558e+02, lst: 1.13026067222577780e+01,
        f107a: 2.73793809740555332e+02, f107: 2.78905963769590869e+02, ap: 5.07362589298731592e+01,
        ap_array: [1.66351606805292995e+01, 2.16351606805292995e+01, 2.66351606805292995e+01, 3.16351606805292995e+01, 3.66351606805292960e+01, 4.16351606805292960e+01, 4.66351606805292960e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.16518866302871741e+07, 2.89693494253488159e+10, 2.08672400629128662e+11, 2.63665376613373680e+10, 1.44302088485921049e+09, 1.19650412805664324e-11, 1.05050387814551941e+06, 1.54900596867807186e+06, 4.73458433152614070e-31, 1.02731846489999998e+03, 4.93694898371761838e+02, 1.19650412805664324e-11] },
    // case 440: alt 160.000 km, lat -88.875, lon -125.635, doy 225, ap 66.526
    Case { doy: 225, sec: 5.09629629629629562e+03, alt: 1.60000000000000000e+02,
        g_lat: -8.88754685547688439e+01, g_long: -1.25634861006761838e+02, lst: 1.41563786008230430e+00,
        f107a: 2.91870732817478370e+02, f107: 2.92729493181355565e+02, ap: 6.65257326140836796e+01,
        ap_array: [2.09829867674858228e+01, 2.59829867674858228e+01, 3.09829867674858228e+01, 3.59829867674858264e+01, 4.09829867674858264e+01, 4.59829867674858264e+01, 5.09829867674858264e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.52604884630600270e+06, 8.30114155093650532e+09, 2.22287168627521248e+10, 1.32000418256643271e+09, 8.33824599434286654e+07, 1.32972057690958506e-12, 5.50754358829305056e+04, 1.51831165343010928e+07, 3.50267537047627388e-17, 1.42674043845850383e+03, 1.03918665720397325e+03, 1.32972057690958506e-12] },
    // case 441: alt 200.000 km, lat -63.161, lon -92.908, doy 134, ap 82.315
    Case { doy: 134, sec: 3.38962962962962993e+04, alt: 2.00000000000000000e+02,
        g_lat: -6.31611828404831357e+01, g_long: -9.29075882794891044e+01, lst: 3.22179864144969930e+00,
        f107a: 7.63381884387801506e+01, f107: 7.23661713820476251e+01, ap: 8.23152062982942141e+01,
        ap_array: [2.53308128544423425e+01, 3.03308128544423425e+01, 3.53308128544423425e+01, 4.03308128544423425e+01, 4.53308128544423425e+01, 5.03308128544423425e+01, 5.53308128544423425e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.76582485381843187e+07, 2.30286101067535591e+09, 3.08944521051716471e+09, 2.95099292265671670e+08, 5.10651570428848080e+06, 2.20980406052311347e-13, 5.13924845838858571e+05, 3.70711232707412448e+06, 1.18377393609000773e-09, 9.28495242891618659e+02, 8.54242039823018217e+02, 2.20980406052311347e-13] },
    // case 442: alt 240.000 km, lat -37.447, lon -60.180, doy 316, ap 98.105
    Case { doy: 316, sec: 6.26962962962962920e+04, alt: 2.40000000000000000e+02,
        g_lat: -3.74468971261974204e+01, g_long: -6.01803155522163848e+01, lst: 1.34036168232678765e+01,
        f107a: 9.44151115157032450e+01, f107: 8.61897007938123352e+01, ap: 9.81046799825047344e+01,
        ap_array: [2.96786389413988658e+01, 3.46786389413988658e+01, 3.96786389413988658e+01, 4.46786389413988658e+01, 4.96786389413988658e+01, 5.46786389413988658e+01, 5.96786389413988658e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.09462340403436404e+06, 2.07332384341625595e+09, 1.29353986194936156e+09, 5.54894843172458485e+07, 4.84013286590974662e+05, 1.19292211436759468e-13, 1.21858206403185453e+05, 4.67825688050867617e+07, 5.28822299130076584e-03, 1.01956495983504669e+03, 9.88490005616404915e+02, 1.19292211436899898e-13] },
    // case 443: alt 250.000 km, lat -11.733, lon -27.453, doy 88, ap 113.894
    Case { doy: 88, sec: 1.46962962962962956e+04, alt: 2.50000000000000000e+02,
        g_lat: -1.17326114119117051e+01, g_long: -2.74530428249436511e+01, lst: 2.25210167175272780e+00,
        f107a: 1.12492034592626311e+02, f107: 1.00013230205577045e+02, ap: 1.13894153666715255e+02,
        ap_array: [3.40264650283553891e+01, 3.90264650283553891e+01, 4.40264650283553891e+01, 4.90264650283553891e+01, 5.40264650283553891e+01, 5.90264650283553891e+01, 6.40264650283553891e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.21573583461606354e+07, 1.86236296393208957e+09, 5.03543557668939531e+08, 2.02188378026446290e+07, 1.85934322024810681e+05, 7.44203736187642578e-14, 2.19591513116167160e+05, 1.65167234030311834e+07, 6.70209991118081022e-02, 9.08805723034293123e+02, 8.92636989098457661e+02, 7.44203736205443319e-14] },
    // case 444: alt 300.000 km, lat +13.982, lon +5.274, doy 271, ap 129.684
    Case { doy: 271, sec: 4.34962962962962847e+04, alt: 3.00000000000000000e+02,
        g_lat: 1.39816743023740031e+01, g_long: 5.27422990232912525e+00, lst: 1.20823045267489686e+01,
        f107a: 1.30568957669549377e+02, f107: 1.13836759617341755e+02, ap: 1.29683627350925775e+02,
        ap_array: [3.83742911153119124e+01, 4.33742911153119124e+01, 4.83742911153119124e+01, 5.33742911153119124e+01, 5.83742911153119124e+01, 6.33742911153119124e+01, 6.83742911153119053e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [8.65538156100037135e+06, 1.03849797969026613e+09, 2.02826293593621522e+08, 5.40770576526210457e+06, 4.60147306771850272e+04, 3.76879111468739593e-14, 9.62774180710909277e+04, 1.42037184863266218e+07, 1.30183008108375418e+02, 1.11826221145646650e+03, 1.09888537735723094e+03, 3.76879146045346547e-14] },
    // case 445: alt 320.000 km, lat +39.696, lon +38.002, doy 179, ap 145.473
    Case { doy: 179, sec: 7.22962962962962774e+04, alt: 3.20000000000000000e+02,
        g_lat: 3.96959600166597113e+01, g_long: 3.80015026296018164e+01, lst: 2.26157380353890893e+01,
        f107a: 1.48645880746472471e+02, f107: 1.27660289029106451e+02, ap: 1.45473101035136324e+02,
        ap_array: [4.27221172022684286e+01, 4.77221172022684286e+01, 5.27221172022684286e+01, 5.77221172022684286e+01, 6.27221172022684286e+01, 6.77221172022684357e+01, 7.27221172022684357e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.55385911386046279e+06, 4.17022645492045522e+08, 1.48054890324412555e+08, 4.14078807413896266e+06, 5.29815518226112181e+04, 1.84324133886407732e-14, 3.53966092506489076e+04, 1.03634940476195700e+07, 1.14194290570000339e+03, 1.06889055501065150e+03, 1.06638993399993888e+03, 1.84324437186443476e-14] },
    // case 446: alt 450.000 km, lat +65.410, lon +70.729, doy 362, ap 161.263
    Case { doy: 362, sec: 2.42962962962962993e+04, alt: 4.50000000000000000e+02,
        g_lat: 6.54102457309454337e+01, g_long: 7.07287753568745359e+01, lst: 1.14642228838739406e+01,
        f107a: 1.66722803823395566e+02, f107: 1.41483818440871175e+02, ap: 1.61262574719346844e+02,
        ap_array: [4.70699432892249519e+01, 5.20699432892249519e+01, 5.70699432892249519e+01, 6.20699432892249519e+01, 6.70699432892249519e+01, 7.20699432892249519e+01, 7.70699432892249519e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.56034265375642292e+06, 1.00973593696716756e+08, 7.55766072042391077e+06, 2.12216780001067906e+05, 9.82981228481691460e+02, 3.14123756450996373e-15, 3.48895368593608655e+04, 2.85809072635569191e+06, 5.02409197948061228e+03, 1.18159038118208468e+03, 1.18134777054258552e+03, 3.14137100439293862e-15] },
    // case 447: alt 68.000 km, lat -85.202, lon +103.456, doy 5, ap 177.052
    Case { doy: 5, sec: 5.30962962962962920e+04, alt: 6.80000000000000000e+01,
        g_lat: -8.52019991670137387e+01, g_long: 1.03456048084147312e+02, lst: 2.16460410656921241e+01,
        f107a: 1.84799726900318632e+02, f107: 1.55307347852635871e+02, ap: 1.77052048403557393e+02,
        ap_array: [5.14177693761814751e+01, 5.64177693761814751e+01, 6.14177693761814751e+01, 6.64177693761814680e+01, 7.14177693761814680e+01, 7.64177693761814680e+01, 8.14177693761814680e+01],
        switches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.96662412813052240e+16, 0.00000000000000000e+00, 2.94141205199243759e+21, 7.76761491636785906e+20, 3.51785902924503409e+19, 1.80314391591615412e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.35184087034242623e+02, 1.80314391591615412e-04] },
    // case 448: alt 95.000 km, lat -59.488, lon +136.183, doy 188, ap 192.842
    Case { doy: 188, sec: 8.18962962962962920e+04, alt: 9.50000000000000000e+01,
        g_lat: -5.94877134527280305e+01, g_long: 1.36183320811420003e+02, lst: 2.27489711934156347e+01,
        f107a: 2.02876649977241726e+02, f107: 1.69130877264400567e+02, ap: 1.92841522087767913e+02,
        ap_array: [5.57655954631379984e+01, 6.07655954631379984e+01, 6.57655954631379984e+01, 7.07655954631379984e+01, 7.57655954631379984e+01, 8.07655954631379984e+01, 8.57655954631379984e+01],
        switches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.31655387905383156e+14, 1.69432325480938560e+17, 1.49947965764109824e+19, 3.61299036783274854e+18, 1.72573486998355392e+17, 9.04840101044943665e-07, 1.79855805707230234e+13, 7.24949088445314331e+10, 4.35045445955050623e-40, 1.02731846489999998e+03, 1.62913451274637424e+02, 9.04840101044943770e-07] },
    // case 449: alt 250.000 km, lat -33.773, lon +168.911, doy 97, ap 208.631
    Case { doy: 97, sec: 8.29629629629629562e+03, alt: 2.50000000000000000e+02,
        g_lat: -3.37734277384423152e+01, g_long: 1.68910593538692751e+02, lst: 1.35652329848840445e+01,
        f107a: 2.20953573054164792e+02, f107: 1.82954406676165291e+02, ap: 2.08630995771978405e+02,
        ap_array: [6.01134215500945146e+01, 6.51134215500945146e+01, 7.01134215500945146e+01, 7.51134215500945146e+01, 8.01134215500945146e+01, 8.51134215500945146e+01, 9.01134215500945146e+01],
        switches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.10374007215722520e+13, 3.78064833148617100e+15, 1.88930468716378250e+15, 4.06506574825607344e+13, 1.33585332105554370e+12, 1.92956956116082959e-10, 2.66133485375797081e+10, 1.03556732680553594e+14, 1.41109426412772475e+05, 1.45636738285912952e+03, 1.36881777652359960e+03, 1.92956956119830785e-10] },
    // case 450: alt 700.000 km, lat -8.059, lon -155.387, doy 279, ap 224.420
    Case { doy: 279, sec: 3.70962962962962993e+04, alt: 7.00000000000000000e+02,
        g_lat: -8.05914202415659986e+00, g_long: -1.55386927122464300e+02, lst: -5.46017258597597532e-02,
        f107a: 2.39030496131087858e+02, f107: 1.96777936087929987e+02, ap: 2.24420469456188954e+02,
        ap_array: [6.44612476370510450e+01, 6.94612476370510450e+01, 7.44612476370510450e+01, 7.94612476370510450e+01, 8.44612476370510450e+01, 8.94612476370510450e+01, 9.44612476370510450e+01],
        switches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.33388433431073486e+12, 1.09818337535047305e+13, 3.56645839355422897e+10, 1.01184988068112269e+08, 2.46026531947872398e+05, 3.29174388901499683e-13, 2.99524737782075119e+10, 5.87224190106155640e+11, 5.37490057558511719e+11, 1.21429671945353107e+03, 1.21429415427724302e+03, 3.43450124830253831e-13] },
    // case 451: alt 657.280 km, lat +17.655, lon -122.660, doy 51, ap 240.210
    Case { doy: 51, sec: 6.58962962962962920e+04, alt: 6.57279999999999973e+02,
        g_lat: 1.76551436901291083e+01, g_long: -1.22659654395191581e+02, lst: 1.01272164559584201e+01,
        f107a: 2.57107419208010924e+02, f107: 2.10601465499694712e+02, ap: 2.40209943140399503e+02,
        ap_array: [6.88090737240075612e+01, 7.38090737240075612e+01, 7.88090737240075612e+01, 8.38090737240075612e+01, 8.88090737240075612e+01, 9.38090737240075612e+01, 9.88090737240075612e+01],
        switches: [0, 0, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.73079345230385382e+06, 5.42406238351953961e+06, 1.72605691438797621e+04, 1.37421083527870195e+02, 8.08144695577386329e-02, 1.66109184591628181e-16, 5.13088697958188350e+04, 1.29899742269589115e+05, 3.58671348964573326e+03, 1.10082837827414687e+03, 1.10081467235480295e+03, 1.66204447701913195e-16] },
    // case 452: alt 45.588 km, lat +43.369, lon -89.932, doy 234, ap 255.999
    Case { doy: 234, sec: 1.78962962962962956e+04, alt: 4.55880000000000081e+01,
        g_lat: 4.33694294044148023e+01, g_long: -8.99323816679188610e+01, lst: 4.97119341563785966e+00,
        f107a: 2.75184342284934019e+02, f107: 2.24424994911459407e+02, ap: 2.55999416824609995e+02,
        ap_array: [7.31568998109640773e+01, 7.81568998109640773e+01, 8.31568998109640773e+01, 8.81568998109640773e+01, 9.31568998109640773e+01, 9.81568998109640773e+01, 1.03156899810964077e+02],
        switches: [0, 1, 0, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.24432562916633514e+11, 0.00000000000000000e+00, 3.34438872306747280e+16, 8.97199162737311000e+15, 4.00039057109715750e+14, 2.05762815735212385e-06, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.63449913385115281e+02, 2.05762815735212385e-06] },
    // case 453: alt 111.868 km, lat +69.084, lon -57.205, doy 142, ap 271.789
    Case { doy: 142, sec: 4.66962962962962847e+04, alt: 1.11867999999999995e+02,
        g_lat: 6.90837151187005247e+01, g_long: -5.72051089406461415e+01, lst: 9.15751948626144596e+00,
        f107a: 2.93261265361857113e+02, f107: 2.38248524323224103e+02, ap: 2.71788890508820487e+02,
        ap_array: [7.75047258979206077e+01, 8.25047258979206077e+01, 8.75047258979206077e+01, 9.25047258979206077e+01, 9.75047258979206077e+01, 1.02504725897920608e+02, 1.07504725897920608e+02],
        switches: [0, 1, 1, 0, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [9.59363554733774997e+06, 2.93067913120170364e+10, 4.64326920118175720e+11, 6.21441171907817230e+10, 3.72163024600335979e+09, 2.59085938277459805e-11, 1.40141543023297074e+06, 5.34501179049489205e+05, 3.74599958580110828e-35, 1.02731846489999998e+03, 6.39650168902102678e+02, 2.59085938277459805e-11] },
    // case 454: alt 139.264 km, lat -81.529, lon -24.478, doy 325, ap 287.578
    Case { doy: 325, sec: 7.54962962962962774e+04, alt: 1.39264000000000010e+02,
        g_lat: -8.15285297792586476e+01, g_long: -2.44778362133733935e+01, lst: 1.93393376680796294e+01,
        f107a: 7.77287209831588513e+01, f107: 2.52072053734988799e+02, ap: 2.87578364193031064e+02,
        ap_array: [8.18525519848771239e+01, 8.68525519848771239e+01, 9.18525519848771239e+01, 9.68525519848771239e+01, 1.01852551984877124e+02, 1.06852551984877124e+02, 1.11852551984877124e+02],
        switches: [0, 1, 1, 1, 0, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.21337177232528944e+06, 1.71237007593334618e+10, 8.52565392400380554e+10, 1.19565775252632103e+10, 6.28737950798472524e+08, 5.09462240771127547e-12, 7.47562850346168852e+05, 8.12374563293206599e+06, 1.32687143056869167e-24, 1.18273856670621785e+03, 6.77416309361371759e+02, 5.09462240771127547e-12] },
    // case 455: alt 561.280 km, lat -55.814, lon +8.249, doy 28, ap 4.199
    Case { doy: 28, sec: 2.74962962962962956e+04, alt: 5.61279999999999973e+02,
        g_lat: -5.58142440649729323e+01, g_long: 8.24943651389935440e+00, lst: 8.18782251656448423e+00,
        f107a: 9.58056440600819315e+01, f107: 2.65895583146753552e+02, ap: 4.19886280798950207e+00,
        ap_array: [8.62003780718336543e+01, 9.12003780718336543e+01, 9.62003780718336543e+01, 1.01200378071833654e+02, 1.06200378071833654e+02, 1.11200378071833654e+02, 1.16200378071833654e+02],
        switches: [0, 1, 1, 1, 1, 0, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.00665876324363286e+06, 1.70710229534048103e+07, 5.75346739605594310e+05, 2.09650303104764098e+04, 3.34787568965147315e+01, 4.97215211102261435e-16, 4.90622613833701398e+04, 3.95231586126376584e+05, 1.13193493095326321e+04, 1.25997239495182521e+03, 1.25913900266022370e+03, 4.97515853019922646e-16] },
    // case 456: alt 33.988 km, lat -30.100, lon +40.977, doy 211, ap 19.988
    Case { doy: 211, sec: 5.62962962962962847e+04, alt: 3.39880000000000067e+01,
        g_lat: -3.00999583506872170e+01, g_long: 4.09767092411720455e+01, lst: 1.56378600823045240e+01,
        f107a: 1.13882567137004997e+02, f107: 2.79719112558518191e+02, ap: 1.99883364922000304e+01,
        ap_array: [9.05482041587901705e+01, 9.55482041587901705e+01, 1.00548204158790170e+02, 1.05548204158790170e+02, 1.10548204158790170e+02, 1.15548204158790170e+02, 1.20548204158790170e+02],
        switches: [0, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.10048941714765088e+12, 0.00000000000000000e+00, 1.63989768183987840e+17, 4.39935351107184160e+16, 1.96156361213277050e+15, 1.00894361413684628e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.34056911965756314e+02, 1.00894361413684628e-05] },
    // case 457: alt 104.268 km, lat -4.386, lon +73.704, doy 120, ap 35.778
    Case { doy: 120, sec: 8.50962962962962774e+04, alt: 1.04268000000000001e+02,
        g_lat: -4.38567263640150884e+00, g_long: 7.37039819684447934e+01, lst: 2.85514588802008404e+01,
        f107a: 1.31959490213928063e+02, f107: 2.93542641970282943e+02, ap: 3.57778101764105543e+01,
        ap_array: [9.48960302457466867e+01, 9.98960302457466867e+01, 1.04896030245746687e+02, 1.09896030245746687e+02, 1.14896030245746687e+02, 1.19896030245746687e+02, 1.24896030245746687e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.93862790306622535e+07, 5.05919090038413635e+11, 3.36418455401843555e+12, 6.29256306095531128e+11, 2.95468848395355377e+10, 2.05193002562074582e-10, 8.55763563261595741e+06, 4.47339491447451641e+05, 7.49577122933782724e-41, 1.02731846489999998e+03, 2.30184083092361163e+02, 2.05193002562074582e-10] },
    // case 458: alt 363.264 km, lat +21.329, lon +106.431, doy 302, ap 51.567
    Case { doy: 302, sec: 2.96296296296296259e+03, alt: 3.63264000000000010e+02,
        g_lat: 2.13286130778842136e+01, g_long: 1.06431254695717541e+02, lst: 7.91846224720421432e+00,
        f107a: 1.50036413290851158e+02, f107: 7.31793201709749610e+01, ap: 5.15672838606210817e+01,
        ap_array: [9.92438563327032170e+01, 1.04243856332703217e+02, 1.09243856332703217e+02, 1.14243856332703217e+02, 1.19243856332703217e+02, 1.24243856332703217e+02, 1.29243856332703217e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.41465657387150321e+07, 3.20718162438184798e+08, 1.30561593966575544e+07, 2.17102167479493655e+05, 7.16114654137771822e+02, 9.29340237506325035e-15, 1.56662909782697156e+05, 2.68951884094524616e+06, 3.69359587557934333e+02, 8.70462003111393642e+02, 8.70243024635673237e+02, 9.29341218525389575e-15] },
    // case 459: alt 465.280 km, lat +47.043, lon +139.159, doy 74, ap 67.357
    Case { doy: 74, sec: 3.17629629629629635e+04, alt: 4.65279999999999973e+02,
        g_lat: 4.70428987921699218e+01, g_long: 1.39158527422990289e+02, lst: 1.81002804290223978e+01,
        f107a: 1.68113336367774252e+02, f107: 8.70028495827396711e+01, ap: 6.73567575448316092e+01,
        ap_array: [3.78071833648393207e+00, 8.78071833648393252e+00, 1.37807183364839325e+01, 1.87807183364839325e+01, 2.37807183364839325e+01, 2.87807183364839325e+01, 3.37807183364839290e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.93621129432710493e+06, 5.40211008465622514e+07, 1.32088555169493193e+06, 1.71244084283090669e+04, 4.20122240119733732e+01, 1.54250251106272981e-15, 7.16236999067961879e+04, 1.10927433785251249e+06, 1.16977067504812567e+04, 1.03753206269415523e+03, 1.03705069575232551e+03, 1.54281320215402270e-15] },
    // case 460: alt 22.388 km, lat +72.757, lon +171.886, doy 256, ap 83.146
    Case { doy: 256, sec: 6.05629629629629562e+04, alt: 2.23880000000000052e+01,
        g_lat: 7.27571845064556442e+01, g_long: 1.71885800150262980e+02, lst: 1.68230452674897109e+01,
        f107a: 1.86190259444697347e+02, f107: 1.00826378994504381e+02, ap: 8.31462312290421153e+01,
        ap_array: [8.12854442344045403e+00, 1.31285444234404540e+01, 1.81285444234404522e+01, 2.31285444234404522e+01, 2.81285444234404522e+01, 3.31285444234404522e+01, 3.81285444234404522e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.54092426567988965e+12, 0.00000000000000000e+00, 9.74697836548073856e+17, 2.61482188610751776e+17, 1.16588481718697860e+16, 5.99680802521068864e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.21522331496940780e+02, 5.99680802521068864e-05] },
    // case 461: alt 96.668 km, lat -77.855, lon -152.412, doy 165, ap 98.936
    Case { doy: 165, sec: 1.25629629629629617e+04, alt: 9.66680000000000064e+01,
        g_lat: -7.78550603915035424e+01, g_long: -1.52411720510894071e+02, lst: -6.67106943323655877e+00,
        f107a: 2.04267182521620413e+02, f107: 1.14649908406269077e+02, ap: 9.89357049132526498e+01,
        ap_array: [1.24763705103969755e+01, 1.74763705103969755e+01, 2.24763705103969755e+01, 2.74763705103969755e+01, 3.24763705103969755e+01, 3.74763705103969755e+01, 4.24763705103969755e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.24521143929113328e+08, 2.97628823505901733e+11, 1.15751266603658926e+13, 2.57224782391943701e+12, 1.28421256538304092e+11, 6.91082746197893063e-10, 1.90106004682821222e+07, 1.39975612856466032e+05, 6.70351925461541422e-45, 1.02731846489999998e+03, 1.78987857369223889e+02, 6.91082746197893063e-10] },
    // case 462: alt 318.464 km, lat -52.141, lon -119.684, doy 348, ap 114.725
    Case { doy: 348, sec: 4.13629629629629635e+04, alt: 3.18464000000000055e+02,
        g_lat: -5.21407746772178271e+01, g_long: -1.19684447783621337e+02, lst: 3.51074874858162289e+00,
        f107a: 2.22344105598543479e+02, f107: 1.28473437818033801e+02, ap: 1.14725178597463170e+02,
        ap_array: [1.68241965973534953e+01, 2.18241965973534953e+01, 2.68241965973534953e+01, 3.18241965973534953e+01, 3.68241965973534917e+01, 4.18241965973534917e+01, 4.68241965973534917e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.64442580282535031e+06, 4.74199920241240621e+08, 2.49010936914044350e+08, 2.97305265897911647e+06, 8.97359487524324359e+04, 2.44404095250755500e-14, 3.55220167852705781e+04, 4.16378302851364762e+06, 8.06477062224176279e+02, 1.21933777868896982e+03, 1.20767264521077846e+03, 2.44404309451063212e-14] },
    // case 463: alt 945.280 km, lat -26.426, lon -86.957, doy 17, ap 130.515
    Case { doy: 17, sec: 7.01629629629629490e+04, alt: 9.45280000000000086e+02,
        g_lat: -2.64264889629321118e+01, g_long: -8.69571750563486034e+01, lst: 1.36925669303998028e+01,
        f107a: 2.40421028675466545e+02, f107: 1.42296967229798497e+02, ap: 1.30514652281673705e+02,
        ap_array: [2.11720226843100185e+01, 2.61720226843100185e+01, 3.11720226843100185e+01, 3.61720226843100221e+01, 4.11720226843100221e+01, 4.61720226843100221e+01, 5.11720226843100221e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.95313586037073634e+05, 6.50324932406198466e+05, 5.27665189865155753e+02, 8.51572812174853366e-01, 4.86288992067687470e-04, 2.20879765976282468e-17, 2.54002766741883606e+04, 3.42396117251284304e+04, 5.11187323192531858e+03, 1.29359480066134870e+03, 1.29359478392695428e+03, 2.22237479506681807e-17] },
    // case 464: alt 10.788 km, lat -0.712, lon -54.230, doy 199, ap 146.304
    Case { doy: 199, sec: 2.21629629629629635e+04, alt: 1.07879999999999985e+01,
        g_lat: -7.12203248646403608e-01, g_long: -5.42299023290759123e+01, lst: 6.15637860082304567e+00,
        f107a: 2.58497951752389667e+02, f107: 1.56120496641563193e+02, ap: 1.46304125965884225e+02,
        ap_array: [2.55198487712665383e+01, 3.05198487712665383e+01, 3.55198487712665383e+01, 4.05198487712665383e+01, 4.55198487712665383e+01, 5.05198487712665383e+01, 5.55198487712665383e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.09283554235251094e+13, 0.00000000000000000e+00, 6.09895144239740416e+18, 1.63616570345195162e+18, 7.29526076782365440e+16, 3.75236710124113955e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.31052243053905897e+02, 3.75236710124113955e-04] },
    // case 465: alt 89.068 km, lat +25.002, lon -21.503, doy 108, ap 162.094
    Case { doy: 108, sec: 5.09629629629629562e+04, alt: 8.90679999999999978e+01,
        g_lat: 2.50020824656393046e+01, g_long: -2.15026296018031644e+01, lst: 1.27228699607028322e+01,
        f107a: 2.76574874829312762e+02, f107: 1.69944026053327889e+02, ap: 1.62093599650094774e+02,
        ap_array: [2.98676748582230580e+01, 3.48676748582230545e+01, 3.98676748582230545e+01, 4.48676748582230545e+01, 4.98676748582230545e+01, 5.48676748582230545e+01, 5.98676748582230545e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], out: [5.00811652865244031e+08, 2.78482116743202393e+11, 6.50890867785672031e+13, 1.38177456249678359e+13, 7.53485830140818848e+11, 3.81677077893409217e-09, 6.19684409982523844e+07, 2.35325269835353829e+05, 4.74741492521998058e-50, 1.02731846489999998e+03, 1.85673912066235005e+02, 3.81677077893409217e-09] },
    // case 466: alt 273.664 km, lat +50.716, lon +11.225, doy 291, ap 177.883
    Case { doy: 291, sec: 7.97629629629629490e+04, alt: 2.73663999999999987e+02,
        g_lat: 5.07163681799250128e+01, g_long: 1.12246431254695835e+01, lst: 2.29046881425210138e+01,
        f107a: 2.94651797906235799e+02, f107: 1.83767555465092613e+02, ap: 1.77883073334305323e+02,
        ap_array: [3.42155009451795920e+01, 3.92155009451795920e+01, 4.42155009451795920e+01, 4.92155009451795920e+01, 5.42155009451795920e+01, 5.92155009451795920e+01, 6.42155009451795991e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], out: [7.67327506280369032e+06, 2.45666993379854679e+09, 8.31085464562206626e+08, 8.92458856858540513e+06, 4.04973029271103209e+05, 1.04910775127548544e-13, 5.11986698001877812e+04, 2.06871515874373168e+07, 2.09449635728989625e+00, 1.15550595582352116e+03, 1.14854202299022359e+03, 1.04910775183178372e-13] },
    // case 467: alt 849.280 km, lat +76.431, lon +43.952, doy 63, ap 193.673
    Case { doy: 63, sec: 6.16296296296296259e+03, alt: 8.49280000000000086e+02,
        g_lat: 7.64306538942107352e+01, g_long: 4.39519158527422746e+01, lst: 4.64206187989475261e+00,
        f107a: 7.91192535275375519e+01, f107: 1.97591084876857309e+02, ap: 1.93672547018515814e+02,
        ap_array: [3.85633270321361081e+01, 4.35633270321361081e+01, 4.85633270321361081e+01, 5.35633270321361081e+01, 5.85633270321361081e+01, 6.35633270321361081e+01, 6.85633270321361010e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], out: [4.96814034576299659e+05, 2.42887808269281813e+05, 7.79959247884256797e+02, 1.20212375960426741e+01, 3.19335383501787333e-03, 9.94030357367979961e-18, 6.08042710846324917e+04, 2.26040535452233189e+03, 3.24085213949798572e+03, 1.23569997395817427e+03, 1.23568352777765404e+03, 1.00263806065048661e-17] },
    // case 468: alt 68.788 km, lat -74.182, lon +76.679, doy 245, ap 209.462
    Case { doy: 245, sec: 3.49629629629629635e+04, alt: 6.87880000000000109e+01,
        g_lat: -7.41815910037484372e+01, g_long: 7.66791885800150226e+01, lst: 9.71193415637860014e+00,
        f107a: 9.71961766044606321e+01, f107: 2.11414614288622033e+02, ap: 2.09462020702726335e+02,
        ap_array: [4.29111531190926243e+01, 4.79111531190926243e+01, 5.29111531190926243e+01, 5.79111531190926243e+01, 6.29111531190926243e+01, 6.79111531190926314e+01, 7.29111531190926314e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], out: [7.57419269772100544e+09, 0.00000000000000000e+00, 1.12437143663862912e+15, 3.02681190793106938e+14, 1.34421484277727188e+13, 6.92318181781369538e-08, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.23275959796893801e+02, 6.92318181781369538e-08] },
    // case 469: alt 81.468 km, lat -48.467, lon +109.406, doy 154, ap 225.251
    Case { doy: 154, sec: 6.37629629629629562e+04, alt: 8.14680000000000035e+01,
        g_lat: -4.84673052894627290e+01, g_long: 1.09406461307287771e+02, lst: 2.50056982435311177e+01,
        f107a: 1.15273099681383712e+02, f107: 2.25238143700386729e+02, ap: 2.25251494386936855e+02,
        ap_array: [4.72589792060491476e+01, 5.22589792060491476e+01, 5.72589792060491476e+01, 6.22589792060491476e+01, 6.72589792060491476e+01, 7.22589792060491476e+01, 7.72589792060491476e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], out: [1.44503429898881102e+09, 6.11319298722951698e+09, 2.06651466124053344e+14, 5.47065051312555781e+13, 2.45755798807708594e+12, 1.26745236176798322e-08, 6.42475828074931279e+07, 5.02902306887185405e+04, 2.78316016374236165e-56, 1.02731846489999998e+03, 1.92194125678933261e+02, 1.26745236176798322e-08] },
    // case 470: alt 228.864 km, lat -22.753, lon +142.134, doy 336, ap 241.041
    Case { doy: 336, sec: 1.57629629629629617e+04, alt: 2.28864000000000004e+02,
        g_lat: -2.27530195751770208e+01, g_long: 1.42133734034560518e+02, lst: 1.38541830920159690e+01,
        f107a: 1.33350022758306807e+02, f107: 2.39061673112151425e+02, ap: 2.41040968071147404e+02,
        ap_array: [5.16068052930056780e+01, 5.66068052930056780e+01, 6.16068052930056780e+01, 6.66068052930056780e+01, 7.16068052930056780e+01, 7.66068052930056780e+01, 8.16068052930056780e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], out: [8.17894044018027745e+06, 3.38413247564784622e+09, 2.35075619438434553e+09, 1.29764570705575600e+08, 1.57966174417796917e+06, 2.07017281418406259e-13, 1.07330046129659298e+05, 3.52454854613649473e+07, 4.99530360249971072e-05, 1.34409441308287865e+03, 1.20570184319228611e+03, 2.07017281418407572e-13] },
    // case 471: alt 753.280 km, lat +2.961, lon +174.861, doy 40, ap 256.830
    Case { doy: 40, sec: 4.45629629629629562e+04, alt: 7.53280000000000086e+02,
        g_lat: 2.96126613910870162e+00, g_long: 1.74861006761833266e+02, lst: 2.40360012738341524e+01,
        f107a: 1.51426945835229873e+02, f107: 2.52885202523916121e+02, ap: 2.56830441755357924e+02,
        ap_array: [5.59546313799621942e+01, 6.09546313799621942e+01, 6.59546313799621942e+01, 7.09546313799621942e+01, 7.59546313799621942e+01, 8.09546313799621942e+01, 8.59546313799621942e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], out: [1.06996219144923892e+06, 1.44772335857857228e+06, 1.78435093283780975e+03, 8.61096594011704219e+00, 3.40591789463124328e-03, 4.69375267219065651e-17, 4.01042918111303152e+04, 5.29895856457945338e+04, 2.58039628986030903e+03, 1.10948172336159837e+03, 1.10947842904648996e+03, 4.70060620473652563e-17] },
    // case 472: alt 57.188 km, lat +28.676, lon -149.437, doy 222, ap 272.620
    Case { doy: 222, sec: 7.33629629629629490e+04, alt: 5.71880000000000095e+01,
        g_lat: 2.86755518533944098e+01, g_long: -1.49436513899323813e+02, lst: 2.03786008230452609e+01,
        f107a: 1.69503868912152939e+02, f107: 2.66708731935680817e+02, ap: 2.72619915439568445e+02,
        ap_array: [6.03024574669187103e+01, 6.53024574669187103e+01, 7.03024574669187103e+01, 7.53024574669187103e+01, 8.03024574669187103e+01, 8.53024574669187103e+01, 9.03024574669187103e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], out: [5.30147692919467316e+10, 0.00000000000000000e+00, 7.90001211374438700e+15, 2.11933624975420675e+15, 9.44959948985534844e+13, 4.86046590713114609e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.46331025539539581e+02, 4.86046590713114609e-07] },
    // case 473: alt 119.468 km, lat +54.390, lon -116.709, doy 131, ap 288.409
    Case { doy: 131, sec: 2.53629629629629635e+04, alt: 1.19468000000000004e+02,
        g_lat: 5.43898375676801038e+01, g_long: -1.16709241172051094e+02, lst: -7.35348588424805882e-01,
        f107a: 1.87580791989076005e+02, f107: 2.80532261347445569e+02, ap: 2.88409389123778965e+02,
        ap_array: [6.46502835538752407e+01, 6.96502835538752407e+01, 7.46502835538752407e+01, 7.96502835538752407e+01, 8.46502835538752407e+01, 8.96502835538752407e+01, 9.46502835538752407e+01],
        switches: [0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.02320043482838012e+06, 1.81871580638088150e+10, 3.56544606393157043e+11, 6.47656966233105774e+10, 6.43904634716732788e+09, 2.09232012456389926e-11, 1.12869400685459631e+06, 3.73173200679514499e+05, 1.70568094672651646e-32, 1.02731846489999998e+03, 5.25848363985549781e+02, 2.09232012456389926e-11] },
    // case 474: alt 130.304 km, lat +80.104, lon -83.982, doy 313, ap 5.030
    Case { doy: 313, sec: 5.41629629629629562e+04, alt: 1.30304000000000002e+02,
        g_lat: 8.01041232819658262e+01, g_long: -8.39819684447783743e+01, lst: 9.44646959339337400e+00,
        f107a: 2.05657715065999099e+02, f107: 2.94355790759210265e+02, ap: 5.02988773873742545e+00,
        ap_array: [6.89981096408317569e+01, 7.39981096408317569e+01, 7.89981096408317569e+01, 8.39981096408317569e+01, 8.89981096408317569e+01, 9.39981096408317569e+01, 9.89981096408317569e+01],
        switches: [0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.11584139623175785e+07, 6.44921405478443298e+10, 1.11648343804758316e+11, 9.93848537906561852e+09, 2.73629627260356247e+08, 7.44887874910129224e-12, 1.05134260911983973e+06, 4.71981161295508593e+06, 3.42628275554592737e-29, 1.14268503565945343e+03, 5.32869989783377150e+02, 7.44887874910129224e-12] },
    // case 475: alt 542.080 km, lat -70.508, lon -51.255, doy 85, ap 20.819
    Case { doy: 85, sec: 8.29629629629629490e+04, alt: 5.42080000000000041e+02,
        g_lat: -7.05081216159933319e+01, g_long: -5.12546957175056548e+01, lst: 1.96282877752115503e+01,
        f107a: 2.23734638142922165e+02, f107: 7.39924689599022969e+01, ap: 2.08193614229479529e+01,
        ap_array: [7.33459357277882873e+01, 7.83459357277882873e+01, 8.33459357277882873e+01, 8.83459357277882873e+01, 9.33459357277882873e+01, 9.83459357277882873e+01, 1.03345935727788287e+02],
        switches: [0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.50691445386111550e+06, 1.12404396771944109e+07, 4.98814548103102788e+04, 2.09889040755971308e+02, 4.17269170795420530e-01, 3.20362360578738152e-16, 7.83509174674385722e+04, 1.16636885702221814e+05, 9.38285518740427360e+04, 8.88716592876478330e+02, 8.88716491342186714e+02, 3.22854446916512768e-16] },
    // case 476: alt 31.668 km, lat -44.794, lon -18.527, doy 268, ap 36.609
    Case { doy: 268, sec: 9.36296296296296350e+03, alt: 3.16680000000000064e+01,
        g_lat: -4.47938359017076237e+01, g_long: -1.85274229902329068e+01, lst: 2.60082304526748942e+00,
        f107a: 2.41811561219845231e+02, f107: 8.78159983716670069e+01, ap: 3.66088351071584768e+01,
        ap_array: [7.76937618147448035e+01, 8.26937618147448035e+01, 8.76937618147448035e+01, 9.26937618147448035e+01, 9.76937618147448035e+01, 1.02693761814744803e+02, 1.07693761814744803e+02],
        switches: [0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.37940071049682788e+12, 0.00000000000000000e+00, 2.05551820146992864e+17, 5.51433867908337760e+16, 2.45870809668825500e+15, 1.26465326836021354e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.36228601405741813e+02, 1.26465326836021354e-05] },
    // case 477: alt 102.748 km, lat -19.080, lon +14.200, doy 177, ap 52.398
    Case { doy: 177, sec: 3.81629629629629635e+04, alt: 1.02748000000000005e+02,
        g_lat: -1.90795501874219156e+01, g_long: 1.41998497370398127e+01, lst: 1.15474796944034761e+01,
        f107a: 2.59888484296768297e+02, f107: 1.01639527783431717e+02, ap: 5.23983087913690042e+01,
        ap_array: [8.20415879017013197e+01, 8.70415879017013197e+01, 9.20415879017013197e+01, 9.70415879017013197e+01, 1.02041587901701320e+02, 1.07041587901701320e+02, 1.12041587901701320e+02],
        switches: [0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [6.85047176676870883e+07, 4.40313063126788574e+11, 3.99966430203952686e+12, 6.55861136477912231e+11, 3.60359116972187500e+10, 2.34831720285547261e-10, 1.05662761180709098e+07, 3.46486955647350987e+05, 7.19548164909611363e-41, 1.02731846489999998e+03, 2.17170142879086995e+02, 2.34831720285547261e-10] },
    // case 478: alt 354.304 km, lat +6.635, lon +46.927, doy 359, ap 68.188
    Case { doy: 359, sec: 6.69629629629629490e+04, alt: 3.54304000000000030e+02,
        g_lat: 6.63473552686379264e+00, g_long: 4.69271224643125322e+01, lst: 2.17292978762216542e+01,
        f107a: 2.77965407373691392e+02, f107: 1.15463057195196413e+02, ap: 6.81877824755795245e+01,
        ap_array: [8.63894139886578500e+01, 9.13894139886578500e+01, 9.63894139886578500e+01, 1.01389413988657850e+02, 1.06389413988657850e+02, 1.11389413988657850e+02, 1.16389413988657850e+02],
        switches: [0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [4.75514448905652203e+06, 6.65584803389731407e+08, 3.90340261629661396e+07, 2.31573410222792503e+05, 4.84847803688235945e+03, 1.97104798362364243e-14, 1.15854652780555407e+05, 7.48090901578565873e+06, 1.90021607116533505e+03, 9.82107411436638245e+02, 9.81756336139250152e+02, 1.97105303059752723e-14] },
    // case 479: alt 446.080 km, lat +32.349, lon +79.654, doy 11, ap 83.977
    Case { doy: 11, sec: 1.89629629629629635e+04, alt: 4.46079999999999984e+02,
        g_lat: 3.23490212411495150e+01, g_long: 7.96543951915853086e+01, lst: 1.05777827247065090e+01,
        f107a: 2.96042330450614486e+02, f107: 1.29286586606961123e+02, ap: 8.39772561597900591e+01,
        ap_array: [9.07372400756143662e+01, 9.57372400756143662e+01, 1.00737240075614366e+02, 1.05737240075614366e+02, 1.10737240075614366e+02, 1.15737240075614366e+02, 1.20737240075614366e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.47290184137700684e+06, 1.44829888594498664e+08, 1.66760424691089243e+06, 7.75688176458400812e+03, 3.16675165636544875e+01, 4.01931107093740502e-15, 1.36156380432257603e+05, 2.50166731370077189e+06, 2.48096806754921017e+04, 9.35871966250375181e+02, 9.35870370548927667e+02, 4.01997001605614642e-15] },
    // case 480: alt 20.068 km, lat +58.063, lon +112.382, doy 194, ap 99.767
    Case { doy: 194, sec: 4.77629629629629562e+04, alt: 2.00680000000000049e+01,
        g_lat: 5.80633069554352232e+01, g_long: 1.12381667918858000e+02, lst: 1.32674897119341537e+01,
        f107a: 8.05097860719162526e+01, f107: 1.43110116018725819e+02, ap: 9.97667298440005794e+01,
        ap_array: [9.50850661625708824e+01, 1.00085066162570882e+02, 1.05085066162570882e+02, 1.10085066162570882e+02, 1.15085066162570882e+02, 1.20085066162570882e+02, 1.25085066162570882e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.00051877235382266e+13, 0.00000000000000000e+00, 1.49092611873811251e+18, 3.99970749873873472e+17, 1.78337127692859320e+16, 9.17289171945746980e-05, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.24657260479063666e+02, 9.17289171945746980e-05] },
    // case 481: alt 95.148 km, lat +83.778, lon +145.109, doy 102, ap 115.556
    Case { doy: 102, sec: 7.65629629629629490e+04, alt: 9.51479999999999961e+01,
        g_lat: 8.37775926697209457e+01, g_long: 1.45108940646130804e+02, lst: 3.09414190883428759e+01,
        f107a: 9.85867091488393328e+01, f107: 1.56933645430490543e+02, ap: 1.15556203528211100e+02,
        ap_array: [9.94328922495274128e+01, 1.04432892249527413e+02, 1.09432892249527413e+02, 1.14432892249527413e+02, 1.19432892249527413e+02, 1.24432892249527413e+02, 1.29432892249527413e+02],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.14611949330799133e+08, 7.15264028975063934e+10, 3.20587264902703164e+13, 8.67832965309688867e+12, 3.78686794258602051e+11, 1.97812851818583378e-09, 4.02075174724953994e+07, 1.53959181855769508e+05, 1.60984835834878487e-46, 1.02731846489999998e+03, 1.79485992398123869e+02, 1.97812851818583378e-09] },
    // case 482: alt 309.504 km, lat -66.835, lon +177.836, doy 285, ap 131.346
    Case { doy: 285, sec: 2.85629629629629635e+04, alt: 3.09504000000000019e+02,
        g_lat: -6.68346522282382409e+01, g_long: 1.77836213373403496e+02, lst: 1.97899039368277236e+01,
        f107a: 1.16663632225762399e+02, f107: 1.70757174842255239e+02, ap: 1.31345677212421634e+02,
        ap_array: [3.96975425330812914e+00, 8.96975425330812826e+00, 1.39697542533081283e+01, 1.89697542533081283e+01, 2.39697542533081283e+01, 2.89697542533081283e+01, 3.39697542533081318e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [2.08109643295948138e+06, 5.25109855534194767e+08, 3.47059298816351235e+08, 2.04044014475069903e+07, 2.43162967644182558e+05, 3.12851380190265012e-14, 4.22177504780358358e+04, 4.00119016052022576e+06, 6.15539205401741967e+01, 1.34078048103469951e+03, 1.29711309066896001e+03, 3.12851396538986274e-14] },
    // case 483: alt 926.080 km, lat -41.120, lon -178.918, doy 57, ap 147.135
    Case { doy: 57, sec: 5.73629629629629562e+04, alt: 9.26080000000000041e+02,
        g_lat: -4.11203665139525256e+01, g_long: -1.78918106686701719e+02, lst: 4.00628259948737409e+00,
        f107a: 1.34740555302685493e+02, f107: 1.84580704254019963e+02, ap: 1.47135150896632155e+02,
        ap_array: [8.31758034026464976e+00, 1.33175803402646498e+01, 1.83175803402646480e+01, 2.33175803402646480e+01, 2.83175803402646480e+01, 3.33175803402646480e+01, 3.83175803402646480e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [3.34934774133915256e+05, 1.25809929887032704e+05, 1.06451783885856585e+02, 5.13184266508980858e-01, 1.88747280380140506e-04, 5.66865401758605059e-18, 2.01566014259155818e+04, 2.78572587668560118e+03, 2.78903041695931388e+04, 1.17874758601454437e+03, 1.17874467958763671e+03, 6.40942049633044476e-18] },
    // case 484: alt 8.468 km, lat -15.406, lon -146.191, doy 239, ap 162.925
    Case { doy: 239, sec: 8.61629629629629490e+04, alt: 8.46799999999999997e+00,
        g_lat: -1.54060807996668103e+01, g_long: -1.46190833959429000e+02, lst: 2.39341563786008180e+01,
        f107a: 1.52817478379608559e+02, f107: 1.98404233665784659e+02, ap: 1.62924624580842675e+02,
        ap_array: [1.26654064272211713e+01, 1.76654064272211713e+01, 2.26654064272211713e+01, 2.76654064272211713e+01, 3.26654064272211713e+01, 3.76654064272211713e+01, 4.26654064272211713e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.41505984776470859e+13, 0.00000000000000000e+00, 8.06926804838334566e+18, 2.16474253933980032e+18, 9.65205497607373920e+16, 4.96460026642674937e-04, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.46701760280788051e+02, 4.96460026642674937e-04] },
    // case 485: alt 87.548 km, lat +10.308, lon -113.464, doy 148, ap 178.714
    Case { doy: 148, sec: 2.37037037037037010e+02, alt: 8.75480000000000018e+01,
        g_lat: 1.03082049146188979e+01, g_long: -1.13463561232156266e+02, lst: -7.49839379407790751e+00,
        f107a: 1.70894401456531625e+02, f107: 2.12227763077549383e+02, ap: 1.78714098265053224e+02,
        ap_array: [1.70132325141776946e+01, 2.20132325141776946e+01, 2.70132325141776946e+01, 3.20132325141776946e+01, 3.70132325141776946e+01, 4.20132325141776946e+01, 4.70132325141776946e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.29219458923169911e+08, 1.43810929531959229e+11, 7.23361169651343594e+13, 1.70334554693646465e+13, 8.47500504994807983e+11, 4.32709715283540639e-09, 6.69807700845959410e+07, 2.03872732005270169e+05, 1.71289246076059589e-49, 1.02731846489999998e+03, 1.59377930972444204e+02, 4.32709715283540639e-09] },
    // case 486: alt 264.704 km, lat +36.022, lon -80.736, doy 331, ap 194.504
    Case { doy: 331, sec: 2.90370370370370365e+04, alt: 2.64704000000000008e+02,
        g_lat: 3.60224906289046203e+01, g_long: -8.07362885048835466e+01, lst: 2.68342438774027325e+00,
        f107a: 1.88971324533454720e+02, f107: 2.26051292489314079e+02, ap: 1.94503571949263744e+02,
        ap_array: [2.13610586011342143e+01, 2.63610586011342143e+01, 3.13610586011342143e+01, 3.63610586011342178e+01, 4.13610586011342178e+01, 4.63610586011342178e+01, 5.13610586011342178e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], out: [1.25367860337354280e+07, 2.14918382859168434e+09, 8.33650747836285830e+08, 2.67680596766396612e+07, 8.91944481171111460e+05, 9.75119129394831182e-14, 2.38526608464935671e+04, 5.03766757330265176e+06, 7.50036767924032510e-01, 1.19946399612380992e+03, 1.16728483311815762e+03, 9.75119129594040901e-14] },
    // case 487: alt 830.080 km, lat +61.737, lon -48.009, doy 34, ap 210.293
    Case { doy: 34, sec: 5.78370370370370365e+04, alt: 8.30080000000000041e+02,
        g_lat: 6.17367763431903143e+01, g_long: -4.80090157776108128e+01, lst: 1.28652425695584576e+01,
        f107a: 2.07048247610377814e+02, f107: 2.39874821901078775e+02, ap: 2.10293045633474236e+02,
        ap_array: [2.57088846880907376e+01, 3.07088846880907376e+01, 3.57088846880907340e+01, 4.07088846880907340e+01, 4.57088846880907340e+01, 5.07088846880907340e+01, 5.57088846880907340e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1], out: [5.64141641881399672e+05, 3.91050585805653222e+06, 4.89526539311796587e+04, 5.69850920800193944e+03, 7.12650719998737414e-01, 1.18783748133384798e-16, 6.82712367578998328e+03, 3.69423732020261232e+05, 1.01470777744110037e+04, 1.55453246147473669e+03, 1.55452592393844589e+03, 1.19053254519073155e-16] },
    // case 488: alt 66.468 km, lat +87.451, lon -15.282, doy 217, ap 226.083
    Case { doy: 217, sec: 9.83703703703703650e+03, alt: 6.64680000000000035e+01,
        g_lat: 8.74510620574760367e+01, g_long: -1.52817430503380649e+01, lst: 2.73251028806584362e+00,
        f107a: 2.25125170687300880e+02, f107: 2.53698351312843471e+02, ap: 2.26082519317684785e+02,
        ap_array: [3.00567107750472573e+01, 3.50567107750472573e+01, 4.00567107750472573e+01, 4.50567107750472573e+01, 5.00567107750472573e+01, 5.50567107750472573e+01, 6.00567107750472573e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1], out: [1.97460302250276985e+10, 0.00000000000000000e+00, 2.94877606821581950e+15, 7.79346053037893125e+14, 3.52816205351356953e+13, 1.80800804705217856e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.39198216972137914e+02, 1.80800804705217856e-07] },
    // case 489: alt 79.948 km, lat -88.351, lon +17.446, doy 125, ap 241.872
    Case { doy: 125, sec: 3.86370370370370365e+04, alt: 7.99479999999999933e+01,
        g_lat: -8.83506872136609758e+01, g_long: 1.74455296769345978e+01, lst: 1.18955455998614816e+01,
        f107a: 2.43202093764223946e+02, f107: 2.67521880724608195e+02, ap: 2.41871993001895333e+02,
        ap_array: [3.44045368620037806e+01, 3.94045368620037806e+01, 4.44045368620037806e+01, 4.94045368620037806e+01, 5.44045368620037806e+01, 5.94045368620037806e+01, 6.44045368620037806e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1], out: [9.39788033754654288e+08, 2.78590568792698979e+08, 1.39350854727703969e+14, 3.50501819363183320e+13, 1.66839062650449927e+12, 8.44968818479686314e-09, 9.01064453884354793e+06, 2.06683666394881038e+04, 3.15606366914412322e-57, 1.02731846489999998e+03, 2.33636106897726819e+02, 8.44968818479686314e-09] },
    // case 490: alt 219.904 km, lat -62.636, lon +50.173, doy 308, ap 257.661
    Case { doy: 308, sec: 6.74370370370370365e+04, alt: 2.19903999999999996e+02,
        g_lat: -6.26364014993752676e+01, g_long: 5.01728024042073457e+01, lst: 2.20773637816796651e+01,
        f107a: 2.61279016841147040e+02, f107: 2.81345410136372891e+02, ap: 2.57661466686105825e+02,
        ap_array: [3.87523629489603039e+01, 4.37523629489603039e+01, 4.87523629489603039e+01, 5.37523629489603039e+01, 5.87523629489603039e+01, 6.37523629489603039e+01, 6.87523629489602968e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1], out: [2.03831279368100245e+06, 2.03037601459803700e+09, 8.15825270611180019e+09, 3.18284042332927346e+08, 5.34016048857776150e+07, 4.53827493733922952e-13, 6.89230778086870396e+03, 1.02607691394542139e+07, 1.09576925441403519e-04, 1.96924149030719059e+03, 1.65488072999385031e+03, 4.53827493733925830e-13] },
    // case 491: alt 734.080 km, lat -36.922, lon +82.900, doy 80, ap 273.451
    Case { doy: 80, sec: 1.94370370370370365e+04, alt: 7.34080000000000041e+02,
        g_lat: -3.69221157850895452e+01, g_long: 8.29000751314800937e+01, lst: 1.09258486301645164e+01,
        f107a: 2.79355939918070135e+02, f107: 2.95168939548137587e+02, ap: 2.73450940370316346e+02,
        ap_array: [4.31001890359168200e+01, 4.81001890359168200e+01, 5.31001890359168200e+01, 5.81001890359168200e+01, 6.31001890359168200e+01, 6.81001890359168272e+01, 7.31001890359168272e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1], out: [1.54777566942262975e+06, 2.25791010397721082e+07, 4.12164799644753570e+05, 2.16869359943162772e+03, 2.91703034280543356e+01, 7.12495027119391819e-16, 7.17913580751381960e+03, 3.58134242224614089e+06, 3.87088774153126462e+05, 1.70984629532156032e+03, 1.70869776450382665e+03, 7.22776104960898920e-16] },
    // case 492: alt 54.868 km, lat -11.208, lon +115.627, doy 262, ap 289.240
    Case { doy: 262, sec: 4.82370370370370438e+04, alt: 5.48680000000000092e+01,
        g_lat: -1.12078300708038370e+01, g_long: 1.15627347858752842e+02, lst: 1.33991769547325106e+01,
        f107a: 2.97432862994993172e+02, f107: 7.48056177488296328e+01, ap: 2.89240414054526866e+02,
        ap_array: [4.74480151228733433e+01, 5.24480151228733433e+01, 5.74480151228733433e+01, 6.24480151228733433e+01, 6.74480151228733433e+01, 7.24480151228733433e+01, 7.74480151228733433e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1], out: [6.01739871357415466e+10, 0.00000000000000000e+00, 8.96684515756007600e+15, 2.40553555041864150e+15, 1.07256918352154844e+14, 5.51683270295488321e-07, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.02731846489999998e+03, 2.56507031669085279e+02, 5.51683270295488321e-07] },
    // case 493: alt 117.948 km, lat +14.506, lon +148.355, doy 171, ap 5.861
    Case { doy: 171, sec: 7.70370370370370365e+04, alt: 1.17948000000000008e+02,
        g_lat: 1.45064556434818712e+01, g_long: 1.48354620586025590e+02, lst: 3.12894849938008832e+01,
        f107a: 8.19003186162949532e+01, f107: 8.86291471605943428e+01, ap: 5.86091266948534706e+00,
        ap_array: [5.17958412098298737e+01, 5.67958412098298737e+01, 6.17958412098298737e+01, 6.67958412098298737e+01, 7.17958412098298737e+01, 7.67958412098298737e+01, 8.17958412098298737e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1], out: [2.90206427735617198e+07, 1.04753008279602707e+11, 4.05118073233332886e+11, 9.40627247734865723e+10, 1.96729088735066390e+09, 2.67395942106487690e-11, 3.35061069281557016e+06, 1.20279207177435537e+06, 7.17663293231616816e-34, 1.02731846489999998e+03, 3.52960907312950781e+02, 2.67395942106487690e-11] },
    // case 494: alt 175.104 km, lat +40.221, lon -175.943, doy 353, ap 21.650
    Case { doy: 353, sec: 3.43703703703703650e+03, alt: 1.75103999999999985e+02,
        g_lat: 4.02207413577675652e+01, g_long: -1.75942900075131490e+02, lst: -1.07747941613873675e+01,
        f107a: 9.99772416932180192e+01, f107: 1.02452676572359053e+02, ap: 2.16503863536958718e+01,
        ap_array: [5.61436672967863899e+01, 6.11436672967863899e+01, 6.61436672967863899e+01, 7.11436672967863899e+01, 7.61436672967863899e+01, 8.11436672967863899e+01, 8.61436672967863899e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1], out: [2.26089003672515452e+07, 8.02135346839177036e+09, 7.43673403587016296e+09, 5.58691384357436061e+08, 5.93161168191660196e+06, 5.89718023155179436e-13, 2.83335908122503839e+05, 3.39646410781604871e+07, 1.75954723998984356e-14, 9.12947149217413084e+02, 7.91076215347783887e+02, 5.89718023155179436e-13] },
    // case 495: alt 638.080 km, lat +65.935, lon -143.216, doy 23, ap 37.440
    Case { doy: 23, sec: 3.22370370370370365e+04, alt: 6.38080000000000041e+02,
        g_lat: 6.59350270720532876e+01, g_long: -1.43215627347858742e+02, lst: -5.92975979569184020e-01,
        f107a: 1.18054164770141114e+02, f107: 1.16276205984123749e+02, ap: 3.74398600379063993e+01,
        ap_array: [6.04914933837429132e+01, 6.54914933837429203e+01, 7.04914933837429203e+01, 7.54914933837429203e+01, 8.04914933837429203e+01, 8.54914933837429203e+01, 9.04914933837429203e+01],
        switches: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], out: [2.06726391247916082e+06, 1.53113238051511976e+06, 5.38728985607943287e+03, 9.07755350740416844e+01, 2.38397487087014491e-02, 5.50601324919143257e-17, 8.18039469130405778e+04, 1.18590672462324856e+04, 4.59737501452189917e+03, 9.75162934262814929e+02, 9.75162115288806149e+02, 5.51822387723000267e-17] },
    ];

    /// With turbulent mixing switched off below `72.5 km`, the model still returns a density.
    ///
    /// [`gtd7`] sizes its own mixing ramp as `d_diffusive / d_mixed − 1`, and `gts7` produces
    /// `d_mixed` only when `switches[15]` is on. The reference keeps that density in a file-scope
    /// `static`, so with mixing off it divides by whatever the previous call happened to leave
    /// there — an answer that depends on the call order — and by zero if there was no previous
    /// call. Below `zmix = 62.5 km` the ramp weight is exactly `0`, and `0 · ∞` is a NaN, so the
    /// whole lower atmosphere comes back as NaN on a fresh process.
    ///
    /// Threading the value out of `gts7` makes the gap explicit and fills it with the only thing
    /// "no mixing" can mean: with no mixed profile to compare against, the diffusive density *is*
    /// the reference density and the ramp is the identity. Asserted here as an exact equality
    /// against the mixing-on evaluation at the same point, since the two differ only in `dnet` and
    /// `dnet` is what `switches[15]` turns off.
    #[test]
    fn switching_off_the_turbopause_still_leaves_a_density_below_it() {
        let mut flags = Nrlmsise00Flags::default();
        let base = Nrlmsise00Input {
            year: 0,
            doy: 172,
            sec: 29000.0,
            alt: 0.0,
            g_lat: 60.0,
            g_long: -70.0,
            lst: 16.0,
            f107a: 150.0,
            f107: 150.0,
            ap: 4.0,
            ap_array: None,
        };
        // Both sides of `zmix`, where the ramp weight goes from zero to nonzero, and the boundary
        // itself; `72.5` is the first altitude `gtd7` hands straight to the thermosphere.
        for alt in [0.0, 30.0, 62.5, 65.0, 72.0] {
            let input = Nrlmsise00Input { alt, ..base };
            flags.switches[15] = 0;
            let off = gtd7d(&input, &flags);
            for (k, d) in off.d.iter().enumerate() {
                assert!(d.is_finite(), "density {k} = {d:e} at {alt} km, mixing off");
            }
            assert!(off.d[5] > 0.0, "mass density {:e} at {alt} km", off.d[5]);

            flags.switches[15] = 1;
            let on = gtd7d(&input, &flags);
            // Mixing off is not the same model — `dnet` no longer blends the two N2 profiles — so
            // the densities differ. What must *not* differ is the temperature, which never sees
            // `switches[15]`, and the finiteness above.
            assert_eq!(off.t, on.t, "the temperature moved at {alt} km");
            assert!(on.d[5] > 0.0);
        }
    }

    /// The turbopause blend is a power mean, and the closed form says so without a logarithm.
    ///
    /// `dnet` computes `dd·(1 + e^{a ln(dm/dd)})^{1/a}` — chosen for range, since the model's
    /// densities span sixty orders — but that expression is algebraically
    /// `(ddᵃ + dmᵃ)^{1/a}`, the power mean of order `a = zhm/(xmm − xm)`. The closed form shares no
    /// operation with the code (no `log`, no `exp`) and carries `a` from its own definition, so a
    /// swapped operator in either the exponent or the blend moves one and not the other.
    ///
    /// The two saturating arms are *approximations*, not limits: at `|ylog| = 10` the discarded
    /// factor is `(1 ± e^{-10})^{1/a}`, about `4.5e-5/a`. They are held to that, which pins both the
    /// `10.0` cut and its direction — a flipped comparison returns the wrong endpoint entirely.
    #[test]
    fn the_turbopause_blend_is_the_power_mean_of_its_two_densities() {
        let mean =
            |dd: f64, dm: f64, a: f64| libm::pow(libm::pow(dd, a) + libm::pow(dm, a), 1.0 / a);
        let mut worst = 0.0f64;
        for (zhm, xmm, xm) in [
            (5.5e1, 2.8e1, 1.6e1),
            (1.0e2, 3.2e1, 4.0e0),
            (7.0e0, 2.0e1, 1.4e1),
            (4.0e1, 2.8e1, 1.0e0),
        ] {
            let a = zhm / (xmm - xm);
            for dd in [1.0e-3, 1.0, 7.5, 1.0e3] {
                for dm in [2.0e-3, 3.0, 1.1e1, 5.0e2] {
                    // Stay clear of the `|ylog| > 10` arms, which are tested on their own below.
                    if libm::fabs(a * libm::log(dm / dd)) > 9.0 {
                        continue;
                    }
                    let (got, want) = (dnet(dd, dm, zhm, xmm, xm), mean(dd, dm, a));
                    worst = nan_max(worst, libm::fabs((got - want) / want));
                }
            }
        }
        assert!(worst < 1.0e-13, "power-mean residual {worst:e}");

        // The saturating arms, from both sides of each cut. `a = 55/(28−16) = 4.583`, so `ylog`
        // reaches ±10 at `dm/dd = e^{±10/a}`.
        let (zhm, xmm, xm) = (5.5e1, 2.8e1, 1.6e1);
        let a = zhm / (xmm - xm);
        for sign in [-1.0, 1.0] {
            for nudge in [0.98, 1.02] {
                let dd = 1.0;
                let dm = libm::exp(sign * 10.0 * nudge / a);
                let (got, want) = (dnet(dd, dm, zhm, xmm, xm), mean(dd, dm, a));
                let miss = libm::fabs((got - want) / want);
                assert!(miss < 1.0e-4, "arm sign {sign} nudge {nudge}: {miss:e}");
                // …and the arm really is the endpoint, not the blend.
                if nudge > 1.0 {
                    assert_eq!(got, if sign < 0.0 { dd } else { dm });
                }
            }
        }

        // The four zero combinations, exactly. `dd = dm = 0` is the "both species vanished" case the
        // reference resolves to 1; the one-sided cases return whichever survived.
        assert_eq!(dnet(0.0, 0.0, zhm, xmm, xm), 1.0);
        assert_eq!(dnet(5.0, 0.0, zhm, xmm, xm), 5.0);
        assert_eq!(dnet(0.0, 7.0, zhm, xmm, xm), 7.0);
        let (got, want) = (dnet(5.0, 7.0, zhm, xmm, xm), mean(5.0, 7.0, a));
        assert!(libm::fabs(got / want - 1.0) < 1.0e-13, "{got} vs {want}");

        // The two cuts on the boundary itself. `zhm = 10` over `xmm − xm = 1` makes `a = 10`
        // exactly, and `log(exp(±1))` round-trips to `±1` in `f64`, so `ylog` lands *on* `±10`
        // rather than near it — the only place a strict comparison is distinguishable from a
        // non-strict one. Neither arm fires at the tie: the reference falls through to the blend.
        let (zhm, xmm, xm) = (1.0e1, 2.0e0, 1.0e0);
        let a = zhm / (xmm - xm);
        for sign in [-1.0f64, 1.0] {
            let (dd, dm) = (1.0, libm::exp(sign));
            assert_eq!(a * libm::log(dm / dd), sign * 10.0, "the tie is not exact");
            let (got, want) = (dnet(dd, dm, zhm, xmm, xm), mean(dd, dm, a));
            assert!(
                libm::fabs(got / want - 1.0) < 1.0e-13,
                "sign {sign}: {got} vs the blend {want}"
            );
            assert_ne!(
                got,
                if sign < 0.0 { dd } else { dm },
                "sign {sign}: the arm"
            );
        }

        // A signed density is not a density, and the model says so by taking its logarithm: the
        // guard above resolves only the *zero* combinations, so a negative argument reaches
        // `log(dm/dd)` and leaves as a NaN. Unreachable from the model — `densu` returns `0.0`
        // rather than a negative density — but it is what fixes the roles of the two zero tests in
        // that guard, which are otherwise interchangeable with any other comparison.
        assert!(dnet(-5.0, 7.0, zhm, xmm, xm).is_nan());
        assert!(dnet(5.0, -7.0, zhm, xmm, xm).is_nan());
    }

    /// The chemistry corrections are logistics in `(alt − zh)/h`, pinned by an identity and by three
    /// landmark points.
    ///
    /// `ln ccor` is `r/(1 + e^{(alt−zh)/h1})`, so `ln ccor(zh+d) + ln ccor(zh−d) = r` for **every**
    /// `d` — the two logistic weights are complements. That identity fixes the roles of `alt`, `zh`
    /// and the `1 +`, but it survives `h1` being multiplied instead of divided, because it is even in
    /// `d/h1`. The landmarks close that: at `alt = zh + h1 ln k` the exponential is exactly `k`, so
    /// `ccor = e^{r/(1+k)}` — a closed form in which `h1` appears alone.
    ///
    /// `ccor2` collapses onto `ccor` when both scale heights agree, since `½(eᵉ + eᵉ) = eᵉ`. That is
    /// an exact, bit-for-bit cross-check of the `½`, the `+`, and the second exponent.
    #[test]
    fn the_chemistry_corrections_are_logistics_with_the_landmarks_to_prove_it() {
        let (r, h1, zh) = (-1.7, 8.5, 137.0);

        for d in [0.0, 1.0, 9.5, 40.0, 200.0] {
            let sum = libm::log(ccor(zh + d, r, h1, zh)) + libm::log(ccor(zh - d, r, h1, zh));
            assert!(libm::fabs(sum - r) < 1.0e-14, "d = {d}: {sum}");
        }
        for k in [1.0f64 / 3.0, 1.0, 3.0, 20.0] {
            let alt = zh + h1 * libm::log(k);
            let (got, want) = (ccor(alt, r, h1, zh), libm::exp(r / (1.0 + k)));
            assert!(
                libm::fabs(got / want - 1.0) < 1.0e-14,
                "k = {k}: {got} vs {want}"
            );
        }
        // The two saturating arms are exact in `f64`: `e^{±70}` puts the correction `30` orders under
        // an ulp of the value it corrects, so the switch cannot be seen in the answer — assert that
        // rather than assume it, from both sides of each cut.
        for e in [-70.5, -69.5, 69.5, 70.5] {
            let alt = zh + h1 * e;
            let want = if e > 0.0 { 1.0 } else { libm::exp(r) };
            assert_eq!(ccor(alt, r, h1, zh), want, "e = {e}");
        }
        // Stronger, and the reason no comparison *inside* those arms is observable: the continuous
        // form reproduces the arm **bit for bit** at the cut, past it, and at an infinite exponent.
        // Above `+70`, `r/(1 + e^e)` is `30` orders below an ulp of zero and `exp` of it is exactly
        // `1`; below `−70`, `1 + e^e` is exactly `1` and the quotient is `r` itself. Whether the
        // reference tests `>` or `≥`, `<` or `≤`, and where it puts the boundary, changes no digit.
        //
        // The `r` loop is what makes that a statement about the code rather than about one
        // coefficient. The upper arm stops being exact once `|r| / e^70` reaches an ulp of one —
        // near `1.4e14` — and `1e10` is four orders below that and ten above any `r` the model
        // builds, all of which are logarithms of ratios of table entries.
        for r in [-1.0e10f64, -1.7, 0.0, 1.7, 1.0e10] {
            for e in [
                f64::NEG_INFINITY,
                -1.0e6,
                -70.5,
                -70.0,
                -69.5,
                0.0,
                69.5,
                70.0,
                70.5,
                1.0e6,
                f64::INFINITY,
            ] {
                let continuous = libm::exp(r / (1.0 + libm::exp(e)));
                assert_eq!(ccor(e, r, 1.0, 0.0), continuous, "ccor at e = {e}, r = {r}");
                assert_eq!(
                    ccor2(e, r, 1.0, 0.0, 1.0),
                    continuous,
                    "ccor2 at e = {e}, r = {r}"
                );
            }
            // `ccor2`'s two arms are `||` over the exponents above and `&&` below, and neither
            // choice is visible while the scale heights agree — the loop above drives `e1 == e2`
            // throughout. With `alt − zh = 1` the exponents are the reciprocals of the two scale
            // heights, so these five pairs put one past a cut and the other short of it, in both
            // directions and on both sides.
            for (h1, h2) in [
                (0.01, 0.2),
                (0.2, 0.01),
                (-0.01, -0.2),
                (-0.2, -0.01),
                (0.01, -0.01),
            ] {
                let (e1, e2) = (1.0 / h1, 1.0 / h2);
                let continuous = libm::exp(r / (1.0 + 0.5 * (libm::exp(e1) + libm::exp(e2))));
                assert_eq!(
                    ccor2(1.0, r, h1, 0.0, h2),
                    continuous,
                    "ccor2 at ({e1}, {e2}), r = {r}"
                );
            }
        }

        // ccor2 with one scale height is ccor.
        for alt in [zh - 300.0, zh - 20.0, zh, zh + 20.0, zh + 300.0] {
            assert_eq!(
                ccor2(alt, r, h1, zh, h1),
                ccor(alt, r, h1, zh),
                "alt = {alt}"
            );
        }
        // …and with two, the denominator is the mean of the two exponentials.
        let h2 = 2.0 * h1;
        for k in [1.0f64 / 3.0, 1.0, 3.0] {
            let alt = zh + h1 * libm::log(k);
            let want = libm::exp(r / (1.0 + 0.5 * (k + libm::sqrt(k))));
            let got = ccor2(alt, r, h1, zh, h2);
            assert!(
                libm::fabs(got / want - 1.0) < 1.0e-14,
                "k = {k}: {got} vs {want}"
            );
        }
        // The upper arm fires when *either* exponent is large, the lower only when *both* are small.
        // The mixed cases are the ones an `||`/`&&` swap gets wrong, so both are checked against the
        // continuous formula they have to agree with.
        for (e1, e2) in [
            (80.0, 1.0),
            (1.0, 80.0),
            (-80.0, -1.0),
            (-1.0, -80.0),
            // …and with one exponent exactly on a cut, where a `>`/`>=` swap decides which side of
            // the boundary the pair falls on.
            (70.0, 1.0),
            (1.0, 70.0),
            (-70.0, -71.0),
            (-71.0, -70.0),
        ] {
            // Fix the offset and read the two scale heights off the exponents, so both stay positive
            // whichever side of `zh` the point is on.
            let delta = 100.0 * if e1 > 0.0 { 1.0 } else { -1.0 };
            let (alt, h1, h2) = (zh + delta, delta / e1, delta / e2);
            let want = libm::exp(r / (1.0 + 0.5 * (libm::exp(e1) + libm::exp(e2))));
            let got = ccor2(alt, r, h1, zh, h2);
            assert!(
                libm::fabs(got - want) <= 4.0 * f64::EPSILON * libm::fabs(want),
                "e1 {e1} e2 {e2}: {got} vs {want}"
            );
        }
    }

    /// A clamped cubic spline through samples of a cubic **is** that cubic — so the polynomial is the
    /// oracle for all three spline routines at once.
    ///
    /// The interpolation conditions, `C²` continuity and the two end slopes determine the spline
    /// uniquely, and the sampled polynomial satisfies every one of them. That makes `P` an exact
    /// external answer for `spline`'s second derivatives, for `splint`'s values *between* knots,
    /// and — integrating it once — for `splini`. Knots are deliberately unequally spaced: with a
    /// uniform grid, `x[i+1] − x[i]` and `x[i] − x[i−1]` are the same number and half the arithmetic
    /// in `spline` becomes untestable. They also deliberately do **not** start at zero, for the same
    /// reason one place further in: `x[1] − x[0]` and `x[1] + x[0]` are the same number when `x[0]`
    /// is `0`, and the clamped end condition is built out of exactly that difference.
    #[test]
    fn a_clamped_spline_through_a_cubic_reproduces_that_cubic_exactly() {
        // P(x) = 0.37x³ − 1.9x² + 2.4x − 0.55, its derivative, second derivative and integral.
        let p = |x: f64| 0.37 * x * x * x - 1.9 * x * x + 2.4 * x - 0.55;
        let dp = |x: f64| 1.11 * x * x - 3.8 * x + 2.4;
        let d2p = |x: f64| 2.22 * x - 3.8;
        let ip = |x: f64| 0.0925 * x * x * x * x - 1.9 / 3.0 * x * x * x + 1.2 * x * x - 0.55 * x;

        let n = 5usize;
        let xa = [0.4, 1.1, 2.3, 3.5, 4.4];
        let ya: [f64; 5] = core::array::from_fn(|i| p(xa[i]));
        let mut y2 = [0.0f64; 5];
        spline(&xa, &ya, n, dp(xa[0]), dp(xa[n - 1]), &mut y2);

        for i in 0..n {
            let want = d2p(xa[i]);
            assert!(
                libm::fabs(y2[i] - want) < 1.0e-12 * libm::fabs(want),
                "y2[{i}] = {} vs {want}",
                y2[i]
            );
        }
        let mut worst_v = 0.0f64;
        let mut worst_i = 0.0f64;
        for k in 0..=40 {
            let x = xa[0] + (xa[n - 1] - xa[0]) * f64::from(k) / 40.0;
            worst_v = nan_max(worst_v, libm::fabs(splint(&xa, &ya, &y2, n, x) - p(x)));
            worst_i = nan_max(
                worst_i,
                libm::fabs(splini(&xa, &ya, &y2, n, x) - (ip(x) - ip(xa[0]))),
            );
        }
        assert!(worst_v < 1.0e-13, "splint residual {worst_v:e}");
        assert!(worst_i < 1.0e-13, "splini residual {worst_i:e}");
        // The knots themselves, exactly what interpolation means.
        for i in 0..n {
            let got = splint(&xa, &ya, &y2, n, xa[i]);
            assert!(
                libm::fabs(got - ya[i]) < 1.0e-13,
                "knot {i}: {got} vs {}",
                ya[i]
            );
        }
        // `splini` starts at zero: below the first knot there is no interval to integrate over.
        assert_eq!(splini(&xa, &ya, &y2, n, xa[0]), 0.0);

        // Past the last knot **neither routine stops**. `splint`'s bisection settles on the final
        // interval and evaluates its cubic there with `b > 1`; `splini` clamps its upper limit to
        // `xa[khi]` on every interval *except* the last, so the last one is integrated out to `x`.
        // Both are extrapolations of the same cubic, so the polynomial is still the exact answer —
        // which is the only reason the omitted clamp is visible at all. The model never asks:
        // `densu` and `densm` clamp the altitude into the node range before they call.
        for x in [xa[n - 1], 4.6, 5.0, 6.0] {
            let (v, i) = (splint(&xa, &ya, &y2, n, x), splini(&xa, &ya, &y2, n, x));
            assert!(
                libm::fabs(v - p(x)) < 1.0e-12,
                "splint({x}) = {v} vs {}",
                p(x)
            );
            assert!(
                libm::fabs(i - (ip(x) - ip(xa[0]))) < 1.0e-12,
                "splini({x}) = {i} vs {}",
                ip(x) - ip(xa[0])
            );
        }

        // The natural arm: a slope above `0.99e30` means "no end condition", i.e. zero curvature
        // there. It is the one branch the model never takes — `densu` and `densm` always pass a
        // finite end slope — so nothing else in this crate would notice if it were wrong.
        let mut y2n = [0.0f64; 5];
        spline(&xa, &ya, n, 1.0e30, 1.0e30, &mut y2n);
        assert_eq!(y2n[0], 0.0);
        assert_eq!(y2n[n - 1], 0.0);
        // A natural spline is *not* the cubic, or the clamped end conditions would be doing nothing.
        assert!(libm::fabs(y2n[1] - d2p(xa[1])) > 1.0e-3);
        // `0.99e30` is the sentinel's *exclusive* floor: the natural arm wants a slope strictly
        // above it, so the threshold value itself still clamps — with that absurd slope as the end
        // condition, which is what makes the two arms tell apart at the tie.
        let mut y2t = [0.0f64; 5];
        spline(&xa, &ya, n, 0.99e30, 0.99e30, &mut y2t);
        assert!(
            y2t[0] < -1.0e30,
            "yp1 on the threshold went natural: {}",
            y2t[0]
        );
        assert!(
            y2t[n - 1] > 1.0e30,
            "ypn on the threshold went natural: {}",
            y2t[n - 1]
        );
    }

    /// Every altitude gate in the two density routines belongs to the branch *below* it.
    ///
    /// Both are piecewise. `densm` hands the caller's own `d0` straight back above `zn2[0]`
    /// (`72.5 km`) and switches to the troposphere nodes at or below `zn3[0]` (`32.5 km`);
    /// `densu` evaluates the Bates profile at and above `zn1[0]` (`120 km`) and the spline below
    /// it. Every one of those comparisons is strict, so the node altitude itself is handled by the
    /// lower branch — a spline evaluated one ulp outside its own node range is the failure that
    /// getting it backwards produces. Nothing else here can see it: the reference vectors and the
    /// sweep are Halton draws, and none of them lands on a boundary exactly.
    #[test]
    fn the_altitude_gates_belong_to_the_branch_below_them() {
        let (gsurf, re) = glatf(45.0);
        let (mn2, mn3) = (4usize, 5usize);
        let (zn2, zn3) = (ZN2, ZN3);
        let tn2 = [208.0, 258.0, 266.0, 230.0];
        let tgn2 = [-2.4, 1.6];
        // The shared `32.5 km` node is deliberately given two *different* temperatures. `gtd7`
        // always sets `tn3[0] = tn2[mn2−1]`, and while that holds, which branch owns the boundary
        // is a question about the last ulp of `1/(1/T)`. Breaking it is what makes the gate visible.
        let tn3 = [245.0, 215.0, 210.0, 220.0, 288.0];
        let tgn3 = [1.6, -6.2];
        let d0 = 3.5e17;
        let densm_at = |alt: f64, xm: f64| {
            let mut held = false;
            let out = densm(
                gsurf, re, alt, d0, xm, mn3, &zn3, &tn3, &tgn3, mn2, &zn2, &tn2, &tgn2, 1.0e4,
                &mut held,
            );
            // A ceiling far above any node here, so the exospheric hold cannot fire and the numbers
            // below are the spline's own.
            assert!(!held, "the spline was held at {alt} km");
            out
        };

        // Above the top node the routine is a pass-through: the caller's density back unchanged,
        // and a temperature slot that was never written.
        assert_eq!(densm_at(zn2[0].next_up(), 0.0), (0.0, 0.0));
        assert_eq!(densm_at(zn2[0].next_up(), 28.9), (d0, 0.0));
        // At the node itself the stratosphere spline runs, and lands on that node's own temperature
        // to within the `1/(1/T)` round trip. The density is still `d0` there — the profile is
        // anchored at the top node, so the integral it exponentiates is empty — which is why the
        // temperature, and not the density, is what tells the two sides of this gate apart.
        let (rho, tz) = densm_at(zn2[0], 28.9);
        assert!(
            libm::fabs(tz / tn2[0] - 1.0) < 1.0e-15,
            "tz {tz} vs node {}",
            tn2[0]
        );
        assert_eq!(
            rho, d0,
            "the profile is not anchored at the caller's density"
        );
        assert_eq!(
            densm_at(zn2[0], 0.0).0,
            tz,
            "xm = 0 returns the temperature"
        );

        // The lower gate the same way round: `zn3[0]` belongs to the troposphere nodes, and one ulp
        // above it to the stratosphere ones.
        let ratio = |alt: f64, node: f64| libm::fabs(densm_at(alt, 0.0).1 / node - 1.0);
        assert!(ratio(zn3[0], tn3[0]) < 1.0e-9, "{}", ratio(zn3[0], tn3[0]));
        assert!(ratio(zn3[0].next_up(), tn2[mn2 - 1]) < 1.0e-9);

        // `densu`'s gate at `za = zn1[0]` is the one with a side effect: the branch below it writes
        // the Bates end node into `tn1[0]`/`tgn1[0]` for the spline that follows to read back. At
        // `alt == za` the reference does not take it, so both slots keep whatever the caller left
        // there — which is how the branch is observable even where the two temperatures agree.
        let zn1 = [120.0, 110.0, 100.0, 90.0, 72.5];
        let (tinf, tlb, zlb, s2) = (1.0e3, 386.0, 120.0, 2.0e-2);
        let sentinel = -1.0;
        let mut tn1 = [sentinel, 350.0, 250.0, 200.0, 190.0];
        let mut tgn1 = [sentinel, -3.0];
        let densu_at = |alt: f64, tn1: &mut [f64; 5], tgn1: &mut [f64; 2]| {
            let mut held = false;
            let out = densu(
                gsurf, re, alt, 2.0e10, tinf, tlb, 0.0, 0.0, zlb, s2, 5, &zn1, tn1, tgn1, &mut held,
            );
            assert!(!held, "the spline was held at {alt} km");
            out
        };
        let at_node = densu_at(zn1[0], &mut tn1, &mut tgn1);
        assert_eq!(
            (tn1[0], tgn1[0]),
            (sentinel, sentinel),
            "the spline branch ran at the node"
        );
        assert_eq!(at_node.1, tlb, "the Bates profile at its own base");
        let below = densu_at(zn1[0].next_down(), &mut tn1, &mut tgn1);
        assert_ne!(tn1[0], sentinel, "the end node was not written below `za`");
        assert_ne!(tgn1[0], sentinel);
        assert!(
            libm::fabs(below.1 / at_node.1 - 1.0) < 1.0e-12,
            "the two branches do not join: {} vs {}",
            below.1,
            at_node.1
        );
    }

    /// Each of the three places the `1/T` spline is evaluated reports its own hold, and none of them
    /// can lose one already reported.
    ///
    /// [`Nrlmsise00Output::out_of_range`] is an OR over four accumulation sites — two in `densm`,
    /// one in `densu`, one in [`temperature_node`] — and an end-to-end test cannot tell them apart:
    /// any single site still raises the flag on its own, so the other three could be silently
    /// dropped. Hence a fixture per site, each with the bound placed so that *only* that site can
    /// engage, and each run twice: once from a clear flag, and once from a flag another site has
    /// already set. The second run is the one that matters — accumulating with `^` instead of `|`
    /// agrees everywhere except there, where it would erase the earlier report.
    #[test]
    fn each_interpolation_site_reports_its_own_hold() {
        let (gsurf, re) = glatf(45.0);
        let (mn2, mn3) = (4usize, 5usize);
        let (zn2, zn3) = (ZN2, ZN3);
        let tn2 = [208.0, 258.0, 266.0, 230.0];
        let tgn2 = [-2.4, 1.6];
        // `tn3[0] == tn2[mn2−1]`, the join `gtd7` always establishes.
        let tn3 = [230.0, 215.0, 210.0, 220.0, 288.0];
        let tgn3 = [1.6, -6.2];
        let densm_at = |alt: f64, t_max: f64, seed: bool| {
            let mut held = seed;
            let (_rho, tz) = densm(
                gsurf, re, alt, 3.5e17, 28.9, mn3, &zn3, &tn3, &tgn3, mn2, &zn2, &tn2, &tgn2,
                t_max, &mut held,
            );
            (tz, held)
        };
        let is = |tz: f64, want: f64| libm::fabs(tz / want - 1.0) <= 2.0 * f64::EPSILON;

        // The mesospheric spline, at its `55 km` node where the profile is `258 K`. A ceiling above
        // every node is inert; one below that node engages, and only here — the stratospheric spline
        // does not run above `zn3[0]`.
        let (tz, held) = densm_at(55.0, 1.0e4, false);
        assert!(is(tz, tn2[1]) && !held, "{tz} K, held {held}");
        let (tz, held) = densm_at(55.0, 240.0, false);
        assert!(is(tz, 240.0) && held, "{tz} K, held {held}");
        assert!(densm_at(55.0, 240.0, true).1, "an earlier hold was erased");
        assert!(densm_at(55.0, 1.0e4, true).1, "an earlier hold was dropped");

        // The stratospheric spline, at the ground where the profile is `288 K`. Below `zn3[0]` the
        // mesospheric spline is evaluated at `zn2[mn2−1]` — its own last node, `230 K` — whatever the
        // altitude, so a `280 K` ceiling can only be reached by the stratospheric one.
        let (tz, held) = densm_at(32.5, 280.0, false);
        assert!(is(tz, tn3[0]) && !held, "{tz} K, held {held}");
        let (tz, held) = densm_at(0.0, 1.0e4, false);
        assert!(is(tz, tn3[4]) && !held, "{tz} K, held {held}");
        let (tz, held) = densm_at(0.0, 280.0, false);
        assert!(is(tz, 280.0) && held, "{tz} K, held {held}");
        assert!(densm_at(0.0, 280.0, true).1, "an earlier hold was erased");
        assert!(densm_at(0.0, 1.0e4, true).1, "an earlier hold was dropped");

        // The thermospheric spline, at its `110 km` node where the profile is `350 K`. Here the
        // ceiling is the routine's own `tinf` rather than a separate argument.
        let zn1 = [120.0, 110.0, 100.0, 90.0, 72.5];
        let densu_at = |tinf: f64, seed: bool| {
            let mut tn1 = [0.0, 350.0, 250.0, 200.0, 190.0];
            let mut tgn1 = [0.0, -3.0];
            let mut held = seed;
            let (_d, tz) = densu(
                gsurf, re, 110.0, 2.0e10, tinf, 386.0, 0.0, 0.0, 120.0, 2.0e-2, 5, &zn1, &mut tn1,
                &mut tgn1, &mut held,
            );
            (tz, held)
        };
        let (tz, held) = densu_at(1.0e3, false);
        assert!(is(tz, 350.0) && !held, "{tz} K, held {held}");
        let (tz, held) = densu_at(300.0, false);
        assert!(is(tz, 300.0) && held, "{tz} K, held {held}");
        assert!(densu_at(300.0, true).1, "an earlier hold was erased");
        assert!(densu_at(1.0e3, true).1, "an earlier hold was dropped");

        // And the node bound, the fourth site, which reports through the same flag.
        let mut held = true;
        assert_eq!(temperature_node(200.0, 0.5, 1000.0, &mut held), 400.0);
        assert!(held, "an earlier hold was dropped");
    }

    /// The model's own arithmetic, unbounded, leaves the `109..120 km` polar storm corner without a
    /// temperature — and so without a density.
    ///
    /// Both mesospheric node forms are quotients whose denominator the fit does not constrain, and
    /// past `ap ≈ 300` at `|φ| ≥ 60°` the geomagnetic terms drive it through zero: the reference C
    /// returns `t[1] = −8186 K` at `113 km / 90°S / ap 400`, prints `dnet log error`, and hands back
    /// a NaN mass density. Swept over `8 578 080` points of the `60..140 km` band — every half
    /// kilometre, `5°` in latitude, `15°` in longitude, four seasons, five `ap` and three `F10.7` —
    /// the unbounded form fails `50 683` of them (`0.59 %`), all inside this box; with
    /// [`temperature_node`] and the `densu` spline guard in place, **zero**, and the coldest
    /// temperature anywhere in the sweep is `113.9 K`.
    ///
    /// This grid is the failing box at the sweep's own resolution. It is a regression test for two
    /// separate divergences from the reference, so it asserts the physics rather than a number: a
    /// temperature is positive and finite, a number density is non-negative and finite.
    #[test]
    fn the_polar_storm_corner_returns_a_temperature_and_a_density() {
        let flags = Nrlmsise00Flags::default();
        let mut hottest = 0.0f64;
        let mut checked = 0usize;
        let mut held = false;
        let mut alt = 109.0;
        while alt <= 120.0 {
            for lat in [
                -90.0, -85.0, -80.0, -75.0, -70.0, -65.0, -60.0, 60.0, 75.0, 90.0,
            ] {
                for g_long in [-180.0, 0.0] {
                    for ap in [300.0, 400.0] {
                        for f107 in [65.0, 250.0] {
                            for doy in [105, 196] {
                                let input = Nrlmsise00Input {
                                    year: 0,
                                    doy,
                                    sec: 29000.0,
                                    alt,
                                    g_lat: lat,
                                    g_long,
                                    lst: 16.0,
                                    f107a: f107,
                                    f107,
                                    ap,
                                    ap_array: None,
                                };
                                let o = gtd7d(&input, &flags);
                                let at = format!(
                                    "{alt} km, {lat}°, {g_long}°, doy {doy}, ap {ap}, F10.7 {f107}"
                                );
                                assert!(
                                    o.t[1] > 0.0 && o.t[1].is_finite(),
                                    "temperature {} at {at}",
                                    o.t[1]
                                );
                                assert!(
                                    o.t[1] <= o.t[0],
                                    "{} K at {at}, over the {} K exosphere",
                                    o.t[1],
                                    o.t[0]
                                );
                                held |= o.out_of_range;
                                for (k, d) in o.d.iter().enumerate() {
                                    assert!(
                                        d.is_finite() && *d >= 0.0,
                                        "density {k} = {d:e} at {at}"
                                    );
                                }
                                hottest = hottest.max(o.t[1]);
                                checked += 1;
                            }
                        }
                    }
                }
            }
            alt += 0.5;
        }
        // Not vacuous: the box really is the disturbed thermosphere, not a patch of zeros, and it
        // does reach the corner the bounds exist for — otherwise the assertions above would be
        // testing the ordinary model and passing for the wrong reason.
        assert_eq!(checked, 3680);
        assert!(hottest > 1_000.0, "hottest {hottest}");
        assert!(held, "no evaluation in the storm box engaged the bound");
    }

    /// The bounds are inert in the quiet atmosphere: over ordinary space weather the model is
    /// returned exactly as it was fitted, and the flag stays clear.
    ///
    /// This is the compatibility half of the guarantee, and it is asserted on the flag rather than
    /// against the reference C because the flag is what the diverging code sets. The reference
    /// agreement itself is pinned at `%.17e` by [`the_sweep_matches_the_reference_c_at_full_precision`]
    /// over the same domain. Swept more finely off-line — 11.0 million points, `0.5 km` resolution
    /// through the mesosphere, `ap` to `400` — the first activation anywhere is at `ap = 160` and
    /// `|φ| = 50°`, so the `ap ≤ 150` box here has margin rather than sitting on the edge.
    #[test]
    fn the_bounds_are_inert_in_the_quiet_atmosphere() {
        let flags = Nrlmsise00Flags::default();
        let mut checked = 0usize;
        let mut alt = 0.0;
        while alt <= 1000.0 {
            let mut lat = -90.0;
            while lat <= 90.0 {
                for ap in [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0] {
                    for f107 in [70.0, 250.0] {
                        let input = Nrlmsise00Input {
                            year: 0,
                            doy: 196,
                            sec: 29000.0,
                            alt,
                            g_lat: lat,
                            g_long: -180.0,
                            lst: 16.0,
                            f107a: f107,
                            f107,
                            ap,
                            ap_array: None,
                        };
                        let o = gtd7d(&input, &flags);
                        let at = format!("{alt} km, {lat}°, ap {ap}, F10.7 {f107}");
                        assert!(!o.out_of_range, "the bound engaged at {at}");
                        assert!(o.t[1] > 0.0 && o.t[1] <= o.t[0], "{} K at {at}", o.t[1]);
                        for (k, d) in o.d.iter().enumerate() {
                            assert!(d.is_finite() && *d >= 0.0, "density {k} = {d:e} at {at}");
                        }
                        checked += 1;
                    }
                }
                lat += 15.0;
            }
            alt += if (60.0..200.0).contains(&alt) {
                2.0
            } else {
                25.0
            };
        }
        assert_eq!(checked, 17_836);
    }

    /// Bounding the temperature does not put a step in the density.
    ///
    /// A clamp that engages over part of an altitude band is exactly the shape of change that can
    /// introduce a discontinuity, and a discontinuous density is worse for a drag integrator than a
    /// coarse one. Asserted by refinement rather than against a threshold: halving the sampling step
    /// halves the largest increment iff the profile is continuous, whereas a jump would survive at
    /// any step. Held columns and quiet ones are held to the same ratio, so the test says the bound
    /// changed nothing about the smoothness rather than merely that the profile is smooth somewhere.
    #[test]
    fn the_bounded_profile_has_no_step_in_it() {
        let flags = Nrlmsise00Flags::default();
        let sampled = |lat: f64, ap: f64, f107: f64, step: f64| {
            let at = |alt: f64| Nrlmsise00Input {
                year: 0,
                doy: 105,
                sec: 29000.0,
                alt,
                g_lat: lat,
                g_long: -180.0,
                lst: 16.0,
                f107a: f107,
                f107,
                ap,
                ap_array: None,
            };
            let (mut worst, mut engaged) = (0.0f64, false);
            let mut z = 100.0;
            let mut previous = gtd7(&at(z), &flags);
            while z < 130.0 {
                z += step;
                let o = gtd7(&at(z), &flags);
                engaged |= o.out_of_range;
                worst = worst.max(libm::fabs(o.d[5] / previous.d[5] - 1.0));
                previous = o;
            }
            (worst, engaged)
        };

        for (lat, ap, f107, expect_held) in [
            (-90.0, 300.0, 250.0, true),
            (-90.0, 400.0, 250.0, true),
            (90.0, 400.0, 70.0, true),
            (70.0, 350.0, 250.0, true),
            (-90.0, 4.0, 150.0, false),
            (45.0, 4.0, 150.0, false),
        ] {
            let (coarse, engaged) = sampled(lat, ap, f107, 0.1);
            let (fine, _) = sampled(lat, ap, f107, 0.05);
            assert_eq!(engaged, expect_held, "at {lat}°, ap {ap}, F10.7 {f107}");
            let ratio = coarse / fine;
            assert!(
                (1.9..2.1).contains(&ratio),
                "step {coarse:e} against {fine:e} is a ratio of {ratio}, not a halving, \
                 at {lat}°, ap {ap}, F10.7 {f107}"
            );
        }
    }

    /// [`temperature_node`] is the identity above the bound and pins to the bound below it.
    ///
    /// Exact equalities, because the point of the construction is that `f64::max` hands back its
    /// left operand *bit for bit* when the model is inside its fitted range — an approximate
    /// assertion here would not distinguish "inert" from "nearly inert", and nearly inert is a
    /// divergence from the reference on every input rather than on the broken ones.
    #[test]
    fn a_temperature_node_is_inert_until_it_would_outrun_the_exosphere() {
        let mut held = false;
        // 200 / 0.5 = 400 K, well under the 1000 K exosphere: untouched, and not reported.
        assert_eq!(temperature_node(200.0, 0.5, 1000.0, &mut held), 200.0 / 0.5);
        // A colder-than-unperturbed node is as far from the bound as it gets, and still exact.
        assert_eq!(temperature_node(200.0, -1.0, 1000.0, &mut held), 100.0);
        // The tie: `1 − 0.75` is exactly `250/1000`, so the bound is reached and not crossed, and
        // the quotient is the same double either way. Written with a representable pair on purpose —
        // `1 − 0.8` is `0.19999999999999996`, which is already past `200/1000` and does engage.
        assert_eq!(temperature_node(250.0, 0.75, 1000.0, &mut held), 1000.0);
        assert!(!held, "the bound reported itself where it changed nothing");

        // Past the bound, and past the pole where the unbounded form is negative, `-0.0`, NaN.
        for variation in [0.8, 0.9, 1.0, 1.5, 4.0, f64::NAN] {
            let mut held = false;
            assert_eq!(
                temperature_node(200.0, variation, 1000.0, &mut held),
                1000.0
            );
            assert!(
                held,
                "the bound engaged at {variation} without reporting it"
            );
        }
    }

    /// Anomalous oxygen is a positive number density everywhere, including where the reference's own
    /// profile has gone negative.
    ///
    /// The `4000 K` end node `tho` is a shape parameter, not a temperature, so the cubic bridging it
    /// to the mesospheric nodes overshoots — and between `113` and `119.5 km` in the polar storm
    /// corner it takes `1/T` below zero. The reference's `(T₁/T)^(1+α)` prefactor then returns a
    /// *negative* density rather than NaN, because anomalous oxygen is the one call with `α = 0` and
    /// an integer exponent: at `113 km`, `−2.1e-59 cm⁻³`. Holding `1/T` at `1/tho` instead keeps the
    /// profile positive and monotone in altitude through the whole band, which is the assertion
    /// here — a sign test alone would pass on a constant, and monotonicity is what says the held
    /// values are still a profile.
    #[test]
    fn anomalous_oxygen_is_positive_where_the_reference_goes_negative() {
        let at = |alt, g_lat, ap, f107| Nrlmsise00Input {
            year: 0,
            doy: 105,
            sec: 29000.0,
            alt,
            g_lat,
            g_long: -180.0,
            lst: 16.0,
            f107a: f107,
            f107,
            ap,
            ap_array: None,
        };
        let flags = Nrlmsise00Flags::default();
        // The band the reference returns negative densities over, walked from below it to above it.
        let mut previous = 0.0;
        let mut z = 108.0;
        while z <= 121.0 {
            let d8 = gts7(&at(z, -90.0, 300.0, 250.0), &flags).d[8];
            assert!(
                d8 > previous,
                "anomalous oxygen {d8} at {z} km, after {previous}"
            );
            previous = d8;
            z += 0.5;
        }
        // A quiet point and a high one, where the reference is well-formed and we match it: without
        // these, "positive" would be satisfied by a routine that had stopped modelling anything.
        assert!(gtd7(&at(113.0, 45.0, 4.0, 150.0), &flags).d[8] > 0.0);
        assert!(gtd7(&at(400.0, -90.0, 300.0, 250.0), &flags).d[8] > 0.0);
    }

    /// The one field [`Nrlmsise00Output::out_of_range`] does not cover is negligible where it is
    /// uncovered.
    ///
    /// Below the thermopause the anomalous-oxygen profile is anchored on the shape parameter `tho`
    /// rather than on a temperature, so it is bounded without being flagged, and that exception is
    /// only defensible while the field is numerically irrelevant there. This puts a number on it, and
    /// contrasts it with the same field where the model does make a claim — the two differ by
    /// thirty-three orders of magnitude, which is the whole argument for the exception.
    #[test]
    fn the_anomalous_oxygen_tail_below_the_thermopause_is_negligible() {
        let flags = Nrlmsise00Flags::default();
        let at = |alt: f64, g_lat: f64, ap: f64, f107: f64, doy: i32| Nrlmsise00Input {
            year: 0,
            doy,
            sec: 29000.0,
            alt,
            g_lat,
            g_long: -180.0,
            lst: 16.0,
            f107a: f107,
            f107,
            ap,
            ap_array: None,
        };
        let mut largest = 0.0f64;
        let mut alt = 72.5;
        while alt < 120.0 {
            let mut lat = -90.0;
            while lat <= 90.0 {
                for ap in [0.0, 200.0, 400.0] {
                    for f107 in [65.0, 300.0] {
                        // `gtd7d` rather than `gtd7`: it is the entry point that folds `d[8]` into
                        // the mass density a drag model integrates, so it is where a tail that was
                        // not negligible would actually do damage.
                        let o = gtd7d(&at(alt, lat, ap, f107, 196), &flags);
                        assert!(o.d[8] >= 0.0, "anomalous oxygen {} at {alt} km", o.d[8]);
                        largest = largest.max(o.d[8]);
                    }
                }
                lat += 15.0;
            }
            alt += 1.0;
        }
        // Bounded, and not by accident: the measured maximum is 1.27e-29 cm⁻³, so a bound two orders
        // above it would still fail on a routine that had started returning ordinary numbers here.
        assert!(largest < 1.0e-28, "largest tail value {largest:e}");
        // …and not vacuous: the band really is evaluated, not returning zeros.
        assert!(largest > 1.0e-31, "largest tail value {largest:e}");
        // Where the model does claim anomalous oxygen, the same field is thirty-three orders larger.
        assert!(gtd7d(&at(400.0, -90.0, 300.0, 250.0, 105), &flags).d[8] > 1.0e3);
    }

    /// Comparator: the authors' 17 printed outputs are what NRL's code prints, and no document
    /// declares them normative (`plans/17` D-18 b). The full-precision sweep above, against the
    /// same C at `1e-14` rather than `1e-5`, is the tripwire that stays.
    #[test]
    #[ignore = "comparator: NRL's printed output, not a verdict"]
    fn gtd7_matches_reference_vectors() {
        let mut inputs = [base_input(); 17];
        inputs[1].doy = 81;
        inputs[2].sec = 75000.0;
        inputs[2].alt = 1000.0;
        inputs[3].alt = 100.0;
        inputs[10].alt = 0.0;
        inputs[11].alt = 10.0;
        inputs[12].alt = 30.0;
        inputs[13].alt = 50.0;
        inputs[14].alt = 70.0;
        inputs[16].alt = 100.0;
        inputs[4].g_lat = 0.0;
        inputs[5].g_long = 0.0;
        inputs[6].lst = 4.0;
        inputs[7].f107a = 70.0;
        inputs[8].f107 = 180.0;
        inputs[9].ap = 40.0;
        inputs[15].ap_array = Some([100.0; 7]);
        inputs[16].ap_array = Some([100.0; 7]);

        let flags_default = Nrlmsise00Flags::default();
        let flags_ap = {
            let mut s = [1i32; 24];
            s[0] = 0;
            s[9] = -1;
            Nrlmsise00Flags { switches: s }
        };

        let mut max_rel = 0.0f64;
        for i in 0..17 {
            let flags = if i < 15 { &flags_default } else { &flags_ap };
            let out = gtd7(&inputs[i], flags);
            let got = [
                out.d[0], out.d[1], out.d[2], out.d[3], out.d[4], out.d[5], out.d[6], out.d[7],
                out.d[8], out.t[0], out.t[1],
            ];
            for j in 0..11 {
                let exp = GOLDEN[i][j];
                let g = got[j];
                // Guard against the ~1e-42 "zero" entries: compare absolutely when the golden ~0.
                let rel = if libm::fabs(exp) < 1.0E-30 {
                    libm::fabs(g)
                } else {
                    libm::fabs((g - exp) / exp)
                };
                assert!(
                    rel < 1.0E-5,
                    "case {i} field {j}: got {g:e}, expected {exp:e}, rel err {rel:e}"
                );
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        std::eprintln!("NRLMSISE-00 max relative error over 17x11 = {max_rel:e}");
        assert!(max_rel < 1.0E-5);
    }
}
