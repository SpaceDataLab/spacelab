//! The Moon as a central body: its gravity field, and the two body-fixed frames it is quoted in.
//!
//! # Why the Moon is not the Earth with different numbers
//!
//! [`crate::GeoPotential`] is already body-agnostic — it takes `μ`, a reference radius and normalized
//! coefficients — so nothing in the recursion changes. What changes is everything around it, and each
//! difference is a way to be wrong by kilometres rather than by rounding:
//!
//! * **The field is far rougher relative to its own mean.** The Moon's `J₂` is 2.03e-4, a fifth of the
//!   Earth's, but its `C₂₂` is 2.24e-5 against the Earth's 1.57e-6 — an order of magnitude larger,
//!   because it is a frozen tidal bulge aimed at the Earth rather than a rotational flattening. A
//!   zonal-only lunar model is not a first approximation, it is a different orbit.
//! * **There is no frozen orbit without the whole field.** Low lunar orbits are unstable on a scale of
//!   weeks under the mascons, which is why degree matters here at altitudes where the Earth would be
//!   content with degree 8.
//! * **The body-fixed frame is not the one lunar maps use.** Gravity models are quoted in the
//!   **principal-axis** frame; cartography, and every "selenographic latitude/longitude" in a paper,
//!   is in the **mean Earth/mean rotation** frame. They differ by a fixed rotation of about 0.03°,
//!   which is 900 m on the surface. [`pa_to_me`] is that rotation and exists so the mistake has
//!   a name.
//! * **The rotation is a libration, not a spin.** The Moon's orientation cannot be written as
//!   `θ₀ + ωt` about a fixed axis the way [`crate::forces::EarthFixed`] writes the Earth's: the
//!   physical libration in longitude alone reaches 8°. The field must be evaluated through a full
//!   time-varying matrix — [`crate::forces::BodyOriented`] — and that matrix comes from an ephemeris,
//!   which is why it is supplied by the caller rather than modelled here.
//!
//! # What this module is not
//!
//! It is not a frame provider. The orientation matrix and the Moon's own position come from
//! `spacelab_ephemeris`, which owns the JPL kernels; this crate stays free of that dependency, the
//! same split [`crate::forces::BodyOriented`] already uses for the Earth.
//!
//! Sources: Lemoine et al. (2014), GRL **41**, doi:10.1002/2014GL060027 (GRGM1200A); Park et al.
//! (2021), AJ **161**, 105 (DE440, and the PA→ME angles); Konopliv et al. (2001), Icarus **150**, 1
//! (the PA/ME distinction and why gravity is quoted in PA).

use alloc::vec::Vec;

use crate::geopotential::GeoPotential;
use crate::lunar_coeffs::{MODEL_COEFFS, MODEL_GM, MODEL_MAX_DEGREE, MODEL_RE};

/// The Moon's mean radius, metres — IAU/LRO-LOLA, the value lunar cartography uses.
///
/// This is **not** [`crate::lunar_coeffs::MODEL_RE`]. The gravity model's reference radius is 1738 km, a
/// round number chosen when the expansion was defined and carried forward ever since; the mean radius
/// of the actual surface is 1737.4 km. The 600 m between them is a convention, not a discrepancy —
/// but substituting one for the other rescales every coefficient by `(R'/R)ⁿ` and quietly changes the
/// field. Use `MODEL_RE` with the coefficients and this one for geometry: shadow, horizon, altitude.
pub const MOON_MEAN_RADIUS: f64 = 1_737_400.0;

/// The Moon's gravitational parameter, m³/s², as GRGM1200A defines it.
pub const MOON_MU: f64 = MODEL_GM;

/// A rotation matrix, row-major: `v_out[i] = Σⱼ M[i][j] · v_in[j]`.
pub type Matrix3 = [[f64; 3]; 3];

/// The three rotation angles taking the mean Earth frame to the DE440 principal axes, **arcseconds**,
/// applied about axes 3, 2, 1 in that order.
///
/// Read verbatim from NAIF's `moon_de440_250416.tf`, `TKFRAME_31009_ANGLES`, which cites Park et al.
/// (2021) and states that its listed angles are the inverse of the published rotation — negated, with
/// the sequence reversed. SPICE's `eul2m` composition of them therefore runs **ME → PA**, not the
/// direction the constant's position in the kernel suggests, and the two differ by twice the angle:
/// 0.06°, or 1.8 km on the surface, in the direction that reads as a small bias rather than a
/// mistake. Which way round is settled by measurement, not by reading: averaging the Earth's
/// direction over a nodal cycle puts it at longitude −71″ in the principal axes and −3″ in the mean
/// Earth frame, which is what the ME frame is defined to do
/// (`spacelab-ephemeris/tests/lunar_orientation.rs`).
pub const ME_TO_PA_ANGLES: [f64; 3] = [67.8526, 78.6944, 0.2785];

/// Radians per arcsecond.
const ARCSEC: f64 = core::f64::consts::PI / (180.0 * 3600.0);

/// The fixed mean-Earth → principal-axis rotation, row-major: `r_PA = M · r_ME`.
///
/// `M = R₁(a₃)·R₂(a₂)·R₃(a₁)` with the angles of [`ME_TO_PA_ANGLES`] and `Rₖ(θ)` the *frame* rotation
/// about axis `k` — the matrix whose rows are the rotated axes, so `Rₖ(θ)ᵢⱼ` carries `+sin` above the
/// diagonal. That is SPICE's `eul2m` convention, which is the convention the kernel's angles are
/// listed in; assembling them as vector rotations instead transposes the result.
///
/// This is the one a caller usually wants: a position given in selenographic coordinates has to come
/// *into* the principal axes before [`embedded`]'s field can be evaluated at it.
///
/// The rotation is small — all three angles are under 80 arcseconds — so `M` is the identity to
/// within 4e-4, and a test that only checks orthogonality would pass with the angles zeroed. What
/// pins its size is the surface displacement, about 900 m at the equator; what pins its *direction*
/// is that ME is defined so the mean Earth direction lies on its prime meridian, and only one of the
/// two candidates removes the 68-arcsecond mean longitude the Earth has in the principal axes.
#[must_use]
pub fn me_to_pa() -> Matrix3 {
    let [a1, a2, a3] = ME_TO_PA_ANGLES;
    mul3(
        &rot(1, a3 * ARCSEC),
        &mul3(&rot(2, a2 * ARCSEC), &rot(3, a1 * ARCSEC)),
    )
}

/// The principal-axis → mean-Earth rotation, the transpose of [`me_to_pa`]: `r_ME = M · r_PA`.
///
/// Cartography's direction: the gravity field and the mascons live in the principal axes, the maps
/// and the published feature coordinates in the mean Earth frame.
#[must_use]
pub fn pa_to_me() -> Matrix3 {
    transpose(&me_to_pa())
}

/// Frame rotation by `theta` radians about axis `k` (1, 2 or 3), row-major.
fn rot(k: u8, theta: f64) -> Matrix3 {
    let (c, s) = (libm::cos(theta), libm::sin(theta));
    match k {
        1 => [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]],
        2 => [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]],
        _ => [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
    }
}

fn mul3(a: &Matrix3, b: &Matrix3) -> Matrix3 {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    m
}

fn transpose(m: &Matrix3) -> Matrix3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// The embedded lunar field ([`crate::lunar_coeffs::MODEL_NAME`]) truncated to `degree`, using the
/// model's own `GM` and reference radius.
///
/// `None` above [`crate::lunar_coeffs::MODEL_MAX_DEGREE`] — the table is a truncation of a degree-1200
/// solution, and asking for a degree it does not carry would silently give a *smoother* field than
/// requested rather than an error.
///
/// The result is in the **principal-axis** frame. Wrap it in [`crate::forces::BodyOriented`] with a
/// GCRF→PA matrix; evaluated at a raw inertial position it smears the whole tesseral field.
#[must_use]
pub fn embedded(degree: usize) -> Option<GeoPotential> {
    if degree > MODEL_MAX_DEGREE {
        return None;
    }
    GeoPotential::from_normalized(MODEL_GM, MODEL_RE, degree, MODEL_COEFFS)
}

/// What a SHADR header record declares about the field that follows it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadrHeader {
    /// Reference radius, metres (the product states kilometres).
    pub reference_radius: f64,
    /// Gravitational parameter, m³/s² (the product states km³/s²).
    pub mu: f64,
    /// Degree of the full model, which is not the degree parsed if the file was truncated.
    pub degree: usize,
    /// Order of the full model.
    pub order: usize,
}

/// A count read out of a header field that the format stores as a float.
///
/// The file is untrusted input, so the conversion is checked rather than cast: a negative, fractional
/// or implausibly large value is a file this parser has not understood, and `NaN` fails the range
/// test as well. Past the guard the cast is exact, which is why it is the only one in the module.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn header_count(value: f64) -> Option<usize> {
    if !(0.0..=1.0e6).contains(&value) || value.fract() != 0.0 {
        return None;
    }
    Some(value as usize)
}

/// Parse a PDS SHADR spherical-harmonic product — the format every GRAIL, MGS and MESSENGER gravity
/// solution is archived in — truncating at `degree`.
///
/// Returns the header alongside the field, because the header is the only place the model's own `GM`
/// and reference radius appear and using any others with these coefficients is a wrong answer that
/// looks plausible. `None` if the header is unreadable, if the coefficients are unnormalized (state
/// flag other than 1), or if `degree` exceeds what [`GeoPotential`] can hold.
///
/// Two properties of the format are load-bearing and neither is inferred:
///
/// * the file is parsed by **content**, not by the record offsets in its PDS label — `gggrx_1200a`'s
///   label declares the coefficients at record 3 and they begin at record 2, so an offset-driven
///   reader silently drops `(1,0)` and shifts nothing else, which no sanity check on `GM` would see;
/// * degrees 0 and 1 are absent by definition, not by truncation. `C₀₀ = 1` is the point mass and is
///   set by the constructor, and the degree-1 terms vanish because the expansion is centred on the
///   centre of mass. A file that carried a non-zero degree-1 term would be describing a different
///   origin, so they are read and checked rather than skipped.
///
/// The header record is required whole. SHADR defines eight fields and the last two are the
/// reference longitude and latitude — the same "different origin" statement the degree-1 check
/// exists to catch, so they are required to be zero rather than ignored.
#[must_use]
pub fn parse_shadr(text: &str, degree: usize) -> Option<(ShadrHeader, GeoPotential)> {
    let mut records = text.lines().filter(|line| !line.trim().is_empty());

    let head: Vec<f64> = records
        .next()?
        .split(',')
        .map(|token| token.trim().parse::<f64>().ok())
        .collect::<Option<Vec<_>>>()?;
    if head.len() < 8 {
        return None;
    }
    // Fields 6 and 7 are the reference longitude and latitude. Anything but the origin describes an
    // expansion this parser is not reading, in the same way a non-zero degree-1 term would.
    #[allow(clippy::float_cmp)]
    if head[6] != 0.0 || head[7] != 0.0 {
        return None;
    }
    // Field 5 is the normalization state: 0 unnormalized, 1 normalized, 2 other. Only 1 is accepted,
    // because feeding unnormalized coefficients to `from_normalized` is off by N_nm — a factor that
    // reaches 1e-3 by degree 10 and looks like a slightly different model rather than a mistake.
    // The comparison is exact on purpose: this field is a flag, not a measurement, and a value that is
    // merely near 1 is a file this parser has not understood.
    #[allow(clippy::float_cmp)]
    if head[5] != 1.0 {
        return None;
    }
    let header = ShadrHeader {
        reference_radius: head[0] * 1.0e3,
        mu: head[1] * 1.0e9,
        degree: header_count(head[3])?,
        order: header_count(head[4])?,
    };

    let mut coeffs: Vec<(usize, usize, f64, f64)> = Vec::new();
    for record in records {
        let mut field = record.split(',').map(str::trim);
        let (n, m) = (
            field.next()?.parse::<usize>().ok()?,
            field.next()?.parse::<usize>().ok()?,
        );
        if n > degree {
            break;
        }
        let (c, s) = (
            field.next()?.parse::<f64>().ok()?,
            field.next()?.parse::<f64>().ok()?,
        );
        if n < 2 {
            // The degree-1 terms are the centre-of-mass offset. Non-zero means the file is referenced
            // to something other than the centre of mass, and the field cannot be used as-is.
            if c != 0.0 || s != 0.0 {
                return None;
            }
            continue;
        }
        coeffs.push((n, m, c, s));
    }
    if coeffs.is_empty() {
        return None;
    }

    let field = GeoPotential::from_normalized(header.mu, header.reference_radius, degree, &coeffs)?;
    Some((header, field))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::unwrap_used)]
    use super::{
        ME_TO_PA_ANGLES, MOON_MEAN_RADIUS, MOON_MU, Matrix3, embedded, me_to_pa, mul3, pa_to_me,
        parse_shadr, rot, transpose,
    };
    use crate::lunar_coeffs::{MODEL_MAX_DEGREE, MODEL_RE};

    fn norm(v: [f64; 3]) -> f64 {
        libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    }

    fn apply(m: &Matrix3, v: [f64; 3]) -> [f64; 3] {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    #[test]
    fn the_principal_axis_to_mean_earth_rotation_moves_the_surface_by_about_nine_hundred_metres() {
        // A rotation of 67.9 + 78.7 arcseconds is a number with no feel to it, and a test that only
        // asserted orthogonality would pass with all three angles set to zero. The displacement it
        // produces on the surface is the quantity that matters and the one a reader can check:
        // 0.03 degrees at 1737.4 km is 900 m, and that is the error made by evaluating a
        // principal-axis gravity field at a selenographic position.
        let m = pa_to_me();
        for point in [
            [MOON_MEAN_RADIUS, 0.0, 0.0],
            [0.0, MOON_MEAN_RADIUS, 0.0],
            [0.0, 0.0, MOON_MEAN_RADIUS],
        ] {
            let moved = apply(&m, point);
            let shift = norm([
                moved[0] - point[0],
                moved[1] - point[1],
                moved[2] - point[2],
            ]);
            println!("PA->ME moves {point:?} by {shift:.1} m");
            assert!(
                (100.0..1200.0).contains(&shift),
                "surface displacement {shift} m is not the ~0.03 degrees the angles describe"
            );
        }

        // The z-axis is the one the two frames disagree about least in *angle* and most in
        // consequence: the pole moves by the 78.7-arcsecond y-rotation alone, 663 m.
        let pole = apply(&m, [0.0, 0.0, MOON_MEAN_RADIUS]);
        let polar_shift = norm([pole[0], pole[1], pole[2] - MOON_MEAN_RADIUS]);
        let expected = MOON_MEAN_RADIUS
            * libm::sqrt(
                ME_TO_PA_ANGLES[1] * ME_TO_PA_ANGLES[1] + ME_TO_PA_ANGLES[2] * ME_TO_PA_ANGLES[2],
            )
            * super::ARCSEC;
        println!("pole moves {polar_shift:.1} m, small-angle prediction {expected:.1} m");
        assert!(
            (polar_shift - expected).abs() < 0.5,
            "the pole displacement is not the y and x angles in quadrature"
        );
    }

    #[test]
    fn the_two_frame_rotations_are_inverses_and_neither_is_the_identity() {
        let there = pa_to_me();
        let back = me_to_pa();
        assert_eq!(
            back,
            transpose(&there),
            "me_to_pa is not pa_to_me transposed"
        );

        // Round-trip to the last bit is not available through two matrix products, but a metre of
        // slack would hide a transposed sub-block; 1e-9 m over a lunar radius is 6e-16 relative.
        let point = [1.0e6, -2.0e5, 9.0e5];
        let round = apply(&back, apply(&there, point));
        for k in 0..3 {
            assert!((round[k] - point[k]).abs() < 1.0e-9, "round trip axis {k}");
        }

        // Guard against the whole thing collapsing to the identity — which is what a units slip
        // (degrees read as arcseconds, or the angles dropped) produces, and which every orthogonality
        // and round-trip check above would still pass.
        let off =
            (there[0][1] * there[0][1] + there[0][2] * there[0][2] + there[1][2] * there[1][2])
                .sqrt();
        println!("off-diagonal magnitude {off:.6e}");
        assert!(off > 1.0e-4, "the rotation collapsed to the identity");
        assert!(
            off < 1.0e-3,
            "the rotation is far larger than 80 arcseconds"
        );
    }

    /// Both directions are orthonormal, both displace the surface by 900 m, and every check above
    /// passes on either — so the one thing that separates them gets its own test.
    ///
    /// The mean Earth direction defines the ME prime meridian, and in the principal axes it sits at
    /// longitude −71″ (measured over a nodal cycle of DE440 in
    /// `spacelab-ephemeris/tests/lunar_orientation.rs`). So `me_to_pa` must carry the ME `+x` axis to
    /// a *negative* longitude of about that size; the transposed matrix carries it to `+68″`, which
    /// is the same distance in the wrong direction and reads as a sign convention rather than an
    /// error.
    #[test]
    fn me_to_pa_carries_the_mean_earth_direction_to_the_negative_longitude_de440_measures() {
        let mean_earth_in_pa = apply(&me_to_pa(), [MOON_MEAN_RADIUS, 0.0, 0.0]);
        let longitude = libm::atan2(mean_earth_in_pa[1], mean_earth_in_pa[0]) / super::ARCSEC;
        println!("the ME prime meridian sits at longitude {longitude:.3}\" in the principal axes");
        assert!(
            (-70.0..-65.0).contains(&longitude),
            "ME +x lands at {longitude:.3}\" in PA, not the −67.85″ the kernel's z-angle gives"
        );
    }

    #[test]
    fn the_embedded_field_carries_the_moons_published_low_degree_shape() {
        let field = embedded(4).unwrap();
        // GRGM1200A's own constants, not a lunar constant from elsewhere: the coefficients are
        // referenced to these and mixing sources rescales the field.
        assert_eq!(field.mu(), MOON_MU);
        assert_eq!(field.reference_radius(), MODEL_RE);
        assert_eq!(MODEL_RE, 1_738_000.0, "the model radius is 1738 km exactly");
        const {
            assert!(
                MODEL_RE > MOON_MEAN_RADIUS,
                "the expansion radius must sit above the mean surface, or the series is evaluated \
                 inside the body at the poles"
            );
        }

        // The Moon's J2 is 2.032e-4 and its C22 is 2.238e-5 -- a ratio of 9, against about 690 for
        // the Earth. That ratio is the single number that says this is a tidally locked body carrying
        // a frozen bulge rather than a rotational figure, and it is what a wrong-body coefficient
        // table would fail first.
        //
        // Both come off the *normalized* store and neither is comparable to a textbook value without
        // its own factor: J2 = -sqrt(5)*C20bar, and C22 = C22bar*sqrt(5/12). Doing that here rather
        // than in an accessor keeps the convention where the published number is quoted.
        let (c20_bar, _) = field.normalized(2, 0).unwrap();
        let (c22_bar, s22_bar) = field.normalized(2, 2).unwrap();
        let j2 = -libm::sqrt(5.0) * c20_bar;
        let c22 = c22_bar * libm::sqrt(5.0 / 12.0);
        println!(
            "lunar J2 = {j2:.6e}, C22 = {c22:.6e}, ratio {:.1}",
            j2 / c22
        );
        assert!((j2 - 2.0322e-4).abs() < 1e-8, "J2 = {j2}");
        assert!((c22 - 2.2381e-5).abs() < 1e-9, "C22 = {c22}");
        assert!((5.0..20.0).contains(&(j2 / c22)), "J2/C22 = {}", j2 / c22);

        // The principal axes diagonalise the degree-2 tensor, so S22 vanishes -- not exactly, because
        // the frame is realised by an ephemeris libration model the fit could not fully absorb, but
        // to 3.4e-5 of C22. In the mean-Earth frame the same ratio is two orders larger, so this is
        // the assertion that the embedded table is in the frame its header claims.
        let tilt = libm::fabs(s22_bar / c22_bar);
        println!("S22bar/C22bar = {tilt:.3e} -- principal axes to that");
        assert!(tilt < 1.0e-4, "degree-2 tensor is not diagonal: {tilt:.3e}");
        assert!(tilt > 0.0, "an exactly zero S22 would mean a zeroed table");

        assert!(embedded(MODEL_MAX_DEGREE).is_some());
        assert!(
            embedded(MODEL_MAX_DEGREE + 1).is_none(),
            "asking above the table must fail rather than silently return a smoother field"
        );
    }

    /// Two records of a well-formed SHADR product, in the real fixed-width layout.
    const SAMPLE: &str = concat!(
        " 1.7380000000000000e+03, 4.9028001224453001e+03, 6.4536052689015518e-15, 1200, 1200,    1, 0.0000000000000000e+00, 0.0000000000000000e+00\n",
        "    1,    0, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00\n",
        "    1,    1, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00\n",
        "    2,    0,-9.0884339347000005e-05, 0.0000000000000000e+00, 1.0000000000000000e-12, 0.0000000000000000e+00\n",
        "    2,    1, 1.4664000000000000e-11, 1.1733000000000000e-09, 1.0000000000000000e-12, 1.0000000000000000e-12\n",
        "    2,    2, 3.4673096471000002e-05, 9.0791900000000005e-10, 1.0000000000000000e-12, 1.0000000000000000e-12\n",
        "    3,    0, 8.4700000000000000e-06, 0.0000000000000000e+00, 1.0000000000000000e-12, 0.0000000000000000e+00\n",
    );

    #[test]
    fn the_shadr_parser_reads_the_header_constants_and_stops_at_the_requested_degree() {
        let (header, field) = parse_shadr(SAMPLE, 2).unwrap();
        // The header states km and km^3/s^2; everything downstream is SI, and getting that conversion
        // wrong is a factor of 1e9 in GM -- which is not subtle, unlike using the label's quoted
        // 4902.80011526323 in place of the file's 4902.8001224453.
        assert_eq!(header.reference_radius, 1_738_000.0);
        assert_eq!(header.mu, 4.902_800_122_445_3e12);
        assert_eq!((header.degree, header.order), (1200, 1200));
        assert_eq!(field.mu(), header.mu);

        // The degree-2 records must have been read, and read into the right slots: C21 and S21 differ
        // by five orders of magnitude in the sample, so a column slip shows up immediately.
        assert_eq!(field.degree(), 2);
        assert_eq!(field.normalized(2, 0).unwrap().0, -9.088_433_934_700_001e-5);
        assert_eq!(field.normalized(2, 1).unwrap(), (1.4664e-11, 1.1733e-09));
        assert_eq!(
            field.normalized(2, 2).unwrap(),
            (3.467_309_647_1e-5, 9.079_190_000_000_001e-10)
        );

        // Truncation at 2 drops the degree-3 record; at 3 it is there. Asking the truncated field for
        // it must be `None` rather than a zero, which is the difference between "not in this model"
        // and "this model says zero".
        assert!(field.normalized(3, 0).is_none(), "degree 3 was truncated");
        let (_, deeper) = parse_shadr(SAMPLE, 3).unwrap();
        assert_eq!(deeper.normalized(3, 0).unwrap().0, 8.47e-06);
    }

    #[test]
    fn the_shadr_parser_refuses_a_file_it_cannot_use_as_it_stands() {
        assert!(parse_shadr("", 4).is_none(), "empty");
        assert!(parse_shadr("1.0, 2.0\n", 4).is_none(), "short header");

        // Normalization state 0 is unnormalized. Accepting it would be wrong by N_nm, which at degree
        // 10 is 1e-3 -- small enough to read as a different model rather than a misread file.
        let unnormalized = SAMPLE.replacen(", 1200, 1200,    1,", ", 1200, 1200,    0,", 1);
        assert!(parse_shadr(&unnormalized, 2).is_none(), "unnormalized");

        // A non-zero degree-1 term means the expansion is not centred on the centre of mass. The
        // field is still a field, but it is not the one the caller is about to add a point mass to.
        let offset = SAMPLE.replacen(
            "    1,    1, 0.0000000000000000e+00",
            "    1,    1, 1.0000000000000000e-09",
            1,
        );
        assert!(parse_shadr(&offset, 2).is_none(), "degree-1 term present");

        // Header only, no coefficients: a truncation that cut too early, not a valid degree-0 field.
        let header_only = SAMPLE.lines().next().unwrap();
        assert!(parse_shadr(header_only, 2).is_none(), "no coefficients");
    }

    /// The header's counts are counts. The file is untrusted, so a field that is not one has to be
    /// refused rather than cast — `-1.0 as usize` saturates to 0 and `2.5 as usize` truncates to 2,
    /// and both produce a field that parses into silence.
    #[test]
    fn parse_shadr_refuses_a_header_whose_counts_are_not_counts() {
        let fractional = SAMPLE.replacen(", 1200, 1200,", ", 1200.5, 1200,", 1);
        assert!(parse_shadr(&fractional, 2).is_none(), "fractional degree");

        let negative = SAMPLE.replacen(", 1200, 1200,", ", 1200, -1,", 1);
        assert!(parse_shadr(&negative, 2).is_none(), "negative order");

        let enormous = SAMPLE.replacen(", 1200, 1200,", ", 1200, 1e9,", 1);
        assert!(parse_shadr(&enormous, 2).is_none(), "order of 1e9");
    }

    /// SHADR's header record is eight fields, and the last two say where the expansion is centred.
    #[test]
    fn parse_shadr_requires_the_whole_header_record_at_the_origin() {
        let head = SAMPLE.lines().next().unwrap();
        let short: String = head.rsplit_once(',').unwrap().0.to_string();
        let truncated = SAMPLE.replacen(head, &short, 1);
        assert!(parse_shadr(&truncated, 2).is_none(), "seven-field header");

        let displaced = SAMPLE.replacen(
            ",    1, 0.0000000000000000e+00, 0.0000000000000000e+00\n",
            ",    1, 9.0000000000000000e+01, 0.0000000000000000e+00\n",
            1,
        );
        assert!(
            parse_shadr(&displaced, 2).is_none(),
            "reference longitude 90"
        );
    }

    /// `rot(k, θ)` must turn about the axis it names. The composition in [`me_to_pa`] is three
    /// arcsecond-scale angles, so confusing two of its axes moves the product by ~1e-6 — below every
    /// tolerance in this module and above the one in `lunar_orientation.rs`.
    #[test]
    fn rot_turns_about_the_axis_it_names() {
        let quarter = core::f64::consts::FRAC_PI_2;
        for (axis, fixed) in [(1u8, 0), (2, 1), (3, 2)] {
            let mut unit = [0.0; 3];
            unit[fixed] = 1.0;
            let turned = apply(&rot(axis, quarter), unit);
            for (component, value) in turned.iter().enumerate() {
                let want = if component == fixed { 1.0 } else { 0.0 };
                assert!(
                    (value - want).abs() < 1e-15,
                    "rot({axis}) moved the axis it turns about: {turned:?}"
                );
            }
        }
        // A quarter turn about z sends x to −y in this convention (the frame rotates, not the vector).
        let turned = apply(&rot(3, quarter), [1.0, 0.0, 0.0]);
        assert!(turned[1] < -0.999, "rot(3) turns the wrong way: {turned:?}");
    }

    /// `mul3` is only ever exercised here on near-identity products, where every off-diagonal term is
    /// ~1e-8 and a dropped one is invisible. This is the same routine on a product that has no small
    /// terms to hide behind.
    #[test]
    fn mul3_multiplies_rows_into_columns() {
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
        let b = [[1.0, 0.0, 2.0], [0.0, 3.0, 1.0], [4.0, 1.0, 0.0]];
        let want = [[13.0, 9.0, 4.0], [28.0, 21.0, 13.0], [47.0, 34.0, 22.0]];
        let got = mul3(&a, &b);
        for (row, wanted) in got.iter().zip(want.iter()) {
            for (value, expected) in row.iter().zip(wanted.iter()) {
                assert!(
                    (value - expected).abs() < 1e-12,
                    "mul3 gave {got:?}, want {want:?}"
                );
            }
        }
    }
}
