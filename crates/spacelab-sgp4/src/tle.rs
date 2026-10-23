//! Producing an element set: SGP4 as a fitted parameter vector, and the 69-column format it prints
//! into.
//!
//! # The two halves
//!
//! [`TleModel`] is the fitting half. It presents the element set as `N` numbers — the six mean
//! elements, optionally `B*` — so that [`PropagatorConverter`] can least-squares them onto a
//! reference trajectory. SGP4 is a fixed listing with no derivative in it, so the Jacobian has to
//! come from [`PropagatorConverter::fit_by_finite_difference`]; the steps in
//! [`TLE_FIT_STEPS`] are the ones this crate's tests were measured with.
//!
//! [`TleBuilder`] is the printing half, and it is where the format stops being an implementation
//! detail. A `sgp4::Elements` can hold a six-digit catalog number, a 400° inclination and a
//! revolution count past a million; a TLE line cannot hold any of them. Everything that does not
//! fit a column is refused rather than truncated, so a [`Tle`] that exists is one that parses back
//! to the elements it was built from.
//!
//! # Catalog numbers past 99 999
//!
//! Five columns hold five digits, and the catalogue has more objects than that. The workaround in
//! circulation is Alpha-5, which re-reads the leading column as a letter worth 10–33 and buys
//! 240 000 more slots — `sgp4::parse_norad_id` reads it, and so this crate ingests it. It is not
//! *written*. Alpha-5 is not part of any published format specification, a consumer that has not
//! implemented it either rejects the line or reads `A0001` as a parse error, and an element set
//! attributed to the wrong object is worse than one that was never emitted. A caller who needs to
//! publish such an object has CCSDS OMM, which carries `NORAD_CAT_ID` as an integer, and
//! [`TleError::CatalogNumberTooLarge`] says so.
//!
//! [`PropagatorConverter`]: spacelab_estimation::PropagatorConverter
//! [`PropagatorConverter::fit_by_finite_difference`]:
//!     spacelab_estimation::PropagatorConverter::fit_by_finite_difference

use core::fmt;
use core::ops::Range;

use sgp4::chrono::{Datelike, NaiveDate, Timelike};
use sgp4::{Classification, Elements};
use spacelab_core::{NoradId, UtcDateTime};
use spacelab_estimation::{PropagatorBuilder, SampledPropagator};

use crate::satellite::{Convention, Satellite};

/// Columns in one line of a TLE, checksum included.
pub const TLE_LINE_LENGTH: usize = 69;

/// Largest catalog number the five numeric columns can hold.
pub const MAX_CATALOG_NUMBER: u64 = 99_999;

/// Seconds in a minute. SGP4's own parameter is minutes; nothing else here is.
const SECONDS_PER_MINUTE: f64 = 60.0;

/// Degrees in a full turn, the modulus the printed angles are reduced by.
const DEGREES_PER_TURN: f64 = 360.0;

/// Central-difference steps for the six fitted mean elements, in the parameter vector's own units:
/// `[mean motion rev/day, eccentricity, inclination °, RAAN °, argument of perigee °, mean anomaly °]`.
///
/// Each step is bounded from below by SGP4's own output granularity — a step that moves the
/// propagated position by less than the model's round-off differences noise — and from above by the
/// quadratic truncation term of the central difference. What says both bounds were met is the
/// self-consistency fit in `tests/tle_conversion.rs`: with these steps, an ISS element set
/// propagated onto a 24-hour grid is recovered from its own ephemeris to `0` rev/day, `3.7e-16` in
/// eccentricity and at worst `8.8e-12` degrees of angle, at a post-fit position RMS of `5.4e-8` m.
pub const TLE_FIT_STEPS: [f64; 6] = [1e-8, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6];

/// [`TLE_FIT_STEPS`] with a seventh entry for `B*`, in earth radii⁻¹.
///
/// `B*` is only observable where there is drag to observe. On the same ISS arc it is recovered to
/// `4.4e-15`; at GNSS altitude the drag column of the Jacobian is identically zero, and a
/// seven-parameter fit there comes back
/// [`ConverterError::Solver`](spacelab_estimation::ConverterError::Solver) naming direction 6
/// rather than inventing a drag term.
pub const TLE_FIT_STEPS_WITH_DRAG: [f64; 7] = [1e-8, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9];

impl SampledPropagator for Satellite {
    /// The TEME state at `t` seconds from the element-set epoch, on the naive UTC coordinate.
    ///
    /// `None` is every reason SGP4 has for not answering — divergence, decay — which the converter
    /// reads as "step away from this parameter vector".
    fn state_at(&self, t: f64) -> Option<[f64; 6]> {
        let state = self.propagate_minutes(t / SECONDS_PER_MINUTE).ok()?;
        let (r, v) = (state.position.vector, state.velocity.vector);
        Some([r.x, r.y, r.z, v.x, v.y, v.z])
    }
}

/// The minimal element set a fit can start from when there is no prior one.
///
/// A precise-orbit product carries no elements, so the template a [`TleModel`] is built around has
/// to be manufactured: this is that template, holding the epoch, the catalog number and a seeded
/// parameter vector, with no drag, no derivatives and no catalog metadata. Everything absent is
/// zero rather than plausible, so that anything the caller does not set cannot be mistaken for
/// something that was fitted.
///
/// It exists so that a caller need not reach for the vendor crate's `chrono` to name an epoch: the
/// workspace's own [`UtcDateTime`] is what goes in.
///
/// # Errors
///
/// [`TleError::EpochOutOfRange`] for a UTC label the Gregorian calendar type cannot hold — which
/// includes a leap-second label, whose `:60` has no representation and would otherwise round into
/// the next minute.
pub fn seed_elements(
    epoch: UtcDateTime,
    norad_id: NoradId,
    parameters: &[f64; 6],
) -> Result<Elements, TleError> {
    #[allow(clippy::cast_possible_truncation)] // the second of a UTC label is in [0, 61)
    let total_nanoseconds = (epoch.second() * 1e9).round() as i64;
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    // non-negative by the line above
    let datetime = NaiveDate::from_ymd_opt(
        epoch.year(),
        u32::from(epoch.month()),
        u32::from(epoch.day()),
    )
    .and_then(|date| {
        date.and_hms_nano_opt(
            u32::from(epoch.hour()),
            u32::from(epoch.minute()),
            (total_nanoseconds / 1_000_000_000) as u32,
            (total_nanoseconds % 1_000_000_000) as u32,
        )
    })
    .ok_or(TleError::EpochOutOfRange { year: epoch.year() })?;

    Ok(Elements {
        object_name: None,
        international_designator: None,
        norad_id: u64::from(norad_id.get()),
        classification: Classification::Unclassified,
        datetime,
        mean_motion_dot: 0.0,
        mean_motion_ddot: 0.0,
        drag_term: 0.0,
        element_set_number: 0,
        inclination: parameters[2],
        right_ascension: parameters[3],
        eccentricity: parameters[1],
        argument_of_perigee: parameters[4],
        mean_anomaly: parameters[5],
        mean_motion: parameters[0],
        revolution_number: 0,
        ephemeris_type: 0,
    })
}

/// An SGP4 element set presented as a parameter vector, for fitting onto a reference trajectory.
///
/// The template supplies everything the fit does not: the epoch the parameters are referred to, the
/// catalog metadata, and — in the six-parameter form — the drag term. Fitting in the element set's
/// *printed* units rather than in radians and SI is deliberate: it makes the finite-difference
/// steps and the formal sigmas readable in the same units the TLE columns hold, and makes the
/// emitted line a rounding of the answer rather than a conversion of it.
///
/// Both parameterisations are implemented for the same type. Six parameters hold `B*` at the
/// template's value; seven estimate it.
#[derive(Debug, Clone, PartialEq)]
pub struct TleModel {
    template: Elements,
    convention: Convention,
}

impl TleModel {
    /// Build a model around a template element set and the SGP4 convention to evaluate it under.
    ///
    /// The convention must be the one the fitted elements will be *propagated* under. A TLE is a
    /// fit, so elements fitted with WGS-72 and evaluated with WGS-84 are wrong by the difference
    /// between the two gravity models, plausibly and silently.
    #[must_use]
    pub fn new(template: Elements, convention: Convention) -> Self {
        Self {
            template,
            convention,
        }
    }

    /// The epoch the parameter vector's times are measured from, as the element set's own UTC label.
    #[must_use]
    pub fn epoch(&self) -> sgp4::chrono::NaiveDateTime {
        self.template.datetime
    }

    /// The template the fitted fields are written back into.
    #[must_use]
    pub const fn template(&self) -> &Elements {
        &self.template
    }

    /// Pack the six fitted fields out of an element set, in the parameter vector's order and units.
    #[must_use]
    pub const fn parameters(elements: &Elements) -> [f64; 6] {
        [
            elements.mean_motion,
            elements.eccentricity,
            elements.inclination,
            elements.right_ascension,
            elements.argument_of_perigee,
            elements.mean_anomaly,
        ]
    }

    /// [`TleModel::parameters`] with `B*` appended.
    #[must_use]
    pub const fn parameters_with_drag(elements: &Elements) -> [f64; 7] {
        [
            elements.mean_motion,
            elements.eccentricity,
            elements.inclination,
            elements.right_ascension,
            elements.argument_of_perigee,
            elements.mean_anomaly,
            elements.drag_term,
        ]
    }

    /// Write a fitted parameter vector back into a full element set, angles reduced to `[0, 360)`.
    ///
    /// The reduction is a printing convention, not a change of orbit: SGP4 reaches every angle
    /// through its sine and cosine, so a mean anomaly of `−34.97°` and one of `325.03°` describe
    /// the same orbit. They are not bit-identical — `(M − 360)·π/180` rounds differently from
    /// `M·π/180` — and the gap that opens is measured at `1.9×10⁻⁹` m of position an hour past the
    /// epoch of the ISS element set, twelve decades below anything SGP4 claims.
    #[must_use]
    pub fn elements(&self, parameters: &[f64; 6]) -> Elements {
        self.write_back(parameters, self.template.drag_term)
    }

    /// [`TleModel::elements`] for the seven-parameter form.
    #[must_use]
    pub fn elements_with_drag(&self, parameters: &[f64; 7]) -> Elements {
        let orbit = [
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
            parameters[5],
        ];
        self.write_back(&orbit, parameters[6])
    }

    fn write_back(&self, orbit: &[f64; 6], drag_term: f64) -> Elements {
        let mut elements = self.template.clone();
        elements.mean_motion = orbit[0];
        elements.eccentricity = orbit[1];
        elements.inclination = orbit[2].rem_euclid(DEGREES_PER_TURN);
        elements.right_ascension = orbit[3].rem_euclid(DEGREES_PER_TURN);
        elements.argument_of_perigee = orbit[4].rem_euclid(DEGREES_PER_TURN);
        elements.mean_anomaly = orbit[5].rem_euclid(DEGREES_PER_TURN);
        elements.drag_term = drag_term;
        elements
    }

    /// The propagator for one parameter vector, or `None` outside the model's domain.
    ///
    /// The angles are passed through unreduced. Reducing them here would be a discontinuity in the
    /// middle of a finite difference — a step that crosses `0°` would produce a `360°` divided
    /// difference — while SGP4 itself only ever takes their sine and cosine.
    fn satellite(&self, orbit: &[f64; 6], drag_term: f64) -> Option<Satellite> {
        if orbit[0] <= 0.0 || !orbit[0].is_finite() || !(0.0..1.0).contains(&orbit[1]) {
            return None;
        }
        let mut elements = self.template.clone();
        elements.mean_motion = orbit[0];
        elements.eccentricity = orbit[1];
        elements.inclination = orbit[2];
        elements.right_ascension = orbit[3];
        elements.argument_of_perigee = orbit[4];
        elements.mean_anomaly = orbit[5];
        elements.drag_term = drag_term;
        Satellite::from_elements(&elements, self.convention).ok()
    }
}

impl PropagatorBuilder<6> for TleModel {
    type Propagator = Satellite;

    fn build(&self, parameters: &[f64; 6]) -> Option<Satellite> {
        self.satellite(parameters, self.template.drag_term)
    }
}

impl PropagatorBuilder<7> for TleModel {
    type Propagator = Satellite;

    fn build(&self, parameters: &[f64; 7]) -> Option<Satellite> {
        let orbit = [
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
            parameters[5],
        ];
        self.satellite(&orbit, parameters[6])
    }
}

/// A two-line element set, as the two lines that were checked against every column width.
///
/// Only [`TleBuilder::build`] constructs one, so its existence is the statement that the fields
/// fit: 69 ASCII columns per line, both checksums correct, and a catalog number the format can
/// name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tle {
    line1: String,
    line2: String,
}

impl Tle {
    /// The first line.
    #[must_use]
    pub fn line1(&self) -> &str {
        &self.line1
    }

    /// The second line.
    #[must_use]
    pub fn line2(&self) -> &str {
        &self.line2
    }
}

impl fmt::Display for Tle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.line1)?;
        write!(f, "{}", self.line2)
    }
}

/// Why an element set could not be printed as a TLE.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TleError {
    /// The catalog number needs more than the five numeric columns the format has.
    ///
    /// Its own variant, and not [`Self::OutOfRange`], because the remedy is different: no rounding
    /// or renormalising recovers it, and the caller has to publish CCSDS OMM instead. See the
    /// module documentation on Alpha-5.
    CatalogNumberTooLarge {
        /// The catalog number that has no five-column representation.
        norad_id: u64,
    },
    /// The epoch year has no two-digit representation.
    ///
    /// Two digits are read as 1957–1999 and 2000–2056, so the format cannot name an epoch outside
    /// that century-and-a-bit at all.
    EpochOutOfRange {
        /// The year the elements are stamped with.
        year: i32,
    },
    /// A field holds a value its columns cannot express, named by the field.
    OutOfRange {
        /// The field, by the name the format gives it.
        field: &'static str,
    },
    /// The international designator is not `<launch year>-<launch number and piece>`.
    ///
    /// This is the CCSDS form `sgp4::Elements` carries — `1998-067A` — which is what has to be
    /// taken apart again to fill columns 10–17.
    MalformedDesignator,
}

impl fmt::Display for TleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CatalogNumberTooLarge { norad_id } => write!(
                f,
                "catalog number {norad_id} needs more than the {MAX_CATALOG_NUMBER} a TLE can name; \
                 publish CCSDS OMM instead"
            ),
            Self::EpochOutOfRange { year } => write!(
                f,
                "epoch year {year} is outside the 1957-2056 range two digits can name"
            ),
            Self::OutOfRange { field } => {
                write!(
                    f,
                    "the {field} does not fit the columns the format gives it"
                )
            }
            Self::MalformedDesignator => write!(
                f,
                "the international designator is not <launch year>-<launch number and piece>"
            ),
        }
    }
}

impl core::error::Error for TleError {}

/// Prints an element set as a TLE, refusing everything the columns cannot hold.
///
/// The orbital half comes from an element set — a fit produces one, so does the parser — and the
/// catalog half from the setters, because a fit produces none of it. Defaulting a catalog number or
/// a revolution count silently would attribute the elements to an object; the defaults here
/// ([`Classification::Unclassified`], no international designator, element set number and
/// revolution count zero) are the ones a TLE prints when the producer has nothing to say.
#[derive(Debug, Clone, PartialEq)]
pub struct TleBuilder {
    elements: Elements,
}

impl TleBuilder {
    /// Start from an element set: the six mean elements, the epoch, the drag terms, and whatever
    /// catalog metadata it already carries.
    #[must_use]
    pub fn from_elements(elements: &Elements) -> Self {
        Self {
            elements: elements.clone(),
        }
    }

    /// Set the catalog number.
    #[must_use]
    pub fn norad_id(mut self, norad_id: NoradId) -> Self {
        self.elements.norad_id = u64::from(norad_id.get());
        self
    }

    /// Set the classification marking in column 8.
    #[must_use]
    pub fn classification(mut self, classification: Classification) -> Self {
        self.elements.classification = classification;
        self
    }

    /// Set the international designator, in the CCSDS form `1998-067A`.
    #[must_use]
    pub fn international_designator(mut self, designator: &str) -> Self {
        self.elements.international_designator = Some(designator.to_owned());
        self
    }

    /// Set the running element-set count in columns 65–68.
    #[must_use]
    pub const fn element_set_number(mut self, number: u64) -> Self {
        self.elements.element_set_number = number;
        self
    }

    /// Set the revolution count at epoch in columns 64–68 of line 2.
    #[must_use]
    pub const fn revolution_number(mut self, number: u64) -> Self {
        self.elements.revolution_number = number;
        self
    }

    /// The element set as it will be printed.
    #[must_use]
    pub const fn elements(&self) -> &Elements {
        &self.elements
    }

    /// Print the two lines, or say which field could not be printed.
    ///
    /// # Errors
    ///
    /// [`TleError::CatalogNumberTooLarge`] past [`MAX_CATALOG_NUMBER`],
    /// [`TleError::EpochOutOfRange`] outside 1957–2056, [`TleError::MalformedDesignator`] for an
    /// international designator that is not `<year>-<rest>`, and [`TleError::OutOfRange`] naming
    /// the first field whose value its columns cannot express.
    pub fn build(&self) -> Result<Tle, TleError> {
        let e = &self.elements;
        if e.norad_id > MAX_CATALOG_NUMBER {
            return Err(TleError::CatalogNumberTooLarge {
                norad_id: e.norad_id,
            });
        }
        let catalog = format!("{:05}", e.norad_id);
        Ok(Tle {
            line1: finish(first_line(e, &catalog)?),
            line2: finish(second_line(e, &catalog)?),
        })
    }
}

/// Line 1: identity, epoch, and the three drag-related quantities.
///
/// The column boundaries are the format's, and the parser this crate reads with checks that
/// columns 2, 9, 18, 33, 44, 53, 62 and 64 (one-based) are blank — so the buffer starts as spaces
/// and only the fields are written into it.
fn first_line(e: &Elements, catalog: &str) -> Result<[u8; TLE_LINE_LENGTH], TleError> {
    let mut line1 = [b' '; TLE_LINE_LENGTH];
    line1[0] = b'1';
    place(&mut line1, 2..7, catalog, "catalog number")?;
    line1[7] = classification_column(&e.classification);
    place(
        &mut line1,
        9..17,
        &designator(e)?,
        "international designator",
    )?;
    place(&mut line1, 18..32, &epoch(e)?, "epoch")?;
    place(
        &mut line1,
        33..43,
        &leading_decimal(e.mean_motion_dot, "mean motion first derivative")?,
        "mean motion first derivative",
    )?;
    place(
        &mut line1,
        44..52,
        &decimal_point_assumed(e.mean_motion_ddot, "mean motion second derivative")?,
        "mean motion second derivative",
    )?;
    place(
        &mut line1,
        53..61,
        &decimal_point_assumed(e.drag_term, "drag term")?,
        "drag term",
    )?;
    if e.ephemeris_type > 9 {
        return Err(TleError::OutOfRange {
            field: "ephemeris type",
        });
    }
    line1[62] = b'0' + e.ephemeris_type;
    place(
        &mut line1,
        64..68,
        &bounded(e.element_set_number, 9_999, 4, "element set number")?,
        "element set number",
    )?;
    Ok(line1)
}

/// Line 2: the six mean elements and the revolution count.
fn second_line(e: &Elements, catalog: &str) -> Result<[u8; TLE_LINE_LENGTH], TleError> {
    let mut line2 = [b' '; TLE_LINE_LENGTH];
    line2[0] = b'2';
    place(&mut line2, 2..7, catalog, "catalog number")?;
    place(
        &mut line2,
        8..16,
        &angle(e.inclination, 180.0, "inclination")?,
        "inclination",
    )?;
    place(
        &mut line2,
        17..25,
        &angle(e.right_ascension, DEGREES_PER_TURN, "right ascension")?,
        "right ascension",
    )?;
    place(
        &mut line2,
        26..33,
        &eccentricity(e.eccentricity)?,
        "eccentricity",
    )?;
    place(
        &mut line2,
        34..42,
        &angle(
            e.argument_of_perigee,
            DEGREES_PER_TURN,
            "argument of perigee",
        )?,
        "argument of perigee",
    )?;
    place(
        &mut line2,
        43..51,
        &angle(e.mean_anomaly, DEGREES_PER_TURN, "mean anomaly")?,
        "mean anomaly",
    )?;
    place(
        &mut line2,
        52..63,
        &mean_motion(e.mean_motion)?,
        "mean motion",
    )?;
    place(
        &mut line2,
        63..68,
        &bounded(e.revolution_number, 99_999, 5, "revolution number")?,
        "revolution number",
    )?;
    Ok(line2)
}

/// Write one field into its columns, refusing anything that is not exactly that wide.
///
/// The last line of defence rather than the first: every field is range-checked by its own
/// formatter, and this catches the case where a formatter and its columns disagree.
fn place(
    line: &mut [u8; TLE_LINE_LENGTH],
    columns: Range<usize>,
    text: &str,
    field: &'static str,
) -> Result<(), TleError> {
    if text.len() != columns.len() || !text.is_ascii() {
        return Err(TleError::OutOfRange { field });
    }
    line[columns].copy_from_slice(text.as_bytes());
    Ok(())
}

/// Append the checksum digit and turn the buffer into a line.
///
/// Digits summed modulo ten with a minus sign counting as one — the rule published with the format,
/// and the one `crate::Satellite::from_tle` checks on the way in.
fn finish(mut line: [u8; TLE_LINE_LENGTH]) -> String {
    let sum: u32 = line[..TLE_LINE_LENGTH - 1]
        .iter()
        .map(|&b| match b {
            b'0'..=b'9' => u32::from(b - b'0'),
            b'-' => 1,
            _ => 0,
        })
        .sum();
    #[allow(clippy::cast_possible_truncation)] // sum % 10 is a single digit
    {
        line[TLE_LINE_LENGTH - 1] = b'0' + (sum % 10) as u8;
    }
    // Every byte written is ASCII: `place` refuses anything else and the rest are digits and spaces.
    String::from_utf8_lossy(&line).into_owned()
}

const fn classification_column(classification: &Classification) -> u8 {
    match classification {
        Classification::Unclassified => b'U',
        Classification::Classified => b'C',
        Classification::Secret => b'S',
    }
}

/// Columns 10–17: two digits of launch year, then the launch number and piece, left-justified.
fn designator(elements: &Elements) -> Result<String, TleError> {
    let Some(text) = elements.international_designator.as_deref() else {
        return Ok(" ".repeat(8));
    };
    let (year_text, rest) = text.split_once('-').ok_or(TleError::MalformedDesignator)?;
    let year: i32 = year_text
        .parse()
        .map_err(|_| TleError::MalformedDesignator)?;
    if !(1957..=2056).contains(&year)
        || rest.len() > 6
        || !rest.chars().all(|c| c.is_ascii_alphanumeric())
    {
        return Err(TleError::MalformedDesignator);
    }
    Ok(format!("{:02}{:<6}", year % 100, rest))
}

/// Columns 19–32: `YYDDD.DDDDDDDD`, the epoch as a two-digit year and a fractional day of year.
///
/// The fraction is written from the time of day directly rather than by rounding `ordinal + f`, so
/// no rounding can carry into the day number. A label within half a unit of the last one — 0.43 ms
/// before midnight — is written as the last instant the eight digits can name rather than as day
/// zero of the next day, which the format has no way to express.
fn epoch(elements: &Elements) -> Result<String, TleError> {
    let datetime = elements.datetime;
    let year = datetime.year();
    if !(1957..=2056).contains(&year) {
        return Err(TleError::EpochOutOfRange { year });
    }
    let seconds =
        f64::from(datetime.num_seconds_from_midnight()) + f64::from(datetime.nanosecond()) * 1e-9;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // bounded to 8 digits below
    let fraction = ((seconds / 86_400.0) * 1e8).round().min(99_999_999.0) as u64;
    Ok(format!(
        "{:02}{:03}.{fraction:08}",
        year % 100,
        datetime.ordinal()
    ))
}

/// Columns 34–43: a sign column, then a leading decimal point and eight digits.
fn leading_decimal(value: f64, field: &'static str) -> Result<String, TleError> {
    if !value.is_finite() {
        return Err(TleError::OutOfRange { field });
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // checked immediately below
    let digits = (value.abs() * 1e8).round() as u64;
    if digits > 99_999_999 {
        return Err(TleError::OutOfRange { field });
    }
    let sign = if value < 0.0 { '-' } else { ' ' };
    Ok(format!("{sign}.{digits:08}"))
}

/// The eight-column `±MMMMM±E` form meaning `±0.MMMMM × 10^±E`, used for `B*` and `n̈`.
///
/// Zero is written ` 00000-0`, which is what the catalogue prints. There is no mathematical reason
/// to prefer it over ` 00000+0` — both parse to zero — but it makes a re-emitted element set
/// byte-identical to the one it came from in the overwhelmingly common case of `n̈ = 0`.
fn decimal_point_assumed(value: f64, field: &'static str) -> Result<String, TleError> {
    if !value.is_finite() {
        return Err(TleError::OutOfRange { field });
    }
    if value == 0.0 {
        return Ok(" 00000-0".to_owned());
    }
    let magnitude = value.abs();
    #[allow(clippy::cast_possible_truncation)] // log10 of a finite non-zero f64 is within i32
    let mut exponent = magnitude.log10().floor() as i32 + 1;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // in [1e4, 1e5) by construction
    let mut mantissa = (magnitude / 10.0_f64.powi(exponent) * 1e5).round() as u64;
    // Rounding can carry the mantissa past its five columns: 9.999_999e-5 is 0.999_999_9e-4, whose
    // five rounded digits are 100000 rather than 99999.
    if mantissa >= 100_000 {
        mantissa /= 10;
        exponent += 1;
    }
    if !(-9..=9).contains(&exponent) {
        return Err(TleError::OutOfRange { field });
    }
    let sign = if value < 0.0 { '-' } else { ' ' };
    let exponent_sign = if exponent < 0 { '-' } else { '+' };
    Ok(format!(
        "{sign}{mantissa:05}{exponent_sign}{}",
        exponent.abs()
    ))
}

/// An angle in eight columns, `NNN.NNNN`, refused outside `[0, limit]`.
fn angle(value: f64, limit: f64, field: &'static str) -> Result<String, TleError> {
    if !value.is_finite() || !(0.0..=limit).contains(&value) {
        return Err(TleError::OutOfRange { field });
    }
    Ok(format!("{value:8.4}"))
}

/// The eccentricity in seven columns, decimal point assumed.
fn eccentricity(value: f64) -> Result<String, TleError> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(TleError::OutOfRange {
            field: "eccentricity",
        });
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // in [0, 1e7] by the check above
    let digits = (value * 1e7).round() as u64;
    if digits > 9_999_999 {
        // Rounding carries out of the field: 0.999_999_95 and above become an eighth digit, which
        // seven columns have nowhere to put. There is no exponent here to carry it into, unlike the
        // decimal-point-assumed fields, so the value is refused rather than renormalised.
        return Err(TleError::OutOfRange {
            field: "eccentricity",
        });
    }
    Ok(format!("{digits:07}"))
}

/// The mean motion in eleven columns, `NN.NNNNNNNN`.
///
/// The negative-zero test is not redundant with the range: `-0.0 >= 0.0` is true, so `-0.0` passes
/// the bound and prints as `-0.00000000`, which is eleven characters wide and therefore accepted by
/// [`place`] — a minus sign in columns 53–63, where the format defines only digits and a point.
fn mean_motion(value: f64) -> Result<String, TleError> {
    if !value.is_finite() || value.is_sign_negative() || !(0.0..100.0).contains(&value) {
        return Err(TleError::OutOfRange {
            field: "mean motion",
        });
    }
    Ok(format!("{value:11.8}"))
}

/// A right-justified count, refused past what its columns hold.
fn bounded(value: u64, limit: u64, width: usize, field: &'static str) -> Result<String, TleError> {
    if value > limit {
        return Err(TleError::OutOfRange { field });
    }
    Ok(format!("{value:>width$}"))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

    use super::*;

    /// The `sgp4` crate's own documented ISS element set — the one its `from_tle` doctest parses.
    const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    fn iss() -> Elements {
        Elements::from_tle(None, LINE1.as_bytes(), LINE2.as_bytes()).expect("a valid element set")
    }

    /// The strongest statement the printer can make: the bytes come back identical.
    #[test]
    fn a_parsed_element_set_prints_back_to_the_lines_it_came_from() {
        let tle = TleBuilder::from_elements(&iss()).build().unwrap();
        assert_eq!(tle.line1(), LINE1);
        assert_eq!(tle.line2(), LINE2);
        assert_eq!(tle.to_string(), format!("{LINE1}\n{LINE2}"));
    }

    #[test]
    fn every_printed_line_is_69_columns_with_a_correct_checksum() {
        let tle = TleBuilder::from_elements(&iss())
            .element_set_number(9_999)
            .revolution_number(99_999)
            .build()
            .unwrap();
        for line in [tle.line1(), tle.line2()] {
            assert_eq!(line.len(), TLE_LINE_LENGTH);
            let sum: u32 = line.as_bytes()[..68]
                .iter()
                .map(|&b| match b {
                    b'0'..=b'9' => u32::from(b - b'0'),
                    b'-' => 1,
                    _ => 0,
                })
                .sum();
            assert_eq!(sum % 10, u32::from(line.as_bytes()[68] - b'0'));
        }
        // The parser is the arbiter of whether the columns were right.
        assert!(Satellite::from_tle(tle.line1(), tle.line2()).is_ok());
    }

    /// The five numeric columns are the whole constraint, and Alpha-5 is not written into them.
    #[test]
    fn a_catalog_number_past_the_five_columns_is_refused_rather_than_alpha5_encoded() {
        assert_eq!(
            TleBuilder::from_elements(&iss())
                .norad_id(NoradId::new(270_123))
                .build()
                .unwrap_err(),
            TleError::CatalogNumberTooLarge { norad_id: 270_123 }
        );
        // The last representable one still prints.
        let tle = TleBuilder::from_elements(&iss())
            .norad_id(NoradId::new(99_999))
            .build()
            .unwrap();
        assert_eq!(&tle.line1()[2..7], "99999");
    }

    #[test]
    fn a_field_that_does_not_fit_its_columns_names_itself() {
        let mut elements = iss();
        elements.eccentricity = 1.5;
        assert_eq!(
            TleBuilder::from_elements(&elements).build().unwrap_err(),
            TleError::OutOfRange {
                field: "eccentricity"
            }
        );

        let mut elements = iss();
        elements.mean_motion = 120.0;
        assert_eq!(
            TleBuilder::from_elements(&elements).build().unwrap_err(),
            TleError::OutOfRange {
                field: "mean motion"
            }
        );

        let mut elements = iss();
        elements.inclination = 190.0;
        assert_eq!(
            TleBuilder::from_elements(&elements).build().unwrap_err(),
            TleError::OutOfRange {
                field: "inclination"
            }
        );

        let mut elements = iss();
        elements.revolution_number = 100_000;
        assert_eq!(
            TleBuilder::from_elements(&elements).build().unwrap_err(),
            TleError::OutOfRange {
                field: "revolution number"
            }
        );
    }

    #[test]
    fn an_epoch_the_two_digit_year_cannot_name_is_refused() {
        let mut elements = iss();
        elements.datetime = sgp4::chrono::NaiveDate::from_ymd_opt(2057, 3, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        assert_eq!(
            TleBuilder::from_elements(&elements).build().unwrap_err(),
            TleError::EpochOutOfRange { year: 2057 }
        );
    }

    #[test]
    fn a_designator_that_is_not_year_dash_rest_is_refused_rather_than_padded() {
        assert_eq!(
            TleBuilder::from_elements(&iss())
                .international_designator("98067A")
                .build()
                .unwrap_err(),
            TleError::MalformedDesignator
        );
        // Absent is not malformed: the columns are simply blank, as the catalogue prints them.
        let mut elements = iss();
        elements.international_designator = None;
        let tle = TleBuilder::from_elements(&elements).build().unwrap();
        assert_eq!(&tle.line1()[9..17], "        ");
    }

    /// The exponential fields are the only ones with a non-obvious encoding, so they get their own
    /// round trip through the parser this crate reads with.
    #[test]
    fn the_decimal_point_assumed_fields_survive_a_round_trip() {
        for drag in [0.0, 1.1606e-4, -1.1606e-4, 5.0e-9, -9.9999e-1] {
            let mut elements = iss();
            elements.drag_term = drag;
            elements.mean_motion_ddot = drag;
            let tle = TleBuilder::from_elements(&elements).build().unwrap();
            let parsed =
                Elements::from_tle(None, tle.line1().as_bytes(), tle.line2().as_bytes()).unwrap();
            let tolerance = drag.abs() * 1e-4 + 1e-18;
            assert!(
                (parsed.drag_term - drag).abs() <= tolerance,
                "B* {drag:e} came back {:e}",
                parsed.drag_term
            );
            assert!((parsed.mean_motion_ddot - drag).abs() <= tolerance);
        }
    }

    /// A carry out of the five mantissa columns must move the exponent, not print six digits.
    #[test]
    fn a_mantissa_that_rounds_past_five_digits_carries_into_the_exponent() {
        let text = decimal_point_assumed(9.999_999e-5, "drag term").unwrap();
        assert_eq!(text, " 10000-3");
        assert_eq!(decimal_point_assumed(0.0, "drag term").unwrap(), " 00000-0");
        assert_eq!(decimal_point_assumed(1.0, "drag term").unwrap(), " 10000+1");
        assert!(decimal_point_assumed(1e-30, "drag term").is_err());
    }

    #[test]
    fn the_parameter_vector_round_trips_through_the_element_set() {
        let elements = iss();
        let model = TleModel::new(elements.clone(), Convention::Reference);
        let packed = TleModel::parameters(&elements);
        let unpacked = model.elements(&packed);
        assert_eq!(unpacked.mean_motion, elements.mean_motion);
        assert_eq!(unpacked.eccentricity, elements.eccentricity);
        assert_eq!(unpacked.mean_anomaly, elements.mean_anomaly);

        let packed = TleModel::parameters_with_drag(&elements);
        assert_eq!(packed[6], elements.drag_term);
        assert_eq!(
            model.elements_with_drag(&packed).drag_term,
            elements.drag_term
        );
    }

    /// A negative angle is a legal parameter and prints as its positive equivalent.
    #[test]
    fn a_wrapped_angle_prints_reduced_and_propagates_to_the_same_state() {
        let elements = iss();
        let model = TleModel::new(elements.clone(), Convention::Reference);
        let mut wrapped = TleModel::parameters(&elements);
        wrapped[5] -= 360.0;
        let reduced = model.elements(&wrapped);
        assert!((reduced.mean_anomaly - elements.mean_anomaly).abs() < 1e-9);

        let direct = <TleModel as PropagatorBuilder<6>>::build(&model, &wrapped).unwrap();
        let plain =
            <TleModel as PropagatorBuilder<6>>::build(&model, &TleModel::parameters(&elements))
                .unwrap();
        let (a, b) = (
            direct.state_at(3600.0).unwrap(),
            plain.state_at(3600.0).unwrap(),
        );
        // Measured: 1.9e-9 m, the difference between rounding `(M − 360)·π/180` and `M·π/180`.
        for k in 0..6 {
            assert!(
                (a[k] - b[k]).abs() < 1e-8,
                "component {k}: {} vs {}",
                a[k],
                b[k]
            );
        }
    }

    #[test]
    fn a_parameter_vector_outside_the_model_domain_is_refused() {
        let model = TleModel::new(iss(), Convention::Reference);
        let mut bad = TleModel::parameters(&iss());
        bad[1] = 1.5;
        assert!(<TleModel as PropagatorBuilder<6>>::build(&model, &bad).is_none());
        bad = TleModel::parameters(&iss());
        bad[0] = -1.0;
        assert!(<TleModel as PropagatorBuilder<6>>::build(&model, &bad).is_none());
    }

    /// A non-finite parameter is the only witness the domain check has to itself.
    ///
    /// Every *numeric* bound sgp4 applies — `kozai_mean_motion <= 0`, `eccentricity < 0 ||
    /// eccentricity >= 1` — is a comparison, and every comparison against a `NaN` is false. So a
    /// `NaN` mean motion or eccentricity passes all of them and comes back as a `Satellite` full of
    /// `NaN`s. The refusal here is the model's own or it does not happen.
    #[test]
    fn a_non_finite_parameter_is_refused_by_the_model_rather_than_by_sgp4() {
        let model = TleModel::new(iss(), Convention::Reference);
        for index in [0, 1] {
            let mut bad = TleModel::parameters(&iss());
            bad[index] = f64::NAN;
            assert!(
                <TleModel as PropagatorBuilder<6>>::build(&model, &bad).is_none(),
                "parameter {index} accepted a NaN"
            );
        }
    }

    /// The epoch is the template's, not the calendar type's default — which is 1970-01-01, an
    /// epoch a TLE can neither hold nor be distinguished from by any propagated state.
    #[test]
    fn the_model_reports_the_template_epoch_and_not_a_default_one() {
        let elements = iss();
        let model = TleModel::new(elements.clone(), Convention::Reference);
        assert_eq!(
            model.epoch().date(),
            NaiveDate::from_ymd_opt(2008, 9, 20).unwrap()
        );
        assert_eq!(model.epoch(), elements.datetime);
    }

    /// The two conversions inside [`seed_elements`] are invisible to anything that reads the epoch
    /// back through the same code: the factor between a fractional second and the nanosecond count
    /// `chrono` wants, and the split of that count into whole seconds and a remainder. The numbers
    /// asserted here are computed from the definition — `12.25 s` is exact in binary, so it is
    /// `12.25 × 10⁹ = 12_250_000_000` ns, which is `12` whole seconds and `250_000_000` ns.
    #[test]
    fn a_seeded_epoch_splits_a_fractional_second_into_seconds_and_nanoseconds() {
        const PARAMETERS: [f64; 6] = [15.5, 0.001, 51.6, 247.0, 130.0, 325.0];
        let epoch = UtcDateTime::new(2008, 9, 20, 12, 25, 12.25).unwrap();
        let elements = seed_elements(epoch, NoradId::new(25_544), &PARAMETERS).unwrap();
        assert_eq!(elements.datetime.second(), 12);
        assert_eq!(elements.datetime.nanosecond(), 250_000_000);
        assert_eq!(
            elements.datetime,
            NaiveDate::from_ymd_opt(2008, 9, 20)
                .unwrap()
                .and_hms_nano_opt(12, 25, 12, 250_000_000)
                .unwrap()
        );

        // A leap-second label has no Gregorian representation, so it is refused rather than rounded
        // into the next minute.
        let leap = UtcDateTime::new(2008, 12, 31, 23, 59, 60.5).unwrap();
        assert_eq!(
            seed_elements(leap, NoradId::new(25_544), &PARAMETERS).unwrap_err(),
            TleError::EpochOutOfRange { year: 2008 }
        );
    }

    /// A refusal that does not say what was refused is a refusal a caller cannot act on.
    #[test]
    fn every_refusal_prints_what_could_not_be_named_and_what_to_do_about_it() {
        assert_eq!(
            TleError::CatalogNumberTooLarge { norad_id: 270_123 }.to_string(),
            "catalog number 270123 needs more than the 99999 a TLE can name; \
             publish CCSDS OMM instead"
        );
        assert_eq!(
            TleError::EpochOutOfRange { year: 2057 }.to_string(),
            "epoch year 2057 is outside the 1957-2056 range two digits can name"
        );
        assert_eq!(
            TleError::OutOfRange {
                field: "eccentricity"
            }
            .to_string(),
            "the eccentricity does not fit the columns the format gives it"
        );
        assert_eq!(
            TleError::MalformedDesignator.to_string(),
            "the international designator is not <launch year>-<launch number and piece>"
        );
    }

    /// One column holds one digit. Ten is not two columns, it is `b'0' + 10`, which is `:`.
    #[test]
    fn an_ephemeris_type_past_one_digit_is_refused_rather_than_printed_as_punctuation() {
        let mut elements = iss();
        elements.ephemeris_type = 9;
        let tle = TleBuilder::from_elements(&elements).build().unwrap();
        assert_eq!(&tle.line1()[62..63], "9");
        assert!(Satellite::from_tle(tle.line1(), tle.line2()).is_ok());
        let parsed =
            Elements::from_tle(None, tle.line1().as_bytes(), tle.line2().as_bytes()).unwrap();
        assert_eq!(parsed.ephemeris_type, 9);

        let mut elements = iss();
        elements.ephemeris_type = 10;
        assert_eq!(
            TleBuilder::from_elements(&elements).build().unwrap_err(),
            TleError::OutOfRange {
                field: "ephemeris type"
            }
        );
    }

    /// `place` is the last line of defence, so it is tested on the two things it is the last line of
    /// defence against: a formatter whose output is the wrong width, and one whose output is not
    /// ASCII. Bytes are columns only while the text is ASCII.
    #[test]
    fn place_refuses_text_that_is_not_exactly_as_wide_as_its_columns() {
        let mut line = [b' '; TLE_LINE_LENGTH];
        let refused = Err(TleError::OutOfRange { field: "field" });
        assert_eq!(place(&mut line, 0..4, "abc", "field"), refused);
        assert_eq!(place(&mut line, 0..4, "abcde", "field"), refused);
        assert_eq!(place(&mut line, 0..2, "é", "field"), refused);
        assert_eq!(line, [b' '; TLE_LINE_LENGTH]);

        assert_eq!(place(&mut line, 0..3, "abc", "field"), Ok(()));
        assert_eq!(&line[..4], b"abc ");
    }

    /// The checksum digit is computed from the 68 columns before it. The column it lands in is an
    /// output, so whatever is sitting there is overwritten and never summed.
    #[test]
    fn the_checksum_covers_the_columns_before_it_and_not_the_one_it_is_written_into() {
        let mut buffer = [b' '; TLE_LINE_LENGTH];
        buffer[0] = b'1';
        buffer[TLE_LINE_LENGTH - 1] = b'7';
        let line = finish(buffer);
        assert_eq!(line.len(), TLE_LINE_LENGTH);
        assert_eq!(&line[TLE_LINE_LENGTH - 1..], "1");
    }

    /// Six columns hold the launch number and piece, left-justified: six characters print, a
    /// seventh has nowhere to go, and neither has anything to do with what those characters are.
    #[test]
    fn a_designator_at_the_last_legal_length_prints_and_one_character_past_it_is_refused() {
        let tle = TleBuilder::from_elements(&iss())
            .international_designator("1998-067ABC")
            .build()
            .unwrap();
        assert_eq!(&tle.line1()[9..17], "98067ABC");
        assert!(Satellite::from_tle(tle.line1(), tle.line2()).is_ok());

        // Too long, and — at a length that fits — not alphanumeric. Both are the designator's own
        // refusal rather than the column's.
        for text in ["1998-067ABCD", "1998-067 A", "1998-067-A"] {
            assert_eq!(
                TleBuilder::from_elements(&iss())
                    .international_designator(text)
                    .build()
                    .unwrap_err(),
                TleError::MalformedDesignator,
                "{text}"
            );
        }
    }

    /// Two digits of launch year name 1957–2056 and nothing else, at both ends.
    #[test]
    fn a_designator_year_outside_the_two_digit_window_is_refused_at_both_ends() {
        for (text, columns) in [("1957-001A", "57001A  "), ("2056-001A", "56001A  ")] {
            let tle = TleBuilder::from_elements(&iss())
                .international_designator(text)
                .build()
                .unwrap();
            assert_eq!(&tle.line1()[9..17], columns);
        }
        for text in ["1956-001A", "2057-001A"] {
            assert_eq!(
                TleBuilder::from_elements(&iss())
                    .international_designator(text)
                    .build()
                    .unwrap_err(),
                TleError::MalformedDesignator,
                "{text}"
            );
        }
    }

    /// Eight digits after an assumed decimal point, and a sign column that is a sign column: zero
    /// is not negative, `.99999999` is the widest value that fits, and what rounds to a ninth digit
    /// is refused rather than printed one column wide.
    #[test]
    fn the_first_derivative_prints_eight_digits_and_refuses_a_ninth() {
        assert_eq!(leading_decimal(0.0, "n dot").unwrap(), " .00000000");
        assert_eq!(leading_decimal(-0.0, "n dot").unwrap(), " .00000000");
        assert_eq!(
            leading_decimal(0.999_999_99, "n dot").unwrap(),
            " .99999999"
        );
        assert_eq!(
            leading_decimal(-0.999_999_99, "n dot").unwrap(),
            "-.99999999"
        );
        for value in [1.0, f64::INFINITY, f64::NAN] {
            assert_eq!(
                leading_decimal(value, "n dot"),
                Err(TleError::OutOfRange { field: "n dot" }),
                "{value}"
            );
        }
    }

    /// The exponent column carries a sign of its own, and it prints at both ends of the one digit
    /// it has. Zero is not a negative exponent — the ` 00000-0` of the zero value is a spelling
    /// choice made before this branch is reached, not the sign rule.
    #[test]
    fn the_decimal_point_assumed_exponent_prints_its_own_sign_at_both_ends() {
        assert_eq!(decimal_point_assumed(0.5, "b star").unwrap(), " 50000+0");
        assert_eq!(decimal_point_assumed(-0.5, "b star").unwrap(), "-50000+0");
        assert_eq!(decimal_point_assumed(5e8, "b star").unwrap(), " 50000+9");
        assert_eq!(decimal_point_assumed(5e-10, "b star").unwrap(), " 50000-9");
        for value in [5e9, 5e-11, f64::NAN] {
            assert_eq!(
                decimal_point_assumed(value, "b star"),
                Err(TleError::OutOfRange { field: "b star" }),
                "{value:e}"
            );
        }
    }

    /// Seven columns, decimal point assumed, and no sign column at all — so the range check is what
    /// keeps a negative eccentricity from being cast to `0` and printed as a circular orbit.
    #[test]
    fn the_eccentricity_prints_both_ends_of_its_seven_columns_and_refuses_what_is_outside_them() {
        assert_eq!(eccentricity(0.0).unwrap(), "0000000");
        assert_eq!(eccentricity(0.999_999_9).unwrap(), "9999999");
        for value in [
            -f64::EPSILON,
            -0.5,
            0.999_999_99,
            1.0,
            f64::NAN,
            f64::INFINITY,
        ] {
            assert_eq!(
                eccentricity(value),
                Err(TleError::OutOfRange {
                    field: "eccentricity"
                }),
                "{value}"
            );
        }
    }

    /// Eleven columns hold `NN.NNNNNNNN`, with no sign column and no third integer digit.
    #[test]
    fn the_mean_motion_prints_at_the_column_width_and_refuses_anything_wider() {
        assert_eq!(mean_motion(0.0).unwrap(), " 0.00000000");
        assert_eq!(mean_motion(99.999_999_99).unwrap(), "99.99999999");
        // `-0.0` is in the list because the range alone does not refuse it: `-0.0 >= 0.0` holds, and
        // `format!("{:11.8}")` then prints `-0.00000000`, eleven columns wide and therefore accepted
        // by `place` — a minus sign in columns 53–63, where the format defines only digits and a point.
        for value in [-0.0, -1.0, 100.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                mean_motion(value),
                Err(TleError::OutOfRange {
                    field: "mean motion"
                }),
                "{value}"
            );
        }

        // Inside the range check and still one column too wide: rounding carries `99.999999999`
        // into three integer digits. `place` is what catches it, and the field it names is still
        // the mean motion's.
        let mut elements = iss();
        elements.mean_motion = 99.999_999_999;
        assert_eq!(
            TleBuilder::from_elements(&elements).build().unwrap_err(),
            TleError::OutOfRange {
                field: "mean motion"
            }
        );
    }
}
