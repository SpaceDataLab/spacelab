//! Orbit Parameter Message — a single Cartesian state, optionally with elements, covariance and
//! planned manoeuvres (CCSDS 502.0-B-3 §3).
//!
//! The OPM is the smallest useful state handover: one epoch, one state vector, and whatever the
//! producer knows about the spacecraft that carries it. Everything past the state vector is optional
//! in the standard and optional here, with one exception made deliberately — see [`Anomaly`].

use spacelab_core::Angle;
use spacelab_core::UtcDateTime;

use crate::kvn::{
    COVARIANCE_ENTRIES, Item, M_PER_KM, M2_PER_KM2, NdmError, OPM_COVARIANCE_KEYWORDS, epoch, scan,
    value, write_epoch,
};

/// Cubic metres per cubic kilometre — the scaling `GM` takes from `km³/s²` to SI.
const M3_PER_KM3: f64 = 1.0e9;

/// The version keyword an OPM must open with.
const VERSION_KEYWORD: &str = "CCSDS_OPM_VERS";

/// Which anomaly the producer chose to state.
///
/// CCSDS 502.0-B-3 §3.2.4.4 admits `TRUE_ANOMALY` or `MEAN_ANOMALY` and requires exactly one of
/// them. Two `Option<Angle>` fields would admit both and neither; a sum type admits exactly what the
/// standard does, and a consumer that forgets which one it is holding cannot compile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Anomaly {
    /// `TRUE_ANOMALY`.
    True(Angle),
    /// `MEAN_ANOMALY`.
    Mean(Angle),
}

/// The optional Keplerian block, in SI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keplerian {
    /// Semi-major axis \[m].
    pub semi_major_axis: f64,
    /// Eccentricity.
    pub eccentricity: f64,
    /// Inclination.
    pub inclination: Angle,
    /// Right ascension of the ascending node.
    pub right_ascension: Angle,
    /// Argument of pericentre.
    pub argument_of_pericenter: Angle,
    /// Whichever anomaly the producer stated.
    pub anomaly: Anomaly,
    /// Gravitational parameter of the central body \[m³/s²].
    pub gm: f64,
}

/// The optional spacecraft-parameters block, in SI. Every keyword in it is individually optional.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Spacecraft {
    /// Mass \[kg].
    pub mass: Option<f64>,
    /// Area exposed to solar radiation \[m²].
    pub solar_rad_area: Option<f64>,
    /// Solar radiation pressure coefficient.
    pub solar_rad_coeff: Option<f64>,
    /// Area exposed to drag \[m²].
    pub drag_area: Option<f64>,
    /// Drag coefficient.
    pub drag_coeff: Option<f64>,
}

impl Spacecraft {
    /// Whether the producer stated any spacecraft parameter at all.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.mass.is_none()
            && self.solar_rad_area.is_none()
            && self.solar_rad_coeff.is_none()
            && self.drag_area.is_none()
            && self.drag_coeff.is_none()
    }
}

/// The optional 6×6 position/velocity covariance, as its lower triangle read row by row.
#[derive(Debug, Clone, PartialEq)]
pub struct Covariance {
    /// `COV_REF_FRAME`, when the producer states one; otherwise the covariance is in `REF_FRAME`.
    pub ref_frame: Option<CovFrame>,
    /// Lower triangle, row by row, in SI: m², m²/s, m²/s².
    pub lower: [f64; COVARIANCE_ENTRIES],
}

/// A covariance reference frame name, kept as the file's own token.
///
/// A `String` rather than an enum: the frame set is an interoperability registry that grows without
/// this crate, and refusing to parse a message because its frame is newer than our enum would make
/// the reader less useful than the file.
pub type CovFrame = String;

/// One planned manoeuvre (CCSDS 502.0-B-3 §3.2.4.6). Every keyword in the block is mandatory.
#[derive(Debug, Clone, PartialEq)]
pub struct Maneuver {
    /// `MAN_EPOCH_IGNITION`.
    pub epoch_ignition: UtcDateTime,
    /// Burn duration \[s]. Zero for an impulsive manoeuvre.
    pub duration: f64,
    /// Mass change \[kg], negative for propellant spent.
    pub delta_mass: f64,
    /// The frame `delta_v` is stated in.
    pub ref_frame: String,
    /// Velocity increment \[m/s].
    pub delta_v: [f64; 3],
}

/// A parsed Orbit Parameter Message, in SI throughout.
#[derive(Debug, Clone, PartialEq)]
pub struct Opm {
    /// `CREATION_DATE`.
    pub creation_date: UtcDateTime,
    /// `ORIGINATOR`.
    pub originator: String,
    /// `MESSAGE_ID`, optional since version 3.0.
    pub message_id: Option<String>,
    /// `OBJECT_NAME`.
    pub object_name: String,
    /// `OBJECT_ID`, the international designator.
    pub object_id: String,
    /// `CENTER_NAME`, the body the state is relative to.
    pub center_name: String,
    /// `REF_FRAME`.
    pub ref_frame: String,
    /// `REF_FRAME_EPOCH`, present only for a frame whose realisation is epoch-dependent.
    pub ref_frame_epoch: Option<UtcDateTime>,
    /// `TIME_SYSTEM`.
    pub time_system: String,
    /// `EPOCH`, the instant the state is valid at.
    pub epoch: UtcDateTime,
    /// Position \[m].
    pub position: [f64; 3],
    /// Velocity \[m/s].
    pub velocity: [f64; 3],
    /// The optional Keplerian block.
    pub keplerian: Option<Keplerian>,
    /// The optional spacecraft-parameters block.
    pub spacecraft: Spacecraft,
    /// The optional covariance block.
    pub covariance: Option<Covariance>,
    /// Planned manoeuvres, in file order.
    pub maneuvers: Vec<Maneuver>,
    /// `USER_DEFINED_*` keywords, as `(suffix, value)` in file order.
    pub user_defined: Vec<(String, String)>,
}

/// Everything an OPM may carry, before any of it is required to be there.
#[derive(Default)]
struct Builder {
    creation_date: Option<UtcDateTime>,
    originator: Option<String>,
    message_id: Option<String>,
    object_name: Option<String>,
    object_id: Option<String>,
    center_name: Option<String>,
    ref_frame: Option<String>,
    ref_frame_epoch: Option<UtcDateTime>,
    time_system: Option<String>,
    epoch: Option<UtcDateTime>,
    position: [Option<f64>; 3],
    velocity: [Option<f64>; 3],
    semi_major_axis: Option<f64>,
    eccentricity: Option<f64>,
    inclination: Option<Angle>,
    right_ascension: Option<Angle>,
    argument_of_pericenter: Option<Angle>,
    anomaly: Option<Anomaly>,
    gm: Option<f64>,
    spacecraft: Spacecraft,
    cov_ref_frame: Option<CovFrame>,
    covariance: [Option<f64>; COVARIANCE_ENTRIES],
    /// The line the last covariance keyword was read from, so a short block can name where it ended.
    cov_line: usize,
    maneuvers: Vec<Maneuver>,
    pending: Option<PendingManeuver>,
    user_defined: Vec<(String, String)>,
}

#[derive(Default)]
struct PendingManeuver {
    epoch_ignition: Option<UtcDateTime>,
    duration: Option<f64>,
    delta_mass: Option<f64>,
    ref_frame: Option<String>,
    delta_v: [Option<f64>; 3],
}

impl PendingManeuver {
    fn finish(self) -> Result<Maneuver, NdmError> {
        let missing = |keyword| NdmError::MissingKeyword { keyword };
        Ok(Maneuver {
            epoch_ignition: self
                .epoch_ignition
                .ok_or_else(|| missing("MAN_EPOCH_IGNITION"))?,
            duration: self.duration.ok_or_else(|| missing("MAN_DURATION"))?,
            delta_mass: self.delta_mass.ok_or_else(|| missing("MAN_DELTA_MASS"))?,
            ref_frame: self.ref_frame.ok_or_else(|| missing("MAN_REF_FRAME"))?,
            delta_v: [
                self.delta_v[0].ok_or_else(|| missing("MAN_DV_1"))?,
                self.delta_v[1].ok_or_else(|| missing("MAN_DV_2"))?,
                self.delta_v[2].ok_or_else(|| missing("MAN_DV_3"))?,
            ],
        })
    }
}

/// Read a CCSDS Orbit Parameter Message in KVN.
///
/// # Errors
///
/// [`NdmError::NotThisMessage`] if the file does not open with `CCSDS_OPM_VERS`, and otherwise the
/// first structural failure; see [`NdmError`].
pub fn parse_opm(text: &str) -> Result<Opm, NdmError> {
    let mut builder = Builder::default();
    let mut opened = false;

    for item in scan(text) {
        let Item::Pair { line, key, value } = item else {
            continue;
        };
        if key == VERSION_KEYWORD {
            opened = true;
            continue;
        }
        if !opened {
            return Err(NdmError::NotThisMessage {
                expected: VERSION_KEYWORD,
            });
        }
        builder.set(line, key, value)?;
    }

    if !opened {
        return Err(NdmError::NotThisMessage {
            expected: VERSION_KEYWORD,
        });
    }
    builder.finish()
}

impl Builder {
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per CCSDS keyword; a table, not logic"
    )]
    fn set(&mut self, line: usize, key: &str, text: &str) -> Result<(), NdmError> {
        // A manoeuvre block opens on its ignition epoch, so any other keyword ends the one in hand.
        if key == "MAN_EPOCH_IGNITION" {
            self.flush_maneuver()?;
            self.pending = Some(PendingManeuver {
                epoch_ignition: Some(epoch(line, "MAN_EPOCH_IGNITION", text)?),
                ..PendingManeuver::default()
            });
            return Ok(());
        }
        if let Some(pending) = self.pending.as_mut() {
            match key {
                "MAN_DURATION" => {
                    pending.duration = Some(value(line, "MAN_DURATION", text)?);
                    return Ok(());
                }
                "MAN_DELTA_MASS" => {
                    pending.delta_mass = Some(value(line, "MAN_DELTA_MASS", text)?);
                    return Ok(());
                }
                "MAN_REF_FRAME" => {
                    pending.ref_frame = Some(text.to_owned());
                    return Ok(());
                }
                "MAN_DV_1" | "MAN_DV_2" | "MAN_DV_3" => {
                    let index = usize::from(key.as_bytes()[7] - b'1');
                    let keyword = ["MAN_DV_1", "MAN_DV_2", "MAN_DV_3"][index];
                    pending.delta_v[index] = Some(value::<f64>(line, keyword, text)? * M_PER_KM);
                    return Ok(());
                }
                _ => self.flush_maneuver()?,
            }
        }

        if let Some(suffix) = key.strip_prefix("USER_DEFINED_") {
            self.user_defined.push((suffix.to_owned(), text.to_owned()));
            return Ok(());
        }

        if let Some(index) = OPM_COVARIANCE_KEYWORDS.iter().position(|name| *name == key) {
            let keyword = OPM_COVARIANCE_KEYWORDS[index];
            self.covariance[index] = Some(value::<f64>(line, keyword, text)? * M2_PER_KM2);
            self.cov_line = line;
            return Ok(());
        }

        match key {
            "CREATION_DATE" => self.creation_date = Some(epoch(line, "CREATION_DATE", text)?),
            "ORIGINATOR" => self.originator = Some(text.to_owned()),
            "MESSAGE_ID" => self.message_id = Some(text.to_owned()),
            "OBJECT_NAME" => self.object_name = Some(text.to_owned()),
            "OBJECT_ID" => self.object_id = Some(text.to_owned()),
            "CENTER_NAME" => self.center_name = Some(text.to_owned()),
            "REF_FRAME" => self.ref_frame = Some(text.to_owned()),
            "REF_FRAME_EPOCH" => self.ref_frame_epoch = Some(epoch(line, "REF_FRAME_EPOCH", text)?),
            "TIME_SYSTEM" => self.time_system = Some(text.to_owned()),
            "EPOCH" => self.epoch = Some(epoch(line, "EPOCH", text)?),
            "X" => self.position[0] = Some(value::<f64>(line, "X", text)? * M_PER_KM),
            "Y" => self.position[1] = Some(value::<f64>(line, "Y", text)? * M_PER_KM),
            "Z" => self.position[2] = Some(value::<f64>(line, "Z", text)? * M_PER_KM),
            "X_DOT" => self.velocity[0] = Some(value::<f64>(line, "X_DOT", text)? * M_PER_KM),
            "Y_DOT" => self.velocity[1] = Some(value::<f64>(line, "Y_DOT", text)? * M_PER_KM),
            "Z_DOT" => self.velocity[2] = Some(value::<f64>(line, "Z_DOT", text)? * M_PER_KM),
            "SEMI_MAJOR_AXIS" => {
                self.semi_major_axis =
                    Some(value::<f64>(line, "SEMI_MAJOR_AXIS", text)? * M_PER_KM);
            }
            "ECCENTRICITY" => self.eccentricity = Some(value(line, "ECCENTRICITY", text)?),
            "INCLINATION" => {
                self.inclination = Some(Angle::from_degrees(value(line, "INCLINATION", text)?));
            }
            "RA_OF_ASC_NODE" => {
                self.right_ascension =
                    Some(Angle::from_degrees(value(line, "RA_OF_ASC_NODE", text)?));
            }
            "ARG_OF_PERICENTER" => {
                self.argument_of_pericenter =
                    Some(Angle::from_degrees(value(line, "ARG_OF_PERICENTER", text)?));
            }
            "TRUE_ANOMALY" => {
                self.anomaly = Some(Anomaly::True(Angle::from_degrees(value(
                    line,
                    "TRUE_ANOMALY",
                    text,
                )?)));
            }
            "MEAN_ANOMALY" => {
                self.anomaly = Some(Anomaly::Mean(Angle::from_degrees(value(
                    line,
                    "MEAN_ANOMALY",
                    text,
                )?)));
            }
            "GM" => self.gm = Some(value::<f64>(line, "GM", text)? * M3_PER_KM3),
            "MASS" => self.spacecraft.mass = Some(value(line, "MASS", text)?),
            "SOLAR_RAD_AREA" => {
                self.spacecraft.solar_rad_area = Some(value(line, "SOLAR_RAD_AREA", text)?);
            }
            "SOLAR_RAD_COEFF" => {
                self.spacecraft.solar_rad_coeff = Some(value(line, "SOLAR_RAD_COEFF", text)?);
            }
            "DRAG_AREA" => self.spacecraft.drag_area = Some(value(line, "DRAG_AREA", text)?),
            "DRAG_COEFF" => self.spacecraft.drag_coeff = Some(value(line, "DRAG_COEFF", text)?),
            "COV_REF_FRAME" => self.cov_ref_frame = Some(text.to_owned()),
            _ => {}
        }
        Ok(())
    }

    fn flush_maneuver(&mut self) -> Result<(), NdmError> {
        if let Some(pending) = self.pending.take() {
            self.maneuvers.push(pending.finish()?);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Opm, NdmError> {
        self.flush_maneuver()?;
        let missing = |keyword| NdmError::MissingKeyword { keyword };
        let component = |slot: Option<f64>, keyword| slot.ok_or_else(|| missing(keyword));

        // A covariance is all 21 entries or none: a partially stated one is a matrix with holes,
        // and zero-filling the holes produces a matrix that is not the producer's and may not even
        // be positive semi-definite.
        let stated = self.covariance.iter().filter(|slot| slot.is_some()).count();
        let covariance = match stated {
            0 => None,
            COVARIANCE_ENTRIES => {
                let mut lower = [0.0; COVARIANCE_ENTRIES];
                for (slot, entry) in self.covariance.iter().zip(lower.iter_mut()) {
                    *entry = slot.unwrap_or_default();
                }
                Some(Covariance {
                    ref_frame: self.cov_ref_frame,
                    lower,
                })
            }
            entries => {
                return Err(NdmError::ShortCovariance {
                    line: self.cov_line,
                    entries,
                });
            }
        };

        // The Keplerian block is likewise all or nothing, save for `GM`, which CCSDS 502.0-B-3
        // table 3-3 marks conditional: it is required only when the central body is not one the
        // recipient can look up. Present with holes is refused; absent entirely is fine.
        let keplerian = if self.semi_major_axis.is_none()
            && self.eccentricity.is_none()
            && self.inclination.is_none()
            && self.right_ascension.is_none()
            && self.argument_of_pericenter.is_none()
            && self.anomaly.is_none()
        {
            None
        } else {
            Some(Keplerian {
                semi_major_axis: self
                    .semi_major_axis
                    .ok_or_else(|| missing("SEMI_MAJOR_AXIS"))?,
                eccentricity: self.eccentricity.ok_or_else(|| missing("ECCENTRICITY"))?,
                inclination: self.inclination.ok_or_else(|| missing("INCLINATION"))?,
                right_ascension: self
                    .right_ascension
                    .ok_or_else(|| missing("RA_OF_ASC_NODE"))?,
                argument_of_pericenter: self
                    .argument_of_pericenter
                    .ok_or_else(|| missing("ARG_OF_PERICENTER"))?,
                anomaly: self.anomaly.ok_or_else(|| missing("TRUE_ANOMALY"))?,
                gm: self.gm.unwrap_or(f64::NAN),
            })
        };

        Ok(Opm {
            creation_date: self.creation_date.ok_or_else(|| missing("CREATION_DATE"))?,
            originator: self.originator.ok_or_else(|| missing("ORIGINATOR"))?,
            message_id: self.message_id,
            object_name: self.object_name.ok_or_else(|| missing("OBJECT_NAME"))?,
            object_id: self.object_id.ok_or_else(|| missing("OBJECT_ID"))?,
            center_name: self.center_name.ok_or_else(|| missing("CENTER_NAME"))?,
            ref_frame: self.ref_frame.ok_or_else(|| missing("REF_FRAME"))?,
            ref_frame_epoch: self.ref_frame_epoch,
            time_system: self.time_system.ok_or_else(|| missing("TIME_SYSTEM"))?,
            epoch: self.epoch.ok_or_else(|| missing("EPOCH"))?,
            position: [
                component(self.position[0], "X")?,
                component(self.position[1], "Y")?,
                component(self.position[2], "Z")?,
            ],
            velocity: [
                component(self.velocity[0], "X_DOT")?,
                component(self.velocity[1], "Y_DOT")?,
                component(self.velocity[2], "Z_DOT")?,
            ],
            keplerian,
            spacecraft: self.spacecraft,
            covariance,
            maneuvers: self.maneuvers,
            user_defined: self.user_defined,
        })
    }
}

/// Write a CCSDS Orbit Parameter Message in KVN, at version 3.0.
///
/// The values are emitted at whatever decimal precision `f64` needs to reproduce itself, so that
/// `parse_opm(&write_opm(&parse_opm(text)?))` returns the first parse bit for bit wherever the unit
/// scaling permits — which is everywhere except the angles; see the crate documentation.
#[must_use]
pub fn write_opm(opm: &Opm) -> String {
    use core::fmt::Write as _;
    let mut out = String::new();
    let mut pair = |key: &str, value: &str| {
        let _ = writeln!(out, "{key} = {value}");
    };

    pair(VERSION_KEYWORD, "3.0");
    let mut date = String::new();
    write_epoch(&mut date, opm.creation_date);
    pair("CREATION_DATE", &date);
    pair("ORIGINATOR", &opm.originator);
    if let Some(id) = &opm.message_id {
        pair("MESSAGE_ID", id);
    }

    pair("OBJECT_NAME", &opm.object_name);
    pair("OBJECT_ID", &opm.object_id);
    pair("CENTER_NAME", &opm.center_name);
    pair("REF_FRAME", &opm.ref_frame);
    if let Some(frame_epoch) = opm.ref_frame_epoch {
        let mut text = String::new();
        write_epoch(&mut text, frame_epoch);
        pair("REF_FRAME_EPOCH", &text);
    }
    pair("TIME_SYSTEM", &opm.time_system);

    let mut text = String::new();
    write_epoch(&mut text, opm.epoch);
    pair("EPOCH", &text);
    for (key, metres) in ["X", "Y", "Z"].iter().zip(opm.position) {
        pair(key, &(metres / M_PER_KM).to_string());
    }
    for (key, metres) in ["X_DOT", "Y_DOT", "Z_DOT"].iter().zip(opm.velocity) {
        pair(key, &(metres / M_PER_KM).to_string());
    }

    if let Some(kep) = &opm.keplerian {
        pair(
            "SEMI_MAJOR_AXIS",
            &(kep.semi_major_axis / M_PER_KM).to_string(),
        );
        pair("ECCENTRICITY", &kep.eccentricity.to_string());
        pair("INCLINATION", &kep.inclination.degrees().to_string());
        pair("RA_OF_ASC_NODE", &kep.right_ascension.degrees().to_string());
        pair(
            "ARG_OF_PERICENTER",
            &kep.argument_of_pericenter.degrees().to_string(),
        );
        match kep.anomaly {
            Anomaly::True(angle) => pair("TRUE_ANOMALY", &angle.degrees().to_string()),
            Anomaly::Mean(angle) => pair("MEAN_ANOMALY", &angle.degrees().to_string()),
        }
        if kep.gm.is_finite() {
            pair("GM", &(kep.gm / M3_PER_KM3).to_string());
        }
    }

    for (key, quantity) in [
        ("MASS", opm.spacecraft.mass),
        ("SOLAR_RAD_AREA", opm.spacecraft.solar_rad_area),
        ("SOLAR_RAD_COEFF", opm.spacecraft.solar_rad_coeff),
        ("DRAG_AREA", opm.spacecraft.drag_area),
        ("DRAG_COEFF", opm.spacecraft.drag_coeff),
    ] {
        if let Some(quantity) = quantity {
            pair(key, &quantity.to_string());
        }
    }

    for maneuver in &opm.maneuvers {
        let mut ignition = String::new();
        write_epoch(&mut ignition, maneuver.epoch_ignition);
        pair("MAN_EPOCH_IGNITION", &ignition);
        pair("MAN_DURATION", &maneuver.duration.to_string());
        pair("MAN_DELTA_MASS", &maneuver.delta_mass.to_string());
        pair("MAN_REF_FRAME", &maneuver.ref_frame);
        for (key, metres) in ["MAN_DV_1", "MAN_DV_2", "MAN_DV_3"]
            .iter()
            .zip(maneuver.delta_v)
        {
            pair(key, &(metres / M_PER_KM).to_string());
        }
    }

    if let Some(covariance) = &opm.covariance {
        if let Some(frame) = &covariance.ref_frame {
            pair("COV_REF_FRAME", frame);
        }
        for (key, entry) in OPM_COVARIANCE_KEYWORDS.iter().zip(covariance.lower) {
            pair(key, &(entry / M2_PER_KM2).to_string());
        }
    }

    for (suffix, text) in &opm.user_defined {
        pair(&format!("USER_DEFINED_{suffix}"), text);
    }

    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;

    /// CCSDS 502.0-B-3 annex G, figure G-1: the simple OPM, state only.
    const SIMPLE: &str = include_str!("../tests/fixtures/ccsds_502_g1.opm");

    #[test]
    fn the_simple_annex_example_reads_as_the_state_it_states() {
        let opm = parse_opm(SIMPLE).unwrap();
        assert_eq!(opm.originator, "JAXA");
        assert_eq!(opm.object_name, "OSPREY 5");
        assert_eq!(opm.ref_frame, "ITRF2000");
        assert_eq!(opm.epoch.second(), 15.1172);
        // Position scales exactly here; velocity does not. `-0.873160 km * 1000` is
        // -873.1600000000001 m, because multiplying a decimal by a thousand is a rounding in
        // binary. What survives exactly is the kilometre the file stated — see the round-trip test.
        assert_eq!(opm.position, [6_503_514.0, 1_239_647.0, -717_490.0]);
        for (component, expected) in opm.velocity.iter().zip([-873.16, 8740.42, -4191.076]) {
            assert!(
                (component - expected).abs() < 1e-9,
                "{component} vs {expected}"
            );
        }
        assert_eq!(opm.spacecraft.mass, Some(3000.0));
        assert_eq!(opm.spacecraft.drag_coeff, Some(2.5));
        assert!(opm.keplerian.is_none());
        assert!(opm.covariance.is_none());
        assert_eq!(opm.maneuvers.len(), 0);
    }

    #[test]
    fn a_message_that_is_not_an_opm_is_named_as_such_rather_than_as_a_missing_epoch() {
        assert_eq!(
            parse_opm("CCSDS_OEM_VERS = 3.0\nORIGINATOR = NASA/JPL\n"),
            Err(NdmError::NotThisMessage {
                expected: "CCSDS_OPM_VERS"
            })
        );
        assert_eq!(
            parse_opm(""),
            Err(NdmError::NotThisMessage {
                expected: "CCSDS_OPM_VERS"
            })
        );
    }

    #[test]
    fn a_mandatory_keyword_left_out_is_named() {
        let text = SIMPLE.replace("REF_FRAME      = ITRF2000\n", "");
        assert_eq!(
            parse_opm(&text),
            Err(NdmError::MissingKeyword {
                keyword: "REF_FRAME"
            })
        );
    }

    #[test]
    fn a_value_that_does_not_parse_names_its_keyword_and_its_line() {
        let text = SIMPLE.replace("6503.514000", "6503.514.000");
        let error = parse_opm(&text).unwrap_err();
        assert_eq!(
            error,
            NdmError::MalformedValue {
                line: 13,
                keyword: "X"
            }
        );
    }

    /// The reason the anomaly is a sum type: a file cannot state both, and a consumer cannot mistake
    /// one for the other.
    #[test]
    fn the_anomaly_is_whichever_one_the_file_stated() {
        let with_true = parse_opm(include_str!("../tests/fixtures/ccsds_502_g2.opm")).unwrap();
        let anomaly = with_true.keplerian.unwrap().anomaly;
        assert!(matches!(anomaly, Anomaly::True(_)));
        let Anomaly::True(angle) = anomaly else {
            unreachable!()
        };
        assert!((angle.degrees() - 41.922_339).abs() < 1e-12);

        let with_mean = include_str!("../tests/fixtures/ccsds_502_g2.opm")
            .replace("TRUE_ANOMALY", "MEAN_ANOMALY");
        let anomaly = parse_opm(&with_mean).unwrap().keplerian.unwrap().anomaly;
        assert!(matches!(anomaly, Anomaly::Mean(_)));
    }

    #[test]
    fn two_manoeuvres_are_two_manoeuvres_and_not_one_with_the_second_overwriting_the_first() {
        let opm = parse_opm(include_str!("../tests/fixtures/ccsds_502_g2.opm")).unwrap();
        assert_eq!(opm.maneuvers.len(), 2);
        assert_eq!(opm.maneuvers[0].ref_frame, "EME2000");
        assert_eq!(opm.maneuvers[1].ref_frame, "RTN");
        assert_eq!(opm.maneuvers[0].duration, 132.60);
        assert_eq!(opm.maneuvers[1].duration, 0.0);
        assert_eq!(opm.maneuvers[0].delta_v[0], -23.257);
        assert_eq!(opm.maneuvers[1].delta_mass, -1.469);
    }

    /// A manoeuvre block that stops halfway is a manoeuvre with an unknown burn, not a manoeuvre
    /// with a zero one.
    #[test]
    fn a_manoeuvre_missing_a_component_is_refused_rather_than_zero_filled() {
        let text = include_str!("../tests/fixtures/ccsds_502_g2.opm")
            .replace("MAN_DV_3          =     -0.00893444", "");
        assert_eq!(
            parse_opm(&text),
            Err(NdmError::MissingKeyword {
                keyword: "MAN_DV_3"
            })
        );
    }

    #[test]
    fn a_covariance_with_holes_in_it_is_refused_rather_than_zero_filled() {
        let text = include_str!("../tests/fixtures/ccsds_502_g4.opm")
            .replace("CZ_DOT_Z_DOT = 6.224444338635500e-10   [km**2/s**2]\n", "");
        assert_eq!(
            parse_opm(&text),
            Err(NdmError::ShortCovariance {
                line: 53,
                entries: 20
            })
        );
    }

    /// Every field of `Spacecraft` on its own is enough to make the block non-empty.
    ///
    /// The five are one `&&` chain, and a chain is only as tested as its shortest witness: with all
    /// five absent every conjunct is true and no mutation of an operator inside it can change the
    /// answer. Exactly one field stated is the case that separates them, and each field gets its own,
    /// because a chain that dropped one term would still pass a test that only ever sets `mass`.
    #[test]
    fn a_spacecraft_block_is_empty_only_when_the_producer_stated_nothing() {
        assert!(Spacecraft::default().is_empty());
        let one_at_a_time = [
            Spacecraft {
                mass: Some(3000.0),
                ..Spacecraft::default()
            },
            Spacecraft {
                solar_rad_area: Some(18.77),
                ..Spacecraft::default()
            },
            Spacecraft {
                solar_rad_coeff: Some(1.0),
                ..Spacecraft::default()
            },
            Spacecraft {
                drag_area: Some(18.77),
                ..Spacecraft::default()
            },
            Spacecraft {
                drag_coeff: Some(2.5),
                ..Spacecraft::default()
            },
        ];
        for spacecraft in one_at_a_time {
            assert!(!spacecraft.is_empty(), "{spacecraft:?}");
        }
        assert!(!parse_opm(SIMPLE).unwrap().spacecraft.is_empty());
    }

    /// A Keplerian block with one keyword in it is a hole, not an absence.
    ///
    /// Absent entirely is legal — figure G-1 states a state vector and no elements — so the reader
    /// distinguishes the two by a six-way conjunction, and the case that tells the two apart is a
    /// message stating exactly one element. Reading it as "no elements stated" would silently drop a
    /// semi-major axis the producer wrote down.
    #[test]
    fn a_keplerian_block_stating_one_element_is_refused_rather_than_read_as_no_elements() {
        for (keyword, missing) in [
            ("SEMI_MAJOR_AXIS   =  41399.5123", "ECCENTRICITY"),
            ("ECCENTRICITY      =      0.020842611", "SEMI_MAJOR_AXIS"),
            ("INCLINATION       =      0.117746", "SEMI_MAJOR_AXIS"),
            ("RA_OF_ASC_NODE    =     17.604721", "SEMI_MAJOR_AXIS"),
            ("ARG_OF_PERICENTER =    218.242943", "SEMI_MAJOR_AXIS"),
            ("TRUE_ANOMALY      =     41.922339", "SEMI_MAJOR_AXIS"),
        ] {
            let text = format!("{SIMPLE}{keyword}\n");
            assert_eq!(
                parse_opm(&text),
                Err(NdmError::MissingKeyword { keyword: missing }),
                "{keyword}"
            );
        }
    }

    /// The two optional header keywords no annex figure states, read and written back.
    ///
    /// `REF_FRAME_EPOCH` is what makes `TOD` or `TEME` a frame rather than a family of them, and
    /// `MESSAGE_ID` is what an operator correlates a message by. Neither is exercised by any of the
    /// three figures, so the witness is figure G-1 with both inserted.
    #[test]
    fn the_message_identifier_and_the_frame_epoch_survive_a_round_trip() {
        let text = SIMPLE
            .replacen(
                "ORIGINATOR     = JAXA\n",
                "ORIGINATOR     = JAXA\nMESSAGE_ID     = OPM 201113719185\n",
                1,
            )
            .replacen(
                "REF_FRAME      = ITRF2000\n",
                "REF_FRAME      = ITRF2000\nREF_FRAME_EPOCH = 2022-12-18T00:00:00\n",
                1,
            );
        let opm = parse_opm(&text).unwrap();
        assert_eq!(opm.message_id.as_deref(), Some("OPM 201113719185"));
        assert_eq!(
            opm.ref_frame_epoch,
            Some(UtcDateTime::new(2022, 12, 18, 0, 0, 0.0).unwrap())
        );

        let written = write_opm(&opm);
        assert!(written.contains("MESSAGE_ID = OPM 201113719185\n"));
        assert_eq!(parse_opm(&written).unwrap(), opm);

        // And figure G-1 as published states neither, so neither is a constant.
        let plain = parse_opm(SIMPLE).unwrap();
        assert!(plain.message_id.is_none());
        assert!(plain.ref_frame_epoch.is_none());
    }

    #[test]
    fn the_covariance_frame_and_a_user_defined_keyword_survive_the_read() {
        let opm = parse_opm(include_str!("../tests/fixtures/ccsds_502_g4.opm")).unwrap();
        let covariance = opm.covariance.unwrap();
        assert_eq!(covariance.ref_frame.as_deref(), Some("RTN"));
        // km² to m² is 1e6, and the entry is the (0,0) of the lower triangle.
        assert!((covariance.lower[0] - 3.331_349_476_038_534e2).abs() < 1e-10);
        assert_eq!(
            opm.user_defined,
            vec![("EARTH_MODEL".to_owned(), "WGS-84".to_owned())]
        );
    }
}
