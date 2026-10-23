//! Conjunction Data Message — how a screening arrives (CCSDS 508.0-B-1).
//!
//! A CDM is two objects, their states at the time of closest approach, and the covariance of each in
//! its own RTN frame. That is exactly the input `spacelab_estimation::conjunction` takes, and until
//! this reader existed the numbers reached it by hand-transcription.
//!
//! # The hard-body radius, and why comments are not discarded
//!
//! Collision probability needs a combined hard-body radius, and **CDM version 1.0 has no keyword for
//! it** — table 3-3 of 508.0-B-1 does not define one. NASA CARA therefore writes it as
//! `COMMENT HBR = 15.0`, in metres, and every one of the reference conjunctions this workspace
//! verifies against carries it that way. So the reader keeps comments, and
//! [`Cdm::commented_hard_body_radius`] reads it back out with the convention named rather than
//! assumed.

use spacelab_core::UtcDateTime;

use crate::kvn::{
    CDM_COVARIANCE_KEYWORDS, COVARIANCE_ENTRIES, Item, M_PER_KM, NdmError, epoch, scan, value,
};

/// The version keyword a CDM must open with.
const VERSION_KEYWORD: &str = "CCSDS_CDM_VERS";

/// The non-standard comment key NASA CARA carries the hard-body radius under.
const HBR_COMMENT_KEY: &str = "HBR";

/// One of the two objects in a conjunction, in SI throughout.
#[derive(Debug, Clone, PartialEq)]
pub struct CdmObject {
    /// `OBJECT_DESIGNATOR`, the catalogue number.
    pub object_designator: String,
    /// `CATALOG_NAME`, the catalogue the designator belongs to.
    pub catalog_name: String,
    /// `OBJECT_NAME`.
    pub object_name: String,
    /// `INTERNATIONAL_DESIGNATOR`.
    pub international_designator: String,
    /// `EPHEMERIS_NAME`, or `NONE`.
    pub ephemeris_name: String,
    /// `COVARIANCE_METHOD`, e.g. `CALCULATED`.
    pub covariance_method: String,
    /// `MANEUVERABLE`: `YES`, `NO` or `N/A`.
    pub maneuverable: String,
    /// `REF_FRAME` the state vector is given in.
    pub ref_frame: String,
    /// Position at TCA \[m].
    pub position: [f64; 3],
    /// Velocity at TCA \[m/s].
    pub velocity: [f64; 3],
    /// The 6×6 RTN covariance, lower triangle row by row, in SI: m², m²/s, m²/s².
    ///
    /// A CDM states this in metres already, unlike an OPM or OEM, so it takes no unit scaling —
    /// 508.0-B-1 table 3-5 against 502.0-B-3 table 3-4.
    pub covariance: [f64; COVARIANCE_ENTRIES],
    /// `AREA_PC`, the area used for the probability computation \[m²].
    pub area_pc: Option<f64>,
    /// `MASS` \[kg].
    pub mass: Option<f64>,
    /// `CD_AREA_OVER_MASS` \[m²/kg].
    pub cd_area_over_mass: Option<f64>,
    /// `CR_AREA_OVER_MASS` \[m²/kg].
    pub cr_area_over_mass: Option<f64>,
}

/// A parsed Conjunction Data Message, in SI throughout.
#[derive(Debug, Clone, PartialEq)]
pub struct Cdm {
    /// `CREATION_DATE`.
    pub creation_date: UtcDateTime,
    /// `ORIGINATOR`.
    pub originator: String,
    /// `MESSAGE_FOR`, the party the message is addressed to.
    pub message_for: Option<String>,
    /// `MESSAGE_ID`.
    pub message_id: Option<String>,
    /// `TCA`, the time of closest approach.
    pub tca: UtcDateTime,
    /// `MISS_DISTANCE` \[m].
    pub miss_distance: f64,
    /// `RELATIVE_SPEED` \[m/s].
    pub relative_speed: Option<f64>,
    /// `RELATIVE_POSITION_{R,T,N}` \[m], object 2 relative to object 1 in object 1's RTN frame.
    pub relative_position: Option<[f64; 3]>,
    /// `RELATIVE_VELOCITY_{R,T,N}` \[m/s].
    pub relative_velocity: Option<[f64; 3]>,
    /// `COLLISION_PROBABILITY`, the originator's own number.
    pub collision_probability: Option<f64>,
    /// `COLLISION_PROBABILITY_METHOD`, e.g. `FOSTER-1992`.
    pub collision_probability_method: Option<String>,
    /// Every `COMMENT` in the message, in file order. See [`Cdm::commented_hard_body_radius`].
    pub comments: Vec<String>,
    /// The first object.
    pub object1: CdmObject,
    /// The second object.
    pub object2: CdmObject,
}

impl Cdm {
    /// The combined hard-body radius \[m], if the producer wrote one into a comment.
    ///
    /// CDM 1.0 has no keyword for it, so this reads NASA CARA's `COMMENT HBR = <metres>`. It is a
    /// *convention*, not a standard, which is why it is a named method returning `Option` rather
    /// than a field that looks as authoritative as `MISS_DISTANCE`: a message from another producer
    /// will not carry it, and a caller then has to supply the radius itself.
    #[must_use]
    pub fn commented_hard_body_radius(&self) -> Option<f64> {
        self.comments.iter().find_map(|comment| {
            let (key, value) = comment.split_once('=')?;
            (key.trim() == HBR_COMMENT_KEY).then(|| value.trim().parse().ok())?
        })
    }
}

/// Read a CCSDS Conjunction Data Message in KVN.
///
/// # Errors
///
/// [`NdmError::NotThisMessage`] if the file does not open with `CCSDS_CDM_VERS`,
/// [`NdmError::ObjectCount`] if it does not describe exactly two objects, and otherwise the first
/// structural failure; see [`NdmError`].
pub fn parse_cdm(text: &str) -> Result<Cdm, NdmError> {
    let mut opened = false;
    let mut header = Header::default();
    let mut sections: Vec<ObjectBuilder> = Vec::new();
    let mut comments = Vec::new();

    for item in scan(text) {
        match item {
            Item::Comment { text, .. } => comments.push(text.to_owned()),
            Item::Bare { .. } => {}
            Item::Pair { line, key, value } => {
                if key == VERSION_KEYWORD {
                    opened = true;
                    continue;
                }
                if !opened {
                    return Err(NdmError::NotThisMessage {
                        expected: VERSION_KEYWORD,
                    });
                }
                // `OBJECT = OBJECTn` is the only section delimiter a CDM has: everything after it
                // belongs to that object until the next one, and the keywords repeat verbatim.
                if key == "OBJECT" {
                    sections.push(ObjectBuilder::default());
                    continue;
                }
                match sections.last_mut() {
                    Some(object) => object.set(line, key, value)?,
                    None => header.set(line, key, value)?,
                }
            }
        }
    }

    if !opened {
        return Err(NdmError::NotThisMessage {
            expected: VERSION_KEYWORD,
        });
    }
    let [object1, object2] = <[ObjectBuilder; 2]>::try_from(sections)
        .map_err(|wrong| NdmError::ObjectCount { found: wrong.len() })?;

    let missing = |keyword| NdmError::MissingKeyword { keyword };
    Ok(Cdm {
        creation_date: header
            .creation_date
            .ok_or_else(|| missing("CREATION_DATE"))?,
        originator: header.originator.ok_or_else(|| missing("ORIGINATOR"))?,
        message_for: header.message_for,
        message_id: header.message_id,
        tca: header.tca.ok_or_else(|| missing("TCA"))?,
        miss_distance: header
            .miss_distance
            .ok_or_else(|| missing("MISS_DISTANCE"))?,
        relative_speed: header.relative_speed,
        relative_position: triple(&header.relative_position),
        relative_velocity: triple(&header.relative_velocity),
        collision_probability: header.collision_probability,
        collision_probability_method: header.collision_probability_method,
        comments,
        object1: object1.finish()?,
        object2: object2.finish()?,
    })
}

/// An optional RTN triple is present only when all three components are.
///
/// A partly stated vector is a direction with a hole in it, and zero-filling the hole names a
/// different direction rather than an unknown one.
fn triple(components: &[Option<f64>; 3]) -> Option<[f64; 3]> {
    Some([components[0]?, components[1]?, components[2]?])
}

#[derive(Default)]
struct Header {
    creation_date: Option<UtcDateTime>,
    originator: Option<String>,
    message_for: Option<String>,
    message_id: Option<String>,
    tca: Option<UtcDateTime>,
    miss_distance: Option<f64>,
    relative_speed: Option<f64>,
    relative_position: [Option<f64>; 3],
    relative_velocity: [Option<f64>; 3],
    collision_probability: Option<f64>,
    collision_probability_method: Option<String>,
}

impl Header {
    fn set(&mut self, line: usize, key: &str, text: &str) -> Result<(), NdmError> {
        match key {
            "CREATION_DATE" => self.creation_date = Some(epoch(line, "CREATION_DATE", text)?),
            "ORIGINATOR" => self.originator = Some(text.to_owned()),
            "MESSAGE_FOR" => self.message_for = Some(text.to_owned()),
            "MESSAGE_ID" => self.message_id = Some(text.to_owned()),
            "TCA" => self.tca = Some(epoch(line, "TCA", text)?),
            "MISS_DISTANCE" => self.miss_distance = Some(value(line, "MISS_DISTANCE", text)?),
            "RELATIVE_SPEED" => self.relative_speed = Some(value(line, "RELATIVE_SPEED", text)?),
            "RELATIVE_POSITION_R" => {
                self.relative_position[0] = Some(value(line, "RELATIVE_POSITION_R", text)?);
            }
            "RELATIVE_POSITION_T" => {
                self.relative_position[1] = Some(value(line, "RELATIVE_POSITION_T", text)?);
            }
            "RELATIVE_POSITION_N" => {
                self.relative_position[2] = Some(value(line, "RELATIVE_POSITION_N", text)?);
            }
            "RELATIVE_VELOCITY_R" => {
                self.relative_velocity[0] = Some(value(line, "RELATIVE_VELOCITY_R", text)?);
            }
            "RELATIVE_VELOCITY_T" => {
                self.relative_velocity[1] = Some(value(line, "RELATIVE_VELOCITY_T", text)?);
            }
            "RELATIVE_VELOCITY_N" => {
                self.relative_velocity[2] = Some(value(line, "RELATIVE_VELOCITY_N", text)?);
            }
            "COLLISION_PROBABILITY" => {
                self.collision_probability = Some(value(line, "COLLISION_PROBABILITY", text)?);
            }
            "COLLISION_PROBABILITY_METHOD" => {
                self.collision_probability_method = Some(text.to_owned());
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Default)]
struct ObjectBuilder {
    object_designator: Option<String>,
    catalog_name: Option<String>,
    object_name: Option<String>,
    international_designator: Option<String>,
    ephemeris_name: Option<String>,
    covariance_method: Option<String>,
    maneuverable: Option<String>,
    ref_frame: Option<String>,
    position: [Option<f64>; 3],
    velocity: [Option<f64>; 3],
    covariance: [Option<f64>; COVARIANCE_ENTRIES],
    cov_line: usize,
    area_pc: Option<f64>,
    mass: Option<f64>,
    cd_area_over_mass: Option<f64>,
    cr_area_over_mass: Option<f64>,
}

impl ObjectBuilder {
    fn set(&mut self, line: usize, key: &str, text: &str) -> Result<(), NdmError> {
        if let Some(index) = CDM_COVARIANCE_KEYWORDS.iter().position(|name| *name == key) {
            let keyword = CDM_COVARIANCE_KEYWORDS[index];
            self.covariance[index] = Some(value(line, keyword, text)?);
            self.cov_line = line;
            return Ok(());
        }
        match key {
            "OBJECT_DESIGNATOR" => self.object_designator = Some(text.to_owned()),
            "CATALOG_NAME" => self.catalog_name = Some(text.to_owned()),
            "OBJECT_NAME" => self.object_name = Some(text.to_owned()),
            "INTERNATIONAL_DESIGNATOR" => self.international_designator = Some(text.to_owned()),
            "EPHEMERIS_NAME" => self.ephemeris_name = Some(text.to_owned()),
            "COVARIANCE_METHOD" => self.covariance_method = Some(text.to_owned()),
            "MANEUVERABLE" => self.maneuverable = Some(text.to_owned()),
            "REF_FRAME" => self.ref_frame = Some(text.to_owned()),
            "X" => self.position[0] = Some(value::<f64>(line, "X", text)? * M_PER_KM),
            "Y" => self.position[1] = Some(value::<f64>(line, "Y", text)? * M_PER_KM),
            "Z" => self.position[2] = Some(value::<f64>(line, "Z", text)? * M_PER_KM),
            "X_DOT" => self.velocity[0] = Some(value::<f64>(line, "X_DOT", text)? * M_PER_KM),
            "Y_DOT" => self.velocity[1] = Some(value::<f64>(line, "Y_DOT", text)? * M_PER_KM),
            "Z_DOT" => self.velocity[2] = Some(value::<f64>(line, "Z_DOT", text)? * M_PER_KM),
            "AREA_PC" => self.area_pc = Some(value(line, "AREA_PC", text)?),
            "MASS" => self.mass = Some(value(line, "MASS", text)?),
            "CD_AREA_OVER_MASS" => {
                self.cd_area_over_mass = Some(value(line, "CD_AREA_OVER_MASS", text)?);
            }
            "CR_AREA_OVER_MASS" => {
                self.cr_area_over_mass = Some(value(line, "CR_AREA_OVER_MASS", text)?);
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(self) -> Result<CdmObject, NdmError> {
        let missing = |keyword| NdmError::MissingKeyword { keyword };
        let component = |slot: Option<f64>, keyword| slot.ok_or_else(|| missing(keyword));

        let stated = self.covariance.iter().filter(|slot| slot.is_some()).count();
        if stated != COVARIANCE_ENTRIES {
            return Err(NdmError::ShortCovariance {
                line: self.cov_line,
                entries: stated,
            });
        }
        let mut covariance = [0.0; COVARIANCE_ENTRIES];
        for (slot, entry) in self.covariance.iter().zip(covariance.iter_mut()) {
            *entry = slot.unwrap_or_default();
        }

        Ok(CdmObject {
            object_designator: self
                .object_designator
                .ok_or_else(|| missing("OBJECT_DESIGNATOR"))?,
            catalog_name: self.catalog_name.ok_or_else(|| missing("CATALOG_NAME"))?,
            object_name: self.object_name.ok_or_else(|| missing("OBJECT_NAME"))?,
            international_designator: self
                .international_designator
                .ok_or_else(|| missing("INTERNATIONAL_DESIGNATOR"))?,
            ephemeris_name: self
                .ephemeris_name
                .ok_or_else(|| missing("EPHEMERIS_NAME"))?,
            covariance_method: self
                .covariance_method
                .ok_or_else(|| missing("COVARIANCE_METHOD"))?,
            maneuverable: self.maneuverable.ok_or_else(|| missing("MANEUVERABLE"))?,
            ref_frame: self.ref_frame.ok_or_else(|| missing("REF_FRAME"))?,
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
            covariance,
            area_pc: self.area_pc,
            mass: self.mass,
            cd_area_over_mass: self.cd_area_over_mass,
            cr_area_over_mass: self.cr_area_over_mass,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;

    /// Drop every line whose keyword is `keyword`, without depending on the file's column layout.
    fn without_lines_starting(text: &str, keyword: &str) -> String {
        text.lines()
            .filter(|line| {
                line.split('=')
                    .next()
                    .is_none_or(|key| key.trim() != keyword)
            })
            .fold(String::new(), |mut acc, line| {
                acc.push_str(line);
                acc.push('\n');
                acc
            })
    }

    /// CCSDS 508.0-B-1 §3.6.2: the CDM with only obligatory keywords.
    const OBLIGATORY: &str = include_str!("../tests/fixtures/ccsds_508_362.cdm");
    /// CCSDS 508.0-B-1 §3.6.3: the same conjunction with the optional keywords too.
    const OPTIONAL: &str = include_str!("../tests/fixtures/ccsds_508_363.cdm");

    #[test]
    fn the_two_objects_of_the_annex_example_are_two_objects_and_not_one_overwritten() {
        let cdm = parse_cdm(OBLIGATORY).unwrap();
        assert_eq!(cdm.originator, "JSPOC");
        assert_eq!(cdm.object1.object_designator, "12345");
        assert_eq!(cdm.object2.object_designator, "30337");
        assert_eq!(cdm.object1.object_name, "SATELLITE A");
        assert_eq!(cdm.object2.object_name, "FENGYUN 1C DEB");
        assert_eq!(cdm.object1.maneuverable, "YES");
        assert_eq!(cdm.object2.maneuverable, "NO");
        // The two states differ in the fourth decimal of a kilometre — 715 m apart, as stated.
        assert_ne!(cdm.object1.position, cdm.object2.position);
        assert_eq!(cdm.miss_distance, 715.0);
    }

    #[test]
    fn a_state_is_read_in_metres_and_a_covariance_is_left_in_them() {
        let cdm = parse_cdm(OBLIGATORY).unwrap();
        // All six components of both objects, because the scaling is written out per keyword and a
        // component nothing asserts is a conversion nothing checks. The literals are the exact
        // doubles the file's kilometres scale to — `4.418769571 * 1000` is not `4418.769571`.
        assert_eq!(
            cdm.object1.position,
            [2_570_097.065, 2_244_654.904, 6_281_497.978]
        );
        assert_eq!(
            cdm.object1.velocity,
            [4_418.769_571_000_001, 4_833.547_742_999_999, -3_526.774_282]
        );
        assert_eq!(
            cdm.object2.position,
            [2_569_540.800_000_000_3, 2_245_093.614, 6_281_599.946]
        );
        assert_eq!(
            cdm.object2.velocity,
            [
                -2_888.612_499_999_999_7,
                -6_007.247_515_999_999,
                3_328.770_172
            ]
        );
        // CR_R, the first lower-triangle entry, in m² and therefore unscaled.
        assert_eq!(cdm.object1.covariance[0], 4.142e1);
        assert_eq!(cdm.object2.covariance[0], 1.337e3);
        // The last entry, CNDOT_NDOT.
        assert_eq!(cdm.object1.covariance[COVARIANCE_ENTRIES - 1], 5.529e-5);
    }

    #[test]
    fn the_optional_relative_state_and_the_originators_own_probability_survive_the_read() {
        let cdm = parse_cdm(OPTIONAL).unwrap();
        assert_eq!(cdm.relative_position, Some([27.4, -70.2, 711.8]));
        assert_eq!(cdm.relative_velocity, Some([-7.2, -14692.0, -1437.2]));
        assert_eq!(cdm.relative_speed, Some(14762.0));
        assert_eq!(cdm.collision_probability, Some(4.835e-5));
        assert_eq!(
            cdm.collision_probability_method.as_deref(),
            Some("FOSTER-1992")
        );
        assert_eq!(cdm.object1.mass, Some(251.6));
        assert_eq!(cdm.object1.area_pc, Some(5.2));
    }

    /// The addressee, the message identifier, and the two ballistic ratios.
    ///
    /// These four are read and stored and nothing downstream in this workspace consumes them yet,
    /// which is exactly why they need an assertion of their own: a keyword no test names can be
    /// dropped from the reader without a single failure.
    #[test]
    fn the_routing_keywords_and_the_ballistic_ratios_survive_the_read() {
        let cdm = parse_cdm(OPTIONAL).unwrap();
        assert_eq!(cdm.message_for.as_deref(), Some("SATELLITE A"));
        assert_eq!(cdm.message_id.as_deref(), Some("201113719185"));
        assert_eq!(cdm.object1.cd_area_over_mass, Some(0.045_663));
        assert_eq!(cdm.object1.cr_area_over_mass, Some(0.0));
        assert_eq!(cdm.object2.cd_area_over_mass, Some(0.118_668));
        assert_eq!(cdm.object2.cr_area_over_mass, Some(0.075_204));
    }

    /// The obligatory-keywords example states no relative position, and the reader must say so
    /// rather than report the origin.
    #[test]
    fn an_absent_relative_state_is_absent_rather_than_zero() {
        let cdm = parse_cdm(OBLIGATORY).unwrap();
        assert_eq!(cdm.relative_position, None);
        assert_eq!(cdm.relative_velocity, None);
        assert_eq!(cdm.relative_speed, None);
    }

    #[test]
    fn a_partly_stated_relative_vector_is_absent_rather_than_zero_filled() {
        let text = without_lines_starting(OPTIONAL, "RELATIVE_POSITION_N");
        let cdm = parse_cdm(&text).unwrap();
        assert_eq!(cdm.relative_position, None);
        // The velocity, which is complete, still reads.
        assert_eq!(cdm.relative_velocity, Some([-7.2, -14692.0, -1437.2]));
    }

    #[test]
    fn a_message_describing_one_object_is_refused() {
        let one = OBLIGATORY.split("= OBJECT2").next().unwrap().to_owned();
        assert_eq!(parse_cdm(&one), Err(NdmError::ObjectCount { found: 1 }));
        assert_eq!(
            parse_cdm(&OBLIGATORY.replace("= OBJECT2", "= OBJECT3")).map(|_| ()),
            Ok(())
        );
    }

    #[test]
    fn a_message_that_is_not_a_cdm_is_named_as_such() {
        assert_eq!(
            parse_cdm("CCSDS_OPM_VERS = 3.0\nORIGINATOR = JAXA\n"),
            Err(NdmError::NotThisMessage {
                expected: "CCSDS_CDM_VERS"
            })
        );
    }

    /// The convention CDM 1.0 has no keyword for.
    #[test]
    fn a_commented_hard_body_radius_is_read_and_a_message_without_one_says_so() {
        assert_eq!(
            parse_cdm(OBLIGATORY).unwrap().commented_hard_body_radius(),
            None
        );
        let text = OBLIGATORY.replace("CREATION_DATE", "COMMENT HBR = 15.0\nCREATION_DATE");
        assert_eq!(
            parse_cdm(&text).unwrap().commented_hard_body_radius(),
            Some(15.0)
        );
    }

    /// A comment that merely mentions the letters is not a hard-body radius, and one that is not a
    /// number is not one either.
    #[test]
    fn only_a_comment_whose_key_is_exactly_hbr_supplies_a_radius() {
        for comment in [
            "COMMENT HBR RADIUS = 15.0",
            "COMMENT THE HBR = 15.0",
            "COMMENT HBR = not a number",
        ] {
            let text = OBLIGATORY.replace("CREATION_DATE", &format!("{comment}\nCREATION_DATE"));
            assert_eq!(
                parse_cdm(&text).unwrap().commented_hard_body_radius(),
                None,
                "{comment}"
            );
        }
    }

    #[test]
    fn an_object_covariance_with_holes_in_it_is_refused() {
        let text = without_lines_starting(OBLIGATORY, "CNDOT_NDOT");
        assert!(matches!(
            parse_cdm(&text),
            Err(NdmError::ShortCovariance { entries: 20, .. })
        ));
    }
}
