//! Instrument scan geometry: where each pixel looks, and when it was sampled.
//!
//! A cross-track radiometer does not photograph a scene. It sweeps one detector (or a short column
//! of them) across the ground while the spacecraft moves beneath it, so *every pixel has its own
//! pointing direction and its own observation time*, and the two are not separable — by the time the
//! mirror reaches the end of a scan the platform has travelled several kilometres. Geolocating such
//! an instrument means evaluating the orbit once per pixel, at that pixel's own instant, along that
//! pixel's own line of sight. [`geoloc`](crate::geoloc) does the second half; this module is the
//! first, and it is a *catalogue*: the numbers are per-instrument facts, not a model.
//!
//! # Provenance
//!
//! Ported from `pyorbital.geoloc_instrument_definitions` (pytroll/pyorbital, **Apache-2.0** —
//! verified against the `LICENSE.txt` in the distribution rather than a repository metadata field,
//! and no GPL file headers remain anywhere in the package after its relicensing). The primary
//! sources pyorbital cites for each instrument are named on each variant below; where it cites none,
//! that is said too, because a number with no source is a number to be careful with.
//!
//! # Angles are roll and pitch, not "across" and "along"
//!
//! pyorbital names its two scan angles for the axes they rotate *about* — `x` along-track, `y`
//! cross-track — which is the opposite of the direction the line of sight then moves. That has
//! bitten enough people that the names here are [`ScanSample::roll`] and [`ScanSample::pitch`]: the
//! angles as the attitude convention in [`geoloc`](crate::geoloc) consumes them, added to any
//! platform roll and pitch the caller supplies. A cross-track scanner sweeps in **roll**.
//!
//! # No allocation, and no arrays
//!
//! pyorbital materialises the whole scan as a `(2, scans, pixels)` array; VIIRS at one granule is
//! 4.7 M doubles before any orbit has been evaluated. Here [`Instrument::sample`] computes one pixel
//! on demand from the same closed forms, so a caller who wants the swath edges pays for two pixels.
//! That is also what lets this be `no_std`.
//!
//! The closed forms are checked in `derivations/scan_catalogue.mac` (edge angles at 120 digits:
//! `derivations/scan_catalogue.gp`) and gated in `tests/scan_gates.rs`.

use libm::atan2;

/// One pixel's pointing and timing within a scan.
///
/// The angles are instrument-frame offsets from the boresight, to be added to the platform's own
/// roll and pitch; `time_offset` is seconds from the start of the *first* scan, so it accumulates
/// across `scan`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanSample {
    /// Rotation about the along-track axis, radians. This is the sweep of a cross-track scanner.
    pub roll: f64,
    /// Rotation about the cross-track axis, radians. Zero for a single-detector scan line; non-zero
    /// where the instrument images several detector rows at once, as VIIRS does.
    pub pitch: f64,
    /// Seconds since the start of the first scan.
    pub time_offset: f64,
}

/// A cross-track or push-broom imager whose scan geometry is tabulated.
///
/// Construct nothing: each variant *is* the instrument, and its dimensions and rates come from
/// [`pixels_per_line`](Self::pixels_per_line), [`detector_lines`](Self::detector_lines) and
/// [`scan_period`](Self::scan_period).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Instrument {
    /// AVHRR/3 at full resolution — 2048 pixels across a ±55.37° swath, 6 scans per second.
    ///
    /// Source: NOAA KLM User's Guide, Appendix J.
    Avhrr,
    /// AVHRR GAC — 409 pixels, each the average of four full-resolution fields of view.
    ///
    /// The averaging is what moves the swath edge: GAC samples run `..1111.2222.----.NNNN..`, so the
    /// outermost *pixel centre* sits at `55.37 · (1023.5 − 3.5)/1023.5`, not at 55.37°. Getting that
    /// factor wrong displaces the swath edge on the ground by about 16 km (15.7 km at 833 km
    /// altitude, `derivations/scan_catalogue.gp`).
    ///
    /// Source: NOAA KLM User's Guide, Appendix J.
    AvhrrGac,
    /// VIIRS I-band — 6400 pixels across, **32 detectors at once**, ±56.28°.
    ///
    /// The only two-dimensional member of the catalogue: the detector column subtends
    /// `atan2(11.87/2, 824.0)` in pitch, so a scan is a strip rather than a line.
    ///
    /// Source: VIIRS User's Guide, NOAA Technical Report NESDIS 142, version 1.2 (2013-09-10). The
    /// earlier 55.84° figure it replaces is the one still found in older references.
    Viirs,
    /// AMSU-A — 30 samples per scan over ±48.3°, one scan every 8 s.
    ///
    /// The only instrument here with a non-zero sync delay before the sweep starts (3.55 ms).
    AmsuA,
    /// MHS — 90 samples over ±49.444°, one scan every 8/3 s.
    ///
    /// Sources: EUMETSAT Metop/MHS instrument pages; NOAA KLM User's Guide (2014 revision).
    /// pyorbital records the sync delay as *unknown* and uses zero; so does this.
    Mhs,
    /// HIRS/4 — 56 samples over ±49.5°, one scan every 6.4 s.
    ///
    /// Sources: EUMETSAT Metop/HIRS instrument pages; NOAA KLM User's Guide (2014 revision).
    Hirs4,
    /// ATMS — 96 samples over ±52.7°, one scan every 8/3 s.
    ///
    /// Sources: Garrett et al., *Assimilation of Suomi-NPP ATMS* (2013); Yang, *Suomi NPP ATMS Scan
    /// Reversal Study*, NOAA/STAR ATMS SDR Working Group.
    Atms,
    /// MWHS-2 — 98 samples over ±53.35°, one scan every 8/3 s of which 1.71 s is the sweep.
    ///
    /// Source: NSSC/CAS instrument description. As with MHS, the sync delay is unknown and taken as
    /// zero.
    Mwhs2,
    /// OLCI — 4000 push-broom samples from +46.5° to −22.1°.
    ///
    /// Deliberately asymmetric: Sentinel-3 tilts OLCI's field of view westward to keep sun glint out
    /// of the swath. See [`scan_period`](Self::scan_period) on why its timing is absent.
    ///
    /// Source: Sentinel-3 OLCI Coverage (ESA).
    Olci,
    /// ASCAT — two 21-sample fans, one either side of the ground track, each spanning 25° to 53°.
    ///
    /// A scatterometer, so there is no swath under the spacecraft at all: the nadir gap between the
    /// two fans is the instrument working as designed, not a coverage defect.
    Ascat,
    /// SLSTR nadir view — 3000 push-broom samples from +46.5° to −22.1°.
    ///
    /// Note that the angular limits are *identical* to [`Olci`](Self::Olci)'s, from the same ESA
    /// coverage page; only the sample count differs. Two instruments agreeing to the digit is worth
    /// treating as one source, not two.
    ///
    /// Source: Sentinel-3 SLSTR Coverage (ESA).
    SlstrNadir,
}

/// AVHRR's outermost field of view, degrees. Also the base for the GAC pixel-centre angle.
const AVHRR_SCAN_ANGLE_DEG: f64 = 55.37;
/// Seconds between the centres of two adjacent full-resolution AVHRR samples.
const AVHRR_SAMPLE_SECONDS: f64 = 0.000_025;
/// VIIRS detector-column half-height (mm) and focal length (mm), whose ratio is the pitch extent.
const VIIRS_DETECTOR_EXTENT_MM: f64 = 11.87;
/// VIIRS focal length, mm.
const VIIRS_FOCAL_LENGTH_MM: f64 = 824.0;
/// Seconds per VIIRS sample column, from a 85.4 s granule of 48 scans over 6400 columns.
const VIIRS_COLUMN_SECONDS: f64 = 0.000_277_994_791_7;
/// Seconds for one VIIRS scan.
const VIIRS_SCAN_SECONDS: f64 = 1.779_166_667;

impl Instrument {
    /// Samples across one scan line.
    #[must_use]
    pub const fn pixels_per_line(self) -> u32 {
        match self {
            Self::Avhrr => 2048,
            Self::AvhrrGac => 409,
            Self::Viirs => 6400,
            Self::AmsuA => 30,
            Self::Mhs => 90,
            Self::Hirs4 => 56,
            Self::Atms => 96,
            Self::Mwhs2 => 98,
            Self::Olci => 4000,
            Self::Ascat => 42,
            Self::SlstrNadir => 3000,
        }
    }

    /// Detector rows imaged simultaneously. One for every instrument here except VIIRS.
    #[must_use]
    pub const fn detector_lines(self) -> u32 {
        match self {
            Self::Viirs => 32,
            _ => 1,
        }
    }

    /// Seconds from one scan to the next, or `None` where the source carries no timing.
    ///
    /// OLCI and SLSTR return `None`, and their [`ScanSample::time_offset`] is always zero. That is
    /// pyorbital's state rather than a simplification here — its OLCI definition has the scan rate
    /// commented out and builds an all-zero time array. A caller that needs per-pixel time for a
    /// Sentinel-3 push-broom has to get it elsewhere, and this says so instead of handing back a
    /// zero that looks like an answer.
    #[must_use]
    pub fn scan_period(self) -> Option<f64> {
        match self {
            Self::Avhrr => Some(1.0 / 6.0),
            Self::AvhrrGac => Some(0.5),
            Self::Viirs => Some(VIIRS_SCAN_SECONDS),
            Self::AmsuA => Some(8.0),
            Self::Mhs | Self::Atms | Self::Mwhs2 => Some(8.0 / 3.0),
            Self::Hirs4 => Some(6.4),
            Self::Ascat => Some(ASCAT_SCAN_SECONDS),
            Self::Olci | Self::SlstrNadir => None,
        }
    }

    /// The pointing and timing of one pixel, or `None` if `line` or `pixel` is off the detector.
    ///
    /// `scan` is unbounded: a scan index is just an elapsed-time multiplier, and how many scans a
    /// pass contains is the caller's business.
    #[must_use]
    pub fn sample(self, scan: u32, line: u32, pixel: u32) -> Option<ScanSample> {
        if line >= self.detector_lines() || pixel >= self.pixels_per_line() {
            return None;
        }
        let p = f64::from(pixel);
        let scan_offset = self.scan_period().unwrap_or(0.0) * f64::from(scan);
        Some(ScanSample {
            roll: self.roll(p),
            pitch: self.pitch(line),
            time_offset: self.sweep_seconds(p) + scan_offset,
        })
    }

    /// The scan angle of sample `p`, radians.
    ///
    /// Every cross-track member normalises the sample index the same way — `p/((n−1)/2) − 1` runs
    /// from exactly −1 to exactly +1 — and multiplies by the *negated* half-swath, so sample 0 looks
    /// to positive roll. The push-broom members interpolate between two unequal limits instead,
    /// which is why they cannot share the expression.
    fn roll(self, p: f64) -> f64 {
        let symmetric = |half_swath_deg: f64| {
            let half = (f64::from(self.pixels_per_line()) - 1.0) / 2.0;
            (p / half - 1.0) * (-half_swath_deg).to_radians()
        };
        let ramp = |from_deg: f64, to_deg: f64, count: f64, index: f64| {
            let (from, to) = (from_deg.to_radians(), to_deg.to_radians());
            from + index * (to - from) / (count - 1.0)
        };
        match self {
            Self::Avhrr => symmetric(AVHRR_SCAN_ANGLE_DEG),
            // The GAC pixel centre, not the field-of-view edge: see the variant's own docs.
            Self::AvhrrGac => symmetric(AVHRR_SCAN_ANGLE_DEG * (1023.5 - 3.5) / 1023.5),
            Self::Viirs => symmetric(56.28),
            Self::AmsuA => symmetric(48.3),
            Self::Mhs => symmetric(49.444),
            Self::Hirs4 => symmetric(49.5),
            Self::Atms => symmetric(52.7),
            Self::Mwhs2 => symmetric(53.35),
            Self::Olci | Self::SlstrNadir => {
                ramp(46.5, -22.1, f64::from(self.pixels_per_line()), p)
            }
            // Two fans of 21, both running outer-to-inner in magnitude but on opposite sides.
            Self::Ascat => {
                if p < 21.0 {
                    ramp(53.0, 25.0, 21.0, p)
                } else {
                    ramp(-25.0, -53.0, 21.0, p - 21.0)
                }
            }
        }
    }

    /// The along-track offset of detector row `line`, radians.
    fn pitch(self, line: u32) -> f64 {
        match self {
            Self::Viirs => {
                let rows = f64::from(self.detector_lines());
                let extent = atan2(VIIRS_DETECTOR_EXTENT_MM / 2.0, VIIRS_FOCAL_LENGTH_MM);
                -(f64::from(line) / ((rows - 1.0) / 2.0) - 1.0) * extent
            }
            _ => 0.0,
        }
    }

    /// Seconds from the start of a scan to sample `p` of it.
    fn sweep_seconds(self, p: f64) -> f64 {
        match self {
            Self::Avhrr => p * AVHRR_SAMPLE_SECONDS,
            // GAC averages four fields of view, so its samples are five times as far apart.
            Self::AvhrrGac => p * 0.000_125,
            Self::Viirs => p * VIIRS_COLUMN_SECONDS,
            // The one non-zero sync delay in the catalogue.
            Self::AmsuA => p * 0.2 + 0.003_55,
            Self::Mhs => p * (8.0 / 3.0 - 1.0) / 90.0,
            Self::Hirs4 => p * 6.4 / 56.0,
            Self::Atms => p * 18e-3,
            Self::Mwhs2 => p * (8.0 / 3.0 - 1.0) / 98.0,
            Self::Ascat => p * ASCAT_SCAN_SECONDS / 42.0,
            Self::Olci | Self::SlstrNadir => 0.0,
        }
    }
}

/// One ASCAT scan, seconds. The sampling interval is this divided by the 42 samples.
const ASCAT_SCAN_SECONDS: f64 = 3.747_474_747_47;
