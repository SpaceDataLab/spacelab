//! **spacelab-astrometry** — from pixels to the angles an orbit determination consumes.
//!
//! An angles-only initial orbit determination takes lines of sight. A telescope produces images.
//! This crate is the reduction between them, in the order it happens:
//!
//! - [`sources`] — find what is in the frame and measure where, sub-pixel, with an uncertainty.
//! - [`quads`] — identify the field against a star catalogue with no prior pointing (blind solve).
//! - [`wcs`] — the world-coordinate system: a gnomonic (TAN) projection with a SIP distortion
//!   polynomial, fitted by least squares, with per-star residuals and a parameter covariance.
//! - [`reduce`] — turn a measured pixel position into a *true* topocentric direction: streak
//!   endpoints with their epochs, refraction, and the two aberrations.
//!
//! # The catalogue gate
//!
//! A plate solve is only as good as the star positions it is fitted against, and those are
//! **someone else's data**. This crate therefore ships none:
//!
//! - **Nothing is vendored.** Tycho-2 is 2.5 million stars and Gaia DR3 is 1.8 *billion*; neither
//!   belongs in a source repository, and both carry attribution terms that a silently-copied excerpt
//!   would strip. Gaia's catalogue data is CC BY-NC 3.0 IGO with the credit line "ESA, Gaia DPAC",
//!   and any use that earns money, even indirectly, needs ESA's licence first; Tycho-2 comes from
//!   the ESA Hipparcos mission with its own citation requirement. Implementing the mathematics of a
//!   plate solve needs neither.
//! - **The catalogue is a trait**, [`CatalogueStar`] plus [`StarCatalogue`], so the crate depends on
//!   the *shape* of a catalogue query — stars near a direction, brighter than a limit — and not on
//!   any particular one. An in-memory implementation is enough for a test; a real one is a fetch and
//!   a cache, and that belongs in `tools/`, beside the fetchers for the wind and tide models, where
//!   it is a build step rather than a dependency.
//! - **Which catalogue** is then the caller's decision, and the trade is worth stating once. Gaia
//!   DR3 gives sub-milliarcsecond positions with proper motions and parallaxes at epoch 2016.0 —
//!   more accuracy than any ground-based plate needs, at the cost of a query service and a real
//!   proper-motion propagation. Tycho-2 gives 60 mas at V < 11, is small enough to hold locally, and
//!   is already more accurate than the centroiding error of a typical survey frame. For satellite
//!   tracking, where the residual is dominated by the streak endpoint and the timing, Tycho-2 is not
//!   the limiting term; for a metre-class astrometric campaign it may be.
//!
//! Whatever the source, the positions handed to this crate must already be **at the observation
//! epoch** — proper motion applied — and that requirement is in [`CatalogueStar`]'s own
//! documentation rather than left to be discovered from a systematic residual.
//!
//! `no_std` + `alloc` + `libm`; the only other dependency is `spacelab-core` for its vector and
//! epoch types.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

extern crate alloc;

pub mod quads;
pub mod reduce;
pub mod sources;
pub mod wcs;

use spacelab_core::Vec3;

/// A catalogue star, as this crate needs it.
///
/// `right_ascension` and `declination` are radians in the catalogue's own equatorial frame — ICRS
/// for anything modern — and they must already be **propagated to the observation epoch** by the
/// catalogue's proper motions. Gaia DR3's epoch is 2016.0 and Tycho-2's is 2000.0; a star with
/// 100 mas/yr of proper motion has moved 2.4 arcseconds since J2000, which is a third of a pixel on
/// a wide-field survey camera and ten pixels on a narrow one. Nothing downstream can detect that
/// error — it appears as a plate solution that fits perfectly and points slightly wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatalogueStar {
    /// Right ascension at the observation epoch, radians.
    pub right_ascension: f64,
    /// Declination at the observation epoch, radians.
    pub declination: f64,
    /// Apparent magnitude in the catalogue's band; smaller is brighter.
    pub magnitude: f64,
}

impl CatalogueStar {
    /// The star's unit direction in the catalogue frame.
    #[must_use]
    pub fn direction(&self) -> Vec3 {
        let cos_dec = libm::cos(self.declination);
        Vec3::new(
            cos_dec * libm::cos(self.right_ascension),
            cos_dec * libm::sin(self.right_ascension),
            libm::sin(self.declination),
        )
    }
}

/// What a star catalogue has to be able to answer.
///
/// One query, deliberately: "the stars within `radius` of `centre`, no fainter than
/// `magnitude_limit`, brightest first". A blind solve calls it once per candidate field and a
/// verification calls it once; anything richer would be shaped by whichever catalogue was
/// implemented first.
pub trait StarCatalogue {
    /// Stars within `radius` radians of the unit direction `centre` and brighter than
    /// `magnitude_limit`, **sorted brightest first**. The order is part of the contract:
    /// [`quads`] builds its asterisms from the brightest stars in a field, and a differently
    /// ordered catalogue would build different asterisms from the same sky.
    fn query(
        &self,
        centre: Vec3,
        radius: f64,
        magnitude_limit: f64,
    ) -> alloc::vec::Vec<CatalogueStar>;
}

/// A catalogue held in memory — the implementation a test uses, and the shape a cached local
/// extract takes.
#[derive(Debug, Clone, Default)]
pub struct InMemoryCatalogue {
    stars: alloc::vec::Vec<CatalogueStar>,
}

impl InMemoryCatalogue {
    /// Take ownership of a star list. It is sorted brightest first here, so that
    /// [`StarCatalogue::query`]'s ordering contract holds however the caller assembled it.
    #[must_use]
    pub fn new(mut stars: alloc::vec::Vec<CatalogueStar>) -> Self {
        stars.sort_by(|a, b| {
            a.magnitude
                .partial_cmp(&b.magnitude)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        Self { stars }
    }

    /// The stars it holds, brightest first.
    #[must_use]
    pub fn stars(&self) -> &[CatalogueStar] {
        &self.stars
    }
}

impl StarCatalogue for InMemoryCatalogue {
    fn query(
        &self,
        centre: Vec3,
        radius: f64,
        magnitude_limit: f64,
    ) -> alloc::vec::Vec<CatalogueStar> {
        let cos_radius = libm::cos(radius);
        self.stars
            .iter()
            .filter(|star| {
                star.magnitude <= magnitude_limit && star.direction().dot(centre) >= cos_radius
            })
            .copied()
            .collect()
    }
}
