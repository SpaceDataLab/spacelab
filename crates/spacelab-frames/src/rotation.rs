//! The frame algebra.
//!
//! The angles in this crate are checked against ERFA numerically; the algebra wrapped around them is
//! proved in `derivations/frame_transform.mac`, because a number at one epoch cannot tell a correct
//! composition law from a lucky one. What that file establishes: [`Transform::inverse`] followed by
//! [`Transform::state`] is the identity on `(r, v)` for **any** proper rotation and **any** angular
//! velocity — no small-angle, no alignment assumption — and it rests on `det R = +1` and nothing
//! else. Feed the same code an *improper* orthogonal matrix and the round trip comes back wrong by
//! exactly `−2 ω × r`, which at LEO is a kilometre per second. [`Rotation::from_matrix`] is where
//! that precondition is asserted and never checked.

use core::marker::PhantomData;
use core::ops::Mul;

use spacelab_core::{Mat3, Vec3};

/// A reference frame.
///
/// Implemented only by the tag types in this module. A frame is never
/// instantiated; it exists to label a type parameter and then vanish.
///
/// The supertraits exist so that `#[derive(Copy)]` on the frame-tagged types does
/// not demand them all over again: a derive on a struct holding
/// `PhantomData<fn(F) -> F>` adds a spurious `F: Copy` bound, even though
/// `PhantomData` is `Copy` unconditionally.
pub trait Frame: Copy + core::fmt::Debug + PartialEq + Eq + 'static {
    /// Short name, for diagnostics.
    const NAME: &'static str;
}

macro_rules! frame {
    ($(#[$m:meta])* $name:ident, $s:literal) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name;
        impl Frame for $name {
            const NAME: &'static str = $s;
        }
    };
}

frame!(
    /// Geocentric Celestial Reference Frame. Quasi-inertial, the IAU's answer to
    /// "what does not rotate".
    ///
    /// Its axes are ICRF3's, reached through the IAU 2006/2000A precession-nutation
    /// model plus the observed celestial-pole offsets `dX, dY` — nothing here fixes
    /// a radio-source position, so the catalogue realization arrives with the EOP
    /// series, as it does for [`Itrf`].
    Gcrf, "GCRF"
);
frame!(
    /// Celestial Intermediate Reference System. GCRF, with precession and nutation
    /// applied.
    Cirs, "CIRS"
);
frame!(
    /// Terrestrial Intermediate Reference System. CIRS, spun by the Earth Rotation
    /// Angle. Differs from [`Pef`] only by the TIO locator, some tens of
    /// microarcseconds.
    Tirs, "TIRS"
);
frame!(
    /// Pseudo-Earth-Fixed. The classical equinox-based counterpart of [`Tirs`]:
    /// what you get after rotating by sidereal time, before polar motion.
    Pef, "PEF"
);
frame!(
    /// International Terrestrial Reference Frame. Bolted to the crust.
    ///
    /// *Which* realization is not this type's to say: it is the one the supplied
    /// Earth-orientation series realizes. Station coordinates must come from the
    /// matching realization — they differ at the centimetre level, which is the
    /// whole budget of an SLR residual.
    ///
    /// **It is the file's vintage that fixes this, not the row's date.** The IERS
    /// replaced 14 C04 with 20 C04 on 2023-02-14 and *reprocessed the whole series*
    /// in ITRF2020: the current product's own header reads "EOP (IERS) 20 C04 TIME
    /// SERIES consistent with ITRF 2020" and its first row is 1962-01-01. So a 2026
    /// download realizes ITRF2020 back to 1962, and a file archived in 2022 realizes
    /// ITRF2014 all the way forward — the opposite of splitting a series by epoch.
    ///
    /// `finals2000A`, which this workspace reads, states none of that: it has no
    /// header at all, every line is a data row, so the realization cannot be
    /// recovered from the bytes. Naming the chain's frame therefore needs the C04
    /// product, which does say it. Where a *product* names its frame the workspace
    /// now reads it — see `spacelab_gnss::ReferenceFrame` for SP3, not linked
    /// because that crate is not a dependency here — and the agreement is checked
    /// where two products' coordinates are differenced.
    Itrf, "ITRF"
);
frame!(
    /// True Equator, Mean Equinox. What SGP4 emits, and what no standard defines
    /// precisely. Neither SOFA nor ERFA offers it.
    Teme, "TEME"
);

/// A rotation from one frame to another.
///
/// `PhantomData<fn(From) -> To>` rather than `PhantomData<(From, To)>`: it gives
/// the correct variance and keeps `Rotation` unconditionally `Send + Sync`
/// regardless of the tag types, which are never values anyway.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation<From: Frame, To: Frame> {
    matrix: Mat3,
    _frames: PhantomData<fn(From) -> To>,
}

impl<From: Frame, To: Frame> Rotation<From, To> {
    /// Wrap a matrix, asserting by construction which frames it connects.
    ///
    /// The only place a frame claim is made rather than checked. Everything
    /// downstream is then checked by the compiler.
    ///
    /// Two things are asserted here and not checked: the frames, and that the matrix
    /// is a *proper* rotation. The second is not decoration —
    /// `derivations/frame_transform.mac` shows [`Transform::inverse`] inverts
    /// [`Transform::state`] only when `det = +1`, and misses by `−2 ω × r` when it is
    /// `−1`. A reflection would typecheck.
    #[must_use]
    pub const fn from_matrix(matrix: Mat3) -> Self {
        Self {
            matrix,
            _frames: PhantomData,
        }
    }

    /// The underlying matrix.
    #[must_use]
    pub const fn matrix(self) -> Mat3 {
        self.matrix
    }

    /// The inverse rotation. For a proper rotation, the transpose.
    #[must_use]
    pub fn inverse(self) -> Rotation<To, From> {
        Rotation::from_matrix(self.matrix.transpose())
    }
}

impl<A: Frame, B: Frame, C: Frame> Mul<Rotation<A, B>> for Rotation<B, C> {
    type Output = Rotation<A, C>;

    /// Compose: apply the right operand first.
    ///
    /// `b_to_c * a_to_b` is `a_to_c`. Any other pairing fails to typecheck, which
    /// is the entire point of this module.
    fn mul(self, rhs: Rotation<A, B>) -> Rotation<A, C> {
        Rotation::from_matrix(self.matrix * rhs.matrix)
    }
}

/// A position, in a known frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position<F: Frame> {
    /// Metres, in frame `F`.
    pub vector: Vec3,
    _frame: PhantomData<fn(F) -> F>,
}

/// A velocity, in a known frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Velocity<F: Frame> {
    /// Metres per second, in frame `F`.
    pub vector: Vec3,
    _frame: PhantomData<fn(F) -> F>,
}

macro_rules! tagged_vector {
    ($name:ident) => {
        impl<F: Frame> $name<F> {
            /// Tag a raw vector with the frame it is expressed in.
            #[must_use]
            pub const fn new(vector: Vec3) -> Self {
                Self {
                    vector,
                    _frame: PhantomData,
                }
            }
        }

        impl<A: Frame, B: Frame> Mul<$name<A>> for Rotation<A, B> {
            type Output = $name<B>;
            fn mul(self, v: $name<A>) -> $name<B> {
                $name::new(self.matrix * v.vector)
            }
        }
    };
}

tagged_vector!(Position);
tagged_vector!(Velocity);

/// Position and velocity, in a known frame, at an implied instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateVector<F: Frame> {
    /// Position.
    pub position: Position<F>,
    /// Velocity.
    pub velocity: Velocity<F>,
}

impl<F: Frame> StateVector<F> {
    /// Tag a raw pair with the frame it is expressed in.
    #[must_use]
    pub const fn new(position: Vec3, velocity: Vec3) -> Self {
        Self {
            position: Position::new(position),
            velocity: Velocity::new(velocity),
        }
    }
}

/// A rotation between frames that turn relative to one another.
///
/// Carries the angular velocity of `To` with respect to `From`, expressed in
/// `From`. A [`Rotation`] alone is enough for a [`Position`]; only this is enough
/// for a [`StateVector`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform<From: Frame, To: Frame> {
    /// The rotation part.
    pub rotation: Rotation<From, To>,
    /// Angular velocity of `To` relative to `From`, expressed in `From`, rad/s.
    pub omega: Vec3,
}

impl<A: Frame, B: Frame> Transform<A, B> {
    /// Assemble from a rotation and the angular velocity between the frames.
    #[must_use]
    pub const fn new(rotation: Rotation<A, B>, omega: Vec3) -> Self {
        Self { rotation, omega }
    }

    /// Apply to a position. The angular velocity plays no part.
    #[must_use]
    pub fn position(self, p: Position<A>) -> Position<B> {
        self.rotation * p
    }

    /// Apply to a full state.
    ///
    /// ```text
    /// r_B = R · r_A
    /// v_B = R · (v_A − ω × r_A)
    /// ```
    ///
    /// The second term is the one everyone forgets. For GCRF→ITRF at the equator it
    /// is 465 m/s: a satellite at rest on the ground would appear to be moving at
    /// Mach 1.4, and a satellite in orbit would have its velocity wrong by six
    /// percent. The result still *looks* like a plausible orbit.
    #[must_use]
    pub fn state(self, s: StateVector<A>) -> StateVector<B> {
        let r = s.position.vector;
        let v = s.velocity.vector;
        StateVector::new(
            self.rotation.matrix() * r,
            self.rotation.matrix() * (v - self.omega.cross(r)),
        )
    }

    /// The inverse transform.
    ///
    /// The angular velocity of `From` relative to `To` is `−ω`, re-expressed in
    /// `To`.
    #[must_use]
    pub fn inverse(self) -> Transform<B, A> {
        Transform::new(
            self.rotation.inverse(),
            self.rotation.matrix() * (-self.omega),
        )
    }
}
