//! Attitude-dependent **box-wing** (panel) spacecraft models for atmospheric drag and solar
//! radiation pressure — a faithful port of Orekit's `BoxAndSolarArraySpacecraft`
//! (`RadiationSensitive` / `DragSensitive`).
//!
//! # Why a panel model
//!
//! The cannonball forces in [`crate::forces`] carry one scalar ballistic (`C_d A/m`) or radiation
//! (`C_r A/m`) coefficient: the projected area is a constant, independent of how the satellite is
//! turned. That is wrong for any real, non-spherical body — a solar array edge-on to the flux and
//! broadside to it differ by an order of magnitude. The box-wing model replaces the scalar with a
//! **per-facet sum**: a convex body of flat facets (typically the six sides of a box) plus one or
//! more solar-array panels, each facet contributing only when the flux or relative wind strikes its
//! lit/windward side, weighted by its projected area and its own optical/aerodynamic coefficients.
//!
//! # The model (Orekit `BoxAndSolarArraySpacecraft`)
//!
//! Each [`Facet`] has an outward unit normal `n̂` in the **spacecraft body frame**, an area `A`,
//! aerodynamic (`drag`, `lift_ratio`) and optical (`absorption C_a`, `reflection C_s`) coefficients,
//! and a single/double-sided flag. Given the flux (SRP) or relative-velocity (drag) direction in the
//! body frame, a facet contributes iff it faces the incoming flux (`n̂ · d̂ < 0`, after the
//! double-sided flip). The attitude — a body→inertial rotation — is the caller's, threaded in exactly
//! like [`crate::forces::BodyOriented`] threads the Earth-orientation matrix: this crate takes no
//! frames or attitude dependency, only applies the matrix the caller supplies.
//!
//! * **Drag** ([`BoxWing::drag_acceleration`]): with `v̂` the unit relative velocity
//!   `v_atm − v_sat` (Orekit's convention, so drag pushes along `+v̂`), each windward facet
//!   (`n̂ · v̂ < 0`) adds `coeff · C_d · A · (n̂·v̂)`-weighted terms, `coeff = ρ · |v|² / (2m)`.
//!   The optional lift term rotates part of the force from the wind direction toward `n̂`.
//! * **SRP** ([`BoxWing::srp_acceleration`]): each sunlit facet (`n̂ · flux < 0`) adds Vallado's
//!   eq. 8-44 absorption/specular/diffuse decomposition,
//!   `−(P·A·cosθ/m)·[(C_a+C_d)·ŝ + 2(C_s·cosθ + C_d/3)·n̂]`, with `ŝ` the satellite→Sun unit,
//!   `cosθ = n̂·ŝ`, and `C_d = 1 − C_a − C_s` the diffuse coefficient.
//!
//! Both are written once over any [`Scalar`], so the Jacobian — and hence the STM — comes from
//! automatic differentiation just as the cannonball forces do. The facet geometry and coefficients
//! are `f64` constants; only the environment (`ρ`, `v_rel`, `flux`) carries the derivative.
//!
//! # Pluggability
//!
//! [`DragModel`] / [`RadiationModel`] are the Strategy interface a drag / SRP force delegates to:
//! [`BoxWing`] is the anisotropic implementation, [`Cannonball`] the isotropic one that reproduces
//! the scalar [`crate::forces::Drag`] / [`crate::forces::SolarRadiationPressure`]. [`PanelDrag`] /
//! [`PanelSrp`] wrap either behind the [`Acceleration`] trait, so a box-wing (or a cannonball) drops
//! into a `Newtonian((…))` composition. The scalar cannonball forces remain the default.

// Panel geometry speaks in single letters — n̂ for the normal, d̂ for the flux direction, the
// facet coefficients (C_a, C_s, C_d) — and the SRP/drag literature names its axes (x, y, z). These
// are the domain's names, not terse locals; and cosθ vs the coefficient C_d read alike by design.
#![allow(clippy::many_single_char_names, clippy::similar_names)]

use spacelab_autodiff::Scalar;

use crate::ephemeris::Ephemeris;
use crate::forces::{Acceleration, Atmosphere, NoWind, Wind};

/// Smallest positive normal `f64` = `2⁻¹⁰²²` = Hipparchus `Precision.SAFE_MIN`, the threshold below
/// which Orekit treats the radiation flux as null (umbra) and returns zero.
const SAFE_MIN: f64 = f64::MIN_POSITIVE;

/// `2⁻⁵³` = Hipparchus `Precision.EPSILON` — the guard for a solar array whose rotation axis points
/// (near) exactly at the Sun. This is **not** Rust's [`f64::EPSILON`] (`2⁻⁵²`); the values differ by
/// a factor of two and the wrong one would move the degenerate-case boundary.
const HIPPARCHUS_EPSILON: f64 = 1.110_223_024_625_156_5e-16;

// ---------------------------------------------------------------------------------------------------
// Facet geometry
// ---------------------------------------------------------------------------------------------------

/// How a facet's outward unit normal is oriented in the spacecraft body frame — Orekit's polymorphic
/// [`Panel.getNormal`](https://www.orekit.org) (Strategy over panel kind).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PanelNormal {
    /// A fixed outward unit normal — Orekit `FixedPanel`, i.e. one side of the box. Normalised on
    /// construction.
    Fixed([f64; 3]),
    /// A Sun-tracking panel — Orekit `PointingPanel`, i.e. a solar array. It rotates about `axis`
    /// (body frame) so its normal is the satellite→Sun direction projected onto the plane
    /// perpendicular to `axis`, normalised — the best the single rotational degree of freedom allows.
    /// Always double-sided. `axis` is normalised on construction.
    SunPointing {
        /// Rotation axis of the array in the body frame (unit).
        axis: [f64; 3],
    },
}

/// One flat facet of the spacecraft body — an Orekit `Panel`. Carries its own aerodynamic and optical
/// coefficients, since a box side and a solar array reflect and drag differently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Facet {
    /// Body-frame outward normal, fixed or Sun-tracking.
    pub normal: PanelNormal,
    /// Facet area, m².
    pub area: f64,
    /// If true the facet intercepts flux from either side (solar arrays); if false only from its
    /// outward normal side (box sides).
    pub double_sided: bool,
    /// Drag coefficient `C_d`.
    pub drag: f64,
    /// Drag lift ratio in `[0, 1]`: the fraction of molecules that reflect specularly (producing
    /// lift) rather than diffusely. `0` = pure drag along the wind; only relevant for reentry.
    pub lift_ratio: f64,
    /// Radiation-pressure absorption coefficient `C_a` in `[0, 1]`.
    pub absorption: f64,
    /// Radiation-pressure specular-reflection coefficient `C_s` in `[0, 1]`. The diffuse coefficient
    /// is `1 − C_a − C_s`.
    pub reflection: f64,
}

impl Facet {
    /// A fixed-normal facet — Orekit `FixedPanel`. The normal is normalised.
    #[must_use]
    pub fn fixed(
        normal: [f64; 3],
        area: f64,
        double_sided: bool,
        drag: f64,
        lift_ratio: f64,
        absorption: f64,
        reflection: f64,
    ) -> Self {
        Self {
            normal: PanelNormal::Fixed(normalize(normal)),
            area,
            double_sided,
            drag,
            lift_ratio,
            absorption,
            reflection,
        }
    }

    /// A Sun-tracking solar-array facet — Orekit `PointingPanel`. Always double-sided; `axis` is
    /// normalised.
    #[must_use]
    pub fn solar_array(
        axis: [f64; 3],
        area: f64,
        drag: f64,
        lift_ratio: f64,
        absorption: f64,
        reflection: f64,
    ) -> Self {
        Self {
            normal: PanelNormal::SunPointing {
                axis: normalize(axis),
            },
            area,
            double_sided: true,
            drag,
            lift_ratio,
            absorption,
            reflection,
        }
    }

    /// The facet's outward unit normal in the body frame, given the unit satellite→Sun direction in
    /// the body frame (used only by [`PanelNormal::SunPointing`]; ignored for a fixed facet).
    pub(crate) fn unit_normal<S: Scalar>(&self, sun_dir_body: &[S; 3]) -> [S; 3] {
        match self.normal {
            PanelNormal::Fixed(n) => [S::from_f64(n[0]), S::from_f64(n[1]), S::from_f64(n[2])],
            PanelNormal::SunPointing { axis } => {
                // Orekit PointingPanel.getNormal: n̂ = normalize(ŝ − (ŝ·axis) axis), the Sun
                // direction projected onto the plane ⟂ axis. `sun_dir_body` is applyTo(ŝ_inertial).
                let ax = [
                    S::from_f64(axis[0]),
                    S::from_f64(axis[1]),
                    S::from_f64(axis[2]),
                ];
                let d = dot3(sun_dir_body, &ax);
                let f = S::from_f64(1.0) - d * d;
                if f.value() < HIPPARCHUS_EPSILON {
                    // Sun (near) along the array axis — Orekit falls back to axis.orthogonal().
                    let o = orthogonal(&axis);
                    return [S::from_f64(o[0]), S::from_f64(o[1]), S::from_f64(o[2])];
                }
                let s = f.sqrt().recip();
                [
                    s * sun_dir_body[0] - s * d * ax[0],
                    s * sun_dir_body[1] - s * d * ax[1],
                    s * sun_dir_body[2] - s * d * ax[2],
                ]
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// The box-wing spacecraft
// ---------------------------------------------------------------------------------------------------

/// A convex box-wing spacecraft: `P` facets in the body frame, a mass, and the two Orekit global
/// multiplicative factors (`1.0` unless being estimated in orbit determination). Const-generic `P`
/// keeps the facet array on the stack and `no_std`, matching the crate's [`crate::forces::Zonal`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxWing<const P: usize> {
    /// The facets composing the body (6 for a bare box, 7 with one solar array).
    pub facets: [Facet; P],
    /// Spacecraft mass, kg.
    pub mass: f64,
    /// Global drag multiplicative factor — Orekit `GLOBAL_DRAG_FACTOR`, default `1.0`.
    pub drag_factor: f64,
    /// Global radiation multiplicative factor — Orekit `GLOBAL_RADIATION_FACTOR`, default `1.0`.
    pub radiation_factor: f64,
}

impl BoxWing<6> {
    /// The six sides of a parallelepiped aligned with the body axes — Orekit `buildBox`. Facet order
    /// is `−X, +X, −Y, +Y, −Z, +Z`; each side is single-sided with the given coefficients.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn box6(
        x_length: f64,
        y_length: f64,
        z_length: f64,
        mass: f64,
        drag: f64,
        lift_ratio: f64,
        absorption: f64,
        reflection: f64,
    ) -> Self {
        let side = |normal, area| {
            Facet::fixed(
                normal, area, false, drag, lift_ratio, absorption, reflection,
            )
        };
        Self {
            facets: [
                side([-1.0, 0.0, 0.0], y_length * z_length),
                side([1.0, 0.0, 0.0], y_length * z_length),
                side([0.0, -1.0, 0.0], x_length * z_length),
                side([0.0, 1.0, 0.0], x_length * z_length),
                side([0.0, 0.0, -1.0], x_length * y_length),
                side([0.0, 0.0, 1.0], x_length * y_length),
            ],
            mass,
            drag_factor: 1.0,
            radiation_factor: 1.0,
        }
    }
}

impl BoxWing<7> {
    /// A box plus one Sun-tracking solar array — Orekit `buildPanels`. The array is appended as the
    /// seventh, double-sided [`PanelNormal::SunPointing`] facet rotating about `array_axis`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn box_with_array(
        x_length: f64,
        y_length: f64,
        z_length: f64,
        array_area: f64,
        array_axis: [f64; 3],
        mass: f64,
        drag: f64,
        lift_ratio: f64,
        absorption: f64,
        reflection: f64,
    ) -> Self {
        let b = BoxWing::<6>::box6(
            x_length, y_length, z_length, mass, drag, lift_ratio, absorption, reflection,
        );
        let array = Facet::solar_array(
            array_axis, array_area, drag, lift_ratio, absorption, reflection,
        );
        Self {
            facets: [
                b.facets[0],
                b.facets[1],
                b.facets[2],
                b.facets[3],
                b.facets[4],
                b.facets[5],
                array,
            ],
            mass,
            drag_factor: 1.0,
            radiation_factor: 1.0,
        }
    }
}

impl<const P: usize> BoxWing<P> {
    /// Body-frame drag acceleration — a direct port of Orekit `BoxAndSolarArraySpacecraft`'s
    /// `dragAcceleration` (the `double` path). `v_rel_body` is the relative velocity `v_atm − v_sat`
    /// (Orekit's convention) in the body frame; `sun_dir_body` is the unit satellite→Sun direction in
    /// the body frame (needed only by Sun-tracking arrays). The result is in the body frame.
    ///
    /// Per windward facet (`n̂·v̂ < 0`, after the double-sided flip): with `f = coeff·C_d·A·(n̂·v̂)`
    /// (`≤ 0`) and `coeff = ρ·|v|²/(2m)`, the contribution is
    /// `(1−lift)·|f|·v̂ + 2·lift·f·n̂`. With `lift = 0` this is the standard projected-area drag
    /// `½ρ|v|²(C_d/m)·A·|n̂·v̂|·v̂` along the wind.
    #[must_use]
    pub fn drag_acceleration<S: Scalar>(
        &self,
        rho: S,
        v_rel_body: &[S; 3],
        sun_dir_body: &[S; 3],
    ) -> [S; 3] {
        let mut force = [S::from_f64(0.0); 3];
        self.for_each_drag_facet(rho, v_rel_body, sun_dir_body, |_, f| {
            force = [force[0] + f[0], force[1] + f[1], force[2] + f[2]];
        });
        let inv_m = S::from_f64(1.0 / self.mass);
        [force[0] * inv_m, force[1] * inv_m, force[2] * inv_m]
    }

    /// Walks the facets the relative wind actually strikes, handing each one its index and its own
    /// body-frame **force**. [`Self::drag_acceleration`] sums these and divides by the mass;
    /// [`crate::torque`] takes their moments about the centre of mass instead, so the windward test
    /// and the Orekit force decomposition live here once rather than in each caller.
    pub(crate) fn for_each_drag_facet<S: Scalar>(
        &self,
        rho: S,
        v_rel_body: &[S; 3],
        sun_dir_body: &[S; 3],
        mut sink: impl FnMut(usize, [S; 3]),
    ) {
        let vnorm2 = dot3(v_rel_body, v_rel_body);
        let inv_vnorm = vnorm2.sqrt().recip();
        let vdir = [
            v_rel_body[0] * inv_vnorm,
            v_rel_body[1] * inv_vnorm,
            v_rel_body[2] * inv_vnorm,
        ];
        // coeff = ρ · dragFactor · |v|² / 2; the mass divides the sum, not each term.
        let coeff = rho * S::from_f64(self.drag_factor) * vnorm2 / S::from_f64(2.0);

        for (index, facet) in self.facets.iter().enumerate() {
            let mut normal = facet.unit_normal(sun_dir_body);
            let mut d = dot3(&normal, &vdir);
            if facet.double_sided && d.value() > 0.0 {
                normal = [-normal[0], -normal[1], -normal[2]];
                d = -d;
            }
            if d.value() < 0.0 {
                let f = coeff * S::from_f64(facet.drag * facet.area) * d; // ≤ 0 here
                // |f| = −f since f ≤ 0 in this branch — bit-identical to Orekit's FastMath.abs(f).
                let a_v = -f * S::from_f64(1.0 - facet.lift_ratio);
                let a_n = f * S::from_f64(2.0 * facet.lift_ratio);
                sink(
                    index,
                    [
                        a_v * vdir[0] + a_n * normal[0],
                        a_v * vdir[1] + a_n * normal[1],
                        a_v * vdir[2] + a_n * normal[2],
                    ],
                );
            }
        }
    }

    /// Body-frame drag acceleration with a **physical** free-molecular drag coefficient — Sentman's
    /// gas–surface model ([`crate::drag_coefficient::sentman_panel`]) evaluated per windward facet,
    /// instead of the fixed per-facet `C_d`/`lift_ratio` of [`Self::drag_acceleration`]. `conditions`
    /// carries the ambient gas (`T_∞`, `M`, from the atmosphere model) and the surface (`α`, `T_w`).
    ///
    /// Per windward facet the force per unit **full** area is `−c_p n̂ + c_τ t̂` — pressure along the
    /// inward normal, shear along the in-plane flow direction `t̂ = (v̂ + γ n̂)/sin δ` — with `c_p`, `c_τ`
    /// from Sentman at the facet incidence `γ = −n̂·v̂` (the incidence is already inside the
    /// coefficients, so the full area is used, not the projected area). Along the wind this reduces to
    /// `coeff·A·C_d`; a lift component appears off normal incidence. The speed ratio (hence the
    /// coefficients) is taken from the `f64` value of the relative speed, so it stays out of the STM
    /// Jacobian exactly as the fixed `C_d` did; `ρ` and `|v|²` carry the derivative.
    #[must_use]
    pub fn sentman_drag_acceleration<S: Scalar>(
        &self,
        rho: S,
        v_rel_body: &[S; 3],
        sun_dir_body: &[S; 3],
        conditions: &crate::drag_coefficient::FreeMolecularConditions,
    ) -> [S; 3] {
        self.gas_surface_drag_acceleration(rho, v_rel_body, sun_dir_body, &(*conditions).into())
    }

    /// [`Self::sentman_drag_acceleration`] with any gas–surface model: each windward facet's
    /// coefficients are `conditions.surface`'s ([`crate::drag_coefficient::SurfaceModel`]), a
    /// clean or partly adsorbate-covered surface re-emitting diffusely or by Lord's kernels.
    /// [`Self::sentman_drag_acceleration`] is this at a clean diffuse surface, bit for bit.
    #[must_use]
    pub fn gas_surface_drag_acceleration<S: Scalar>(
        &self,
        rho: S,
        v_rel_body: &[S; 3],
        sun_dir_body: &[S; 3],
        conditions: &crate::drag_coefficient::GasSurfaceConditions,
    ) -> [S; 3] {
        let vnorm2 = dot3(v_rel_body, v_rel_body);
        let inv_vnorm = vnorm2.sqrt().recip();
        let vdir = [
            v_rel_body[0] * inv_vnorm,
            v_rel_body[1] * inv_vnorm,
            v_rel_body[2] * inv_vnorm,
        ];
        let coeff = rho * S::from_f64(self.drag_factor) * vnorm2 / S::from_f64(2.0 * self.mass);
        let speed_ratio = crate::drag_coefficient::speed_ratio(
            vnorm2.sqrt().value(),
            conditions.atmosphere_temperature,
            conditions.mean_molar_mass,
        );

        let mut acc = [S::from_f64(0.0); 3];
        for facet in &self.facets {
            let mut normal = facet.unit_normal(sun_dir_body);
            let mut d = dot3(&normal, &vdir);
            if facet.double_sided && d.value() > 0.0 {
                normal = [-normal[0], -normal[1], -normal[2]];
                d = -d;
            }
            if d.value() < 0.0 {
                let gamma = -d.value(); // cos incidence ∈ (0, 1]
                let c = conditions.surface.panel(
                    speed_ratio,
                    gamma,
                    conditions.wall_temperature,
                    conditions.atmosphere_temperature,
                );
                let area = coeff * S::from_f64(facet.area);
                // Pressure force along −n̂ (full area; incidence is inside c_p).
                let cp = S::from_f64(c.pressure);
                acc = [
                    acc[0] - area * cp * normal[0],
                    acc[1] - area * cp * normal[1],
                    acc[2] - area * cp * normal[2],
                ];
                // Shear force c_τ·t̂ with t̂ = (v̂ + γ n̂)/sin δ; c_τ ∝ sin δ, so c_τ/sin δ is finite and
                // the term vanishes cleanly at normal incidence (no defined tangential direction there).
                let sin_delta = libm::sqrt((1.0 - gamma * gamma).max(0.0));
                if sin_delta > 1.0e-9 {
                    let shear = area * S::from_f64(c.shear / sin_delta);
                    let g = S::from_f64(gamma);
                    acc = [
                        acc[0] + shear * (vdir[0] + g * normal[0]),
                        acc[1] + shear * (vdir[1] + g * normal[1]),
                        acc[2] + shear * (vdir[2] + g * normal[2]),
                    ];
                }
            }
        }
        acc
    }

    /// Body-frame solar-radiation-pressure acceleration — a direct port of Orekit
    /// `BoxAndSolarArraySpacecraft`'s `radiationPressureAcceleration` (the `double` path, Vallado
    /// eq. 8-44). `flux_body` is the radiation flux (satellite→Sun points **into** it; it is the
    /// Sun→satellite flux vector) in the body frame; `sun_dir_body` is the unit satellite→Sun
    /// direction in the body frame. Returns the body-frame acceleration; `0` in umbra
    /// (`|flux|² < SAFE_MIN`).
    ///
    /// Per sunlit facet (`n̂·flux < 0`, after the double-sided flip), with `psr = |flux|`
    /// (× `radiation_factor`), diffuse `C_d = 1 − C_a − C_s`:
    /// `cN = 2·A·(n̂·flux)·(C_d/3 − C_s·(n̂·flux)/psr)`, `cS = (A·(n̂·flux)/psr)·(C_s − 1)`,
    /// force `+= cN·n̂ + cS·flux`, finally scaled by `1/m`.
    #[must_use]
    pub fn srp_acceleration<S: Scalar>(&self, flux_body: &[S; 3], sun_dir_body: &[S; 3]) -> [S; 3] {
        let mut force = [S::from_f64(0.0); 3];
        self.for_each_srp_facet(flux_body, sun_dir_body, |_, f| {
            force = [force[0] + f[0], force[1] + f[1], force[2] + f[2]];
        });
        let inv_m = S::from_f64(1.0 / self.mass);
        [force[0] * inv_m, force[1] * inv_m, force[2] * inv_m]
    }

    /// Walks the sunlit facets, handing each one its index and its own body-frame **force**.
    /// [`Self::srp_acceleration`] sums these and divides by the mass; [`crate::torque`] takes their
    /// moments about the centre of mass instead, so the umbra gate and the Orekit reflection model
    /// live here once rather than in each caller.
    pub(crate) fn for_each_srp_facet<S: Scalar>(
        &self,
        flux_body: &[S; 3],
        sun_dir_body: &[S; 3],
        mut sink: impl FnMut(usize, [S; 3]),
    ) {
        // Umbra gate on |flux|², from the value alone so it does not enter the Jacobian — the shadow
        // boundary is a measure-zero set the STM is not defined across, exactly as the cannonball SRP.
        if dot3(flux_body, flux_body).value() < SAFE_MIN {
            return;
        }
        let rf = S::from_f64(self.radiation_factor);
        let flux_sat = [flux_body[0] * rf, flux_body[1] * rf, flux_body[2] * rf];
        let inv_psr = dot3(&flux_sat, &flux_sat).sqrt().recip();

        for (index, facet) in self.facets.iter().enumerate() {
            let mut normal = facet.unit_normal(sun_dir_body);
            let mut d = dot3(&normal, &flux_sat);
            if facet.double_sided && d.value() > 0.0 {
                normal = [-normal[0], -normal[1], -normal[2]];
                d = -d;
            }
            if d.value() < 0.0 {
                let diffuse = 1.0 - (facet.absorption + facet.reflection);
                // cN = 2·A·d·(C_d/3 − C_s·d/psr);  cS = (A·d/psr)·(C_s − 1).
                let inner =
                    S::from_f64(diffuse / 3.0) - S::from_f64(facet.reflection) * d * inv_psr;
                let c_n = S::from_f64(2.0 * facet.area) * d * inner;
                let c_s =
                    S::from_f64(facet.area) * d * inv_psr * S::from_f64(facet.reflection - 1.0);
                sink(
                    index,
                    [
                        c_n * normal[0] + c_s * flux_sat[0],
                        c_n * normal[1] + c_s * flux_sat[1],
                        c_n * normal[2] + c_s * flux_sat[2],
                    ],
                );
            }
        }
    }

    /// Inertial-frame drag acceleration: rotate the inertial relative velocity and Sun direction into
    /// the body frame with `inertial_to_body` (Orekit `attitude.getRotation().getMatrix()`, i.e.
    /// `v_body = M·v_inertial`), apply [`BoxWing::drag_acceleration`], rotate the result back with
    /// `Mᵀ`. `v_rel_inertial = v_atm − v_sat`; `sun_dir_inertial` is the unit satellite→Sun vector.
    #[must_use]
    pub fn drag_acceleration_inertial<S: Scalar>(
        &self,
        rho: S,
        v_rel_inertial: &[S; 3],
        sun_dir_inertial: &[S; 3],
        inertial_to_body: &[[f64; 3]; 3],
    ) -> [S; 3] {
        let v_body = to_body(inertial_to_body, v_rel_inertial);
        let sun_body = to_body(inertial_to_body, sun_dir_inertial);
        let a_body = self.drag_acceleration(rho, &v_body, &sun_body);
        to_inertial(inertial_to_body, &a_body)
    }

    /// Inertial-frame SRP acceleration: the [`BoxWing::drag_acceleration_inertial`] round-trip applied
    /// to [`BoxWing::srp_acceleration`]. `flux_inertial` is the Sun→satellite flux vector.
    #[must_use]
    pub fn srp_acceleration_inertial<S: Scalar>(
        &self,
        flux_inertial: &[S; 3],
        sun_dir_inertial: &[S; 3],
        inertial_to_body: &[[f64; 3]; 3],
    ) -> [S; 3] {
        let flux_body = to_body(inertial_to_body, flux_inertial);
        let sun_body = to_body(inertial_to_body, sun_dir_inertial);
        let a_body = self.srp_acceleration(&flux_body, &sun_body);
        to_inertial(inertial_to_body, &a_body)
    }
}

// ---------------------------------------------------------------------------------------------------
// Pluggable spacecraft models
// ---------------------------------------------------------------------------------------------------

/// A spacecraft's aerodynamic response — the Strategy an attitude-aware drag force delegates to,
/// mirroring Orekit's `DragSensitive`. `v_rel_body` is `v_atm − v_sat` in the body frame,
/// `sun_dir_body` the unit satellite→Sun direction in the body frame; the result is in the body
/// frame. [`BoxWing`] is the anisotropic implementation, [`Cannonball`] the isotropic one.
pub trait DragModel {
    /// Body-frame drag acceleration.
    fn drag<S: Scalar>(&self, rho: S, v_rel_body: &[S; 3], sun_dir_body: &[S; 3]) -> [S; 3];
}

/// A spacecraft's radiation response — the Strategy an SRP force delegates to, mirroring Orekit's
/// `RadiationSensitive`. `flux_body` is the Sun→satellite flux vector in the body frame; the result
/// is in the body frame.
pub trait RadiationModel {
    /// Body-frame radiation-pressure acceleration.
    fn radiation<S: Scalar>(&self, flux_body: &[S; 3], sun_dir_body: &[S; 3]) -> [S; 3];
}

impl<const P: usize> DragModel for BoxWing<P> {
    fn drag<S: Scalar>(&self, rho: S, v_rel_body: &[S; 3], sun_dir_body: &[S; 3]) -> [S; 3] {
        self.drag_acceleration(rho, v_rel_body, sun_dir_body)
    }
}

impl<const P: usize> RadiationModel for BoxWing<P> {
    fn radiation<S: Scalar>(&self, flux_body: &[S; 3], sun_dir_body: &[S; 3]) -> [S; 3] {
        self.srp_acceleration(flux_body, sun_dir_body)
    }
}

/// The isotropic cannonball as a pluggable spacecraft model — Orekit `IsotropicDrag` /
/// `IsotropicRadiationSingleCoefficient`. Rotation-invariant, so it ignores the attitude and the
/// body/inertial distinction, reproducing the scalar [`crate::forces::Drag`] /
/// [`crate::forces::SolarRadiationPressure`]. Keeps the box-wing and the cannonball interchangeable
/// behind [`DragModel`] / [`RadiationModel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cannonball {
    /// Spacecraft mass, kg.
    pub mass: f64,
    /// Drag coefficient `C_d`.
    pub drag_coeff: f64,
    /// Cross-sectional area, m².
    pub area: f64,
    /// Radiation-pressure coefficient `C_r`.
    pub cr: f64,
}

impl DragModel for Cannonball {
    fn drag<S: Scalar>(&self, rho: S, v_rel_body: &[S; 3], _sun_dir_body: &[S; 3]) -> [S; 3] {
        // Orekit IsotropicDrag: a = (|v|·ρ·C_d·A / (2m))·v, v = v_atm − v_sat (so a is along +v).
        let vnorm = dot3(v_rel_body, v_rel_body).sqrt();
        let k = vnorm * rho * S::from_f64(self.drag_coeff * self.area / (2.0 * self.mass));
        [k * v_rel_body[0], k * v_rel_body[1], k * v_rel_body[2]]
    }
}

impl RadiationModel for Cannonball {
    fn radiation<S: Scalar>(&self, flux_body: &[S; 3], _sun_dir_body: &[S; 3]) -> [S; 3] {
        // Orekit IsotropicRadiationSingleCoefficient: a = (C_r·A/m)·flux (flux points Sun→sat).
        let k = S::from_f64(self.cr * self.area / self.mass);
        [k * flux_body[0], k * flux_body[1], k * flux_body[2]]
    }
}

// ---------------------------------------------------------------------------------------------------
// Force wrappers (Acceleration trait)
// ---------------------------------------------------------------------------------------------------

/// Aerodynamic drag on a pluggable [`DragModel`], as an inertial-frame [`Acceleration`] for a
/// `Newtonian((…))` composition. The atmosphere co-rotates about `+z` at `earth_rotation_rate`
/// (`v_atm = ω × r`); the relative velocity `v_atm − v_sat` (Orekit convention) and the unit
/// satellite→Sun direction are rotated into the body frame by `attitude(t)` (inertial→body,
/// `v_body = M·v_inertial`), the model is evaluated, and the result rotated back.
///
/// Mirrors [`crate::forces::BodyOriented`] (attitude as a closure, no frames dependency) and carries
/// the Sun ephemeris that Sun-tracking arrays need; for a fixed box or a [`Cannonball`] the Sun
/// direction is simply unused by the model.
pub struct PanelDrag<M, A, R, E, W = NoWind> {
    /// The spacecraft aerodynamic model.
    pub model: M,
    /// The atmospheric density model.
    pub atmosphere: A,
    /// Earth rotation rate `ω`, rad/s (the atmosphere co-rotates about `+z`).
    pub earth_rotation_rate: f64,
    /// Inertial→body attitude matrix at time `t` (`v_body = M·v_inertial`).
    pub attitude: R,
    /// The Sun's position over time (for Sun-tracking arrays).
    pub sun: E,
    /// The neutral wind on top of co-rotation; [`NoWind`] for a rigidly co-rotating atmosphere.
    ///
    /// A panel model needs this more than the cannonball [`crate::forces::Drag`] does, not less: the
    /// wind turns the relative-velocity vector, and it is the *direction* of that vector in the body
    /// frame that picks out which facets are windward.
    pub wind: W,
}

impl<M, A, R, E, W> Acceleration for PanelDrag<M, A, R, E, W>
where
    M: DragModel,
    A: Atmosphere,
    R: Fn(f64) -> [[f64; 3]; 3],
    E: Ephemeris,
    W: Wind,
{
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let omega = S::from_f64(self.earth_rotation_rate);
        let w = self.wind.velocity(t.value(), r);
        // relativeVelocity = v_atm − v_sat, v_atm = ω × r + w = (−ω y, ω x, 0) + w.
        let v_rel = [
            w[0] - omega * r[1] - v[0],
            w[1] + omega * r[0] - v[1],
            w[2] - v[2],
        ];
        let rho = self.atmosphere.density(t.value(), r);
        let m = (self.attitude)(t.value());
        let sun_dir = unit_sun_dir(&self.sun, t.value(), r);
        let a_body = self
            .model
            .drag(rho, &to_body(&m, &v_rel), &to_body(&m, &sun_dir));
        to_inertial(&m, &a_body)
    }
}

/// Solar radiation pressure on a pluggable [`RadiationModel`], as an inertial-frame [`Acceleration`].
/// The flux is the cannonball form `ν·k_ref/d² · d̂` with `d` the satellite→Sun vector and `ν ∈ {0, 1}`
/// the cylindrical-shadow illumination of [`crate::forces::SolarRadiationPressure`]; it and the unit
/// satellite→Sun direction are rotated into the body frame by `attitude(t)`, the model evaluated, and
/// the result rotated back. (The conical umbra/penumbra refinement is out of scope for the wrapper;
/// the facet physics is validated directly against Orekit.)
pub struct PanelSrp<M, R, E> {
    /// The spacecraft radiation model.
    pub model: M,
    /// Reference flux `k_ref = P_ref · d_ref²`, N·m² — the flux at unit distance.
    pub k_ref: f64,
    /// Earth radius for the cylindrical shadow, m.
    pub earth_radius: f64,
    /// Inertial→body attitude matrix at time `t`.
    pub attitude: R,
    /// The Sun's position over time.
    pub sun: E,
}

impl<M, R, E> Acceleration for PanelSrp<M, R, E>
where
    M: RadiationModel,
    R: Fn(f64) -> [[f64; 3]; 3],
    E: Ephemeris,
{
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let sun = self.sun.position(t.value());
        if crate::ephemeris::is_absent(sun) {
            return [S::from_f64(0.0); 3];
        }

        // Cylindrical shadow, from the f64 value: lit unless behind Earth and inside its radius.
        let sat = [r[0].value(), r[1].value(), r[2].value()];
        let sun_mag = (sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]).sqrt();
        let along = (sat[0] * sun[0] + sat[1] * sun[1] + sat[2] * sun[2]) / sun_mag;
        let perp2 = (sat[0] * sat[0] + sat[1] * sat[1] + sat[2] * sat[2]) - along * along;
        let nu = if along < 0.0 && perp2 < self.earth_radius * self.earth_radius {
            0.0
        } else {
            1.0
        };

        // flux = ν·k_ref/|d|² · (d̂), d = r_sun − r_sat, pointing Sun→sat is −d̂; Orekit's flux vector
        // is ν·k_ref/|sunSat|³ · sunSat with sunSat = r_sat − r_sun, i.e. along −d.
        let sun_sat = [
            r[0] - S::from_f64(sun[0]),
            r[1] - S::from_f64(sun[1]),
            r[2] - S::from_f64(sun[2]),
        ];
        let d2 = dot3(&sun_sat, &sun_sat);
        let inv_d3 = (d2 * d2.sqrt()).recip();
        let scale = S::from_f64(nu * self.k_ref) * inv_d3;
        let flux = [scale * sun_sat[0], scale * sun_sat[1], scale * sun_sat[2]];

        let m = (self.attitude)(t.value());
        let sun_dir = unit_sun_dir(&self.sun, t.value(), r);
        let a_body = self
            .model
            .radiation(&to_body(&m, &flux), &to_body(&m, &sun_dir));
        to_inertial(&m, &a_body)
    }
}

// ---------------------------------------------------------------------------------------------------
// Small vector / rotation helpers
// ---------------------------------------------------------------------------------------------------

pub(crate) fn dot3<S: Scalar>(a: &[S; 3], b: &[S; 3]) -> S {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `M · v` — rotate an inertial vector into the body frame.
pub(crate) fn to_body<S: Scalar>(m: &[[f64; 3]; 3], v: &[S; 3]) -> [S; 3] {
    [
        S::from_f64(m[0][0]) * v[0] + S::from_f64(m[0][1]) * v[1] + S::from_f64(m[0][2]) * v[2],
        S::from_f64(m[1][0]) * v[0] + S::from_f64(m[1][1]) * v[1] + S::from_f64(m[1][2]) * v[2],
        S::from_f64(m[2][0]) * v[0] + S::from_f64(m[2][1]) * v[1] + S::from_f64(m[2][2]) * v[2],
    ]
}

/// `Mᵀ · a` — rotate a body-frame vector back into the inertial frame.
pub(crate) fn to_inertial<S: Scalar>(m: &[[f64; 3]; 3], a: &[S; 3]) -> [S; 3] {
    [
        S::from_f64(m[0][0]) * a[0] + S::from_f64(m[1][0]) * a[1] + S::from_f64(m[2][0]) * a[2],
        S::from_f64(m[0][1]) * a[0] + S::from_f64(m[1][1]) * a[1] + S::from_f64(m[2][1]) * a[2],
        S::from_f64(m[0][2]) * a[0] + S::from_f64(m[1][2]) * a[1] + S::from_f64(m[2][2]) * a[2],
    ]
}

/// Unit satellite→Sun direction in the inertial frame, from the value of the position (the array
/// normal enters the Jacobian only weakly and Orekit's `PointingPanel` reads the plain state).
pub(crate) fn unit_sun_dir<S: Scalar, E: Ephemeris>(sun: &E, t: f64, r: &[S; 3]) -> [S; 3] {
    let s = sun.position(t);
    let d = [
        s[0] - r[0].value(),
        s[1] - r[1].value(),
        s[2] - r[2].value(),
    ];
    let u = normalize(d);
    [S::from_f64(u[0]), S::from_f64(u[1]), S::from_f64(u[2])]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let inv = 1.0 / libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// An arbitrary unit vector orthogonal to `v` — Hipparchus `Vector3D.orthogonal()`. Cross the axis
/// with whichever body axis it is least aligned to, so the result is well-conditioned.
fn orthogonal(v: &[f64; 3]) -> [f64; 3] {
    let (x, y, z) = (v[0], v[1], v[2]);
    let threshold = 0.6 * libm::sqrt(x * x + y * y + z * z);
    if libm::fabs(x) <= threshold {
        let inv = 1.0 / libm::sqrt(y * y + z * z);
        [0.0, inv * z, -inv * y]
    } else if libm::fabs(y) <= threshold {
        let inv = 1.0 / libm::sqrt(x * x + z * z);
        [-inv * z, 0.0, inv * x]
    } else {
        let inv = 1.0 / libm::sqrt(x * x + y * y);
        [inv * y, -inv * x, 0.0]
    }
}

#[cfg(test)]
mod tests {
    // Verbatim Orekit `%.17e` golden vectors (orekit_box_wing.csv) — kept at full precision.
    #![allow(
        clippy::float_cmp,
        clippy::unreadable_literal,
        clippy::excessive_precision
    )]
    use super::*;
    use crate::ephemeris::Ephemeris;
    use crate::forces::Acceleration;
    use spacelab_testkit::nan_max;

    // Box-wing golden fixtures from Orekit 13 `BoxAndSolarArraySpacecraft`
    // (`tools/orekit-oracle`, orekit_box_wing.csv). A convex box (1.5×2.0×3.0 m, mass 1000 kg,
    // C_d = 2.2, absorption 0.3, reflection 0.4 ⇒ diffuse 0.3) and a box + Sun-tracking solar array
    // (20 m², axis +Y), each evaluated at a fixed attitude over several (relative-velocity, flux)
    // geometries by calling Orekit's own `dragAcceleration` / `radiationPressureAcceleration` directly.
    // The attitude matrix M (inertial→body) is emitted as columns = applyTo(basis), so M·v equals the
    // Hipparchus rotation exactly; the port applies M inbound and Mᵀ outbound. A lift-enabled box
    // (liftRatio 0.25) exercises the drag lift term.
    const RHO: f64 = 2.0e-12;

    fn box_only() -> BoxWing<6> {
        BoxWing::<6>::box6(1.5, 2.0, 3.0, 1000.0, 2.2, 0.0, 0.3, 0.4)
    }
    fn box_lift() -> BoxWing<6> {
        BoxWing::<6>::box6(1.5, 2.0, 3.0, 1000.0, 2.2, 0.25, 0.3, 0.4)
    }
    fn box_array() -> BoxWing<7> {
        BoxWing::<7>::box_with_array(
            1.5,
            2.0,
            3.0,
            20.0,
            [0.0, 1.0, 0.0],
            1000.0,
            2.2,
            0.0,
            0.3,
            0.4,
        )
    }

    /// Per-component relative error < 1e-9 against Orekit, reporting the actual max.
    fn assert_close(label: &str, kind: &str, got: [f64; 3], want: [f64; 3]) -> f64 {
        let mut max_rel = 0.0_f64;
        for i in 0..3 {
            let rel = if want[i] == 0.0 {
                got[i].abs()
            } else {
                (got[i] - want[i]).abs() / want[i].abs()
            };
            max_rel = nan_max(max_rel, rel);
        }
        std::println!("{label} {kind}: max rel err {max_rel:.3e}");
        assert!(
            max_rel < 1e-9,
            "{label} {kind}: got {got:?} want {want:?} max rel {max_rel:.3e}"
        );
        max_rel
    }

    #[allow(clippy::too_many_arguments)]
    fn check<const P: usize>(
        label: &str,
        model: &BoxWing<P>,
        m: [[f64; 3]; 3],
        relvel: [f64; 3],
        flux: [f64; 3],
        sundir: [f64; 3],
        exp_drag: [f64; 3],
        exp_srp: [f64; 3],
    ) -> f64 {
        let drag = model.drag_acceleration_inertial(RHO, &relvel, &sundir, &m);
        let srp = model.srp_acceleration_inertial(&flux, &sundir, &m);
        let ed = assert_close(label, "drag", drag, exp_drag);
        let es = assert_close(label, "srp", srp, exp_srp);
        ed.max(es)
    }

    /// The box-wing drag + SRP accelerations match Orekit's `BoxAndSolarArraySpacecraft` to < 1e-9
    /// relative across five geometries: a bare box at near-identity and tilted attitudes, a
    /// lift-enabled box (validating the drag lift term), and a box + Sun-tracking array at both
    /// attitudes (validating the `PointingPanel` normal and the double-sided panel).
    #[test]
    #[allow(clippy::too_many_lines)] // five embedded Orekit golden geometries, one per fixture block
    fn box_wing_matches_orekit() {
        let mut worst = 0.0_f64;

        worst = worst.max(check(
            "box_id",
            &box_only(),
            [
                [
                    9.99999999999994200e-01,
                    -7.07827974419919800e-08,
                    8.05621714697613400e-08,
                ],
                [
                    7.07827947785733800e-08,
                    9.99999999999997100e-01,
                    3.30604145422213640e-08,
                ],
                [
                    -8.05621738098697200e-08,
                    -3.30604088398055230e-08,
                    9.99999999999996200e-01,
                ],
            ],
            [7400.0, 900.0, -1200.0],
            [
                2.72242175241453040e-06,
                -3.17615871115028600e-06,
                1.81494783494302070e-06,
            ],
            [1.0, 0.0, 0.0],
            [
                8.47374050370470000e-07,
                1.03059006126138250e-07,
                -1.37412008168184340e-07,
            ],
            [
                2.39900762653069940e-08,
                -2.58978296625905040e-08,
                1.14370087300053210e-08,
            ],
        ));

        worst = worst.max(check(
            "box_tilt",
            &box_only(),
            [
                [
                    7.81639107922839700e-01,
                    -4.82929352591077100e-01,
                    3.94739845178505130e-01,
                ],
                [
                    5.50117295349308300e-01,
                    8.32030097196060800e-01,
                    -7.13924275919757800e-02,
                ],
                [
                    -2.93957932914131400e-01,
                    2.72956329411641900e-01,
                    9.16015052229478500e-01,
                ],
            ],
            [-2600.0, 6100.0, 3300.0],
            [
                -1.37150180009253020e-06,
                9.14334533395020300e-07,
                -4.25165558028684400e-06,
            ],
            [1.0, 0.0, 0.0],
            [
                -3.07462203268989630e-07,
                7.21353630746475900e-07,
                3.90240488764487150e-07,
            ],
            [
                -1.44462782012073580e-08,
                8.92491129137265700e-09,
                -2.95535110056117300e-08,
            ],
        ));

        worst = worst.max(check(
            "lift_tilt",
            &box_lift(),
            [
                [
                    7.81639107922839700e-01,
                    -4.82929352591077100e-01,
                    3.94739845178505130e-01,
                ],
                [
                    5.50117295349308300e-01,
                    8.32030097196060800e-01,
                    -7.13924275919757800e-02,
                ],
                [
                    -2.93957932914131400e-01,
                    2.72956329411641900e-01,
                    9.16015052229478500e-01,
                ],
            ],
            [7400.0, 900.0, -1200.0],
            [
                2.72242175241453040e-06,
                -3.17615871115028600e-06,
                1.81494783494302070e-06,
            ],
            [1.0, 0.0, 0.0],
            [
                1.05063598970169920e-06,
                1.04319104803891920e-07,
                -1.05929245207931560e-07,
            ],
            [
                3.00124364514176600e-08,
                -2.76854288282366800e-08,
                1.78562818589368200e-08,
            ],
        ));

        worst = worst.max(check(
            "arr_id",
            &box_array(),
            [
                [
                    9.99999999999994200e-01,
                    -7.07827974419919800e-08,
                    8.05621714697613400e-08,
                ],
                [
                    7.07827947785733800e-08,
                    9.99999999999997100e-01,
                    3.30604145422213640e-08,
                ],
                [
                    -8.05621738098697200e-08,
                    -3.30604088398055230e-08,
                    9.99999999999996200e-01,
                ],
            ],
            [7400.0, 900.0, -1200.0],
            [
                -7.95287837940697900e-07,
                4.26471318372766600e-06,
                1.84870015260790400e-06,
            ],
            [
                1.68646344367932440e-01,
                -9.04362237043323200e-01,
                -3.92029788079080400e-01,
            ],
            [
                2.15843841394434670e-06,
                2.62512780074312440e-07,
                -3.50017040099083300e-07,
            ],
            [
                -1.72684851234705200e-08,
                5.55753998440712800e-08,
                3.92753913253922540e-08,
            ],
        ));

        worst = worst.max(check(
            "arr_tilt",
            &box_array(),
            [
                [
                    7.81639107922839700e-01,
                    -4.82929352591077100e-01,
                    3.94739845178505130e-01,
                ],
                [
                    5.50117295349308300e-01,
                    8.32030097196060800e-01,
                    -7.13924275919757800e-02,
                ],
                [
                    -2.93957932914131400e-01,
                    2.72956329411641900e-01,
                    9.16015052229478500e-01,
                ],
            ],
            [-2600.0, 6100.0, 3300.0],
            [
                -7.95287837940697900e-07,
                4.26471318372766600e-06,
                1.84870015260790400e-06,
            ],
            [
                1.68646344367932440e-01,
                -9.04362237043323200e-01,
                -3.92029788079080400e-01,
            ],
            [
                -1.05934405235560420e-06,
                2.48538412283430180e-06,
                1.34455206645134470e-06,
            ],
            [
                -5.24378345200247630e-08,
                1.00335642218016190e-07,
                6.21245126259656200e-08,
            ],
        ));

        std::println!("box-wing worst-case relative error vs Orekit: {worst:.3e}");
    }

    fn norm(v: [f64; 3]) -> f64 {
        libm::sqrt(dot3(&v, &v))
    }

    fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn scaled(k: f64, v: [f64; 3]) -> [f64; 3] {
        [k * v[0], k * v[1], k * v[2]]
    }

    /// A general inertial→body attitude (a `Rz·Rx·Rz` Euler rotation): no entry is `0` or `±1`, so
    /// a dropped, transposed or partially applied rotation cannot slip through.
    fn general_attitude() -> [[f64; 3]; 3] {
        let (sa, ca) = libm::sincos(0.7);
        let (sb, cb) = libm::sincos(0.4);
        let (sg, cg) = libm::sincos(1.1);
        [
            [cg * ca - sg * cb * sa, cg * sa + sg * cb * ca, sg * sb],
            [-sg * ca - cg * cb * sa, -sg * sa + cg * cb * ca, cg * sb],
            [sb * sa, -sb * ca, cb],
        ]
    }

    fn assert_vectors_agree(label: &str, got: [f64; 3], want: [f64; 3]) {
        assert!(
            norm(want) > 0.0,
            "{label}: the reference is zero, nothing is being tested"
        );
        let miss = norm(sub(got, want)) / norm(want);
        assert!(
            miss < 1e-14,
            "{label}: {miss:.3e} relative, got {got:?} want {want:?}"
        );
    }

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    /// The six sides in Orekit's `buildBox` order, each with the product of the two lengths it does
    /// **not** span. The Orekit fixtures above use `1.5 × 2.0 × 3.0`, and there `1.5 * 3.0` and
    /// `1.5 + 3.0` are both `4.5`: the ±Y area cannot tell a product from a sum. These three
    /// lengths avoid that curve (`b = a/(a−1)`) on all three pairs. (G3)
    #[test]
    fn box6_gives_each_side_the_product_of_its_two_transverse_lengths() {
        let (x, y, z) = (1.25, 3.5, 7.0);
        let bw = BoxWing::<6>::box6(x, y, z, 1000.0, 2.2, 0.0, 0.3, 0.4);
        let want = [
            ([-1.0, 0.0, 0.0], y * z),
            ([1.0, 0.0, 0.0], y * z),
            ([0.0, -1.0, 0.0], x * z),
            ([0.0, 1.0, 0.0], x * z),
            ([0.0, 0.0, -1.0], x * y),
            ([0.0, 0.0, 1.0], x * y),
        ];
        for (i, (facet, (normal, area))) in bw.facets.iter().zip(want).enumerate() {
            assert_eq!(facet.normal, PanelNormal::Fixed(normal), "facet {i} normal");
            assert_eq!(facet.area, area, "facet {i} area");
            assert!(!facet.double_sided, "facet {i}: a box side is single-sided");
        }
    }

    /// Hipparchus `Vector3D.orthogonal()` crosses with whichever body axis the vector is least
    /// aligned to, so the zero component names the branch that ran and pins the well-conditioning
    /// choice, not merely the perpendicularity. Two of the four cases sit exactly on a `<=`
    /// boundary: `‖(3,4,0)‖ = 5` and `0.6 * 5.0` rounds to exactly `3.0`. `(5,1,10)` sits close
    /// enough to the threshold that shrinking the norm moves it out of the first branch, and
    /// `(4,1,3)` is the only case in the middle branch whose `z` is neither `0` nor `2` — the two
    /// values where `z * z` and `z + z` agree.
    #[test]
    fn orthogonal_is_a_well_conditioned_unit_perpendicular() {
        for (v, zero_at) in [
            ([0.1, 3.0, -4.0], 0_usize),
            ([5.0, 1.0, 10.0], 0),
            ([3.0, 4.0, 0.0], 0),
            ([4.0, 1.0, 3.0], 1),
            ([4.0, 3.0, 0.0], 1),
            ([4.0, 4.0, 2.0], 2),
        ] {
            let o = orthogonal(&v);
            assert!(
                libm::fabs(norm(o) - 1.0) < 1e-15,
                "‖orthogonal({v:?})‖ = {}",
                norm(o)
            );
            assert!(
                libm::fabs(dot3(&o, &v)) < 1e-15 * norm(v),
                "orthogonal({v:?})·v = {}",
                dot3(&o, &v)
            );
            assert_eq!(
                o[zero_at], 0.0,
                "orthogonal({v:?}) = {o:?} took a branch that crosses with the wrong axis"
            );
        }
    }

    /// Four properties determine a [`PanelNormal::SunPointing`] normal uniquely, and all four are
    /// the definition rather than a second copy of the code: unit, perpendicular to the rotation
    /// axis, in the span of the Sun direction and the axis, and on the sunlit side. The Orekit
    /// fixtures put the axis on body `+Y`, where two of the three components are identically zero
    /// and every arithmetic mutation in them is invisible; this axis is general. (G2)
    #[test]
    fn a_sun_pointing_array_turns_as_far_toward_the_sun_as_one_axis_allows() {
        let axis = normalize([1.0, 2.0, 3.0]);
        let facet = Facet::solar_array(axis, 20.0, 2.2, 0.0, 0.3, 0.4);
        assert!(
            facet.double_sided,
            "an array intercepts flux from either side"
        );
        for sun in [[0.6, -0.48, 0.64], [-0.2, 0.3, 0.93], [1.0, 0.0, 0.0]] {
            let s = normalize(sun);
            let n = facet.unit_normal(&s);
            assert!(libm::fabs(norm(n) - 1.0) < 1e-15, "‖n̂‖ = {}", norm(n));
            assert!(
                libm::fabs(dot3(&n, &axis)) < 1e-15,
                "n̂·axis = {}, the array must turn about its axis",
                dot3(&n, &axis)
            );
            assert!(
                libm::fabs(dot3(&n, &cross(s, axis))) < 1e-15,
                "n̂ left the plane spanned by the Sun and the axis"
            );
            assert!(dot3(&n, &s) > 0.0, "n̂ points away from the Sun: {n:?}");
        }
    }

    /// With the Sun exactly along the rotation axis the projection vanishes and Orekit falls back
    /// to `axis.orthogonal()`. `normalize([1,2,3])·itself` is exactly `1.0`, so `f = 1 − d²` is
    /// exactly zero here and the branch is reached rather than approached.
    #[test]
    fn an_array_edge_on_to_the_sun_falls_back_to_an_arbitrary_perpendicular() {
        let axis = normalize([1.0, 2.0, 3.0]);
        assert_eq!(
            dot3(&axis, &axis),
            1.0,
            "the degenerate branch needs d² = 1 exactly"
        );
        let facet = Facet::solar_array(axis, 20.0, 2.2, 0.0, 0.3, 0.4);
        for sun in [axis, [-axis[0], -axis[1], -axis[2]]] {
            assert_eq!(
                facet.unit_normal(&sun),
                orthogonal(&axis),
                "the Sun-along-axis fallback must be Hipparchus's orthogonal()"
            );
        }
    }

    /// A symmetric box in a uniform flux along a body axis produces a force purely along that axis —
    /// the transverse facets cancel by symmetry. Sanity that the facet sum has no spurious coupling.
    #[test]
    fn symmetric_box_axial_flux_is_axial() {
        let bw = BoxWing::<6>::box6(2.0, 2.0, 2.0, 500.0, 2.0, 0.0, 0.2, 0.3);
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let flux = [0.0, 0.0, 4.5e-6]; // +Z
        let a = bw.srp_acceleration_inertial(&flux, &[0.0, 0.0, -1.0], &identity);
        assert!(
            a[0].abs() < 1e-18 && a[1].abs() < 1e-18,
            "transverse leak: {a:?}"
        );
        assert!(a[2] > 0.0, "SRP must push along +Z (away from Sun): {a:?}");
    }

    /// [`Cannonball`] as a [`DragModel`] / [`RadiationModel`] reproduces the closed-form isotropic
    /// drag `−½(C_d A/m)ρ|v|v` and SRP `(C_r A/m)·flux`, and is rotation-invariant (the "cannonball
    /// path"). Fed the Orekit relative-velocity convention `v_atm − v_sat = −v` (static atmosphere),
    /// the drag opposes `v`. (G3)
    #[test]
    fn cannonball_reduces_to_closed_form() {
        let c = Cannonball {
            mass: 800.0,
            drag_coeff: 2.3,
            area: 5.0,
            cr: 1.4,
        };
        let v = [7000.0, -1500.0, 300.0];
        let v_rel = [-v[0], -v[1], -v[2]]; // v_atm − v_sat with v_atm = 0
        let drag = c.drag(RHO, &v_rel, &[1.0, 0.0, 0.0]);
        let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let k = 0.5 * c.drag_coeff * c.area / c.mass * RHO * speed;
        for i in 0..3 {
            assert!((drag[i] + k * v[i]).abs() < 1e-18, "drag[{i}] {}", drag[i]);
        }
        let flux = [1.0e-6, -2.0e-6, 0.5e-6];
        let srp = c.radiation(&flux, &[1.0, 0.0, 0.0]);
        let kr = c.cr * c.area / c.mass;
        for i in 0..3 {
            assert!((srp[i] - kr * flux[i]).abs() < 1e-24, "srp[{i}] {}", srp[i]);
        }
    }

    /// Physical Sentman box-wing drag: a flow along body `−x` leaves only the `+X` facet windward at
    /// normal incidence, so the force is exactly `coeff·A·C_d` along the wind with `C_d` the Sentman
    /// coefficient, and no transverse (lift/shear) leak. Ties the box-wing wiring to the verified
    /// [`crate::drag_coefficient`] module (the coefficient value itself is anchored by that module's
    /// benchmark table).
    #[test]
    fn sentman_box_drag_matches_the_single_windward_facet() {
        use crate::drag_coefficient::{FreeMolecularConditions, sentman_panel, speed_ratio};
        let bw = BoxWing::<6>::box6(1.0, 1.0, 1.0, 100.0, 2.2, 0.0, 0.2, 0.3);
        let v_rel = [-7600.0, 0.0, 0.0]; // v_atm − v_sat: flow along −x ⇒ +X facet windward, δ = 0
        let cond = FreeMolecularConditions {
            atmosphere_temperature: 1000.0,
            mean_molar_mass: 0.016,
            accommodation: 1.0,
            wall_temperature: 300.0,
        };
        let rho = 1.0e-12;
        let a = bw.sentman_drag_acceleration(rho, &v_rel, &[0.0, 0.0, 1.0], &cond);

        let cd = sentman_panel(speed_ratio(7600.0, 1000.0, 0.016), 1.0, 1.0, 300.0, 1000.0).drag;
        assert!((2.0..2.2).contains(&cd), "cd {cd}"); // physical band, not the fixed 2.2
        let coeff = rho * 7600.0 * 7600.0 / (2.0 * 100.0); // ρ|v|²/(2m), area_x = 1
        assert!(
            (a[0] + coeff * cd).abs() < 1e-18,
            "a_x {} vs {}",
            a[0],
            -coeff * cd
        );
        assert!(
            a[1].abs() < 1e-18 && a[2].abs() < 1e-18,
            "transverse leak {a:?}"
        );
    }

    #[test]
    fn gas_surface_box_drag_uses_the_surface_model_on_each_windward_facet() {
        use crate::drag_coefficient::{
            GasSurfaceConditions, Reemission, SurfaceModel, speed_ratio,
        };
        let bw = BoxWing::<6>::box6(1.0, 1.0, 1.0, 100.0, 2.2, 0.0, 0.2, 0.3);
        let v_rel = [-7600.0, 0.0, 0.0];
        let surface = SurfaceModel {
            clean: Reemission::Cll {
                normal_accommodation: 0.5,
                tangential_accommodation: 0.9,
            },
            adsorbate: Reemission::Diffuse { accommodation: 1.0 },
            coverage: 0.4,
        };
        let cond = GasSurfaceConditions {
            atmosphere_temperature: 1000.0,
            mean_molar_mass: 0.016,
            wall_temperature: 300.0,
            surface,
        };
        let rho = 1.0e-12;
        let a = bw.gas_surface_drag_acceleration(rho, &v_rel, &[0.0, 0.0, 1.0], &cond);
        let cd = surface
            .panel(speed_ratio(7600.0, 1000.0, 0.016), 1.0, 300.0, 1000.0)
            .drag;
        // Between the diffuse 2.15 and the CLL clean surface's 3.45 at this speed ratio.
        assert!((2.5..3.0).contains(&cd), "cd {cd}");
        let coeff = rho * 7600.0 * 7600.0 / (2.0 * 100.0);
        assert!(
            (a[0] + coeff * cd).abs() < 1e-18,
            "a_x {} vs {}",
            a[0],
            -coeff * cd
        );
        assert!(
            a[1].abs() < 1e-18 && a[2].abs() < 1e-18,
            "transverse leak {a:?}"
        );

        // A clean diffuse surface is Sentman's model, bit for bit.
        let sentman = sentman_conditions();
        let v_oblique = [-5200.0, 3100.0, -4400.0];
        let sun = [0.35, 0.62, -0.70];
        assert_eq!(
            bw.gas_surface_drag_acceleration(rho, &v_oblique, &sun, &sentman.into()),
            bw.sentman_drag_acceleration(rho, &v_oblique, &sun, &sentman)
        );
    }

    /// A Sun far off every axis, so no component of the satellite→Sun vector, the flux or the
    /// shadow test is identically zero.
    const SUN_AT: [f64; 3] = [0.6e11, -1.2e11, 0.5e11];
    const EARTH_RADIUS: f64 = 6.378_137e6;
    const K_REF: f64 = 1.0e17;

    fn sentman_conditions() -> crate::drag_coefficient::FreeMolecularConditions {
        crate::drag_coefficient::FreeMolecularConditions {
            atmosphere_temperature: 950.0,
            mean_molar_mass: 0.0165,
            accommodation: 0.85,
            wall_temperature: 310.0,
        }
    }

    /// A Sun at the origin is [`crate::ephemeris::is_absent`]'s sentinel: no Sun, no pressure.
    ///
    /// Left to run, the cylindrical-shadow test divides by `‖s‖` and gets `NaN`, which fails
    /// `along < 0.0` and so reports the satellite *lit*; the flux is then built from
    /// `sun_sat = r − 0 = r` and comes out finite and enormous, `k_ref/‖r‖²` with the AU-scaled
    /// `k_ref`. A shadow model that answers "lit" on a NaN is the part worth naming: the failure
    /// does not stop at the term that produced it.
    #[test]
    fn an_absent_sun_exerts_no_panel_pressure() {
        let a = PanelSrp {
            model: box_array(),
            k_ref: K_REF,
            earth_radius: EARTH_RADIUS,
            attitude: |_t| general_attitude(),
            sun: FixedBody([0.0; 3]),
        }
        .acceleration::<f64>(0.0, &[3.0e6, -6.0e6, 2.5e6], &[0.0; 3]);
        assert_eq!(a, [0.0; 3], "absent Sun: {a:?}");
    }

    /// [`PanelSrp`]'s own work is the inverse-square flux and the cylindrical shadow; the facet
    /// physics it delegates to is already pinned against Orekit above. Asserting the composition
    /// separates the two, and the flux is built here as *magnitude × direction* rather than the
    /// `1/(d²√d²)` the code uses, so agreeing is evidence and not a transcription. (S1)
    ///
    /// The existing wrapper test put the satellite on the Sun line with an isotropic model, where
    /// two components vanish identically, the shadow branch is unreachable, and the only assertion
    /// on the third is its **sign** — which every arithmetic mutation survives.
    #[test]
    fn panel_srp_builds_the_inverse_square_flux_and_the_shadow_then_delegates() {
        let m = general_attitude();
        let bw = box_array();
        let ps = PanelSrp {
            model: bw,
            k_ref: K_REF,
            earth_radius: EARTH_RADIUS,
            attitude: |_t| general_attitude(),
            sun: FixedBody(SUN_AT),
        };

        // Sunward, deep in the umbra, and anti-sunward but clear of the shadow cylinder.
        for (label, r, lit) in [
            ("sunward", [3.0e6, -6.0e6, 2.5e6], true),
            ("umbra", [-3.0e6, 6.0e6, -2.5e6], false),
            (
                "behind but outside the cylinder",
                [7.0e6, 6.0e6, -2.5e6],
                true,
            ),
        ] {
            let got = ps.acceleration(0.0, &r, &[0.0; 3]);
            if !lit {
                assert_eq!(got, [0.0; 3], "{label}: the umbra must return exactly zero");
                continue;
            }
            let sun_sat = sub(r, SUN_AT);
            let d = norm(sun_sat);
            let flux = scaled(K_REF / (d * d) / d, sun_sat);
            let sat_sun = sub(SUN_AT, r);
            let want =
                bw.srp_acceleration_inertial(&flux, &scaled(1.0 / norm(sat_sun), sat_sun), &m);
            assert_vectors_agree(label, got, want);
        }
    }

    /// The cylindrical shadow is a *decision*, and the three fixtures above sit far from its wall —
    /// every arithmetic mutation of `‖sun‖`, of `along` or of `perp² = ‖sat‖² − along²` still leaves
    /// them on the side they started. Walking the wall pins it instead: one part in `1e9` of the
    /// cylinder radius either side of it has to flip the satellite between dark and lit, and the
    /// same offset on the *sunward* side has to stay lit however far inside the cylinder it is.
    #[test]
    fn the_cylindrical_shadow_flips_across_the_terminator_cylinder() {
        let sun_dir = scaled(1.0 / norm(SUN_AT), SUN_AT);
        let across = cross(sun_dir, [0.0, 0.0, 1.0]);
        let perp_dir = scaled(1.0 / norm(across), across);

        for (label, along, perp, lit) in [
            (
                "just inside the cylinder",
                -7.0e6,
                EARTH_RADIUS * (1.0 - 1e-9),
                false,
            ),
            (
                "just outside the cylinder",
                -7.0e6,
                EARTH_RADIUS * (1.0 + 1e-9),
                true,
            ),
            (
                "the same offset, sunward",
                7.0e6,
                EARTH_RADIUS * (1.0 - 1e-9),
                true,
            ),
        ] {
            let r = core::array::from_fn(|i| along * sun_dir[i] + perp * perp_dir[i]);
            let got = PanelSrp {
                model: box_array(),
                k_ref: K_REF,
                earth_radius: EARTH_RADIUS,
                attitude: |_t| general_attitude(),
                sun: FixedBody(SUN_AT),
            }
            .acceleration(0.0, &r, &[0.0; 3]);
            if lit {
                assert!(norm(got) > 0.0, "{label}: shadowed, expected sunlight");
            } else {
                assert_eq!(got, [0.0; 3], "{label}: lit, expected the umbra");
            }
        }
    }

    /// `perp² < R²` is strict, and unlike the ties elsewhere in this module the tie here is
    /// reachable and constructible exactly. With the Sun on `+x` at `2⁴⁰ m`, a satellite at
    /// `(−2²³, R, 0)` and `R = 2¹²`: `‖sun‖ = 2⁴⁰` and `sat·sun = −2⁶³` are exact, so `along = −2²³`
    /// is exact, `‖sat‖² = 2⁴⁶ + 2²⁴` is exact, and `perp²` is exactly `2²⁴ = R²`. Grazing the
    /// cylinder wall is not being inside it.
    #[test]
    fn a_satellite_exactly_on_the_shadow_cylinder_is_lit() {
        let got = PanelSrp {
            model: box_array(),
            k_ref: K_REF,
            earth_radius: 4096.0,
            attitude: |_t| general_attitude(),
            sun: FixedBody([1_099_511_627_776.0, 0.0, 0.0]),
        }
        .acceleration(0.0, &[-8_388_608.0, 4096.0, 0.0], &[0.0; 3]);
        assert!(norm(got) > 0.0, "a satellite on the wall must stay lit");
    }

    /// [`PanelDrag`]'s own work is the co-rotating relative velocity `ω × r − v` and the unit
    /// satellite→Sun direction. The cross product is written here as a cross product, not as the
    /// expanded `(−ωy, ωx, 0)` the code carries, and the state is off every axis so all three
    /// components are live — the existing wrapper test used a static atmosphere (`ω = 0`) and a
    /// model that ignores the Sun direction entirely.
    #[test]
    fn panel_drag_builds_the_corotating_relative_velocity_then_delegates() {
        const OMEGA: f64 = 7.292_115e-5;
        let m = general_attitude();
        let bw = box_array();
        let pd = PanelDrag {
            model: bw,
            atmosphere: ConstDensity(RHO),
            earth_rotation_rate: OMEGA,
            wind: NoWind,
            attitude: |_t| general_attitude(),
            sun: FixedBody(SUN_AT),
        };
        let r = [4.1e6, 3.3e6, -2.7e6];
        let v = [-2100.0, 6400.0, 3000.0];

        let got = pd.acceleration(0.0, &r, &v);
        let v_rel = sub(cross([0.0, 0.0, OMEGA], r), v);
        let sat_sun = sub(SUN_AT, r);
        let want =
            bw.drag_acceleration_inertial(RHO, &v_rel, &scaled(1.0 / norm(sat_sun), sat_sun), &m);
        assert_vectors_agree("PanelDrag", got, want);
    }

    /// A wind that rotates rigidly about the pole, `w = Ω ẑ × r`.
    struct SolidBodyWind {
        rate: f64,
    }

    impl Wind for SolidBodyWind {
        fn velocity<S: Scalar>(&self, _t: f64, r: &[S; 3]) -> [S; 3] {
            let rate = S::from_f64(self.rate);
            [
                S::from_f64(0.0) - rate * r[1],
                rate * r[0],
                S::from_f64(0.0),
            ]
        }
    }

    /// A solid-body wind enters `v_rel` at exactly the place co-rotation does, so the two must be the
    /// same force. Nothing else here can see a wind that arrives with the wrong sign or in the wrong
    /// frame: the panel model takes `v_rel`'s *direction* to pick the windward facets, so a flipped
    /// wind reads as an atmosphere counter-rotating at twice the wind speed and still produces a
    /// perfectly plausible drag magnitude.
    #[test]
    fn a_solid_body_wind_is_a_change_of_co_rotation_rate() {
        const OMEGA: f64 = 7.292_115e-5;
        let extra = 150.0 / 6.778e6; // a 150 m/s zonal wind at 400 km, as a rotation rate
        let r = [4.1e6, 3.3e6, -2.7e6];
        let v = [-2100.0, 6400.0, 3000.0];

        let winded = PanelDrag {
            model: box_array(),
            atmosphere: ConstDensity(RHO),
            earth_rotation_rate: OMEGA,
            wind: SolidBodyWind { rate: extra },
            attitude: |_t| general_attitude(),
            sun: FixedBody(SUN_AT),
        }
        .acceleration(0.0, &r, &v);
        let faster = PanelDrag {
            model: box_array(),
            atmosphere: ConstDensity(RHO),
            earth_rotation_rate: OMEGA + extra,
            wind: NoWind,
            attitude: |_t| general_attitude(),
            sun: FixedBody(SUN_AT),
        }
        .acceleration(0.0, &r, &v);
        assert_vectors_agree("PanelDrag wind", winded, faster);
    }

    /// Orekit's two global multiplicative factors are `1.0` in every fixture, and `ρ · 1.0` cannot
    /// tell a product from a quotient. Both scale their acceleration linearly: the factor multiplies
    /// the drag coefficient outright, and for radiation it cancels out of the specular ratio
    /// `d/psr` and out of the diffuse term, leaving the force proportional to it. (G4)
    #[test]
    fn the_global_factors_scale_their_acceleration_linearly() {
        let base = box_only();
        let m = general_attitude();
        let v_rel = [-6100.0, 3200.0, -1500.0];
        let sun_dir = scaled(1.0 / norm([0.3, -0.9, 0.4]), [0.3, -0.9, 0.4]);
        let flux = [1.1e-6, -3.0e-6, 0.7e-6];
        let a_drag = base.drag_acceleration_inertial(RHO, &v_rel, &sun_dir, &m);
        let a_srp = base.srp_acceleration_inertial(&flux, &sun_dir, &m);

        for factor in [0.5, 2.5] {
            let dragged = BoxWing {
                drag_factor: factor,
                ..base
            };
            assert_vectors_agree(
                "drag_factor",
                dragged.drag_acceleration_inertial(RHO, &v_rel, &sun_dir, &m),
                scaled(factor, a_drag),
            );
            let radiated = BoxWing {
                radiation_factor: factor,
                ..base
            };
            assert_vectors_agree(
                "radiation_factor",
                radiated.srp_acceleration_inertial(&flux, &sun_dir, &m),
                scaled(factor, a_srp),
            );
            assert_vectors_agree(
                "drag_factor, Sentman",
                dragged.sentman_drag_acceleration(RHO, &v_rel, &sun_dir, &sentman_conditions()),
                scaled(
                    factor,
                    base.sentman_drag_acceleration(RHO, &v_rel, &sun_dir, &sentman_conditions()),
                ),
            );
        }
    }

    /// Sentman box-wing drag at oblique incidence, resolved the other way round. The code assembles
    /// each facet's force as a pressure along `−n̂` plus a shear along the in-plane flow direction;
    /// the identical force resolved **along the flow and perpendicular to it** is `C_d v̂ − C_l û_L`
    /// with `û_L = (n̂ + γ v̂)/sin δ`, and `C_d`, `C_l` come from the benchmark-pinned coefficient
    /// module rather than from here. The two decompositions share no expression.
    ///
    /// The existing Sentman test drove a cube with a body-axis wind at normal incidence: `sin δ = 0`
    /// so the whole shear block never ran, two of the three components were identically zero, and
    /// with no double-sided facet the windward flip was unreachable.
    #[test]
    fn sentman_box_wing_drag_resolves_along_the_flow_at_oblique_incidence() {
        use crate::drag_coefficient::{sentman_panel, speed_ratio};

        // A general array axis: `box_array`'s `+Y` puts an exact zero in the array normal's middle
        // component, where negating it on the windward flip is invisible.
        let bw = BoxWing::<7>::box_with_array(
            1.5,
            2.0,
            3.0,
            20.0,
            [1.0, 2.0, 3.0],
            1000.0,
            2.2,
            0.0,
            0.3,
            0.4,
        );
        let cond = sentman_conditions();
        let rho = 3.1e-12;
        let v_rel = [-5200.0, 3100.0, -4400.0];
        let sun_dir = scaled(1.0 / norm([0.35, 0.62, -0.70]), [0.35, 0.62, -0.70]);
        let got = bw.sentman_drag_acceleration(rho, &v_rel, &sun_dir, &cond);

        let speed = norm(v_rel);
        let v_hat = scaled(1.0 / speed, v_rel);
        let coeff = rho * speed * speed / (2.0 * bw.mass);
        let s = speed_ratio(speed, cond.atmosphere_temperature, cond.mean_molar_mass);

        let mut want = [0.0; 3];
        let (mut flipped, mut oblique) = (0, 0);
        for facet in &bw.facets {
            let mut n = facet.unit_normal(&sun_dir);
            if facet.double_sided && dot3(&n, &v_hat) > 0.0 {
                n = scaled(-1.0, n);
                flipped += 1;
            }
            let gamma = -dot3(&n, &v_hat);
            if gamma <= 0.0 {
                continue;
            }
            let c = sentman_panel(
                s,
                gamma,
                cond.accommodation,
                cond.wall_temperature,
                cond.atmosphere_temperature,
            );
            let sin_delta = libm::sqrt((1.0 - gamma * gamma).max(0.0));
            let mut f = scaled(c.drag, v_hat);
            if sin_delta > 0.0 {
                oblique += 1;
                let lift_dir = scaled(
                    1.0 / sin_delta,
                    [
                        n[0] + gamma * v_hat[0],
                        n[1] + gamma * v_hat[1],
                        n[2] + gamma * v_hat[2],
                    ],
                );
                f = sub(f, scaled(c.lift, lift_dir));
            }
            want = [
                want[0] + coeff * facet.area * f[0],
                want[1] + coeff * facet.area * f[1],
                want[2] + coeff * facet.area * f[2],
            ];
        }
        assert_eq!(
            flipped, 1,
            "the double-sided array must be flipped by this wind"
        );
        assert!(
            oblique >= 3,
            "only {oblique} facets are off normal incidence"
        );
        assert_vectors_agree("Sentman box-wing", got, want);
    }

    fn one_facet(normal: [f64; 3], double_sided: bool) -> BoxWing<1> {
        BoxWing::<1> {
            facets: [Facet::fixed(
                normal,
                12.0,
                double_sided,
                2.2,
                0.15,
                0.3,
                0.4,
            )],
            mass: 640.0,
            drag_factor: 1.0,
            radiation_factor: 1.0,
        }
    }

    /// A double-sided facet intercepts flux from either side, so writing its normal the other way
    /// round must change nothing — that is the whole purpose of the flip in each facet loop. Exactly
    /// one of the two spellings takes the branch, so the pair pins the negation itself and not just
    /// its existence. A single-sided facet is the opposite: one spelling is windward, the other is
    /// shadowed and contributes nothing. No Orekit fixture flips a facet. (G4)
    #[test]
    fn a_double_sided_facet_does_not_care_which_way_its_normal_is_written() {
        let n = normalize([0.4, -0.7, 0.55]);
        let flip = scaled(-1.0, n);
        let v_rel = [-6100.0, 3200.0, -1500.0];
        let flux = [1.1e-6, -3.0e-6, 0.7e-6];
        let sun_dir = scaled(1.0 / norm([0.3, -0.9, 0.4]), [0.3, -0.9, 0.4]);

        let drag = one_facet(n, true).drag_acceleration(RHO, &v_rel, &sun_dir);
        let srp = one_facet(n, true).srp_acceleration(&flux, &sun_dir);
        assert_vectors_agree(
            "double-sided drag",
            one_facet(flip, true).drag_acceleration(RHO, &v_rel, &sun_dir),
            drag,
        );
        assert_vectors_agree(
            "double-sided SRP",
            one_facet(flip, true).srp_acceleration(&flux, &sun_dir),
            srp,
        );

        for (kind, windward, shadowed) in [
            (
                "drag",
                one_facet(n, false).drag_acceleration(RHO, &v_rel, &sun_dir),
                one_facet(flip, false).drag_acceleration(RHO, &v_rel, &sun_dir),
            ),
            (
                "SRP",
                one_facet(flip, false).srp_acceleration(&flux, &sun_dir),
                one_facet(n, false).srp_acceleration(&flux, &sun_dir),
            ),
        ] {
            assert!(
                norm(windward) > 0.0,
                "single-sided {kind}: the lit side contributes nothing"
            );
            assert_eq!(
                shadowed, [0.0; 3],
                "single-sided {kind}: the shadowed side contributes"
            );
        }
    }

    /// Orekit returns zero when the flux is null, gated on `|flux|² < Precision.SAFE_MIN` — the
    /// smallest positive normal `f64`. Without that early return `1/|flux|` is infinite and the sum
    /// is NaN, so a zero flux must give exactly zero; and the comparison is strict, so `|flux|²`
    /// exactly *at* the threshold must still be evaluated. `2⁻⁵¹¹` squares to `2⁻¹⁰²²` with no
    /// rounding, which puts the boundary within reach rather than near it.
    #[test]
    fn a_null_radiation_flux_returns_zero_and_the_threshold_is_strict() {
        let bw = box_only();
        let sun_dir = scaled(1.0 / norm([0.3, -0.9, 0.4]), [0.3, -0.9, 0.4]);
        assert_eq!(bw.srp_acceleration(&[0.0; 3], &sun_dir), [0.0; 3]);

        let edge = [libm::sqrt(SAFE_MIN), 0.0, 0.0];
        assert_eq!(
            edge[0] * edge[0],
            SAFE_MIN,
            "the boundary must be reached exactly"
        );
        let a = bw.srp_acceleration(&edge, &sun_dir);
        assert!(
            a.iter().any(|c| *c != 0.0),
            "|flux|² = SAFE_MIN is not below SAFE_MIN, so the facet sum must run: {a:?}"
        );
    }

    struct FixedBody([f64; 3]);
    impl Ephemeris for FixedBody {
        fn position(&self, _t: f64) -> [f64; 3] {
            self.0
        }
    }
    struct ConstDensity(f64);
    impl Atmosphere for ConstDensity {
        fn density<S: Scalar>(&self, _t: f64, _r: &[S; 3]) -> S {
            S::from_f64(self.0)
        }
    }

    /// The [`PanelDrag`] / [`PanelSrp`] wrappers wire a pluggable model into the [`Acceleration`]
    /// composition. With a [`Cannonball`] model, identity attitude and a static atmosphere, `PanelDrag`
    /// must reproduce the closed-form cannonball drag; `PanelSrp` the closed-form cannonball SRP. This
    /// is the "plugs into the existing forces" path.
    #[test]
    fn wrappers_wire_a_pluggable_model() {
        let c = Cannonball {
            mass: 800.0,
            drag_coeff: 2.3,
            area: 5.0,
            cr: 1.4,
        };
        let identity = |_t: f64| [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let sun = FixedBody([1.4e11, 0.0, 0.0]); // far off +X, satellite lit
        let r = [7.0e6, 0.0, 0.0];
        let v = [0.0, 7500.0, 0.0];

        // Drag: static atmosphere (ω = 0 ⇒ v_atm = 0 ⇒ v_rel = −v), so a = −½(C_d A/m)ρ|v|v.
        let pd = PanelDrag {
            model: c,
            atmosphere: ConstDensity(RHO),
            earth_rotation_rate: 0.0,
            wind: NoWind,
            attitude: identity,
            sun: FixedBody([1.4e11, 0.0, 0.0]),
        };
        let a = pd.acceleration(0.0, &r, &v);
        let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let k = 0.5 * c.drag_coeff * c.area / c.mass * RHO * speed;
        for i in 0..3 {
            assert!((a[i] + k * v[i]).abs() < 1e-18, "PanelDrag[{i}] {}", a[i]);
        }

        // SRP: flux = k_ref/|d|³·(sat − sun); a = (C_r A/m)·flux, pointing away from the Sun (−X here).
        let k_ref = 1.0e17;
        let ps = PanelSrp {
            model: c,
            k_ref,
            earth_radius: 6.378137e6,
            attitude: identity,
            sun,
        };
        let a = ps.acceleration(0.0, &r, &v);
        assert!(a[0] < 0.0, "SRP must push along −X (away from Sun): {a:?}");
        assert!(
            a[1].abs() < 1e-20 && a[2].abs() < 1e-20,
            "off-axis SRP leak: {a:?}"
        );
    }

    /// The box-wing drag monomorphises under [`spacelab_autodiff::Dual`] (the STM path is real, not
    /// f64-only) and its automatic-differentiation Jacobian `∂a/∂v_rel` matches central finite
    /// differences — the same AD that gives every other force its state-transition matrix.
    #[test]
    fn ad_jacobian_matches_finite_differences() {
        use spacelab_autodiff::Dual;

        let model = box_only();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let sun = [1.0, 0.0, 0.0];
        let v = [7400.0, 900.0, -1200.0];

        // Seed the three relative-velocity components as independent dual variables.
        let vd = [
            Dual::<3>::variable(v[0], 0),
            Dual::<3>::variable(v[1], 1),
            Dual::<3>::variable(v[2], 2),
        ];
        let sund = [
            Dual::<3>::constant(sun[0]),
            Dual::<3>::constant(sun[1]),
            Dual::<3>::constant(sun[2]),
        ];
        let a = model.drag_acceleration_inertial(Dual::<3>::constant(RHO), &vd, &sund, &identity);
        let a_f64 = model.drag_acceleration_inertial(RHO, &v, &sun, &identity);
        for i in 0..3 {
            assert!(
                (a[i].value() - a_f64[i]).abs() < 1e-20,
                "value drift on axis {i}"
            );
        }

        let h = 1.0; // 1 m/s about ~7.5 km/s — the active-facet set is stable across it
        for j in 0..3 {
            let mut vp = v;
            let mut vm = v;
            vp[j] += h;
            vm[j] -= h;
            let ap = model.drag_acceleration_inertial(RHO, &vp, &sun, &identity);
            let am = model.drag_acceleration_inertial(RHO, &vm, &sun, &identity);
            for i in 0..3 {
                let fd = (ap[i] - am[i]) / (2.0 * h);
                let ad = a[i].grad()[j];
                assert!(
                    (fd - ad).abs() <= 1e-6 * ad.abs() + 1e-18,
                    "∂a[{i}]/∂v[{j}]: AD {ad:.6e} vs FD {fd:.6e}"
                );
            }
        }
    }
}
