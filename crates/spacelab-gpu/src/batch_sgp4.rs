//! A near-Earth SGP4 reimplementation whose per-satellite constants can be uploaded to the GPU.
//!
//! The `sgp4` crate keeps its initialized `Constants` fields `pub(crate)`, so a GPU batch SGP4 cannot
//! reuse them — it needs its own init. This is a faithful port of `sgp4` 2.4's near-Earth arithmetic
//! on WGS-72, the document's geopotential (Vallado et al. 2006), refusing where the document refuses
//! (codes 1, 4 and 6). It is **near-Earth only**: deep-space element sets (Brouwer mean motion
//! ≤ 2π/225 min⁻¹, i.e. period ≥ 225 min) are rejected — the branchy resonance and luni-solar
//! machinery does not vectorise. The CPU path here is the correctness oracle for the GPU kernel and
//! is held to the document's printed output, `tcppver.out`.

use bytemuck::{Pod, Zeroable};

// WGS-72: the element sets are fitted with it, and `Satellite`'s reference convention uses it.
const AE: f64 = sgp4::WGS72.ae;
const KE: f64 = sgp4::WGS72.ke;
const J2: f64 = sgp4::WGS72.j2;
const J3: f64 = sgp4::WGS72.j3;
const J4: f64 = sgp4::WGS72.j4;
const PI: f64 = core::f64::consts::PI;

/// The per-satellite SGP4 constants (f64), initialised once from an element set. The near-Earth SGP4
/// propagation is stateless in these — a prediction at any minute-offset is a pure function of them.
#[derive(Clone, Copy, Debug)]
pub struct Sgp4Sat {
    inclination: f64,
    right_ascension_0: f64,
    eccentricity_0: f64,
    argument_of_perigee_0: f64,
    mean_anomaly_0: f64,
    mean_motion_0: f64,
    right_ascension_dot: f64,
    argument_of_perigee_dot: f64,
    mean_anomaly_dot: f64,
    c1: f64,
    c4: f64,
    k0: f64,
    k1: f64,
    a0: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    k6: f64,
    high_altitude: bool,
    c5: f64,
    d2: f64,
    d3: f64,
    d4: f64,
    eta: f64,
    k7: f64,
    k8: f64,
    k9: f64,
    k10: f64,
    elliptic: bool,
    k11: f64,
    k12: f64,
    k13: f64,
}

impl Sgp4Sat {
    /// Initialise from a parsed element set, or `None` if it is deep-space or out of range. Mirrors
    /// `sgp4::Constants::from_elements` (un-Kozai → `Constants::new` → `near_earth::constants`).
    #[must_use]
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    pub fn from_elements(el: &sgp4::Elements) -> Option<Self> {
        let inc = el.inclination * (PI / 180.0);
        let raan = el.right_ascension * (PI / 180.0);
        let ecc = el.eccentricity;
        let aop = el.argument_of_perigee * (PI / 180.0);
        let ma = el.mean_anomaly * (PI / 180.0);
        let kozai_n = el.mean_motion * (PI / 720.0); // rev/day → rad/min
        let bstar = el.drag_term;
        if !(0.0..1.0).contains(&ecc) || kozai_n <= 0.0 {
            return None;
        }

        // Un-Kozai: TLE Kozai mean motion → Brouwer n₀".
        let a1 = (KE / kozai_n).powf(2.0 / 3.0);
        let p0 = 0.75 * J2 * (3.0 * inc.cos().powi(2) - 1.0) / (1.0 - ecc.powi(2)).powf(1.5);
        let d1 = p0 / a1.powi(2);
        let d0 =
            p0 / (a1 * (1.0 - d1.powi(2) - d1 * (1.0 / 3.0 + 134.0 * d1.powi(2) / 81.0))).powi(2);
        let n0 = kozai_n / (1.0 + d0);
        // Near-Earth only (Brouwer mean motion above the deep-space threshold).
        if n0 <= 2.0 * PI / 225.0 {
            return None;
        }

        // Constants::new.
        let p1 = inc.cos();
        let p2 = 1.0 - ecc.powi(2);
        let k6 = 3.0 * p1.powi(2) - 1.0;
        let a0 = (KE / n0).powf(2.0 / 3.0);
        let p3 = a0 * (1.0 - ecc);
        let p4 = AE * (p3 - 1.0);
        let p5 = if p4 < 98.0 {
            20.0
        } else if p4 < 156.0 {
            p4 - 78.0
        } else {
            78.0
        };
        let s = p5 / AE + 1.0;
        let p6 = ((120.0 - p5) / AE).powi(4);
        let xi = 1.0 / (a0 - s);
        let p7 = p6 * xi.powi(4);
        let eta = a0 * ecc * xi;
        let p8 = (1.0 - eta.powi(2)).abs();
        let p9 = p7 / p8.powf(3.5);
        let c1 = bstar
            * (p9
                * n0
                * (a0 * (1.0 + 1.5 * eta.powi(2) + ecc * eta * (4.0 + eta.powi(2)))
                    + 0.375 * J2 * xi / p8 * k6 * (8.0 + 3.0 * eta.powi(2) * (8.0 + eta.powi(2)))));
        let p10 = 1.0 / (a0 * p2).powi(2);
        let b0 = p2.sqrt();
        let p11 = 1.5 * J2 * p10 * n0;
        let p12 = 0.5 * p11 * J2 * p10;
        let p13 = -0.46875 * J4 * p10.powi(2) * n0;
        let p14 = -p11 * p1
            + (0.5 * p12 * (4.0 - 19.0 * p1.powi(2)) + 2.0 * p13 * (3.0 - 7.0 * p1.powi(2))) * p1;
        let k14 = -0.5 * p11 * (1.0 - 5.0 * p1.powi(2))
            + 0.0625 * p12 * (7.0 - 114.0 * p1.powi(2) + 395.0 * p1.powi(4))
            + p13 * (3.0 - 36.0 * p1.powi(2) + 49.0 * p1.powi(4));
        let p15 = n0
            + 0.5 * p11 * b0 * k6
            + 0.0625 * p12 * b0 * (13.0 - 78.0 * p1.powi(2) + 137.0 * p1.powi(4));
        let c4 = bstar
            * (2.0
                * n0
                * p9
                * a0
                * p2
                * (eta * (2.0 + 0.5 * eta.powi(2)) + ecc * (0.5 + 2.0 * eta.powi(2))
                    - J2 * xi / (a0 * p8)
                        * (-3.0
                            * k6
                            * (1.0 - 2.0 * ecc * eta + eta.powi(2) * (1.5 - 0.5 * ecc * eta))
                            + 0.75
                                * (1.0 - p1.powi(2))
                                * (2.0 * eta.powi(2) - ecc * eta * (1.0 + eta.powi(2)))
                                * (2.0 * aop).cos())));
        let k0 = 3.5 * p2 * (-p11 * p1) * c1;
        let k1 = 1.5 * c1;

        // near_earth::constants.
        let k2 = -0.5 * (J3 / J2) * inc.sin();
        let k3 = 1.0 - p1.powi(2);
        let k4 = 7.0 * p1.powi(2) - 1.0;
        let k5 = if (1.0 + p1).abs() > 1.5e-12 {
            -0.25 * (J3 / J2) * inc.sin() * (3.0 + 5.0 * p1) / (1.0 + p1)
        } else {
            -0.25 * (J3 / J2) * inc.sin() * (3.0 + 5.0 * p1) / 1.5e-12
        };
        let high_altitude = p3 >= 220.0 / AE + 1.0;
        let mut sat = Self {
            inclination: inc,
            right_ascension_0: raan,
            eccentricity_0: ecc,
            argument_of_perigee_0: aop,
            mean_anomaly_0: ma,
            mean_motion_0: n0,
            right_ascension_dot: p14,
            argument_of_perigee_dot: k14,
            mean_anomaly_dot: p15,
            c1,
            c4,
            k0,
            k1,
            a0,
            k2,
            k3,
            k4,
            k5,
            k6,
            high_altitude,
            c5: 0.0,
            d2: 0.0,
            d3: 0.0,
            d4: 0.0,
            eta,
            k7: 0.0,
            k8: 0.0,
            k9: 0.0,
            k10: 0.0,
            elliptic: false,
            k11: 0.0,
            k12: 0.0,
            k13: 0.0,
        };
        if high_altitude {
            let d2 = 4.0 * a0 * xi * c1.powi(2);
            let p16 = d2 * xi * c1 / 3.0;
            sat.d2 = d2;
            sat.d3 = (17.0 * a0 + s) * p16;
            sat.d4 = 0.5 * p16 * a0 * xi * (221.0 * a0 + 31.0 * s) * c1;
            sat.c5 = bstar
                * (2.0
                    * p9
                    * a0
                    * p2
                    * (1.0 + 2.75 * (eta.powi(2) + eta * ecc) + eta * ecc * eta.powi(2)));
            sat.k7 = ma.sin();
            sat.k8 = d2 + 2.0 * c1.powi(2);
            sat.k9 = 0.25 * (3.0 * sat.d3 + c1 * (12.0 * d2 + 10.0 * c1.powi(2)));
            sat.k10 = 0.2
                * (3.0 * sat.d4
                    + 12.0 * c1 * sat.d3
                    + 6.0 * d2.powi(2)
                    + 15.0 * c1.powi(2) * (2.0 * d2 + c1.powi(2)));
            if ecc > 1.0e-4 {
                sat.elliptic = true;
                sat.k11 = (1.0 + eta * ma.cos()).powi(3);
                sat.k12 = bstar * (-2.0 * p7 * xi * (J3 / J2) * n0 * inc.sin() / ecc) * aop.cos();
                sat.k13 = -2.0 / 3.0 * p7 * bstar / (ecc * eta);
            }
        }
        Some(sat)
    }

    /// TEME position (km) and velocity (km/s) at `t` minutes since epoch, or `None` where the
    /// document refuses. Mirrors `sgp4::Constants::propagate_from_state` (near-Earth branch).
    #[must_use]
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    pub fn propagate(&self, t: f64) -> Option<([f64; 3], [f64; 3])> {
        let p22 = self.right_ascension_0 + self.right_ascension_dot * t + self.k0 * t.powi(2);
        let p23 = self.argument_of_perigee_0 + self.argument_of_perigee_dot * t;
        let p24 = self.mean_anomaly_0 + self.mean_anomaly_dot * t;
        let (argument_of_perigee, mean_anomaly, a, p27) = if self.high_altitude {
            let (argument_of_perigee, p26) = if self.elliptic {
                let p25 =
                    self.k13 * ((1.0 + self.eta * p24.cos()).powi(3) - self.k11) + self.k12 * t;
                (p23 - p25, p24 + p25)
            } else {
                (p23, p24)
            };
            (
                argument_of_perigee,
                p26 + self.mean_motion_0
                    * (self.k1 * t.powi(2)
                        + self.k8 * t.powi(3)
                        + t.powi(4) * (self.k9 + t * self.k10)),
                self.a0
                    * (1.0
                        - self.c1 * t
                        - self.d2 * t.powi(2)
                        - self.d3 * t.powi(3)
                        - self.d4 * t.powi(4))
                    .powi(2),
                self.eccentricity_0 - (self.c4 * t + self.c5 * (p26.sin() - self.k7)),
            )
        } else {
            (
                p23,
                p24 + self.mean_motion_0 * self.k1 * t.powi(2),
                self.a0 * (1.0 - self.c1 * t).powi(2),
                self.eccentricity_0 - self.c4 * t,
            )
        };
        if !(-0.001..1.0).contains(&p27) {
            return None;
        }
        let eccentricity = p27.max(1.0e-6);
        let mean_motion = KE / a.powf(1.5);
        let inclination = self.inclination;

        let axn = eccentricity * argument_of_perigee.cos();
        let p37 = 1.0 / (a * (1.0 - eccentricity.powi(2)));
        let ayn = eccentricity * argument_of_perigee.sin() + p37 * self.k2;
        let p38 = (mean_anomaly + argument_of_perigee + p37 * self.k5 * axn) % (2.0 * PI);
        let mut ew = p38;
        for _ in 0..10 {
            let delta = (p38 - ayn * ew.cos() + axn * ew.sin() - ew)
                / (1.0 - ew.cos() * axn - ew.sin() * ayn);
            if delta.abs() < 1.0e-12 {
                break;
            }
            ew += delta.clamp(-0.95, 0.95);
        }
        let p39 = axn.powi(2) + ayn.powi(2);
        let pl = a * (1.0 - p39);
        if pl < 0.0 {
            return None;
        }
        let p40 = axn * ew.sin() - ayn * ew.cos();
        let r = a * (1.0 - (axn * ew.cos() + ayn * ew.sin()));
        let r_dot = a.sqrt() * p40 / r;
        let b = (1.0 - p39).sqrt();
        let p41 = p40 / (1.0 + b);
        let p42 = a / r * (ew.sin() - ayn - axn * p41);
        let p43 = a / r * (ew.cos() - axn + ayn * p41);
        let u = p42.atan2(p43);
        let p44 = 2.0 * p43 * p42;
        let p45 = 1.0 - 2.0 * p42.powi(2);
        let p46 = 0.5 * J2 / pl / pl;
        let rk = r * (1.0 - 1.5 * p46 * b * self.k6) + 0.5 * (0.5 * J2 / pl) * self.k3 * p45;
        // The document's code 6: the radius has fallen below one Earth radius.
        if rk < 1.0 {
            return None;
        }
        let uk = u - 0.25 * p46 * self.k4 * p44;
        let inclination_k = inclination + 1.5 * p46 * inclination.cos() * inclination.sin() * p45;
        let right_ascension_k = p22 + 1.5 * p46 * inclination.cos() * p44;
        let rk_dot = r_dot - mean_motion * (0.5 * J2 / pl) * self.k3 * p44 / KE;
        let rfk_dot =
            pl.sqrt() / r + mean_motion * (0.5 * J2 / pl) * (self.k3 * p45 + 1.5 * self.k6) / KE;
        let u0 = -right_ascension_k.sin() * inclination_k.cos() * uk.sin()
            + right_ascension_k.cos() * uk.cos();
        let u1 = right_ascension_k.cos() * inclination_k.cos() * uk.sin()
            + right_ascension_k.sin() * uk.cos();
        let u2 = inclination_k.sin() * uk.sin();
        let f = AE * KE / 60.0;
        Some((
            [rk * u0 * AE, rk * u1 * AE, rk * u2 * AE],
            [
                (rk_dot * u0
                    + rfk_dot
                        * (-right_ascension_k.sin() * inclination_k.cos() * uk.cos()
                            - right_ascension_k.cos() * uk.sin()))
                    * f,
                (rk_dot * u1
                    + rfk_dot
                        * (right_ascension_k.cos() * inclination_k.cos() * uk.cos()
                            - right_ascension_k.sin() * uk.sin()))
                    * f,
                (rk_dot * u2 + rfk_dot * (inclination_k.sin() * uk.cos())) * f,
            ],
        ))
    }
}

/// The [`Sgp4Sat`] constants flattened to `f32` for GPU upload. Field order and the 3-word tail
/// padding match the WGSL `Sgp4` struct byte-for-byte (all scalars, so a 144-byte, 4-aligned stride).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Sgp4Init {
    inclination: f32,
    right_ascension_0: f32,
    eccentricity_0: f32,
    argument_of_perigee_0: f32,
    mean_anomaly_0: f32,
    mean_motion_0: f32,
    right_ascension_dot: f32,
    argument_of_perigee_dot: f32,
    mean_anomaly_dot: f32,
    c1: f32,
    c4: f32,
    k0: f32,
    k1: f32,
    a0: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    k5: f32,
    k6: f32,
    eta: f32,
    c5: f32,
    d2: f32,
    d3: f32,
    d4: f32,
    k7: f32,
    k8: f32,
    k9: f32,
    k10: f32,
    k11: f32,
    k12: f32,
    k13: f32,
    high_altitude: u32,
    elliptic: u32,
    _pad: [u32; 3],
}

impl Sgp4Sat {
    /// Flatten to the `f32` GPU upload struct.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_gpu(self) -> Sgp4Init {
        Sgp4Init {
            inclination: self.inclination as f32,
            right_ascension_0: self.right_ascension_0 as f32,
            eccentricity_0: self.eccentricity_0 as f32,
            argument_of_perigee_0: self.argument_of_perigee_0 as f32,
            mean_anomaly_0: self.mean_anomaly_0 as f32,
            mean_motion_0: self.mean_motion_0 as f32,
            right_ascension_dot: self.right_ascension_dot as f32,
            argument_of_perigee_dot: self.argument_of_perigee_dot as f32,
            mean_anomaly_dot: self.mean_anomaly_dot as f32,
            c1: self.c1 as f32,
            c4: self.c4 as f32,
            k0: self.k0 as f32,
            k1: self.k1 as f32,
            a0: self.a0 as f32,
            k2: self.k2 as f32,
            k3: self.k3 as f32,
            k4: self.k4 as f32,
            k5: self.k5 as f32,
            k6: self.k6 as f32,
            eta: self.eta as f32,
            c5: self.c5 as f32,
            d2: self.d2 as f32,
            d3: self.d3 as f32,
            d4: self.d4 as f32,
            k7: self.k7 as f32,
            k8: self.k8 as f32,
            k9: self.k9 as f32,
            k10: self.k10 as f32,
            k11: self.k11 as f32,
            k12: self.k12 as f32,
            k13: self.k13 as f32,
            high_altitude: u32::from(self.high_altitude),
            elliptic: u32::from(self.elliptic),
            _pad: [0; 3],
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    // Near-Earth verification element sets (the ISS and a few LEO objects), NORAD-published TLEs.
    const TLES: &[(&str, &str)] = &[
        (
            "1 25544U 98067A   20194.88612269 -.00002218  00000-0 -31515-4 0  9992",
            "2 25544  51.6461 221.2784 0001413  89.1723 280.4612 15.49507896236008",
        ),
        (
            "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
            "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
        ),
        (
            "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985",
            "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774",
        ),
    ];

    type State = ([f64; 3], [f64; 3]);

    /// The `sgp4` crate on WGS-72, as `Satellite`'s reference convention builds it, with the
    /// document's code 6 applied: the crate answers below one Earth radius, the document does not.
    fn crate_reference(el: &sgp4::Elements) -> impl Fn(f64) -> Option<State> {
        let constants = sgp4::Constants::from_elements_afspc_compatibility_mode(el).unwrap();
        move |t| {
            let p = constants.propagate(sgp4::MinutesSinceEpoch(t)).ok()?;
            let r = p.position.iter().map(|x| x * x).sum::<f64>().sqrt();
            (r >= sgp4::WGS72.ae).then_some((p.position, p.velocity))
        }
    }

    #[test]
    fn init_and_propagate_match_the_sgp4_crate() {
        let mut checked = 0;
        for (l1, l2) in TLES {
            let elements = sgp4::Elements::from_tle(None, l1.as_bytes(), l2.as_bytes()).unwrap();
            let reference = crate_reference(&elements);
            let Some(sat) = Sgp4Sat::from_elements(&elements) else {
                continue; // deep-space, skipped
            };
            for &t in &[0.0, 10.0, 100.0, 720.0, 1440.0, -360.0] {
                match (sat.propagate(t), reference(t)) {
                    (Some((r, v)), Some((position, velocity))) => {
                        let dr = ((r[0] - position[0]).powi(2)
                            + (r[1] - position[1]).powi(2)
                            + (r[2] - position[2]).powi(2))
                        .sqrt();
                        let dv = ((v[0] - velocity[0]).powi(2)
                            + (v[1] - velocity[1]).powi(2)
                            + (v[2] - velocity[2]).powi(2))
                        .sqrt();
                        assert!(
                            dr < 1e-6 && dv < 1e-9,
                            "t={t}: position off by {dr:.3e} km, velocity by {dv:.3e} km/s vs the sgp4 crate"
                        );
                        checked += 1;
                    }
                    (None, None) => {}
                    (m, r) => panic!("t={t}: solvability disagreement (mine {m:?}, crate {r:?})"),
                }
            }
        }
        assert!(
            checked >= 6,
            "expected several near-Earth comparisons, got {checked}"
        );
    }

    /// The `k5` singularity guard, which the official verification set never reaches.
    ///
    /// `k5` divides by `1 + cos i`, so the AFSPC code substitutes `1.5e-12` for the denominator
    /// when the inclination is within that of 180 degrees. Nothing in `SGP4-VER.TLE` is more
    /// inclined than 98.4 degrees, so neither the guard nor the branch behind it was ever executed:
    /// eleven mutations of that line survived the whole suite. These two element sets are the ISS
    /// set with its inclination replaced by exactly 180 and exactly 0 degrees (checksums recomputed)
    /// — one on each side of the guard, since `1 + cos i` vanishes at 180 and `1 - cos i` at 0.
    /// Compared against the `sgp4` crate, which carries the same guard, bit for bit.
    #[allow(clippy::float_cmp)] // the two implementations agree exactly, and that is the assertion
    #[test]
    fn the_inclination_singularity_agrees_with_the_sgp4_crate() {
        const LINE1: &str = "1 25544U 98067A   20194.88612269 -.00002218  00000-0 -31515-4 0  9992";
        let cases = [
            "2 25544 180.0000 221.2784 0001413  89.1723 280.4612 15.49507896236004",
            "2 25544   0.0000 221.2784 0001413  89.1723 280.4612 15.49507896236005",
        ];
        for line2 in cases {
            let el = sgp4::Elements::from_tle(None, LINE1.as_bytes(), line2.as_bytes()).unwrap();
            let Some(sat) = Sgp4Sat::from_elements(&el) else {
                panic!(
                    "near-Earth element set refused at inclination {}",
                    el.inclination
                )
            };
            let reference = crate_reference(&el);
            let mut compared = 0u32;
            for step in 0..=48 {
                let t = f64::from(step) * 30.0;
                match (sat.propagate(t), reference(t)) {
                    (Some((r, v)), Some((position, velocity))) => {
                        assert_eq!(r, position, "inclination {}, t = {t}", el.inclination);
                        assert_eq!(v, velocity, "inclination {}, t = {t}", el.inclination);
                        compared += 1;
                    }
                    (None, None) => {}
                    (m, r) => panic!("solvability disagreement at t = {t}: {m:?} vs {r:?}"),
                }
            }
            assert!(compared > 40, "only {compared} states compared");
        }
    }

    /// The two branch boundaries a four-decimal TLE field cannot land on.
    ///
    /// Both switch on a quantity the element-set format quantises too coarsely to reach, so the
    /// element sets here are built directly rather than parsed. The oracle is the `sgp4` crate,
    /// which draws both boundaries in the same places.
    #[allow(clippy::float_cmp)] // the two implementations agree exactly, and that is the assertion
    #[test]
    fn the_branch_boundaries_a_tle_cannot_express_agree_with_the_sgp4_crate() {
        const LINE1: &str = "1 25544U 98067A   20194.88612269 -.00002218  00000-0 -31515-4 0  9992";
        const LINE2: &str = "2 25544  51.6461 221.2784 0001413  89.1723 280.4612 15.49507896236008";
        let good = sgp4::Elements::from_tle(None, LINE1.as_bytes(), LINE2.as_bytes()).unwrap();
        // The guard band around a retrograde-polar orbit is 1.5e-12 wide in `1 + cos i`, which a
        // TLE's four decimal places of inclination cannot land inside: 179.9999 degrees gives
        // 1.52e-12 and 180.0000 gives ~0, where `sin i` is ~1e-16 and `k5` underflows to nothing.
        // Built directly, an inclination one part in 1e5 of a degree short of 180 puts `k5` at
        // order one and the branch's arithmetic where a test can see it.
        let edge = sgp4::Elements {
            inclination: 180.0 - 9.6e-5,
            ..good.clone()
        };
        let Some(sat) = Sgp4Sat::from_elements(&edge) else {
            panic!("the guard-band element set was refused")
        };
        let reference = crate_reference(&edge);
        for step in 0..=24 {
            let t = f64::from(step) * 60.0;
            if let (Some((r, v)), Some((position, velocity))) = (sat.propagate(t), reference(t)) {
                assert_eq!(r, position, "guard band, t = {t}");
                assert_eq!(v, velocity, "guard band, t = {t}");
            }
        }

        // The short-period corrections switch on at an eccentricity of exactly 1e-4, a tie a
        // four-decimal TLE field cannot express either.
        for eccentricity in [1.0e-4, 1.0e-4 - f64::EPSILON * 1e-4] {
            let el = sgp4::Elements {
                eccentricity,
                ..good.clone()
            };
            let Some(sat) = Sgp4Sat::from_elements(&el) else {
                panic!("element set at eccentricity {eccentricity} was refused")
            };
            if let (Some((r, _)), Some((position, _))) =
                (sat.propagate(90.0), crate_reference(&el)(90.0))
            {
                assert_eq!(r, position, "eccentricity {eccentricity}");
            }
        }
    }

    /// Element sets the near-Earth model cannot represent are refused, not propagated.
    ///
    /// The two conditions are independent — an unbounded eccentricity and a non-positive mean
    /// motion each make the un-Kozai step meaningless on their own — so the guard is a disjunction,
    /// and a conjunction would let either through.
    #[test]
    fn from_elements_refuses_an_element_set_it_cannot_model() {
        const LINE1: &str = "1 25544U 98067A   20194.88612269 -.00002218  00000-0 -31515-4 0  9992";
        const LINE2: &str = "2 25544  51.6461 221.2784 0001413  89.1723 280.4612 15.49507896236008";
        let good = sgp4::Elements::from_tle(None, LINE1.as_bytes(), LINE2.as_bytes()).unwrap();
        assert!(
            Sgp4Sat::from_elements(&good).is_some(),
            "the baseline set was refused"
        );
        for (eccentricity, mean_motion, what) in [
            (1.0, good.mean_motion, "unit eccentricity"),
            (1.5, good.mean_motion, "hyperbolic eccentricity"),
            (-0.1, good.mean_motion, "negative eccentricity"),
            (good.eccentricity, 0.0, "zero mean motion"),
            (good.eccentricity, -1.0, "negative mean motion"),
        ] {
            let el = sgp4::Elements {
                eccentricity,
                mean_motion,
                ..good.clone()
            };
            assert!(Sgp4Sat::from_elements(&el).is_none(), "{what} was accepted");
        }
    }

    /// The `pl < 0` bail-out — the osculating semi-latus rectum going negative, after which the
    /// `sqrt(1 − axn² − ayn²)` below it is imaginary — is reachable from the published sets, but
    /// only well outside the grids they publish. The comparison above spans those grids and never
    /// reaches it; six of the nine near-Earth sets do, the nearest at 26 hours before its own
    /// epoch. Which epochs those are is the assertion: the `sgp4` crate refuses exactly the same
    /// ones, so this is a differential check on the guard rather than a restatement of it.
    #[test]
    fn the_epochs_with_no_solution_are_the_ones_the_sgp4_crate_also_refuses() {
        const CASES: &str = include_str!("../../spacelab-sgp4/tests/fixtures/sgp4_ver_tle.txt");
        let (mut checked, mut refused) = (0_u32, 0_u32);
        for line in CASES
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let f: Vec<&str> = line.split('|').collect();
            let Ok(el) = sgp4::Elements::from_tle(None, f[0].as_bytes(), f[1].as_bytes()) else {
                continue;
            };
            let Some(sat) = Sgp4Sat::from_elements(&el) else {
                continue;
            };
            let reference = crate_reference(&el);
            for k in -600..=600_i32 {
                let t = f64::from(k) * 1440.0; // days out, either side of epoch
                let ours = sat.propagate(t).is_none();
                let theirs = reference(t).is_none();
                assert_eq!(ours, theirs, "{} at t = {t} min", el.norad_id);
                checked += 1;
                refused += u32::from(ours);
            }
        }
        assert_eq!(checked, 10_809, "epochs compared");
        assert!(refused > 0, "no epoch exercised the no-solution branch");
    }

    /// The official AIAA 2006-6753 verification element sets, on their published time grids.
    ///
    /// The three TLEs above are three ordinary LEO objects, and every branch in `from_elements`
    /// that discriminates decides the same way for all of them: each perigee is far above the
    /// 156 km rung of the `s` ladder, each Brouwer mean motion far above the deep-space cutoff,
    /// each inclination far from 180°, each `high_altitude`. A mutation of any of those
    /// discriminants cannot change the answer, and 100 mutants of this file survive that fixture.
    ///
    /// The verification set spans the regimes on purpose — 9 near-Earth objects from 87 to 133
    /// minutes including two retrograde and Vallado's decaying 88888, and 20 deep-space objects
    /// the port must refuse rather than mis-propagate.
    ///
    /// The comparison is exact, because that is what it measures: over all 157 answered states of
    /// the published grids the port and the crate land on the same doubles, this being a
    /// transcription of the same arithmetic in the same order rather than an independent
    /// derivation. Any drift at all is therefore a real divergence, not accumulated rounding.
    /// At the document's three code-6 states, 28872 at 55 and 60 min and 29141 at 440, both refuse,
    /// the crate through `crate_reference`.
    #[test]
    fn the_official_verification_sets_agree_with_the_sgp4_crate() {
        const CASES: &str = include_str!("../../spacelab-sgp4/tests/fixtures/sgp4_ver_tle.txt");
        let (mut near_earth, mut deep_space, mut compared, mut unparsed) = (0, 0, 0, 0);
        let (mut worst_r, mut worst_v) = (0.0_f64, 0.0_f64);

        for line in CASES
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let f: Vec<&str> = line.split('|').collect();
            // A few of the published sets carry deliberately dirty header fields that the `sgp4`
            // crate's strict parser refuses; they cannot be compared against a reference that
            // will not read them.
            let Ok(el) = sgp4::Elements::from_tle(None, f[0].as_bytes(), f[1].as_bytes()) else {
                unparsed += 1;
                continue;
            };
            // Deep space is a Brouwer period of 225 minutes or more. The un-Kozai correction is
            // parts in 1e4 and no set here sits near the cut — 133 minutes and 295 minutes are the
            // two nearest — so the TLE's own Kozai period decides the expected verdict.
            let id = el.norad_id;
            let deep = 1440.0 / el.mean_motion >= 225.0;

            let Some(sat) = Sgp4Sat::from_elements(&el) else {
                assert!(
                    deep,
                    "{id} refused, but its period is {:.1} min",
                    1440.0 / el.mean_motion
                );
                deep_space += 1;
                continue;
            };
            assert!(
                !deep,
                "{id} accepted, but its period is {:.1} min",
                1440.0 / el.mean_motion
            );
            near_earth += 1;

            let reference = crate_reference(&el);
            let (first, last, step): (f64, f64, f64) = (
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
                f[4].parse().unwrap(),
            );
            let mut t = first;
            while t <= last {
                match (sat.propagate(t), reference(t)) {
                    (Some((r, v)), Some((position, velocity))) => {
                        let d = |a: [f64; 3], b: [f64; 3]| {
                            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                                .sqrt()
                        };
                        worst_r = spacelab_testkit::nan_max(worst_r, d(r, position));
                        worst_v = spacelab_testkit::nan_max(worst_v, d(v, velocity));
                        compared += 1;
                    }
                    (None, None) => {}
                    (m, r) => panic!("{id} at t={t}: solvability disagreement ({m:?} vs {r:?})"),
                }
                t += step;
            }
        }

        assert_eq!(near_earth, 9, "near-Earth sets in the verification file");
        assert_eq!(deep_space, 20, "deep-space sets in the verification file");
        assert_eq!(unparsed, 4, "sets the sgp4 crate refuses to read");
        assert_eq!(compared, 157, "state comparisons over the published grids");
        assert!(
            worst_r <= 0.0 && worst_v <= 0.0,
            "worst divergence {worst_r:.3e} km, {worst_v:.3e} km/s vs the sgp4 crate"
        );
    }

    /// Every refusal the document's model makes on the near-Earth sets every five minutes for a
    /// day, past where its driver stops, as runs: set (row of `sgp4_ver_tle.txt`, from 0), first
    /// minute, last minute, class (1: codes 1–4, 6: code 6). Derived by
    /// `derivations/sgp4_document.mac` (5), which reads it back. The model is a function of t
    /// alone: after its first refusal 28872 (set 25) dips below one Earth radius once a revolution
    /// and answers between, and 29141 (set 26) answers at 1405 min.
    #[rustfmt::skip]
    const DAY_REFUSALS: [f64; 84] = [
        11.0, 490.0, 1440.0, 1.0,
        25.0, 55.0, 65.0, 6.0,
        25.0, 140.0, 155.0, 6.0,
        25.0, 230.0, 240.0, 6.0,
        25.0, 315.0, 330.0, 6.0,
        25.0, 400.0, 415.0, 6.0,
        25.0, 490.0, 500.0, 6.0,
        25.0, 575.0, 590.0, 6.0,
        25.0, 660.0, 675.0, 6.0,
        25.0, 750.0, 760.0, 6.0,
        25.0, 835.0, 845.0, 6.0,
        25.0, 920.0, 930.0, 6.0,
        25.0, 1005.0, 1020.0, 6.0,
        25.0, 1090.0, 1105.0, 6.0,
        25.0, 1175.0, 1190.0, 6.0,
        25.0, 1260.0, 1275.0, 6.0,
        25.0, 1345.0, 1360.0, 6.0,
        25.0, 1430.0, 1440.0, 6.0,
        26.0, 425.0, 1375.0, 6.0,
        26.0, 1380.0, 1400.0, 1.0,
        26.0, 1410.0, 1440.0, 6.0,
    ];

    /// The port refuses exactly where the document's model refuses, every five minutes for a day.
    #[test]
    fn the_port_refuses_exactly_where_the_document_refuses_through_a_day() {
        const CASES: &str = include_str!("../../spacelab-sgp4/tests/fixtures/sgp4_ver_tle.txt");
        let (mut sets, mut refused) = (0, 0);
        for (k, line) in CASES
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .enumerate()
        {
            let f: Vec<&str> = line.split('|').collect();
            let Ok(el) = sgp4::Elements::from_tle(None, f[0].as_bytes(), f[1].as_bytes()) else {
                continue;
            };
            let Some(sat) = Sgp4Sat::from_elements(&el) else {
                continue;
            };
            sets += 1;
            let set = f64::from(u32::try_from(k).unwrap());
            for minute in (0..=288_u16).map(|j| f64::from(j * 5)) {
                let document = DAY_REFUSALS
                    .chunks(4)
                    .any(|r| (r[0] - set).abs() < 0.5 && (r[1]..=r[2]).contains(&minute));
                assert_eq!(
                    sat.propagate(minute).is_none(),
                    document,
                    "set {k} ({}) at {minute} min",
                    el.norad_id
                );
                refused += usize::from(document);
            }
        }
        assert_eq!((sets, refused), (9, 455), "near-Earth sets and refusals");
    }

    /// (G5) The document's printed output, `tcppver.out` (AIAA 2006-6753, Appendix E), for every
    /// near-Earth set: each printed state to a millimetre and a micrometre per second, and each
    /// section ending where this port first refuses.
    ///
    /// The document's driver prints t = 0 and then each grid time until its first error. Every
    /// near-Earth refusal run in `derivations/sgp4_document.mac` reaches the end of its grid
    /// (22312 and 28350 code 1, 28872 and 29141 code 6), so refusing each grid time past the
    /// printed rows is the document's verdict as well.
    #[test]
    fn the_published_output_is_reproduced_and_ends_where_the_document_refuses() {
        const CASES: &str = include_str!("../../spacelab-sgp4/tests/fixtures/sgp4_ver_tle.txt");
        const PUBLISHED: &str = include_str!("../../spacelab-sgp4/tests/fixtures/tcppver.out");
        let mut sections: Vec<Vec<[f64; 7]>> = Vec::new();
        for line in PUBLISHED.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() == 2 && fields[1] == "xx" {
                sections.push(Vec::new());
            } else if fields.len() >= 7 {
                let mut row = [0.0; 7];
                for (x, f) in row.iter_mut().zip(&fields) {
                    *x = f.parse().unwrap();
                }
                sections.last_mut().unwrap().push(row);
            }
        }
        let sets: Vec<&str> = CASES
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .collect();
        assert_eq!(sections.len(), sets.len(), "one published section per set");

        let d = |a: [f64; 3], b: [f64; 3]| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        let (mut near_earth, mut printed, mut refused) = (0, 0, 0);
        let (mut worst_r, mut worst_v) = (0.0_f64, 0.0_f64);
        for (set, rows) in sets.iter().zip(&sections) {
            let f: Vec<&str> = set.split('|').collect();
            let Ok(el) = sgp4::Elements::from_tle(None, f[0].as_bytes(), f[1].as_bytes()) else {
                continue;
            };
            let Some(sat) = Sgp4Sat::from_elements(&el) else {
                continue;
            };
            near_earth += 1;
            let id = el.norad_id;

            // The driver's grid: t = 0, then the first time (one step back unless it is 0)
            // stepped forward in binary64, the last step clamped to the last time.
            let (first, last, step): (f64, f64, f64) = (
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
                f[4].parse().unwrap(),
            );
            let mut grid = vec![0.0];
            let mut t = if first.abs() > 1e-8 {
                first - step
            } else {
                first
            };
            while t < last {
                t = (t + step).min(last);
                grid.push(t);
            }
            assert!(rows.len() <= grid.len(), "{id}: more rows than grid times");

            for (row, &t) in rows.iter().zip(&grid) {
                assert!(
                    (row[0] - t).abs() < 1e-6,
                    "{id}: printed {} min, grid {t}",
                    row[0]
                );
                let Some((r, v)) = sat.propagate(t) else {
                    panic!("{id} refused at {t} min, which the document prints")
                };
                worst_r = spacelab_testkit::nan_max(worst_r, d(r, [row[1], row[2], row[3]]));
                worst_v = spacelab_testkit::nan_max(worst_v, d(v, [row[4], row[5], row[6]]));
                printed += 1;
            }
            for &t in &grid[rows.len()..] {
                assert!(
                    sat.propagate(t).is_none(),
                    "{id} answered at {t} min, past the document's first refusal"
                );
                refused += 1;
            }
        }

        println!(
            "tcppver.out, near-Earth: {printed} rows, worst {:.3} µm and {:.4} µm/s; {refused} refused",
            worst_r * 1e9,
            worst_v * 1e9
        );
        assert_eq!(near_earth, 9, "near-Earth sets in the document");
        assert_eq!(
            (printed, refused),
            (158, 64),
            "printed rows and refusals past them"
        );
        assert!(
            worst_r < 1e-6 && worst_v < 1e-9,
            "worst {worst_r:.3e} km, {worst_v:.3e} km/s against the printed output"
        );
    }
}
