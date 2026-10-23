//! Kinematically-regular constellations — a constellation as the orbit of one seed satellite under a
//! finite rotation group (Mozhaev's group-theoretic view of symmetric satellite networks).
//!
//! A set of satellites is *kinematically regular* if some group `H` of transformations maps it to
//! itself at every instant. The construction here takes a seed phase state `s₀` and a finite rotation
//! group `{R_k}`: satellite `k` is `R_k · s₀` (optionally advanced in mean anomaly). Because the
//! two-body flow commutes with rotations — the central field is isotropic, so `φ_t(R·s) = R·φ_t(s)` —
//! the set `{R_k · φ_τ(s₀)}` is carried to `{R_k · φ_{t+τ}(s₀)}` as time advances, i.e. it stays the
//! orbit of the group at *every* time. Regularity is therefore automatic, not a phasing constraint.
//!
//! Where this sits relative to the other families: the cyclic group `C_n` about the polar axis just
//! shifts right ascension, so [`Symmetry::CyclicPolar`] is a single-axis Walker (verified below). The
//! genuinely distinct cases are the **dihedral** `D_n` (a 180° equatorial flip added — a north/south
//! symmetry Walker lattices cannot impose) and the **polyhedral** groups
//! ([`Symmetry::Tetrahedral`]), whose orbit-plane symmetry no `T/P/F` lattice expresses.

use spacelab_core::{Angle, Mat3};
use spacelab_elements::KeplerianElements;

use crate::mozhaev::{MozhaevSymbol, PointGroup};

const TAU: f64 = core::f64::consts::TAU;
const PI: f64 = core::f64::consts::PI;

/// The three cyclic coordinate permutations (rotations by 0, 120°, 240° about the body diagonal
/// `(1,1,1)`) — the rotation part of the tetrahedral group, before signs.
const TETRA_PERMS: [Mat3; 3] = [
    Mat3::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    Mat3::from_rows([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    Mat3::from_rows([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
];

/// The four sign diagonals with an even number of minus signs (determinant `+1`) — combined with the
/// permutations they give the 12 proper rotations of the tetrahedral group `T ≅ A₄`.
const TETRA_SIGNS: [Mat3; 4] = [
    Mat3::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    Mat3::from_rows([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
    Mat3::from_rows([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
    Mat3::from_rows([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
];

/// The symmetry group whose orbit of the seed is the constellation.
#[derive(Clone, Copy, Debug)]
pub enum Symmetry {
    /// Cyclic `C_n` about the polar axis, satellite `k` staggered by `k·stagger` in mean anomaly.
    /// This only shifts right ascension, so it is a single-axis Walker of `order` planes.
    CyclicPolar {
        /// Group order `n` (number of satellites).
        order: u32,
        /// Per-step mean-anomaly stagger.
        stagger: Angle,
    },
    /// Dihedral `D_n`: the `C_n` copies plus their images under a 180° flip about the equatorial
    /// x-axis — `2n` satellites with a north/south symmetry the cyclic (Walker) families cannot
    /// impose.
    DihedralPolar {
        /// Cyclic order `n`; the constellation has `2n` satellites.
        order: u32,
        /// Per-step mean-anomaly stagger of the cyclic part.
        stagger: Angle,
    },
    /// The 12 proper rotations of the tetrahedral group applied to the seed's orbit plane — an
    /// orbit-plane symmetry no Walker/lattice `T/P/F` expresses.
    Tetrahedral,
    /// The seed's orbit under an arbitrary Mozhaev [`PointGroup`] (`Cₙ`, `Dₙ`, `T`, `O`, `Y`) — the
    /// generative counterpart of the classification: one satellite per group rotation, no stagger.
    Group(PointGroup),
}

/// A kinematically-regular constellation: the orbit of `seed` under `symmetry`.
#[derive(Clone, Copy, Debug)]
pub struct KinematicallyRegular {
    /// The generating satellite `s₀`.
    pub seed: KeplerianElements,
    /// The symmetry group.
    pub symmetry: Symmetry,
}

impl KinematicallyRegular {
    /// Number of satellites: `n` (cyclic), `2n` (dihedral) or `12` (tetrahedral).
    #[must_use]
    pub const fn total(&self) -> u32 {
        match self.symmetry {
            Symmetry::CyclicPolar { order, .. } => order,
            Symmetry::DihedralPolar { order, .. } => 2 * order,
            Symmetry::Tetrahedral => 12,
            Symmetry::Group(group) => group.order(),
        }
    }

    /// A kinematically-regular system named by a Mozhaev symbol: the seed's orbit under the symbol's
    /// spatial rotation component `H_R`.
    #[must_use]
    pub const fn from_symbol(seed: KeplerianElements, symbol: &MozhaevSymbol) -> Self {
        Self {
            seed,
            symmetry: Symmetry::Group(symbol.component),
        }
    }

    /// Apply rotation `rot` to the seed advanced by `mean_shift`, and emit the resulting orbit.
    fn emit_element<E: FnMut(KeplerianElements)>(&self, rot: Mat3, mean_shift: f64, emit: &mut E) {
        let m = Angle::from_radians(self.seed.mean_anomaly().radians() + mean_shift);
        let (r, v) = self.seed.at_mean_anomaly(m).to_state();
        emit(KeplerianElements::from_state(
            rot * r,
            rot * v,
            self.seed.mu,
        ));
    }

    /// Emit the constellation's elements — one per group element.
    pub fn generate<E: FnMut(KeplerianElements)>(&self, mut emit: E) {
        match self.symmetry {
            Symmetry::CyclicPolar { order, stagger } => {
                for k in 0..order {
                    // Active rotation about +z (frame rotation by the negative angle), so right
                    // ascension advances with k, as in a Walker.
                    let rot = Mat3::rot3(-f64::from(k) * TAU / f64::from(order));
                    self.emit_element(rot, f64::from(k) * stagger.radians(), &mut emit);
                }
            }
            Symmetry::DihedralPolar { order, stagger } => {
                let flip = Mat3::rot1(PI); // 180° about the equatorial x-axis
                for k in 0..order {
                    let rz = Mat3::rot3(-f64::from(k) * TAU / f64::from(order));
                    let shift = f64::from(k) * stagger.radians();
                    self.emit_element(rz, shift, &mut emit);
                    self.emit_element(rz * flip, shift, &mut emit);
                }
            }
            Symmetry::Tetrahedral => {
                for perm in TETRA_PERMS {
                    for sign in TETRA_SIGNS {
                        self.emit_element(perm * sign, 0.0, &mut emit);
                    }
                }
            }
            Symmetry::Group(group) => {
                group.rotations(|rot| self.emit_element(rot, 0.0, &mut emit));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::unwrap_used)]
    use super::*;
    use crate::{Walker, WalkerPattern};
    use spacelab_core::Vec3;

    const MU: f64 = 3.986_004_418e14;
    const RE: f64 = 6_378_137.0;

    fn seed() -> KeplerianElements {
        KeplerianElements {
            semi_major_axis: RE + 12_000e3,
            eccentricity: 0.1,
            inclination: Angle::from_radians(63.434_f64.to_radians()),
            raan: Angle::from_radians(0.0),
            arg_periapsis: Angle::from_radians(30.0_f64.to_radians()),
            true_anomaly: Angle::from_radians(0.0),
            mu: MU,
        }
    }

    fn collect(kr: &KinematicallyRegular) -> Vec<KeplerianElements> {
        let mut v = Vec::new();
        kr.generate(|e| v.push(e));
        v
    }

    /// A constellation as the multiset of unit position directions at time `t`.
    fn dir_set(sats: &[KeplerianElements], t: f64) -> Vec<[f64; 3]> {
        sats.iter()
            .map(|e| {
                let r = e.propagate(t).to_state().0;
                let n = r.norm();
                [r.x / n, r.y / n, r.z / n]
            })
            .collect()
    }

    /// Multiset equality by greedy nearest-match — robust to ordering and float noise (no reliance on
    /// a total order over the tie-prone directions).
    fn sets_match(a: &[[f64; 3]], b: &[[f64; 3]]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut used = vec![false; b.len()];
        for p in a {
            let hit = b
                .iter()
                .enumerate()
                .find(|(i, q)| !used[*i] && (0..3).all(|k| (p[k] - q[k]).abs() < 1e-6));
            match hit {
                Some((i, _)) => used[i] = true,
                None => return false,
            }
        }
        true
    }

    #[test]
    fn the_reported_total_is_the_number_of_elements_emitted() {
        for symmetry in [
            Symmetry::CyclicPolar {
                order: 5,
                stagger: Angle::from_radians(0.2),
            },
            Symmetry::DihedralPolar {
                order: 4,
                stagger: Angle::from_radians(0.3),
            },
            Symmetry::Tetrahedral,
            Symmetry::Group(PointGroup::Octahedral),
        ] {
            let kr = KinematicallyRegular {
                seed: seed(),
                symmetry,
            };
            assert_eq!(
                kr.total() as usize,
                collect(&kr).len(),
                "{symmetry:?} reports a count it does not emit"
            );
        }
    }

    #[test]
    fn the_dihedral_arm_is_the_cyclic_progression_and_its_equatorial_flip() {
        // The dihedral branch repeats the cyclic branch's expression verbatim, and every mutant of
        // the cyclic copy is caught while every mutant of the dihedral copy survives. The whole
        // difference is the assertion: the dihedral test checks `len() == 8` and that `a` and `e`
        // are preserved — quantities invariant under every rotation and every mean-anomaly shift —
        // so the node spacing and the stagger are never observed. Rebuild both arms from the seed.
        // A pure z-rotation-invariance property would *not* see the sign of the progression; only
        // reconstructing the rotation does.
        let order = 4_u32;
        let stagger = 0.37_f64;
        let kr = KinematicallyRegular {
            seed: seed(),
            symmetry: Symmetry::DihedralPolar {
                order,
                stagger: Angle::from_radians(stagger),
            },
        };
        let got = collect(&kr);
        assert_eq!(got.len(), 2 * order as usize);

        let flip = Mat3::rot1(180.0_f64.to_radians());
        for k in 0..order {
            let rz = Mat3::rot3(-f64::from(k) * 360.0_f64.to_radians() / f64::from(order));
            let m = Angle::from_radians(seed().mean_anomaly().radians() + f64::from(k) * stagger);
            let (r, _) = seed().at_mean_anomaly(m).to_state();
            for (idx, rot) in [(2 * k as usize, rz), (2 * k as usize + 1, rz * flip)] {
                let want = rot * r;
                let d = got[idx].to_state().0 - want;
                assert!(d.norm() < 1e-6, "element {idx} is {} m away", d.norm());
            }
        }
    }

    #[test]
    fn cyclic_polar_is_a_single_axis_walker() {
        // C_6 about the pole with a 60° stagger == Walker delta 6/6/1 (one satellite per plane, planes
        // at 2πk/6, mean anomaly staggered k·60°). Compare as RAAN/true-anomaly sets.
        let n = 6u32;
        let a = RE + 20_000e3;
        let inc = Angle::from_radians(55.0_f64.to_radians());
        let s0 = KeplerianElements {
            semi_major_axis: a,
            eccentricity: 0.0,
            inclination: inc,
            raan: Angle::from_radians(0.0),
            arg_periapsis: Angle::from_radians(0.0),
            true_anomaly: Angle::from_radians(0.0),
            mu: MU,
        };
        let kr = KinematicallyRegular {
            seed: s0,
            symmetry: Symmetry::CyclicPolar {
                order: n,
                stagger: Angle::from_radians(TAU / f64::from(n)),
            },
        };
        let walker = Walker {
            pattern: WalkerPattern::Delta,
            inclination: inc,
            planes: n,
            per_plane: 1,
            phasing: 1,
            semi_major_axis: a,
            mu: MU,
        };
        assert!(sets_match(&dir_set(&collect(&kr), 0.0), &{
            let mut w = Vec::new();
            walker.generate(|e| w.push(e));
            dir_set(&w, 0.0)
        }));
    }

    #[test]
    fn dihedral_and_tetrahedral_have_the_right_counts_and_shared_orbit_size() {
        let dihedral = KinematicallyRegular {
            seed: seed(),
            symmetry: Symmetry::DihedralPolar {
                order: 4,
                stagger: Angle::from_radians(0.3),
            },
        };
        let tetra = KinematicallyRegular {
            seed: seed(),
            symmetry: Symmetry::Tetrahedral,
        };
        assert_eq!(collect(&dihedral).len(), 8);
        assert_eq!(collect(&tetra).len(), 12);
        // Rotations preserve orbit size and shape: every satellite keeps the seed's a and e.
        for s in collect(&dihedral).iter().chain(&collect(&tetra)) {
            assert!((s.semi_major_axis - seed().semi_major_axis).abs() < 1e-3);
            assert!((s.eccentricity - seed().eccentricity).abs() < 1e-9);
        }
    }

    #[test]
    fn point_group_orbit_and_from_symbol_build_the_group() {
        use crate::mozhaev::{GroupType, MozhaevSymbol, PointGroup};

        // The seed's orbit under the octahedral group has 24 satellites and is invariant under a
        // group element (a 90° rotation about z, which lies in O).
        let kr = KinematicallyRegular {
            seed: seed(),
            symmetry: Symmetry::Group(PointGroup::Octahedral),
        };
        let sats = collect(&kr);
        assert_eq!(sats.len(), 24);
        assert_eq!(kr.total(), 24);
        let base = dir_set(&sats, 0.0);
        let g = Mat3::rot3(-core::f64::consts::FRAC_PI_2); // active +90° about z ∈ O
        let rotated: Vec<[f64; 3]> = base
            .iter()
            .map(|d| {
                let r = g * Vec3::new(d[0], d[1], d[2]);
                [r.x, r.y, r.z]
            })
            .collect();
        assert!(
            sets_match(&base, &rotated),
            "octahedral orbit not group-invariant"
        );

        // from_symbol builds the orbit under the symbol's spatial component H_R.
        let symbol = MozhaevSymbol {
            order: 12,
            component: PointGroup::Tetrahedral,
            intersection: PointGroup::Tetrahedral,
            kappa: None,
            group_type: GroupType::First,
        };
        let from_sym = KinematicallyRegular::from_symbol(seed(), &symbol);
        assert_eq!(collect(&from_sym).len(), 12);
    }

    #[test]
    fn tetrahedral_set_is_invariant_under_its_generators_at_all_times() {
        // The defining kinematic-regularity property: a group generator applied to every satellite
        // permutes the constellation — and, because the flow commutes with rotations, at every time,
        // not just the epoch. Check a 120° body-diagonal rotation and a 180° axis rotation, at t=0 and
        // a quarter period later.
        let kr = KinematicallyRegular {
            seed: seed(),
            symmetry: Symmetry::Tetrahedral,
        };
        let sats = collect(&kr);
        let period = TAU * (seed().semi_major_axis.powi(3) / MU).sqrt();
        let generators = [TETRA_PERMS[1], TETRA_SIGNS[1]];
        for g in generators {
            for &t in &[0.0, 0.25 * period] {
                let base = dir_set(&sats, t);
                // Apply g to every satellite's direction; the set must be unchanged.
                let rotated: Vec<[f64; 3]> = base
                    .iter()
                    .map(|d| {
                        let r = g * Vec3::new(d[0], d[1], d[2]);
                        [r.x, r.y, r.z]
                    })
                    .collect();
                assert!(
                    sets_match(&base, &rotated),
                    "not invariant under a generator at t={t}"
                );
            }
        }
    }
}
