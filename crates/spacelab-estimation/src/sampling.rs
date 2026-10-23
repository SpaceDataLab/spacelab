//! Deterministic pseudo-random sampling for the Monte-Carlo estimators.
//!
//! Every other filter in this crate is a deterministic function of its inputs. The ensemble and
//! particle filters are not: they carry a *sample*, and a sample needs a source of randomness. That
//! source is made an explicit argument rather than hidden behind a global, for two reasons that are
//! not style preferences:
//!
//! * **A filter run must be reproducible.** A covariance measured from an ensemble is a random
//!   variable; a test that asserts on one cannot distinguish a regression from a different draw
//!   unless the draw is pinned. Threading `&mut Rng` makes the seed part of the call.
//! * **The crate is `no_std`.** There is no entropy source to fall back on, and there should not be
//!   one: seeding from the clock would make a filter irreproducible by construction.
//!
//! The generator is **xoshiro256++** seeded through **`SplitMix64`** (Blackman & Vigna 2018) — 256
//! bits of state, a period of 2²⁵⁶−1, and passing `BigCrush`. `spacelab-earthvision` carries the same
//! twenty lines for its Monte-Carlo radiative transfer; the duplication is deliberate, because the
//! alternative is a dependency edge from an estimation crate to an imaging crate.
//!
//! Normal deviates come from the **Box–Muller** transform, which is exact rather than approximate
//! (unlike a sum-of-uniforms) and needs no tables (unlike the ziggurat). Its cost is two logarithms
//! and a sine-cosine pair per *pair* of deviates, so the second value is cached.

// Every `as f64` below is applied to a value the preceding shift or mask has already reduced below
// 2⁵³, so the conversion is exact rather than rounded.
#![allow(clippy::cast_precision_loss)]

/// A seeded xoshiro256++ generator with a cached second Box–Muller deviate.
///
/// Reproducible: the same seed and the same sequence of calls produce the same numbers, on every
/// platform, because every operation is integer arithmetic and IEEE-754 double arithmetic with no
/// libm call in the uniform path.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
    /// The unused half of the last Box–Muller pair, if there is one.
    spare_normal: Option<f64>,
}

impl Rng {
    /// Seed the generator. Any `u64` is a valid seed, including zero — `SplitMix64` expands it into
    /// four words before xoshiro sees it, so an all-zero state (which xoshiro cannot leave) is
    /// unreachable.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut state = [0u64; 4];
        let mut i = 0;
        while i < 4 {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            state[i] = x ^ (x >> 31);
            i += 1;
        }
        Self {
            state,
            spare_normal: None,
        }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// A uniform draw on `[0, 1)`, with the 53 bits a double can hold.
    pub fn unit(&mut self) -> f64 {
        // 2⁻⁵³ exactly; the shift discards the low bits, which are the weakest in any xoshiro variant.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// A uniform draw on `(0, 1]` — the half-open end that a logarithm can survive.
    fn open_unit(&mut self) -> f64 {
        open_unit_from_bits(self.next_u64() >> 11)
    }

    /// A draw from `N(0, 1)`.
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        let radius = libm::sqrt(-2.0 * libm::log(self.open_unit()));
        let angle = core::f64::consts::TAU * self.unit();
        self.spare_normal = Some(radius * libm::sin(angle));
        radius * libm::cos(angle)
    }

    /// A draw from `N(0, A Aᵀ)`, given any square root `A` of the covariance (a Cholesky factor is
    /// the usual one).
    ///
    /// The factor is applied as `A ξ` with `ξ ~ N(0, I)`, which is why *any* square root does: the
    /// distribution depends on `A Aᵀ` and not on `A`.
    pub fn correlated_normal<const N: usize>(
        &mut self,
        covariance_sqrt: &[[f64; N]; N],
    ) -> [f64; N] {
        let mut xi = [0.0; N];
        for value in &mut xi {
            *value = self.standard_normal();
        }
        let mut out = [0.0; N];
        for i in 0..N {
            let mut acc = 0.0;
            for j in 0..N {
                acc += covariance_sqrt[i][j] * xi[j];
            }
            out[i] = acc;
        }
        out
    }
}

/// Map 53 random bits onto `(0, 1]`.
///
/// Split out from [`Rng::open_unit`] because the draw this exists for — all 53 bits zero, one time
/// in 2⁵³ — is one no test can wait for. As a function of the bits it is one call away, and the
/// difference between `(0, 1]` and `[0, 1)` is the difference between a finite normal deviate and an
/// infinite one.
const fn open_unit_from_bits(bits: u64) -> f64 {
    (bits + 1) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::many_single_char_names,
        clippy::items_after_statements
    )]
    use super::*;

    /// The same seed replays the same stream, and different seeds do not.
    #[test]
    fn the_generator_is_reproducible_and_seed_dependent() {
        let mut a = Rng::new(20_260_814);
        let mut b = Rng::new(20_260_814);
        let mut c = Rng::new(20_260_815);
        let (mut same, mut differ) = (0, 0);
        for _ in 0..1000 {
            let (x, y, z) = (a.next_u64(), b.next_u64(), c.next_u64());
            if x == y {
                same += 1;
            }
            if x != z {
                differ += 1;
            }
        }
        assert_eq!(same, 1000, "the same seed must replay exactly");
        assert_eq!(
            differ, 1000,
            "1000 collisions between two seeds is not chance"
        );
    }

    /// The generator against the published `xoshiro256++` stream.
    ///
    /// Everything else here is a *statistic*, and a damaged generator hides inside one: a shift that
    /// goes the wrong way or a state update that ANDs where it should XOR still delivers moments
    /// within their standard error, because the state space is large enough that almost any linear
    /// map over it looks random at four moments. The stream is the one thing about the generator
    /// that is exactly specified. The fixture comes from `tools/gen_xoshiro_oracle.py`, written from
    /// Blackman & Vigna's algorithm rather than from this code, and it is integer arithmetic
    /// throughout, so it is bit-exact.
    #[test]
    fn the_generator_reproduces_the_published_xoshiro256pp_stream() {
        const STREAM: [(u64, [u64; 4]); 3] = [
            (
                0,
                [
                    0x5317_5d61_490b_23df,
                    0x61da_6f3d_c380_d507,
                    0x5c0f_df91_ec9a_7bfc,
                    0x02ee_bf8c_3bbe_5e1a,
                ],
            ),
            (
                1,
                [
                    0xcfc5_d07f_6f03_c29b,
                    0xbf42_4132_963f_e08d,
                    0x19a3_7d57_57aa_f520,
                    0xbf08_119f_05cd_56d6,
                ],
            ),
            (
                42,
                [
                    0xd076_4d4f_4476_689f,
                    0x519e_4174_576f_3791,
                    0xfbe0_7cfb_0c24_ed8c,
                    0xb37d_9f60_0cd8_35b8,
                ],
            ),
        ];
        for (seed, want) in STREAM {
            let mut rng = Rng::new(seed);
            for (i, w) in want.into_iter().enumerate() {
                assert_eq!(rng.next_u64(), w, "seed {seed}, draw {i}");
            }
        }
    }

    /// The `(0, 1]` mapping at the draw that makes it necessary, and at the other end.
    ///
    /// All 53 bits zero happens once in 2⁵³ draws — never, in any test that waits for it, and
    /// eventually, in a filter that runs. There `open_unit` must return `2⁻⁵³` rather than zero,
    /// because `ln 0` is `−∞` and one infinite deviate poisons every member of the ensemble it is
    /// drawn for.
    #[test]
    fn the_open_unit_mapping_excludes_zero_and_includes_one() {
        assert_eq!(open_unit_from_bits(0), 1.0 / 9_007_199_254_740_992.0);
        assert_eq!(open_unit_from_bits((1 << 53) - 1), 1.0);
        assert!(libm::log(open_unit_from_bits(0)).is_finite());
    }

    /// The uniform draw stays inside its stated interval, and the logarithm-safe one is never zero.
    ///
    /// `open_unit` returning zero would make `standard_normal` return infinity, which propagates into
    /// every ensemble member at once — a failure that shows up as a covariance of NaN three steps
    /// later, far from its cause.
    #[test]
    fn the_uniform_draws_respect_their_intervals() {
        let mut rng = Rng::new(7);
        for _ in 0..100_000 {
            let u = rng.unit();
            assert!((0.0..1.0).contains(&u), "unit() returned {u}");
            let o = rng.open_unit();
            assert!(o > 0.0 && o <= 1.0, "open_unit() returned {o}");
        }
    }

    /// The first four moments of the normal draw, against their exact values.
    ///
    /// Mean 0, variance 1, skewness 0, kurtosis 3. With 200 000 draws the standard error of the mean
    /// is 2.2e-3 and of the variance 3.2e-3, so the bounds below are ~4σ — tight enough to catch a
    /// missing `√2` or a dropped cosine, loose enough not to depend on the seed.
    #[test]
    fn the_normal_draw_has_the_right_moments() {
        let mut rng = Rng::new(1_234_567);
        const COUNT: usize = 200_000;
        let (mut m1, mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0, 0.0);
        for _ in 0..COUNT {
            let x = rng.standard_normal();
            m1 += x;
            m2 += x * x;
            m3 += x * x * x;
            m4 += x * x * x * x;
        }
        let n = COUNT as f64;
        let (mean, var) = (m1 / n, m2 / n);
        let (skew, kurt) = (m3 / n, m4 / n);
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
        assert!(skew.abs() < 0.03, "skewness {skew}");
        assert!((kurt - 3.0).abs() < 0.10, "kurtosis {kurt}");
    }

    /// A correlated draw is the factor applied to standard normals, in draw order.
    ///
    /// The covariance test below cannot see this: `A ξ` and `−A ξ` have the same covariance, and so
    /// does `A` applied to the draws in the other order. Both are conventions rather than
    /// mathematics, and both have to hold for a filter run to replay — which is the property the
    /// whole module exists for.
    #[test]
    fn the_correlated_draw_is_the_factor_times_the_standard_draws() {
        let l = [[2.0, 0.0], [1.0, 2.0]];
        let mut applied = Rng::new(4242);
        let got = applied.correlated_normal(&l);
        let mut direct = Rng::new(4242);
        let xi = [direct.standard_normal(), direct.standard_normal()];
        assert_eq!(got, [2.0 * xi[0], xi[0] + 2.0 * xi[1]]);
    }

    /// A correlated draw reproduces the covariance it was given, including the off-diagonal sign.
    #[test]
    fn the_correlated_draw_reproduces_its_covariance() {
        // L Lᵀ = [[4, 2], [2, 5]] — correlation +0.447.
        let l = [[2.0, 0.0], [1.0, 2.0]];
        let mut rng = Rng::new(99);
        const COUNT: usize = 200_000;
        let mut sum = [0.0; 2];
        let mut cov = [[0.0; 2]; 2];
        for _ in 0..COUNT {
            let x = rng.correlated_normal(&l);
            for i in 0..2 {
                sum[i] += x[i];
                for j in 0..2 {
                    cov[i][j] += x[i] * x[j];
                }
            }
        }
        let n = COUNT as f64;
        let want = [[4.0, 2.0], [2.0, 5.0]];
        for i in 0..2 {
            assert!(
                (sum[i] / n).abs() < 0.03,
                "component {i} mean {}",
                sum[i] / n
            );
            for j in 0..2 {
                let got = cov[i][j] / n;
                assert!(
                    (got - want[i][j]).abs() < 0.08,
                    "cov[{i}][{j}] = {got}, want {}",
                    want[i][j]
                );
            }
        }
    }
}
