/* H-libm: how far libm 0.2.16's elementary functions are from the true
   value, measured against Arb balls, and held as a named hypothesis
   (plans/18 Ф28.J; docs/verification.md §1.1).

   A measurement is a tested hypothesis, not a proof. The constants in
   h_libm.txt are the largest errors this file finds, rounded up to the next
   multiple of 1/64 ulp, and 1/2 for sqrt and cbrt, which are correctly
   rounded. A larger error may exist at an input no sample hits;
   the published largest known errors (below) are lower bounds of the true
   maximum too. Every result that uses a constant says "under H-libm".

       cc -O2 libm_hypothesis_arb.c -o libm_hypothesis_arb \
           $(pkg-config --cflags --libs flint)
       ./libm_hypothesis_arb gen \
           | cargo run -q --release -p spacelab-core --example libm_eval \
           | ./libm_hypothesis_arb check h_libm.txt
       ./libm_hypothesis_arb hard        the hardest arguments, for the .gp file

   gen    prints `row function a1 [a2]`, each double as the 16 hex digits of
          its bits, for every row of h_libm.txt. The inputs are deterministic
          (splitmix64, a fixed seed) and use no libm of their own: a random
          double is built from a random binade and random mantissa bits, so
          every platform draws the same ones.
   check  reads those lines with libm's answer appended, evaluates the
          function in ball arithmetic on the same exact double, and computes
          the error |r - f| / ulp(f) as a ball. ulp(f) = 2^(max(e, -1022) - 52)
          with e = floor(log2 |f|), taken from the ball's lower end: near a
          power of two the smaller ulp is used, which can only make the error
          larger. The precision doubles from 192 bits until the error's radius
          is below 2^-70 ulp; an input that cannot be decided fails. A row
          fails if any upper end of its error is above its constant, if it has
          no points, or if a line names a row the table does not have. Exit 0
          only if every row holds.
   hard   prints the hardest argument of every binade (below) and the hardest
          of all, for libm_hypothesis.gp to recompute in PARI/GP.

   The hard arguments: the doubles nearest multiples of pi/2, where the
   reduction of sin, cos and tan cancels. For x = m 2^k, m in [2^52, 2^53),
   x is close to j pi/2 when m theta is close to an integer, theta the
   fractional part of 2^(k+1)/pi. The best m are the denominators of the
   continued fraction of theta, its semiconvergents and the smallest multiple
   of a smaller denominator in the range; for each k from -52 (x in [1, 2)) to
   971 the four nearest such m are kept, with m - 1 and m + 1, and both signs.
   The doubles on either side of j pi/2 for j = 1 ... 256 are added.

   The published inputs: Gladman, Innocente, Mather, Ozaki and Zimmermann,
   "Accuracy of Mathematical Functions in Single, Double, Double Extended, and
   Quadruple Precision", August 2026 edition, Tables 3, 7 and 9: the inputs of
   the largest known errors of Musl 1.2.6, OpenLibm 0.8.7 and FreeBSD 15.1,
   whose code libm 0.2.16 ports (FreeBSD msun for sin, cos, tan, asin, acos,
   atan, atan2, exp, log; musl for hypot). Each lies in the `all` row of its
   function; atan2's and hypot's are also scaled by a power of two, which
   changes no rounding inside the function, into the proof rows they fit. */

#include <flint/arb.h>
#include <flint/fmpz.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- deterministic inputs ------------------------------------------------ */

static uint64_t state = 0x48d159e26af37bc0ULL; /* fixed seed */

static uint64_t next(void) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static double from_bits(uint64_t b) {
  double x;
  memcpy(&x, &b, sizeof x);
  return x;
}

static uint64_t to_bits(double x) {
  uint64_t b;
  memcpy(&b, &x, sizeof b);
  return b;
}

/* uniform on [a, b], by IEEE operations only */
static double uniform(double a, double b) {
  return a + (b - a) * ((double)(next() >> 11) * 0x1p-53);
}

/* a double whose biased exponent field is uniform on [lo, hi] (0 is the
   subnormals) and whose mantissa bits are random */
static double field(int lo, int hi) {
  uint64_t e = (uint64_t)lo + next() % (uint64_t)(hi - lo + 1);
  return from_bits((e << 52) | (next() >> 12));
}

/* log-uniform on [lo, hi], lo > 0, both normal: a random binade between
   theirs, random mantissa bits, redrawn until inside */
static double logu(double lo, double hi) {
  int elo = (int)((to_bits(lo) >> 52) & 0x7ff),
      ehi = (int)((to_bits(hi) >> 52) & 0x7ff);
  for (;;) {
    double x = field(elo, ehi);
    if (x >= lo && x <= hi)
      return x;
  }
}

static double sign(double x) { return (next() & 1) ? -x : x; }

static void emit1(const char *row, const char *fn, double x) {
  printf("%s %s %016llx\n", row, fn, (unsigned long long)to_bits(x));
}

static void emit2(const char *row, const char *fn, double a, double b) {
  printf("%s %s %016llx %016llx\n", row, fn, (unsigned long long)to_bits(a),
         (unsigned long long)to_bits(b));
}

/* the double nearest j pi / 2 */
static double nearest_multiple(slong j) {
  arb_t t;
  double d;
  arb_init(t);
  arb_const_pi(t, 256);
  arb_mul_si(t, t, j, 256);
  arb_mul_2exp_si(t, t, -1);
  d = arf_get_d(arb_midref(t), ARF_RND_NEAR);
  arb_clear(t);
  return d;
}

/* ---- the hard arguments: doubles nearest multiples of pi/2 -------------- */

#define KEEP 4
#define KMIN (-52)
#define KMAX 971

typedef struct {
  uint64_t m;
  int k;
  double dist; /* |x - j pi/2|, x = m 2^k */
} hard_t;

static hard_t hard[(KMAX - KMIN + 1) * KEEP];
static int nhard;
static hard_t hardest;

static void find_hard(void) {
  fmpz_t num, den, a, q, qp, qn, two52, two53, c, t;
  arb_t pi, alpha, s;
  fmpz_init(num);
  fmpz_init(den);
  fmpz_init(a);
  fmpz_init(q);
  fmpz_init(qp);
  fmpz_init(qn);
  fmpz_init(two52);
  fmpz_init(two53);
  fmpz_init(c);
  fmpz_init(t);
  arb_init(pi);
  arb_init(alpha);
  arb_init(s);
  fmpz_one(two52);
  fmpz_mul_2exp(two52, two52, 52);
  fmpz_mul_2exp(two53, two52, 1);
  nhard = 0;
  hardest.dist = 1;
  for (int k = KMIN; k <= KMAX; k++) {
    slong prec = (k + 1 > 0 ? k + 1 : 0) + 320;
    uint64_t cand[256];
    int ncand = 0;
    hard_t best[KEEP];
    int nbest = 0;
    /* alpha = 2^(k+1) / pi; theta = frac(alpha) = num / 2^240 */
    arb_const_pi(pi, prec + 16);
    arb_one(alpha);
    arb_mul_2exp_si(alpha, alpha, k + 1);
    arb_div(alpha, alpha, pi, prec);
    arb_mul_2exp_si(s, alpha, 240);
    if (mag_cmp_2exp_si(arb_radref(s), -8) >= 0) {
      fprintf(stderr, "hard: alpha not accurate at k = %d\n", k);
      exit(2);
    }
    arf_get_fmpz(num, arb_midref(s), ARF_RND_FLOOR);
    fmpz_fdiv_r_2exp(num, num, 240);
    fmpz_one(den);
    fmpz_mul_2exp(den, den, 240);
    /* continued fraction of num / den: denominators qp (previous), q */
    fmpz_zero(qp);
    fmpz_one(q);
    while (!fmpz_is_zero(num) && fmpz_cmp(q, two53) < 0 && ncand < 250) {
      /* next partial quotient a; the remainder becomes (den, num) */
      fmpz_fdiv_qr(a, t, den, num);
      fmpz_swap(den, num);
      fmpz_swap(num, t);
      /* q itself */
      if (fmpz_cmp(q, two52) >= 0)
        cand[ncand++] = fmpz_get_ui(q);
      /* the smallest multiple of q in [2^52, 2^53) */
      if (fmpz_cmp(q, two52) < 0) {
        fmpz_cdiv_q(c, two52, q);
        fmpz_mul(c, c, q);
        if (fmpz_cmp(c, two53) < 0)
          cand[ncand++] = fmpz_get_ui(c);
      }
      /* the largest semiconvergent qp + t q below 2^53, t <= a */
      fmpz_sub_ui(c, two53, 1);
      fmpz_sub(c, c, qp);
      fmpz_fdiv_q(c, c, q);
      if (fmpz_cmp(c, a) > 0)
        fmpz_set(c, a);
      if (fmpz_sgn(c) > 0) {
        fmpz_mul(c, c, q);
        fmpz_add(c, c, qp);
        if (fmpz_cmp(c, two52) >= 0 && fmpz_cmp(c, two53) < 0)
          cand[ncand++] = fmpz_get_ui(c);
      }
      /* the next denominator */
      fmpz_mul(qn, a, q);
      fmpz_add(qn, qn, qp);
      fmpz_swap(qp, q);
      fmpz_swap(q, qn);
    }
    if (fmpz_cmp(q, two52) >= 0 && fmpz_cmp(q, two53) < 0 && ncand < 256)
      cand[ncand++] = fmpz_get_ui(q);
    /* keep the KEEP nearest */
    for (int i = 0; i < ncand; i++) {
      arb_t x;
      fmpz_t j;
      double d;
      int dup = 0;
      for (int b = 0; b < nbest; b++)
        dup |= best[b].m == cand[i];
      if (dup)
        continue;
      arb_init(x);
      fmpz_init(j);
      arb_mul_ui(x, alpha, cand[i], prec);
      arf_get_fmpz(j, arb_midref(x), ARF_RND_NEAR);
      arb_sub_fmpz(x, x, j, prec);
      arb_abs(x, x);
      arb_mul(x, x, pi, prec);
      arb_mul_2exp_si(x, x, -1);
      d = arf_get_d(arb_midref(x), ARF_RND_NEAR);
      arb_clear(x);
      fmpz_clear(j);
      if (nbest < KEEP) {
        best[nbest].m = cand[i];
        best[nbest].k = k;
        best[nbest++].dist = d;
      } else {
        int w = 0;
        for (int b = 1; b < KEEP; b++)
          if (best[b].dist > best[w].dist)
            w = b;
        if (d < best[w].dist) {
          best[w].m = cand[i];
          best[w].dist = d;
        }
      }
    }
    for (int b = 0; b < nbest; b++) {
      hard[nhard++] = best[b];
      if (best[b].dist < hardest.dist)
        hardest = best[b];
    }
  }
  fmpz_clear(num);
  fmpz_clear(den);
  fmpz_clear(a);
  fmpz_clear(q);
  fmpz_clear(qp);
  fmpz_clear(qn);
  fmpz_clear(two52);
  fmpz_clear(two53);
  fmpz_clear(c);
  fmpz_clear(t);
  arb_clear(pi);
  arb_clear(alpha);
  arb_clear(s);
}

/* ---- the rows: one block per row of h_libm.txt -------------------------- */

#define N 200000  /* points per general row */
#define P 1000000 /* points per proof row */

static const char *trig[] = {"sin", "cos", "tan", "sincos_sin", "sincos_cos"};

static void gen(void) {
  const double tau = 0x1.921fb54442d18p+2; /* core::f64::consts::TAU */
  char row[64];
  find_hard();

  /* Kepler: libm::sincos(E), E in [-0.001, 6.3] (kepler_roundoff.gappa) */
  for (int i = 0; i < P; i++) {
    double e = uniform(-0.001, 6.3);
    emit1("kepler_sin", "sincos_sin", e);
    emit1("kepler_cos", "sincos_cos", e);
  }
  for (int i = 0; i < P / 4; i++) {
    double e = (next() & 1) ? -logu(0x1p-60, 0.001) : logu(0x1p-60, 0.001);
    emit1("kepler_sin", "sincos_sin", e);
    emit1("kepler_cos", "sincos_cos", e);
  }
  for (int j = 1; j <= 4; j++) {
    double x = nearest_multiple(j), y = x;
    for (int s = 0; s < 16; s++) {
      emit1("kepler_sin", "sincos_sin", x);
      emit1("kepler_cos", "sincos_cos", x);
      emit1("kepler_sin", "sincos_sin", y);
      emit1("kepler_cos", "sincos_cos", y);
      x = nextafter(x, 7);
      y = nextafter(y, 0);
    }
  }
  for (int h = 0; h < nhard; h++) {
    double x = ldexp((double)hard[h].m, hard[h].k);
    if (x <= 6.3) {
      emit1("kepler_sin", "sincos_sin", x);
      emit1("kepler_cos", "sincos_cos", x);
    }
  }
  emit1("kepler_sin", "sincos_sin", -0.001);
  emit1("kepler_cos", "sincos_cos", -0.001);
  emit1("kepler_sin", "sincos_sin", 6.3);
  emit1("kepler_cos", "sincos_cos", 6.3);

  /* Vermeille: hypot(d, z), d in [1 mm, 6.1e7 m], z in [1 mm, 4.3e7 m] */
  for (int i = 0; i < P; i++)
    emit2("vermeille_hypot", "hypot", logu(1e-3, 6.1e7), logu(1e-3, 4.3e7));
  for (int i = 0; i < P / 4; i++) {
    double d = logu(1e-3, 4.3e7), z = d * uniform(1 - 0x1p-8, 1 + 0x1p-8);
    if (z >= 1e-3 && z <= 4.3e7)
      emit2("vermeille_hypot", "hypot", d, z);
  }
  emit2("vermeille_hypot", "hypot", 0x1.00014d4b1c6b9p+23,
        0x1.000105ba9bf4p+23);
  emit2("vermeille_hypot", "hypot", 0x1.6a0a41410b1abp+24,
        ldexp(0x0.a24afe71b539fp-1022, 1028));
  emit2("vermeille_hypot", "hypot", 0x1.6a0a41410b1abp+24,
        0x1.4495fce36a73ep+5);

  /* Vermeille: atan2(z, d + hyp), 0 < z <= d + hyp <= 1.3e8 m, z >= 1 mm */
  for (int i = 0; i < 2 * P; i++) {
    double w = logu(1e-3, 1.3e8), z = w * logu(0x1p-37, 1);
    if (z >= 1e-3 && z <= 4.3e7 && z <= w)
      emit2("vermeille_atan2", "atan2", z, w);
  }
  emit2("vermeille_atan2", "atan2", 0x1.358bb5eb25bdcp+13,
        0x1.2f86b82481a0ap+15);

  /* Vermeille: cbrt of 1 + s + sqrt(s (2 + s)), s in [0, 6.34e-4] */
  for (int i = 0; i < N / 2; i++)
    emit1("vermeille_cbrt", "cbrt", uniform(1, 1.0364));
  emit1("vermeille_cbrt", "cbrt", 1.0364);

  /* The shadow lens: atan2(heron, d^2 -+ squares), heron > 0 */
  for (int i = 0; i < P; i++)
    emit2("shadow_atan2", "atan2", logu(0x1p-60, 16), sign(logu(0x1p-60, 16)));
  emit2("shadow_atan2", "atan2", 0x1.358bb5eb25bdcp+0, 0x1.2f86b82481a0ap+2);

  /* The shadow lens: sin(x) in x_minus_sin's direct branch, x in [2, 2 pi] */
  for (int i = 0; i < P / 2; i++)
    emit1("shadow_sin", "sin", uniform(2, tau));
  for (int j = 2; j <= 4; j++) {
    double x = nearest_multiple(j), y = x;
    for (int s = 0; s < 16; s++) {
      if (x <= tau)
        emit1("shadow_sin", "sin", x);
      if (y >= 2 && y <= tau)
        emit1("shadow_sin", "sin", y);
      x = nextafter(x, 7);
      y = nextafter(y, 0);
    }
  }

  /* sqrt, every positive double */
  for (int i = 0; i < N / 2; i++)
    emit1("sqrt_all", "sqrt", field(0, 2046));
  emit1("sqrt_all", "sqrt", 0x1.fffffffffffffp-1);

  /* sin, cos, tan and both halves of sincos, every finite argument; and the
     hard arguments */
  for (int f = 0; f < 5; f++) {
    snprintf(row, sizeof row, "%s_all", trig[f]);
    for (int i = 0; i < N / 2; i++)
      emit1(row, trig[f], uniform(-tau, tau));
    for (int i = 0; i < N / 4; i++)
      emit1(row, trig[f], sign(logu(0x1p-26, 0x1p21)));
    for (int i = 0; i < N / 4; i++)
      emit1(row, trig[f], sign(field(1, 2046)));
    emit1(row, trig[f], 0x1.4d84db080b9fdp+21);  /* sin */
    emit1(row, trig[f], -0x1.34e729fd08086p+21); /* cos */
    emit1(row, trig[f], 0x1.3f9605aaeb51bp+21);  /* tan */
    snprintf(row, sizeof row, "%s_hard", trig[f]);
    for (int h = 0; h < nhard; h++)
      for (int dm = -1; dm <= 1; dm++) {
        double x =
            ldexp((double)(hard[h].m + (uint64_t)(int64_t)dm), hard[h].k);
        emit1(row, trig[f], x);
        emit1(row, trig[f], -x);
      }
    for (int j = 1; j <= 256; j++) {
      double x = nearest_multiple(j);
      emit1(row, trig[f], x);
      emit1(row, trig[f], nextafter(x, 0));
      emit1(row, trig[f], nextafter(x, INFINITY));
    }
  }

  /* atan2, every quadrant and scale */
  for (int i = 0; i < N; i++) {
    double x = field(1, 2046), y = x * logu(0x1p-60, 0x1p60);
    if (isfinite(y) && y != 0)
      emit2("atan2_all", "atan2", sign(y), sign(x));
  }
  emit2("atan2_all", "atan2", -0x1.358bb5eb25bdcp+813, 0x1.2f86b82481a0ap+815);

  /* hypot, every scale, subnormals included */
  for (int i = 0; i < N; i++) {
    double x = field(0, 2045), y = x * logu(0x1p-60, 1);
    if (y != 0)
      emit2("hypot_all", "hypot", sign((next() & 1) ? x : y),
            sign((next() & 1) ? y : x));
  }
  emit2("hypot_all", "hypot", 0x1.00014d4b1c6b9p-1015, -0x1.000105ba9bf4p-1015);
  emit2("hypot_all", "hypot", 0x1.6a0a41410b1abp-1004,
        -0x0.a24afe71b539fp-1022);
  emit2("hypot_all", "hypot", 0x1.6a0a41410b1abp-1004,
        -0x1.4495fce36a73ep-1023);

  /* exp, every argument with a finite nonzero result */
  for (int i = 0; i < N / 2; i++)
    emit1("exp_all", "exp",
          uniform(-0x1.74910d52d3051p+9, 0x1.62e42fefa39efp+9));
  for (int i = 0; i < N / 2; i++)
    emit1("exp_all", "exp", sign(logu(0x1p-60, 512)));
  emit1("exp_all", "exp", 0x1.2e8f20cf3cbe7p+8);
  emit1("exp_all", "exp", -0x1.18209ecd19a8cp+6);

  /* log, every positive double, and the neighbourhood of 1 */
  for (int i = 0; i < N / 2; i++)
    emit1("log_all", "log", field(0, 2046));
  for (int i = 0; i < N / 2; i++) {
    double t = logu(0x1p-52, 0.5);
    emit1("log_all", "log", (next() & 1) ? 1 + t : 1 - t);
  }
  emit1("log_all", "log", 0x1.48ae5a67204f5p+0);
  emit1("log_all", "log", 0x1.dc0b586f2b26p-1);

  /* acos and asin on [-1, 1]: uniform, next to +-1, and small */
  for (int f = 0; f < 2; f++) {
    const char *fn = f ? "asin" : "acos";
    snprintf(row, sizeof row, "%s_all", fn);
    for (int i = 0; i < N / 2; i++)
      emit1(row, fn, uniform(-1, 1));
    for (int i = 0; i < N / 4; i++)
      emit1(row, fn, sign(1 - logu(0x1p-53, 0.5)));
    for (int i = 0; i < N / 4; i++)
      emit1(row, fn, sign(logu(0x1p-60, 0.5)));
    emit1(row, fn, -0x1.0068b067c6feep-1); /* acos */
    emit1(row, fn, -0x1.004d1c5a9400bp-1); /* asin */
  }

  /* atan, every finite argument */
  for (int i = 0; i < N / 2; i++)
    emit1("atan_all", "atan", sign(logu(0x1p-30, 0x1p30)));
  for (int i = 0; i < N / 2; i++)
    emit1("atan_all", "atan", sign(field(1, 2046)));
  emit1("atan_all", "atan", 0x1.62ff6a1682c25p-1);
}

/* ---- the check ----------------------------------------------------------- */

static int exact(arb_t f, const char *fn, double a1, double a2, slong prec) {
  arb_t x, y;
  int ok = 1;
  arb_init(x);
  arb_init(y);
  arb_set_d(x, a1);
  arb_set_d(y, a2);
  if (!strcmp(fn, "sin") || !strcmp(fn, "sincos_sin"))
    arb_sin(f, x, prec);
  else if (!strcmp(fn, "cos") || !strcmp(fn, "sincos_cos"))
    arb_cos(f, x, prec);
  else if (!strcmp(fn, "tan"))
    arb_tan(f, x, prec);
  else if (!strcmp(fn, "exp"))
    arb_exp(f, x, prec);
  else if (!strcmp(fn, "log"))
    arb_log(f, x, prec);
  else if (!strcmp(fn, "acos"))
    arb_acos(f, x, prec);
  else if (!strcmp(fn, "asin"))
    arb_asin(f, x, prec);
  else if (!strcmp(fn, "atan"))
    arb_atan(f, x, prec);
  else if (!strcmp(fn, "sqrt"))
    arb_sqrt(f, x, prec);
  else if (!strcmp(fn, "cbrt")) {
    arb_abs(y, x);
    arb_root_ui(f, y, 3, prec);
    if (a1 < 0)
      arb_neg(f, f);
  } else if (!strcmp(fn, "atan2"))
    arb_atan2(f, x, y, prec);
  else if (!strcmp(fn, "hypot"))
    arb_hypot(f, x, y, prec);
  else
    ok = 0;
  arb_clear(x);
  arb_clear(y);
  return ok;
}

/* err = |r - f| / ulp(f) as a ball, decided to 2^-70 ulp; 0 if it cannot be */
static int ulp_error(arb_t err, const char *fn, double a1, double a2,
                     double r) {
  arb_t f, t;
  arf_t lo;
  int done = 0;
  arb_init(f);
  arb_init(t);
  arf_init(lo);
  for (slong prec = 192; prec <= 16384 && !done; prec *= 2) {
    slong e;
    if (!exact(f, fn, a1, a2, prec))
      break;
    if (!arb_is_finite(f) || arb_contains_zero(f))
      continue;
    arb_abs(t, f);
    arb_get_lbound_arf(lo, t, prec);
    e = arf_abs_bound_lt_2exp_si(lo) - 1;
    if (e < -1022)
      e = -1022;
    arb_set_d(t, r);
    arb_sub(t, t, f, prec);
    arb_abs(t, t);
    arb_mul_2exp_si(t, t, 52 - e);
    if (arb_is_finite(t) && mag_cmp_2exp_si(arb_radref(t), -70) < 0) {
      arb_set(err, t);
      done = 1;
    }
  }
  arb_clear(f);
  arb_clear(t);
  arf_clear(lo);
  return done;
}

#define MAXROWS 64

typedef struct {
  char row[48], fn[16];
  double constant, max, a1, a2, r;
  int args;
  long n, above, undecided;
} stat_t;

static stat_t rows[MAXROWS];
static int nrows;

static int read_table(const char *path) {
  FILE *in = fopen(path, "r");
  char line[512];
  if (!in) {
    fprintf(stderr, "cannot open %s\n", path);
    return 0;
  }
  while (fgets(line, sizeof line, in)) {
    stat_t *s = &rows[nrows];
    if (line[0] == '#' || line[0] == '\n')
      continue;
    if (nrows == MAXROWS ||
        sscanf(line, "%47s %15s %lf", s->row, s->fn, &s->constant) != 3) {
      fprintf(stderr, "malformed table line: %s", line);
      fclose(in);
      return 0;
    }
    s->max = -1;
    nrows++;
  }
  fclose(in);
  return nrows > 0;
}

static int check(const char *table) {
  char line[256], row[48], fn[16];
  unsigned long long b1, b2, br;
  arb_t err;
  arf_t up;
  long total = 0;
  int failed = 0;
  if (!read_table(table))
    return 2;
  arb_init(err);
  arf_init(up);
  while (fgets(line, sizeof line, stdin)) {
    int k = sscanf(line, "%47s %15s %llx %llx %llx", row, fn, &b1, &b2, &br);
    int i, args = k == 5 ? 2 : 1;
    double a1, a2 = 0, r, e;
    if (k == 4)
      br = b2;
    else if (k != 5) {
      fprintf(stderr, "malformed data line: %s", line);
      return 2;
    }
    for (i = 0; i < nrows; i++)
      if (!strcmp(rows[i].row, row) && !strcmp(rows[i].fn, fn))
        break;
    if (i == nrows) {
      fprintf(stderr, "row %s %s is not in %s\n", row, fn, table);
      return 2;
    }
    a1 = from_bits(b1);
    if (args == 2)
      a2 = from_bits(b2);
    r = from_bits(br);
    rows[i].n++;
    rows[i].args = args;
    total++;
    if (!ulp_error(err, fn, a1, a2, r)) {
      rows[i].undecided++;
      fprintf(stderr, "UNDECIDED %s", line);
      continue;
    }
    arb_get_ubound_arf(up, err, 64);
    e = arf_get_d(up, ARF_RND_UP);
    if (e > rows[i].constant)
      rows[i].above++;
    if (e > rows[i].max) {
      rows[i].max = e;
      rows[i].a1 = a1;
      rows[i].a2 = a2;
      rows[i].r = r;
    }
  }
  for (int i = 0; i < nrows; i++) {
    stat_t *s = &rows[i];
    int bad = s->n == 0 || s->above > 0 || s->undecided > 0;
    failed += bad;
    printf("ROW %-18s %-10s n=%-7ld max=%.6f ulp at %016llx", s->row, s->fn,
           s->n, s->max, (unsigned long long)to_bits(s->a1));
    if (s->args == 2)
      printf(" %016llx", (unsigned long long)to_bits(s->a2));
    printf(" -> %016llx  constant %.6f  %s", (unsigned long long)to_bits(s->r),
           s->constant, bad ? "ABOVE" : "holds");
    if (s->n == 0)
      printf(" (no points)");
    else if (s->above || s->undecided)
      printf(" (%ld above, %ld undecided)", s->above, s->undecided);
    printf("\n");
  }
  if (failed)
    printf("REFUTED %d of %d rows over %ld points\n", failed, nrows, total);
  else
    printf("HOLDS %d rows over %ld points\n", nrows, total);
  arb_clear(err);
  arf_clear(up);
  return failed ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && !strcmp(argv[1], "gen")) {
    gen();
    return 0;
  }
  if (argc == 2 && !strcmp(argv[1], "hard")) {
    find_hard();
    for (int h = 0; h < nhard; h++)
      printf("HARD %llu %d %.6e\n", (unsigned long long)hard[h].m, hard[h].k,
             hard[h].dist);
    printf("HARDEST %llu %d %.6e\n", (unsigned long long)hardest.m, hardest.k,
           hardest.dist);
    return 0;
  }
  if (argc == 3 && !strcmp(argv[1], "check"))
    return check(argv[2]);
  fprintf(stderr, "usage: %s gen | hard | check TABLE\n", argv[0]);
  return 2;
}
