/* The certificate for spacelab-fa's exact tables (plans/17 Ф28.A): every
   25-digit literal of ARGUMENTS_EXACT, ERA_EXACT, XY_EXACT, S_EXACT, NUT_EXACT,
   TDB_EXACT, and of spacelab-frames' POM_EXACT and CHAIN_EXACT, is its value
   correctly rounded to the digits it shows, decided in Arb ball arithmetic.

   `fundamental_arguments.mac`, `cio_nutation_series.mac`, `tdb_series.mac` and
   `gcrf_itrf_chain.mac` wrote those literals from 40-digit bigfloats, which
   bound nothing. This
   reads the coefficients `fa_series_arb.mac` prints as exact rationals and
   evaluates the same quantities at 320 bits: the IERS (2010) eqs (5.43) and
   (5.44) reduced with C's fmod, the ERA reduced to one turn, X and Y by eq
   (5.16), s by eq (5.18) at the doubles of XY_EXACT, the IAU 2000A nutation
   and its IAU 2006 rescaling, the Fairhead-Bretagnon TDB - TT, the transposed
   polar-motion matrix, and the GCRF -> ITRF chain W^T R3(ERA) Q of eq (5.1)
   on the exact X, Y and s with the pole offsets added. Each
   literal L of a value V must satisfy |V - L| + rad < half a unit in L's last
   digit. A literal that cannot be decided fails, as does a run that reads
   fewer than the 266 literals or any block short.

     cc -O2 fa_series_arb.c -o fa_series_arb \
         $(pkg-config --cflags --libs flint)
     timeout 600 maxima --very-quiet -Q -b fa_series_arb.mac | ./fa_series_arb
*/

#include <flint/arb.h>
#include <flint/arb_mat.h>
#include <flint/fmpq.h>
#include <flint/fmpz.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PREC 320
#define MAXTOK 128
#define NEPOCH 6
#define NSINGLE                                                                \
  17 /* EPOCHS PA ERA ERADAYS XP YP SPOLY LSELP LSD K06                        \
       TDBEND JPLT2 TDBT POMIN CHAINUT1 CHAINEOP SPRIME */

typedef struct {
  int m[14];
  int code;
  fmpq_t amp;
} xy_term;
typedef struct {
  int m[8];
  int pt;
  fmpq_t s, c;
} s_term;
typedef struct {
  int m[13];
  fmpq_t k[6];
} n_term;

static fmpq_t epochs[NEPOCH], del[5][5], pla[8][2], pa[2], era_c[2],
    era_days[NEPOCH];
static fmpq_t xp[6], yp[6], spoly[6], lselp[5], lsd[5], pl_lin[5][2], k06[2];
static xy_term *xyt;
static s_term *st;
static n_term *ls, *pl;
static int nxy, ns, nls, npl;
static char *lit_args[84], *lit_era[NEPOCH], *lit_xy[2 * NEPOCH],
    *lit_s[NEPOCH], *lit_nut[4 * NEPOCH];
static fmpq_t fh[787][3], jplm[4][3], jpl_t2, tdb_t[8];
static int tdb_end[5], nfh, njpl;
static char *lit_tdb[8];
static fmpq_t pom_in[24], chain_ut1[NEPOCH], chain_eop[4 * NEPOCH], sprime;
static char *lit_pom[72], *lit_chain[9 * NEPOCH];
/* the exact CIP X, Y and the series part of s at each epoch, for the chain */
static arb_t cip_x[NEPOCH], cip_y[NEPOCH], s_series[NEPOCH];
static int checked, failed;
static double worst_margin;

static void die(const char *what) {
  fprintf(stderr, "fa_series_arb: %s\n", what);
  exit(1);
}

static void rat(fmpq_t q, const char *tok) {
  if (fmpq_set_str(q, tok, 10) != 0)
    die(tok);
}

static int split(char *line, char **tok) {
  int n = 0;
  for (char *p = strtok(line, " \t\r\n"); p && n < MAXTOK;
       p = strtok(NULL, " \t\r\n"))
    tok[n++] = p;
  return n;
}

static void rats(fmpq_t *dst, char **tok, int n, int want, const char *tag) {
  if (n != want)
    die(tag);
  for (int i = 0; i < n; i++)
    rat(dst[i], tok[i]);
}

static void lits(char **dst, char **tok, int n, int want, const char *tag) {
  if (n != want)
    die(tag);
  for (int i = 0; i < n; i++)
    dst[i] = strdup(tok[i]);
}

/* sum c[k] t^k, exact */
static void poly_q(fmpq_t out, fmpq_t *c, int n, const fmpq_t t) {
  fmpq_zero(out);
  for (int k = n - 1; k >= 0; k--) {
    fmpq_mul(out, out, t);
    fmpq_add(out, out, c[k]);
  }
}

static void arcsec_to_rad(arb_t r, const fmpq_t x) {
  arb_t pi;
  arb_init(pi);
  arb_const_pi(pi, PREC);
  arb_set_fmpq(r, x, PREC);
  arb_mul(r, r, pi, PREC);
  arb_div_ui(r, r, 648000, PREC);
  arb_clear(pi);
}

static void arcsec_poly_rad(arb_t r, fmpq_t *c, int n, const fmpq_t t) {
  fmpq_t x;
  fmpq_init(x);
  poly_q(x, c, n, t);
  arcsec_to_rad(r, x);
  fmpq_clear(x);
}

static void linear_rad(arb_t r, fmpq_t *c, const fmpq_t t) {
  fmpq_t x;
  fmpq_init(x);
  poly_q(x, c, 2, t);
  arb_set_fmpq(r, x, PREC);
  fmpq_clear(x);
}

/* The fourteen arguments of eqs (5.43), (5.44), unreduced, in radians. */
static void arguments(arb_t *a, const fmpq_t t) {
  for (int k = 0; k < 5; k++)
    arcsec_poly_rad(a[k], del[k], 5, t);
  for (int k = 0; k < 8; k++)
    linear_rad(a[5 + k], pla[k], t);
  fmpq_t x;
  fmpq_init(x);
  poly_q(x, pa, 2, t);
  fmpq_mul(x, x, t);
  arb_set_fmpq(a[13], x, PREC);
  fmpq_clear(x);
}

/* x - trunc(x / m) m for a ball x and m = 2 pi, the integer decided or the run
 * fails */
static void fmod_two_pi(arb_t r, const arb_t x) {
  arb_t m, y;
  fmpz_t z;
  arb_init(m);
  arb_init(y);
  fmpz_init(z);
  arb_const_pi(m, PREC);
  arb_mul_2exp_si(m, m, 1);
  arb_div(y, x, m, PREC);
  if (arb_is_nonnegative(y))
    arb_floor(y, y, PREC);
  else if (arb_is_negative(y))
    arb_ceil(y, y, PREC);
  else
    die("fmod: the sign of a planetary argument is undecided");
  if (!arb_get_unique_fmpz(z, y))
    die("fmod: the turn count of a planetary argument is undecided");
  arb_mul_fmpz(y, m, z, PREC);
  arb_sub(r, x, y, PREC);
  arb_clear(m);
  arb_clear(y);
  fmpz_clear(z);
}

/* L is V rounded to its own digits: |V - L| + rad < half a unit in L's last
 * place. */
static void check(const char *what, int i, const arb_t v, const char *lit) {
  /* an exact zero is written 0.0 and must be one */
  if (!strcmp(lit, "0.0") || !strcmp(lit, "-0.0")) {
    if (!arb_is_zero(v)) {
      printf("UNDECIDED %s[%d]: literal %s, value ", what, i, lit);
      arb_printn(v, 40, 0);
      printf("\n");
      failed++;
    }
    checked++;
    return;
  }
  const char *e = strpbrk(lit, "eE");
  const char *dot = strchr(lit, '.');
  if (!e || !dot)
    die(lit);
  long exp10 = strtol(e + 1, NULL, 10);
  long digits = (long)(e - dot - 1);
  arb_t l, d, h;
  arb_init(l);
  arb_init(d);
  arb_init(h);
  if (arb_set_str(l, lit, PREC) != 0)
    die(lit);
  arb_sub(d, v, l, PREC);
  arb_abs(d, d);
  long k = exp10 - digits;
  arb_set_ui(h, 10);
  arb_pow_ui(h, h, (ulong)(k < 0 ? -k : k), PREC);
  if (k < 0)
    arb_inv(h, h, PREC);
  arb_mul_2exp_si(h, h, -1);
  if (!arb_lt(d, h)) {
    printf("UNDECIDED %s[%d]: literal %s, value ", what, i, lit);
    arb_printn(v, 40, 0);
    printf("\n");
    failed++;
  } else {
    arb_div(d, d, h, PREC);
    double margin =
        arf_get_d(arb_midref(d), ARF_RND_UP) + mag_get_d(arb_radref(d));
    if (margin > worst_margin)
      worst_margin = margin;
  }
  checked++;
  arb_clear(l);
  arb_clear(d);
  arb_clear(h);
}

static void sum_term(arb_t acc, const arb_t coef, const arb_t trig) {
  arb_t v;
  arb_init(v);
  arb_mul(v, coef, trig, PREC);
  arb_add(acc, acc, v, PREC);
  arb_clear(v);
}

static void dot(arb_t arg, const int *m, arb_t *a, int n) {
  arb_zero(arg);
  for (int k = 0; k < n; k++)
    if (m[k] != 0)
      arb_addmul_si(arg, a[k], m[k], PREC);
}

static void epoch(int e) {
  const fmpq_t *t = (const fmpq_t *)&epochs[e];
  arb_t a[14], r, arg, s, c, coef, tt, x, y, dp, de, w[6];
  for (int k = 0; k < 14; k++)
    arb_init(a[k]);
  arb_init(r), arb_init(arg), arb_init(s), arb_init(c), arb_init(coef),
      arb_init(tt);
  arb_init(x), arb_init(y), arb_init(dp), arb_init(de);
  for (int j = 0; j < 6; j++)
    arb_init(w[j]);
  arguments(a, *t);
  arb_set_fmpq(tt, *t, PREC);

  /* ARGUMENTS_EXACT: the Delaunay arguments reduced in arcseconds, exactly; the
     planetary ones modulo 2 pi; p_A as is */
  fmpq_t xq;
  fmpz_t z;
  fmpq_init(xq);
  fmpz_init(z);
  for (int k = 0; k < 5; k++) {
    poly_q(xq, del[k], 5, *t);
    fmpz_mul_ui(z, fmpq_denref(xq), 1296000);
    fmpz_tdiv_q(z, fmpq_numref(xq), z);
    fmpz_mul_ui(z, z, 1296000);
    fmpq_sub_fmpz(xq, xq, z);
    arcsec_to_rad(r, xq);
    check("ARGUMENTS_EXACT", 14 * e + k, r, lit_args[14 * e + k]);
  }
  for (int k = 5; k < 13; k++) {
    fmod_two_pi(r, a[k]);
    check("ARGUMENTS_EXACT", 14 * e + k, r, lit_args[14 * e + k]);
  }
  check("ARGUMENTS_EXACT", 14 * e + 13, a[13], lit_args[14 * e + 13]);

  /* ERA_EXACT: 2 pi (c0 + c1 Tu mod 1) */
  poly_q(xq, era_c, 2, era_days[e]);
  fmpz_fdiv_q(z, fmpq_numref(xq), fmpq_denref(xq));
  fmpq_sub_fmpz(xq, xq, z);
  arb_set_fmpq(r, xq, PREC);
  arb_const_pi(arg, PREC);
  arb_mul_2exp_si(arg, arg, 1);
  arb_mul(r, r, arg, PREC);
  check("ERA_EXACT", e, r, lit_era[e]);
  fmpq_clear(xq);
  fmpz_clear(z);

  /* XY_EXACT: eq (5.16), amplitudes in microarcseconds */
  arb_zero(x);
  arb_zero(y);
  for (int i = 0; i < nxy; i++) {
    dot(arg, xyt[i].m, a, 14);
    arb_set_fmpq(coef, xyt[i].amp, PREC);
    arb_pow_ui(r, tt, (unsigned)(xyt[i].code >> 2), PREC);
    arb_mul(coef, coef, r, PREC);
    if ((xyt[i].code >> 1) & 1)
      arb_cos(c, arg, PREC);
    else
      arb_sin(c, arg, PREC);
    sum_term((xyt[i].code & 1) ? y : x, coef, c);
  }
  for (int v = 0; v < 2; v++) {
    fmpq_t p;
    fmpq_init(p);
    poly_q(p, v == 0 ? xp : yp, 6, *t);
    arb_set_fmpq(r, p, PREC);
    arb_div_ui(coef, v == 0 ? x : y, 1000000, PREC);
    arb_add(r, r, coef, PREC);
    arb_const_pi(coef, PREC);
    arb_mul(r, r, coef, PREC);
    arb_div_ui(r, r, 648000, PREC);
    check("XY_EXACT", 2 * e + v, r, lit_xy[2 * e + v]);
    arb_set(v == 0 ? cip_x[e] : cip_y[e], r);
    fmpq_clear(p);
  }

  /* S_EXACT: eq (5.18) at the doubles XY_EXACT parses to */
  arb_t *a8[8] = {&a[0], &a[1], &a[2], &a[3], &a[4], &a[6], &a[7], &a[13]};
  arb_t b8[8];
  for (int k = 0; k < 8; k++) {
    arb_init(b8[k]);
    arb_set(b8[k], *a8[k]);
  }
  for (int j = 0; j < 6; j++)
    arb_set_fmpq(w[j], spoly[j], PREC);
  for (int i = 0; i < ns; i++) {
    dot(arg, st[i].m, b8, 8);
    arb_sin_cos(s, c, arg, PREC);
    arb_set_fmpq(coef, st[i].s, PREC);
    sum_term(w[st[i].pt], coef, s);
    arb_set_fmpq(coef, st[i].c, PREC);
    sum_term(w[st[i].pt], coef, c);
  }
  arb_zero(r);
  for (int j = 5; j >= 0; j--) {
    arb_mul(r, r, tt, PREC);
    arb_add(r, r, w[j], PREC);
  }
  arb_const_pi(coef, PREC);
  arb_mul(r, r, coef, PREC);
  arb_div_ui(r, r, 648000, PREC);
  arb_set(s_series[e], r);
  arb_set_d(x, strtod(lit_xy[2 * e], NULL));
  arb_set_d(y, strtod(lit_xy[2 * e + 1], NULL));
  arb_mul(x, x, y, PREC);
  arb_mul_2exp_si(x, x, -1);
  arb_sub(r, r, x, PREC);
  check("S_EXACT", e, r, lit_s[e]);
  for (int k = 0; k < 8; k++)
    arb_clear(b8[k]);

  /* NUT_EXACT: the luni-solar series on MHB2000's l' and D, the planetary on
     its linear arguments, in 0.1 microarcseconds; then the IAU 2006 factors */
  arb_t lsa[5], pla_[13];
  for (int k = 0; k < 5; k++)
    arb_init(lsa[k]);
  for (int k = 0; k < 13; k++)
    arb_init(pla_[k]);
  arb_set(lsa[0], a[0]);
  arcsec_poly_rad(lsa[1], lselp, 5, *t);
  arb_set(lsa[2], a[2]);
  arcsec_poly_rad(lsa[3], lsd, 5, *t);
  arb_set(lsa[4], a[4]);
  for (int k = 0; k < 4; k++)
    linear_rad(pla_[k], pl_lin[k], *t);
  for (int k = 0; k < 7; k++)
    arb_set(pla_[4 + k], a[5 + k]);
  linear_rad(pla_[11], pl_lin[4], *t);
  arb_set(pla_[12], a[13]);
  arb_zero(dp);
  arb_zero(de);
  for (int i = 0; i < nls; i++) {
    dot(arg, ls[i].m, lsa, 5);
    arb_sin_cos(s, c, arg, PREC);
    arb_set_fmpq(coef, ls[i].k[1], PREC);
    arb_mul(coef, coef, tt, PREC);
    arb_set_fmpq(r, ls[i].k[0], PREC);
    arb_add(coef, coef, r, PREC);
    sum_term(dp, coef, s);
    arb_set_fmpq(coef, ls[i].k[2], PREC);
    sum_term(dp, coef, c);
    arb_set_fmpq(coef, ls[i].k[4], PREC);
    arb_mul(coef, coef, tt, PREC);
    arb_set_fmpq(r, ls[i].k[3], PREC);
    arb_add(coef, coef, r, PREC);
    sum_term(de, coef, c);
    arb_set_fmpq(coef, ls[i].k[5], PREC);
    sum_term(de, coef, s);
  }
  for (int i = 0; i < npl; i++) {
    dot(arg, pl[i].m, pla_, 13);
    arb_sin_cos(s, c, arg, PREC);
    arb_set_fmpq(coef, pl[i].k[0], PREC);
    sum_term(dp, coef, s);
    arb_set_fmpq(coef, pl[i].k[1], PREC);
    sum_term(dp, coef, c);
    arb_set_fmpq(coef, pl[i].k[2], PREC);
    sum_term(de, coef, s);
    arb_set_fmpq(coef, pl[i].k[3], PREC);
    sum_term(de, coef, c);
  }
  arb_const_pi(coef, PREC);
  for (int v = 0; v < 2; v++) {
    arb_ptr q = v == 0 ? dp : de;
    arb_mul(q, q, coef, PREC);
    arb_div_ui(q, q, 648000, PREC);
    arb_div_ui(q, q, 10000000, PREC);
  }
  check("NUT_EXACT", 4 * e + 0, dp, lit_nut[4 * e + 0]);
  check("NUT_EXACT", 4 * e + 1, de, lit_nut[4 * e + 1]);
  arb_set_fmpq(r, k06[1], PREC);
  arb_mul(r, r, tt, PREC);
  arb_add_ui(r, r, 1, PREC);
  arb_mul(de, de, r, PREC);
  arb_set_fmpq(coef, k06[0], PREC);
  arb_add(r, r, coef, PREC);
  arb_mul(dp, dp, r, PREC);
  check("NUT_EXACT", 4 * e + 2, dp, lit_nut[4 * e + 2]);
  check("NUT_EXACT", 4 * e + 3, de, lit_nut[4 * e + 3]);
  for (int k = 0; k < 5; k++)
    arb_clear(lsa[k]);
  for (int k = 0; k < 13; k++)
    arb_clear(pla_[k]);

  for (int k = 0; k < 14; k++)
    arb_clear(a[k]);
  arb_clear(r), arb_clear(arg), arb_clear(s), arb_clear(c), arb_clear(coef),
      arb_clear(tt);
  arb_clear(x), arb_clear(y), arb_clear(dp), arb_clear(de);
  for (int j = 0; j < 6; j++)
    arb_clear(w[j]);
}

/* A sin(w t + phi), A in seconds */
static void tdb_term(arb_t acc, fmpq_t *r, const fmpq_t t) {
  fmpq_t x;
  arb_t a, sn;
  fmpq_init(x);
  arb_init(a);
  arb_init(sn);
  fmpq_mul(x, r[1], t);
  fmpq_add(x, x, r[2]);
  arb_set_fmpq(a, x, PREC);
  arb_sin(sn, a, PREC);
  arb_set_fmpq(a, r[0], PREC);
  sum_term(acc, a, sn);
  fmpq_clear(x);
  arb_clear(a);
  arb_clear(sn);
}

/* TDB_EXACT: five blocks multiplying t^0 .. t^4, the JPL-mass terms and t^2 */
static void tdb(int e) {
  arb_t w, g, tt;
  arb_init(w);
  arb_init(g);
  arb_init(tt);
  arb_set_fmpq(tt, tdb_t[e], PREC);
  for (int k = 4, lo; k >= 0; k--) {
    lo = k == 0 ? 0 : tdb_end[k - 1];
    arb_zero(g);
    for (int i = lo; i < tdb_end[k]; i++)
      tdb_term(g, fh[i], tdb_t[e]);
    arb_mul(w, w, tt, PREC);
    arb_add(w, w, g, PREC);
  }
  for (int i = 0; i < 4; i++)
    tdb_term(w, jplm[i], tdb_t[e]);
  arb_set_fmpq(g, jpl_t2, PREC);
  arb_mul(g, g, tt, PREC);
  arb_mul(g, g, tt, PREC);
  arb_add(w, w, g, PREC);
  check("TDB_EXACT", e, w, lit_tdb[e]);
  arb_clear(w);
  arb_clear(g);
  arb_clear(tt);
}

/* acc <- acc R(a), R1, R2, R3 the passive frame rotations of cio_rotation.mac
   for axis 0, 1, 2 */
static void rot_mul(arb_mat_t acc, int axis, const arb_t a) {
  int i = (axis + 1) % 3, j = (axis + 2) % 3;
  arb_mat_t r;
  arb_t s, c;
  arb_mat_init(r, 3, 3);
  arb_init(s);
  arb_init(c);
  arb_sin_cos(s, c, a, PREC);
  arb_mat_one(r);
  arb_set(arb_mat_entry(r, i, i), c);
  arb_set(arb_mat_entry(r, j, j), c);
  arb_set(arb_mat_entry(r, i, j), s);
  arb_neg(arb_mat_entry(r, j, i), s);
  arb_mat_mul(acc, acc, r, PREC);
  arb_mat_clear(r);
  arb_clear(s);
  arb_clear(c);
}

static void check_matrix(const char *what, int row, const arb_mat_t m,
                         char **lit) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      check(what, 9 * row + 3 * i + j, arb_mat_entry(m, i, j),
            lit[9 * row + 3 * i + j]);
}

/* POM_EXACT: W^T = R1(-yp) R2(-xp) R3(s') at the POM_INPUTS doubles */
static void pom(int k) {
  arb_mat_t w;
  arb_t a;
  arb_mat_init(w, 3, 3);
  arb_init(a);
  arb_mat_one(w);
  arb_set_fmpq(a, pom_in[3 * k + 1], PREC);
  arb_neg(a, a);
  rot_mul(w, 0, a);
  arb_set_fmpq(a, pom_in[3 * k], PREC);
  arb_neg(a, a);
  rot_mul(w, 1, a);
  arb_set_fmpq(a, pom_in[3 * k + 2], PREC);
  rot_mul(w, 2, a);
  check_matrix("POM_EXACT", k, w, lit_pom);
  arb_mat_clear(w);
  arb_clear(a);
}

/* CHAIN_EXACT: W^T(xp, yp, s'(t)) R3(ERA(UT1)) R3(-(E + s)) R2(d) R3(E), with
   E = atan2(Y', X'), d = atan(sqrt(r^2 / (1 - r^2))), X' and Y' the series
   plus dX, dY, and s eq (5.18) on the series X, Y */
static void chain(int e) {
  arb_mat_t m;
  arb_t a, x, y, s, r2, d, big_e;
  fmpq_t q;
  fmpz_t z;
  arb_mat_init(m, 3, 3);
  arb_init(a), arb_init(x), arb_init(y), arb_init(s), arb_init(r2), arb_init(d),
      arb_init(big_e);
  fmpq_init(q);
  fmpz_init(z);
  arb_mat_one(m);
  arcsec_to_rad(a, chain_eop[4 * e + 1]);
  arb_neg(a, a);
  rot_mul(m, 0, a);
  arcsec_to_rad(a, chain_eop[4 * e]);
  arb_neg(a, a);
  rot_mul(m, 1, a);
  fmpq_mul(q, sprime, epochs[e]);
  arcsec_to_rad(a, q);
  rot_mul(m, 2, a);
  poly_q(q, era_c, 2, chain_ut1[e]);
  fmpz_fdiv_q(z, fmpq_numref(q), fmpq_denref(q));
  fmpq_sub_fmpz(q, q, z);
  arb_set_fmpq(a, q, PREC);
  arb_const_pi(d, PREC);
  arb_mul_2exp_si(d, d, 1);
  arb_mul(a, a, d, PREC);
  rot_mul(m, 2, a);
  arcsec_to_rad(a, chain_eop[4 * e + 2]);
  arb_add(x, cip_x[e], a, PREC);
  arcsec_to_rad(a, chain_eop[4 * e + 3]);
  arb_add(y, cip_y[e], a, PREC);
  arb_mul(s, cip_x[e], cip_y[e], PREC);
  arb_mul_2exp_si(s, s, -1);
  arb_sub(s, s_series[e], s, PREC);
  arb_atan2(big_e, y, x, PREC);
  arb_sqr(r2, x, PREC);
  arb_addmul(r2, y, y, PREC);
  arb_sub_ui(a, r2, 1, PREC);
  arb_neg(a, a);
  arb_div(d, r2, a, PREC);
  arb_sqrt(d, d, PREC);
  arb_atan(d, d, PREC);
  arb_add(a, big_e, s, PREC);
  arb_neg(a, a);
  rot_mul(m, 2, a);
  rot_mul(m, 1, d);
  rot_mul(m, 2, big_e);
  check_matrix("CHAIN_EXACT", e, m, lit_chain);
  arb_mat_clear(m);
  arb_clear(a), arb_clear(x), arb_clear(y), arb_clear(s), arb_clear(r2),
      arb_clear(d), arb_clear(big_e);
  fmpq_clear(q);
  fmpz_clear(z);
}

int main(void) {
  char *line = NULL, *tok[MAXTOK];
  size_t cap = 0;
  int begun = 0, ended = 0, nd = 0, np = 0, nlin = 0;
  unsigned seen = 0; /* one bit per single-line tag */
  int cxy = 8192, cst = 128, cls = 1024, cpl = 1024;
  xyt = malloc(sizeof *xyt * cxy);
  st = malloc(sizeof *st * cst);
  ls = malloc(sizeof *ls * cls);
  pl = malloc(sizeof *pl * cpl);
  for (int i = 0; i < NEPOCH; i++)
    fmpq_init(epochs[i]), fmpq_init(era_days[i]);
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++)
      fmpq_init(del[i][j]);
  for (int i = 0; i < 8; i++)
    fmpq_init(pla[i][0]), fmpq_init(pla[i][1]);
  for (int i = 0; i < 6; i++)
    fmpq_init(xp[i]), fmpq_init(yp[i]), fmpq_init(spoly[i]);
  for (int i = 0; i < 5; i++)
    fmpq_init(lselp[i]), fmpq_init(lsd[i]), fmpq_init(pl_lin[i][0]),
        fmpq_init(pl_lin[i][1]);
  fmpq_init(pa[0]), fmpq_init(pa[1]), fmpq_init(era_c[0]), fmpq_init(era_c[1]);
  fmpq_init(k06[0]), fmpq_init(k06[1]);
  for (int i = 0; i < 787; i++)
    for (int j = 0; j < 3; j++)
      fmpq_init(fh[i][j]);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++)
      fmpq_init(jplm[i][j]);
  for (int i = 0; i < 8; i++)
    fmpq_init(tdb_t[i]);
  fmpq_init(jpl_t2);
  for (int i = 0; i < 24; i++)
    fmpq_init(pom_in[i]), fmpq_init(chain_eop[i]);
  for (int i = 0; i < NEPOCH; i++)
    fmpq_init(chain_ut1[i]), arb_init(cip_x[i]), arb_init(cip_y[i]),
        arb_init(s_series[i]);
  fmpq_init(sprime);

  while (getline(&line, &cap, stdin) != -1) {
    int n = split(line, tok);
    if (n == 0)
      continue;
    if (!begun) {
      begun = strcmp(tok[0], "BEGIN") == 0;
      continue;
    }
    char *tag = tok[0];
    /* Maxima's batch mode echoes each statement; a tag is upper case, digits
     * and '_' only */
    if (strspn(tag, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != strlen(tag))
      continue;
    char **v = tok + 1;
    n--;
    if (!strcmp(tag, "END")) {
      ended = 1;
      break;
    } else if (!strcmp(tag, "EPOCHS") && (seen |= 1u << 0))
      rats(epochs, v, n, NEPOCH, tag);
    else if (!strcmp(tag, "DELAUNAY") && nd < 5)
      rats(del[nd++], v, n, 5, tag);
    else if (!strcmp(tag, "PLANETARY") && np < 8)
      rats(pla[np++], v, n, 2, tag);
    else if (!strcmp(tag, "PA") && (seen |= 1u << 1))
      rats(pa, v, n, 2, tag);
    else if (!strcmp(tag, "ERA") && (seen |= 1u << 2))
      rats(era_c, v, n, 2, tag);
    else if (!strcmp(tag, "ERADAYS") && (seen |= 1u << 3))
      rats(era_days, v, n, NEPOCH, tag);
    else if (!strcmp(tag, "XP") && (seen |= 1u << 4))
      rats(xp, v, n, 6, tag);
    else if (!strcmp(tag, "YP") && (seen |= 1u << 5))
      rats(yp, v, n, 6, tag);
    else if (!strcmp(tag, "SPOLY") && (seen |= 1u << 6))
      rats(spoly, v, n, 6, tag);
    else if (!strcmp(tag, "LSELP") && (seen |= 1u << 7))
      rats(lselp, v, n, 5, tag);
    else if (!strcmp(tag, "LSD") && (seen |= 1u << 8))
      rats(lsd, v, n, 5, tag);
    else if (!strncmp(tag, "PL_A", 4) && nlin < 5)
      rats(pl_lin[nlin++], v, n, 2, tag);
    else if (!strcmp(tag, "K06") && (seen |= 1u << 9))
      rats(k06, v, n, 2, tag);
    else if (!strcmp(tag, "XYT")) {
      if (n != 16 || nxy == cxy)
        die(tag);
      for (int k = 0; k < 14; k++)
        xyt[nxy].m[k] = atoi(v[k]);
      xyt[nxy].code = atoi(v[14]);
      fmpq_init(xyt[nxy].amp);
      rat(xyt[nxy].amp, v[15]);
      nxy++;
    } else if (!strcmp(tag, "ST")) {
      if (n != 11 || ns == cst)
        die(tag);
      for (int k = 0; k < 8; k++)
        st[ns].m[k] = atoi(v[k]);
      st[ns].pt = atoi(v[8]);
      if (st[ns].pt < 0 || st[ns].pt > 5)
        die(tag);
      fmpq_init(st[ns].s), fmpq_init(st[ns].c);
      rat(st[ns].s, v[9]);
      rat(st[ns].c, v[10]);
      ns++;
    } else if (!strcmp(tag, "LS")) {
      if (n != 11 || nls == cls)
        die(tag);
      for (int k = 0; k < 5; k++)
        ls[nls].m[k] = atoi(v[k]);
      for (int k = 0; k < 6; k++) {
        fmpq_init(ls[nls].k[k]);
        rat(ls[nls].k[k], v[5 + k]);
      }
      nls++;
    } else if (!strcmp(tag, "PL")) {
      if (n != 17 || npl == cpl)
        die(tag);
      for (int k = 0; k < 13; k++)
        pl[npl].m[k] = atoi(v[k]);
      for (int k = 0; k < 4; k++) {
        fmpq_init(pl[npl].k[k]);
        rat(pl[npl].k[k], v[13 + k]);
      }
      npl++;
    } else if (!strcmp(tag, "LIT_ARGUMENTS"))
      lits(lit_args, v, n, 84, tag);
    else if (!strcmp(tag, "LIT_ERA"))
      lits(lit_era, v, n, NEPOCH, tag);
    else if (!strcmp(tag, "LIT_XY"))
      lits(lit_xy, v, n, 2 * NEPOCH, tag);
    else if (!strcmp(tag, "LIT_S"))
      lits(lit_s, v, n, NEPOCH, tag);
    else if (!strcmp(tag, "LIT_NUT"))
      lits(lit_nut, v, n, 4 * NEPOCH, tag);
    else if (!strcmp(tag, "TDBEND") && (seen |= 1u << 10)) {
      if (n != 5)
        die(tag);
      for (int k = 0; k < 5; k++)
        tdb_end[k] = atoi(v[k]);
    } else if (!strcmp(tag, "FH") && nfh < 787)
      rats(fh[nfh++], v, n, 3, tag);
    else if (!strcmp(tag, "JPL") && njpl < 4)
      rats(jplm[njpl++], v, n, 3, tag);
    else if (!strcmp(tag, "JPLT2") && (seen |= 1u << 11))
      rats(&jpl_t2, v, n, 1, tag);
    else if (!strcmp(tag, "TDBT") && (seen |= 1u << 12))
      rats(tdb_t, v, n, 8, tag);
    else if (!strcmp(tag, "LIT_TDB"))
      lits(lit_tdb, v, n, 8, tag);
    else if (!strcmp(tag, "POMIN") && (seen |= 1u << 13))
      rats(pom_in, v, n, 24, tag);
    else if (!strcmp(tag, "CHAINUT1") && (seen |= 1u << 14))
      rats(chain_ut1, v, n, NEPOCH, tag);
    else if (!strcmp(tag, "CHAINEOP") && (seen |= 1u << 15))
      rats(chain_eop, v, n, 4 * NEPOCH, tag);
    else if (!strcmp(tag, "SPRIME") && (seen |= 1u << 16))
      rats(&sprime, v, n, 1, tag);
    else if (!strcmp(tag, "LIT_POM"))
      lits(lit_pom, v, n, 72, tag);
    else if (!strcmp(tag, "LIT_CHAIN"))
      lits(lit_chain, v, n, 9 * NEPOCH, tag);
    /* what tdb_series.mac prints of its own between the blocks */
    else if (strcmp(tag, "TDB_EXACT") && strcmp(tag, "DONE"))
      die(tag);
  }
  if (!ended || nd != 5 || np != 8 || nlin != 5 || nxy != 4755 || ns != 66 ||
      nls != 678 || npl != 687 || seen != (1u << NSINGLE) - 1 || !lit_args[0] ||
      !lit_era[0] || !lit_xy[0] || !lit_s[0] || !lit_nut[0] || nfh != 787 ||
      njpl != 4 || tdb_end[4] != 787 || !lit_tdb[0] || !lit_pom[0] ||
      !lit_chain[0])
    die("incomplete input");
  for (int e = 0; e < NEPOCH; e++)
    epoch(e);
  for (int e = 0; e < 8; e++)
    tdb(e);
  for (int k = 0; k < 8; k++)
    pom(k);
  for (int e = 0; e < NEPOCH; e++)
    chain(e);
  if (failed || checked != 266) {
    printf("FAILED: %d of %d literals undecided or wrong\n", failed, checked);
    return 1;
  }
  printf("CERTIFIED %d literals; the worst lies %.3g of a half unit in its "
         "last digit from its ball\n",
         checked, worst_margin);
  flint_cleanup();
  return 0;
}
