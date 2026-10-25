/* The certificate for `shadow_lens.gp`'s reference values: each one inside an
   Arb ball.

   Neither PARI's 200-digit reals nor Maxima's bigfloats bound their own error,
   so a reference asserted at the last bit of a double needs a radius. This
   reads the rows the PARI script prints, evaluates the same quantities in ball
   arithmetic at 400 bits on the same doubles -- the textbook lens formula, its
   plateaus decided by exact comparisons, and x - sin x -- and requires every
   double-double reference to lie within 2^-100 of the ball, relative. A row
   that cannot be decided fails, as does a run that reads no rows.

       cc -O2 shadow_lens_arb.c -o shadow_lens_arb \
           $(pkg-config --cflags --libs flint)
       gp -q -f shadow_lens.gp </dev/null | ./shadow_lens_arb
*/

#include <flint/arb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PREC 400

/* The numbers of one `(a, b, ...),` row; returns how many were read. */
static int parse(const char *line, double *v, int max) {
  const char *p = strchr(line, '(');
  int n = 0;
  if (!p)
    return 0;
  p++;
  while (n < max) {
    char *end;
    double x = strtod(p, &end);
    if (end == p)
      break;
    v[n++] = x;
    p = end;
    while (*p == ',' || *p == ' ')
      p++;
  }
  return n;
}

/* |x - (hi + lo)| < 2^-100 |x|, certainly; for an exact zero ball, hi = lo = 0.
 */
static int within(const arb_t x, double hi, double lo) {
  arb_t ref, t, tol;
  int ok;
  if (arb_is_zero(x))
    return hi == 0.0 && lo == 0.0;
  arb_init(ref);
  arb_init(t);
  arb_init(tol);
  arb_set_d(ref, hi);
  arb_set_d(t, lo);
  arb_add(ref, ref, t, PREC);
  arb_sub(t, x, ref, PREC);
  arb_abs(t, t);
  arb_abs(tol, x);
  arb_mul_2exp_si(tol, tol, -100);
  ok = arb_lt(t, tol);
  arb_clear(ref);
  arb_clear(t);
  arb_clear(tol);
  return ok;
}

/* r^2 acos((d^2 + r^2 - o^2)/(2 d r)), one term of the textbook area. */
static void sector(arb_t out, const arb_t d, const arb_t r, const arb_t o) {
  arb_t num, den;
  arb_init(num);
  arb_init(den);
  arb_mul(num, d, d, PREC);
  arb_addmul(num, r, r, PREC);
  arb_submul(num, o, o, PREC);
  arb_mul(den, d, r, PREC);
  arb_mul_2exp_si(den, den, 1);
  arb_div(num, num, den, PREC);
  arb_acos(out, num, PREC);
  arb_mul(out, out, r, PREC);
  arb_mul(out, out, r, PREC);
  arb_clear(num);
  arb_clear(den);
}

/* The lens row (re, rs, d, A_hi, A_lo, nu_hi, nu_lo): 1 if both references are
 * inside. */
static int lens_row(const double *v) {
  arb_t re, rs, d, s, m, a, nu, t, pi;
  int ok;
  arb_init(re);
  arb_init(rs);
  arb_init(d);
  arb_init(s);
  arb_init(m);
  arb_init(a);
  arb_init(nu);
  arb_init(t);
  arb_init(pi);
  arb_set_d(re, v[0]);
  arb_set_d(rs, v[1]);
  arb_set_d(d, v[2]);
  arb_const_pi(pi, PREC);
  /* Exact: the operands are doubles within 2^60 of each other, and PREC is 400.
   */
  arb_add(s, re, rs, PREC);
  arb_sub(m, re, rs, PREC);
  arb_abs(m, m);
  if (!arb_is_exact(s) || !arb_is_exact(m)) {
    printf("inexact sum\n");
    exit(2);
  }
  if (arb_ge(d, s)) {
    arb_zero(a);
    arb_one(nu);
  } else if (arb_le(d, m)) {
    if (arb_lt(re, rs)) {
      /* The occulter inside the Sun's disc: an annulus is left. */
      arb_mul(a, re, re, PREC);
      arb_mul(a, a, pi, PREC);
      arb_div(nu, re, rs, PREC);
      arb_mul(nu, nu, nu, PREC);
      arb_sub_si(nu, nu, 1, PREC);
      arb_neg(nu, nu);
    } else {
      arb_mul(a, rs, rs, PREC);
      arb_mul(a, a, pi, PREC);
      arb_zero(nu);
    }
  } else if (arb_lt(d, s) && arb_gt(d, m)) {
    sector(a, d, rs, re);
    sector(t, d, re, rs);
    arb_add(a, a, t, PREC);
    /* sqrt((s - d)(d - re + rs)(d + re - rs)(d + s)) / 2 */
    arb_sub(t, s, d, PREC);
    {
      arb_t f;
      arb_init(f);
      arb_sub(f, d, re, PREC);
      arb_add(f, f, rs, PREC);
      arb_mul(t, t, f, PREC);
      arb_add(f, d, re, PREC);
      arb_sub(f, f, rs, PREC);
      arb_mul(t, t, f, PREC);
      arb_add(f, d, s, PREC);
      arb_mul(t, t, f, PREC);
      arb_clear(f);
    }
    arb_sqrt(t, t, PREC);
    arb_mul_2exp_si(t, t, -1);
    arb_sub(a, a, t, PREC);
    arb_mul(t, rs, rs, PREC);
    arb_mul(t, t, pi, PREC);
    arb_div(nu, a, t, PREC);
    arb_sub_si(nu, nu, 1, PREC);
    arb_neg(nu, nu);
  } else {
    printf("undecided comparison\n");
    exit(2);
  }
  ok = within(a, v[3], v[4]) && within(nu, v[5], v[6]);
  arb_clear(re);
  arb_clear(rs);
  arb_clear(d);
  arb_clear(s);
  arb_clear(m);
  arb_clear(a);
  arb_clear(nu);
  arb_clear(t);
  arb_clear(pi);
  return ok;
}

/* The row (x, hi, lo) of x - sin x. */
static int series_row(const double *v) {
  arb_t x, t;
  int ok;
  arb_init(x);
  arb_init(t);
  arb_set_d(x, v[0]);
  arb_sin(t, x, PREC);
  arb_sub(t, x, t, PREC);
  ok = within(t, v[1], v[2]);
  arb_clear(x);
  arb_clear(t);
  return ok;
}

int main(void) {
  char line[4096];
  long lens = 0, series = 0, bad = 0;
  while (fgets(line, sizeof line, stdin)) {
    double v[8];
    int n = parse(line, v, 8);
    int ok;
    if (n == 8) {
      ok = lens_row(v);
      lens++;
    } else if (n == 3) {
      ok = series_row(v);
      series++;
    } else {
      continue;
    }
    if (!ok) {
      bad++;
      printf("outside its ball: %s", line);
    }
  }
  printf("%ld lens rows and %ld x - sin x rows, %ld outside their Arb ball\n",
         lens, series, bad);
  flint_cleanup();
  return bad != 0 || lens == 0 || series == 0;
}
