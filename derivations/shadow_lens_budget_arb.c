/* The two suprema shadow_lens_budget.mac leaves to interval arithmetic,
   certified by bisection in Arb ball arithmetic (plans/17 Ф28.D).

   (a) g(x) = 2 amp(x) + X(x) + f(x) <= 9.05 on [2, 2 pi], where amp(x) =
       x (1 - cos x)/(x - sin x), X(x) = 1 + 2|sin x|/(x - sin x) the round-off
       of `x - sin(x)` in units of u, f(x) = 4|cos(x/2)| sin^3(x/2)/(x - sin x).
   (b) F(a, b) = B - 2 kappa <= 18 on the triangle's core a, b >= 1/50,
       a + b <= pi, in the chord frame of shadow_lens_budget.mac:
         F = [2 sin(a+b) ((9/2)|cos(a-b)| - cos(a-b) cot a cot b)
              + 2|cos a| sin a + 2|cos b| sin b + 4a + 4b
              + X(2a) A_S(a) + X(2b) A_E(b)] / (A_S + A_E) + 3,
       A_S(a) = (2a - sin 2a)/(2 sin^2 a), X(x) = 1.245 below x = 2.
   A box whose upper bound is not under the target is split in two along its
   longer side; one that is still undecided at width 2^-40 fails the run. A_S
   rises with a (shadow_lens_budget.mac (4)), so its enclosure on a box is the
   hull of its values at the ends, which avoids the cancellation in 2a - sin 2a.
   Boxes that reach past a + b = pi are evaluated whole: F's formula is finite
   there, so their bound covers the part inside.

       cc -O2 shadow_lens_budget_arb.c -o shadow_lens_budget_arb \
           $(pkg-config --cflags --libs flint)
       ./shadow_lens_budget_arb
*/

#include <flint/arb.h>
#include <stdio.h>
#include <stdlib.h>

#define PREC 128
#define MAXBOX 4000000

typedef struct {
  double a0, a1, b0, b1;
} box;

static arb_t X1;

/* x - sin x, 2 sin^2(x/2) = 1 - cos x and the rest at a ball x */
static void x_round_off(arb_t out, const arb_t x) {
  arb_t s, v;
  arb_init(s);
  arb_init(v);
  arb_sin(s, x, PREC);
  arb_sub(v, x, s, PREC);
  arb_abs(s, s);
  arb_mul_2exp_si(s, s, 1);
  arb_div(out, s, v, PREC);
  arb_add_ui(out, out, 1, PREC);
  arb_clear(s);
  arb_clear(v);
}

static void g_of(arb_t out, const arb_t x) {
  arb_t v, t, h, c;
  arb_init(v);
  arb_init(t);
  arb_init(h);
  arb_init(c);
  arb_sin(t, x, PREC);
  arb_sub(v, x, t, PREC); /* x - sin x */
  arb_cos(t, x, PREC);
  arb_sub_ui(t, t, 1, PREC);
  arb_neg(t, t);
  arb_mul(t, t, x, PREC);
  arb_div(t, t, v, PREC);
  arb_mul_2exp_si(out, t, 1); /* 2 amp */
  arb_mul_2exp_si(h, x, -1);
  arb_sin(t, h, PREC);
  arb_pow_ui(t, t, 3, PREC);
  arb_cos(c, h, PREC);
  arb_abs(c, c);
  arb_mul(t, t, c, PREC);
  arb_mul_2exp_si(t, t, 2);
  arb_div(t, t, v, PREC);
  arb_add(out, out, t, PREC); /* + f */
  x_round_off(t, x);
  arb_add(out, out, t, PREC); /* + X */
  arb_clear(v);
  arb_clear(t);
  arb_clear(h);
  arb_clear(c);
}

static void interval(arb_t x, double lo, double hi) {
  arf_t l, h;
  arf_init(l);
  arf_init(h);
  arf_set_d(l, lo);
  arf_set_d(h, hi);
  arb_set_interval_arf(x, l, h, PREC);
  arf_clear(l);
  arf_clear(h);
}

/* A_S at a point a */
static void segment(arb_t out, double a) {
  arb_t x, s;
  arb_init(x);
  arb_init(s);
  arb_set_d(x, a);
  arb_sin(s, x, PREC);
  arb_sqr(s, s, PREC);
  arb_mul_2exp_si(s, s, 1);
  arb_mul_2exp_si(x, x, 1);
  arb_sin(out, x, PREC);
  arb_sub(out, x, out, PREC);
  arb_div(out, out, s, PREC);
  arb_clear(x);
  arb_clear(s);
}

/* X(2a) on a box of a: 1.245 below a = 1 (less a margin for the computed
   angle's rounding), the direct branch's above, their larger where the box
   straddles. */
static void x_of(arb_t out, double a0, double a1) {
  const double edge = 1.0 - 0x1p-40;
  arb_t x;
  if (a1 < edge) {
    arb_set(out, X1);
    return;
  }
  arb_init(x);
  interval(x, 2.0 * (a0 < edge ? edge : a0), 2.0 * a1);
  x_round_off(out, x);
  if (a0 < edge)
    arb_max(out, out, X1, PREC);
  arb_clear(x);
}

static void f_of(arb_t out, const box *q) {
  arb_t a, b, sa, ca, sb, cb, t, u, as, ae, xs;
  arb_init(a);
  arb_init(b);
  arb_init(sa);
  arb_init(ca);
  arb_init(sb);
  arb_init(cb);
  arb_init(t);
  arb_init(u);
  arb_init(as);
  arb_init(ae);
  arb_init(xs);
  interval(a, q->a0, q->a1);
  interval(b, q->b0, q->b1);
  arb_sin_cos(sa, ca, a, PREC);
  arb_sin_cos(sb, cb, b, PREC);
  /* 2 sin(a+b) ((9/2)|cos(a-b)| - cos(a-b) cos a cos b/(sin a sin b)) */
  arb_sub(t, a, b, PREC);
  arb_cos(t, t, PREC);
  arb_mul(u, ca, cb, PREC);
  arb_mul(u, u, t, PREC);
  arb_div(u, u, sa, PREC);
  arb_div(u, u, sb, PREC);
  arb_abs(t, t);
  arb_mul_ui(t, t, 9, PREC);
  arb_mul_2exp_si(t, t, -1);
  arb_sub(t, t, u, PREC);
  arb_add(u, a, b, PREC);
  arb_sin(u, u, PREC);
  arb_mul(t, t, u, PREC);
  arb_mul_2exp_si(out, t, 1);
  /* + 2|cos a| sin a + 2|cos b| sin b + 4a + 4b */
  arb_abs(t, ca);
  arb_mul(t, t, sa, PREC);
  arb_abs(u, cb);
  arb_addmul(t, u, sb, PREC);
  arb_add(t, t, a, PREC);
  arb_add(t, t, a, PREC);
  arb_add(t, t, b, PREC);
  arb_add(t, t, b, PREC);
  arb_mul_2exp_si(t, t, 1);
  arb_add(out, out, t, PREC);
  /* the segments, by monotonicity */
  segment(as, q->a0);
  segment(t, q->a1);
  arb_union(as, as, t, PREC);
  segment(ae, q->b0);
  segment(t, q->b1);
  arb_union(ae, ae, t, PREC);
  x_of(xs, q->a0, q->a1);
  arb_addmul(out, xs, as, PREC);
  x_of(xs, q->b0, q->b1);
  arb_addmul(out, xs, ae, PREC);
  arb_add(t, as, ae, PREC);
  arb_div(out, out, t, PREC);
  arb_add_ui(out, out, 3, PREC);
  arb_clear(a);
  arb_clear(b);
  arb_clear(sa);
  arb_clear(ca);
  arb_clear(sb);
  arb_clear(cb);
  arb_clear(t);
  arb_clear(u);
  arb_clear(as);
  arb_clear(ae);
  arb_clear(xs);
}

static int below(const arb_t v, const arb_t target) {
  return arb_lt(v, target);
}

int main(void) {
  const double eps = 0.02, pi = 3.141592653589793 + 4e-16; /* above pi */
  static box stack[MAXBOX];
  long n = 0, top = 0, ng = 0;
  double best = 0, ba = 0, bb = 0;
  arb_t v, target, x;
  arb_init(v);
  arb_init(target);
  arb_init(x);
  arb_init(X1);
  arb_set_ui(X1, 249);
  arb_div_ui(X1, X1, 200, PREC);

  /* (a) g on [2, 2 pi] */
  arb_set_ui(target, 905);
  arb_div_ui(target, target, 100, PREC);
  {
    double lo[64], hi[64];
    int k = 0;
    lo[k] = 2.0;
    hi[k++] = 2.0 * pi;
    while (k > 0) {
      double l = lo[--k], h = hi[k];
      interval(x, l, h);
      g_of(v, x);
      ng++;
      if (below(v, target))
        continue;
      if (h - l < 0x1p-40 || k + 2 > 64) {
        printf("FAIL g on [%.17g, %.17g]\n", l, h);
        return 1;
      }
      lo[k] = l;
      hi[k++] = 0.5 * (l + h);
      lo[k] = 0.5 * (l + h);
      hi[k++] = h;
    }
  }

  /* (b) F on the core */
  arb_set_ui(target, 18);
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 64; j++) {
      double a0 = eps + (pi - 2 * eps) * i / 64,
             a1 = eps + (pi - 2 * eps) * (i + 1) / 64;
      double b0 = eps + (pi - 2 * eps) * j / 64,
             b1 = eps + (pi - 2 * eps) * (j + 1) / 64;
      if (a0 + b0 >= pi)
        continue;
      stack[top++] = (box){a0, a1, b0, b1};
    }
  while (top > 0) {
    box q = stack[--top];
    f_of(v, &q);
    n++;
    if (below(v, target)) {
      box m = {0.5 * (q.a0 + q.a1), 0.5 * (q.a0 + q.a1), 0.5 * (q.b0 + q.b1),
               0.5 * (q.b0 + q.b1)};
      if (m.a0 + m.b0 < pi) {
        double mid;
        f_of(v, &m);
        mid = arf_get_d(arb_midref(v), ARF_RND_NEAR);
        if (mid > best) {
          best = mid;
          ba = m.a0;
          bb = m.b0;
        }
      }
      continue;
    }
    if ((q.a1 - q.a0 < 0x1p-40 && q.b1 - q.b0 < 0x1p-40) || top + 2 > MAXBOX) {
      printf("FAIL F on a [%.17g, %.17g] b [%.17g, %.17g]\n", q.a0, q.a1, q.b0,
             q.b1);
      return 1;
    }
    if (q.a1 - q.a0 >= q.b1 - q.b0) {
      double m = 0.5 * (q.a0 + q.a1);
      stack[top++] = (box){q.a0, m, q.b0, q.b1};
      if (m + q.b0 < pi)
        stack[top++] = (box){m, q.a1, q.b0, q.b1};
    } else {
      double m = 0.5 * (q.b0 + q.b1);
      stack[top++] = (box){q.a0, q.a1, q.b0, m};
      if (q.a0 + m < pi)
        stack[top++] = (box){q.a0, q.a1, m, q.b1};
    }
  }
  printf("g <= 9.05 on [2, 2 pi] (%ld intervals); B - 2 kappa <= 18 on the "
         "core a, b >= 1/50 "
         "(%ld boxes, largest midpoint value %.4f at a = %.4f, b = %.4f)\n",
         ng, n, best, ba, bb);
  arb_clear(v);
  arb_clear(target);
  arb_clear(x);
  arb_clear(X1);
  flint_cleanup();
  return 0;
}
