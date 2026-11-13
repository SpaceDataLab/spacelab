/* The Arb certificate for the tables derivations write through `certified`
   (rust_literals.mac; plans/17 Ф28.A). A derivation decides each 25-digit
   literal against a 40-digit Maxima bigfloat, which bounds nothing; it also
   prints the literal's exact value, and this program evaluates that value in
   Arb ball arithmetic and requires the literal to be it correctly rounded to
   the digits it shows: |V - L| + rad < half a unit in L's last digit, and a
   literal of zeros an exact zero. An undecided literal is retried at higher
   precision before it fails the run.

   Input: the derivations' output; every line but `ARB name index literal
   token...` is skipped. The tokens are the exact value in reverse Polish
   notation: an integer or a rational p/q; `pi`, `e`; `addN`, `mulN` over the
   top N; `powK` an integer power, `rootQ` the principal Q-th root of a
   non-negative ball, `powx` x^y = exp(y log x) for x > 0; `atan2` of (y, x);
   and sin cos tan asin acos atan exp log abs sinh cosh tanh erf erfc.

   A value may also be a root: `ARBVAR name lo hi f... ; f'...` defines `name`
   as the root of f(x) in [lo, hi] (dyadic rationals; `x` is the token of the
   unknown). It is certified, at each precision, by f(lo) and f(hi) of opposite
   decided signs and f' bounded away from zero on [lo, hi], so the root is
   unique; bisection on decided signs narrows it to a ball, and `$name` pushes
   that ball in any later value. `ARBLET name token...` names a value shared by
   later ones, evaluated once at each precision: a value built by iteration
   (RK4 steps) then stays linear in its length. A root or a let may use only
   names defined on earlier lines.

   Doubles: `ARBF64 name index literal token...` claims that the double the
   decimal literal parses to (correctly rounded, ties to even) is the double
   the value rounds to; `ARBD candidate token...` that the value rounds to the
   double `candidate`, a dyadic rational. Both ends of the value's ball must
   round to the same 53-bit significand (FLINT's ARF_RND_NEAR is ties to even),
   that double must be the claimed one, and normal or zero. With -v it prints
   each ball, its precision and the literal.

     cc -O2 arb_balls.c -o arb_balls $(pkg-config --cflags --libs flint)
     timeout 300 maxima --very-quiet -Q -b derivation.mac | ./arb_balls
*/
#include <ctype.h>
#include <flint/arb.h>
#include <flint/arb_hypgeom.h>
#include <flint/fmpq.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPREC 3
static const slong precs[NPREC] = {256, 1024, 4096};

/* a named value, a root (its bracket, f and f' in RPN) or a let (its value in
   f), and its ball at each precision once certified */
typedef struct {
  char *name;
  int let;
  fmpq_t lo, hi;
  char **f, **d;
  int nf, nd, state[NPREC];
  arb_t ball[NPREC];
} var_t;
static var_t *vars;
static int nvars;
static int checked, failed, verbose, dchecked, dfailed;
static double worst_margin, worst_dmargin;
static slong worst_prec, worst_dprec;

static void die(const char *what, const char *detail) {
  fprintf(stderr, "arb_balls: %s: %s\n", what, detail);
  exit(1);
}

static int is_rational(const char *t) {
  const char *p = t + (*t == '-');
  int digits = 0, slash = 0;
  for (; *p; p++) {
    if (isdigit((unsigned char)*p))
      digits++;
    else if (*p == '/' && digits && !slash)
      slash = 1, digits = 0;
    else
      return 0;
  }
  return digits > 0;
}

static int var_ball(arb_t out, const char *name, int pi);

/* The value of the RPN tokens tok[0..n) at precs[pi] bits, x the unknown's
   ball where a root's f is evaluated (NULL elsewhere); 0 when malformed. */
static int evaluate(arb_t out, char **tok, int n, int pi, const arb_t x) {
  slong prec = precs[pi];
  arb_ptr st = _arb_vec_init(n);
  int sp = 0, ok = 1;
  for (int i = 0; i < n && ok; i++) {
    const char *t = tok[i];
    long k;
    if (!strcmp(t, "x") && x)
      arb_set(st + sp++, x);
    else if (t[0] == '$')
      ok = var_ball(st + sp++, t + 1, pi);
    else if (is_rational(t)) {
      fmpq_t q;
      fmpq_init(q);
      ok = fmpq_set_str(q, t, 10) == 0 && fmpz_sgn(fmpq_denref(q)) > 0;
      if (ok)
        fmpq_canonicalise(q), arb_set_fmpq(st + sp++, q, prec);
      fmpq_clear(q);
    } else if (!strcmp(t, "pi"))
      arb_const_pi(st + sp++, prec);
    else if (!strcmp(t, "e"))
      arb_const_e(st + sp++, prec);
    else if ((!strncmp(t, "add", 3) || !strncmp(t, "mul", 3)) &&
             (k = strtol(t + 3, NULL, 10)) >= 2 && k <= sp) {
      for (long j = sp - k + 1; j < sp; j++)
        if (t[0] == 'a')
          arb_add(st + sp - k, st + sp - k, st + j, prec);
        else
          arb_mul(st + sp - k, st + sp - k, st + j, prec);
      sp -= (int)k - 1;
    } else if (!strncmp(t, "pow", 3) && t[3] != 'x' && sp >= 1) {
      fmpz_t z;
      fmpz_init(z);
      ok = fmpz_set_str(z, t + 3, 10) == 0;
      if (ok)
        arb_pow_fmpz(st + sp - 1, st + sp - 1, z, prec);
      fmpz_clear(z);
    } else if (!strncmp(t, "root", 4) && (k = strtol(t + 4, NULL, 10)) >= 2 &&
               sp >= 1) {
      if (!arb_is_nonnegative(st + sp - 1))
        arb_indeterminate(st + sp - 1);
      else
        arb_root_ui(st + sp - 1, st + sp - 1, (ulong)k, prec);
    } else if (!strcmp(t, "powx") && sp >= 2) {
      /* x^y = exp(y log x), x > 0: arb_pow is not used, F52 */
      if (!arb_is_positive(st + sp - 2))
        arb_indeterminate(st + sp - 2);
      else {
        arb_log(st + sp - 2, st + sp - 2, prec);
        arb_mul(st + sp - 2, st + sp - 2, st + sp - 1, prec);
        arb_exp(st + sp - 2, st + sp - 2, prec);
      }
      sp--;
    } else if (!strcmp(t, "atan2") && sp >= 2) {
      arb_atan2(st + sp - 2, st + sp - 2, st + sp - 1, prec);
      sp--;
    } else if (sp >= 1) {
      arb_ptr y = st + sp - 1;
      if (!strcmp(t, "sin"))
        arb_sin(y, y, prec);
      else if (!strcmp(t, "cos"))
        arb_cos(y, y, prec);
      else if (!strcmp(t, "tan"))
        arb_tan(y, y, prec);
      else if (!strcmp(t, "asin"))
        arb_asin(y, y, prec);
      else if (!strcmp(t, "acos"))
        arb_acos(y, y, prec);
      else if (!strcmp(t, "atan"))
        arb_atan(y, y, prec);
      else if (!strcmp(t, "exp"))
        arb_exp(y, y, prec);
      else if (!strcmp(t, "log"))
        arb_log(y, y, prec);
      else if (!strcmp(t, "abs"))
        arb_abs(y, y);
      else if (!strcmp(t, "sinh"))
        arb_sinh(y, y, prec);
      else if (!strcmp(t, "cosh"))
        arb_cosh(y, y, prec);
      else if (!strcmp(t, "tanh"))
        arb_tanh(y, y, prec);
      else if (!strcmp(t, "erf"))
        arb_hypgeom_erf(y, y, prec);
      else if (!strcmp(t, "erfc"))
        arb_hypgeom_erfc(y, y, prec);
      else
        ok = 0;
    } else
      ok = 0;
  }
  ok = ok && sp == 1;
  if (ok)
    arb_set(out, st);
  _arb_vec_clear(st, n);
  return ok;
}

/* a <- q exactly, for q a dyadic rational; 0 otherwise */
static int dyadic(arf_t a, const fmpq_t q) {
  const fmpz *den = fmpq_denref(q);
  slong v = (slong)fmpz_val2(den);
  if ((slong)fmpz_bits(den) - 1 != v)
    return 0;
  arf_set_fmpz(a, fmpq_numref(q));
  arf_mul_2exp_si(a, a, -v);
  return 1;
}

/* the sign of f at the exact point a: -1, 1, or 0 when undecided */
static int sign_at(var_t *v, const arf_t a, int pi) {
  arb_t x, y;
  arb_init(x), arb_init(y);
  arb_set_arf(x, a);
  int s = 0;
  if (!evaluate(y, v->f, v->nf, pi, x))
    die("malformed root function of", v->name);
  if (arb_is_positive(y))
    s = 1;
  else if (arb_is_negative(y))
    s = -1;
  arb_clear(x), arb_clear(y);
  return s;
}

static var_t *find_var(const char *name) {
  for (int i = 0; i < nvars; i++)
    if (!strcmp(vars[i].name, name))
      return &vars[i];
  return NULL;
}

/* A root's ball at precs[pi]: certified unique in its bracket, then bisected
   while the midpoint's sign is decided. */
static void root_ball(var_t *v, int pi) {
  slong prec = precs[pi];
  arf_t a, b, m;
  arb_t ab, d;
  arf_init(a), arf_init(b), arf_init(m);
  arb_init(ab), arb_init(d);
  if (dyadic(a, v->lo) && dyadic(b, v->hi)) {
    int sa = sign_at(v, a, pi), sb = sign_at(v, b, pi);
    arb_set_interval_arf(ab, a, b, prec);
    if (!evaluate(d, v->d, v->nd, pi, ab))
      die("malformed root derivative of", v->name);
    if (sa * sb < 0 && !arb_contains_zero(d)) {
      for (slong it = 0; it < prec + 16; it++) {
        arf_add(m, a, b, ARF_PREC_EXACT, ARF_RND_DOWN);
        arf_mul_2exp_si(m, m, -1);
        int sm = sign_at(v, m, pi);
        if (sm == 0)
          break;
        if (sm == sa)
          arf_swap(a, m);
        else
          arf_swap(b, m);
      }
      arb_set_interval_arf(v->ball[pi], a, b, prec);
      v->state[pi] = 1;
    }
  }
  arf_clear(a), arf_clear(b), arf_clear(m);
  arb_clear(ab), arb_clear(d);
}

/* The ball of the root or let `name` at precs[pi], computed once. */
static int var_ball(arb_t out, const char *name, int pi) {
  var_t *v = find_var(name);
  if (!v)
    die("undefined name", name);
  if (v->state[pi] == 0) {
    v->state[pi] = -1;
    if (!v->let)
      root_ball(v, pi);
    else if (evaluate(v->ball[pi], v->f, v->nf, pi, NULL))
      v->state[pi] = 1;
    if (verbose || v->state[pi] < 0) {
      printf(
          "%s %s %s at %ld bits: ", v->state[pi] > 0 ? "BALL" : "UNCERTIFIED",
          v->let ? "let" : "root", v->name, (long)precs[pi]);
      arb_printn(v->ball[pi], 40, 0);
      printf("\n");
    }
  }
  if (v->state[pi] < 0)
    return 0;
  arb_set(out, v->ball[pi]);
  return 1;
}

/* 1 when V decides L, 0 when not yet; *margin is |V - L| + rad over half a
   unit in L's last digit */
static int decides(const arb_t v, const char *lit, slong prec, double *margin) {
  char mant[512];
  const char *e = strpbrk(lit, "eE");
  size_t len = e ? (size_t)(e - lit) : strlen(lit);
  if (len == 0 || len >= sizeof mant)
    die("literal", lit);
  memcpy(mant, lit, len);
  mant[len] = 0;
  if (strspn(mant, "-+0.") == len) {
    *margin = 0;
    return arb_is_zero(v);
  }
  const char *dot = strchr(mant, '.');
  long k = (e ? strtol(e + 1, NULL, 10) : 0) -
           (dot ? (long)(mant + len - dot - 1) : 0);
  arb_t l, d, h;
  arb_init(l), arb_init(d), arb_init(h);
  if (arb_set_str(l, lit, prec) != 0)
    die("literal", lit);
  arb_sub(d, v, l, prec);
  arb_abs(d, d);
  arb_set_ui(h, 10);
  arb_pow_ui(h, h, (ulong)(k < 0 ? -k : k), prec);
  if (k < 0)
    arb_inv(h, h, prec);
  arb_mul_2exp_si(h, h, -1);
  int ok = arb_lt(d, h);
  arb_div(d, d, h, prec);
  *margin = arf_get_d(arb_midref(d), ARF_RND_UP) + mag_get_d(arb_radref(d));
  arb_clear(l), arb_clear(d), arb_clear(h);
  return ok;
}

static void certify(char **tok, int n) {
  const char *name = tok[0], *index = tok[1], *lit = tok[2];
  arb_t v;
  arb_init(v);
  double margin = 0;
  slong prec = 0;
  int ok = 0;
  for (int p = 0; p < NPREC && !ok; p++) {
    prec = precs[p];
    if (!evaluate(v, tok + 3, n - 3, p, NULL)) {
      /* an uncertified root: try the next precision */
      arb_indeterminate(v);
      continue;
    }
    ok = decides(v, lit, prec, &margin);
  }
  if (!ok || verbose) {
    printf("%s %s[%s]: literal %s, value ", ok ? "BALL" : "UNDECIDED", name,
           index, lit);
    arb_printn(v, 40, 0);
    printf(" at %ld bits\n", (long)prec);
  }
  if (!ok)
    failed++;
  else if (margin > worst_margin)
    worst_margin = margin, worst_prec = prec;
  checked++;
  arb_clear(v);
}

/* q <- the decimal literal [-]digits[.digits][e[-]digits] exactly; 0 when
   malformed */
static int decimal(fmpq_t q, const char *lit) {
  const char *p = lit + (*lit == '-');
  fmpz_t m, t;
  fmpz_init(m), fmpz_init(t);
  long frac = 0, ex = 0;
  int digits = 0, point = 0, ok = 1;
  for (; *p && *p != 'e' && *p != 'E'; p++)
    if (*p == '.' && !point)
      point = 1;
    else if (isdigit((unsigned char)*p)) {
      fmpz_mul_ui(m, m, 10);
      fmpz_add_ui(m, m, (ulong)(*p - '0'));
      digits++, frac += point;
    } else
      ok = 0;
  if (*p) {
    char *end;
    ex = strtol(p + 1, &end, 10);
    ok = ok && end != p + 1 && !*end;
  }
  ok = ok && digits > 0;
  if (ok) {
    long k = ex - frac;
    fmpz_set_ui(t, 10);
    fmpz_pow_ui(t, t, (ulong)(k < 0 ? -k : k));
    if (k < 0)
      fmpq_set_fmpz_frac(q, m, t);
    else {
      fmpz_mul(m, m, t);
      fmpq_set_fmpz(q, m);
    }
    if (*lit == '-')
      fmpq_neg(q, q);
  }
  fmpz_clear(m), fmpz_clear(t);
  return ok;
}

/* 1 when every point of v rounds to one double, written to d */
static int rounds_to(arf_t d, const arb_t v, slong prec) {
  arf_t lo, hi;
  arf_init(lo), arf_init(hi);
  int ok = arb_is_finite(v);
  if (ok) {
    arb_get_lbound_arf(lo, v, prec);
    arb_get_ubound_arf(hi, v, prec);
    arf_set_round(lo, lo, 53, ARF_RND_NEAR);
    arf_set_round(hi, hi, 53, ARF_RND_NEAR);
    ok = arf_equal(lo, hi);
    arf_set(d, lo);
  }
  arf_clear(lo), arf_clear(hi);
  return ok;
}

static int normal(const arf_t d) {
  return arf_is_zero(d) ||
         (arf_cmpabs_2exp_si(d, -1022) >= 0 && arf_cmpabs_2exp_si(d, 1024) < 0);
}

/* (|V - D| + rad) over half the gap from D to the next double on V's side */
static double ulp_margin(const arb_t v, const arf_t d, slong prec) {
  if (arf_is_zero(d))
    return 0;
  slong e = arf_abs_bound_lt_2exp_si(d);
  int below = arf_cmpabs(arb_midref(v), d) < 0;
  arb_t t;
  arb_init(t);
  arb_set_arf(t, d);
  arb_sub(t, v, t, prec);
  arb_abs(t, t);
  arb_mul_2exp_si(t, t, -(e - 54 - (below && arf_bits(d) == 1)));
  double m = arf_get_d(arb_midref(t), ARF_RND_UP) + mag_get_d(arb_radref(t));
  arb_clear(t);
  return m;
}

/* ARBF64 name index literal token... (f64) or ARBD candidate token... */
static void certify_double(char **tok, int n, int f64) {
  int skip = f64 ? 3 : 1, decided = 0, ok = 0;
  fmpq_t q;
  arf_t want, got;
  arb_t v;
  fmpq_init(q), arf_init(want), arf_init(got), arb_init(v);
  if (f64 ? !decimal(q, tok[2]) : fmpq_set_str(q, tok[0], 10) != 0)
    die(f64 ? "literal" : "candidate", f64 ? tok[2] : tok[0]);
  fmpq_canonicalise(q);
  /* a literal rounds; a candidate must already be a double */
  int inexact = arf_set_fmpq(want, q, 53, ARF_RND_NEAR);
  slong prec = 0;
  for (int p = 0; p < NPREC && !decided; p++) {
    prec = precs[p];
    if (!evaluate(v, tok + skip, n - skip, p, NULL)) {
      arb_indeterminate(v);
      continue;
    }
    decided = rounds_to(got, v, prec);
  }
  ok = decided && arf_equal(got, want) && normal(want) && (f64 || !inexact);
  double margin = ok ? ulp_margin(v, want, prec) : 0;
  if (!ok || verbose) {
    const char *label = ok ? "BALL" : decided ? "WRONG" : "UNDECIDED";
    if (f64)
      printf("%s double %s[%s]: literal %s, value ", label, tok[0], tok[1],
             tok[2]);
    else
      printf("%s double: candidate %s, value ", label, tok[0]);
    arb_printn(v, 40, 0);
    printf(" at %ld bits\n", (long)prec);
  }
  if (!ok)
    dfailed++;
  else if (margin > worst_dmargin)
    worst_dmargin = margin, worst_dprec = prec;
  dchecked++;
  fmpq_clear(q), arf_clear(want), arf_clear(got), arb_clear(v);
}

/* tok[0..n) copied, each `$name` in it already defined: names go strictly
   forward, so no value depends on itself */
static char **copy_tokens(char **tok, int n) {
  char **c = malloc(sizeof *c * (size_t)n);
  for (int i = 0; i < n; i++) {
    if (tok[i][0] == '$' && !find_var(tok[i] + 1))
      die("undefined name", tok[i] + 1);
    c[i] = strdup(tok[i]);
  }
  return c;
}

static var_t *new_var(const char *name, int let) {
  if (find_var(name))
    die("name defined twice", name);
  vars = realloc(vars, sizeof *vars * (size_t)(nvars + 1));
  var_t *v = &vars[nvars++];
  memset(v, 0, sizeof *v);
  v->name = strdup(name);
  v->let = let;
  fmpq_init(v->lo), fmpq_init(v->hi);
  for (int p = 0; p < NPREC; p++)
    arb_init(v->ball[p]);
  return v;
}

/* ARBVAR name lo hi f... ; f'... */
static void define_root(char **tok, int n) {
  int semi = -1;
  for (int i = 3; i < n; i++)
    if (!strcmp(tok[i], ";"))
      semi = i;
  if (n < 6 || semi < 4 || semi == n - 1)
    die("malformed root", n ? tok[0] : "");
  var_t *v = new_var(tok[0], 0);
  if (fmpq_set_str(v->lo, tok[1], 10) || fmpq_set_str(v->hi, tok[2], 10))
    die("malformed root bracket", tok[0]);
  fmpq_canonicalise(v->lo), fmpq_canonicalise(v->hi);
  v->nf = semi - 3, v->nd = n - semi - 1;
  v->f = copy_tokens(tok + 3, v->nf);
  v->d = copy_tokens(tok + semi + 1, v->nd);
}

/* ARBLET name token... */
static void define_let(char **tok, int n) {
  if (n < 2)
    die("malformed let", n ? tok[0] : "");
  var_t *v = new_var(tok[0], 1);
  v->nf = n - 1;
  v->f = copy_tokens(tok + 1, v->nf);
}

int main(int argc, char **argv) {
  verbose = argc > 1 && !strcmp(argv[1], "-v");
  char *line = NULL;
  size_t cap = 0;
  int ntok = 1024;
  char **tok = malloc(sizeof *tok * (size_t)ntok);
  while (getline(&line, &cap, stdin) > 0) {
    int root = !strncmp(line, "ARBVAR ", 7), let = !strncmp(line, "ARBLET ", 7),
        f64 = !strncmp(line, "ARBF64 ", 7), dbl = !strncmp(line, "ARBD ", 5);
    if (strncmp(line, "ARB ", 4) && !root && !let && !f64 && !dbl)
      continue;
    int n = 0;
    for (char *t = strtok(line + (root || let || f64 ? 7
                                  : dbl              ? 5
                                                     : 4),
                          " \t\r\n");
         t; t = strtok(NULL, " \t\r\n")) {
      if (n == ntok)
        tok = realloc(tok, sizeof *tok * (size_t)(ntok *= 2));
      tok[n++] = t;
    }
    if (root)
      define_root(tok, n);
    else if (let)
      define_let(tok, n);
    else if ((f64 && n < 4) || (dbl && n < 2))
      die("short line", line);
    else if (f64 || dbl)
      certify_double(tok, n, f64);
    else if (n < 4)
      die("short line", line);
    else
      certify(tok, n);
  }
  free(tok);
  free(line);
  if (failed || dfailed || (!checked && !dchecked)) {
    printf("FAILED: %d of %d literals and %d of %d doubles undecided or "
           "wrong\n",
           failed, checked, dfailed, dchecked);
    return 1;
  }
  printf("CERTIFIED %d literals, %d doubles", checked, dchecked);
  if (checked)
    printf("; the worst lies %.3g of a half unit in its last digit from its "
           "ball (%ld bits)",
           worst_margin, (long)worst_prec);
  if (dchecked)
    printf("%s the worst double %.3g of a half ulp (%ld bits)",
           checked ? "," : ";", worst_dmargin, (long)worst_dprec);
  printf("\n");
  flint_cleanup();
  return 0;
}
