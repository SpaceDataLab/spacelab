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
   and sin cos tan asin acos atan exp log abs sinh cosh tanh. With -v it
   prints each ball, its precision and the literal.

     cc -O2 arb_balls.c -o arb_balls $(pkg-config --cflags --libs flint)
     timeout 300 maxima --very-quiet -Q -b derivation.mac | ./arb_balls
*/
#include <ctype.h>
#include <flint/arb.h>
#include <flint/fmpq.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const slong precs[] = {256, 1024, 4096};
static int checked, failed, verbose;
static double worst_margin;
static slong worst_prec;

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

/* The value of the RPN tokens tok[0..n) at prec bits; 0 when malformed. */
static int evaluate(arb_t out, char **tok, int n, slong prec) {
  arb_ptr st = _arb_vec_init(n);
  int sp = 0, ok = 1;
  for (int i = 0; i < n && ok; i++) {
    const char *t = tok[i];
    long k;
    if (is_rational(t)) {
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
      arb_ptr x = st + sp - 1;
      if (!strcmp(t, "sin"))
        arb_sin(x, x, prec);
      else if (!strcmp(t, "cos"))
        arb_cos(x, x, prec);
      else if (!strcmp(t, "tan"))
        arb_tan(x, x, prec);
      else if (!strcmp(t, "asin"))
        arb_asin(x, x, prec);
      else if (!strcmp(t, "acos"))
        arb_acos(x, x, prec);
      else if (!strcmp(t, "atan"))
        arb_atan(x, x, prec);
      else if (!strcmp(t, "exp"))
        arb_exp(x, x, prec);
      else if (!strcmp(t, "log"))
        arb_log(x, x, prec);
      else if (!strcmp(t, "abs"))
        arb_abs(x, x);
      else if (!strcmp(t, "sinh"))
        arb_sinh(x, x, prec);
      else if (!strcmp(t, "cosh"))
        arb_cosh(x, x, prec);
      else if (!strcmp(t, "tanh"))
        arb_tanh(x, x, prec);
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
  for (size_t p = 0; p < sizeof precs / sizeof *precs && !ok; p++) {
    prec = precs[p];
    if (!evaluate(v, tok + 3, n - 3, prec))
      die("malformed value of", name);
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

int main(int argc, char **argv) {
  verbose = argc > 1 && !strcmp(argv[1], "-v");
  char *line = NULL;
  size_t cap = 0;
  int ntok = 1024;
  char **tok = malloc(sizeof *tok * (size_t)ntok);
  while (getline(&line, &cap, stdin) > 0) {
    if (strncmp(line, "ARB ", 4))
      continue;
    int n = 0;
    for (char *t = strtok(line + 4, " \t\r\n"); t;
         t = strtok(NULL, " \t\r\n")) {
      if (n == ntok)
        tok = realloc(tok, sizeof *tok * (size_t)(ntok *= 2));
      tok[n++] = t;
    }
    if (n < 4)
      die("short line", line);
    certify(tok, n);
  }
  free(tok);
  free(line);
  if (failed || !checked) {
    printf("FAILED: %d of %d literals undecided or wrong\n", failed, checked);
    return 1;
  }
  printf("CERTIFIED %d literals; the worst lies %.3g of a half unit in its "
         "last digit from its ball (%ld bits)\n",
         checked, worst_margin, (long)worst_prec);
  flint_cleanup();
  return 0;
}
