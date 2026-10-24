#!/usr/bin/env bash
# Re-derive everything the code claims was derived, and fail loudly if it is not.
#
# A comment saying "proved in Maxima" is worth nothing unless something re-runs Maxima. This is
# that something. It is not part of `cargo test` — it needs a computer algebra system — so it runs
# as its own CI job, and any failure here means a doc comment is now a lie.
#
#   ./derivations/verify.sh
#
# Every invocation is wrapped in a timeout: Maxima and FriCAS both hang on malformed input, and
# FriCAS hangs on a missing `)quit` regardless.

set -uo pipefail
cd "$(dirname "$0")"

PASS=0
FAIL=0
SKIP=0

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; printf '        %s\n' "$2"; FAIL=$((FAIL + 1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s (%s not installed)\n' "$1" "$2"; SKIP=$((SKIP + 1)); }

have() { command -v "$1" >/dev/null 2>&1; }

printf '\nRe-deriving every claim the source code attributes to a computer algebra system.\n\n'

# ---------------------------------------------------------------------------------------------
# 1. The footprint radius equals SaVi's published closed form. Maxima, then FriCAS, independently.
# ---------------------------------------------------------------------------------------------
if have maxima; then
    out=$(timeout 120 maxima --very-quiet -b footprint3.mac 2>&1)
    zeros=$(printf '%s' "$out" | grep -cE '==> 0|identity: 0')
    if [ "$zeros" -ge 3 ]; then
        ok "maxima footprint3.mac: mask, cone and discriminant identities all reduce to 0"
    else
        bad "maxima footprint3.mac" "expected 3 identities reducing to 0, found $zeros"
    fi

    out=$(timeout 120 maxima --very-quiet -b ellipsoid_normal.mac 2>&1)
    if printf '%s' "$out" | grep -q 'zenith direction is'; then
        ok "maxima ellipsoid_normal.mac: grad f of the ellipsoid is (x, y, gamma*z)"
    else
        bad "maxima ellipsoid_normal.mac" "the gradient derivation did not run"
    fi

    out=$(timeout 120 maxima --very-quiet -b step_bound.mac 2>&1)
    if printf '%s' "$out" | grep -q 'a cosine: 0'; then
        ok "maxima step_bound.mac: the Cauchy-Schwarz step in the |dg/dt| bound checks out"
    else
        bad "maxima step_bound.mac" "the bound derivation did not verify"
    fi

    out=$(timeout 120 maxima --very-quiet -b cull_bound.mac 2>&1)
    if printf '%s' "$out" | grep -q 'negative => cos decreases'; then
        ok "maxima cull_bound.mac: d(cos lam_max)/dL < 0, so R_polar gives the conservative cap"
    else
        bad "maxima cull_bound.mac" "the cull monotonicity derivation did not verify"
    fi

    out=$(timeout 120 maxima --very-quiet -b dopri5_order.mac 2>&1)
    zeros=$(printf '%s' "$out" | grep -cE ': 0 *$')
    if [ "$zeros" -ge 18 ]; then
        ok "maxima dopri5_order.mac: the DOPRI5 tableau satisfies every order condition through 5"
    else
        bad "maxima dopri5_order.mac" "expected >=18 order conditions to vanish, found $zeros"
    fi

    out=$(timeout 120 maxima --very-quiet -b dop853_order.mac 2>&1)
    # every order condition must vanish to ~1e-28 (the literals' precision), proving order 8
    big=$(printf '%s' "$out" | grep -oE ': [0-9.]+e-[0-9]+' | grep -oE 'e-[0-9]+' | sed 's/e-//' | sort -n | head -1)
    if [ -n "$big" ] && [ "$big" -ge 27 ]; then
        ok "maxima dop853_order.mac: the full-precision DOP853 literals satisfy order 8 to ~1e-${big}"
    else
        bad "maxima dop853_order.mac" "an order condition did not vanish (smallest exponent e-${big:-?})"
    fi

    out=$(timeout 120 maxima --very-quiet -b zonal_gradient.mac 2>&1)
    zeros=$(printf '%s' "$out" | grep -cE 'identity a[xz]: +0')
    if [ "$zeros" -ge 4 ]; then
        ok "maxima zonal_gradient.mac: J2 and J3 accelerations equal -grad U (4 identities to 0)"
    else
        bad "maxima zonal_gradient.mac" "expected 4 zonal gradient identities to vanish, found $zeros"
    fi

    out=$(timeout 120 maxima --very-quiet -b third_body_battin.mac 2>&1)
    if printf '%s' "$out" | grep -qE 'battin identity: +0'; then
        ok "maxima third_body_battin.mac: the cancellation-free third-body form equals the direct one"
    else
        bad "maxima third_body_battin.mac" "the Battin identity did not reduce to 0"
    fi

    out=$(timeout 120 maxima --very-quiet -b yoshida_order.mac 2>&1)
    if printf '%s' "$out" | grep -qE 'consistency: +1' && printf '%s' "$out" | grep -qE 'order4: +0'; then
        ok "maxima yoshida_order.mac: 2w1+w0=1 and 2w1^3+w0^3=0, so yoshida4 is 4th order"
    else
        bad "maxima yoshida_order.mac" "a Yoshida 4th-order condition failed"
    fi

    out=$(timeout 120 maxima --very-quiet -b drag_dissipation.mac 2>&1)
    if printf '%s' "$out" | grep -qE 'drag trace identity: +0'; then
        ok "maxima drag_dissipation.mac: tr(da/dv) = -2(Cd A/m)rho|v_rel| < 0, so det Phi contracts"
    else
        bad "maxima drag_dissipation.mac" "the drag Jacobian trace did not match"
    fi

    out=$(timeout 120 maxima --very-quiet -b adams_coeffs.mac 2>&1)
    if printf '%s' "$out" | grep -qE 'AB8 sum .*: +1' && printf '%s' "$out" | grep -qE 'AM8 sum .*: +1' && [ "$(printf '%s' "$out" | grep -cE 'leading.*: +0')" -ge 2 ]; then
        ok "maxima adams_coeffs.mac: AB8/AM8 sum to 1 and the leading coefficients derive correctly"
    else
        bad "maxima adams_coeffs.mac" "an Adams coefficient identity failed"
    fi

    out=$(timeout 120 maxima --very-quiet -b suzuki_triplejump.mac 2>&1)
    cancels=$(printf '%s' "$out" | grep -cE 'cancellation p = [357] : +0')
    roots=$(printf '%s' "$out" | grep -cE 'root - closed form \(=0\?\): +0')
    if [ "$cancels" -ge 3 ] && [ "$roots" -ge 3 ] && printf '%s' "$out" | grep -qE 'yoshida8 cancellation.*: +0'; then
        ok "maxima suzuki_triplejump.mac: z=1/(2-2^(1/(2k+1))) is the unique root that lifts yoshida4->6->8"
    else
        bad "maxima suzuki_triplejump.mac" "a triple-jump coefficient identity failed (cancels=$cancels roots=$roots)"
    fi

    out=$(timeout 120 maxima --very-quiet -b gauss_jackson_coeffs.mac 2>&1)
    # both series must sum to 1, the m=9 residuals vanish (order >= 8) and m=10 residuals do not (=8)
    if printf '%s' "$out" | grep -qE 'Stormer sum .*: +1' && printf '%s' "$out" | grep -qE 'Cowell sum .*: +1' \
       && [ "$(printf '%s' "$out" | grep -cE 'm=9 residual.*: +0')" -ge 2 ] \
       && ! printf '%s' "$out" | grep -qE 'm=10 residual.*: +0 *$'; then
        ok "maxima gauss_jackson_coeffs.mac: Stormer/Cowell weights sum to 1 and are exactly order 8"
    else
        bad "maxima gauss_jackson_coeffs.mac" "a Gauss-Jackson moment condition failed"
    fi

    out=$(timeout 120 maxima --very-quiet -b radau5_coeffs.mac 2>&1)
    bcond=$(printf '%s' "$out" | grep -cE 'B\( [1-5] \) residual .*: +0')
    stiff=$(printf '%s' "$out" | grep -cE 'stiff-accurate b [1-3].*: +0')
    if [ "$bcond" -ge 5 ] && [ "$stiff" -ge 3 ] && [ "$(printf '%s' "$out" | grep -cE 'row-sum a.*: +0')" -ge 3 ]; then
        ok "maxima radau5_coeffs.mac: Radau IIA tableau is order 5 (B(1..5)=0), stiffly accurate, rows sum to c"
    else
        bad "maxima radau5_coeffs.mac" "a Radau IIA order/collocation condition failed (B=$bcond stiff=$stiff)"
    fi

    out=$(timeout 120 maxima --very-quiet -b rosenbrock_grk4t.mac 2>&1)
    # the GRK4T stability function must match exp(z) through z^4 (5 MATCH lines: z^0..z^4)
    matches=$(printf '%s' "$out" | grep -cE 'MATCH \(order cond\)')
    if [ "$matches" -ge 5 ]; then
        ok "maxima rosenbrock_grk4t.mac: the GRK4T stage recurrence reproduces exp(z) through z^4 (order 4)"
    else
        bad "maxima rosenbrock_grk4t.mac" "GRK4T stability function did not match exp(z) to order 4 (matches=$matches)"
    fi

    out=$(timeout 120 maxima --very-quiet -b ias15_nodes.mac 2>&1)
    # eight Gauss-Radau nodes (h0=0, h7<1) and the c/d conversion matrices are mutually inverse
    cdinv=$(printf '%s' "$out" | grep -oE 'c.d - I largest entry \(=0\?\): [0-9.]+b-[0-9]+' | grep -oE 'b-[0-9]+' | tr -d 'b-')
    if printf '%s' "$out" | grep -qE 'node count .*: +8' && [ -n "$cdinv" ] && [ "$cdinv" -ge 30 ]; then
        ok "maxima ias15_nodes.mac: 8 Gauss-Radau nodes derived, c and d conversion matrices invert to ~1e-${cdinv}"
    else
        bad "maxima ias15_nodes.mac" "IAS15 node/matrix derivation failed (c.d-I exponent e-${cdinv:-?})"
    fi

    out=$(timeout 120 maxima --very-quiet -b whfast_stumpff.mac 2>&1)
    # Stumpff identities (closed forms/recurrences/doubling) plus the Kepler-drift derivation
    # (radius, Kepler equation, symplectic Wronskian) must all reduce to 0.
    stumpff=$(printf '%s' "$out" | grep -cE '=0\?\): 0( |$)')
    if [ "$stumpff" -ge 13 ] && printf '%s' "$out" | grep -qE 'Wronskian.*: +0'; then
        ok "maxima whfast_stumpff.mac: Stumpff identities and the symplectic Kepler f/g map all reduce to 0"
    else
        bad "maxima whfast_stumpff.mac" "a Stumpff/Kepler identity failed ($stumpff reduced to 0)"
    fi

    out=$(timeout 120 maxima --very-quiet -b adams_nordsieck.mac 2>&1)
    # the order-5 Nordsieck l-vector is derived (char poly x^5(x-1), stable) and the Pascal predictor is exact
    if printf '%s' "$out" | grep -qE 'char poly M0.*\(=0\?\): 0' \
       && printf '%s' "$out" | grep -qE 'l-vector - .*\(=0\?\): \[0, 0, 0, 0, 0, 0\]' \
       && printf '%s' "$out" | grep -qE 'Pascal.*\(=0\?\): 0'; then
        ok "maxima adams_nordsieck.mac: the order-5 Nordsieck l-vector is derived (spurious roots at 0), Pascal exact"
    else
        bad "maxima adams_nordsieck.mac" "the Nordsieck l-vector derivation or Pascal predictor failed"
    fi

    out=$(timeout 120 maxima --very-quiet -b ark_order.mac 2>&1)
    # all 18 ARK4(3)6L[2]SA order-4 conditions vanish to ~1e-27 (the coefficients' precision)
    exp=$(printf '%s' "$out" | grep -oE 'residual.*: [0-9.]+b-[0-9]+' | grep -oE 'b-[0-9]+' | tr -d 'b-')
    if [ -n "$exp" ] && [ "$exp" -ge 20 ]; then
        ok "maxima ark_order.mac: the ARK4(3)6L[2]SA IMEX pair satisfies all 18 order-4 conditions to ~1e-${exp}"
    else
        bad "maxima ark_order.mac" "an ARK order-4 condition did not vanish (exponent e-${exp:-?})"
    fi

    out=$(timeout 120 maxima --very-quiet -b mercurius_changeover.mac 2>&1)
    # the smootherstep changeover: 0->1 partition of unity, C^2 (first two derivatives vanish at both ends)
    z=$(printf '%s' "$out" | grep -cE '\(=0\?\): 0 *$')
    if [ "$z" -ge 8 ]; then
        ok "maxima mercurius_changeover.mac: the changeover is a C^2 partition of unity (S, S', S'' pinned at both ends)"
    else
        bad "maxima mercurius_changeover.mac" "a changeover smoothness condition failed ($z/8)"
    fi

    out=$(timeout 120 maxima --very-quiet -b mri_gark_order.mac 2>&1)
    # MRI-GARK-ERK33a: internal consistency + base order 3 + the order-3 coupling condition
    if printf '%s' "$out" | grep -qE 'Gamma\^0 . 1 - Dc \(=0\?\): \[0, 0, 0\]' \
       && printf '%s' "$out" | grep -qE 'coupling.*- 1/6 \(=0\?\): 0'; then
        ok "maxima mri_gark_order.mac: MRI-GARK-ERK33a is internally consistent and order 3"
    else
        bad "maxima mri_gark_order.mac" "an MRI-GARK order condition failed"
    fi

    out=$(timeout 120 maxima --very-quiet -b saba_coeffs.mac 2>&1)
    # SABA drift lengths and kick weights each sum to 1 (n=1..4, to ~1e-33), and SABA2 matches its form
    sums=$(printf '%s' "$out" | grep -cE 'max \|sum - 1\| \(~0\?\): (0\.0|[0-9.]+b-[0-9][0-9])')
    if [ "$sums" -ge 4 ] && printf '%s' "$out" | grep -qE 'SABA2 c1.*\(=0\?\): 0\.0' && printf '%s' "$out" | grep -qE 'SABA2 c2.*\(=0\?\): 0\.0'; then
        ok "maxima saba_coeffs.mac: SABA_n Gauss-Legendre drift/kick coefficients sum to 1, SABA2 closed form checks"
    else
        bad "maxima saba_coeffs.mac" "a SABA coefficient identity failed (sums=$sums)"
    fi

    out=$(timeout 120 maxima --very-quiet -b ks_regularization.mac 2>&1)
    # the KS map: |r|=|u|^2, the constraint, L^T L = |u|^2 I, and the physical bilinear relation
    ks=$(printf '%s' "$out" | grep -cE 'KS .*\(=0\?\): 0')
    if [ "$ks" -ge 4 ]; then
        ok "maxima ks_regularization.mac: |r|=|u|^2, L^T L=|u|^2 I, and the KS bilinear relation all hold"
    else
        bad "maxima ks_regularization.mac" "a KS identity failed ($ks/4 reduced to 0)"
    fi

    out=$(timeout 120 maxima --very-quiet -b lie_so3.mac 2>&1)
    # the SO(3) machinery: commutator=cross, Rodrigues orthogonal+unit-det, dexpinv Bernoulli coeffs
    zeros=$(printf '%s' "$out" | grep -cE '\(=0\?\): 0 *$')
    if [ "$zeros" -ge 6 ] && printf '%s' "$out" | grep -qE 'orthogonality.*\(=0\?\): 0'; then
        ok "maxima lie_so3.mac: so(3)=cross, Rodrigues is a rotation, dexpinv is the Bernoulli series"
    else
        bad "maxima lie_so3.mac" "an SO(3) Lie-group identity failed ($zeros reduced to 0)"
    fi

    out=$(timeout 120 maxima --very-quiet -b taylor_jets.mac 2>&1)
    # the jet recurrences (recip, sqrt, powf) must match the true Taylor coefficients, and the ODE
    # bootstrap on y'=y^2 must reproduce the geometric series (all a_k − 1 = 0)
    jets=$(printf '%s' "$out" | grep -cE '\(=0\?\): 0 *$')
    if [ "$jets" -ge 3 ] && printf '%s' "$out" | grep -qE 'ode bootstrap.*\[0(, 0)+\]'; then
        ok "maxima taylor_jets.mac: jet recip/sqrt/powf recurrences match Taylor, ODE bootstrap reproduces the series"
    else
        bad "maxima taylor_jets.mac" "a jet recurrence or the ODE bootstrap failed ($jets recurrences matched)"
    fi

    out=$(timeout 240 maxima --very-quiet -b process_noise.mac 2>&1)
    # Van Loan must reproduce the integral definition on a 3-state plant (two residuals to 0), and the
    # scalar Gauss-Markov stationary variance must come out as q*tau/2 -- which is what fixes
    # q = 2 sigma^2 / tau. The scalar case is the one that breaks first: a 1x1 matrix product
    # simplifies to a scalar unless `scalarmatrixp` is off, and then `integrate` returns a noun form
    # and the limit fails several lines later.
    resid=$(printf '%s' "$out" | grep -cE 'residual: 0 *$')
    if [ "$resid" -ge 2 ] && printf '%s' "$out" | grep -qE 'stationary variance:' \
       && printf '%s' "$out" | grep -q 'q tau'; then
        ok "maxima process_noise.mac: Van Loan equals the definition, and the stationary variance is q*tau/2"
    else
        bad "maxima process_noise.mac" "expected 2 zero residuals and the q*tau/2 limit, found $resid residuals"
    fi

    # -----------------------------------------------------------------------------------------
    # 1b. The rest of the .mac corpus. Each of these files ends in a claim -- a residual that must
    # vanish, a ratio that must be exact, a flag that must be true -- and until now nothing re-ran
    # any of them, so the claim was only as good as the day it was written.
    #
    # These use `batchload` where the block above uses `-b`, for a reason that cost two false
    # passes to learn: `-b` echoes every statement before its result, so a grep can match a
    # `print` call's own argument and report a proof that never ran. `batchload` echoes nothing.
    # `display2d:false` and a wide `linel` then keep each result on one line, so a grep sees the
    # whole value instead of Maxima's 78-column wrap.
    #
    # Where a file only prints symbolic forms, the claim is evaluated against the bindings the
    # file leaves behind (macf's second argument) rather than read out of its output.
    # -----------------------------------------------------------------------------------------
    # `--batch-string` still echoes the wrapper's own statements, and `display2d:false` is itself
    # the token one of the checks below counts -- so drop the three preamble echoes by exact match.
    # Statements passed in $2 echo too; those call sites anchor their token to the start of a line,
    # where an echoed `print(TOKEN, ...)` cannot reach it.
    macf() {
        timeout 300 maxima --very-quiet --batch-string \
            "display2d:false\$ linel:2000\$ batchload(\"$1\")\$ ${2-}" 2>&1 \
            | grep -vxF -e 'display2d:false' -e 'linel:2000' -e "batchload(\"$1\")"
    }
    # smallest negative power of ten among the matches of $1 in $2 -- how close to zero the file got
    tozero() { printf '%s' "$2" | grep -oE "$1" | grep -oE '[eb]-[0-9]+' | grep -oE '[0-9]+' | sort -n | head -1; }

    out=$(macf cio_frame_branch.mac)
    n=$(printf '%s' "$out" | grep -c '^VERIFIED:')
    if [ "$n" -eq 2 ]; then
        ok "maxima cio_frame_branch.mac: Q collapses to R3(-s) for every E at d = 0, and d = 0 exactly at r2 = 0"
    else
        bad "maxima cio_frame_branch.mac" "expected 2 VERIFIED lines, each reached only past an error() guard, found $n"
    fi

    out=$(macf consensus_delay.mac)
    if printf '%s' "$out" | grep -q 'A: tau_max \* lam (must be pi/2) : %pi/2' \
       && printf '%s' "$out" | grep -q 'A: residual of the transcendental equation : 0\.0b0' \
       && printf '%s' "$out" | grep -q 'C_6 lam_max / k : 4'; then
        ok "maxima consensus_delay.mac: the consensus critical delay is pi/(2 lam_max), and lam_max(C_6) = 4k"
    else
        bad "maxima consensus_delay.mac" "the Olfati-Saber bound did not come out as pi/(2 lam_max)"
    fi

    out=$(macf csa_fm_rate.mac)
    if printf '%s' "$out" | grep -q 'r doubled  *2\.0b0' \
       && printf '%s' "$out" | grep -q 'lambda doubled 8\.0b0' \
       && printf '%s' "$out" | grep -q 'f_eta doubled  *4\.0b0' \
       && printf '%s' "$out" | grep -q 'v doubled  *2\.5b-1' \
       && printf '%s' "$out" | grep -q 'D doubled  *1\.25b-1'; then
        ok "maxima csa_fm_rate.mac: K_m scales as r lambda^3 f^2 / (v^2 D^3) -- 2, 8, 4, 1/4, 1/8 exactly"
    else
        bad "maxima csa_fm_rate.mac" "a scaling exponent of the chirp-scaling FM rate moved"
    fi

    out=$(macf dde_method_of_steps.mac)
    cont=$(printf '%s' "$out" | grep -cE "^x'? +continuous at t = [1-8] : 0 *$")
    resid=$(printf '%s' "$out" | grep -cE '^[0-8]  <= t <=  [1-9]  :  0 *$')
    if [ "$cont" -eq 16 ] && [ "$resid" -eq 8 ] \
       && printf '%s' "$out" | grep -q '0  <= t <=  1  :  1-t' \
       && printf '%s' "$out" | grep -q '1  <= t <=  2  :  (t^2-4\*t+3)/2' \
       && printf '%s' "$out" | grep -qE "x'' jump at t = 1 [^:]*: 1 *$" \
       && printf '%s' "$out" | grep -qE "x'' jump at t = 2 [^:]*: 0 *$"; then
        ok "maxima dde_method_of_steps.mac: 8 exact arcs solve x'(t) = -x(t-1), and the x'' jump sits at t = 1 only"
    else
        bad "maxima dde_method_of_steps.mac" "expected 16 continuity zeros and 8 residual zeros, found $cont and $resid"
    fi

    out=$(macf delay_margin.mac)
    # the third entry of each [w_c, tau_max, |char. eq.|] triple: the residual, not the margin
    worst=$(tozero ',[0-9.]+b-[0-9]+\] *$' "$out")
    if printf '%s' "$out" | grep -q 'crossing_u(w^2) - crossing (must be 0) : 0' \
       && printf '%s' "$out" | grep -q 'substituting it back (must be 0) : 0' \
       && printf '%s' "$out" | grep -q '^tau_max = 0\.0b0' \
       && [ -n "$worst" ] && [ "$worst" -ge 30 ]; then
        ok "maxima delay_margin.mac: w_c solves the crossing polynomial (char. eq. ~1e-${worst}), and kw = 0 gives tau_max = 0"
    else
        bad "maxima delay_margin.mac" "the imaginary-axis crossing did not close (char. eq. ~1e-${worst:-?})"
    fi

    # footprint.mac derives the roots that footprint3.mac then assumes. Nothing checked that the
    # two files were still talking about the same root; this does.
    out=$(macf footprint.mac 'u : sqrt(sin(a)^2 + L^2 - 1)$ v : sqrt(1 - L^2*sin(a)^2)$
print(MASKROOT, ratsimp(rhs(sol[2]) - (sin(a)*u + 1 - sin(a)^2)/L))$
print(CONEROOT, trigsimp(ratsimp(rhs(solc[2]) - (L - L*(1 - sin(a)^2) + cos(a)*v))))$')
    if printf '%s' "$out" | grep -qE '^MASKROOT 0 *$' && printf '%s' "$out" | grep -qE '^CONEROOT 0 *$'; then
        ok "maxima footprint.mac: both roots solved from the geometry are the ones footprint3.mac proves SaVi's forms from"
    else
        bad "maxima footprint.mac" "a derived root no longer matches the root footprint3.mac assumes"
    fi

    # radcan cannot close either difference (BIBLIOGRAPHY.md records why). The mathematical claim
    # is still that they are zero, so evaluate them.
    out=$(macf footprint2.mac 'fpprec : 60$
grid : [[11/10, %pi/12], [11/10, %pi/6], [3/2, %pi/12], [2, %pi/12], [5, %pi/30]]$
print(MASKNUM, lmax(makelist(abs(bfloat(subst([L = g[1], a = g[2]], rho_ours^2 - rho_savi^2))), g, grid)))$
print(CONENUM, lmax(makelist(abs(bfloat(subst([L = g[1], a = g[2]], rho_cone_ours^2 - rho_cone_savi^2))), g, grid)))$')
    worst=$(tozero '^(MASKNUM|CONENUM) [^ ]+' "$out")
    if [ -n "$worst" ] && [ "$worst" -ge 40 ]; then
        ok "maxima footprint2.mac: both unclosed radicals are numerically zero to ~1e-${worst} over 5 (L, a) pairs at 60 digits"
    else
        bad "maxima footprint2.mac" "rho_ours^2 - rho_savi^2 is not zero numerically (worst ~1e-${worst:-?})"
    fi

    out=$(macf gbs_extrapolation.mac)
    if printf '%s' "$out" | grep -q '^ALL ORDERS CONFIRMED (2,4,6,8,10)' \
       && ! printf '%s' "$out" | grep -q 'MISMATCH'; then
        ok "maxima gbs_extrapolation.mac: the extrapolation diagonal T[j][j] is order 2j for j = 1..5"
    else
        bad "maxima gbs_extrapolation.mac" "an extrapolation-diagonal order is not 2j"
    fi

    out=$(macf j2_secular_rates.mac)
    n=$(printf '%s' "$out" | grep -cE 'derived - coded = +0 *$')
    if [ "$n" -eq 4 ]; then
        ok "maxima j2_secular_rates.mac: Mdot, Wdot, wdot and the frozen eccentricity all equal the coded forms"
    else
        bad "maxima j2_secular_rates.mac" "expected 4 'derived - coded = 0' lines, found $n"
    fi

    out=$(macf lqr_riccati.mac)
    zeros=$(printf '%s' "$out" | grep -cE ' : 0 *$|: matrix\(\[0,0\],\[0,0\]\) *$')
    stable=$(printf '%s' "$out" | grep -c 'Re = -')
    if [ "$zeros" -eq 5 ] && [ "$stable" -eq 6 ]; then
        ok "maxima lqr_riccati.mac: three CARE residuals vanish and all 6 CW closed-loop eigenvalues are in the left half-plane"
    else
        bad "maxima lqr_riccati.mac" "expected 5 zero residuals and 6 stable eigenvalues, found $zeros and $stable"
    fi

    out=$(macf min_energy_double_integrator.mac)
    t=$(printf '%s' "$out" | grep -cE ': +true *$')
    f=$(printf '%s' "$out" | grep -c 'false')
    if [ "$t" -eq 10 ] && [ "$f" -eq 0 ]; then
        ok "maxima min_energy_double_integrator.mac: Gramian, control law, boundary conditions and costates all check out (10/10)"
    else
        bad "maxima min_energy_double_integrator.mac" "expected 10 true and 0 false, found $t and $f"
    fi

    out=$(macf multistatic_observables.mac)
    ident=$(printf '%s' "$out" | grep -cE '= (0|\[0,0,0\]) +\(0 confirms')
    sheet=$(tozero 'hyperboloid residual[^]]*\]' "$out")
    fd=$(tozero 'central difference - closed form = [^ ]+' "$out")
    if [ "$ident" -eq 3 ] && [ -n "$sheet" ] && [ "$sheet" -ge 30 ] && [ -n "$fd" ] && [ "$fd" -ge 12 ]; then
        ok "maxima multistatic_observables.mac: rate form and both gradients are exact; hyperboloid ~1e-${sheet}, finite difference ~1e-${fd}"
    else
        bad "maxima multistatic_observables.mac" "expected 3 exact identities, found $ident (sheet ~1e-${sheet:-?}, fd ~1e-${fd:-?})"
    fi

    out=$(macf richardson_acceleration.mac)
    if printf '%s' "$out" | grep -q '^ALL ORDERS CONFIRMED: m-point cancels m-1 terms' \
       && ! printf '%s' "$out" | grep -q 'ORDER MISMATCH'; then
        ok "maxima richardson_acceleration.mac: the m-point Richardson diagonal cancels m-1 error terms"
    else
        bad "maxima richardson_acceleration.mac" "an m-point diagonal did not reach order m"
    fi

    # Section 5 substitutes the series back into the exact EOM in the bookkeeping the
    # construction assumes -- amplitudes O(eps), detuning lam^2 - c2 = O(eps^2) -- so orders 1 and 2
    # must vanish outright, and order 3 must vanish in every harmonic except the resonant pair,
    # which need only satisfy Px = k Py because the linear operator at frequency lam is singular.
    # Measured 1e-15 on both libration points against an eps^4 truncation of O(6), asserted at
    # 1e-13 for two decades of headroom.
    out=$(macf richardson_halo.mac)
    lin=$(tozero 'orders eps\^1 and eps\^2 = [0-9.]+e-[0-9]+' "$out")
    nres=$(tozero 'at order eps\^3 = [0-9.]+e-[0-9]+' "$out")
    solv=$(tozero 'Px - k Py\| / \|Px\| = [0-9.]+e-[0-9]+' "$out")
    if printf '%s' "$out" | grep -q 'P3(u) = (5\*uu\^3-3\*uu)/2 *P4(u) = (35\*uu\^4-30\*uu\^2+3)/8' \
       && printf '%s' "$out" | grep -q 'lambda\^2 roots: \[LL = -((sqrt(9\*c2\^2-8\*c2)+c2-2)/2),LL = (sqrt(9\*c2\^2-8\*c2)-c2+2)/2\]' \
       && printf '%s' "$out" | grep -q 'k from balance: \[kk = (lam\^2+2\*c2+1)/(2\*lam)\]' \
       && [ "$(printf '%s' "$out" | grep -c '^  L [12] :')" -eq 6 ] \
       && [ "$(printf '%s' "$out" | grep -c 'eps\^4 truncation of [0-9]\+\.')" -eq 2 ] \
       && [ -n "$lin" ] && [ "$lin" -ge 13 ] \
       && [ -n "$nres" ] && [ "$nres" -ge 13 ] \
       && [ -n "$solv" ] && [ "$solv" -ge 13 ] \
       && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima richardson_halo.mac: the third-order series annihilates orders eps^1..eps^2 (~1e-${lin}) and every non-resonant harmonic of eps^3 (~1e-${nres}), the resonant pair satisfying Px = k Py to ~1e-${solv}"
    else
        bad "maxima richardson_halo.mac" "the series no longer satisfies the EOM (orders 1-2 ~1e-${lin:-?}, non-resonant eps^3 ~1e-${nres:-?}, solvability ~1e-${solv:-?})"
    fi

    out=$(macf shapiro_delay.mac)
    quad=$(tozero 'quadrature - closed form = [^ ]+' "$out")
    if printf '%s' "$out" | grep -q '^d/dl F(l) - 1/r(l) = 0 ' \
       && printf '%s' "$out" | grep -q '^difference from the IERS form = 0 ' \
       && [ -n "$quad" ] && [ "$quad" -ge 15 ]; then
        ok "maxima shapiro_delay.mac: the antiderivative and the IERS closed form are exact, and quadrature agrees to ~1e-${quad} m"
    else
        bad "maxima shapiro_delay.mac" "the closed form does not reproduce the integral (quadrature ~1e-${quad:-?} m)"
    fi

    out=$(macf src_fm_rate.mac)
    if printf '%s' "$out" | grep -q 'lambda x2, f_eta /2 (mig fixed) 5\.0b-1' \
       && printf '%s' "$out" | grep -q 'lambda x2, v x2 *(mig fixed) 5\.0b-1' \
       && printf '%s' "$out" | grep -q 'f_eta x2, v x2 *(mig fixed) 1\.0b0' \
       && printf '%s' "$out" | grep -q 'R0 x2 *5\.0b-1'; then
        ok "maxima src_fm_rate.mac: K_src scales as v^2 lambda^-3 f^-2 r^-1 at fixed range migration"
    else
        bad "maxima src_fm_rate.mac" "a scaling exponent of the secondary range compression FM rate moved"
    fi

    out=$(macf nadir_deflection.mac)
    if printf '%s' "$out" | grep -q 'deflection closed form agrees with the sweep' \
       && printf '%s' "$out" | grep -q 'rate bound holds: the sweep never exceeds'; then
        ok "maxima nadir_deflection.mac: atan(e^2/(2 sqrt(1-e^2))) is the max deflection and e^2/(1-e^2) bounds its rate"
    else
        bad "maxima nadir_deflection.mac" "the closed form or the rate bound disagreed with its own sweep"
    fi


    out=$(macf gpt3_reduction.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 9 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima gpt3_reduction.mac: the barometric law at Tv, the humidity inversion and (p/p0)^(1+lambda) are the shipped lines; Tv is exact in q"
    else
        bad "maxima gpt3_reduction.mac" "expected 9 identities to hold, found $holds"
    fi

    out=$(macf sentman_panel.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 9 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima sentman_panel.mac: the Maxwellian's moments are the shipped c_p and c_t on both sides, and v_ratio is alpha's definition"
    else
        bad "maxima sentman_panel.mac" "expected 9 identities and limits to hold, found $holds"
    fi

    out=$(macf knocke_lambert.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 8 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima knocke_lambert.mac: Lambert's cosine gives the view factor (R/r)^2 exactly; Orekit's gives 1.97x it at 500 km and 2M at the surface"
    else
        bad "maxima knocke_lambert.mac" "expected 8 identities to hold, found $holds"
    fi

    out=$(macf shadow_lens.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 8 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima shadow_lens.mac: Orekit's lens formula is the circle-intersection area, continuous onto both plateaus"
    else
        bad "maxima shadow_lens.mac" "expected 8 identities to hold, found $holds"
    fi

    out=$(macf decimal_year.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 3 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima decimal_year.mac: the ICGEM epoch day count matches the Julian Day Number on all 73414 days of 1900-2100"
    else
        bad "maxima decimal_year.mac" "expected 3 checks to hold, found $holds"
    fi

    out=$(macf lunar_frames.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 13 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima lunar_frames.mac: the PA/ME rotation is the kernel's inverse sequence, and every quoted number follows from its three angles"
    else
        bad "maxima lunar_frames.mac" "expected 13 checks to hold, found $holds"
    fi

    out=$(macf stt_variational.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 11 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima stt_variational.mac: the variational equations hold on a closed-form flow, and the STT moments are exact for a full P0"
    else
        bad "maxima stt_variational.mac" "expected 11 checks to hold, found $holds"
    fi

    out=$(macf gravity_gradient.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 7 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima gravity_gradient.mac: 3 mu/R^5 (R x I R) is the second-order moment of the mass elements, and the slender body's pendulum is stable"
    else
        bad "maxima gravity_gradient.mac" "expected 7 checks to hold, found $holds"
    fi

    out=$(macf solid_tide_love.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 11 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima solid_tide_love.mac: the DEHANTTIDEINEL terms are the Conventions' eqs 7.5-7.11, and the pole tide's 33/9 mm follow from h and l"
    else
        bad "maxima solid_tide_love.mac" "expected 11 checks to hold, found $holds"
    fi

    out=$(macf light_time_partials.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 4 ]; then
        ok "maxima light_time_partials.mac: the light-time partial is a rank-one row update, and a range collapses to 1/(1+rhodot/c)"
    else
        bad "maxima light_time_partials.mac" "expected 4 identities to hold, found $holds"
    fi

    out=$(macf stumpff_quartering.mac)
    t=$(printf '%s' "$out" | grep -cE ': +true *$')
    z=$(printf '%s' "$out" | grep -cE '= +0 *$')
    if [ "$t" -eq 3 ] && [ "$z" -eq 2 ]; then
        ok "maxima stumpff_quartering.mac: both quartering identities and 1 - zS = sin(x)/x hold exactly"
    else
        bad "maxima stumpff_quartering.mac" "expected 3 true and 2 vanishing identities, found $t and $z"
    fi
    out=$(timeout 120 maxima --very-quiet -b sdtm_bates.mac 2>&1)
    zeros=$(printf '%s' "$out" | grep -cE ': 0 *$')
    closed=$(printf '%s' "$out" | grep -oE 'closed form: *[0-9.]+' | grep -oE '[0-9.]+$')
    quad=$(printf '%s' "$out" | grep -oE 'quadrature: *[0-9.]+' | grep -oE '[0-9.]+$')
    if [ "$zeros" -ge 4 ] && [ -n "$closed" ] && [ -n "$quad" ] \
        && awk -v a="$closed" -v b="$quad" 'BEGIN { d = (a - b) / a; if (d < 0) d = -d; exit !(d < 1e-10) }'; then
        ok "maxima sdtm_bates.mac: Bates profile integral, diffusive equilibrium and the density gradient reduce to 0; closed form = quadrature"
    else
        bad "maxima sdtm_bates.mac" "expected 4 identities at 0 and closed form == quadrature, found $zeros / $closed vs $quad"
    fi

else
    skip "maxima derivations" maxima
fi

# ---------------------------------------------------------------------------------------------
# 2. The same two identities in a CAS of a different lineage. `-nosman` or it hangs.
# ---------------------------------------------------------------------------------------------
if have fricas; then
    out=$(timeout 120 fricas -nosman -nox < footprint.input 2>&1)
    zeros=$(printf '%s' "$out" | grep -cE '^ *\([0-9]+\) *0 *$')
    if [ "$zeros" -ge 2 ]; then
        ok "fricas footprint.input: both identities reduce to 0, independently of Maxima"
    else
        bad "fricas footprint.input" "expected 2 zeros, found $zeros"
    fi
else
    skip "fricas cross-check" fricas
fi

# ---------------------------------------------------------------------------------------------
# 3. The exact footprint boundary, at 40 digits, must still match the committed table.
# ---------------------------------------------------------------------------------------------
if have gp; then
    if [ -f edge_precision.txt ]; then
        fresh=$(timeout 120 gp -q edge_precision.gp 2>/dev/null | grep -vE 'realprecision')
        if [ "$fresh" = "$(cat edge_precision.txt)" ]; then
            ok "pari edge_precision.gp: the 40-digit boundary table is reproduced exactly"
        else
            bad "pari edge_precision.gp" "the regenerated table differs from edge_precision.txt"
        fi
    else
        bad "pari edge_precision.gp" "edge_precision.txt is missing"
    fi

    # WGS84_GAMMA. PARI gives the exact decimal; the source holds the nearest f64. Comparing the
    # printed digits is wrong -- 1.0067394967422764... rounds to the double whose shortest repr is
    # ...765. Compare the *bit patterns*, which is the only question that has an answer.
    lit=$(grep -oE '1\.006_739_496_[0-9_]+' ../crates/spacelab-sensors/src/ellipsoid_grid.rs | head -1 | tr -d '_')
    exact=$(timeout 60 gp -q -f /dev/stdin <<'PARI' 2>/dev/null | tail -1
\p 60
f = 1/298.257223563;
printf("%.45g", 1/(1-f)^2);
quit
PARI
)
    if python3 -c "
import sys, struct
lit, exact = sys.argv[1], sys.argv[2].strip()
a = struct.pack('>d', float(lit))
b = struct.pack('>d', float(exact))
sys.exit(0 if a == b else 1)
" "$lit" "$exact"; then
        ok "pari WGS84_GAMMA: the source literal is the nearest f64 to 1/(1-f)^2 (bit-identical)"
    else
        bad "pari WGS84_GAMMA" "source '$lit' is not the nearest f64 to PARI's '$exact'"
    fi
else
    skip "pari checks" gp
fi

# ---------------------------------------------------------------------------------------------
# 4. The |dg/dt| bound must survive being attacked, not merely derived.
# ---------------------------------------------------------------------------------------------
if have octave; then
    out=$(timeout 900 octave --no-gui --quiet step_bound_check.m 2>&1)
    if printf '%s' "$out" | grep -q 'bound holds'; then
        ratio=$(printf '%s' "$out" | grep -oE 'worst  max\|dg/dt\| / bound  = [0-9.]+' | grep -oE '[0-9.]+$')
        ok "octave step_bound_check.m: bound never violated over 3.2e8 samples (worst ratio ${ratio:-?})"
    else
        bad "octave step_bound_check.m" "the bound was violated, or the script did not run"
    fi

    out=$(timeout 600 octave --no-gui --quiet predicate_identical.m 2>&1)
    if printf '%s' "$out" | grep -q 'true .* true'; then
        ok "octave predicate_identical.m: sqrt-free, one-sqrt and asin forms decide identically"
    else
        bad "octave predicate_identical.m" "the three predicate forms diverged"
    fi
else
    skip "octave falsification tests" octave
fi

# ---------------------------------------------------------------------------------------------
# 5. The generators that prove nothing. `csa_phases.mac`, `geopotential.mac` and the six
# `tools/gen_*_oracle.mac` exist only to print high-precision reference values, so there is no
# residual in them to drive to zero -- but there is a claim: the values were pasted into a test,
# and if a generator now prints something else the paste is stale. Numeric comparison, because a
# generator prints 20 or 60 digits where the test stores an f64.
# ---------------------------------------------------------------------------------------------
if have maxima && have python3; then
    if out=$(timeout 1800 python3 oracle_constants.py 2>&1); then
        n=$(printf '%s' "$out" | grep -oE '^[0-9]+ reference values' | grep -oE '^[0-9]+')
        ok "oracle_constants.py: all ${n:-?} generated reference values are still held somewhere in the tree"
    else
        bad "oracle_constants.py" "$(printf '%s' "$out" | grep MISSING | head -3)"
    fi
else
    skip "oracle_constants.py" "maxima and python3"
fi

printf '\n  %d passed, %d failed, %d skipped\n\n' "$PASS" "$FAIL" "$SKIP"
[ "$FAIL" -eq 0 ] || exit 1
