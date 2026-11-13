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
    if [ "$zeros" -ge 18 ] && printf '%s' "$out" | grep -q '^dense output is order 4 at every theta: true' && printf '%s' "$out" | grep -q '^rust D1, D3..D7 are exactly d: true'; then
        ok "maxima dopri5_order.mac: the DOPRI5 tableau satisfies every order condition through 5, its dense output is order 4 at every theta, and dopri5.rs holds its weights exactly"
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

    out=$(timeout 120 maxima --very-quiet -Q -b nrlmsise00_structure.mac 2>&1 </dev/null)
    zeros=$(printf '%s\n' "$out" | grep -cE ': +0 *$')
    if [ "$zeros" -eq 20 ] && ! printf '%s' "$out" | grep -q 'Lisp error'; then
        ok "maxima nrlmsise00_structure.mac: Bates-Walker, diffusive equilibrium, spline slopes, dnet and ccor limits (20 identities)"
    else
        bad "maxima nrlmsise00_structure.mac" "expected 20 identities reducing to 0, found $zeros"
    fi

    out=$(timeout 120 maxima --very-quiet -b adams_coeffs.mac 2>&1)
    if printf '%s' "$out" | grep -qE 'AB8 sum .*: +1' && printf '%s' "$out" | grep -qE 'AM8 sum .*: +1' && [ "$(printf '%s' "$out" | grep -cE 'leading.*: +0')" -ge 2 ] && [ "$(printf '%s' "$out" | grep -cE '^rust (AB8|AM8) is .*: true')" -eq 4 ]; then
        ok "maxima adams_coeffs.mac: AB8/AM8 sum to 1, the leading coefficients derive correctly, and abm.rs holds every coefficient rounded to its digits"
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
    if [ "$bcond" -ge 5 ] && [ "$stiff" -ge 3 ] && [ "$(printf '%s' "$out" | grep -cE 'row-sum a.*: +0')" -ge 3 ] && [ "$(printf '%s' "$out" | grep -cE '^rust (C|A|B) is .*: true')" -eq 3 ]; then
        ok "maxima radau5_coeffs.mac: Radau IIA tableau is order 5 (B(1..5)=0), stiffly accurate, rows sum to c, and radau.rs holds it rounded to its digits"
    else
        bad "maxima radau5_coeffs.mac" "a Radau IIA order/collocation condition failed (B=$bcond stiff=$stiff)"
    fi

    out=$(timeout 120 maxima --very-quiet -b rosenbrock_grk4t.mac 2>&1)
    # the GRK4T stability function must match exp(z) through z^4 (5 MATCH lines: z^0..z^4)
    matches=$(printf '%s' "$out" | grep -cE 'MATCH \(order cond\)')
    if [ "$matches" -ge 5 ] && [ "$(printf '%s' "$out" | grep -cE '^(rust .*|A Gamma .*): true')" -eq 4 ]; then
        ok "maxima rosenbrock_grk4t.mac: the GRK4T stage recurrence reproduces exp(z) through z^4 (order 4), and rosenbrock.rs holds the same doubles"
    else
        bad "maxima rosenbrock_grk4t.mac" "GRK4T stability function did not match exp(z) to order 4 (matches=$matches)"
    fi

    out=$(timeout 120 maxima --very-quiet -b ias15_nodes.mac 2>&1)
    # eight Gauss-Radau nodes (h0=0, h7<1) and the c/d conversion matrices are mutually inverse
    cdinv=$(printf '%s' "$out" | grep -oE 'c.d - I largest entry \(=0\?\): [0-9.]+b-[0-9]+' | grep -oE 'b-[0-9]+' | tr -d 'b-')
    rust=$(printf '%s' "$out" | grep -cE '^rust (H|CC|DD) is the nearest doubles to .*: true')
    if printf '%s' "$out" | grep -qE 'node count .*: +8' && [ -n "$cdinv" ] && [ "$cdinv" -ge 30 ] && [ "$rust" -eq 3 ]; then
        ok "maxima ias15_nodes.mac: 8 Gauss-Radau nodes derived, c and d conversion matrices invert to ~1e-${cdinv}, and ias15.rs holds them to the last digit shown, which parse to their nearest doubles"
    else
        bad "maxima ias15_nodes.mac" "IAS15 node/matrix derivation failed (c.d-I exponent e-${cdinv:-?}, ${rust:-0} of 3 Rust tables match)"
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
    rust=$(printf '%s' "$out" | grep -cE '^rust (C|B|AEX|AIM) is exactly .*: true')
    if [ -n "$exp" ] && [ "$exp" -ge 20 ] && [ "$rust" -eq 4 ]; then
        ok "maxima ark_order.mac: the ARK4(3)6L[2]SA IMEX pair satisfies all 18 order-4 conditions to ~1e-${exp}, and ark.rs holds its rationals exactly"
    else
        bad "maxima ark_order.mac" "an ARK order-4 condition did not vanish (exponent e-${exp:-?}), or ark.rs differs (${rust:-0} of 4 tables match)"
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
    if [ "$sums" -ge 4 ] && printf '%s' "$out" | grep -qE 'SABA2 c1.*\(=0\?\): 0\.0' && printf '%s' "$out" | grep -qE 'SABA2 c2.*\(=0\?\): 0\.0' && [ "$(printf '%s' "$out" | grep -cE '^rust SABA[24]_[CD] is .*: true')" -eq 4 ]; then
        ok "maxima saba_coeffs.mac: SABA_n Gauss-Legendre drift/kick coefficients sum to 1, SABA2 closed form checks, and saba.rs holds them rounded to their digits"
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

    out=$(macf kepler_acceleration_partials.mac)
    if printf '%s' "$out" | grep -q 'jacobian: 9 of 9 residuals are 0' \
       && printf '%s' "$out" | grep -q 'hessian: 27 of 27 residuals are 0'; then
        ok "maxima kepler_acceleration_partials.mac: the two-body Jacobian and Hessian closed forms, 36 residuals 0"
    else
        bad "maxima kepler_acceleration_partials.mac" "a closed form of the two-body partials no longer reduces to 0"
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

    out=$(macf cll_panel.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9][a-z]?\) holds:')
    if [ "$holds" -eq 9 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima cll_panel.mac: Lord's kernels integrate to 1, the mean reflected normal speed is (sqrt(pi a)/2) 1F1(-1/2;1;-q), a = 1 is Sentman's full accommodation and the hyperthermal limit is 2(1 + sqrt(1 - a))"
    else
        bad "maxima cll_panel.mac" "expected 9 identities and limits to hold, found $holds"
    fi

    out=$(macf knocke_lambert.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 27 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima knocke_lambert.mac: Lambert's cosine gives the view factor (R/r)^2 exactly; Orekit's gives 1.97x it at 500 km and 2M at the surface; the central cap is R^2 sin^2 D / d^2, a crown the difference of two caps, the lumped element the cap's D^2 term; an exitance varying across r pushes the flux across it by a closed form; a crown's view factor, first moment and cross flux are closed forms in the slant distance, three or more azimuths sum a linear exitance exactly, and a thin crown's moments are held at 30 digits"
    else
        bad "maxima knocke_lambert.mac" "expected 27 identities to hold, found $holds"
    fi

    out=$(macf sgp4_eccentricity_boundary.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 7 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima sgp4_eccentricity_boundary.mac: SGP4's drag terms above e = 1e-4 shift M and w oppositely, moving the state a e delta sin M radially and 2 a e delta cos M along track, of order a coef B* whatever e"
    else
        bad "maxima sgp4_eccentricity_boundary.mac" "expected 7 identities to hold, found $holds"
    fi

    out=$(macf shadow_lens.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 15 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima shadow_lens.mac: the shipped segment form and Orekit's lens are the circle-intersection area, the series is x - sin x's, and dA/dd is minus the chord"
    else
        bad "maxima shadow_lens.mac" "expected 15 identities to hold, found $holds"
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
    if [ "$holds" -eq 10 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima solid_tide_love.mac: the DEHANTTIDEINEL terms are the Conventions' eqs 7.5-7.11, and the pole tide's 33/9 mm follow from h and l"
    else
        bad "maxima solid_tide_love.mac" "expected 10 checks to hold, found $holds"
    fi

    out=$(macf solid_tide_iers.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 8 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima solid_tide_iers.mac: the force is eqs (6.6)-(6.8), (6.22) and (6.24); 1.333e-9 and the ocean coefficients follow from k2 and Desai; the G3 goldens are the chapter's potential at 40 digits"
    else
        bad "maxima solid_tide_iers.mac" "expected 8 checks to hold, found $holds"
    fi

    out=$(macf ocean_tide_iers.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 4 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima ocean_tide_iers.mac: the force is eq (6.15) under the Doodson-to-Delaunay map of eq 5.43; the G3 goldens are its field and gradient at 40 digits, less only theta's f64 rounding"
    else
        bad "maxima ocean_tide_iers.mac" "expected 4 checks to hold, found $holds"
    fi

    out=$(macf relativity_iers.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 6 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima relativity_iers.mac: the forces are eq (10.12)'s three lines; averaged they give Einstein's perigee advance, the Lense-Thirring rates, the geodesic precession and the chapter's 0.8, 180 and 19 mas/y; the G3 goldens at 40 digits"
    else
        bad "maxima relativity_iers.mac" "expected 6 checks to hold, found $holds"
    fi

    out=$(macf maneuver.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 4 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima maneuver.mac: the mass channel is the rocket equation, the RTN basis and argument of latitude their definitions, the empirical polynomial its Horner loop; the G3 goldens at 40 digits"
    else
        bad "maxima maneuver.mac" "expected 4 checks to hold, found $holds"
    fi

    out=$(macf gnss_mapping.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 6 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima gnss_mapping.mac: GMF and VMF1 are IERS GMF.F and VMF1_HT.F, which reproduce their own test cases, on both hemispheres; VMF3 is TU Wien's vmf3.m; the G3 goldens at 40 digits"
    else
        bad "maxima gnss_mapping.mac" "expected 6 checks to hold, found $holds"
    fi

    out=$(macf vmf_grid.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 3 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima vmf_grid.mac: the shipped weights are the bilinear interpolant, the fixture's rows run in the order the reshape undoes; the G3 goldens are the file's own rows"
    else
        bad "maxima vmf_grid.mac" "expected 3 checks to hold, found $holds"
    fi

    out=$(macf modip_grid.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 4 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima modip_grid.mac: the third-order interpolation is the cubic through its four samples, the Galileo file's wrap rows and columns are the sphere's, the G3 goldens the bicubic through the sixteen nearest samples"
    else
        bad "maxima modip_grid.mac" "expected 4 checks to hold, found $holds"
    fi

    out=$(macf earth_rotation_rate.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 3 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima earth_rotation_rate.mac: EARTH_ROTATION_RATE is the nearest double to the ERA's rate, slowed by 1 - LOD/86400 per SI second; the G3 goldens at 40 digits"
    else
        bad "maxima earth_rotation_rate.mac" "expected 3 checks to hold, found $holds"
    fi

    out=$(macf brouwer_lyddane.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9][ab]?\) holds:')
    if [ "$holds" -eq 9 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima brouwer_lyddane.mac: the J2^2/J4 secular rates by Lie-Deprit, the short-period terms the brackets of a generator solving the homological equation, the long-period terms those of the g-generator, G cos i kept"
    else
        bad "maxima brouwer_lyddane.mac" "expected 9 checks to hold, found $holds"
    fi

    out=$(macf eckstein_hechler.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9][ab]?\) holds:')
    if [ "$holds" -eq 11 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima eckstein_hechler.mac: the first-order short-period terms the brackets of Brouwer's generator to first order in e, da by the energy integral, the second-order rates at e = 0, the averaged long-period dynamics, the J2 part of the eccentricity ellipse"
    else
        bad "maxima eckstein_hechler.mac" "expected 11 checks to hold, found $holds"
    fi

    out=$(macf light_time_down_leg.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 3 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima light_time_down_leg.mac: the two-body orbit by f and g, the down leg the fixed point of c tau = |r(t_r - tau) - s|; the G3 goldens at 40 digits"
    else
        bad "maxima light_time_down_leg.mac" "expected 3 checks to hold, found $holds"
    fi


    out=$(macf ocean_loading.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 11 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima ocean_loading.mac: HARDISP's arguments are SOFA's to 2.2e-6 arcsec and loading.rs writes them, its constituents Darwin's, its spline clamped at parabolic slopes"
    else
        bad "maxima ocean_loading.mac" "expected 11 checks to hold, found $holds"
    fi

    out=$(macf binding_defaults.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 5 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima binding_defaults.mac: every default the Python and C bindings write is its source's value to the digits it shows"
    else
        bad "maxima binding_defaults.mac" "expected 5 checks to hold, found $holds"
    fi

    out=$(macf constants_readback.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 6 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima constants_readback.mac: WGS 84's copies, 2^±53, the conjunction width, the GPU SGP4 set, SGP4's resonance window and Mars's GM are their sources' values"
    else
        bad "maxima constants_readback.mac" "expected 6 checks to hold, found $holds"
    fi

    out=$(macf edl_entry_fixture.mac)
    if printf '%s' "$out" | grep -q '^(1) holds:' && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima edl_entry_fixture.mac: edl.rs's entry-equation fixtures are the 60-digit evaluation at the inputs' doubles"
    else
        bad "maxima edl_entry_fixture.mac" "the fixtures do not hold"
    fi

    # DSST's coefficients and truncation bounds: each recurrence is its closed form, and every row the
    # module's gate reads is the closed form's value as this run prints it.
    out=$(macf dsst_coefficients.mac)
    holds=$(printf '%s' "$out" | grep -cE "^\([0-9]'?\) holds:")
    fixture=../crates/spacelab-propagation/tests/fixtures/dsst_coefficients.csv
    if [ "$holds" -eq 8 ] && printf '%s' "$out" | grep -q '^DONE' \
        && diff -q <(printf '%s\n' "$out" | grep -E '^(dnl|rnml|vns|qns|gs|hs|jacobi_value|jacobi_deriv),') \
            <(grep -v '^#' "$fixture") >/dev/null; then
        ok "maxima dsst_coefficients.mac: V, Q, G/H, Jacobi, D and R recurrences are their closed forms; the 7712 fixture rows are their values"
    else
        bad "maxima dsst_coefficients.mac" "expected 8 checks and the fixture's rows, found $holds checks"
    fi

    # spacelab-fa's fundamental arguments and ERA: every coefficient is the published decimal, and the
    # exact tables fundamental.rs and cio.rs hold the functions to are the polynomials in exact arithmetic.
    out=$(macf fundamental_arguments.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 3 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima fundamental_arguments.mac: the 14 arguments and the ERA are the published decimals; ARGUMENTS_EXACT and ERA_EXACT are their exact values"
    else
        bad "maxima fundamental_arguments.mac" "expected 3 checks to hold, found $holds"
    fi

    # X, Y, s and the IAU 2000A/2006A nutation summed in 40 digits; the file loads
    # fundamental_arguments.mac, so its three checks and DONE come first.
    out=$(macf cio_nutation_series.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 6 ] && [ "$(printf '%s' "$out" | grep -c '^DONE')" -eq 2 ]; then
        ok "maxima cio_nutation_series.mac: XY_EXACT, S_EXACT and NUT_EXACT are the CIO and nutation series summed exactly; the IAU 2006 factors are the published decimals"
    else
        bad "maxima cio_nutation_series.mac" "expected 6 checks to hold, found $holds"
    fi

    # TDB - TT: the Fairhead & Bretagnon series tdb.rs sums, in 40-digit arithmetic; TDB_EXACT is its value.
    out=$(macf tdb_series.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 2 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima tdb_series.mac: the 787-term series and the JPL-mass adjustment summed exactly; TDB_EXACT is their value at eight epochs"
    else
        bad "maxima tdb_series.mac" "expected 2 checks to hold, found $holds"
    fi

    # The certificate for the spacelab-fa tables the three files above wrote from unbounded bigfloats:
    # Arb balls decide that each of the 140 literals is its value rounded to its own digits.
    if have cc && pkg-config --exists flint 2>/dev/null; then
        tmp=$(mktemp -d)
        # shellcheck disable=SC2046 # pkg-config prints separate flags
        if cc -O2 fa_series_arb.c -o "$tmp/arb" $(pkg-config --cflags --libs flint) 2>"$tmp/cc.log" \
            && cert=$(macf fa_series_arb.mac | timeout 120 "$tmp/arb" 2>&1) \
            && printf '%s' "$cert" | grep -q '^CERTIFIED 140 literals'; then
            ok "arb fa_series_arb.c: $cert"
        else
            bad "arb fa_series_arb.c" "$(cat "$tmp/cc.log")${cert:-}"
        fi
        rm -rf "$tmp"
    else
        skip "arb fa_series_arb.c" "a C compiler and FLINT"
    fi

    # The IERS Conventions (2010) Tables 5.3a/5.3b, which `tools/fetch-specs.sh` fetches as
    # digest-named snapshots; the pin is each file's SHA-256, so a table the IERS revised fails here.
    tab53a=../data/specs/iers-tab5.3a.6da73bfe.txt
    tab53b=../data/specs/iers-tab5.3b.f0dff02c.txt
    if [ -f "$tab53a" ] && [ -f "$tab53b" ]; then
        if ! printf '%s  %s\n%s  %s\n' \
            6da73bfe10873ac815520d00fffd67114d647a34afebc5946cfc275e73693f32 "$tab53a" \
            f0dff02c78809b629cc64e2a9fbeffaea5ae20f67e1a62a0ed966f8624807557 "$tab53b" \
            | sha256sum --check --quiet >/dev/null 2>&1; then
            bad "maxima nutation_iers.mac" "the IERS Table 5.3a/5.3b snapshots are not the pinned files"
        else
            out=$(macf nutation_iers.mac)
            holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
            if [ "$holds" -eq 4 ] && printf '%s' "$out" | grep -q '^DONE'; then
                ok "maxima nutation_iers.mac: nut_tables.rs is IERS Tables 5.3a/5.3b row for row; only the 11 out-of-phase rates A''' and B''' are the IERS's alone"
            else
                bad "maxima nutation_iers.mac" "expected 4 checks to hold, found $holds"
            fi
        fi
    else
        skip "maxima nutation_iers.mac" "tools/fetch-specs.sh output"
    fi

    # The IERS Conventions (2010) Tables 5.2a, 5.2b and 5.2d, fetched and pinned the same way.
    tab52a=../data/specs/iers-tab5.2a.19549252.txt
    tab52b=../data/specs/iers-tab5.2b.1f17a3a6.txt
    tab52d=../data/specs/iers-tab5.2d.fe94c83e.txt
    if [ -f "$tab52a" ] && [ -f "$tab52b" ] && [ -f "$tab52d" ]; then
        if ! printf '%s  %s\n%s  %s\n%s  %s\n' \
            19549252df9eb77c8237dbf5b749de82b7fd7713a8b60b35371538cd36b6aa1d "$tab52a" \
            1f17a3a6ad0b468705b3323bfe72d0320ce996798ec7cba0cdde4405d88f14d9 "$tab52b" \
            fe94c83e1ef6f92b15b3f007779ae70c0984c06ee45d5511b4c821ee9a0ecded "$tab52d" \
            | sha256sum --check --quiet >/dev/null 2>&1; then
            bad "maxima cio_iers.mac" "the IERS Table 5.2a/5.2b/5.2d snapshots are not the pinned files"
        else
            out=$(macf cio_iers.mac)
            holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
            if [ "$holds" -eq 3 ] && printf '%s' "$out" | grep -q '^DONE'; then
                ok "maxima cio_iers.mac: cio_tables.rs is IERS Tables 5.2a/5.2b/5.2d term for term, polynomials included"
            else
                bad "maxima cio_iers.mac" "expected 3 checks to hold, found $holds"
            fi
        fi
    else
        skip "maxima cio_iers.mac" "tools/fetch-specs.sh output"
    fi

    # IERS Conventions chapter 6 (2018-02-01) and Desai's ocean pole-tide coefficients, fetched and
    # pinned the same way; the chapter's text and the coefficients are written beside them.
    icc6=../data/specs/iers-icc6.abb3c0b0.pdf
    desai=../data/specs/iers-desai-scopole.075efd2e.gz
    if [ -f "$icc6" ] && [ -f "$desai" ] && command -v pdftotext >/dev/null; then
        if ! printf '%s  %s\n%s  %s\n' \
            abb3c0b0b6d2079b19893d6872a9d8e7320516d9a4736acb5eecaa79c4543388 "$icc6" \
            075efd2ecbf2cd9137ffcbff8a60a75cd9d64a6a10cf271ddd57075257c9d3dc "$desai" \
            | sha256sum --check --quiet >/dev/null 2>&1; then
            bad "maxima tide_step2_iers.mac" "the IERS chapter 6 or Desai snapshot is not the pinned file"
        else
            pdftotext -layout "$icc6" "${icc6%.pdf}.txt"
            gzip -dc "$desai" > "${desai%.gz}.txt"
            out=$(macf tide_step2_iers.mac)
            holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
            if [ "$holds" -eq 6 ] && printf '%s' "$out" | grep -q '^DONE'; then
                ok "maxima tide_step2_iers.mac: tide_step2_tables.rs is Tables 6.5a/b/c row for row; every row's Delaunay multipliers follow from its Doodson number and its amplitudes from its dk"
            else
                bad "maxima tide_step2_iers.mac" "expected 6 checks to hold, found $holds"
            fi
        fi
    else
        skip "maxima tide_step2_iers.mac" "tools/fetch-specs.sh output and pdftotext"
    fi

    out=$(macf sgp4_regime.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 2 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima sgp4_regime.mac: Regime is the document's branch for all 33 published sets, none within 1e-6 of an edge"
    else
        bad "maxima sgp4_regime.mac" "expected 2 checks to hold, found $holds"
    fi

    out=$(macf sgp4_document.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 5 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima sgp4_document.mac: the document's SGP4 at 40 digits reproduces tcppver.out to < 1 mm, ends every published section where it first refuses, and DOCUMENT_REFUSALS and DAY_REFUSALS are its every refusal"
    else
        bad "maxima sgp4_document.mac" "expected 5 checks to hold, found $holds"
    fi

    out=$(macf hansen_zonal.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 4 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima hansen_zonal.mac: Collins' recurrence gives the zonal Hansen kernels' defining orbit average exactly, s <= 8"
    else
        bad "maxima hansen_zonal.mac" "expected 4 checks to hold, found $holds"
    fi

    out=$(macf dsst_zonal_average.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 4 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima dsst_zonal_average.mac: DSST's zonal U is the orbit average for J2..J6, its partials are U's, and J2 gives Kozai's secular rates"
    else
        bad "maxima dsst_zonal_average.mac" "expected 4 checks to hold, found $holds"
    fi

    out=$(macf dsst_third_body_average.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 5 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima dsst_third_body_average.mac: the third-body Hansen kernels are their definition, U is the point-mass orbit average for n = 2..6, its partials are U's"
    else
        bad "maxima dsst_third_body_average.mac" "expected 5 checks to hold, found $holds"
    fi

    out=$(macf dsst_tesseral_expansion.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 7 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima dsst_tesseral_expansion.mac: the tesseral Hansen, G/H, Jacobi and inclination layers multiply out to the order-m harmonic, n <= 5; its partials are its terms'"
    else
        bad "maxima dsst_tesseral_expansion.mac" "expected 7 checks to hold, found $holds"
    fi

    out=$(macf dsst_gaussian_short_period.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 7 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima dsst_gaussian_short_period.mac: rho_j + i sigma_j are the lambda-means of exp(ijL), U1/V1/U2/V2 are the product to sum, and the shipped assembly is F = cos L's closed form to its e^13 cut"
    else
        bad "maxima dsst_gaussian_short_period.mac" "expected 7 checks to hold, found $holds"
    fi

    out=$(macf dsst_srp_conical_shadow.mac)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]+\) holds:')
    if [ "$holds" -eq 25 ] && printf '%s' "$out" | grep -q '^DONE'; then
        ok "maxima dsst_srp_conical_shadow.mac: the shipped S is the conical shadow on the orbit, its quartic in tan((L - L0)/2) and the roots it keeps, the precheck, and the smoothstep of the penumbra rule"
    else
        bad "maxima dsst_srp_conical_shadow.mac" "expected 25 checks to hold, found $holds"
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

    # The DSST Gaussian short periods: the second engine beside dsst_gaussian_short_period.mac.
    out=$(timeout 600 gp -q -f dsst_gaussian_short_period.gp </dev/null 2>&1)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 6 ] && ! printf '%s' "$out" | grep -qF '***' && printf '%s' "$out" | grep -q '^DONE'; then
        ok "pari dsst_gaussian_short_period.gp: rho_j + i sigma_j by residues, the product to sum at JMAX = 12 in exact rationals, F = cos L's closed forms for every ellipse"
    else
        bad "pari dsst_gaussian_short_period.gp" "expected 6 checks to hold and no error, found $holds"
    fi

    # The DSST SRP rates over the conical shadow: the average at 80 digits by two routes, and every
    # row it prints is in the file gated on it, verbatim.
    out=$(timeout 600 gp -q -f dsst_srp_conical_average.gp </dev/null 2>&1)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    rows=$(printf '%s\n' "$out" | grep -E '^ +\(')
    n=$(printf '%s\n' "$rows" | grep -c .)
    if [ "$holds" -eq 5 ] && [ "$n" -eq 34 ] && ! printf '%s' "$out" | grep -qF '***' \
        && printf '%s' "$out" | grep -q '^DONE'; then
        missing=$(printf '%s\n' "$rows" | grep -cvxF -f <(cat ../crates/spacelab-propagation/tests/dsst_srp_conical_reference.rs ../crates/spacelab-propagation/src/dsst/srp.rs))
        if [ "$missing" -eq 0 ]; then
            ok "pari dsst_srp_conical_average.gp: the SRP mean rates and Fourier coefficients over the conical shadow, at 80 digits by two routes, are in the tests verbatim"
        else
            bad "pari dsst_srp_conical_average.gp" "$missing printed rows are not in tests/dsst_srp_conical_reference.rs or src/dsst/srp.rs"
        fi
    else
        bad "pari dsst_srp_conical_average.gp" "expected 5 checks and 34 rows and no error, found $holds and $n"
    fi

    # The lens reference: every row the script prints is in the file gated on it, verbatim, and
    # every value lies inside an Arb ball where a C compiler and FLINT are installed.
    lens=$(timeout 300 gp -q -f shadow_lens.gp </dev/null 2>&1)
    rows=$(printf '%s\n' "$lens" | grep -E '^    \(')
    n=$(printf '%s\n' "$rows" | grep -c .)
    if [ "$n" -eq 184 ] && ! printf '%s' "$lens" | grep -qF '***'; then
        missing=$(printf '%s\n' "$rows" | grep -cvxF -f <(cat ../crates/spacelab-propagation/tests/shadow_lens.rs ../crates/spacelab-propagation/src/shadow.rs))
        if [ "$missing" -eq 0 ]; then
            ok "pari shadow_lens.gp: the 171 lens and 13 x - sin x reference rows, at 200 digits on exact doubles, are in the tests verbatim"
        else
            bad "pari shadow_lens.gp" "$missing printed rows are not in tests/shadow_lens.rs or src/shadow.rs"
        fi
    else
        bad "pari shadow_lens.gp" "expected 184 rows and no error, found $n"
    fi
    if have cc && pkg-config --exists flint 2>/dev/null; then
        tmp=$(mktemp -d)
        # shellcheck disable=SC2046 # pkg-config prints separate flags
        if cc -O2 shadow_lens_arb.c -o "$tmp/arb" $(pkg-config --cflags --libs flint) 2>"$tmp/cc.log" \
            && cert=$(printf '%s\n' "$lens" | timeout 120 "$tmp/arb"); then
            ok "arb shadow_lens_arb.c: $cert"
        else
            bad "arb shadow_lens_arb.c" "$(cat "$tmp/cc.log")${cert:-}"
        fi
        rm -rf "$tmp"
    else
        skip "arb shadow_lens_arb.c" "a C compiler and FLINT"
    fi

    # Chan's series at 80 digits by two routes; every row it prints is in the test, verbatim.
    out=$(timeout 300 gp -q -f conjunction_chan_series.gp </dev/null 2>&1)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:.*, 0 off$')
    rows=$(printf '%s\n' "$out" | grep -E '^        \(')
    n=$(printf '%s\n' "$rows" | grep -c .)
    if [ "$holds" -eq 2 ] && [ "$n" -eq 16 ] && ! printf '%s' "$out" | grep -qF '***' \
        && printf '%s' "$out" | grep -q '^DONE'; then
        missing=$(printf '%s\n' "$rows" | grep -cvxF -f ../crates/spacelab-estimation/src/conjunction.rs)
        if [ "$missing" -eq 0 ]; then
            ok "pari conjunction_chan_series.gp: Chan's series by the Poisson mixture and the Rice integral, every published value that series rounded, the 16 rows in the test verbatim"
        else
            bad "pari conjunction_chan_series.gp" "$missing printed rows are not in src/conjunction.rs"
        fi
    else
        bad "pari conjunction_chan_series.gp" "expected 2 clean checks and 16 rows and no error, found $holds and $n"
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

# ---------------------------------------------------------------------------------------------
# 6. NRLMSISE-00's coefficient tables are NRL's own DATA statements. `tables.rs` claims to be
# generated from BLOCK DATA GTD7BK of `nrlmsise00_sub.for`; this re-runs the generator on the pinned
# file and compares every byte. The pin is the file's SHA-256, so a source that moved fails here
# instead of passing against something else. `tools/fetch-nrlmsise00-fortran.sh` fetches it.
# ---------------------------------------------------------------------------------------------
NRL_FOR="${NRLMSISE00_DIR:-../data/oracles/nrlmsise00}/nrlmsise00_sub.for"
NRL_SHA256=5fc43b10dbd88bdbfbb7c3cf49b9a10ce1016f6183fd3b7396205d43d4ae9f0e
if [ -f "$NRL_FOR" ] && have python3; then
    sha=$(sha256sum "$NRL_FOR" | cut -d' ' -f1)
    regenerated=$(mktemp)
    if [ "$sha" != "$NRL_SHA256" ]; then
        bad "nrlmsise00_tables.py" "$NRL_FOR has sha256 $sha, not the pinned $NRL_SHA256"
    elif ! timeout 300 python3 -I nrlmsise00_tables.py "$NRL_FOR" "$regenerated" >/dev/null 2>&1; then
        bad "nrlmsise00_tables.py" "the generator failed on the pinned NRL FORTRAN"
    elif cmp -s "$regenerated" ../crates/spacelab-propagation/src/nrlmsise00/tables.rs; then
        ok "nrlmsise00_tables.py: tables.rs is byte for byte BLOCK DATA GTD7BK of the pinned NRL FORTRAN"
    else
        bad "nrlmsise00_tables.py" "tables.rs differs from what the pinned NRL FORTRAN regenerates"
    fi
    rm -f "$regenerated"
else
    skip "nrlmsise00_tables.py" "tools/fetch-nrlmsise00-fortran.sh output"
fi

# H3's constants: `h3_constants.mac` claims to be `h3_constants.py`'s reading of Uber's H3 v4.5.0
# `faceijk.c` and `constants.h`. This re-runs the reader on the pinned snapshots `tools/fetch-specs.sh`
# fetches and compares every byte; `h3_definition.mac`, below, proves what the numbers must be.
H3_FACEIJK=../data/specs/h3-faceijk.7828ec42.c
H3_FACEIJK_SHA256=7828ec42137b8134dd1ec209a0d2f393d17893850051dd10a6e63e5b4a36388c
H3_CONSTANTS=../data/specs/h3-constants.5f4068fa.h
H3_CONSTANTS_SHA256=5f4068fa886e1e67b1581d5795f2b034d6c6fadc2c84141b802db2710e3d6450
if [ -f "$H3_FACEIJK" ] && [ -f "$H3_CONSTANTS" ] && have python3; then
    regenerated=$(mktemp)
    if [ "$(sha256sum "$H3_FACEIJK" | cut -d' ' -f1)" != "$H3_FACEIJK_SHA256" ] \
        || [ "$(sha256sum "$H3_CONSTANTS" | cut -d' ' -f1)" != "$H3_CONSTANTS_SHA256" ]; then
        bad "h3_constants.py" "the H3 snapshots are not the pinned v4.5.0 files"
    elif ! timeout 60 python3 -I h3_constants.py "$H3_FACEIJK" "$H3_CONSTANTS" "$regenerated" >/dev/null 2>&1; then
        bad "h3_constants.py" "the reader failed on the pinned H3 files"
    elif cmp -s "$regenerated" h3_constants.mac; then
        ok "h3_constants.py: h3_constants.mac is byte for byte H3 v4.5.0's face centres, axis azimuths and constants"
    else
        bad "h3_constants.py" "h3_constants.mac differs from what the pinned H3 files regenerate"
    fi
    rm -f "$regenerated"
else
    skip "h3_constants.py" "tools/fetch-specs.sh output"
fi

# The FES2004 ocean-tide table: `ocean_tide_fes2004.rs` claims to be generated from the
# `fes2004_Cnm-Snm.dat` the IERS Conventions Centre publishes with chapter 6. This re-runs the
# generator on the pinned snapshot `tools/fetch-specs.sh` fetches and compares every byte.
FES2004_DAT=../data/specs/iers-fes2004.620dc48f.dat
FES2004_SHA256=620dc48fd4d97baa14f532287b19788bd55dc2056e0b478ab0f928b951c194cf
if [ -f "$FES2004_DAT" ] && have python3; then
    regenerated=$(mktemp)
    if [ "$(sha256sum "$FES2004_DAT" | cut -d' ' -f1)" != "$FES2004_SHA256" ]; then
        bad "ocean_tide_fes2004.py" "$FES2004_DAT is not the pinned IERS file"
    elif ! timeout 300 python3 -I ocean_tide_fes2004.py "$FES2004_DAT" "$regenerated" >/dev/null 2>&1; then
        bad "ocean_tide_fes2004.py" "the generator failed on the pinned IERS file"
    elif cmp -s "$regenerated" ../crates/spacelab-propagation/src/ocean_tide_fes2004.rs; then
        ok "ocean_tide_fes2004.py: ocean_tide_fes2004.rs is byte for byte the IERS Conventions Centre's FES2004 file"
    else
        bad "ocean_tide_fes2004.py" "ocean_tide_fes2004.rs differs from what the pinned IERS file regenerates"
    fi
    rm -f "$regenerated"
else
    skip "ocean_tide_fes2004.py" "tools/fetch-specs.sh output"
fi

# SGP4 as the document defines it: `sgp4_vallado.mac` claims to be python-sgp4's `propagation.py`
# (Vallado's routines, MIT) translated statement for statement into Maxima. This re-runs the
# translator on the pinned snapshot `tools/fetch-specs.sh` fetches and compares every byte;
# `sgp4_document.mac`, above, evaluates the translation.
SGP4_PY=../data/specs/python-sgp4-propagation.fa851ca7.py
SGP4_PY_SHA256=fa851ca7d8710eb54d684de61bc84cc8a3687241518e4ee4cfe8249e0fb33a92
if [ -f "$SGP4_PY" ] && have python3; then
    regenerated=$(mktemp)
    if [ "$(sha256sum "$SGP4_PY" | cut -d' ' -f1)" != "$SGP4_PY_SHA256" ]; then
        bad "sgp4_vallado_maxima.py" "$SGP4_PY is not the pinned python-sgp4 file"
    elif ! timeout 120 python3 -I sgp4_vallado_maxima.py "$SGP4_PY" "$regenerated" >/dev/null 2>&1; then
        bad "sgp4_vallado_maxima.py" "the translator failed on the pinned python-sgp4 file"
    elif cmp -s "$regenerated" sgp4_vallado.mac; then
        ok "sgp4_vallado_maxima.py: sgp4_vallado.mac is the pinned propagation.py, translated statement for statement"
    else
        bad "sgp4_vallado_maxima.py" "sgp4_vallado.mac differs from what the pinned propagation.py translates to"
    fi
    rm -f "$regenerated"
else
    skip "sgp4_vallado_maxima.py" "tools/fetch-specs.sh output"
fi

# GPT3's grid as TU Wien publishes it (CC BY 4.0): the vendored asset must be their `gpt3_5.grd`
# with its CRLF line ends made LF, byte for byte. `gpt3_evaluation.mac`, in section 7, evaluates it.
GPT3_GRD=../data/specs/gpt3-5-grd.3bf35c61.grd
GPT3_GRD_SHA256=3bf35c611e7cd5c42a0650ee14652a8f36430127ec213ed7b7b867bd29245b3d
if [ -f "$GPT3_GRD" ]; then
    if [ "$(sha256sum "$GPT3_GRD" | cut -d' ' -f1)" != "$GPT3_GRD_SHA256" ]; then
        bad "gpt3_5.grd" "$GPT3_GRD is not the pinned TU Wien file"
    elif tr -d '\r' < "$GPT3_GRD" | cmp -s - ../crates/spacelab-gnss/assets/gpt3_5.grd; then
        ok "gpt3_5.grd: the vendored GPT3 grid is TU Wien's file, line ends made LF, byte for byte"
    else
        bad "gpt3_5.grd" "the vendored asset differs from TU Wien's gpt3_5.grd"
    fi
else
    skip "gpt3_5.grd" "tools/fetch-specs.sh output"
fi

# The DSST's Gauss-Legendre tables: gauss_legendre_tables.gp generates them from P_n and prints the
# Rust text; the block in gaussian.rs must be that output byte for byte.
if have gp; then
    fresh=$(timeout 120 gp -q -f gauss_legendre_tables.gp </dev/null 2>&1)
    held=$(grep -B1 -E '^const (POINTS|WEIGHTS)_[0-9]+:' ../crates/spacelab-propagation/src/dsst/gaussian.rs | grep -v '^--$')
    if [ -n "$fresh" ] && [ "$fresh" = "$held" ]; then
        ok "pari gauss_legendre_tables.gp: gaussian.rs holds the nearest doubles to the Gauss-Legendre nodes and weights"
    else
        bad "pari gauss_legendre_tables.gp" "gaussian.rs differs from the regenerated Gauss-Legendre tables"
    fi
else
    skip "pari gauss_legendre_tables.gp" gp
fi

# ---------------------------------------------------------------------------------------------
# 7. Ф17.B: each module's gate, derived in Maxima and checked by a second engine. A Maxima file
# prints one `<name>: 0` line per identity and the count is pinned; a PARI or GAP file must run
# without an error and print no `false`, `FAIL` or `not ok`.
# ---------------------------------------------------------------------------------------------
mac_zeros() { # file count what
    local out zeros
    out=$(timeout 300 maxima --very-quiet --suppress-input-echo -Q -b "$1" </dev/null 2>&1)
    zeros=$(printf '%s\n' "$out" | grep -cE ': +0 *$')
    if [ "$zeros" -eq "$2" ] && ! printf '%s' "$out" | grep -q 'Lisp error'; then
        ok "maxima $1: $3 ($2 identities)"
    else
        bad "maxima $1" "expected $2 identities reducing to 0, found $zeros"
    fi
}
second_engine() { # file what
    local out rc
    case "$1" in
        *.gp) out=$(timeout 300 gp -q -f "$1" </dev/null 2>&1); rc=$? ;;
        *.g) out=$(timeout 300 gap -q -A --quitonbreak "$1" 2>&1); rc=$? ;;
    esac
    if [ "$rc" -eq 0 ] && [ -n "$out" ] && ! printf '%s' "$out" | grep -qF '***' \
        && ! printf '%s' "$out" | grep -qiE '\bfalse\b|\bfail|not ok|^Error|Syntax error'; then
        ok "${1##*.} $1: $2"
    else
        bad "${1##*.} $1" "an error, or a check printed false/FAIL (exit $rc)"
    fi
}
if have maxima; then
    mac_zeros gauss_legendre_tables.mac 29 "the DSST Gauss-Legendre tables are the nearest doubles to the roots of P_n and their weights; each rule is exact to degree 2n - 1"
    mac_zeros access_horizon_crossing.mac 4 "a circular pass rises and sets at acos(Re cos e / r) - e"
    mac_zeros root_reference_roots.mac 12 "Cardano, Dottie and W(1) roots the root finder is gated on"
    mac_zeros event_geometry.mac 9 "apparent radius, occulted-rim extremes, elevation and angle forms of the event detectors"
    mac_zeros box_wing_momentum.mac 13 "box-wing facet forces from the momentum balance: Lambert's 2/3, eq. 8-44, drag with specular lift, the Sun-tracking normal"
    mac_zeros metrics_periodic_gaps.mac 8 "coverage metrics of a periodic pass pattern in exact rationals"
    mac_zeros fom_weighted_grid.mac 7 "weighted figures of merit on an exact grid, and their weight-scaling invariance"
    mac_zeros photometry_closed_forms.mac 17 "the diffuse-sphere phase law, V zero point, air mass and limiting magnitude"
    mac_zeros observability_diurnal_arc.mac 15 "the diurnal arc solves the altitude equation; its culminations bound the polar branches; air-mass values; refraction monotone"
    mac_zeros elevation_footprint.mac 8 "the elevation footprint edge from its defining vectors"
    mac_zeros cone_footprint.mac 15 "the cone footprint edge, its tangency with the horizon, the limb, 1 - e^2 = (1 - f)^2"
    mac_zeros comms_link_budget.mac 7 "slant range, free-space loss and the dB link budget"
    mac_zeros sar_incidence.mac 7 "SAR incidence, band edges and look side"
    mac_zeros scan_catalogue.mac 12 "scan-angle catalogue end points, antisymmetry and VIIRS extent"
    mac_zeros wgs84_geodesy.mac 13 "WGS 84 derived constants and both geodetic maps"
    mac_zeros keplerian_elements.mac 31 "element <-> state maps invert each other on every branch; Kepler's equation"
    mac_zeros brouwer_m2_phipps.mac 7 "Phipps' along-track drag drift of a, e and the mean anomaly from Kepler's third law"
    mac_zeros zonal_secular.mac 6 "the first-order J2 secular rates of both zonal mean-element propagators from Lagrange's equations"
    mac_zeros design_orbit.mac 5 "repeat-ground-track and sun-synchronous conditions, critical inclination"
    mac_zeros kepler_universal.mac 9 "universal-variable Kepler equation and f gdot - fdot g = 1"
    mac_zeros lifetime_exponential_decay.mac 5 "the exact erfi decay epoch solves the circular decay ODE"
    mac_zeros split_conformal_coverage.mac 7 "split-conformal rank law, rounding bound, Beta and Beta-binomial laws"
    mac_zeros cluster_design_effect.mac 13 "the design effect, Fisher's variance of its rho, the enumerated cluster bootstrap, the AR(1) factor"
    mac_zeros tle_kozai_recovery.mac 3 "a TLE's Kozai recovery against the relation it inverts: first order, then (4/3) d^2"
    mac_zeros harris_priester_law.mac 16 "the Harris-Priester law: exponential brackets, half-angle power, apex values"
    mac_zeros soc_street_geometry.mac 18 "street-of-coverage widths, co- and counter-rotating node steps, seam and stagger"
    mac_zeros walker_phasing.mac 5 "Walker T/P/F relative phasing as an exact lattice"
    mac_zeros lattice_flower.mac 6 "the lattice flower's Hermite-normal-form system"
    mac_zeros flower_phasing.mac 6 "flower time phasing along the common track and its semi-major axis"
    mac_zeros draim_tetrahedral.mac 5 "Draim's tetrahedral elements, semi-major axis and symmetries"
    mac_zeros shell_surveillance.mac 4 "surveillance-shell characteristic velocity and elevation"
    mac_zeros analysis_alpha_closed_form.mac 7 "the alpha figure on small configurations in closed form"
    mac_zeros mozhaev_point_groups.mac 12 "Mozhaev point groups are the rotation groups of their polyhedra"
    mac_zeros symmetric_regularity.mac 6 "orbit-stabilizer plane counts and staggered-arm regularity"
    mac_zeros rk4_order.mac 9 "RK4 meets every order condition through 4 and none of order 5"
    mac_zeros rkn1210_order.mac 43 "the shipped RKN12(10) tableau meets the Nystrom conditions through 12 and 10"
    mac_zeros gauss_jackson8_generating.mac 48 "Stormer, Cowell and AM8 weights from their generating functions"
    mac_zeros symplectic_composition.mac 14 "leapfrog and the Yoshida compositions: order and time symmetry"
    mac_zeros taylor_jet_recurrences.mac 15 "the Taylor jet recurrences are the exact series"
    mac_zeros cr3bp.mac 20 "CR3BP field as grad Omega plus Coriolis, Jacobi constant, the Lagrange points"
    mac_zeros cr3bp_variational.mac 33 "the variational equations, symplectic form and the collinear Hessian"
    mac_zeros cw.mac 19 "Clohessy-Wiltshire: closed form, unit determinant, limit of exact relative motion"
    mac_zeros lowthrust.mac 12 "low-thrust energy rate, tangential steering, Edelbaum's quasi-circular rate"
    mac_zeros optimal_pmp.mac 16 "minimum-energy PMP: optimal control, Hamilton's equations, primer equation"
    mac_zeros lambert_universal.mac 30 "Lambert in universal variables: y(z), dF/dz, Kepler's equation, the Wronskian"
    mac_zeros iod_classical.mac 29 "Gibbs, Herrick-Gibbs, Gauss and Laplace on their own models"
    mac_zeros transfer_excess_speed.mac 10 "hyperbolic excess speed and C3 on exact conics"
    mac_zeros patched_conics.mac 10 "sphere of influence, flyby hyperbola and the B-plane"
    mac_zeros synodic_frame.mac 14 "the rotating-frame map, its inverse and the transport theorem"
    mac_zeros targeting_hohmann.mac 6 "the apoapsis constraint's root is Hohmann's burn; least-squares point"
    mac_zeros powered_zem_zev.mac 15 "ZEM/ZEV guidance is the open-loop optimum along its own trajectory"
    mac_zeros edl_allen_eggers.mac 7 "Allen-Eggers ballistic entry and its peak deceleration"
    mac_zeros atmosphere_lift.mac 25 "the density lift to SI and its radial gradient -rho/H"
    mac_zeros sdtm_uncertainty.mac 14 "the shipped interval's normal quantiles and log symmetry"
    mac_zeros teme_c1.mac 3 "TEME to ITRF by AIAA 2006-6753 Eq. (C-1): ROT3 composes and fixes the pole; the Appendix C example at 40 digits"
    mac_zeros sdtm_no_cooling.mac 12 "the heating and nitric-oxide cooling step: exact N step and mean, frozen D step, fixed points, local error from h^3"
    mac_zeros sdtm_flux_excess.mac 10 "the saturated daily flux excess D tanh(d/D): the flux rewrite returns it; odd, unit slope, series, bounded by D, linear as D -> inf, monotone"
    mac_zeros sdtm_storm_gain.mac 9 "the storm state's flux gain (F10.7a/150)^kappa: unity at 150, elasticity kappa, reciprocal; a gain on a minor storm and a shift on a large one; the cooling state homogeneous of degree one"
    mac_zeros sdtm_solar_wind.mac 8 "Newell's solar-wind coupling: the half-angle form at the clock angle, symbolic and in every quadrant; zero due north, V^(4/3)|Bz|^(2/3) due south; even in By; homogeneous of degree 4/3 in V and 2/3 in B"
    mac_zeros sdtm_no_step.mac 4 "the exact step of heating with nitric-oxide cooling under a held drive: the rate's integral in closed form, the integrating factor's state equation and start"
    mac_zeros sdtm_column.mac 12 "the column above a height is n T/b in diffusive equilibrium for any T, the Bates column integrated directly, and SDTM-1's column O/N2 at the N2 depth free of Tinf and sigma"
    mac_zeros sdtm_composition.mac 4 "the composition rival n_O0 e^c: d ln rho/dc is the oxygen mass fraction, its complement the rest, the column O/N2 scales as e^c, and the response at a third height with two held"
    mac_zeros sdtm_overcooling.mac 14 "the post-storm rivals: the delta form's frozen step, its shipped limit, steady state and sign region, the odd log regressor; the linear form's exact step, linear steady state and undershoot"
    mac_zeros sdtm_nitric_flux.mac 8 "the nitric-oxide time scaled by the flux: the shipped time at kn = 0, elasticity -kn, the held step exact with the shipped mean, a steady state free of the time, the ratio at two fluxes"
    mac_zeros sdtm_flux_memory.mac 9 "the daily flux's first-order memory: the exact daily step, the shipped flux as tau -> 0, a fixed point, lfilter's recursion, a spike's flux conserved, a held step's approach, the spike's peak"
    mac_zeros gaussian_scores.mac 18 "the normal CRPS's closed form from E|X - y| - E|X - X'|/2, its derivative and its value at 0 against the defining integral, the log score, and the convexity of the continuous sigma's fit"
    mac_zeros central_difference.mac 3 "the nine-node central difference: its weights exact to degree 8, the Lagrange slope at the centre, the first missed term -h^8 f^(9)/630"
    mac_zeros drag_error_growth.mac 6 "a constant along-track force in Clohessy-Wiltshire and in mean elements: the along-track error of a density error eps grows as (3/2) eps a_D t^2"
    mac_zeros sdtm_ablation.mac 12 "SDTM-1's ablation ladder: each rung is the one above it in a limit; share, kappa -> 0, D, H_s -> inf, the kernel's fast time and the filter's time -> 0, the storm regressors with the state"
    mac_zeros density_calibration_ridge.mac 104 "the ridge calibration fit solves its normal equations exactly"
    mac_zeros wind_model_vsh.mac 222 "the vector spherical harmonic basis by Rodrigues' formula; the ridge stationarity"
    mac_zeros attitude_law_frames.mac 75 "every attitude law is an orthonormal right-handed frame"
    mac_zeros analytic_ephemeris.mac 33 "the analytic Sun and Moon series and their precession to J2000"
    mac_zeros thermosphere_legendre_terms.mac 30 "Legendre and local-time terms of a thermosphere expansion"
    mac_zeros kalman_square_root.mac 10 "the square-root Kalman array: Joseph, information form, the trace-minimising gain"
    mac_zeros rts_smoother.mac 5 "RTS and fixed-lag smoothing equal the joint-Gaussian conditional"
    mac_zeros consider_covariance.mac 4 "the consider covariance is the error covariance and reduces at Pcc = 0"
    mac_zeros srukf_square_root.mac 5 "Cholesky update and downdate add or remove exactly one outer product"
    mac_zeros unscented_transform.mac 9 "sigma points reproduce mean and covariance; exactness on quadratics"
    mac_zeros hinf_riccati.mac 6 "the H-infinity Riccati recursion, Kalman at gamma = 0, the prior's threshold"
    mac_zeros chi_square_quantiles.mac 4 "NEES/NIS quadratic forms and exact chi-square quantiles"
    mac_zeros robust_weights.mac 22 "Huber and Student-t weights from their defining equations, limits and efficiency"
    mac_zeros lie_quaternion.mac 12 "quaternion product, passive Rodrigues matrix, exp and log"
    mac_zeros se3_exponential.mac 22 "SE(3) exponential, both Jacobians and the adjoint from their series"
    mac_zeros se23_invariant_navigation.mac 19 "SE2(3) group-affine strapdown and the invariant error dynamics"
    mac_zeros iekf_attitude.mac 8 "invariant EKF: state-independent error transition and H = [r]x"
    mac_zeros mekf_attitude.mac 8 "multiplicative EKF: error kinematics, H, Q and the norm-preserving reset"
    mac_zeros imm_mixing.mac 9 "IMM mixing weights are distributions and conserve the bank's moments"
    mac_zeros gaussian_mixture_moments.mac 9 "mixture collapse and merging preserve the first two moments"
    mac_zeros particle_resampling.mac 14 "systematic, stratified and residual resampling are unbiased"
    mac_zeros ensemble_kalman_gain.mac 5 "the ensemble gain is the Kalman gain of the sample covariance"
    mac_zeros sampling_transforms.mac 13 "Box-Muller, Cholesky sampling and the xoshiro256++ recurrences"
    mac_zeros conjunction_pc.mac 32 "Foster, Patera and Chan collision probability against the defining integral"
    mac_zeros screening_tca.mac 19 "closest approach as the root of the range rate; the radial filter bound"
    mac_zeros lincov_recursion.mac 15 "linear covariance analysis from the error dynamics"
    mac_zeros measurement_modifiers.mac 8 "Shapiro delay in PPN form and its partials"
    mac_zeros multistatic_fdoa.mac 5 "FDOA as the first-order term of the relativistic Doppler law"
    mac_zeros radiometric_observables.mac 18 "Moyer's ramped Doppler and range, the corona delay and the plasma coefficient"
    mac_zeros exact_predicates.mac 18 "orient and in-sphere predicates as exact determinants; the filter bound"
    mac_zeros hull_delaunay.mac 10 "Delaunay and convex hull against their brute-force definitions"
    mac_zeros spherical_triangle.mac 14 "spherical excess as the solid angle of a triangle"
    mac_zeros igrf_synthesis.mac 37 "IGRF field as minus the gradient of the Schmidt-normalised potential"
    # Diurnal aberration at aberration_erfa.rs's five stations: speed and the shift across the motion.
    out=$(timeout 300 maxima --very-quiet --suppress-input-echo -Q -b diurnal_aberration.mac </dev/null 2>&1)
    holds=$(printf '%s' "$out" | grep -cE '^\([0-9]\) holds:')
    if [ "$holds" -eq 2 ] && printf '%s' "$out" | grep -q '^DONE' && ! printf '%s' "$out" | grep -q 'Lisp error'; then
        ok "maxima diurnal_aberration.mac: DIURNAL_EXACT is omega times the distance from the axis and asin(v/c) across the motion"
    else
        bad "maxima diurnal_aberration.mac" "expected 2 checks to hold, found $holds"
    fi
    mac_zeros geomagnetic_dipole.mac 22 "the centred dipole's pole, moment and field"
    mac_zeros attitude_control.mac 18 "attitude error kinematics, the Lyapunov derivative and the PD loop"
    mac_zeros gmst82.mac 10 "GMST 1982 as the exact rational polynomial"
    mac_zeros cio_rotation.mac 8 "the CIO-based rotation as the IERS closed form in X, Y, s"
    mac_zeros gnss_combinations.mac 32 "IF, GF, wide/narrow-lane and MW combinations as exact rationals of the ICD carriers"
    mac_zeros gnss_cycle_slip.mac 13 "the slip predictor is the least-squares polynomial extrapolation"
    mac_zeros gnss_dop.mac 31 "DOP from the cofactor traces; closed forms for zenith-plus-ring geometries"
    mac_zeros h3_definition.mac 12 "the H3 grid by its definition: an icosahedron's faces, axes on the vertices, u = (3 - sqrt 5)/2, the aperture-7 turn, 2 + 120 7^r cells, 24 cell centres"
    mac_zeros gpt3_evaluation.mac 13 "GPT3 from its definition: the grid's harmonics, height reduction and bilinear cell reproduce gpt3() at 10 sites"
    mac_zeros gps_broadcast_orbit.mac 8 "the IS-GPS-200 user algorithm, its velocity and the relativistic term"
    mac_zeros glonass_pz90_motion.mac 10 "the PZ-90 equations of motion as the J2 gradient plus rotating-frame terms"
    mac_zeros slr_two_way_range.mac 4 "the SLR legs as roots of the light-time equations"
    mac_zeros kronrod_g7k15.mac 92 "the ICD's Gauss-Kronrod G7/K15 nodes and weights from their defining equations"
    mac_zeros nequick_ray.mac 30 "the NeQuick ray perigee and slant geometry"
    mac_zeros nequick_fourier.mac 8 "the CCIR Fourier collapse of the ICD maps"
    mac_zeros nequick_parameters.mac 18 "the NeQuick profile parameters and the solar declination"
    mac_zeros gauss2_richardson.mac 18 "the two-point Gauss rule and its Richardson extrapolation"
    mac_zeros swath_sphere.mac 12 "the swath edge as the near root of the line-of-sight quadratic"
    mac_zeros observation_geometry.mac 11 "ENU sun/sensor angles, the scattering angle and Kasten-Young air mass"
    mac_zeros stereo_disparity.mac 11 "disparity f B / z on a rectified pair and the sub-pixel parabola vertex"
    mac_zeros plane_sweep_homography.mac 9 "the plane-induced homography and the gain-invariant patch score"
    mac_zeros pinhole_collinearity.mac 20 "the collinearity equations, their Jacobians and the Schur step"
    mac_zeros rpc00b.mac 8 "the RPC00B cubic basis, its gradient and the Newton inverse"
    mac_zeros rpc_bias_affine.mac 4 "the RPC bias-compensation normal equations"
    mac_zeros ortho_bilinear_dem.mac 12 "bilinear DEM interpolation and the ray-DEM root"
    mac_zeros ray_ellipsoid_geolocation.mac 25 "the ray-ellipsoid hit, geodetic latitude and the subpoint"
    mac_zeros gnomonic_tan_plate.mac 11 "the TAN projection of Paper II, the plate fit and SIP"
    mac_zeros quad_hash.mac 13 "the quad code and its similarity invariance"
    mac_zeros source_moments.mac 8 "the MAD-to-sigma factor and the blob moments"
    mac_zeros relativistic_aberration.mac 6 "relativistic aberration as a Lorentz boost"
    mac_zeros detection_box_geometry.mac 10 "IoU, greedy NMS, anchors and the box parameterisation"
    mac_zeros harris_structure_tensor.mac 5 "the Harris response from the structure tensor"
    mac_zeros edge_mtf.mac 8 "the MTF of a Gaussian edge and the WGS 84 radii of curvature"
    mac_zeros pansharpen_component_substitution.mac 8 "Gram-Schmidt and Brovey component substitution"
    mac_zeros fft_radix2.mac 5 "the radix-2 FFT as the DFT, its inverse and Parseval"
    mac_zeros polsar_decomposition.mac 19 "coherency matrix and H/A/alpha of the canonical scatterers"
    mac_zeros radon_transform.mac 14 "the Ram-Lak filter, the Gaussian projection and filtered back-projection"
    mac_zeros psinsar_sbas.mac 12 "the SBAS normal equations and the height sensitivity"
    mac_zeros insar_unwrap.mac 16 "coherence, residues, the DCT Poisson unwrapping and the forward DCT's defining sum"
    mac_zeros sar_range_doppler.mac 18 "the chirp matched filter, its sinc limit and the range-Doppler terms"
    mac_zeros csa_point_target.mac 6 "the chirp-scaling phase multiplies compose to a pure delay"
    mac_zeros tomosar_beamforming.mac 16 "Bartlett and Capon spectra and the tomographic steering phase"
    mac_zeros phase_functions.mac 34 "phase-function normalisation, asymmetry and Legendre moments"
    mac_zeros mie_lorenz.mac 30 "the Lorenz-Mie coefficients and efficiencies, Rayleigh limits"
    mac_zeros gas_k_distribution.mac 10 "the k-distribution band transmittance and its merge rule"
    mac_zeros slab_single_scattering.mac 12 "the slab's first-collision expectation and conservation"
    mac_zeros six_s_coupling_brdf.mac 15 "the 6S coupling inverse, Gauss-Legendre on [0, 1] and the BRDF kernels"
    mac_zeros ice_geometric_optics.mac 22 "the ice-crystal geometric optics: Fresnel, halos, J1 and Airy"
else
    skip "Ф17.B Maxima gates" maxima
fi
if have maxima; then
    out=$(timeout 300 maxima --very-quiet --suppress-input-echo -Q -b jacchia_bowman_quartic.mac </dev/null 2>&1)
    zeros=$(printf '%s\n' "$out" | grep -cE '= +0 *$')
    if [ "$zeros" -eq 5 ] && ! printf '%s' "$out" | grep -q 'Lisp error'; then
        ok "maxima jacchia_bowman_quartic.mac: Jacchia's quartic below 125 km solves eqs. (10)-(11) (5 residuals)"
    else
        bad "maxima jacchia_bowman_quartic.mac" "expected 5 residuals reducing to 0, found $zeros"
    fi
    mac_zeros jacchia_bowman_hermite.mac 4 "the 1000-1500 km density factor meets its four end conditions"
    mac_zeros jacchia_bowman_diurnal.mac 6 "eq. (17) in one bracket, the diurnal extremes, Boole's weights"
    for f in jacchia_bowman_static_reference jacchia_bowman_full_model_terms; do
        out=$(timeout 600 maxima --very-quiet --suppress-input-echo -Q -b "$f.mac" </dev/null 2>&1)
        if [ $? -eq 0 ] && ! printf '%s' "$out" | grep -qE 'Lisp error|incorrect syntax'; then
            ok "maxima $f.mac: the independent reference the Jacchia-Bowman tests compare against"
        else
            bad "maxima $f.mac" "the reference generator failed"
        fi
    done
fi
if have gp; then
    for f in access_horizon_crossing root_reference_roots event_geometry box_wing_momentum brouwer_m2_phipps h3_definition photometry_closed_forms observability_diurnal_arc \
        elevation_footprint cone_footprint comms_link_budget sar_incidence scan_catalogue \
        wgs84_geodesy keplerian_elements design_orbit kepler_universal lifetime_exponential_decay split_conformal_coverage cluster_design_effect tle_kozai_recovery harris_priester_law \
        soc_street_geometry flower_phasing draim_tetrahedral shell_surveillance analysis_alpha_closed_form \
        rk4_order rkn1210_order gauss_jackson8_generating symplectic_composition taylor_jet_recurrences \
        cr3bp cr3bp_variational cw lowthrust optimal_pmp lambert_universal iod_classical patched_conics \
        targeting_hohmann powered_zem_zev edl_allen_eggers sdtm_uncertainty sdtm_no_cooling sdtm_flux_excess sdtm_storm_gain sdtm_solar_wind sdtm_ablation sdtm_no_step sdtm_column sdtm_composition sdtm_overcooling sdtm_nitric_flux sdtm_flux_memory gaussian_scores drag_error_growth central_difference \
        density_calibration_ridge \
        wind_model_vsh attitude_law_frames analytic_ephemeris thermosphere_legendre_terms \
        chi_square_quantiles robust_weights se3_exponential particle_resampling sampling_transforms \
        conjunction_pc screening_tca lincov_recursion measurement_modifiers exact_predicates \
        hull_delaunay spherical_triangle igrf_synthesis geomagnetic_dipole attitude_control gmst82 \
        cio_rotation gnss_combinations gpt3_evaluation gps_broadcast_orbit glonass_pz90_motion slr_two_way_range \
        kronrod_g7k15 cll_panel nequick_fourier gauss2_richardson swath_sphere observation_geometry \
        stereo_disparity pinhole_collinearity rpc_bias_affine ortho_bilinear_dem \
        ray_ellipsoid_geolocation gnomonic_tan_plate quad_hash source_moments \
        relativistic_aberration detection_box_geometry harris_structure_tensor edge_mtf \
        pansharpen_component_substitution fft_radix2 polsar_decomposition radon_transform \
        psinsar_sbas insar_unwrap sar_range_doppler tomosar_beamforming phase_functions mie_lorenz \
        gas_k_distribution slab_single_scattering six_s_coupling_brdf ice_geometric_optics; do
        second_engine "$f.gp" "second engine for $f.mac"
    done
    second_engine linear_filters.gp "second engine for the Kalman, RTS, consider, H-infinity and UT fixtures"
    second_engine multistatic_observables.gp "multistatic rates as numerical derivatives of the separation"
    second_engine radiometric_corona.gp "the corona electron column in arc length against its closed form"
    second_engine rpc00b_inverse.gp "the RPC inverse roots on the Pleiades fixture at 60 digits"
else
    skip "Ф17.B PARI gates" gp
fi
if have gap; then
    for f in metrics_periodic_gaps fom_weighted_grid kernel_lattice walker_phasing lattice_flower flower_phasing \
        mozhaev_point_groups symmetric_regularity gnss_cycle_slip gnss_dop; do
        second_engine "$f.g" "exact counts for the $f gate"
    done
else
    skip "Ф17.B GAP gates" gap
fi

printf '\n  %d passed, %d failed, %d skipped\n\n' "$PASS" "$FAIL" "$SKIP"
[ "$FAIL" -eq 0 ] || exit 1
