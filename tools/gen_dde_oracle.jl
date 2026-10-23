# DelayDiffEq.jl as an independent numerical opinion on x'(t) = -x(t - 1), x(t) = 1 for t <= 0.
#
# The primary oracle for this problem is the exact closed form in derivations/dde_method_of_steps.mac
# -- Maxima builds the piecewise polynomials by rational integration and checks its own recursion
# symbolically (x and x' continuous at every breakpoint, the DDE residual identically zero). Nothing
# numerical can beat that. What DelayDiffEq adds is a *different implementation of the same idea*:
# an independent method of steps, with its own breakpoint tracking, written by people who were not
# reading our code. If the two agree to their tolerances the driver's structure is confirmed as well
# as its arithmetic.
#
# Reference-only, never a runtime dependency, and never installed into a project environment:
#
#   julia --startup-file=no --project=/tmp/dde-oracle tools/gen_dde_oracle.jl
#
# DelayDiffEq.jl is MIT. `Vern9` lives in OrdinaryDiffEqVerner, which DelayDiffEq depends on but
# does not re-export, so it is imported by name rather than assumed.

using DelayDiffEq
using OrdinaryDiffEqVerner: Vern9
using Printf

const TAU = 1.0

history(p, t) = [1.0]

prob = DDEProblem(
    (du, u, h, p, t) -> (du[1] = -h(p, t - TAU)[1]),
    [1.0],
    history,
    (0.0, 8.0);
    constant_lags = [TAU],
)

sol = solve(prob, MethodOfSteps(Vern9()); abstol = 1e-14, reltol = 1e-14)

# Maxima's closed form, 40 significant digits truncated to Float64 -- the same 16 values the Rust
# test asserts against.
reference = [
    (0.5, 5.0e-1),
    (1.0, 0.0),
    (1.5, -3.75e-1),
    (2.0, -5.0e-1),
    (2.5, -3.958333333333333333e-1),
    (3.0, -1.666666666666666667e-1),
    (3.5, 6.510416666666666667e-2),
    (4.0, 2.083333333333333333e-1),
    (4.5, 2.315104166666666667e-1),
    (5.0, 1.583333333333333333e-1),
    (5.5, 4.351128472222222222e-2),
    (6.0, -5.694444444444444444e-2),
    (6.5, -1.078791542658730159e-1),
    (7.0, -1.029761904761904762e-1),
    (7.5, -5.935785202752976190e-2),
    (8.0, -4.538690476190476190e-3),
]

worst = 0.0
println("   t        DelayDiffEq            Maxima closed form     |difference|")
for (t, exact) in reference
    got = sol(t)[1]
    d = abs(got - exact)
    global worst = max(worst, d)
    @printf("%5.1f   %+.17e   %+.17e   %.3e\n", t, got, exact, d)
end
@printf("\nworst |DelayDiffEq - closed form| over the 16 points: %.3e\n", worst)
@printf("steps taken: %d\n", length(sol.t))
