#= The scalar transfer equation in a homogeneous plane-parallel slab over a black surface, solved by
   doubling and adding (plans/17 Ф19.G): the reference spacelab-earthvision's Monte Carlo is held to.

   Directions are the double-Gauss rule, N Gauss-Legendre nodes on each hemisphere's (0, 1), plus
   zero-weight nodes at the cosines a test asks about, so their rows and columns come out of the same
   matrices without entering any integral. The reflection function R and the diffuse transmission T
   are expanded in azimuth, R = sum_m (2 - delta_m0) R_m cos(m phi), and for every m the layers
   compose the same way, (A o B)_m = 2 int_0^1 A_m B_m mu dmu, so with C = diag(2 w mu) (zero at the
   extra nodes) and E = diag(exp(-t/mu)) two identical layers of depth t make one of depth 2t by the
   adding equations of Hansen & Travis (1974, Space Sci. Rev. 16, 527):

       Q = R C R,  S = (I - Q C)^-1 Q,  D = T + S E + S C T,  U = R E + R C D,
       R2 = R + E U + T C U,  T2 = E D + T E + T C D.

   The start is a layer of depth tau / 2^n thin enough for single scattering,
   R_m = omega dtau p_m(-mu mu') / (2 mu mu'), T_m = omega dtau p_m(+mu mu') / (2 mu mu'), where p_m is
   the m-th cosine coefficient of p(cos Theta) in azimuth, cos Theta = -+ mu mu' + s s' cos phi,
   taken by the trapezoid rule on a periodic integrand. The phase functions are the module's own,
   normalised to int_-1^1 p dmu = 1: Henyey-Greenstein, its two-lobe mixture and Rayleigh's 3/8 (1 + mu^2).
   R_m is the module's pi I / (mu_s F) (derivations/slab_single_scattering.mac (1)).

   The path reflectance at an azimuth adds the exact single-scattering term,
   omega p(cos Theta) (1 - exp(-tau (1/mu_s + 1/mu_v))) / (2 (mu_s + mu_v)), to the cosine series of
   the multiple scattering alone, which needs far fewer orders than the whole. Rayleigh's p has no
   order above 2.

   Modes, from derivations/:
       timeout 1800 julia --startup-file=no --project=@. slab_doubling.jl slab
       ... aeronet       writes crates/spacelab-earthvision/tests/fixtures/aeronet_coupling_doubling.txt
       ... rayleigh      writes crates/spacelab-earthvision/tests/fixtures/rayleigh_doubling.txt
       ... reduced       writes slab_doubling_reduced.txt, which slab_doubling.mac recomputes
       ... convergence   the studies the bounds rest on
   Standard library only.                                                                          =#

using LinearAlgebra
using Printf

const FIXTURES =
    joinpath(@__DIR__, "..", "crates", "spacelab-earthvision", "tests", "fixtures")

# ---- quadrature ------------------------------------------------------------------------------------

"Gauss-Legendre nodes and weights on (0, 1), by Newton on P_n in 256-bit arithmetic."
function gauss01(n::Int)
    setprecision(BigFloat, 256) do
        x = zeros(BigFloat, n)
        w = zeros(BigFloat, n)
        for i = 1:n
            z = cos(BigFloat(pi) * (i - big"0.25") / (n + big"0.5"))
            dp = big"0.0"
            for _ = 1:200
                p0, p1 = one(z), z
                for k = 2:n
                    p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
                end
                dp = n * (z * p1 - p0) / (z^2 - 1)
                dz = p1 / dp
                z -= dz
                abs(dz) < big"1e-70" && break
            end
            x[i] = (1 - z) / 2
            w[i] = 1 / ((1 - z^2) * dp^2)
        end
        Float64.(x), Float64.(w)
    end
end

# The four-node rule coupling_from_slab integrates the spherical albedo on, as atmospheric_correction.rs
# writes it.
const GAUSS_01 = [
    (
        parse(Float64, "6.943184420297371238802675555359524745214e-2"),
        parse(Float64, "1.739274225687269286865319746109997036177e-1"),
    ),
    (
        parse(Float64, "3.300094782075718675986671204483776563997e-1"),
        parse(Float64, "3.260725774312730713134680253890002963823e-1"),
    ),
    (
        parse(Float64, "6.699905217924281324013328795516223436003e-1"),
        parse(Float64, "3.260725774312730713134680253890002963823e-1"),
    ),
    (
        parse(Float64, "9.305681557970262876119732444464047525479e-1"),
        parse(Float64, "1.739274225687269286865319746109997036177e-1"),
    ),
]

# ---- phase functions -------------------------------------------------------------------------------

abstract type Phase end
struct HG <: Phase
    g::Float64
end
struct DoubleHG <: Phase
    forward::Float64
    backward::Float64
    weight::Float64
end
struct Rayleigh <: Phase end

hg(g, mu) = (1 - g^2) / (2 * (1 + g^2 - 2g * mu)^1.5)
value(p::HG, mu) = hg(p.g, mu)
value(p::DoubleHG, mu) = p.weight * hg(p.forward, mu) + (1 - p.weight) * hg(p.backward, mu)
value(::Rayleigh, mu) = 0.375 * (1 + mu^2)

"""
The cosine coefficients p_m, m = 0..M, of p(sign mu mu' + s s' cos phi) over phi, for every pair of
nodes: `(1/pi) int_0^pi p cos(m phi) dphi` by the trapezoid rule on K points of the period.
"""
function fourier(phase::Phase, mu::Vector{Float64}, sign::Float64, M::Int, K::Int)
    n = length(mu)
    s = sqrt.(max.(0.0, 1 .- mu .^ 2))
    half = K ÷ 2
    phi = [2pi * k / K for k = 0:half]
    weight = [k == 0 || k == half ? 1.0 : 2.0 for k = 0:half] ./ K
    basis = [weight[k+1] * cos(m * phi[k+1]) for k = 0:half, m = 0:M]
    out = zeros(n, n, M + 1)
    row = zeros(half + 1)
    for j = 1:n, i = 1:n
        for k = 0:half
            row[k+1] = value(
                phase,
                clamp(sign * mu[i] * mu[j] + s[i] * s[j] * cos(phi[k+1]), -1.0, 1.0),
            )
        end
        out[i, j, :] = basis' * row
    end
    out
end

# ---- doubling --------------------------------------------------------------------------------------

"""
R_m and T_m of a slab of depth `tau` and albedo `omega`, from the thin start `pr`, `pt` (the phase
coefficients of this order), doubled `n` times.
"""
function slab_order(pr, pt, mu, c, tau, omega, n)
    dtau = tau / 2.0^n
    scale = omega * dtau ./ (2 .* mu .* mu')
    R = pr .* scale
    T = pt .* scale
    t = dtau
    for _ = 1:n
        e = exp.(-t ./ mu)
        Q = R * (c .* R)
        S = (I - Q .* c') \ Q
        D = T + S .* e' + S * (c .* T)
        U = R .* e' + R * (c .* D)
        R, T = R + e .* U + T * (c .* U), e .* D + T .* e' + T * (c .* D)
        t *= 2
    end
    R, T
end

"""
Every order of a slab on the nodes `mu` (the first `N` of them the Gauss nodes): the vector of
(R_m, T_m) for m = 0..M.
"""
function solve(phase::Phase, tau, omega, mu, c; M = 0, K = 512, n = 34)
    pr = fourier(phase, mu, -1.0, M, K)
    pt = fourier(phase, mu, 1.0, M, K)
    [slab_order(pr[:, :, m+1], pt[:, :, m+1], mu, c, tau, omega, n) for m = 0:M], pr
end

"The double-Gauss rule with zero-weight extra nodes appended: (mu, C)."
function nodes(N, extra)
    x, w = gauss01(N)
    vcat(x, extra), vcat(2 .* w .* x, zeros(length(extra)))
end

plane_albedo(R0, c, j) = sum(c .* R0[:, j])
total_transmittance(T0, c, j, mu, tau) = sum(c .* T0[:, j]) + exp(-tau / mu[j])

"The path reflectance R(mu_v, mu_s, phi): exact single scattering plus the multiple-scattering series."
function path(orders, pr, phase, tau, omega, mu, iv, is, phi)
    mv, ms = mu[iv], mu[is]
    attenuation = 1 - exp(-tau * (1 / ms + 1 / mv))
    cos_theta = -ms * mv + sqrt(1 - ms^2) * sqrt(1 - mv^2) * cos(phi)
    single =
        omega * value(phase, clamp(cos_theta, -1.0, 1.0)) * attenuation / (2 * (ms + mv))
    multiple = 0.0
    for (m, (R, _)) in enumerate(orders)
        order = m - 1
        single_m = omega * pr[iv, is, m] * attenuation / (2 * (ms + mv))
        multiple += (order == 0 ? 1.0 : 2.0) * (R[iv, is] - single_m) * cos(order * phi)
    end
    single + multiple
end

# ---- the seven slab cases of phase_function_slab.rs -------------------------------------------------

const SLAB_CASES = [
    ("isotropic thin", 0.0, 1.0, 1.0, HG(0.0)),
    ("isotropic", 1.0, 1.0, 1.0, HG(0.0)),
    ("isotropic absorbing", 1.0, 0.9, 0.6, HG(0.0)),
    ("hg g=0.5", 2.0, 1.0, 0.8, HG(0.5)),
    ("hg g=0.85", 4.0, 0.99, 0.7, HG(0.85)),
    ("hg g=-0.4", 1.5, 0.85, 1.0, HG(-0.4)),
    ("two-lobe", 3.0, 0.95, 0.75, DoubleHG(0.85, -0.6, 0.6)),
]

function slab_case(tau, omega, mu0, phase; N = 128, n = 34)
    tau == 0.0 && return (0.0, 1.0)
    mu, c = nodes(N, [mu0])
    orders, _ = solve(phase, tau, omega, mu, c; n = n)
    R0, T0 = orders[1]
    j = length(mu)
    plane_albedo(R0, c, j), total_transmittance(T0, c, j, mu, tau)
end

function slab_mode()
    for (name, tau, omega, mu0, phase) in SLAB_CASES
        r, t = slab_case(tau, omega, mu0, phase; n = 40)
        @printf("%-20s R %.15f  T %.15f\n", name, r, t)
    end
end

# ---- the AERONET atmospheres ------------------------------------------------------------------------

const MU_VIEW = 0.8
const AZIMUTHS_DEG = (0.0, 90.0, 180.0)

data_lines(path) = [l for l in readlines(path) if !startswith(l, "#") && !isempty(strip(l))]

"T-down, T-up, the four-node and the exact spherical albedo, and the path at the three azimuths."
function coupling(tau, omega, g, mu_sun; N = 96, M = 40, n = 34)
    extra = vcat([mu_sun, MU_VIEW], first.(GAUSS_01))
    mu, c = nodes(N, extra)
    phase = HG(g)
    orders, pr = solve(phase, tau, omega, mu, c; M = M, n = n)
    R0, T0 = orders[1]
    is, iv = N + 1, N + 2
    down = total_transmittance(T0, c, is, mu, tau)
    up = total_transmittance(T0, c, iv, mu, tau)
    albedo4 = sum(
        w * plane_albedo(R0, c, N + 2 + k) * 2 * m for (k, (m, w)) in enumerate(GAUSS_01)
    )
    albedo = sum(c[j] * plane_albedo(R0, c, j) for j = 1:N)
    paths =
        [path(orders, pr, phase, tau, omega, mu, iv, is, deg2rad(a)) for a in AZIMUTHS_DEG]
    down, up, albedo4, albedo, paths
end

function aeronet_mode()
    rows = data_lines(joinpath(FIXTURES, "aeronet_atmosphere.txt"))
    out = [
        "# The coupling terms of the AERONET atmospheres of aeronet_atmosphere.txt, by doubling and adding.",
        "# Generated by derivations/slab_doubling.jl aeronet -- do not edit; derivations/slab_doubling.mac",
        "# recomputes the method on a reduced input.",
        "# 96 Gauss nodes per hemisphere, 2^-34 of the depth to start, 41 azimuthal orders of the",
        "# multiple scattering; Henyey-Greenstein in closed form.",
        "# View cosine 0.8; azimuths 0, 90, 180 degrees from the forward-scattering half-plane.",
        "# spherical_albedo_g4 is int_0^1 R(mu) 2 mu dmu on coupling_from_slab's four nodes;",
        "# spherical_albedo is the same integral on the 96 nodes.",
        "#",
        "# site  date  time_utc  wavelength_nm  transmittance_down  transmittance_up  spherical_albedo_g4  spherical_albedo  path_reflectance_0deg  path_reflectance_90deg  path_reflectance_180deg",
    ]
    for line in rows
        f = split(line)
        site, date, time, sza, nm = f[1], f[3], f[4], parse(Float64, f[5]), f[6]
        tau, omega, g = parse.(Float64, f[7:9])
        down, up, a4, a, paths = coupling(tau, omega, g, cos(deg2rad(sza)))
        push!(
            out,
            @sprintf(
                "%-16s %s %s %6d %.12f %.12f %.12f %.12f %.12f %.12f %.12f",
                site,
                date,
                time,
                parse(Int, nm),
                down,
                up,
                a4,
                a,
                paths...
            )
        )
    end
    write(joinpath(FIXTURES, "aeronet_coupling_doubling.txt"), join(out, "\n") * "\n")
    println("aeronet_coupling_doubling.txt: ", length(rows), " rows")
end

# ---- Rayleigh at the geometries of the libRadtran table -----------------------------------------------

function rayleigh_mode(; N = 64, n = 34)
    rows = [
        parse.(Float64, split(l)) for
        l in data_lines(joinpath(FIXTURES, "pyspectral_rayleigh.txt"))
    ]
    out = [
        "# Conservative Rayleigh slab, path reflectance pi I / (mu_s F) over a black surface, by doubling and adding.",
        "# Generated by derivations/slab_doubling.jl rayleigh -- do not edit; derivations/slab_doubling.mac",
        "# recomputes the method on a reduced input. Geometry and Bodhaine depths as pyspectral_rayleigh.txt;",
        "# 64 Gauss nodes per hemisphere, 2^-34 of the depth to start, orders 0-2 (Rayleigh has no other).",
        "#",
        "# wavelength_nm  optical_depth  mu_sun  mu_view  azimuth_deg  path_reflectance",
    ]
    phase = Rayleigh()
    for tau in unique(r[2] for r in rows)
        group = [r for r in rows if r[2] == tau]
        extra = unique(vcat([r[3] for r in group], [r[4] for r in group]))
        mu, c = nodes(N, extra)
        orders, pr = solve(phase, tau, 1.0, mu, c; M = 2, n = n)
        index(x) = N + findfirst(==(x), extra)
        for r in group
            value = path(
                orders,
                pr,
                phase,
                tau,
                1.0,
                mu,
                index(r[4]),
                index(r[3]),
                deg2rad(r[5]),
            )
            push!(
                out,
                @sprintf(
                    "%7.1f %.12f %.6f %.6f %6.1f %.12f",
                    r[1],
                    tau,
                    r[3],
                    r[4],
                    r[5],
                    value
                )
            )
        end
    end
    write(joinpath(FIXTURES, "rayleigh_doubling.txt"), join(out, "\n") * "\n")
    println("rayleigh_doubling.txt: ", length(rows), " rows")
end

# ---- the reduced input slab_doubling.mac recomputes ----------------------------------------------------

# Three Gauss nodes, whose exact values Maxima knows, plus 3/5 and 4/5; ten doublings; K = 64.
const REDUCED = [
    ("isotropic", 1.0, 0.9, HG(0.0), 0),
    ("hg", 2.0, 0.95, HG(0.5), 3),
    ("rayleigh", 0.3, 1.0, Rayleigh(), 2),
]

function reduced_mode()
    out = String[]
    for (_, tau, omega, phase, M) in REDUCED
        mu, c = nodes(3, [0.6, 0.8])
        orders, _ = solve(phase, tau, omega, mu, c; M = M, K = 64, n = 10)
        for (R, T) in orders, j = 4:5, i = 4:5
            push!(out, @sprintf("%.17e %.17e", R[i, j], T[i, j]))
        end
    end
    write(joinpath(@__DIR__, "slab_doubling_reduced.txt"), join(out, "\n") * "\n")
    println("slab_doubling_reduced.txt: ", length(out), " rows")
end

# ---- convergence ---------------------------------------------------------------------------------------

function convergence_mode()
    println(
        "slab cases: the shipped N = 128, n = 40, and a 2^-34 start, against N = 256, n = 40",
    )
    for (name, tau, omega, mu0, phase) in SLAB_CASES[2:end]
        best = slab_case(tau, omega, mu0, phase; N = 256, n = 40)
        shipped = slab_case(tau, omega, mu0, phase; N = 128, n = 40)
        start = slab_case(tau, omega, mu0, phase; N = 128, n = 34)
        @printf(
            "  %-20s shipped %.2e  2^-34 start %.2e  R + T - 1 %s\n",
            name,
            maximum(abs.(shipped .- best)),
            maximum(abs.(start .- best)),
            omega == 1.0 ? @sprintf("%.2e", sum(best) - 1) : "-"
        )
    end

    rows = [split(l) for l in data_lines(joinpath(FIXTURES, "aeronet_atmosphere.txt"))]
    column(k) = [parse(Float64, r[k]) for r in rows]
    tau, omega, g = column(7), column(8), column(9)
    println("AERONET: the shipped N = 96, M = 40, n = 34 against N = 192, M = 80, n = 40")
    for (label, i) in (
        ("deepest", argmax(tau)),
        ("most forward", argmax(g)),
        ("most absorbing", argmin(omega)),
    )
        mu_sun = cos(deg2rad(parse(Float64, rows[i][5])))
        best = coupling(tau[i], omega[i], g[i], mu_sun; N = 192, M = 80, n = 40)
        got = coupling(tau[i], omega[i], g[i], mu_sun)
        d = maximum(abs.(vcat(collect(got[1:4]) .- collect(best[1:4]), got[5] .- best[5])))
        @printf(
            "  %-15s tau %.4f omega %.4f g %.4f  worst %.2e\n",
            label,
            tau[i],
            omega[i],
            g[i],
            d
        )
    end

    rows = [
        parse.(Float64, split(l)) for
        l in data_lines(joinpath(FIXTURES, "pyspectral_rayleigh.txt"))
    ]
    println(
        "Rayleigh: the shipped N = 64, n = 34 against N = 128, n = 40, at every geometry of the table",
    )
    for depth in (maximum(r[2] for r in rows), minimum(r[2] for r in rows))
        group = [r for r in rows if r[2] == depth]
        extra = unique(vcat([r[3] for r in group], [r[4] for r in group]))
        values = map([(64, 34), (128, 40)]) do (N, n)
            mu, c = nodes(N, extra)
            orders, pr = solve(Rayleigh(), depth, 1.0, mu, c; M = 2, n = n)
            index(x) = N + findfirst(==(x), extra)
            [
                path(
                    orders,
                    pr,
                    Rayleigh(),
                    depth,
                    1.0,
                    mu,
                    index(r[4]),
                    index(r[3]),
                    deg2rad(r[5]),
                ) for r in group
            ]
        end
        @printf("  tau %.4f  worst %.2e\n", depth, maximum(abs.(values[1] .- values[2])))
    end
end
const MODES = Dict(
    "slab" => slab_mode,
    "aeronet" => aeronet_mode,
    "rayleigh" => rayleigh_mode,
    "reduced" => reduced_mode,
    "convergence" => convergence_mode,
)
MODES[isempty(ARGS) ? "slab" : ARGS[1]]()
