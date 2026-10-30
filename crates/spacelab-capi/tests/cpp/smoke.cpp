/*
 * The C++ side of the boundary, exercised as a consumer sees it.
 *
 * This is not a second copy of the Rust suite. Those tests already prove that each entry point is
 * its crate function; what cannot be proved from inside Rust is everything that only exists once a
 * C++ compiler is involved -- that the header parses under `-Wall -Wextra -Wpedantic -Werror`,
 * that the structs really do have the layout both sides assume, that the static library links,
 * that a failure arrives as an exception and not as a status nobody read, and that the handle
 * releases itself. Those are the claims here, and the numbers are checked against the same values
 * the Rust tests assert so that a disagreement is about the boundary and not about the physics.
 *
 * No test framework, deliberately: a dependency to assert `a == b` would be a dependency in the
 * one place this repository is trying to demonstrate it has none.
 */

#include "spacelab.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char *what, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, line, what);
        ++failures;
    }
}

void check_near(double got, double want, double tolerance, const char *what, int line) {
    if (!(std::fabs(got - want) <= tolerance)) {
        std::fprintf(stderr, "FAIL %s:%d: %s -- got %.17g, want %.17g +- %.3g\n", __FILE__, line,
                     what, got, want, tolerance);
        ++failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_NEAR(got, want, tol) check_near((got), (want), (tol), #got, __LINE__)

/* The element set the Rust suite uses, so the two sides are answering the same question. Pointers
 * rather than `std::string`, so that nothing in this file allocates before `main` runs. */
constexpr const char *LINE1 =
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
constexpr const char *LINE2 =
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

spacelab::state leo() {
    const double a = 6778137.0;
    const double speed = std::sqrt(spacelab::mu_earth / a);
    const double inc = 51.6 * M_PI / 180.0;
    return spacelab::state{
        .position = {.x = a, .y = 0.0, .z = 0.0},
        .velocity = {.x = 0.0, .y = speed * std::cos(inc), .z = speed * std::sin(inc)}};
}

/* The distance of a position from the centre. Named because three of these tests want it and the
 * open-coded form is where the missing parentheses were. */
double radius_of(const spacelab::vec3 &r) {
    return std::sqrt((r.x * r.x) + (r.y * r.y) + (r.z * r.z));
}

/*
 * Layout. Every one of these is an assumption both sides make and neither compiler checks: the
 * Rust `#[repr(C)]` structs and these C ones are separate declarations that happen to agree, and
 * "happen to" is the part worth a test. A `static_assert` that `sl_state` is six packed doubles is
 * also what makes it legal to hand `std::vector<state>::data()` to the batch call.
 */
void the_structs_have_the_layout_both_sides_assume() {
    static_assert(sizeof(spacelab::vec3) == 3 * sizeof(double));
    static_assert(sizeof(spacelab::state) == 6 * sizeof(double));
    static_assert(alignof(spacelab::state) == alignof(double));
    static_assert(offsetof(spacelab::state, position) == 0);
    static_assert(offsetof(spacelab::state, velocity) == 3 * sizeof(double));
    static_assert(sizeof(spacelab::geodetic) == 3 * sizeof(double));
    static_assert(sizeof(spacelab::latlon) == 2 * sizeof(double));

    /* Trivially copyable, or none of the above buys anything: it is what lets a vector of them be
     * passed straight across the boundary with no marshalling step to get wrong. */
    static_assert(std::is_trivially_copyable_v<spacelab::state>);
    static_assert(std::is_standard_layout_v<spacelab::state>);

    /* The library reads every enum as a 32-bit integer. A C enum is only as wide as the compiler
     * decides, so these two are the check that this translation unit agrees -- they fail under
     * -fshort-enums if the _FORCE_INT32 enumerators are ever dropped from the header. */
    static_assert(sizeof(::sl_precision) == sizeof(std::int32_t));
    static_assert(sizeof(::sl_walker_pattern) == sizeof(std::int32_t));
    static_assert(sizeof(::sl_sensor_kind) == sizeof(std::int32_t));
    static_assert(offsetof(::sl_walker_params, inclination_rad) == sizeof(double));

    CHECK(spacelab::abi_version() == SL_ABI_VERSION);
    CHECK(!spacelab::version().empty());
}

/* Geodesy, against the same measured z the Rust test pins. Radians, both sides. */
void the_geodetic_round_trip_agrees_with_the_rust_measurement() {
    const spacelab::geodetic g{.latitude_rad = 35.0 * M_PI / 180.0,
                               .longitude_rad = 139.0 * M_PI / 180.0,
                               .height_m = 42.0};
    const spacelab::vec3 r = spacelab::geodetic_to_itrf(g);
    CHECK_NEAR(r.z, 3637891.0, 1.0);

    const spacelab::geodetic back = spacelab::itrf_to_geodetic(r);
    CHECK_NEAR(back.latitude_rad, g.latitude_rad, 1e-12);
    CHECK_NEAR(back.longitude_rad, g.longitude_rad, 1e-12);
    CHECK_NEAR(back.height_m, g.height_m, 1e-6);
}

/*
 * The handle releases itself, the batch is the singles, and a bad element set throws rather than
 * returning a zero state. The scope around the first satellite is the point of the class: nothing
 * in this function calls a close.
 */
void the_sgp4_class_owns_its_handle_and_batches_correctly() {
    const std::vector<double> offsets{0.0, 45.0, 90.0, -30.0};
    std::vector<spacelab::state> batch;
    std::vector<spacelab::state> singles;

    {
        const spacelab::sgp4 sat(LINE1, LINE2);
        batch = sat.at(std::span<const double>(offsets));
        for (const double m : offsets) {
            singles.push_back(sat.at(m));
        }
    }

    CHECK(batch.size() == offsets.size());
    for (std::size_t i = 0; i < batch.size(); ++i) {
        CHECK(batch.at(i).position.x == singles.at(i).position.x);
        CHECK(batch.at(i).position.y == singles.at(i).position.y);
        CHECK(batch.at(i).position.z == singles.at(i).position.z);
        CHECK(batch.at(i).velocity.x == singles.at(i).velocity.x);
    }

    /* The ISS is where the ISS is. Measured at the element set's own epoch: 6 720 189.23 m from
     * the centre, which is 342 km up -- an ISS altitude, and not the zero a mis-linked call or a
     * dropped out-parameter would leave in the caller's struct. */
    CHECK_NEAR(radius_of(batch.at(0).position), 6720189.23, 0.01);

    CHECK(spacelab::sgp4_propagate(LINE1, LINE2, 45.0).position.x == singles.at(1).position.x);

    /* Propagating no epochs is an empty answer, not a failure. The C entry point refuses the null
     * an empty span carries, and it is right to; the wrapper knows the length and answers. */
    {
        const spacelab::sgp4 sat(LINE1, LINE2);
        CHECK(sat.at(std::span<const double>{}).empty());
    }

    bool threw = false;
    try {
        const spacelab::sgp4 broken("...", "...");
        (void)broken;
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_PARSE);
        CHECK(std::string(e.what()).find("invalid TLE") != std::string::npos);
    }
    CHECK(threw);
}

/* Lambert between two points of one orbit returns that orbit's own velocities, and an impossible
 * transfer arrives as a domain exception carrying the code a caller can branch on. */
void lambert_returns_the_orbit_it_was_asked_about() {
    const spacelab::state start = leo();
    const double tof = 1800.0;
    const spacelab::state arrival = spacelab::kepler_propagate(start, spacelab::mu_earth, tof);

    const spacelab::transfer t =
        spacelab::lambert(start.position, arrival.position, tof, spacelab::mu_earth);
    CHECK_NEAR(t.departure.x, start.velocity.x, 1e-6);
    CHECK_NEAR(t.departure.y, start.velocity.y, 1e-6);
    CHECK_NEAR(t.departure.z, start.velocity.z, 1e-6);
    CHECK_NEAR(t.arrival.y, arrival.velocity.y, 1e-6);

    bool threw = false;
    try {
        (void)spacelab::lambert(start.position, arrival.position, 0.0, spacelab::mu_earth);
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_DOMAIN);
    }
    CHECK(threw);
}

/* The two-call protocol, hidden behind a vector. The infeasible design still surfaces as the
 * domain error the C API names rather than as the empty vector it would be easy to return. */
void the_generators_size_their_own_buffers() {
    spacelab::walker_params w = spacelab::default_walker_params();
    CHECK(w.mu == spacelab::mu_earth);
    w.pattern = SL_WALKER_DELTA;
    w.inclination_rad = 56.0 * M_PI / 180.0;
    w.planes = 6;
    w.per_plane = 4;
    w.phasing = 1;
    w.semi_major_axis_m = 29600e3;

    const std::vector<spacelab::state> walker = spacelab::walker(w);
    CHECK(walker.size() == 24);
    for (const spacelab::state &s : walker) {
        CHECK_NEAR(radius_of(s.position), w.semi_major_axis_m, 1.0);
    }

    spacelab::soc_params p = spacelab::default_soc_params();
    CHECK(p.earth_radius_m == spacelab::re_earth);
    p.inclination_rad = M_PI / 2.0;
    p.altitude_m = 1200e3;
    p.min_elevation_rad = 10.0 * M_PI / 180.0;

    const spacelab::soc_geometry g = spacelab::street_of_coverage_geometry(p);
    const std::vector<spacelab::state> soc = spacelab::street_of_coverage(p);
    CHECK(soc.size() == static_cast<std::size_t>(g.planes) * g.per_plane);

    p.min_elevation_rad = M_PI / 2.0;
    bool threw = false;
    try {
        (void)spacelab::street_of_coverage(p);
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_DOMAIN);
        CHECK(std::string(e.what()).find("no street tiles") != std::string::npos);
    }
    CHECK(threw);
}

/* Design and measurement, the total functions among them returning values rather than throwing. */
void the_design_and_measurement_functions_answer_in_radians() {
    CHECK_NEAR(spacelab::critical_inclination(), std::acos(1.0 / std::sqrt(5.0)), 1e-12);

    const double sso = spacelab::sun_synchronous_inclination(7078e3);
    CHECK(sso > M_PI / 2.0);
    CHECK_NEAR(sso * 180.0 / M_PI, 98.2, 0.5);

    bool threw = false;
    try {
        (void)spacelab::sun_synchronous_inclination(42164e3);
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_DOMAIN);
    }
    CHECK(threw);

    /* Zero is not a repeat cycle, and the C++ default arguments must not paper over it. */
    threw = false;
    try {
        (void)spacelab::repeat_ground_track_sma(0, 3, 1.7);
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_INVALID_ARGUMENT);
    }
    CHECK(threw);

    const spacelab::state sat = leo();
    const spacelab::vec3 station{.x = 4194303.0, .y = 172460.0, .z = 4780100.0};
    const double lat = 48.8 * M_PI / 180.0;
    const double lon = 2.35 * M_PI / 180.0;
    CHECK(spacelab::range(sat, station) > 0.0);
    CHECK(std::fabs(spacelab::elevation(sat, station, lat, lon)) <= M_PI / 2.0);
    CHECK(std::fabs(spacelab::azimuth(sat, station, lat, lon)) <= 2.0 * M_PI);
    CHECK(std::isfinite(spacelab::range_rate(sat, station)));

    const spacelab::latlon start{.latitude_rad = 0.4, .longitude_rad = -1.2};
    const spacelab::latlon end = spacelab::ground_point_at(start, 0.05, 1.9);
    CHECK(std::fabs(end.latitude_rad) <= M_PI / 2.0);
    CHECK(spacelab::horizon_look_angle(spacelab::re_earth, 500e3) > 0.0);
}

/* The frame chain runs, and the two frames differ by the precession the Rust test measured. */
void the_frame_chain_runs_and_the_two_frames_differ_by_precession() {
    const spacelab::state s = leo();
    const std::int64_t ut1 = 813456789012345678LL;
    const std::int64_t tt = 813456789081184678LL;

    const spacelab::state teme = spacelab::teme_to_itrf(s, ut1, tt, 1.2e-7, -3.4e-7, 1.5e-3);
    const spacelab::state gcrf =
        spacelab::gcrf_to_itrf(s, ut1, tt, 1.2e-7, -3.4e-7, 5.6e-10, -7.8e-10, 1.5e-3);

    const double dx = teme.position.x - gcrf.position.x;
    const double dy = teme.position.y - gcrf.position.y;
    CHECK_NEAR(std::sqrt((dx * dx) + (dy * dy)), 39072.5, 1.0);

    /* Defaulted Earth-orientation arguments: the header says zeros give a GMST-only rotation, so
     * the short call must be the long call with zeros and not something else. */
    const spacelab::state bare = spacelab::teme_to_itrf(s, ut1, tt);
    const spacelab::state explicit_zeros = spacelab::teme_to_itrf(s, ut1, tt, 0.0, 0.0, 0.0);
    CHECK(bare.position.x == explicit_zeros.position.x);
    CHECK(bare.position.x != teme.position.x);
}

/*
 * The dense ephemeris as a C++ consumer holds it: a factory that can fail, a destructor that
 * cannot be forgotten, and a span-checked query. The scope is the point -- nothing here closes.
 */
void the_ephemeris_class_owns_its_handle_and_refuses_to_extrapolate() {
    const spacelab::state start = leo();
    const std::vector<double> times{0.0, 600.0, 1200.0, 1800.0};
    std::vector<spacelab::state> sampled;

    {
        const spacelab::result<spacelab::ephemeris> eph =
            spacelab::ephemeris::j2(start, 1800.0, spacelab::mu_earth, spacelab::j2_earth);
        CHECK(static_cast<bool>(eph));
        if (!eph) {
            return;
        }

        const auto [t0, t1] = eph->span();
        CHECK(t0 == 0.0);
        CHECK(t1 == 1800.0);
        const auto [accepted, rejected] = eph->step_counts();
        CHECK(accepted > 0);
        CHECK(rejected < accepted);

        sampled = eph->at(std::span<const double>(times));
        CHECK(sampled.size() == times.size());
        for (std::size_t i = 0; i < times.size(); ++i) {
            CHECK(eph->at(times.at(i)).position.x == sampled.at(i).position.x);
        }

        bool threw = false;
        try {
            (void)eph->at(-1.0);
        } catch (const spacelab::error &e) {
            threw = true;
            CHECK(e.status() == SL_ERR_DOMAIN);
        }
        CHECK(threw);

        /* An empty request is an empty answer, as it is for the batch propagator. */
        CHECK(eph->at(std::span<const double>{}).empty());
    }

    /* The orbit is still an orbit: a J2 LEO does not change its radius by kilometres in half an
     * hour, and this is what a mis-wired interpolant would break. */
    for (const spacelab::state &s : sampled) {
        CHECK_NEAR(radius_of(s.position), 6778137.0, 25000.0);
    }

    /* A degree the model does not carry, and a libration point that does not exist. */
    bool threw = false;
    try {
        (void)spacelab::geopotential_acceleration(100000, start.position);
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_DOMAIN);
    }
    CHECK(threw);

    const std::array<spacelab::vec3, 5> points = spacelab::libration_points();
    CHECK(points.at(0).y == 0.0);
    CHECK(points.at(3).y > 0.0);
    CHECK(spacelab::richardson_halo(2, 0.002).position.x > 0.0);

    /* L3 is outside the third-order theory, and the wrapper says so rather than returning a state
     * that looks like an orbit. */
    threw = false;
    try {
        (void)spacelab::richardson_halo(3, 0.002);
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_DOMAIN);
    }
    CHECK(threw);
}

/*
 * DSST through the C++ face: the element sets come back as a `std::vector<equinoctial>` sized by
 * the wrapper, and the numbers are the ones the C suite and the Rust tests pin, so all three are
 * provably running the same arithmetic.
 */
void the_dsst_wrapper_sizes_its_own_answer_and_matches_the_other_suites() {
    const spacelab::dsst_params p = spacelab::default_dsst_params();
    CHECK(p.zonal_count == 4);
    CHECK(std::size(p.zonal) == SL_DSST_MAX_ZONAL);

    const double inclination = 51.6 * M_PI / 180.0;
    const spacelab::equinoctial initial{7.0e6, 1.0e-3, 0.0, std::tan(inclination / 2.0), 0.0, 0.0};
    const std::vector<double> times{1800.0, 86400.0};

    const std::vector<spacelab::equinoctial> mean = spacelab::dsst_zonal_mean(p, initial, times);
    CHECK(mean.size() == 2);
    CHECK_NEAR(mean.at(0).at(0), 7.0e6, 1e-6);
    CHECK_NEAR(mean.at(1).at(0), 7.0e6, 1e-6);
    CHECK_NEAR(mean.at(0).at(5), 1.940206765692, 1e-9);
    CHECK_NEAR(mean.at(1).at(4), -3.765956823994e-2, 1e-12);

    const std::vector<spacelab::equinoctial> osculating =
        spacelab::dsst_zonal_osculating(p, initial, times);
    CHECK_NEAR(osculating.at(0).at(0), 6.9957388684e6, 1.0);

    CHECK_NEAR(spacelab::dsst_mean(p, initial, times).at(1).at(0), 6.9999999844e6, 1.0);
    CHECK_NEAR(spacelab::dsst_osculating(p, initial, times).at(0).at(0), 6.9957388043e6, 1.0);

    /* No epochs is an empty answer, as it is for `sgp4::at`. */
    CHECK(spacelab::dsst_zonal_mean(p, initial, std::span<const double>{}).empty());

    spacelab::dsst_params broken = p;
    broken.zonal_count = 0;
    const spacelab::result<std::vector<spacelab::equinoctial>> refused =
        spacelab::try_dsst_zonal_mean(broken, initial, times);
    CHECK(!refused);
    CHECK(refused.error() == SL_ERR_INVALID_ARGUMENT);
}

/*
 * The least-squares wrapper takes an ordinary C++ callable, so the model can be a lambda that
 * captures. The problem is the one the C and Rust suites solve: six unit measurements, whose
 * answer and covariance are known in closed form.
 */
void the_least_squares_wrapper_takes_a_lambda_and_propagates_its_exception() {
    const std::array<double, 6> target{1.0, -2.0, 3.5, 0.25, -0.75, 10.0};
    const double weight = 4.0;
    std::uint32_t calls = 0;

    auto model = [&](std::span<const double, 6> state, std::span<spacelab::residual_row> rows) {
        ++calls;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            for (std::size_t j = 0; j < 6; ++j) {
                rows[i].jacobian[j] = (i == j) ? 1.0 : 0.0;
            }
            rows[i].residual = state[i] - target.at(i);
            rows[i].weight = weight;
        }
    };

    const spacelab::estimate solved = spacelab::gauss_newton(model, {0, 0, 0, 0, 0, 0}, 6);
    CHECK(solved.converged);
    CHECK(calls >= solved.iterations);
    for (std::size_t i = 0; i < 6; ++i) {
        CHECK_NEAR(solved.state[i], target.at(i), 1e-9);
        CHECK_NEAR(solved.covariance[(i * 6) + i], 1.0 / weight, 1e-12);
    }

    /* Options are a plain struct with the C defaults in it. */
    spacelab::gauss_newton_options options = spacelab::default_gauss_newton_options();
    options.max_iterations = 1;
    const spacelab::estimate capped = spacelab::gauss_newton(model, {0, 0, 0, 0, 0, 0}, 6, options);
    CHECK(!capped.converged);
    CHECK(capped.iterations == 1);

    /* Too few rows to determine six directions is a status, not a wrong answer. */
    const spacelab::result<spacelab::estimate> narrow =
        spacelab::try_gauss_newton(model, {0, 0, 0, 0, 0, 0}, 1);
    CHECK(!narrow);
    CHECK(narrow.error() == SL_ERR_DOMAIN);

    /*
     * A model that throws is the reason the wrapper exists rather than the raw callback: the
     * exception comes back out here, from this `try`, instead of unwinding through Rust.
     */
    bool rethrown = false;
    auto angry = [](std::span<const double, 6>, std::span<spacelab::residual_row>) {
        throw std::runtime_error("the arc has a gap in it");
    };
    try {
        (void)spacelab::try_gauss_newton(angry, {0, 0, 0, 0, 0, 0}, 6);
    } catch (const std::runtime_error &e) {
        rethrown = std::string(e.what()) == "the arc has a gap in it";
    }
    CHECK(rethrown);
}

/* The third RAII handle, and the incremental half of the estimation API. */
void the_information_filter_owns_its_handle_and_accumulates() {
    const std::array<double, 6> target{1.0, -2.0, 3.5, 0.25, -0.75, 10.0};

    spacelab::result<spacelab::srif> opened = spacelab::srif::open();
    CHECK(static_cast<bool>(opened));
    if (!opened) {
        return;
    }
    spacelab::srif &f = *opened;

    const spacelab::result<std::array<double, 6>> too_soon = f.try_solve();
    CHECK(!too_soon);
    CHECK(too_soon.error() == SL_ERR_DOMAIN);

    for (std::size_t i = 0; i < 6; ++i) {
        std::array<double, 6> row{};
        row.at(i) = 1.0;
        f.add_row(row, target.at(i), 4.0);
    }

    const std::array<double, 6> x = f.solve();
    const std::array<double, 36> covariance = f.covariance();
    for (std::size_t i = 0; i < 6; ++i) {
        CHECK_NEAR(x.at(i), target.at(i), 1e-12);
        CHECK_NEAR(covariance.at((i * 6) + i), 0.25, 1e-12);
    }

    const std::pair<double, std::uint32_t> stats = f.residual_stats();
    CHECK(stats.second == 6);
    CHECK(std::fabs(stats.first) < 1e-18);

    /* A prior of full rank already answers, without a single row folded in. */
    std::array<double, 36> information_sqrt{};
    for (std::size_t i = 0; i < 6; ++i) {
        information_sqrt.at((i * 6) + i) = 2.0;
    }
    const spacelab::result<spacelab::srif> seeded = spacelab::srif::open(target, information_sqrt);
    CHECK(static_cast<bool>(seeded));
    if (seeded) {
        CHECK_NEAR(seeded->solve().at(0), target.at(0), 1e-12);
        CHECK_NEAR(seeded->covariance().at(0), 0.25, 1e-15);
    }
}

/*
 * The three added families, and the one property that ties them to an already-verified generator:
 * a circular lattice (No, Nso, Nc) = (P, S, F) is a Walker Delta i: T/P/F, satellite for
 * satellite. Draim's geometry is checked against Kepler's third law rather than against itself.
 */
void the_new_families_generate_what_they_promise() {
    spacelab::lattice_params l = spacelab::default_lattice_params();
    CHECK(l.mu == spacelab::mu_earth);
    l.orbits = 4;
    l.sats_per_orbit = 6;
    l.config = 3;
    l.semi_major_axis_m = spacelab::re_earth + 20000e3;
    l.inclination_rad = 56.0 * M_PI / 180.0;

    spacelab::walker_params w = spacelab::default_walker_params();
    w.inclination_rad = l.inclination_rad;
    w.planes = l.orbits;
    w.per_plane = l.sats_per_orbit;
    w.phasing = l.config;
    w.semi_major_axis_m = l.semi_major_axis_m;

    const std::vector<spacelab::state> lattice = spacelab::lattice_flower(l);
    const std::vector<spacelab::state> walker = spacelab::walker(w);
    CHECK(lattice.size() == 24);
    CHECK(lattice.size() == walker.size());
    for (std::size_t i = 0; i < lattice.size(); ++i) {
        CHECK_NEAR(lattice[i].position.x, walker[i].position.x, 1e-6);
        CHECK_NEAR(lattice[i].position.y, walker[i].position.y, 1e-6);
        CHECK_NEAR(lattice[i].position.z, walker[i].position.z, 1e-6);
    }

    spacelab::flower_params f = spacelab::default_flower_params();
    f.petals = 3;
    f.days = 1;
    f.satellites = 6;
    f.eccentricity = 0.2;
    f.inclination_rad = 63.4 * M_PI / 180.0;
    const std::vector<spacelab::state> petals = spacelab::flower(f);
    CHECK(petals.size() == 6);

    const spacelab::draim_params d = spacelab::default_draim_params();
    CHECK(d.period_s == 172800.0);
    const spacelab::draim_orbit orbit = spacelab::draim_geometry(d);
    const double n = 2.0 * M_PI / d.period_s;
    const double want_a = std::cbrt(spacelab::mu_earth / (n * n));
    CHECK_NEAR(orbit.semi_major_axis_m, want_a, want_a * 1e-12);
    CHECK_NEAR(orbit.perigee_radius_m, want_a * (1.0 - d.eccentricity), 1e-6);
    CHECK_NEAR(orbit.apogee_radius_m, want_a * (1.0 + d.eccentricity), 1e-6);
    CHECK(spacelab::draim(d).size() == 4);

    /* The search answers with a design, so the design goes straight back into the generator, and
     * the constellation it builds is measured against the footprint that was asked for. */
    const double footprint = 0.5;
    const spacelab::walker_params best =
        spacelab::minimum_walker(footprint, 60.0 * M_PI / 180.0, spacelab::re_earth + 1200e3,
                                 spacelab::mu_earth, 1, 40, 120, 12);
    const std::vector<spacelab::state> cheapest = spacelab::walker(best);
    CHECK(cheapest.size() == static_cast<std::size_t>(best.planes) * best.per_plane);
    CHECK(spacelab::alpha_characteristic(cheapest, spacelab::mu_earth, 1, 120, 12) <= footprint);
}

/*
 * Coverage, through the wrapper's three named sensor constructors -- which is the point of them:
 * the C caller fills in an `sl_sensor` and can set a field the tag does not name, and a C++ caller
 * cannot reach that state at all. The metrics are then checked against the windows behind them.
 */
void the_coverage_wrapper_agrees_with_the_windows_it_summarises() {
    const std::array<double, 2> lats = {0.0, 0.4};
    const std::array<double, 2> lons = {0.0, 1.5};
    const std::vector<spacelab::state> cells = spacelab::grid_cells(lats, lons);
    CHECK(cells.size() == 4);
    for (const spacelab::state &c : cells) {
        CHECK_NEAR(radius_of(c.position), spacelab::re_earth, 1e-6);
    }

    const spacelab::sensor mask = spacelab::elevation_mask(10.0 * M_PI / 180.0);
    CHECK(mask.kind == SL_SENSOR_ELEVATION_MASK);

    // The same mask stated the way a station states it. `elevation_mask_apparent` has to land below
    // the geometric constructor given the same number, because the atmosphere lifts the satellite.
    const spacelab::sensor apparent_mask = spacelab::elevation_mask_apparent(10.0 * M_PI / 180.0);
    CHECK(apparent_mask.kind == SL_SENSOR_ELEVATION_MASK);
    CHECK(apparent_mask.min_elevation_rad < mask.min_elevation_rad);
    CHECK_NEAR(apparent_mask.min_elevation_rad, spacelab::refraction_geometric(10.0 * M_PI / 180.0),
               1e-15);
    CHECK(spacelab::refraction_apparent(0.0) > 0.0);

    // Diurnal aberration through the throwing wrappers. The station is on the equator at longitude
    // zero, so its velocity is due east at 465.1 m/s and a source at the zenith shifts 0.320" into
    // the motion; `aberration_remove` is the exact inverse, not a sign flip.
    const spacelab::vec3 station{.x = spacelab::re_earth, .y = 0.0, .z = 0.0};
    const spacelab::vec3 spin{.x = 0.0, .y = 0.0, .z = SL_EARTH_ROTATION_RATE};
    const spacelab::vec3 carried = spacelab::diurnal_velocity(station, spin);
    CHECK_NEAR(carried.y, 465.101, 1e-3);
    CHECK_NEAR(carried.x, 0.0, 1e-12);

    const spacelab::vec3 zenith{.x = 1.0, .y = 0.0, .z = 0.0};
    const spacelab::vec3 aberrated = spacelab::aberration_apply(carried, zenith);
    CHECK_NEAR(std::atan2(aberrated.y, aberrated.x) * 180.0 * 3600.0 / M_PI, 0.320, 1e-3);
    const spacelab::vec3 back = spacelab::aberration_remove(carried, aberrated);
    CHECK_NEAR(back.x, 1.0, 1e-15);
    CHECK_NEAR(back.y, 0.0, 1e-15);
    const spacelab::ground_track orbit{.semi_major_axis_m = spacelab::re_earth + 700e3,
                                       .inclination_rad = 51.6 * M_PI / 180.0};
    const double horizon = 86400.0;

    const std::vector<spacelab::coverage_metrics> metrics =
        spacelab::coverage(orbit, mask, cells, horizon);
    CHECK(metrics.size() == cells.size());

    bool seen = false;
    for (std::size_t k = 0; k < cells.size(); ++k) {
        const std::vector<spacelab::window> windows =
            spacelab::access_windows(orbit, mask, cells[k], horizon);
        CHECK(windows.size() == metrics[k].passes);
        double covered = 0.0;
        for (const spacelab::window &v : windows) {
            CHECK(v.rise_s < v.set_s);
            covered += v.set_s - v.rise_s;
        }
        CHECK_NEAR(metrics[k].coverage_fraction, covered / horizon, 1e-9);
        seen = seen || !windows.empty();
    }
    CHECK(seen);

    const spacelab::constellation_fom fom = spacelab::figures_of_merit(metrics);
    CHECK(fom.point_count == metrics.size());
    CHECK(fom.min_coverage <= fom.mean_coverage);

    /* A SAR whose near edge is past its far one is a domain error the wrapper throws, not a pair
     * the library silently swaps. */
    bool threw = false;
    try {
        (void)spacelab::access_windows(orbit, spacelab::side_looking_sar(0.8, 0.4), cells[0],
                                       horizon);
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_DOMAIN);
    }
    CHECK(threw);

    /* And a weight array that is not one weight per cell is refused here rather than read past. */
    const std::array<double, 2> too_few = {1.0, 1.0};
    const spacelab::result<spacelab::constellation_fom> mismatched =
        spacelab::try_figures_of_merit(metrics, too_few);
    CHECK(!mismatched);
    CHECK(mismatched.error() == SL_ERR_INVALID_ARGUMENT);
}

/*
 * A minimal well-formed RINEX 3 observation file, the same one the C smoke test reads. Two GPS
 * satellites in the first epoch is the case a lookup on either half of the identifier alone cannot
 * tell apart, and G07's second record carries the receiver's own loss-of-lock bit.
 */
constexpr std::string_view rinex_observations =
    "     3.05           OBSERVATION DATA    M: MIXED            RINEX VERSION / TYPE\n"
    "G    2 C1C L1C                                              SYS / # / OBS TYPES\n"
    "R    1 C1C                                                  SYS / # / OBS TYPES\n"
    "  4460894.9000  2682361.5000 -3674748.5000                  APPROX POSITION XYZ\n"
    "    30.000                                                  INTERVAL\n"
    "END                                                         END OF HEADER\n"
    "> 2024 01 09 12 00  0.0000000  0  3\n"
    "G07  23629347.915 6 124144623.325 7\n"
    "R05  21000000.000\n"
    "G09  23626000.000 5 124130000.000 6\n"
    "> 2024 01 09 12 00 30.0000000  0  1\n"
    "G07  23629400.00016 124144900.000 7\n";

/*
 * GNSS through the wrapper, where the RINEX length comes from the string rather than from a
 * `strlen` the caller writes, and a failure is an exception rather than a status.
 */
void the_gnss_wrapper_reads_a_file_and_refuses_a_geometry() {
    /* GPT3 needs no file at all, and the grid reader refuses text that is not a grid. */
    const spacelab::gpt3_weather weather = spacelab::gpt3(0.84, 0.29, 329.0, 180.0);
    CHECK(weather.pressure > 80e3 && weather.pressure < 110e3);
    CHECK(weather.temperature > 220.0 && weather.temperature < 320.0);
    CHECK(!spacelab::try_vmf_grid_coefficients("not a grid", 0.84, 0.29));

    const std::string header = "     3.04           RINEX VERSION / TYPE\n";
    CHECK(spacelab::rinex_gnss_positions(header, 0.0).empty());
    CHECK(spacelab::rinex_glonass_positions(header, 0.0).empty());

    /* An observation file, where the wrapper's `string_view` carries the length and the descriptor
     * is copied to the NUL-terminated token the C function takes. */
    const spacelab::obs_header meta = spacelab::rinex_obs_header(rinex_observations);
    CHECK_NEAR(meta.version, 3.05, 1e-12);
    CHECK(meta.file_system == 'M');
    CHECK(meta.time_system == SL_TIME_SYSTEM_GPS);
    CHECK_NEAR(meta.interval, 30.0, 1e-12);
    CHECK(std::isnan(meta.antenna_delta_hen[1]));
    CHECK(spacelab::rinex_obs_descriptors(rinex_observations, 'G').size() == 2);
    CHECK(spacelab::rinex_obs_descriptors(rinex_observations, 'E').empty());

    const std::vector<spacelab::obs_sample> series =
        spacelab::rinex_obs_series(rinex_observations, 'G', 7, "C1C");
    CHECK(series.size() == 2);
    CHECK_NEAR(series[0].value, 23629347.915, 1e-6);
    CHECK(series[0].lost_lock == 0 && series[1].lost_lock == 1);
    /* G09 sits behind G07 in the same epoch, so a match that stopped at the constellation letter
     * would report G07's range under G09's name. */
    CHECK_NEAR(spacelab::rinex_obs_series(rinex_observations, 'G', 9, "C1C").at(0).value,
               23626000.0, 1e-6);
    CHECK(!spacelab::try_rinex_obs_series(rinex_observations, 'R', 5, "L1C"));

    const double l1 = 1575.42e6;
    CHECK_NEAR(spacelab::ionospheric_path_delay(20.0, l1), 40.3e16 * 20.0 / (l1 * l1), 1e-12);
    CHECK_NEAR(spacelab::glonass_l1_frequency(-7), (1602.0 - (7 * 0.5625)) * 1e6, 1e-6);
    CHECK_NEAR(spacelab::ionospheric_mapping_factor(M_PI / 2.0), 1.0, 1e-12);

    std::vector<spacelab::vec3> satellites;
    for (std::size_t i = 0; i < 5; ++i) {
        const double theta = 0.25 * static_cast<double>(i);
        const double phi = 1.3 * static_cast<double>(i);
        const double a = spacelab::re_earth + 20200e3;
        satellites.push_back({.x = a * std::cos(theta),
                              .y = a * std::sin(theta) * std::cos(phi),
                              .z = a * std::sin(theta) * std::sin(phi)});
    }
    const spacelab::vec3 receiver{.x = spacelab::re_earth, .y = 0.0, .z = 0.0};
    const spacelab::dop d = spacelab::dilution_of_precision(receiver, satellites);
    CHECK_NEAR(d.gdop * d.gdop, (d.pdop * d.pdop) + (d.tdop * d.tdop), 1e-9);

    bool threw = false;
    try {
        (void)spacelab::dilution_of_precision(receiver,
                                              std::span<const spacelab::vec3>(satellites).first(3));
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_DOMAIN);
    }
    CHECK(threw);

    /* The mapping functions, and the domain error a ray along the horizon is. */
    const spacelab::mapping niell = spacelab::niell_mapping(0.7, 200.0, 90.0, M_PI / 2.0);
    CHECK_NEAR(niell.hydrostatic, 1.0, 1e-2);
    CHECK_NEAR(spacelab::gmf_mapping(0.7, 0.2, 200.0, 90.0, M_PI / 2.0).wet, 1.0, 1e-2);
    CHECK(!spacelab::try_vmf3_mapping(1.2e-3, 6e-4, 0.7, 0.2, 200.0, 90.0, 0.0));

    /* NeQuick is the slow one, so it is called once, and bounded rather than pinned. */
    const spacelab::geodetic station{.latitude_rad = 0.7, .longitude_rad = 0.2, .height_m = 100.0};
    const spacelab::geodetic satellite{
        .latitude_rad = 0.9, .longitude_rad = 0.6, .height_m = 20200e3};
    const std::array<double, 3> alpha{236.831641, -0.39362878, 0.00402826613};
    const double stec = spacelab::nequick_galileo_stec(alpha, 4, 43200.0, station, satellite);
    CHECK(stec > 1.0 && stec < 500.0);
    CHECK(!spacelab::try_nequick_galileo_stec(alpha, 13, 43200.0, station, satellite));
}

/*
 * The atmospheres, tides and density half through the C++ face: the fixed-length out-parameters
 * come back as `std::array`, the two-call ones as `std::vector`, and the mean-pole tag has a
 * default so a caller who does not know the 2018 amendment gets the current pole rather than the
 * retired one.
 */
void the_atmosphere_wrapper_answers_what_the_formulas_say() {
    CHECK(spacelab::nrlmsise00_density(172, 29000.0, 200.0, 0.0, 0.0, 150.0, 150.0, 4.0) >
          spacelab::nrlmsise00_density(172, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 4.0));

    /* SDTM-1 through the span overload: a constant index history sits on the cooling state's steady
     * state, below the index, the same at the default 3-hourly cadence and at an hourly one. */
    const std::array<double, 16> quiet_ap{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                          5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
    const double quiet_storm = spacelab::sdtm1_storm_response(quiet_ap);
    CHECK(quiet_storm > 0.0 && quiet_storm < 5.0);
    CHECK(std::abs(spacelab::sdtm1_storm_response(quiet_ap, 3600.0) - quiet_storm) <
          1e-12 * quiet_storm);
    const double sdtm_400 =
        spacelab::sdtm1_density(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0);
    CHECK(spacelab::sdtm1_density(172.0, 29000.0, 200.0, 0.0, 0.0, 150.0, 150.0, 5.0) > sdtm_400);
    CHECK(spacelab::sdtm1_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0,
                                                 5.0) > 586.0);
    const std::array<double, SL_SDTM1_INTERVAL> band =
        spacelab::sdtm1_density_interval(sdtm_400, 150.0, 150.0, 5.0, 1.6448536269514727);
    CHECK(band[0] < sdtm_400);
    CHECK(band[1] > sdtm_400);
    const std::array<double, SL_SDTM1_ENVELOPE> envelope = spacelab::sdtm1_envelope();
    CHECK(envelope[0] < 400.0 && 400.0 < envelope[1]);
    CHECK(envelope[2] < 150.0 && 150.0 < envelope[3]);

    /* The solar-wind field through its four spans: hours without the solar wind drive by their
     * ap60, a strong southward field drives the same hours higher, and spans of different lengths
     * are no history and answer NaN. */
    std::array<double, 16> sw_speed{};
    std::array<double, 16> sw_by{};
    std::array<double, 16> sw_bz{};
    sw_speed.fill(std::numeric_limits<double>::quiet_NaN());
    sw_by.fill(std::numeric_limits<double>::quiet_NaN());
    sw_bz.fill(std::numeric_limits<double>::quiet_NaN());
    const double sw_quiet =
        spacelab::sdtm1_solar_wind_storm_response(quiet_ap, sw_speed, sw_by, sw_bz);
    CHECK(sw_quiet > 0.0 && sw_quiet < 5.0);
    for (std::size_t i = 10; i < sw_speed.size(); ++i) {
        sw_speed[i] = 700.0;
        sw_by[i] = 0.0;
        sw_bz[i] = -20.0;
    }
    CHECK(spacelab::sdtm1_solar_wind_storm_response(quiet_ap, sw_speed, sw_by, sw_bz) > sw_quiet);
    CHECK(std::isnan(spacelab::sdtm1_solar_wind_storm_response(
        quiet_ap, std::span<const double>(sw_speed).first(15), sw_by, sw_bz)));
    const double sw_400 =
        spacelab::sdtm1_solar_wind_density(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0);
    CHECK(spacelab::sdtm1_solar_wind_density(172.0, 29000.0, 200.0, 0.0, 0.0, 150.0, 150.0, 5.0) >
          sw_400);
    CHECK(spacelab::sdtm1_solar_wind_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0,
                                                            150.0, 5.0) > 0.0);
    const std::array<double, SL_SDTM1_INTERVAL> sw_band =
        spacelab::sdtm1_solar_wind_density_interval(sw_400, 150.0, 150.0, 5.0, 1.6448536269514727);
    CHECK(sw_band[0] < sw_400 && sw_400 < sw_band[1]);
    const std::array<double, SL_SDTM1_ENVELOPE> sw_envelope = spacelab::sdtm1_solar_wind_envelope();
    CHECK(sw_envelope[0] < 400.0 && 400.0 < sw_envelope[1]);

    const std::array<double, 5> delaunay{1.1, 2.2, 3.3, 4.4, 5.5};
    const std::array<double, 5> once = spacelab::solid_tide_step2(0.7, delaunay);
    const std::array<double, 5> turned =
        spacelab::solid_tide_step2(0.7 + (2.0 * std::numbers::pi), delaunay);
    for (std::size_t i = 0; i < once.size(); ++i) {
        CHECK_NEAR(once[i], turned[i], 1e-20);
    }

    const double t = 20.0 * 365.25 * 86400.0;
    const std::array<double, 2> now = spacelab::solid_pole_tide(0.0, 1e-7, t);
    const std::array<double, 2> retired =
        spacelab::solid_pole_tide(0.0, 1e-7, t, SL_MEAN_POLE_IERS2010);
    CHECK(std::abs(retired[0] - now[0]) > 0.1 * std::abs(now[0]));
    CHECK(spacelab::ocean_pole_tide(0.0, 1e-7, t)[0] != 0.0);

    const spacelab::zonal mean = spacelab::conventional_zonal_coefficients(25.0);
    const spacelab::zonal secular = spacelab::time_variable_zonal_correction(25.0);
    CHECK_NEAR(mean.c20 - secular.c20, -0.48416948e-3, 1e-15);
    CHECK_NEAR(secular.c20, 11.6e-12 * 25.0, 1e-24);

    const double ratio = spacelab::molecular_speed_ratio(7600.0, 1000.0, 0.016);
    CHECK_NEAR(ratio, 7.4549, 1e-3);
    const spacelab::drag_panel panel =
        spacelab::sentman_drag_coefficients(ratio, 1.0, 1.0, 300.0, 1000.0);
    CHECK_NEAR(panel.shear, 0.0, 1e-12);
    CHECK_NEAR(panel.drag, 2.14822, 1e-4);
    CHECK(!spacelab::try_sentman_drag_coefficients(ratio, 1.0, 1.0, 0.0, 1000.0));

    const std::array<double, SL_DENSITY_DRIVERS> drivers =
        spacelab::space_weather_drivers(150.0, 130.0, 9.0);
    CHECK_NEAR(drivers[0], 140.0, 1e-12);
    CHECK_NEAR(drivers[3], 3.0, 1e-12);

    /* A header with no data rows sizes to nothing rather than failing, and a day the file does not
     * hold is a status the caller branches on. */
    const std::string_view csv = "DATE,BSRN,ND,KP1,KP2\n";
    CHECK(spacelab::space_weather_days(csv).empty());
    CHECK(!spacelab::try_space_weather_ap_array(csv, 2020, 1, 1, 0.0));
    CHECK(spacelab::density_data_samples(csv).empty());

    const std::array<spacelab::density_sample, 2> samples{
        spacelab::density_sample{
            .latitude_rad = -1.5, .local_solar_time_h = 1.0, .log_ratio = 0.25, .weight = 1.0},
        spacelab::density_sample{
            .latitude_rad = 1.5, .local_solar_time_h = 23.0, .log_ratio = -0.5, .weight = 1.0}};
    const std::vector<double> coefficients = spacelab::density_correction_fit(samples, 1e-6);
    CHECK(coefficients.size() == SL_DENSITY_CORRECTION_COEFFICIENTS);
    CHECK(spacelab::density_correction_factor(coefficients, -1.5, 1.0) > 0.0);
    CHECK(!spacelab::try_density_correction_factor(std::span<const double>(coefficients).first(4),
                                                   -1.5, 1.0));

    const std::vector<double> grid = spacelab::density_grid_snapshot(samples, 2, 2);
    CHECK(grid.size() == 4);
    CHECK_NEAR(grid[0], 0.25, 1e-15);
    CHECK_NEAR(grid[3], -0.5, 1e-15);

    const spacelab::swath cut = spacelab::swath_geometry(spacelab::re_earth, 700e3, 0.35);
    CHECK_NEAR(cut.earth_central_angle_rad + 0.35 + (std::numbers::pi / 2.0) + cut.elevation_rad,
               std::numbers::pi, 1e-12);
    CHECK(!spacelab::try_swath_geometry(spacelab::re_earth, 700e3, 1.4));

    /* The analytical propagator's round trip through mean elements is the identity at the epoch. */
    const double speed = std::sqrt(spacelab::mu_earth / 7.0e6);
    const double inclination = 51.6 * std::numbers::pi / 180.0;
    const spacelab::brouwer_lyddane_params bl{
        .initial = {.position = {.x = 7.0e6, .y = 0.0, .z = 0.0},
                    .velocity = {.x = 0.0,
                                 .y = speed * std::cos(inclination),
                                 .z = speed * std::sin(inclination)}},
        .mu = spacelab::mu_earth,
        .earth_radius_m = spacelab::re_earth,
        .c20 = -1.08262668e-3,
        .c30 = 2.53265649e-6,
        .c40 = 1.61962159e-6,
        .c50 = 2.27296083e-7,
        .m2 = 0.0};
    const std::array<double, 2> times{0.0, 86400.0};
    const std::vector<spacelab::state> arc = spacelab::brouwer_lyddane_propagate(bl, times);
    CHECK(arc.size() == 2);
    CHECK_NEAR(arc[0].position.x, bl.initial.position.x, 10.0);
    CHECK_NEAR(arc[0].position.y, bl.initial.position.y, 10.0);
    CHECK(arc[1].position.x != arc[0].position.x);
}

/*
 * The rest of the force models through the C++ face: each factory returns a `result<ephemeris>`
 * whose handle closes itself, and the failing one returns a status rather than a half-built object.
 */
void every_force_model_is_a_factory_that_owns_its_handle() {
    const double speed = std::sqrt(spacelab::mu_earth / 7.0e6);
    const double inclination = 51.6 * std::numbers::pi / 180.0;
    const double arc = 600.0;
    const double j2 = 1.08262668e-3;
    const spacelab::state s0{.position = {.x = 7.0e6, .y = 0.0, .z = 0.0},
                             .velocity = {.x = 0.0,
                                          .y = speed * std::cos(inclination),
                                          .z = speed * std::sin(inclination)}};
    const spacelab::eop orientation{.epoch_tt_ns = 725760069184000000LL,
                                    .epoch_ut1_ns = 725760000000000000LL,
                                    .xp_rad = 1.0e-6,
                                    .yp_rad = 2.0e-6,
                                    .dx_rad = 0.0,
                                    .dy_rad = 0.0};

    const auto radius = [](const spacelab::state &s) {
        return std::sqrt((s.position.x * s.position.x) + (s.position.y * s.position.y) +
                         (s.position.z * s.position.z));
    };

    const spacelab::result<spacelab::ephemeris> dense =
        spacelab::ephemeris::j2_dense(s0, arc, spacelab::mu_earth, j2);
    CHECK(static_cast<bool>(dense));
    CHECK(std::abs(radius(dense->at(arc)) - 7.0e6) < 50e3);

    const spacelab::exponential_atmosphere atmosphere{.ballistic_coefficient = 100.0,
                                                      .reference_density = 1e-12,
                                                      .reference_altitude_m = 500e3,
                                                      .scale_height_m = 60e3,
                                                      .earth_radius_m = spacelab::re_earth};
    CHECK(static_cast<bool>(spacelab::ephemeris::drag(s0, arc, 4, atmosphere)));

    const spacelab::nrlmsise_drag thermosphere{.doy = 172,
                                               .seconds_of_day = 29000.0,
                                               .ballistic_coefficient = 100.0,
                                               .f107 = 150.0,
                                               .f107a = 150.0,
                                               .ap = 4.0,
                                               .mu = spacelab::mu_earth};
    CHECK(static_cast<bool>(
        spacelab::ephemeris::drag_nrlmsise00(s0, arc, orientation, thermosphere, 1e-10)));

    const spacelab::sdtm1_drag sdtm1{.doy = 172.0,
                                     .seconds_of_day = 29000.0,
                                     .ballistic_coefficient = 0.02,
                                     .f107 = 128.0,
                                     .f107a = 141.0,
                                     .mu = spacelab::mu_earth};
    std::array<double, 120> quiet_ap{};
    quiet_ap.fill(5.0);
    CHECK(static_cast<bool>(
        spacelab::ephemeris::drag_sdtm1(s0, arc, orientation, sdtm1, quiet_ap, 10800.0, 1e-10)));
    std::array<double, 48> hours_ap60{};
    std::array<double, 48> hours_speed{};
    std::array<double, 48> hours_by{};
    std::array<double, 48> hours_bz{};
    hours_ap60.fill(5.0);
    hours_speed.fill(400.0);
    hours_by.fill(2.0);
    hours_bz.fill(-3.0);
    CHECK(static_cast<bool>(spacelab::ephemeris::drag_sdtm1_solar_wind(
        s0, arc, orientation, sdtm1, hours_ap60, hours_speed, hours_by, hours_bz, 1e-10)));
    /* Spans of different lengths are refused before the library is asked. */
    CHECK(!spacelab::ephemeris::drag_sdtm1_solar_wind(
        s0, arc, orientation, sdtm1, hours_ap60, std::span<const double>(hours_speed).first(47),
        hours_by, hours_bz));

    CHECK(static_cast<bool>(spacelab::ephemeris::geopotential_itrf(s0, arc, 4, orientation)));
    CHECK(static_cast<bool>(spacelab::ephemeris::geopotential_tide(
        s0, arc, 4, orientation, 0.0, 1.32712440018e20, 4.902800066e12)));
    CHECK(static_cast<bool>(spacelab::ephemeris::j2_itrf(s0, arc, spacelab::mu_earth, j2,
                                                         spacelab::re_earth, orientation)));
    CHECK(static_cast<bool>(spacelab::ephemeris::luni_solar(s0, arc, 4, 0.0, 1e-7)));
    CHECK(static_cast<bool>(
        spacelab::ephemeris::relativistic(s0, arc, 0.0, spacelab::mu_earth, 1.32712440018e20)));

    const std::array<spacelab::eop_sample, 2> series{
        spacelab::eop_sample{.t_seconds = -arc,
                             .xp_rad = orientation.xp_rad,
                             .yp_rad = orientation.yp_rad,
                             .dut1_seconds = -0.069184,
                             .dx_rad = 0.0,
                             .dy_rad = 0.0},
        spacelab::eop_sample{.t_seconds = 2.0 * arc,
                             .xp_rad = orientation.xp_rad,
                             .yp_rad = orientation.yp_rad,
                             .dut1_seconds = -0.069184,
                             .dx_rad = 0.0,
                             .dy_rad = 0.0}};
    const spacelab::result<spacelab::ephemeris> interpolated = spacelab::ephemeris::j2_itrf_series(
        s0, arc, spacelab::mu_earth, j2, spacelab::re_earth, orientation.epoch_tt_ns, 0.0, series);
    CHECK(static_cast<bool>(interpolated));

    /* One row cannot be interpolated, and the failure is a status rather than a handle. */
    CHECK(!spacelab::ephemeris::j2_itrf_series(
        s0, arc, spacelab::mu_earth, j2, spacelab::re_earth, orientation.epoch_tt_ns, 0.0,
        std::span<const spacelab::eop_sample>(series).first(1)));

    const spacelab::result<spacelab::state> ordinary = spacelab::try_propagate_j2_precision(
        s0, spacelab::mu_earth, j2, spacelab::re_earth, 1.0, 600, SL_PRECISION_DOUBLE);
    const spacelab::result<spacelab::state> extended = spacelab::try_propagate_j2_precision(
        s0, spacelab::mu_earth, j2, spacelab::re_earth, 1.0, 600, SL_PRECISION_DOUBLE_DOUBLE);
    CHECK(static_cast<bool>(ordinary));
    CHECK(static_cast<bool>(extended));
    CHECK(std::abs(ordinary->position.x - extended->position.x) < 1e-3);
}

/*
 * The radar surface through the wrapper: every variable-length answer sizes itself, the spans
 * carry their own lengths, and a pair whose two images disagree in length is refused before the
 * C ABI is ever reached.
 */
void the_radar_wrapper_sizes_its_own_answers_and_finds_the_scatterer() {
    const std::vector<spacelab::complex> chirp = spacelab::lfm_chirp(32, 5.0e11, 3.0e7);
    CHECK(chirp.size() == 32);
    for (const spacelab::complex &c : chirp) {
        CHECK_NEAR(std::hypot(c.re, c.im), 1.0, 1e-12);
    }

    const std::vector<spacelab::complex> compressed = spacelab::matched_filter(chirp, chirp);
    double peak = 0.0;
    for (const spacelab::complex &c : compressed) {
        peak = std::max(peak, std::hypot(c.re, c.im));
    }
    CHECK_NEAR(peak, 32.0, 1e-6 * 32.0);

    /* Two images of different lengths are not a pair, and the wrapper says so itself rather than
     * handing the C ABI a length it would have to guess at. */
    const std::vector<spacelab::complex> shorter(chirp.begin(), chirp.begin() + 8);
    CHECK(!spacelab::try_interferogram(chirp, shorter));

    /* A ramp wraps and unwraps back to itself, and all three unwrappers agree on a raster that
     * has no residues to disagree about. */
    constexpr std::size_t rows = 8;
    constexpr std::size_t cols = 8;
    std::vector<double> truth(rows * cols);
    std::vector<double> wrapped(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double phase = (0.30 * static_cast<double>(r)) + (0.22 * static_cast<double>(c));
            truth[(r * cols) + c] = phase;
            wrapped[(r * cols) + c] = spacelab::wrap_phase(phase);
        }
    }
    const std::vector<std::int8_t> charges = spacelab::residues(wrapped, rows, cols);
    CHECK(charges.size() == (rows - 1) * (cols - 1));
    CHECK(std::ranges::all_of(charges, [](std::int8_t q) { return q == 0; }));

    for (const std::vector<double> &recovered :
         {spacelab::unwrap_phase_2d(wrapped, rows, cols),
          spacelab::unwrap_phase_2d_branch_cut(wrapped, rows, cols),
          spacelab::unwrap_phase_2d_lstsq(wrapped, rows, cols)}) {
        CHECK(recovered.size() == rows * cols);
        const double offset = recovered[0] - truth[0];
        for (std::size_t k = 0; k < recovered.size(); ++k) {
            CHECK_NEAR(recovered[k] - truth[k], offset, 1e-9);
        }
    }

    /* One scatterer, injected at a known elevation, and both spectra have to find it there. */
    const spacelab::tomo_params params{.wavelength = 0.031,
                                       .slant_range = 700000.0,
                                       .loading = 0.1,
                                       .lambda = 0.05,
                                       .max_iter = 600,
                                       .threshold_rel = 0.5};
    const std::vector<double> baselines{-160.0, -110.0, -70.0, -30.0, 20.0, 65.0, 120.0, 175.0};
    std::vector<double> elevations;
    for (int k = -20; k <= 20; ++k) {
        elevations.push_back(static_cast<double>(k) * 2.0);
    }
    const double truth_elevation = 10.0;
    std::vector<spacelab::complex> steering;
    for (const double b : baselines) {
        const double phase = -4.0 * std::numbers::pi * b * truth_elevation /
                             (params.wavelength * params.slant_range);
        steering.push_back(spacelab::complex{.re = std::cos(phase), .im = std::sin(phase)});
    }
    std::vector<spacelab::complex> covariance(steering.size() * steering.size());
    for (std::size_t i = 0; i < steering.size(); ++i) {
        for (std::size_t j = 0; j < steering.size(); ++j) {
            covariance[(i * steering.size()) + j] = spacelab::complex{
                .re = (steering[i].re * steering[j].re) + (steering[i].im * steering[j].im),
                .im = (steering[i].im * steering[j].re) - (steering[i].re * steering[j].im)};
        }
    }

    for (const std::vector<double> &power :
         {spacelab::capon_spectrum(covariance, baselines, elevations, params),
          spacelab::bartlett_spectrum(covariance, baselines, elevations, params)}) {
        CHECK(power.size() == elevations.size());
        const auto best = static_cast<std::size_t>(
            std::ranges::distance(power.begin(), std::ranges::max_element(power)));
        CHECK_NEAR(elevations[best], truth_elevation, 1e-9);
    }

    const std::vector<spacelab::scatterer> found =
        spacelab::sl1mmer(steering, baselines, elevations, params);
    CHECK(found.size() == 1);
    if (found.size() == 1) {
        CHECK_NEAR(found[0].elevation_m, truth_elevation, 1e-9);
    }

    /* Polarimetry: a depolarised target is the maximum-entropy one, and the Pauli powers of a
     * pure surface return sit entirely in the first channel. */
    std::vector<spacelab::complex> t3(SL_COHERENCY_ENTRIES,
                                      spacelab::complex{.re = 0.0, .im = 0.0});
    for (std::size_t i = 0; i < 3; ++i) {
        t3[(i * 3) + i] = spacelab::complex{.re = 1.0 / 3.0, .im = 0.0};
    }
    const spacelab::cloude_pottier decomposition = spacelab::cloude_pottier_decompose(t3);
    CHECK_NEAR(decomposition.entropy, 1.0, 1e-12);
    CHECK_NEAR(decomposition.anisotropy, 0.0, 1e-12);
    /* A t3 of the wrong size is not a coherency matrix, and the wrapper refuses it itself. */
    CHECK(!spacelab::try_cloude_pottier_decompose(std::span(t3).first(4)));

    const spacelab::polarimetric_sample surface{.s_hh = {.re = 1.0, .im = 0.0},
                                                .s_hv = {.re = 0.0, .im = 0.0},
                                                .s_vv = {.re = 1.0, .im = 0.0}};
    const std::array<double, 3> powers = spacelab::pauli_powers(surface);
    CHECK_NEAR(powers[0], 2.0, 1e-12);
    CHECK_NEAR(powers[1], 0.0, 1e-12);
    CHECK_NEAR(powers[2], 0.0, 1e-12);

    /* A profile with no main lobe is a status, and here it is an exception rather than a NaN. */
    const std::vector<double> flat(16, 1.0);
    CHECK(!spacelab::try_resolution_3db_samples(flat));
}

/*
 * The optical surface through the wrapper: spans carry the lengths the C ABI would otherwise take
 * separately, the ragged corner lists stay explicit because C has no other shape for them, and a
 * mismatched pair is refused before the boundary is reached.
 */
void the_optical_wrapper_triangulates_detects_and_suppresses() {
    constexpr double focal = 200.0;
    const std::array<double, 9> identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const std::array<double, 3> world{1.2, -0.7, 14.0};
    std::vector<spacelab::camera> cameras;
    std::vector<double> pixels;
    for (const double baseline : {0.0, 3.0}) {
        spacelab::camera c{};
        std::ranges::copy(identity, std::begin(c.rotation));
        c.center[0] = baseline;
        c.center[1] = 0.0;
        c.center[2] = 0.0;
        c.focal = focal;
        c.principal[0] = 50.0;
        c.principal[1] = 40.0;
        cameras.push_back(c);
        pixels.push_back((focal * (world[0] - baseline) / world[2]) + 50.0);
        pixels.push_back((focal * world[1] / world[2]) + 40.0);
    }

    const spacelab::vec3 point = spacelab::triangulate(cameras, pixels);
    CHECK_NEAR(point.x, world[0], 1e-9);
    CHECK_NEAR(point.y, world[1], 1e-9);
    CHECK_NEAR(point.z, world[2], 1e-9);
    /* One pixel per camera is what "a view" means; anything else the wrapper refuses itself. */
    CHECK(!spacelab::try_triangulate(cameras, std::span(pixels).first(3)));

    /* One corner per view and one match between them is one track, and it is the point above. */
    const std::vector<spacelab::corner> corners{{.x = pixels[0], .y = pixels[1], .response = 1.0},
                                                {.x = pixels[2], .y = pixels[3], .response = 1.0}};
    const std::vector<std::size_t> per_view{1, 1};
    const std::vector<spacelab::feature_match> matches{
        {.view_a = 0, .corner_a = 0, .view_b = 1, .corner_b = 0}};
    const std::vector<spacelab::track_point> tracks =
        spacelab::sfm_triangulate(corners, per_view, matches, cameras);
    CHECK(tracks.size() == 1);
    if (tracks.size() == 1) {
        CHECK(tracks[0].resolved);
        CHECK_NEAR(tracks[0].position.z, world[2], 1e-9);
    }

    /* A drawn corner is found where it was drawn. */
    constexpr std::size_t rows = 24;
    constexpr std::size_t cols = 24;
    std::vector<double> image(rows * cols, 0.0);
    for (std::size_t y = 12; y < rows; ++y) {
        for (std::size_t x = 12; x < cols; ++x) {
            image[(y * cols) + x] = 1.0;
        }
    }
    const spacelab::harris_config harris{
        .k = 0.04, .window_radius = 2, .threshold_rel = 0.3, .nms_radius = 3, .max_corners = 8};
    const std::vector<spacelab::corner> found = spacelab::harris_corners(image, rows, cols, harris);
    CHECK(!found.empty());
    if (!found.empty()) {
        const spacelab::corner &best =
            *std::ranges::max_element(found, {}, &spacelab::corner::response);
        CHECK(std::abs(best.x - 12.0) <= 2.0);
        CHECK(std::abs(best.y - 12.0) <= 2.0);
    }

    /* A rectified pair built by sliding one image sideways has that slide as its disparity. */
    std::vector<double> left(rows * cols);
    std::vector<double> right(rows * cols);
    constexpr std::size_t shift = 3;
    for (std::size_t y = 0; y < rows; ++y) {
        for (std::size_t x = 0; x < cols; ++x) {
            const auto texture = [](std::size_t yy, std::size_t xx) {
                return std::sin(0.9 * static_cast<double>(xx)) +
                       std::cos(0.7 * static_cast<double>(yy));
            };
            left[(y * cols) + x] = texture(y, x);
            right[(y * cols) + x] = texture(y, x + shift);
        }
    }
    const spacelab::sgm_config sgm{.d_max = 8, .radius = 2, .p1 = 1.0, .p2 = 8.0};
    const std::vector<double> disparity = spacelab::semi_global_match(left, right, rows, cols, sgm);
    CHECK(disparity.size() == rows * cols);
    CHECK_NEAR(disparity[(12 * cols) + 16], static_cast<double>(shift), 0.6);
    /* A pair whose two images differ in length is not a pair. */
    CHECK(!spacelab::try_semi_global_match(left, std::span(right).first(8), rows, cols, sgm));

    /* Suppression and the anchor grid, through the spans that carry their own lengths. */
    const std::vector<spacelab::bbox> boxes{{.x1 = 0.0, .y1 = 0.0, .x2 = 2.0, .y2 = 2.0},
                                            {.x1 = 1.0, .y1 = 0.0, .x2 = 3.0, .y2 = 2.0},
                                            {.x1 = 10.0, .y1 = 10.0, .x2 = 12.0, .y2 = 12.0}};
    const std::vector<double> scores{0.9, 0.8, 0.7};
    const std::vector<std::size_t> kept = spacelab::nms(boxes, scores, 0.3);
    CHECK(kept == std::vector<std::size_t>({0, 2}));
    CHECK(!spacelab::try_nms(boxes, std::span(scores).first(2), 0.3));
    CHECK_NEAR(spacelab::box_iou(boxes[0], boxes[1]), 1.0 / 3.0, 1e-15);

    const std::vector<double> scales{4.0, 8.0};
    const std::vector<double> ratios{0.5, 1.0, 2.0};
    const std::vector<spacelab::bbox> anchors =
        spacelab::generate_anchors(2, 2, 16.0, scales, ratios);
    CHECK(anchors.size() == std::size_t{2} * 2 * scales.size() * ratios.size());
    const spacelab::bbox decoded = spacelab::decode_box(anchors[5], {0.0, 0.0, 0.0, 0.0});
    CHECK_NEAR(decoded.x1, anchors[5].x1, 1e-12);
    CHECK_NEAR(decoded.y2, anchors[5].y2, 1e-12);

    /* The atmosphere and the two pan-sharpeners, on the shapes the wrapper derives itself. */
    const spacelab::atmospheric_coupling coupling{.path_reflectance = 0.08,
                                                  .transmittance_down = 0.83,
                                                  .transmittance_up = 0.87,
                                                  .spherical_albedo = 0.11};
    CHECK_NEAR(spacelab::surface_from_toa(coupling, spacelab::toa_from_surface(coupling, 0.42)),
               0.42, 1e-12);

    const std::vector<double> ms{0.20, 0.31, 0.15, 0.35, 0.22, 0.51, 0.11, 0.47, 0.26};
    const std::vector<double> pan{0.55, 0.61, 0.48};
    const std::vector<double> brovey = spacelab::brovey_pan_sharpen(ms, 3, pan);
    CHECK(brovey.size() == ms.size());
    for (std::size_t p = 0; p < pan.size(); ++p) {
        CHECK_NEAR(brovey[p] + brovey[3 + p] + brovey[6 + p], pan[p], 1e-12);
    }
    CHECK(spacelab::gram_schmidt_pan_sharpen(ms, 3, pan).size() == ms.size());
}

/* A head-on conjunction with isotropic covariances, the miss vector across the relative track. */
spacelab::conjunction conjunction_case(double miss) {
    return spacelab::conjunction{
        .relative_position = {.x = 0.6 * miss, .y = 0.0, .z = 0.8 * miss},
        .relative_velocity = {.x = 0.0, .y = 14000.0, .z = 0.0},
        .primary_covariance = {1e4, 0.0, 0.0, 0.0, 1e4, 0.0, 0.0, 0.0, 1e4},
        .secondary_covariance = {1e4, 0.0, 0.0, 0.0, 1e4, 0.0, 0.0, 0.0, 1e4},
        .primary_radius = 5.0,
        .secondary_radius = 5.0};
}

/* A cannonball at `altitude` under an exponential atmosphere pinned at 300 km. */
spacelab::decay_params decay_case() {
    spacelab::decay_params p = spacelab::default_decay_params();
    p.area_over_mass = 0.02;
    p.reference_density = 2e-11;
    p.reference_radius = SL_RE_EARTH + 300e3;
    p.scale_height = 60e3;
    p.horizon = 400.0 * 86400.0;
    p.tolerance = 3600.0;
    return p;
}

/*
 * The radiometric observables through the wrapper, and the `try_`/throwing split that only exists
 * on this side of the boundary.
 */
void the_radiometric_wrapper_splits_its_refusals_from_its_answers() {
    static_assert(sizeof(::sl_band) == sizeof(std::int32_t));
    static_assert(std::is_trivially_copyable_v<spacelab::terminal>);
    static_assert(std::is_same_v<spacelab::band, ::sl_band>);

    const spacelab::state sat{.position = {.x = 4200000.0, .y = 5300000.0, .z = 1600000.0},
                              .velocity = {.x = -5200.0, .y = 3900.0, .z = 900.0}};
    const spacelab::terminal tx{.position = {.x = 4194303.0, .y = 172460.0, .z = 4780100.0},
                                .velocity = {.x = 0.0, .y = 0.0, .z = 0.0},
                                .epoch = 0.0};
    spacelab::terminal rx = tx;
    rx.position = {.x = -3950200.0, .y = 2522400.0, .z = 4311700.0};
    rx.epoch = 0.04;

    CHECK(spacelab::bistatic_range(sat, tx, 0.02, rx) > 0.0);
    /* Swapping the receivers negates the arrival difference, which says the two arguments are not
     * one argument written twice. */
    spacelab::terminal first = tx;
    first.epoch = 0.03;
    const double tdoa = spacelab::tdoa(sat, 0.0, first, rx);
    CHECK_NEAR(spacelab::tdoa(sat, 0.0, rx, first), -tdoa, 1e-15);
    CHECK(std::isfinite(spacelab::fdoa(sat, 0.0, first, rx, 8.4e9)));
    CHECK(std::isfinite(spacelab::delta_dor(sat, 0.0, first, rx, 1.2, -0.4)));

    /* The two halves of every fallible entry point: `try_` reports a refusal as a status and never
     * throws, its sibling throws. Nothing else in this file can check that pairing. */
    spacelab::terminal early = rx;
    early.epoch = -1.0;
    const auto refused = spacelab::try_bistatic_range(sat, tx, 0.02, early);
    CHECK(!refused.has_value());
    CHECK(refused.error() == SL_ERR_INVALID_ARGUMENT);
    try {
        (void)spacelab::bistatic_range(sat, tx, 0.02, early);
        CHECK(false);
    } catch (const spacelab::error &e) {
        CHECK(e.status() == SL_ERR_INVALID_ARGUMENT);
        CHECK(std::string_view(e.what()).find("sl_bistatic_range") != std::string_view::npos);
    }

    /* One flat ramp, spanned by a `std::span` the wrapper sizes from the container. Over it the two
     * Doppler observables sum to `M * f_reference` exactly. */
    const double exciter = 7.15e9;
    const std::array<spacelab::ramp_segment, 1> table{
        spacelab::ramp_segment{.start = 0.0, .end = 1e4, .start_frequency = exciter, .rate = 0.0}};
    const spacelab::doppler_link link{.uplink = SL_BAND_X,
                                      .downlink = SL_BAND_X,
                                      .reference_frequency = exciter,
                                      .reference_band = SL_BAND_X,
                                      .count_interval = 60.0,
                                      .subtracted = true};
    const double counted = spacelab::dsn_averaged_doppler(table, link, 100.0, 160.0);
    const double instantaneous =
        spacelab::dsn_measured_frequency(table, SL_BAND_X, SL_BAND_X, 100.0, 1.0);
    CHECK_NEAR(counted + instantaneous, (880.0 / 749.0) * exciter, 1e-2);
    /* Moyer's S-band denominator is 221 against X's 749, so the range units are in that ratio. */
    const double wide = spacelab::dsn_sequential_range(table, SL_BAND_X, 100.0, 101.0, 26);
    const double s_band = spacelab::dsn_sequential_range(table, SL_BAND_S, 100.0, 101.0, 26);
    CHECK_NEAR(s_band / wide, 749.0 / 221.0, 1e-12);

    /* The corona series defaults to Aksim & Pavlov when the span is empty, and the caller's own
     * span replaces it -- so the two answers must differ. */
    const spacelab::vec3 near_sun{.x = -1.5e11, .y = 6.96e9, .z = 0.0};
    const spacelab::vec3 far_sun{.x = 2.3e11, .y = 6.96e9, .z = 0.0};
    const double group = spacelab::solar_corona_delay(near_sun, far_sun, 8.4e9);
    CHECK(group > 0.0);
    CHECK_NEAR(spacelab::solar_corona_delay(near_sun, far_sun, 8.4e9, false), -group, 0.0);
    const std::array<spacelab::corona_term, 1> mine{
        spacelab::corona_term{.coefficient = 1.0e12, .exponent = 2.0}};
    CHECK(spacelab::solar_corona_delay(near_sun, far_sun, 8.4e9, true, mine) != group);

    /* Total, so a null answers NaN and the wrapper marks it `noexcept` rather than wrapping it. */
    CHECK(spacelab::shapiro_delay(6.878e6, 2.6561e7, 2.0e7, SL_MU_EARTH) > 0.0);
    spacelab::state remote = sat;
    remote.position.x += 12000.0;
    CHECK(spacelab::inter_satellite_range(sat, remote) > 11000.0);
    CHECK(std::isfinite(spacelab::inter_satellite_range_rate(sat, remote)));
}

/* The decay scan and the element-set printer, each returning an owning C++ value. */
void the_lifetime_wrapper_owns_its_quantiles_and_its_two_lines() {
    static_assert(sizeof(::sl_lifetime_outcome) == sizeof(std::int32_t));

    const spacelab::decay_params p = decay_case();
    /* `default_decay_params` leaves the four an atmosphere cannot be guessed for as NaN, so the
     * caller who forgets one is refused rather than answered on a silent zero. */
    CHECK(std::isnan(spacelab::default_decay_params().reference_density));
    CHECK(spacelab::default_decay_params().chunk == 86400.0);

    const std::array<double, 3> zonals{0.0, 0.0, -SL_J2_EARTH};
    std::array<double, 6> initial{SL_RE_EARTH + 300e3, 0.0, 0.0, 0.0, 0.0, 0.0};

    const spacelab::lifetime low = spacelab::orbital_lifetime(p, initial, zonals);
    CHECK(low.outcome == SL_LIFETIME_REENTRY);
    CHECK(low.seconds > 0.0 && low.seconds < p.horizon);
    CHECK(low.perigee_altitude_m <= p.reentry_altitude + 1e3);

    /* 700 km higher is `exp(-700/60)` of the density, so the object outlives the horizon and the
     * second field is the horizon rather than a decay epoch nobody observed. */
    initial[0] = SL_RE_EARTH + 1000e3;
    const spacelab::lifetime high = spacelab::orbital_lifetime(p, initial, zonals);
    CHECK(high.outcome == SL_LIFETIME_SURVIVES);
    CHECK_NEAR(high.seconds, p.horizon, 0.0);
    initial[0] = SL_RE_EARTH + 300e3;

    /* The quantiles and the counts arrive together, because the counts are what say how much of
     * the distribution the quantiles are conditioned on. */
    const std::array<double, 3> scales{0.5, 1.0, 2.0};
    const std::array<double, 3> quantiles{0.0, 0.5, 1.0};
    const spacelab::lifetime_quantiles spread =
        spacelab::orbital_lifetime_ensemble(p, initial, zonals, scales, quantiles);
    CHECK(spread.seconds.size() == quantiles.size());
    CHECK(spread.ensemble.members == 3 && spread.ensemble.survivors == 0);
    CHECK(std::is_sorted(spread.seconds.begin(), spread.seconds.end()));
    CHECK_NEAR(spacelab::ballistic_log_sensitivity(p, initial, zonals, 0.05), -1.0, 0.2);

    /* Fewer than three zonals is a refusal the `try_` half reports without throwing. */
    const std::array<double, 2> too_few{0.0, 0.0};
    const auto refused = spacelab::try_orbital_lifetime(p, initial, too_few);
    CHECK(!refused.has_value());
    CHECK(refused.error() == SL_ERR_INVALID_ARGUMENT);

    /* The two lines arrive as `std::string`, which is what a caller writes to a file -- the C side
     * hands back a fixed 70-byte field and the wrapper is what trims it at the terminator. */
    const spacelab::utc_datetime epoch{
        .year = 2008, .month = 9, .day = 20, .hour = 12, .minute = 25, .second = 12.25};
    const std::array<double, 6> parameters{15.72125391, 0.0006703, 51.6416,
                                           247.4627,    130.5360,  325.0288};
    const spacelab::tle_lines lines =
        spacelab::tle(epoch, 25544, parameters, -1.1606e-4, "1998-067A", 292, 56353);
    CHECK(lines.line1 == "1 25544U 98067A   08264.51750289  .00000000  00000-0 -11606-3 0  2921");
    CHECK(lines.line2 == "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537");
    CHECK(lines.line1.size() == 69 && lines.line2.size() == 69);
    /* An empty designator is the wrapper's own null, not a zero-length string pushed across. */
    CHECK(spacelab::tle(epoch, 25544, parameters).line1.size() == 69);
}

/* The carrier observables, and the two detectors whose answers the wrapper sizes for the caller. */
void the_gnss_wrapper_sizes_its_verdicts_and_its_candidates() {
    static_assert(sizeof(::sl_slip) == sizeof(std::int32_t));

    const double f1 = 1575.42e6;
    const double f2 = 1227.60e6;
    const spacelab::lanes lanes = spacelab::lane_wavelengths(f1, f2);
    CHECK_NEAR(lanes.wide_lane, 0.8619, 1e-4);
    CHECK_NEAR(lanes.narrow_lane, 0.1070, 1e-4);
    /* Equal frequencies have no lane: the `try_` half says so, the throwing half throws. */
    CHECK(!spacelab::try_lane_wavelengths(f1, f1).has_value());
    try {
        (void)spacelab::lane_wavelengths(f1, f1);
        CHECK(false);
    } catch (const spacelab::error &e) {
        CHECK(e.status() == SL_ERR_INVALID_ARGUMENT);
    }
    CHECK_NEAR(spacelab::carrier_frequency('G', '1'), f1, 0.0);
    CHECK(std::isnan(spacelab::carrier_frequency('R', '1')));

    const double geometric = 22100000.0;
    const double i1 = 4.5;
    const double i2 = i1 * (f1 / f2) * (f1 / f2);
    const double p1 = geometric + i1;
    const double p2 = geometric + i2;
    const double l1 = (geometric - i1) + (lanes.lambda1 * 1234.0);
    const double l2 = (geometric - i2) + (lanes.lambda2 * 987.0);
    CHECK_NEAR(spacelab::ionosphere_free(f1, f2, p1, p2), geometric, 1e-6);
    CHECK_NEAR(spacelab::wide_lane(f1, f2, p1, p2) + spacelab::narrow_lane(f1, f2, p1, p2),
               2.0 * geometric, 1e-6);
    CHECK_NEAR(spacelab::melbourne_wubbena(f1, f2, p1, p2, l1, l2, true), 1234.0 - 987.0, 1e-6);
    CHECK_NEAR(spacelab::geometry_free(p1, p2), i1 - i2, 1e-8);
    CHECK_NEAR(spacelab::graphic(p1, l1), 0.5 * (p1 + l1), 0.0);

    /* A series drifting 0.02 m a sample with a 0.30 m step at index 20. The wrapper returns one
     * verdict per sample in a vector it sized itself, which is the whole point of it. */
    std::vector<double> times;
    std::vector<double> values;
    for (int i = 0; i < 40; ++i) {
        times.push_back(30.0 * static_cast<double>(i));
        values.push_back((0.02 * static_cast<double>(i)) + (i >= 20 ? 0.30 : 0.0));
    }
    const std::vector<spacelab::slip_verdict> first_difference =
        spacelab::detect_cycle_slips_geometry_free(times, values, 0.10, 120.0);
    CHECK(first_difference.size() == times.size());
    CHECK(first_difference[0].verdict == SL_SLIP_ARC_START);
    CHECK(std::isnan(first_difference[0].statistic));
    CHECK(first_difference[20].verdict == SL_SLIP_SLIPPED);
    CHECK_NEAR(first_difference[20].statistic, 0.32, 1e-12);
    CHECK(std::count_if(
              first_difference.begin(), first_difference.end(),
              [](const spacelab::slip_verdict &v) { return v.verdict == SL_SLIP_SLIPPED; }) == 1);
    const std::vector<spacelab::slip_verdict> fitted =
        spacelab::detect_cycle_slips_polynomial(times, values, 1, 5, 0.10, 120.0);
    /* The fit tests nothing until its window fills and clears the window after a slip, so the two
     * detectors disagree on the sample after the step. That is why there are two. */
    CHECK(fitted[4].verdict == SL_SLIP_ARC_START);
    CHECK(fitted[20].verdict == SL_SLIP_SLIPPED);
    CHECK(fitted[21].verdict == SL_SLIP_ARC_START);
    CHECK(
        !spacelab::try_detect_cycle_slips_polynomial(times, values, 3, 4, 0.10, 120.0).has_value());

    const std::array<double, 3> flt{5.45, 3.10, 2.97};
    const std::array<double, 9> covariance{6.290, 5.978, 0.544, 5.978, 6.292,
                                           2.340, 0.544, 2.340, 6.288};
    const spacelab::decorrelation reduced = spacelab::ambiguity_decorrelation(flt, covariance);
    CHECK(reduced.transformation.size() == 9);
    CHECK(reduced.conditional_variances.size() == 3);
    /* `Z` is unimodular, so every entry is an integer -- the one property no wrong reduction has. */
    for (double z : reduced.transformation) {
        CHECK(std::fabs(z - std::round(z)) < 1e-9);
    }
    const spacelab::candidate bootstrapped = spacelab::ambiguity_bootstrap(flt, covariance);
    const std::vector<spacelab::candidate> best = spacelab::resolve_ambiguities(flt, covariance, 2);
    CHECK(best.size() == 2);
    CHECK(best[0].ambiguities.size() == flt.size());
    CHECK(best[0].squared_norm <= best[1].squared_norm);
    /* The search is at least as good as the bootstrap, which is the whole reason it exists. */
    CHECK(best[0].squared_norm <= bootstrapped.squared_norm);
    /* A shape the wrapper can see is wrong is refused before the call, not at the boundary. */
    const std::array<double, 4> mismatched{1.0, 0.0, 0.0, 1.0};
    CHECK(!spacelab::try_ambiguity_decorrelation(flt, mismatched).has_value());
}

/*
 * The conjunction wrapper, including the piece only C++ has: a sampler that throws, whose exception
 * has to survive a round trip through a C function pointer and arrive at the caller intact.
 */
void the_conjunction_wrapper_carries_an_exception_back_through_c() {
    const spacelab::conjunction geometry = conjunction_case(50.0);
    const spacelab::probability_estimates estimates = spacelab::collision_probability(geometry);
    CHECK(std::fabs(estimates.patera - estimates.foster) < 0.02 * estimates.foster);
    CHECK(std::fabs(estimates.chan - estimates.foster) < 0.02 * estimates.foster);
    CHECK(estimates.maximum >= estimates.foster);
    /* Head-on with an isotropic joint covariance `2*sigma^2` and no miss, the integral is
     * elementary: `1 - exp(-R^2/2s^2)`. */
    const double exact = 1.0 - std::exp(-(10.0 * 10.0) / (2.0 * 2.0 * 100.0 * 100.0));
    CHECK_NEAR(spacelab::collision_probability(conjunction_case(0.0)).foster, exact, 2e-4 * exact);
    /* Further away is less likely, which is the one monotonicity the model rests on. */
    CHECK(spacelab::collision_probability(conjunction_case(300.0)).foster < estimates.foster);
    /* A degenerate encounter velocity has no encounter plane: a domain refusal, not a zero. */
    spacelab::conjunction still = geometry;
    still.relative_velocity.y = 0.0;
    const auto degenerate = spacelab::try_collision_probability(still);
    CHECK(!degenerate.has_value());
    CHECK(degenerate.error() == SL_ERR_DOMAIN);

    /* Feed the domain check its own assumption, through a capturing lambda -- which is the thing a
     * C function pointer cannot take and the trampoline exists for. */
    const auto rectilinear = [&geometry](double t, ::sl_relative_sample &out) {
        out.position.x = geometry.relative_position.x + (geometry.relative_velocity.x * t);
        out.position.y = geometry.relative_position.y + (geometry.relative_velocity.y * t);
        out.position.z = geometry.relative_position.z + (geometry.relative_velocity.z * t);
        out.velocity = geometry.relative_velocity;
        for (std::size_t i = 0; i < 9; ++i) {
            out.covariance[i] = geometry.primary_covariance[i] + geometry.secondary_covariance[i];
        }
        return true;
    };
    const spacelab::usage_violations clear =
        spacelab::conjunction_usage_violations(geometry, 5400.0, rectilinear);
    CHECK(!clear.any);
    CHECK_NEAR(clear.correction_factor, 1.0, 1e-6);
    /* The period is the scale both ratios are stated on, so halving it doubles them exactly. */
    const spacelab::usage_violations halved =
        spacelab::conjunction_usage_violations(geometry, 2700.0, rectilinear);
    CHECK_NEAR(halved.offset_indicator, 2.0 * clear.offset_indicator, 1e-9);
    CHECK_NEAR(halved.extended_indicator, 2.0 * clear.extended_indicator, 1e-9);
    /* A sampler that answers `false` is a refusal, reported as a status rather than as a check
     * that happened to find nothing wrong. */
    const auto never = [](double, ::sl_relative_sample &) { return false; };
    const auto blind = spacelab::try_conjunction_usage_violations(geometry, 5400.0, never);
    CHECK(!blind.has_value());
    CHECK(blind.error() == SL_ERR_COMPUTE);

    /* The scan, with its own capturing lambda. `10000*sin(2*pi*t/600)` along track against a
     * constant 500 m across it puts the minima at 300, 600 and 900 s. The wrapper runs both halves
     * of the two-call protocol, so the vector arrives already the right length. */
    const double rate = 2.0 * std::numbers::pi / 600.0;
    const auto drifting = [rate](double t, ::sl_state &out) {
        out.position = {.x = 10000.0 * std::sin(rate * t), .y = 500.0, .z = 0.0};
        out.velocity = {.x = 10000.0 * rate * std::cos(rate * t), .y = 0.0, .z = 0.0};
        return true;
    };
    const std::vector<spacelab::close_approach> found =
        spacelab::closest_approaches(drifting, 10.0, 1000.0, 30.0, 1e-4);
    CHECK(found.size() == 3);
    for (std::size_t i = 0; i < found.size(); ++i) {
        CHECK_NEAR(found[i].tca, 300.0 * static_cast<double>(i + 1), 0.5);
        CHECK_NEAR(found[i].distance, 500.0, 1e-3);
        CHECK_NEAR(found[i].relative_speed, 10000.0 * rate, 1e-3);
    }
    /* The step is a resolution as much as a bound: extrema 300 s apart need a step that can
     * bracket them separately, and a coarser one reports fewer than there are. */
    CHECK(spacelab::closest_approaches(drifting, 10.0, 1000.0, 400.0, 1e-4).size() < found.size());

    /*
     * The claim this whole file exists for: a sampler that throws is not a sampler that returned
     * false. The exception crosses a C function pointer -- where unwinding is undefined -- by being
     * caught in the trampoline, parked in an `exception_ptr` and rethrown once the C call has
     * returned. A wrapper that lost it would report `SL_ERR_COMPUTE` and the caller would read
     * "no ephemeris" as "no conjunctions".
     */
    bool rethrown = false;
    const auto truncated = [rate, &drifting](double t, ::sl_state &out) {
        if (t > 500.0) {
            throw std::runtime_error("no ephemeris past 500 s");
        }
        (void)rate;
        return drifting(t, out);
    };
    try {
        (void)spacelab::closest_approaches(truncated, 10.0, 1000.0, 30.0, 1e-4);
    } catch (const std::runtime_error &e) {
        rethrown = std::string_view(e.what()).find("500 s") != std::string_view::npos;
    }
    CHECK(rethrown);

    /* Filter 1 of an all-on-all screen: a certainty in one direction and nothing in the other. */
    CHECK(spacelab::radially_separated(SL_RE_EARTH + 400e3, SL_RE_EARTH + 420e3,
                                       SL_RE_EARTH + 800e3, SL_RE_EARTH + 900e3, 5000.0));
    CHECK(!spacelab::radially_separated(SL_RE_EARTH + 400e3, SL_RE_EARTH + 900e3,
                                        SL_RE_EARTH + 800e3, SL_RE_EARTH + 420e3, 1.0));
}

/*
 * The control pillar through the C++ wrapper.
 *
 * The claims here are the ones only the wrapper makes: that the multi-value returns arrive as
 * aggregates rather than as out-parameters nobody checked, that `loop_run` sizes its own sample
 * vector through the two-call protocol, and that a refusal arrives as a `spacelab::error` carrying
 * the status the C ABI produced. The physics is pinned in Rust and in `smoke.c`; what is checked
 * against numbers here is only what would still be right if the wrapper had crossed two fields.
 */
void the_control_wrapper_closes_the_loop_and_carries_its_refusals() {
    static_assert(sizeof(spacelab::latency) == 3 * sizeof(double));
    static_assert(sizeof(spacelab::attitude_state) == 12 * sizeof(double));
    static_assert(offsetof(spacelab::attitude_state, rate) == 9 * sizeof(double));
    static_assert(sizeof(spacelab::loop_sample) == 5 * sizeof(double));
    static_assert(sizeof(spacelab::delay_margin) == 5 * sizeof(double));
    static_assert(sizeof(spacelab::element_feedback) == 8 * sizeof(double));
    static_assert(std::is_trivially_copyable_v<spacelab::loop_sample>);
    static_assert(sizeof(::sl_attitude_law) == sizeof(std::int32_t));
    static_assert(sizeof(::sl_consensus_protocol) == sizeof(std::int32_t));

    const std::array<double, 9> identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const std::array<double, 9> inertia{4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
    const double c = std::cos(0.3), s = std::sin(0.3);
    const std::array<double, 9> turned{1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c};
    const spacelab::vec3 rest{0.0, 0.0, 0.0};
    const spacelab::latency latency{0.2, 0.1, 0.0};

    CHECK_NEAR(spacelab::latency_total(latency), 0.3, 1e-15);
    const std::size_t lag = spacelab::latency_steps(latency, 0.1);
    CHECK(lag == 3);

    /* The MRP and the principal angle come back as one aggregate, and `|sigma| = tan(theta/4)`
     * checks that the wrapper did not swap the two out-parameters. */
    const spacelab::attitude_error error =
        spacelab::attitude_error_mrp(turned, identity, rest, rest);
    CHECK_NEAR(error.angle_rad, 0.3, 1e-12);
    CHECK_NEAR(std::sqrt((error.mrp.x * error.mrp.x) + (error.mrp.y * error.mrp.y) +
                         (error.mrp.z * error.mrp.z)),
               std::tan(0.3 / 4.0), 1e-12);

    /* The other chart, as its own aggregate: `sigma = q_v / (1 + q0)` relates the two, so the
     * wrapper cannot have crossed the scalar part with a vector one and still satisfy this. */
    const std::array<double, 4> q =
        spacelab::attitude_error_quaternion(turned, identity, rest, rest);
    CHECK(q[0] >= 0.0);
    CHECK_NEAR(2.0 * std::acos(q[0]), 0.3, 1e-12);
    CHECK_NEAR(q[1] / (1.0 + q[0]), error.mrp.x, 1e-15);

    /* `(k_theta, k_omega)`, in that order: the quaternion law's stiffness is twice the MRP law's
     * at the same nominal gains and its rate gain is the same, which distinguishes the pair from
     * its own transposition. */
    const spacelab::small_angle_gains quaternion =
        spacelab::attitude_small_angle_gains(SL_LAW_QUATERNION_PD, 8.0, 12.0);
    const spacelab::small_angle_gains mrp =
        spacelab::attitude_small_angle_gains(SL_LAW_MRP_LYAPUNOV, 8.0, 12.0);
    CHECK_NEAR(quaternion.angle, 2.0 * mrp.angle, 1e-15);
    CHECK(quaternion.rate == mrp.rate);

    /* `(omega_n, zeta)`, likewise: at the critically damped gain the second is one and the first
     * is not, so a swap could not pass. */
    const double derivative = spacelab::quaternion_pd_critically_damped(8.0, 4.0);
    const spacelab::pd_response response = spacelab::quaternion_pd_response(8.0, derivative, 4.0);
    CHECK_NEAR(response.damping_ratio, 1.0, 1e-14);
    CHECK_NEAR(response.natural_frequency_rad_s, std::sqrt(8.0 / (2.0 * 4.0)), 1e-14);

    /* The loop sizes its own sample vector -- the branch of the two-call protocol `smoke.c`
     * writes out by hand -- and returns the final state alongside it. */
    spacelab::attitude_state initial{};
    spacelab::attitude_state seed{};
    std::copy(turned.begin(), turned.end(), std::begin(initial.rotation));
    std::copy(identity.begin(), identity.end(), std::begin(seed.rotation));
    initial.rate = rest;
    seed.rate = rest;
    const spacelab::loop_run run = spacelab::attitude_loop_run(
        SL_LAW_QUATERNION_PD, 8.0, 12.0, latency, 0.1, initial, seed, identity, inertia, 40);
    CHECK(run.samples.size() == 40);
    CHECK(std::all_of(run.samples.begin(), run.samples.begin() + static_cast<std::ptrdiff_t>(lag),
                      [](const spacelab::loop_sample &v) { return v.torque.x == 0.0; }));
    CHECK(run.samples[lag].torque.x != 0.0);
    CHECK_NEAR(run.final_state.rotation[0], 1.0, 1e-6);

    /* Every field of the margin is filled, and the normalised form is the product of the two the
     * caller can check for themselves. */
    const spacelab::delay_margin margin =
        spacelab::delay_margin_of_law(SL_LAW_QUATERNION_PD, 8.0, 12.0, 4.0);
    CHECK_NEAR(margin.normalised, margin.seconds * margin.natural_frequency_rad_s, 1e-12);
    CHECK(spacelab::delay_margin_of_gains(4.0, quaternion.angle, quaternion.rate).seconds ==
          margin.seconds);
    const spacelab::binding_axis worst = spacelab::worst_principal_axis(
        SL_LAW_QUATERNION_PD, 8.0, 12.0, spacelab::vec3{6.0, 2.0, 9.0});
    CHECK(worst.axis < 3);
    CHECK(worst.margin.seconds <= margin.seconds);

    /* The LQR's three answers come back together, so the residual the gain was accepted at cannot
     * be dropped on the floor. */
    const std::array<double, 4> plant_a{0.0, 1.0, 0.0, 0.0};
    const std::array<double, 2> plant_b{0.0, 1.0};
    const std::array<double, 4> weight_q{1.0, 0.0, 0.0, 1.0};
    const std::array<double, 1> weight_r{1.0};
    const spacelab::lqr_solution solved = spacelab::lqr(plant_a, plant_b, weight_q, weight_r, 2, 1);
    CHECK(solved.gain.size() == 2);
    CHECK(solved.riccati.size() == 4);
    CHECK(solved.residual < 1e-9);
    CHECK_NEAR(solved.gain[0], solved.riccati[2], 1e-12);

    const spacelab::linear_plant cw =
        spacelab::clohessy_wiltshire_plant(2.0 * std::numbers::pi / 5400.0);
    CHECK(cw.a.size() == 36 && cw.b.size() == 18);
    CHECK(std::all_of(cw.b.begin(), cw.b.begin() + 9, [](double v) { return v == 0.0; }));

    /* The station-keeping law is a plain aggregate the caller fills, so this is the check that the
     * wrapper passes it through unreordered: one controlled element, `V = 1/2 w de^2`. */
    const spacelab::element_feedback law{{1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1e-6, 1e-4};
    const std::array<double, 6> elements{7.0e6, 0.0, 0.0, 0.0, 0.0, 0.0};
    const std::array<double, 6> drift{100.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK_NEAR(spacelab::element_feedback_lyapunov(law, drift), 0.5 * 100.0 * 100.0, 1e-9);
    CHECK(spacelab::element_feedback_decrease_rate(law, elements, SL_MU_EARTH, 0.0, drift) <= 0.0);
    const spacelab::vec3 push =
        spacelab::element_feedback_acceleration(law, elements, SL_MU_EARTH, 0.0, drift);
    CHECK(std::sqrt((push.x * push.x) + (push.y * push.y) + (push.z * push.z)) <=
          1e-4 * (1.0 + 1e-12));

    /* The consensus surface sizes its spectrum from the agent count the caller states, and the
     * published bound is recovered from that spectrum rather than from a literal. */
    const std::array<double, 16> path{0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                                      0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0};
    const std::vector<double> spectrum = spacelab::laplacian_spectrum(path, 4);
    CHECK(spectrum.size() == 4);
    CHECK(spacelab::algebraic_connectivity(path, 4) == spectrum[1]);
    const double bound = spacelab::consensus_delay_bound(path, 4, SL_CONSENSUS_DELAYED_RELATIVE);
    CHECK_NEAR(bound, std::numbers::pi / (2.0 * spectrum[3]), 1e-12);

    /* The modes size themselves from the agent count and arrive alongside the eigenvalues, so a
     * caller cannot take one and silently lose the other. */
    const spacelab::laplacian_modes eigen = spacelab::laplacian_eigen(path, 4);
    CHECK(eigen.values == spectrum);
    CHECK(eigen.modes.size() == 16);
    for (std::size_t k = 0; k < 4; ++k) {
        const std::span<const double> mode{eigen.modes.data() + (4 * k), 4};
        double squared = 0.0;
        for (const double v : mode) {
            squared += v * v;
        }
        CHECK_NEAR(squared, 1.0, 1e-12);
    }

    /* Connectivity is a `bool` and not a refusal, which is the whole reason it is separate from
     * every other entry point in this section. */
    CHECK(spacelab::graph_is_connected(path, 4));
    std::array<double, 16> split = path;
    split[(4 * 1) + 2] = 0.0;
    split[(4 * 2) + 1] = 0.0;
    CHECK(!spacelab::graph_is_connected(split, 4));

    /* Cayley-Hamilton on the companion matrix of `(l+1)(l+2)(l+3)`: the coefficients are checked
     * against the polynomial they encode, which is the one thing a transposed read could not
     * survive. */
    const std::array<double, 9> companion{0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -6.0, -11.0, -6.0};
    const std::vector<double> coefficients = spacelab::characteristic_polynomial(companion, 3);
    CHECK(coefficients.size() == 3);
    CHECK_NEAR(coefficients[0], 6.0, 1e-12);
    CHECK_NEAR(coefficients[1], 11.0, 1e-12);
    CHECK_NEAR(coefficients[2], 6.0, 1e-12);

    const std::array<double, 4> start{1.0, -2.0, 3.0, -0.5};
    const spacelab::consensus_result settled = spacelab::consensus_run(
        path, SL_CONSENSUS_DELAYED_RELATIVE, 0.7 * bound, start, 40.0, 1e-9);
    CHECK(settled.final_state.size() == 4);
    CHECK(settled.accepted > 0);
    CHECK(settled.disagreement < 1e-3 * spacelab::consensus_disagreement(start));
    /* An undirected Laplacian annihilates the all-ones vector, so the mean is an invariant of the
     * run whatever the delay does to the rest of it. */
    const double mean = std::accumulate(start.begin(), start.end(), 0.0) / 4.0;
    const double ended =
        std::accumulate(settled.final_state.begin(), settled.final_state.end(), 0.0) / 4.0;
    CHECK_NEAR(ended, mean, 1e-6);

    const std::array<double, 2> delays{0.25, 0.4};
    const std::vector<double> mesh = spacelab::delay_breakpoints(delays, 0.0, 1.0, 2);
    CHECK(!mesh.empty());
    CHECK(std::is_sorted(mesh.begin(), mesh.end()));
    CHECK(mesh.front() > 0.0 && mesh.back() < 1.0);

    /* And a refusal is an exception carrying the status the C ABI produced -- the wrapper's whole
     * reason for existing over the two-call protocol by hand. */
    bool threw = false;
    try {
        (void)spacelab::latency_steps(latency, 0.08);
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_DOMAIN);
    }
    CHECK(threw);
    threw = false;
    try {
        /* A shape this ABI has no instantiation for, refused at the boundary rather than reshaped
         * -- and the size check is the wrapper's own, before the call. */
        (void)spacelab::lqr(plant_a, plant_b, weight_q, weight_r, 5, 2);
    } catch (const spacelab::error &e) {
        threw = true;
        CHECK(e.status() == SL_ERR_INVALID_ARGUMENT);
    }
    CHECK(threw);
}

} // namespace

/*
 * The RAII form of the cloud masker. `open` is the factory rather than the constructor for the same
 * reason `sgp4::open` is -- a build without the `onnx` feature has to be able to say so without
 * throwing, and `SL_ERR_UNSUPPORTED` is that answer. The gate builds the default library, so this
 * asserts the shape of the refusal; with the feature on the Rust suite checks the mask itself.
 */
void the_cloud_masker_reports_a_build_without_inference() {
    const std::array<std::uint8_t, 4> model{0, 0, 0, 0};
    const spacelab::result<spacelab::cloud_masker> opened =
        spacelab::cloud_masker::open(model, 0.5f);
    if (!opened) {
        CHECK(opened.error() == SL_ERR_UNSUPPORTED || opened.error() == SL_ERR_PARSE);
    }
    /* Whatever the build, `open` is total: it either yields a handle that frees itself or a status,
     * and there is no third state in which something has been allocated and lost. */
    CHECK(opened.has_value() || opened.error() != SL_OK);
}

int main() {
    /* Every failure this program expects arrives as a `spacelab::error` inside a `try`. One that
     * escapes to here is a test that was wrong about what it was calling, and it is caught rather
     * than left to `std::terminate`, which reports nothing. */
    try {
        the_structs_have_the_layout_both_sides_assume();
        the_geodetic_round_trip_agrees_with_the_rust_measurement();
        the_sgp4_class_owns_its_handle_and_batches_correctly();
        lambert_returns_the_orbit_it_was_asked_about();
        the_generators_size_their_own_buffers();
        the_design_and_measurement_functions_answer_in_radians();
        the_frame_chain_runs_and_the_two_frames_differ_by_precession();
        the_ephemeris_class_owns_its_handle_and_refuses_to_extrapolate();
        the_dsst_wrapper_sizes_its_own_answer_and_matches_the_other_suites();
        the_least_squares_wrapper_takes_a_lambda_and_propagates_its_exception();
        the_information_filter_owns_its_handle_and_accumulates();
        the_new_families_generate_what_they_promise();
        the_coverage_wrapper_agrees_with_the_windows_it_summarises();
        the_gnss_wrapper_reads_a_file_and_refuses_a_geometry();
        the_atmosphere_wrapper_answers_what_the_formulas_say();
        every_force_model_is_a_factory_that_owns_its_handle();
        the_radar_wrapper_sizes_its_own_answers_and_finds_the_scatterer();
        the_optical_wrapper_triangulates_detects_and_suppresses();
        the_radiometric_wrapper_splits_its_refusals_from_its_answers();
        the_lifetime_wrapper_owns_its_quantiles_and_its_two_lines();
        the_gnss_wrapper_sizes_its_verdicts_and_its_candidates();
        the_conjunction_wrapper_carries_an_exception_back_through_c();
        the_cloud_masker_reports_a_build_without_inference();
        the_control_wrapper_closes_the_loop_and_carries_its_refusals();
    } catch (const spacelab::error &e) {
        std::fprintf(stderr, "FAIL: uncaught spacelab::error, status %d: %s\n", e.status(),
                     e.what());
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
        return EXIT_FAILURE;
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("spacelab C++ smoke: all checks passed");
    return EXIT_SUCCESS;
}
