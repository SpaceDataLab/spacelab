/*
 * The C++ header compiled with `-fno-exceptions`, exercised through the `try_` half of the API.
 *
 * A header-only wrapper whose every failure path is a `throw` is unusable to the consumers that
 * turn exceptions off -- games, embedded targets, anything with a `-fno-exceptions` house rule --
 * and "use the C header instead" gives up the RAII handle and the vector sizing, which is most of
 * what the wrapper is for. This translation unit is the proof that they do not have to: it never
 * names a throwing entry point, and it is compiled with exceptions disabled, so if one appeared in
 * a path it uses the build would fail rather than the test.
 *
 * The claims are the ones the `try_` layer adds, not the physics, which `smoke.cpp` already pins:
 * that a failure arrives as a `result` carrying the same status the C API returned, that a success
 * carries the same value the throwing form would have returned, and that the handle still releases
 * itself when nothing can throw.
 */

#include "spacelab.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef __cpp_exceptions
#error "this translation unit exists to test the -fno-exceptions build and must be compiled with it"
#endif

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

constexpr const char *LINE1 =
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
constexpr const char *LINE2 =
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

double radius_of(const spacelab::vec3 &r) {
    return std::sqrt((r.x * r.x) + (r.y * r.y) + (r.z * r.z));
}

/*
 * `result<T>` in both of its flavours. Which one this is depends on the standard: the C++20 rung
 * of the build gets the stand-in and the C++23 and C++26 rungs get `std::expected`, so the two
 * paths are both compiled rather than one of them being a claim.
 */
void a_result_carries_either_a_value_or_a_status() {
    const spacelab::geodetic g{
        .latitude_rad = 35.0 * M_PI / 180.0,
        .longitude_rad = 139.0 * M_PI / 180.0,
        .height_m = 42.0,
    };

    const spacelab::result<spacelab::vec3> good = spacelab::try_geodetic_to_itrf(g);
    CHECK(static_cast<bool>(good));
    CHECK_NEAR(good->z, 3637891.0, 1.0);
    CHECK_NEAR((*good).z, good->z, 0.0);

    /* Plain `if (r)` rather than `and_then`: the monadic operations are `std::expected`'s, and
     * this file has to compile on the C++20 rung where `result` is the stand-in. */
    const spacelab::result<spacelab::geodetic> back = spacelab::try_itrf_to_geodetic(*good);
    CHECK(static_cast<bool>(back));
    if (back) {
        CHECK_NEAR(back->latitude_rad, g.latitude_rad, 1e-12);
        const spacelab::result<spacelab::vec3> round_trip = spacelab::try_geodetic_to_itrf(*back);
        CHECK(static_cast<bool>(round_trip));
        CHECK_NEAR(round_trip->x, good->x, 1e-6);
    }
}

/* The handle without a constructor: `open` is what a build with no exceptions has instead. */
void the_handle_still_owns_itself_without_exceptions() {
    std::vector<spacelab::state> batch;

    {
        const spacelab::result<spacelab::sgp4> sat = spacelab::sgp4::open(LINE1, LINE2);
        CHECK(static_cast<bool>(sat));
        if (!sat) {
            return;
        }

        const std::vector<double> offsets{0.0, 45.0, 90.0, -30.0};
        const spacelab::result<std::vector<spacelab::state>> many =
            sat->try_at(std::span<const double>(offsets));
        CHECK(static_cast<bool>(many));
        batch = *many;

        for (std::size_t i = 0; i < offsets.size(); ++i) {
            const spacelab::result<spacelab::state> one = sat->try_at(offsets.at(i));
            CHECK(static_cast<bool>(one));
            CHECK(one->position.x == batch.at(i).position.x);
        }

        /* Empty is an answer here too, and it is the wrapper that knows that. */
        const spacelab::result<std::vector<spacelab::state>> none =
            sat->try_at(std::span<const double>{});
        CHECK(static_cast<bool>(none));
        CHECK(none->empty());
    }

    CHECK(batch.size() == 4);
    CHECK_NEAR(radius_of(batch.at(0).position), 6720189.23, 0.01);
}

/* Every failure the throwing half would have raised is a status here, with the same value. */
void a_failure_is_a_status_the_caller_can_branch_on() {
    const spacelab::result<spacelab::sgp4> broken = spacelab::sgp4::open("...", "...");
    CHECK(!broken);
    CHECK(broken.error() == SL_ERR_PARSE);
    CHECK(spacelab::last_error_message().find("invalid TLE") != std::string::npos);

    /* A domain error is the reason this half of the API exists: no arc for that time of flight is
     * an ordinary answer in a porkchop sweep, and aborting the process over it would be absurd. */
    const spacelab::vec3 r1{.x = 6778137.0, .y = 0.0, .z = 0.0};
    const spacelab::vec3 r2{.x = 0.0, .y = 6778137.0, .z = 0.0};
    const spacelab::result<spacelab::transfer> impossible =
        spacelab::try_lambert(r1, r2, 0.0, spacelab::mu_earth);
    CHECK(!impossible);
    CHECK(impossible.error() == SL_ERR_DOMAIN);

    const spacelab::result<double> no_sso = spacelab::try_sun_synchronous_inclination(42164e3);
    CHECK(!no_sso);
    CHECK(no_sso.error() == SL_ERR_DOMAIN);

    const spacelab::result<double> not_a_cycle = spacelab::try_repeat_ground_track_sma(0, 3, 1.7);
    CHECK(!not_a_cycle);
    CHECK(not_a_cycle.error() == SL_ERR_INVALID_ARGUMENT);

    /* And the total functions are unchanged: they never had a status to carry. */
    CHECK_NEAR(spacelab::critical_inclination(), std::acos(1.0 / std::sqrt(5.0)), 1e-12);
    CHECK(spacelab::abi_version() == SL_ABI_VERSION);
    CHECK(!spacelab::version().empty());
}

/* The two-call protocol is still hidden, and an infeasible design still names itself. */
void the_generators_still_size_their_own_buffers() {
    spacelab::walker_params w = spacelab::default_walker_params();
    w.pattern = SL_WALKER_DELTA;
    w.inclination_rad = 56.0 * M_PI / 180.0;
    w.planes = 6;
    w.per_plane = 4;
    w.phasing = 1;
    w.semi_major_axis_m = 29600e3;

    const spacelab::result<std::vector<spacelab::state>> shells = spacelab::try_walker(w);
    CHECK(static_cast<bool>(shells));
    CHECK(shells->size() == 24);
    for (const spacelab::state &s : *shells) {
        CHECK_NEAR(radius_of(s.position), w.semi_major_axis_m, 1.0);
    }

    spacelab::soc_params p = spacelab::default_soc_params();
    p.inclination_rad = M_PI / 2.0;
    p.altitude_m = 1200e3;
    p.min_elevation_rad = 10.0 * M_PI / 180.0;
    const spacelab::result<spacelab::soc_geometry> geometry =
        spacelab::try_street_of_coverage_geometry(p);
    CHECK(static_cast<bool>(geometry));

    const spacelab::result<std::vector<spacelab::state>> soc = spacelab::try_street_of_coverage(p);
    CHECK(static_cast<bool>(soc));
    CHECK(soc->size() == static_cast<std::size_t>(geometry->planes) * geometry->per_plane);

    p.min_elevation_rad = M_PI / 2.0;
    const spacelab::result<std::vector<spacelab::state>> impossible =
        spacelab::try_street_of_coverage(p);
    CHECK(!impossible);
    CHECK(impossible.error() == SL_ERR_DOMAIN);
}

/* The second RAII handle, built and queried without a single throwing name in sight. */
void the_ephemeris_is_a_factory_here_too() {
    const spacelab::state start{
        .position = {.x = 6778137.0, .y = 0.0, .z = 0.0},
        .velocity = {.x = 0.0, .y = 4767.5, .z = 6027.6},
    };

    const spacelab::result<spacelab::ephemeris> eph =
        spacelab::ephemeris::geopotential(start, 600.0, 8);
    CHECK(static_cast<bool>(eph));
    if (!eph) {
        return;
    }

    const spacelab::result<std::pair<double, double>> span = eph->try_span();
    CHECK(static_cast<bool>(span));
    CHECK(span->first == 0.0);
    CHECK(span->second == 600.0);

    const spacelab::result<spacelab::state> mid = eph->try_at(300.0);
    CHECK(static_cast<bool>(mid));
    CHECK(radius_of(mid->position) > 6000e3);

    const spacelab::result<spacelab::state> off = eph->try_at(601.0);
    CHECK(!off);
    CHECK(off.error() == SL_ERR_DOMAIN);

    const spacelab::result<spacelab::ephemeris> too_deep =
        spacelab::ephemeris::geopotential(start, 600.0, 100000);
    CHECK(!too_deep);
    CHECK(too_deep.error() == SL_ERR_DOMAIN);
}

/* DSST without exceptions: only the `try_` half exists, and it is the whole API here. */
void the_dsst_wrapper_still_sizes_its_own_answer() {
    const spacelab::dsst_params p = spacelab::default_dsst_params();
    const double inclination = 51.6 * M_PI / 180.0;
    const spacelab::equinoctial initial{7.0e6, 1.0e-3, 0.0, std::tan(inclination / 2.0), 0.0, 0.0};
    const std::vector<double> times{1800.0, 86400.0};

    const spacelab::result<std::vector<spacelab::equinoctial>> mean =
        spacelab::try_dsst_zonal_mean(p, initial, times);
    CHECK(static_cast<bool>(mean));
    if (mean) {
        CHECK(mean->size() == 2);
        CHECK_NEAR(mean->at(0).at(0), 7.0e6, 1e-6);
        CHECK_NEAR(mean->at(0).at(5), 1.940206765692, 1e-9);
    }

    const spacelab::result<std::vector<spacelab::equinoctial>> full =
        spacelab::try_dsst_osculating(p, initial, times);
    CHECK(static_cast<bool>(full));
    if (full) {
        CHECK_NEAR(full->at(0).at(0), 6.9957388043e6, 1.0);
    }

    spacelab::dsst_params broken = p;
    broken.zonal_count = SL_DSST_MAX_ZONAL + 1;
    const spacelab::result<std::vector<spacelab::equinoctial>> refused =
        spacelab::try_dsst_zonal_mean(broken, initial, times);
    CHECK(!refused);
    CHECK(refused.error() == SL_ERR_INVALID_ARGUMENT);
}

/*
 * The least-squares model is a lambda here too. The trampoline that carries it across the C
 * boundary compiles down to nothing but the call when there is no exception machinery to guard,
 * and this translation unit is what proves that path builds at all.
 */
void the_least_squares_wrapper_takes_a_lambda_without_exceptions() {
    const std::array<double, 6> target{1.0, -2.0, 3.5, 0.25, -0.75, 10.0};
    auto model = [&](std::span<const double, 6> state, std::span<spacelab::residual_row> rows) {
        for (std::size_t i = 0; i < rows.size(); ++i) {
            for (std::size_t j = 0; j < 6; ++j) {
                rows[i].jacobian[j] = (i == j) ? 1.0 : 0.0;
            }
            rows[i].residual = state[i] - target.at(i);
            rows[i].weight = 4.0;
        }
    };

    const spacelab::result<spacelab::estimate> solved =
        spacelab::try_gauss_newton(model, {0, 0, 0, 0, 0, 0}, 6);
    CHECK(static_cast<bool>(solved));
    if (solved) {
        CHECK(solved->converged);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK_NEAR(solved->state[i], target.at(i), 1e-9);
            CHECK_NEAR(solved->covariance[(i * 6) + i], 0.25, 1e-12);
        }
    }

    const spacelab::result<spacelab::estimate> narrow =
        spacelab::try_gauss_newton(model, {0, 0, 0, 0, 0, 0}, 1);
    CHECK(!narrow);
    CHECK(narrow.error() == SL_ERR_DOMAIN);
}

/* And the filter handle still releases itself with nothing to throw. */
void the_information_filter_still_owns_itself() {
    const std::array<double, 6> target{1.0, -2.0, 3.5, 0.25, -0.75, 10.0};
    std::array<double, 6> solved{};

    {
        spacelab::result<spacelab::srif> opened = spacelab::srif::open();
        CHECK(static_cast<bool>(opened));
        if (!opened) {
            return;
        }
        for (std::size_t i = 0; i < 6; ++i) {
            std::array<double, 6> row{};
            row.at(i) = 1.0;
            CHECK(opened->try_add_row(row, target.at(i), 4.0) == SL_OK);
        }
        const spacelab::result<std::array<double, 6>> x = opened->try_solve();
        CHECK(static_cast<bool>(x));
        if (x) {
            solved = *x;
        }
        const spacelab::result<std::pair<double, std::uint32_t>> stats =
            opened->try_residual_stats();
        CHECK(static_cast<bool>(stats));
        CHECK(stats->second == 6);
    }

    for (std::size_t i = 0; i < 6; ++i) {
        CHECK_NEAR(solved.at(i), target.at(i), 1e-12);
    }
}

/* The new families, through the `try_` half: a design that exists is a value, one that does not is
 * a status, and neither needs an exception. */
void the_new_families_still_generate_what_they_promise() {
    spacelab::lattice_params l = spacelab::default_lattice_params();
    l.orbits = 4;
    l.sats_per_orbit = 6;
    l.config = 3;
    l.semi_major_axis_m = spacelab::re_earth + 20000e3;
    l.inclination_rad = 56.0 * M_PI / 180.0;
    const spacelab::result<std::vector<spacelab::state>> lattice = spacelab::try_lattice_flower(l);
    CHECK(static_cast<bool>(lattice));
    CHECK(lattice->size() == 24);

    spacelab::flower_params f = spacelab::default_flower_params();
    f.petals = 3;
    f.days = 1;
    f.satellites = 6;
    f.eccentricity = 0.2;
    f.inclination_rad = 63.4 * M_PI / 180.0;
    const spacelab::result<std::vector<spacelab::state>> petals = spacelab::try_flower(f);
    CHECK(static_cast<bool>(petals));
    CHECK(petals->size() == 6);

    const spacelab::draim_params d = spacelab::default_draim_params();
    const spacelab::result<spacelab::draim_orbit> orbit = spacelab::try_draim_geometry(d);
    CHECK(static_cast<bool>(orbit));
    const double n = 2.0 * M_PI / d.period_s;
    CHECK_NEAR(orbit->semi_major_axis_m, std::cbrt(spacelab::mu_earth / (n * n)), 1.0);
    const spacelab::result<std::vector<spacelab::state>> four = spacelab::try_draim(d);
    CHECK(static_cast<bool>(four));
    CHECK(four->size() == 4);

    /* A footprint four satellites cannot reach is a status, and the caller branches on it. */
    const spacelab::result<spacelab::walker_params> impossible = spacelab::try_minimum_walker(
        0.05, 60.0 * M_PI / 180.0, spacelab::re_earth + 1200e3, spacelab::mu_earth, 1, 4, 60, 6);
    CHECK(!impossible);
    CHECK(impossible.error() == SL_ERR_DOMAIN);
}

/* Coverage without exceptions: the same agreement between the metrics and the windows. */
void the_coverage_wrapper_still_agrees_with_its_windows() {
    const std::array<double, 2> lats = {0.0, 0.4};
    const std::array<double, 2> lons = {0.0, 1.5};
    const spacelab::result<std::vector<spacelab::state>> cells =
        spacelab::try_grid_cells(lats, lons);
    CHECK(static_cast<bool>(cells));
    CHECK(cells->size() == 4);

    const spacelab::sensor mask = spacelab::elevation_mask(10.0 * M_PI / 180.0);

    const spacelab::result<spacelab::sensor> apparent_mask =
        spacelab::try_elevation_mask_apparent(10.0 * M_PI / 180.0);
    CHECK(static_cast<bool>(apparent_mask));
    CHECK(apparent_mask->min_elevation_rad < mask.min_elevation_rad);
    // And the failure travels through the composed wrapper rather than being swallowed by it.
    CHECK(!spacelab::try_elevation_mask_apparent(-0.1));
    CHECK(spacelab::try_elevation_mask_apparent(-0.1).error() == SL_ERR_DOMAIN);

    // Aberration, and its two refusals, without exceptions: a velocity at the speed of light is a
    // domain error and a zero-length direction is not a direction.
    const spacelab::vec3 station{.x = spacelab::re_earth, .y = 0.0, .z = 0.0};
    const spacelab::vec3 spin{.x = 0.0, .y = 0.0, .z = SL_EARTH_ROTATION_RATE};
    const spacelab::result<spacelab::vec3> carried = spacelab::try_diurnal_velocity(station, spin);
    CHECK(static_cast<bool>(carried));
    CHECK(carried->y > 465.0 && carried->y < 465.2);
    const spacelab::vec3 zenith{.x = 1.0, .y = 0.0, .z = 0.0};
    CHECK(static_cast<bool>(spacelab::try_aberration_apply(*carried, zenith)));
    const spacelab::vec3 superluminal{.x = 0.0, .y = 3.0e8, .z = 0.0};
    CHECK(!spacelab::try_aberration_apply(superluminal, zenith));
    CHECK(spacelab::try_aberration_apply(superluminal, zenith).error() == SL_ERR_DOMAIN);
    const spacelab::vec3 nowhere{.x = 0.0, .y = 0.0, .z = 0.0};
    CHECK(spacelab::try_aberration_remove(*carried, nowhere).error() == SL_ERR_INVALID_ARGUMENT);

    const spacelab::ground_track orbit{.semi_major_axis_m = spacelab::re_earth + 700e3,
                                       .inclination_rad = 51.6 * M_PI / 180.0};
    const double horizon = 86400.0;

    const spacelab::result<std::vector<spacelab::coverage_metrics>> metrics =
        spacelab::try_coverage(orbit, mask, *cells, horizon);
    CHECK(static_cast<bool>(metrics));
    CHECK(metrics->size() == cells->size());

    const spacelab::result<std::vector<spacelab::window>> windows =
        spacelab::try_access_windows(orbit, mask, (*cells)[0], horizon);
    CHECK(static_cast<bool>(windows));
    CHECK(windows->size() == (*metrics)[0].passes);

    const spacelab::result<spacelab::constellation_fom> fom =
        spacelab::try_figures_of_merit(*metrics);
    CHECK(static_cast<bool>(fom));
    CHECK(fom->point_count == metrics->size());

    const spacelab::result<std::vector<spacelab::window>> refused = spacelab::try_access_windows(
        orbit, spacelab::side_looking_sar(0.8, 0.4), (*cells)[0], horizon);
    CHECK(!refused);
    CHECK(refused.error() == SL_ERR_DOMAIN);
}

/* GNSS without exceptions: the two RINEX readers, and the geometry that has no answer. */
void the_gnss_wrapper_still_reads_a_file_and_refuses_a_geometry() {
    const auto weather = spacelab::try_gpt3(0.84, 0.29, 329.0, 180.0);
    CHECK(static_cast<bool>(weather));
    CHECK(weather->pressure > 80e3 && weather->pressure < 110e3);
    CHECK(!spacelab::try_vmf_grid_coefficients("not a grid", 0.84, 0.29));

    const std::string header = "     3.04           RINEX VERSION / TYPE\n";
    const spacelab::result<std::vector<spacelab::gnss_state>> none =
        spacelab::try_rinex_gnss_positions(header, 0.0);
    CHECK(static_cast<bool>(none));
    CHECK(none->empty());

    /* The observation reader answers through the same `result`, and a descriptor the header never
     * declared is a status rather than a throw. */
    constexpr std::string_view observations =
        "     3.05           OBSERVATION DATA    M: MIXED            RINEX VERSION / TYPE\n"
        "G    1 C1C                                                  SYS / # / OBS TYPES\n"
        "END                                                         END OF HEADER\n"
        "> 2024 01 09 12 00  0.0000000  0  1\n"
        "G07  23629347.915 6\n";
    const spacelab::result<spacelab::obs_header> meta =
        spacelab::try_rinex_obs_header(observations);
    CHECK(static_cast<bool>(meta));
    CHECK(meta->file_system == 'M');
    const spacelab::result<std::vector<spacelab::obs_sample>> series =
        spacelab::try_rinex_obs_series(observations, 'G', 7, "C1C");
    CHECK(static_cast<bool>(series));
    CHECK(series->size() == 1);
    CHECK_NEAR((*series)[0].value, 23629347.915, 1e-6);
    const spacelab::result<std::vector<spacelab::obs_sample>> absent =
        spacelab::try_rinex_obs_series(observations, 'G', 7, "L1C");
    CHECK(!absent);
    CHECK(absent.error() == SL_ERR_INVALID_ARGUMENT);

    const double l1 = 1575.42e6;
    CHECK_NEAR(spacelab::ionospheric_path_delay(20.0, l1), 40.3e16 * 20.0 / (l1 * l1), 1e-12);
    CHECK_NEAR(spacelab::glonass_l2_frequency(0), 1246.0e6, 1e-6);

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
    const spacelab::result<spacelab::dop> d =
        spacelab::try_dilution_of_precision(receiver, satellites);
    CHECK(static_cast<bool>(d));
    CHECK_NEAR(d->gdop * d->gdop, (d->pdop * d->pdop) + (d->tdop * d->tdop), 1e-9);

    const spacelab::result<spacelab::dop> too_few = spacelab::try_dilution_of_precision(
        receiver, std::span<const spacelab::vec3>(satellites).first(3));
    CHECK(!too_few);
    CHECK(too_few.error() == SL_ERR_DOMAIN);

    const spacelab::result<spacelab::mapping> zenith =
        spacelab::try_niell_mapping(0.7, 200.0, 90.0, M_PI / 2.0);
    CHECK(static_cast<bool>(zenith));
    CHECK_NEAR(zenith->hydrostatic, 1.0, 1e-2);
}

/* The same half without exceptions: every fallible call is a `result` the caller branches on. */
void the_atmosphere_wrapper_still_answers_without_exceptions() {
    CHECK(spacelab::nrlmsise00_density(172, 29000.0, 200.0, 0.0, 0.0, 150.0, 150.0, 4.0) >
          spacelab::nrlmsise00_density(172, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 4.0));
    CHECK_NEAR(spacelab::molecular_speed_ratio(7600.0, 1000.0, 0.016), 7.4549, 1e-3);

    /* SDTM-1's three total functions need no `result`; its interval does, and here it is the branch
     * rather than the throw. */
    const std::array<double, 4> quiet_ap{5.0, 5.0, 5.0, 5.0};
    const double quiet_storm = spacelab::sdtm1_storm_response(quiet_ap);
    CHECK(quiet_storm > 0.0 && quiet_storm < 5.0);
    const double sdtm_400 =
        spacelab::sdtm1_density(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0);
    CHECK(spacelab::sdtm1_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0,
                                                 5.0) > 586.0);
    const spacelab::result<std::array<double, SL_SDTM1_INTERVAL>> band =
        spacelab::try_sdtm1_density_interval(sdtm_400, 150.0, 150.0, 5.0, 1.6448536269514727);
    CHECK(static_cast<bool>(band));
    CHECK((*band)[0] < sdtm_400);
    CHECK((*band)[1] > sdtm_400);

    const double t = 20.0 * 365.25 * 86400.0;
    const spacelab::result<std::array<double, 2>> now = spacelab::try_solid_pole_tide(0.0, 1e-7, t);
    const spacelab::result<std::array<double, 2>> retired =
        spacelab::try_solid_pole_tide(0.0, 1e-7, t, SL_MEAN_POLE_IERS2010);
    CHECK(static_cast<bool>(now));
    CHECK(static_cast<bool>(retired));
    CHECK(std::abs((*retired)[0] - (*now)[0]) > 0.1 * std::abs((*now)[0]));

    const spacelab::result<spacelab::zonal> mean =
        spacelab::try_conventional_zonal_coefficients(25.0);
    const spacelab::result<spacelab::zonal> secular =
        spacelab::try_time_variable_zonal_correction(25.0);
    CHECK(static_cast<bool>(mean));
    CHECK(static_cast<bool>(secular));
    CHECK_NEAR(mean->c20 - secular->c20, -0.48416948e-3, 1e-15);

    CHECK(!spacelab::try_sentman_drag_coefficients(7.4549, 1.0, 1.0, 0.0, 1000.0));
    CHECK(!spacelab::try_swath_geometry(spacelab::re_earth, 700e3, 1.4));

    const std::string_view csv = "DATE,BSRN,ND,KP1,KP2\n";
    const spacelab::result<std::vector<spacelab::space_weather_day>> days =
        spacelab::try_space_weather_days(csv);
    CHECK(static_cast<bool>(days));
    CHECK(days->empty());
    CHECK(!spacelab::try_space_weather_ap_array(csv, 2020, 1, 1, 0.0));

    const std::array<spacelab::density_sample, 2> samples{
        spacelab::density_sample{
            .latitude_rad = -1.5, .local_solar_time_h = 1.0, .log_ratio = 0.25, .weight = 1.0},
        spacelab::density_sample{
            .latitude_rad = 1.5, .local_solar_time_h = 23.0, .log_ratio = -0.5, .weight = 1.0}};
    const spacelab::result<std::vector<double>> grid =
        spacelab::try_density_grid_snapshot(samples, 2, 2);
    CHECK(static_cast<bool>(grid));
    CHECK(grid->size() == 4);
    CHECK_NEAR((*grid)[0], 0.25, 1e-15);

    const double speed = std::sqrt(spacelab::mu_earth / 7.0e6);
    const spacelab::brouwer_lyddane_params bl{
        .initial = {.position = {.x = 7.0e6, .y = 0.0, .z = 0.0},
                    .velocity = {.x = 0.0, .y = speed, .z = 0.0}},
        .mu = -1.0,
        .earth_radius_m = spacelab::re_earth,
        .c20 = -1.08262668e-3,
        .c30 = 0.0,
        .c40 = 0.0,
        .c50 = 0.0,
        .m2 = 0.0};
    const std::array<double, 1> times{0.0};
    CHECK(!spacelab::try_brouwer_lyddane_propagate(bl, times));
}

/* The same factories without exceptions: `try_at` rather than `at`, and every failure a status. */
void every_force_model_is_still_a_factory_without_exceptions() {
    const double speed = std::sqrt(spacelab::mu_earth / 7.0e6);
    const double arc = 600.0;
    const spacelab::state s0{.position = {.x = 7.0e6, .y = 0.0, .z = 0.0},
                             .velocity = {.x = 0.0, .y = speed, .z = 0.0}};
    const spacelab::eop orientation{.epoch_tt_ns = 725760069184000000LL,
                                    .epoch_ut1_ns = 725760000000000000LL,
                                    .xp_rad = 1.0e-6,
                                    .yp_rad = 2.0e-6,
                                    .dx_rad = 0.0,
                                    .dy_rad = 0.0};

    const spacelab::result<spacelab::ephemeris> itrf =
        spacelab::ephemeris::geopotential_itrf(s0, arc, 4, orientation);
    CHECK(static_cast<bool>(itrf));
    const spacelab::result<spacelab::state> at_end = itrf->try_at(arc);
    CHECK(static_cast<bool>(at_end));
    CHECK(at_end->position.x != s0.position.x);

    CHECK(static_cast<bool>(
        spacelab::ephemeris::relativistic(s0, arc, 0.0, spacelab::mu_earth, 1.32712440018e20)));
    CHECK(!spacelab::ephemeris::j2_dense(s0, 0.0, spacelab::mu_earth));
    CHECK(!spacelab::ephemeris::geopotential_itrf(s0, arc, 4, orientation, 0.0));

    const spacelab::result<spacelab::state> extended = spacelab::try_propagate_j2_precision(
        s0, spacelab::mu_earth, 1.08262668e-3, spacelab::re_earth, 1.0, 100,
        SL_PRECISION_DOUBLE_DOUBLE);
    CHECK(static_cast<bool>(extended));
    /* Casting past the enum is the point: that the tag is refused rather than defaulted is the
     * property. NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) */
    CHECK(!spacelab::try_propagate_j2_precision(s0, spacelab::mu_earth, 0.0, spacelab::re_earth,
                                                1.0, 10, static_cast<spacelab::precision>(7)));
}

/*
 * The radar surface without exceptions: every answer arrives as a `result`, including the two
 * refusals -- a mismatched pair and a profile with no main lobe -- that the throwing suite reaches
 * by catching.
 */
void the_radar_wrapper_still_sizes_its_own_answers() {
    const auto chirp = spacelab::try_lfm_chirp(32, 5.0e11, 3.0e7);
    CHECK(static_cast<bool>(chirp));
    CHECK(chirp->size() == 32);
    for (const spacelab::complex &c : *chirp) {
        CHECK_NEAR(std::hypot(c.re, c.im), 1.0, 1e-12);
    }

    const auto compressed = spacelab::try_matched_filter(*chirp, *chirp);
    CHECK(static_cast<bool>(compressed));
    double peak = 0.0;
    for (const spacelab::complex &c : *compressed) {
        peak = std::max(peak, std::hypot(c.re, c.im));
    }
    CHECK_NEAR(peak, 32.0, 1e-6 * 32.0);

    const std::vector<spacelab::complex> shorter(chirp->begin(), chirp->begin() + 8);
    CHECK(!spacelab::try_interferogram(*chirp, shorter));

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
    const auto charges = spacelab::try_residues(wrapped, rows, cols);
    CHECK(static_cast<bool>(charges));
    CHECK(charges->size() == (rows - 1) * (cols - 1));
    CHECK(std::ranges::all_of(*charges, [](std::int8_t q) { return q == 0; }));

    const auto recovered = spacelab::try_unwrap_phase_2d(wrapped, rows, cols);
    CHECK(static_cast<bool>(recovered));
    const double offset = (*recovered)[0] - truth[0];
    for (std::size_t k = 0; k < recovered->size(); ++k) {
        CHECK_NEAR((*recovered)[k] - truth[k], offset, 1e-9);
    }

    const spacelab::rda_geometry geometry{.prf = 1700.0,
                                          .velocity = 7100.0,
                                          .wavelength = 0.055,
                                          .r0_slant = 850000.0,
                                          .range_spacing = 4.5};
    CHECK_NEAR(spacelab::azimuth_fm_rate(geometry),
               2.0 * geometry.velocity * geometry.velocity /
                   (geometry.wavelength * geometry.r0_slant),
               1e-3);
    CHECK(!spacelab::try_src_fm_rate(0.0, geometry));
    CHECK(static_cast<bool>(spacelab::try_src_fm_rate(400.0, geometry)));

    const std::vector<double> flat(16, 1.0);
    CHECK(!spacelab::try_resolution_3db_samples(flat));
}

/*
 * The optical surface without exceptions: the degenerate rig and the mismatched pair arrive as
 * empty results rather than as throws, which is the only way this translation unit could see them.
 */
void the_optical_wrapper_still_triangulates_and_suppresses() {
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

    const auto point = spacelab::try_triangulate(cameras, pixels);
    CHECK(static_cast<bool>(point));
    CHECK_NEAR(point->z, world[2], 1e-9);

    /* Two copies of one camera see one ray; there is no point where a ray meets itself. */
    const std::vector<spacelab::camera> twice{cameras[0], cameras[0]};
    const std::vector<double> same{pixels[0], pixels[1], pixels[0], pixels[1]};
    CHECK(!spacelab::try_triangulate(twice, same));
    CHECK(!spacelab::try_triangulate(cameras, std::span(pixels).first(3)));

    const std::vector<spacelab::bbox> boxes{{.x1 = 0.0, .y1 = 0.0, .x2 = 2.0, .y2 = 2.0},
                                            {.x1 = 1.0, .y1 = 0.0, .x2 = 3.0, .y2 = 2.0},
                                            {.x1 = 10.0, .y1 = 10.0, .x2 = 12.0, .y2 = 12.0}};
    const std::vector<double> scores{0.9, 0.8, 0.7};
    const auto kept = spacelab::try_nms(boxes, scores, 0.3);
    CHECK(static_cast<bool>(kept));
    CHECK(*kept == std::vector<std::size_t>({0, 2}));
    CHECK(!spacelab::try_nms(boxes, std::span(scores).first(2), 0.3));
    CHECK_NEAR(spacelab::box_iou(boxes[0], boxes[0]), 1.0, 1e-15);

    const std::vector<double> scales{4.0, 8.0};
    const std::vector<double> ratios{0.5, 1.0, 2.0};
    const auto anchors = spacelab::try_generate_anchors(2, 2, 16.0, scales, ratios);
    CHECK(static_cast<bool>(anchors));
    CHECK(anchors->size() == std::size_t{2} * 2 * scales.size() * ratios.size());

    const spacelab::atmospheric_coupling coupling{.path_reflectance = 0.08,
                                                  .transmittance_down = 0.83,
                                                  .transmittance_up = 0.87,
                                                  .spherical_albedo = 0.11};
    CHECK_NEAR(spacelab::surface_from_toa(coupling, spacelab::toa_from_surface(coupling, 0.42)),
               0.42, 1e-12);

    const std::vector<double> ms{0.20, 0.31, 0.15, 0.35, 0.22, 0.51, 0.11, 0.47, 0.26};
    const std::vector<double> pan{0.55, 0.61, 0.48};
    const auto brovey = spacelab::try_brovey_pan_sharpen(ms, 3, pan);
    CHECK(static_cast<bool>(brovey));
    for (std::size_t p = 0; p < pan.size(); ++p) {
        CHECK_NEAR((*brovey)[p] + (*brovey)[3 + p] + (*brovey)[6 + p], pan[p], 1e-12);
    }
}

/* The masker is a factory here too: without exceptions there is no constructor that could report a
 * missing `onnx` build, so `open` is the only form and it hands back a status instead. */
void the_cloud_masker_is_a_factory_without_exceptions() {
    const std::array<std::uint8_t, 4> model{0, 0, 0, 0};
    const spacelab::result<spacelab::cloud_masker> opened =
        spacelab::cloud_masker::open(model, 0.5f);
    CHECK(opened.has_value() || opened.error() != SL_OK);
}

/*
 * The two callback scans without exceptions. Their trampolines carry an `std::exception_ptr` under
 * `__cpp_exceptions` and nothing at all without it, so this is the build where the branch that
 * parks a thrown exception is compiled out entirely -- and a sampler that cannot answer has only
 * one way left to say so, which is the `false` it returns.
 */
void the_scans_still_take_a_lambda_without_exceptions() {
    const spacelab::conjunction geometry{
        .relative_position = {.x = 30.0, .y = 0.0, .z = 40.0},
        .relative_velocity = {.x = 0.0, .y = 14000.0, .z = 0.0},
        .primary_covariance = {1e4, 0.0, 0.0, 0.0, 1e4, 0.0, 0.0, 0.0, 1e4},
        .secondary_covariance = {1e4, 0.0, 0.0, 0.0, 1e4, 0.0, 0.0, 0.0, 1e4},
        .primary_radius = 5.0,
        .secondary_radius = 5.0};

    const spacelab::result<spacelab::probability_estimates> estimates =
        spacelab::try_collision_probability(geometry);
    CHECK(estimates.has_value());
    CHECK(estimates->maximum >= estimates->foster);

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
    const spacelab::result<spacelab::usage_violations> clear =
        spacelab::try_conjunction_usage_violations(geometry, 5400.0, rectilinear);
    CHECK(clear.has_value());
    CHECK(!clear->any);
    CHECK_NEAR(clear->correction_factor, 1.0, 1e-6);

    /* A sampler that cannot answer returns `false`, and that arrives as a status the caller can
     * branch on rather than as a scan that quietly found nothing. */
    const auto never = [](double, ::sl_relative_sample &) { return false; };
    const spacelab::result<spacelab::usage_violations> blind =
        spacelab::try_conjunction_usage_violations(geometry, 5400.0, never);
    CHECK(!blind.has_value());
    CHECK(blind.error() == SL_ERR_COMPUTE);

    const double rate = 2.0 * std::numbers::pi / 600.0;
    const auto drifting = [rate](double t, ::sl_state &out) {
        out.position = {.x = 10000.0 * std::sin(rate * t), .y = 500.0, .z = 0.0};
        out.velocity = {.x = 10000.0 * rate * std::cos(rate * t), .y = 0.0, .z = 0.0};
        return true;
    };
    /* Both halves of the two-call protocol still run inside the wrapper, so the vector arrives the
     * right length without the caller ever seeing a capacity. */
    const spacelab::result<std::vector<spacelab::close_approach>> found =
        spacelab::try_closest_approaches(drifting, 10.0, 1000.0, 30.0, 1e-4);
    CHECK(found.has_value());
    CHECK(found->size() == 3);
    CHECK_NEAR((*found)[0].distance, 500.0, 1e-3);
    const auto refusing = [](double, ::sl_state &) { return false; };
    CHECK(!spacelab::try_closest_approaches(refusing, 10.0, 1000.0, 30.0, 1e-4).has_value());
    CHECK(spacelab::radially_separated(SL_RE_EARTH + 400e3, SL_RE_EARTH + 420e3,
                                       SL_RE_EARTH + 800e3, SL_RE_EARTH + 900e3, 5000.0));
}

/* The observables, the decay scan and the carrier combinations, each through its `try_` half. */
void the_new_observables_still_report_a_refusal_as_a_status() {
    const double exciter = 7.15e9;
    const std::array<spacelab::ramp_segment, 1> table{
        spacelab::ramp_segment{.start = 0.0, .end = 1e4, .start_frequency = exciter, .rate = 0.0}};
    const spacelab::doppler_link link{.uplink = SL_BAND_X,
                                      .downlink = SL_BAND_X,
                                      .reference_frequency = exciter,
                                      .reference_band = SL_BAND_X,
                                      .count_interval = 60.0,
                                      .subtracted = true};
    const spacelab::result<double> counted =
        spacelab::try_dsn_averaged_doppler(table, link, 100.0, 160.0);
    const spacelab::result<double> instantaneous =
        spacelab::try_dsn_measured_frequency(table, SL_BAND_X, SL_BAND_X, 100.0, 1.0);
    CHECK(counted.has_value() && instantaneous.has_value());
    CHECK_NEAR(*counted + *instantaneous, (880.0 / 749.0) * exciter, 1e-2);
    /* An interval the table does not cover is out of the observable's domain, and that status is
     * the only way this build can be told so. */
    CHECK(spacelab::try_dsn_sequential_range(table, SL_BAND_X, 1e5, 1e5 + 1.0, 26).error() ==
          SL_ERR_DOMAIN);

    spacelab::decay_params p = spacelab::default_decay_params();
    p.area_over_mass = 0.02;
    p.reference_density = 2e-11;
    p.reference_radius = SL_RE_EARTH + 300e3;
    p.scale_height = 60e3;
    p.horizon = 400.0 * 86400.0;
    p.tolerance = 3600.0;
    const std::array<double, 3> zonals{0.0, 0.0, -SL_J2_EARTH};
    const std::array<double, 6> initial{SL_RE_EARTH + 300e3, 0.0, 0.0, 0.0, 0.0, 0.0};
    const spacelab::result<spacelab::lifetime> low =
        spacelab::try_orbital_lifetime(p, initial, zonals);
    CHECK(low.has_value());
    CHECK(low->outcome == SL_LIFETIME_REENTRY);
    const std::array<double, 2> too_few{0.0, 0.0};
    CHECK(spacelab::try_orbital_lifetime(p, initial, too_few).error() == SL_ERR_INVALID_ARGUMENT);

    const spacelab::utc_datetime epoch{
        .year = 2008, .month = 9, .day = 20, .hour = 12, .minute = 25, .second = 12.25};
    const std::array<double, 6> parameters{15.72125391, 0.0006703, 51.6416,
                                           247.4627,    130.5360,  325.0288};
    const spacelab::result<spacelab::tle_lines> lines =
        spacelab::try_tle_lines(epoch, 25544, parameters, -1.1606e-4, "1998-067A", 292, 56353);
    CHECK(lines.has_value());
    CHECK(lines->line1.size() == 69 && lines->line2.size() == 69);
    /* Five numeric columns cannot name this satellite, and a status is how the caller finds out. */
    CHECK(spacelab::try_tle_lines(epoch, 270123, parameters).error() == SL_ERR_INVALID_ARGUMENT);

    const double f1 = 1575.42e6;
    const double f2 = 1227.60e6;
    const spacelab::result<spacelab::lanes> lanes = spacelab::try_lane_wavelengths(f1, f2);
    CHECK(lanes.has_value());
    CHECK_NEAR(lanes->wide_lane, 0.8619, 1e-4);
    CHECK(spacelab::try_lane_wavelengths(f1, f1).error() == SL_ERR_INVALID_ARGUMENT);

    const double geometric = 22100000.0;
    const double i1 = 4.5;
    const double i2 = i1 * (f1 / f2) * (f1 / f2);
    const spacelab::result<double> free_of_ionosphere =
        spacelab::try_ionosphere_free(f1, f2, geometric + i1, geometric + i2);
    CHECK(free_of_ionosphere.has_value());
    CHECK_NEAR(*free_of_ionosphere, geometric, 1e-6);
    CHECK_NEAR(spacelab::geometry_free(geometric + i1, geometric + i2), i1 - i2, 1e-8);

    std::vector<double> times;
    std::vector<double> values;
    for (int i = 0; i < 40; ++i) {
        times.push_back(30.0 * static_cast<double>(i));
        values.push_back((0.02 * static_cast<double>(i)) + (i >= 20 ? 0.30 : 0.0));
    }
    const spacelab::result<std::vector<spacelab::slip_verdict>> verdicts =
        spacelab::try_detect_cycle_slips_geometry_free(times, values, 0.10, 120.0);
    CHECK(verdicts.has_value());
    CHECK(verdicts->size() == times.size());
    CHECK((*verdicts)[20].verdict == SL_SLIP_SLIPPED);

    const std::array<double, 3> flt{5.45, 3.10, 2.97};
    const std::array<double, 9> covariance{6.290, 5.978, 0.544, 5.978, 6.292,
                                           2.340, 0.544, 2.340, 6.288};
    const spacelab::result<std::vector<spacelab::candidate>> best =
        spacelab::try_resolve_ambiguities(flt, covariance, 2);
    CHECK(best.has_value());
    CHECK(best->size() == 2);
    CHECK((*best)[0].squared_norm <= (*best)[1].squared_norm);
    const std::array<double, 4> mismatched{1.0, 0.0, 0.0, 1.0};
    CHECK(spacelab::try_ambiguity_decorrelation(flt, mismatched).error() ==
          SL_ERR_INVALID_ARGUMENT);
}

/*
 * The control surface through the `try_` layer only.
 *
 * Two claims that the throwing suite cannot make: that every control refusal is reachable as a
 * status a `-fno-exceptions` consumer can branch on, and that the sizing wrappers -- `loop_run`,
 * `laplacian_spectrum`, `consensus_run`, `delay_breakpoints` -- still allocate correctly when the
 * failure path cannot throw. The three walls this ABI adds over the crate are checked here rather
 * than in `smoke.cpp`, because a wall whose only expression is an exception is a wall the consumers
 * that matter most to it cannot see.
 */
void the_control_surface_reports_every_wall_as_a_status() {
    const std::array<double, 9> identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const std::array<double, 9> inertia{4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
    const spacelab::latency latency{0.2, 0.1, 0.0};
    const spacelab::latency backwards{-0.1, 0.0, 0.0};
    const spacelab::latency far_behind{60.0, 0.0, 0.0};

    const spacelab::result<double> total = spacelab::try_latency_total(latency);
    CHECK(total.has_value());
    CHECK_NEAR(*total, 0.3, 1e-15);
    CHECK(spacelab::try_latency_total(backwards).error() == SL_ERR_INVALID_ARGUMENT);
    /* A delay of 3.75 steps is a different problem from 3 or 4, and this is where a consumer that
     * cannot catch learns which one it has. */
    CHECK(spacelab::try_latency_steps(latency, 0.08).error() == SL_ERR_DOMAIN);
    const spacelab::result<std::size_t> lag = spacelab::try_latency_steps(latency, 0.1);
    CHECK(lag.has_value() && *lag == 3);

    /* Wall one: the 512-sample loop history. A lag of 600 steps does not fit it, and wrapping the
     * ring would silently read a state from a previous era. */
    spacelab::attitude_state initial{};
    std::copy(identity.begin(), identity.end(), std::begin(initial.rotation));
    initial.rate = spacelab::vec3{0.0, 0.0, 0.1};
    spacelab::attitude_state seed = initial;
    std::copy(identity.begin(), identity.end(), std::begin(seed.rotation));
    seed.rate = spacelab::vec3{0.0, 0.0, 0.0};
    CHECK(spacelab::try_attitude_loop_run(SL_LAW_MRP_LYAPUNOV, 8.0, 12.0, far_behind, 0.1, initial,
                                          seed, identity, inertia, 40)
              .error() == SL_ERR_DOMAIN);
    const spacelab::result<spacelab::loop_run> run = spacelab::try_attitude_loop_run(
        SL_LAW_MRP_LYAPUNOV, 8.0, 12.0, latency, 0.1, initial, seed, identity, inertia, 40);
    CHECK(run.has_value());
    CHECK(run->samples.size() == 40);
    /* The body starts spinning and the law is asked to stop it, so the run ends slower than it
     * began -- the loop is closed, not merely wired. */
    CHECK(std::abs(run->final_state.rate.z) < 0.1);

    /* Wall two: the four instantiated LQR shapes. The wrapper checks the span lengths itself and
     * refuses before the call, which is why these four-element arrays are safe to offer as a 5x2
     * plant. */
    const std::array<double, 4> plant_a{0.0, 1.0, 0.0, 0.0};
    const std::array<double, 2> plant_b{0.0, 1.0};
    const std::array<double, 4> weight_q{1.0, 0.0, 0.0, 1.0};
    const std::array<double, 1> weight_r{1.0};
    CHECK(spacelab::try_lqr(plant_a, plant_b, weight_q, weight_r, 5, 2).error() ==
          SL_ERR_INVALID_ARGUMENT);
    CHECK(spacelab::try_lqr(plant_a, plant_a, weight_q, weight_r, 2, 1).error() ==
          SL_ERR_INVALID_ARGUMENT);
    const spacelab::result<spacelab::lqr_solution> solved =
        spacelab::try_lqr(plant_a, plant_b, weight_q, weight_r, 2, 1);
    CHECK(solved.has_value());
    CHECK(solved->residual < 1e-9);

    /* Wall three: the agent count. Two to twelve is what this ABI is monomorphised for, and one
     * past either end is a status rather than a reshaped answer. */
    const std::array<double, 16> path{0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                                      0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0};
    CHECK(spacelab::try_laplacian_spectrum(path, 13).error() == SL_ERR_INVALID_ARGUMENT);
    CHECK(spacelab::try_laplacian_spectrum(path, 1).error() == SL_ERR_INVALID_ARGUMENT);
    /* And the wrapper's own span check, which fires before the ABI ever sees the call. */
    CHECK(spacelab::try_laplacian_spectrum(path, 3).error() == SL_ERR_INVALID_ARGUMENT);
    const spacelab::result<std::vector<double>> spectrum =
        spacelab::try_laplacian_spectrum(path, 4);
    CHECK(spectrum.has_value() && spectrum->size() == 4);

    const spacelab::result<double> bound =
        spacelab::try_consensus_delay_bound(path, 4, SL_CONSENSUS_DELAYED_RELATIVE);
    CHECK(bound.has_value());
    const std::array<double, 4> start{1.0, -2.0, 3.0, -0.5};
    /* A run that ends before the first message arrives is a refusal, not an empty answer -- and a
     * domain error rather than a malformed argument, because every number in it is individually
     * admissible and only their relation is not. */
    CHECK(spacelab::try_consensus_run(path, SL_CONSENSUS_DELAYED_RELATIVE, 0.7 * *bound, start,
                                      0.1 * *bound, 1e-9)
              .error() == SL_ERR_DOMAIN);
    const spacelab::result<spacelab::consensus_result> settled = spacelab::try_consensus_run(
        path, SL_CONSENSUS_DELAYED_RELATIVE, 0.7 * *bound, start, 40.0, 1e-9);
    CHECK(settled.has_value());
    CHECK(settled->final_state.size() == 4);
    CHECK(settled->accepted > 0);

    /* The modes wrapper checks its own span before the call, the same way `try_lqr` does. */
    CHECK(spacelab::try_laplacian_modes(path, 3).error() == SL_ERR_INVALID_ARGUMENT);
    const spacelab::result<spacelab::laplacian_modes> eigen =
        spacelab::try_laplacian_modes(path, 4);
    CHECK(eigen.has_value());
    CHECK(eigen->values.size() == 4 && eigen->modes.size() == 16);
    CHECK(eigen->values == *spectrum);

    /* Connectivity is the one answer in this section that is a value rather than a status, and a
     * consumer that cannot catch needs it to stay one: cutting a link of the path graph gives
     * `false`, while an asymmetric matrix is still `SL_ERR_DOMAIN`. */
    const spacelab::result<bool> whole = spacelab::try_graph_is_connected(path, 4);
    CHECK(whole.has_value() && *whole);
    std::array<double, 16> split = path;
    split[(4 * 1) + 2] = 0.0;
    split[(4 * 2) + 1] = 0.0;
    const spacelab::result<bool> cut = spacelab::try_graph_is_connected(split, 4);
    CHECK(cut.has_value() && !*cut);
    split[(4 * 1) + 2] = 1.0;
    CHECK(spacelab::try_graph_is_connected(split, 4).error() == SL_ERR_DOMAIN);

    /* The matrix-order wall, refused for its own reason and reported as its own status. */
    const std::array<double, 9> companion{0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -6.0, -11.0, -6.0};
    CHECK(spacelab::try_characteristic_polynomial(companion, 2).error() == SL_ERR_INVALID_ARGUMENT);
    const spacelab::result<std::vector<double>> coefficients =
        spacelab::try_characteristic_polynomial(companion, 3);
    CHECK(coefficients.has_value());
    CHECK_NEAR((*coefficients)[0], 6.0, 1e-12);

    /* A span that runs backwards has no mesh, and the sizing wrapper reports that rather than
     * allocating nothing and calling it an answer. */
    const std::array<double, 2> delays{0.25, 0.4};
    CHECK(spacelab::try_delay_breakpoints(delays, 1.0, 0.0, 2).error() == SL_ERR_DOMAIN);
    const spacelab::result<std::vector<double>> mesh =
        spacelab::try_delay_breakpoints(delays, 0.0, 1.0, 2);
    CHECK(mesh.has_value() && !mesh->empty());
    CHECK(std::is_sorted(mesh->begin(), mesh->end()));

    /* The station-keeping law's every-weight-zero case, which is the one a caller reaches by
     * forgetting to fill the aggregate at all. */
    const spacelab::element_feedback empty{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1e-6, 1e-4};
    const std::array<double, 6> drift{100.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK(spacelab::try_element_feedback_lyapunov(empty, drift).error() == SL_ERR_DOMAIN);
    const spacelab::element_feedback law{{1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1e-6, 1e-4};
    const spacelab::result<double> lyapunov = spacelab::try_element_feedback_lyapunov(law, drift);
    CHECK(lyapunov.has_value());
    CHECK_NEAR(*lyapunov, 0.5 * 100.0 * 100.0, 1e-9);

    /* An unknown law tag is a status here too, and it is the enum-width case as well: a narrowed
     * enum would not carry 7 across the boundary to be refused. */
    CHECK(
        spacelab::try_attitude_small_angle_gains(static_cast<spacelab::attitude_law>(7), 8.0, 12.0)
            .error() == SL_ERR_INVALID_ARGUMENT);
    const spacelab::result<spacelab::small_angle_gains> gains =
        spacelab::try_attitude_small_angle_gains(SL_LAW_QUATERNION_PD, 8.0, 12.0);
    CHECK(gains.has_value());
    const spacelab::result<spacelab::delay_margin> margin =
        spacelab::try_delay_margin_of_gains(4.0, gains->angle, gains->rate);
    CHECK(margin.has_value());
    CHECK_NEAR(margin->normalised, margin->seconds * margin->natural_frequency_rad_s, 1e-12);
    /* A body cannot have a zero principal moment, and the refusal names the domain rather than
     * the argument. */
    CHECK(spacelab::try_worst_principal_axis(SL_LAW_QUATERNION_PD, 8.0, 12.0,
                                             spacelab::vec3{1.0, 0.0, 1.0})
              .error() == SL_ERR_DOMAIN);
}

} // namespace

int main() {
    a_result_carries_either_a_value_or_a_status();
    the_handle_still_owns_itself_without_exceptions();
    a_failure_is_a_status_the_caller_can_branch_on();
    the_generators_still_size_their_own_buffers();
    the_ephemeris_is_a_factory_here_too();
    the_dsst_wrapper_still_sizes_its_own_answer();
    the_least_squares_wrapper_takes_a_lambda_without_exceptions();
    the_information_filter_still_owns_itself();
    the_new_families_still_generate_what_they_promise();
    the_coverage_wrapper_still_agrees_with_its_windows();
    the_gnss_wrapper_still_reads_a_file_and_refuses_a_geometry();
    the_atmosphere_wrapper_still_answers_without_exceptions();
    every_force_model_is_still_a_factory_without_exceptions();
    the_radar_wrapper_still_sizes_its_own_answers();
    the_optical_wrapper_still_triangulates_and_suppresses();
    the_scans_still_take_a_lambda_without_exceptions();
    the_new_observables_still_report_a_refusal_as_a_status();
    the_cloud_masker_is_a_factory_without_exceptions();
    the_control_surface_reports_every_wall_as_a_status();

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("spacelab C++ no-exceptions smoke: all checks passed");
    return EXIT_SUCCESS;
}
