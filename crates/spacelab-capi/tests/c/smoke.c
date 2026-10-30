/*
 * The C side of the boundary, exercised as a C consumer sees it.
 *
 * `spacelab.h` is the contract and the C++ header is one reading of it; this is the other. It is
 * not a translation of `smoke.cpp` -- it asserts what only a plain C consumer reaches:
 *
 *   - that the header parses as C, not merely as C++ (a `bool` parameter, an anonymous struct
 *     typedef and an enum with an out-of-range sentinel all mean different things to the two
 *     front-ends, and this repository ships the header to both);
 *   - the two-call length protocol written out by hand, including the short-buffer branch, which
 *     the C++ wrapper sizes its way past and therefore never takes;
 *   - the totality rule -- a value-returning function handed a null answers NaN rather than
 *     trapping -- which the C++ wrapper marks `noexcept` and never tests;
 *   - the enum widths at run time, where C99 has no `static_assert` to check them at compile time.
 *
 * Written to C99 so that the whole ladder from C99 to the newest standard compiles it; the build
 * runs it at both ends of that ladder. Nothing here is POSIX: `M_PI` is not in the C standard, so
 * this file carries its own pi rather than quietly depending on a glibc extension.
 */

#include "spacelab.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const double PI = 3.14159265358979323846;

static int failures = 0;

static void check(int condition, const char *what, int line) {
    if (!condition) {
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, line, what);
        ++failures;
    }
}

static void check_near(double got, double want, double tolerance, const char *what, int line) {
    if (!(fabs(got - want) <= tolerance)) {
        fprintf(stderr, "FAIL %s:%d: %s -- got %.17g, want %.17g +- %.3g\n", __FILE__, line, what,
                got, want, tolerance);
        ++failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_NEAR(got, want, tol) check_near((got), (want), (tol), #got, __LINE__)

/* The element set the Rust and C++ suites use, so all three answer the same question. */
static const char *const LINE1 =
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
static const char *const LINE2 =
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

static double radius_of(sl_vec3 r) {
    return sqrt((r.x * r.x) + (r.y * r.y) + (r.z * r.z));
}

/*
 * Widths and offsets. The C++ smoke test static-asserts these; C99 cannot, and the C consumer is
 * the one who suffers if they are wrong, so it checks them at run time instead. The enum sizes are
 * the `-fshort-enums` trap: without the `_FORCE_INT32` enumerators these read 1 under that flag,
 * and the build compiles this file both ways.
 */
static void the_layout_is_the_one_the_library_assumes(void) {
    CHECK(sizeof(sl_vec3) == 3 * sizeof(double));
    CHECK(sizeof(sl_state) == 6 * sizeof(double));
    CHECK(offsetof(sl_state, position) == 0);
    CHECK(offsetof(sl_state, velocity) == 3 * sizeof(double));
    CHECK(sizeof(sl_geodetic) == 3 * sizeof(double));
    CHECK(sizeof(sl_latlon) == 2 * sizeof(double));

    CHECK(sizeof(sl_precision) == 4);
    CHECK(sizeof(sl_walker_pattern) == 4);
    CHECK(sizeof(sl_sensor_kind) == 4);
    CHECK(sizeof(sl_band) == 4);
    CHECK(sizeof(sl_lifetime_outcome) == 4);
    CHECK(sizeof(sl_slip) == 4);
    CHECK(sizeof(sl_attitude_law) == 4);
    CHECK(sizeof(sl_consensus_protocol) == 4);
    CHECK(offsetof(sl_walker_params, inclination_rad) == sizeof(double));

    CHECK(sizeof(sl_latency) == 3 * sizeof(double));
    CHECK(sizeof(sl_attitude_state) == 12 * sizeof(double));
    CHECK(offsetof(sl_attitude_state, rate) == 9 * sizeof(double));
    CHECK(sizeof(sl_loop_sample) == 5 * sizeof(double));
    CHECK(offsetof(sl_loop_sample, torque) == 2 * sizeof(double));
    CHECK(sizeof(sl_delay_margin) == 5 * sizeof(double));
    CHECK(sizeof(sl_element_feedback) == 8 * sizeof(double));
    CHECK(offsetof(sl_element_feedback, proportional) == 6 * sizeof(double));

    CHECK(sl_abi_version() == SL_ABI_VERSION);
}

/* The version string's own two-call protocol: ask the length, then fill. */
static void the_version_string_sizes_itself(void) {
    size_t needed = sl_version_string(NULL, 0);
    char *buf = NULL;
    CHECK(needed > 1);

    buf = (char *)malloc(needed);
    if (buf == NULL) {
        CHECK(0);
        return;
    }
    CHECK(sl_version_string(buf, needed) == needed);
    CHECK(strlen(buf) == needed - 1);
    free(buf);
}

/* Geodesy, against the same measured z the Rust and C++ tests pin. Radians, all three. */
static void the_geodetic_round_trip_agrees_with_the_other_two_suites(void) {
    sl_geodetic g;
    sl_geodetic back;
    sl_vec3 r;

    g.latitude_rad = 35.0 * PI / 180.0;
    g.longitude_rad = 139.0 * PI / 180.0;
    g.height_m = 42.0;

    CHECK(sl_geodetic_to_itrf(&g, &r) == SL_OK);
    CHECK_NEAR(r.z, 3637891.0, 1.0);

    CHECK(sl_itrf_to_geodetic(&r, &back) == SL_OK);
    CHECK_NEAR(back.latitude_rad, g.latitude_rad, 1e-12);
    CHECK_NEAR(back.longitude_rad, g.longitude_rad, 1e-12);
    CHECK_NEAR(back.height_m, g.height_m, 1e-6);
}

/* Open, propagate one, propagate many, close -- the discipline the C++ class exists to automate. */
static void the_handle_opens_propagates_and_closes(void) {
    const double offsets[4] = {0.0, 45.0, 90.0, -30.0};
    sl_state batch[4];
    sl_sgp4 *sat = NULL;
    sl_state one;

    CHECK(sl_sgp4_open(LINE1, LINE2, &sat) == SL_OK);
    if (sat == NULL) {
        CHECK(0);
        return;
    }

    CHECK(sl_sgp4_propagate_batch(sat, offsets, 4, batch) == SL_OK);
    for (size_t i = 0; i < 4; ++i) {
        CHECK(sl_sgp4_propagate_at(sat, offsets[i], &one) == SL_OK);
        CHECK(one.position.x == batch[i].position.x);
        CHECK(one.velocity.z == batch[i].velocity.z);
    }

    /* The ISS is where the ISS is: 6 720 189.23 m from the centre at the element set's epoch. */
    CHECK_NEAR(radius_of(batch[0].position), 6720189.23, 0.01);

    sl_sgp4_close(sat);

    /* Closing a null is the documented no-op, so a cleanup path does not need a branch. */
    sl_sgp4_close(NULL);
}

/*
 * A failure is a status plus a message, and neither is anything the caller has to free. The
 * out-parameter must be left alone on failure -- checked by writing a sentinel first, because a
 * function that half-fills its answer is worse than one that fails.
 */
static void a_failure_is_a_status_and_a_readable_message(void) {
    sl_state out;
    sl_sgp4 *sat = (sl_sgp4 *)0x1;
    char message[256] = {0};

    out.position.x = -12345.0;
    const sl_status status = sl_sgp4_propagate("...", "...", 0.0, &out);
    CHECK(status == SL_ERR_PARSE);
    CHECK(out.position.x == -12345.0);

    const size_t needed = sl_last_error_message(NULL, 0);
    CHECK(needed > 1);
    CHECK(sl_last_error_message(message, sizeof(message)) > 1);
    CHECK(strstr(message, "invalid TLE") != NULL);

    /* A buffer too small truncates and still terminates; it does not write past the end, and it
     * still reports the length the whole message needs rather than the part that fitted. */
    CHECK(sl_last_error_message(message, 4) == needed);
    CHECK(strlen(message) == 3);

    /* And a rejected open leaves the handle untouched rather than half-built. */
    CHECK(sl_sgp4_open("...", "...", &sat) == SL_ERR_PARSE);
    CHECK(sat == (sl_sgp4 *)0x1);

    CHECK(sl_sgp4_propagate_at(NULL, 0.0, &out) == SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_sgp4_propagate(LINE1, LINE2, 0.0, NULL) == SL_ERR_INVALID_ARGUMENT);
}

/*
 * The two-call protocol, written the way a C caller writes it, plus the branch the C++ wrapper
 * cannot reach: a buffer that is too small comes back as `SL_ERR_BUFFER_TOO_SMALL` with the needed
 * length reported and not one element written.
 */
static void the_generator_reports_its_length_before_it_needs_a_buffer(void) {
    sl_walker_params p;
    sl_state *states = NULL;
    sl_state one;
    size_t len = 0;

    sl_walker_params_init(&p);
    CHECK(p.mu == SL_MU_EARTH);
    CHECK(p.pattern == SL_WALKER_DELTA);

    p.inclination_rad = 56.0 * PI / 180.0;
    p.planes = 6;
    p.per_plane = 4;
    p.phasing = 1;
    p.semi_major_axis_m = 29600e3;

    CHECK(sl_walker(&p, NULL, 0, &len) == SL_OK);
    CHECK(len == 24);

    one.position.x = -999.0;
    len = 0;
    CHECK(sl_walker(&p, &one, 1, &len) == SL_ERR_BUFFER_TOO_SMALL);
    CHECK(len == 24);
    CHECK(one.position.x == -999.0);

    states = (sl_state *)malloc(len * sizeof(sl_state));
    if (states == NULL) {
        CHECK(0);
        return;
    }
    CHECK(sl_walker(&p, states, len, &len) == SL_OK);
    for (size_t i = 0; i < len; ++i) {
        CHECK_NEAR(radius_of(states[i].position), p.semi_major_axis_m, 1.0);
    }
    free(states);
}

/* An infeasible street is a domain error, and the geometry solver says so before any allocation. */
static void an_infeasible_constellation_is_a_domain_error(void) {
    sl_soc_params p;
    sl_soc_geometry g;
    size_t len = 0;

    sl_soc_params_init(&p);
    CHECK(p.earth_radius_m == SL_RE_EARTH);
    p.inclination_rad = PI / 2.0;
    p.altitude_m = 1200e3;
    p.min_elevation_rad = 10.0 * PI / 180.0;

    CHECK(sl_street_of_coverage_geometry(&p, &g) == SL_OK);
    CHECK(g.planes > 0);
    CHECK(sl_street_of_coverage(&p, NULL, 0, &len) == SL_OK);
    CHECK(len == (size_t)g.planes * g.per_plane);

    p.min_elevation_rad = PI / 2.0;
    CHECK(sl_street_of_coverage_geometry(&p, &g) == SL_ERR_DOMAIN);
}

/*
 * The totality rule. These functions have no status to return, so the header promises NaN for an
 * input with no answer -- including a null pointer, which is the one a C caller reaches by
 * accident. A trap here would take the consumer's process down with no message.
 */
static void the_total_functions_answer_nan_rather_than_trapping(void) {
    sl_state sat;
    sl_vec3 station;
    sl_latlon start;
    sl_latlon end;

    sat.position.x = 6778137.0;
    sat.position.y = 0.0;
    sat.position.z = 0.0;
    sat.velocity.x = 0.0;
    sat.velocity.y = 7668.0;
    sat.velocity.z = 0.0;
    station.x = 4194303.0;
    station.y = 172460.0;
    station.z = 4780100.0;

    CHECK(sl_range(&sat, &station) > 0.0);
    CHECK(isnan(sl_range(NULL, &station)));
    CHECK(isnan(sl_range(&sat, NULL)));
    CHECK(isnan(sl_range_rate(NULL, NULL)));
    CHECK(isnan(sl_elevation(NULL, &station, 0.0, 0.0)));
    CHECK(isnan(sl_azimuth(&sat, NULL, 0.0, 0.0)));

    /* A NaN input is an input with no answer, and it comes back as one. */
    CHECK(isnan(sl_horizon_look_angle(SL_RE_EARTH, (double)NAN)));
    CHECK(sl_horizon_look_angle(SL_RE_EARTH, 500e3) > 0.0);

    CHECK_NEAR(sl_critical_inclination(), acos(1.0 / sqrt(5.0)), 1e-12);
    CHECK(sl_frozen_eccentricity(7078e3, 1.7, SL_RE_EARTH, SL_J2_EARTH, SL_J3_EARTH) != 0.0);

    start.latitude_rad = 0.4;
    start.longitude_rad = -1.2;
    end = sl_ground_point_at(start, 0.05, 1.9);
    CHECK(fabs(end.latitude_rad) <= PI / 2.0);
}

/*
 * The dense ephemeris from C: build, query, sample, close. The interpolant is the second opaque
 * handle in this ABI, and the reason it is worth a C test of its own is that C is where the caller
 * carries the `close` themselves -- the C++ wrapper does it in a destructor and can never get the
 * pairing wrong.
 */
static void the_ephemeris_handle_interpolates_and_closes(void) {
    const double a = 6778137.0;
    const double speed = sqrt(SL_MU_EARTH / a);
    sl_state start;
    sl_ephemeris *eph = NULL;
    sl_state at_start;
    sl_state at_end;
    sl_state batch[3];
    const double times[3] = {0.0, 900.0, 1800.0};
    double t0 = -1.0;
    double t1 = -1.0;
    uint32_t accepted = 0;
    uint32_t rejected = 0;

    start.position.x = a;
    start.position.y = 0.0;
    start.position.z = 0.0;
    start.velocity.x = 0.0;
    start.velocity.y = speed * cos(51.6 * PI / 180.0);
    start.velocity.z = speed * sin(51.6 * PI / 180.0);

    CHECK(sl_ephemeris_j2(&start, 1800.0, SL_MU_EARTH, SL_J2_EARTH, SL_RE_EARTH, &eph) == SL_OK);
    if (eph == NULL) {
        CHECK(0);
        return;
    }

    CHECK(sl_ephemeris_span(eph, &t0, &t1) == SL_OK);
    CHECK(t0 == 0.0);
    CHECK(t1 == 1800.0);
    CHECK(sl_ephemeris_step_counts(eph, &accepted, &rejected) == SL_OK);
    CHECK(accepted > 0);

    CHECK(sl_ephemeris_at(eph, 0.0, &at_start) == SL_OK);
    CHECK_NEAR(radius_of(at_start.position), a, 1e-6);
    CHECK(sl_ephemeris_at(eph, 1800.0, &at_end) == SL_OK);
    /* Measured: a J2 orbit started circular is not circular, and half an hour in it is 9.9 km
     * lower. Pinned to the value rather than to a band, because a band this wide would also pass
     * for an integration that had quietly lost the J2 term. */
    CHECK_NEAR(radius_of(at_end.position), 6768202.477, 0.01);

    CHECK(sl_ephemeris_sample(eph, times, 3, batch) == SL_OK);
    CHECK(batch[0].position.x == at_start.position.x);
    CHECK(batch[2].position.x == at_end.position.x);

    /* Off the span there is no answer, and asking for one leaves the caller's struct alone. */
    at_end.position.x = -7.0;
    CHECK(sl_ephemeris_at(eph, 1800.1, &at_end) == SL_ERR_DOMAIN);
    CHECK(at_end.position.x == -7.0);

    sl_ephemeris_close(eph);
    sl_ephemeris_close(NULL);
}

/* The three-body and gravity-field entry points, against values that are their own check. */
static void the_field_and_three_body_answers_are_the_shape_they_should_be(void) {
    sl_vec3 r;
    sl_vec3 g;
    sl_vec3 points[5];
    sl_state halo;
    double mag;
    size_t i;

    r.x = 6878137.0;
    r.y = 0.0;
    r.z = 0.0;
    CHECK(sl_geopotential_acceleration(12, &r, &g) == SL_OK);
    mag = sqrt((g.x * g.x) + (g.y * g.y) + (g.z * g.z));
    /* Measured at degree 12: 8.4373638769 m/s^2, against a Newtonian 8.4255087096 at the same
     * radius. The 0.0119 between them is J2 doing its job on the equator, and it is what a
     * binding that had silently fallen back to a point mass would lose. */
    CHECK_NEAR(mag, 8.4373638769, 1e-9);
    CHECK(mag > SL_MU_EARTH / (6878137.0 * 6878137.0));
    CHECK(g.x < 0.0);

    /* A degree the embedded model does not carry is a refusal, not a truncation. */
    CHECK(sl_geopotential_acceleration(100000, &r, &g) == SL_ERR_DOMAIN);

    CHECK(sl_libration_points(0.012150585, points) == SL_OK);
    for (i = 0; i < 3; ++i) {
        CHECK(points[i].y == 0.0);
    }
    CHECK(points[3].y > 0.0);
    CHECK(points[4].y < 0.0);

    CHECK(sl_richardson_halo(1, 0.002, true, 0.012150585, &halo) == SL_OK);
    CHECK(halo.position.x > 0.0);
    CHECK(sl_richardson_halo(9, 0.002, true, 0.012150585, &halo) == SL_ERR_DOMAIN);
}

/*
 * DSST in C: the parameter block is a value, the elements are flat doubles, and `6 * n` of them
 * come back. The claims are the ones the theory makes -- the zonal mean rate does not move the
 * semi-major axis, the node regresses for a prograde orbit, and the short-period corrections put
 * back the kilometres the averaging took out -- plus one pinned number per entry point, so the C
 * and C++ suites and the Rust tests are provably running the same arithmetic.
 */
static void the_mean_elements_are_mean_and_the_osculating_ones_are_not(void) {
    sl_dsst_params p;
    sl_dsst_params_init(&p);
    CHECK(p.zonal_count == 4);
    CHECK(sizeof p.zonal / sizeof p.zonal[0] == SL_DSST_MAX_ZONAL);

    const double inclination = 51.6 * PI / 180.0;
    const double initial[6] = {7.0e6, 1.0e-3, 0.0, tan(inclination / 2.0), 0.0, 0.0};
    const double times[2] = {1800.0, 86400.0};
    double mean[12] = {0};
    double osculating[12] = {0};

    CHECK(sl_dsst_zonal_mean(&p, initial, times, 2, mean) == SL_OK);
    CHECK_NEAR(mean[0], 7.0e6, 1e-6);
    CHECK_NEAR(mean[6], 7.0e6, 1e-6);
    CHECK_NEAR(mean[5], 1.940206765692, 1e-9);
    CHECK_NEAR(mean[10], -3.765956823994e-2, 1e-12);
    CHECK(mean[10] < 0.0);

    CHECK(sl_dsst_zonal_osculating(&p, initial, times, 2, osculating) == SL_OK);
    CHECK_NEAR(osculating[0], 6.9957388684e6, 1.0);
    CHECK(fabs(osculating[0] - mean[0]) > 1e3);

    double full[12] = {0};
    CHECK(sl_dsst_mean(&p, initial, times, 2, full) == SL_OK);
    CHECK_NEAR(full[6], 6.9999999844e6, 1.0);
    CHECK(full[6] != mean[6]);

    CHECK(sl_dsst_osculating(&p, initial, times, 2, full) == SL_OK);
    CHECK_NEAR(full[0], 6.9957388043e6, 1.0);

    /* An unusable field is named, and nothing is written when it is. */
    double untouched[6] = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
    p.zonal_count = 0;
    CHECK(sl_dsst_zonal_mean(&p, initial, times, 1, untouched) == SL_ERR_INVALID_ARGUMENT);
    CHECK(untouched[0] == -1.0);
    sl_dsst_params_init(&p);
    CHECK(sl_dsst_zonal_mean(&p, initial, times, 0, untouched) == SL_ERR_INVALID_ARGUMENT);
    CHECK(untouched[0] == -1.0);
    sl_dsst_params_init(NULL); /* documented no-op */
}

/*
 * The least-squares callback, and the reason `sl_residual_fn` takes a `void *`: this model is a
 * struct on the caller's stack and the library never learns what is in it.
 */
typedef struct diagonal_model {
    double target[6];
    double weight;
    unsigned calls;
} diagonal_model;

static sl_status diagonal_rows(void *user_data, const double *state, sl_residual_row *rows,
                               size_t row_count) {
    diagonal_model *m = (diagonal_model *)user_data;
    m->calls += 1U;
    for (size_t i = 0; i < row_count; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            rows[i].jacobian[j] = (i == j) ? 1.0 : 0.0;
        }
        rows[i].residual = state[i] - m->target[i];
        rows[i].weight = m->weight;
    }
    return SL_OK;
}

static sl_status refusing_rows(void *user_data, const double *state, sl_residual_row *rows,
                               size_t row_count) {
    (void)user_data;
    (void)state;
    (void)rows;
    (void)row_count;
    return SL_ERR_PARSE;
}

/*
 * Six independent unit measurements of a six-state: the solution is the target and the covariance
 * is `diag(1/w)`, both known before the solver runs, so this checks the answer and not merely that
 * one came back.
 */
static void the_least_squares_solve_reaches_the_answer_it_should(void) {
    diagonal_model model;
    const double target[6] = {1.0, -2.0, 3.5, 0.25, -0.75, 10.0};
    for (size_t i = 0; i < 6; ++i) {
        model.target[i] = target[i];
    }
    model.weight = 4.0;
    model.calls = 0U;

    const double x0[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    sl_estimate out;
    CHECK(sl_gauss_newton(NULL, x0, diagonal_rows, &model, 6, &out) == SL_OK);
    CHECK(out.converged);
    CHECK(model.calls >= out.iterations);
    for (size_t i = 0; i < 6; ++i) {
        CHECK_NEAR(out.state[i], target[i], 1e-9);
        for (size_t j = 0; j < 6; ++j) {
            CHECK_NEAR(out.covariance[(i * 6) + j], (i == j) ? 0.25 : 0.0, 1e-12);
        }
    }

    /* The defaults are readable, and a capped run is an answer rather than a failure. */
    sl_gauss_newton_options options;
    sl_gauss_newton_options_init(&options);
    CHECK(options.max_iterations > 0U);
    options.max_iterations = 1U;
    CHECK(sl_gauss_newton(&options, x0, diagonal_rows, &model, 6, &out) == SL_OK);
    CHECK(out.iterations == 1U);
    CHECK(!out.converged);

    /* A model that refuses stops the solve; too few rows leave the state undetermined. */
    CHECK(sl_gauss_newton(NULL, x0, refusing_rows, &model, 6, &out) == SL_ERR_COMPUTE);
    CHECK(sl_gauss_newton(NULL, x0, diagonal_rows, &model, 1, &out) == SL_ERR_DOMAIN);
    CHECK(sl_gauss_newton(NULL, x0, NULL, &model, 6, &out) == SL_ERR_INVALID_ARGUMENT);
    sl_gauss_newton_options_init(NULL); /* documented no-op */
}

/* The same system folded in one row at a time, which is what the filter is for. */
static void the_information_filter_accumulates_and_solves(void) {
    const double target[6] = {1.0, -2.0, 3.5, 0.25, -0.75, 10.0};
    sl_srif *f = NULL;
    CHECK(sl_srif_open(NULL, NULL, &f) == SL_OK);
    CHECK(f != NULL);
    if (f == NULL) {
        return;
    }

    double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK(sl_srif_solve(f, x) == SL_ERR_DOMAIN); /* nothing folded in yet */

    for (size_t i = 0; i < 6; ++i) {
        double row[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        row[i] = 1.0;
        CHECK(sl_srif_add_row(f, row, target[i], 4.0) == SL_OK);
    }

    CHECK(sl_srif_solve(f, x) == SL_OK);
    double covariance[36] = {0};
    CHECK(sl_srif_covariance(f, covariance) == SL_OK);
    for (size_t i = 0; i < 6; ++i) {
        CHECK_NEAR(x[i], target[i], 1e-12);
        CHECK_NEAR(covariance[(i * 6) + i], 0.25, 1e-12);
    }

    double sum_of_squares = -1.0;
    uint32_t rows = 0;
    CHECK(sl_srif_residual_stats(f, &sum_of_squares, &rows) == SL_OK);
    CHECK(rows == 6);
    CHECK(fabs(sum_of_squares) < 1e-18);

    sl_srif_close(f);
    sl_srif_close(NULL); /* documented no-op */

    /* Half a prior is a mistake rather than a shorthand, and a refused open writes nothing. */
    sl_srif *half = NULL;
    CHECK(sl_srif_open(target, NULL, &half) == SL_ERR_INVALID_ARGUMENT);
    CHECK(half == NULL);
}

/*
 * The lattice contains the Walker: a circular lattice (No, Nso, Nc) = (P, S, F) is a Walker Delta
 * i: T/P/F, satellite for satellite. Checked here as well as in Rust, because the C caller reaches
 * both generators through structs they fill in by hand, and a field wired to the wrong member of
 * the parameter block is exactly what this comparison catches.
 */
static void the_lattice_is_the_walker_it_contains(void) {
    sl_lattice_params l;
    sl_walker_params w;
    sl_state *from_lattice = NULL;
    sl_state *from_walker = NULL;
    size_t len = 0;
    size_t other = 0;
    size_t i = 0;
    const double a = SL_RE_EARTH + 20000e3;
    const double inc = 56.0 * PI / 180.0;

    sl_lattice_params_init(&l);
    CHECK(l.mu == SL_MU_EARTH);
    l.orbits = 4;
    l.sats_per_orbit = 6;
    l.config = 3;
    l.semi_major_axis_m = a;
    l.inclination_rad = inc;

    sl_walker_params_init(&w);
    w.inclination_rad = inc;
    w.planes = 4;
    w.per_plane = 6;
    w.phasing = 3;
    w.semi_major_axis_m = a;

    CHECK(sl_lattice_flower(&l, NULL, 0, &len) == SL_OK);
    CHECK(len == 24);
    CHECK(sl_walker(&w, NULL, 0, &other) == SL_OK);
    CHECK(other == len);

    from_lattice = (sl_state *)malloc(len * sizeof(sl_state));
    from_walker = (sl_state *)malloc(len * sizeof(sl_state));
    if (from_lattice == NULL || from_walker == NULL) {
        CHECK(0);
        free(from_lattice);
        free(from_walker);
        return;
    }
    CHECK(sl_lattice_flower(&l, from_lattice, len, &len) == SL_OK);
    CHECK(sl_walker(&w, from_walker, len, &other) == SL_OK);
    for (i = 0; i < len; ++i) {
        CHECK_NEAR(from_lattice[i].position.x, from_walker[i].position.x, 1e-6);
        CHECK_NEAR(from_lattice[i].position.y, from_walker[i].position.y, 1e-6);
        CHECK_NEAR(from_lattice[i].position.z, from_walker[i].position.z, 1e-6);
    }
    free(from_lattice);
    free(from_walker);
}

/* Draim's four, and the orbit the geometry says they fly -- against Kepler's third law. */
static void the_draim_design_flies_the_orbit_it_reports(void) {
    sl_draim_params d;
    sl_draim_orbit orbit;
    sl_state four[4];
    size_t len = 0;
    size_t i = 0;
    double want_a = 0.0;

    sl_draim_params_init(&d);
    CHECK(d.period_s == 172800.0);
    CHECK(d.eccentricity == 0.28);
    CHECK_NEAR(d.inclination_rad, 33.0 * PI / 180.0, 1e-15);

    CHECK(sl_draim_geometry(&d, &orbit) == SL_OK);
    want_a = cbrt(SL_MU_EARTH * (d.period_s / (2.0 * PI)) * (d.period_s / (2.0 * PI)));
    CHECK_NEAR(orbit.semi_major_axis_m, want_a, want_a * 1e-12);
    CHECK_NEAR(orbit.perigee_radius_m, want_a * (1.0 - d.eccentricity), 1e-6);
    CHECK_NEAR(orbit.apogee_radius_m, want_a * (1.0 + d.eccentricity), 1e-6);

    CHECK(sl_draim(&d, NULL, 0, &len) == SL_OK);
    CHECK(len == 4);
    CHECK(sl_draim(&d, four, 4, &len) == SL_OK);
    for (i = 0; i < len; ++i) {
        const double r = radius_of(four[i].position);
        CHECK(r >= orbit.perigee_radius_m - 1.0);
        CHECK(r <= orbit.apogee_radius_m + 1.0);
    }
}

/*
 * Coverage: the grid is the points it names, the metrics are the windows they summarise, and the
 * sensor tag is checked before the fields it does not name are read.
 */
static void the_coverage_engine_answers_what_the_windows_say(void) {
    const double lats[2] = {0.0, 0.4};
    const double lons[2] = {0.0, 1.5};
    const double a = SL_RE_EARTH + 700e3;
    const double inc = 51.6 * PI / 180.0;
    const double horizon = 86400.0;
    sl_sensor sensor;
    sl_state cells[4];
    sl_coverage_metrics metrics[4];
    sl_constellation_fom fom;
    sl_window *windows = NULL;
    size_t len = 0;
    size_t k = 0;
    size_t j = 0;
    int passes_seen = 0;

    sl_sensor_init(&sensor, SL_SENSOR_ELEVATION_MASK);
    sensor.min_elevation_rad = 10.0 * PI / 180.0;

    /* The mask is a geometric elevation; a station's "visible above 10 degrees" is the apparent
       one. Bennett's bend at 10 degrees is 5.3 arcminutes, so the geometric mask sits *below* the
       apparent one, and Saemundsson takes it back within the few arcseconds by which the two
       published fits disagree. Both directions and both domain refusals, here rather than only in
       Rust, because a C caller who cannot convert the threshold cannot correct the bias at all. */
    {
        double geometric = 0.0;
        double back = 0.0;
        const double apparent = 10.0 * PI / 180.0;
        CHECK(sl_refraction_geometric(apparent, 1010.0, 283.0, &geometric) == SL_OK);
        CHECK(geometric < apparent);
        CHECK_NEAR((apparent - geometric) * 180.0 * 60.0 / PI, 5.3, 0.1);
        CHECK(sl_refraction_apparent(geometric, 1010.0, 283.0, &back) == SL_OK);
        CHECK_NEAR((back - apparent) * 180.0 * 3600.0 / PI, 0.0, 6.0);
        CHECK(sl_refraction_geometric(-0.1, 1010.0, 283.0, &geometric) == SL_ERR_DOMAIN);
        CHECK(sl_refraction_apparent(apparent, 0.0, 283.0, &back) == SL_ERR_DOMAIN);
        CHECK(sl_refraction_apparent(apparent, 1010.0, 283.0, NULL) == SL_ERR_INVALID_ARGUMENT);
    }

    /* Diurnal aberration. A station on the equator at longitude zero is carried due east at 465.1
       m/s, and that shifts a source at its zenith by 0.320 arcseconds towards the motion. The
       Earth's 29.8 km/s orbital velocity -- the 20.5" constant of aberration -- does not enter,
       because the station and the satellite share it. Here rather than only in Rust for the same
       reason as the refraction block above: a C caller who cannot reach the correction cannot
       apply it. */
    {
        sl_vec3 station;
        sl_vec3 omega;
        sl_vec3 velocity;
        sl_vec3 zenith;
        sl_vec3 seen;
        sl_vec3 back;
        sl_vec3 zero;
        double shift_arcsec = 0.0;
        station.x = SL_RE_EARTH;
        station.y = 0.0;
        station.z = 0.0;
        omega.x = 0.0;
        omega.y = 0.0;
        omega.z = SL_EARTH_ROTATION_RATE;
        zenith.x = 1.0;
        zenith.y = 0.0;
        zenith.z = 0.0;
        zero.x = 0.0;
        zero.y = 0.0;
        zero.z = 0.0;

        CHECK(sl_diurnal_velocity(&station, &omega, &velocity) == SL_OK);
        CHECK_NEAR(radius_of(velocity), 465.101, 1e-3);
        CHECK_NEAR(velocity.y, 465.101, 1e-3); /* due east, not merely of the right size */

        CHECK(sl_aberration_apply(&velocity, &zenith, &seen) == SL_OK);
        CHECK_NEAR(radius_of(seen), 1.0, 1e-15);
        /* The shift is transverse and the motion is along +y, so `atan2` reads it directly rather
           than through an `acos` that would lose half the mantissa at this angle. */
        shift_arcsec = atan2(seen.y, seen.x) * 180.0 * 3600.0 / PI;
        CHECK_NEAR(shift_arcsec, 0.320, 1e-3);

        CHECK(sl_aberration_remove(&velocity, &seen, &back) == SL_OK);
        CHECK_NEAR(back.x, 1.0, 1e-15);
        CHECK_NEAR(back.y, 0.0, 1e-15);

        /* An observer at the speed of light is a domain error, not a NaN; a direction of zero
           length is not a direction; a null out-pointer is refused before anything is computed. */
        velocity.y = 3.0e8;
        CHECK(sl_aberration_apply(&velocity, &zenith, &seen) == SL_ERR_DOMAIN);
        velocity.y = 465.101;
        CHECK(sl_aberration_remove(&velocity, &zero, &back) == SL_ERR_INVALID_ARGUMENT);
        CHECK(sl_aberration_apply(&velocity, &zenith, NULL) == SL_ERR_INVALID_ARGUMENT);
        CHECK(sl_diurnal_velocity(&station, NULL, &velocity) == SL_ERR_INVALID_ARGUMENT);
    }

    CHECK(sl_grid_cells(lats, 2, lons, 2, NULL, 0, &len) == SL_OK);
    CHECK(len == 4);
    CHECK(sl_grid_cells(lats, 2, lons, 2, cells, 4, &len) == SL_OK);
    for (k = 0; k < len; ++k) {
        CHECK_NEAR(radius_of(cells[k].position), SL_RE_EARTH, 1e-6);
        CHECK_NEAR(radius_of(cells[k].velocity), 1.0, 1e-12);
    }

    CHECK(sl_coverage_scan(a, inc, 0.0, 0.0, &sensor, cells, len, horizon, 1.0, 0, metrics) ==
          SL_OK);
    for (k = 0; k < len; ++k) {
        double covered = 0.0;
        size_t count = 0;
        CHECK(sl_access_windows(a, inc, 0.0, 0.0, &sensor, &cells[k], horizon, 1.0, NULL, 0,
                                &count) == SL_OK);
        CHECK(count == (size_t)metrics[k].passes);
        if (count == 0) {
            continue;
        }
        windows = (sl_window *)malloc(count * sizeof(sl_window));
        if (windows == NULL) {
            CHECK(0);
            return;
        }
        CHECK(sl_access_windows(a, inc, 0.0, 0.0, &sensor, &cells[k], horizon, 1.0, windows, count,
                                &count) == SL_OK);
        for (j = 0; j < count; ++j) {
            CHECK(windows[j].rise_s < windows[j].set_s);
            covered += windows[j].set_s - windows[j].rise_s;
        }
        free(windows);
        CHECK_NEAR(metrics[k].coverage_fraction, covered / horizon, 1e-9);
        passes_seen = 1;
    }
    CHECK(passes_seen);

    CHECK(sl_figures_of_merit(metrics, len, NULL, &fom) == SL_OK);
    CHECK(fom.point_count == (uint32_t)len);
    CHECK(fom.min_coverage <= fom.mean_coverage);

    /* A tag the header does not define is refused before any field is read. */
    sl_sensor_init(&sensor, 7);
    CHECK(sl_access_windows(a, inc, 0.0, 0.0, &sensor, &cells[0], horizon, 1.0, NULL, 0, &len) ==
          SL_ERR_INVALID_ARGUMENT);

    /* A SAR whose near edge is past its far one is a domain error, not a silent swap. */
    sl_sensor_init(&sensor, SL_SENSOR_SIDE_LOOKING_SAR);
    sensor.near_incidence_rad = 0.8;
    sensor.far_incidence_rad = 0.4;
    CHECK(sl_access_windows(a, inc, 0.0, 0.0, &sensor, &cells[0], horizon, 1.0, NULL, 0, &len) ==
          SL_ERR_DOMAIN);
}

/*
 * GNSS, ionosphere and troposphere. The C caller reaches these with hand-sized buffers and raw
 * `double` arrays, so what is checked here is the shape of that: an explicit text length rather
 * than a NUL scan, four-element coefficient arrays, and the two-call protocol over a record type
 * wider than an `sl_state`.
 */
static void the_gnss_models_answer_what_their_formulas_say(void) {
    const double l1 = 1575.42e6;
    const double l2 = 1227.60e6;
    const char *header = "     3.04           RINEX VERSION / TYPE\n";
    double alpha[4] = {0.0, 0.0, 0.0, 0.0};
    double beta[4] = {0.0, 0.0, 0.0, 0.0};
    double night = 0.0;
    double delay = 0.0;
    size_t len = 0;
    sl_vec3 receiver;
    sl_vec3 satellites[5];
    sl_vec3 field;
    sl_dop dop;
    sl_mapping mapping;
    sl_gpt3_weather weather;
    sl_vmf_coefficients vienna;
    size_t i = 0;
    int k = 0;

    /* The FDMA frequencies are a closed form, and an out-of-range channel is NaN rather than a
     * wrap onto a channel that exists. */
    for (k = -7; k <= 6; ++k) {
        CHECK_NEAR(sl_glonass_l1_frequency(k), (1602.0 + (k * 0.5625)) * 1e6, 1e-6);
        CHECK_NEAR(sl_glonass_l2_frequency(k), (1246.0 + (k * 0.4375)) * 1e6, 1e-6);
    }
    CHECK(isnan(sl_glonass_l1_frequency(1000)));

    /* A RINEX header with no records is zero records and SL_OK, not a parse failure. */
    CHECK(sl_rinex_gnss_positions(header, strlen(header), 0.0, NULL, 0, &len) == SL_OK);
    CHECK(len == 0);
    CHECK(sl_rinex_glonass_positions(header, strlen(header), 0.0, NULL, 0, &len) == SL_OK);
    CHECK(len == 0);

    /* Five satellites spread over the sky, and the two identities the five DOP figures obey. */
    receiver.x = SL_RE_EARTH;
    receiver.y = 0.0;
    receiver.z = 0.0;
    for (i = 0; i < 5; ++i) {
        const double theta = 0.25 * (double)i;
        const double phi = 1.3 * (double)i;
        const double a = SL_RE_EARTH + 20200e3;
        satellites[i].x = a * cos(theta);
        satellites[i].y = a * sin(theta) * cos(phi);
        satellites[i].z = a * sin(theta) * sin(phi);
    }
    CHECK(sl_dilution_of_precision(&receiver, satellites, 5, &dop) == SL_OK);
    CHECK_NEAR(dop.gdop * dop.gdop, (dop.pdop * dop.pdop) + (dop.tdop * dop.tdop), 1e-9);
    CHECK_NEAR(dop.pdop * dop.pdop, (dop.hdop * dop.hdop) + (dop.vdop * dop.vdop), 1e-9);
    CHECK(sl_dilution_of_precision(&receiver, satellites, 3, &dop) == SL_ERR_DOMAIN);

    /* First order is its own closed form, and falls as 1/f^2. */
    delay = sl_ionospheric_path_delay(20.0, l1);
    CHECK_NEAR(delay, 40.3e16 * 20.0 / (l1 * l1), 1e-12);
    CHECK_NEAR(sl_ionospheric_path_delay(20.0, l2) / delay, (l1 * l1) / (l2 * l2), 1e-9);

    /* With zero coefficients Klobuchar is its 5 ns floor times the obliquity factor. */
    CHECK(sl_klobuchar_iono_delay(alpha, beta, 0.0, 0.0, PI / 2.0, 0.0, 0.0, &night) == SL_OK);
    CHECK_NEAR(night, 5e-9 * 299792458.0 * (1.0 + (16.0 * 0.03 * 0.03 * 0.03)), 1e-6);

    /* Every mapping function is 1 at the zenith and refuses the horizon. */
    CHECK(sl_niell_mapping(0.7, 200.0, 90.0, PI / 2.0, &mapping) == SL_OK);
    CHECK_NEAR(mapping.hydrostatic, 1.0, 1e-2);
    CHECK(sl_gmf_mapping(0.7, 0.2, 200.0, 90.0, PI / 2.0, &mapping) == SL_OK);
    CHECK_NEAR(mapping.wet, 1.0, 1e-2);
    CHECK(sl_vmf1_mapping(1.2e-3, 6e-4, 0.7, 200.0, 90.0, PI / 2.0, &mapping) == SL_OK);
    CHECK(sl_vmf3_mapping(1.2e-3, 6e-4, 0.7, 0.2, 200.0, 90.0, PI / 2.0, &mapping) == SL_OK);
    CHECK(sl_niell_mapping(0.7, 200.0, 90.0, 0.0, &mapping) == SL_ERR_DOMAIN);

    /* And the field the second-order ionospheric term leans on is tens of microtesla. */
    receiver.x = SL_RE_EARTH * cos(0.6);
    receiver.y = 0.0;
    receiver.z = SL_RE_EARTH * sin(0.6);
    CHECK(sl_igrf_field_ecef(&receiver, 2026.0, &field) == SL_OK);
    CHECK(radius_of(field) > 20e-6);
    CHECK(radius_of(field) < 70e-6);

    /* GPT3 is blind -- no file, no forecast -- so it answers a plausible atmosphere anywhere. */
    CHECK(sl_gpt3(0.84, 0.29, 329.0, 180.0, &weather) == SL_OK);
    CHECK(weather.pressure > 80e3 && weather.pressure < 110e3);
    CHECK(weather.temperature > 220.0 && weather.temperature < 320.0);
    CHECK(weather.ah > 0.0 && weather.aw > 0.0);
    CHECK(sl_gpt3(0.84, 0.29, 329.0, 180.0, NULL) == SL_ERR_INVALID_ARGUMENT);

    /* The grid reader takes text with its length, and text that is not a grid does not parse. */
    CHECK(sl_vmf_grid_coefficients("not a grid", 10, 0.84, 0.29, &vienna) == SL_ERR_PARSE);
}

/*
 * A minimal well-formed RINEX 3 observation file, written column by column rather than produced by
 * the reader -- a fixture generated from the reader would agree with the reader's own mistakes.
 *
 * Two GPS satellites in the first epoch is the case a lookup on either half of the identifier alone
 * cannot tell apart: G09 sits behind G07 and carries a different range. G07's second record is the
 * one with the quality digits closed up -- `00016` is the value, then LLI 1, then strength 6 -- so
 * the receiver declares a loss of lock there and nowhere else.
 */
static const char *const RINEX_OBS =
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
 * The observation reader. Nothing about a value names it except its position in the constellation's
 * declared descriptor list, so what is checked here is that the position is honoured on both halves
 * of the satellite identifier and on both constellations, and that a descriptor the header never
 * declared is refused rather than answered with an empty series.
 */
static void the_rinex_observation_reader_finds_a_value_by_its_declared_slot(void) {
    const size_t text_len = strlen(RINEX_OBS);
    sl_obs_header header;
    sl_obs_descriptor codes[4];
    sl_obs_sample samples[4];
    size_t len = 0;

    CHECK(sl_rinex_obs_header(RINEX_OBS, text_len, &header) == SL_OK);
    CHECK_NEAR(header.version, 3.05, 1e-12);
    CHECK(header.file_system == 'M');
    CHECK(header.marker_name[0] == '\0');
    CHECK_NEAR(header.interval, 30.0, 1e-12);
    CHECK_NEAR(header.approx_position[0], 4460894.9, 1e-6);
    CHECK_NEAR(header.approx_position[2], -3674748.5, 1e-6);
    /* The header declares no antenna eccentricity, and absent is NaN in all three components. */
    CHECK(isnan(header.antenna_delta_hen[0]));
    CHECK(isnan(header.antenna_delta_hen[1]));
    CHECK(isnan(header.antenna_delta_hen[2]));
    /* No `TIME OF FIRST OBS` either, and the format's default is GPS rather than "unknown". */
    CHECK(header.time_system == SL_TIME_SYSTEM_GPS);
    CHECK(sl_rinex_obs_header(RINEX_OBS, text_len, NULL) == SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_rinex_obs_header("not a RINEX file", 16, &header) == SL_ERR_INVALID_ARGUMENT);

    /* Two descriptors for GPS, one for GLONASS, and none at all for a constellation the file never
     * mentions -- which is zero results and SL_OK, not a failure. */
    CHECK(sl_rinex_obs_descriptors(RINEX_OBS, text_len, 'G', codes, 4, &len) == SL_OK);
    CHECK(len == 2);
    CHECK(strcmp(codes[0].code, "C1C") == 0);
    CHECK(strcmp(codes[1].code, "L1C") == 0);
    CHECK(sl_rinex_obs_descriptors(RINEX_OBS, text_len, 'R', codes, 4, &len) == SL_OK);
    CHECK(len == 1);
    CHECK(sl_rinex_obs_descriptors(RINEX_OBS, text_len, 'E', NULL, 0, &len) == SL_OK);
    CHECK(len == 0);
    CHECK(sl_rinex_obs_descriptors(RINEX_OBS, text_len, '7', codes, 4, &len) ==
          SL_ERR_INVALID_ARGUMENT);

    /* G07 is observed at both epochs, thirty seconds apart, and only the second record carries the
     * receiver's own loss-of-lock bit. */
    CHECK(sl_rinex_obs_series(RINEX_OBS, text_len, 'G', 7, "C1C", NULL, 0, &len) == SL_OK);
    CHECK(len == 2);
    CHECK(sl_rinex_obs_series(RINEX_OBS, text_len, 'G', 7, "C1C", samples, 1, &len) ==
          SL_ERR_BUFFER_TOO_SMALL);
    CHECK(len == 2);
    CHECK(sl_rinex_obs_series(RINEX_OBS, text_len, 'G', 7, "C1C", samples, 4, &len) == SL_OK);
    CHECK_NEAR(samples[0].value, 23629347.915, 1e-6);
    CHECK_NEAR(samples[1].value, 23629400.000, 1e-6);
    CHECK_NEAR(samples[1].seconds_j2000 - samples[0].seconds_j2000, 30.0, 1e-9);
    CHECK(samples[0].lost_lock == 0);
    CHECK(samples[1].lost_lock == 1);

    /* The second slot is a different quantity in a different unit, read out of the same records. */
    CHECK(sl_rinex_obs_series(RINEX_OBS, text_len, 'G', 7, "L1C", samples, 4, &len) == SL_OK);
    CHECK(len == 2);
    CHECK_NEAR(samples[0].value, 124144623.325, 1e-6);

    /* G09 is the satellite a match that stopped at the constellation letter would report G07 for. */
    CHECK(sl_rinex_obs_series(RINEX_OBS, text_len, 'G', 9, "C1C", samples, 4, &len) == SL_OK);
    CHECK(len == 1);
    CHECK_NEAR(samples[0].value, 23626000.0, 1e-6);

    /* And a PRN a match that stopped at the number would report R05's range for. */
    CHECK(sl_rinex_obs_series(RINEX_OBS, text_len, 'R', 5, "C1C", samples, 4, &len) == SL_OK);
    CHECK(len == 1);
    CHECK_NEAR(samples[0].value, 21000000.0, 1e-6);

    /* GLONASS declares no L1C here. That is "asked for the wrong thing", not "not observed". */
    CHECK(sl_rinex_obs_series(RINEX_OBS, text_len, 'R', 5, "L1C", samples, 4, &len) ==
          SL_ERR_INVALID_ARGUMENT);
    /* A satellite the file never carries is observed nowhere, which is an empty series and SL_OK. */
    CHECK(sl_rinex_obs_series(RINEX_OBS, text_len, 'G', 31, "C1C", NULL, 0, &len) == SL_OK);
    CHECK(len == 0);
    /* A PRN past the byte the format writes it in, and a null descriptor, are both refusals. */
    CHECK(sl_rinex_obs_series(RINEX_OBS, text_len, 'G', 300, "C1C", NULL, 0, &len) ==
          SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_rinex_obs_series(RINEX_OBS, text_len, 'G', 7, NULL, NULL, 0, &len) ==
          SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_rinex_obs_series(NULL, 0, 'G', 7, "C1C", NULL, 0, &len) == SL_ERR_INVALID_ARGUMENT);
}

/*
 * The atmospheres, tides and density-calibration half, as a C consumer sees it: fixed-length
 * out-parameters written by hand, the two-call protocol including the short-buffer branch, and the
 * enum tag that only C can pass out of range.
 */
static void the_atmosphere_models_answer_what_their_formulas_say(void) {
    const double delaunay[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    const char *csv = "DATE,BSRN,ND,KP1,KP2\n";
    double step2_a[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double step2_b[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double secular_pole[2] = {0.0, 0.0};
    double retired_pole[2] = {0.0, 0.0};
    double ocean[2] = {0.0, 0.0};
    double drivers[SL_DENSITY_DRIVERS];
    double ap_array[SL_AP_ARRAY_LEN];
    double coefficients[SL_DENSITY_CORRECTION_COEFFICIENTS];
    double grid[4];
    double factor = 0.0;
    double t = 20.0 * 365.25 * 86400.0;
    sl_zonal mean_now;
    sl_zonal mean_later;
    sl_zonal secular_now;
    sl_zonal secular_later;
    sl_drag_panel panel;
    sl_swath swath;
    sl_density_sample samples[2];
    size_t len = 0;
    int i = 0;
    double ap_history[16];
    double storm_history[16];
    double quiet_storm = 0.0;
    double interval[SL_SDTM1_INTERVAL];
    double envelope[SL_SDTM1_ENVELOPE];
    double density_400 = 0.0;
    double sw_ap60[16];
    double sw_speed[16];
    double sw_by[16];
    double sw_bz[16];
    double wind_storm = 0.0;

    /* The thermosphere thins with height, in kg/m^3 and not the model's own g/cm^3 -- and a NaN
     * argument is NaN out, which the model on its own does not do. */
    CHECK(sl_nrlmsise00_density(172, 29000.0, 200.0, 0.0, 0.0, 150.0, 150.0, 4.0) >
          sl_nrlmsise00_density(172, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 4.0));
    CHECK(sl_nrlmsise00_density(172, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 4.0) > 1e-13);
    CHECK(sl_nrlmsise00_density(172, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 4.0) < 1e-11);
    CHECK(isnan(sl_nrlmsise00_density(172, 29000.0, NAN, 0.0, 0.0, 150.0, 150.0, 4.0)));

    /* SDTM-1, this library's own model. Its storm state cools: a constant index history sits on the
     * steady state H/(1 + H/H_s) from the first step, below the index itself and the same at any
     * cadence. An `ap` 300 tail moves it up without reaching it. */
    for (i = 0; i < 16; ++i) {
        ap_history[i] = 5.0;
        storm_history[i] = i < 12 ? 5.0 : 300.0;
    }
    quiet_storm = sl_sdtm1_storm_response(ap_history, 16, 10800.0);
    CHECK(quiet_storm > 0.0 && quiet_storm < 5.0);
    CHECK(fabs(sl_sdtm1_storm_response(ap_history, 16, 3600.0) - quiet_storm) <
          1e-12 * quiet_storm);
    CHECK(sl_sdtm1_storm_response(storm_history, 16, 10800.0) > quiet_storm);
    CHECK(sl_sdtm1_storm_response(storm_history, 16, 10800.0) < 300.0);
    CHECK(sl_sdtm1_storm_response(ap_history, 0, 10800.0) == 0.0);
    CHECK(isnan(sl_sdtm1_storm_response(NULL, 16, 10800.0)));
    CHECK(isnan(sl_sdtm1_storm_response(ap_history, 16, 0.0)));

    /* Thins with height, in a band the measured corpus spans at 400 km, and a NaN day of year is NaN
     * out -- the argument the model would otherwise carry into a plausible mid-season number. */
    CHECK(sl_sdtm1_density(172.0, 29000.0, 200.0, 0.0, 0.0, 150.0, 150.0, 5.0) >
          sl_sdtm1_density(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0));
    CHECK(sl_sdtm1_density(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0) > 1e-13);
    CHECK(sl_sdtm1_density(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0) < 1e-11);
    CHECK(isnan(sl_sdtm1_density(NAN, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0)));

    /* Fifteen degrees of longitude is one hour of local solar time, which is the convention a caller
     * has to be able to rely on: 15 E an hour earlier in UT is the same place in the model, and holds
     * the density to 1.7e-4 relative -- the annual harmonic over 1/24 of a day -- while the same 15 E
     * at the same UT is an hour later in local time and worth 9.3 %. */
    density_400 = sl_sdtm1_density(172.0, 43200.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0);
    CHECK(fabs(sl_sdtm1_density(172.0, 39600.0, 400.0, 0.0, 15.0, 150.0, 150.0, 5.0) / density_400 -
               1.0) < 1e-3);
    CHECK(fabs(sl_sdtm1_density(172.0, 43200.0, 400.0, 0.0, 15.0, 150.0, 150.0, 5.0) / density_400 -
               1.0) > 0.05);

    /* T_inf is above the 380 K boundary by construction and inside the 586-1528 K the fit produced
     * over the corpus; a storm raises it. */
    CHECK(sl_sdtm1_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0) >
          586.0);
    CHECK(sl_sdtm1_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0) <
          1528.0);
    CHECK(sl_sdtm1_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 250.0) >
          sl_sdtm1_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0));
    CHECK(isnan(sl_sdtm1_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, NAN, 5.0)));

    /* The interval brackets the density multiplicatively, so z = 0 collapses it onto the value and a
     * wider z only widens it; a null out pointer is refused rather than written through. */
    density_400 = sl_sdtm1_density(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0);
    CHECK(sl_sdtm1_density_interval(density_400, 150.0, 150.0, 5.0, 0.0, interval) == SL_OK);
    CHECK_NEAR(interval[0], density_400, 1e-24);
    CHECK_NEAR(interval[1], density_400, 1e-24);
    CHECK(sl_sdtm1_density_interval(density_400, 150.0, 150.0, 5.0, 1.6448536269514727, interval) ==
          SL_OK);
    CHECK(interval[0] < density_400);
    CHECK(interval[1] > density_400);
    CHECK_NEAR(interval[0] * interval[1], density_400 * density_400, 1e-36);
    CHECK(sl_sdtm1_density_interval(density_400, 150.0, 150.0, 5.0, 1.0, NULL) ==
          SL_ERR_INVALID_ARGUMENT);

    /* The envelope is kilometres and flux units in the order the header names, and it holds the
     * 400 km and 150 sfu evaluated above; a null out pointer is refused. */
    CHECK(sl_sdtm1_envelope(envelope) == SL_OK);
    CHECK(envelope[0] < 400.0 && 400.0 < envelope[1] && envelope[1] < 600.0);
    CHECK(envelope[2] < 150.0 && 150.0 < envelope[3]);
    CHECK(sl_sdtm1_envelope(NULL) == SL_ERR_INVALID_ARGUMENT);

    /* SDTM-1's solar-wind field runs its storm state hourly. An hour without the solar wind -- a NaN
     * speed, By or Bz -- drives by its ap60, so a quiet ap60 history without it sits on its steady
     * state below 5, as the 3-hourly field's does. A strong southward field over the last six hours
     * drives the same history higher, and a due-north one lower. A fill value is no solar wind and
     * answers NaN, as a null series does; n = 0 is an empty history. */
    for (i = 0; i < 16; ++i) {
        sw_ap60[i] = 5.0;
        sw_speed[i] = NAN;
        sw_by[i] = NAN;
        sw_bz[i] = NAN;
    }
    quiet_storm = sl_sdtm1_solar_wind_storm_response(sw_ap60, sw_speed, sw_by, sw_bz, 16);
    CHECK(quiet_storm > 0.0 && quiet_storm < 5.0);
    for (i = 10; i < 16; ++i) {
        sw_speed[i] = 700.0;
        sw_by[i] = 0.0;
        sw_bz[i] = -20.0;
    }
    wind_storm = sl_sdtm1_solar_wind_storm_response(sw_ap60, sw_speed, sw_by, sw_bz, 16);
    CHECK(wind_storm > quiet_storm);
    for (i = 10; i < 16; ++i) {
        sw_bz[i] = 20.0;
    }
    CHECK(sl_sdtm1_solar_wind_storm_response(sw_ap60, sw_speed, sw_by, sw_bz, 16) < quiet_storm);
    sw_speed[12] = 9999.0;
    CHECK(isnan(sl_sdtm1_solar_wind_storm_response(sw_ap60, sw_speed, sw_by, sw_bz, 16)));
    CHECK(sl_sdtm1_solar_wind_storm_response(sw_ap60, sw_speed, sw_by, sw_bz, 0) == 0.0);
    CHECK(isnan(sl_sdtm1_solar_wind_storm_response(sw_ap60, NULL, sw_by, sw_bz, 16)));

    /* Its point functions take the 3-hourly field's arguments in the same units and answer its own
     * values; its interval and envelope are its own fit's, and a null out pointer is refused. */
    density_400 = sl_sdtm1_solar_wind_density(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0);
    CHECK(sl_sdtm1_solar_wind_density(172.0, 29000.0, 200.0, 0.0, 0.0, 150.0, 150.0, 5.0) >
          density_400);
    CHECK(density_400 > 1e-13 && density_400 < 1e-11);
    CHECK(density_400 != sl_sdtm1_density(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0));
    CHECK(isnan(sl_sdtm1_solar_wind_density(NAN, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 5.0)));
    CHECK(sl_sdtm1_solar_wind_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0,
                                                     250.0) >
          sl_sdtm1_solar_wind_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0, 150.0,
                                                     5.0));
    CHECK(isnan(sl_sdtm1_solar_wind_exospheric_temperature(172.0, 29000.0, 400.0, 0.0, 0.0, 150.0,
                                                           NAN, 5.0)));
    CHECK(sl_sdtm1_solar_wind_density_interval(density_400, 150.0, 150.0, 5.0, 1.6448536269514727,
                                               interval) == SL_OK);
    CHECK(interval[0] < density_400 && density_400 < interval[1]);
    CHECK(sl_sdtm1_solar_wind_density_interval(density_400, 150.0, 150.0, 5.0, 1.0, NULL) ==
          SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_sdtm1_solar_wind_envelope(envelope) == SL_OK);
    CHECK(envelope[0] < 400.0 && 400.0 < envelope[1] && envelope[1] < 600.0);
    CHECK(envelope[2] < 150.0 && 150.0 < envelope[3]);
    CHECK(sl_sdtm1_solar_wind_envelope(NULL) == SL_ERR_INVALID_ARGUMENT);

    /* Step-2 is trigonometric in the sidereal argument, so a whole turn is the identity. */
    CHECK(sl_solid_tide_step2(0.7, delaunay, step2_a) == SL_OK);
    CHECK(sl_solid_tide_step2(0.7 + (2.0 * PI), delaunay, step2_b) == SL_OK);
    for (i = 0; i < 5; ++i) {
        CHECK_NEAR(step2_a[i], step2_b[i], 1e-20);
    }
    CHECK(sl_solid_tide_step2(NAN, delaunay, step2_a) == SL_ERR_INVALID_ARGUMENT);

    /* The two conventional poles are ~60 mas apart, which is why the tag is an argument and not a
     * default nobody sees -- and a tag outside the enum is refused rather than defaulted. */
    CHECK(sl_solid_pole_tide(0.0, 1e-7, t, SL_MEAN_POLE_SECULAR2018, secular_pole) == SL_OK);
    CHECK(sl_solid_pole_tide(0.0, 1e-7, t, SL_MEAN_POLE_IERS2010, retired_pole) == SL_OK);
    CHECK(fabs(retired_pole[0] - secular_pole[0]) > 0.1 * fabs(secular_pole[0]));
    CHECK(sl_ocean_pole_tide(0.0, 1e-7, t, SL_MEAN_POLE_SECULAR2018, ocean) == SL_OK);
    CHECK(ocean[0] != 0.0);
    /* Casting past the enum is the point: C is the only rung that can hand the library a tag
     * outside the enumeration, and that it is refused rather than defaulted is the property.
     * NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) */
    CHECK(sl_solid_pole_tide(0.0, 1e-7, t, (sl_mean_pole)7, ocean) == SL_ERR_INVALID_ARGUMENT);

    /* conventional(t) - secular(t) is the epoch mean, so it does not move with t. */
    CHECK(sl_conventional_zonal_coefficients(0.0, &mean_now) == SL_OK);
    CHECK(sl_conventional_zonal_coefficients(25.0, &mean_later) == SL_OK);
    CHECK(sl_time_variable_zonal_correction(0.0, &secular_now) == SL_OK);
    CHECK(sl_time_variable_zonal_correction(25.0, &secular_later) == SL_OK);
    CHECK_NEAR(mean_now.c20 - secular_now.c20, mean_later.c20 - secular_later.c20, 1e-24);
    CHECK_NEAR(mean_now.c20 - secular_now.c20, -0.48416948e-3, 1e-15); /* IERS Table 6.2 */
    CHECK_NEAR(secular_later.c20, 11.6e-12 * 25.0, 1e-24);

    /* Head-on there is no tangential component to shear against, and a non-positive temperature is
     * a domain error rather than a NaN nobody notices. */
    CHECK_NEAR(sl_molecular_speed_ratio(7600.0, 1000.0, 0.016), 7.4549, 1e-3);
    CHECK(sl_sentman_drag_coefficients(7.4549, 1.0, 1.0, 300.0, 1000.0, &panel) == SL_OK);
    CHECK_NEAR(panel.shear, 0.0, 1e-12);
    CHECK_NEAR(panel.drag, 2.14822, 1e-4);
    CHECK(sl_sentman_drag_coefficients(7.4549, 1.0, 1.0, 0.0, 1000.0, &panel) == SL_ERR_DOMAIN);

    /* [flux, ap, flux^2, sqrt(ap), 1] on the mean of the two fluxes. */
    CHECK(sl_space_weather_drivers(150.0, 130.0, 9.0, drivers) == SL_OK);
    CHECK_NEAR(drivers[0], 140.0, 1e-12);
    CHECK_NEAR(drivers[2], 140.0 * 140.0, 1e-9);
    CHECK_NEAR(drivers[3], 3.0, 1e-12);
    CHECK_NEAR(drivers[4], 1.0, 1e-15);

    /* A header with no data rows is zero days and SL_OK, and a day the file does not hold is a
     * domain error rather than a silently zeroed array. */
    CHECK(sl_space_weather_days(csv, strlen(csv), NULL, 0, &len) == SL_OK);
    CHECK(len == 0);
    CHECK(sl_space_weather_ap_array(csv, strlen(csv), 2020, 1, 1, 0.0, ap_array) == SL_ERR_DOMAIN);
    CHECK(sl_density_data_samples(csv, strlen(csv), NULL, 0, &len) == SL_OK);
    CHECK(len == 0);

    /* The fit is two-call and always as long as the model has coefficients; a short buffer says so
     * and writes nothing, which is the branch the C++ wrapper sizes its way past. */
    samples[0].latitude_rad = -1.5;
    samples[0].local_solar_time_h = 1.0;
    samples[0].log_ratio = 0.25;
    samples[0].weight = 1.0;
    samples[1].latitude_rad = 1.5;
    samples[1].local_solar_time_h = 23.0;
    samples[1].log_ratio = -0.5;
    samples[1].weight = 1.0;
    len = 0;
    CHECK(sl_density_correction_fit(samples, 2, 1e-6, NULL, 0, &len) == SL_OK);
    CHECK(len == SL_DENSITY_CORRECTION_COEFFICIENTS);
    coefficients[0] = 12345.0;
    CHECK(sl_density_correction_fit(samples, 2, 1e-6, coefficients, 1, &len) ==
          SL_ERR_BUFFER_TOO_SMALL);
    CHECK(len == SL_DENSITY_CORRECTION_COEFFICIENTS);
    CHECK(coefficients[0] == 12345.0);
    CHECK(sl_density_correction_fit(samples, 2, 1e-6, coefficients,
                                    SL_DENSITY_CORRECTION_COEFFICIENTS, &len) == SL_OK);
    CHECK(sl_density_correction_factor(coefficients, SL_DENSITY_CORRECTION_COEFFICIENTS, -1.5, 1.0,
                                       &factor) == SL_OK);
    CHECK(factor > 0.0);
    CHECK(sl_density_correction_factor(coefficients, 4, -1.5, 1.0, &factor) ==
          SL_ERR_INVALID_ARGUMENT);

    /* Row-major lat_bin * n_lst + lst_bin, and the cells no sample reaches are exactly zero. */
    CHECK(sl_density_grid_snapshot(samples, 2, 2, 2, grid, 4, &len) == SL_OK);
    CHECK(len == 4);
    CHECK_NEAR(grid[0], 0.25, 1e-15);
    CHECK_NEAR(grid[3], -0.5, 1e-15);
    CHECK(grid[1] == 0.0);
    CHECK(grid[2] == 0.0);
    CHECK(sl_density_grid_snapshot(samples, 2, 0, 2, NULL, 0, &len) == SL_ERR_INVALID_ARGUMENT);

    /* The sub-satellite/target/centre triangle closes on pi, and a look angle past the horizon has
     * no ground point to report. */
    CHECK(sl_swath_geometry(SL_RE_EARTH, 700000.0, 0.35, &swath) == SL_OK);
    CHECK_NEAR(swath.earth_central_angle_rad + 0.35 + (PI / 2.0) + swath.elevation_rad, PI, 1e-12);
    CHECK_NEAR(swath.ground_range_m, SL_RE_EARTH * swath.earth_central_angle_rad, 1e-6);
    CHECK(sl_swath_geometry(SL_RE_EARTH, 700000.0, 1.4, &swath) == SL_ERR_DOMAIN);
}

/* The analytical propagator, whose round trip through mean elements is the identity at the epoch
 * and whose node regresses the way J2 says it must. */
static void the_analytical_propagator_flies_the_orbit_it_was_given(void) {
    sl_brouwer_lyddane_params p;
    sl_state out[2];
    double times[2] = {0.0, 86400.0};
    double speed = sqrt(SL_MU_EARTH / 7000000.0);
    double inclination = 51.6 * PI / 180.0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double h0x = 0.0;
    double h0y = 0.0;
    double h1x = 0.0;
    double h1y = 0.0;
    double drift = 0.0;

    p.initial.position.x = 7000000.0;
    p.initial.position.y = 0.0;
    p.initial.position.z = 0.0;
    p.initial.velocity.x = 0.0;
    p.initial.velocity.y = speed * cos(inclination);
    p.initial.velocity.z = speed * sin(inclination);
    p.mu = SL_MU_EARTH;
    p.earth_radius_m = SL_RE_EARTH;
    p.c20 = -1.08262668e-3;
    p.c30 = 2.53265649e-6;
    p.c40 = 1.61962159e-6;
    p.c50 = 2.27296083e-7;
    p.m2 = 0.0;

    CHECK(sl_brouwer_lyddane_propagate(&p, times, 2, out) == SL_OK);
    dx = out[0].position.x - p.initial.position.x;
    dy = out[0].position.y - p.initial.position.y;
    dz = out[0].position.z - p.initial.position.z;
    CHECK(sqrt((dx * dx) + (dy * dy) + (dz * dz)) < 10.0);

    /* Omega = atan2(h_x, -h_y) from the angular momentum, and under J2 it regresses. */
    h0x = (out[0].position.y * out[0].velocity.z) - (out[0].position.z * out[0].velocity.y);
    h0y = (out[0].position.z * out[0].velocity.x) - (out[0].position.x * out[0].velocity.z);
    h1x = (out[1].position.y * out[1].velocity.z) - (out[1].position.z * out[1].velocity.y);
    h1y = (out[1].position.z * out[1].velocity.x) - (out[1].position.x * out[1].velocity.z);
    drift = (atan2(h1x, -h1y) - atan2(h0x, -h0y)) * 180.0 / PI;
    CHECK(drift < -3.0);
    CHECK(drift > -8.0);

    p.mu = -1.0;
    CHECK(sl_brouwer_lyddane_propagate(&p, times, 2, out) == SL_ERR_DOMAIN);
}

/*
 * The rest of the force models, as a C consumer builds them: one handle type, one release
 * function, and a refusal that leaves the caller's pointer untouched rather than dangling.
 */
static void the_force_models_all_build_the_same_handle(void) {
    sl_state s0;
    sl_state at_end;
    sl_state ordinary;
    sl_state extended;
    sl_eop eop;
    sl_eop_sample series[2];
    sl_exponential_atmosphere atmosphere;
    sl_nrlmsise_drag drag;
    sl_sdtm1_drag sdtm1;
    double quiet_ap[120];
    double hours_ap60[48];
    double hours_speed[48];
    double hours_by[48];
    double hours_bz[48];
    sl_ephemeris *e = NULL;
    double speed = sqrt(SL_MU_EARTH / 7000000.0);
    double inclination = 51.6 * PI / 180.0;
    double arc = 600.0;
    double j2 = 1.08262668e-3;
    int i = 0;

    s0.position.x = 7000000.0;
    s0.position.y = 0.0;
    s0.position.z = 0.0;
    s0.velocity.x = 0.0;
    s0.velocity.y = speed * cos(inclination);
    s0.velocity.z = speed * sin(inclination);

    eop.epoch_tt_ns = 725760069184000000LL;
    eop.epoch_ut1_ns = 725760000000000000LL;
    eop.xp_rad = 1.0e-6;
    eop.yp_rad = 2.0e-6;
    eop.dx_rad = 0.0;
    eop.dy_rad = 0.0;

    atmosphere.ballistic_coefficient = 100.0;
    atmosphere.reference_density = 1e-12;
    atmosphere.reference_altitude_m = 500000.0;
    atmosphere.scale_height_m = 60000.0;
    atmosphere.earth_radius_m = SL_RE_EARTH;

    drag.doy = 172;
    drag.seconds_of_day = 29000.0;
    drag.ballistic_coefficient = 100.0;
    drag.f107 = 150.0;
    drag.f107a = 150.0;
    drag.ap = 4.0;
    drag.mu = SL_MU_EARTH;

    sdtm1.doy = 172.0;
    sdtm1.seconds_of_day = 29000.0;
    sdtm1.ballistic_coefficient = 0.02;
    sdtm1.f107 = 128.0;
    sdtm1.f107a = 141.0;
    sdtm1.mu = SL_MU_EARTH;
    for (i = 0; i < 120; ++i) {
        quiet_ap[i] = 5.0;
    }
    for (i = 0; i < 48; ++i) {
        hours_ap60[i] = 5.0;
        hours_speed[i] = 400.0;
        hours_by[i] = 2.0;
        hours_bz[i] = -3.0;
    }

    for (i = 0; i < 2; ++i) {
        series[i].t_seconds = i == 0 ? -arc : 2.0 * arc;
        series[i].xp_rad = eop.xp_rad;
        series[i].yp_rad = eop.yp_rad;
        series[i].dut1_seconds = -0.069184;
        series[i].dx_rad = 0.0;
        series[i].dy_rad = 0.0;
    }

    /* Each constructor is a different force model behind the same handle, and each answer is a
     * bounded orbit rather than a diverged one. */
    CHECK(sl_ephemeris_j2_dense(&s0, arc, SL_MU_EARTH, j2, SL_RE_EARTH, 1e-11, &e) == SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    CHECK(fabs(sqrt((at_end.position.x * at_end.position.x) +
                    (at_end.position.y * at_end.position.y) +
                    (at_end.position.z * at_end.position.z)) -
               7000000.0) < 50000.0);
    sl_ephemeris_close(e);
    e = NULL;

    CHECK(sl_ephemeris_drag(&s0, arc, 4, 0.0, &atmosphere, 1e-11, &e) == SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    sl_ephemeris_close(e);
    e = NULL;

    CHECK(sl_ephemeris_drag_nrlmsise00(&s0, arc, &eop, &drag, 1e-10, &e) == SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    sl_ephemeris_close(e);
    e = NULL;

    /* A 7000 km radius is above SDTM-1's envelope, and the arc is integrated rather than refused.
     * A null or empty history, or a value in it that is not finite, is refused, and the refusal
     * leaves the pointer untouched. */
    CHECK(sl_ephemeris_drag_sdtm1(&s0, arc, &eop, &sdtm1, quiet_ap, 120, 10800.0, 1e-10, &e) ==
          SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    sl_ephemeris_close(e);
    e = NULL;
    CHECK(sl_ephemeris_drag_sdtm1(&s0, arc, &eop, &sdtm1, NULL, 120, 10800.0, 1e-10, &e) ==
          SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_ephemeris_drag_sdtm1(&s0, arc, &eop, &sdtm1, quiet_ap, 0, 10800.0, 1e-10, &e) ==
          SL_ERR_INVALID_ARGUMENT);
    quiet_ap[3] = NAN;
    CHECK(sl_ephemeris_drag_sdtm1(&s0, arc, &eop, &sdtm1, quiet_ap, 120, 10800.0, 1e-10, &e) ==
          SL_ERR_INVALID_ARGUMENT);
    CHECK(e == NULL);

    /* The solar-wind field's drag takes four hourly series of one length. A null one, n = 0, or an
     * hour that is no hour -- a fill value for Bz -- is refused, and the pointer stays untouched. */
    CHECK(sl_ephemeris_drag_sdtm1_solar_wind(&s0, arc, &eop, &sdtm1, hours_ap60, hours_speed,
                                             hours_by, hours_bz, 48, 1e-10, &e) == SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    sl_ephemeris_close(e);
    e = NULL;
    CHECK(sl_ephemeris_drag_sdtm1_solar_wind(&s0, arc, &eop, &sdtm1, hours_ap60, NULL, hours_by,
                                             hours_bz, 48, 1e-10, &e) == SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_ephemeris_drag_sdtm1_solar_wind(&s0, arc, &eop, &sdtm1, hours_ap60, hours_speed,
                                             hours_by, hours_bz, 0, 1e-10,
                                             &e) == SL_ERR_INVALID_ARGUMENT);
    hours_bz[7] = -999.9;
    CHECK(sl_ephemeris_drag_sdtm1_solar_wind(&s0, arc, &eop, &sdtm1, hours_ap60, hours_speed,
                                             hours_by, hours_bz, 48, 1e-10,
                                             &e) == SL_ERR_INVALID_ARGUMENT);
    CHECK(e == NULL);

    CHECK(sl_ephemeris_geopotential_itrf(&s0, arc, 4, &eop, 1e-11, &e) == SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    sl_ephemeris_close(e);
    e = NULL;

    CHECK(sl_ephemeris_geopotential_tide(&s0, arc, 4, &eop, 0.0, 1.32712440018e20, 4.902800066e12,
                                         1e-11, &e) == SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    sl_ephemeris_close(e);
    e = NULL;

    CHECK(sl_ephemeris_j2_itrf(&s0, arc, SL_MU_EARTH, j2, SL_RE_EARTH, &eop, 1e-11, &e) == SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    sl_ephemeris_close(e);
    e = NULL;

    CHECK(sl_ephemeris_j2_itrf_series(&s0, arc, SL_MU_EARTH, j2, SL_RE_EARTH, eop.epoch_tt_ns, 0.0,
                                      series, 2, 1e-11, &e) == SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    sl_ephemeris_close(e);
    e = NULL;

    CHECK(sl_ephemeris_luni_solar(&s0, arc, 4, 0.0, 1e-7, 0.0, 1e-11, &e) == SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    sl_ephemeris_close(e);
    e = NULL;

    CHECK(sl_ephemeris_relativistic(&s0, arc, 0.0, SL_MU_EARTH, 1.32712440018e20, 1e-11, &e) ==
          SL_OK);
    CHECK(sl_ephemeris_at(e, arc, &at_end) == SL_OK);
    sl_ephemeris_close(e);
    e = NULL;

    /* A refused constructor writes nothing, so `e` is still the null it was set to -- which is
     * what makes `sl_ephemeris_close(e)` after a failure safe rather than a double free. */
    CHECK(sl_ephemeris_j2_itrf_series(&s0, arc, SL_MU_EARTH, j2, SL_RE_EARTH, eop.epoch_tt_ns, 0.0,
                                      series, 1, 1e-11, &e) == SL_ERR_INVALID_ARGUMENT);
    CHECK(e == NULL);
    CHECK(sl_ephemeris_geopotential_itrf(&s0, 0.0, 4, &eop, 1e-11, &e) == SL_ERR_INVALID_ARGUMENT);
    CHECK(e == NULL);
    sl_ephemeris_close(e); /* documented no-op on NULL */

    /* The arithmetic is the parameter: two runs of the same integrator, and the gap between them
     * is the round-off the ordinary answer carries. */
    CHECK(sl_propagate_j2_precision(&s0, SL_MU_EARTH, j2, SL_RE_EARTH, 1.0, 600,
                                    SL_PRECISION_DOUBLE, &ordinary) == SL_OK);
    CHECK(sl_propagate_j2_precision(&s0, SL_MU_EARTH, j2, SL_RE_EARTH, 1.0, 600,
                                    SL_PRECISION_DOUBLE_DOUBLE, &extended) == SL_OK);
    CHECK(fabs(ordinary.position.x - extended.position.x) < 1e-3);
    /* Casting past the enum is the point: C is the only rung that can hand the library a tag
     * outside the enumeration, and that it is refused rather than defaulted is the property.
     * NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) */
    CHECK(sl_propagate_j2_precision(&s0, SL_MU_EARTH, j2, SL_RE_EARTH, 1.0, 10, (sl_precision)7,
                                    &ordinary) == SL_ERR_INVALID_ARGUMENT);
}

/*
 * The radar surface, as only C reaches it: the two-call protocol written out by hand for a
 * complex answer, the short-buffer branch, the total functions that answer NaN for a null, and a
 * domain refusal where the mathematics has no answer to give.
 */
static void the_radar_surface_answers_what_its_formulas_say(void) {
    static const size_t CHIRP = 32;
    sl_complex chirp[32];
    sl_complex compressed[128];
    sl_complex short_buffer[3];
    sl_complex pair[16];
    double coherence[16];
    double wrapped[64];
    double unwrapped[64];
    int8_t charges[49];
    size_t len = 0;
    size_t i = 0;
    size_t rows = 8;
    size_t cols = 8;
    double peak = 0.0;
    double expected = 0.0;
    double rate = 0.0;
    double migration = 0.0;
    double src = 0.0;
    sl_rda_geometry geometry;

    /* A chirp is all phase: the modulation moves the frequency, never the amplitude. */
    CHECK(sl_lfm_chirp(CHIRP, 5.0e11, 3.0e7, NULL, 0, &len) == SL_OK);
    CHECK(len == CHIRP);
    CHECK(sl_lfm_chirp(CHIRP, 5.0e11, 3.0e7, chirp, CHIRP, &len) == SL_OK);
    for (i = 0; i < CHIRP; ++i) {
        CHECK_NEAR(sqrt((chirp[i].re * chirp[i].re) + (chirp[i].im * chirp[i].im)), 1.0, 1e-12);
    }

    /* The short-buffer branch: the length is reported, the buffer is left alone. */
    short_buffer[0].re = 42.0;
    short_buffer[0].im = 42.0;
    len = 0;
    CHECK(sl_lfm_chirp(CHIRP, 5.0e11, 3.0e7, short_buffer, 3, &len) == SL_ERR_BUFFER_TOO_SMALL);
    CHECK(len == CHIRP);
    CHECK(short_buffer[0].re == 42.0 && short_buffer[0].im == 42.0);

    /* Correlating a signal with itself compresses to its own energy -- 32 unit samples. */
    CHECK(sl_matched_filter(chirp, CHIRP, chirp, CHIRP, NULL, 0, &len) == SL_OK);
    CHECK(len <= 128);
    CHECK(sl_matched_filter(chirp, CHIRP, chirp, CHIRP, compressed, len, &len) == SL_OK);
    for (i = 0; i < len; ++i) {
        double magnitude =
            sqrt((compressed[i].re * compressed[i].re) + (compressed[i].im * compressed[i].im));
        if (magnitude > peak) {
            peak = magnitude;
        }
    }
    CHECK_NEAR(peak, (double)CHIRP, 1e-6 * (double)CHIRP);

    /* An image interfered with itself is real-positive, and perfectly coherent with itself. */
    for (i = 0; i < 16; ++i) {
        pair[i].re = 2.0 * cos((double)i * 0.37);
        pair[i].im = 2.0 * sin((double)i * 0.37);
    }
    CHECK(sl_interferogram(pair, pair, 16, compressed, 128, &len) == SL_OK);
    CHECK(len == 16);
    for (i = 0; i < len; ++i) {
        CHECK(compressed[i].re > 0.0);
        CHECK_NEAR(compressed[i].im, 0.0, 1e-12);
    }
    CHECK(sl_coherence(pair, pair, 4, 4, 1, coherence, 16, &len) == SL_OK);
    CHECK(len == 16);
    for (i = 0; i < len; ++i) {
        CHECK_NEAR(coherence[i], 1.0, 1e-12);
    }

    /* A ramp under pi per sample wraps and unwraps back to itself, and has no residues at all. */
    for (i = 0; i < rows; ++i) {
        size_t j = 0;
        for (j = 0; j < cols; ++j) {
            double phase = (0.30 * (double)i) + (0.22 * (double)j);
            wrapped[(i * cols) + j] = sl_wrap_phase(phase);
            unwrapped[(i * cols) + j] = phase;
        }
    }
    CHECK(sl_residues(wrapped, rows, cols, charges, 49, &len) == SL_OK);
    CHECK(len == (rows - 1) * (cols - 1));
    for (i = 0; i < len; ++i) {
        CHECK(charges[i] == 0);
    }
    CHECK(sl_unwrap_phase_2d(wrapped, rows, cols, coherence, 16, &len) == SL_ERR_BUFFER_TOO_SMALL);
    CHECK(len == rows * cols);
    {
        double recovered[64];
        double offset = 0.0;
        CHECK(sl_unwrap_phase_2d(wrapped, rows, cols, recovered, 64, &len) == SL_OK);
        offset = recovered[0] - unwrapped[0];
        for (i = 0; i < len; ++i) {
            CHECK_NEAR(recovered[i] - unwrapped[i], offset, 1e-9);
        }
    }

    /* The rates are closed forms, and the two total ones answer NaN rather than trapping. */
    geometry.prf = 1700.0;
    geometry.velocity = 7100.0;
    geometry.wavelength = 0.055;
    geometry.r0_slant = 850000.0;
    geometry.range_spacing = 4.5;
    rate = sl_azimuth_fm_rate(&geometry);
    expected =
        2.0 * geometry.velocity * geometry.velocity / (geometry.wavelength * geometry.r0_slant);
    CHECK_NEAR(rate, expected, 1e-9 * expected);
    migration = sl_range_migration(600.0, &geometry) / sl_range_migration(300.0, &geometry);
    CHECK_NEAR(migration, 4.0, 1e-9);
    CHECK(isnan(sl_azimuth_fm_rate(NULL)));
    CHECK(isnan(sl_range_migration(1.0, NULL)));

    /* At the band centre the range-azimuth coupling vanishes, so there is no rate to report. */
    CHECK(sl_src_fm_rate(0.0, &geometry, &src) == SL_ERR_DOMAIN);
    CHECK(sl_src_fm_rate(400.0, &geometry, &src) == SL_OK);
    CHECK(src != 0.0);

    /* A flat profile never falls to half power, so it has no width -- a status, not a NaN. */
    for (i = 0; i < 16; ++i) {
        coherence[i] = 1.0;
    }
    CHECK(sl_resolution_3db_samples(coherence, 16, &peak) == SL_ERR_DOMAIN);
}

/*
 * The optical surface, as a C consumer reaches it: a struct carrying a borrowed pointer, the
 * two-call protocol for a struct answer, the totality rule on the kernels and the overlap, and a
 * degenerate rig refused with a status rather than a plausible number.
 */
static void the_optical_surface_answers_what_its_geometry_says(void) {
    static const double FOCAL = 200.0;
    sl_camera cameras[2];
    double pixels[4];
    double world[3] = {1.2, -0.7, 14.0};
    sl_vec3 point;
    sl_bbox boxes[3];
    sl_bbox decoded;
    sl_bbox anchors[64];
    double scores[3] = {0.9, 0.8, 0.7};
    double zero_delta[4] = {0.0, 0.0, 0.0, 0.0};
    double scales[2] = {4.0, 8.0};
    double ratios[3] = {0.5, 1.0, 2.0};
    size_t kept[3];
    size_t len = 0;
    size_t i = 0;
    sl_atmospheric_coupling coupling;
    double ms[9] = {0.20, 0.31, 0.15, 0.35, 0.22, 0.51, 0.11, 0.47, 0.26};
    double pan[3] = {0.55, 0.61, 0.48};
    double sharpened[9];
    double ross = 0.0;
    double li = 0.0;
    double combined = 0.0;

    for (i = 0; i < 2; ++i) {
        cameras[i].rotation[0] = 1.0;
        cameras[i].rotation[1] = 0.0;
        cameras[i].rotation[2] = 0.0;
        cameras[i].rotation[3] = 0.0;
        cameras[i].rotation[4] = 1.0;
        cameras[i].rotation[5] = 0.0;
        cameras[i].rotation[6] = 0.0;
        cameras[i].rotation[7] = 0.0;
        cameras[i].rotation[8] = 1.0;
        cameras[i].center[0] = (i == 0) ? 0.0 : 3.0;
        cameras[i].center[1] = 0.0;
        cameras[i].center[2] = 0.0;
        cameras[i].focal = FOCAL;
        cameras[i].principal[0] = 50.0;
        cameras[i].principal[1] = 40.0;
        pixels[2 * i] = (FOCAL * (world[0] - cameras[i].center[0]) / world[2]) + 50.0;
        pixels[(2 * i) + 1] = (FOCAL * world[1] / world[2]) + 40.0;
    }

    /* Triangulation inverts the projection it was given. */
    CHECK(sl_triangulate(cameras, pixels, 2, &point) == SL_OK);
    CHECK_NEAR(point.x, world[0], 1e-9);
    CHECK_NEAR(point.y, world[1], 1e-9);
    CHECK_NEAR(point.z, world[2], 1e-9);

    /* One camera twice is one ray twice, and a ray does not intersect itself in a point. */
    cameras[1] = cameras[0];
    pixels[2] = pixels[0];
    pixels[3] = pixels[1];
    CHECK(sl_triangulate(cameras, pixels, 2, &point) == SL_ERR_DOMAIN);
    CHECK(sl_triangulate(cameras, pixels, 1, &point) == SL_ERR_INVALID_ARGUMENT);

    /* The overlap is a ratio, so a box against itself is one and against a distant one is zero. */
    boxes[0].x1 = 0.0;
    boxes[0].y1 = 0.0;
    boxes[0].x2 = 2.0;
    boxes[0].y2 = 2.0;
    boxes[1].x1 = 1.0;
    boxes[1].y1 = 0.0;
    boxes[1].x2 = 3.0;
    boxes[1].y2 = 2.0;
    boxes[2].x1 = 10.0;
    boxes[2].y1 = 10.0;
    boxes[2].x2 = 12.0;
    boxes[2].y2 = 12.0;
    CHECK_NEAR(sl_box_iou(&boxes[0], &boxes[0]), 1.0, 1e-15);
    CHECK_NEAR(sl_box_iou(&boxes[0], &boxes[2]), 0.0, 1e-15);
    CHECK_NEAR(sl_box_iou(&boxes[0], &boxes[1]), 1.0 / 3.0, 1e-15);
    CHECK(isnan(sl_box_iou(NULL, &boxes[0])));

    /* Suppression drops the weaker of an overlapping pair and leaves the distant one alone. */
    CHECK(sl_nms(boxes, scores, 3, 0.3, NULL, 0, &len) == SL_OK);
    CHECK(len == 2);
    CHECK(sl_nms(boxes, scores, 3, 0.3, kept, 3, &len) == SL_OK);
    CHECK(kept[0] == 0 && kept[1] == 2);

    /* The grid is the product of its four dimensions, and a zero delta is the identity. */
    CHECK(sl_generate_anchors(2, 2, 16.0, scales, 2, ratios, 3, NULL, 0, &len) == SL_OK);
    CHECK(len == (size_t)2 * 2 * 2 * 3);
    CHECK(sl_generate_anchors(2, 2, 16.0, scales, 2, ratios, 3, anchors, 64, &len) == SL_OK);
    CHECK(sl_decode_box(&anchors[5], zero_delta, &decoded) == SL_OK);
    CHECK_NEAR(decoded.x1, anchors[5].x1, 1e-12);
    CHECK_NEAR(decoded.y2, anchors[5].y2, 1e-12);

    /* The reflectance is exactly its two kernels weighted -- no third term hiding in it. */
    ross = sl_ross_thick_kernel(0.3, 0.5, 0.7);
    li = sl_li_sparse_reciprocal_kernel(0.3, 0.5, 0.7);
    combined = sl_brdf_reflectance(0.12, 0.03, 0.008, 0.3, 0.5, 0.7);
    CHECK_NEAR(combined, 0.12 + (0.03 * ross) + (0.008 * li), 1e-15);

    /* The two 6S directions undo one another, and a null coupling is a NaN, not a guess. */
    coupling.path_reflectance = 0.08;
    coupling.transmittance_down = 0.83;
    coupling.transmittance_up = 0.87;
    coupling.spherical_albedo = 0.11;
    CHECK_NEAR(sl_atmospheric_surface_from_toa(&coupling,
                                               sl_atmospheric_toa_from_surface(&coupling, 0.42)),
               0.42, 1e-12);
    CHECK(isnan(sl_atmospheric_toa_from_surface(NULL, 0.3)));

    /* Brovey is a ratio substitution, so the sharpened bands sum back to the panchromatic. */
    CHECK(sl_brovey_pan_sharpen(ms, 3, 3, pan, sharpened, 9, &len) == SL_OK);
    CHECK(len == 9);
    for (i = 0; i < 3; ++i) {
        CHECK_NEAR(sharpened[i] + sharpened[3 + i] + sharpened[6 + i], pan[i], 1e-12);
    }
}

/* The cloud-mask family is here in every build, and says which build flag it is missing when the
 * library was compiled without `onnx`. That is the property worth testing from C: a consumer links
 * one library and must not have to discover at load time which Cargo features it was built with.
 * The gate builds the default library, so the expected answer is `SL_ERR_UNSUPPORTED`; with the
 * feature on the same call opens a model and the Rust suite checks the mask it produces. */
static void the_cloud_mask_symbols_are_here_whatever_the_build(void) {
    const uint8_t model[4] = {0, 0, 0, 0};
    const float tile[3] = {0.0f, 0.0f, 0.0f};
    sl_cloud_masker *masker = NULL;
    size_t len = 7;
    sl_status s = sl_cloud_masker_open(model, sizeof model, 0.5f, &masker);

    CHECK(s == SL_ERR_UNSUPPORTED || s == SL_OK);
    if (s == SL_ERR_UNSUPPORTED) {
        CHECK(masker == NULL);
    } else {
        sl_cloud_masker_close(masker);
    }

    /* A null is refused before the feature is consulted, and refusing writes neither out-parameter. */
    CHECK(sl_cloud_masker_open(NULL, sizeof model, 0.5f, &masker) == SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_cloud_masker_mask(NULL, tile, 1, 1, 3, NULL, 0, &len) == SL_ERR_INVALID_ARGUMENT);
    CHECK(len == 7);
    /* Closing nothing is not an error, so a failed open needs no special path. */
    sl_cloud_masker_close(NULL);
}

/*
 * A pair of relative-motion samplers, as a C consumer writes them: a bare function pointer and a
 * `void *`, with no closure to capture anything in.
 *
 * The trajectory is `10000*sin(2*pi*t/600)` along track against a constant 500 m across it, so the
 * minima sit where the sine crosses zero -- at 300, 600 and 900 s -- each at 500 m and at the
 * relative speed `10000*2*pi/600`. Nothing is read back from the scan.
 */
static int drifting_state(void *user_data, double t, sl_state *out) {
    const double rate = 2.0 * PI / 600.0;
    (void)user_data;
    out->position.x = 10000.0 * sin(rate * t);
    out->position.y = 500.0;
    out->position.z = 0.0;
    out->velocity.x = 10000.0 * rate * cos(rate * t);
    out->velocity.y = 0.0;
    out->velocity.z = 0.0;
    return SL_OK;
}

static int refusing_state(void *user_data, double t, sl_state *out) {
    (void)user_data;
    (void)t;
    (void)out;
    return SL_ERR_COMPUTE;
}

/* The conjunction's own relative motion, propagated the way the 2-D model assumes it moves. */
static int rectilinear_sample(void *user_data, double t, sl_relative_sample *out) {
    const sl_conjunction *g = (const sl_conjunction *)user_data;
    int i;

    out->position.x = g->relative_position.x + (g->relative_velocity.x * t);
    out->position.y = g->relative_position.y + (g->relative_velocity.y * t);
    out->position.z = g->relative_position.z + (g->relative_velocity.z * t);
    out->velocity = g->relative_velocity;
    for (i = 0; i < 9; ++i) {
        out->covariance[i] = g->primary_covariance[i] + g->secondary_covariance[i];
    }
    return SL_OK;
}

/* A head-on conjunction with isotropic covariances, the miss vector across the relative track. */
static sl_conjunction conjunction_case(double miss) {
    sl_conjunction c;
    int i;

    c.relative_position.x = 0.6 * miss;
    c.relative_position.y = 0.0;
    c.relative_position.z = 0.8 * miss;
    c.relative_velocity.x = 0.0;
    c.relative_velocity.y = 14000.0;
    c.relative_velocity.z = 0.0;
    for (i = 0; i < 9; ++i) {
        c.primary_covariance[i] = (i % 4 == 0) ? 1e4 : 0.0;
        c.secondary_covariance[i] = c.primary_covariance[i];
    }
    c.primary_radius = 5.0;
    c.secondary_radius = 5.0;
    return c;
}

/*
 * The radiometric surface: two observables of one link that must sum to the turn-around ratio, and
 * the geometries whose answers are their own definition.
 */
static void the_radiometric_observables_answer_what_their_geometry_says(void) {
    const double light = 299792458.0;
    const double exciter = 7.15e9;
    const double turnaround = 880.0 / 749.0; /* M(X, X) from Moyer's table 13-1 */
    sl_ramp_segment flat;
    sl_doppler_link link;
    sl_state sat;
    sl_state remote;
    sl_terminal tx;
    sl_terminal rx;
    sl_vec3 near_sun;
    sl_vec3 far_sun;
    double bistatic = 0.0;
    double tdoa = 0.0;
    double swapped = 0.0;
    double counted = 0.0;
    double instantaneous = 0.0;
    double wide = 0.0;
    double s_band = 0.0;
    double group = 0.0;
    double phase = 0.0;
    double shapiro = 0.0;

    flat.start = 0.0;
    flat.end = 1e4;
    flat.start_frequency = exciter;
    flat.rate = 0.0;

    sat.position.x = 4200000.0;
    sat.position.y = 5300000.0;
    sat.position.z = 1600000.0;
    sat.velocity.x = -5200.0;
    sat.velocity.y = 3900.0;
    sat.velocity.z = 900.0;

    tx.position.x = 4194303.0;
    tx.position.y = 172460.0;
    tx.position.z = 4780100.0;
    tx.velocity.x = 0.0;
    tx.velocity.y = 0.0;
    tx.velocity.z = 0.0;
    tx.epoch = 0.0;
    rx = tx;
    rx.position.x = -3950200.0;
    rx.position.y = 2522400.0;
    rx.position.z = 4311700.0;
    rx.epoch = 0.04;

    /* Two legs of one path, so the bistatic range is bounded by twice the longer of them. */
    CHECK(sl_bistatic_range(&sat, &tx, 0.02, &rx, &bistatic) == SL_OK);
    CHECK(bistatic > 0.0 && bistatic < 4.0e7);

    /* An arrival difference is a projected baseline over c, so it cannot exceed the baseline. Both
     * terminals receive after the emission, which the geometry requires and checks. */
    tx.epoch = 0.03;
    CHECK(sl_tdoa(&sat, 0.0, &tx, &rx, &tdoa) == SL_OK);
    CHECK(fabs(tdoa) < 1.5e7 / light);
    /* Swapping the receivers negates it, which is what says the two arguments are not one. */
    CHECK(sl_tdoa(&sat, 0.0, &rx, &tx, &swapped) == SL_OK);
    CHECK_NEAR(swapped, -tdoa, 1e-15);

    /* The epochs must increase along the signal path, and a receiver that hears the bounce early
     * is a malformed request rather than a negative leg. */
    tx.epoch = 0.0;
    rx.epoch = -1.0;
    CHECK(sl_bistatic_range(&sat, &tx, 0.02, &rx, &bistatic) == SL_ERR_INVALID_ARGUMENT);
    rx.epoch = 0.04;

    /* Moyer's conversion is `221/2d`: `d = 749` for X and `221` for S, so the ratio is `749/221`. */
    CHECK(sl_dsn_sequential_range(&flat, 1, SL_BAND_X, 100.0, 101.0, 26, &wide) == SL_OK);
    CHECK(sl_dsn_sequential_range(&flat, 1, SL_BAND_S, 100.0, 101.0, 26, &s_band) == SL_OK);
    CHECK_NEAR(wide, (221.0 / 1498.0) * exciter, 1.0);
    CHECK_NEAR(s_band / wide, 749.0 / 221.0, 1e-12);

    link.uplink = SL_BAND_X;
    link.downlink = SL_BAND_X;
    link.reference_frequency = exciter;
    link.reference_band = SL_BAND_X;
    link.count_interval = 60.0;
    link.subtracted = true;
    /* Over a flat exciter the two Doppler observables sum to `M * f_reference` exactly, which is
     * one equation across two entry points and the reason both are called here. */
    CHECK(sl_dsn_averaged_doppler(&flat, 1, &link, 100.0, 160.0, &counted) == SL_OK);
    CHECK(sl_dsn_measured_frequency(&flat, 1, SL_BAND_X, SL_BAND_X, 100.0, 1.0, &instantaneous) ==
          SL_OK);
    CHECK_NEAR(counted + instantaneous, turnaround * exciter, 1e-2);

    /* A band tag the header does not define is refused rather than taken as S. */
    CHECK(sl_dsn_sequential_range(&flat, 1, (sl_band)7, 100.0, 101.0, 26, &wide) ==
          SL_ERR_INVALID_ARGUMENT);

    /* Ten solar radii off the Sun-probe line: the group delay is positive and the phase advance is
     * its exact negative, which is the whole content of the two signatures. */
    near_sun.x = -1.5e11;
    near_sun.y = 6.96e9;
    near_sun.z = 0.0;
    far_sun.x = 2.3e11;
    far_sun.y = 6.96e9;
    far_sun.z = 0.0;
    CHECK(sl_solar_corona_delay(&near_sun, &far_sun, 8.4e9, true, NULL, 0, &group) == SL_OK);
    CHECK(sl_solar_corona_delay(&near_sun, &far_sun, 8.4e9, false, NULL, 0, &phase) == SL_OK);
    CHECK(group > 0.0);
    CHECK_NEAR(phase, -group, 0.0);
    /* A ray through the photosphere has no honest correction, so it is a domain refusal. */
    near_sun.y = 0.0;
    far_sun.y = 0.0;
    CHECK(sl_solar_corona_delay(&near_sun, &far_sun, 8.4e9, true, NULL, 0, &group) ==
          SL_ERR_DOMAIN);

    /* Shapiro is symmetric in its endpoints and linear in the central mass. */
    shapiro = sl_shapiro_delay(6.878e6, 2.6561e7, 2.0e7, SL_MU_EARTH);
    CHECK_NEAR(sl_shapiro_delay(2.6561e7, 6.878e6, 2.0e7, SL_MU_EARTH), shapiro, 0.0);
    CHECK_NEAR(sl_shapiro_delay(6.878e6, 2.6561e7, 2.0e7, 2.0 * SL_MU_EARTH), 2.0 * shapiro, 0.0);
    CHECK(shapiro > 0.001 && shapiro < 0.05);

    /* Totality: the two inter-satellite observables answer NaN for a null rather than trapping. */
    remote = sat;
    remote.position.x += 12000.0;
    CHECK(sl_inter_satellite_range(&sat, &remote) > 11000.0);
    CHECK(isnan(sl_inter_satellite_range(NULL, &remote)));
    CHECK(isnan(sl_inter_satellite_range_rate(&sat, NULL)));
}

/* Re-entry, survival and the printed element set -- the three answers #310 and #311 produce. */
static void the_lifetime_and_element_set_answer_in_their_own_units(void) {
    const double zonals[3] = {0.0, 0.0, -SL_J2_EARTH};
    const double scales[3] = {0.5, 1.0, 2.0};
    const double quantiles[3] = {0.0, 0.5, 1.0};
    const double parameters[6] = {15.72125391, 0.0006703, 51.6416, 247.4627, 130.5360, 325.0288};
    double initial[6];
    double seconds[3];
    double slope = 0.0;
    sl_decay_params p;
    sl_lifetime low;
    sl_lifetime high;
    sl_lifetime_ensemble ensemble;
    sl_utc_datetime epoch;
    sl_tle tle;

    sl_decay_params_init(&p);
    p.area_over_mass = 0.02;
    p.reference_density = 2e-11;
    p.reference_radius = SL_RE_EARTH + 300e3;
    p.scale_height = 60e3;
    p.horizon = 400.0 * 86400.0;
    p.tolerance = 3600.0;
    /* `_init` leaves the four an atmosphere cannot be guessed for as NaN, so a caller who forgets
     * one is refused rather than answered on a silent zero. */
    CHECK(isnan(p.horizon) == 0);
    CHECK(p.chunk == 86400.0);
    CHECK(p.density_scale == 1.0);

    initial[0] = SL_RE_EARTH + 300e3;
    initial[1] = 0.0;
    initial[2] = 0.0;
    initial[3] = 0.0;
    initial[4] = 0.0;
    initial[5] = 0.0;

    CHECK(sl_orbital_lifetime(&p, initial, zonals, 3, &low) == SL_OK);
    CHECK(low.outcome == SL_LIFETIME_REENTRY);
    CHECK(low.seconds > 0.0 && low.seconds < p.horizon);
    CHECK(low.perigee_altitude_m <= p.reentry_altitude + 1e3);

    /* 700 km higher is `exp(-700/60)` of the density, so the same object outlives the horizon --
     * and the second field is then the horizon rather than a decay epoch nobody observed. */
    initial[0] = SL_RE_EARTH + 1000e3;
    CHECK(sl_orbital_lifetime(&p, initial, zonals, 3, &high) == SL_OK);
    CHECK(high.outcome == SL_LIFETIME_SURVIVES);
    CHECK_NEAR(high.seconds, p.horizon, 0.0);
    initial[0] = SL_RE_EARTH + 300e3;

    CHECK(sl_orbital_lifetime_ensemble(&p, initial, zonals, 3, scales, 3, quantiles, 3, seconds,
                                       &ensemble) == SL_OK);
    CHECK(ensemble.members == 3 && ensemble.survivors == 0 && ensemble.unresolved == 0);
    CHECK(seconds[0] <= seconds[1] && seconds[1] <= seconds[2]);

    /* `T ~ 1/(S/m)` to first order, so the log-log slope is -1 and its sign is the point. */
    CHECK(sl_ballistic_log_sensitivity(&p, initial, zonals, 3, 0.05, &slope) == SL_OK);
    CHECK_NEAR(slope, -1.0, 0.2);

    /* Fewer than three zonals is refused here rather than asserted on inside the field model,
     * where an assertion across the ABI would abort this process instead of returning. */
    CHECK(sl_orbital_lifetime(&p, initial, zonals, 2, &low) == SL_ERR_INVALID_ARGUMENT);

    epoch.year = 2008;
    epoch.month = 9;
    epoch.day = 20;
    epoch.hour = 12;
    epoch.minute = 25;
    epoch.second = 12.25;
    CHECK(sl_tle_lines(&epoch, 25544, parameters, -1.1606e-4, "1998-067A", 292, 56353, &tle) ==
          SL_OK);
    /* A column layout has no tolerance: a field one column out still parses and is another orbit. */
    CHECK(strcmp(tle.line1,
                 "1 25544U 98067A   08264.51750289  .00000000  00000-0 -11606-3 0  2921") == 0);
    CHECK(strcmp(tle.line2,
                 "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537") == 0);
    CHECK(strlen(tle.line1) == 69);
    CHECK(strlen(tle.line2) == 69);
    /* Five numeric columns cannot name this satellite, and it is refused rather than truncated. */
    CHECK(sl_tle_lines(&epoch, 270123, parameters, 0.0, NULL, 0, 0, &tle) ==
          SL_ERR_INVALID_ARGUMENT);
}

/* The carrier observables: the combinations, the two detectors and the ambiguity search. */
static void the_gnss_observables_combine_and_detect(void) {
    const double f1 = 1575.42e6;
    const double f2 = 1227.60e6;
    const double geometric = 22100000.0;
    const double i1 = 4.5;
    const double flt[3] = {5.45, 3.10, 2.97};
    const double covariance[9] = {6.290, 5.978, 0.544, 5.978, 6.292, 2.340, 0.544, 2.340, 6.288};
    double times[40];
    double values[40];
    double transformation[9];
    double conditional[3];
    double norms[2];
    double squared_norm = 0.0;
    double i2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    double l1 = 0.0;
    double l2 = 0.0;
    double free_of_ionosphere = 0.0;
    double wl = 0.0;
    double nl = 0.0;
    double cycles = 0.0;
    int64_t bootstrapped[3];
    int64_t best[6];
    size_t resolved = 0;
    sl_lanes lanes;
    sl_slip_verdict verdicts[40];
    int i;
    int slipped = 0;

    CHECK(sl_lane_wavelengths(f1, f2, &lanes) == SL_OK);
    /* The published GPS L1/L2 lanes: the wide one is why its ambiguity is resolved first. */
    CHECK_NEAR(lanes.wide_lane, 0.8619, 1e-4);
    CHECK_NEAR(lanes.narrow_lane, 0.1070, 1e-4);
    /* Two equal frequencies have no lane, and that is a refusal rather than an infinity. */
    CHECK(sl_lane_wavelengths(f1, f1, &lanes) == SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_lane_wavelengths(f1, f2, &lanes) == SL_OK);

    /* The RINEX letters, and the FDMA hole that answers NaN rather than being absent. */
    CHECK_NEAR(sl_carrier_frequency('G', '1'), f1, 0.0);
    CHECK(isnan(sl_carrier_frequency('R', '1')));

    i2 = i1 * (f1 / f2) * (f1 / f2);
    p1 = geometric + i1;
    p2 = geometric + i2;
    l1 = (geometric - i1) + (lanes.lambda1 * 1234.0);
    l2 = (geometric - i2) + (lanes.lambda2 * 987.0);

    /* The combination exists to remove exactly this, and it removes it to the last micron. */
    CHECK(sl_ionosphere_free(f1, f2, p1, p2, &free_of_ionosphere) == SL_OK);
    CHECK_NEAR(free_of_ionosphere, geometric, 1e-6);
    /* On code the two lanes carry the ionosphere with opposite sign and equal magnitude, so they
     * straddle the geometric range and their sum is exactly twice it. */
    CHECK(sl_wide_lane(f1, f2, p1, p2, &wl) == SL_OK);
    CHECK(sl_narrow_lane(f1, f2, p1, p2, &nl) == SL_OK);
    CHECK(wl < geometric && geometric < nl);
    CHECK_NEAR(wl + nl, 2.0 * geometric, 1e-6);
    /* Everything geometric and everything dispersive cancels, so what is left is `N1 - N2`. */
    CHECK(sl_melbourne_wubbena(f1, f2, p1, p2, l1, l2, true, &cycles) == SL_OK);
    CHECK_NEAR(cycles, 1234.0 - 987.0, 1e-6);
    CHECK_NEAR(sl_geometry_free(p1, p2), i1 - i2, 1e-8);
    CHECK_NEAR(sl_graphic(p1, l1), 0.5 * (p1 + l1), 0.0);

    /* A series drifting 0.02 m a sample with a 0.30 m step at index 20: a slip the 0.10 m
     * threshold sees and the drift does not trip. */
    for (i = 0; i < 40; ++i) {
        times[i] = 30.0 * (double)i;
        values[i] = (0.02 * (double)i) + (i >= 20 ? 0.30 : 0.0);
    }
    CHECK(sl_detect_cycle_slips_geometry_free(times, values, 40, 0.10, 120.0, verdicts) == SL_OK);
    CHECK(verdicts[0].verdict == SL_SLIP_ARC_START);
    /* Nothing was tested at an arc start, and the statistic says so rather than reading zero. */
    CHECK(isnan(verdicts[0].statistic));
    CHECK(verdicts[19].verdict == SL_SLIP_CONTINUOUS);
    CHECK(verdicts[20].verdict == SL_SLIP_SLIPPED);
    CHECK_NEAR(verdicts[20].statistic, 0.32, 1e-12);
    for (i = 0; i < 40; ++i) {
        slipped += (verdicts[i].verdict == SL_SLIP_SLIPPED) ? 1 : 0;
    }
    CHECK(slipped == 1);

    /* The fit tests nothing until its window fills, and clears the window after a slip -- so the
     * two detectors disagree on the sample after the step, which is why there are two. */
    CHECK(sl_detect_cycle_slips_polynomial(times, values, 40, 1, 5, 0.10, 120.0, verdicts) ==
          SL_OK);
    CHECK(verdicts[4].verdict == SL_SLIP_ARC_START);
    CHECK(verdicts[5].verdict == SL_SLIP_CONTINUOUS);
    CHECK(verdicts[20].verdict == SL_SLIP_SLIPPED);
    CHECK(verdicts[21].verdict == SL_SLIP_ARC_START);
    /* `window < order + 2` leaves no residual to test, so it is refused. */
    CHECK(sl_detect_cycle_slips_polynomial(times, values, 40, 3, 4, 0.10, 120.0, verdicts) ==
          SL_ERR_INVALID_ARGUMENT);

    CHECK(sl_ambiguity_decorrelation(flt, 3, covariance, transformation, conditional) == SL_OK);
    /* `Z` is unimodular, so every entry is an integer and every conditional variance is positive.
     * That is the one property no incorrect reduction has. */
    for (i = 0; i < 9; ++i) {
        CHECK(fabs(transformation[i] - floor(transformation[i] + 0.5)) < 1e-9);
    }
    for (i = 0; i < 3; ++i) {
        CHECK(conditional[i] > 0.0);
    }
    CHECK(sl_ambiguity_bootstrap(flt, 3, covariance, bootstrapped, &squared_norm) == SL_OK);
    CHECK(sl_resolve_ambiguities(flt, 3, covariance, 2, best, norms, &resolved) == SL_OK);
    CHECK(resolved == 2);
    CHECK(norms[0] <= norms[1]);
    /* The search is at least as good as the bootstrap, which is the whole reason it exists. */
    CHECK(norms[0] <= squared_norm);
    /* A covariance that is not one is refused before anything is factorised. */
    conditional[0] = -1.0;
    CHECK(sl_ambiguity_decorrelation(flt, 1, conditional, transformation, conditional) ==
          SL_ERR_INVALID_ARGUMENT);
}

/*
 * The conjunction surface, including the two callback scans and the short-buffer branch of the
 * two-call protocol -- which the C++ wrapper sizes its way past and therefore never takes.
 */
static void the_conjunction_surface_estimates_and_screens(void) {
    sl_conjunction geometry = conjunction_case(50.0);
    sl_conjunction centred = conjunction_case(0.0);
    sl_conjunction far = conjunction_case(300.0);
    sl_probability_estimates estimates;
    sl_probability_estimates other;
    sl_usage_violations violations;
    sl_close_approach found[4];
    sl_close_approach few[2];
    double exact = 0.0;
    double rate = 2.0 * PI / 600.0;
    size_t sized = 0;
    size_t len = 0;
    size_t few_len = 0;
    int i;

    CHECK(sl_collision_probability(&geometry, &estimates) == SL_OK);
    /* Three routes to one integral agree; the fourth is a bound over a scaling of the covariance
     * rather than a fourth estimate of the same thing, so it sits above all three. */
    CHECK(fabs(estimates.patera - estimates.foster) < 0.02 * estimates.foster);
    CHECK(fabs(estimates.chan - estimates.foster) < 0.02 * estimates.foster);
    CHECK(estimates.maximum >= estimates.foster);
    /* Head-on with an isotropic joint covariance `2*sigma^2` and no miss, the integral is
     * elementary: `1 - exp(-R^2/2s^2)`. The one geometry whose answer is not a quadrature. */
    exact = 1.0 - exp(-(10.0 * 10.0) / (2.0 * 2.0 * 100.0 * 100.0));
    CHECK(sl_collision_probability(&centred, &other) == SL_OK);
    CHECK_NEAR(other.foster, exact, 2e-4 * exact);
    /* Further away is less likely, which is the one monotonicity the whole model rests on. */
    CHECK(sl_collision_probability(&far, &other) == SL_OK);
    CHECK(other.foster < estimates.foster);
    /* A degenerate encounter velocity has no encounter plane: a domain refusal, not a zero. */
    centred.relative_velocity.y = 0.0;
    CHECK(sl_collision_probability(&centred, &other) == SL_ERR_DOMAIN);

    /* Feed the domain check its own assumption: rectilinear samples report a rectilinear
     * encounter, and the correction that implies is unity. */
    CHECK(sl_conjunction_usage_violations(&geometry, 5400.0, rectilinear_sample, &geometry,
                                          &violations) == SL_OK);
    CHECK(!violations.any);
    CHECK(!violations.offset && !violations.extended && !violations.inaccuracy);
    CHECK_NEAR(violations.correction_factor, 1.0, 1e-6);
    /* A sampler that refuses is reported as a refusal, not as a check that found nothing wrong. */
    CHECK(sl_conjunction_usage_violations(&geometry, 5400.0, NULL, NULL, &violations) ==
          SL_ERR_INVALID_ARGUMENT);

    /* The two-call protocol, written out: ask the length, then fill. Sizing runs the scan, so the
     * two calls have to agree on how many minima there are. */
    CHECK(sl_closest_approaches(drifting_state, NULL, 10.0, 1000.0, 30.0, 1e-4, NULL, 0, &sized) ==
          SL_OK);
    CHECK(sized == 3);
    CHECK(sl_closest_approaches(drifting_state, NULL, 10.0, 1000.0, 30.0, 1e-4, found, 4, &len) ==
          SL_OK);
    CHECK(len == sized);
    for (i = 0; i < (int)len; ++i) {
        CHECK_NEAR(found[i].tca, 300.0 * (double)(i + 1), 0.5);
        CHECK_NEAR(found[i].distance, 500.0, 1e-3);
        CHECK_NEAR(found[i].relative_speed, 10000.0 * rate, 1e-3);
    }
    /* The short-buffer branch: a refusal that still reports the length, so the caller can allocate
     * and ask again rather than silently receiving two of three. */
    CHECK(sl_closest_approaches(drifting_state, NULL, 10.0, 1000.0, 30.0, 1e-4, few, 2, &few_len) ==
          SL_ERR_BUFFER_TOO_SMALL);
    CHECK(few_len == 3);
    /* A sampler that cannot cover the span ends the scan through the status rather than through an
     * empty answer -- the difference between "nothing happened" and "we could not look". */
    CHECK(sl_closest_approaches(refusing_state, NULL, 10.0, 1000.0, 30.0, 1e-4, found, 4, &len) ==
          SL_ERR_COMPUTE);
    /* An unusable scan request is refused by the scan itself. */
    CHECK(sl_closest_approaches(drifting_state, NULL, 10.0, 1000.0, -30.0, 1e-4, NULL, 0, &sized) ==
          SL_ERR_INVALID_ARGUMENT);

    /* Filter 1 of an all-on-all screen: a certainty in one direction, nothing in the other, and a
     * NaN examined rather than silently discarded. Total, so there is no status to read. */
    CHECK(sl_radially_separated(SL_RE_EARTH + 400e3, SL_RE_EARTH + 420e3, SL_RE_EARTH + 800e3,
                                SL_RE_EARTH + 900e3, 5000.0));
    CHECK(!sl_radially_separated(SL_RE_EARTH + 400e3, SL_RE_EARTH + 420e3, SL_RE_EARTH + 800e3,
                                 SL_RE_EARTH + 900e3, 400e3));
    CHECK(!sl_radially_separated(SL_RE_EARTH + 400e3, SL_RE_EARTH + 900e3, SL_RE_EARTH + 800e3,
                                 SL_RE_EARTH + 420e3, 1.0));
    CHECK(!sl_radially_separated(NAN, SL_RE_EARTH + 420e3, SL_RE_EARTH + 800e3, SL_RE_EARTH + 900e3,
                                 5000.0));
}

/*
 * The closed-loop control surface, exercised the way only a plain C consumer reaches it.
 *
 * The Rust suite already proves each entry point is its crate function, and `smoke.cpp` proves the
 * C++ wrapper is a faithful reading of the header. What is left over for this file is what neither
 * of those touches: the two-call length protocol written out by hand including the short-buffer
 * branch, the null-pointer refusals the C++ wrapper cannot express because it never has a null to
 * pass, and the three walls that are the ABI's own invention rather than the crate's -- the four
 * instantiated LQR shapes, the agent count, and the 512-sample loop history -- which therefore have
 * no Rust test above them.
 */
static void the_control_surface_closes_its_own_loop(void) {
    /* Inertial->body for a rotation of 0.3 rad about the body x axis, row-major. */
    static const double IDENTITY[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    static const double INERTIA[9] = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
    const double c = cos(0.3), s = sin(0.3);
    const double turned[9] = {1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c};
    const sl_latency latency = {0.2, 0.1, 0.0};
    const sl_latency backwards = {-0.1, 0.0, 0.0};
    const sl_latency far_behind = {60.0, 0.0, 0.0};
    sl_attitude_state initial;
    sl_attitude_state seed;
    sl_attitude_state settled;
    sl_loop_sample samples[40];
    sl_loop_sample few[2];
    sl_delay_margin margin;
    sl_delay_margin direct;
    double total = -1.0;
    double angle_gain = 0.0, rate_gain = 0.0;
    size_t lag = 0, sized = 0, len = 0, few_len = 0, axis = 4;
    int i = 0;

    /* Three numbers, not one, and their sum is what a margin is compared against. */
    CHECK(sl_latency_total(&latency, &total) == SL_OK);
    CHECK_NEAR(total, 0.3, 1e-15);
    CHECK(sl_latency_total(NULL, &total) == SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_latency_total(&latency, NULL) == SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_latency_total(&backwards, &total) == SL_ERR_INVALID_ARGUMENT);
    CHECK(sl_latency_steps(&latency, 0.1, &lag) == SL_OK);
    CHECK(lag == 3);
    /* 0.3 s is not a whole number of 0.08 s steps, and the answer is a refusal rather than a
     * rounded step count labelled with the caller's tau. */
    CHECK(sl_latency_steps(&latency, 0.08, &lag) == SL_ERR_DOMAIN);
    CHECK(sl_latency_steps(&latency, 0.0, &lag) == SL_ERR_INVALID_ARGUMENT);

    /* The MRP is `tan(theta/4)` along the rotation axis, so the vector and the angle check each
     * other rather than a table. */
    {
        sl_vec3 mrp = {0.0, 0.0, 0.0};
        const sl_vec3 rest = {0.0, 0.0, 0.0};
        double principal = -1.0;
        CHECK(sl_attitude_error_mrp(turned, IDENTITY, &rest, &rest, &mrp, &principal) == SL_OK);
        CHECK_NEAR(principal, 0.3, 1e-12);
        CHECK_NEAR(radius_of(mrp), tan(0.3 / 4.0), 1e-12);
        CHECK_NEAR(mrp.x, -tan(0.3 / 4.0), 1e-12);
        CHECK(sl_attitude_error_mrp(NULL, IDENTITY, &rest, &rest, &mrp, &principal) ==
              SL_ERR_INVALID_ARGUMENT);
        CHECK(sl_attitude_error_mrp(turned, IDENTITY, &rest, &rest, NULL, &principal) ==
              SL_ERR_INVALID_ARGUMENT);

        /* The same rotation in the other chart. `sigma = q_v / (1 + q0)` is what relates them, so
         * the two entry points check each other rather than each checking a table, and `q0 >= 0`
         * is the short-way convention that makes the principal angle unambiguous. */
        {
            double q[4] = {0.0, 0.0, 0.0, 0.0};
            CHECK(sl_attitude_error_quaternion(turned, IDENTITY, &rest, &rest, q) == SL_OK);
            CHECK(q[0] >= 0.0);
            CHECK_NEAR((q[0] * q[0]) + (q[1] * q[1]) + (q[2] * q[2]) + (q[3] * q[3]), 1.0, 1e-15);
            CHECK_NEAR(2.0 * acos(q[0]), 0.3, 1e-12);
            CHECK_NEAR(q[1] / (1.0 + q[0]), mrp.x, 1e-15);
            CHECK_NEAR(q[2] / (1.0 + q[0]), mrp.y, 1e-15);
            CHECK_NEAR(q[3] / (1.0 + q[0]), mrp.z, 1e-15);
            CHECK(sl_attitude_error_quaternion(turned, IDENTITY, &rest, &rest, NULL) ==
                  SL_ERR_INVALID_ARGUMENT);
        }
    }

    /* Both laws command nothing at the reference, and an unknown tag is refused rather than
     * defaulted. The out-of-range cast is the `-fshort-enums` case as well: an enum narrowed to a
     * byte would not carry 7 across the boundary. */
    {
        const sl_vec3 rest = {0.0, 0.0, 0.0};
        sl_vec3 torque = {1.0, 1.0, 1.0};
        double lyapunov = -1.0;
        CHECK(sl_attitude_torque(SL_LAW_QUATERNION_PD, 8.0, 3.0, IDENTITY, IDENTITY, &rest, &rest,
                                 INERTIA, &torque) == SL_OK);
        CHECK(torque.x == 0.0 && torque.y == 0.0 && torque.z == 0.0);
        CHECK(sl_attitude_torque(SL_LAW_MRP_LYAPUNOV, 8.0, 3.0, IDENTITY, IDENTITY, &rest, &rest,
                                 INERTIA, &torque) == SL_OK);
        CHECK(torque.x == 0.0 && torque.y == 0.0 && torque.z == 0.0);
        CHECK(sl_attitude_torque((sl_attitude_law)7, 8.0, 3.0, IDENTITY, IDENTITY, &rest, &rest,
                                 INERTIA, &torque) == SL_ERR_INVALID_ARGUMENT);
        /* An inertia tensor that is not positive definite is a domain error, not a NaN. */
        CHECK(sl_attitude_torque(SL_LAW_QUATERNION_PD, 8.0, 3.0, IDENTITY, IDENTITY, &rest, &rest,
                                 IDENTITY, &torque) == SL_OK);
        CHECK(sl_attitude_lyapunov(8.0, 3.0, IDENTITY, IDENTITY, &rest, &rest, INERTIA,
                                   &lyapunov) == SL_OK);
        CHECK(lyapunov == 0.0);
        CHECK(sl_attitude_lyapunov(8.0, 3.0, turned, IDENTITY, &rest, &rest, INERTIA, &lyapunov) ==
              SL_OK);
        CHECK(lyapunov > 0.0);
    }

    /* `P = sqrt(2 I K)` is what a damping ratio of one means, so the two entry points round-trip
     * through each other rather than through a literal. */
    {
        double derivative = 0.0, frequency = 0.0, damping = 0.0;
        CHECK(sl_quaternion_pd_critically_damped(8.0, 4.0, &derivative) == SL_OK);
        CHECK_NEAR(derivative, sqrt(2.0 * 4.0 * 8.0), 1e-12);
        CHECK(sl_quaternion_pd_response(8.0, derivative, 4.0, &frequency, &damping) == SL_OK);
        CHECK_NEAR(damping, 1.0, 1e-14);
        CHECK_NEAR(frequency, sqrt(8.0 / (2.0 * 4.0)), 1e-14);
        CHECK(sl_quaternion_pd_critically_damped(-8.0, 4.0, &derivative) == SL_ERR_DOMAIN);
    }

    /* The loop, under a latency of three steps. The controller reads `seed` until the delay has
     * elapsed, so seeding at the reference while starting the body away from it makes the wiring
     * observable: exactly `lag` commanded moments are zero and the next one is not. */
    memcpy(initial.rotation, turned, sizeof turned);
    initial.rate.x = initial.rate.y = initial.rate.z = 0.0;
    memcpy(seed.rotation, IDENTITY, sizeof IDENTITY);
    seed.rate = initial.rate;

    CHECK(sl_attitude_loop_run(SL_LAW_QUATERNION_PD, 8.0, 12.0, &latency, 0.1, &initial, &seed,
                               IDENTITY, INERTIA, 40, NULL, 0, &sized, NULL) == SL_OK);
    CHECK(sized == 40);
    CHECK(sl_attitude_loop_run(SL_LAW_QUATERNION_PD, 8.0, 12.0, &latency, 0.1, &initial, &seed,
                               IDENTITY, INERTIA, 40, samples, 40, &len, &settled) == SL_OK);
    CHECK(len == sized);
    for (i = 0; i < (int)lag; ++i) {
        CHECK(samples[i].torque.x == 0.0);
        CHECK(samples[i].error_angle_rad == 0.0);
        CHECK_NEAR(samples[i].time_s, 0.1 * (double)i, 1e-12);
    }
    CHECK(samples[lag].torque.x != 0.0);
    /* The rotation the run ends on is still a rotation: its first row is a unit vector. */
    CHECK_NEAR(sqrt((settled.rotation[0] * settled.rotation[0]) +
                    (settled.rotation[1] * settled.rotation[1]) +
                    (settled.rotation[2] * settled.rotation[2])),
               1.0, 1e-12);
    /* The short-buffer branch, which the C++ wrapper sizes its way past and never takes. */
    CHECK(sl_attitude_loop_run(SL_LAW_QUATERNION_PD, 8.0, 12.0, &latency, 0.1, &initial, &seed,
                               IDENTITY, INERTIA, 40, few, 2, &few_len,
                               NULL) == SL_ERR_BUFFER_TOO_SMALL);
    CHECK(few_len == 40);
    /* A lag of 600 steps does not fit the 512-sample history this ABI gives the loop, and the
     * refusal is the point: wrapping the ring would read a state from a previous era. */
    CHECK(sl_attitude_loop_run(SL_LAW_QUATERNION_PD, 8.0, 12.0, &far_behind, 0.1, &initial, &seed,
                               IDENTITY, INERTIA, 40, NULL, 0, &sized, NULL) == SL_ERR_DOMAIN);

    /* The margin of a named law is the margin of its own linearisation -- not of the nominal
     * gains, which are a different loop for the MRP law. */
    CHECK(sl_attitude_small_angle_gains(SL_LAW_QUATERNION_PD, 8.0, 12.0, &angle_gain, &rate_gain) ==
          SL_OK);
    CHECK(sl_delay_margin_of_gains(4.0, angle_gain, rate_gain, &direct) == SL_OK);
    CHECK(sl_delay_margin_of_law(SL_LAW_QUATERNION_PD, 8.0, 12.0, 4.0, &margin) == SL_OK);
    CHECK(margin.seconds == direct.seconds);
    CHECK_NEAR(margin.normalised, margin.seconds * margin.natural_frequency_rad_s, 1e-12);
    CHECK(margin.crossing_frequency_rad_s > 0.0);
    /* The MRP law linearises to a quarter of its nominal stiffness and the quaternion law to a
     * half, so at the same gains the first is exactly twice the second. */
    {
        double mrp_angle = 0.0, mrp_rate = 0.0;
        CHECK(sl_attitude_small_angle_gains(SL_LAW_MRP_LYAPUNOV, 8.0, 12.0, &mrp_angle,
                                            &mrp_rate) == SL_OK);
        CHECK_NEAR(angle_gain, 2.0 * mrp_angle, 1e-15);
        CHECK(rate_gain == mrp_rate);
    }
    /* An undamped loop tolerates no delay at all, and that answer is computed rather than
     * refused. */
    CHECK(sl_delay_margin_of_gains(4.0, 8.0, 0.0, &direct) == SL_OK);
    CHECK(direct.seconds == 0.0);
    CHECK(direct.damping_ratio == 0.0);

    /* The binding axis is the one with the smallest margin, and a body cannot have a zero
     * principal moment. */
    {
        const sl_vec3 moments = {6.0, 2.0, 9.0};
        const sl_vec3 flat = {1.0, 0.0, 1.0};
        sl_delay_margin each[3];
        CHECK(sl_worst_principal_axis(SL_LAW_QUATERNION_PD, 8.0, 12.0, &moments, &axis, &margin) ==
              SL_OK);
        CHECK(sl_delay_margin_of_law(SL_LAW_QUATERNION_PD, 8.0, 12.0, 6.0, &each[0]) == SL_OK);
        CHECK(sl_delay_margin_of_law(SL_LAW_QUATERNION_PD, 8.0, 12.0, 2.0, &each[1]) == SL_OK);
        CHECK(sl_delay_margin_of_law(SL_LAW_QUATERNION_PD, 8.0, 12.0, 9.0, &each[2]) == SL_OK);
        CHECK(axis < 3);
        CHECK(margin.seconds <= each[0].seconds);
        CHECK(margin.seconds <= each[1].seconds);
        CHECK(margin.seconds <= each[2].seconds);
        CHECK(margin.seconds == each[axis].seconds);
        CHECK(sl_worst_principal_axis(SL_LAW_QUATERNION_PD, 8.0, 12.0, &flat, &axis, &margin) ==
              SL_ERR_DOMAIN);
    }

    /* The double integrator, where the whole check fits in 2x2 arithmetic: with `R = I` the gain
     * is `B^T P`, which for `B = [0 1]^T` is the second row of P. */
    {
        const double plant_a[4] = {0.0, 1.0, 0.0, 0.0};
        const double plant_b[2] = {0.0, 1.0};
        const double weight_q[4] = {1.0, 0.0, 0.0, 1.0};
        const double weight_r[1] = {1.0};
        double gain[2] = {0.0, 0.0};
        double riccati[4] = {0.0, 0.0, 0.0, 0.0};
        double residual = 1.0;
        CHECK(sl_lqr(plant_a, plant_b, weight_q, weight_r, 2, 1, gain, riccati, &residual) ==
              SL_OK);
        CHECK(residual < 1e-9);
        CHECK_NEAR(gain[0], riccati[2], 1e-12);
        CHECK_NEAR(gain[1], riccati[3], 1e-12);
        /* Every out-matrix is optional; asking for nothing but the residual is legal. */
        CHECK(sl_lqr(plant_a, plant_b, weight_q, weight_r, 2, 1, NULL, NULL, &residual) == SL_OK);
        /* A shape this ABI has no instantiation for is refused before a single element is read,
         * which is why these four-element arrays are safe to hand it. */
        CHECK(sl_lqr(plant_a, plant_b, weight_q, weight_r, 5, 2, NULL, NULL, &residual) ==
              SL_ERR_INVALID_ARGUMENT);
        CHECK(sl_lqr(NULL, plant_b, weight_q, weight_r, 2, 1, NULL, NULL, &residual) ==
              SL_ERR_INVALID_ARGUMENT);
    }

    /* The Clohessy-Wiltshire plant: the bottom half of B is the identity, so the three controls
     * are the three accelerations, and the 6x3 shape is one the LQR is instantiated for. */
    {
        double plant_a[36];
        double plant_b[18];
        double weight_q[36];
        double gain[18];
        double residual = 1.0;
        const double mean_motion = 2.0 * PI / 5400.0;
        int j = 0;
        CHECK(sl_clohessy_wiltshire_plant(mean_motion, plant_a, plant_b) == SL_OK);
        for (i = 0; i < 9; ++i) {
            CHECK(plant_b[i] == 0.0);
        }
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                CHECK(plant_b[9 + (3 * i) + j] == (i == j ? 1.0 : 0.0));
            }
        }
        for (i = 0; i < 36; ++i) {
            weight_q[i] = (i % 7 == 0) ? 1.0 : 0.0;
        }
        CHECK(sl_lqr(plant_a, plant_b, weight_q, IDENTITY, 6, 3, gain, NULL, &residual) == SL_OK);
        CHECK(residual < 1e-9);
        CHECK(sl_clohessy_wiltshire_plant(0.0, plant_a, plant_b) == SL_ERR_DOMAIN);
    }

    /* Cayley-Hamilton: `A^n + c_{n-1} A^{n-1} + ... + c_0 I = 0`. That is the check with no second
     * implementation in it -- the coefficients are verified against the matrix they came from. */
    {
        /* The companion matrix of `(l+1)(l+2)(l+3) = l^3 + 6l^2 + 11l + 6`. */
        const double companion[9] = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -6.0, -11.0, -6.0};
        const double want[3] = {6.0, 11.0, 6.0};
        double coefficients[3] = {0.0, 0.0, 0.0};
        double power[9];
        double next[9];
        double total[9];
        int j = 0, k = 0;
        CHECK(sl_characteristic_polynomial(companion, 3, coefficients) == SL_OK);
        for (i = 0; i < 3; ++i) {
            CHECK_NEAR(coefficients[i], want[i], 1e-12);
        }
        for (i = 0; i < 9; ++i) {
            power[i] = (i % 4 == 0) ? 1.0 : 0.0; /* the identity */
            total[i] = 0.0;
        }
        for (k = 0; k < 4; ++k) {
            for (i = 0; i < 9; ++i) {
                total[i] += (k < 3 ? coefficients[k] : 1.0) * power[i];
            }
            for (i = 0; i < 3; ++i) {
                int col = 0;
                for (col = 0; col < 3; ++col) {
                    double sum = 0.0;
                    for (j = 0; j < 3; ++j) {
                        sum += power[(3 * i) + j] * companion[(3 * j) + col];
                    }
                    next[(3 * i) + col] = sum;
                }
            }
            for (i = 0; i < 9; ++i) {
                power[i] = next[i];
            }
        }
        for (i = 0; i < 9; ++i) {
            CHECK_NEAR(total[i], 0.0, 1e-9);
        }
        /* The matrix-order wall, which is the same two numbers as the agent wall and not the
         * same wall -- so it is refused here for its own reason and by its own check. */
        CHECK(sl_characteristic_polynomial(companion, 1, coefficients) == SL_ERR_INVALID_ARGUMENT);
        CHECK(sl_characteristic_polynomial(companion, 13, coefficients) == SL_ERR_INVALID_ARGUMENT);
        CHECK(sl_characteristic_polynomial(NULL, 3, coefficients) == SL_ERR_INVALID_ARGUMENT);
    }

    /* The station-keeping law: a single controlled element makes `V = 1/2 w de^2` checkable by
     * hand, and the decrease rate is what "Lyapunov" means rather than what it is called. */
    {
        const sl_element_feedback law = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1e-6, 1e-4};
        const sl_element_feedback unweighted = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1e-6, 1e-4};
        const double elements[6] = {7.0e6, 0.0, 0.0, 0.0, 0.0, 0.0};
        const double error[6] = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        sl_vec3 acceleration = {0.0, 0.0, 0.0};
        double lyapunov = -1.0, rate = 1.0;
        CHECK(sl_element_feedback_lyapunov(&law, error, &lyapunov) == SL_OK);
        CHECK_NEAR(lyapunov, 0.5 * 100.0 * 100.0, 1e-9);
        CHECK(sl_element_feedback_acceleration(&law, elements, SL_MU_EARTH, 0.0, error,
                                               &acceleration) == SL_OK);
        CHECK(radius_of(acceleration) <= 1e-4 * (1.0 + 1e-12));
        CHECK(sl_element_feedback_decrease_rate(&law, elements, SL_MU_EARTH, 0.0, error, &rate) ==
              SL_OK);
        CHECK(rate <= 0.0);
        /* Every weight zero controls nothing, and that is a domain error rather than a law that
         * silently does nothing. */
        CHECK(sl_element_feedback_lyapunov(&unweighted, error, &lyapunov) == SL_ERR_DOMAIN);
        CHECK(sl_element_feedback_lyapunov(NULL, error, &lyapunov) == SL_ERR_INVALID_ARGUMENT);
    }

    /* The consensus surface. The path graph on four agents has a spectrum whose trace is the
     * total link weight, whose second entry is the algebraic connectivity, and whose largest
     * entry sets the published delay bound -- so the four entry points check each other. */
    {
        const double path[16] = {0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                                 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0};
        const double state[4] = {1.0, -2.0, 3.0, -0.5};
        double spectrum[4] = {0.0, 0.0, 0.0, 0.0};
        double fiedler = 0.0, bound = 0.0, spread = 0.0, trace = 0.0;
        double settled_state[4] = {0.0, 0.0, 0.0, 0.0};
        double disagreement = -1.0;
        uint32_t accepted = 0, rejected = 0;
        double mean = 0.0, ended = 0.0;

        CHECK(sl_laplacian_spectrum(path, 4, spectrum) == SL_OK);
        for (i = 0; i < 4; ++i) {
            trace += spectrum[i];
        }
        CHECK_NEAR(trace, 6.0, 1e-12); /* three links counted twice */
        CHECK_NEAR(spectrum[0], 0.0, 1e-12);
        CHECK(sl_algebraic_connectivity(path, 4, &fiedler) == SL_OK);
        CHECK(fiedler == spectrum[1]);
        CHECK(sl_consensus_delay_bound(path, 4, SL_CONSENSUS_DELAYED_RELATIVE, &bound) == SL_OK);
        CHECK_NEAR(bound, PI / (2.0 * spectrum[3]), 1e-12);
        /* The other protocol keeps the agent's own state current, and no delay destroys it. */
        CHECK(sl_consensus_delay_bound(path, 4, SL_CONSENSUS_CURRENT_SELF, &bound) == SL_OK);
        CHECK(bound > 0.0);
        CHECK(sl_consensus_delay_bound(path, 4, (sl_consensus_protocol)9, &bound) ==
              SL_ERR_INVALID_ARGUMENT);
        /* 13 agents is past this ABI's instantiation wall, and 1 short of its floor. */
        CHECK(sl_laplacian_spectrum(path, 13, spectrum) == SL_ERR_INVALID_ARGUMENT);
        CHECK(sl_laplacian_spectrum(path, 1, spectrum) == SL_ERR_INVALID_ARGUMENT);

        CHECK(sl_consensus_disagreement(state, 4, &spread) == SL_OK);
        CHECK(spread > 0.0);

        /* The eigenvectors the spectrum leaves behind. `L v = lambda v` for every mode is the
         * check with no second implementation in it -- L is rebuilt here from the adjacency the
         * caller supplied, not read back through the binding under test. */
        {
            double values[4];
            double modes[16];
            double laplacian[16];
            int j = 0, k = 0;
            CHECK(sl_laplacian_modes(path, 4, values, modes) == SL_OK);
            for (i = 0; i < 4; ++i) {
                CHECK(values[i] == spectrum[i]);
            }
            for (i = 0; i < 4; ++i) {
                double degree = 0.0;
                for (j = 0; j < 4; ++j) {
                    degree += path[(4 * i) + j];
                    laplacian[(4 * i) + j] = -path[(4 * i) + j];
                }
                laplacian[(4 * i) + i] += degree;
            }
            for (k = 0; k < 4; ++k) {
                double squared = 0.0;
                for (i = 0; i < 4; ++i) {
                    double row = 0.0;
                    for (j = 0; j < 4; ++j) {
                        row += laplacian[(4 * i) + j] * modes[(4 * k) + j];
                    }
                    CHECK_NEAR(row, values[k] * modes[(4 * k) + i], 1e-12);
                    squared += modes[(4 * k) + i] * modes[(4 * k) + i];
                }
                CHECK_NEAR(squared, 1.0, 1e-12); /* unit, so the mode is a direction */
            }
            /* Both out-pointers are optional, and asking for neither is still a validation. */
            CHECK(sl_laplacian_modes(path, 4, NULL, NULL) == SL_OK);
            CHECK(sl_laplacian_modes(path, 13, values, modes) == SL_ERR_INVALID_ARGUMENT);
        }

        /* Connectivity is combinatorial, so cutting one link of a path graph disconnects it and
         * the answer is `false` rather than the refusal every other entry point here gives. */
        {
            double split[16];
            bool connected = false;
            for (i = 0; i < 16; ++i) {
                split[i] = path[i];
            }
            CHECK(sl_graph_is_connected(path, 4, &connected) == SL_OK);
            CHECK(connected);
            split[(4 * 1) + 2] = 0.0;
            split[(4 * 2) + 1] = 0.0;
            CHECK(sl_graph_is_connected(split, 4, &connected) == SL_OK);
            CHECK(!connected);
            /* And every other refusal is still a refusal: an asymmetric matrix is not a
             * disconnected graph, it is not a graph. */
            split[(4 * 1) + 2] = 1.0;
            CHECK(sl_graph_is_connected(split, 4, &connected) == SL_ERR_DOMAIN);
            CHECK(sl_graph_is_connected(path, 4, NULL) == SL_ERR_INVALID_ARGUMENT);
            /* The disconnected graph the other entry points refuse, so the two agree on which
             * question each is answering. */
            split[(4 * 1) + 2] = 0.0;
            CHECK(sl_laplacian_spectrum(split, 4, spectrum) == SL_ERR_DOMAIN);
        }

        CHECK(sl_laplacian_spectrum(path, 4, spectrum) == SL_OK);
        CHECK(sl_consensus_delay_bound(path, 4, SL_CONSENSUS_DELAYED_RELATIVE, &bound) == SL_OK);
        /* Below the bound the disagreement collapses, and the mean is conserved either way --
         * an undirected graph's Laplacian annihilates the all-ones vector. */
        CHECK(sl_consensus_run(path, 4, SL_CONSENSUS_DELAYED_RELATIVE, 0.7 * bound, state, 40.0,
                               1e-9, settled_state, &disagreement, &accepted, &rejected) == SL_OK);
        CHECK(disagreement < 1e-3 * spread);
        CHECK(accepted > 0);
        for (i = 0; i < 4; ++i) {
            mean += state[i] / 4.0;
            ended += settled_state[i] / 4.0;
        }
        CHECK_NEAR(ended, mean, 1e-6);
        /* A run that ends before the first message arrives is a refusal, not an empty answer --
         * and a domain error rather than a malformed argument, because every number in it is
         * individually admissible and only their relation is not. */
        CHECK(sl_consensus_run(path, 4, SL_CONSENSUS_DELAYED_RELATIVE, 0.7 * bound, state,
                               0.1 * bound, 1e-9, NULL, NULL, NULL, NULL) == SL_ERR_DOMAIN);
    }

    /* The breakpoint mesh a delayed run must not step over. `t1` is the caller's own and is never
     * emitted; every value is strictly inside the span and strictly increasing. */
    {
        const double delays[2] = {0.25, 0.4};
        double mesh[16];
        double one[1];
        size_t mesh_len = 0;
        CHECK(sl_delay_breakpoints(delays, 2, 0.0, 1.0, 2, NULL, 0, &mesh_len) == SL_OK);
        CHECK(mesh_len > 0);
        CHECK(sl_delay_breakpoints(delays, 2, 0.0, 1.0, 2, mesh, 16, &mesh_len) == SL_OK);
        for (i = 0; i < (int)mesh_len; ++i) {
            CHECK(mesh[i] > 0.0 && mesh[i] < 1.0);
            if (i > 0) {
                CHECK(mesh[i] > mesh[i - 1]);
            }
        }
        CHECK(sl_delay_breakpoints(delays, 2, 0.0, 1.0, 2, one, 1, &mesh_len) ==
              SL_ERR_BUFFER_TOO_SMALL);
        CHECK(mesh_len > 1);
        /* A span that runs backwards is refused rather than answered with an empty mesh. */
        CHECK(sl_delay_breakpoints(delays, 2, 1.0, 0.0, 2, NULL, 0, &mesh_len) == SL_ERR_DOMAIN);
    }
}

int main(void) {
    the_layout_is_the_one_the_library_assumes();
    the_version_string_sizes_itself();
    the_geodetic_round_trip_agrees_with_the_other_two_suites();
    the_handle_opens_propagates_and_closes();
    a_failure_is_a_status_and_a_readable_message();
    the_generator_reports_its_length_before_it_needs_a_buffer();
    an_infeasible_constellation_is_a_domain_error();
    the_total_functions_answer_nan_rather_than_trapping();
    the_ephemeris_handle_interpolates_and_closes();
    the_field_and_three_body_answers_are_the_shape_they_should_be();
    the_mean_elements_are_mean_and_the_osculating_ones_are_not();
    the_least_squares_solve_reaches_the_answer_it_should();
    the_information_filter_accumulates_and_solves();
    the_lattice_is_the_walker_it_contains();
    the_draim_design_flies_the_orbit_it_reports();
    the_coverage_engine_answers_what_the_windows_say();
    the_gnss_models_answer_what_their_formulas_say();
    the_rinex_observation_reader_finds_a_value_by_its_declared_slot();
    the_atmosphere_models_answer_what_their_formulas_say();
    the_analytical_propagator_flies_the_orbit_it_was_given();
    the_force_models_all_build_the_same_handle();
    the_radar_surface_answers_what_its_formulas_say();
    the_optical_surface_answers_what_its_geometry_says();
    the_radiometric_observables_answer_what_their_geometry_says();
    the_lifetime_and_element_set_answer_in_their_own_units();
    the_gnss_observables_combine_and_detect();
    the_conjunction_surface_estimates_and_screens();
    the_cloud_mask_symbols_are_here_whatever_the_build();
    the_control_surface_closes_its_own_loop();

    if (failures != 0) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    puts("spacelab C smoke: all checks passed");
    return EXIT_SUCCESS;
}
