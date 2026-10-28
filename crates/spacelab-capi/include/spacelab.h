/*
 * spacelab.h -- the C ABI of the spacelab space-data computation core.
 *
 * This header is the contract. Rust cannot export C++ -- C++ has no stable ABI and name mangling
 * is a compiler's private business -- so the boundary is C, and `spacelab.hpp` next to this file
 * is a header-only C++ wrapper over exactly these symbols. Use whichever suits; same code.
 *
 * Four rules hold everywhere, so that nothing has to be remembered per function.
 *
 * 1. UNITS ARE SI AND ANGLES ARE RADIANS. Every length is metres, every duration seconds, every
 *    angle radians, every rate per second. There is no degrees-valued field anywhere in this
 *    header. (The Python bridge does use degrees for geodetic latitude and longitude. That is a
 *    deliberate divergence: the mixed-unit API is the one that gets misread, and Python's is
 *    fixed by its own compatibility. If you use both, convert at the seam, once.)
 *
 * 2. A FUNCTION RETURNING `sl_status` CAN FAIL; ONE RETURNING A VALUE IS TOTAL. On any status
 *    other than `SL_OK` the out-parameters are left untouched -- never half-written -- and
 *    `sl_last_error_message` says what happened. A total function has no failure to report:
 *    given a NaN it returns NaN, which is the answer. The one exception is
 *    `SL_ERR_BUFFER_TOO_SMALL`, which does set `*out_len` -- telling you the length you need is
 *    the entire point of returning it.
 *
 * 3. THIS LIBRARY NEVER HANDS YOU MEMORY TO FREE. Fixed-size results are written through
 *    out-parameters you own. Variable-size results use the two-call protocol: call once with
 *    `out == NULL` to learn the length, allocate, call again. Nothing crosses the allocator
 *    boundary, so there is no way to free with the wrong deallocator and no way to leak by
 *    forgetting a destructor. Handles (`sl_sgp4`) are the exception and they are opaque: the
 *    library allocates and the matching `_close` frees, both on this side of the boundary.
 *
 * 4. NO PANIC ESCAPES. A Rust panic unwinding into C++ is undefined behaviour, so every entry
 *    point catches its own; you get `SL_ERR_PANIC` and a message. Treat it as a bug report.
 *
 * Thread safety: every function here is safe to call concurrently from any number of threads.
 * The error message is thread-local, so a failure on one thread is never seen on another. An
 * `sl_sgp4` handle is immutable once opened and may be shared and propagated concurrently; only
 * `sl_sgp4_close` must not race with a call still using it.
 */

#ifndef SPACELAB_H
#define SPACELAB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- ABI version ---- */

/*
 * Incremented whenever anything below changes meaning, layout or signature. Compare it against
 * `sl_abi_version()` at start-up if you load the shared object dynamically; a static link cannot
 * skew, but a `dlopen` can, and a struct that grew a field is not detectable any other way.
 */
#define SL_ABI_VERSION 2U

uint32_t sl_abi_version(void);

/*
 * The library version, NUL-terminated, written into `buf`. Returns the number of bytes the full
 * string needs *including* the terminator, so `sl_version_string(NULL, 0)` sizes it. Truncated
 * but always terminated if `cap` is too small.
 */
size_t sl_version_string(char *buf, size_t cap);

/* ---- Status ---- */

typedef int32_t sl_status;

/* The call succeeded and the out-parameters are written. */
#define SL_OK 0

/* A pointer was NULL, a string had no terminator in range or was not UTF-8, or a count was zero
 * where zero has no meaning. The call did nothing. */
#define SL_ERR_INVALID_ARGUMENT 1

/* The inputs were well-formed but the answer does not exist: no Lambert arc connects those two
 * positions in that time, no inclination makes that orbit Sun-synchronous, that street cannot be
 * tiled. Not a bug -- a question with no answer. */
#define SL_ERR_DOMAIN 2

/* Input text did not parse: a malformed two-line element set. */
#define SL_ERR_PARSE 3

/* The computation started and failed: an SGP4 orbit that decayed, a geodetic conversion inside
 * the 43 km geocentric singularity. */
#define SL_ERR_COMPUTE 4

/* The buffer you supplied is shorter than the result. `*out_len` is set to the length needed and
 * nothing was written. */
#define SL_ERR_BUFFER_TOO_SMALL 5

/* A panic was caught at the boundary. This is a bug in the library; the message is its payload. */
#define SL_ERR_PANIC 6

/* The function is part of this ABI but the library you linked was built without the optional
 * feature behind it -- today only `onnx`, which the `sl_cloud_mask*` family needs. Every symbol
 * declared here exists in every build, so this is a run-time answer rather than a link error: a
 * symbol table that changes with a build flag is invisible to `sl_abi_version` and fails a
 * `dlopen` consumer at load, where there is nothing useful to say. */
#define SL_ERR_UNSUPPORTED 7

/*
 * The message describing the most recent failure *on this thread*, NUL-terminated, written into
 * `buf`. Returns the bytes the full message needs including the terminator, so
 * `sl_last_error_message(NULL, 0)` sizes it. Truncated but always terminated if `cap` is short.
 * Returns 1 -- the terminator alone -- when nothing has failed on this thread.
 *
 * Only failures write it; successful calls leave it alone. Branch on the status, never on the
 * message being non-empty, and read it before making the next call.
 */
size_t sl_last_error_message(char *buf, size_t cap);

/* ---- Types ---- */

/* A Cartesian triple: metres for a position, metres per second for a velocity. */
typedef struct {
    double x;
    double y;
    double z;
} sl_vec3;

/* A position and velocity in one frame, metres and metres per second. */
typedef struct {
    sl_vec3 position;
    sl_vec3 velocity;
} sl_state;

/* WGS-84 geodetic coordinates. Height is above the ellipsoid, not the geoid. */
typedef struct {
    double latitude_rad;
    double longitude_rad;
    double height_m;
} sl_geodetic;

/* A point on the ellipsoid, where height is not part of the answer. */
typedef struct {
    double latitude_rad;
    double longitude_rad;
} sl_latlon;

/*
 * Which arithmetic a propagator runs in. `SL_PRECISION_DOUBLE_DOUBLE` is roughly 32 significant
 * digits for about twice the arithmetic: at a small step the `double` run stops improving at its
 * round-off floor while this one keeps falling.
 */
typedef enum {
    SL_PRECISION_DOUBLE = 0,
    SL_PRECISION_DOUBLE_DOUBLE = 1,
    SL_PRECISION_FORCE_INT32 = 0x7FFFFFFF
} sl_precision;

/* How a Walker constellation spreads its planes in RAAN: `DELTA` over 2*pi, `STAR` over pi. */
typedef enum {
    SL_WALKER_DELTA = 0,
    SL_WALKER_STAR = 1,
    SL_WALKER_PATTERN_FORCE_INT32 = 0x7FFFFFFF
} sl_walker_pattern;

/*
 * Both enums above carry a `_FORCE_INT32` enumerator. It is not a value to pass; it is what makes
 * the enum four bytes wide no matter who compiles this header.
 *
 * The width of a C enum is the implementation's choice, and `-fshort-enums` -- the *default* on
 * most ARM bare-metal toolchains -- picks the smallest type that fits. Measured with gcc 15 on
 * x86-64: without the enumerator `sizeof(sl_precision)` is 4 normally and **1** under
 * `-fshort-enums`; with it, 4 under both, in C99 and in C++20. That matters twice over. As an
 * argument, `sl_propagate_two_body` would be handed one byte where the library reads four. As the
 * type of `sl_walker_params.pattern`, at offset 0, it would leave three bytes of uninitialised
 * padding exactly where the library reads the tag -- a wrong answer with nothing to report it.
 *
 * `0x7FFFFFFF` fits in `int` and nothing narrower, which is the whole trick. A test in this crate
 * checks that both enumerators are still here, the C++ smoke test static-asserts the widths and the
 * C smoke test reads them at run time; both are compiled with `-fshort-enums` as well as without.
 *
 * C23 offers the direct spelling -- `enum sl_precision : int32_t` -- and it is not used here. The
 * guarantee this header makes is to every consumer, and a consumer compiling as C99 or C17 would
 * get the fixed width only if their standard supported it. A promise that holds on the newest
 * standard and quietly lapses on an older one is worse than the sentinel, which holds on all of
 * them. Measured on this header, gcc 16.1.1 and clang 22.1.8, C99 through C23 and C++20 through
 * C++26: four bytes on every rung, with `-fshort-enums` and without.
 */

/* ---- Constants ---- */

/*
 * The values the library itself uses, at the precision it holds them -- not rounded for the
 * header. A test in this crate compares each literal below against the Rust constant it mirrors
 * and fails on any drift, so the header cannot quietly fall behind the code.
 */

/* Earth's gravitational parameter GM, m^3/s^2. */
#define SL_MU_EARTH 3.986004418e14

/* WGS-84 equatorial radius, m. */
#define SL_RE_EARTH 6378137.0

/* Earth's second zonal harmonic, unnormalised. */
#define SL_J2_EARTH 1.08262668e-3

/* Earth's third zonal harmonic, unnormalised. */
#define SL_J3_EARTH (-2.53265365e-6)

/* Earth's rotation rate, rad/s. */
#define SL_EARTH_ROTATION_RATE 7.292115146706979e-5

/* ---- Time and frames ---- */

/*
 * Epochs are integer nanoseconds since J2000 on a named scale, which is exact for any date this
 * library is useful on: an int64 nanosecond count spans +-292 years around 2000. Two scales are
 * needed at once because they drive different physics -- UT1 turns the Earth, TT drives
 * precession and nutation -- and conflating them is worth hundreds of metres on the ground.
 */

/*
 * TEME (the frame SGP4 outputs) to ITRF (Earth-fixed). `xp_rad` and `yp_rad` are the polar-motion
 * angles and `lod_seconds` the excess length of day; pass zero for all three to rotate by
 * Greenwich mean sidereal time alone, which is what you have with no Earth-orientation data.
 */
sl_status sl_teme_to_itrf(const sl_state *in, int64_t ut1_ns, int64_t tt_ns, double xp_rad,
                          double yp_rad, double lod_seconds, sl_state *out);

/*
 * GCRF (GCRS) to ITRF by the rigorous IAU 2006/2000A CIO chain -- precession, nutation, Earth
 * rotation, polar motion. `dx_rad` and `dy_rad` are the observed celestial-pole offsets. Unlike
 * `sl_teme_to_itrf` the pole is tracked exactly rather than approximated by a fixed axis; this is
 * the frame the tesseral geopotential must be evaluated in. Validated against pyerfa to < 5e-14.
 */
sl_status sl_gcrf_to_itrf(const sl_state *in, int64_t ut1_ns, int64_t tt_ns, double xp_rad,
                          double yp_rad, double dx_rad, double dy_rad, double lod_seconds,
                          sl_state *out);

/*
 * ITRF Cartesian position to WGS-84 geodetic, by Vermeille's closed form. This is the subpoint
 * step of a ground track: feed it what `sl_teme_to_itrf` gave you. Returns `SL_ERR_COMPUTE` only
 * inside the 43 km geocentric singularity, which nothing in orbit occupies.
 */
sl_status sl_itrf_to_geodetic(const sl_vec3 *r_itrf, sl_geodetic *out);

/* WGS-84 geodetic to an ITRF Cartesian position -- the exact inverse, for placing stations. */
sl_status sl_geodetic_to_itrf(const sl_geodetic *g, sl_vec3 *out);

/* ---- SGP4 ---- */

/*
 * A parsed two-line element set. Opaque: the layout is not part of this ABI and will change.
 * Immutable once opened, so one handle may be shared across threads.
 */
typedef struct sl_sgp4 sl_sgp4;

/*
 * Parse a TLE. `line1` and `line2` are NUL-terminated; anything longer than 4096 bytes without a
 * terminator is rejected rather than read past. On success `*out` owns a handle that
 * `sl_sgp4_close` must free. Returns `SL_ERR_PARSE` if the element set is malformed.
 */
sl_status sl_sgp4_open(const char *line1, const char *line2, sl_sgp4 **out);

/* Release a handle. `sl_sgp4_close(NULL)` is a no-op, so it is safe in unconditional cleanup. */
void sl_sgp4_close(sl_sgp4 *sat);

/*
 * Propagate to a signed offset in minutes from the element set's own epoch, in TEME. Returns
 * `SL_ERR_COMPUTE` when SGP4 itself fails -- a decayed orbit, most often.
 */
sl_status sl_sgp4_propagate_at(const sl_sgp4 *sat, double minutes, sl_state *out);

/*
 * The same for `n` offsets at once, writing `n` states. This is why the handle exists: the
 * element set is parsed once instead of once per epoch. `minutes` and `out` must not overlap.
 * Fails on the first offset that fails, having written nothing.
 */
sl_status sl_sgp4_propagate_batch(const sl_sgp4 *sat, const double *minutes, size_t n,
                                  sl_state *out);

/*
 * Parse and propagate in one call, for a single epoch with no reason to keep a handle. Exactly
 * `sl_sgp4_open` then `sl_sgp4_propagate_at` then `sl_sgp4_close`.
 */
sl_status sl_sgp4_propagate(const char *line1, const char *line2, double minutes, sl_state *out);

/* ---- Two-body motion ---- */

/*
 * Advance a state by `dt` seconds under a point mass, by the universal-variable Kepler solution:
 * exact for the two-body problem, and independent of step size because it does not step.
 */
sl_status sl_kepler_propagate(const sl_state *in, double mu, double dt, sl_state *out);

/*
 * Advance a state by `steps` fixed steps of `dt` seconds under a point mass, with the 8th-order
 * time-symmetric symplectic method. Use this rather than `sl_kepler_propagate` when you want the
 * numerical answer -- to measure an integrator's own error, or to compare precisions.
 */
sl_status sl_propagate_two_body(const sl_state *in, double mu, double dt, uint64_t steps,
                                sl_precision precision, sl_state *out);

/*
 * Lambert's problem: the two velocities connecting `r1` to `r2` in `tof` seconds under `mu`.
 * `prograde` selects the short-way branch. Returns `SL_ERR_DOMAIN` when no arc exists.
 */
sl_status sl_lambert(const sl_vec3 *r1, const sl_vec3 *r2, double tof, double mu, bool prograde,
                     sl_vec3 *v1, sl_vec3 *v2);

/* ---- Measurements from the ground ---- */

/*
 * All four take the satellite state and the station in the same Earth-fixed frame. They are
 * total: a NULL pointer or a NaN input gives NaN back, because there is no answer to report and
 * no failure to describe. `latitude_rad` and `longitude_rad` orient the station's local horizon.
 */

/* Range from station to satellite, m. */
double sl_range(const sl_state *sat, const sl_vec3 *station_ecef);

/* Range rate, m/s, positive when opening. */
double sl_range_rate(const sl_state *sat, const sl_vec3 *station_ecef);

/* Topocentric elevation above the local horizon, rad. */
double sl_elevation(const sl_state *sat, const sl_vec3 *station_ecef, double latitude_rad,
                    double longitude_rad);

/* Topocentric azimuth, rad, from north through east. */
double sl_azimuth(const sl_state *sat, const sl_vec3 *station_ecef, double latitude_rad,
                  double longitude_rad);

/* ---- Footprint geometry ---- */

/* The limb look angle: the largest off-nadir angle whose line of sight still meets Earth, rad. */
double sl_horizon_look_angle(double earth_radius_m, double altitude_m);

/*
 * Where a great-circle arc of `central_angle_rad` from `start` on initial `azimuth_rad` arrives.
 * Sweep the azimuth over 2*pi to trace a footprint circle. Total.
 */
sl_latlon sl_ground_point_at(sl_latlon start, double central_angle_rad, double azimuth_rad);

/* ---- Orbit design ---- */

/* The prograde critical inclination, acos(1/sqrt(5)) ~ 63.43 deg, where apsidal precession dies. */
double sl_critical_inclination(void);

/* The frozen-orbit eccentricity at argument of perigee 90 deg, balancing the J2/J3 drift. */
double sl_frozen_eccentricity(double semi_major_axis_m, double inclination_rad,
                              double earth_radius_m, double j2, double j3);

/*
 * The semi-major axis whose ground track repeats after `revs` revolutions in `days` nodal days.
 * Both counts must be non-zero: zero is not a repeat cycle, and letting it through would return
 * an infinity that looks like an orbit until something divides by it.
 */
sl_status sl_repeat_ground_track_sma(uint32_t revs, uint32_t days, double inclination_rad,
                                     double eccentricity, double mu, double earth_radius_m,
                                     double j2, double earth_rotation_rate, double *out);

/*
 * The Sun-synchronous inclination for this orbit, rad. Returns `SL_ERR_DOMAIN` when the orbit is
 * too high for J2 to precess the node a full turn a year -- past roughly 6 000 km altitude there
 * is no such inclination, and that is a fact about the orbit, not a failure of the solver.
 */
sl_status sl_sun_synchronous_inclination(double semi_major_axis_m, double eccentricity, double mu,
                                         double earth_radius_m, double j2, double *out);

/* ---- Constellations ---- */

/*
 * Walker `i: T/P/F`. Fill with `sl_walker_params_init` and then override -- a struct literal you
 * wrote by hand silently gains a zero when this grows a field, and zero is a valid-looking mu.
 */
typedef struct {
    sl_walker_pattern pattern;
    double inclination_rad;
    uint32_t planes;
    uint32_t per_plane;
    uint32_t phasing;
    double semi_major_axis_m;
    double mu;
} sl_walker_params;

/* Earth mu, delta pattern, everything else zero. */
void sl_walker_params_init(sl_walker_params *p);

/*
 * The constellation's inertial states at the epoch, plane-major -- `planes * per_plane` of them.
 *
 * Two-call protocol: pass `out == NULL` to have `*out_len` set to the count with nothing written,
 * then allocate and call again with `cap` elements. A short `cap` gives
 * `SL_ERR_BUFFER_TOO_SMALL`, `*out_len` set to what is needed, and nothing written.
 */
sl_status sl_walker(const sl_walker_params *p, sl_state *out, size_t cap, size_t *out_len);

/*
 * Street of Coverage: a near-polar constellation sized for continuous single-fold coverage above
 * `min_elevation_rad`. `per_plane == 0` means "the fewest that still tile the street".
 */
typedef struct {
    double inclination_rad;
    double altitude_m;
    double min_elevation_rad;
    double earth_radius_m;
    double mu;
    uint32_t per_plane;
} sl_soc_params;

/* Earth mu and radius, `per_plane = 0`, the rest zero. */
void sl_soc_params_init(sl_soc_params *p);

/* The solved geometry. Counts are exact; every angle is radians. */
typedef struct {
    uint32_t planes;
    uint32_t per_plane;
    double footprint_half_angle_rad;
    double street_half_width_rad;
    double raan_step_co_rad;
    double raan_step_counter_rad;
    double raan_step_rad;
    double raan_span_rad;
    double inter_plane_phase_rad;
} sl_soc_geometry;

/*
 * Solve the geometry without generating the states. Returns `SL_ERR_DOMAIN` when the
 * constellation is infeasible: a non-positive footprint, an inclination too far from polar for
 * the streets to close, or a `per_plane` below the ceil(pi/theta) floor.
 */
sl_status sl_street_of_coverage_geometry(const sl_soc_params *p, sl_soc_geometry *out);

/*
 * The constellation's inertial states, plane-major. Same two-call protocol as `sl_walker`, and
 * the same `SL_ERR_DOMAIN` for an infeasible geometry -- where the Python binding returns an
 * empty list and leaves you to guess why, this says so.
 */
sl_status sl_street_of_coverage(const sl_soc_params *p, sl_state *out, size_t cap, size_t *out_len);

/* ---- Numerical propagation ---- */

/*
 * A dense ephemeris: an interpolant of a propagated orbit, queryable anywhere on its span without
 * re-integrating. Built by one of the constructors below and released by `sl_ephemeris_close`.
 *
 * Which integrator built it is not part of the type, because it is not part of the question: a
 * caller wants the state at a time, and chose the physics when they chose the constructor.
 */
typedef struct sl_ephemeris sl_ephemeris;

/*
 * Propagate `in` for `t1_seconds` under point-mass gravity `mu` plus J2 (`earth_radius_m` the
 * reference radius) with the 15th-order Gauss-Radau IAS15 integrator, keeping the interpolant.
 * Set `j2 = 0` for pure two-body. Position-only forces, so no drag.
 *
 * `t1_seconds` may be negative, which propagates backwards; it may not be zero, which would be a
 * span with nothing in it. Returns `SL_ERR_COMPUTE` if the integration fails.
 */
sl_status sl_ephemeris_j2(const sl_state *in, double t1_seconds, double mu, double j2,
                          double earth_radius_m, sl_ephemeris **out);

/*
 * The same, under the embedded tesseral geopotential truncated to `degree`, evaluated
 * Earth-fixed at each step so the longitude-dependent terms are right: the body rotates as
 * `theta = theta0_rad + omega*t` with omega Earth's rate. `tolerance` is the per-step relative and
 * absolute error the adaptive DOP853 aims for; 1e-9 is a reasonable default.
 *
 * Slower at high degree. Returns `SL_ERR_DOMAIN` if `degree` exceeds the embedded model.
 */
sl_status sl_ephemeris_geopotential(const sl_state *in, double t1_seconds, uint32_t degree,
                                    double theta0_rad, double tolerance, sl_ephemeris **out);

/*
 * The rest of the force models the same handle can be built over. Each takes the same arc
 * (`t1_seconds`, non-zero and finite) and the same DOP853 tolerance, and differs only in what it
 * integrates; the handle they return is read through `sl_ephemeris_at` and its neighbours like any
 * other.
 *
 * `sl_ephemeris_j2_dense` is the same two forces as `sl_ephemeris_j2` under DOP853 rather than
 * IAS15. Reach for IAS15 when the arc is long and the forces depend on position alone -- it holds
 * the round-off floor over thousands of revolutions; reach for DOP853 when a force reads velocity
 * or time, which IAS15's Gauss-Radau formulation does not admit.
 */
sl_status sl_ephemeris_j2_dense(const sl_state *in, double t1_seconds, double mu, double j2,
                                double earth_radius_m, double tolerance, sl_ephemeris **out);

/* A scale-height atmosphere and the ballistic coefficient through it. */
typedef struct sl_exponential_atmosphere {
    double ballistic_coefficient; /* m / (Cd A), kg/m^2 */
    double reference_density;     /* at the reference altitude, kg/m^3 */
    double reference_altitude_m;  /* above the surface */
    double scale_height_m;        /* the e-folding height */
    double earth_radius_m;        /* the spherical Earth the altitude is measured from */
} sl_exponential_atmosphere;

/* The Earth-fixed geopotential plus drag through that atmosphere. `theta0_rad` is the sidereal
 * phase at the epoch, advanced at the uniform Earth rotation rate. */
sl_status sl_ephemeris_drag(const sl_state *in, double t1_seconds, uint32_t degree,
                            double theta0_rad, const sl_exponential_atmosphere *atmosphere,
                            double tolerance, sl_ephemeris **out);

/*
 * The Earth-orientation quantities one arc is rotated with, held fixed over the arc.
 *
 * There is no length-of-day field, and its absence is deliberate: the GCRF->ITRF rotation these
 * propagators use is independent of LOD, which only scales the angular velocity they discard. A
 * field the library reads and ignores is worse than no field, because a caller who fills it in
 * believes something is happening. (The Python bridge does take one, and drops it.)
 */
typedef struct sl_eop {
    int64_t epoch_tt_ns;  /* the arc's start on TT, nanoseconds since J2000 */
    int64_t epoch_ut1_ns; /* the same instant on UT1 */
    double xp_rad;        /* polar motion */
    double yp_rad;
    double dx_rad; /* celestial-pole offset */
    double dy_rad;
} sl_eop;

/* Drag through NRLMSISE-00 rather than through a scale height: the density is read at the geodetic
 * point the state rotates to at each stage, which is why this one needs the frame. */
typedef struct sl_nrlmsise_drag {
    int32_t doy;                  /* day of year at the arc's epoch */
    double seconds_of_day;        /* UT seconds into it */
    double ballistic_coefficient; /* m / (Cd A), kg/m^2 */
    double f107;                  /* daily 10.7 cm flux, sfu */
    double f107a;                 /* its 81-day mean */
    double ap;                    /* the daily planetary amplitude */
    double mu;                    /* m^3/s^2 */
} sl_nrlmsise_drag;

sl_status sl_ephemeris_drag_nrlmsise00(const sl_state *in, double t1_seconds, const sl_eop *eop,
                                       const sl_nrlmsise_drag *drag, double tolerance,
                                       sl_ephemeris **out);

/*
 * The geopotential evaluated in the true ITRF -- rotated by the interpolated CIO chain rather than
 * by a uniform sidereal angle, which is the difference between kilometres and metres over a day --
 * and the same plus the full IERS 2010 degree-2 solid Earth tide: Step 1, the frequency-dependent
 * Step 2, and the solid and ocean pole tides on the dC21/dS21 band.
 */
sl_status sl_ephemeris_geopotential_itrf(const sl_state *in, double t1_seconds, uint32_t degree,
                                         const sl_eop *eop, double tolerance, sl_ephemeris **out);
sl_status sl_ephemeris_geopotential_tide(const sl_state *in, double t1_seconds, uint32_t degree,
                                         const sl_eop *eop, double epoch_j2000_seconds,
                                         double mu_sun, double mu_moon, double tolerance,
                                         sl_ephemeris **out);

/* Point-mass gravity plus J2 evaluated in the true ITRF. */
sl_status sl_ephemeris_j2_itrf(const sl_state *in, double t1_seconds, double mu, double j2,
                               double earth_radius_m, const sl_eop *eop, double tolerance,
                               sl_ephemeris **out);

/* One row of an Earth-orientation series. */
typedef struct sl_eop_sample {
    double t_seconds; /* from the arc's epoch */
    double xp_rad;    /* polar motion */
    double yp_rad;
    double dut1_seconds; /* UT1 - UTC */
    double dx_rad;       /* celestial-pole offset */
    double dy_rad;
} sl_eop_sample;

/*
 * The same over a series rather than one fixed set: the rotation is rebuilt from the interpolated
 * EOP at every stage, which is what a multi-day arc needs. The series must be strictly increasing
 * in `t_seconds` and hold at least two rows, and every field of every row must be finite; a query
 * outside it saturates at the nearest end rather than extrapolating a quantity that is measured and
 * not modelled.
 */
sl_status sl_ephemeris_j2_itrf_series(const sl_state *in, double t1_seconds, double mu, double j2,
                                      double earth_radius_m, int64_t epoch_utc_ns,
                                      double tt_minus_utc_s, const sl_eop_sample *series, size_t n,
                                      double tolerance, sl_ephemeris **out);

/*
 * The Earth-fixed geopotential plus Sun and Moon third-body attraction and solar radiation
 * pressure, with the built-in analytic ephemerides advanced along the J2000 timeline by
 * `epoch_j2000_seconds`. `srp_reference_acceleration` is the acceleration at one astronomical unit,
 * m/s^2 -- that is, the area-to-mass ratio and the reflectivity rolled into one number.
 */
sl_status sl_ephemeris_luni_solar(const sl_state *in, double t1_seconds, uint32_t degree,
                                  double epoch_j2000_seconds, double srp_reference_acceleration,
                                  double theta0_rad, double tolerance, sl_ephemeris **out);

/* Point-mass gravity plus the three IERS 2010 relativistic corrections: Schwarzschild, Lense-
 * Thirring and de Sitter. */
sl_status sl_ephemeris_relativistic(const sl_state *in, double t1_seconds,
                                    double epoch_j2000_seconds, double mu, double mu_sun,
                                    double tolerance, sl_ephemeris **out);

/*
 * Fixed-step 8th-order symplectic (Yoshida) propagation under point-mass gravity plus J2, in `f64`
 * or in double-double. The arithmetic is the parameter, which is the point of the call: run it
 * twice and the difference is the round-off the ordinary answer carries.
 *
 * Unlike everything above this returns a state rather than a handle -- a fixed-step symplectic
 * integrator has no dense interpolant to hand out.
 */
sl_status sl_propagate_j2_precision(const sl_state *in, double mu, double j2, double earth_radius_m,
                                    double dt_seconds, uint64_t steps, sl_precision precision,
                                    sl_state *out);

/* Release an ephemeris. `sl_ephemeris_close(NULL)` is a no-op. */
void sl_ephemeris_close(sl_ephemeris *e);

/*
 * The interpolated state at `t_seconds` from the start of the span. Returns `SL_ERR_DOMAIN` off
 * the span rather than extrapolating -- an interpolant asked outside its own data is not an
 * answer, and a wrong orbit that looks like one is the expensive kind of wrong.
 */
sl_status sl_ephemeris_at(const sl_ephemeris *e, double t_seconds, sl_state *out);

/*
 * The states at `n` epochs, in one crossing. `out` holds `n` states, which the caller sizes -- the
 * length is an input here, so there is no protocol to run. Every epoch is checked against the span
 * before any state is written, so a refusal leaves `out` untouched rather than partly filled.
 */
sl_status sl_ephemeris_sample(const sl_ephemeris *e, const double *times_seconds, size_t n,
                              sl_state *out);

/* The span the interpolant covers, seconds from the start. `t1` is negative for a backwards arc. */
sl_status sl_ephemeris_span(const sl_ephemeris *e, double *out_t0, double *out_t1);

/* Accepted and rejected step counts -- how hard the problem was. */
sl_status sl_ephemeris_step_counts(const sl_ephemeris *e, uint32_t *out_accepted,
                                   uint32_t *out_rejected);

/*
 * The tesseral geopotential acceleration at a body-fixed position, m/s^2. A degree beyond the
 * embedded model is `SL_ERR_DOMAIN`, and the message names the highest the model carries.
 */
sl_status sl_geopotential_acceleration(uint32_t degree, const sl_vec3 *r_body_fixed, sl_vec3 *out);

/* ---- Three-body motion ---- */

/*
 * The five libration points of the circular restricted three-body problem for `mass_ratio`, in
 * synodic nondimensional coordinates, written to `out[0..5]` as L1, L2, L3, L4, L5. The mass ratio
 * must lie in (0, 1); Earth-Moon is 0.012150585.
 */
sl_status sl_libration_points(double mass_ratio, sl_vec3 *out);

/*
 * Richardson's third-order analytical guess for a halo orbit about libration point `li` with
 * out-of-plane amplitude `amplitude_z`, in the same synodic nondimensional coordinates.
 * `northern` picks the branch. A starting point for differential correction, not a periodic orbit.
 *
 * Points are numbered as the literature names them, so L1 is `1`. The third-order theory covers
 * **L1 and L2 only**; anything else is `SL_ERR_DOMAIN`, including the `0` a zero-based habit would
 * reach for.
 */
sl_status sl_richardson_halo(uint32_t li, double amplitude_z, bool northern, double mass_ratio,
                             sl_state *out);

/* ---- Constellation families ---- */

/*
 * Three more constellation designs, alongside the Walker and Street-of-Coverage above. Each takes a
 * parameter block with an `_init` that writes the documented defaults, and each generates through
 * the same two-call protocol: pass `out == NULL` to learn the length, then a buffer that size.
 *
 * Flower (Mortari et al. 2004) phases `satellites` spacecraft along one `(petals, days)` repeating
 * ground track. The 2-D Lattice Flower is the integer-lattice generalisation that contains both
 * Flower and Walker — a circular lattice with `config = F` reproduces a Walker `i: T/P/F`. Draim's
 * is the four-satellite tetrahedral design that achieves continuous global coverage with the fewest
 * spacecraft known.
 */
typedef struct sl_flower_params {
    uint32_t petals;     /* N_p of the repeating ground track */
    uint32_t days;       /* N_d, the days the track takes to close */
    uint32_t satellites; /* N_s phased along the common track */
    double eccentricity;
    double inclination_rad;
    double arg_periapsis_rad;
    double raan_rad;         /* the reference satellite's node */
    double mean_anomaly_rad; /* the reference satellite's anomaly */
    double frame_rate_rad_s; /* the rotating frame's rate; default is Earth's sidereal rate */
    double mu;
} sl_flower_params;

typedef struct sl_lattice_params {
    uint32_t orbits;         /* N_o right-ascension slots */
    uint32_t sats_per_orbit; /* N_so in each slot */
    uint32_t config;         /* N_c = Walker's F; the literature's N_c' is -N_c mod N_o */
    double semi_major_axis_m;
    double eccentricity;
    double inclination_rad;
    double arg_periapsis_rad;
    double raan0_rad; /* the first slot's node */
    double mu;
} sl_lattice_params;

typedef struct sl_draim_params {
    double period_s;
    double eccentricity;
    double inclination_rad;
    double raan0_rad;
    double mu;
} sl_draim_params;

/*
 * Fill with the defaults. Flower and lattice default every count and angle to zero — a design has
 * to be stated — and only the frame rate and `mu` to something useful. Draim defaults to the
 * patent's worked example: a 48-hour orbit at `e = 0.28` and `i = 33 degrees`.
 * `..._init(NULL)` is a no-op in each case.
 */
void sl_flower_params_init(sl_flower_params *p);
void sl_lattice_params_init(sl_lattice_params *p);
void sl_draim_params_init(sl_draim_params *p);

sl_status sl_flower(const sl_flower_params *p, sl_state *out, size_t cap, size_t *out_len);
sl_status sl_lattice_flower(const sl_lattice_params *p, sl_state *out, size_t cap, size_t *out_len);
sl_status sl_draim(const sl_draim_params *p, sl_state *out, size_t cap, size_t *out_len);

/* The geometry of the orbit a Draim design flies. Radii are from the centre, not the surface. */
typedef struct sl_draim_orbit {
    double semi_major_axis_m;
    double perigee_radius_m;
    double apogee_radius_m;
} sl_draim_orbit;

sl_status sl_draim_geometry(const sl_draim_params *p, sl_draim_orbit *out);

/*
 * The alpha-characteristic of a delta system (Mozhaev): the smallest sub-satellite footprint
 * half-angle, in radians, at which those `n` satellites continuously `fold`-fold cover the sphere.
 * Evaluated on a `grid_points` x `time_samples` space-time grid, so both control the cost and the
 * sharpness of the answer; 500 and 60 are reasonable.
 */
sl_status sl_alpha_characteristic(const sl_state *states, size_t n, double mu, uint32_t fold,
                                  uint32_t grid_points, uint32_t time_samples, double *out_rad);

/*
 * The fewest-satellite Walker Delta whose alpha-characteristic clears `max_footprint_rad` — the
 * cheapest constellation that continuously `fold`-fold covers the sphere with that sensor, at that
 * orbit. `SL_ERR_DOMAIN` if no design of at most `max_total` satellites reaches it.
 *
 * The answer is written as a complete `sl_walker_params`, so it can be handed straight to
 * `sl_walker` rather than reassembled from three numbers.
 */
sl_status sl_minimum_walker(double max_footprint_rad, double inclination_rad,
                            double semi_major_axis_m, double mu, uint32_t fold, uint32_t max_total,
                            uint32_t grid_points, uint32_t time_samples, sl_walker_params *out);

/* ---- Coverage ---- */

/* Which constraint decides whether a ground point is covered. */
typedef enum sl_sensor_kind {
    SL_SENSOR_ELEVATION_MASK = 0,           /* a minimum elevation above the local horizon */
    SL_SENSOR_SIDE_LOOKING_SAR = 1,         /* an incidence band, near and far */
    SL_SENSOR_AGILE_OPTICAL = 2,            /* an optical instrument slewable off nadir */
    SL_SENSOR_KIND_FORCE_INT32 = 0x7FFFFFFF /* see the note on sl_precision */
} sl_sensor_kind;

/*
 * The instrument a coverage query is asked about.
 *
 * One tagged struct rather than three types, because C has no way to spell an overload: the two
 * entry points below would otherwise each need three. Only the fields the tag names are read.
 * `sl_sensor_init` writes the tag and leaves the rest at zero, so filling in the one field that
 * belongs to your kind is the whole of the work. `sl_sensor_init(NULL, ...)` is a no-op.
 *
 * A tag outside the enum is `SL_ERR_INVALID_ARGUMENT`; angles the instrument cannot accept -- a
 * SAR whose near edge is past its far one, an elevation mask below the horizon -- are
 * `SL_ERR_DOMAIN`, and the message names which.
 */
typedef struct sl_sensor {
    int32_t kind;              /* an sl_sensor_kind */
    double min_elevation_rad;  /* SL_SENSOR_ELEVATION_MASK */
    double near_incidence_rad; /* SL_SENSOR_SIDE_LOOKING_SAR */
    double far_incidence_rad;  /* SL_SENSOR_SIDE_LOOKING_SAR */
    double max_off_nadir_rad;  /* SL_SENSOR_AGILE_OPTICAL */
} sl_sensor;

void sl_sensor_init(sl_sensor *s, int32_t kind);

/*
 * Atmospheric refraction between the two elevations a station has, at `pressure_hpa` and
 * `temperature_k`; `SL_ERR_DOMAIN` outside each fit's own range. Both fits are published at
 * 1010 hPa and 283 K and scale by `(P/1010)(283/T)`.
 *
 * `sl_refraction_apparent` is Saemundsson (1986), true -> apparent, fitted in the true altitude;
 * `sl_refraction_geometric` is Bennett (1982), apparent -> true, fitted in the apparent one. They
 * are not each other's inverse -- a round trip closes to a few arcseconds near the horizon -- which
 * is why both are here rather than one and a sign.
 *
 * `sl_refraction_geometric` is what an `SL_SENSOR_ELEVATION_MASK` needs. `min_elevation_rad` is a
 * *geometric* elevation, the straight line of sight before the atmosphere bends it, while a
 * station's published "visible above 10 degrees" is the *apparent* one; the two differ by 5.3' at
 * 10 degrees and 34' at the horizon, which is 0.81 s to 20.3 s of pass edge with the same sign on
 * every edge of every pass. Converting the threshold once is exact rather than an approximation --
 * `h -> h + R(h)` is strictly increasing, so one apparent mask is exactly one geometric mask -- and
 * it costs nothing per sample, which is why this is a scalar conversion and not a field on
 * `sl_sensor`.
 */
sl_status sl_refraction_apparent(double true_altitude_rad, double pressure_hpa,
                                 double temperature_k, double *out_apparent_rad);
sl_status sl_refraction_geometric(double apparent_altitude_rad, double pressure_hpa,
                                  double temperature_k, double *out_true_rad);

/*
 * Stellar aberration: the special-relativistic map between the direction light *has* and the
 * direction an observer moving at `velocity_mps` *sees*. Both vectors are in the same frame as the
 * velocity; the output is a unit vector whatever the input's length was. `SL_ERR_DOMAIN` for a
 * velocity at or above the speed of light, `SL_ERR_INVALID_ARGUMENT` for a direction of zero length
 * or a non-finite component.
 *
 * `sl_aberration_apply` goes natural -> apparent, which is what predicts where a moving observer
 * will see a source; `sl_aberration_remove` goes apparent -> natural, which is what reduces a
 * measurement. They are exact inverses -- the reverse boost, not a sign flip on a small-angle term.
 *
 * The velocity that matters for a satellite observed from the ground is *not* the Earth's 29.8 km/s
 * orbital motion, the famous 20.5" constant of aberration: the station and the satellite share that
 * velocity and a shared velocity cancels in a geocentric frame. What is left is the station's own
 * rotation about the Earth's axis, which `sl_diurnal_velocity` builds as `omega x r` -- 465.1 m/s at
 * the equator, falling as cos(latitude). That is worth 0.320" at the equator and 0.262" at 34.9
 * degrees, a hundred times an astrometric plate's centroiding error, and it turns with the hour
 * angle rather than averaging out. Pass the station's position in the observation frame and
 * `{0, 0, SL_EARTH_ROTATION_RATE}`; taking the rotation as a vector rather than a scalar rate is
 * what lets polar motion tilt the pole.
 *
 * Verified against ERFA `ab` (the SOFA lineage) on twenty station-and-direction pairs, worst
 * separation 6.2e-17 rad.
 */
sl_status sl_diurnal_velocity(const sl_vec3 *station_m, const sl_vec3 *angular_velocity_rad_s,
                              sl_vec3 *out_velocity_mps);
sl_status sl_aberration_apply(const sl_vec3 *velocity_mps, const sl_vec3 *natural,
                              sl_vec3 *out_apparent);
sl_status sl_aberration_remove(const sl_vec3 *velocity_mps, const sl_vec3 *apparent,
                               sl_vec3 *out_natural);

/*
 * What one ground cell saw over the horizon.
 *
 * The struct is `sl_coverage_metrics` and the function that fills it is `sl_coverage_scan`, and the
 * two names differ because they have to: a typedef name and a function name share one namespace in
 * C, so `sl_coverage_metrics` could not be both. C++ hides the type behind the function instead of
 * rejecting the pair, which is why only the C99 build catches it -- and why it is in the matrix.
 */
typedef struct sl_coverage_metrics {
    uint32_t passes;
    double coverage_fraction; /* of the horizon the cell was covered for */
    double max_gap_s;         /* the longest wait between passes */
    double mean_gap_s;
    double mean_response_time_s; /* the mean wait from an arbitrary instant to the next pass */
} sl_coverage_metrics;

/* The whole-constellation figures of merit. */
typedef struct sl_constellation_fom {
    uint32_t point_count;
    double mean_coverage;
    double min_coverage; /* the worst cell */
    double worst_max_gap_s;
    double mean_max_gap_s;
    double mean_response_time_s;
    double worst_response_time_s;
    uint32_t fully_covered_points;
} sl_constellation_fom;

/* One access interval, seconds from the start of the scan. */
typedef struct sl_window {
    double rise_s;
    double set_s;
} sl_window;

/*
 * Ground cells on a spherical Earth from a latitude and a longitude grid, both in RADIANS -- where
 * the Python binding takes degrees, because the rest of this API is radians and a C caller has no
 * keyword argument to remind them which.
 *
 * Writes `lat_count * lon_count` cells, longitude-major within each latitude, through the two-call
 * protocol. Each cell arrives as an `sl_state` whose `position` is the centre in ECEF metres and
 * whose `velocity` is the outward unit zenith -- a reused shape rather than a fourth struct, and
 * the one the two entry points below take.
 */
sl_status sl_grid_cells(const double *lats_rad, size_t lat_count, const double *lons_rad,
                        size_t lon_count, sl_state *out, size_t cap, size_t *out_len);

/*
 * Per-cell coverage and revisit metrics for one circular orbit `(a, i, RAAN, phase)` over
 * `[0, horizon_s]` seconds, written to `out[0..n]` aligned with `cells`.
 *
 * The ground track is analytic -- a circular orbit in an Earth-fixed frame, identical to the Python
 * binding's, so the two answer the same question. A coverage sweep asks about geometry over days,
 * where the sensor model changes the answer far more than a J2 secular drift would, and an
 * integration here would put a step-size choice between the caller and a figure of merit.
 *
 * What it is not is a fixed-cadence sampler: the step is bounded by the sensor's own maximum rate,
 * so no pass is stepped over however short, and every edge is refined to `tol_s` seconds by Brent.
 * `threads` splits the cells; 0 means one.
 */
sl_status sl_coverage_scan(double a_m, double inclination_rad, double raan_rad, double phase_rad,
                           const sl_sensor *sensor, const sl_state *cells, size_t n,
                           double horizon_s, double tol_s, uint32_t threads,
                           sl_coverage_metrics *out);

/*
 * The raw access windows over a single cell -- the engine's primary product, which
 * `sl_coverage_scan` only aggregates. Two-call protocol.
 */
sl_status sl_access_windows(double a_m, double inclination_rad, double raan_rad, double phase_rad,
                            const sl_sensor *sensor, const sl_state *cell, double horizon_s,
                            double tol_s, sl_window *out, size_t cap, size_t *out_len);

/*
 * Aggregate per-cell metrics into whole-constellation figures of merit. `weights` is NULL to weight
 * every cell equally, or `n` values -- `cos(latitude)` is the usual choice, so that a lat/lon
 * grid's crowded poles do not outvote its equator.
 */
sl_status sl_figures_of_merit(const sl_coverage_metrics *cells, size_t n, const double *weights,
                              sl_constellation_fom *out);

/* ---- Semi-analytical propagation (DSST) ---- */

/*
 * The most zonal harmonics `sl_dsst_params` carries, J2 through J17.
 *
 * An inline array rather than a pointer and a count, so the parameter block is a value you put on
 * the stack and copy, with nothing to free and no lifetime to outlive. DSST's zonal theory is a
 * low-degree one: the default stops at J5 and Orekit's at J6. The tesseral field, which is where
 * degree 120 lives, is a different entry point.
 */
#define SL_DSST_MAX_ZONAL 16

/* The field, the spacecraft and the epoch a DSST propagation runs against. */
typedef struct sl_dsst_params {
    double mu;                       /* central-body GM, m^3/s^2 */
    double earth_radius_m;           /* central-body equatorial radius, metres */
    double epoch_j2000_seconds;      /* the initial epoch on the J2000 timeline */
    double reflectivity;             /* radiation-pressure coefficient Cr */
    double area_over_mass;           /* cross-section over mass, m^2/kg */
    double mu_sun;                   /* solar GM, m^3/s^2 */
    double mu_moon;                  /* lunar GM, m^3/s^2 */
    double zonal[SL_DSST_MAX_ZONAL]; /* [J2, J3, ...], indexed from degree 2 */
    uint32_t zonal_count;            /* how many of `zonal` are read; 1..=SL_DSST_MAX_ZONAL */
} sl_dsst_params;

/*
 * Fill `p` with the defaults: EIGEN-6S mu and radius, the zonals through J5, Cr = 1.5,
 * area/mass = 0.02 m^2/kg, epoch J2000 itself, and the DE-series solar and lunar GM. These are the
 * same defaults the Python binding takes, so a call that names no field answers identically in
 * both. `sl_dsst_params_init(NULL)` is a no-op.
 */
void sl_dsst_params_init(sl_dsst_params *p);

/*
 * DSST works in *mean equinoctial* elements `[a, k, h, q, p, lambda]`, with `k = e cos(varpi)`,
 * `h = e sin(varpi)`, `q = tan(i/2) cos(Omega)`, `p = tan(i/2) sin(Omega)` and
 * `lambda = M + varpi`; `a` is metres and the rest are dimensionless or radians.
 *
 * Every one of the four calls below takes that state in `initial`, propagates it to each of the `n`
 * epochs in `times_seconds` (seconds from the initial epoch, in any order, positive or negative),
 * and writes `6 * n` doubles to `out` — one sextuple per epoch, in the order asked for.
 *
 * The `_mean` calls answer in mean elements: the orbit-averaged drift, integrated with steps far
 * longer than an orbit. That is what makes DSST cheap over long arcs, and it is the right answer
 * for secular questions — node regression, apsidal precession, a repeat-ground-track drift budget.
 * The `_osculating` calls add each contribution's analytic short-period correction, so what comes
 * back is the instantaneous orbit rather than its average.
 *
 * `sl_dsst_zonal_*` uses the zonal geopotential alone. `sl_dsst_mean` and `sl_dsst_osculating` add
 * Sun and Moon third-body attraction and solar radiation pressure, with the Sun and Moon taken from
 * the built-in analytic ephemerides advanced from `epoch_j2000_seconds`. The tesseral field is
 * zero in both: a tesseral mean rate is identically zero away from a resonance, so for every orbit
 * except a repeat-ground-track one this is the whole conservative mean rate. Drag is not part of
 * these propagators.
 *
 * A non-finite input, a `zonal_count` outside 1..=SL_DSST_MAX_ZONAL or `n == 0` is
 * `SL_ERR_INVALID_ARGUMENT`; a non-positive semi-major axis is `SL_ERR_DOMAIN`. Nothing is written
 * to `out` unless every epoch was accepted.
 */
sl_status sl_dsst_zonal_mean(const sl_dsst_params *params, const double *initial,
                             const double *times_seconds, size_t n, double *out);
sl_status sl_dsst_zonal_osculating(const sl_dsst_params *params, const double *initial,
                                   const double *times_seconds, size_t n, double *out);
sl_status sl_dsst_mean(const sl_dsst_params *params, const double *initial,
                       const double *times_seconds, size_t n, double *out);
sl_status sl_dsst_osculating(const sl_dsst_params *params, const double *initial,
                             const double *times_seconds, size_t n, double *out);

/* ---- Estimation ---- */

/* One scalar observation, linearised at the current iterate. */
typedef struct sl_residual_row {
    double jacobian[6]; /* d(residual)/d(state) */
    double residual;    /* computed(state) - observed, in the measurement's own units */
    double weight;      /* 1/sigma^2 */
} sl_residual_row;

/*
 * The callback `sl_gauss_newton` calls once per iterate. Fill all `row_count` rows from `state`
 * and return `SL_OK`; return anything else to abandon the solve, which then reports
 * `SL_ERR_COMPUTE` and names the code you returned.
 *
 * `state` is six doubles and `rows` is exactly `row_count` entries — both are the library's, valid
 * for the duration of the call and no longer. `user_data` is whatever was handed to
 * `sl_gauss_newton`, unchanged.
 *
 * The row count is fixed for the whole solve rather than reported per call, because an orbit
 * determination has the observations it has: a model that changed its measurement count between
 * iterations would be minimising a different problem each time. It also means the buffer is
 * allocated once, by this library, and your callback never allocates.
 *
 * The callback must not throw or longjmp out. It runs underneath Rust frames, and unwinding
 * through them is undefined behaviour, not a stack trace.
 */
typedef sl_status (*sl_residual_fn)(void *user_data, const double *state, sl_residual_row *rows,
                                    size_t row_count);

/* Stopping criteria and damping schedule. */
typedef struct sl_gauss_newton_options {
    uint32_t max_iterations;    /* give up after this many */
    double cost_tolerance;      /* converged below this fractional fall in the weighted cost */
    double step_tolerance;      /* converged below this correction, relative to the state norm */
    double initial_lambda;      /* initial Levenberg-Marquardt damping; 0.0 is pure Gauss-Newton */
    double absolute_cost_floor; /* converged once the weighted cost reaches this */
} sl_gauss_newton_options;

/*
 * Fill `o` with the defaults, read from the Rust solver rather than written down twice.
 * `sl_gauss_newton_options_init(NULL)` is a no-op.
 */
void sl_gauss_newton_options_init(sl_gauss_newton_options *o);

/* A solved state, its formal covariance, and what the solve cost. */
typedef struct sl_estimate {
    double state[6];
    double covariance[36]; /* 6x6 row-major; the square roots of the diagonal are the 1-sigma */
    double cost;           /* the final weighted sum of squared residuals */
    uint32_t iterations;
    bool converged; /* a criterion was met, as against the iteration cap being reached */
} sl_estimate;

/*
 * Gauss-Newton / Levenberg-Marquardt least squares over a six-state: minimise the weighted sum of
 * squared residuals your `model` reports, from `x0`.
 *
 * `options` may be NULL, which takes the defaults. A rank-deficient normal system — observations
 * that never make some direction of the state observable — is `SL_ERR_DOMAIN`, and reaching
 * `max_iterations` is not a failure: it returns `SL_OK` with `converged` false, because a
 * half-converged state and its covariance are still the best answer available and throwing them
 * away would leave the caller with nothing to look at.
 */
sl_status sl_gauss_newton(const sl_gauss_newton_options *options, const double *x0,
                          sl_residual_fn model, void *user_data, size_t row_count,
                          sl_estimate *out);

/*
 * A square-root information filter over a six-state: fold scalar measurements in one at a time and
 * solve whenever you like, then keep folding.
 *
 * Where `sl_gauss_newton` re-linearises the whole arc at each iterate, this accumulates a linear
 * system incrementally and never revisits a row — the right shape when the measurements arrive
 * over time, or when there are more of them than fit in memory at once. Rows are rotated into an
 * upper-triangular information square root by Givens rotations, so nothing squares the condition
 * number the way forming `H^T W H` does.
 *
 * The handle is opaque and owns its own memory; every one from `sl_srif_open` needs an
 * `sl_srif_close`.
 */
typedef struct sl_srif sl_srif;

/*
 * Open a filter, empty or with a prior.
 *
 * `x0` and `information_sqrt` are both NULL for an empty filter — no information, so a solve fails
 * until enough rows have been folded in — or both non-NULL for a prior, where `information_sqrt`
 * is the upper-triangular `chol(P0^-1)`, 36 doubles row-major. One without the other is
 * `SL_ERR_INVALID_ARGUMENT` rather than a shorthand: a prior state with no prior information says
 * nothing, and prior information about no state has nowhere to sit.
 */
sl_status sl_srif_open(const double *x0, const double *information_sqrt, sl_srif **out);

/* Release a filter. `sl_srif_close(NULL)` is a no-op. */
void sl_srif_close(sl_srif *f);

/*
 * Fold one scalar measurement `z = jacobian . x + noise`, weighted `1/sigma^2`, into the filter.
 * A zero weight is accepted and contributes nothing, which is how a rejected observation is
 * carried without changing the row indexing.
 */
sl_status sl_srif_add_row(sl_srif *f, const double *jacobian, double z, double weight);

/*
 * The state the folded rows imply, six doubles to `out`. `SL_ERR_DOMAIN` while the rows leave some
 * direction of the state unobservable — with a six-state that means fewer than six independent
 * rows, which is a statement about their geometry and not only their number.
 */
sl_status sl_srif_solve(const sl_srif *f, double *out);

/* The 6x6 formal covariance, 36 doubles row-major. Same rank condition as the solve. */
sl_status sl_srif_covariance(const sl_srif *f, double *out);

/*
 * How well the rows agreed: the post-fit sum of squares rotated past the state, and how many rows
 * produced it. Divided by `rows - 6` this is the variance factor, and a value far from 1 says the
 * weights are not the measurement noise.
 */
sl_status sl_srif_residual_stats(const sl_srif *f, double *out_sum_of_squares, uint32_t *out_rows);

/* ---- GNSS ---- */

/* Which constellation a broadcast record belongs to. A PRN alone is ambiguous in a mixed file. */
typedef enum sl_gnss_system {
    SL_GNSS_GPS = 0,
    SL_GNSS_GALILEO = 1,
    SL_GNSS_BEIDOU = 2,
    SL_GNSS_QZSS = 3,
    SL_GNSS_NAVIC = 4,
    SL_GNSS_SYSTEM_FORCE_INT32 = 0x7FFFFFFF /* see the note on sl_precision */
} sl_gnss_system;

/* One satellite's broadcast position and velocity, ECEF, with what identifies it. */
typedef struct sl_gnss_state {
    int32_t system; /* an sl_gnss_system */
    uint32_t prn;
    sl_state state;
} sl_gnss_state;

/* One GLONASS satellite, PZ-90. GLONASS numbers by orbital slot rather than by PRN. */
typedef struct sl_glonass_state {
    uint32_t slot;
    sl_state state;
} sl_glonass_state;

/*
 * Propagate every record in a RINEX-3 navigation file to `tk_seconds` (GNSS) or `dt_seconds`
 * (GLONASS) from each record's own reference epoch. Two-call protocol.
 *
 * `text` is passed with an explicit length rather than NUL-terminated -- the only place in this
 * header that is so. A navigation file is a file, not a token: `sl_sgp4_open` scans at most
 * `MAX_C_STRING` bytes for a terminator, on purpose, and a RINEX file is routinely longer than
 * that bound. `strlen(text)` is the right thing to pass for a NUL-terminated buffer.
 *
 * A file with no records of that kind is zero results and `SL_OK`, not a failure: a GPS-only file
 * genuinely has no GLONASS in it. Text that is not UTF-8 is `SL_ERR_INVALID_ARGUMENT`.
 */
sl_status sl_rinex_gnss_positions(const char *text, size_t text_len, double tk_seconds,
                                  sl_gnss_state *out, size_t cap, size_t *out_len);
sl_status sl_rinex_glonass_positions(const char *text, size_t text_len, double dt_seconds,
                                     sl_glonass_state *out, size_t cap, size_t *out_len);

/* The time scale a `TIME OF FIRST OBS` record names. GPS is what the format prescribes when the
 * record is absent or names nothing, so it is the value an absent record reports -- not a
 * separate "unknown". */
typedef enum sl_time_system {
    SL_TIME_SYSTEM_GPS = 0,
    SL_TIME_SYSTEM_GLO = 1,
    SL_TIME_SYSTEM_GAL = 2,
    SL_TIME_SYSTEM_TAI = 3,
    SL_TIME_SYSTEM_UTC = 4,
    SL_TIME_SYSTEM_QZS = 5,
    SL_TIME_SYSTEM_BDT = 6,
    SL_TIME_SYSTEM_IRN = 7,
    SL_TIME_SYSTEM_FORCE_INT32 = 0x7FFFFFFF /* see the note on sl_precision */
} sl_time_system;

/*
 * The header records of a RINEX 3 observation file.
 *
 * `interval`, `approx_position` and `antenna_delta_hen` are NaN -- all three components, for the
 * two triples -- where the header declares no such record. That is this ABI's way of saying
 * absent, the same one `sl_decay_params_init` uses, and it is why they are not pointers.
 *
 * `antenna_delta_hen` is height above the marker, then east, then north eccentricity. It is not a
 * Cartesian triple, which is why it is a plain array where `approx_position` could have been an
 * `sl_vec3`; both are written the same way so neither invites the other's reading.
 */
typedef struct sl_obs_header {
    double version;
    double interval;             /* seconds */
    double approx_position[3];   /* metres, the file's terrestrial frame */
    double antenna_delta_hen[3]; /* metres: height, east, north */
    sl_time_system time_system;  /* the scale `seconds_j2000` is counted in */
    char file_system;            /* 'G', 'R', 'E', 'C', 'J', 'I', 'S', or 'M' for mixed */
    char marker_name[61];        /* NUL-terminated; empty where the record is absent */
} sl_obs_header;

/* One declared observation descriptor, e.g. "C1C", NUL-terminated. The `I` and `X` descriptors are
 * written with a blank third column in the file and reported here without it, which is also the
 * spelling `sl_rinex_obs_series` matches on. */
typedef struct sl_obs_descriptor {
    char code[4];
} sl_obs_descriptor;

/* One satellite's observation of one descriptor, at one epoch. */
typedef struct sl_obs_sample {
    double seconds_j2000; /* the file's own time system, not converted */
    double value;         /* the unit its descriptor carries: metres for `C`, *cycles* for `L` */
    int32_t lost_lock;    /* 1 where the receiver declares a loss of lock, 0 otherwise */
} sl_obs_sample;

/*
 * Read a RINEX 3 observation file: its header, the descriptors one constellation declares, or one
 * satellite's series of one descriptor. `text` is length-counted for the reason
 * `sl_rinex_gnss_positions` states. The last two are two-call.
 *
 * There is deliberately no entry point for the whole parsed file. It is a per-constellation
 * descriptor map over epochs of per-satellite *ragged* slot lists, and that shape reaches C only
 * as an opaque handle; these three are what a caller opens the file for. Each re-parses the text,
 * which is the trade `sl_rinex_gnss_positions` already makes.
 *
 * `system` and `prn` are the two halves of a RINEX identifier -- 'G' and 9 for `G09`. A
 * constellation the file never mentions declares no descriptors, which is zero results and
 * `SL_OK`. A descriptor the header never declared for that constellation is
 * `SL_ERR_INVALID_ARGUMENT` rather than an empty series: it is the difference between "not
 * observed" and "asked for the wrong thing". Epochs the satellite is absent from, or carries no
 * value at, are skipped.
 */
sl_status sl_rinex_obs_header(const char *text, size_t text_len, sl_obs_header *out);
sl_status sl_rinex_obs_descriptors(const char *text, size_t text_len, char system,
                                   sl_obs_descriptor *out, size_t cap, size_t *out_len);
sl_status sl_rinex_obs_series(const char *text, size_t text_len, char system, uint32_t prn,
                              const char *code, sl_obs_sample *out, size_t cap, size_t *out_len);

/*
 * The GLONASS FDMA carrier frequencies for channel `k`: `(1602.0 + k*0.5625)` MHz on L1 and
 * `(1246.0 + k*0.4375)` MHz on L2. Total -- a channel outside the signed-byte range the standard
 * uses returns NaN, which is the answer.
 */
double sl_glonass_l1_frequency(int32_t channel);
double sl_glonass_l2_frequency(int32_t channel);

/* The five dilution-of-precision figures. */
typedef struct sl_dop {
    double gdop; /* position and clock together */
    double pdop;
    double hdop;
    double vdop;
    double tdop;
} sl_dop;

/*
 * Dilution of precision for a receiver seeing `n` satellites, all ECEF metres. Local vertical is
 * the geocentric radial. Fewer than four satellites, or a geometry whose design matrix is
 * rank-deficient, is `SL_ERR_DOMAIN` -- where the Python binding returns `None` and leaves the
 * caller to work out which of the two applied.
 */
sl_status sl_dilution_of_precision(const sl_vec3 *receiver, const sl_vec3 *satellites, size_t n,
                                   sl_dop *out);

/* Earth's magnetic field at an ECEF position, tesla, from IGRF-14 degree and order 13. `year` is
 * decimal; the model is valid over roughly 2025-2030. */
sl_status sl_igrf_field_ecef(const sl_vec3 *position_m, double year, sl_vec3 *out_tesla);

/* ---- Ionosphere ---- */

/*
 * The broadcast (Klobuchar) L1 ionospheric slant delay, metres, from the eight broadcast
 * coefficients. `alpha` and `beta` are four doubles each.
 */
sl_status sl_klobuchar_iono_delay(const double *alpha, const double *beta, double latitude_rad,
                                  double longitude_rad, double elevation_rad, double azimuth_rad,
                                  double gps_seconds_of_day, double *out_metres);

/* The first-order ionospheric range delay, `40.3 * TEC / f^2`, metres, from a slant TEC in TECU
 * and a carrier frequency in hertz. Total. */
double sl_ionospheric_path_delay(double stec_tecu, double frequency_hz);

/*
 * The second-order ionospheric group delay, metres -- the geomagnetic `1/f^3` term Orekit omits,
 * which survives an ionosphere-free combination as an L1/L2 asymmetry. Signed. Total.
 *
 * `pierce_height_m` places the thin shell; 350 000 is the usual choice. The geomagnetic field at
 * the pierce point is the IGRF-14 centred dipole, which is an approximation the full model above
 * does not make -- it is good to a few per cent over most of the Earth and worse over the South
 * Atlantic anomaly.
 */
double sl_second_order_ionospheric_delay(double latitude_rad, double longitude_rad,
                                         double elevation_rad, double azimuth_rad, double stec_tecu,
                                         double frequency_hz, double pierce_height_m);

/* The thin-shell obliquity factor mapping a vertical TEC to a slant one. Total. */
double sl_ionospheric_mapping_factor(double elevation_rad, double ionosphere_height_m,
                                     double earth_radius_m);

/*
 * NeQuick slant total electron content along the receiver-to-satellite ray, TECU. The Galileo
 * variant is driven by the three broadcast effective-ionisation coefficients.
 * `month` is 1-12 and `seconds_in_utc_day` is within one day; both endpoints are geodetic, and as
 * everywhere here the angles are radians.
 *
 * These are the slowest functions in this header by a wide margin -- the ray is integrated
 * through a profile, not evaluated -- so a per-satellite loop over a full constellation is a
 * measurable cost, not a rounding error.
 */
sl_status sl_nequick_galileo_stec(const double *alpha, uint32_t month, double seconds_in_utc_day,
                                  const sl_geodetic *receiver, const sl_geodetic *satellite,
                                  double *out_tecu);

/* ---- Troposphere ---- */

/*
 * The Saastamoinen zenith delays, metres: the hydrostatic one from surface pressure (hPa),
 * geodetic latitude and ellipsoidal height, the wet one from surface temperature (K) and the
 * water-vapour partial pressure (hPa). Both total.
 */
double sl_saastamoinen_zenith_hydrostatic(double pressure_hpa, double latitude_rad,
                                          double height_m);
double sl_saastamoinen_zenith_wet(double temperature_k, double vapour_pressure_hpa);

/* A mapping function's two factors: what each zenith delay is multiplied by at that elevation. */
typedef struct sl_mapping {
    double hydrostatic;
    double wet;
} sl_mapping;

/*
 * The tropospheric mapping functions, in the order they were published: Niell (1996) from latitude,
 * height and season; GMF (2006), which adds longitude; and VMF1 and VMF3, which take the `a`
 * coefficients an operational grid supplies instead of deriving them.
 *
 * `elevation_rad` at or below the horizon is `SL_ERR_DOMAIN`: a mapping function is a ratio to the
 * zenith delay along a ray that leaves the atmosphere, and at the horizon it does not.
 */
sl_status sl_niell_mapping(double latitude_rad, double height_m, double day_of_year,
                           double elevation_rad, sl_mapping *out);
sl_status sl_gmf_mapping(double latitude_rad, double longitude_rad, double height_m,
                         double day_of_year, double elevation_rad, sl_mapping *out);
sl_status sl_vmf1_mapping(double ah, double aw, double latitude_rad, double height_m,
                          double day_of_year, double elevation_rad, sl_mapping *out);
sl_status sl_vmf3_mapping(double ah, double aw, double latitude_rad, double longitude_rad,
                          double height_m, double day_of_year, double elevation_rad,
                          sl_mapping *out);

/* ---- Atmospheres, tides, density calibration and space weather ---- */

/*
 * NRLMSISE-00 total mass density, kg/m^3. `doy` is the day of year, `sec` UT seconds into the day,
 * the altitude kilometres and the two angles degrees -- the model's own units, not this header's
 * SI, because they are what the published subroutine takes and translating them here would hide a
 * conversion inside a binding. Total.
 *
 * The local solar time the model needs is derived from `sec` and the longitude.
 *
 * A non-finite argument answers NaN. The published subroutine does not propagate one -- measured,
 * a NaN altitude comes back as 2.9e-3 kg/m^3, which is a sea-level density and so a plausible
 * number rather than an obvious failure.
 */
double sl_nrlmsise00_density(int32_t doy, double sec, double altitude_km, double latitude_deg,
                             double longitude_deg, double f107, double f107a, double ap);

/* How many doubles `sl_sdtm1_density_interval` writes. */
#define SL_SDTM1_INTERVAL 2

/*
 * SDTM-1 total mass density, kg/m^3 -- this library's own thermosphere model, fitted to open
 * accelerometer densities. Same argument shape and same units as `sl_nrlmsise00_density`, except
 * that the day of year is a double so it can carry a fraction, and that the geomagnetic argument is
 * not an index: `storm` is `sl_sdtm1_storm_response` over the `ap` history.
 *
 * `f107` is the previous day's observed 10.7 cm flux and `f107a` its 81-day TRAILING mean -- the
 * fit's convention, not the centred mean NRLMSISE-00 specifies. Total; a non-finite argument
 * answers NaN.
 */
double sl_sdtm1_density(double doy, double sec, double altitude_km, double latitude_deg,
                        double longitude_deg, double f107, double f107a, double storm);

/*
 * SDTM-1 exospheric temperature T_inf, K, at the same point and drivers. The quantity the fit
 * parameterizes -- the density is a Bates profile above it. Total; non-finite in, NaN out.
 */
double sl_sdtm1_exospheric_temperature(double doy, double sec, double altitude_km,
                                       double latitude_deg, double longitude_deg, double f107,
                                       double f107a, double storm);

/*
 * SDTM-1's geomagnetic storm driver: the shipped heating and nitric-oxide cooling state run over an
 * index history. `history` is the `ap` series oldest first, ending at the interval BEFORE the sample,
 * readable for `n` doubles; `cadence_s` is that series' step, 10800 for CelesTrak's 3-hourly `ap`,
 * which is what the fit used. Nitric oxide remembers a storm for days, and a truncated history reads
 * as a different storm: pass eleven days, 88 intervals.
 *
 * `n == 0` answers 0. A null `history`, or a cadence that is not finite and positive, answers NaN.
 */
double sl_sdtm1_storm_response(const double *history, size_t n, double cadence_s);

/*
 * SDTM-1's log-normal density interval about `density`: `[lower, upper]` in kg/m^3,
 * `SL_SDTM1_INTERVAL` doubles out. `z` is the standard-normal multiplier -- 1.645 for 90 %, 1.960
 * for 95 %, 2.576 for 99 %. Which flux x storm regime applies is decided by `f107a` and `storm`, so
 * they must be the values `density` was evaluated with.
 */
sl_status sl_sdtm1_density_interval(double density, double f107, double f107a, double storm,
                                    double z, double *out);

/* The field a Brouwer-Lyddane mean-element theory is built over, and the state it starts from. */
typedef struct sl_brouwer_lyddane_params {
    sl_state initial;      /* osculating, metres and metres per second */
    double mu;             /* m^3/s^2 */
    double earth_radius_m; /* what the coefficients are normalized to */
    double c20;            /* normalized zonal coefficients, degrees 2 to 5 */
    double c30;
    double c40;
    double c50;
    double m2; /* the Lyddane drag term: a secular mean-motion rate, rad/s^2 */
} sl_brouwer_lyddane_params;

/*
 * Brouwer-Lyddane analytical propagation: the osculating state at each of `n` offsets from the
 * epoch, written into `n` `sl_state`. Unlike the two-call functions below the length is the `n` the
 * caller passed, so there is nothing to size.
 */
sl_status sl_brouwer_lyddane_propagate(const sl_brouwer_lyddane_params *p,
                                       const double *times_seconds, size_t n, sl_state *out);

/*
 * Which conventional pole the pole-tide wobble variables are measured from.
 *
 * This is not a detail: the two poles are ~60 mas apart, which is larger than the polar motion the
 * pole tide responds to, and at a representative (xp, yp) they give `dC21` opposite signs. The
 * default everywhere in this library is the 2018 secular pole; `SL_MEAN_POLE_IERS2010` exists
 * because the Orekit fixtures were taken against the retired one.
 */
typedef enum sl_mean_pole {
    SL_MEAN_POLE_SECULAR2018 = 0,         /* Chapter 7 Eq. 21, as amended 2018-02-01 */
    SL_MEAN_POLE_IERS2010 = 1,            /* the retired Table 7.7 mean pole */
    SL_MEAN_POLE_FORCE_INT32 = 0x7FFFFFFF /* see the note on sl_precision */
} sl_mean_pole;

/*
 * IERS 2010 solid-tide Step-2 frequency corrections: `[dC20, dC21, dS21, dC22, dS22]` out, from the
 * Greenwich mean sidereal time and the five Delaunay arguments in. All radians; `delaunay` is
 * readable for five doubles and `out` writable for five.
 */
sl_status sl_solid_tide_step2(double gmst_rad, const double *delaunay, double *out);

/*
 * The two pole tides: `[dC21, dS21]` out, two doubles each, from the polar motion in radians at `t`
 * seconds since J2000 (TT). They superpose on the same degree-2 order-1 band, so a caller who sums
 * them must give both the same `mean_pole` -- IERS 2010 section 6.4 and section 6.5 share the
 * definition of the wobble variables, and mixing the two poles is not a small error.
 */
sl_status sl_solid_pole_tide(double xp_rad, double yp_rad, double t_seconds, sl_mean_pole mean_pole,
                             double *out);
sl_status sl_ocean_pole_tide(double xp_rad, double yp_rad, double t_seconds, sl_mean_pole mean_pole,
                             double *out);

/* Three low-degree normalized zonal geopotential coefficients. */
typedef struct sl_zonal {
    double c20;
    double c30;
    double c40;
} sl_zonal;

/*
 * The IERS 2010 conventional zonal coefficients at `years_since_j2000` decimal years -- mean plus
 * secular drift, C20 zero-tide -- and the secular corrections on their own, which are what a caller
 * adds to a base field's C20/C30/C40 for time-variable gravity.
 */
sl_status sl_conventional_zonal_coefficients(double years_since_j2000, sl_zonal *out);
sl_status sl_time_variable_zonal_correction(double years_since_j2000, sl_zonal *out);

/* Sentman's free-molecular coefficients for one flat panel. */
typedef struct sl_drag_panel {
    double pressure; /* normal */
    double shear;    /* tangential */
    double drag;     /* along the incident flow */
    double lift;     /* across it */
} sl_drag_panel;

/*
 * The physical drag coefficients of a flat panel in free-molecular flow, and the molecular speed
 * ratio they are parameterised by. `cos_incidence` is the cosine of the angle between the panel
 * normal and the flow, `accommodation` the energy accommodation coefficient in [0, 1]. The speed
 * ratio is total; the panel refuses a non-positive temperature with `SL_ERR_DOMAIN`.
 */
sl_status sl_sentman_drag_coefficients(double speed_ratio, double cos_incidence,
                                       double accommodation, double wall_temperature_k,
                                       double atmosphere_temperature_k, sl_drag_panel *out);
double sl_molecular_speed_ratio(double relative_speed_ms, double temperature_k,
                                double mean_molar_mass);

/* How many drivers `sl_space_weather_drivers` writes. */
#define SL_DENSITY_DRIVERS 5

/* How many entries the NRLMSISE-00 storm-time ap array holds. */
#define SL_AP_ARRAY_LEN 7

/* How many coefficients a degree-4 density correction carries. */
#define SL_DENSITY_CORRECTION_COEFFICIENTS 25

/*
 * The nonlinear density drivers a regression is built over: `[flux, ap, flux^2, sqrt(ap), 1]`,
 * `SL_DENSITY_DRIVERS` doubles out.
 */
sl_status sl_space_weather_drivers(double f107, double f107_81, double ap, double *out);

/*
 * The seven-element NRLMSISE-00 storm-time ap array for one day and time, built from the 3-hourly
 * AP1..AP8 history in a CelesTrak `SW-All.csv`. Layout: `[daily Ap, current 3h ap, -3h, -6h, -9h,
 * mean(-12..-33h), mean(-36..-57h)]`, `SL_AP_ARRAY_LEN` doubles out. Feed it to the model's
 * array-Ap mode to resolve the storm-time surge the daily Ap smooths out.
 *
 * The day and its ~2.5-day look-back must both be in the file; that they are not is `SL_ERR_DOMAIN`
 * rather than a silent zero.
 */
sl_status sl_space_weather_ap_array(const char *csv_text, size_t text_len, int32_t year,
                                    uint32_t month, uint32_t day, double seconds_of_day,
                                    double *out);

/* One day of a CelesTrak `SW-All.csv`. */
typedef struct sl_space_weather_day {
    int32_t year;
    uint32_t month;
    uint32_t day;
    double ap;            /* the daily planetary amplitude */
    double f107_observed; /* 10.7 cm solar flux, sfu */
    double f107_center81; /* its 81-day centred mean, sfu */
} sl_space_weather_day;

/* Every day the file holds. Two-call protocol. */
sl_status sl_space_weather_days(const char *csv_text, size_t text_len, sl_space_weather_day *out,
                                size_t cap, size_t *out_len);

/* One density-ratio observation the calibration is fitted to. */
typedef struct sl_density_sample {
    double latitude_rad;
    double local_solar_time_h; /* [0, 24) */
    double log_ratio;          /* ln(rho_measured / rho_model) */
    double weight;
} sl_density_sample;

/* One height-normalized absolute log-density from a density file. */
typedef struct sl_density_observation {
    int32_t year;
    uint32_t month;
    uint32_t day;
    double latitude_rad;
    double local_solar_time_h;
    double ln_density_ref; /* ln rho lifted to the reference altitude */
} sl_density_observation;

/*
 * The two reductions of a TU Delft density file. `sl_density_data_samples` gives the orbit-mean
 * ratio to the background model -- the residual a correction is fitted to; `sl_density_altitude
 * _normalized` keeps the absolute amplitude, lifting `ln rho` to `reference_altitude_m` through an
 * exponential `scale_height_m`, which is the signal that tracks solar flux. Both two-call.
 *
 * The text is length-counted for the reason `sl_rinex_gnss_positions` states: it is a file, and a
 * file is longer than the terminator scan a NUL-terminated argument is bounded by.
 */
sl_status sl_density_data_samples(const char *text, size_t text_len, sl_density_sample *out,
                                  size_t cap, size_t *out_len);
sl_status sl_density_altitude_normalized(const char *text, size_t text_len,
                                         double reference_altitude_m, double scale_height_m,
                                         sl_density_observation *out, size_t cap, size_t *out_len);

/*
 * The HASDM-style density correction: fit `SL_DENSITY_CORRECTION_COEFFICIENTS` coefficients to the
 * samples by ridge regression, then evaluate the multiplicative factor they give at one latitude
 * and local solar time. `n` in the evaluator is the coefficient count and must be exactly that
 * constant -- a shorter vector is a different model, not a truncated one.
 *
 * A singular set of normal equations is `SL_ERR_COMPUTE`. The fit is two-call.
 */
sl_status sl_density_correction_fit(const sl_density_sample *samples, size_t n, double ridge,
                                    double *out, size_t cap, size_t *out_len);
sl_status sl_density_correction_factor(const double *coefficients, size_t n, double latitude_rad,
                                       double local_solar_time_h, double *out);

/*
 * Bin the samples into an `n_lat x n_lst` field of the mean `log_ratio`, row-major
 * `lat_bin * n_lst + lst_bin` with empty cells zero -- a density-field snapshot for POD. Two-call;
 * the result is always `n_lat * n_lst` long.
 */
sl_status sl_density_grid_snapshot(const sl_density_sample *samples, size_t n, size_t n_lat,
                                   size_t n_lst, double *out, size_t cap, size_t *out_len);

/* The geometry one look angle cuts out of a spherical Earth. */
typedef struct sl_swath {
    double earth_central_angle_rad; /* sub-satellite point to target */
    double elevation_rad;           /* of the spacecraft, seen from the target */
    double ground_range_m;
    double slant_range_m;
} sl_swath;

/*
 * That geometry, from the Earth radius, the altitude and the look angle off nadir. A look angle
 * past the horizon has no ground point at all, which is `SL_ERR_DOMAIN` and not a NaN: the answer
 * does not exist, which is a different statement from having computed one badly.
 */
sl_status sl_swath_geometry(double earth_radius_m, double altitude_m, double look_angle_rad,
                            sl_swath *out);

/*
 * ---------------------------------------------------------------------------------------------
 * Synthetic aperture radar: focusing, interferometry, tomography, polarimetry
 * ---------------------------------------------------------------------------------------------
 */

/* One complex sample, laid out as C's `double _Complex` is -- real part first. */
typedef struct sl_complex {
    double re;
    double im;
} sl_complex;

/* The acquisition a chirp-scaling focus is run against. */
typedef struct sl_csa_params {
    double prf;               /* pulse repetition frequency, Hz */
    double velocity;          /* effective platform velocity, m/s */
    double wavelength;        /* m */
    double r_ref;             /* reference slant range the scaling is anchored at, m */
    double range_sample_rate; /* Hz */
    double range_fm_rate;     /* transmitted chirp rate, Hz/s */
} sl_csa_params;

/* The acquisition the range-Doppler rates are derived from. */
typedef struct sl_rda_geometry {
    double prf;           /* Hz */
    double velocity;      /* m/s */
    double wavelength;    /* m */
    double r0_slant;      /* slant range of closest approach, m */
    double range_spacing; /* m per range sample */
} sl_rda_geometry;

/* The tomographic geometry an elevation spectrum is formed over. */
typedef struct sl_tomo_params {
    double wavelength;    /* m */
    double slant_range;   /* m */
    double loading;       /* diagonal loading of the covariance, relative */
    double lambda;        /* the sparsity weight SL1MMER solves under */
    size_t max_iter;      /* iteration cap of the sparse solve */
    double threshold_rel; /* what fraction of the peak still counts as a scatterer */
} sl_tomo_params;

/* One elevation and the complex reflectivity SL1MMER put there. */
typedef struct sl_scatterer {
    double elevation_m;
    sl_complex reflectivity;
} sl_scatterer;

/* One look's three-component Pauli scattering vector. */
typedef struct sl_polarimetric_sample {
    sl_complex s_hh;
    sl_complex s_hv;
    sl_complex s_vv;
} sl_polarimetric_sample;

/* The three parameters of the H/A/alpha decomposition. */
typedef struct sl_cloude_pottier {
    double entropy;
    double anisotropy;
    double alpha_rad;
} sl_cloude_pottier;

/* What an SBAS inversion resolves a stack into. */
typedef struct sl_deformation {
    double velocity_ms; /* line-of-sight, m/s */
    double height_m;    /* residual topographic height */
} sl_deformation;

/* How many complex entries a 3x3 coherency matrix holds. */
#define SL_COHERENCY_ENTRIES 9

/* A linear frequency-modulated chirp of `n` samples. Two-call. */
sl_status sl_lfm_chirp(size_t n, double chirp_rate, double sample_rate, sl_complex *out, size_t cap,
                       size_t *out_len);

/* Correlate a signal against a replica -- the compression step of every SAR pipeline. Two-call. */
sl_status sl_matched_filter(const sl_complex *signal, size_t signal_len, const sl_complex *replica,
                            size_t replica_len, sl_complex *out, size_t cap, size_t *out_len);

/*
 * The range-Doppler rates of one acquisition: the range cell migration in metres, and the azimuth
 * FM rate in Hz/s. Both are total -- a null geometry answers NaN. The secondary range compression
 * rate is not: past the Doppler bandwidth it does not exist, which is `SL_ERR_DOMAIN` rather than a
 * NaN, because a caller who asked outside the band asked the wrong question.
 */
double sl_range_migration(double f_eta, const sl_rda_geometry *geometry);
double sl_azimuth_fm_rate(const sl_rda_geometry *geometry);
sl_status sl_src_fm_rate(double f_eta, const sl_rda_geometry *geometry, double *out);

/*
 * Focus a raw SAR aperture with the chirp-scaling algorithm, and the single-line phase multiply it
 * is built from. The focus is two-call and its answer is `na * nr` samples, row-major by azimuth
 * line; `sl_apply_chirp_scaling` answers the `n` samples it was given.
 *
 * The focus transforms along both axes with a radix-2 FFT, so `na` and `nr` must both be powers of
 * two. An aperture that is not is `SL_ERR_INVALID_ARGUMENT` rather than a truncated scene.
 */
sl_status sl_chirp_scaling_focus(const sl_complex *raw, size_t na, size_t nr,
                                 const sl_csa_params *params, sl_complex *out, size_t cap,
                                 size_t *out_len);
sl_status sl_apply_chirp_scaling(const sl_complex *line, size_t n, double chirp_rate,
                                 double scaling, double tau_ref, double sample_rate,
                                 sl_complex *out, size_t cap, size_t *out_len);

/*
 * The interferogram of two co-registered images -- `s1 * conj(s2)` sample by sample -- and the
 * windowed coherence of the same pair. The coherence takes the raster shape because its window is
 * two-dimensional; the interferogram does not, because it is pointwise. Two-call, both.
 */
sl_status sl_interferogram(const sl_complex *s1, const sl_complex *s2, size_t n, sl_complex *out,
                           size_t cap, size_t *out_len);
sl_status sl_coherence(const sl_complex *s1, const sl_complex *s2, size_t rows, size_t cols,
                       size_t half_window, double *out, size_t cap, size_t *out_len);

/* Wrap a phase into `(-pi, pi]`. Total. */
double sl_wrap_phase(double phi_rad);

/* Unwrap a phase profile by accumulating branch jumps. Two-call. */
sl_status sl_unwrap_phase_1d(const double *wrapped, size_t n, double *out, size_t cap,
                             size_t *out_len);

/*
 * The three two-dimensional unwrappers, in the order they were published: quality-guided flood
 * fill, Goldstein branch cuts, and the weighted least-squares solution. Each takes a row-major
 * `rows x cols` raster and answers one of the same shape. Two-call, each.
 *
 * The three disagree on purpose. Flood fill is fastest and follows the residues wherever they lead;
 * branch cuts refuse to cross a cut and so leave inconsistent regions alone; least squares spreads
 * the inconsistency over the whole raster instead of localising it. Which one is right is a
 * property of the scene, not of the library.
 */
sl_status sl_unwrap_phase_2d(const double *wrapped, size_t rows, size_t cols, double *out,
                             size_t cap, size_t *out_len);
sl_status sl_unwrap_phase_2d_branch_cut(const double *wrapped, size_t rows, size_t cols,
                                        double *out, size_t cap, size_t *out_len);
sl_status sl_unwrap_phase_2d_lstsq(const double *wrapped, size_t rows, size_t cols, double *out,
                                   size_t cap, size_t *out_len);

/*
 * The residue charge at each `2x2` loop of the wrapped raster: `+1`, `-1` or zero, row-major over
 * the `(rows - 1) x (cols - 1)` loops. A raster with no residues is one every unwrapper agrees on,
 * and the count of non-zero entries is how hard the unwrapping is going to be. The charge is
 * `int8_t` because it is one of three values, and a wider type would invite a caller to believe
 * otherwise. Two-call.
 */
sl_status sl_residues(const double *wrapped, size_t rows, size_t cols, int8_t *out, size_t cap,
                      size_t *out_len);

/*
 * The Capon (minimum-variance) and Bartlett (beamforming) elevation spectra of one resolution cell.
 * `covariance` is the `baseline_count x baseline_count` sample covariance, row-major; the answer is
 * one power per requested elevation. Bartlett ignores `loading`, which is the whole difference
 * between them: Capon inverts the covariance and needs the diagonal loaded to do it, Bartlett does
 * not invert anything. Two-call, both.
 */
sl_status sl_capon_spectrum(const sl_complex *covariance, const double *baselines,
                            size_t baseline_count, const double *elevations, size_t elevation_count,
                            const sl_tomo_params *params, double *out, size_t cap, size_t *out_len);
sl_status sl_bartlett_spectrum(const sl_complex *covariance, const double *baselines,
                               size_t baseline_count, const double *elevations,
                               size_t elevation_count, const sl_tomo_params *params, double *out,
                               size_t cap, size_t *out_len);

/*
 * SL1MMER: the sparse elevation reconstruction, and the dense spectrum it is thresholded from.
 * Unlike the two spectra above, `y` is the *data* vector -- one complex sample per baseline, not a
 * covariance -- because SL1MMER solves for the scatterers directly rather than through a second
 * moment. The reconstruction answers however many scatterers cleared `threshold_rel`, which is why
 * it is two-call even though the spectrum's length is known in advance.
 */
sl_status sl_sl1mmer(const sl_complex *y, const double *baselines, size_t baseline_count,
                     const double *elevations, size_t elevation_count, const sl_tomo_params *params,
                     sl_scatterer *out, size_t cap, size_t *out_len);
sl_status sl_sl1mmer_spectrum(const sl_complex *y, const double *baselines, size_t baseline_count,
                              const double *elevations, size_t elevation_count,
                              const sl_tomo_params *params, double *out, size_t cap,
                              size_t *out_len);

/*
 * The PS-InSAR candidate score of every pixel in an amplitude stack, row-major
 * `epoch * pixels + pixel`, and the indices it selects. Two-call, both: the dispersion is `pixels`
 * long, the selection however many cleared the threshold.
 */
sl_status sl_amplitude_dispersion(const double *stack, size_t epochs, size_t pixels, double *out,
                                  size_t cap, size_t *out_len);
sl_status sl_select_persistent_scatterers(const double *dispersion, size_t n, double threshold,
                                          size_t *out, size_t cap, size_t *out_len);

/*
 * Invert an SBAS stack for the line-of-sight velocity and the residual topographic height. The
 * three arrays are one entry per interferogram and must agree in length; a design matrix that
 * cannot be inverted -- fewer than two interferograms, or baselines that do not separate the two
 * unknowns -- is `SL_ERR_COMPUTE`.
 */
sl_status sl_sbas_invert(const double *phases, const double *temporal_baselines,
                         const double *perp_baselines, size_t n, double wavelength,
                         double slant_range, double incidence_rad, sl_deformation *out);

/*
 * Polarimetry: the coherency matrix of a set of looks, the H/A/alpha decomposition of one such
 * matrix, and the three Pauli channel powers -- surface, double-bounce, volume -- of a single
 * scattering vector. The matrix is row-major and always `SL_COHERENCY_ENTRIES` long, which is why
 * it is the one two-call function here whose length never varies.
 */
sl_status sl_coherency_matrix(const sl_polarimetric_sample *samples, size_t n, sl_complex *out,
                              size_t cap, size_t *out_len);
sl_status sl_cloude_pottier_decompose(const sl_complex *t3, sl_cloude_pottier *out);
sl_status sl_pauli_powers(const sl_polarimetric_sample *sample, double *out);

/*
 * The 3 dB main-lobe width of an impulse response, in samples, and its peak-to-sidelobe ratio in
 * decibels -- the two numbers a compression is judged by. A profile with no resolvable main lobe
 * has neither, which is `SL_ERR_DOMAIN`: the Python bridge used to panic here, and a C caller has
 * no exception to catch, so a status is the only honest answer.
 */
sl_status sl_resolution_3db_samples(const double *magnitude, size_t n, double *out);
sl_status sl_peak_sidelobe_ratio_db(const double *magnitude, size_t n, double *out);

/*
 * ---------------------------------------------------------------------------------------------
 * Optical Earth observation: multi-view geometry, stereo, BRDF, pan-sharpening, detection
 * ---------------------------------------------------------------------------------------------
 */

/* A calibrated pinhole camera, as the collinearity equations take it. */
typedef struct sl_camera {
    double rotation[9];  /* world to camera, row-major */
    double center[3];    /* camera centre in world coordinates */
    double focal;        /* pixels */
    double principal[2]; /* [c_x, c_y], pixels */
} sl_camera;

/*
 * One image and the camera that took it. The pointer is borrowed for the duration of the call and
 * is not retained, so the caller may free it the moment the call returns.
 */
typedef struct sl_mvs_view {
    const double *image; /* row-major, `rows * cols` doubles */
    sl_camera camera;
} sl_mvs_view;

/* The tuning of a Semi-Global Match. */
typedef struct sl_sgm_config {
    size_t d_max;  /* disparities searched, d in [0, d_max) */
    size_t radius; /* census window radius; the window side is 2*radius + 1 */
    double p1;     /* penalty for a one-step disparity change -- what keeps a slope smooth */
    double p2;     /* penalty for a larger jump -- what lets a depth edge survive */
} sl_sgm_config;

/* The five knobs of the Harris corner detector. */
typedef struct sl_harris_config {
    double k;             /* of the response `det - k * trace^2` */
    size_t window_radius; /* structure-tensor window */
    double threshold_rel; /* keep a corner at least this strong, relative to the strongest */
    size_t nms_radius;    /* non-maximum suppression radius */
    size_t max_corners;   /* cap on how many are returned */
} sl_harris_config;

/* One detected corner and how strong it was. */
typedef struct sl_corner {
    double x; /* pixel column */
    double y; /* pixel row */
    double response;
} sl_corner;

/* The same feature seen in two views, by index into their corner lists. */
typedef struct sl_feature_match {
    size_t view_a;
    size_t corner_a;
    size_t view_b;
    size_t corner_b;
} sl_feature_match;

/* What a feature track triangulated to -- or, when `resolved` is false, that it did not. */
typedef struct sl_track_point {
    sl_vec3 position; /* meaningful only when `resolved` */
    bool resolved;    /* false when the track's rays are parallel and never meet */
} sl_track_point;

/* A ground point whose image coordinates are known. */
typedef struct sl_ground_control_point {
    double latitude;  /* degrees */
    double longitude; /* degrees */
    double height;    /* metres above the ellipsoid */
    double row;       /* measured image row, pixels */
    double col;       /* measured image column, pixels */
} sl_ground_control_point;

/* A pixel, as a sensor model answers one. */
typedef struct sl_image_point {
    double row;
    double col;
} sl_image_point;

/* The four 6S terms that stand between a surface and a sensor. */
typedef struct sl_atmospheric_coupling {
    double path_reflectance; /* what the atmosphere reflects before the ground is reached */
    double transmittance_down;
    double transmittance_up;
    double spherical_albedo; /* of the atmosphere, seen from below */
} sl_atmospheric_coupling;

/* An axis-aligned detection box. */
typedef struct sl_bbox {
    double x1;
    double y1;
    double x2;
    double y2;
} sl_bbox;

/*
 * Dense disparity by Semi-Global Matching of a rectified pair. Two-call; the answer is
 * `rows * cols` disparities, row-major, one per pixel of the left image.
 *
 * The census descriptor packs one bit per window neighbour into a 64-bit word, so `radius` may not
 * exceed 3: a wider window has more neighbours than there are bits, and losing the outer ones
 * silently would be a different algorithm wearing this one's name.
 */
sl_status sl_semi_global_match(const double *left, const double *right, size_t rows, size_t cols,
                               const sl_sgm_config *config, double *out, size_t cap,
                               size_t *out_len);

/*
 * Dense depth by plane-sweep multi-view stereo: for every pixel of the reference image, the depth
 * hypothesis that best matches the neighbours photometrically. Two-call; the answer is
 * `rows * cols` depths in the reference camera's frame, zero where nothing was consistent.
 */
sl_status sl_plane_sweep_mvs(const sl_mvs_view *reference, const sl_mvs_view *neighbors,
                             size_t neighbor_count, size_t rows, size_t cols, const double *depths,
                             size_t depth_count, size_t patch_radius, double *out, size_t cap,
                             size_t *out_len);

/*
 * Triangulate one world point from two or more views of it; `pixels` is `(u, v)` per camera.
 *
 * Rays that do not meet are a degenerate configuration rather than a bad input, so parallel
 * baselines are `SL_ERR_DOMAIN`: there is no point to answer, which is a different statement from
 * having computed one badly.
 */
sl_status sl_triangulate(const sl_camera *cameras, const double *pixels, size_t n, sl_vec3 *out);

/*
 * Build feature tracks across views and triangulate each. Two-call; the answer is one
 * `sl_track_point` per track.
 *
 * `corners` is the views' corner lists laid end to end and `corners_per_view` says how long each
 * one is, because C has no ragged array. A match naming a view or corner outside those lists is
 * `SL_ERR_INVALID_ARGUMENT`: the track builder indexes them directly, and the alternative to
 * checking here is a panic, which is not a value a C caller can receive.
 */
sl_status sl_sfm_triangulate(const sl_corner *corners, const size_t *corners_per_view,
                             size_t view_count, const sl_feature_match *matches, size_t match_count,
                             const sl_camera *cameras, sl_track_point *out, size_t cap,
                             size_t *out_len);

/*
 * Harris corners of one grayscale image. Two-call; the answer is however many cleared the
 * threshold, at most `max_corners`.
 */
sl_status sl_harris_corners(const double *image, size_t rows, size_t cols,
                            const sl_harris_config *config, sl_corner *out, size_t cap,
                            size_t *out_len);

/*
 * Fit a Grodecki-Dial bias correction to an RPC from ground control, and answer where the
 * corrected model puts one ground point.
 *
 * `affine` chooses the six-parameter fit over the two shifts; the affine one needs three control
 * points to be determined and the shift one needs one. `rpb_text` is the `.RPB` the vendor ships,
 * passed as text with its length rather than as a path -- this library does not read files.
 */
sl_status sl_rpc_block_adjust(const char *rpb_text, size_t text_len,
                              const sl_ground_control_point *gcps, size_t n, double latitude,
                              double longitude, double height, bool affine, sl_image_point *out);

/*
 * The two MODIS RTLSR kernels and the reflectance they combine into,
 * `f_iso + f_vol*K_vol + f_geo*K_geo`. All three are total: the kernels are trigonometry and have
 * an answer at every angle.
 */
double sl_ross_thick_kernel(double theta_s, double theta_v, double rel_azimuth);
double sl_li_sparse_reciprocal_kernel(double theta_s, double theta_v, double rel_azimuth);
double sl_brdf_reflectance(double f_iso, double f_vol, double f_geo, double theta_s, double theta_v,
                           double rel_azimuth);

/*
 * The 6S forward and inverse: what a surface of a given reflectance shows at the top of the
 * atmosphere, and what a top-of-atmosphere reflectance implies about the surface. Total, and exact
 * inverses of one another; a null coupling answers NaN.
 */
double sl_atmospheric_toa_from_surface(const sl_atmospheric_coupling *coupling,
                                       double surface_reflectance);
double sl_atmospheric_surface_from_toa(const sl_atmospheric_coupling *coupling,
                                       double toa_reflectance);

/*
 * The two pan-sharpeners: Brovey's ratio substitution and Gram-Schmidt component substitution.
 * `ms_bands` is `band_count` bands of `pixels` samples laid end to end, and the answer has the
 * same shape. Two-call, both.
 */
sl_status sl_brovey_pan_sharpen(const double *ms_bands, size_t band_count, size_t pixels,
                                const double *pan, double *out, size_t cap, size_t *out_len);
sl_status sl_gram_schmidt_pan_sharpen(const double *ms_bands, size_t band_count, size_t pixels,
                                      const double *pan, double *out, size_t cap, size_t *out_len);

/*
 * The detection head: greedy non-maximum suppression, the overlap it suppresses on, the decode of
 * a predicted delta against its anchor, and the anchor grid itself.
 *
 * `sl_nms` is two-call and answers the kept indices in descending score order; `sl_generate_anchors`
 * is two-call and answers `feat_rows * feat_cols * scale_count * ratio_count` boxes. `sl_box_iou`
 * is total and answers NaN for a null.
 */
sl_status sl_nms(const sl_bbox *boxes, const double *scores, size_t n, double iou_threshold,
                 size_t *out, size_t cap, size_t *out_len);
double sl_box_iou(const sl_bbox *a, const sl_bbox *b);
sl_status sl_decode_box(const sl_bbox *anchor, const double *delta, sl_bbox *out);
sl_status sl_generate_anchors(size_t feat_rows, size_t feat_cols, double stride,
                              const double *scales, size_t scale_count, const double *aspect_ratios,
                              size_t ratio_count, sl_bbox *out, size_t cap, size_t *out_len);

/* Everything the GPT3 climatology knows about one point and one day. */
typedef struct sl_gpt3_weather {
    double ah;                   /* hydrostatic VMF3 mapping coefficient */
    double aw;                   /* wet VMF3 mapping coefficient */
    double pressure;             /* Pa */
    double temperature;          /* K */
    double water_vapor_pressure; /* Pa */
    double tm;                   /* weighted mean temperature, K */
    double lambda;               /* water-vapour decrease factor */
    double gnh;                  /* hydrostatic north gradient */
    double geh;                  /* hydrostatic east gradient */
    double gnw;                  /* wet north gradient */
    double gew;                  /* wet east gradient */
} sl_gpt3_weather;

/* The Vienna mapping coefficients and the zenith delays shipped with them. */
typedef struct sl_vmf_coefficients {
    double ah;  /* hydrostatic mapping coefficient */
    double aw;  /* wet mapping coefficient */
    double zhd; /* zenith hydrostatic delay, m */
    double zwd; /* zenith wet delay, m */
} sl_vmf_coefficients;

/*
 * The GPT3 empirical troposphere at one point and one day of the year -- blind in the sense that
 * matters: it reads no file and needs no forecast, only the 5x5 degree climatology compiled into
 * the library. Feed `ah`/`aw` to `sl_vmf3_mapping`.
 *
 * `day_of_year` is Orekit's continuous seasonal argument: physical days since 31 December of the
 * previous year at 00:00 UTC.
 */
sl_status sl_gpt3(double latitude_rad, double longitude_rad, double day_of_year, double height_m,
                  sl_gpt3_weather *out);

/*
 * Bilinearly interpolate an operational VMF grid at one point. The grid arrives as its own text
 * with a length, the way RINEX does, because it is a file rather than a token and this library does
 * not read files.
 */
sl_status sl_vmf_grid_coefficients(const char *grid_text, size_t text_len, double latitude_rad,
                                   double longitude_rad, sl_vmf_coefficients *out);

/* ---- Cloud masking ---- */

/*
 * A loaded cloud-segmentation model and the probability cut it thresholds at. Opaque, like
 * `sl_sgp4`, and immutable once opened, so one handle may be masked with from several threads.
 *
 * Everything in this section needs the library to have been built with the `onnx` Cargo feature.
 * Without it the symbols are still here and every one of them returns `SL_ERR_UNSUPPORTED`; check
 * once at start-up by opening a model rather than per call.
 */
typedef struct sl_cloud_masker sl_cloud_masker;

/*
 * Load a segmentation model from the bytes of a `.onnx` file. `threshold` is the cloud-probability
 * cut at or above which a pixel is called cloudy; it is not held to [0, 1], because a cut above
 * every probability is how you ask for "nothing is cloudy" and below every one for "everything
 * is". Returns `SL_ERR_PARSE` if the bytes are not a model the runtime can load and optimize.
 *
 * `sl_cloud_masker_close` frees the handle. Loading optimizes the graph, which is the expensive
 * part: open once and mask many tiles.
 */
sl_status sl_cloud_masker_open(const uint8_t *onnx_bytes, size_t onnx_len, float threshold,
                               sl_cloud_masker **out);

/* Release a handle from `sl_cloud_masker_open`. Null is fine and does nothing. */
void sl_cloud_masker_close(sl_cloud_masker *masker);

/*
 * Mask one `channels x height x width` tile, row-major, implicit batch of 1. The result is one byte
 * per pixel of the *model's own output grid* -- `1` cloudy, `0` clear -- which is not in general
 * `height * width`: a segmentation net that pools twice returns a quarter as many pixels each way.
 * Bytes rather than `bool` because this is an array crossing an ABI, and `uint8_t` has one width
 * everywhere while `_Bool` is only promised to have one on this platform.
 *
 * Two-call protocol, with one caveat this is the only function here to carry: sizing runs the
 * model. Prefer allocating `height * width` and calling once -- that fits every model that does not
 * upsample -- and re-call on `SL_ERR_BUFFER_TOO_SMALL`, which sets `*out_len` to what is needed.
 *
 * Returns `SL_ERR_COMPUTE` if the model rejects the shape it was handed.
 */
sl_status sl_cloud_masker_mask(const sl_cloud_masker *masker, const float *tile, size_t channels,
                               size_t height, size_t width, uint8_t *out, size_t cap,
                               size_t *out_len);

/*
 * Open, mask, close. The convenient form, and the wrong one inside a loop -- it re-optimizes the
 * graph for every tile.
 */
sl_status sl_cloud_mask(const uint8_t *onnx_bytes, size_t onnx_len, float threshold,
                        const float *tile, size_t channels, size_t height, size_t width,
                        uint8_t *out, size_t cap, size_t *out_len);

/* ---- Radiometric and multistatic observables ---- */

/*
 * One end of a link: where it is, how it moves, and when the signal was there. Position and
 * velocity are in the same inertial frame as the satellite state they are used with; the epoch is
 * seconds in whatever timescale the caller keeps the link in, and only differences of it are read.
 */
typedef struct {
    sl_vec3 position;
    sl_vec3 velocity;
    double epoch;
} sl_terminal;

/* Moyer's three DSN bands, which fix the turn-around ratios. */
typedef enum {
    SL_BAND_S = 0,
    SL_BAND_X = 1,
    SL_BAND_KA = 2,
    SL_BAND_FORCE_INT32 = 0x7FFFFFFF /* see the note on sl_precision */
} sl_band;

/* One linear ramp of a station's exciter. A table is an array of these, in time order. */
typedef struct {
    double start;           /* start of validity, seconds */
    double end;             /* end of it, seconds */
    double start_frequency; /* transmitted frequency at `start`, Hz */
    double rate;            /* ramp rate over the segment, Hz/s */
} sl_ramp_segment;

/*
 * What an n-way averaged Doppler count is taken against.
 *
 * `reference_band` is carried separately from `uplink` because the receiver's reference need not be
 * stated against the band actually transmitted on, and neither has a default: a wrong one is a
 * silent change of answer rather than an error. `subtracted` selects the DSN convention, in which
 * the ramp term enters negative; it is Tudat's too.
 */
typedef struct {
    sl_band uplink;
    sl_band downlink;
    double reference_frequency; /* Hz */
    sl_band reference_band;
    double count_interval; /* seconds */
    bool subtracted;
} sl_doppler_link;

/* One term `coefficient * (r/R_sun)^-exponent` of the coronal electron-density series. */
typedef struct {
    double coefficient; /* electrons per cubic metre */
    double exponent;
} sl_corona_term;

/*
 * Bistatic range `rho_tx + rho_rx` and its time derivative -- one transmitter, one reflecting
 * satellite, one receiver. `bounce` is the epoch of the reflection, which is the epoch `satellite`
 * belongs to. Returns `SL_ERR_INVALID_ARGUMENT` if the three epochs do not increase along the
 * signal path.
 */
sl_status sl_bistatic_range(const sl_state *satellite, const sl_terminal *transmitter,
                            double bounce, const sl_terminal *receiver, double *out);

sl_status sl_bistatic_range_rate(const sl_state *satellite, const sl_terminal *transmitter,
                                 double bounce, const sl_terminal *receiver, double *out);

/*
 * Time and frequency difference of arrival across a two-terminal baseline. `emission` is the epoch
 * the satellite state belongs to, and each terminal's epoch must be strictly after it. TDOA is
 * `(rho_second - rho_first)/c` in seconds; FDOA is `(f/c)(rhodot_first - rhodot_second)` in hertz.
 */
sl_status sl_tdoa(const sl_state *satellite, double emission, const sl_terminal *first,
                  const sl_terminal *second, double *out);

sl_status sl_fdoa(const sl_state *satellite, double emission, const sl_terminal *first,
                  const sl_terminal *second, double frequency, double *out);

/*
 * Delta-DOR against a quasar, seconds -- the arrival-time difference across the baseline,
 * differenced again against a radio source at infinity so the station clocks and both atmospheres
 * cancel. The catalogue angles are radians, in the frame the states are expressed in.
 */
sl_status sl_delta_dor(const sl_state *satellite, double emission, const sl_terminal *first,
                       const sl_terminal *second, double right_ascension, double declination,
                       double *out);

/*
 * The inter-satellite link's one-way range and range rate, metres and m/s; the rate is positive
 * when the two are separating. Total: a null pointer answers NaN rather than a status, because
 * there is no status to answer with.
 */
double sl_inter_satellite_range(const sl_state *local, const sl_state *remote);

double sl_inter_satellite_range_rate(const sl_state *local, const sl_state *remote);

/*
 * The Shapiro (gravitational) delay on one leg, metres. The two radii are the endpoints' distances
 * from the central body and `path` the straight-line distance between them; `gm` is the body's
 * gravitational parameter, m^3/s^2, and has no default because a wrong one rescales the answer
 * silently. One leg at Earth: 0.669 mm to a 500 km satellite at the zenith, 12.66 mm to GPS.
 */
double sl_shapiro_delay(double r_emitter, double r_receiver, double path, double gm);

/*
 * DSN sequential range in range units, wrapped at `2^(C+6)`. `transmit` and `receive` are the
 * epochs the ranging code left the transmitting station and reached the receiving one, in the ramp
 * table's own timescale; `uplink` sets the scale, since a range unit is defined against the uplink
 * carrier, and `lowest_ranging_component` is Moyer's `C`.
 *
 * `SL_ERR_INVALID_ARGUMENT` for a ramp table that is empty, out of order or does not cover the
 * interval; `SL_ERR_DOMAIN` for an interval that does not run forwards or a `C` whose modulus is
 * not representable.
 */
sl_status sl_dsn_sequential_range(const sl_ramp_segment *segments, size_t segment_count,
                                  sl_band uplink, double transmit, double receive,
                                  int32_t lowest_ranging_component, double *out);

/*
 * DSN n-way averaged Doppler, hertz -- `M(reference_band, downlink) * reference_frequency`, less
 * `(M(uplink, downlink)/count_interval)` times the integral of the transmitted frequency, taken
 * between the two transmission epochs.
 */
sl_status sl_dsn_averaged_doppler(const sl_ramp_segment *segments, size_t segment_count,
                                  const sl_doppler_link *link, double transmit_start,
                                  double transmit_end, double *out);

/*
 * DSN measured frequency, hertz -- `M(uplink, downlink) * f_T(transmit_epoch) * transmit_rate`, the
 * instantaneous counterpart of the averaged Doppler and not independent of it: over a flat exciter
 * referred to the same band and frequency the two sum to `M * f_reference`. `transmit_rate` is
 * `dt_T/dt_R`, which comes out of the light-time solution and so is the caller's -- `1` for a
 * stationary link, `1 - rhodot/c` to first order for a receding one.
 */
sl_status sl_dsn_measured_frequency(const sl_ramp_segment *segments, size_t segment_count,
                                    sl_band uplink, sl_band downlink, double transmit_epoch,
                                    double transmit_rate, double *out);

/*
 * Solar-corona range correction along one leg, metres: the electron-column term along the straight
 * line between two *heliocentric* positions, scaled by `40.3/f^2`. `series` may be null, which
 * takes Aksim & Pavlov (2022) -- Tudat's default too. `group_delay` selects the sign: a ranging
 * observable arrives late, a carrier-phase or Doppler one early, by the same magnitude.
 *
 * Returns `SL_ERR_DOMAIN` if the line of sight passes within one solar radius of the Sun's centre.
 * There is no honest correction for a signal through the photosphere, and returning zero or NaN
 * would each be a different way of not saying so.
 */
sl_status sl_solar_corona_delay(const sl_vec3 *transmitter_from_sun,
                                const sl_vec3 *receiver_from_sun, double frequency,
                                bool group_delay, const sl_corona_term *series, size_t series_len,
                                double *out);

/* ---- Orbital lifetime, re-entry and element-set generation ---- */

/* What became of one object over one horizon. */
typedef enum {
    SL_LIFETIME_REENTRY = 0,    /* the mean perigee altitude reached the threshold */
    SL_LIFETIME_SURVIVES = 1,   /* still up at the horizon, by an amount this run does not know */
    SL_LIFETIME_UNRESOLVED = 2, /* the propagation stopped being steppable above the threshold */
    SL_LIFETIME_OUTCOME_FORCE_INT32 = 0x7FFFFFFF /* see the note on sl_precision */
} sl_lifetime_outcome;

/*
 * The gravity field, the spacecraft, the atmosphere and the scan. `sl_decay_params_init` fills it
 * with `cd = 2.2`, `area_over_mass = 0.01`, `chunk = 86400`, `tolerance = 60`, `density_scale = 1`,
 * Earth's `mu` and radius, `reentry_altitude = 120000` and Earth's rotation rate -- and NaN for the
 * four an atmosphere and a horizon cannot be guessed for, so a caller who forgets one is refused
 * rather than answered.
 *
 * The atmosphere is exponential about the spherical radius, co-rotating rigidly about `z`. That is
 * named rather than implied: driving the mean-element quadrature with an empirical model over a
 * decade needs the inertial-to-geodetic rotation carried through every node, which is a separate
 * piece of work, and quoting a decade of decay from a model whose geographic arguments were quietly
 * approximated would be worse than not offering it.
 */
typedef struct {
    double cd;
    double area_over_mass;    /* m^2/kg */
    double reference_density; /* kg/m^3 at `reference_radius` */
    double reference_radius;  /* m */
    double scale_height;      /* m */
    double horizon;           /* how far ahead to look, seconds */
    double chunk;             /* how far the scan advances between perigee tests, seconds */
    double tolerance;         /* how finely the crossing chunk is bisected, seconds */
    double density_scale;     /* a multiplier on the density; the ensemble's own axis */
    double mu;                /* m^3/s^2 */
    double body_radius;       /* m */
    double reentry_altitude;  /* the mean perigee altitude counted as re-entry, m */
    double rotation_rate;     /* rad/s */
} sl_decay_params;

/* The scan's answer. `seconds` is the crossing, the horizon or where the scan stopped, by outcome. */
typedef struct {
    sl_lifetime_outcome outcome;
    double seconds;
    double perigee_altitude_m;
} sl_lifetime;

/*
 * How an ensemble ended. A non-zero `unresolved` is a finding rather than a statistic: those
 * members stopped being propagable before reaching the threshold, and every quantile reported is
 * conditioned on that.
 */
typedef struct {
    size_t members;
    size_t survivors;
    size_t unresolved;
} sl_lifetime_ensemble;

/* Fill `p` with the defaults documented above. Null is fine and does nothing. */
void sl_decay_params_init(sl_decay_params *p);

/*
 * Predict one object's re-entry epoch under drag and a zonal field.
 *
 * `initial` is the *mean* equinoctial state `[a, k, h, q, p, lambda]`, metres and radians, and the
 * answer is seconds from that epoch. `zonals` are the unnormalised `Cn0` indexed by degree, so
 * `{0, 0, C20}` is J2 alone and `C20 = -J2`; at least three are needed.
 *
 * A propagation that breaks down is *not* an error: it is `SL_LIFETIME_UNRESOLVED`, because it is
 * an outcome of the physics as often as of the arguments.
 */
sl_status sl_orbital_lifetime(const sl_decay_params *params, const double *initial,
                              const double *zonals, size_t zonal_count, sl_lifetime *out);

/*
 * The same decay under a set of density scale factors, reported as quantiles. One member per entry
 * of `density_scales`, which is where the driver uncertainty enters: over a decade the answer is
 * dominated by how active the Sun turns out to be, and to first order that enters as a factor on
 * the density the object flies through.
 *
 * `out_seconds` is written with one epoch per entry of `quantiles`. A quantile the sample does not
 * determine -- past the censoring fraction, or from an empty ensemble -- is NaN, because the
 * alternative is the largest decay the run happened to see, which understates the answer by however
 * long the censored members go on to last.
 */
sl_status sl_orbital_lifetime_ensemble(const sl_decay_params *params, const double *initial,
                                       const double *zonals, size_t zonal_count,
                                       const double *density_scales, size_t scale_count,
                                       const double *quantiles, size_t quantile_count,
                                       double *out_seconds, sl_lifetime_ensemble *out);

/*
 * `d ln(lifetime) / d ln(S/m)` by central difference on the area-to-mass ratio -- the elasticity a
 * disposal study needs, in the units the requirement is written in. `fraction` is the relative step
 * taken either side. Refuses if either side fails to re-enter within the horizon, because a
 * sensitivity between two epochs one of which does not exist is not a number.
 */
sl_status sl_ballistic_log_sensitivity(const sl_decay_params *params, const double *initial,
                                       const double *zonals, size_t zonal_count, double fraction,
                                       double *out);

/* A civil epoch, which is what a TLE's columns carry. */
typedef struct {
    int32_t year;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint8_t minute;
    double second;
} sl_utc_datetime;

/* Bytes each TLE line needs: 69 columns and the terminator. */
#define SL_TLE_LINE_BYTES 70

/* The two lines, each NUL-terminated. */
typedef struct {
    char line1[SL_TLE_LINE_BYTES];
    char line2[SL_TLE_LINE_BYTES];
} sl_tle;

/*
 * Print an element set as the two lines of a TLE.
 *
 * `parameters` are the six fitted fields in the element set's own *printed* units and order -- mean
 * motion in revolutions per day, eccentricity, then inclination, right ascension, argument of
 * perigee and mean anomaly in degrees. `drag_term` is `B*`, the only drag quantity SGP4 reads; the
 * two mean-motion derivatives are printed as zero, because nothing here fits them and a plausible
 * value would be indistinguishable from a fitted one.
 *
 * `international_designator` may be null. Both checksums are correct and both lines are 69 columns.
 * `SL_ERR_INVALID_ARGUMENT` for a catalog number past 99999, an epoch outside 1957-2056, a
 * malformed designator, and any field whose columns cannot express its value.
 */
sl_status sl_tle_lines(const sl_utc_datetime *epoch, uint32_t norad_id, const double *parameters,
                       double drag_term, const char *international_designator,
                       uint64_t element_set_number, uint64_t revolution_number, sl_tle *out);

/* ---- GNSS observables ---- */

/* The four wavelengths of a frequency pair, metres. */
typedef struct {
    double lambda1;     /* c/f1 */
    double lambda2;     /* c/f2 */
    double wide_lane;   /* c/(f1 - f2) -- 0.8619 m on GPS L1/L2 */
    double narrow_lane; /* c/(f1 + f2) -- 0.1070 m on the same pair */
} sl_lanes;

/* What a cycle-slip detector made of one sample. */
typedef enum {
    SL_SLIP_ARC_START = 0, /* nothing to compare this sample against yet */
    SL_SLIP_CONTINUOUS = 1,
    SL_SLIP_SLIPPED = 2,
    SL_SLIP_FORCE_INT32 = 0x7FFFFFFF /* see the note on sl_precision */
} sl_slip;

/* One verdict. `statistic` is NaN at an arc start, where there is none. */
typedef struct {
    sl_slip verdict;
    double statistic;
} sl_slip_verdict;

/*
 * The carrier frequency of `band` on constellation `system`, hertz, or NaN where that constellation
 * does not broadcast it. Both arguments are the RINEX letters -- 'G', 'E', 'C', 'J', 'I', 'R' for
 * the system and '1', '2', '5', '6', '7', '8' for the band.
 *
 * GLONASS L1/L2 answer NaN and are not an omission: they are FDMA, so the frequency depends on the
 * satellite's channel, which `sl_glonass_l1_frequency` takes.
 */
double sl_carrier_frequency(char system, char band);

/* The four lane wavelengths of a frequency pair. */
sl_status sl_lane_wavelengths(double f1, double f2, sl_lanes *out);

/*
 * The dual-frequency combinations, all metres. Each applies to a code pair or a phase pair alike;
 * on phases the surviving ambiguity of the ionosphere-free combination is not an integer.
 */
sl_status sl_ionosphere_free(double f1, double f2, double x1, double x2, double *out);

sl_status sl_wide_lane(double f1, double f2, double x1, double x2, double *out);

sl_status sl_narrow_lane(double f1, double f2, double x1, double x2, double *out);

/*
 * Melbourne-Wubbena: the wide-lane phase less the narrow-lane code, from pseudoranges `p1`/`p2` and
 * carrier phases `l1`/`l2`, all metres. In metres, or in wide-lane cycles with `cycles` -- the unit
 * the arc is read in, since over a clean arc the value hovers about an integer and a step in it is
 * a wide-lane slip.
 */
sl_status sl_melbourne_wubbena(double f1, double f2, double p1, double p2, double l1, double l2,
                               bool cycles, double *out);

/*
 * Geometry-free `x1 - x2` and GRAPHIC `(p + l)/2`, metres. Total. The first takes no frequencies
 * because everything non-dispersive cancels; the second is the single-frequency ionosphere-free
 * observable.
 */
double sl_geometry_free(double x1, double x2);

double sl_graphic(double p, double l);

/*
 * Run a cycle-slip detector over a series, writing one verdict per sample. `out` is writable for
 * `count` verdicts; there is no sizing call, because the answer is one per input.
 *
 * The first-difference detector's `threshold` is on the raw first difference in the units of
 * `values`, so it has to be chosen for the sampling interval: on the geometry-free combination at
 * 30 s, a few centimetres separates real ionospheric change from the 0.05-0.11 m a one-cycle L1 or
 * L2 slip moves it by. A gap longer than `max_gap` seconds restarts the arc rather than testing
 * across it.
 *
 * The polynomial detector fits the last `window` samples at degree `order` and tests the next
 * sample against the prediction, which is what sees a slip smaller than the series' own drift over
 * the window. Its first `window` samples are `SL_SLIP_ARC_START`, and `window` must be at least
 * `order + 2`.
 */
sl_status sl_detect_cycle_slips_geometry_free(const double *times, const double *values,
                                              size_t count, double threshold, double max_gap,
                                              sl_slip_verdict *out);

sl_status sl_detect_cycle_slips_polynomial(const double *times, const double *values, size_t count,
                                           size_t order, size_t window, double threshold,
                                           double max_gap, sl_slip_verdict *out);

/*
 * Decorrelate an ambiguity problem. `covariance` is the row-major `n x n` covariance and
 * `float_ambiguities` the `n` float values. `out_transformation` receives the integer-valued `Z` of
 * LAMBDA, row-major and `n x n`; `out_conditional` receives the `n` conditional variances of
 * `Q_z = L^T D L`, whose product is the volume of the search ellipsoid, so their spread is the
 * measure of how much the decorrelation bought.
 */
sl_status sl_ambiguity_decorrelation(const double *float_ambiguities, size_t n,
                                     const double *covariance, double *out_transformation,
                                     double *out_conditional);

/*
 * The bootstrapped integer solution and its objective -- sequential conditional rounding in the
 * decorrelated parametrisation, one pass and no search. It is not in general the integer
 * least-squares solution; it is what bounds the search, and its success rate is the classical lower
 * bound on the ILS success rate.
 */
sl_status sl_ambiguity_bootstrap(const double *float_ambiguities, size_t n,
                                 const double *covariance, int64_t *out_ambiguities,
                                 double *out_squared_norm);

/*
 * The best integer ambiguity vectors by the LAMBDA integer least-squares search, best first, in the
 * caller's own parametrisation. The ratio of the second squared norm to the first is the classical
 * acceptance test.
 *
 * `out_ambiguities` is writable for `count * n` integers and `out_squared_norms` for `count`
 * doubles; `*out_len` is how many were actually written, which is at most `count`. Returns
 * `SL_ERR_COMPUTE` if the enumeration reaches its node budget without exhausting the ellipsoid -- a
 * truncated list would be a wrong answer that looks like a right one.
 */
sl_status sl_resolve_ambiguities(const double *float_ambiguities, size_t n,
                                 const double *covariance, size_t count, int64_t *out_ambiguities,
                                 double *out_squared_norms, size_t *out_len);

/* ---- Conjunction assessment ---- */

/*
 * The relative geometry at TCA and the two hard-body radii. The state is relative -- `r2 - r1` and
 * `v2 - v1`, metres and m/s in one common inertial frame -- and each covariance is that object's
 * 3x3 position covariance, row-major, m^2, in the same frame. The radii are equivalent-sphere
 * radii and their sum is the collision cross section.
 */
typedef struct {
    sl_vec3 relative_position;
    sl_vec3 relative_velocity;
    double primary_covariance[9];
    double secondary_covariance[9];
    double primary_radius;
    double secondary_radius;
} sl_conjunction;

/*
 * The 2-D collision probability by all four estimators. All four are returned rather than one
 * chosen here: Foster, Patera and Chan integrate the same encounter-plane Gaussian by different
 * routes and agree to a few percent where the 2-D model holds, so their spread is itself the
 * diagnostic. The fourth is the *maximum* over a scaling of the covariance -- a bound rather than
 * an estimate, and the one to act on when the covariance is not trusted.
 * `sl_conjunction_usage_violations` is what says whether any of them applies at all.
 */
typedef struct {
    double foster;
    double patera;
    double chan;
    double maximum;
} sl_probability_estimates;

/* A relative state and the *joint* covariance `C1 + C2` at one epoch. */
typedef struct {
    sl_vec3 position;
    sl_vec3 velocity;
    double covariance[9];
} sl_relative_sample;

/*
 * Whether the 2-D collision probability is inside its own model's domain. A flag says the estimate
 * is not known to be accurate, not that it is wrong; the thresholds are deliberately conservative.
 * `correction_factor` is the rough multiplicative correction that falls out of the fourth check:
 * the 2-D probability times it is the corrected one.
 */
typedef struct {
    bool any;
    bool non_positive_definite;
    bool offset;
    bool extended;
    bool inaccuracy;
    double offset_indicator;
    double extended_indicator;
    double inaccuracy_indicator;
    double correction_factor;
} sl_usage_violations;

/* One local minimum of the separation. */
typedef struct {
    double tca;            /* seconds, in the scan's own timescale */
    double distance;       /* metres */
    double relative_speed; /* m/s */
} sl_close_approach;

/*
 * The callbacks the two scans call. Fill `out` with the relative state -- object 2 minus object 1,
 * in one common inertial frame -- at `t`, and return `SL_OK`. Anything else ends the scan and is
 * reported to the caller as `SL_ERR_COMPUTE`, so a sampler that failed is never mistaken for a scan
 * that found nothing. Neither may unwind.
 */
typedef sl_status (*sl_relative_state_fn)(void *user_data, double t, sl_state *out);

typedef sl_status (*sl_relative_sample_fn)(void *user_data, double t, sl_relative_sample *out);

/* The 2-D collision probability at one conjunction, by all four estimators. */
sl_status sl_collision_probability(const sl_conjunction *geometry, sl_probability_estimates *out);

/*
 * Whether the 2-D collision probability is inside its own model's domain.
 *
 * `minimum_period` is the shorter of the two objects' orbital periods in seconds -- the scale that
 * turns an offset and a duration into the dimensionless ratios the published thresholds are stated
 * on. There is no default for it, because a guessed period rescales both thresholds silently.
 * `sample(t)` supplies the relative state and the joint covariance `t` seconds from TCA, under
 * whatever dynamics the caller propagates with.
 */
sl_status sl_conjunction_usage_violations(const sl_conjunction *geometry, double minimum_period,
                                          sl_relative_sample_fn sample, void *user_data,
                                          sl_usage_violations *out);

/*
 * Every closest approach of a relative trajectory in `[start, end]`, in time order. The scan
 * samples every `step` seconds and refines each bracket by bisection to `tolerance`, so the cost is
 * logarithmic in `step/tolerance` per approach and does not depend on how the separation behaves
 * inside the bracket. The step is the resolution: two minima closer together than one step are
 * reported as one.
 *
 * Two-call protocol, with the caveat `sl_cloud_masker_mask` also carries -- sizing runs the scan,
 * so `relative` is invoked by both calls.
 */
sl_status sl_closest_approaches(sl_relative_state_fn relative, void *user_data, double start,
                                double end, double step, double tolerance, sl_close_approach *out,
                                size_t cap, size_t *out_len);

/*
 * Whether two orbits are far enough apart in radius that no conjunction is possible -- the first
 * and cheapest stage of an all-on-all screen (Hoots, Crawford & Roehrich 1984, filter 1). All five
 * arguments are metres: the two orbits' geocentric perigee and apogee distances, and the screening
 * distance below which a miss is worth propagating.
 *
 * `true` is a geometric certainty and not a heuristic -- the two spherical shells the orbits are
 * confined to cannot come within `threshold` of each other at any epoch, whatever the phasing or
 * the plane. The converse is not claimed: most pairs that survive the filter still do not conjunct.
 * A non-finite input answers `false`, so a corrupt element set is examined rather than silently
 * discarded.
 */
bool sl_radially_separated(double perigee_a, double apogee_a, double perigee_b, double apogee_b,
                           double threshold);

/* ---- Closed-loop control ---- */

/*
 * The three latencies of a feedback loop, seconds, each finite and non-negative.
 *
 * Three numbers rather than one `tau` because they have different physical origins and different
 * fixes: a star tracker's integration time is bought down with aperture, compute latency with a
 * faster processor, actuator lag with a different wheel. Collapsing them gives a margin nobody can
 * act on.
 */
typedef struct sl_latency {
    /* The measurement describes the state this long ago. */
    double sensor_s;
    /* Deciding takes this long. */
    double compute_s;
    /* The actuator takes this long to respond. */
    double actuator_s;
} sl_latency;

/* Which attitude law. */
typedef enum sl_attitude_law {
    /* Wie's quaternion PD, `u = -K q_v - P dw`. Locally stable, and the cheaper of the two. */
    SL_LAW_QUATERNION_PD = 0,
    /* Schaub & Junkins' MRP Lyapunov law, which adds the gyroscopic feed-forward and is stable
     * for every initial attitude rather than only for small ones. */
    SL_LAW_MRP_LYAPUNOV = 1,
    SL_LAW_FORCE_INT32 = 0x7FFFFFFF /* see the note on sl_precision */
} sl_attitude_law;

/* A rigid body's orientation and rate: one point of the plant's state. */
typedef struct sl_attitude_state {
    /* Inertial->body change of basis, row-major. */
    double rotation[9];
    /* Body rate in body axes, rad/s. */
    sl_vec3 rate;
} sl_attitude_state;

/* One step of a closed-loop run. */
typedef struct sl_loop_sample {
    /* Seconds since the run started. */
    double time_s;
    /* The principal angle of the error the controller acted on -- of the *delayed* state, when
     * there is latency. */
    double error_angle_rad;
    /* The moment commanded at this step, N*m, body axes. */
    sl_vec3 torque;
} sl_loop_sample;

/* The delay a linearised loop tolerates, and the loop it was measured on. */
typedef struct sl_delay_margin {
    /* The largest feedback delay the loop survives, seconds. Compare against
     * `sl_latency_total`. */
    double seconds;
    /* The frequency the pole pair crosses at, rad/s -- what the loop rings at when the delay is
     * exactly the margin. */
    double crossing_frequency_rad_s;
    /* The loop's undamped natural frequency, rad/s. */
    double natural_frequency_rad_s;
    /* The loop's damping ratio. */
    double damping_ratio;
    /* `tau_max * omega_n`, which depends only on the damping ratio. The form to quote: it says
     * the margin is a fixed fraction of the loop's own period, so buying bandwidth spends margin
     * proportionally. */
    double normalised;
} sl_delay_margin;

/* The station-keeping law: Bryson weights, a proportional gain and an acceleration limit. */
typedef struct sl_element_feedback {
    /* One weight per equinoctial element, in the DSST's `[a, k, h, q, p, lambda]` order. Each
     * finite and non-negative with at least one strictly positive; a zero declares an element
     * uncontrolled, which is the usual case. */
    double weights[6];
    /* The proportional gain, finite and strictly positive. */
    double proportional;
    /* The largest acceleration the thruster can produce, m/s^2. */
    double max_acceleration;
} sl_element_feedback;

/* What an agent does with a message that arrived late. */
typedef enum sl_consensus_protocol {
    /* The relative measurement itself is delayed, so both terms are equally old. This is the
     * protocol Olfati-Saber & Murray bound at `pi / (2 lambda_max(L))`. */
    SL_CONSENSUS_DELAYED_RELATIVE = 0,
    /* The agent's own state is current and only the neighbour's is late. No delay destroys its
     * stability; the convergence rate still falls with delay. */
    SL_CONSENSUS_CURRENT_SELF = 1,
    SL_CONSENSUS_FORCE_INT32 = 0x7FFFFFFF /* see the note on sl_precision */
} sl_consensus_protocol;

/* The total latency `tau_s + tau_c + tau_a`, seconds. */
sl_status sl_latency_total(const sl_latency *latency, double *out);

/*
 * How many whole steps of `step` the total latency is.
 *
 * `SL_ERR_DOMAIN` when it is not a whole number of them to within four ulps. Refusing is the
 * point: a loop whose delay is 2.5 steps is a different problem from one whose delay is 2 or 3,
 * and rounding here would answer a question nobody asked while labelling it with the caller's tau.
 */
sl_status sl_latency_steps(const sl_latency *latency, double step, size_t *out);

/*
 * The attitude error between two frames: the modified Rodrigues vector of the error rotation and
 * its principal angle. Both matrices are row-major inertial-from-body [NB] (body vectors to
 * inertial; the transpose of an attitude law's inertial->body matrix), and the error is
 * [NR]^T [NB]; both rates are body rates in body axes, rad/s.
 */
sl_status sl_attitude_error_mrp(const double rotation[9], const double reference[9],
                                const sl_vec3 *rate, const sl_vec3 *rate_reference,
                                sl_vec3 *out_mrp, double *out_angle_rad);

/*
 * The same error as the error quaternion `(q0, q1, q2, q3)`, with `q0 >= 0`.
 *
 * A separate entry point rather than two more out-parameters on `sl_attitude_error_mrp`, because the
 * two are the same rotation in different charts and a caller wants one of them: the MRP is what the
 * Lyapunov law acts on, the quaternion is what most flight software speaks. `(q1, q2, q3)` is what
 * a proportional law acts on. `out` receives four doubles.
 */
sl_status sl_attitude_error_quaternion(const double rotation[9], const double reference[9],
                                       const sl_vec3 *rate, const sl_vec3 *rate_reference,
                                       double out[4]);

/* The commanded body moment, N*m. `inertia` is the nine row-major entries of the tensor, kg*m^2,
 * and must be symmetric and positive definite. */
sl_status sl_attitude_torque(sl_attitude_law law, double proportional, double derivative,
                             const double rotation[9], const double reference[9],
                             const sl_vec3 *rate, const sl_vec3 *rate_reference,
                             const double inertia[9], sl_vec3 *out);

/*
 * `V = 1/2 dw^T I dw + 2K ln(1 + sigma^T sigma)`, the Lyapunov function the MRP law drives
 * downhill. There is no law argument: the quaternion PD law is stable in the small-angle
 * linearisation and this function is not its certificate.
 */
sl_status sl_attitude_lyapunov(double proportional, double derivative, const double rotation[9],
                               const double reference[9], const sl_vec3 *rate,
                               const sl_vec3 *rate_reference, const double inertia[9], double *out);

/*
 * `(k_theta, k_omega)` in N*m/rad and N*m*s/rad -- the law's linearisation about the reference.
 *
 * The MRP law linearises to a quarter of its nominal stiffness, because `sigma = q_v/(1+q0)` tends
 * to `dtheta/4`; the quaternion law to a half. Two laws at the same nominal gains are not the same
 * loop, and their delay margins differ accordingly.
 */
sl_status sl_attitude_small_angle_gains(sl_attitude_law law, double proportional, double derivative,
                                        double *out_angle_gain, double *out_rate_gain);

/* The derivative gain that puts a principal axis of moment `moment` at a damping ratio of one:
 * `P = sqrt(2 I K)`. */
sl_status sl_quaternion_pd_critically_damped(double proportional, double moment, double *out);

/* `(omega_n, zeta)` for the quaternion PD law on a principal axis of moment `moment`. */
sl_status sl_quaternion_pd_response(double proportional, double derivative, double moment,
                                    double *out_natural_frequency_rad_s, double *out_damping_ratio);

/*
 * Run a rigid-body attitude loop at a fixed rate under a fixed latency, and report every step.
 *
 * The controller reads the state as it was `sl_latency_steps` steps ago, and reads `seed` until
 * that much time has elapsed. `out` receives one sample per step, in order, under the same
 * two-call protocol as everywhere else: `sl_attitude_loop_run(..., NULL, 0, &len, NULL)` sizes it.
 * `out_final` may be null if only the samples are wanted.
 *
 * `SL_ERR_DOMAIN` unless the total latency is a whole number of `step`s, and unless the resulting
 * lag fits the 512-sample history this ABI gives the loop.
 */
sl_status sl_attitude_loop_run(sl_attitude_law law, double proportional, double derivative,
                               const sl_latency *latency, double step,
                               const sl_attitude_state *initial, const sl_attitude_state *seed,
                               const double reference[9], const double inertia[9], size_t steps,
                               sl_loop_sample *out, size_t cap, size_t *out_len,
                               sl_attitude_state *out_final);

/*
 * The delay margin of one linearised principal axis, from its moment and its two gains.
 *
 * A zero rate gain is admitted rather than refused, because the answer it produces -- zero margin,
 * an undamped loop tolerates no delay at all -- is the correct one and is worth being able to
 * compute.
 */
sl_status sl_delay_margin_of_gains(double moment, double angle_gain, double rate_gain,
                                   sl_delay_margin *out);

/*
 * The delay margin of a named law on a principal axis of moment `moment`.
 *
 * This is the composition to reach for rather than `sl_delay_margin_of_gains` on hand-computed
 * gains: it takes the law's own linearisation, so the MRP law's quarter-stiffness is applied for
 * you.
 */
sl_status sl_delay_margin_of_law(sl_attitude_law law, double proportional, double derivative,
                                 double moment, sl_delay_margin *out);

/*
 * The binding axis of a three-axis body: the index of the principal moment with the smallest
 * margin, and that margin.
 *
 * `moments` are the three *principal* moments, kg*m^2. A non-diagonal inertia tensor has to be
 * diagonalised by the caller; this will not pretend a tensor's diagonal is its spectrum.
 */
sl_status sl_worst_principal_axis(sl_attitude_law law, double proportional, double derivative,
                                  const sl_vec3 *moments, size_t *out_axis, sl_delay_margin *out);

/*
 * The Clohessy-Wiltshire relative plant at `mean_motion` rad/s, row-major: `out_a` is 6x6 and
 * `out_b` is 6x3, in the LVLH state order `[x, y, z, xdot, ydot, zdot]`.
 */
sl_status sl_clohessy_wiltshire_plant(double mean_motion, double out_a[36], double out_b[18]);

/*
 * The infinite-horizon LQR gain for `xdot = Ax + Bu` under `J = integral (x^T Q x + u^T R u) dt`.
 *
 * All four inputs are row-major: `a` and `q` are `n*n`, `r` is `m*m`, and `b` is `n*m` indexed
 * `b[state][control]`. `out_gain` receives `K` as `m*n` so that `u = -Kx`, `out_riccati` receives
 * `P` as `n*n`, and `out_residual` the relative Riccati residual the solution was accepted at --
 * the measured accuracy of the gain rather than a promise about it. Either out-matrix may be null.
 *
 * The Rust solver is generic in the two dimensions at compile time, so this ABI reaches the shapes
 * it was instantiated for: `(n, m)` in `(2,1)`, `(3,1)`, `(4,2)`, `(6,3)`. The last is the
 * Clohessy-Wiltshire plant and the first the double integrator. Anything else is
 * `SL_ERR_INVALID_ARGUMENT` rather than a silently reshaped answer.
 */
sl_status sl_lqr(const double *a, const double *b, const double *q, const double *r, size_t n,
                 size_t m, double *out_gain, double *out_riccati, double *out_residual);

/*
 * The monic characteristic polynomial of a square matrix: the coefficients of `lambda^0` through
 * `lambda^{n-1}`, in that order. The `lambda^n` coefficient is 1 and is not returned, so `out`
 * receives `n` doubles for an `n x n` input.
 *
 * This is what makes a gain from `sl_lqr` checkable without an eigensolver: the closed-loop matrix
 * `A - BK` is stable exactly when this polynomial's roots are all in the left half-plane, and the
 * coefficients are real and order-free where the eigenvalues are neither -- which is what lets them
 * be compared against a CAS directly. Faddeev-LeVerrier, so the cost is matrix products and traces.
 *
 * The recurrence divides by `k` and so loses roughly `n` digits of the smallest coefficient on a
 * matrix whose eigenvalues are widely spread; on the 6x6 closed loops here that is measured at
 * `1e-12` relative against a 40-digit reference, and for a much larger or stiffer matrix it would
 * be the wrong tool.
 *
 * `n` must be between 2 and 12 -- the same numbers as the consensus wall and not the same wall,
 * since this one is the matrix order rather than an agent count.
 */
sl_status sl_characteristic_polynomial(const double *m, size_t n, double *out);

/* `V = 1/2 delta-oe^T W delta-oe` for the station-keeping law. */
sl_status sl_element_feedback_lyapunov(const sl_element_feedback *law, const double error[6],
                                       double *out);

/*
 * The station-keeping acceleration in the inertial orbit frame, m/s^2 -- the frame every DSST
 * force returns its acceleration in, so this sums with them directly.
 *
 * `elements` are the osculating equinoctial elements the variational matrix is evaluated at, in
 * the DSST's `[a, k, h, q, p, lambda]` order with `a` in metres; `error` is `oe - oe_target` in
 * the same order. The mean-longitude component is wrapped to `(-pi, pi]` before use: an unwrapped
 * error of `2pi - eps` would command nearly a full revolution of drift to reach an orbit that is
 * `eps` away.
 */
sl_status sl_element_feedback_acceleration(const sl_element_feedback *law, const double elements[6],
                                           double mu, double true_longitude_rad,
                                           const double error[6], sl_vec3 *out);

/*
 * `Vdot` along the closed loop, which the station-keeping law makes non-positive by construction.
 * Computed from the same `B(oe)` the command used, so a caller can measure the decrease rather
 * than take the derivation's word for it.
 */
sl_status sl_element_feedback_decrease_rate(const sl_element_feedback *law,
                                            const double elements[6], double mu,
                                            double true_longitude_rad, const double error[6],
                                            double *out);

/*
 * The Laplacian spectrum of an interaction graph, ascending. `adjacency` is the row-major weight
 * matrix of `agents^2` entries -- the weights are control gains, so they need not be one -- and
 * `out` receives `agents` eigenvalues.
 *
 * `SL_ERR_DOMAIN` unless the graph is symmetric, non-negative, hollow and connected: a
 * disconnected graph has a repeated zero eigenvalue and never reaches one consensus at all, which
 * is a different answer from a small one.
 *
 * `agents` must be between 2 and 12, the range the const-generic eigensolver is instantiated for.
 */
sl_status sl_laplacian_spectrum(const double *adjacency, size_t agents, double *out);

/* The algebraic connectivity `lambda_2` -- the convergence rate of undelayed consensus. */
sl_status sl_algebraic_connectivity(const double *adjacency, size_t agents, double *out);

/*
 * The same eigenvalues, and the unit eigenvectors `sl_laplacian_spectrum` leaves behind.
 *
 * `out_values` receives `agents` eigenvalues ascending and `out_modes` the `agents^2` entries of
 * the eigenvector matrix, row-major, so that mode `k` is `out_modes[k * agents .. (k+1) * agents]`.
 * Either may be null.
 *
 * The eigenvectors are what the disagreement decomposes into: each mode evolves as the scalar
 * delay equation `y' = -lambda_k y(t - tau)` under `SL_CONSENSUS_DELAYED_RELATIVE`, independently
 * of every other. Watching the mode that crosses first is not the same as watching whichever mode
 * happens to dominate the aggregate norm.
 */
sl_status sl_laplacian_modes(const double *adjacency, size_t agents, double *out_values,
                             double *out_modes);

/*
 * Whether every agent can reach every other over links of non-zero gain.
 *
 * Combinatorial, by breadth-first search, and therefore exact. `sl_algebraic_connectivity` answers
 * the same question as `lambda_2 > tolerance`, and there is no tolerance anybody can choose for an
 * arbitrary weighting -- which is why this is a separate entry point and not a documented
 * threshold on that one.
 *
 * Unlike the rest of this section it admits a disconnected graph, since that is the answer it
 * exists to give; it still refuses one that is not symmetric, non-negative and hollow.
 */
sl_status sl_graph_is_connected(const double *adjacency, size_t agents, bool *out);

/*
 * The largest link delay at which a protocol still reaches consensus, seconds, or `HUGE_VAL` for
 * `SL_CONSENSUS_CURRENT_SELF`, which has no such bound.
 */
sl_status sl_consensus_delay_bound(const double *adjacency, size_t agents,
                                   sl_consensus_protocol protocol, double *out);

/*
 * The disagreement `||x - mean(x) 1||`, which is what consensus drives to zero. The mean is
 * subtracted rather than a designated leader's value, because no agent is a leader and the mean is
 * the quantity the dynamics conserve.
 */
sl_status sl_consensus_disagreement(const double *state, size_t agents, double *out);

/*
 * Integrate delayed consensus over a fixed graph to `t1`, by the method of steps.
 *
 * `out_final` receives the `agents` states there and may be null; `out_disagreement`,
 * `out_accepted` and `out_rejected` receive the disagreement at `t1` and the step counts summed
 * over every arc of the mesh, and each may be null. A breakpoint is placed at every propagated
 * discontinuity `k*tau`: a delayed solution has a jump in `x'` at the origin, in `x''` at `tau`
 * and in `x'''` at `2 tau`, and a mesh that steps over one loses the integrator's order.
 *
 * `delay` must be finite and strictly positive. A zero delay is an ordinary differential equation
 * and belongs to a different driver, and the method of steps needs a positive shortest delay to
 * bound its mesh.
 */
sl_status sl_consensus_run(const double *adjacency, size_t agents, sl_consensus_protocol protocol,
                           double delay, const double *initial, double t1, double tolerance,
                           double *out_final, double *out_disagreement, uint32_t *out_accepted,
                           uint32_t *out_rejected);

/*
 * Every propagated discontinuity of a delayed system strictly inside `(t0, t1)`, sorted, merged
 * and *not* including either endpoint -- these are the interior mesh points, and `t1` is the
 * caller's already.
 *
 * A delayed system smooths one order per delay interval, so the jumps sit at `t0 + sum k_i tau_i`
 * for every multi-index with `1 <= sum k_i <= levels`. Two-call protocol, as everywhere else.
 */
sl_status sl_delay_breakpoints(const double *delays, size_t n, double t0, double t1, size_t levels,
                               double *out, size_t cap, size_t *out_len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPACELAB_H */
