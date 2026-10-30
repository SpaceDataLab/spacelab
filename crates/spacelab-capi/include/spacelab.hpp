/*
 * spacelab.hpp -- the C++ face of the spacelab core. Header-only; include it and link the static
 * library. C++20.
 *
 * There is nothing under here but `spacelab.h`. Rust cannot export C++ -- C++ has no stable ABI --
 * so every "C++ binding" to a Rust library is a C boundary with a header over it, and this is the
 * header. Read `spacelab.h` for what the functions mean, the units they take, and what each
 * failure code says; this file only changes how you hold them:
 *
 *   - failures become exceptions, so there is no status code to forget to check;
 *   - the SGP4 handle becomes a class, so there is no `close` to forget to call;
 *   - the two-call length protocol becomes a `std::vector`, so there is no buffer to size;
 *   - the defaults C cannot express become default arguments.
 *
 * What it does not change: the units. Everything is SI and every angle is radians, here as there.
 *
 * The types are the C structs, not wrappers around them. `spacelab::state` *is* `sl_state`, so it
 * can be memcpy'd, put in a `std::vector` handed straight to `sl_sgp4_propagate_batch`, or passed
 * to C code, with no conversion layer to go wrong.
 *
 * ---- Exceptions, and building without them ----
 *
 * Every fallible operation comes in two forms. `try_f(...)` returns a `result<T>` and never throws;
 * `f(...)` returns `T` and throws `spacelab::error` on failure. The second is written in terms of
 * the first, so they cannot disagree.
 *
 * Under `-fno-exceptions` the throwing half is not declared at all -- the `try_` half, the RAII
 * handle and the vector sizing all remain. That is a real difference in the declarations a
 * translation unit sees, so **do not compile some of your translation units with exceptions and
 * others without and link them together**; that is an ODR violation, here as everywhere else. The
 * alternative would have been to keep the throwing names and abort inside them, which turns an
 * ordinary domain error -- no Lambert arc exists for that time of flight -- into a dead process.
 *
 * `result<T>` is `std::expected<T, sl_status>` where the standard library has it, and a stand-in
 * with the same subset of its interface (`operator bool`, `operator*`, `operator->`, `error()`)
 * where it does not. Both are tested: the C++20 rung of the build gets the stand-in and the C++23
 * and C++26 rungs get the standard type.
 */

#ifndef SPACELAB_HPP
#define SPACELAB_HPP

#include "spacelab.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
#define SPACELAB_HAS_STD_EXPECTED 1
#include <expected>
#else
#define SPACELAB_HAS_STD_EXPECTED 0
#endif

namespace spacelab {

/* ---- Types ---- */

using vec3 = ::sl_vec3;
using state = ::sl_state;
using geodetic = ::sl_geodetic;
using latlon = ::sl_latlon;
using walker_params = ::sl_walker_params;
using soc_params = ::sl_soc_params;
using soc_geometry = ::sl_soc_geometry;
using flower_params = ::sl_flower_params;
using lattice_params = ::sl_lattice_params;
using draim_params = ::sl_draim_params;
using draim_orbit = ::sl_draim_orbit;
using sensor = ::sl_sensor;
using coverage_metrics = ::sl_coverage_metrics;
using constellation_fom = ::sl_constellation_fom;
using window = ::sl_window;
using gnss_state = ::sl_gnss_state;
using glonass_state = ::sl_glonass_state;
using gnss_system = ::sl_gnss_system;
using time_system = ::sl_time_system;
using obs_header = ::sl_obs_header;
using obs_descriptor = ::sl_obs_descriptor;
using obs_sample = ::sl_obs_sample;
using dop = ::sl_dop;
using mapping = ::sl_mapping;
using brouwer_lyddane_params = ::sl_brouwer_lyddane_params;
using mean_pole = ::sl_mean_pole;
using zonal = ::sl_zonal;
using drag_panel = ::sl_drag_panel;
using density_sample = ::sl_density_sample;
using density_observation = ::sl_density_observation;
using space_weather_day = ::sl_space_weather_day;
using swath = ::sl_swath;
using eop = ::sl_eop;
using eop_sample = ::sl_eop_sample;
using exponential_atmosphere = ::sl_exponential_atmosphere;
using nrlmsise_drag = ::sl_nrlmsise_drag;
using sdtm1_drag = ::sl_sdtm1_drag;
using precision = ::sl_precision;
using complex = ::sl_complex;
using csa_params = ::sl_csa_params;
using rda_geometry = ::sl_rda_geometry;
using tomo_params = ::sl_tomo_params;
using scatterer = ::sl_scatterer;
using polarimetric_sample = ::sl_polarimetric_sample;
using cloude_pottier = ::sl_cloude_pottier;
using deformation = ::sl_deformation;
using camera = ::sl_camera;
using mvs_view = ::sl_mvs_view;
using sgm_config = ::sl_sgm_config;
using harris_config = ::sl_harris_config;
using corner = ::sl_corner;
using feature_match = ::sl_feature_match;
using track_point = ::sl_track_point;
using ground_control_point = ::sl_ground_control_point;
using image_point = ::sl_image_point;
using atmospheric_coupling = ::sl_atmospheric_coupling;
using bbox = ::sl_bbox;
using gpt3_weather = ::sl_gpt3_weather;
using vmf_coefficients = ::sl_vmf_coefficients;
using walker_pattern = ::sl_walker_pattern;
using dsst_params = ::sl_dsst_params;
using residual_row = ::sl_residual_row;
using gauss_newton_options = ::sl_gauss_newton_options;
using estimate = ::sl_estimate;
using terminal = ::sl_terminal;
using band = ::sl_band;
using ramp_segment = ::sl_ramp_segment;
using doppler_link = ::sl_doppler_link;
using corona_term = ::sl_corona_term;
using lifetime_outcome = ::sl_lifetime_outcome;
using decay_params = ::sl_decay_params;
using lifetime = ::sl_lifetime;
using lifetime_ensemble = ::sl_lifetime_ensemble;
using utc_datetime = ::sl_utc_datetime;
using lanes = ::sl_lanes;
using slip = ::sl_slip;
using slip_verdict = ::sl_slip_verdict;
using conjunction = ::sl_conjunction;
using probability_estimates = ::sl_probability_estimates;
using relative_sample = ::sl_relative_sample;
using usage_violations = ::sl_usage_violations;
using close_approach = ::sl_close_approach;

/*
 * A mean equinoctial element set `[a, k, h, q, p, lambda]`, which is what DSST works in.
 *
 * An `std::array` rather than a struct with six named fields: `sl_dsst_*` takes and returns these
 * as flat doubles, and the static assertion below is what makes a `std::vector` of them a buffer
 * the C side can fill directly, with no per-element copy on the way back.
 */
using equinoctial = std::array<double, 6>;
static_assert(sizeof(equinoctial) == 6 * sizeof(double),
              "a vector<equinoctial> is handed to C as 6*n contiguous doubles");

/* The C macros again, as constants that obey scope and type. */
inline constexpr double mu_earth = SL_MU_EARTH;
inline constexpr double re_earth = SL_RE_EARTH;
inline constexpr double j2_earth = SL_J2_EARTH;
inline constexpr double j3_earth = SL_J3_EARTH;
inline constexpr double earth_rotation_rate = SL_EARTH_ROTATION_RATE;

/* ---- Results ---- */

#if SPACELAB_HAS_STD_EXPECTED

template <class T> using result = std::expected<T, ::sl_status>;

#else

/*
 * The subset of `std::expected` this header uses, for standard libraries that do not have it.
 * Deliberately not a general-purpose type: no `value()`, because the only thing it could do on a
 * failure is throw, and this type exists partly so that a build without exceptions has an answer.
 * Dereferencing a failed result is a precondition violation, exactly as it is for `std::expected`.
 */
template <class T> class result {
  public:
    result(T value) : value_(std::move(value)) {} // NOLINT(google-explicit-constructor)

    [[nodiscard]] static result failure(::sl_status status) {
        result r;
        r.status_ = status;
        return r;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return status_ == SL_OK;
    }

    [[nodiscard]] bool has_value() const noexcept {
        return status_ == SL_OK;
    }

    [[nodiscard]] ::sl_status error() const noexcept {
        return status_;
    }

    [[nodiscard]] const T &operator*() const & noexcept {
        return *value_;
    }

    [[nodiscard]] T &operator*() & noexcept {
        return *value_;
    }

    [[nodiscard]] T &&operator*() && noexcept {
        return std::move(*value_);
    }

    [[nodiscard]] const T *operator->() const noexcept {
        return &*value_;
    }

    [[nodiscard]] T *operator->() noexcept {
        return &*value_;
    }

  private:
    result() = default;

    std::optional<T> value_;
    ::sl_status status_ = SL_OK;
};

#endif

namespace detail {

/* The one spelling of "this failed" that works for both flavours of `result`. */
template <class T> [[nodiscard]] result<T> failure(::sl_status status) {
#if SPACELAB_HAS_STD_EXPECTED
    return std::unexpected(status);
#else
    return result<T>::failure(status);
#endif
}

} // namespace detail

/* ---- Errors ---- */

/*
 * The message the library recorded for the most recent failure on this thread, or an empty string
 * if nothing has failed. Rarely needed directly: `error::what()` already carries it.
 */
[[nodiscard]] inline std::string last_error_message() {
    const std::size_t needed = ::sl_last_error_message(nullptr, 0);
    std::string message(needed, '\0');
    ::sl_last_error_message(message.data(), needed);
    message.pop_back(); /* the NUL the C API writes; std::string carries its own length */
    return message;
}

/*
 * A failed call. `status()` is the `SL_ERR_*` code, kept because that is what a caller branches on
 * -- `SL_ERR_DOMAIN` is a question with no answer and often expected, `SL_ERR_PANIC` is a bug and
 * never is -- and `what()` is the library's own description of it.
 *
 * Declared even where exceptions are off, so that a `-fno-exceptions` consumer can still name the
 * type; only the code that throws it is conditional.
 */
class error : public std::runtime_error {
  public:
    explicit error(::sl_status status)
        : std::runtime_error(last_error_message()), status_(status) {}

    [[nodiscard]] ::sl_status status() const noexcept {
        return status_;
    }

  private:
    ::sl_status status_;
};

#if defined(__cpp_exceptions)
namespace detail {

/* The whole of the throwing layer: every `f` below is `unwrap(try_f(...))`. */
template <class T> [[nodiscard]] T unwrap(result<T> r) {
    if (!r) {
        throw error(r.error());
    }
    return *std::move(r);
}

} // namespace detail
#endif

/* ---- Library version ---- */

[[nodiscard]] inline std::uint32_t abi_version() noexcept {
    return ::sl_abi_version();
}

[[nodiscard]] inline std::string version() {
    const std::size_t needed = ::sl_version_string(nullptr, 0);
    std::string text(needed, '\0');
    ::sl_version_string(text.data(), needed);
    text.pop_back();
    return text;
}

/* ---- Time and frames ---- */

[[nodiscard]] inline result<state> try_teme_to_itrf(const state &in, std::int64_t ut1_ns,
                                                    std::int64_t tt_ns, double xp_rad = 0.0,
                                                    double yp_rad = 0.0, double lod_seconds = 0.0) {
    state out{};
    const ::sl_status s = ::sl_teme_to_itrf(&in, ut1_ns, tt_ns, xp_rad, yp_rad, lod_seconds, &out);
    return s == SL_OK ? result<state>(out) : detail::failure<state>(s);
}

[[nodiscard]] inline result<state> try_gcrf_to_itrf(const state &in, std::int64_t ut1_ns,
                                                    std::int64_t tt_ns, double xp_rad = 0.0,
                                                    double yp_rad = 0.0, double dx_rad = 0.0,
                                                    double dy_rad = 0.0, double lod_seconds = 0.0) {
    state out{};
    const ::sl_status s =
        ::sl_gcrf_to_itrf(&in, ut1_ns, tt_ns, xp_rad, yp_rad, dx_rad, dy_rad, lod_seconds, &out);
    return s == SL_OK ? result<state>(out) : detail::failure<state>(s);
}

[[nodiscard]] inline result<geodetic> try_itrf_to_geodetic(const vec3 &r_itrf) {
    geodetic out{};
    const ::sl_status s = ::sl_itrf_to_geodetic(&r_itrf, &out);
    return s == SL_OK ? result<geodetic>(out) : detail::failure<geodetic>(s);
}

[[nodiscard]] inline result<vec3> try_geodetic_to_itrf(const geodetic &g) {
    vec3 out{};
    const ::sl_status s = ::sl_geodetic_to_itrf(&g, &out);
    return s == SL_OK ? result<vec3>(out) : detail::failure<vec3>(s);
}

#if defined(__cpp_exceptions)

[[nodiscard]] inline state teme_to_itrf(const state &in, std::int64_t ut1_ns, std::int64_t tt_ns,
                                        double xp_rad = 0.0, double yp_rad = 0.0,
                                        double lod_seconds = 0.0) {
    return detail::unwrap(try_teme_to_itrf(in, ut1_ns, tt_ns, xp_rad, yp_rad, lod_seconds));
}

[[nodiscard]] inline state gcrf_to_itrf(const state &in, std::int64_t ut1_ns, std::int64_t tt_ns,
                                        double xp_rad = 0.0, double yp_rad = 0.0,
                                        double dx_rad = 0.0, double dy_rad = 0.0,
                                        double lod_seconds = 0.0) {
    return detail::unwrap(
        try_gcrf_to_itrf(in, ut1_ns, tt_ns, xp_rad, yp_rad, dx_rad, dy_rad, lod_seconds));
}

[[nodiscard]] inline geodetic itrf_to_geodetic(const vec3 &r_itrf) {
    return detail::unwrap(try_itrf_to_geodetic(r_itrf));
}

[[nodiscard]] inline vec3 geodetic_to_itrf(const geodetic &g) {
    return detail::unwrap(try_geodetic_to_itrf(g));
}

#endif

/* ---- SGP4 ---- */

/*
 * A parsed element set. Constructing it parses; destroying it releases. There is no way to leak
 * one and no way to close one twice, which is the whole reason this class exists over the handle.
 *
 * Movable, not copyable: the handle is unique, and a copy would have to re-parse -- silently
 * doubling the work the batch call exists to avoid. Construct a second one if you want a second.
 *
 * `open` is the constructor for a build without exceptions: a constructor has no way to report a
 * malformed element set except by throwing, so the factory is the form that always exists.
 */
class sgp4 {
  public:
    [[nodiscard]] static result<sgp4> open(const std::string &line1, const std::string &line2) {
        ::sl_sgp4 *raw = nullptr;
        const ::sl_status s = ::sl_sgp4_open(line1.c_str(), line2.c_str(), &raw);
        return s == SL_OK ? result<sgp4>(sgp4(raw)) : detail::failure<sgp4>(s);
    }

    [[nodiscard]] result<state> try_at(double minutes) const {
        state out{};
        const ::sl_status s = ::sl_sgp4_propagate_at(handle_.get(), minutes, &out);
        return s == SL_OK ? result<state>(out) : detail::failure<state>(s);
    }

    [[nodiscard]] result<std::vector<state>> try_at(std::span<const double> minutes) const {
        // An empty span has a null `data()`, and the C rule -- correctly, for C -- is that a null
        // is an invalid argument. Here the length is in the type, so "nothing to propagate" is an
        // answer rather than a mistake, and returning it is not the same as hiding an error.
        if (minutes.empty()) {
            return std::vector<state>{};
        }
        std::vector<state> out(minutes.size());
        const ::sl_status s =
            ::sl_sgp4_propagate_batch(handle_.get(), minutes.data(), minutes.size(), out.data());
        return s == SL_OK ? result<std::vector<state>>(std::move(out))
                          : detail::failure<std::vector<state>>(s);
    }

#if defined(__cpp_exceptions)
    sgp4(const std::string &line1, const std::string &line2)
        : sgp4(detail::unwrap(open(line1, line2))) {}

    [[nodiscard]] state at(double minutes) const {
        return detail::unwrap(try_at(minutes));
    }

    [[nodiscard]] std::vector<state> at(std::span<const double> minutes) const {
        return detail::unwrap(try_at(minutes));
    }
#endif

  private:
    explicit sgp4(::sl_sgp4 *raw) : handle_(raw) {}

    struct closer {
        void operator()(::sl_sgp4 *p) const noexcept {
            ::sl_sgp4_close(p);
        }
    };

    std::unique_ptr<::sl_sgp4, closer> handle_;
};

/* Parse and propagate in one call; see `sl_sgp4_propagate`. */
[[nodiscard]] inline result<state> try_sgp4_propagate(const std::string &line1,
                                                      const std::string &line2, double minutes) {
    state out{};
    const ::sl_status s = ::sl_sgp4_propagate(line1.c_str(), line2.c_str(), minutes, &out);
    return s == SL_OK ? result<state>(out) : detail::failure<state>(s);
}

/* ---- Two-body motion ---- */

[[nodiscard]] inline result<state> try_kepler_propagate(const state &in, double mu, double dt) {
    state out{};
    const ::sl_status s = ::sl_kepler_propagate(&in, mu, dt, &out);
    return s == SL_OK ? result<state>(out) : detail::failure<state>(s);
}

[[nodiscard]] inline result<state> try_propagate_two_body(const state &in, double mu, double dt,
                                                          std::uint64_t steps,
                                                          precision p = SL_PRECISION_DOUBLE) {
    state out{};
    const ::sl_status s = ::sl_propagate_two_body(&in, mu, dt, steps, p, &out);
    return s == SL_OK ? result<state>(out) : detail::failure<state>(s);
}

/* The two velocities of a Lambert transfer, departure first. `SL_ERR_DOMAIN` when no arc exists,
 * which for a porkchop sweep is an ordinary outcome and not a fault. */
struct transfer {
    vec3 departure;
    vec3 arrival;
};

[[nodiscard]] inline result<transfer> try_lambert(const vec3 &r1, const vec3 &r2, double tof,
                                                  double mu, bool prograde = true) {
    transfer out{};
    const ::sl_status s = ::sl_lambert(&r1, &r2, tof, mu, prograde, &out.departure, &out.arrival);
    return s == SL_OK ? result<transfer>(out) : detail::failure<transfer>(s);
}

#if defined(__cpp_exceptions)

[[nodiscard]] inline state sgp4_propagate(const std::string &line1, const std::string &line2,
                                          double minutes) {
    return detail::unwrap(try_sgp4_propagate(line1, line2, minutes));
}

[[nodiscard]] inline state kepler_propagate(const state &in, double mu, double dt) {
    return detail::unwrap(try_kepler_propagate(in, mu, dt));
}

[[nodiscard]] inline state propagate_two_body(const state &in, double mu, double dt,
                                              std::uint64_t steps,
                                              precision p = SL_PRECISION_DOUBLE) {
    return detail::unwrap(try_propagate_two_body(in, mu, dt, steps, p));
}

[[nodiscard]] inline transfer lambert(const vec3 &r1, const vec3 &r2, double tof, double mu,
                                      bool prograde = true) {
    return detail::unwrap(try_lambert(r1, r2, tof, mu, prograde));
}

#endif

/* ---- Measurements from the ground ---- */

[[nodiscard]] inline double range(const state &sat, const vec3 &station_ecef) noexcept {
    return ::sl_range(&sat, &station_ecef);
}

[[nodiscard]] inline double range_rate(const state &sat, const vec3 &station_ecef) noexcept {
    return ::sl_range_rate(&sat, &station_ecef);
}

[[nodiscard]] inline double elevation(const state &sat, const vec3 &station_ecef,
                                      double latitude_rad, double longitude_rad) noexcept {
    return ::sl_elevation(&sat, &station_ecef, latitude_rad, longitude_rad);
}

[[nodiscard]] inline double azimuth(const state &sat, const vec3 &station_ecef, double latitude_rad,
                                    double longitude_rad) noexcept {
    return ::sl_azimuth(&sat, &station_ecef, latitude_rad, longitude_rad);
}

/* ---- Footprint geometry ---- */

[[nodiscard]] inline double horizon_look_angle(double earth_radius_m, double altitude_m) noexcept {
    return ::sl_horizon_look_angle(earth_radius_m, altitude_m);
}

[[nodiscard]] inline latlon ground_point_at(latlon start, double central_angle_rad,
                                            double azimuth_rad) noexcept {
    return ::sl_ground_point_at(start, central_angle_rad, azimuth_rad);
}

/* ---- Orbit design ---- */

[[nodiscard]] inline double critical_inclination() noexcept {
    return ::sl_critical_inclination();
}

[[nodiscard]] inline double frozen_eccentricity(double semi_major_axis_m, double inclination_rad,
                                                double earth_radius_m = re_earth,
                                                double j2 = j2_earth,
                                                double j3 = j3_earth) noexcept {
    return ::sl_frozen_eccentricity(semi_major_axis_m, inclination_rad, earth_radius_m, j2, j3);
}

[[nodiscard]] inline result<double>
try_repeat_ground_track_sma(std::uint32_t revs, std::uint32_t days, double inclination_rad,
                            double eccentricity = 0.0, double mu = mu_earth,
                            double earth_radius_m = re_earth, double j2 = j2_earth,
                            double rotation_rate = earth_rotation_rate) {
    double out = 0.0;
    const ::sl_status s = ::sl_repeat_ground_track_sma(revs, days, inclination_rad, eccentricity,
                                                       mu, earth_radius_m, j2, rotation_rate, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<double>
try_sun_synchronous_inclination(double semi_major_axis_m, double eccentricity = 0.0,
                                double mu = mu_earth, double earth_radius_m = re_earth,
                                double j2 = j2_earth) {
    double out = 0.0;
    const ::sl_status s = ::sl_sun_synchronous_inclination(semi_major_axis_m, eccentricity, mu,
                                                           earth_radius_m, j2, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

#if defined(__cpp_exceptions)

[[nodiscard]] inline double repeat_ground_track_sma(std::uint32_t revs, std::uint32_t days,
                                                    double inclination_rad,
                                                    double eccentricity = 0.0, double mu = mu_earth,
                                                    double earth_radius_m = re_earth,
                                                    double j2 = j2_earth,
                                                    double rotation_rate = earth_rotation_rate) {
    return detail::unwrap(try_repeat_ground_track_sma(revs, days, inclination_rad, eccentricity, mu,
                                                      earth_radius_m, j2, rotation_rate));
}

[[nodiscard]] inline double sun_synchronous_inclination(double semi_major_axis_m,
                                                        double eccentricity = 0.0,
                                                        double mu = mu_earth,
                                                        double earth_radius_m = re_earth,
                                                        double j2 = j2_earth) {
    return detail::unwrap(
        try_sun_synchronous_inclination(semi_major_axis_m, eccentricity, mu, earth_radius_m, j2));
}

#endif

/* ---- Constellations ---- */

/* The defaults `sl_walker_params_init` writes. Start from this and override what you mean. */
[[nodiscard]] inline walker_params default_walker_params() noexcept {
    walker_params p{};
    ::sl_walker_params_init(&p);
    return p;
}

/* The defaults `sl_soc_params_init` writes. */
[[nodiscard]] inline soc_params default_soc_params() noexcept {
    soc_params p{};
    ::sl_soc_params_init(&p);
    return p;
}

/*
 * Both generators run the C two-call protocol for you: ask the length, size the vector, fill it.
 * The second call cannot come back short, because the vector was sized from the first.
 */
[[nodiscard]] inline result<std::vector<state>> try_walker(const walker_params &p) {
    std::size_t len = 0;
    ::sl_status s = ::sl_walker(&p, nullptr, 0, &len);
    if (s != SL_OK) {
        return detail::failure<std::vector<state>>(s);
    }
    std::vector<state> out(len);
    s = ::sl_walker(&p, out.data(), out.size(), &len);
    return s == SL_OK ? result<std::vector<state>>(std::move(out))
                      : detail::failure<std::vector<state>>(s);
}

[[nodiscard]] inline result<soc_geometry> try_street_of_coverage_geometry(const soc_params &p) {
    soc_geometry out{};
    const ::sl_status s = ::sl_street_of_coverage_geometry(&p, &out);
    return s == SL_OK ? result<soc_geometry>(out) : detail::failure<soc_geometry>(s);
}

[[nodiscard]] inline result<std::vector<state>> try_street_of_coverage(const soc_params &p) {
    std::size_t len = 0;
    ::sl_status s = ::sl_street_of_coverage(&p, nullptr, 0, &len);
    if (s != SL_OK) {
        return detail::failure<std::vector<state>>(s);
    }
    std::vector<state> out(len);
    s = ::sl_street_of_coverage(&p, out.data(), out.size(), &len);
    return s == SL_OK ? result<std::vector<state>>(std::move(out))
                      : detail::failure<std::vector<state>>(s);
}

#if defined(__cpp_exceptions)

[[nodiscard]] inline std::vector<state> walker(const walker_params &p) {
    return detail::unwrap(try_walker(p));
}

[[nodiscard]] inline soc_geometry street_of_coverage_geometry(const soc_params &p) {
    return detail::unwrap(try_street_of_coverage_geometry(p));
}

[[nodiscard]] inline std::vector<state> street_of_coverage(const soc_params &p) {
    return detail::unwrap(try_street_of_coverage(p));
}

#endif

/* The defaults the three `_init` entry points write. */
[[nodiscard]] inline flower_params default_flower_params() noexcept {
    flower_params p{};
    ::sl_flower_params_init(&p);
    return p;
}

[[nodiscard]] inline lattice_params default_lattice_params() noexcept {
    lattice_params p{};
    ::sl_lattice_params_init(&p);
    return p;
}

[[nodiscard]] inline draim_params default_draim_params() noexcept {
    draim_params p{};
    ::sl_draim_params_init(&p);
    return p;
}

namespace detail {

/*
 * The two-call protocol, once, for every variable-length result below. `call` is invoked as
 * `call(out, cap, &len)` -- first with a null `out` to learn the length, then with a vector sized
 * from it, which is why the second call cannot come back short.
 */
template <class T, class Call> [[nodiscard]] inline result<std::vector<T>> sized(Call call) {
    using many = std::vector<T>;
    std::size_t len = 0;
    ::sl_status s = call(static_cast<T *>(nullptr), std::size_t{0}, &len);
    if (s != SL_OK) {
        return failure<many>(s);
    }
    many out(len);
    s = call(out.data(), out.size(), &len);
    return s == SL_OK ? result<many>(std::move(out)) : failure<many>(s);
}

} // namespace detail

[[nodiscard]] inline result<std::vector<state>> try_flower(const flower_params &p) {
    return detail::sized<state>([&p](state *out, std::size_t cap, std::size_t *len) {
        return ::sl_flower(&p, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<state>> try_lattice_flower(const lattice_params &p) {
    return detail::sized<state>([&p](state *out, std::size_t cap, std::size_t *len) {
        return ::sl_lattice_flower(&p, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<state>> try_draim(const draim_params &p) {
    return detail::sized<state>([&p](state *out, std::size_t cap, std::size_t *len) {
        return ::sl_draim(&p, out, cap, len);
    });
}

[[nodiscard]] inline result<draim_orbit> try_draim_geometry(const draim_params &p) {
    draim_orbit out{};
    const ::sl_status s = ::sl_draim_geometry(&p, &out);
    return s == SL_OK ? result<draim_orbit>(out) : detail::failure<draim_orbit>(s);
}

[[nodiscard]] inline result<double> try_alpha_characteristic(std::span<const state> states,
                                                             double mu = mu_earth,
                                                             std::uint32_t fold = 1,
                                                             std::uint32_t grid_points = 500,
                                                             std::uint32_t time_samples = 60) {
    double out = 0.0;
    const ::sl_status s = ::sl_alpha_characteristic(states.data(), states.size(), mu, fold,
                                                    grid_points, time_samples, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<walker_params>
try_minimum_walker(double max_footprint_rad, double inclination_rad, double semi_major_axis_m,
                   double mu = mu_earth, std::uint32_t fold = 1, std::uint32_t max_total = 40,
                   std::uint32_t grid_points = 200, std::uint32_t time_samples = 20) {
    walker_params out{};
    const ::sl_status s = ::sl_minimum_walker(max_footprint_rad, inclination_rad, semi_major_axis_m,
                                              mu, fold, max_total, grid_points, time_samples, &out);
    return s == SL_OK ? result<walker_params>(out) : detail::failure<walker_params>(s);
}

#if defined(__cpp_exceptions)

[[nodiscard]] inline std::vector<state> flower(const flower_params &p) {
    return detail::unwrap(try_flower(p));
}

[[nodiscard]] inline std::vector<state> lattice_flower(const lattice_params &p) {
    return detail::unwrap(try_lattice_flower(p));
}

[[nodiscard]] inline std::vector<state> draim(const draim_params &p) {
    return detail::unwrap(try_draim(p));
}

[[nodiscard]] inline draim_orbit draim_geometry(const draim_params &p) {
    return detail::unwrap(try_draim_geometry(p));
}

[[nodiscard]] inline double alpha_characteristic(std::span<const state> states,
                                                 double mu = mu_earth, std::uint32_t fold = 1,
                                                 std::uint32_t grid_points = 500,
                                                 std::uint32_t time_samples = 60) {
    return detail::unwrap(try_alpha_characteristic(states, mu, fold, grid_points, time_samples));
}

[[nodiscard]] inline walker_params
minimum_walker(double max_footprint_rad, double inclination_rad, double semi_major_axis_m,
               double mu = mu_earth, std::uint32_t fold = 1, std::uint32_t max_total = 40,
               std::uint32_t grid_points = 200, std::uint32_t time_samples = 20) {
    return detail::unwrap(try_minimum_walker(max_footprint_rad, inclination_rad, semi_major_axis_m,
                                             mu, fold, max_total, grid_points, time_samples));
}

#endif

/* ---- Coverage ---- */

/*
 * Three named constructors rather than one `sl_sensor` to fill in by hand. `sl_sensor` is a tagged
 * struct because C cannot overload, and the hazard that comes with it -- setting a field the tag
 * does not name, and getting the default zero for the one it does -- is one C++ can simply not
 * have. Each of these sets the tag and exactly the fields that tag reads.
 */
[[nodiscard]] inline sensor elevation_mask(double min_elevation_rad) noexcept {
    sensor s{};
    ::sl_sensor_init(&s, SL_SENSOR_ELEVATION_MASK);
    s.min_elevation_rad = min_elevation_rad;
    return s;
}

[[nodiscard]] inline sensor side_looking_sar(double near_incidence_rad,
                                             double far_incidence_rad) noexcept {
    sensor s{};
    ::sl_sensor_init(&s, SL_SENSOR_SIDE_LOOKING_SAR);
    s.near_incidence_rad = near_incidence_rad;
    s.far_incidence_rad = far_incidence_rad;
    return s;
}

[[nodiscard]] inline sensor agile_optical(double max_off_nadir_rad) noexcept {
    sensor s{};
    ::sl_sensor_init(&s, SL_SENSOR_AGILE_OPTICAL);
    s.max_off_nadir_rad = max_off_nadir_rad;
    return s;
}

/*
 * `min_elevation_rad` above is a *geometric* elevation and a station's published "visible above 10
 * degrees" is the *apparent* one -- worth 0.81 s to 20.3 s of pass edge, with the same sign on every
 * edge of every pass. These two convert between them, and `elevation_mask_apparent` is the
 * constructor to reach for when the mask came from a station's spec rather than from geometry.
 * Converting the threshold once is exact, not an approximation: `h -> h + R(h)` is strictly
 * increasing, so one apparent mask is exactly one geometric mask.
 */
[[nodiscard]] inline result<double> try_refraction_apparent(double true_altitude_rad,
                                                            double pressure_hpa = 1010.0,
                                                            double temperature_k = 283.0) {
    double out = 0.0;
    const ::sl_status s =
        ::sl_refraction_apparent(true_altitude_rad, pressure_hpa, temperature_k, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<double> try_refraction_geometric(double apparent_altitude_rad,
                                                             double pressure_hpa = 1010.0,
                                                             double temperature_k = 283.0) {
    double out = 0.0;
    const ::sl_status s =
        ::sl_refraction_geometric(apparent_altitude_rad, pressure_hpa, temperature_k, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<sensor> try_elevation_mask_apparent(double apparent_elevation_rad,
                                                                double pressure_hpa = 1010.0,
                                                                double temperature_k = 283.0) {
    const result<double> geometric =
        try_refraction_geometric(apparent_elevation_rad, pressure_hpa, temperature_k);
    if (!geometric) {
        return detail::failure<sensor>(geometric.error());
    }
    return elevation_mask(*geometric);
}

/*
 * Stellar aberration. The velocity a satellite observation needs is the station's own rotation,
 * `diurnal_velocity(station, {0, 0, SL_EARTH_ROTATION_RATE})`, and not the Earth's orbital motion:
 * station and satellite share the latter and a shared velocity cancels in a geocentric frame. What
 * is left is 465.1 m/s at the equator, worth 0.320" there and 0.262" at 34.9 degrees. `apply` is
 * natural -> apparent and `remove` is apparent -> natural; they are exact inverses.
 */
[[nodiscard]] inline result<vec3> try_diurnal_velocity(const vec3 &station_m,
                                                       const vec3 &angular_velocity_rad_s) {
    vec3 out{};
    const ::sl_status s = ::sl_diurnal_velocity(&station_m, &angular_velocity_rad_s, &out);
    return s == SL_OK ? result<vec3>(out) : detail::failure<vec3>(s);
}

[[nodiscard]] inline result<vec3> try_aberration_apply(const vec3 &velocity_mps,
                                                       const vec3 &natural) {
    vec3 out{};
    const ::sl_status s = ::sl_aberration_apply(&velocity_mps, &natural, &out);
    return s == SL_OK ? result<vec3>(out) : detail::failure<vec3>(s);
}

[[nodiscard]] inline result<vec3> try_aberration_remove(const vec3 &velocity_mps,
                                                        const vec3 &apparent) {
    vec3 out{};
    const ::sl_status s = ::sl_aberration_remove(&velocity_mps, &apparent, &out);
    return s == SL_OK ? result<vec3>(out) : detail::failure<vec3>(s);
}

/* The orbit a coverage query is asked about -- one circular orbit, as `sl_coverage_scan` takes
 * it. Bundled because four bare doubles in a row is four chances to transpose two of them. */
struct ground_track {
    double semi_major_axis_m = 0.0;
    double inclination_rad = 0.0;
    double raan_rad = 0.0;
    double phase_rad = 0.0;
};

/* Ground cells from a latitude and a longitude grid, both in radians. */
[[nodiscard]] inline result<std::vector<state>> try_grid_cells(std::span<const double> lats_rad,
                                                               std::span<const double> lons_rad) {
    return detail::sized<state>([&](state *out, std::size_t cap, std::size_t *len) {
        return ::sl_grid_cells(lats_rad.data(), lats_rad.size(), lons_rad.data(), lons_rad.size(),
                               out, cap, len);
    });
}

/*
 * Per-cell coverage and revisit metrics. The result is sized here rather than by the caller,
 * because it is one entry per cell and there is no other length it could be.
 */
[[nodiscard]] inline result<std::vector<coverage_metrics>>
try_coverage(const ground_track &orbit, const sensor &instrument, std::span<const state> cells,
             double horizon_s, double tol_s = 1.0, std::uint32_t threads = 0) {
    using many = std::vector<coverage_metrics>;
    if (cells.empty()) {
        return many{};
    }
    many out(cells.size());
    const ::sl_status s = ::sl_coverage_scan(
        orbit.semi_major_axis_m, orbit.inclination_rad, orbit.raan_rad, orbit.phase_rad,
        &instrument, cells.data(), cells.size(), horizon_s, tol_s, threads, out.data());
    return s == SL_OK ? result<many>(std::move(out)) : detail::failure<many>(s);
}

/* The raw access windows over one cell. */
[[nodiscard]] inline result<std::vector<window>>
try_access_windows(const ground_track &orbit, const sensor &instrument, const state &cell,
                   double horizon_s, double tol_s = 1.0) {
    return detail::sized<window>([&](window *out, std::size_t cap, std::size_t *len) {
        return ::sl_access_windows(orbit.semi_major_axis_m, orbit.inclination_rad, orbit.raan_rad,
                                   orbit.phase_rad, &instrument, &cell, horizon_s, tol_s, out, cap,
                                   len);
    });
}

/*
 * Aggregate per-cell metrics. An empty `weights` weights every cell equally; otherwise it must be
 * one weight per cell, and `cos(latitude)` is the usual choice.
 */
[[nodiscard]] inline result<constellation_fom>
try_figures_of_merit(std::span<const coverage_metrics> cells,
                     std::span<const double> weights = {}) {
    constellation_fom out{};
    if (!weights.empty() && weights.size() != cells.size()) {
        return detail::failure<constellation_fom>(SL_ERR_INVALID_ARGUMENT);
    }
    const ::sl_status s = ::sl_figures_of_merit(cells.data(), cells.size(),
                                                weights.empty() ? nullptr : weights.data(), &out);
    return s == SL_OK ? result<constellation_fom>(out) : detail::failure<constellation_fom>(s);
}

#if defined(__cpp_exceptions)

[[nodiscard]] inline double refraction_apparent(double true_altitude_rad,
                                                double pressure_hpa = 1010.0,
                                                double temperature_k = 283.0) {
    return detail::unwrap(try_refraction_apparent(true_altitude_rad, pressure_hpa, temperature_k));
}

[[nodiscard]] inline double refraction_geometric(double apparent_altitude_rad,
                                                 double pressure_hpa = 1010.0,
                                                 double temperature_k = 283.0) {
    return detail::unwrap(
        try_refraction_geometric(apparent_altitude_rad, pressure_hpa, temperature_k));
}

[[nodiscard]] inline sensor elevation_mask_apparent(double apparent_elevation_rad,
                                                    double pressure_hpa = 1010.0,
                                                    double temperature_k = 283.0) {
    return detail::unwrap(
        try_elevation_mask_apparent(apparent_elevation_rad, pressure_hpa, temperature_k));
}

[[nodiscard]] inline vec3 diurnal_velocity(const vec3 &station_m,
                                           const vec3 &angular_velocity_rad_s) {
    return detail::unwrap(try_diurnal_velocity(station_m, angular_velocity_rad_s));
}

[[nodiscard]] inline vec3 aberration_apply(const vec3 &velocity_mps, const vec3 &natural) {
    return detail::unwrap(try_aberration_apply(velocity_mps, natural));
}

[[nodiscard]] inline vec3 aberration_remove(const vec3 &velocity_mps, const vec3 &apparent) {
    return detail::unwrap(try_aberration_remove(velocity_mps, apparent));
}

[[nodiscard]] inline std::vector<state> grid_cells(std::span<const double> lats_rad,
                                                   std::span<const double> lons_rad) {
    return detail::unwrap(try_grid_cells(lats_rad, lons_rad));
}

[[nodiscard]] inline std::vector<coverage_metrics>
coverage(const ground_track &orbit, const sensor &instrument, std::span<const state> cells,
         double horizon_s, double tol_s = 1.0, std::uint32_t threads = 0) {
    return detail::unwrap(try_coverage(orbit, instrument, cells, horizon_s, tol_s, threads));
}

[[nodiscard]] inline std::vector<window> access_windows(const ground_track &orbit,
                                                        const sensor &instrument, const state &cell,
                                                        double horizon_s, double tol_s = 1.0) {
    return detail::unwrap(try_access_windows(orbit, instrument, cell, horizon_s, tol_s));
}

[[nodiscard]] inline constellation_fom figures_of_merit(std::span<const coverage_metrics> cells,
                                                        std::span<const double> weights = {}) {
    return detail::unwrap(try_figures_of_merit(cells, weights));
}

#endif

/* ---- Numerical propagation ---- */

/*
 * A dense ephemeris, owned. Same bargain as `sgp4`: the handle is released when this goes out of
 * scope, and there is no way to hold a closed one. The two constructors are factories rather than
 * constructors for the same reason `sgp4::open` is -- an integration that fails has to be able to
 * say so without throwing.
 */
class ephemeris {
  public:
    [[nodiscard]] static result<ephemeris> j2(const state &in, double t1_seconds, double mu,
                                              double j2 = 0.0, double earth_radius_m = re_earth) {
        ::sl_ephemeris *raw = nullptr;
        const ::sl_status s = ::sl_ephemeris_j2(&in, t1_seconds, mu, j2, earth_radius_m, &raw);
        return s == SL_OK ? result<ephemeris>(ephemeris(raw)) : detail::failure<ephemeris>(s);
    }

    [[nodiscard]] static result<ephemeris> geopotential(const state &in, double t1_seconds,
                                                        std::uint32_t degree,
                                                        double theta0_rad = 0.0,
                                                        double tolerance = 1e-9) {
        ::sl_ephemeris *raw = nullptr;
        const ::sl_status s =
            ::sl_ephemeris_geopotential(&in, t1_seconds, degree, theta0_rad, tolerance, &raw);
        return s == SL_OK ? result<ephemeris>(ephemeris(raw)) : detail::failure<ephemeris>(s);
    }

    /* The same two forces as `j2`, under DOP853 rather than IAS15. */
    [[nodiscard]] static result<ephemeris> j2_dense(const state &in, double t1_seconds, double mu,
                                                    double j2 = 0.0,
                                                    double earth_radius_m = re_earth,
                                                    double tolerance = 1e-9) {
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_j2_dense(&in, t1_seconds, mu, j2, earth_radius_m, tolerance, raw);
        });
    }

    [[nodiscard]] static result<ephemeris> drag(const state &in, double t1_seconds,
                                                std::uint32_t degree,
                                                const exponential_atmosphere &atmosphere,
                                                double theta0_rad = 0.0, double tolerance = 1e-9) {
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_drag(&in, t1_seconds, degree, theta0_rad, &atmosphere, tolerance,
                                       raw);
        });
    }

    [[nodiscard]] static result<ephemeris> drag_nrlmsise00(const state &in, double t1_seconds,
                                                           const eop &orientation,
                                                           const nrlmsise_drag &model,
                                                           double tolerance = 1e-9) {
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_drag_nrlmsise00(&in, t1_seconds, &orientation, &model, tolerance,
                                                  raw);
        });
    }

    [[nodiscard]] static result<ephemeris>
    drag_sdtm1(const state &in, double t1_seconds, const eop &orientation, const sdtm1_drag &model,
               std::span<const double> ap_history, double cadence_s = 10800.0,
               double tolerance = 1e-9) {
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_drag_sdtm1(&in, t1_seconds, &orientation, &model,
                                             ap_history.data(), ap_history.size(), cadence_s,
                                             tolerance, raw);
        });
    }

    [[nodiscard]] static result<ephemeris>
    drag_sdtm1_solar_wind(const state &in, double t1_seconds, const eop &orientation,
                          const sdtm1_drag &model, std::span<const double> ap60,
                          std::span<const double> speed_km_s, std::span<const double> by_nt,
                          std::span<const double> bz_nt, double tolerance = 1e-9) {
        if (speed_km_s.size() != ap60.size() || by_nt.size() != ap60.size() ||
            bz_nt.size() != ap60.size()) {
            return detail::failure<ephemeris>(SL_ERR_INVALID_ARGUMENT);
        }
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_drag_sdtm1_solar_wind(
                &in, t1_seconds, &orientation, &model, ap60.data(), speed_km_s.data(), by_nt.data(),
                bz_nt.data(), ap60.size(), tolerance, raw);
        });
    }

    [[nodiscard]] static result<ephemeris> geopotential_itrf(const state &in, double t1_seconds,
                                                             std::uint32_t degree,
                                                             const eop &orientation,
                                                             double tolerance = 1e-9) {
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_geopotential_itrf(&in, t1_seconds, degree, &orientation,
                                                    tolerance, raw);
        });
    }

    [[nodiscard]] static result<ephemeris>
    geopotential_tide(const state &in, double t1_seconds, std::uint32_t degree,
                      const eop &orientation, double epoch_j2000_seconds, double mu_sun,
                      double mu_moon, double tolerance = 1e-9) {
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_geopotential_tide(&in, t1_seconds, degree, &orientation,
                                                    epoch_j2000_seconds, mu_sun, mu_moon, tolerance,
                                                    raw);
        });
    }

    [[nodiscard]] static result<ephemeris> j2_itrf(const state &in, double t1_seconds, double mu,
                                                   double j2, double earth_radius_m,
                                                   const eop &orientation,
                                                   double tolerance = 1e-9) {
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_j2_itrf(&in, t1_seconds, mu, j2, earth_radius_m, &orientation,
                                          tolerance, raw);
        });
    }

    [[nodiscard]] static result<ephemeris>
    j2_itrf_series(const state &in, double t1_seconds, double mu, double j2, double earth_radius_m,
                   std::int64_t epoch_utc_ns, double tt_minus_utc_s,
                   std::span<const eop_sample> series, double tolerance = 1e-9) {
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_j2_itrf_series(&in, t1_seconds, mu, j2, earth_radius_m,
                                                 epoch_utc_ns, tt_minus_utc_s, series.data(),
                                                 series.size(), tolerance, raw);
        });
    }

    [[nodiscard]] static result<ephemeris>
    luni_solar(const state &in, double t1_seconds, std::uint32_t degree, double epoch_j2000_seconds,
               double srp_reference_acceleration, double theta0_rad = 0.0,
               double tolerance = 1e-9) {
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_luni_solar(&in, t1_seconds, degree, epoch_j2000_seconds,
                                             srp_reference_acceleration, theta0_rad, tolerance,
                                             raw);
        });
    }

    [[nodiscard]] static result<ephemeris> relativistic(const state &in, double t1_seconds,
                                                        double epoch_j2000_seconds, double mu,
                                                        double mu_sun, double tolerance = 1e-9) {
        return build([&](::sl_ephemeris **raw) {
            return ::sl_ephemeris_relativistic(&in, t1_seconds, epoch_j2000_seconds, mu, mu_sun,
                                               tolerance, raw);
        });
    }

    [[nodiscard]] result<state> try_at(double t_seconds) const {
        state out{};
        const ::sl_status s = ::sl_ephemeris_at(handle_.get(), t_seconds, &out);
        return s == SL_OK ? result<state>(out) : detail::failure<state>(s);
    }

    [[nodiscard]] result<std::vector<state>> try_at(std::span<const double> times_seconds) const {
        if (times_seconds.empty()) {
            return std::vector<state>{};
        }
        std::vector<state> out(times_seconds.size());
        const ::sl_status s = ::sl_ephemeris_sample(handle_.get(), times_seconds.data(),
                                                    times_seconds.size(), out.data());
        return s == SL_OK ? result<std::vector<state>>(std::move(out))
                          : detail::failure<std::vector<state>>(s);
    }

    /* `[t0, t1]`, seconds from the start; `t1` is negative for a backwards arc. */
    [[nodiscard]] result<std::pair<double, double>> try_span() const {
        double t0 = 0.0;
        double t1 = 0.0;
        const ::sl_status s = ::sl_ephemeris_span(handle_.get(), &t0, &t1);
        return s == SL_OK ? result<std::pair<double, double>>(std::pair(t0, t1))
                          : detail::failure<std::pair<double, double>>(s);
    }

    /* `(accepted, rejected)` steps -- how hard the problem was. */
    [[nodiscard]] result<std::pair<std::uint32_t, std::uint32_t>> try_step_counts() const {
        std::uint32_t accepted = 0;
        std::uint32_t rejected = 0;
        const ::sl_status s = ::sl_ephemeris_step_counts(handle_.get(), &accepted, &rejected);
        using pair = std::pair<std::uint32_t, std::uint32_t>;
        return s == SL_OK ? result<pair>(pair(accepted, rejected)) : detail::failure<pair>(s);
    }

#if defined(__cpp_exceptions)
    [[nodiscard]] state at(double t_seconds) const {
        return detail::unwrap(try_at(t_seconds));
    }

    [[nodiscard]] std::vector<state> at(std::span<const double> times_seconds) const {
        return detail::unwrap(try_at(times_seconds));
    }

    [[nodiscard]] std::pair<double, double> span() const {
        return detail::unwrap(try_span());
    }

    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> step_counts() const {
        return detail::unwrap(try_step_counts());
    }
#endif

  private:
    /* Every constructor above differs only in the C call it makes; the raw pointer never escapes
     * this function, so a failure cannot leave a caller holding one. */
    template <class Call> [[nodiscard]] static result<ephemeris> build(Call call) {
        ::sl_ephemeris *raw = nullptr;
        const ::sl_status s = call(&raw);
        return s == SL_OK ? result<ephemeris>(ephemeris(raw)) : detail::failure<ephemeris>(s);
    }

    explicit ephemeris(::sl_ephemeris *raw) : handle_(raw) {}

    struct closer {
        void operator()(::sl_ephemeris *p) const noexcept {
            ::sl_ephemeris_close(p);
        }
    };

    std::unique_ptr<::sl_ephemeris, closer> handle_;
};

/*
 * Fixed-step 8th-order symplectic propagation under point-mass gravity plus J2, in `f64` or in
 * double-double. Run it twice and the difference is the round-off the ordinary answer carries.
 */
[[nodiscard]] inline result<state>
try_propagate_j2_precision(const state &in, double mu, double j2, double earth_radius_m,
                           double dt_seconds, std::uint64_t steps, precision arithmetic) {
    state out{};
    const ::sl_status s = ::sl_propagate_j2_precision(&in, mu, j2, earth_radius_m, dt_seconds,
                                                      steps, arithmetic, &out);
    return s == SL_OK ? result<state>(out) : detail::failure<state>(s);
}

[[nodiscard]] inline result<vec3> try_geopotential_acceleration(std::uint32_t degree,
                                                                const vec3 &r_body_fixed) {
    vec3 out{};
    const ::sl_status s = ::sl_geopotential_acceleration(degree, &r_body_fixed, &out);
    return s == SL_OK ? result<vec3>(out) : detail::failure<vec3>(s);
}

/* ---- Three-body motion ---- */

/* L1, L2, L3, L4, L5 in synodic nondimensional coordinates. Earth-Moon is the default ratio. */
[[nodiscard]] inline result<std::array<vec3, 5>>
try_libration_points(double mass_ratio = 0.012150585) {
    std::array<vec3, 5> out{};
    const ::sl_status s = ::sl_libration_points(mass_ratio, out.data());
    using five = std::array<vec3, 5>;
    return s == SL_OK ? result<five>(out) : detail::failure<five>(s);
}

[[nodiscard]] inline result<state> try_richardson_halo(std::uint32_t li, double amplitude_z,
                                                       bool northern = true,
                                                       double mass_ratio = 0.012150585) {
    state out{};
    const ::sl_status s = ::sl_richardson_halo(li, amplitude_z, northern, mass_ratio, &out);
    return s == SL_OK ? result<state>(out) : detail::failure<state>(s);
}

#if defined(__cpp_exceptions)

[[nodiscard]] inline vec3 geopotential_acceleration(std::uint32_t degree,
                                                    const vec3 &r_body_fixed) {
    return detail::unwrap(try_geopotential_acceleration(degree, r_body_fixed));
}

[[nodiscard]] inline std::array<vec3, 5> libration_points(double mass_ratio = 0.012150585) {
    return detail::unwrap(try_libration_points(mass_ratio));
}

[[nodiscard]] inline state richardson_halo(std::uint32_t li, double amplitude_z,
                                           bool northern = true, double mass_ratio = 0.012150585) {
    return detail::unwrap(try_richardson_halo(li, amplitude_z, northern, mass_ratio));
}

#endif

/* ---- Semi-analytical propagation (DSST) ---- */

/* The defaults `sl_dsst_params_init` writes. */
[[nodiscard]] inline dsst_params default_dsst_params() noexcept {
    dsst_params p{};
    ::sl_dsst_params_init(&p);
    return p;
}

namespace detail {

/* The four DSST entry points differ only in which one they are, so they share one body. */
using dsst_entry = ::sl_status (*)(const ::sl_dsst_params *, const double *, const double *,
                                   std::size_t, double *);

[[nodiscard]] inline result<std::vector<equinoctial>>
dsst_call(dsst_entry entry, const dsst_params &p, const equinoctial &initial,
          std::span<const double> times_seconds) {
    using many = std::vector<equinoctial>;
    if (times_seconds.empty()) {
        return many{};
    }
    many out(times_seconds.size());
    const ::sl_status s =
        entry(&p, initial.data(), times_seconds.data(), times_seconds.size(), out.data()->data());
    return s == SL_OK ? result<many>(std::move(out)) : detail::failure<many>(s);
}

} // namespace detail

/*
 * Mean and osculating DSST propagation; see `spacelab.h` for what the elements mean and what each
 * of the four includes. Asking for no epochs is an empty vector rather than a failure, as it is
 * for `sgp4::try_at`.
 */
[[nodiscard]] inline result<std::vector<equinoctial>>
try_dsst_zonal_mean(const dsst_params &p, const equinoctial &initial,
                    std::span<const double> times_seconds) {
    return detail::dsst_call(&::sl_dsst_zonal_mean, p, initial, times_seconds);
}

[[nodiscard]] inline result<std::vector<equinoctial>>
try_dsst_zonal_osculating(const dsst_params &p, const equinoctial &initial,
                          std::span<const double> times_seconds) {
    return detail::dsst_call(&::sl_dsst_zonal_osculating, p, initial, times_seconds);
}

[[nodiscard]] inline result<std::vector<equinoctial>>
try_dsst_mean(const dsst_params &p, const equinoctial &initial,
              std::span<const double> times_seconds) {
    return detail::dsst_call(&::sl_dsst_mean, p, initial, times_seconds);
}

[[nodiscard]] inline result<std::vector<equinoctial>>
try_dsst_osculating(const dsst_params &p, const equinoctial &initial,
                    std::span<const double> times_seconds) {
    return detail::dsst_call(&::sl_dsst_osculating, p, initial, times_seconds);
}

/* ---- Estimation ---- */

/* The defaults `sl_gauss_newton_options_init` writes. */
[[nodiscard]] inline gauss_newton_options default_gauss_newton_options() noexcept {
    gauss_newton_options o{};
    ::sl_gauss_newton_options_init(&o);
    return o;
}

namespace detail {

/*
 * What `sl_gauss_newton` is actually handed as `user_data`.
 *
 * The C callback is a plain function pointer, and a function template cannot have C language
 * linkage, so the caller's model is type-erased into `invoke` and one non-template trampoline does
 * the crossing. `failure` is why the trampoline exists at all rather than the model being passed
 * straight through: a C++ model that throws would otherwise unwind through Rust frames, which is
 * undefined behaviour and not a stack trace. The exception is caught, the solve is abandoned
 * through the status the C API already has for it, and it is rethrown on the way out — so the
 * caller sees their own exception, from their own `try`, at the point they called us.
 */
struct model_bridge {
    void *model;
    void (*invoke)(void *model, const double *state, ::sl_residual_row *rows, std::size_t n);
#if defined(__cpp_exceptions)
    std::exception_ptr failure;
#endif
};

extern "C" inline ::sl_status spacelab_residual_trampoline(void *user_data, const double *state,
                                                           ::sl_residual_row *rows,
                                                           std::size_t row_count) {
    auto *bridge = static_cast<model_bridge *>(user_data);
#if defined(__cpp_exceptions)
    try {
        bridge->invoke(bridge->model, state, rows, row_count);
    } catch (...) {
        bridge->failure = std::current_exception();
        return SL_ERR_COMPUTE;
    }
#else
    bridge->invoke(bridge->model, state, rows, row_count);
#endif
    return SL_OK;
}

} // namespace detail

/*
 * Gauss-Newton / Levenberg-Marquardt least squares over a six-state.
 *
 * `model` is anything callable as `model(std::span<const double, 6> state,
 * std::span<residual_row> rows)`: fill every row from `state`. It is called once per iterate, and
 * both spans are valid only for the duration of that call. Throwing from it is allowed and does
 * what you would want — the solve stops and the exception comes back out of this function — which
 * is not true of the raw C entry point.
 */
template <class Model>
[[nodiscard]] inline result<estimate>
try_gauss_newton(Model &&model, const std::array<double, 6> &x0, std::size_t row_count,
                 const gauss_newton_options &options = default_gauss_newton_options()) {
    using bare = std::remove_reference_t<Model>;
    detail::model_bridge bridge{};
    /* Round-tripped to `bare *` below, so a const model stays const and needs `operator() const`. */
    bridge.model = const_cast<void *>(static_cast<const void *>(std::addressof(model)));
    bridge.invoke = [](void *m, const double *state, ::sl_residual_row *rows, std::size_t n) {
        (*static_cast<bare *>(m))(std::span<const double, 6>(state, 6),
                                  std::span<::sl_residual_row>(rows, n));
    };

    estimate out{};
    const ::sl_status s = ::sl_gauss_newton(
        &options, x0.data(), &detail::spacelab_residual_trampoline, &bridge, row_count, &out);
#if defined(__cpp_exceptions)
    if (bridge.failure) {
        std::rethrow_exception(bridge.failure);
    }
#endif
    return s == SL_OK ? result<estimate>(out) : detail::failure<estimate>(s);
}

/*
 * A square-root information filter over a six-state; see `spacelab.h` for when to reach for it
 * rather than `try_gauss_newton`.
 */
class srif {
  public:
    /* An empty filter: no information, so `try_solve` fails until enough rows are folded in. */
    [[nodiscard]] static result<srif> open() {
        ::sl_srif *raw = nullptr;
        const ::sl_status s = ::sl_srif_open(nullptr, nullptr, &raw);
        return s == SL_OK ? result<srif>(srif(raw)) : detail::failure<srif>(s);
    }

    /* Seeded with a prior state and its information square root `chol(P0^-1)`, row-major. */
    [[nodiscard]] static result<srif> open(const std::array<double, 6> &x0,
                                           const std::array<double, 36> &information_sqrt) {
        ::sl_srif *raw = nullptr;
        const ::sl_status s = ::sl_srif_open(x0.data(), information_sqrt.data(), &raw);
        return s == SL_OK ? result<srif>(srif(raw)) : detail::failure<srif>(s);
    }

    /*
     * Fold one scalar measurement in. A status rather than a `result`, because there is no value
     * for one to carry -- the answer is the filter, and it is `this`.
     */
    ::sl_status try_add_row(std::span<const double, 6> jacobian, double z, double weight) {
        return ::sl_srif_add_row(handle_.get(), jacobian.data(), z, weight);
    }

    [[nodiscard]] result<std::array<double, 6>> try_solve() const {
        using six = std::array<double, 6>;
        six out{};
        const ::sl_status s = ::sl_srif_solve(handle_.get(), out.data());
        return s == SL_OK ? result<six>(out) : detail::failure<six>(s);
    }

    [[nodiscard]] result<std::array<double, 36>> try_covariance() const {
        using thirty_six = std::array<double, 36>;
        thirty_six out{};
        const ::sl_status s = ::sl_srif_covariance(handle_.get(), out.data());
        return s == SL_OK ? result<thirty_six>(out) : detail::failure<thirty_six>(s);
    }

    /* `(post-fit sum of squares, rows folded in)`. */
    [[nodiscard]] result<std::pair<double, std::uint32_t>> try_residual_stats() const {
        using pair = std::pair<double, std::uint32_t>;
        double sum_of_squares = 0.0;
        std::uint32_t rows = 0;
        const ::sl_status s = ::sl_srif_residual_stats(handle_.get(), &sum_of_squares, &rows);
        return s == SL_OK ? result<pair>(pair(sum_of_squares, rows)) : detail::failure<pair>(s);
    }

#if defined(__cpp_exceptions)
    [[nodiscard]] std::array<double, 6> solve() const {
        return detail::unwrap(try_solve());
    }

    [[nodiscard]] std::array<double, 36> covariance() const {
        return detail::unwrap(try_covariance());
    }

    [[nodiscard]] std::pair<double, std::uint32_t> residual_stats() const {
        return detail::unwrap(try_residual_stats());
    }

    void add_row(std::span<const double, 6> jacobian, double z, double weight) {
        const ::sl_status s = try_add_row(jacobian, z, weight);
        if (s != SL_OK) {
            throw error(s);
        }
    }
#endif

  private:
    explicit srif(::sl_srif *raw) : handle_(raw) {}

    struct closer {
        void operator()(::sl_srif *p) const noexcept {
            ::sl_srif_close(p);
        }
    };

    std::unique_ptr<::sl_srif, closer> handle_;
};

#if defined(__cpp_exceptions)

[[nodiscard]] inline std::vector<equinoctial>
dsst_zonal_mean(const dsst_params &p, const equinoctial &initial,
                std::span<const double> times_seconds) {
    return detail::unwrap(try_dsst_zonal_mean(p, initial, times_seconds));
}

[[nodiscard]] inline std::vector<equinoctial>
dsst_zonal_osculating(const dsst_params &p, const equinoctial &initial,
                      std::span<const double> times_seconds) {
    return detail::unwrap(try_dsst_zonal_osculating(p, initial, times_seconds));
}

[[nodiscard]] inline std::vector<equinoctial>
dsst_mean(const dsst_params &p, const equinoctial &initial, std::span<const double> times_seconds) {
    return detail::unwrap(try_dsst_mean(p, initial, times_seconds));
}

[[nodiscard]] inline std::vector<equinoctial>
dsst_osculating(const dsst_params &p, const equinoctial &initial,
                std::span<const double> times_seconds) {
    return detail::unwrap(try_dsst_osculating(p, initial, times_seconds));
}

template <class Model>
[[nodiscard]] inline estimate
gauss_newton(Model &&model, const std::array<double, 6> &x0, std::size_t row_count,
             const gauss_newton_options &options = default_gauss_newton_options()) {
    return detail::unwrap(try_gauss_newton(std::forward<Model>(model), x0, row_count, options));
}

#endif

/* ---- GNSS ---- */

/*
 * The two RINEX entry points take a `std::string_view`, so the length the C API wants comes from
 * the string rather than from a `strlen` the caller writes out. A view over a file read into a
 * `std::string` is the expected call; nothing here retains it past the call.
 */
[[nodiscard]] inline result<std::vector<gnss_state>> try_rinex_gnss_positions(std::string_view text,
                                                                              double tk_seconds) {
    return detail::sized<gnss_state>([&](gnss_state *out, std::size_t cap, std::size_t *len) {
        return ::sl_rinex_gnss_positions(text.data(), text.size(), tk_seconds, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<glonass_state>>
try_rinex_glonass_positions(std::string_view text, double dt_seconds) {
    return detail::sized<glonass_state>([&](glonass_state *out, std::size_t cap, std::size_t *len) {
        return ::sl_rinex_glonass_positions(text.data(), text.size(), dt_seconds, out, cap, len);
    });
}

/*
 * The observation-file readers take the descriptor as a `std::string_view` too, and copy it into a
 * `std::string` because the C function takes a NUL-terminated token rather than a length. A view
 * carries no terminator and a descriptor is three characters, so the copy is both necessary and
 * inside every implementation's small-string buffer.
 */
[[nodiscard]] inline result<obs_header> try_rinex_obs_header(std::string_view text) {
    obs_header out{};
    const ::sl_status s = ::sl_rinex_obs_header(text.data(), text.size(), &out);
    return s == SL_OK ? result<obs_header>(out) : detail::failure<obs_header>(s);
}

[[nodiscard]] inline result<std::vector<obs_descriptor>>
try_rinex_obs_descriptors(std::string_view text, char system) {
    return detail::sized<obs_descriptor>(
        [&](obs_descriptor *out, std::size_t cap, std::size_t *len) {
            return ::sl_rinex_obs_descriptors(text.data(), text.size(), system, out, cap, len);
        });
}

[[nodiscard]] inline result<std::vector<obs_sample>>
try_rinex_obs_series(std::string_view text, char system, std::uint32_t prn, std::string_view code) {
    std::string owned(code);
    return detail::sized<obs_sample>([&](obs_sample *out, std::size_t cap, std::size_t *len) {
        return ::sl_rinex_obs_series(text.data(), text.size(), system, prn, owned.c_str(), out, cap,
                                     len);
    });
}

[[nodiscard]] inline double glonass_l1_frequency(std::int32_t channel) noexcept {
    return ::sl_glonass_l1_frequency(channel);
}

[[nodiscard]] inline double glonass_l2_frequency(std::int32_t channel) noexcept {
    return ::sl_glonass_l2_frequency(channel);
}

[[nodiscard]] inline result<dop> try_dilution_of_precision(const vec3 &receiver,
                                                           std::span<const vec3> satellites) {
    dop out{};
    const ::sl_status s =
        ::sl_dilution_of_precision(&receiver, satellites.data(), satellites.size(), &out);
    return s == SL_OK ? result<dop>(out) : detail::failure<dop>(s);
}

[[nodiscard]] inline result<vec3> try_igrf_field_ecef(const vec3 &position_m, double year) {
    vec3 out{};
    const ::sl_status s = ::sl_igrf_field_ecef(&position_m, year, &out);
    return s == SL_OK ? result<vec3>(out) : detail::failure<vec3>(s);
}

/* ---- Ionosphere ---- */

[[nodiscard]] inline result<double>
try_klobuchar_iono_delay(const std::array<double, 4> &alpha, const std::array<double, 4> &beta,
                         double latitude_rad, double longitude_rad, double elevation_rad,
                         double azimuth_rad, double gps_seconds_of_day) {
    double out = 0.0;
    const ::sl_status s =
        ::sl_klobuchar_iono_delay(alpha.data(), beta.data(), latitude_rad, longitude_rad,
                                  elevation_rad, azimuth_rad, gps_seconds_of_day, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline double ionospheric_path_delay(double stec_tecu, double frequency_hz) noexcept {
    return ::sl_ionospheric_path_delay(stec_tecu, frequency_hz);
}

[[nodiscard]] inline double
second_order_ionospheric_delay(double latitude_rad, double longitude_rad, double elevation_rad,
                               double azimuth_rad, double stec_tecu, double frequency_hz,
                               double pierce_height_m = 350e3) noexcept {
    return ::sl_second_order_ionospheric_delay(latitude_rad, longitude_rad, elevation_rad,
                                               azimuth_rad, stec_tecu, frequency_hz,
                                               pierce_height_m);
}

[[nodiscard]] inline double ionospheric_mapping_factor(double elevation_rad,
                                                       double ionosphere_height_m = 350e3,
                                                       double earth_radius_m = re_earth) noexcept {
    return ::sl_ionospheric_mapping_factor(elevation_rad, ionosphere_height_m, earth_radius_m);
}

[[nodiscard]] inline result<double> try_nequick_galileo_stec(const std::array<double, 3> &alpha,
                                                             std::uint32_t month,
                                                             double seconds_in_utc_day,
                                                             const geodetic &receiver,
                                                             const geodetic &satellite) {
    double out = 0.0;
    const ::sl_status s = ::sl_nequick_galileo_stec(alpha.data(), month, seconds_in_utc_day,
                                                    &receiver, &satellite, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

/* ---- Troposphere ---- */

[[nodiscard]] inline double saastamoinen_zenith_hydrostatic(double pressure_hpa,
                                                            double latitude_rad,
                                                            double height_m) noexcept {
    return ::sl_saastamoinen_zenith_hydrostatic(pressure_hpa, latitude_rad, height_m);
}

[[nodiscard]] inline double saastamoinen_zenith_wet(double temperature_k,
                                                    double vapour_pressure_hpa) noexcept {
    return ::sl_saastamoinen_zenith_wet(temperature_k, vapour_pressure_hpa);
}

[[nodiscard]] inline result<mapping> try_niell_mapping(double latitude_rad, double height_m,
                                                       double day_of_year, double elevation_rad) {
    mapping out{};
    const ::sl_status s =
        ::sl_niell_mapping(latitude_rad, height_m, day_of_year, elevation_rad, &out);
    return s == SL_OK ? result<mapping>(out) : detail::failure<mapping>(s);
}

[[nodiscard]] inline result<mapping> try_gmf_mapping(double latitude_rad, double longitude_rad,
                                                     double height_m, double day_of_year,
                                                     double elevation_rad) {
    mapping out{};
    const ::sl_status s =
        ::sl_gmf_mapping(latitude_rad, longitude_rad, height_m, day_of_year, elevation_rad, &out);
    return s == SL_OK ? result<mapping>(out) : detail::failure<mapping>(s);
}

[[nodiscard]] inline result<mapping> try_vmf1_mapping(double ah, double aw, double latitude_rad,
                                                      double height_m, double day_of_year,
                                                      double elevation_rad) {
    mapping out{};
    const ::sl_status s =
        ::sl_vmf1_mapping(ah, aw, latitude_rad, height_m, day_of_year, elevation_rad, &out);
    return s == SL_OK ? result<mapping>(out) : detail::failure<mapping>(s);
}

[[nodiscard]] inline result<mapping> try_vmf3_mapping(double ah, double aw, double latitude_rad,
                                                      double longitude_rad, double height_m,
                                                      double day_of_year, double elevation_rad) {
    mapping out{};
    const ::sl_status s = ::sl_vmf3_mapping(ah, aw, latitude_rad, longitude_rad, height_m,
                                            day_of_year, elevation_rad, &out);
    return s == SL_OK ? result<mapping>(out) : detail::failure<mapping>(s);
}

/* ---- Atmospheres, tides, density calibration and space weather ---- */

[[nodiscard]] inline double nrlmsise00_density(std::int32_t doy, double sec, double altitude_km,
                                               double latitude_deg, double longitude_deg,
                                               double f107, double f107a, double ap) noexcept {
    return ::sl_nrlmsise00_density(doy, sec, altitude_km, latitude_deg, longitude_deg, f107, f107a,
                                   ap);
}

[[nodiscard]] inline double sdtm1_density(double doy, double sec, double altitude_km,
                                          double latitude_deg, double longitude_deg, double f107,
                                          double f107a, double storm) noexcept {
    return ::sl_sdtm1_density(doy, sec, altitude_km, latitude_deg, longitude_deg, f107, f107a,
                              storm);
}

[[nodiscard]] inline double sdtm1_exospheric_temperature(double doy, double sec, double altitude_km,
                                                         double latitude_deg, double longitude_deg,
                                                         double f107, double f107a,
                                                         double storm) noexcept {
    return ::sl_sdtm1_exospheric_temperature(doy, sec, altitude_km, latitude_deg, longitude_deg,
                                             f107, f107a, storm);
}

[[nodiscard]] inline double sdtm1_storm_response(std::span<const double> history,
                                                 double cadence_s = 10800.0) noexcept {
    return ::sl_sdtm1_storm_response(history.data(), history.size(), cadence_s);
}

[[nodiscard]] inline result<std::array<double, SL_SDTM1_INTERVAL>>
try_sdtm1_density_interval(double density, double f107, double f107a, double storm, double z) {
    using interval = std::array<double, SL_SDTM1_INTERVAL>;
    interval out{};
    const ::sl_status s = ::sl_sdtm1_density_interval(density, f107, f107a, storm, z, out.data());
    return s == SL_OK ? result<interval>(out) : detail::failure<interval>(s);
}

[[nodiscard]] inline result<std::array<double, SL_SDTM1_ENVELOPE>> try_sdtm1_envelope() {
    using envelope = std::array<double, SL_SDTM1_ENVELOPE>;
    envelope out{};
    const ::sl_status s = ::sl_sdtm1_envelope(out.data());
    return s == SL_OK ? result<envelope>(out) : detail::failure<envelope>(s);
}

[[nodiscard]] inline double sdtm1_solar_wind_density(double doy, double sec, double altitude_km,
                                                     double latitude_deg, double longitude_deg,
                                                     double f107, double f107a,
                                                     double storm) noexcept {
    return ::sl_sdtm1_solar_wind_density(doy, sec, altitude_km, latitude_deg, longitude_deg, f107,
                                         f107a, storm);
}

[[nodiscard]] inline double
sdtm1_solar_wind_exospheric_temperature(double doy, double sec, double altitude_km,
                                        double latitude_deg, double longitude_deg, double f107,
                                        double f107a, double storm) noexcept {
    return ::sl_sdtm1_solar_wind_exospheric_temperature(doy, sec, altitude_km, latitude_deg,
                                                        longitude_deg, f107, f107a, storm);
}

/* Four series of different lengths are not a history, and answer NaN as an invalid hour does. */
[[nodiscard]] inline double
sdtm1_solar_wind_storm_response(std::span<const double> ap60, std::span<const double> speed_km_s,
                                std::span<const double> by_nt,
                                std::span<const double> bz_nt) noexcept {
    if (speed_km_s.size() != ap60.size() || by_nt.size() != ap60.size() ||
        bz_nt.size() != ap60.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return ::sl_sdtm1_solar_wind_storm_response(ap60.data(), speed_km_s.data(), by_nt.data(),
                                                bz_nt.data(), ap60.size());
}

[[nodiscard]] inline result<std::array<double, SL_SDTM1_INTERVAL>>
try_sdtm1_solar_wind_density_interval(double density, double f107, double f107a, double storm,
                                      double z) {
    using interval = std::array<double, SL_SDTM1_INTERVAL>;
    interval out{};
    const ::sl_status s =
        ::sl_sdtm1_solar_wind_density_interval(density, f107, f107a, storm, z, out.data());
    return s == SL_OK ? result<interval>(out) : detail::failure<interval>(s);
}

[[nodiscard]] inline result<std::array<double, SL_SDTM1_ENVELOPE>> try_sdtm1_solar_wind_envelope() {
    using envelope = std::array<double, SL_SDTM1_ENVELOPE>;
    envelope out{};
    const ::sl_status s = ::sl_sdtm1_solar_wind_envelope(out.data());
    return s == SL_OK ? result<envelope>(out) : detail::failure<envelope>(s);
}

[[nodiscard]] inline result<std::vector<state>>
try_brouwer_lyddane_propagate(const brouwer_lyddane_params &p, std::span<const double> times) {
    std::vector<state> out(times.size());
    const ::sl_status s =
        ::sl_brouwer_lyddane_propagate(&p, times.data(), times.size(), out.data());
    return s == SL_OK ? result<std::vector<state>>(std::move(out))
                      : detail::failure<std::vector<state>>(s);
}

[[nodiscard]] inline result<std::array<double, 5>>
try_solid_tide_step2(double gmst_rad, const std::array<double, 5> &delaunay) {
    using five = std::array<double, 5>;
    five out{};
    const ::sl_status s = ::sl_solid_tide_step2(gmst_rad, delaunay.data(), out.data());
    return s == SL_OK ? result<five>(out) : detail::failure<five>(s);
}

[[nodiscard]] inline result<std::array<double, 2>>
try_solid_pole_tide(double xp_rad, double yp_rad, double t_seconds,
                    mean_pole pole = SL_MEAN_POLE_SECULAR2018) {
    using two = std::array<double, 2>;
    two out{};
    const ::sl_status s = ::sl_solid_pole_tide(xp_rad, yp_rad, t_seconds, pole, out.data());
    return s == SL_OK ? result<two>(out) : detail::failure<two>(s);
}

[[nodiscard]] inline result<std::array<double, 2>>
try_ocean_pole_tide(double xp_rad, double yp_rad, double t_seconds,
                    mean_pole pole = SL_MEAN_POLE_SECULAR2018) {
    using two = std::array<double, 2>;
    two out{};
    const ::sl_status s = ::sl_ocean_pole_tide(xp_rad, yp_rad, t_seconds, pole, out.data());
    return s == SL_OK ? result<two>(out) : detail::failure<two>(s);
}

[[nodiscard]] inline result<zonal> try_conventional_zonal_coefficients(double years_since_j2000) {
    zonal out{};
    const ::sl_status s = ::sl_conventional_zonal_coefficients(years_since_j2000, &out);
    return s == SL_OK ? result<zonal>(out) : detail::failure<zonal>(s);
}

[[nodiscard]] inline result<zonal> try_time_variable_zonal_correction(double years_since_j2000) {
    zonal out{};
    const ::sl_status s = ::sl_time_variable_zonal_correction(years_since_j2000, &out);
    return s == SL_OK ? result<zonal>(out) : detail::failure<zonal>(s);
}

[[nodiscard]] inline result<drag_panel>
try_sentman_drag_coefficients(double speed_ratio, double cos_incidence, double accommodation,
                              double wall_temperature_k, double atmosphere_temperature_k) {
    drag_panel out{};
    const ::sl_status s =
        ::sl_sentman_drag_coefficients(speed_ratio, cos_incidence, accommodation,
                                       wall_temperature_k, atmosphere_temperature_k, &out);
    return s == SL_OK ? result<drag_panel>(out) : detail::failure<drag_panel>(s);
}

[[nodiscard]] inline double molecular_speed_ratio(double relative_speed_ms, double temperature_k,
                                                  double mean_molar_mass) noexcept {
    return ::sl_molecular_speed_ratio(relative_speed_ms, temperature_k, mean_molar_mass);
}

[[nodiscard]] inline result<std::array<double, SL_DENSITY_DRIVERS>>
try_space_weather_drivers(double f107, double f107_81, double ap) {
    using drivers = std::array<double, SL_DENSITY_DRIVERS>;
    drivers out{};
    const ::sl_status s = ::sl_space_weather_drivers(f107, f107_81, ap, out.data());
    return s == SL_OK ? result<drivers>(out) : detail::failure<drivers>(s);
}

[[nodiscard]] inline result<std::array<double, SL_AP_ARRAY_LEN>>
try_space_weather_ap_array(std::string_view csv_text, std::int32_t year, std::uint32_t month,
                           std::uint32_t day, double seconds_of_day) {
    using ap_array = std::array<double, SL_AP_ARRAY_LEN>;
    ap_array out{};
    const ::sl_status s = ::sl_space_weather_ap_array(csv_text.data(), csv_text.size(), year, month,
                                                      day, seconds_of_day, out.data());
    return s == SL_OK ? result<ap_array>(out) : detail::failure<ap_array>(s);
}

[[nodiscard]] inline result<std::vector<space_weather_day>>
try_space_weather_days(std::string_view csv_text) {
    return detail::sized<space_weather_day>(
        [&](space_weather_day *out, std::size_t cap, std::size_t *len) {
            return ::sl_space_weather_days(csv_text.data(), csv_text.size(), out, cap, len);
        });
}

[[nodiscard]] inline result<std::vector<density_sample>>
try_density_data_samples(std::string_view text) {
    return detail::sized<density_sample>(
        [&](density_sample *out, std::size_t cap, std::size_t *len) {
            return ::sl_density_data_samples(text.data(), text.size(), out, cap, len);
        });
}

[[nodiscard]] inline result<std::vector<density_observation>>
try_density_altitude_normalized(std::string_view text, double reference_altitude_m,
                                double scale_height_m) {
    return detail::sized<density_observation>(
        [&](density_observation *out, std::size_t cap, std::size_t *len) {
            return ::sl_density_altitude_normalized(text.data(), text.size(), reference_altitude_m,
                                                    scale_height_m, out, cap, len);
        });
}

[[nodiscard]] inline result<std::vector<double>>
try_density_correction_fit(std::span<const density_sample> samples, double ridge) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_density_correction_fit(samples.data(), samples.size(), ridge, out, cap, len);
    });
}

[[nodiscard]] inline result<double>
try_density_correction_factor(std::span<const double> coefficients, double latitude_rad,
                              double local_solar_time_h) {
    double out = 0.0;
    const ::sl_status s = ::sl_density_correction_factor(coefficients.data(), coefficients.size(),
                                                         latitude_rad, local_solar_time_h, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<std::vector<double>>
try_density_grid_snapshot(std::span<const density_sample> samples, std::size_t n_lat,
                          std::size_t n_lst) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_density_grid_snapshot(samples.data(), samples.size(), n_lat, n_lst, out, cap,
                                          len);
    });
}

[[nodiscard]] inline result<swath> try_swath_geometry(double earth_radius_m, double altitude_m,
                                                      double look_angle_rad) {
    swath out{};
    const ::sl_status s = ::sl_swath_geometry(earth_radius_m, altitude_m, look_angle_rad, &out);
    return s == SL_OK ? result<swath>(out) : detail::failure<swath>(s);
}

/* ---- Synthetic aperture radar ---- */

[[nodiscard]] inline result<std::vector<complex>> try_lfm_chirp(std::size_t n, double chirp_rate,
                                                                double sample_rate) {
    return detail::sized<complex>([&](complex *out, std::size_t cap, std::size_t *len) {
        return ::sl_lfm_chirp(n, chirp_rate, sample_rate, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<complex>>
try_matched_filter(std::span<const complex> signal, std::span<const complex> replica) {
    return detail::sized<complex>([&](complex *out, std::size_t cap, std::size_t *len) {
        return ::sl_matched_filter(signal.data(), signal.size(), replica.data(), replica.size(),
                                   out, cap, len);
    });
}

[[nodiscard]] inline double range_migration(double f_eta, const rda_geometry &geometry) noexcept {
    return ::sl_range_migration(f_eta, &geometry);
}

[[nodiscard]] inline double azimuth_fm_rate(const rda_geometry &geometry) noexcept {
    return ::sl_azimuth_fm_rate(&geometry);
}

[[nodiscard]] inline result<double> try_src_fm_rate(double f_eta, const rda_geometry &geometry) {
    double out = 0.0;
    const ::sl_status s = ::sl_src_fm_rate(f_eta, &geometry, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<std::vector<complex>>
try_chirp_scaling_focus(std::span<const complex> raw, std::size_t na, std::size_t nr,
                        const csa_params &params) {
    return detail::sized<complex>([&](complex *out, std::size_t cap, std::size_t *len) {
        return ::sl_chirp_scaling_focus(raw.data(), na, nr, &params, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<complex>>
try_apply_chirp_scaling(std::span<const complex> line, double chirp_rate, double scaling,
                        double tau_ref, double sample_rate) {
    return detail::sized<complex>([&](complex *out, std::size_t cap, std::size_t *len) {
        return ::sl_apply_chirp_scaling(line.data(), line.size(), chirp_rate, scaling, tau_ref,
                                        sample_rate, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<complex>> try_interferogram(std::span<const complex> s1,
                                                                    std::span<const complex> s2) {
    if (s1.size() != s2.size()) {
        return detail::failure<std::vector<complex>>(SL_ERR_INVALID_ARGUMENT);
    }
    return detail::sized<complex>([&](complex *out, std::size_t cap, std::size_t *len) {
        return ::sl_interferogram(s1.data(), s2.data(), s1.size(), out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<double>> try_coherence(std::span<const complex> s1,
                                                               std::span<const complex> s2,
                                                               std::size_t rows, std::size_t cols,
                                                               std::size_t half_window) {
    if (s1.size() != s2.size()) {
        return detail::failure<std::vector<double>>(SL_ERR_INVALID_ARGUMENT);
    }
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_coherence(s1.data(), s2.data(), rows, cols, half_window, out, cap, len);
    });
}

[[nodiscard]] inline double wrap_phase(double phi_rad) noexcept {
    return ::sl_wrap_phase(phi_rad);
}

[[nodiscard]] inline result<std::vector<double>>
try_unwrap_phase_1d(std::span<const double> wrapped) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_unwrap_phase_1d(wrapped.data(), wrapped.size(), out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<double>>
try_unwrap_phase_2d(std::span<const double> wrapped, std::size_t rows, std::size_t cols) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_unwrap_phase_2d(wrapped.data(), rows, cols, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<double>>
try_unwrap_phase_2d_branch_cut(std::span<const double> wrapped, std::size_t rows,
                               std::size_t cols) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_unwrap_phase_2d_branch_cut(wrapped.data(), rows, cols, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<double>>
try_unwrap_phase_2d_lstsq(std::span<const double> wrapped, std::size_t rows, std::size_t cols) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_unwrap_phase_2d_lstsq(wrapped.data(), rows, cols, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<std::int8_t>>
try_residues(std::span<const double> wrapped, std::size_t rows, std::size_t cols) {
    return detail::sized<std::int8_t>([&](std::int8_t *out, std::size_t cap, std::size_t *len) {
        return ::sl_residues(wrapped.data(), rows, cols, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<double>>
try_capon_spectrum(std::span<const complex> covariance, std::span<const double> baselines,
                   std::span<const double> elevations, const tomo_params &params) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_capon_spectrum(covariance.data(), baselines.data(), baselines.size(),
                                   elevations.data(), elevations.size(), &params, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<double>>
try_bartlett_spectrum(std::span<const complex> covariance, std::span<const double> baselines,
                      std::span<const double> elevations, const tomo_params &params) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_bartlett_spectrum(covariance.data(), baselines.data(), baselines.size(),
                                      elevations.data(), elevations.size(), &params, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<scatterer>> try_sl1mmer(std::span<const complex> y,
                                                                std::span<const double> baselines,
                                                                std::span<const double> elevations,
                                                                const tomo_params &params) {
    return detail::sized<scatterer>([&](scatterer *out, std::size_t cap, std::size_t *len) {
        return ::sl_sl1mmer(y.data(), baselines.data(), baselines.size(), elevations.data(),
                            elevations.size(), &params, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<double>>
try_sl1mmer_spectrum(std::span<const complex> y, std::span<const double> baselines,
                     std::span<const double> elevations, const tomo_params &params) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_sl1mmer_spectrum(y.data(), baselines.data(), baselines.size(),
                                     elevations.data(), elevations.size(), &params, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<double>>
try_amplitude_dispersion(std::span<const double> stack, std::size_t epochs, std::size_t pixels) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_amplitude_dispersion(stack.data(), epochs, pixels, out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<std::size_t>>
try_select_persistent_scatterers(std::span<const double> dispersion, double threshold) {
    return detail::sized<std::size_t>([&](std::size_t *out, std::size_t cap, std::size_t *len) {
        return ::sl_select_persistent_scatterers(dispersion.data(), dispersion.size(), threshold,
                                                 out, cap, len);
    });
}

[[nodiscard]] inline result<deformation> try_sbas_invert(std::span<const double> phases,
                                                         std::span<const double> temporal_baselines,
                                                         std::span<const double> perp_baselines,
                                                         double wavelength, double slant_range,
                                                         double incidence_rad) {
    if (phases.size() != temporal_baselines.size() || phases.size() != perp_baselines.size()) {
        return detail::failure<deformation>(SL_ERR_INVALID_ARGUMENT);
    }
    deformation out{};
    const ::sl_status s =
        ::sl_sbas_invert(phases.data(), temporal_baselines.data(), perp_baselines.data(),
                         phases.size(), wavelength, slant_range, incidence_rad, &out);
    return s == SL_OK ? result<deformation>(out) : detail::failure<deformation>(s);
}

[[nodiscard]] inline result<std::vector<complex>>
try_coherency_matrix(std::span<const polarimetric_sample> samples) {
    return detail::sized<complex>([&](complex *out, std::size_t cap, std::size_t *len) {
        return ::sl_coherency_matrix(samples.data(), samples.size(), out, cap, len);
    });
}

[[nodiscard]] inline result<cloude_pottier>
try_cloude_pottier_decompose(std::span<const complex> t3) {
    if (t3.size() != SL_COHERENCY_ENTRIES) {
        return detail::failure<cloude_pottier>(SL_ERR_INVALID_ARGUMENT);
    }
    cloude_pottier out{};
    const ::sl_status s = ::sl_cloude_pottier_decompose(t3.data(), &out);
    return s == SL_OK ? result<cloude_pottier>(out) : detail::failure<cloude_pottier>(s);
}

[[nodiscard]] inline result<std::array<double, 3>>
try_pauli_powers(const polarimetric_sample &sample) {
    using powers = std::array<double, 3>;
    powers out{};
    const ::sl_status s = ::sl_pauli_powers(&sample, out.data());
    return s == SL_OK ? result<powers>(out) : detail::failure<powers>(s);
}

[[nodiscard]] inline result<double> try_resolution_3db_samples(std::span<const double> magnitude) {
    double out = 0.0;
    const ::sl_status s = ::sl_resolution_3db_samples(magnitude.data(), magnitude.size(), &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<double> try_peak_sidelobe_ratio_db(std::span<const double> magnitude) {
    double out = 0.0;
    const ::sl_status s = ::sl_peak_sidelobe_ratio_db(magnitude.data(), magnitude.size(), &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

/* ---- Optical Earth observation ---- */

[[nodiscard]] inline result<std::vector<double>>
try_semi_global_match(std::span<const double> left, std::span<const double> right, std::size_t rows,
                      std::size_t cols, const sgm_config &config) {
    if (left.size() != right.size()) {
        return detail::failure<std::vector<double>>(SL_ERR_INVALID_ARGUMENT);
    }
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_semi_global_match(left.data(), right.data(), rows, cols, &config, out, cap,
                                      len);
    });
}

[[nodiscard]] inline result<std::vector<double>>
try_plane_sweep_mvs(const mvs_view &reference, std::span<const mvs_view> neighbors,
                    std::size_t rows, std::size_t cols, std::span<const double> depths,
                    std::size_t patch_radius) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_plane_sweep_mvs(&reference, neighbors.data(), neighbors.size(), rows, cols,
                                    depths.data(), depths.size(), patch_radius, out, cap, len);
    });
}

[[nodiscard]] inline result<vec3> try_triangulate(std::span<const camera> cameras,
                                                  std::span<const double> pixels) {
    if (pixels.size() != 2 * cameras.size()) {
        return detail::failure<vec3>(SL_ERR_INVALID_ARGUMENT);
    }
    vec3 out{};
    const ::sl_status s = ::sl_triangulate(cameras.data(), pixels.data(), cameras.size(), &out);
    return s == SL_OK ? result<vec3>(out) : detail::failure<vec3>(s);
}

[[nodiscard]] inline result<std::vector<track_point>>
try_sfm_triangulate(std::span<const corner> corners, std::span<const std::size_t> corners_per_view,
                    std::span<const feature_match> matches, std::span<const camera> cameras) {
    if (cameras.size() != corners_per_view.size()) {
        return detail::failure<std::vector<track_point>>(SL_ERR_INVALID_ARGUMENT);
    }
    return detail::sized<track_point>([&](track_point *out, std::size_t cap, std::size_t *len) {
        return ::sl_sfm_triangulate(corners.data(), corners_per_view.data(),
                                    corners_per_view.size(), matches.data(), matches.size(),
                                    cameras.data(), out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<corner>> try_harris_corners(std::span<const double> image,
                                                                    std::size_t rows,
                                                                    std::size_t cols,
                                                                    const harris_config &config) {
    return detail::sized<corner>([&](corner *out, std::size_t cap, std::size_t *len) {
        return ::sl_harris_corners(image.data(), rows, cols, &config, out, cap, len);
    });
}

[[nodiscard]] inline result<image_point>
try_rpc_block_adjust(std::string_view rpb_text, std::span<const ground_control_point> gcps,
                     double latitude, double longitude, double height, bool affine) {
    image_point out{};
    const ::sl_status s =
        ::sl_rpc_block_adjust(rpb_text.data(), rpb_text.size(), gcps.data(), gcps.size(), latitude,
                              longitude, height, affine, &out);
    return s == SL_OK ? result<image_point>(out) : detail::failure<image_point>(s);
}

[[nodiscard]] inline double ross_thick_kernel(double theta_s, double theta_v,
                                              double rel_azimuth) noexcept {
    return ::sl_ross_thick_kernel(theta_s, theta_v, rel_azimuth);
}

[[nodiscard]] inline double li_sparse_reciprocal_kernel(double theta_s, double theta_v,
                                                        double rel_azimuth) noexcept {
    return ::sl_li_sparse_reciprocal_kernel(theta_s, theta_v, rel_azimuth);
}

[[nodiscard]] inline double brdf_reflectance(double f_iso, double f_vol, double f_geo,
                                             double theta_s, double theta_v,
                                             double rel_azimuth) noexcept {
    return ::sl_brdf_reflectance(f_iso, f_vol, f_geo, theta_s, theta_v, rel_azimuth);
}

[[nodiscard]] inline double toa_from_surface(const atmospheric_coupling &coupling,
                                             double surface_reflectance) noexcept {
    return ::sl_atmospheric_toa_from_surface(&coupling, surface_reflectance);
}

[[nodiscard]] inline double surface_from_toa(const atmospheric_coupling &coupling,
                                             double toa_reflectance) noexcept {
    return ::sl_atmospheric_surface_from_toa(&coupling, toa_reflectance);
}

[[nodiscard]] inline result<std::vector<double>>
try_brovey_pan_sharpen(std::span<const double> ms_bands, std::size_t band_count,
                       std::span<const double> pan) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_brovey_pan_sharpen(ms_bands.data(), band_count, pan.size(), pan.data(), out,
                                       cap, len);
    });
}

[[nodiscard]] inline result<std::vector<double>>
try_gram_schmidt_pan_sharpen(std::span<const double> ms_bands, std::size_t band_count,
                             std::span<const double> pan) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_gram_schmidt_pan_sharpen(ms_bands.data(), band_count, pan.size(), pan.data(),
                                             out, cap, len);
    });
}

[[nodiscard]] inline result<std::vector<std::size_t>>
try_nms(std::span<const bbox> boxes, std::span<const double> scores, double iou_threshold) {
    if (boxes.size() != scores.size()) {
        return detail::failure<std::vector<std::size_t>>(SL_ERR_INVALID_ARGUMENT);
    }
    return detail::sized<std::size_t>([&](std::size_t *out, std::size_t cap, std::size_t *len) {
        return ::sl_nms(boxes.data(), scores.data(), boxes.size(), iou_threshold, out, cap, len);
    });
}

[[nodiscard]] inline double box_iou(const bbox &a, const bbox &b) noexcept {
    return ::sl_box_iou(&a, &b);
}

[[nodiscard]] inline result<bbox> try_decode_box(const bbox &anchor,
                                                 const std::array<double, 4> &delta) {
    bbox out{};
    const ::sl_status s = ::sl_decode_box(&anchor, delta.data(), &out);
    return s == SL_OK ? result<bbox>(out) : detail::failure<bbox>(s);
}

[[nodiscard]] inline result<std::vector<bbox>>
try_generate_anchors(std::size_t feat_rows, std::size_t feat_cols, double stride,
                     std::span<const double> scales, std::span<const double> aspect_ratios) {
    return detail::sized<bbox>([&](bbox *out, std::size_t cap, std::size_t *len) {
        return ::sl_generate_anchors(feat_rows, feat_cols, stride, scales.data(), scales.size(),
                                     aspect_ratios.data(), aspect_ratios.size(), out, cap, len);
    });
}

[[nodiscard]] inline result<gpt3_weather> try_gpt3(double latitude_rad, double longitude_rad,
                                                   double day_of_year, double height_m) {
    gpt3_weather out{};
    const ::sl_status s = ::sl_gpt3(latitude_rad, longitude_rad, day_of_year, height_m, &out);
    return s == SL_OK ? result<gpt3_weather>(out) : detail::failure<gpt3_weather>(s);
}

[[nodiscard]] inline result<vmf_coefficients>
try_vmf_grid_coefficients(std::string_view grid_text, double latitude_rad, double longitude_rad) {
    vmf_coefficients out{};
    const ::sl_status s = ::sl_vmf_grid_coefficients(grid_text.data(), grid_text.size(),
                                                     latitude_rad, longitude_rad, &out);
    return s == SL_OK ? result<vmf_coefficients>(out) : detail::failure<vmf_coefficients>(s);
}

/*
 * A loaded cloud-segmentation model. Same shape as `sgp4`: `open` parses, the destructor releases,
 * movable but not copyable because loading optimizes the graph and a copy would silently redo it.
 *
 * `open` returns `SL_ERR_UNSUPPORTED` when the library was built without the `onnx` feature. That
 * is the one call worth checking; a handle you hold is a model that loaded.
 */
class cloud_masker {
  public:
    [[nodiscard]] static result<cloud_masker> open(std::span<const std::uint8_t> onnx_bytes,
                                                   float threshold) {
        ::sl_cloud_masker *raw = nullptr;
        const ::sl_status s =
            ::sl_cloud_masker_open(onnx_bytes.data(), onnx_bytes.size(), threshold, &raw);
        return s == SL_OK ? result<cloud_masker>(cloud_masker(raw))
                          : detail::failure<cloud_masker>(s);
    }

    /*
     * `1` cloudy, `0` clear, over the model's own output grid.
     *
     * Not `detail::sized`, deliberately: that protocol calls twice, and here the sizing call runs
     * the network. The guess is `height * width`, which is right for every model that does not
     * upsample, and the retry costs a second inference only when it was wrong.
     */
    [[nodiscard]] result<std::vector<std::uint8_t>> try_mask(std::span<const float> tile,
                                                             std::size_t channels,
                                                             std::size_t height,
                                                             std::size_t width) const {
        std::vector<std::uint8_t> out(height * width);
        std::size_t len = 0;
        ::sl_status s = ::sl_cloud_masker_mask(handle_.get(), tile.data(), channels, height, width,
                                               out.data(), out.size(), &len);
        if (s == SL_ERR_BUFFER_TOO_SMALL) {
            out.resize(len);
            s = ::sl_cloud_masker_mask(handle_.get(), tile.data(), channels, height, width,
                                       out.data(), out.size(), &len);
        }
        if (s != SL_OK) {
            return detail::failure<std::vector<std::uint8_t>>(s);
        }
        out.resize(len);
        return result<std::vector<std::uint8_t>>(std::move(out));
    }

#if defined(__cpp_exceptions)
    explicit cloud_masker(std::span<const std::uint8_t> onnx_bytes, float threshold)
        : cloud_masker(detail::unwrap(open(onnx_bytes, threshold))) {}

    [[nodiscard]] std::vector<std::uint8_t> mask(std::span<const float> tile, std::size_t channels,
                                                 std::size_t height, std::size_t width) const {
        return detail::unwrap(try_mask(tile, channels, height, width));
    }
#endif

  private:
    explicit cloud_masker(::sl_cloud_masker *raw) : handle_(raw) {}

    struct closer {
        void operator()(::sl_cloud_masker *p) const noexcept {
            ::sl_cloud_masker_close(p);
        }
    };

    std::unique_ptr<::sl_cloud_masker, closer> handle_;
};

namespace detail {

/*
 * What the two conjunction scans are handed as `user_data`.
 *
 * Same shape as `model_bridge` and for the same reasons: a function template cannot have C language
 * linkage, so the caller's sampler is type-erased into `invoke` and one non-template trampoline per
 * callback signature does the crossing. A C++ sampler that threw would otherwise unwind through
 * Rust frames, which is undefined behaviour and not a stack trace, so the exception is caught, the
 * scan is abandoned through the status the C API already has for it, and it is rethrown on the way
 * out -- at the point the caller called us, from their own `try`.
 *
 * `invoke` returns `bool` rather than `void`: a sampler that cannot produce a state at `t` is an
 * ordinary answer for a trajectory that is only known over part of the span, and saying so is not
 * the same as throwing.
 */
struct relative_state_bridge {
    void *sampler;
    bool (*invoke)(void *sampler, double t, ::sl_state *out);
#if defined(__cpp_exceptions)
    std::exception_ptr failure;
#endif
};

struct relative_sample_bridge {
    void *sampler;
    bool (*invoke)(void *sampler, double t, ::sl_relative_sample *out);
#if defined(__cpp_exceptions)
    std::exception_ptr failure;
#endif
};

extern "C" inline ::sl_status spacelab_relative_state_trampoline(void *user_data, double t,
                                                                 ::sl_state *out) {
    auto *bridge = static_cast<relative_state_bridge *>(user_data);
#if defined(__cpp_exceptions)
    try {
        return bridge->invoke(bridge->sampler, t, out) ? SL_OK : SL_ERR_COMPUTE;
    } catch (...) {
        bridge->failure = std::current_exception();
        return SL_ERR_COMPUTE;
    }
#else
    return bridge->invoke(bridge->sampler, t, out) ? SL_OK : SL_ERR_COMPUTE;
#endif
}

extern "C" inline ::sl_status spacelab_relative_sample_trampoline(void *user_data, double t,
                                                                  ::sl_relative_sample *out) {
    auto *bridge = static_cast<relative_sample_bridge *>(user_data);
#if defined(__cpp_exceptions)
    try {
        return bridge->invoke(bridge->sampler, t, out) ? SL_OK : SL_ERR_COMPUTE;
    } catch (...) {
        bridge->failure = std::current_exception();
        return SL_ERR_COMPUTE;
    }
#else
    return bridge->invoke(bridge->sampler, t, out) ? SL_OK : SL_ERR_COMPUTE;
#endif
}

/* The one-`double`-out shape five of the observables and four of the combinations share. */
template <class Call> [[nodiscard]] inline result<double> one_double(Call call) {
    double out = 0.0;
    const ::sl_status s = call(&out);
    return s == SL_OK ? result<double>(out) : failure<double>(s);
}

} // namespace detail

/* The defaults `sl_decay_params_init` writes, as a value a caller can copy and amend. */
[[nodiscard]] inline decay_params default_decay_params() {
    decay_params p{};
    ::sl_decay_params_init(&p);
    return p;
}

[[nodiscard]] inline result<double> try_bistatic_range(const state &satellite,
                                                       const terminal &transmitter, double bounce,
                                                       const terminal &receiver) {
    return detail::one_double([&](double *out) {
        return ::sl_bistatic_range(&satellite, &transmitter, bounce, &receiver, out);
    });
}

[[nodiscard]] inline result<double> try_bistatic_range_rate(const state &satellite,
                                                            const terminal &transmitter,
                                                            double bounce,
                                                            const terminal &receiver) {
    return detail::one_double([&](double *out) {
        return ::sl_bistatic_range_rate(&satellite, &transmitter, bounce, &receiver, out);
    });
}

[[nodiscard]] inline result<double> try_tdoa(const state &satellite, double emission,
                                             const terminal &first, const terminal &second) {
    return detail::one_double(
        [&](double *out) { return ::sl_tdoa(&satellite, emission, &first, &second, out); });
}

[[nodiscard]] inline result<double> try_fdoa(const state &satellite, double emission,
                                             const terminal &first, const terminal &second,
                                             double frequency) {
    return detail::one_double([&](double *out) {
        return ::sl_fdoa(&satellite, emission, &first, &second, frequency, out);
    });
}

[[nodiscard]] inline result<double> try_delta_dor(const state &satellite, double emission,
                                                  const terminal &first, const terminal &second,
                                                  double right_ascension, double declination) {
    return detail::one_double([&](double *out) {
        return ::sl_delta_dor(&satellite, emission, &first, &second, right_ascension, declination,
                              out);
    });
}

[[nodiscard]] inline double inter_satellite_range(const state &local, const state &remote) {
    return ::sl_inter_satellite_range(&local, &remote);
}

[[nodiscard]] inline double inter_satellite_range_rate(const state &local, const state &remote) {
    return ::sl_inter_satellite_range_rate(&local, &remote);
}

[[nodiscard]] inline double shapiro_delay(double r_emitter, double r_receiver, double path,
                                          double gm) {
    return ::sl_shapiro_delay(r_emitter, r_receiver, path, gm);
}

[[nodiscard]] inline result<double>
try_dsn_sequential_range(std::span<const ramp_segment> segments, band uplink, double transmit,
                         double receive, std::int32_t lowest_ranging_component) {
    return detail::one_double([&](double *out) {
        return ::sl_dsn_sequential_range(segments.data(), segments.size(), uplink, transmit,
                                         receive, lowest_ranging_component, out);
    });
}

[[nodiscard]] inline result<double> try_dsn_averaged_doppler(std::span<const ramp_segment> segments,
                                                             const doppler_link &link,
                                                             double transmit_start,
                                                             double transmit_end) {
    return detail::one_double([&](double *out) {
        return ::sl_dsn_averaged_doppler(segments.data(), segments.size(), &link, transmit_start,
                                         transmit_end, out);
    });
}

[[nodiscard]] inline result<double>
try_dsn_measured_frequency(std::span<const ramp_segment> segments, band uplink, band downlink,
                           double transmit_epoch, double transmit_rate) {
    return detail::one_double([&](double *out) {
        return ::sl_dsn_measured_frequency(segments.data(), segments.size(), uplink, downlink,
                                           transmit_epoch, transmit_rate, out);
    });
}

/* An empty `series` takes Aksim & Pavlov (2022), which is what a null pointer means to the C API. */
[[nodiscard]] inline result<double>
try_solar_corona_delay(const vec3 &transmitter_from_sun, const vec3 &receiver_from_sun,
                       double frequency, bool group_delay = true,
                       std::span<const corona_term> series = {}) {
    return detail::one_double([&](double *out) {
        return ::sl_solar_corona_delay(&transmitter_from_sun, &receiver_from_sun, frequency,
                                       group_delay, series.empty() ? nullptr : series.data(),
                                       series.size(), out);
    });
}

[[nodiscard]] inline result<lifetime> try_orbital_lifetime(const decay_params &params,
                                                           const std::array<double, 6> &initial,
                                                           std::span<const double> zonals) {
    lifetime out{};
    const ::sl_status s =
        ::sl_orbital_lifetime(&params, initial.data(), zonals.data(), zonals.size(), &out);
    return s == SL_OK ? result<lifetime>(out) : detail::failure<lifetime>(s);
}

/*
 * The ensemble's quantiles and how its members ended, together -- the counts are what say how much
 * of the distribution the quantiles are conditioned on, so returning them apart would invite
 * reading one without the other.
 */
struct lifetime_quantiles {
    std::vector<double> seconds;
    lifetime_ensemble ensemble;
};

[[nodiscard]] inline result<lifetime_quantiles>
try_orbital_lifetime_ensemble(const decay_params &params, const std::array<double, 6> &initial,
                              std::span<const double> zonals,
                              std::span<const double> density_scales,
                              std::span<const double> quantiles) {
    lifetime_quantiles out{std::vector<double>(quantiles.size()), lifetime_ensemble{}};
    const ::sl_status s = ::sl_orbital_lifetime_ensemble(
        &params, initial.data(), zonals.data(), zonals.size(), density_scales.data(),
        density_scales.size(), quantiles.data(), quantiles.size(), out.seconds.data(),
        &out.ensemble);
    return s == SL_OK ? result<lifetime_quantiles>(std::move(out))
                      : detail::failure<lifetime_quantiles>(s);
}

[[nodiscard]] inline result<double>
try_ballistic_log_sensitivity(const decay_params &params, const std::array<double, 6> &initial,
                              std::span<const double> zonals, double fraction = 0.01) {
    return detail::one_double([&](double *out) {
        return ::sl_ballistic_log_sensitivity(&params, initial.data(), zonals.data(), zonals.size(),
                                              fraction, out);
    });
}

/* The two lines as `std::string`, which is what a caller writes to a file. */
struct tle_lines {
    std::string line1;
    std::string line2;
};

[[nodiscard]] inline result<tle_lines>
try_tle_lines(const utc_datetime &epoch, std::uint32_t norad_id,
              const std::array<double, 6> &parameters, double drag_term = 0.0,
              std::string_view international_designator = {}, std::uint64_t element_set_number = 0,
              std::uint64_t revolution_number = 0) {
    /* `std::string_view` is not NUL-terminated; the C API takes a C string or a null. */
    const std::string designator(international_designator);
    ::sl_tle raw{};
    const ::sl_status s =
        ::sl_tle_lines(&epoch, norad_id, parameters.data(), drag_term,
                       international_designator.empty() ? nullptr : designator.c_str(),
                       element_set_number, revolution_number, &raw);
    return s == SL_OK ? result<tle_lines>(tle_lines{std::string(raw.line1), std::string(raw.line2)})
                      : detail::failure<tle_lines>(s);
}

[[nodiscard]] inline double carrier_frequency(char system, char band_letter) {
    return ::sl_carrier_frequency(system, band_letter);
}

[[nodiscard]] inline result<lanes> try_lane_wavelengths(double f1, double f2) {
    lanes out{};
    const ::sl_status s = ::sl_lane_wavelengths(f1, f2, &out);
    return s == SL_OK ? result<lanes>(out) : detail::failure<lanes>(s);
}

[[nodiscard]] inline result<double> try_ionosphere_free(double f1, double f2, double x1,
                                                        double x2) {
    return detail::one_double(
        [&](double *out) { return ::sl_ionosphere_free(f1, f2, x1, x2, out); });
}

[[nodiscard]] inline result<double> try_wide_lane(double f1, double f2, double x1, double x2) {
    return detail::one_double([&](double *out) { return ::sl_wide_lane(f1, f2, x1, x2, out); });
}

[[nodiscard]] inline result<double> try_narrow_lane(double f1, double f2, double x1, double x2) {
    return detail::one_double([&](double *out) { return ::sl_narrow_lane(f1, f2, x1, x2, out); });
}

[[nodiscard]] inline result<double> try_melbourne_wubbena(double f1, double f2, double p1,
                                                          double p2, double l1, double l2,
                                                          bool cycles = false) {
    return detail::one_double(
        [&](double *out) { return ::sl_melbourne_wubbena(f1, f2, p1, p2, l1, l2, cycles, out); });
}

[[nodiscard]] inline double geometry_free(double x1, double x2) {
    return ::sl_geometry_free(x1, x2);
}

[[nodiscard]] inline double graphic(double p, double l) {
    return ::sl_graphic(p, l);
}

/*
 * The two detectors write one verdict per sample, so the vector is sized from the input rather than
 * through the two-call protocol -- there is nothing for a sizing call to discover.
 */
[[nodiscard]] inline result<std::vector<slip_verdict>>
try_detect_cycle_slips_geometry_free(std::span<const double> times, std::span<const double> values,
                                     double threshold, double max_gap) {
    using many = std::vector<slip_verdict>;
    if (times.size() != values.size()) {
        return detail::failure<many>(SL_ERR_INVALID_ARGUMENT);
    }
    many out(times.size());
    const ::sl_status s = ::sl_detect_cycle_slips_geometry_free(
        times.data(), values.data(), times.size(), threshold, max_gap, out.data());
    return s == SL_OK ? result<many>(std::move(out)) : detail::failure<many>(s);
}

[[nodiscard]] inline result<std::vector<slip_verdict>>
try_detect_cycle_slips_polynomial(std::span<const double> times, std::span<const double> values,
                                  std::size_t order, std::size_t window, double threshold,
                                  double max_gap) {
    using many = std::vector<slip_verdict>;
    if (times.size() != values.size()) {
        return detail::failure<many>(SL_ERR_INVALID_ARGUMENT);
    }
    many out(times.size());
    const ::sl_status s = ::sl_detect_cycle_slips_polynomial(
        times.data(), values.data(), times.size(), order, window, threshold, max_gap, out.data());
    return s == SL_OK ? result<many>(std::move(out)) : detail::failure<many>(s);
}

/* The integer transformation `Z` and the conditional variances, row-major and `n` long. */
struct decorrelation {
    std::vector<double> transformation;
    std::vector<double> conditional_variances;
};

[[nodiscard]] inline result<decorrelation>
try_ambiguity_decorrelation(std::span<const double> float_ambiguities,
                            std::span<const double> covariance) {
    const std::size_t n = float_ambiguities.size();
    if (n == 0 || covariance.size() != n * n) {
        return detail::failure<decorrelation>(SL_ERR_INVALID_ARGUMENT);
    }
    decorrelation out{std::vector<double>(n * n), std::vector<double>(n)};
    const ::sl_status s =
        ::sl_ambiguity_decorrelation(float_ambiguities.data(), n, covariance.data(),
                                     out.transformation.data(), out.conditional_variances.data());
    return s == SL_OK ? result<decorrelation>(std::move(out)) : detail::failure<decorrelation>(s);
}

/* One integer ambiguity vector and the quadratic form it attains. */
struct candidate {
    std::vector<std::int64_t> ambiguities;
    double squared_norm;
};

[[nodiscard]] inline result<candidate>
try_ambiguity_bootstrap(std::span<const double> float_ambiguities,
                        std::span<const double> covariance) {
    const std::size_t n = float_ambiguities.size();
    if (n == 0 || covariance.size() != n * n) {
        return detail::failure<candidate>(SL_ERR_INVALID_ARGUMENT);
    }
    candidate out{std::vector<std::int64_t>(n), 0.0};
    const ::sl_status s = ::sl_ambiguity_bootstrap(float_ambiguities.data(), n, covariance.data(),
                                                   out.ambiguities.data(), &out.squared_norm);
    return s == SL_OK ? result<candidate>(std::move(out)) : detail::failure<candidate>(s);
}

[[nodiscard]] inline result<std::vector<candidate>>
try_resolve_ambiguities(std::span<const double> float_ambiguities,
                        std::span<const double> covariance, std::size_t count = 2) {
    using many = std::vector<candidate>;
    const std::size_t n = float_ambiguities.size();
    if (n == 0 || count == 0 || covariance.size() != n * n) {
        return detail::failure<many>(SL_ERR_INVALID_ARGUMENT);
    }
    std::vector<std::int64_t> flat(count * n);
    std::vector<double> norms(count);
    std::size_t len = 0;
    const ::sl_status s = ::sl_resolve_ambiguities(float_ambiguities.data(), n, covariance.data(),
                                                   count, flat.data(), norms.data(), &len);
    if (s != SL_OK) {
        return detail::failure<many>(s);
    }
    many out;
    out.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
        out.push_back(candidate{
            std::vector<std::int64_t>(flat.begin() + static_cast<std::ptrdiff_t>(i * n),
                                      flat.begin() + static_cast<std::ptrdiff_t>((i + 1) * n)),
            norms[i]});
    }
    return result<many>(std::move(out));
}

[[nodiscard]] inline result<probability_estimates>
try_collision_probability(const conjunction &geometry) {
    probability_estimates out{};
    const ::sl_status s = ::sl_collision_probability(&geometry, &out);
    return s == SL_OK ? result<probability_estimates>(out)
                      : detail::failure<probability_estimates>(s);
}

/*
 * `sample` is any callable `bool(double t, sl_relative_sample &out)`: fill `out` and answer `true`,
 * or answer `false` where the trajectory is not known. See `spacelab.h` for what the flags mean.
 */
template <class Sampler>
[[nodiscard]] inline result<usage_violations>
try_conjunction_usage_violations(const conjunction &geometry, double minimum_period,
                                 Sampler &&sample) {
    using bare = std::remove_reference_t<Sampler>;
    detail::relative_sample_bridge bridge{};
    /* Round-tripped to `bare *` below, so a const sampler stays const and needs `operator() const`. */
    bridge.sampler = const_cast<void *>(static_cast<const void *>(std::addressof(sample)));
    bridge.invoke = [](void *s, double t, ::sl_relative_sample *out) {
        return (*static_cast<bare *>(s))(t, *out);
    };

    usage_violations out{};
    const ::sl_status s = ::sl_conjunction_usage_violations(
        &geometry, minimum_period, &detail::spacelab_relative_sample_trampoline, &bridge, &out);
#if defined(__cpp_exceptions)
    if (bridge.failure) {
        std::rethrow_exception(bridge.failure);
    }
#endif
    return s == SL_OK ? result<usage_violations>(out) : detail::failure<usage_violations>(s);
}

/*
 * `relative` is any callable `bool(double t, sl_state &out)`. It is invoked by both halves of the
 * two-call protocol, so it must give the same answer at the same `t` -- a sampler that advances an
 * internal cursor would report a different set of approaches on the second pass.
 */
template <class Sampler>
[[nodiscard]] inline result<std::vector<close_approach>>
try_closest_approaches(Sampler &&relative, double start, double end, double step,
                       double tolerance) {
    using bare = std::remove_reference_t<Sampler>;
    detail::relative_state_bridge bridge{};
    bridge.sampler = const_cast<void *>(static_cast<const void *>(std::addressof(relative)));
    bridge.invoke = [](void *s, double t, ::sl_state *out) {
        return (*static_cast<bare *>(s))(t, *out);
    };

    auto found =
        detail::sized<close_approach>([&](close_approach *out, std::size_t cap, std::size_t *len) {
            return ::sl_closest_approaches(&detail::spacelab_relative_state_trampoline, &bridge,
                                           start, end, step, tolerance, out, cap, len);
        });
#if defined(__cpp_exceptions)
    if (bridge.failure) {
        std::rethrow_exception(bridge.failure);
    }
#endif
    return found;
}

[[nodiscard]] inline bool radially_separated(double perigee_a, double apogee_a, double perigee_b,
                                             double apogee_b, double threshold) {
    return ::sl_radially_separated(perigee_a, apogee_a, perigee_b, apogee_b, threshold);
}

/* ---- Closed-loop control ---- */

using latency = ::sl_latency;
using attitude_law = ::sl_attitude_law;
using attitude_state = ::sl_attitude_state;
using loop_sample = ::sl_loop_sample;
using delay_margin = ::sl_delay_margin;
using element_feedback = ::sl_element_feedback;
using consensus_protocol = ::sl_consensus_protocol;

[[nodiscard]] inline result<double> try_latency_total(const latency &l) {
    double out = 0.0;
    const ::sl_status s = ::sl_latency_total(&l, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<std::size_t> try_latency_steps(const latency &l, double step) {
    std::size_t out = 0;
    const ::sl_status s = ::sl_latency_steps(&l, step, &out);
    return s == SL_OK ? result<std::size_t>(out) : detail::failure<std::size_t>(s);
}

/* The error between two frames: the MRP of the error rotation and its principal angle. */
struct attitude_error {
    vec3 mrp;
    double angle_rad;
};

[[nodiscard]] inline result<attitude_error>
try_attitude_error_mrp(const std::array<double, 9> &rotation,
                       const std::array<double, 9> &reference, const vec3 &rate,
                       const vec3 &rate_reference) {
    attitude_error out{};
    const ::sl_status s = ::sl_attitude_error_mrp(rotation.data(), reference.data(), &rate,
                                                  &rate_reference, &out.mrp, &out.angle_rad);
    return s == SL_OK ? result<attitude_error>(out) : detail::failure<attitude_error>(s);
}

[[nodiscard]] inline result<std::array<double, 4>>
try_attitude_error_quaternion(const std::array<double, 9> &rotation,
                              const std::array<double, 9> &reference, const vec3 &rate,
                              const vec3 &rate_reference) {
    std::array<double, 4> out{};
    const ::sl_status s = ::sl_attitude_error_quaternion(rotation.data(), reference.data(), &rate,
                                                         &rate_reference, out.data());
    return s == SL_OK ? result<std::array<double, 4>>(out)
                      : detail::failure<std::array<double, 4>>(s);
}

[[nodiscard]] inline result<vec3> try_attitude_torque(attitude_law law, double proportional,
                                                      double derivative,
                                                      const std::array<double, 9> &rotation,
                                                      const std::array<double, 9> &reference,
                                                      const vec3 &rate, const vec3 &rate_reference,
                                                      const std::array<double, 9> &inertia) {
    vec3 out{};
    const ::sl_status s =
        ::sl_attitude_torque(law, proportional, derivative, rotation.data(), reference.data(),
                             &rate, &rate_reference, inertia.data(), &out);
    return s == SL_OK ? result<vec3>(out) : detail::failure<vec3>(s);
}

[[nodiscard]] inline result<double>
try_attitude_lyapunov(double proportional, double derivative, const std::array<double, 9> &rotation,
                      const std::array<double, 9> &reference, const vec3 &rate,
                      const vec3 &rate_reference, const std::array<double, 9> &inertia) {
    double out = 0.0;
    const ::sl_status s =
        ::sl_attitude_lyapunov(proportional, derivative, rotation.data(), reference.data(), &rate,
                               &rate_reference, inertia.data(), &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

/* A law's linearisation about the reference: torque per unit angle error and per unit rate error. */
struct small_angle_gains {
    double angle;
    double rate;
};

[[nodiscard]] inline result<small_angle_gains>
try_attitude_small_angle_gains(attitude_law law, double proportional, double derivative) {
    small_angle_gains out{};
    const ::sl_status s =
        ::sl_attitude_small_angle_gains(law, proportional, derivative, &out.angle, &out.rate);
    return s == SL_OK ? result<small_angle_gains>(out) : detail::failure<small_angle_gains>(s);
}

[[nodiscard]] inline result<double> try_quaternion_pd_critically_damped(double proportional,
                                                                        double moment) {
    double out = 0.0;
    const ::sl_status s = ::sl_quaternion_pd_critically_damped(proportional, moment, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

/* The second-order response a gain pair gives a principal axis. */
struct pd_response {
    double natural_frequency_rad_s;
    double damping_ratio;
};

[[nodiscard]] inline result<pd_response>
try_quaternion_pd_response(double proportional, double derivative, double moment) {
    pd_response out{};
    const ::sl_status s = ::sl_quaternion_pd_response(
        proportional, derivative, moment, &out.natural_frequency_rad_s, &out.damping_ratio);
    return s == SL_OK ? result<pd_response>(out) : detail::failure<pd_response>(s);
}

/* Everything one closed-loop run produced. */
struct loop_run {
    std::vector<loop_sample> samples;
    attitude_state final_state;
};

[[nodiscard]] inline result<loop_run>
try_attitude_loop_run(attitude_law law, double proportional, double derivative, const latency &l,
                      double step, const attitude_state &initial, const attitude_state &seed,
                      const std::array<double, 9> &reference, const std::array<double, 9> &inertia,
                      std::size_t steps) {
    loop_run out{};
    auto samples =
        detail::sized<loop_sample>([&](loop_sample *buf, std::size_t cap, std::size_t *len) {
            return ::sl_attitude_loop_run(law, proportional, derivative, &l, step, &initial, &seed,
                                          reference.data(), inertia.data(), steps, buf, cap, len,
                                          &out.final_state);
        });
    if (!samples) {
        return detail::failure<loop_run>(samples.error());
    }
    out.samples = *std::move(samples);
    return result<loop_run>(std::move(out));
}

[[nodiscard]] inline result<delay_margin>
try_delay_margin_of_gains(double moment, double angle_gain, double rate_gain) {
    delay_margin out{};
    const ::sl_status s = ::sl_delay_margin_of_gains(moment, angle_gain, rate_gain, &out);
    return s == SL_OK ? result<delay_margin>(out) : detail::failure<delay_margin>(s);
}

[[nodiscard]] inline result<delay_margin>
try_delay_margin_of_law(attitude_law law, double proportional, double derivative, double moment) {
    delay_margin out{};
    const ::sl_status s = ::sl_delay_margin_of_law(law, proportional, derivative, moment, &out);
    return s == SL_OK ? result<delay_margin>(out) : detail::failure<delay_margin>(s);
}

/* The principal axis whose margin binds, and that margin. */
struct binding_axis {
    std::size_t axis;
    delay_margin margin;
};

[[nodiscard]] inline result<binding_axis> try_worst_principal_axis(attitude_law law,
                                                                   double proportional,
                                                                   double derivative,
                                                                   const vec3 &moments) {
    binding_axis out{};
    const ::sl_status s =
        ::sl_worst_principal_axis(law, proportional, derivative, &moments, &out.axis, &out.margin);
    return s == SL_OK ? result<binding_axis>(out) : detail::failure<binding_axis>(s);
}

/* The Clohessy-Wiltshire plant, row-major. */
struct linear_plant {
    std::array<double, 36> a;
    std::array<double, 18> b;
};

[[nodiscard]] inline result<linear_plant> try_clohessy_wiltshire_plant(double mean_motion) {
    linear_plant out{};
    const ::sl_status s = ::sl_clohessy_wiltshire_plant(mean_motion, out.a.data(), out.b.data());
    return s == SL_OK ? result<linear_plant>(out) : detail::failure<linear_plant>(s);
}

/* An optimal gain, its Riccati solution and the residual it was accepted at. */
struct lqr_solution {
    std::vector<double> gain;
    std::vector<double> riccati;
    double residual;
};

[[nodiscard]] inline result<lqr_solution>
try_lqr(std::span<const double> a, std::span<const double> b, std::span<const double> q,
        std::span<const double> r, std::size_t n, std::size_t m) {
    if (a.size() != n * n || b.size() != n * m || q.size() != n * n || r.size() != m * m) {
        return detail::failure<lqr_solution>(SL_ERR_INVALID_ARGUMENT);
    }
    lqr_solution out{std::vector<double>(m * n), std::vector<double>(n * n), 0.0};
    const ::sl_status s = ::sl_lqr(a.data(), b.data(), q.data(), r.data(), n, m, out.gain.data(),
                                   out.riccati.data(), &out.residual);
    return s == SL_OK ? result<lqr_solution>(std::move(out)) : detail::failure<lqr_solution>(s);
}

/* `m` is the flat row-major matrix; its order is the square root of its length. */
[[nodiscard]] inline result<std::vector<double>>
try_characteristic_polynomial(std::span<const double> m, std::size_t n) {
    if (m.size() != n * n) {
        return detail::failure<std::vector<double>>(SL_ERR_INVALID_ARGUMENT);
    }
    std::vector<double> out(n);
    const ::sl_status s = ::sl_characteristic_polynomial(m.data(), n, out.data());
    return s == SL_OK ? result<std::vector<double>>(std::move(out))
                      : detail::failure<std::vector<double>>(s);
}

[[nodiscard]] inline result<double>
try_element_feedback_lyapunov(const element_feedback &law, const std::array<double, 6> &error) {
    double out = 0.0;
    const ::sl_status s = ::sl_element_feedback_lyapunov(&law, error.data(), &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<vec3>
try_element_feedback_acceleration(const element_feedback &law,
                                  const std::array<double, 6> &elements, double mu,
                                  double true_longitude_rad, const std::array<double, 6> &error) {
    vec3 out{};
    const ::sl_status s = ::sl_element_feedback_acceleration(
        &law, elements.data(), mu, true_longitude_rad, error.data(), &out);
    return s == SL_OK ? result<vec3>(out) : detail::failure<vec3>(s);
}

[[nodiscard]] inline result<double>
try_element_feedback_decrease_rate(const element_feedback &law,
                                   const std::array<double, 6> &elements, double mu,
                                   double true_longitude_rad, const std::array<double, 6> &error) {
    double out = 0.0;
    const ::sl_status s = ::sl_element_feedback_decrease_rate(
        &law, elements.data(), mu, true_longitude_rad, error.data(), &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

/* `adjacency` is the flat row-major weight matrix; the agent count is its square root. */
[[nodiscard]] inline result<std::vector<double>>
try_laplacian_spectrum(std::span<const double> adjacency, std::size_t agents) {
    if (adjacency.size() != agents * agents) {
        return detail::failure<std::vector<double>>(SL_ERR_INVALID_ARGUMENT);
    }
    std::vector<double> out(agents);
    const ::sl_status s = ::sl_laplacian_spectrum(adjacency.data(), agents, out.data());
    return s == SL_OK ? result<std::vector<double>>(std::move(out))
                      : detail::failure<std::vector<double>>(s);
}

/* The Laplacian eigenvalues and the unit eigenvectors, `modes[k * agents .. (k+1) * agents]`. */
struct laplacian_modes {
    std::vector<double> values;
    std::vector<double> modes;
};

[[nodiscard]] inline result<laplacian_modes> try_laplacian_modes(std::span<const double> adjacency,
                                                                 std::size_t agents) {
    if (adjacency.size() != agents * agents) {
        return detail::failure<laplacian_modes>(SL_ERR_INVALID_ARGUMENT);
    }
    laplacian_modes out{std::vector<double>(agents), std::vector<double>(agents * agents)};
    const ::sl_status s =
        ::sl_laplacian_modes(adjacency.data(), agents, out.values.data(), out.modes.data());
    return s == SL_OK ? result<laplacian_modes>(std::move(out))
                      : detail::failure<laplacian_modes>(s);
}

[[nodiscard]] inline result<bool> try_graph_is_connected(std::span<const double> adjacency,
                                                         std::size_t agents) {
    if (adjacency.size() != agents * agents) {
        return detail::failure<bool>(SL_ERR_INVALID_ARGUMENT);
    }
    bool out = false;
    const ::sl_status s = ::sl_graph_is_connected(adjacency.data(), agents, &out);
    return s == SL_OK ? result<bool>(out) : detail::failure<bool>(s);
}

[[nodiscard]] inline result<double> try_algebraic_connectivity(std::span<const double> adjacency,
                                                               std::size_t agents) {
    if (adjacency.size() != agents * agents) {
        return detail::failure<double>(SL_ERR_INVALID_ARGUMENT);
    }
    double out = 0.0;
    const ::sl_status s = ::sl_algebraic_connectivity(adjacency.data(), agents, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<double> try_consensus_delay_bound(std::span<const double> adjacency,
                                                              std::size_t agents,
                                                              consensus_protocol protocol) {
    if (adjacency.size() != agents * agents) {
        return detail::failure<double>(SL_ERR_INVALID_ARGUMENT);
    }
    double out = 0.0;
    const ::sl_status s = ::sl_consensus_delay_bound(adjacency.data(), agents, protocol, &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

[[nodiscard]] inline result<double> try_consensus_disagreement(std::span<const double> state) {
    double out = 0.0;
    const ::sl_status s = ::sl_consensus_disagreement(state.data(), state.size(), &out);
    return s == SL_OK ? result<double>(out) : detail::failure<double>(s);
}

/* Where a delayed consensus run ended, and what it cost. */
struct consensus_result {
    std::vector<double> final_state;
    double disagreement;
    std::uint32_t accepted;
    std::uint32_t rejected;
};

[[nodiscard]] inline result<consensus_result>
try_consensus_run(std::span<const double> adjacency, consensus_protocol protocol, double delay,
                  std::span<const double> initial, double t1, double tolerance) {
    const std::size_t agents = initial.size();
    if (adjacency.size() != agents * agents) {
        return detail::failure<consensus_result>(SL_ERR_INVALID_ARGUMENT);
    }
    consensus_result out{std::vector<double>(agents), 0.0, 0, 0};
    const ::sl_status s =
        ::sl_consensus_run(adjacency.data(), agents, protocol, delay, initial.data(), t1, tolerance,
                           out.final_state.data(), &out.disagreement, &out.accepted, &out.rejected);
    return s == SL_OK ? result<consensus_result>(std::move(out))
                      : detail::failure<consensus_result>(s);
}

[[nodiscard]] inline result<std::vector<double>>
try_delay_breakpoints(std::span<const double> delays, double t0, double t1, std::size_t levels) {
    return detail::sized<double>([&](double *out, std::size_t cap, std::size_t *len) {
        return ::sl_delay_breakpoints(delays.data(), delays.size(), t0, t1, levels, out, cap, len);
    });
}

#if defined(__cpp_exceptions)

[[nodiscard]] inline std::vector<gnss_state> rinex_gnss_positions(std::string_view text,
                                                                  double tk_seconds) {
    return detail::unwrap(try_rinex_gnss_positions(text, tk_seconds));
}

[[nodiscard]] inline std::vector<glonass_state> rinex_glonass_positions(std::string_view text,
                                                                        double dt_seconds) {
    return detail::unwrap(try_rinex_glonass_positions(text, dt_seconds));
}

[[nodiscard]] inline obs_header rinex_obs_header(std::string_view text) {
    return detail::unwrap(try_rinex_obs_header(text));
}

[[nodiscard]] inline std::vector<obs_descriptor> rinex_obs_descriptors(std::string_view text,
                                                                       char system) {
    return detail::unwrap(try_rinex_obs_descriptors(text, system));
}

[[nodiscard]] inline std::vector<obs_sample>
rinex_obs_series(std::string_view text, char system, std::uint32_t prn, std::string_view code) {
    return detail::unwrap(try_rinex_obs_series(text, system, prn, code));
}

[[nodiscard]] inline dop dilution_of_precision(const vec3 &receiver,
                                               std::span<const vec3> satellites) {
    return detail::unwrap(try_dilution_of_precision(receiver, satellites));
}

[[nodiscard]] inline vec3 igrf_field_ecef(const vec3 &position_m, double year) {
    return detail::unwrap(try_igrf_field_ecef(position_m, year));
}

[[nodiscard]] inline double klobuchar_iono_delay(const std::array<double, 4> &alpha,
                                                 const std::array<double, 4> &beta,
                                                 double latitude_rad, double longitude_rad,
                                                 double elevation_rad, double azimuth_rad,
                                                 double gps_seconds_of_day) {
    return detail::unwrap(try_klobuchar_iono_delay(alpha, beta, latitude_rad, longitude_rad,
                                                   elevation_rad, azimuth_rad, gps_seconds_of_day));
}

[[nodiscard]] inline double nequick_galileo_stec(const std::array<double, 3> &alpha,
                                                 std::uint32_t month, double seconds_in_utc_day,
                                                 const geodetic &receiver,
                                                 const geodetic &satellite) {
    return detail::unwrap(
        try_nequick_galileo_stec(alpha, month, seconds_in_utc_day, receiver, satellite));
}

[[nodiscard]] inline mapping niell_mapping(double latitude_rad, double height_m, double day_of_year,
                                           double elevation_rad) {
    return detail::unwrap(try_niell_mapping(latitude_rad, height_m, day_of_year, elevation_rad));
}

[[nodiscard]] inline mapping gmf_mapping(double latitude_rad, double longitude_rad, double height_m,
                                         double day_of_year, double elevation_rad) {
    return detail::unwrap(
        try_gmf_mapping(latitude_rad, longitude_rad, height_m, day_of_year, elevation_rad));
}

[[nodiscard]] inline mapping vmf1_mapping(double ah, double aw, double latitude_rad,
                                          double height_m, double day_of_year,
                                          double elevation_rad) {
    return detail::unwrap(
        try_vmf1_mapping(ah, aw, latitude_rad, height_m, day_of_year, elevation_rad));
}

[[nodiscard]] inline mapping vmf3_mapping(double ah, double aw, double latitude_rad,
                                          double longitude_rad, double height_m, double day_of_year,
                                          double elevation_rad) {
    return detail::unwrap(try_vmf3_mapping(ah, aw, latitude_rad, longitude_rad, height_m,
                                           day_of_year, elevation_rad));
}

[[nodiscard]] inline std::vector<state> brouwer_lyddane_propagate(const brouwer_lyddane_params &p,
                                                                  std::span<const double> times) {
    return detail::unwrap(try_brouwer_lyddane_propagate(p, times));
}

[[nodiscard]] inline std::array<double, 5> solid_tide_step2(double gmst_rad,
                                                            const std::array<double, 5> &delaunay) {
    return detail::unwrap(try_solid_tide_step2(gmst_rad, delaunay));
}

[[nodiscard]] inline std::array<double, 2>
solid_pole_tide(double xp_rad, double yp_rad, double t_seconds,
                mean_pole pole = SL_MEAN_POLE_SECULAR2018) {
    return detail::unwrap(try_solid_pole_tide(xp_rad, yp_rad, t_seconds, pole));
}

[[nodiscard]] inline std::array<double, 2>
ocean_pole_tide(double xp_rad, double yp_rad, double t_seconds,
                mean_pole pole = SL_MEAN_POLE_SECULAR2018) {
    return detail::unwrap(try_ocean_pole_tide(xp_rad, yp_rad, t_seconds, pole));
}

[[nodiscard]] inline zonal conventional_zonal_coefficients(double years_since_j2000) {
    return detail::unwrap(try_conventional_zonal_coefficients(years_since_j2000));
}

[[nodiscard]] inline zonal time_variable_zonal_correction(double years_since_j2000) {
    return detail::unwrap(try_time_variable_zonal_correction(years_since_j2000));
}

[[nodiscard]] inline drag_panel sentman_drag_coefficients(double speed_ratio, double cos_incidence,
                                                          double accommodation,
                                                          double wall_temperature_k,
                                                          double atmosphere_temperature_k) {
    return detail::unwrap(try_sentman_drag_coefficients(
        speed_ratio, cos_incidence, accommodation, wall_temperature_k, atmosphere_temperature_k));
}

[[nodiscard]] inline std::array<double, SL_SDTM1_INTERVAL>
sdtm1_density_interval(double density, double f107, double f107a, double storm, double z) {
    return detail::unwrap(try_sdtm1_density_interval(density, f107, f107a, storm, z));
}

[[nodiscard]] inline std::array<double, SL_SDTM1_ENVELOPE> sdtm1_envelope() {
    return detail::unwrap(try_sdtm1_envelope());
}

[[nodiscard]] inline std::array<double, SL_SDTM1_INTERVAL>
sdtm1_solar_wind_density_interval(double density, double f107, double f107a, double storm,
                                  double z) {
    return detail::unwrap(try_sdtm1_solar_wind_density_interval(density, f107, f107a, storm, z));
}

[[nodiscard]] inline std::array<double, SL_SDTM1_ENVELOPE> sdtm1_solar_wind_envelope() {
    return detail::unwrap(try_sdtm1_solar_wind_envelope());
}

[[nodiscard]] inline std::array<double, SL_DENSITY_DRIVERS>
space_weather_drivers(double f107, double f107_81, double ap) {
    return detail::unwrap(try_space_weather_drivers(f107, f107_81, ap));
}

[[nodiscard]] inline std::array<double, SL_AP_ARRAY_LEN>
space_weather_ap_array(std::string_view csv_text, std::int32_t year, std::uint32_t month,
                       std::uint32_t day, double seconds_of_day) {
    return detail::unwrap(try_space_weather_ap_array(csv_text, year, month, day, seconds_of_day));
}

[[nodiscard]] inline std::vector<space_weather_day> space_weather_days(std::string_view csv_text) {
    return detail::unwrap(try_space_weather_days(csv_text));
}

[[nodiscard]] inline std::vector<density_sample> density_data_samples(std::string_view text) {
    return detail::unwrap(try_density_data_samples(text));
}

[[nodiscard]] inline std::vector<density_observation>
density_altitude_normalized(std::string_view text, double reference_altitude_m,
                            double scale_height_m) {
    return detail::unwrap(
        try_density_altitude_normalized(text, reference_altitude_m, scale_height_m));
}

[[nodiscard]] inline std::vector<double>
density_correction_fit(std::span<const density_sample> samples, double ridge) {
    return detail::unwrap(try_density_correction_fit(samples, ridge));
}

[[nodiscard]] inline double density_correction_factor(std::span<const double> coefficients,
                                                      double latitude_rad,
                                                      double local_solar_time_h) {
    return detail::unwrap(
        try_density_correction_factor(coefficients, latitude_rad, local_solar_time_h));
}

[[nodiscard]] inline std::vector<double>
density_grid_snapshot(std::span<const density_sample> samples, std::size_t n_lat,
                      std::size_t n_lst) {
    return detail::unwrap(try_density_grid_snapshot(samples, n_lat, n_lst));
}

[[nodiscard]] inline swath swath_geometry(double earth_radius_m, double altitude_m,
                                          double look_angle_rad) {
    return detail::unwrap(try_swath_geometry(earth_radius_m, altitude_m, look_angle_rad));
}

[[nodiscard]] inline state propagate_j2_precision(const state &in, double mu, double j2,
                                                  double earth_radius_m, double dt_seconds,
                                                  std::uint64_t steps, precision arithmetic) {
    return detail::unwrap(
        try_propagate_j2_precision(in, mu, j2, earth_radius_m, dt_seconds, steps, arithmetic));
}

[[nodiscard]] inline std::vector<complex> lfm_chirp(std::size_t n, double chirp_rate,
                                                    double sample_rate) {
    return detail::unwrap(try_lfm_chirp(n, chirp_rate, sample_rate));
}

[[nodiscard]] inline std::vector<complex> matched_filter(std::span<const complex> signal,
                                                         std::span<const complex> replica) {
    return detail::unwrap(try_matched_filter(signal, replica));
}

[[nodiscard]] inline double src_fm_rate(double f_eta, const rda_geometry &geometry) {
    return detail::unwrap(try_src_fm_rate(f_eta, geometry));
}

[[nodiscard]] inline std::vector<complex> chirp_scaling_focus(std::span<const complex> raw,
                                                              std::size_t na, std::size_t nr,
                                                              const csa_params &params) {
    return detail::unwrap(try_chirp_scaling_focus(raw, na, nr, params));
}

[[nodiscard]] inline std::vector<complex> apply_chirp_scaling(std::span<const complex> line,
                                                              double chirp_rate, double scaling,
                                                              double tau_ref, double sample_rate) {
    return detail::unwrap(try_apply_chirp_scaling(line, chirp_rate, scaling, tau_ref, sample_rate));
}

[[nodiscard]] inline std::vector<complex> interferogram(std::span<const complex> s1,
                                                        std::span<const complex> s2) {
    return detail::unwrap(try_interferogram(s1, s2));
}

[[nodiscard]] inline std::vector<double> coherence(std::span<const complex> s1,
                                                   std::span<const complex> s2, std::size_t rows,
                                                   std::size_t cols, std::size_t half_window) {
    return detail::unwrap(try_coherence(s1, s2, rows, cols, half_window));
}

[[nodiscard]] inline std::vector<double> unwrap_phase_1d(std::span<const double> wrapped) {
    return detail::unwrap(try_unwrap_phase_1d(wrapped));
}

[[nodiscard]] inline std::vector<double> unwrap_phase_2d(std::span<const double> wrapped,
                                                         std::size_t rows, std::size_t cols) {
    return detail::unwrap(try_unwrap_phase_2d(wrapped, rows, cols));
}

[[nodiscard]] inline std::vector<double>
unwrap_phase_2d_branch_cut(std::span<const double> wrapped, std::size_t rows, std::size_t cols) {
    return detail::unwrap(try_unwrap_phase_2d_branch_cut(wrapped, rows, cols));
}

[[nodiscard]] inline std::vector<double> unwrap_phase_2d_lstsq(std::span<const double> wrapped,
                                                               std::size_t rows, std::size_t cols) {
    return detail::unwrap(try_unwrap_phase_2d_lstsq(wrapped, rows, cols));
}

[[nodiscard]] inline std::vector<std::int8_t> residues(std::span<const double> wrapped,
                                                       std::size_t rows, std::size_t cols) {
    return detail::unwrap(try_residues(wrapped, rows, cols));
}

[[nodiscard]] inline std::vector<double> capon_spectrum(std::span<const complex> covariance,
                                                        std::span<const double> baselines,
                                                        std::span<const double> elevations,
                                                        const tomo_params &params) {
    return detail::unwrap(try_capon_spectrum(covariance, baselines, elevations, params));
}

[[nodiscard]] inline std::vector<double> bartlett_spectrum(std::span<const complex> covariance,
                                                           std::span<const double> baselines,
                                                           std::span<const double> elevations,
                                                           const tomo_params &params) {
    return detail::unwrap(try_bartlett_spectrum(covariance, baselines, elevations, params));
}

[[nodiscard]] inline std::vector<scatterer> sl1mmer(std::span<const complex> y,
                                                    std::span<const double> baselines,
                                                    std::span<const double> elevations,
                                                    const tomo_params &params) {
    return detail::unwrap(try_sl1mmer(y, baselines, elevations, params));
}

[[nodiscard]] inline std::vector<double> sl1mmer_spectrum(std::span<const complex> y,
                                                          std::span<const double> baselines,
                                                          std::span<const double> elevations,
                                                          const tomo_params &params) {
    return detail::unwrap(try_sl1mmer_spectrum(y, baselines, elevations, params));
}

[[nodiscard]] inline std::vector<double>
amplitude_dispersion(std::span<const double> stack, std::size_t epochs, std::size_t pixels) {
    return detail::unwrap(try_amplitude_dispersion(stack, epochs, pixels));
}

[[nodiscard]] inline std::vector<std::size_t>
select_persistent_scatterers(std::span<const double> dispersion, double threshold) {
    return detail::unwrap(try_select_persistent_scatterers(dispersion, threshold));
}

[[nodiscard]] inline deformation sbas_invert(std::span<const double> phases,
                                             std::span<const double> temporal_baselines,
                                             std::span<const double> perp_baselines,
                                             double wavelength, double slant_range,
                                             double incidence_rad) {
    return detail::unwrap(try_sbas_invert(phases, temporal_baselines, perp_baselines, wavelength,
                                          slant_range, incidence_rad));
}

[[nodiscard]] inline std::vector<complex>
coherency_matrix(std::span<const polarimetric_sample> samples) {
    return detail::unwrap(try_coherency_matrix(samples));
}

[[nodiscard]] inline cloude_pottier cloude_pottier_decompose(std::span<const complex> t3) {
    return detail::unwrap(try_cloude_pottier_decompose(t3));
}

[[nodiscard]] inline std::array<double, 3> pauli_powers(const polarimetric_sample &sample) {
    return detail::unwrap(try_pauli_powers(sample));
}

[[nodiscard]] inline double resolution_3db_samples(std::span<const double> magnitude) {
    return detail::unwrap(try_resolution_3db_samples(magnitude));
}

[[nodiscard]] inline double peak_sidelobe_ratio_db(std::span<const double> magnitude) {
    return detail::unwrap(try_peak_sidelobe_ratio_db(magnitude));
}

[[nodiscard]] inline std::vector<double> semi_global_match(std::span<const double> left,
                                                           std::span<const double> right,
                                                           std::size_t rows, std::size_t cols,
                                                           const sgm_config &config) {
    return detail::unwrap(try_semi_global_match(left, right, rows, cols, config));
}

[[nodiscard]] inline std::vector<double>
plane_sweep_mvs(const mvs_view &reference, std::span<const mvs_view> neighbors, std::size_t rows,
                std::size_t cols, std::span<const double> depths, std::size_t patch_radius) {
    return detail::unwrap(
        try_plane_sweep_mvs(reference, neighbors, rows, cols, depths, patch_radius));
}

[[nodiscard]] inline vec3 triangulate(std::span<const camera> cameras,
                                      std::span<const double> pixels) {
    return detail::unwrap(try_triangulate(cameras, pixels));
}

[[nodiscard]] inline std::vector<track_point>
sfm_triangulate(std::span<const corner> corners, std::span<const std::size_t> corners_per_view,
                std::span<const feature_match> matches, std::span<const camera> cameras) {
    return detail::unwrap(try_sfm_triangulate(corners, corners_per_view, matches, cameras));
}

[[nodiscard]] inline std::vector<corner> harris_corners(std::span<const double> image,
                                                        std::size_t rows, std::size_t cols,
                                                        const harris_config &config) {
    return detail::unwrap(try_harris_corners(image, rows, cols, config));
}

[[nodiscard]] inline image_point rpc_block_adjust(std::string_view rpb_text,
                                                  std::span<const ground_control_point> gcps,
                                                  double latitude, double longitude, double height,
                                                  bool affine) {
    return detail::unwrap(
        try_rpc_block_adjust(rpb_text, gcps, latitude, longitude, height, affine));
}

[[nodiscard]] inline std::vector<double> brovey_pan_sharpen(std::span<const double> ms_bands,
                                                            std::size_t band_count,
                                                            std::span<const double> pan) {
    return detail::unwrap(try_brovey_pan_sharpen(ms_bands, band_count, pan));
}

[[nodiscard]] inline std::vector<double> gram_schmidt_pan_sharpen(std::span<const double> ms_bands,
                                                                  std::size_t band_count,
                                                                  std::span<const double> pan) {
    return detail::unwrap(try_gram_schmidt_pan_sharpen(ms_bands, band_count, pan));
}

[[nodiscard]] inline std::vector<std::size_t>
nms(std::span<const bbox> boxes, std::span<const double> scores, double iou_threshold) {
    return detail::unwrap(try_nms(boxes, scores, iou_threshold));
}

[[nodiscard]] inline bbox decode_box(const bbox &anchor, const std::array<double, 4> &delta) {
    return detail::unwrap(try_decode_box(anchor, delta));
}

[[nodiscard]] inline std::vector<bbox> generate_anchors(std::size_t feat_rows,
                                                        std::size_t feat_cols, double stride,
                                                        std::span<const double> scales,
                                                        std::span<const double> aspect_ratios) {
    return detail::unwrap(
        try_generate_anchors(feat_rows, feat_cols, stride, scales, aspect_ratios));
}

[[nodiscard]] inline gpt3_weather gpt3(double latitude_rad, double longitude_rad,
                                       double day_of_year, double height_m) {
    return detail::unwrap(try_gpt3(latitude_rad, longitude_rad, day_of_year, height_m));
}

[[nodiscard]] inline vmf_coefficients
vmf_grid_coefficients(std::string_view grid_text, double latitude_rad, double longitude_rad) {
    return detail::unwrap(try_vmf_grid_coefficients(grid_text, latitude_rad, longitude_rad));
}

[[nodiscard]] inline double bistatic_range(const state &satellite, const terminal &transmitter,
                                           double bounce, const terminal &receiver) {
    return detail::unwrap(try_bistatic_range(satellite, transmitter, bounce, receiver));
}

[[nodiscard]] inline double bistatic_range_rate(const state &satellite, const terminal &transmitter,
                                                double bounce, const terminal &receiver) {
    return detail::unwrap(try_bistatic_range_rate(satellite, transmitter, bounce, receiver));
}

[[nodiscard]] inline double tdoa(const state &satellite, double emission, const terminal &first,
                                 const terminal &second) {
    return detail::unwrap(try_tdoa(satellite, emission, first, second));
}

[[nodiscard]] inline double fdoa(const state &satellite, double emission, const terminal &first,
                                 const terminal &second, double frequency) {
    return detail::unwrap(try_fdoa(satellite, emission, first, second, frequency));
}

[[nodiscard]] inline double delta_dor(const state &satellite, double emission,
                                      const terminal &first, const terminal &second,
                                      double right_ascension, double declination) {
    return detail::unwrap(
        try_delta_dor(satellite, emission, first, second, right_ascension, declination));
}

[[nodiscard]] inline double dsn_sequential_range(std::span<const ramp_segment> segments,
                                                 band uplink, double transmit, double receive,
                                                 std::int32_t lowest_ranging_component) {
    return detail::unwrap(
        try_dsn_sequential_range(segments, uplink, transmit, receive, lowest_ranging_component));
}

[[nodiscard]] inline double dsn_averaged_doppler(std::span<const ramp_segment> segments,
                                                 const doppler_link &link, double transmit_start,
                                                 double transmit_end) {
    return detail::unwrap(try_dsn_averaged_doppler(segments, link, transmit_start, transmit_end));
}

[[nodiscard]] inline double dsn_measured_frequency(std::span<const ramp_segment> segments,
                                                   band uplink, band downlink,
                                                   double transmit_epoch, double transmit_rate) {
    return detail::unwrap(
        try_dsn_measured_frequency(segments, uplink, downlink, transmit_epoch, transmit_rate));
}

[[nodiscard]] inline double solar_corona_delay(const vec3 &transmitter_from_sun,
                                               const vec3 &receiver_from_sun, double frequency,
                                               bool group_delay = true,
                                               std::span<const corona_term> series = {}) {
    return detail::unwrap(try_solar_corona_delay(transmitter_from_sun, receiver_from_sun, frequency,
                                                 group_delay, series));
}

[[nodiscard]] inline lifetime orbital_lifetime(const decay_params &params,
                                               const std::array<double, 6> &initial,
                                               std::span<const double> zonals) {
    return detail::unwrap(try_orbital_lifetime(params, initial, zonals));
}

[[nodiscard]] inline lifetime_quantiles
orbital_lifetime_ensemble(const decay_params &params, const std::array<double, 6> &initial,
                          std::span<const double> zonals, std::span<const double> density_scales,
                          std::span<const double> quantiles) {
    return detail::unwrap(
        try_orbital_lifetime_ensemble(params, initial, zonals, density_scales, quantiles));
}

[[nodiscard]] inline double ballistic_log_sensitivity(const decay_params &params,
                                                      const std::array<double, 6> &initial,
                                                      std::span<const double> zonals,
                                                      double fraction = 0.01) {
    return detail::unwrap(try_ballistic_log_sensitivity(params, initial, zonals, fraction));
}

[[nodiscard]] inline tle_lines tle(const utc_datetime &epoch, std::uint32_t norad_id,
                                   const std::array<double, 6> &parameters, double drag_term = 0.0,
                                   std::string_view international_designator = {},
                                   std::uint64_t element_set_number = 0,
                                   std::uint64_t revolution_number = 0) {
    return detail::unwrap(try_tle_lines(epoch, norad_id, parameters, drag_term,
                                        international_designator, element_set_number,
                                        revolution_number));
}

[[nodiscard]] inline lanes lane_wavelengths(double f1, double f2) {
    return detail::unwrap(try_lane_wavelengths(f1, f2));
}

[[nodiscard]] inline double ionosphere_free(double f1, double f2, double x1, double x2) {
    return detail::unwrap(try_ionosphere_free(f1, f2, x1, x2));
}

[[nodiscard]] inline double wide_lane(double f1, double f2, double x1, double x2) {
    return detail::unwrap(try_wide_lane(f1, f2, x1, x2));
}

[[nodiscard]] inline double narrow_lane(double f1, double f2, double x1, double x2) {
    return detail::unwrap(try_narrow_lane(f1, f2, x1, x2));
}

[[nodiscard]] inline double melbourne_wubbena(double f1, double f2, double p1, double p2, double l1,
                                              double l2, bool cycles = false) {
    return detail::unwrap(try_melbourne_wubbena(f1, f2, p1, p2, l1, l2, cycles));
}

[[nodiscard]] inline std::vector<slip_verdict>
detect_cycle_slips_geometry_free(std::span<const double> times, std::span<const double> values,
                                 double threshold, double max_gap) {
    return detail::unwrap(try_detect_cycle_slips_geometry_free(times, values, threshold, max_gap));
}

[[nodiscard]] inline std::vector<slip_verdict>
detect_cycle_slips_polynomial(std::span<const double> times, std::span<const double> values,
                              std::size_t order, std::size_t window, double threshold,
                              double max_gap) {
    return detail::unwrap(
        try_detect_cycle_slips_polynomial(times, values, order, window, threshold, max_gap));
}

[[nodiscard]] inline decorrelation
ambiguity_decorrelation(std::span<const double> float_ambiguities,
                        std::span<const double> covariance) {
    return detail::unwrap(try_ambiguity_decorrelation(float_ambiguities, covariance));
}

[[nodiscard]] inline candidate ambiguity_bootstrap(std::span<const double> float_ambiguities,
                                                   std::span<const double> covariance) {
    return detail::unwrap(try_ambiguity_bootstrap(float_ambiguities, covariance));
}

[[nodiscard]] inline std::vector<candidate>
resolve_ambiguities(std::span<const double> float_ambiguities, std::span<const double> covariance,
                    std::size_t count = 2) {
    return detail::unwrap(try_resolve_ambiguities(float_ambiguities, covariance, count));
}

[[nodiscard]] inline probability_estimates collision_probability(const conjunction &geometry) {
    return detail::unwrap(try_collision_probability(geometry));
}

template <class Sampler>
[[nodiscard]] inline usage_violations
conjunction_usage_violations(const conjunction &geometry, double minimum_period, Sampler &&sample) {
    return detail::unwrap(
        try_conjunction_usage_violations(geometry, minimum_period, std::forward<Sampler>(sample)));
}

template <class Sampler>
[[nodiscard]] inline std::vector<close_approach>
closest_approaches(Sampler &&relative, double start, double end, double step, double tolerance) {
    return detail::unwrap(
        try_closest_approaches(std::forward<Sampler>(relative), start, end, step, tolerance));
}

/* ---- Closed-loop control ---- */

[[nodiscard]] inline double latency_total(const latency &l) {
    return detail::unwrap(try_latency_total(l));
}

[[nodiscard]] inline std::size_t latency_steps(const latency &l, double step) {
    return detail::unwrap(try_latency_steps(l, step));
}

[[nodiscard]] inline attitude_error attitude_error_mrp(const std::array<double, 9> &rotation,
                                                       const std::array<double, 9> &reference,
                                                       const vec3 &rate,
                                                       const vec3 &rate_reference) {
    return detail::unwrap(try_attitude_error_mrp(rotation, reference, rate, rate_reference));
}

[[nodiscard]] inline std::array<double, 4>
attitude_error_quaternion(const std::array<double, 9> &rotation,
                          const std::array<double, 9> &reference, const vec3 &rate,
                          const vec3 &rate_reference) {
    return detail::unwrap(try_attitude_error_quaternion(rotation, reference, rate, rate_reference));
}

[[nodiscard]] inline vec3 attitude_torque(attitude_law law, double proportional, double derivative,
                                          const std::array<double, 9> &rotation,
                                          const std::array<double, 9> &reference, const vec3 &rate,
                                          const vec3 &rate_reference,
                                          const std::array<double, 9> &inertia) {
    return detail::unwrap(try_attitude_torque(law, proportional, derivative, rotation, reference,
                                              rate, rate_reference, inertia));
}

[[nodiscard]] inline double attitude_lyapunov(double proportional, double derivative,
                                              const std::array<double, 9> &rotation,
                                              const std::array<double, 9> &reference,
                                              const vec3 &rate, const vec3 &rate_reference,
                                              const std::array<double, 9> &inertia) {
    return detail::unwrap(try_attitude_lyapunov(proportional, derivative, rotation, reference, rate,
                                                rate_reference, inertia));
}

[[nodiscard]] inline small_angle_gains
attitude_small_angle_gains(attitude_law law, double proportional, double derivative) {
    return detail::unwrap(try_attitude_small_angle_gains(law, proportional, derivative));
}

[[nodiscard]] inline double quaternion_pd_critically_damped(double proportional, double moment) {
    return detail::unwrap(try_quaternion_pd_critically_damped(proportional, moment));
}

[[nodiscard]] inline pd_response quaternion_pd_response(double proportional, double derivative,
                                                        double moment) {
    return detail::unwrap(try_quaternion_pd_response(proportional, derivative, moment));
}

[[nodiscard]] inline loop_run
attitude_loop_run(attitude_law law, double proportional, double derivative, const latency &l,
                  double step, const attitude_state &initial, const attitude_state &seed,
                  const std::array<double, 9> &reference, const std::array<double, 9> &inertia,
                  std::size_t steps) {
    return detail::unwrap(try_attitude_loop_run(law, proportional, derivative, l, step, initial,
                                                seed, reference, inertia, steps));
}

[[nodiscard]] inline delay_margin delay_margin_of_gains(double moment, double angle_gain,
                                                        double rate_gain) {
    return detail::unwrap(try_delay_margin_of_gains(moment, angle_gain, rate_gain));
}

[[nodiscard]] inline delay_margin delay_margin_of_law(attitude_law law, double proportional,
                                                      double derivative, double moment) {
    return detail::unwrap(try_delay_margin_of_law(law, proportional, derivative, moment));
}

[[nodiscard]] inline binding_axis worst_principal_axis(attitude_law law, double proportional,
                                                       double derivative, const vec3 &moments) {
    return detail::unwrap(try_worst_principal_axis(law, proportional, derivative, moments));
}

[[nodiscard]] inline linear_plant clohessy_wiltshire_plant(double mean_motion) {
    return detail::unwrap(try_clohessy_wiltshire_plant(mean_motion));
}

[[nodiscard]] inline lqr_solution lqr(std::span<const double> a, std::span<const double> b,
                                      std::span<const double> q, std::span<const double> r,
                                      std::size_t n, std::size_t m) {
    return detail::unwrap(try_lqr(a, b, q, r, n, m));
}

[[nodiscard]] inline std::vector<double> characteristic_polynomial(std::span<const double> m,
                                                                   std::size_t n) {
    return detail::unwrap(try_characteristic_polynomial(m, n));
}

[[nodiscard]] inline double element_feedback_lyapunov(const element_feedback &law,
                                                      const std::array<double, 6> &error) {
    return detail::unwrap(try_element_feedback_lyapunov(law, error));
}

[[nodiscard]] inline vec3 element_feedback_acceleration(const element_feedback &law,
                                                        const std::array<double, 6> &elements,
                                                        double mu, double true_longitude_rad,
                                                        const std::array<double, 6> &error) {
    return detail::unwrap(
        try_element_feedback_acceleration(law, elements, mu, true_longitude_rad, error));
}

[[nodiscard]] inline double element_feedback_decrease_rate(const element_feedback &law,
                                                           const std::array<double, 6> &elements,
                                                           double mu, double true_longitude_rad,
                                                           const std::array<double, 6> &error) {
    return detail::unwrap(
        try_element_feedback_decrease_rate(law, elements, mu, true_longitude_rad, error));
}

[[nodiscard]] inline std::vector<double> laplacian_spectrum(std::span<const double> adjacency,
                                                            std::size_t agents) {
    return detail::unwrap(try_laplacian_spectrum(adjacency, agents));
}

[[nodiscard]] inline laplacian_modes laplacian_eigen(std::span<const double> adjacency,
                                                     std::size_t agents) {
    return detail::unwrap(try_laplacian_modes(adjacency, agents));
}

[[nodiscard]] inline bool graph_is_connected(std::span<const double> adjacency,
                                             std::size_t agents) {
    return detail::unwrap(try_graph_is_connected(adjacency, agents));
}

[[nodiscard]] inline double algebraic_connectivity(std::span<const double> adjacency,
                                                   std::size_t agents) {
    return detail::unwrap(try_algebraic_connectivity(adjacency, agents));
}

[[nodiscard]] inline double consensus_delay_bound(std::span<const double> adjacency,
                                                  std::size_t agents, consensus_protocol protocol) {
    return detail::unwrap(try_consensus_delay_bound(adjacency, agents, protocol));
}

[[nodiscard]] inline double consensus_disagreement(std::span<const double> state) {
    return detail::unwrap(try_consensus_disagreement(state));
}

[[nodiscard]] inline consensus_result consensus_run(std::span<const double> adjacency,
                                                    consensus_protocol protocol, double delay,
                                                    std::span<const double> initial, double t1,
                                                    double tolerance) {
    return detail::unwrap(try_consensus_run(adjacency, protocol, delay, initial, t1, tolerance));
}

[[nodiscard]] inline std::vector<double>
delay_breakpoints(std::span<const double> delays, double t0, double t1, std::size_t levels) {
    return detail::unwrap(try_delay_breakpoints(delays, t0, t1, levels));
}

#endif

} // namespace spacelab

#endif /* SPACELAB_HPP */
