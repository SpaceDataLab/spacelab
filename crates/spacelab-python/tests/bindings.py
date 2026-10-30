#!/usr/bin/env python
"""Boundary tests for the spacelab bindings.

The physics under the bridge is already checked against external oracles by the Rust suite
(docs/verification.md). What no Rust test can reach is the boundary itself: whether an argument
landed in the slot it was named for, whether metres arrived where metres were expected, whether a
flat list was reshaped along the axis the caller meant. Those are the mistakes a binding makes, and
they are invisible to every test on either side of it.

So the assertions here are deliberately *self-checking* rather than golden constants copied out of
the implementation: a solver is fed the answer to its own question (Lambert's velocities must fly
the transfer they claim to solve), an invariant is asserted that only the correct wiring can satisfy
(the zonal mean semi-major axis is constant; a coherency matrix is Hermitian; a wrapped ramp unwraps
to the ramp), or a bias is injected and the routine must recover exactly that bias. A copied
constant would agree with a swapped argument. These do not.

Companion to smoke.py, which covers the rest of the surface; `test_every_export_is_named_by_a_test`
holds the two together.

Build and run:
    cd crates/spacelab-python-harness
    maturin build --release -o /tmp/w
    uv run --no-project --with /tmp/w/*.whl python tests/bindings.py
"""

import math
import os
import sys
import sysconfig
from itertools import pairwise

import spacedatalab as sl

MU = 3.986004418e14
RE = 6378137.0

# J2..J4, un-normalised, as the DSST and Brouwer-Lyddane entry points want them.
ZONALS = [1.08262668e-3, -2.53265649e-6, -1.61962159e-6]


def close(a, b, tol=1e-6):
    assert abs(a - b) <= tol * max(1.0, abs(b)), f"{a} != {b}"


def norm(v):
    return sum(c * c for c in v) ** 0.5


def far(a, b):
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def circular_state(altitude, inclination):
    """An inertial circular state at `altitude` above the equatorial radius."""
    r = RE + altitude
    speed = (MU / r) ** 0.5
    return [r, 0.0, 0.0], [
        0.0,
        speed * math.cos(inclination),
        speed * math.sin(inclination),
    ]


def specific_energy(state):
    r, v = state
    return sum(c * c for c in v) / 2.0 - MU / norm(r)


def raan(r, v):
    """Right ascension of the ascending node from the angular momentum direction."""
    hx = r[1] * v[2] - r[2] * v[1]
    hy = r[2] * v[0] - r[0] * v[2]
    return math.atan2(hx, -hy)


def texture(n, seed):
    """A deterministic pseudo-random image; the same 64-bit LCG smoke.py uses."""
    out = []
    for _ in range(n):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        out.append((seed >> 40) / float(1 << 24))
    return out


def steering(baselines, elevation, wavelength, distance):
    """The TomoSAR steering vector of a scatterer at `elevation`, as (re, im) pairs."""
    out = []
    for b in baselines:
        phase = -(4.0 * math.pi * b / (wavelength * distance)) * elevation
        out.append((math.cos(phase), math.sin(phase)))
    return out


def test_lambert_closes_the_transfer_it_solves():
    # The strongest available check needs no reference: fly the velocity Lambert returns and the
    # arc must arrive where Lambert was told to aim. A swapped r1/r2 or a wrong mu fails this.
    r1, v1_true = circular_state(622e3, 0.0)
    r2, v2_true = sl.kepler_propagate(r1, v1_true, MU, 1500.0)
    v1, v2 = sl.lambert(r1, r2, 1500.0, MU, True)
    assert far(v1, v1_true) < 1e-9, far(v1, v1_true)
    assert far(v2, v2_true) < 1e-9, far(v2, v2_true)
    arrival, _ = sl.kepler_propagate(r1, v1, MU, 1500.0)
    assert far(arrival, r2) < 1e-6, far(arrival, r2)


def test_nrlmsise00_density_falls_with_altitude_and_uses_every_argument():
    base = {"doy": 80, "sec": 43200.0, "lat_deg": 35.0, "lon_deg": 140.0}
    profile = [
        sl.nrlmsise00_density(alt_km=a, **base)
        for a in (200.0, 300.0, 400.0, 600.0, 800.0)
    ]
    assert all(x > y for x, y in pairwise(profile)), profile
    assert 1e-13 < profile[2] < 5e-11, profile[2]  # ~400 km, quiet sun

    # Each geodetic and space-weather argument must reach the model; a dropped or duplicated one
    # leaves the answer unchanged, which is exactly what a mis-ordered binding looks like.
    reference = sl.nrlmsise00_density(alt_km=400.0, **base)
    for name, value in (
        ("doy", 260),
        ("sec", 3600.0),
        ("lat_deg", -35.0),
        ("lon_deg", -40.0),
        ("f107", 220.0),
        ("f107a", 210.0),
        ("ap", 60.0),
    ):
        varied = {**base, "alt_km": 400.0}
        varied[name] = value
        assert sl.nrlmsise00_density(**varied) != reference, name


def test_sdtm1_is_a_profile_a_temperature_a_kernel_and_an_interval():
    base = {"doy": 80.0, "sec": 43200.0, "lat_deg": 35.0, "lon_deg": 140.0}
    profile = [
        sl.sdtm1_density(alt_km=a, **base) for a in (200.0, 300.0, 400.0, 600.0, 800.0)
    ]
    assert all(x > y for x, y in pairwise(profile)), profile
    assert 1e-13 < profile[2] < 5e-11, profile[2]  # ~400 km, mid-cycle defaults

    # Fifteen degrees of longitude is one hour of local solar time, and that is the convention a
    # caller has to be able to rely on: 15 degrees east an hour earlier in UT is the same place in the
    # model and reproduces the density to 2.2e-4 relative -- the annual harmonic over 1/24 of a day,
    # which is all that is left -- while the same 15 degrees at the same UT is an hour later in local
    # time and worth 12.0 %. Asserted as a ratio rather than a wrapped-longitude symmetry, because a
    # symmetry of the input cannot pin the constant it is built out of.
    here = sl.sdtm1_density(alt_km=400.0, **base)
    same_place = sl.sdtm1_density(
        **{**base, "alt_km": 400.0, "sec": 39600.0, "lon_deg": 155.0}
    )
    an_hour_later = sl.sdtm1_density(**{**base, "alt_km": 400.0, "lon_deg": 155.0})
    assert abs(same_place / here - 1.0) < 1e-3, (here, same_place)
    assert abs(an_hour_later / here - 1.0) > 0.05, (here, an_hour_later)

    # Each argument must reach the model; a dropped or duplicated one leaves the answer unchanged,
    # which is exactly what a mis-ordered binding looks like.
    reference = sl.sdtm1_density(alt_km=400.0, **base)
    for name, value in (
        ("doy", 260.0),
        ("sec", 3600.0),
        ("lat_deg", -35.0),
        ("lon_deg", -40.0),
        ("f107", 220.0),
        ("f107a", 210.0),
        ("storm", 60.0),
    ):
        varied = {**base, "alt_km": 400.0}
        varied[name] = value
        assert sl.sdtm1_density(**varied) != reference, name

    # T_inf is above the fixed 380 K boundary by construction, inside the range the fit produced
    # over the corpus, and higher under a storm.
    t_inf = sl.sdtm1_exospheric_temperature(alt_km=400.0, **base)
    assert 586.0 < t_inf < 1528.0, t_inf
    assert sl.sdtm1_exospheric_temperature(alt_km=400.0, storm=250.0, **base) > t_inf

    # The storm state cools: a constant index history sits on the steady state H/(1 + H/H_s) from
    # the first step, below the index itself and the same at any cadence. A rising tail moves it up
    # without reaching the peak.
    quiet = sl.sdtm1_storm_response([5.0] * 16)
    assert 0.0 < quiet < 5.0, quiet
    assert (
        abs(sl.sdtm1_storm_response([5.0] * 16, cadence_s=3600.0) - quiet)
        < 1e-12 * quiet
    )
    assert sl.sdtm1_storm_response([]) == 0.0
    surge = sl.sdtm1_storm_response([5.0] * 12 + [300.0] * 4)
    assert quiet < surge < 300.0, surge

    # The interval is multiplicative about the density, so z = 0 collapses it and the geometric mean
    # of the two ends is the density itself.
    lo, hi = sl.sdtm1_density_interval(reference)
    assert lo < reference < hi, (lo, reference, hi)
    assert abs(lo * hi / reference**2 - 1.0) < 1e-12, (lo, hi)
    same = sl.sdtm1_density_interval(reference, z=0.0)
    assert same == (reference, reference), same
    # Both regime arguments must reach the table, or the interval is one number pretending to be six.
    # The quiet low-flux cell is wider than the quiet high-flux one (sigma 0.31 against 0.23). Which
    # way the storm bin moves the interval is a property of the fit, not of the bridge -- the pilot
    # field had it marginally narrower, the corpus field has it wider -- so only that moving the bin
    # moves the interval is asserted.
    low_flux = sl.sdtm1_density_interval(reference, f107a=80.0)
    mid_flux = sl.sdtm1_density_interval(reference, f107a=150.0)
    assert low_flux[1] - low_flux[0] > mid_flux[1] - mid_flux[0], (
        low_flux,
        mid_flux,
    )
    stormy = sl.sdtm1_density_interval(reference, f107a=150.0, storm=60.0)
    assert stormy != mid_flux, (mid_flux, stormy)


def test_brouwer_lyddane_reproduces_its_epoch_and_regresses_the_node():
    r0 = [6800e3, 1200e3, 2100e3]
    v0 = [-1500.0, 6200.0, 3900.0]
    day = 86400.0
    states = sl.brouwer_lyddane_propagate(r0, v0, [0.0, day])
    assert far(states[0][0], r0) < 1e-6, far(states[0][0], r0)
    assert far(states[0][1], v0) < 1e-9, far(states[0][1], v0)

    # J2 regresses the node of a prograde orbit; this one at about -5.2 deg/day. The sign is the
    # test: it is what an inverted c20 or a swapped position/velocity would flip.
    drift = raan(*states[1]) - raan(*states[0])
    rate = math.degrees((drift + math.pi) % (2.0 * math.pi) - math.pi)
    assert -7.0 < rate < -3.0, rate


def test_dsst_zonal_mean_holds_the_semi_major_axis():
    initial = [7000e3, 1e-3, 2e-3, 0.4, 0.1, 0.5]
    span = 10.0 * 86400.0
    mean = sl.dsst_zonal_propagate_mean(initial, [0.0, span], j=ZONALS)
    assert mean[0] == initial, mean[0]

    # The defining property of the zonal mean elements: a conservative zonal field produces no
    # secular drift in a, so the averaged semi-major axis is constant to the bit.
    assert mean[1][0] == initial[0], mean[1][0] - initial[0]

    # The mean longitude advances at the mean motion, to within the J2 correction to it.
    revolutions = span / (2.0 * math.pi * (initial[0] ** 3 / MU) ** 0.5)
    advance = (mean[1][5] - mean[0][5]) / (2.0 * math.pi)
    close(advance, revolutions, 1e-3)


def test_dsst_zonal_osculating_adds_a_short_period_signature():
    initial = [7000e3, 1e-3, 2e-3, 0.4, 0.1, 0.5]
    times = [0.0, 10.0 * 86400.0]
    mean = sl.dsst_zonal_propagate_mean(initial, times, j=ZONALS)
    osculating = sl.dsst_zonal_propagate_osculating(initial, times, j=ZONALS)

    # The short-period correction to a is a few km at LEO under J2 -- large enough that a binding
    # returning the mean elements twice is caught, small enough that a wrong scale is too.
    for m, o in zip(mean, osculating, strict=True):
        offset = abs(o[0] - m[0])
        assert 1e3 < offset < 3e4, offset


def test_dsst_full_wires_the_third_body_and_radiation_pressure():
    initial = [7000e3, 1e-3, 2e-3, 0.4, 0.1, 0.5]
    times = [0.0, 5.0 * 86400.0]
    mean = sl.dsst_propagate_mean_full(initial, times, 0.0, j=ZONALS)
    assert mean[0] == initial, mean[0]
    assert abs(mean[1][0] - initial[0]) < 1e3, mean[1][0] - initial[0]

    # Radiation pressure enters through the area-to-mass ratio and the Sun through the epoch; if
    # either argument is dropped on the way in, the propagation is insensitive to it.
    heavier = sl.dsst_propagate_mean_full(
        initial, times, 0.0, j=ZONALS, area_over_mass=0.2
    )
    assert far(heavier[1], mean[1]) > 1e-3, far(heavier[1], mean[1])
    later = sl.dsst_propagate_mean_full(initial, times, 180.0 * 86400.0, j=ZONALS)
    assert far(later[1], mean[1]) > 1e-3, far(later[1], mean[1])

    osculating = sl.dsst_propagate_osculating_full(initial, times, 0.0, j=ZONALS)
    for m, o in zip(mean, osculating, strict=True):
        assert 1e3 < abs(o[0] - m[0]) < 3e4, o[0] - m[0]


def test_solid_tide_step2_is_periodic_in_sidereal_time():
    delaunay = [1.0, 2.0, 3.0, 4.0, 5.0]
    corrections = sl.solid_tide_step2(0.7, delaunay)
    assert len(corrections) == 5

    # Every Step-2 term is a sine or cosine of m*(gmst + pi) minus a combination of the Delaunay
    # arguments, so the whole correction is exactly 2*pi-periodic in gmst. Nothing but the correct
    # argument reaching the correct trigonometric slot reproduces that.
    shifted = sl.solid_tide_step2(0.7 + 2.0 * math.pi, delaunay)
    assert far(corrections, shifted) < 1e-20, far(corrections, shifted)

    # Millimetre-scale corrections to the degree-2 coefficients: the C21/S21 diurnal band dominates.
    assert 1e-11 < abs(corrections[1]) < 1e-8, corrections[1]
    assert 1e-11 < abs(corrections[2]) < 1e-8, corrections[2]

    for i in range(5):
        varied = list(delaunay)
        varied[i] += 0.5
        assert far(sl.solid_tide_step2(0.7, varied), corrections) > 0.0, i


def test_pole_tides_are_affine_in_the_pole_and_ordered_by_size():
    # Both pole tides are affine in the polar motion measured from whichever conventional pole is
    # selected, so the midpoint identity f(a) + f(b) = 2 f((a+b)/2) holds exactly under either --
    # and it holds without this test needing to know where that pole is.
    for tide in (sl.solid_pole_tide, sl.ocean_pole_tide):
        for pole in ("secular2018", "iers2010"):
            lo = tide(-2e-6, 1e-6, 0.0, pole)
            hi = tide(4e-6, 5e-6, 0.0, pole)
            mid = tide(1e-6, 3e-6, 0.0, pole)
            for k in range(2):
                close(lo[k] + hi[k], 2.0 * mid[k], 1e-9)

    # The ocean pole tide is the smaller partner on the same C21/S21 band -- about a sixth of the
    # solid one. Swapping the two bindings would invert that ordering.
    solid = sl.solid_pole_tide(1e-6, 2e-6, 0.0)
    ocean = sl.ocean_pole_tide(1e-6, 2e-6, 0.0)
    for k in range(2):
        ratio = ocean[k] / solid[k]
        assert 0.05 < ratio < 0.5, ratio

    # The default is the 2018 secular pole, not the retired Table 7.7 one, and the two are far
    # enough apart that a silently-reverted default cannot hide behind a tolerance.
    for tide in (sl.solid_pole_tide, sl.ocean_pole_tide):
        assert tide(1e-6, 2e-6, 7.57339269184e8) == tide(
            1e-6, 2e-6, 7.57339269184e8, "secular2018"
        )
        legacy = tide(1e-6, 2e-6, 7.57339269184e8, "iers2010")
        assert far(tide(1e-6, 2e-6, 7.57339269184e8), legacy) > 1e-11, legacy
        try:
            tide(1e-6, 2e-6, 0.0, "table7.7")
        except ValueError:
            pass
        else:
            raise AssertionError(f"{tide.__name__} accepted an unknown mean pole")


def test_dense_ephemerides_interpolate_their_own_span():
    r0, v0 = circular_state(500e3, math.radians(51.6))
    span = 3600.0
    for ephemeris, expected in (
        (sl.propagate_dense(r0, v0, span, MU, 0.0), sl.DenseEphemeris),
        (sl.propagate_ias15_dense(r0, v0, span, MU, 0.0), sl.Ias15Ephemeris),
    ):
        assert isinstance(ephemeris, expected), type(ephemeris)
        assert ephemeris.span() == (0.0, span), ephemeris.span()
        accepted, rejected = ephemeris.step_counts()
        assert accepted > 0 and rejected >= 0, (accepted, rejected)
        assert far(ephemeris.eval(0.0)[0], r0) < 1e-6
        assert far(ephemeris.eval(span)[0], ephemeris.final_state()[0]) < 1e-9

        # The interpolant is not merely self-consistent: at an interior time it must agree with the
        # two-body solution the same state defines, which is what makes eval() an ephemeris rather
        # than a lookup of the nearest step.
        truth, _ = sl.kepler_propagate(r0, v0, MU, 1234.0)
        assert far(ephemeris.eval(1234.0)[0], truth) < 1e-3, far(
            ephemeris.eval(1234.0)[0], truth
        )


def test_relativistic_and_tidal_propagators_add_their_force():
    r0, v0 = circular_state(500e3, math.radians(51.6))
    span = 3600.0

    # Schwarzschild plus Lense-Thirring plus de Sitter move a LEO orbit by centimetres in an hour.
    # Both bounds matter: zero means the force never reached the propagator, metres means it
    # reached it with the wrong scale.
    point_mass = sl.propagate_dense(r0, v0, span, MU, 0.0)
    relativistic = sl.propagate_relativistic_dense(r0, v0, span, 0.0)
    shift = far(relativistic.final_state()[0], point_mass.final_state()[0])
    assert 1e-3 < shift < 1.0, shift

    # The solid Earth tide deforms the low-degree field; against the same geopotential without it,
    # an hour of LEO shows hundreds of metres to kilometres.
    geopotential = sl.propagate_geopotential_dense(r0, v0, span, 4)
    tidal = sl.propagate_geopotential_tide_dense(r0, v0, span, 4, 0, 0, 0.0)
    shift = far(tidal.final_state()[0], geopotential.final_state()[0])
    assert 1.0 < shift < 1e5, shift


def test_nrlmsise00_drag_removes_orbital_energy():
    r0, v0 = circular_state(400e3, math.radians(51.6))
    span = 6.0 * 3600.0
    start = specific_energy((r0, v0))
    drag = sl.propagate_drag_nrlmsise00_dense(r0, v0, span, 0, 0, 80, 43200.0, 0.02)
    loss = specific_energy(drag.final_state()) - start
    assert -1e5 < loss < -1.0, loss

    # A larger ballistic coefficient must lose more, and a conservative propagation none at all.
    heavier = sl.propagate_drag_nrlmsise00_dense(r0, v0, span, 0, 0, 80, 43200.0, 0.2)
    assert specific_energy(heavier.final_state()) < specific_energy(drag.final_state())
    conservative = sl.propagate_dense(r0, v0, span, MU, 0.0)
    assert abs(specific_energy(conservative.final_state()) - start) < 1.0


def test_sdtm1_drag_removes_orbital_energy_inside_its_envelope():
    (low, high), (flux_low, flux_high) = sl.sdtm1_envelope()
    assert 200.0 < low < 400.0 < high < 600.0, (low, high)
    assert 50.0 < flux_low < 150.0 < flux_high < 300.0, (flux_low, flux_high)
    r0, v0 = circular_state(400e3, math.radians(51.6))
    span = 6.0 * 3600.0
    start = specific_energy((r0, v0))
    quiet = [5.0] * 120
    drag = sl.propagate_drag_sdtm1_dense(r0, v0, span, 0, 0, 80, 43200.0, 0.02, quiet)
    loss = specific_energy(drag.final_state()) - start
    assert -1e5 < loss < -1.0, loss

    # A larger ballistic coefficient loses more, and so does a storm, which heats the thermosphere.
    heavier = sl.propagate_drag_sdtm1_dense(r0, v0, span, 0, 0, 80, 43200.0, 0.2, quiet)
    assert specific_energy(heavier.final_state()) < specific_energy(drag.final_state())
    stormy = sl.propagate_drag_sdtm1_dense(
        r0, v0, span, 0, 0, 80, 43200.0, 0.02, [5.0] * 100 + [150.0] * 20
    )
    assert specific_energy(stormy.final_state()) < specific_energy(drag.final_state())

    # The history is the storm state's input, so one that cannot be a history is refused, by name:
    # an error from the integration instead would come from a state the refusal let through.
    for bad, cadence_s, named in (
        ([], 10800.0, "ap_history is empty"),
        ([5.0, math.nan], 10800.0, "ap_history[1] is NaN"),
        (quiet, 0.0, "cadence_s is 0,"),
    ):
        try:
            sl.propagate_drag_sdtm1_dense(
                r0, v0, span, 0, 0, 80, 43200.0, 0.02, bad, cadence_s=cadence_s
            )
        except ValueError as refused:
            assert named in str(refused), f"{bad[:3]} at {cadence_s} s: {refused}"
            continue
        raise AssertionError(f"{bad[:3]} at {cadence_s} s was accepted")


def test_topocentric_azimuth_and_range_rate_use_the_local_frame():
    station = [
        RE,
        0.0,
        0.0,
    ]  # the equator on the prime meridian, so east is +y and north is +z
    for offset, expected in (([0.0, 0.0, 500e3], 0.0), ([0.0, 500e3, 0.0], 90.0)):
        sat = [station[i] + offset[i] for i in range(3)] + [0.0, 0.0, 0.0]
        close(math.degrees(sl.azimuth(sat, station, 0.0, 0.0)), expected, 1e-9)

    # Range rate is positive when opening; a purely tangential velocity opens nothing.
    up = [RE + 500e3, 0.0, 0.0]
    close(sl.range_rate([*up, 100.0, 0.0, 0.0], station), 100.0, 1e-12)
    close(sl.range_rate([*up, -100.0, 0.0, 0.0], station), -100.0, 1e-12)
    assert abs(sl.range_rate([*up, 0.0, 100.0, 0.0], station)) < 1e-9


def test_impulse_response_metrics_match_the_rectangular_window():
    # The impulse response of a rectangular window is a sinc, whose peak-sidelobe ratio is the
    # textbook -13.26 dB and whose -3 dB width is 0.886 in the sinc argument. Both are external
    # constants, so this checks the binding against something neither side of it computed.
    spacing = 0.025
    samples = 801
    magnitude = []
    for i in range(samples):
        x = (i - samples // 2) * spacing
        magnitude.append(
            1.0 if x == 0.0 else abs(math.sin(math.pi * x) / (math.pi * x))
        )
    close(sl.peak_sidelobe_ratio_db(magnitude), -13.26, 1e-3)
    close(sl.resolution_3db_samples(magnitude), 0.886 / spacing, 1e-3)


def test_interferogram_and_coherence_carry_the_phase_difference():
    first = [0.3, 1.1, -2.0, 2.8]
    second = [0.1, -0.4, 2.9, -1.7]
    re, im = sl.interferogram(
        [math.cos(p) for p in first],
        [math.sin(p) for p in first],
        [math.cos(p) for p in second],
        [math.sin(p) for p in second],
    )
    # s1 . conj(s2) has phase phi1 - phi2; conjugating the wrong operand flips the sign.
    for k in range(4):
        close(math.atan2(im[k], re[k]), sl.wrap_phase(first[k] - second[k]), 1e-12)

    rows, cols = 5, 5
    signal = [float(i % 7) for i in range(rows * cols)]
    zeros = [0.0] * (rows * cols)
    identical = sl.coherence(signal, zeros, signal, zeros, rows, cols, 1)
    assert all(c == 1.0 for c in identical), identical

    # Independent phase noise decorrelates; a coherence that stayed at one would mean the second
    # image never reached the estimator.
    noise = texture(rows * cols, 4242)
    scrambled_re = [math.cos(2.0 * math.pi * p) for p in noise]
    scrambled_im = [math.sin(2.0 * math.pi * p) for p in noise]
    decorrelated = sl.coherence(
        signal, zeros, scrambled_re, scrambled_im, rows, cols, 1
    )
    assert decorrelated[rows * cols // 2] < 0.9, decorrelated[rows * cols // 2]


def test_coherency_matrix_is_hermitian():
    samples = [
        (1.0, 0.2, 0.3, -0.1, 0.7, 0.4),
        (0.9, -0.3, 0.2, 0.5, 1.1, 0.0),
        (1.2, 0.1, -0.4, 0.2, 0.8, -0.2),
    ]
    re, im = sl.coherency_matrix(samples)
    assert len(re) == 9 and len(im) == 9

    # T3 = <k k^H> is Hermitian by construction, exactly -- a transposed row-major unpacking or a
    # dropped conjugate breaks it, and nothing else does.
    for i in range(3):
        for j in range(3):
            assert re[3 * i + j] == re[3 * j + i], (i, j)
            assert im[3 * i + j] == -im[3 * j + i], (i, j)
    trace = re[0] + re[4] + re[8]
    assert trace > 0.0, trace

    # The trace is the total scattered power: |hh|^2 + |vv|^2 + 2|hv|^2 averaged over the samples.
    power = sum(
        hr * hr + hi * hi + vr * vr + vi * vi + 2.0 * (xr * xr + xi * xi)
        for hr, hi, xr, xi, vr, vi in samples
    ) / len(samples)
    close(trace, power, 1e-12)


def test_tomosar_spectra_peak_at_the_scatterer():
    wavelength, distance = 0.056, 700e3
    baselines = [-100.0, -66.0, -33.0, 0.0, 33.0, 66.0, 100.0]
    truth = 40.0
    a = steering(baselines, truth, wavelength, distance)
    n = len(baselines)
    cov_re = [a[i][0] * a[j][0] + a[i][1] * a[j][1] for i in range(n) for j in range(n)]
    cov_im = [a[i][1] * a[j][0] - a[i][0] * a[j][1] for i in range(n) for j in range(n)]
    grid = [float(s) for s in range(-200, 201, 5)]

    for spectrum in (
        sl.bartlett_spectrum(cov_re, cov_im, baselines, wavelength, distance, grid),
        sl.capon_spectrum(cov_re, cov_im, baselines, wavelength, distance, grid, 1e-3),
        sl.sl1mmer_spectrum(
            [c[0] for c in a],
            [c[1] for c in a],
            baselines,
            wavelength,
            distance,
            grid,
            0.4,
            4000,
        ),
    ):
        assert len(spectrum) == len(grid)
        assert grid[spectrum.index(max(spectrum))] == truth, grid[
            spectrum.index(max(spectrum))
        ]


def test_persistent_scatterers_are_the_stable_pixels():
    # Two pixels over four acquisitions: the first perfectly stable, the second swinging. The
    # dispersion of a constant amplitude is exactly zero, whatever the constant.
    stack = [[1.0, 1.0], [1.0, 2.0], [1.0, 0.5], [1.0, 3.0]]
    dispersion = sl.amplitude_dispersion(stack, 2)
    assert dispersion[0] == 0.0, dispersion[0]
    assert dispersion[1] > 0.4, dispersion[1]
    assert sl.select_persistent_scatterers(dispersion, 0.1) == [0]
    assert sl.select_persistent_scatterers(dispersion, 10.0) == [0, 1]
    assert sl.select_persistent_scatterers(dispersion, -1.0) == []


def test_azimuth_fm_rate_and_range_migration_scale_as_the_geometry():
    prf, velocity, wavelength, slant, spacing = 1500.0, 7000.0, 0.056, 800e3, 1.0
    rate = sl.azimuth_fm_rate(prf, velocity, wavelength, slant, spacing)
    assert rate > 0.0, rate

    # K_a is quadratic in the platform speed and inversely linear in wavelength and slant range.
    # Any pair of these five arguments arriving swapped breaks at least one of the three laws.
    close(
        sl.azimuth_fm_rate(prf, 2.0 * velocity, wavelength, slant, spacing) / rate,
        4.0,
        1e-12,
    )
    close(
        sl.azimuth_fm_rate(prf, velocity, 2.0 * wavelength, slant, spacing) / rate,
        0.5,
        1e-12,
    )
    close(
        sl.azimuth_fm_rate(prf, velocity, wavelength, 2.0 * slant, spacing) / rate,
        0.5,
        1e-12,
    )

    # Range migration vanishes at zero Doppler and grows with it, quadratically at small offsets.
    assert sl.range_migration(0.0, prf, velocity, wavelength, slant, spacing) == 0.0
    near = sl.range_migration(150.0, prf, velocity, wavelength, slant, spacing)
    far_off = sl.range_migration(300.0, prf, velocity, wavelength, slant, spacing)
    assert 0.0 < near < far_off, (near, far_off)
    close(far_off / near, 4.0, 1e-2)


def test_two_dimensional_unwrapping_recovers_a_ramp():
    rows, cols = 12, 15
    truth = [0.4 * r + 0.3 * c for r in range(rows) for c in range(cols)]
    wrapped = [sl.wrap_phase(p) for p in truth]

    # A ramp steep enough to wrap several times, but residue-free, so path-following and branch-cut
    # must both return it exactly -- up to the global constant unwrapping cannot know.
    for unwrap in (sl.unwrap_phase_2d, sl.unwrap_phase_2d_branch_cut):
        out = unwrap(wrapped, rows, cols)
        assert len(out) == rows * cols
        offset = out[0] - truth[0]
        assert max(abs(u - offset - t) for u, t in zip(out, truth, strict=True)) < 1e-12


def test_harris_corners_finds_the_corners_of_a_square():
    height, width = 40, 40
    image = [0.0] * (height * width)
    for y in range(12, 28):
        for x in range(10, 26):
            image[y * width + x] = 1.0
    corners = sl.harris_corners(image, height, width, 0.04, 2, 0.01, 3, 50)
    assert len(corners) == 4, corners

    # (x, y) and not (y, x): the square is 10..25 across and 12..27 down, so a transposed binding
    # would put the corners at (13, 11) instead of (11, 13).
    found = sorted((round(c[0]), round(c[1])) for c in corners)
    assert found == [(11, 13), (11, 26), (24, 13), (24, 26)], found
    assert all(c[2] > 0.0 for c in corners), corners


def test_plane_sweep_recovers_a_fronto_parallel_plane():
    height, width, focal, baseline, depth = 24, 48, 100.0, 1.0, 10.0
    eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    reference_camera = sl.PinholeCamera(
        eye, [0.0, 0.0, 0.0], focal, [width / 2, height / 2]
    )
    neighbour_camera = sl.PinholeCamera(
        eye, [baseline, 0.0, 0.0], focal, [width / 2, height / 2]
    )

    # A fronto-parallel plane at `depth` seen by two cameras offset along x is a pure horizontal
    # shift of focal*baseline/depth pixels -- ten here, well separated from the other candidates.
    shift = round(focal * baseline / depth)
    reference = texture(height * width, 99)
    neighbour = [0.0] * (height * width)
    for y in range(height):
        for x in range(width):
            source = x + shift
            neighbour[y * width + x] = (
                reference[y * width + source]
                if source < width
                else reference[y * width + x]
            )

    depths = [5.0, 10.0, 20.0, 40.0]
    disparity = sl.plane_sweep_mvs(
        reference,
        reference_camera,
        neighbour,
        neighbour_camera,
        height,
        width,
        depths,
        3,
    )
    assert len(disparity) == height * width
    interior = [
        disparity[y * width + x]
        for y in range(4, height - 4)
        for x in range(4, width - shift - 4)
    ]
    assert sum(1 for d in interior if d == depth) / len(interior) > 0.6, interior[:20]


def test_sfm_tracks_triangulate_to_their_world_points():
    eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    cameras = [
        sl.PinholeCamera(eye, [0.0, 0.0, 0.0], 1000.0, [320.0, 240.0]),
        sl.PinholeCamera(eye, [5.0, 0.0, 0.0], 1000.0, [320.0, 240.0]),
    ]
    world = [(1.3, -0.7, 12.0), (-2.0, 1.5, 20.0), (0.4, 0.9, 15.0)]

    # Project known points into both views, hand back the correspondences, and demand the points.
    # A track builder that paired the wrong corners would triangulate to something else entirely.
    corners = [[(*camera.project(p), 1.0) for p in world] for camera in cameras]
    matches = [(0, i, 1, i) for i in range(len(world))]
    tracks = sl.sfm_triangulate(corners, matches, cameras)
    assert len(tracks) == len(world), tracks
    for got, want in zip(tracks, world, strict=True):
        assert got is not None
        assert far(got, want) < 1e-9, (got, want)


def test_rpc_block_adjust_removes_an_injected_bias():
    fixture = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "spacelab-earthvision",
        "tests",
        "fixtures",
        "pleiades_reunion.rpb",
    )
    with open(fixture, encoding="utf-8") as handle:
        rpc = sl.Rpc.parse_rpb(handle.read())

    # Generate control points from the RPC's own forward map, displaced by a known constant. The
    # adjustment must return exactly that constant -- not a fit to it, since the bias is uniform.
    row_bias, col_bias = 7.0, -4.0
    ground = [
        (-21.2316, 55.7119, 1295.0),
        (-21.2400, 55.7000, 1300.0),
        (-21.2200, 55.7200, 1290.0),
        (-21.2500, 55.6900, 1310.0),
    ]
    control = []
    for lat, lon, height in ground:
        row, col = rpc.forward(lat, lon, height)
        control.append((lat, lon, height, row + row_bias, col + col_bias))

    query = (-21.2350, 55.7050, 1298.0)
    raw_row, raw_col = rpc.forward(*query)
    for affine in (False, True):
        row, col = sl.rpc_block_adjust(rpc, control, query, affine)
        close(row - raw_row, row_bias, 1e-6)
        close(col - raw_col, col_bias, 1e-6)


def test_gram_schmidt_pan_sharpen_is_idempotent_on_its_own_mean():
    bands = [[1.0, 2.0, 3.0], [4.0, 1.0, 2.0], [2.0, 3.0, 1.0]]
    synthetic = [sum(b[i] for b in bands) / len(bands) for i in range(3)]

    # Component substitution replaces the first Gram-Schmidt component with the pan band. Feed it
    # the pan it would have synthesised itself and there is nothing to substitute: the bands come
    # back unchanged. A mis-shaped band/pixel unpacking cannot satisfy that.
    out = sl.gram_schmidt_pan_sharpen(bands, synthetic)
    assert len(out) == len(bands)
    for sharpened, original in zip(out, bands, strict=True):
        assert far(sharpened, original) < 1e-12, far(sharpened, original)

    # A pan carrying detail the bands do not have must change the result. It has to be detail and
    # not brightness: the pan is histogram-matched to the synthetic one first, so adding a constant
    # is removed again before substitution and would prove nothing about the wiring.
    detailed = sl.gram_schmidt_pan_sharpen(bands, [4.0, 1.0, 2.5])
    assert max(far(d, o) for d, o in zip(detailed, bands, strict=True)) > 0.1


def test_inter_satellite_link_agrees_with_the_single_station_models():
    """Both states are variables here, and one of them held still must reproduce a fixed station."""
    local = [7000e3, 0.0, 0.0, 0.0, 7546.0, 0.0]
    remote = [6800e3, 500e3, -200e3, 0.0, 0.0, 0.0]
    separation = sl.inter_satellite_range(local, remote)
    close(separation, sl.range(local, remote[:3]), 1e-12)
    close(
        sl.inter_satellite_range_rate(local, remote),
        sl.range_rate(local, remote[:3]),
        1e-12,
    )
    # Neither observable can tell which end is which: `Δr` and `Δv` both negate, so the norm and
    # the projection of one on the other are invariant. A binding that dropped an argument would
    # satisfy this by accident, which is why the fixed-station comparison above comes first.
    close(separation, sl.inter_satellite_range(remote, local), 1e-12)
    close(
        sl.inter_satellite_range_rate(local, remote),
        sl.inter_satellite_range_rate(remote, local),
        1e-12,
    )


def test_a_bistatic_link_that_folds_onto_one_station_is_the_monostatic_one_doubled():
    """Transmitter and receiver at the same place: two legs of equal length, so exactly twice."""
    sat = [7000e3, 100e3, 200e3, -30.0, 7500.0, 120.0]
    station = [6378137.0, 0.0, 0.0]
    at_rest = (station, [0.0, 0.0, 0.0], 0.0)
    folded = (station, [0.0, 0.0, 0.0], 2.0)
    close(
        sl.bistatic_range(sat, at_rest, 1.0, folded),
        2.0 * sl.range(sat, station),
        1e-12,
    )
    close(
        sl.bistatic_range_rate(sat, at_rest, 1.0, folded),
        2.0 * sl.range_rate(sat, station),
        1e-12,
    )
    # The epochs are not decoration: a reflection that precedes its own transmission is refused
    # rather than propagated as a negative-length leg.
    try:
        sl.bistatic_range(sat, at_rest, -1.0, folded)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-causal bistatic leg was accepted")


def test_tdoa_and_fdoa_are_the_two_legs_differenced():
    """Both are differences of quantities the single-station models already produce."""
    light = 299792458.0
    sat = [7000e3, 300e3, -150e3, 40.0, 7480.0, 90.0]
    first_pos = [6378137.0, 0.0, 0.0]
    second_pos = [4460e3, 2682e3, -3674e3]
    first = (first_pos, [0.0, 0.0, 0.0], 0.05)
    second = (second_pos, [0.0, 0.0, 0.0], 0.06)
    difference = sl.range(sat, second_pos) - sl.range(sat, first_pos)
    close(sl.tdoa(sat, 0.0, first, second) * light, difference, 1e-9)
    carrier = 8.4e9
    rate_difference = sl.range_rate(sat, first_pos) - sl.range_rate(sat, second_pos)
    close(
        sl.fdoa(sat, 0.0, first, second, carrier),
        carrier * rate_difference / light,
        1e-9,
    )
    # Swapping the terminals negates both, which is what says the two arguments are not the same
    # argument -- a binding that read `first` twice would return zero here and pass nothing else.
    close(sl.tdoa(sat, 0.0, second, first), -sl.tdoa(sat, 0.0, first, second), 1e-9)


def test_the_dsn_range_unit_is_the_uplink_scaled_round_trip_and_it_wraps():
    """A constant exciter makes the integral a rectangle, so the range unit is a closed form."""
    exciter = 7.15e9
    flat = [(0.0, 1e4, exciter, 0.0)]
    round_trip = 1.0
    # Moyer's conversion is `221/2d` with `d = 749` for an X-band uplink; over a flat exciter the
    # integral is `f·Δt`, so the whole observable is one product -- and this is the wiring test,
    # because a swapped band or a swapped epoch changes it and nothing else would notice.
    unwrapped = (221.0 / (2.0 * 749.0)) * exciter * round_trip
    # `C = 26` puts the modulus `2^32` above both bands' answers, so neither wraps and the ratio
    # between them is the ratio of the two conversion factors and nothing else.
    wide = sl.dsn_sequential_range(flat, "X", 100.0, 100.0 + round_trip, 26)
    close(wide, unwrapped, 1e-12)
    # An S-band uplink has denominator 221, so its factor is exactly 1/2 and the ratio against the
    # X-band answer is 749/221 -- the two bands are the same argument slot and different answers.
    s_band = sl.dsn_sequential_range(flat, "S", 100.0, 100.0 + round_trip, 26)
    close(s_band / wide, 749.0 / 221.0, 1e-12)
    # The modulus is `2^(C+6)`, so dropping `C` by one halves it; the S-band value sits between the
    # two, and the wrapped answer is the unwrapped one less exactly one modulus.
    wrapped = sl.dsn_sequential_range(flat, "S", 100.0, 100.0 + round_trip, 25)
    close(s_band - wrapped, 2.0**31, 1e-12)
    try:
        sl.dsn_sequential_range(flat, "L", 100.0, 101.0, 24)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown band was accepted")


def test_the_doppler_count_and_the_measured_frequency_sum_to_the_reference():
    """Two observables of one link, and the identity between them is exact over a flat exciter."""
    exciter = 7.15e9
    flat = [(0.0, 1e4, exciter, 0.0)]
    interval = 60.0
    turnaround = 880.0 / 749.0  # M(X, X) from Moyer's table 13-1.
    for rate in (1.0, 1.0 - 1e-5, 1.0 + 3e-6):
        counted = sl.dsn_averaged_doppler(
            flat, "X", "X", exciter, "X", interval, 100.0, 100.0 + interval * rate
        )
        instantaneous = sl.dsn_measured_frequency(flat, "X", "X", 100.0, rate)
        close(counted + instantaneous, turnaround * exciter, 1e-13)
    # The reference band is carried separately from the transmitted one, and it is not decoration:
    # referring the same receiver to S-band scales only the reference term.
    against_s = sl.dsn_averaged_doppler(
        flat, "X", "X", exciter, "S", interval, 100.0, 100.0 + interval
    )
    against_x = sl.dsn_averaged_doppler(
        flat, "X", "X", exciter, "X", interval, 100.0, 100.0 + interval
    )
    close(against_s - against_x, (880.0 / 221.0 - turnaround) * exciter, 1e-13)


def test_delta_dor_is_the_arrival_difference_less_the_quasars_own():
    """A source at infinity contributes the projected baseline and nothing else."""
    light = 299792458.0
    goldstone = [-2353621.4, -4641341.5, 3677052.3]
    canberra = [-4460894.9, 2682361.5, -3674748.5]
    first = (goldstone, [0.0, 0.0, 0.0], 0.05)
    second = (canberra, [0.0, 0.0, 0.0], 0.06)
    spacecraft = [1.2e11, 3.0e10, -8.0e9, 0.0, 0.0, 0.0]
    right_ascension, declination = 1.1, -0.4
    source = [
        math.cos(declination) * math.cos(right_ascension),
        math.cos(declination) * math.sin(right_ascension),
        math.sin(declination),
    ]
    baseline = [b - a for a, b in zip(goldstone, canberra, strict=True)]
    projected = sum(b * s for b, s in zip(baseline, source, strict=True)) / light
    observed = sl.delta_dor(
        spacecraft, 0.0, first, second, right_ascension, declination
    )
    close(observed - sl.tdoa(spacecraft, 0.0, first, second), projected, 1e-9)
    # Both halves negate when the terminals swap, so the whole observable does -- and the two
    # catalogue angles enter the source direction rather than the geometry, so an observable that
    # ignored either would still satisfy the identity above with a different `projected`.
    close(
        sl.delta_dor(spacecraft, 0.0, second, first, right_ascension, declination),
        -observed,
        1e-9,
    )


def test_the_corona_delay_is_signed_by_the_observable_and_falls_as_one_over_frequency_squared():
    """Two arguments that only a wiring error can confuse: the frequency and the sign convention."""
    au = 1.495978707e11
    solar_radius = 696e6
    impact = 10.0 * solar_radius
    receiver = [au, 0.0, 0.0]
    depth = -1.5 * au
    offset = impact * (au - depth) / math.sqrt(au * au - impact * impact)
    transmitter = [depth, offset, 0.0]
    at_x = sl.solar_corona_delay(transmitter, receiver, 8.4e9)
    assert at_x > 0.0, f"a group delay is a delay: {at_x}"
    close(sl.solar_corona_delay(transmitter, receiver, 4.2e9) / at_x, 4.0, 1e-12)
    close(
        sl.solar_corona_delay(transmitter, receiver, 8.4e9, False) / at_x, -1.0, 1e-12
    )
    # The series is the model, and doubling every coefficient doubles the column linearly.
    single = sl.solar_corona_delay(transmitter, receiver, 8.4e9, True, [(1.0e12, 2.0)])
    doubled = sl.solar_corona_delay(transmitter, receiver, 8.4e9, True, [(2.0e12, 2.0)])
    close(doubled / single, 2.0, 1e-12)
    # A line of sight through the photosphere has no honest correction, and saying so is the
    # binding's job as much as the model's.
    inside = [0.0, 0.5 * solar_radius, 0.0]
    beyond = [1e11, 0.5 * solar_radius, 0.0]
    try:
        sl.solar_corona_delay(beyond, inside, 8.4e9)
    except ValueError:
        pass
    else:
        raise AssertionError("a line of sight inside the Sun was accepted")


def _decay_case(altitude, **overrides):
    """A cannonball at `altitude` under an exponential atmosphere pinned at 300 km.

    The arguments are collected once because every lifetime entry point takes the same fifteen, and
    a test that retyped them would be testing its own transcription.
    """
    case = {
        "initial": [RE + altitude, 0.0, 0.0, 0.0, 0.0, 0.0],
        "zonals": [0.0, 0.0, -ZONALS[0]],
        "cd": 2.2,
        "area_over_mass": 0.02,
        "reference_density": 2e-11,
        "reference_radius": RE + 300e3,
        "scale_height": 60e3,
        "horizon": 400.0 * 86400.0,
        "chunk": 86400.0,
        "tolerance": 3600.0,
    }
    case.update(overrides)
    return case


def test_orbital_lifetime_separates_its_three_outcomes():
    """Re-entry, survival and breakdown are three answers, and the binding must not merge them."""
    outcome, seconds, perigee = sl.orbital_lifetime(**_decay_case(300e3))
    assert outcome == "reentry", f"a 300 km cannonball should come down: {outcome}"
    assert 0.0 < seconds < 400.0 * 86400.0, seconds
    assert perigee <= 120e3 + 1e3, f"reported at {perigee} m, threshold 120 km"
    # Same object, same atmosphere, 700 km higher: the density there is `exp(-700/60)` of the
    # reference, so it outlives the horizon -- and the second field is then the horizon itself
    # rather than a decay epoch that was never observed.
    high, horizon, _ = sl.orbital_lifetime(**_decay_case(1000e3))
    assert high == "survives", high
    close(horizon, 400.0 * 86400.0, 1e-12)
    # The density scale is the ensemble's own axis, and it acts the way the physics says: more air,
    # sooner down.
    _, denser, _ = sl.orbital_lifetime(**_decay_case(300e3, density_scale=2.0))
    assert denser < seconds, f"{denser} is not sooner than {seconds}"


def test_the_lifetime_ensemble_brackets_its_own_members():
    """Quantiles of a three-member ensemble are the three members, in order."""
    scales = [0.5, 1.0, 2.0]
    lifetimes = sorted(
        sl.orbital_lifetime(**_decay_case(300e3, density_scale=s))[1] for s in scales
    )
    quantiles, members, survivors, unresolved = sl.orbital_lifetime_ensemble(
        **_decay_case(300e3), density_scales=scales, quantiles=[0.0, 0.5, 1.0]
    )
    assert (members, survivors, unresolved) == (3, 0, 0)
    assert all(q is not None for q in quantiles), quantiles
    assert list(quantiles) == sorted(quantiles), quantiles
    close(quantiles[0], lifetimes[0], 1e-9)
    close(quantiles[-1], lifetimes[-1], 1e-9)


def test_the_ballistic_sensitivity_is_minus_one_for_a_drag_dominated_decay():
    """`T ∝ 1/(S/m)` to first order, so the log-log slope is -1 and its sign is the whole point."""
    slope = sl.ballistic_log_sensitivity(**_decay_case(300e3), fraction=0.05)
    assert abs(slope + 1.0) < 0.2, f"d ln T / d ln (S/m) = {slope}, expected about -1"


def _checksum(line):
    """The TLE's modulo-10 checksum: digits count for themselves, minus signs for one."""
    return (
        sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68]) % 10
    )


def test_the_emitted_tle_puts_every_field_in_its_own_columns():
    """The format is a column layout, and a binding's mistake is a field one column out."""
    epoch = (2008, 9, 20, 12, 25, 12.25)
    parameters = [15.72125391, 0.0006703, 51.6416, 247.4627, 130.5360, 325.0288]
    line1, line2 = sl.tle_lines(
        epoch,
        25544,
        parameters,
        drag_term=-1.1606e-4,
        international_designator="1998-067A",
        element_set_number=292,
        revolution_number=56353,
    )
    for line in (line1, line2):
        assert len(line) == 69, f"{len(line)} columns: {line!r}"
        assert int(line[68]) == _checksum(line), f"checksum: {line!r}"
    assert line1[0] == "1" and line2[0] == "2"
    assert int(line1[2:7]) == 25544 and int(line2[2:7]) == 25544
    assert line1[9:17].strip() == "98067A", line1
    assert int(line1[64:68]) == 292, line1
    # Line 2 carries the six elements, and reading them back out of their own columns is the check
    # -- a swapped pair would still produce a well-formed line with a correct checksum.
    close(float(line2[8:16]), parameters[2], 1e-9)
    close(float(line2[17:25]), parameters[3], 1e-9)
    close(float("0." + line2[26:33]), parameters[1], 1e-4)
    close(float(line2[34:42]), parameters[4], 1e-9)
    close(float(line2[43:51]), parameters[5], 1e-9)
    close(float(line2[52:63]), parameters[0], 1e-9)
    assert int(line2[63:68]) == 56353, line2
    # `B*` is printed in the format's own implied-decimal-point exponential form.
    mantissa, exponent = float(line1[53:59]), int(line1[59:61])
    close(mantissa * 1e-5 * 10.0**exponent, -1.1606e-4, 1e-4)
    # An angle outside `[0, 360)` is a printing question and not an orbit change: SGP4 reaches every
    # angle through its sine and cosine, so the emitted line reduces it rather than refusing it.
    wrapped = list(parameters)
    wrapped[5] -= 360.0
    assert sl.tle_lines(epoch, 25544, wrapped)[1][43:51] == line2[43:51]
    # A catalog number the five columns cannot hold is refused rather than truncated -- which is the
    # whole reason the number is checked before the line is built.
    try:
        sl.tle_lines(epoch, 270123, parameters)
    except ValueError:
        pass
    else:
        raise AssertionError("a six-digit catalog number was printed into five columns")


def _rinex_obs_fixture(gps_codes="C1C L1C"):
    """A minimal well-formed RINEX 3 observation file: two epochs, two constellations.

    Written column by column rather than borrowed, because the columns are the format: a fixture
    generated by the reader it tests would agree with the reader's own mistakes.
    """

    def pad(body, label):
        return f"{body:<60}{label}"

    return "\n".join(
        [
            pad(
                "     3.05           OBSERVATION DATA    M: MIXED",
                "RINEX VERSION / TYPE",
            ),
            pad(f"G    2 {gps_codes}", "SYS / # / OBS TYPES"),
            pad("R    1 C1C", "SYS / # / OBS TYPES"),
            pad("  4460894.9000  2682361.5000 -3674748.5000", "APPROX POSITION XYZ"),
            pad("    30.000", "INTERVAL"),
            pad("END", "END OF HEADER"),
            "> 2024 01 09 12 00  0.0000000  0  2",
            "G07  23629347.915 6 124144623.325 7",
            "R05  21000000.000",
            "> 2024 01 09 12 00 30.0000000  0  1",
            "G07  23629400.00016 124144900.000 7",
        ]
    )


def test_the_rinex_header_declares_the_slot_every_observation_lands_in():
    """The descriptor list is positional, and that position is the only thing naming a value."""
    text = _rinex_obs_fixture()
    header, epochs = sl.rinex_obs(text)
    version, file_system, marker, approx, antenna, interval, time_system, obs_types = (
        header
    )
    close(version, 3.05, 1e-12)
    assert file_system == "M" and marker == "", header
    assert far(approx, [4460894.9, 2682361.5, -3674748.5]) < 1e-4, approx
    assert antenna is None, antenna
    close(interval, 30.0, 1e-12)
    # No `TIME OF FIRST OBS`, and the format's default is GPS rather than "unknown".
    assert time_system == "GPS", time_system
    assert obs_types == [("G", ["C1C", "L1C"]), ("R", ["C1C"])], obs_types
    # The declared interval and the epochs' own spacing are two independent readings of the file.
    close(epochs[1][0] - epochs[0][0], interval, 1e-9)
    first, second = epochs
    assert first[1] is False and first[2] is None, first
    assert [sat for sat, _ in first[3]] == ["G07", "R05"], first
    assert first[3][0][1] == [(23629347.915, None, 6), (124144623.325, None, 7)], first
    # `R` declares one descriptor, so its record has one slot -- not two padded with `None`.
    assert first[3][1][1] == [(21000000.0, None, None)], first
    assert second[3][0][1][0] == (23629400.0, 1, 6), second
    # Reversing the declared order moves the values with it. Nothing about the data records changed,
    # so a reader matching values to descriptors by anything other than position would disagree here
    # -- and would be wrong on every real file, where `C1C` and `C1W` are the same magnitude.
    reversed_header, reversed_epochs = sl.rinex_obs(_rinex_obs_fixture("L1C C1C"))
    assert reversed_header[7][0] == ("G", ["L1C", "C1C"]), reversed_header
    assert reversed_epochs[0][3][0][1] == first[3][0][1], reversed_epochs


def test_the_observation_series_reads_one_descriptor_out_of_its_own_column():
    """The series is what the combinations and the slip detectors consume, and it is a lookup."""
    text = _rinex_obs_fixture()
    series = sl.rinex_obs_series(text, "G07", "C1C")
    assert [round(t, 6) for t, _, _ in series] == [
        round(t, 6) for t, *_ in sl.rinex_obs(text)[1]
    ]
    close(series[0][1], 23629347.915, 1e-12)
    close(series[1][1], 23629400.0, 1e-12)
    # The loss-of-lock bit is the receiver's own declaration of a slip, and it is read from the
    # column beside the value rather than inferred from it.
    assert series[0][2] is False and series[1][2] is True, series
    # `R05` is observed once. An absent satellite is absent, not zero -- collapsing the two would
    # put a step of 21000 km into the arc.
    assert len(sl.rinex_obs_series(text, "R05", "C1C")) == 1
    # Asking for a descriptor the header never declared is a different mistake from asking for one
    # that was declared and not observed, and only the first is refusable.
    for sat, code in (("R05", "L1C"), ("07", "C1C")):
        try:
            sl.rinex_obs_series(text, sat, code)
        except ValueError:
            continue
        raise AssertionError(f"{sat} {code} was accepted")


def test_the_dual_frequency_combinations_recover_the_bias_they_were_built_from():
    """Every combination here is a claim about what cancels; injecting it is the way to check."""
    light = 299792458.0
    f1 = sl.carrier_frequency("G", "1")
    f2 = sl.carrier_frequency("G", "2")
    close(f1, 1575.42e6, 1e-12)
    close(f2, 1227.60e6, 1e-12)
    # GLONASS L1 is FDMA: the frequency belongs to the satellite's channel, not to the band, so the
    # honest answer is that this function cannot give one.
    assert sl.carrier_frequency("R", "1") is None
    assert sl.carrier_frequency("G", "9") is None
    close(sl.glonass_l1_frequency(0), 1602.0e6, 1e-12)
    lambda1, lambda2, wide, narrow = sl.lane_wavelengths(f1, f2)
    close(lambda1, light / f1, 1e-12)
    close(lambda2, light / f2, 1e-12)
    close(wide, light / (f1 - f2), 1e-12)
    close(narrow, light / (f1 + f2), 1e-12)

    rho, iono1 = 22_100_000.0, 4.5
    iono2 = iono1 * (f1 / f2) ** 2
    p1, p2 = rho + iono1, rho + iono2
    n1, n2 = 5.0, 3.0
    l1, l2 = rho - iono1 + n1 * lambda1, rho - iono2 + n2 * lambda2
    # The first-order ionosphere is what the pair was formed to remove, and what is left is the
    # range that went in.
    close(sl.ionosphere_free(f1, f2, p1, p2), rho, 1e-12)
    # ... and the geometry-free combination is the part that cancelled, with nothing else in it.
    close(sl.geometry_free(p1, p2), iono1 - iono2, 1e-9)
    # All three are unit-gain on anything common to both bands -- a clock offset, a troposphere --
    # which is what makes the residues above the whole content.
    for combination in (sl.ionosphere_free, sl.wide_lane, sl.narrow_lane):
        close(combination(f1, f2, rho, rho), rho, 1e-12)
    # Melbourne-Wübbena drops geometry, clocks and ionosphere alike, so what survives an arc is the
    # wide-lane ambiguity -- exactly the integer difference that was put in, in cycles.
    close(sl.melbourne_wubbena(f1, f2, p1, p2, l1, l2, cycles=True), n1 - n2, 1e-6)
    close(sl.melbourne_wubbena(f1, f2, p1, p2, l1, l2), (n1 - n2) * wide, 1e-6)
    # GRAPHIC needs no second band, and half the phase ambiguity is the price.
    close(sl.graphic(p1, l1), rho + 0.5 * n1 * lambda1, 1e-12)
    # High frequency first is an invariant, not a convention: a swapped pair returns a negative
    # wide-lane wavelength and a sign-flipped ambiguity, so it is refused instead.
    try:
        sl.lane_wavelengths(f2, f1)
    except ValueError:
        pass
    else:
        raise AssertionError("an ascending frequency pair was accepted")


def test_the_polynomial_detector_sees_the_slip_the_first_difference_cannot():
    """Two detectors because one is not enough, and this is the case that shows why."""
    times = [30.0 * k for k in range(20)]
    curve = [0.0005 * t * t for t in times]
    jump = 1.0
    values = [v + (jump if k >= 10 else 0.0) for k, v in enumerate(curve)]
    # The clean series' own first differences run from 0.45 to 16.65, so any threshold that does not
    # fire on the curve itself is far above the jump. This detector is blind here by construction.
    plain = sl.detect_cycle_slips_geometry_free(times, values, 20.0, 60.0)
    assert plain[0] == ("arc_start", None), plain
    assert all(tag == "continuous" for tag, _ in plain[1:]), plain
    close(plain[1][1], curve[1] - curve[0], 1e-9)
    # An order-2 fit reproduces a quadratic exactly, so its prediction residual *is* the jump.
    fitted = sl.detect_cycle_slips_polynomial(times, values, 2, 5, 0.5, 60.0)
    assert [k for k, (tag, _) in enumerate(fitted) if tag == "slipped"] == [10], fitted
    close(fitted[10][1], jump, 1e-6)
    # Nothing is tested until the window fills, and the arc restarts after the break rather than
    # fitting across it.
    assert all(tag == "arc_start" for tag, _ in fitted[:5]), fitted
    assert all(tag == "arc_start" for tag, _ in fitted[11:15]), fitted
    # A gap the detector was told not to test across restarts the arc where the gap is.
    gapped = sl.detect_cycle_slips_geometry_free(
        times[:10] + [t + 7200.0 for t in times[10:]], values, 20.0, 60.0
    )
    assert gapped[10] == ("arc_start", None), gapped
    # Two series, not one interleaved list; and a window that cannot determine the polynomial would
    # fit through every sample and never fire, so it is refused at construction.
    for call in (
        lambda: sl.detect_cycle_slips_geometry_free(times, values[:-1], 20.0, 60.0),
        lambda: sl.detect_cycle_slips_polynomial(times, values, 2, 3, 0.5, 60.0),
        lambda: sl.detect_cycle_slips_geometry_free(times, values, -1.0, 60.0),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("a malformed detector call was accepted")


def test_the_ambiguity_search_reports_its_objective_in_the_callers_parametrisation():
    """LAMBDA solves a transformed problem; the answer has to come back through the transform."""
    # Strongly correlated, so the decorrelation is not the identity and a candidate that skipped the
    # inverse transform would fail the objective below rather than pass it by luck.
    covariance = [1.2, 1.1, 1.1, 1.05]
    ambiguities = [5.45, -3.10]
    determinant = covariance[0] * covariance[3] - covariance[1] * covariance[2]
    inverse = [
        covariance[3] / determinant,
        -covariance[1] / determinant,
        -covariance[2] / determinant,
        covariance[0] / determinant,
    ]

    def objective(integers):
        d = [a - b for a, b in zip(integers, ambiguities, strict=True)]
        return sum(d[i] * inverse[2 * i + j] * d[j] for i in range(2) for j in range(2))

    candidates = sl.resolve_ambiguities(ambiguities, covariance, count=4)
    assert len(candidates) == 4, candidates
    for integers, squared_norm in candidates:
        close(squared_norm, objective(integers), 1e-9)
    norms = [n for _, n in candidates]
    assert norms == sorted(norms), norms
    # `Z` is unimodular, so the decorrelation moves the ellipsoid without changing its volume: the
    # conditional variances still multiply to the determinant they started from.
    transformation, conditional = sl.ambiguity_decorrelation(ambiguities, covariance)
    close(conditional[0] * conditional[1], determinant, 1e-9)
    close(
        abs(
            transformation[0] * transformation[3]
            - transformation[1] * transformation[2]
        ),
        1.0,
        1e-12,
    )
    # Bootstrapping is conditional rounding, and with no correlation to condition on it is plain
    # rounding -- which is also the search's answer there, so the two entry points must agree.
    diagonal = [0.01, 0.0, 0.0, 0.04]
    float_pair = [5.3, -2.8]
    rounded, cost = sl.ambiguity_bootstrap(float_pair, diagonal)
    assert rounded == [5, -3], rounded
    close(cost, 0.3**2 / 0.01 + 0.2**2 / 0.04, 1e-9)
    best, best_cost = sl.resolve_ambiguities(float_pair, diagonal, count=1)[0]
    assert best == rounded, (best, rounded)
    close(best_cost, cost, 1e-9)
    # A matrix that is not a covariance is refused rather than factorised into a distance that is
    # not one, and a covariance of the wrong side is refused rather than read as far as it goes.
    for call in (
        lambda: sl.resolve_ambiguities([1.0, 2.0], [1.0, 0.0, 0.0, -1.0]),
        lambda: sl.ambiguity_bootstrap([1.0], [1.0, 0.0, 0.0, 1.0]),
        lambda: sl.ambiguity_decorrelation([], []),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("an unusable ambiguity problem was accepted")


def _drifting_pair(amplitude=10000.0, period=600.0, miss=500.0):
    """A relative trajectory whose closest approaches are known in closed form.

    Along-track the separation is `amplitude*sin(2*pi*t/period)` and across-track it is a constant
    `miss`, so the minima sit exactly where the sine crosses zero, each at distance `miss` and at the
    relative speed `amplitude*2*pi/period`. Nothing here is read back from the scan.
    """
    rate = 2.0 * math.pi / period

    def relative(t):
        return (
            [amplitude * math.sin(rate * t), miss, 0.0],
            [amplitude * rate * math.cos(rate * t), 0.0, 0.0],
        )

    return relative, rate


def test_the_screen_finds_the_approaches_it_was_given_and_says_where_it_cannot():
    """A scan is a resolution as much as a search, and both halves are the deliverable."""
    relative, rate = _drifting_pair()
    found = sl.closest_approaches(relative, 10.0, 1000.0, 30.0, 1e-4)
    assert [round(tca) for tca, _, _ in found] == [300, 600, 900], found
    for _, distance, speed in found:
        close(distance, 500.0, 1e-6)
        close(speed, 10000.0 * rate, 1e-6)
    # The step is the resolution: successive extrema here are 150 s apart, and a step that cannot
    # bracket them separately reports fewer approaches than there are. The documented failure mode,
    # pinned rather than assumed absent.
    coarse = sl.closest_approaches(relative, 10.0, 1000.0, 400.0, 1e-4)
    assert len(coarse) < len(found), coarse

    # An ephemeris that stops covering the span stops the scan there. What was found before it still
    # stands, and the caller's own exception is what surfaces -- not a `ValueError` about NaN.
    def truncated(t):
        if t > 500.0:
            raise RuntimeError("no ephemeris past 500 s")
        return relative(t)

    try:
        sl.closest_approaches(truncated, 10.0, 1000.0, 30.0, 1e-4)
    except RuntimeError as e:
        assert "500 s" in str(e), e
    else:
        raise AssertionError("the sampler's own failure was swallowed")
    for call in (
        lambda: sl.closest_approaches(relative, 10.0, 1000.0, -30.0, 1e-4),
        lambda: sl.closest_approaches(relative, 1000.0, 10.0, 30.0, 1e-4),
        lambda: sl.closest_approaches(relative, 10.0, 1000.0, 30.0, 0.0),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("an unusable scan request was accepted")


def test_the_radial_filter_separates_only_shells_that_cannot_meet():
    """Filter 1 of an all-on-all screen: a certainty in one direction and nothing in the other."""
    low, high = (RE + 400e3, RE + 420e3), (RE + 800e3, RE + 900e3)
    assert sl.radially_separated(low[0], low[1], high[0], high[1], 5000.0)
    assert sl.radially_separated(high[0], high[1], low[0], low[1], 5000.0)
    # The gap here is 380 km, so a screening distance beyond it must not separate the pair.
    assert not sl.radially_separated(low[0], low[1], high[0], high[1], 400e3)
    # Overlapping shells are never separable, whatever the threshold.
    assert not sl.radially_separated(low[0], high[1], high[0], low[1], 1.0)
    # A corrupt element set is examined rather than discarded, so the answer to a NaN is `False`.
    assert not sl.radially_separated(float("nan"), low[1], high[0], high[1], 5000.0)


def _conjunction(miss=50.0, sigma=100.0, speed=14000.0, radius=5.0):
    """A head-on conjunction with isotropic covariances, the miss vector across the relative track."""
    isotropic = [[sigma * sigma if i == j else 0.0 for j in range(3)] for i in range(3)]
    return {
        "relative_position": [0.6 * miss, 0.0, 0.8 * miss],
        "relative_velocity": [0.0, speed, 0.0],
        "primary_covariance": isotropic,
        "secondary_covariance": isotropic,
        "primary_radius": radius,
        "secondary_radius": radius,
    }


def test_the_collision_probability_is_the_encounter_plane_integral_three_ways():
    """Three routes to one integral: they must agree, and at zero miss they have a closed form."""
    case = _conjunction()
    foster, patera, chan, maximum = sl.collision_probability(**case)
    for estimate in (patera, chan):
        assert abs(estimate - foster) < 0.02 * foster, (foster, patera, chan)
    # The maximum over a scaling of the covariance is a bound, not a fourth estimate of the same
    # thing, so it sits above all three.
    assert maximum >= max(foster, patera, chan), (foster, maximum)
    # Head-on with an isotropic joint covariance `2*sigma^2` and no miss, the integral is elementary:
    # `1 - exp(-R^2 / 2s^2)`. This is the one geometry where the answer is not itself a quadrature.
    centred = _conjunction(miss=0.0)
    exact = 1.0 - math.exp(-(10.0**2) / (2.0 * 2.0 * 100.0**2))
    for estimate in sl.collision_probability(**centred)[:3]:
        close(estimate, exact, 2e-4)
    # Only the sum of the radii is a cross section, and the relative state has no preferred sign.
    close(
        sl.collision_probability(
            **(_conjunction() | {"primary_radius": 1.0, "secondary_radius": 9.0})
        )[0],
        foster,
        1e-12,
    )
    negated = _conjunction()
    negated["relative_position"] = [-c for c in negated["relative_position"]]
    negated["relative_velocity"] = [-c for c in negated["relative_velocity"]]
    close(sl.collision_probability(**negated)[0], foster, 1e-12)
    # Further away is less likely, which is the one monotonicity the whole model rests on.
    assert sl.collision_probability(**_conjunction(miss=300.0))[0] < foster


def test_the_domain_check_clears_the_encounter_it_was_built_from():
    """Feed the check its own assumption: rectilinear samples must report a rectilinear encounter."""
    case = _conjunction()
    joint = [
        [
            case["primary_covariance"][i][j] + case["secondary_covariance"][i][j]
            for j in range(3)
        ]
        for i in range(3)
    ]

    def rectilinear(scale=1.0):
        def sample(t):
            position = [
                r + v * t
                for r, v in zip(
                    case["relative_position"], case["relative_velocity"], strict=True
                )
            ]
            return (position, case["relative_velocity"]), [
                [scale * c for c in row] for row in joint
            ]

        return sample

    period = 5400.0
    flags, indicators, correction = sl.conjunction_usage_violations(
        **case, minimum_period=period, sample=rectilinear()
    )
    any_flag, npd, offset, extended, inaccuracy = flags
    assert (any_flag, npd, offset, extended, inaccuracy) == (
        False,
        False,
        False,
        False,
        False,
    ), flags
    # The curvilinear curve is the rectilinear one, so the correction it implies is unity and the
    # disagreement indicator is zero. The residual is the bisection's own width, not a modelling gap.
    close(correction, 1.0, 1e-6)
    assert indicators[2] < 1e-6, indicators
    # The period is the scale both ratios are stated on, and halving it doubles them exactly -- which
    # is what says it entered where the thresholds expect it and not somewhere else.
    _, halved, _ = sl.conjunction_usage_violations(
        **case, minimum_period=0.5 * period, sample=rectilinear()
    )
    close(halved[0], 2.0 * indicators[0], 1e-9)
    close(halved[1], 2.0 * indicators[1], 1e-9)
    # A short enough period puts the encounter outside the model's extended-duration limit, and the
    # flag must follow the indicator across the published threshold rather than sit beside it.
    short = indicators[1] * period / 0.021
    flagged, short_indicators, _ = sl.conjunction_usage_violations(
        **case, minimum_period=short, sample=rectilinear()
    )
    assert short_indicators[1] > 0.02 and flagged[3] and flagged[0], (
        short_indicators,
        flagged,
    )
    # A curvilinear trajectory whose covariance grows away from TCA is a real disagreement, and the
    # correction factor is what reports it.
    _, grown, corrected = sl.conjunction_usage_violations(
        **case, minimum_period=period, sample=rectilinear(scale=4.0)
    )
    assert grown[2] > 1e-3, grown
    assert abs(corrected - 1.0) > 1e-3, corrected

    def broken(t):
        raise RuntimeError("no propagation available")

    try:
        sl.conjunction_usage_violations(**case, minimum_period=period, sample=broken)
    except RuntimeError as e:
        assert "no propagation" in str(e), e
    else:
        raise AssertionError("the sampler's own failure was swallowed")
    try:
        sl.conjunction_usage_violations(
            **case, minimum_period=0.0, sample=rectilinear()
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a conjunction was assessed against no orbital period")


def test_the_shapiro_delay_is_symmetric_and_linear_in_the_mass():
    """The one #313 correction that is a function rather than a decorator chain."""
    r_station, altitude = RE, 500e3
    zenith = sl.shapiro_delay(r_station, r_station + altitude, altitude, MU)
    # `derivations/shapiro_delay.mac` puts this leg at 0.669 mm, against a bigfloat reference.
    close(1000.0 * zenith, 0.669, 1e-3)
    # The two endpoints enter as a sum, so the leg does not know which end emitted.
    close(
        sl.shapiro_delay(r_station + altitude, r_station, altitude, MU), zenith, 1e-12
    )
    # `2GM/c^2` is the whole prefactor, so the delay is linear in the central mass and nothing else
    # in the expression is.
    close(
        sl.shapiro_delay(r_station, r_station + altitude, altitude, 2.0 * MU),
        2.0 * zenith,
        1e-12,
    )
    # A longer path between the same endpoints grazes closer, and is delayed more.
    grazing = sl.shapiro_delay(r_station, r_station + altitude, 2.0 * altitude, MU)
    assert grazing > zenith, (grazing, zenith)


# ── Control: the loop, the delay margin, and the consensus bound ─────────────────────────────


def identity3():
    return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def rotation_x(angle):
    """Inertial->body for a rotation of `angle` about the body x axis, row-major."""
    c, s = math.cos(angle), math.sin(angle)
    return [1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c]


def matmul(a, b, n, k, m):
    """Flat row-major `n×k` times `k×m`."""
    return [
        sum(a[i * k + t] * b[t * m + j] for t in range(k))
        for i in range(n)
        for j in range(m)
    ]


def transpose(m, rows, cols):
    """Flat row-major `rows×cols` transposed."""
    return [m[i * cols + j] for j in range(cols) for i in range(rows)]


def test_latency_is_three_numbers_and_refuses_a_fractional_step():
    assert sl.latency_total(0.1, 0.02, 0.05) == 0.1 + 0.02 + 0.05
    # A whole number of steps is admitted and counted; a fraction of one is refused rather
    # than rounded, because a delay of 2.5 steps is a different problem from 2 or 3.
    assert sl.latency_steps(0.1, 0.02, 0.08, 0.05) == 4
    assert sl.latency_steps(0.0, 0.0, 0.0, 0.05) == 0
    for bad in ((0.1, 0.02, 0.07, 0.05), (0.1, 0.0, 0.0, 0.0), (-0.1, 0.0, 0.0, 0.05)):
        try:
            sl.latency_steps(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not have produced a step count")


def test_the_attitude_error_is_the_rotation_between_the_two_frames():
    zero_mrp, zero_angle = sl.attitude_error_mrp(
        identity3(), identity3(), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    )
    assert zero_mrp == [0.0, 0.0, 0.0]
    assert zero_angle == 0.0

    # `sigma = tan(theta/4)` about the rotation axis is the MRP's definition, so the principal
    # angle and the MRP norm check each other rather than a table.
    theta = 0.7
    mrp, angle = sl.attitude_error_mrp(
        rotation_x(theta), identity3(), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    )
    close(angle, theta, 1e-12)
    close(norm(mrp), math.tan(theta / 4.0), 1e-12)

    # The same rotation in the other chart. `sigma = q_v / (1 + q0)` is what relates them, so the
    # two exports check each other rather than each checking a table, and `q0 >= 0` is the short-way
    # convention that makes the principal angle unambiguous.
    q = sl.attitude_error_quaternion(
        rotation_x(theta), identity3(), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    )
    assert len(q) == 4
    assert q[0] >= 0.0
    close(norm(q), 1.0, 1e-15)
    close(2.0 * math.acos(q[0]), theta, 1e-12)
    for i in range(3):
        close(q[1 + i] / (1.0 + q[0]), mrp[i], 1e-15)


def test_both_laws_command_nothing_at_the_reference_and_a_restoring_moment_off_it():
    inertia = [4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0]
    rest = [0.0, 0.0, 0.0]
    for law in ("quaternion_pd", "mrp_lyapunov"):
        at_rest = sl.attitude_torque(
            law, 8.0, 3.0, identity3(), identity3(), rest, rest, inertia
        )
        assert at_rest == rest, (law, at_rest)

        # In the small-angle limit the commanded moment is the linearisation the margin
        # machinery is told to use, so this pins the two bindings against each other: a gain
        # that disagreed with `attitude_small_angle_gains` would break one of them.
        angle_gain, rate_gain = sl.attitude_small_angle_gains(law, 8.0, 3.0)
        theta = 1.0e-5
        torque = sl.attitude_torque(
            law, 8.0, 3.0, rotation_x(theta), identity3(), rest, rest, inertia
        )
        close(torque[0], -angle_gain * theta, 1e-4)

        spun = sl.attitude_torque(
            law, 8.0, 3.0, identity3(), identity3(), [theta, 0.0, 0.0], rest, inertia
        )
        close(spun[0], -rate_gain * theta, 1e-9)

    # The MRP law linearises to a quarter of its nominal stiffness and the quaternion law to a
    # half, so at the same gains one is exactly twice the other.
    quaternion, _ = sl.attitude_small_angle_gains("quaternion_pd", 8.0, 3.0)
    mrp, _ = sl.attitude_small_angle_gains("mrp_lyapunov", 8.0, 3.0)
    close(quaternion, 2.0 * mrp, 1e-15)

    try:
        sl.attitude_torque(
            "pid", 8.0, 3.0, identity3(), identity3(), rest, rest, inertia
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown law must be refused")


def test_the_lyapunov_function_is_the_kinetic_term_plus_the_attitude_term():
    inertia = [4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0]
    rest = [0.0, 0.0, 0.0]
    assert (
        sl.attitude_lyapunov(8.0, 3.0, identity3(), identity3(), rest, rest, inertia)
        == 0.0
    )

    # Adding a body rate adds exactly the rotational kinetic energy it carries, and nothing
    # else -- the attitude term does not see the rate.
    rate = [0.1, -0.2, 0.05]
    kinetic = 0.5 * sum(w * inertia[3 * i + i] * w for i, w in enumerate(rate))
    turned = sl.attitude_lyapunov(
        8.0, 3.0, rotation_x(0.4), identity3(), rest, rest, inertia
    )
    spun = sl.attitude_lyapunov(
        8.0, 3.0, rotation_x(0.4), identity3(), rate, rest, inertia
    )
    assert turned > 0.0
    close(spun - turned, kinetic, 1e-12)


def test_the_critically_damped_gain_round_trips_to_a_damping_ratio_of_one():
    for gain, moment in ((8.0, 4.0), (0.5, 120.0), (1e3, 1e-2)):
        derivative = sl.quaternion_pd_critically_damped(gain, moment)
        frequency, damping = sl.quaternion_pd_response(gain, derivative, moment)
        close(damping, 1.0, 1e-14)
        # `omega_n = sqrt(K/2I)`: four times the gain doubles it, four times the moment halves it.
        quadrupled, _ = sl.quaternion_pd_response(4.0 * gain, derivative, moment)
        close(quadrupled, 2.0 * frequency, 1e-14)
        heavier, _ = sl.quaternion_pd_response(gain, derivative, 4.0 * moment)
        close(heavier, 0.5 * frequency, 1e-14)
    # A non-positive moment is a domain error and not a NaN pair, the way its C twin
    # `sl_quaternion_pd_response` and every other margin binding treat it.
    for bad in (0.0, -4.0, float("nan")):
        try:
            sl.quaternion_pd_response(8.0, 3.0, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"the response accepted a moment of {bad}")


def test_the_loop_reads_the_state_it_had_one_latency_ago():
    inertia = [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0]
    rest = [0.0, 0.0, 0.0]
    step = 0.1
    lag = sl.latency_steps(0.2, 0.1, 0.0, step)
    assert lag == 3

    # The seed is the history the controller reads until the delay has elapsed. Seeding it at
    # the reference while starting the body away from it makes the wiring observable: the first
    # `lag` commanded moments must be exactly zero, and the next one must not be.
    times, angles, torques, _, _ = sl.attitude_loop_run(
        "quaternion_pd",
        8.0,
        12.0,
        0.2,
        0.1,
        0.0,
        step,
        rotation_x(0.3),
        rest,
        identity3(),
        rest,
        identity3(),
        inertia,
        40,
    )
    assert len(times) == 40
    assert len(torques) == 120
    close(times[1] - times[0], step, 1e-15)
    for n in range(lag):
        assert torques[3 * n : 3 * n + 3] == [0.0, 0.0, 0.0], n
    assert torques[3 * lag] != 0.0
    close(angles[0], 0.0, 1e-15)  # the seed's error, not the body's

    # And the loop closes: from the same start with no latency the error is driven down.
    _, undelayed, _, final_rotation, final_rate = sl.attitude_loop_run(
        "mrp_lyapunov",
        8.0,
        12.0,
        0.0,
        0.0,
        0.0,
        step,
        rotation_x(0.3),
        rest,
        rotation_x(0.3),
        rest,
        identity3(),
        inertia,
        400,
    )
    close(undelayed[0], 0.3, 1e-12)
    assert undelayed[-1] < 1e-3 * undelayed[0], (undelayed[0], undelayed[-1])
    assert norm(final_rate) < 1e-3
    # A rotation matrix stays one: `R Rᵀ` is still the identity after 400 Lie-group steps.
    product = matmul(final_rotation, transpose(final_rotation, 3, 3), 3, 3, 3)
    assert far(product, identity3()) < 1e-12, product


def test_the_delay_margin_is_the_law_s_own_linearisation():
    moment = 4.0
    for law in ("quaternion_pd", "mrp_lyapunov"):
        angle_gain, rate_gain = sl.attitude_small_angle_gains(law, 8.0, 12.0)
        direct = sl.delay_margin_of_gains(moment, angle_gain, rate_gain)
        through = sl.delay_margin_of_law(law, 8.0, 12.0, moment)
        assert direct == through, (law, direct, through)

        seconds, crossing, frequency, damping, normalised = direct
        close(normalised, seconds * frequency, 1e-14)
        # At the margin the loop rings at the crossing frequency, which is where the open loop
        # has unit gain -- so the phase the delay contributes there is exactly the phase margin.
        assert 0.0 < crossing < 10.0 * frequency
        assert damping > 0.0

    # A zero rate gain is admitted and its margin is zero: an undamped loop tolerates no delay.
    seconds, _, _, damping, _ = sl.delay_margin_of_gains(moment, 8.0, 0.0)
    assert seconds == 0.0 and damping == 0.0


def test_the_binding_axis_is_the_one_with_the_smallest_margin():
    moments = [6.0, 2.0, 9.0]
    axis, seconds, crossing = sl.worst_principal_axis(
        "quaternion_pd", 8.0, 12.0, moments
    )
    each = [sl.delay_margin_of_law("quaternion_pd", 8.0, 12.0, j)[0] for j in moments]
    assert axis == each.index(min(each)), (axis, each)
    close(seconds, min(each), 1e-15)
    assert crossing > 0.0
    try:
        sl.worst_principal_axis("quaternion_pd", 8.0, 12.0, [1.0, 0.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("a zero principal moment is not a body")


def test_the_lqr_gain_solves_its_own_riccati_equation():
    # The double integrator, where the whole check fits in 2x2 arithmetic: K must be
    # `R^-1 B^T P`, and P must annihilate `A^T P + P A - P B R^-1 B^T P + Q`.
    a = [0.0, 1.0, 0.0, 0.0]
    b = [0.0, 1.0]
    q = [1.0, 0.0, 0.0, 1.0]
    r = [1.0]
    gain, riccati, residual = sl.lqr(a, b, q, r)
    assert residual < 1e-9, residual
    assert far(gain, matmul(b, riccati, 1, 2, 2)) < 1e-12

    at_p = matmul(transpose(a, 2, 2), riccati, 2, 2, 2)
    p_a = matmul(riccati, a, 2, 2, 2)
    p_b = matmul(riccati, b, 2, 2, 1)
    quadratic = matmul(p_b, matmul(b, riccati, 1, 2, 2), 2, 1, 2)
    zero = [at_p[i] + p_a[i] - quadratic[i] + q[i] for i in range(4)]
    assert max(abs(z) for z in zero) < 1e-12, zero

    # And the Clohessy-Wiltshire plant, the shape the relative-motion controller uses: B is the
    # bottom half identity, so the three controls are the three accelerations.
    mean_motion = 2.0 * math.pi / 5400.0
    plant_a, plant_b = sl.clohessy_wiltshire_plant(mean_motion)
    assert len(plant_a) == 36 and len(plant_b) == 18
    assert plant_b[:9] == [0.0] * 9
    assert plant_b[9:] == identity3()
    gain, _, residual = sl.lqr(
        plant_a, plant_b, [1.0 if i % 7 == 0 else 0.0 for i in range(36)], identity3()
    )
    assert residual < 1e-9, residual
    assert len(gain) == 18

    try:
        sl.lqr([0.0] * 25, [0.0] * 5, [0.0] * 25, [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("an unmonomorphised shape must be refused, not guessed")


def test_the_station_keeping_law_never_increases_its_own_lyapunov_function():
    weights = [1.0e-6, 1.0e4, 1.0e4, 0.0, 0.0, 1.0e2]
    gain, limit = 1.0e-4, 1.0e-5
    elements = [7.0e6, 1.0e-3, 2.0e-4, 5.0e-2, -3.0e-2, 0.4]
    mu = MU

    assert sl.element_feedback_lyapunov(weights, gain, limit, [0.0] * 6) == 0.0
    error = [120.0, 1.0e-5, -2.0e-5, 3.0e-4, 1.0e-4, 2.0e-4]
    single = sl.element_feedback_lyapunov(weights, gain, limit, error)
    doubled = sl.element_feedback_lyapunov(
        weights, gain, limit, [2.0 * e for e in error]
    )
    close(doubled, 4.0 * single, 1e-12)  # V is quadratic in the error

    for longitude in (0.0, 1.1, 2.7, -0.9):
        command = sl.element_feedback_acceleration(
            weights, gain, limit, elements, mu, longitude, error
        )
        assert norm(command) <= limit * (1.0 + 1e-12), command
        rate = sl.element_feedback_decrease_rate(
            weights, gain, limit, elements, mu, longitude, error
        )
        assert rate <= 0.0, (longitude, rate)
    # No error, no command and no motion of V.
    assert sl.element_feedback_acceleration(
        weights, gain, limit, elements, mu, 0.3, [0.0] * 6
    ) == [0.0, 0.0, 0.0]
    assert (
        sl.element_feedback_decrease_rate(
            weights, gain, limit, elements, mu, 0.3, [0.0] * 6
        )
        == 0.0
    )


def path_graph(n):
    """The path `1 - 2 - ... - n` at unit gain, flattened row-major."""
    a = [0.0] * (n * n)
    for i in range(n - 1):
        a[i * n + i + 1] = 1.0
        a[(i + 1) * n + i] = 1.0
    return a


def test_the_laplacian_spectrum_carries_the_graph_s_own_invariants():
    # P3 in closed form: {0, 1, 3}.
    spectrum = sl.laplacian_spectrum(path_graph(3))
    for got, want in zip(spectrum, [0.0, 1.0, 3.0], strict=True):
        close(got, want, 1e-12)
    close(sl.algebraic_connectivity(path_graph(3)), spectrum[1], 1e-15)

    # For any graph the constant vector is in the kernel and the trace of L is the total edge
    # weight counted twice -- neither depends on which graph this is.
    for n in (2, 5, 8, 12):
        a = path_graph(n)
        spectrum = sl.laplacian_spectrum(a)
        assert len(spectrum) == n
        close(spectrum[0], 0.0, 1e-12)
        close(sum(spectrum), sum(a), 1e-10)
        assert spectrum[1] > 0.0  # connected

    disconnected = [0.0] * 16
    disconnected[1] = disconnected[4] = 1.0
    disconnected[11] = disconnected[14] = 1.0
    try:
        sl.laplacian_spectrum(disconnected)
    except ValueError:
        pass
    else:
        raise AssertionError("two components never reach one consensus")


def test_the_modes_are_what_the_spectrum_leaves_behind():
    adjacency = path_graph(4)
    spectrum = sl.laplacian_spectrum(adjacency)
    values, modes = sl.laplacian_modes(adjacency)
    assert values == spectrum

    # `L v = lambda v` for every mode, and the modes are orthonormal -- the two properties that
    # make them a decomposition rather than four arbitrary vectors. `L = D - A` is rebuilt here in
    # Python so the check does not run through the binding it is checking.
    degrees = [sum(adjacency[4 * i : 4 * i + 4]) for i in range(4)]
    laplacian = [
        (degrees[i] if i == j else 0.0) - adjacency[4 * i + j]
        for i in range(4)
        for j in range(4)
    ]
    for k, value in enumerate(values):
        v = modes[4 * k : 4 * k + 4]
        close(norm(v), 1.0, 1e-12)
        assert far(matmul(laplacian, v, 4, 4, 1), [value * x for x in v]) < 1e-12
    for k in range(4):
        for j in range(k + 1, 4):
            close(
                sum(modes[4 * k + i] * modes[4 * j + i] for i in range(4)), 0.0, 1e-12
            )

    # Connectivity is combinatorial, and it is a separate binding because it is the one question
    # in this family whose answer is a value rather than a refusal: every other entry point here
    # rejects a disconnected graph outright, since a repeated zero eigenvalue is not a small
    # convergence rate but no consensus at all.
    assert sl.graph_is_connected(adjacency)
    assert sl.algebraic_connectivity(adjacency) > 0.0
    split = list(adjacency)
    split[4 * 1 + 2] = 0.0
    split[4 * 2 + 1] = 0.0
    assert not sl.graph_is_connected(split)
    for refuses in (
        sl.algebraic_connectivity,
        sl.laplacian_spectrum,
        sl.laplacian_modes,
    ):
        try:
            refuses(split)
        except ValueError:
            continue
        raise AssertionError(f"{refuses.__name__} must refuse a disconnected graph")

    # And a matrix that is not symmetric is not a disconnected graph -- it is not a graph, so it
    # is a refusal from `graph_is_connected` too rather than a `False`.
    split[4 * 1 + 2] = 1.0
    try:
        sl.graph_is_connected(split)
    except ValueError:
        pass
    else:
        raise AssertionError("an asymmetric weight matrix is not a graph")


def test_the_characteristic_polynomial_annihilates_its_own_matrix():
    # Cayley-Hamilton: `A^N + c_{N-1} A^{N-1} + ... + c_0 I = 0`. That is the check with no second
    # implementation in it -- the coefficients are verified against the matrix they came from.
    a = [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        -6.0,
        -11.0,
        -6.0,
    ]  # companion of (l+1)(l+2)(l+3)
    coefficients = sl.characteristic_polynomial(a)
    assert len(coefficients) == 3
    for got, want in zip(coefficients, [6.0, 11.0, 6.0], strict=True):
        close(got, want, 1e-12)

    powers = [identity3()]
    for _ in range(3):
        powers.append(matmul(powers[-1], a, 3, 3, 3))
    total = list(powers[3])
    for k, c in enumerate(coefficients):
        for i in range(9):
            total[i] += c * powers[k][i]
    assert max(abs(v) for v in total) < 1e-9, total

    # And the reason it exists: a closed-loop plant is stable exactly when this polynomial's roots
    # are, and for the double integrator under its own LQR gain every coefficient is positive --
    # the Routh necessary condition, checkable without an eigensolver.
    gain, _, _ = sl.lqr([0.0, 1.0, 0.0, 0.0], [0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0])
    closed = [
        0.0 - (0.0 * gain[0]),
        1.0 - (0.0 * gain[1]),
        0.0 - (1.0 * gain[0]),
        0.0 - (1.0 * gain[1]),
    ]
    assert all(c > 0.0 for c in sl.characteristic_polynomial(closed))

    try:
        sl.characteristic_polynomial([1.0, 2.0, 3.0])
    except ValueError:
        pass
    else:
        raise AssertionError("a non-square matrix has no characteristic polynomial")


def test_the_delay_bound_is_the_published_one_and_the_run_respects_it():
    a = path_graph(4)
    spectrum = sl.laplacian_spectrum(a)
    bound = sl.consensus_delay_bound(a, "delayed_relative")
    # Olfati-Saber & Murray 2004: tau_max = pi / (2 lambda_max).
    close(bound, math.pi / (2.0 * spectrum[-1]), 1e-13)
    assert sl.consensus_delay_bound(a, "current_self") == math.inf

    start = [1.0, -2.0, 0.5, 0.5]
    opening = sl.consensus_disagreement(start)
    close(opening, norm([x - sum(start) / 4.0 for x in start]), 1e-15)
    assert sl.consensus_disagreement([3.0] * 4) == 0.0
    # Shifting every agent by the same amount is not disagreement.
    close(sl.consensus_disagreement([x + 7.0 for x in start]), opening, 1e-13)

    # Below the bound the disagreement collapses; above it, it grows. The mean is conserved
    # either way, because the dynamics only ever move differences.
    for delay, converges in ((0.7 * bound, True), (1.3 * bound, False)):
        final, disagreement, accepted, rejected = sl.consensus_run(
            a, "delayed_relative", delay, start, 40.0 * delay, 1e-11
        )
        assert accepted > 0 and rejected < accepted
        close(sum(final) / 4.0, sum(start) / 4.0, 1e-8)
        if converges:
            assert disagreement < 1e-3 * opening, (delay, disagreement)
        else:
            assert disagreement > opening, (delay, disagreement)

    try:
        sl.consensus_run(a, "delayed_relative", 0.0, start, 1.0, 1e-9)
    except ValueError:
        pass
    else:
        raise AssertionError("a zero delay is an ODE, not a DDE")


def test_the_breakpoints_are_where_the_solution_stops_being_smooth():
    # One delay: the jumps propagate to tau, 2tau, ... and nowhere else, and the endpoint is
    # not one of them -- these are the interior mesh points, and `t1` is the caller's already.
    got = sl.delay_breakpoints([0.25], 0.0, 1.0, 8)
    for value, want in zip(got, [0.25, 0.5, 0.75], strict=True):
        close(value, want, 1e-15)
    assert all(b < c for b, c in pairwise(got))

    # Two incommensurate delays: every sum of at most `levels` of them, so the count grows and
    # the single-delay set stays inside it.
    two = sl.delay_breakpoints([0.25, 0.4], 0.0, 1.0, 3)
    assert len(two) > len(sl.delay_breakpoints([0.25], 0.0, 1.0, 3))
    assert all(b < c for b, c in pairwise(two))
    assert all(0.0 < b < 1.0 for b in two)

    try:
        sl.delay_breakpoints([0.0], 0.0, 1.0, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("a zero delay has no breakpoints to propagate")


def test_threads_get_the_serial_answer_and_leave_the_gil_off():
    """On a free-threaded build the module must run truly parallel.

    PyO3 has declared modules free-threading-safe by default since 0.28, and the bridge has no
    `unsafe`. A `gil_used = true` anywhere in the module tree would make a free-threaded
    interpreter re-enable the GIL at import with one warning -- easy to miss, and every threaded
    caller would silently serialise. Built that way, this test fails. On a GIL build the second
    assertion is vacuous and the first still checks that concurrent calls do not share state.
    """
    from concurrent.futures import ThreadPoolExecutor

    def work(k):
        r, v = sl.kepler_propagate(
            [7_000e3, 0.0, 0.0], [0.0, 7_546.0, 0.0], MU, 60.0 * k
        )
        return r, v, sl.geopotential_acceleration(20, r)

    serial = [work(k) for k in range(64)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        threaded = list(pool.map(work, range(64)))
    assert threaded == serial
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is not None and sysconfig.get_config_var("Py_GIL_DISABLED"):
        assert not is_gil_enabled(), "importing spacelab re-enabled the GIL"


def test_every_export_is_named_by_a_test():
    """No public binding may exist without a test file naming it.

    Naming is not testing, and this does not pretend otherwise -- it is a tripwire, not a proof. It
    exists because the failure it guards against already happened: 36 of the module's exports had
    never been mentioned by any test, and nothing reported that until they were counted.
    """
    here = os.path.dirname(__file__)
    sources = ""
    for name in ("smoke.py", "bindings.py"):
        with open(os.path.join(here, name), encoding="utf-8") as handle:
            sources += handle.read()
    unnamed = sorted(
        name for name in dir(sl) if not name.startswith("_") and name not in sources
    )
    assert not unnamed, f"{len(unnamed)} exports no test names: {unnamed}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("all binding tests passed")
