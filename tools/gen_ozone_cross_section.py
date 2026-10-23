#!/usr/bin/env python3
"""The ozone absorption cross-section, slit-convolved, for the RadCalNet comparison.

`radiative_transfer::Atmosphere` is one grey scattering slab with no gaseous absorption, so a
comparison against MODTRAN -- which carries every absorber -- can only be made where the gases are
absent, or where the absorber can be applied from outside as a two-way exponential. Ozone is the one
absorber for which the second is honest: the Chappuis band is a smooth continuum with no line
structure, and the layer sits at 20-25 km, above both the aerosol and most of the Rayleigh column.
So `exp(-tau_3(lambda)(1/mu_s + 1/mu_v))` above the slab is the right shape, and this file supplies
the `sigma_3(lambda)` in it.

That is a strict strengthening of `docs/verification.md` Section 5.1, where the same absorber is
carried as **one fitted scalar per wavelength**. Here the column is RadCalNet's own measurement and
the cross-section is published, so nothing about the ozone is free.

**Provenance.**

  * Data: Serdyuchenko, A., Gorshelev, V., Weber, M., Chehade, W., and Burrows, J. P.: *High spectral
    resolution ozone absorption cross-sections -- Part 2: Temperature dependence*, Atmos. Meas. Tech.
    **7**, 625-636 (2014), DOI 10.5194/amt-7-625-2014. The "2013 version" files, 213-1100 nm at
    0.01 nm.
  * Served by: the MPI-Mainz UV/VIS Spectral Atlas of Gaseous Molecules of Atmospheric Interest,
    <https://www.uv-vis-spectral-atlas-mainz.org>, which asks to be cited as Keller-Rudek, H.,
    Moortgat, G. K., Sander, R., and Soerensen, R.: *The MPI-Mainz UV/VIS spectral atlas of gaseous
    molecules of atmospheric interest*, Earth Syst. Sci. Data **5**, 365-373 (2013), DOI
    10.5194/essd-5-365-2013. Both citations are carried into the fixture header.
  * Units: cm^2 per molecule, against wavelength in nm.

**Convolved, because the comparison is against convolved data.** RadCalNet's MODTRAN deck (Card 4 of
the site default parameters, `V1: 400 V2: 2500 DV: 10 FWHM: 10 FLAGS: NTAA`) reports on a 10 nm grid
through a **triangular slit of 10 nm FWHM**, so a cross-section read at the band centre would be a
different quantity from the reflectance it multiplies. Each value here is the triangular-slit average
over the 20 nm base.

**Three choices, each costed rather than argued.** Every one of them is reported as what it does to
the *two-way transmittance* at the largest ozone column in the RadCalNet fixture, because that is the
quantity the comparison's residual is in. A percentage of a cross-section that is itself at the
measurement's noise floor reads alarming and means nothing -- the first draft of this script reported
"253 %" for a term worth 1e-3.

  * **Slit-averaging the cross-section rather than the transmittance.** MODTRAN convolves the
    radiance, so the operation these values compose with is not the one it performs. Measured, by
    doing both: **8.86e-06**.
  * **Temperature.** The atlas publishes 193-293 K in 10 K steps and the absorbing layer is
    stratospheric, near 220 K, not the 293 K a room-temperature file would give. Both are written and
    the test uses 223 K; the difference is **1.12e-03**.
  * **Air against vacuum.** The atlas states its wavelengths are in air unless marked otherwise. The
    shift near the Chappuis band is about 0.17 nm, 1.7 % of the slit width, and costs **1.45e-04**.

**The tabulation runs negative in the Wulf-band tail**, where the cross-section is below the
measurement's own noise, and a negative cross-section is not a small error but an amplifier. Those
are clamped to zero, as is everything past the 1100 nm end of the tabulation; the most negative slit
average anywhere is -9.4e-26 cm^2, a two-way optical depth of 2e-06, which is the bound on what the
clamp discards.

    uv run python tools/gen_ozone_cross_section.py

writes `crates/spacelab-earthvision/tests/fixtures/ozone_cross_section.txt`. Needs the network, and
nothing else -- stdlib only, no credentials.
"""

from __future__ import annotations

import math
import pathlib
import urllib.parse
import urllib.request

ATLAS = "https://uv-vis-spectral-atlas-mainz.org/uvvis_data/cross_sections/Ozone/"
FILE = "O3_Serdyuchenko(2014)_{temperature}K_213-1100nm(2013 version).txt"

OUT = pathlib.Path(__file__).resolve().parents[1] / (
    "crates/spacelab-earthvision/tests/fixtures/ozone_cross_section.txt"
)

# RadCalNet's own published grid, 400-2500 nm at 10 nm -- the grid of `radcalnet_absorption.txt`,
# which the 14 wavelengths of `radcalnet_toa.txt` are a subset of. Tabulating all of it rather than
# the 14 is what lets a comparison run at every wavelength RadCalNet publishes: ozone is the only
# per-wavelength input that was not already available across the whole grid, and the Chappuis band
# sits at 450-750 nm, squarely inside the part a 14-point table would have had to skip. Kept here
# rather than imported so each generator is runnable on its own; the Rust test asserts the two agree.
WAVELENGTHS_NM = [400 + 10 * index for index in range(211)]

# Stratospheric first, room temperature second. The order is the fixture's column order.
TEMPERATURES_K = [223, 293]

# MODTRAN Card 4 of RadCalNet's site default parameters: FWHM 10 nm, FLAGS 'NTAA' -- nanometres,
# triangular slit, absolute width. A triangle of 20 nm base has 10 nm FWHM.
SLIT_FWHM_NM = 10.0

# The atlas tabulation stops here. Beyond it the fixture carries a zero cross-section, and the bound
# on what that discards is the value measured at this wavelength (printed, and in the header).
ATLAS_MAX_NM = 1100.0

# Air-vacuum shift near the Chappuis band, used only to print how little the convolved value moves.
AIR_VACUUM_SHIFT_NM = 0.17

# 1 DU is 0.01 mm of pure ozone at 273.15 K and 1013.25 hPa, so it is the Loschmidt constant
# (CODATA 2018, 2.686 780 111e25 m^-3) times 1e-5 m, in cm^-2.
DOBSON_PER_CM2 = 2.686780111e25 * 1.0e-5 * 1.0e-4

# The largest ozone column in radcalnet_toa.txt, used only to say which wavelengths the absorber
# reaches at all -- the diagnostics below are meaningless where sigma is at its own noise floor.
REFERENCE_COLUMN_DU = 415.0

# Below this two-way optical depth at `REFERENCE_COLUMN_DU` the ozone term is under the fixture's
# quantisation (RadCalNet reports reflectance to 1e-4 on a signal of order 0.3) and the tabulated
# cross-section is at its own measurement noise, where a ratio diagnostic says nothing.
REACH_FLOOR = 1.0e-3


def fetch(temperature: int) -> list[tuple[float, float]]:
    """Wavelength (nm) and cross-section (cm^2) pairs, ascending, from the atlas."""
    url = ATLAS + urllib.parse.quote(FILE.format(temperature=temperature))
    with urllib.request.urlopen(url, timeout=180) as response:
        text = response.read().decode("ascii")
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2:
            rows.append((float(fields[0]), float(fields[1])))
    if len(rows) < 10_000:
        raise SystemExit(
            f"{url}: {len(rows)} rows, expected the full 0.01 nm tabulation"
        )
    return rows


def slit_average(
    rows: list[tuple[float, float]], centre: float, shift: float = 0.0
) -> float:
    """Triangular-slit average of the cross-section, `SLIT_FWHM_NM` wide, centred on `centre`."""
    half_base = SLIT_FWHM_NM
    weighted = 0.0
    total = 0.0
    for wavelength, sigma in rows:
        offset = abs(wavelength + shift - centre)
        if offset >= half_base:
            continue
        weight = 1.0 - offset / half_base
        weighted += weight * sigma
        total += weight
    if total == 0.0:
        raise SystemExit(f"no tabulated points within {half_base} nm of {centre} nm")
    return weighted / total


def reach(sigma: float) -> float:
    """Two-way ozone optical depth at nadir through `REFERENCE_COLUMN_DU`, the absorber's largest bite."""
    return 2.0 * sigma * REFERENCE_COLUMN_DU * DOBSON_PER_CM2


def transmittance(sigma: float) -> float:
    """The two-way transmittance that optical depth produces -- the quantity the residual is in."""
    return math.exp(-reach(sigma))


def transmittance_inside_slit(rows: list[tuple[float, float]], centre: float) -> float:
    """Convolve the *transmittance*, not the cross-section.

    Exponentiating a slit-averaged cross-section is not the same operation as slit-averaging the
    transmittance, and MODTRAN does the second. Doing both is the only honest way to say how much
    that matters, rather than arguing that the band is smooth enough.
    """
    weighted = 0.0
    total = 0.0
    for wavelength, sigma in rows:
        offset = abs(wavelength - centre)
        if offset >= SLIT_FWHM_NM:
            continue
        weight = 1.0 - offset / SLIT_FWHM_NM
        weighted += weight * transmittance(max(0.0, sigma))
        total += weight
    return weighted / total


def main() -> None:
    tabulations = {t: fetch(t) for t in TEMPERATURES_K}
    reference = tabulations[TEMPERATURES_K[0]]

    # The tabulation runs negative in the Wulf-band tail, where the cross-section is below the
    # measurement's own noise. A negative cross-section is not a small error, it is an amplifier, so
    # the fixture clamps; the smallest slit average anywhere on the tabulation is the floor these
    # data can resolve, and the bound on what the clamp and the zeros beyond 1100 nm discard.
    floor = min(
        slit_average(reference, float(centre))
        for centre in range(
            int(WAVELENGTHS_NM[0]), int(ATLAS_MAX_NM - SLIT_FWHM_NM) + 1, 10
        )
    )

    # All three diagnostics are reported as a change in two-way transmittance, because that is the
    # quantity the comparison's residual is in. A percentage of a cross-section that is itself at the
    # noise floor reads alarming and means nothing.
    worst_order = 0.0
    worst_shift = 0.0
    worst_temperature = 0.0
    rows = []
    for wavelength in WAVELENGTHS_NM:
        if wavelength + SLIT_FWHM_NM > ATLAS_MAX_NM:
            rows.append((wavelength, [0.0] * len(TEMPERATURES_K)))
            continue
        sigmas = [
            max(0.0, slit_average(tabulations[t], float(wavelength)))
            for t in TEMPERATURES_K
        ]
        rows.append((wavelength, sigmas))
        base = transmittance(sigmas[0])
        worst_order = max(
            worst_order,
            abs(transmittance_inside_slit(reference, float(wavelength)) - base),
        )
        worst_temperature = max(
            worst_temperature, abs(transmittance(sigmas[-1]) - base)
        )
        shifted = slit_average(reference, float(wavelength), AIR_VACUUM_SHIFT_NM)
        worst_shift = max(worst_shift, abs(transmittance(max(0.0, shifted)) - base))

    reached = [w for w, s in rows if reach(s[0]) >= REACH_FLOOR]

    lines = [
        "# Ozone absorption cross-section, convolved to RadCalNet's 10 nm FWHM triangular slit.",
        "# Generated by tools/gen_ozone_cross_section.py -- do not edit.",
        "#",
        "# Serdyuchenko, Gorshelev, Weber, Chehade and Burrows, 'High spectral resolution ozone",
        "# absorption cross-sections - Part 2: Temperature dependence', Atmos. Meas. Tech. 7,",
        "# 625-636 (2014), doi:10.5194/amt-7-625-2014, '2013 version' files, 213-1100 nm at 0.01 nm.",
        "# Served by the MPI-Mainz UV/VIS Spectral Atlas of Gaseous Molecules of Atmospheric",
        "# Interest, https://www.uv-vis-spectral-atlas-mainz.org, which asks to be cited as",
        "# Keller-Rudek, Moortgat, Sander and Soerensen, Earth Syst. Sci. Data 5, 365-373 (2013),",
        "# doi:10.5194/essd-5-365-2013. Both citations are the condition of use.",
        "#",
        "# Units: cm^2 per molecule. Wavelengths in air, as the atlas tabulates them.",
        "#",
        f"# Ozone reaches {len(reached)} of these {len(rows)} wavelengths at all: at the largest column",
        f"# in radcalnet_toa.txt ({REFERENCE_COLUMN_DU:.0f} DU) the other {len(rows) - len(reached)} carry",
        f"# a two-way optical depth under {REACH_FLOOR:.0e}, below RadCalNet's own 1e-4 reflectance",
        "# quantisation.",
        "#",
        "# Three choices that would otherwise be assumptions, each measured as what it does to the",
        f"# two-way transmittance at {REFERENCE_COLUMN_DU:.0f} DU and nadir -- the quantity the",
        "# comparison's residual is in, rather than a percentage of a cross-section that may itself be",
        "# at its noise floor:",
        f"#   * slit-averaging the transmittance instead of the cross-section: {worst_order:.2e}. That",
        "#     is the operation MODTRAN actually performs, and it is not the one this file's values",
        "#     compose with; this is the whole error of the swap.",
        f"#   * 293 K instead of 223 K: {worst_temperature:.2e}. The Chappuis band is a continuum and",
        "#     barely notices the temperature of the layer. Both are written; the test uses 223 K,",
        "#     which is where the absorbing column sits.",
        f"#   * an air-to-vacuum shift of {AIR_VACUUM_SHIFT_NM} nm: {worst_shift:.2e}, so the atlas's",
        "#     air/vacuum convention does not matter at this slit width.",
        "#",
        f"# Beyond {ATLAS_MAX_NM:.0f} nm the tabulation ends and the cross-section is written as zero;",
        "# so is any slit average that comes out negative, which happens in the Wulf-band tail where",
        "# the cross-section is under its own measurement noise. The smallest slit average anywhere on",
        f"# this tabulation is {floor:.3e} cm^2, a two-way optical depth of {abs(reach(floor)):.1e} at",
        f"# {REFERENCE_COLUMN_DU:.0f} DU: that is the floor these data resolve, and the bound on what",
        "# the clamp and the zeros discard.",
        "#",
        "# wavelength_nm  sigma_223K_cm2  sigma_293K_cm2",
    ]
    for wavelength, sigmas in rows:
        lines.append(f"{wavelength:13d} {sigmas[0]:16.6e} {sigmas[1]:15.6e}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"{OUT}: {len(rows)} wavelengths, {len(reached)} reached by ozone")
    print(
        f"  transmittance cost -- convolution order {worst_order:.2e}, temperature "
        f"{worst_temperature:.2e}, air/vacuum {worst_shift:.2e}; tabulation floor {floor:.3e} cm^2"
    )


if __name__ == "__main__":
    main()
