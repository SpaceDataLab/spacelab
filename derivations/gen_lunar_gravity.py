#!/usr/bin/env -S uv run --quiet python
"""Generate the embedded lunar spherical-harmonic gravity table from a PDS SHADR product.

The Moon's field is not the Earth's field with different numbers: it is an order of magnitude
rougher relative to its own mean, the degree-2 terms carry the frozen tidal bulge rather than a
rotational flattening, and — the part that bites — it is defined in the **principal-axis** frame,
which is not the mean-Earth frame most lunar maps use. All three facts are in the emitted header
so the consumer cannot lose them.

Default model is GRGM1200A (Lemoine et al. 2014, GRL, doi:10.1002/2014GL060027), the GSFC degree-1200
solution from the complete GRAIL data set. NASA PDS products are US Government works, not subject to
copyright, so unlike HWM14 or NRLMSIS this one can be embedded rather than only fetched.

Provenance: the values are DATA read from the authoritative PDS `.tab`, emitted as Python `repr()`
decimals — the shortest string that round-trips to the same f64 — so the committed literal is the
nearest f64 to the source value.

The transfer is a byte-range request, not a full download. The product is FIXED_LENGTH with 122-byte
records, so the prefix holding degrees up to `degree` is computable exactly and is under a megabyte
against the full 88 MB.

Run:  timeout 300 uv run --no-project python derivations/gen_lunar_gravity.py [MODEL] [degree]
"""

from __future__ import annotations

import math
import pathlib
import sys
import urllib.request

PDS = "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr"

# PDS SHADR gravity products. `frame` is the body-fixed frame the coefficients are referenced to,
# which for every GRAIL solution is the principal axes of the ephemeris named in `pa_ephemeris`.
MODELS = {
    "GRGM1200A": {
        "file": "gggrx_1200a_sha.tab",
        "pa_ephemeris": "DE430",
        "reference": "Lemoine et al. 2014, GRL 41, doi:10.1002/2014GL060027",
    },
    "GRGM900C": {
        "file": "gggrx_0900c_sha.tab",
        "pa_ephemeris": "DE421",
        "reference": "Lemoine et al. 2013, JGR-Planets 118, doi:10.1002/jgre.20118",
    },
    "GL1500E": {
        "file": "jggrx_1500e_sha.tab",
        "pa_ephemeris": "DE430",
        "reference": "Konopliv et al. 2014, GRL 41, doi:10.1002/2014GL060569",
    },
}

RECORD_BYTES = 122
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates/spacelab-propagation/src/lunar_coeffs.rs"
)


def prefix_bytes(degree: int) -> int:
    """Bytes covering the header record plus every coefficient record of degree <= `degree`.

    The product runs n from 1, one record per (n, m) with 0 <= m <= n, so the count through degree
    N is sum_{n=1..N} (n + 1). One record of slack absorbs the file starting where its own label
    says it does not (the label declares the coefficients at record 3; they begin at record 2).
    """
    records = 1 + sum(n + 1 for n in range(1, degree + 1)) + 1
    return records * RECORD_BYTES


def icgem(
    model: str,
    spec: dict[str, str],
    gm: float,
    re: float,
    degree: int,
    coeffs: list[tuple[int, int, float, float]],
) -> str:
    """Render the same coefficients as an ICGEM `.gfc`, for feeding an external propagator.

    Written for NEPTUNE, which carries AIUB-GRL350A and reads it as an ordinary `.gfc`. Comparing
    two propagators on two different GRAIL solutions measures the solutions as much as the code, so
    the oracle run is given *these* coefficients -- the ones spacelab embeds -- under the filename
    NEPTUNE fixes. Its reader takes GM and the reference radius from this header rather than from
    its own constants, so both sides then agree on all three.

    Three things its reader requires and ICGEM does not: every row carries formal-error columns
    (emitted as zeros, since the truncation drops the product's), degrees 0 and 1 are present (C00
    is the central term, and leaving it out would delete the monopole), and no line before
    `end_of_head` may contain the substrings it scans for.
    """
    head = [
        "begin_of_head ==================================================================",
        "product_type                  gravity_field",
        "body                          moon",
        f"modelname                     {model}",
        f"gravity_constant              {gm!r}",
        f"radius                        {re!r}",
        f"max_degree                    {degree}",
        "errors                          no",
        "norm                          fully_normalized",
        "tide_system                     zero_tide",
        "key    L    M         C                   S             sigma C      sigma S",
        "end_of_head ====================================================================",
    ]
    rows = [
        "gfc    0    0  1.000000000000E+00  0.000000000000E+00  0.0E+00  0.0E+00",
        "gfc    1    0  0.000000000000E+00  0.000000000000E+00  0.0E+00  0.0E+00",
        "gfc    1    1  0.000000000000E+00  0.000000000000E+00  0.0E+00  0.0E+00",
    ]
    rows += [
        f"gfc {n:4d} {m:4d} {c: .18E} {s: .18E}  0.0E+00  0.0E+00"
        for (n, m, c, s) in coeffs
    ]
    provenance = [
        f"# {model} ({spec['reference']}), truncated to degree {degree}.",
        "# GENERATED by derivations/gen_lunar_gravity.py from the PDS SHADR product"
        f" {spec['file']}.",
        f"# Referenced to the Moon's principal axes as realised by JPL {spec['pa_ephemeris']}.",
    ]
    return "\n".join(provenance + head + rows) + "\n"


def main() -> int:
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    gfc_out = next(
        (
            arg.removeprefix("--icgem=")
            for arg in sys.argv[1:]
            if arg.startswith("--icgem=")
        ),
        None,
    )
    model = args[0] if args else "GRGM1200A"
    degree = int(args[1]) if len(args) > 1 else 40
    if model not in MODELS:
        print(f"unknown model {model!r}; known: {', '.join(MODELS)}")
        return 2
    spec = MODELS[model]
    url = f"{PDS}/{spec['file']}"
    want = prefix_bytes(degree)

    print(f"fetching {model} from PDS, first {want} bytes (degrees <= {degree}) …")
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{want - 1}"})
    with urllib.request.urlopen(request, timeout=180) as resp:
        if resp.status != 206:
            print(
                f"  server ignored the range request (status {resp.status}); refusing 88 MB"
            )
            return 1
        blob = resp.read().decode("ascii")

    records = [line for line in blob.replace("\r", "").split("\n") if line.strip()]
    head = [tok.strip() for tok in records[0].split(",")]
    re_km, gm_km3, _sigma_gm, model_degree, model_order, normalized = (
        float(head[0]),
        float(head[1]),
        float(head[2]),
        int(head[3]),
        int(head[4]),
        int(head[5]),
    )
    assert normalized == 1, f"coefficients are not normalized (state {normalized})"
    assert model_order == model_degree, f"non-square field {model_degree}x{model_order}"

    coeffs: list[tuple[int, int, float, float]] = []
    for record in records[1:]:
        field = [tok.strip() for tok in record.split(",")]
        if len(field) < 4:
            continue
        n, m = int(field[0]), int(field[1])
        if n > degree:
            break
        if (
            n >= 2
        ):  # C00 = 1 is set by the constructor; the degree-1 terms are zero by definition
            coeffs.append((n, m, float(field[2]), float(field[3])))

    # J2 and C22 are the two numbers the lunar field is known by, and they are the check that the
    # frame and the normalisation both came through: the Moon's J2 is ~2.03e-4, a fifth of the
    # Earth's, and its C22 is ~2.2e-5 — enormous next to the Earth's 2.4e-6, because it is a frozen
    # tidal bulge pointed at the Earth rather than a rotational figure. A field that arrived in the
    # mean-Earth frame instead of the principal axes would show it here first.
    c20 = next(c for (n, m, c, _s) in coeffs if (n, m) == (2, 0))
    c22_bar = next(c for (n, m, c, _s) in coeffs if (n, m) == (2, 2))
    s22_bar = next(s for (n, m, _c, s) in coeffs if (n, m) == (2, 2))
    c21_bar = next(c for (n, m, c, _s) in coeffs if (n, m) == (2, 1))
    s21_bar = next(s for (n, m, _c, s) in coeffs if (n, m) == (2, 1))
    j2 = -math.sqrt(5.0) * c20
    # J2 comes straight off the normalized C20, but C22 does not: the textbook 2.238e-5 is the
    # *unnormalized* number, and reaching it needs N22 = sqrt(2 * 5 * 0! / 4!) = sqrt(5/12).
    # Comparing the normalized coefficient against the textbook value is the mistake this exists
    # to stop, and it is a 55 % one.
    c22 = c22_bar * math.sqrt(5.0 / 12.0)
    print(
        f"  GM = {gm_km3:.10f} km^3/s^2   Re = {re_km:.3f} km   degree {model_degree}"
    )
    print(f"  C20 = {c20:.10e} -> J2 = {j2:.6e} (expect ~2.032e-4)")
    print(f"  C22bar = {c22_bar:.10e} -> C22 = {c22:.6e} (expect ~2.238e-5)")
    assert abs(j2 - 2.032e-4) < 1e-6, f"J2 sanity failed: {j2}"
    assert abs(c22 - 2.238e-5) < 1e-8, f"C22 sanity failed: {c22}"

    # In a principal-axis frame the degree-2 inertia tensor is diagonal, so C21, S21 and S22 vanish.
    # They do not vanish exactly here, and that residual is a measurement rather than a defect: the
    # PA frame is realised by the ephemeris' libration model, and the fit keeps whatever it could
    # not absorb into it. So report the size instead of asserting a zero the data does not contain.
    # It is still the check that the frame is the one the label claims -- rotating into the
    # mean-Earth frame would put ~1e-2 of C22 into these terms, two orders larger.
    tilt = max(abs(c21_bar), abs(s21_bar), abs(s22_bar)) / abs(c22_bar)
    print(f"  C21 = {c21_bar:.4e}  S21 = {s21_bar:.4e}  S22 = {s22_bar:.4e}")
    print(
        f"  worst off-diagonal degree-2 term is {tilt:.2e} of C22bar -- principal axes to that"
    )
    assert tilt < 1.0e-4, (
        f"degree-2 tensor is not diagonal to 1e-4 ({tilt:.2e}) -- wrong frame?"
    )
    print(f"  parsed {len(coeffs)} coefficients (degrees 2..{degree})")

    gm = gm_km3 * 1.0e9  # km^3/s^2 -> m^3/s^2
    re = re_km * 1.0e3

    if gfc_out is not None:
        target = pathlib.Path(gfc_out)
        target.write_text(icgem(model, spec, gm, re, degree, coeffs))
        print(f"  wrote {target}  ({len(coeffs) + 3} gfc rows)")
        return 0

    lines = [
        "//! Embedded lunar spherical-harmonic gravity coefficients (fully normalized), truncated.",
        "//!",
        f"//! GENERATED by `derivations/gen_lunar_gravity.py` from the PDS SHADR product `{spec['file']}`",
        f"//! ({model}, {spec['reference']}). Do not edit by hand.",
        "//!",
        "//! **The coefficients are referenced to the Moon's principal-axis frame**, as realised by",
        f"//! JPL {spec['pa_ephemeris']} — not to the mean-Earth/mean-rotation frame lunar cartography uses. The",
        "//! two differ by a fixed rotation of about 0.03 degrees, which is 900 m on the surface; evaluating",
        "//! this field at a mean-Earth position is a silent kilometre-scale error, not a rounding one.",
        "//! See [`crate::lunar::PA_TO_ME`].",
        "//!",
        "//! `MODEL_GM` and `MODEL_RE` are the model's own defining constants, read from the product's",
        "//! header record rather than from its PDS label — the label's prose quotes a different GM",
        "//! (4902.80011526323 against the file's value below) and the file is what the coefficients",
        "//! were fitted with.",
        "",
        "#![allow(clippy::unreadable_literal, clippy::doc_markdown)] // generated numeric data",
        "",
        "/// Name of the embedded lunar gravity model.",
        f'pub const MODEL_NAME: &str = "{model}";',
        "/// Model gravitational parameter, m³/s².",
        f"pub const MODEL_GM: f64 = {gm!r};",
        "/// Model reference radius, m.",
        f"pub const MODEL_RE: f64 = {re!r};",
        "/// Highest degree/order in [`MODEL_COEFFS`].",
        f"pub const MODEL_MAX_DEGREE: usize = {degree};",
        "",
        "/// Fully-normalized `(n, m, C̄_nm, S̄_nm)` for degrees 2..=MODEL_MAX_DEGREE.",
        "#[rustfmt::skip]",
        "pub const MODEL_COEFFS: &[(usize, usize, f64, f64)] = &[",
    ]
    lines += [f"    ({n}, {m}, {c!r}, {s!r})," for (n, m, c, s) in coeffs]
    lines.append("];")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT.relative_to(OUT.parents[3])}  (model = {model})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
