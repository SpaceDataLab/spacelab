# IERS Conventions Fortran oracle

The reference implementations of the models `spacelab-slr` ports by hand: the Mendes–Pavlis
optical refraction pair (`FCULZD_HPA`, `FCUL_A`, chapter 9), the solid Earth tide station
displacement (`DEHANTTIDEINEL`, chapter 7), and the tidal arguments the ocean loading model runs on
(`TDFRPH`, from the `HARDISP` set). They are the IERS Conventions Centre's own code, so they are
fetched rather than vendored; what is committed here is the driver and this record.

`run.sh` fetches them from `https://iers-conventions.obspm.fr/content/`, compiles twice — default
reals and `-fdefault-real-8` — runs `oracle.f`, and prints every value at seventeen digits.

## What it is used for

| routine | consumed by | agreement |
|---|---|---|
| `FCULZD_HPA` | `troposphere::mendes_pavlis_zenith_delay` | 1 ULP against the compiled routine |
| `FCUL_A` | `troposphere::fcul_a_mapping` | matches the published value exactly |
| `DEHANTTIDEINEL` | `displacement::solid_earth_tide_displacement` | 1e-12 m on both published cases |
| `TDFRPH` | `loading::DoodsonArguments` | bit-exact, 4 constituents at the `HARDISP` test epoch |
| `HARDISP` (published values) | `loading::ocean_loading_topocentric` | 4.98e-7 m over 144 values, half the last digit it prints |

## The finding: `FCULZD_HPA`'s header disagrees with its own code

At the routine's own test point — McDonald Observatory, 30.67166667°, 2010.344 m, 798.4188 hPa,
14.322 hPa of water vapour, 532 nm:

| output | header comment | what the code returns | ratio |
|---|---|---|---|
| `FCUL_ZTD` | 1.935225924846803114 | 1.9352297249679732 | 1 + 1.963e-6 |
| `FCUL_ZHD` | 1.932992176591644462 | 1.9329959722362896 | 1 + 1.964e-6 |
| `FCUL_ZWD` | 0.2233748255158703871e-2 | 0.22337527316835832e-2 | 1 + 2.004e-6 |

Three things pin this down as documentation rot rather than anything a caller can act on:

- **It is a constant ratio, not an offset.** All three outputs are scaled by the same 2 parts in
  10⁶, and in Mendes–Pavlis exactly one quantity divides all three — the `f(φ, H)` gravity factor.
- **It is not a precision artefact.** The default and `-fdefault-real-8` builds differ by 1.6e-12 m
  on `FCUL_ZHD`, six orders below the gap, and both disagree with the header by the same 3.8 µm.
- **`FCUL_A`, from the same chapter and the same author, matches its header to the last digit**
  (3.800243667312344087), so the discrepancy is specific to this routine and not to the
  transcription of test values in general.

The Rust port therefore asserts what the routine computes. 3.8 µm is four orders below the
centimetre the SLR residual test resolves, so nothing downstream turns on it — but a port that had
been tuned to the header instead would have been wrong by a fixed 2 ppm of the whole troposphere,
and there would have been no test to say so.

## `DEHANTTIDEINEL`

Both published cases are reproduced to 1e-12 m, which is a picometre — round-off on a decimetre.
Case 2 is dated 2012 July 13, past the 2012-07-01 leap second, so `TAI − UTC` is **35 s** there and
34 s in case 1. Using 34 s for both moves the answer by 5e-10 m: five hundred times the tolerance,
invisible to any physical reasoning, and immediately obvious against this oracle.

## Regenerate

```
tools/iers-oracle/run.sh                        # scratch directory
IERS_ORACLE_DIR=/tmp/iers tools/iers-oracle/run.sh   # keep the sources and digests
```

Needs `gfortran` and network access. The fetched sources' SHA-256 digests are written to
`SHA256SUMS` in the working directory, so a change upstream is visible on the next run.
