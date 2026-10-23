# Bibliography — the sources we derive from

Every routine in `spacelab-fa` is derived from published mathematics. This is the
reading list.

It was extracted mechanically from the citation blocks of the IAU SOFA 2023-10-11
distribution (`doc/manual.lis`), then deduplicated. **Bibliographic citations are
facts, not expression.** Nothing here is SOFA's code, and nothing here carries
SOFA's licence obligations.

The point of the list: when we implement a routine, we read the paper — not
anyone's implementation of it. SOFA's own `intro.lis` states that consistency with
the IERS Conventions was a founding stipulation, so SOFA and `spacelab-fa` draw on
the same primary sources. We differ in what we do with them.

## Cornerstones

| For | Source |
|---|---|
| GMST, IAU 1982 (`gmst82`) | Aoki et al., *Astron. Astrophys.* **105**, 359–361 (1982) |
| Earth Rotation Angle, UT1 | Capitaine, Guinot & McCarthy, *Astron. Astrophys.* **355**, 398–405 (2000) |
| CIP X, Y series, IAU 2006 | Capitaine, Wallace & Chapront, *Astron. Astrophys.* **412**, 567 (2003) |
| Precession, IAU 2006 | Capitaine, Wallace & Chapront, *Astron. Astrophys.* **412**, 567 (2003) |
| Nutation, IAU 2000A (MHB2000) | Mathews, Herring & Buffett, *J. Geophys. Res.* **107**, B4 (2002) |
| CIO locator `s` | Capitaine, Chapront, Lambert & Wallace (2003) |
| Coefficient tables | **IERS Conventions**, Tables 5.2a/b/d, 5.3a/b — taken from IERS, never from a code repository |
| Textbook grounding | Urban & Seidelmann (eds.), *Explanatory Supplement to the Astronomical Almanac*, 3rd ed. (2013); Wakker, *Fundamentals of Astrodynamics*, TU Delft (freely distributed) |

## Analytic orbit theory — for the coverage engine, never as a propagator

The coverage engine needs a closed-form ground track: to derive its coarse-grid step
rather than guess it, to seed the root finder, and to compute repeat cycles in closed
form. See `notes/01-epicyclic-orbit-theory.md`.

| For | Source | Domain |
|---|---|---|
| Epicyclic motion, zonal | Hashida, Y. & Palmer, P. L., *Epicyclic Motion of Satellites about an Oblate Planet* | `e = O(J₂)` |
| Epicyclic motion, tesseral | Hashida & Palmer, *Epicyclic Motion of Satellites Under Rotating Potential* | `e = O(J₂)`, arbitrary degree |
| Fitting the epicyclic parameters | Palmer, P. L. & Gilani, S. A. A., SSC11-VII-5, *Epicycle Parameter Filter for Long Term Orbital Parameter Estimation* | — |
| Tesseral resonance and repeat-track stability | O'Donnell, K. & Palmer, P. L., *Absolute and Relative Orbit Analysis for Resonant Satellite Motion* | — |
| Zonal perturbations, **untruncated in eccentricity** | Gooding, R. H., *Perturbations, Untruncated in Eccentricity, for an Orbit in an Axi-Symmetric Gravitational Field*, RAE Tech. Memo. Space 379 (1991) | any `e`, zonal only |

Gooding removes the near-circular restriction the Surrey papers carry; the Surrey papers
carry the tesseral terms Gooding does not. Together they cover the two cases a coverage
engine meets.

## Root finding

Brent's method, from **Brent, R. P. (1973), *Algorithms for Minimization without
Derivatives*, ch. 4** (after Dekker, 1969) — or the `roots` crate, BSD-2-Clause.
**Never from *Numerical Recipes***; see `CONTRIBUTING.md`.

## Full extracted list

1. 1984 Astronomical Almanac, pp B39-B41.
2. Aoki et al., Astron.Astrophys., 105, 359-361 (1982).
3. Aoki, S. et al., 1983, "Conversion matrix of epoch B1950.0 FK4-based positions of stars to epoch J2000.0 positions in accordance with the new IAU resolutions". Astron.Astrophys. 128, 263-267.
4. Calabretta M.R. & Greisen, E.W., 2002, "Representations of celestial coordinates in FITS", Astron.Astrophys. 395, 1077
5. Capitaine, N., Chapront, J., Lambert, S. and Wallace, P., "Expressions for the Celestial Intermediate Pole and Celestial Ephemeris Origin consistent with the IAU 2000A precession- nutation model", Astron.Astrophys. 400, 1145-1154 (2003)
6. Capitaine, N. & Gontier, A.-M., 1993, Astron.Astrophys., 275, 645-650.
7. Capitaine, N. & Gontier, A.-M., Astron.Astrophys., 275, 645-650 (1993)
8. Capitaine N., Guinot B. and McCarthy D.D, 2000, Astron. Astrophys., 355, 398-405.
9. Capitaine, N. & Wallace, P.T., 2006, Astron.Astrophys. 450, 855
10. Capitaine, N., Wallace, P.T. and McCarthy, D.D., "Expressions to implement the IAU 2000 definition of UT1", Astron.Astrophys., 406, 1135-1149 (2003)
11. Capitaine, N., Wallace, P.T. & Chapront, J., 2003, Astron.Astrophys., 412, 567
12. Capitaine, N., Wallace, P.T. & Chapront, J., 2003, Astron. Astrophys. 432, 355
13. Capitaine, N., Wallace, P.T. & Chapront, J., 2005, Astron.Astrophys. 432, 355
14. Chapront, J., Chapront-Touze, M. & Francou, G. 2002, Astron.Astrophys. 387, 700
15. Chapront, J., Chapront-Touze, M. & Francou, G., Astron. Astrophys., 387, 700, 2002.
16. Chapront-Touze, M., Francou, G., and Laskar, J., Astron.Astrophys., 282, 663 (1994).
17. Copyright (C) 2023 IAU SOFA Board. See notes at end.
18. Crane, R.K., Meeks, M.L. (ed), "Refraction Effects in the Neutral Atmosphere", Methods of Experimental Physics: Astrophysics 12B, Academic Press, 1976.
19. Defined in sofam.h: DAU astronomical unit (m) DJC days per Julian century DJ00 reference epoch (J2000.0), Julian Date DD2R degrees to radians
20. Department of Defense World Geodetic System 1984, National Imagery and Mapping Agency Technical Report 8350.2, Third Edition, p3-2.
21. Explanatory Supplement to the Astronomical Almanac, P. Kenneth Seidelmann (ed), University Science Books (1992)
22. Fairhead, L., & Bretagnon, P., Astron.Astrophys., 229, 240-247 (1990).
23. F.Mignard & M.Froeschle, 2000, Astron.Astrophys. 354, 732-739.
24. F.Mignard & M.Froeschle, Astron.Astrophys., 354, 732-739 (2000).
25. Fukushima, T., "Transformation from Cartesian to geodetic coordinates accelerated by Halley's method", J.Geodesy (2006) 79: 689-693
26. Gill, Adrian E., "Atmosphere-Ocean Dynamics", Academic Press, 1982.
27. Green, R.M., Spherical Astronomy, Cambridge University Press, (1985) Section 4.5, p96.
28. Green, R.M., "Spherical Astronomy", Cambridge University Press, 1987.
29. Green, R.M., "Spherical Astronomy", Cambridge University Press, 1987, Chapter 13.
30. Hilton, J. et al., 2006, Celest.Mech.Dyn.Astron. 94, 351
31. Hohenkerk, C.Y., & Sinclair, A.T., NAO Technical Note No. 63, 1985.
32. IAG Resolutions adopted at the XXIIth General Assembly in Birmingham, 1999, Resolution 3.
33. IAU 2000 Resolution B1.9
34. IAU 2006 Resolution 3
35. IAU 2006 Resolution B3
36. IAU Resolution C7, Recommendation 3 (1994)
37. IAU: Trans. International Astronomical Union, Vol. XXIVB; Proc. 24th General Assembly, Manchester, UK. Resolutions B1.3, B1.6. (2000)
38. Kaplan,G.H., 1981. USNO circular no. 163, pA2.
39. Kinoshita, H. and Souchay J. 1990, Celest.Mech. and Dyn.Astron. 48, 187
40. Klein, A., A Generalized Kahan-Babuska-Summation-Algorithm. Computing, 76, 279-293 (2006), Section 3.
41. Klioner, Sergei A., "A practical relativistic model for micro- arcsecond astrometry in space", Astr. J. 125, 1580-1597 (2003).
42. Last revision: 2017 September 12
43. Lieske, J.H., 1979, Astron.Astrophys. 73, 282.
44. Lieske, J.H., 1979, Astron.Astrophys. 73, 282, equations (6) & (7), p283.
45. Lieske, J.H., Lederle, T., Fricke, W. & Morando, B. 1977, Astron.Astrophys. 58, 1-16
46. Lieske, J.H., Lederle, T., Fricke, W. & Morando, B., "Expressions for the precession quantities based upon the IAU (1976) System of Astronomical Constants", Astron.Astrophys., 58, 1-16 (1977)
47. Luzum, B., private communication, 2001 (Fortran code MHB_2000_SHORT)
48. Mathews, P.M., Herring, T.A., Buffet, B.A. 2002, J.Geophys.Res. 107, B4. The MHB_2000 code itself was obtained on 9th September 2002 from ftp//maia.usno.navy.mil/conv2000/chapter5/IAU2000A.
49. Mathews, P.M., Herring, T.A., Buffet, B.A., "Modeling of nutation and precession: New nutation series for nonrigid Earth and insights into the Earth's interior", J.Geophys.Res., 107, B4, 2002. The MHB2000 code itself was obtained on 2002 September 9 from ftp://maia.usno.navy.mil/conv2000/chapter5/IAU2000A.
50. McCarthy, D.D. & Luzum, B.J., "An abridged model of the precession-nutation of the celestial pole", Celestial Mechanics & Dynamical Astronomy, 85, 37-49 (2003)
51. McCarthy, D. D., Petit, G. (eds.), 2004, IERS Conventions (2003), IERS Technical Note No. 32, BKG
52. McCarthy, D. D., Petit, G. (eds.), IERS Conventions (2003), IERS Technical Note No. 32, BKG (2004)
53. Meeus, J., Astronomical Algorithms, 2nd edition, Willmann-Bell, 1998, p337.
54. Moritz, H., Bull. Geodesique 66-2, 187 (1992).
55. Moyer, T.D., Cel.Mech., 23, 33 (1981).
56. Murray, C.A., Vectorial Astrometry, Adam Hilger (1983).
57. Perryman M.A.C. & ESA, 1997, ESA SP-1200, The Hipparcos and Tycho catalogues. Astrometric and photometric star catalogues derived from the ESA Hipparcos Space Astrometry Mission. ESA Publications Division, Noordwijk, Netherlands.
58. Rueger, J.M., "Refractive Index Formulae for Electronic Distance Measurement with Radio and Millimetre Waves", in Unisurv Report S-68, School of Surveying and Spatial Information Systems, University of New South Wales, Sydney, Australia, 2002.
59. Seidelmann, P.K. (ed), 1992, "Explanatory Supplement to the Astronomical Almanac", ISBN 0-935702-68-7.
60. Seidelmann, P.K. et al., Explanatory Supplement to the Astronomical Almanac, Chapter 2, University Science Books (1992).
61. Simon, J.-L., Bretagnon, P., Chapront, J., Chapront-Touze, M., Francou, G., Laskar, J. 1994, Astron.Astrophys. 282, 663-683
62. Simon, J.L., Bretagnon, P., Chapront, J., Chapront-Touze, M., Francou, G. & Laskar, J., Astron.Astrophys., 1994, 282, 663
63. Smart, W.M., "Spherical Astronomy", Cambridge University Press, 6th edition (Green, 1977), p49.
64. Smith, C.A. et al., 1989, "The transformation of astrometric catalog systems to the equinox J2000.0". Astron.J. 97, 265.
65. SOFA release 2023-10-11
66. Souchay, J., Loysel, B., Kinoshita, H., Folgueira, M. 1999, Astron.Astrophys.Supp.Ser. 135, 111
67. Standish, E.M., 1982, "Conversion of positions and proper motions from B1950.0 to the IAU system at J2000.0". Astron.Astrophys., 115, 1, 20-22.
68. Stone, Ronald C., P.A.S.P. 108, 1051-1058, 1996.
69. Stumpff, P., 1985, Astron.Astrophys. 144, 232-240.
70. The Department of Defense World Geodetic System 1972, World Geodetic System Committee, May 1974.
71. Transactions of the International Astronomical Union, XVIII B, 67 (1983).
72. Urban, S. & Seidelmann, P. K. (eds), Explanatory Supplement to the Astronomical Almanac, 3rd ed., University Science Books (2013).
73. Vondrak, J., Capitaine, N. and Wallace, P., 2011, New precession expressions, valid for long time intervals, Astron.Astrophys. 534, A22
74. Vondrak, J., Capitaine, N. and Wallace, P., 2012, New precession expressions, valid for long time intervals (Corrigendum), Astron.Astrophys. 541, C1
75. Wallace, P.T. & Capitaine, N., 2006, Astron.Astrophys. 459, 981
76. Wallace, P.T., "Software for Implementing the IAU 2000 Resolutions", in IERS Workshop 5.1 (2002)
77. Yallop, B.D. et al., 1989, "Transformation of mean star places from FK4 B1950.0 to FK5 J2000.0 using matrices in 6-space". Astron.J. 97, 274.

---

## Coverage geometry (2026-07-10)

| Script | Tool | What it establishes |
|---|---|---|
| `footprint.mac` | Maxima | Derives `sin(eps)` and `cos(theta)` from vectors; solves for the footprint edge. |
| `footprint2.mac` | Maxima | First attempt at the identity. `radcan` does not close nested radicals — recorded because the failure is instructive. |
| `footprint3.mac` | Maxima | **Proof.** Eliminating `u = sqrt(L²−1+s²)` and `v = sqrt(1−L²s²)` reduces `rho_ours² − rho_savi²` to `u² − (L²−1+s²)` and `1 − L²s² − v²`, i.e. to zero. Also `L²cos²α − L² + 1 ≡ 1 − L²sin²α`: SaVi's horizon branch *is* the discriminant vanishing. |
| `footprint.input` | FriCAS | **Independent confirmation.** Same two differences, `simplify` returns `0` directly. Axiom lineage, not Maxima's. |
| `ellipsoid_normal.mac` | Maxima | `∇f ∝ (x, y, γz)` with `γ = a²/b²`: the zenith need not be stored. |
| `step_bound.mac` | Maxima | `\|dg/dt\| ≤ (1+s)·\|v_rel\|` by Cauchy–Schwarz, and `v_rel ≤ v_perigee + ω·r_apogee`. |
| `step_bound_check.m` | Octave | **Falsification.** 3.2×10⁸ `(state, cell)` samples, 400 orbits, `e ≤ 0.7`. Bound never violated; 5 % slack at the worst sample. |
| `predicate_check.m` | Octave | The sqrt-free predicate against the `asin` reference over 2×10⁶ random geometries. Zero disagreements — and zero samples near the boundary, which is why the next script exists. |
| `predicate_boundary.m` | Octave | Geometry placed *on* the derived boundary and perturbed. The three forms misclassify identically. |
| `predicate_identical.m` | Octave | Pointwise identity of `fast`, `mid` and `asin` over ±2000 ulp of the boundary at five altitudes. Single clean transition, no fuzzy band. Also: 1 nrad of elevation error moves the footprint edge ~5 mm. |

### Constants checked, not remembered

`WGS84_GAMMA = a²/b² = 1/(1−f)²` was computed in PARI/GP at 40 digits, two independent ways
(from `f`, and from `e² = 2f − f²`), and rounded once:

```
1.006739496742276434954782158956759376656…   ->   1.0067394967422765   (0x1.01b9adfe197ddp+0)
```

The literal first typed into `ellipsoid_grid.rs` was `1.006739496756586`, wrong in the eleventh
digit — about 70 µm of footprint edge, and invisible to every test that does not compare bits.
`tests/ellipsoid_grid.rs` now recomputes it from the two WGS-84 defining constants and asserts
`to_bits` equality.
