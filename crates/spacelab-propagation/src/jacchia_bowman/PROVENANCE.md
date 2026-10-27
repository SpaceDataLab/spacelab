# Provenance of every number in `jacchia_bowman`

Document keys: **SR313** = Jacchia (1970), SAO Special Report 313; **SR332** = Jacchia (1971),
SAO Special Report 332; **B06** = Bowman et al. (2008), JASTP 70, 774–793 (JB2006); **B08** =
Bowman et al. (2008), AIAA 2008-6438 (JB2008). Pages are the numbers printed on the sheet
(journal or report page); where a PDF page index is meant it is written "PDF n". Every value was re-read from the page image (rendered PNG), not from the OCR text layer.

"derived (file)" = derived from printed equations in Maxima, file under `derivations/` at the
repository root: d1 `jacchia_bowman_quartic.mac`, d2 `jacchia_bowman_hermite.mac`, d3
`jacchia_bowman_diurnal.mac`, d5 `jacchia_bowman_static_reference.mac`, d6
`jacchia_bowman_full_model_terms.mac`.
"census" = the table, written before this code, that chose for each part of the model the paper
edition that prints it, and marked the numbers no paper prints as "code only" without recording
them; those items are left out. It is not published.
"mathematics" = a mathematical constant, a calendar rule or a unit factor.

## Static model (`constants.rs`, `static_model.rs`, `profile.rs`)

| constant (code) | value | role | document | page | equation / table |
|---|---|---|---|---|---|
| `Q0_N2` | 0.78110 | sea-level volume fraction N2 | SR313 | p. 3 | Sec. 2 table |
| `Q0_O2` | 0.20955 | sea-level volume fraction O2 | SR313 | p. 3 | Sec. 2 table |
| `Q0_AR` | 0.00934 | sea-level volume fraction Ar (SR313 value, census Sec. 1) | SR313 | p. 3 | Sec. 2 table |
| `Q0_HE` | 0.00001289 | sea-level volume fraction He (SR313 value, census Sec. 1) | SR313 | p. 3 | Sec. 2 table |
| `M_N2` | 28.0134 g/mol | molar mass N2 | SR313 | p. 3 | Sec. 2 table |
| `M_O2` | 31.9988 g/mol | molar mass O2 | SR313 | p. 3 | Sec. 2 table |
| `M_AR` | 39.948 g/mol | molar mass Ar | SR313 | p. 3 | Sec. 2 table |
| `M_HE` | 4.0026 g/mol | molar mass He | SR313 | p. 3 | Sec. 2 table |
| `M_O` | M(O2)/2 = 15.9994 g/mol | molar mass O | derived (half of printed M(O2)) | p. 3 | — |
| `M_H` | **1.00797 g/mol** | molar mass H | **not in the papers.** IUPAC 1961 standard atomic weight of H, the table that also gives all four printed weights (N 14.0067, O 15.9994, Ar 39.948, He 4.0026). Census Sec. 2 directs "IUPAC". Value from the agent's knowledge; to be verified against the IUPAC 1961 publication | — | — |
| `M_BAR_0` | 28.960 g/mol | sea-level mean molecular mass | SR313 | p. 3 | Sec. 2 |
| `G_PER_KG` | 1e3 | g ↔ kg | mathematics (unit) | — | — |
| `M_BAR_COEFFS[0..7]` | 28.15204, −0.085586, +1.2840e−4, −1.0056e−5, −1.0210e−5, +1.5044e−6, +9.9826e−8 | M̄(z) polynomial in (z − 100) | SR313 | p. 4–5 | eq. (1) |
| `M_BAR_Z_REF` | 100 km | expansion point of eq. (1) | SR313 | p. 4 | eq. (1) |
| `AVOGADRO` | 6.02257e23 mol⁻¹ | Avogadro's number | SR313 | p. 5 | after eq. (4) |
| `R_STAR` | 8.31432 J K⁻¹ mol⁻¹ | gas constant | SR313 | p. 7 | after eq. (5) |
| `Z0` | 90 km | lower boundary | SR313 | p. 7, 9 | Sec. 3, 4 |
| `T0` | 183 K | temperature at 90 km | SR313 | p. 7, 9 | Sec. 3, 4 |
| `RHO0` | 3.46e−9 g cm⁻³ × 1e3 = 3.46e−6 kg m⁻³ | density at 90 km | SR313 | p. 7 | Sec. 3 |
| `KG_M3_PER_G_CM3` | 1e3 | g cm⁻³ → kg m⁻³ | mathematics (unit) | — | — |
| `Z_HOMOPAUSE` | 105 km | mixing/diffusion boundary (SR313 value, census Sec. 1) | SR313 | p. 3, 7 | Sec. 2, 3 |
| `ZX` | 125 km | inflection height | SR313 | p. 9 | eq. (9) |
| `Z_HYDROGEN` | 500 km | hydrogen boundary height | SR313; SR332 | p. 7; p. 9 | eq. (7) |
| `G0` | 980.665 cm s⁻² × 1e−2 = 9.80665 m s⁻² | surface gravity | SR313 | p. 8 | eq. (8) |
| `M_S2_PER_CM_S2` | 1e−2 | cm s⁻² → m s⁻² | mathematics (unit) | — | — |
| `R_E` | 6.356766e8 cm / 1e5 = 6356.766 km | radius in eq. (8) | SR313 | p. 8 | eq. (8) |
| `CM_PER_KM`, `M_PER_KM` | 1e5, 1e3 | units | mathematics (unit) | — | — |
| `TX_A` | 444.3807 K | a of T_x | SR313 | p. 9 | eq. (9) |
| `TX_B` | 0.02385 | b of T_x (printed "bT", read bT∞) | SR313 | p. 9 | eq. (9) |
| `TX_C` | −392.8292 K | c of T_x | SR313 | p. 9 | eq. (9) |
| `TX_K` | −0.0021357 K⁻¹ | k̄ of T_x | SR313 | p. 9 | eq. (9) |
| `GX_FACTOR` | 1.90 | G_x = 1.90 (T_x − T_0)/(z_x − z_0) | SR313 | p. 10 | eq. (11) |
| `QUARTIC_C3_OVER_G` | −17/23275 | c₃/G_x of the quartic | derived (`derivations/jacchia_bowman_quartic.mac`) | p. 9–10 | eqs. (10)–(11) |
| `QUARTIC_C4_OVER_G` | −8/814625 | c₄/G_x of the quartic | derived (`derivations/jacchia_bowman_quartic.mac`) | p. 9–10 | eqs. (10)–(11) |
| c₁ = G_x, c₂ = 0 (in `profile.rs`) | — | quartic | derived (`derivations/jacchia_bowman_quartic.mac`) | p. 9–10 | eqs. (10)–(11) |
| `PROFILE_B` | 4.5e−6 km⁻ⁿ | B of the arctan profile | SR313 | p. 10 | eq. (13) |
| `PROFILE_N` | 2.5 | exponent n | SR313 | p. 10 | eq. (13) |
| 2/π in A = (2/π)(T∞ − T_x) | `FRAC_2_PI` (std) | profile amplitude | SR313 p. 10 eq. (13); mathematics | — | — |
| `ALPHA_HE` | −0.38 | thermal diffusion factor He | SR313 | p. 7 | after eq. (6) |
| α = 0 (other species, in `static_model.rs`) | 0 | thermal diffusion, all but He | SR313 | p. 7 | after eq. (6) |
| `H500_COEFFS` | 73.13, −39.40, 5.5 | log₁₀ n(H) at 500 km | SR313 (numbers); SR332 (argument T₅₀₀) | p. 7; p. 9 | eq. (7) |
| `LOG10_CM3_PER_M3` | 6 | cm⁻³ → m⁻³ | mathematics (unit) | — | — |
| 2 in n(O) = 2N(1 − M̄/M̄₀); 1 in n(O₂) | — | mixing composition | SR313 | p. 5 | eq. (4) |
| `DEFAULT_MAX_STEP_KM` | 1 km | quadrature panel width (numerical method, not physics) | own choice, gated by test `quadrature_converges` | — | — |
| Boole weights 14/45, 64/45, 24/45 (`quadrature.rs`) | — | five-point Newton–Cotes | derived (`derivations/jacchia_bowman_diurnal.mac`) | — | — |

## Variations (`variations.rs`, `model.rs`, `calendar.rs`)

| constant (code) | value | role | document | page | equation / table |
|---|---|---|---|---|---|
| `TC_COEFFS` | 392.4, 3.227, 0.298, 2.259, 0.312, 0.178 | Tc | B08 | p. 5 | eq. (2) |
| `FS_WEIGHT_REF` | 240 | W_T = (F̄10/240)^¼ | B08 | p. 5 | eq. (1) |
| `FS_WEIGHT_EXP` | 1/4 (fourth root, from the page image) | W_T | B08 | p. 5 | eq. (1) |
| `DIURNAL_M` | 2.5 | m | SR313 | p. 19 | text after eq. (17) |
| `DIURNAL_N` | 3.0 | n | SR313 | p. 19 | text |
| `DIURNAL_BETA_DEG` | −37° | β | SR313 | p. 19 | text |
| `DIURNAL_P_DEG` | +6° | p | SR313 | p. 19 | text |
| `DIURNAL_GAMMA_DEG` | +43° | γ | SR313 | p. 19 | text |
| `DIURNAL_R` | 0.31 | R ("a good average") | SR313 | p. 19 | text |
| ½ in η = ½\|φ − δ\|, θ = ½\|φ + δ\|, τ/2 | — | diurnal angles | SR313 | p. 18 | eqs. (15)–(16) |
| π, 2π (`PI`, `TAU`, std) | full precision | angles | mathematics | — | — |
| `HOURS_PER_DAY` | 24 | θ = LST/24; LST from hour angle | B06 p. 784; mathematics | — | — |
| `SEASLAT_AMP` | 0.02 km⁻¹ | S amplitude | SR313 | p. 26 | eq. (24) |
| `SEASLAT_DECAY` | 0.045 km⁻¹ | S decay (linear exponent, J70 form) | SR313 | p. 26 | eq. (24) |
| `SEASLAT_Z_REF` | 90 km | S reference height | SR313 | p. 26 | eq. (24) |
| `SEASLAT_PHASE` | 1.72 rad | P = sin(2πΦ + 1.72) (census Sec. 1) | SR332 | p. 46 | eq. (24) |
| `MJD_1958_JAN_1` | 36204 | Φ epoch; calendar anchor | SR332 | p. 44 | eq. (23) |
| `TROPICAL_YEAR_DAYS` | 365.2422 | Φ | SR332 | p. 44 | eq. (23) |
| `FSMJ_WEIGHTS` | 1.00, −0.70, −0.04 | F̄SMJ (daily 81-day values) | B08 | p. 8 | eq. (4) |
| `SEMIANNUAL_B` | 2.69e−01, −1.18e−02, 2.78e−02, −2.78e−02, 3.47e−04 | F(z) | B08 | p. 9 | Table 1 |
| `SEMIANNUAL_Z_SCALE_KM` | 1000 | z = height/1000 | B08 | p. 8 | eq. (5) |
| `FSM_WEIGHTS` | 1.00, −0.75, −0.37 | F̄SM | B08 | p. 9 | eq. (6) |
| `SEMIANNUAL_C` | −3.63e−01, 8.51e−02, 2.40e−01, −1.90e−01, −2.55e−01, −1.79e−02, 5.65e−04, −6.41e−04, −3.42e−03, −1.25e−03 | G(t) | B08 | p. 10 | Table 2 |
| `OMEGA_DAY_OFFSET` | 1.0 | ϑ = (t − 1.0)/365 | B06 | p. 780 | under eq. (4) |
| `OMEGA_YEAR_DAYS` | 365 | ϑ | B06 | p. 780 | under eq. (4) |
| `DTC_F_OFFSET`, `DTC_F_SCALE` | 100, 100 | F = (F10 − 100)/100 | B06 | p. 784 | list above eq. (7) |
| `DTC_Z_LOW`, `DTC_Z_MID`, `DTC_Z_HIGH` | 200, 250, 700 km | bands of eqs. (8), (7) | B06 | p. 784–785 | eqs. (7)–(8) |
| `DTC_H7_SCALE` | 100 | H = z/100 | B06 | p. 784 | eq. (7) |
| `DTC_H8_SCALE` | 50 (and offset 200) | H = (z − 200)/50 | B06 | p. 785 | eq. (8) |
| `DTC_B[1..19]` | as printed, 9 digits | ΔTc, 250–700 km | B06 | p. 786 | Table 4 "B value" |
| `DTC_C[1..23]` | as printed, 9 digits | ΔTc, 200–250 km | B06 | p. 786 | Table 4 "C value" |
| `HIGH_ALT_C` | 2.200e−01, −2.000e−03, 1.150e−03, −2.110e−06 | F_ρ above 1500 km | B06 | p. 790 | Table 5, eq. (10) |
| `HIGH_ALT_Z_LOW`, `HIGH_ALT_Z_HIGH` | 1000, 1500 km | spline interval | B06 | p. 789 | text, eq. (11) |
| 3, 2, 1 in the cubic, width 500 used once | a₂ = 3F₁₅₀₀ − D − 3, a₃ = D − 2F₁₅₀₀ + 2, D = 500(C₃ + C₄F̄10) | 1000–1500 km factor | derived (`derivations/jacchia_bowman_hermite.mac`) from B06 p. 789 end conditions; B06 eq. (11) prints 500 twice (erratum) | p. 789–790 | eq. (11) |
| `ANCHOR_YEAR` | 1958 | calendar anchor (MJD 36204 = 1958 Jan 1) | SR332 | p. 44 | eq. (23) |
| 4, 100, 400, 365, 366, `DAYS_PER_400_YEARS` = 146097 | — | Gregorian leap rule | mathematics (calendar definition) | — | — |

## Test vectors (in `tests/`)

Transcribed from page images; each test's doc comment cites document, page and table.

| file | source | pages (PDF) |
|---|---|---|
| `g5_sr313_table5.rs` | SR313 Table 5, T∞ = 600, 1000, 1900 K, 17 heights 90–2500 km, all columns but scale height | PDF 51–52 (600 K), 59 and 61 (1000 K; PDF 60 is a microfiche target), 78–79 (1900 K) |
| `g5_sr313_examples.rs` | SR313 Sec. 15 worked example; Table 1 (two panels, hours 12–23); Table 4a and 4b; hours 2.87 h and 14.08 h | p. 19, 35–37, 46 (Table 1, PDF 45), 51 (Tables 4a, 4b) |
| `g2_model.rs` | B06 Fig. 15 (graphical, read by eye) | p. 789 |
| `g1_g2_static.rs`, `g2_model.rs` | Maxima outputs `d1`–`d6` | `derivations/jacchia_bowman_*.mac` |
