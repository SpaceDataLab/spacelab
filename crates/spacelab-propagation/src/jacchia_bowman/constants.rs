//! Every numerical constant of the model, with its printed source.
//!
//! Document keys: SR313 = Jacchia (1970), SAO Special Report 313; SR332 = Jacchia (1971),
//! SAO Special Report 332; B06 = Bowman et al. (2008), JASTP 70, 774-793 (JB2006);
//! B08 = Bowman et al. (2008), AIAA 2008-6438 (JB2008). Pages are the numbers printed on the
//! sheet. Values are copied as printed; unit factors are written out explicitly.
//!
//! Gate: `derivations/jacchia_bowman_quartic.mac` for the derived values, and the `(G1)`/`(G2)` tests in
//! `tests/jacchia_bowman_g1_g2_static.rs`.

// ---------------------------------------------------------------------------------------------
// Composition, SR313 p. 3, Sec. 2 table.

/// Sea-level volume fraction of N2 (SR313 p. 3).
pub const Q0_N2: f64 = 0.78110;
/// Sea-level volume fraction of O2 (SR313 p. 3).
pub const Q0_O2: f64 = 0.20955;
/// Sea-level volume fraction of Ar (SR313 p. 3; the SR313 value, census Sec. 1).
pub const Q0_AR: f64 = 0.00934;
/// Sea-level volume fraction of He, artificially raised (SR313 p. 3; census Sec. 1).
pub const Q0_HE: f64 = 0.00001289;

/// Grams per kilogram: converts the printed molecular weights (g/mol) to kg/mol.
pub const G_PER_KG: f64 = 1.0e3;

/// Molar mass of N2, kg/mol (SR313 p. 3: 28.0134).
pub const M_N2: f64 = 28.0134 / G_PER_KG;
/// Molar mass of O2, kg/mol (SR313 p. 3: 31.9988).
pub const M_O2: f64 = 31.9988 / G_PER_KG;
/// Molar mass of Ar, kg/mol (SR313 p. 3: 39.948).
pub const M_AR: f64 = 39.948 / G_PER_KG;
/// Molar mass of He, kg/mol (SR313 p. 3: 4.0026).
pub const M_HE: f64 = 4.0026 / G_PER_KG;
/// Molar mass of atomic O, kg/mol: half of M(O2) (derived; the papers print no value for O).
pub const M_O: f64 = M_O2 / 2.0;
/// Molar mass of atomic H, kg/mol. NOT printed in the papers (census Sec. 2, "AMW: H").
/// External standard, taken per the census: IUPAC 1961 atomic weight of H, 1.00797, the table
/// from which all four printed weights above come (N 14.0067, O 15.9994, Ar 39.948, He 4.0026).
pub const M_H: f64 = 1.00797 / G_PER_KG;
/// Sea-level mean molecular mass, kg/mol (SR313 p. 3: 28.960).
pub const M_BAR_0: f64 = 28.960 / G_PER_KG;

/// Coefficients c_0..c_6 of the mean molecular mass polynomial, g/mol, in powers of (z - 100 km),
/// valid 90 < z < 105 km (SR313 p. 4-5, eq. (1)).
pub const M_BAR_COEFFS: [f64; 7] = [
    28.15204, -0.085586, 1.2840e-4, -1.0056e-5, -1.0210e-5, 1.5044e-6, 9.9826e-8,
];
/// Expansion point of eq. (1), km (SR313 p. 4).
pub const M_BAR_Z_REF: f64 = 100.0;

/// Avogadro's number, mol^-1 (SR313 p. 5: A = 6.02257e23 for rho in g cm^-3, i.e. per mole).
pub const AVOGADRO: f64 = 6.02257e23;
/// Universal gas constant, J K^-1 mol^-1 (SR313 p. 7: 8.31432 joules (K)^-1 mol^-1).
pub const R_STAR: f64 = 8.31432;

// ---------------------------------------------------------------------------------------------
// Boundary conditions and heights, SR313 p. 3, 7, 9.

/// Lower boundary height z_0, km (SR313 p. 7, 9).
pub const Z0: f64 = 90.0;
/// Temperature at z_0, K (SR313 p. 7, 9: 183 K).
pub const T0: f64 = 183.0;
/// Kilograms per gram over cubic metres per cubic centimetre: g cm^-3 -> kg m^-3 is x 1e3.
const KG_M3_PER_G_CM3: f64 = 1.0e3;
/// Density at z_0, kg m^-3 (SR313 p. 7: 3.46e-9 g cm^-3).
pub const RHO0: f64 = 3.46e-9 * KG_M3_PER_G_CM3;
/// Homopause: mixing below, diffusion above, km (SR313 p. 3, 7: 105 km).
pub const Z_HOMOPAUSE: f64 = 105.0;
/// Inflection height z_x of the temperature profile, km (SR313 p. 9: 125 km).
pub const ZX: f64 = 125.0;
/// Height of the hydrogen boundary condition, km (SR313 p. 7; SR332 p. 9: 500 km).
pub const Z_HYDROGEN: f64 = 500.0;

// ---------------------------------------------------------------------------------------------
// Gravity, SR313 p. 8, eq. (8).

/// Metres per second squared per centimetre per second squared.
const M_S2_PER_CM_S2: f64 = 1.0e-2;
/// Surface gravity g_0, m s^-2 (SR313 p. 8: 980.665 cm s^-2).
pub const G0: f64 = 980.665 * M_S2_PER_CM_S2;
/// Centimetres per kilometre.
const CM_PER_KM: f64 = 1.0e5;
/// Earth radius R_e of eq. (8), km (SR313 p. 8: 6.356766e8 cm).
pub const R_E: f64 = 6.356766e8 / CM_PER_KM;
/// Metres per kilometre: the integrands are in km, the physics in SI.
pub const M_PER_KM: f64 = 1.0e3;

// ---------------------------------------------------------------------------------------------
// Temperature profile, SR313 p. 9-10, eqs. (9)-(13).

/// a of eq. (9), K (SR313 p. 9).
pub const TX_A: f64 = 444.3807;
/// b of eq. (9), dimensionless (SR313 p. 9; printed "bT", read bT_inf, census Sec. 4).
pub const TX_B: f64 = 0.02385;
/// c of eq. (9), K (SR313 p. 9).
pub const TX_C: f64 = -392.8292;
/// k-bar of eq. (9), K^-1 (SR313 p. 9).
pub const TX_K: f64 = -0.0021357;
/// G_x = 1.90 (T_x - T_0)/(z_x - z_0) (SR313 p. 10, eq. (11)).
pub const GX_FACTOR: f64 = 1.90;
/// c_3 / G_x of the quartic (10): -17/23275, derived from eq. (11) (derivations/jacchia_bowman_quartic.mac).
pub const QUARTIC_C3_OVER_G: f64 = -17.0 / 23275.0;
/// c_4 / G_x of the quartic (10): -8/814625, derived from eq. (11) (derivations/jacchia_bowman_quartic.mac).
pub const QUARTIC_C4_OVER_G: f64 = -8.0 / 814625.0;
/// B of eq. (13), km^-n (SR313 p. 10: 4.5e-6 for z in km).
pub const PROFILE_B: f64 = 4.5e-6;
/// Exponent n of eq. (13) (SR313 p. 10: 2.5).
pub const PROFILE_N: f64 = 2.5;

// ---------------------------------------------------------------------------------------------
// Diffusion, SR313 p. 7.

/// Thermal diffusion factor of He (SR313 p. 7: alpha = -0.38); 0 for all other constituents.
pub const ALPHA_HE: f64 = -0.38;

/// Hydrogen at 500 km, log10 n(H) [cm^-3] = H0 + H1 log10 T500 + H2 (log10 T500)^2
/// (SR313 p. 7 eq. (7) numbers; argument T500 as in SR332 p. 9 eq. (7), census Sec. 1).
pub const H500_COEFFS: [f64; 3] = [73.13, -39.40, 5.5];
/// log10 of cubic centimetres per cubic metre: n(H) cm^-3 -> m^-3.
pub const LOG10_CM3_PER_M3: f64 = 6.0;

// ---------------------------------------------------------------------------------------------
// Diurnal variation, SR313 p. 18-19, eqs. (15)-(17).

/// m of eq. (15) (SR313 p. 19: 2.5).
pub const DIURNAL_M: f64 = 2.5;
/// n of eq. (16) (SR313 p. 19: 3.0).
pub const DIURNAL_N: f64 = 3.0;
/// beta of eq. (16), degrees (SR313 p. 19: -37 deg).
pub const DIURNAL_BETA_DEG: f64 = -37.0;
/// p of eq. (16), degrees (SR313 p. 19: +6 deg).
pub const DIURNAL_P_DEG: f64 = 6.0;
/// gamma of eq. (16), degrees (SR313 p. 19: +43 deg).
pub const DIURNAL_GAMMA_DEG: f64 = 43.0;
/// R, constant (SR313 p. 19: "a good average is 0.31"; census Sec. 1).
pub const DIURNAL_R: f64 = 0.31;

// ---------------------------------------------------------------------------------------------
// Seasonal-latitudinal variation of the lower thermosphere: SR313 p. 26 eq. (24) amplitude,
// SR332 p. 44, 46 eqs. (23)-(24) phase (census Sec. 1).

/// Amplitude factor of S, per km (SR313 p. 26: 0.02).
pub const SEASLAT_AMP: f64 = 0.02;
/// Decay rate of S, per km (SR313 p. 26: 0.045).
pub const SEASLAT_DECAY: f64 = 0.045;
/// Reference height of S, km (SR313 p. 26: z - 90).
pub const SEASLAT_Z_REF: f64 = 90.0;
/// Phase constant of P = sin(2 pi Phi + 1.72), rad (SR332 p. 46, eq. (24)).
pub const SEASLAT_PHASE: f64 = 1.72;
/// MJD of 1958 January 1 (SR332 p. 44, eq. (23)).
pub const MJD_1958_JAN_1: f64 = 36204.0;
/// Tropical year in days in Phi = (t - 36204)/365.2422 (SR332 p. 44, eq. (23)).
pub const TROPICAL_YEAR_DAYS: f64 = 365.2422;

// ---------------------------------------------------------------------------------------------
// Tc, B08 p. 5, eqs. (1)-(2).

/// Normalisation of the weight W_T = (F10bar/240)^(1/4) (B08 p. 5, eq. (1)).
pub const FS_WEIGHT_REF: f64 = 240.0;
/// Exponent of W_T: fourth root (B08 p. 5, eq. (1), read from the page image).
pub const FS_WEIGHT_EXP: f64 = 0.25;
/// Tc = 392.4 + 3.227 FSbar + 0.298 dF10 + 2.259 dS10 + 0.312 dM10 + 0.178 dY10, K
/// (B08 p. 5, eq. (2)); order: constant, FSbar, dF10, dS10, dM10, dY10.
pub const TC_COEFFS: [f64; 6] = [392.4, 3.227, 0.298, 2.259, 0.312, 0.178];

// ---------------------------------------------------------------------------------------------
// Semiannual variation, B08 p. 7-10, eqs. (3)-(7), Tables 1-2; omega from B06 p. 780.

/// FSMJ = 1.00 F10bar - 0.70 S10bar - 0.04 M10bar (B08 p. 8, eq. (4), daily 81-day values).
pub const FSMJ_WEIGHTS: [f64; 3] = [1.00, -0.70, -0.04];
/// B1..B5 of F(z) (B08 p. 9, Table 1, as printed).
pub const SEMIANNUAL_B: [f64; 5] = [2.69e-01, -1.18e-02, 2.78e-02, -2.78e-02, 3.47e-04];
/// Kilometres per unit of z in F(z): z = height/1000 (B08 p. 8).
pub const SEMIANNUAL_Z_SCALE_KM: f64 = 1000.0;
/// FSM = 1.00 F10bar - 0.75 S10bar - 0.37 M10bar (B08 p. 9, eq. (6)).
pub const FSM_WEIGHTS: [f64; 3] = [1.00, -0.75, -0.37];
/// C1..C10 of G(t) (B08 p. 10, Table 2, as printed).
pub const SEMIANNUAL_C: [f64; 10] = [
    -3.63e-01, 8.51e-02, 2.40e-01, -1.90e-01, -2.55e-01, -1.79e-02, 5.65e-04, -6.41e-04, -3.42e-03,
    -1.25e-03,
];
/// Day-of-year offset in theta = (t - 1.0)/365 (B06 p. 780, under eq. (4)).
pub const OMEGA_DAY_OFFSET: f64 = 1.0;
/// Year length in theta = (t - 1.0)/365, days (B06 p. 780).
pub const OMEGA_YEAR_DAYS: f64 = 365.0;

// ---------------------------------------------------------------------------------------------
// B06 local-time / latitude / height corrections, p. 784-786, eqs. (7)-(8), Table 4.

/// F = (F10 - 100)/100 (B06 p. 784): offset and scale, sfu.
pub const DTC_F_OFFSET: f64 = 100.0;
/// See [`DTC_F_OFFSET`].
pub const DTC_F_SCALE: f64 = 100.0;
/// theta = local solar time (h) / 24 (B06 p. 784).
pub const HOURS_PER_DAY: f64 = 24.0;
/// Lower edge of eq. (8), km (B06 p. 784-785: 250 km >= z >= 200 km).
pub const DTC_Z_LOW: f64 = 200.0;
/// Boundary between eqs. (8) and (7), km (B06 p. 784: 700 km > z >= 250 km).
pub const DTC_Z_MID: f64 = 250.0;
/// Upper edge of eq. (7), exclusive, km (B06 p. 784).
pub const DTC_Z_HIGH: f64 = 700.0;
/// H = z/100 in eq. (7) (B06 p. 784).
pub const DTC_H7_SCALE: f64 = 100.0;
/// H = (z - 200)/50 in eq. (8) (B06 p. 785).
pub const DTC_H8_SCALE: f64 = 50.0;
/// B1..B19 of eq. (7) (B06 p. 786, Table 4, "B value", all digits).
pub const DTC_B: [f64; 19] = [
    -0.457512297e+01,
    -0.512114909e+01,
    -0.693003609e+02,
    0.203716701e+03,
    0.703316291e+03,
    -0.194349234e+04,
    0.110651308e+04,
    -0.174378996e+03,
    0.188594601e+04,
    -0.709371517e+04,
    0.922454523e+04,
    -0.384508073e+04,
    -0.645841789e+01,
    0.409703319e+02,
    -0.482006560e+03,
    0.181870931e+04,
    -0.237389204e+04,
    0.996703815e+03,
    0.361416936e+02,
];
/// C1..C23 of eq. (8) (B06 p. 786, Table 4, "C value", all digits).
pub const DTC_C: [f64; 23] = [
    -0.155986211e+02,
    -0.512114909e+01,
    -0.693003609e+02,
    0.203716701e+03,
    0.703316291e+03,
    -0.194349234e+04,
    0.110651308e+04,
    -0.220835117e+03,
    0.143256989e+04,
    -0.318481844e+04,
    0.328981513e+04,
    -0.135332119e+04,
    0.199956489e+02,
    -0.127093998e+02,
    0.212825156e+02,
    -0.275555432e+01,
    0.110234982e+02,
    0.148881951e+03,
    -0.751640284e+03,
    0.637876542e+03,
    0.127093998e+02,
    -0.212825156e+02,
    0.275555432e+01,
];

// ---------------------------------------------------------------------------------------------
// B06 high-altitude density factor, p. 789-790, eqs. (10)-(11), Table 5.

/// C1..C4 of F_rho = C1 + C2 F10bar + C3 z + C4 z F10bar, z in km (B06 p. 790, Table 5).
pub const HIGH_ALT_C: [f64; 4] = [2.200e-01, -2.000e-03, 1.150e-03, -2.110e-06];
/// Height below which the factor is 1.0, km (B06 p. 789).
pub const HIGH_ALT_Z_LOW: f64 = 1000.0;
/// Height above which eq. (10) applies, km (B06 p. 789).
pub const HIGH_ALT_Z_HIGH: f64 = 1500.0;
