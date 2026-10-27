#![allow(dead_code)]

/// log10 of a number density in m^-3 converted to cm^-3 (the tables' unit).
pub fn log10_per_cm3(n_m3: f64) -> f64 {
    n_m3.log10() - 6.0
}

/// log10 of a density in kg m^-3 converted to g cm^-3 (the tables' unit).
pub fn log10_g_cm3(rho_kg_m3: f64) -> f64 {
    rho_kg_m3.log10() - 3.0
}
