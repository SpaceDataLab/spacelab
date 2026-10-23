//! Monthly CCIR coefficient maps for the F2 critical frequency and M(3000)F2 factor.
//!
//! Ported from Orekit's `CCIRLoader` (parse) and `NeQuickModel.flatten` (cache layout).
//! Each `ccirNN.asc` file holds 2858 numbers: the first 1976 are the F2 coefficients
//! (2 solar-activity rows × 76 columns × 13 harmonics), the remaining 882 are the
//! M(3000)F2 coefficients (2 × 49 × 9). The file lays them out row-major with the
//! harmonic index innermost; the flattened cache re-orders them to
//! `[column][harmonic][row]` so that the Fourier evaluation walks contiguous memory.

use std::sync::OnceLock;

/// Number of F2 coefficients preceding the M(3000)F2 block in each file.
const NUMBER_F2_COEFFICIENTS: usize = 1976;

/// Solar-activity rows (low/high) in every CCIR array.
const ROWS: usize = 2;

/// Columns of the F2 coefficient array.
const F2_COLUMNS: usize = 76;

/// Harmonics (depth) of the F2 coefficient array.
const F2_DEPTH: usize = 13;

/// Columns of the M(3000)F2 coefficient array.
const FM3_COLUMNS: usize = 49;

/// Harmonics (depth) of the M(3000)F2 coefficient array.
const FM3_DEPTH: usize = 9;

/// Embedded monthly files. Index `month - 1` selects `ccir(month + 10).asc`.
const CCIR_FILES: [&str; 12] = [
    include_str!("../assets/ccir11.asc"),
    include_str!("../assets/ccir12.asc"),
    include_str!("../assets/ccir13.asc"),
    include_str!("../assets/ccir14.asc"),
    include_str!("../assets/ccir15.asc"),
    include_str!("../assets/ccir16.asc"),
    include_str!("../assets/ccir17.asc"),
    include_str!("../assets/ccir18.asc"),
    include_str!("../assets/ccir19.asc"),
    include_str!("../assets/ccir20.asc"),
    include_str!("../assets/ccir21.asc"),
    include_str!("../assets/ccir22.asc"),
];

/// Flattened CCIR coefficients for one month, laid out `[column][harmonic][row]`.
pub(crate) struct CcirCoefficients {
    /// F2 coefficients, length `ROWS × F2_COLUMNS × F2_DEPTH` = 1976.
    pub flatten_f2: Vec<f64>,
    /// M(3000)F2 coefficients, length `ROWS × FM3_COLUMNS × FM3_DEPTH` = 882.
    pub flatten_fm3: Vec<f64>,
}

/// The twelve parsed months, filled on first use.
///
/// The files are `include_str!`ed, so `load` was re-tokenising 2 858 ASCII floats on every call —
/// 0.4 ms, which the STEC ladder pays once per `stec` against 8.4 µs per Gauss node. Nothing in a
/// month's coefficients depends on the call, so the parse belongs behind a latch.
static MONTHS: [OnceLock<CcirCoefficients>; 12] = [const { OnceLock::new() }; 12];

/// Parse and flatten the CCIR file for `month` (1..=12), once per month per process.
pub(crate) fn load(month: u32) -> &'static CcirCoefficients {
    MONTHS[(month - 1) as usize].get_or_init(|| {
        let file = CCIR_FILES[(month - 1) as usize];
        let tokens: Vec<f64> = file
            .split_whitespace()
            .filter_map(|token| token.parse::<f64>().ok())
            .collect();
        let (f2, fm3) = tokens.split_at(NUMBER_F2_COEFFICIENTS);
        CcirCoefficients {
            flatten_f2: flatten(f2, F2_COLUMNS, F2_DEPTH),
            flatten_fm3: flatten(fm3, FM3_COLUMNS, FM3_DEPTH),
        }
    })
}

/// Re-order a row-major `[row][column][harmonic]` block (harmonic innermost) into the
/// cache layout `[column][harmonic][row]` (row innermost) expected by the Fourier step.
fn flatten(tokens: &[f64], columns: usize, depth: usize) -> Vec<f64> {
    let mut flat = vec![0.0; ROWS * columns * depth];
    for row in 0..ROWS {
        for column in 0..columns {
            for d in 0..depth {
                flat[(column * depth + d) * ROWS + row] =
                    tokens[row * columns * depth + column * depth + d];
            }
        }
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both indices in `load` are written from the same `month - 1`, and December is the only call
    /// that tells `month - 1` from `month`: it is the one that puts the wrong index past the end of
    /// a twelve-element array. The crate's other tests drive months 4 and 10, so the cache index
    /// could be off by one and every STEC assertion would still hold — a neighbour's coefficients
    /// are wrong, not absurd.
    #[test]
    fn every_month_loads_its_own_coefficients() {
        let mut first_f2 = Vec::with_capacity(12);
        for month in 1..=12_u32 {
            let cached = load(month);
            let tokens: Vec<f64> = CCIR_FILES[(month - 1) as usize]
                .split_whitespace()
                .filter_map(|token| token.parse::<f64>().ok())
                .collect();
            assert_eq!(tokens.len(), 2858, "month {month} is not a whole CCIR file");
            let (f2, fm3) = tokens.split_at(NUMBER_F2_COEFFICIENTS);
            assert_eq!(
                cached.flatten_f2,
                flatten(f2, F2_COLUMNS, F2_DEPTH),
                "month {month} cached another month's F2 coefficients"
            );
            assert_eq!(
                cached.flatten_fm3,
                flatten(fm3, FM3_COLUMNS, FM3_DEPTH),
                "month {month} cached another month's M(3000)F2 coefficients"
            );
            first_f2.push(cached.flatten_f2[0]);
        }

        // And the twelve are twelve, not one file read twelve times: pairwise distinct.
        for i in 0..12 {
            for j in (i + 1)..12 {
                assert_ne!(
                    first_f2[i].to_bits(),
                    first_f2[j].to_bits(),
                    "months {} and {} load the same coefficients",
                    i + 1,
                    j + 1
                );
            }
        }
    }
}
