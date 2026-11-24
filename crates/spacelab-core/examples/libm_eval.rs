//! The shipped `libm`'s answer at each input `derivations/libm_hypothesis_arb.c` generates, for the
//! measurement that holds the hypothesis H-libm (`docs/verification.md` §1.1).
//!
//! ```text
//! libm_hypothesis_arb gen | cargo run -q --release -p spacelab-core --example libm_eval \
//!     | libm_hypothesis_arb check h_libm.txt
//! ```
//!
//! Each input line is `row function x [y]`, every double as the 16 hex digits of its bits; the
//! output line repeats it and appends the result's bits. The `libm` is the one `Cargo.lock` pins for
//! the whole workspace, so the measurement is of the code every crate calls. Nothing here decides
//! anything: the ball arithmetic and the comparison with the stated constants are the C file's.

use std::io::{self, BufRead, BufWriter, Write};

fn bits(word: &str) -> Result<f64, String> {
    u64::from_str_radix(word, 16)
        .map(f64::from_bits)
        .map_err(|e| format!("not 16 hex digits: {word:?} ({e})"))
}

fn eval(function: &str, x: f64, y: Option<f64>) -> Result<f64, String> {
    let two = |f: fn(f64, f64) -> f64| {
        y.map(|y| f(x, y))
            .ok_or_else(|| format!("{function} takes two arguments"))
    };
    Ok(match function {
        "sin" => libm::sin(x),
        "cos" => libm::cos(x),
        "tan" => libm::tan(x),
        "sincos_sin" => libm::sincos(x).0,
        "sincos_cos" => libm::sincos(x).1,
        "exp" => libm::exp(x),
        "log" => libm::log(x),
        "acos" => libm::acos(x),
        "asin" => libm::asin(x),
        "atan" => libm::atan(x),
        "cbrt" => libm::cbrt(x),
        "sqrt" => libm::sqrt(x),
        "atan2" => two(libm::atan2)?,
        "hypot" => two(libm::hypot)?,
        _ => return Err(format!("unknown function {function:?}")),
    })
}

fn main() -> Result<(), String> {
    let stdin = io::stdin();
    let mut out = BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let words: Vec<&str> = line.split_whitespace().collect();
        let (x, y) = match words.as_slice() {
            [_, _, x] => (bits(x)?, None),
            [_, _, x, y] => (bits(x)?, Some(bits(y)?)),
            _ => return Err(format!("malformed line: {line:?}")),
        };
        let r = eval(words[1], x, y)?;
        writeln!(out, "{line} {:016x}", r.to_bits()).map_err(|e| e.to_string())?;
    }
    out.flush().map_err(|e| e.to_string())
}
