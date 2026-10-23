//! Cost of the GCRF↔ITRF transform, split into its slow (TT) and fast (UT1) parts.
//!
//! `cargo run -p spacelab-frames --release --features std --example cio_bench`
//!
//! The `std` feature is what brings in `CioInterpolator`; without it only the exact transform is
//! timed, and the example says so rather than quietly printing half the comparison.
//!
//! A high-fidelity Earth-fixed propagation calls `gcrf_to_itrf` once per integrator stage. Its cost is
//! dominated by the IAU 2006/2000A CIP series (`cip_xy`, ~1600 nutation terms) driven by TT — which
//! varies on *day* scales, not the seconds of an integration step. This measures the full transform
//! against the isolated fast part (Earth Rotation Angle, driven by UT1) to size the win from evaluating
//! the CIP once per step and only the ERA per stage.

#![allow(clippy::cast_precision_loss, clippy::expect_used)]

use std::hint::black_box;
use std::time::Instant;

use spacelab_core::{Angle, Epoch, Tt, Ut1};
use spacelab_frames::gcrf_to_itrf;

fn main() {
    // A representative present-epoch state (ns since J2000): ~2024, with realistic EOP.
    let base_ns: i128 = 24 * 365 * 86_400 * 1_000_000_000;
    let (xp, yp) = (Angle::from_radians(1.0e-6), Angle::from_radians(1.5e-6));
    let (dx, dy) = (Angle::from_radians(2.0e-10), Angle::from_radians(-1.0e-10));

    // Warm up.
    for _ in 0..1000 {
        let tt = Epoch::<Tt>::from_ns_since_j2000(base_ns);
        let ut1 = Epoch::<Ut1>::from_ns_since_j2000(base_ns);
        black_box(gcrf_to_itrf(ut1, tt, xp, yp, dx, dy, 0.0));
    }

    let reps = 50_000;
    // Full transform, epoch advanced each call as a real propagation would.
    let t0 = Instant::now();
    for i in 0..reps {
        let dt = i128::from(i);
        let tt = Epoch::<Tt>::from_ns_since_j2000(base_ns + dt);
        let ut1 = Epoch::<Ut1>::from_ns_since_j2000(base_ns + dt);
        black_box(gcrf_to_itrf(
            black_box(ut1),
            black_box(tt),
            xp,
            yp,
            dx,
            dy,
            0.0,
        ));
    }
    let full = t0.elapsed().as_secs_f64() / f64::from(reps);

    println!("GCRF→ITRF (IAU 2006/2000A CIO), warm loop:");
    println!(
        "  full transform:    {:>7.3} µs/call   ({:>8.0} calls/s)",
        full * 1e6,
        1.0 / full
    );

    // The interpolated provider (std-feature): the CIP is precomputed on a grid, so a query is a
    // handful of multiplies plus the exact ERA. Run with `--features std` to see it.
    #[cfg(feature = "std")]
    {
        use spacelab_frames::CioInterpolator;
        let span = f64::from(reps) * 1e-9; // arc covered by the epoch sweep below
        let cio = CioInterpolator::new(
            Epoch::<Ut1>::from_ns_since_j2000(base_ns),
            Epoch::<Tt>::from_ns_since_j2000(base_ns),
            span.max(6.0 * 3600.0),
            xp,
            yp,
            dx,
            dy,
            0.0,
        );
        for _ in 0..1000 {
            black_box(cio.rotation(black_box(0.0)));
        }
        let t1 = Instant::now();
        for i in 0..reps {
            black_box(cio.rotation(black_box(f64::from(i) * 1e-3)));
        }
        let interp = t1.elapsed().as_secs_f64() / f64::from(reps);
        println!(
            "  interpolated:      {:>7.3} µs/call   ({:>8.0} calls/s)   {:.0}× faster",
            interp * 1e6,
            1.0 / interp,
            full / interp
        );
    }

    #[cfg(not(feature = "std"))]
    println!("  interpolated:      not built — re-run with `--features std` for the comparison.");

    println!(
        "  The CIP series varies on day scales; `CioInterpolator` evaluates it once per grid node and"
    );
    println!("  interpolates per stage, applying the exact ERA — matched to the f64 floor.");
    println!(
        "  Whether that matters depends on the force model it shares a stage with; spacelab-book's"
    );
    println!(
        "  `frames_bench` times this transform against the geopotential on one named machine."
    );
}
