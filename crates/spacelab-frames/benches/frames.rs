//! Criterion benchmarks for the Earth-orientation frame transform — the per-integrator-stage cost of a
//! high-fidelity Earth-fixed propagation.
//!
//! `cargo bench -p spacelab-frames` (add `--features std` for the `CioInterpolator` comparison).
//!
//! `gcrf_to_itrf` is dominated by the IAU 2006/2000A CIP series (~1600 nutation terms) driven by TT,
//! which in a real propagation varies only on day scales. The `std`-gated `CioInterpolator` evaluates
//! that series once per grid node and interpolates per stage, applying the exact ERA — the intended win.

use criterion::{Criterion, criterion_group, criterion_main};
use spacelab_core::{Angle, Epoch, Tt, Ut1};
use spacelab_frames::gcrf_to_itrf;
use std::hint::black_box;

/// A representative present-epoch state (~2024) with realistic polar motion and CIP offsets.
fn scene() -> (i128, Angle, Angle, Angle, Angle) {
    let base_ns: i128 = 24 * 365 * 86_400 * 1_000_000_000;
    (
        base_ns,
        Angle::from_radians(1.0e-6),
        Angle::from_radians(1.5e-6),
        Angle::from_radians(2.0e-10),
        Angle::from_radians(-1.0e-10),
    )
}

fn bench_gcrf_to_itrf(c: &mut Criterion) {
    let (base_ns, xp, yp, dx, dy) = scene();
    c.bench_function("gcrf_to_itrf/full_cio", |b| {
        let mut i: i128 = 0;
        b.iter(|| {
            i += 1; // advance the epoch each call, as a propagation would
            let tt = Epoch::<Tt>::from_ns_since_j2000(base_ns + i);
            let ut1 = Epoch::<Ut1>::from_ns_since_j2000(base_ns + i);
            black_box(gcrf_to_itrf(
                black_box(ut1),
                black_box(tt),
                xp,
                yp,
                dx,
                dy,
                0.0,
            ))
        });
    });
}

#[cfg(feature = "std")]
fn bench_cio_interpolator(c: &mut Criterion) {
    use spacelab_frames::CioInterpolator;
    let (base_ns, xp, yp, dx, dy) = scene();
    let cio = CioInterpolator::new(
        Epoch::<Ut1>::from_ns_since_j2000(base_ns),
        Epoch::<Tt>::from_ns_since_j2000(base_ns),
        6.0 * 3600.0,
        xp,
        yp,
        dx,
        dy,
        0.0,
    );
    c.bench_function("gcrf_to_itrf/interpolated", |b| {
        let mut t = 0.0_f64;
        b.iter(|| {
            t += 1.0e-3;
            black_box(cio.rotation(black_box(t)))
        });
    });
}

#[cfg(feature = "std")]
criterion_group!(benches, bench_gcrf_to_itrf, bench_cio_interpolator);
#[cfg(not(feature = "std"))]
criterion_group!(benches, bench_gcrf_to_itrf);
criterion_main!(benches);
