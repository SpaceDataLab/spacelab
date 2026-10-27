//! The Radon transform, its adjoint, and filtered back-projection.
//!
//! Images and sinograms are row-major `f64` slices. An image is `size × size`; a sinogram is
//! `angles.len() × detectors`, row `a` holding the projection at `angles[a]`. Detector offsets are in
//! pixel units, centred: detector `d` samples the line at signed offset `d − (detectors−1)/2` from the
//! image centre. Angles are radians.
//!
//! Gate: `derivations/radon_transform.mac` derives the projection of a disc (the chord
//! `2√(r² − s²)`) and of an anisotropic Gaussian, the projection-slice theorem on it, the Ram-Lak
//! taps as the band-limited ramp `∫|ν| e^{2πiνn} dν`, and the inversion `f(0) = ∫₀^π ∫|ν| P_θ dν dθ`;
//! `derivations/radon_transform.gp` re-checks the taps and the inversion at 60 digits and bounds the
//! sampling errors the gates rely on. The module tests
//! `the_projection_of_a_sampled_gaussian_is_its_closed_form` (G3),
//! `the_ramp_filter_convolves_each_projection_with_the_ram_lak_kernel` (G2) and
//! `filtered_back_projection_inverts_the_analytic_sinogram_of_a_gaussian` (G2) hold the shipped
//! code to them.

// Grid indices are small non-negative integer-valued f64s; the single letters (s, t, x, y, d) are the
// notation of the Radon geometry.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names
)]

const PI: f64 = core::f64::consts::PI;

/// Bilinear sample of a `size × size` row-major image at continuous pixel `(col, row)`; 0 outside.
fn sample(image: &[f64], size: usize, col: f64, row: f64) -> f64 {
    let last = (size - 1) as f64;
    if col < 0.0 || row < 0.0 || col > last || row > last {
        return 0.0;
    }
    let c0 = col as usize;
    let r0 = row as usize;
    let c1 = (c0 + 1).min(size - 1);
    let r1 = (r0 + 1).min(size - 1);
    let (fc, fr) = (col - c0 as f64, row - r0 as f64);
    let top = image[r0 * size + c0] * (1.0 - fc) + image[r0 * size + c1] * fc;
    let bot = image[r1 * size + c0] * (1.0 - fc) + image[r1 * size + c1] * fc;
    top * (1.0 - fr) + bot * fr
}

/// Forward Radon transform: the line integrals of `image` (`size × size`) at each of `angles` and
/// `detectors` offsets, written row-major into `sinogram` (`angles.len() × detectors`).
///
/// # Panics
///
/// If `image` or `sinogram` is smaller than required.
pub fn radon_forward(
    image: &[f64],
    size: usize,
    angles: &[f64],
    detectors: usize,
    sinogram: &mut [f64],
) {
    assert!(image.len() >= size * size, "image smaller than size²");
    assert!(
        sinogram.len() >= angles.len() * detectors,
        "sinogram too small"
    );
    let center = (size as f64 - 1.0) / 2.0;
    let det_center = (detectors as f64 - 1.0) / 2.0;
    let n_t = 2 * size; // samples along each ray
    let t_max = size as f64; // half ray length — covers the image at any angle
    let dt = 2.0 * t_max / n_t as f64;
    for (ai, &theta) in angles.iter().enumerate() {
        let (ct, st) = (libm::cos(theta), libm::sin(theta));
        for d in 0..detectors {
            let s = d as f64 - det_center;
            let mut sum = 0.0;
            for k in 0..n_t {
                let t = -t_max + (k as f64 + 0.5) * dt;
                let x = s * ct - t * st;
                let y = s * st + t * ct;
                sum += sample(image, size, x + center, y + center);
            }
            sinogram[ai * detectors + d] = sum * dt;
        }
    }
}

/// Adjoint (unfiltered) back-projection: smear each projection back across the `size × size` image,
/// accumulating into `out`. This is the transpose of [`radon_forward`], not its inverse — the result
/// is the blurred `1/r`-smeared reconstruction. Use [`filtered_back_projection`] for a sharp inverse.
///
/// # Panics
///
/// If `out` is smaller than `size²`.
pub fn back_project(
    sinogram: &[f64],
    angles: &[f64],
    detectors: usize,
    size: usize,
    out: &mut [f64],
) {
    assert!(out.len() >= size * size, "output smaller than size²");
    let center = (size as f64 - 1.0) / 2.0;
    let det_center = (detectors as f64 - 1.0) / 2.0;
    let last = (detectors - 1) as f64;
    for row in 0..size {
        for col in 0..size {
            let (x, y) = (col as f64 - center, row as f64 - center);
            let mut acc = 0.0;
            for (ai, &theta) in angles.iter().enumerate() {
                let s = x * libm::cos(theta) + y * libm::sin(theta) + det_center;
                if s >= 0.0 && s <= last {
                    let d0 = s as usize;
                    let d1 = (d0 + 1).min(detectors - 1);
                    let f = s - d0 as f64;
                    acc += sinogram[ai * detectors + d0] * (1.0 - f)
                        + sinogram[ai * detectors + d1] * f;
                }
            }
            out[row * size + col] = acc;
        }
    }
}

/// The discrete Ram-Lak ramp kernel `h[n]`: `1/4` at 0, `0` for even `n`, `−1/(π²n²)` for odd `n`.
fn ramlak(n: isize) -> f64 {
    if n == 0 {
        0.25
    } else if n % 2 == 0 {
        0.0
    } else {
        let nf = n as f64;
        -1.0 / (PI * PI * nf * nf)
    }
}

/// Convolve each projection (sinogram row) with the Ram-Lak ramp filter, in place. `scratch` (length
/// ≥ `detectors`) holds one filtered row. This is the frequency-weighting that turns back-projection
/// into a true inverse.
///
/// # Panics
///
/// If `scratch` is shorter than `detectors`.
pub fn ramp_filter(sinogram: &mut [f64], n_angles: usize, detectors: usize, scratch: &mut [f64]) {
    assert!(scratch.len() >= detectors, "scratch shorter than detectors");
    for a in 0..n_angles {
        let base = a * detectors;
        for (i, slot) in scratch.iter_mut().take(detectors).enumerate() {
            let mut acc = 0.0;
            for j in 0..detectors {
                acc += sinogram[base + j] * ramlak(i as isize - j as isize);
            }
            *slot = acc;
        }
        sinogram[base..base + detectors].copy_from_slice(&scratch[..detectors]);
    }
}

/// Filtered back-projection: ramp-filter the `sinogram` in place, back-project it into `out`, and
/// scale by `π / n_angles` — the standard inverse Radon reconstruction. `scratch` is the ramp-filter
/// workspace (length ≥ `detectors`).
///
/// # Panics
///
/// If `out < size²` or `scratch < detectors`.
pub fn filtered_back_projection(
    sinogram: &mut [f64],
    angles: &[f64],
    detectors: usize,
    size: usize,
    out: &mut [f64],
    scratch: &mut [f64],
) {
    ramp_filter(sinogram, angles.len(), detectors, scratch);
    back_project(sinogram, angles, detectors, size, out);
    let scale = PI / angles.len() as f64;
    for v in out.iter_mut().take(size * size) {
        *v *= scale;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    /// A filled disk of value 1, radius `r` pixels, centred in a `size × size` image.
    fn disk(size: usize, r: f64) -> Vec<f64> {
        let c = (size as f64 - 1.0) / 2.0;
        (0..size * size)
            .map(|k| {
                let (x, y) = ((k % size) as f64 - c, (k / size) as f64 - c);
                if x * x + y * y <= r * r { 1.0 } else { 0.0 }
            })
            .collect()
    }

    fn argmax(image: &[f64]) -> usize {
        let mut best = 0;
        for (i, &v) in image.iter().enumerate() {
            if v > image[best] {
                best = i;
            }
        }
        best
    }

    #[test]
    fn radon_of_a_disk_is_the_chord_length_and_angle_independent() {
        let (size, r, detectors) = (65, 12.0, 65);
        let img = disk(size, r);
        let angles = [0.0, PI / 2.0, PI / 4.0];
        let mut sino = vec![0.0; angles.len() * detectors];
        radon_forward(&img, size, &angles, detectors, &mut sino);

        let center = detectors / 2; // s = 0
        for a in 0..angles.len() {
            // Through the centre the chord is the diameter, 2r (± a pixel of discretisation).
            assert!(
                (sino[a * detectors + center] - 2.0 * r).abs() < 2.0,
                "θ#{a} s=0: {} vs {}",
                sino[a * detectors + center],
                2.0 * r
            );
            // At offset s = 8 the chord is 2√(r²−s²).
            let expect = 2.0 * (r * r - 64.0).sqrt();
            assert!((sino[a * detectors + center + 8] - expect).abs() < 2.0);
        }
        // A disk is rotationally symmetric ⇒ every projection is the same.
        assert!((sino[center] - sino[detectors + center]).abs() < 1.0);
    }

    #[test]
    fn back_projection_concentrates_a_central_point() {
        let (size, detectors) = (65, 65);
        let mut img = vec![0.0; size * size];
        let center_px = (size / 2) * size + size / 2;
        img[center_px] = 1.0;
        let angles: Vec<f64> = (0..60).map(|k| PI * f64::from(k) / 60.0).collect();
        let mut sino = vec![0.0; angles.len() * detectors];
        radon_forward(&img, size, &angles, detectors, &mut sino);
        let mut recon = vec![0.0; size * size];
        back_project(&sino, &angles, detectors, size, &mut recon);
        assert_eq!(
            argmax(&recon),
            center_px,
            "back-projection peak not at the point"
        );
    }

    #[test]
    fn filtered_back_projection_reconstructs_an_off_centre_point() {
        let (size, detectors) = (65, 65);
        let (pc, pr) = (45usize, 30usize);
        let mut img = vec![0.0; size * size];
        let point = pr * size + pc;
        img[point] = 1.0;
        let angles: Vec<f64> = (0..90).map(|k| PI * f64::from(k) / 90.0).collect();
        let mut sino = vec![0.0; angles.len() * detectors];
        radon_forward(&img, size, &angles, detectors, &mut sino);

        let mut recon = vec![0.0; size * size];
        let mut scratch = vec![0.0; detectors];
        filtered_back_projection(
            &mut sino,
            &angles,
            detectors,
            size,
            &mut recon,
            &mut scratch,
        );
        // The reconstruction peaks at (or immediately next to) the source pixel.
        let peak = argmax(&recon);
        let (rc, rr) = (peak % size, peak / size);
        assert!(
            rc.abs_diff(pc) <= 1 && rr.abs_diff(pr) <= 1,
            "peak at ({rc},{rr}) not ({pc},{pr})"
        );
        // And it stands well above the mean (a real reconstruction, not a smear).
        let mean = recon.iter().sum::<f64>() / recon.len() as f64;
        assert!(recon[peak] > 8.0 * mean.abs().max(1e-9));
    }

    /// (G2) The ramp filter is a convolution with the Ram-Lak kernel, tap for tap — the band-limited
    /// ramp `h[n] = ∫_{−1/2}^{1/2} |ν| e^{2πiνn} dν`, derived in `derivations/radon_transform.mac`.
    ///
    /// A unit impulse in a projection comes back *as* the kernel, so the whole of `ramlak` is on
    /// display at once: `1/4` at zero lag, nothing at even lags, `-1/(pi^2 n^2)` at odd ones. Two rows
    /// with the impulse at different detectors also pin the row stride. Exact, because every other
    /// term of the sum is a product with zero.
    #[test]
    fn the_ramp_filter_convolves_each_projection_with_the_ram_lak_kernel() {
        // h[|n|], n = 0..6: 1/4, -1/pi^2, 0, -1/(9 pi^2), 0, -1/(25 pi^2), 0.
        const H: [f64; 7] = [
            0.25,
            -0.101_321_183_642_337_78,
            0.0,
            -0.011_257_909_293_593_086,
            0.0,
            -0.004_052_847_345_693_51,
            0.0,
        ];
        let detectors = 9;
        let mut sino = vec![0.0; 2 * detectors];
        sino[3] = 1.0;
        sino[detectors + 6] = 1.0;
        let mut scratch = vec![0.0; detectors];
        ramp_filter(&mut sino, 2, detectors, &mut scratch);
        for (row, j0) in [(0usize, 3usize), (1, 6)] {
            for i in 0..detectors {
                let want = H[i.abs_diff(j0)];
                assert_eq!(sino[row * detectors + i], want, "row {row}, tap {i} - {j0}");
            }
        }
    }

    /// The projection at zero degrees, against the sum it has to be.
    ///
    /// At theta = 0 the rays run straight down the columns, and the ray samples land on exact or
    /// half-way rows depending only on the parity of `size` -- so two fixtures state the whole of
    /// `sample`: where its bounds are (an odd `size` puts a ray on column 0 and on the last column
    /// exactly; an even one puts the rows there) and what its bilinear weights are (the other parity
    /// puts the samples half-way between two taps). The disk test's ±2 tolerance on a chord of 24 can
    /// see none of it.
    #[test]
    fn the_zero_degree_projection_is_the_column_sum_it_has_to_be() {
        // size 5, detectors 5: columns land on 0..4 exactly, rows half-way at 0.5, 1.5, 2.5, 3.5.
        let img: Vec<f64> = (1..=25).map(f64::from).collect();
        let mut sino = vec![0.0; 5];
        radon_forward(&img, 5, &[0.0], 5, &mut sino);
        for d in 0..5 {
            let want: f64 = (0..4)
                .map(|j| f64::midpoint(img[j * 5 + d], img[(j + 1) * 5 + d]))
                .sum();
            assert_eq!(sino[d], want, "size 5, detector {d}");
        }

        // size 5, detectors 4: both weights are a half, so the sample is the mean of all four taps --
        // the only arrangement in which the second row of the bilinear carries any weight at all.
        let mut sino = vec![0.0; 4];
        radon_forward(&img, 5, &[0.0], 4, &mut sino);
        for d in 0..4 {
            let want: f64 = (0..4)
                .map(|j| {
                    (img[j * 5 + d]
                        + img[j * 5 + d + 1]
                        + img[(j + 1) * 5 + d]
                        + img[(j + 1) * 5 + d + 1])
                        / 4.0
                })
                .sum();
            assert_eq!(sino[d], want, "size 5, detectors 4, detector {d}");
        }

        // size 4, detectors 5: rows land on 0..3 exactly, columns half-way at -0.5 .. 3.5 -- so the
        // first and last detector look outside the image entirely and must come back as zero.
        let img: Vec<f64> = (1..=16).map(f64::from).collect();
        let mut sino = vec![0.0; 5];
        radon_forward(&img, 4, &[0.0], 5, &mut sino);
        assert_eq!((sino[0], sino[4]), (0.0, 0.0));
        for d in 1..4 {
            let want: f64 = (0..4)
                .map(|r| f64::midpoint(img[r * 4 + d - 1], img[r * 4 + d]))
                .sum();
            assert_eq!(sino[d], want, "size 4, detector {d}");
        }
    }

    /// Back-projection smears each projection back along its own ray, interpolated between detectors.
    ///
    /// `back_projection_concentrates_a_central_point` only locates the brightest pixel, which survives
    /// a wrong row stride, a dropped neighbour and a sign flip between the two interpolation taps.
    /// Here every pixel of the output is stated: with `detectors = size + 1` the ray for pixel column
    /// `c` falls exactly half-way between detectors `c` and `c+1`, and a second projection at 90
    /// degrees does the same down the rows, so the answer is the sum of two half-way means.
    #[test]
    fn back_projection_smears_each_projection_along_its_own_ray() {
        let (size, detectors) = (4usize, 5usize);
        let sino: Vec<f64> = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0].into();
        let angles = [0.0, PI / 2.0];
        let mut out = vec![0.0; size * size];
        back_project(&sino, &angles, detectors, size, &mut out);
        for row in 0..size {
            for col in 0..size {
                let want = f64::midpoint(sino[col], sino[col + 1])
                    + f64::midpoint(sino[detectors + row], sino[detectors + row + 1]);
                assert!(
                    (out[row * size + col] - want).abs() < 1e-12,
                    "pixel ({col}, {row}) = {}, want {want}",
                    out[row * size + col]
                );
            }
        }
    }

    /// A ray that falls off either end of the detector array contributes nothing.
    ///
    /// With three detectors across a five-pixel image the outermost columns project past both ends of
    /// the array, so the guard is what keeps the read in bounds -- and the interpolation partner of
    /// the last detector is the last detector, not the one after it.
    #[test]
    fn a_ray_that_misses_the_detector_array_contributes_nothing() {
        let (size, detectors) = (5usize, 3usize);
        let sino = [7.0, 2.0, 5.0];
        let mut out = vec![0.0; size * size];
        back_project(&sino, &[0.0], detectors, size, &mut out);
        for row in 0..size {
            for col in 0..size {
                let want = if (1..=3).contains(&col) {
                    sino[col - 1]
                } else {
                    0.0
                };
                assert_eq!(out[row * size + col], want, "pixel ({col}, {row})");
            }
        }
    }

    /// Filtered back-projection returns the disk's amplitude, not just its shape.
    ///
    /// `filtered_back_projection_reconstructs_an_off_centre_point` checks where the peak is and that
    /// it stands above the mean, so the `pi/n_angles` normalisation -- and whether it is applied to
    /// the whole image -- is invisible to it. A unit disk reconstructs to unit amplitude: measured,
    /// the interior lands within 0.03 of 1 and the exterior within 0.03 of 0, against the factor of
    /// 38 that dropping or misapplying the scale costs.
    #[test]
    fn filtered_back_projection_returns_the_amplitude_of_a_unit_disk() {
        let (size, detectors, r) = (65usize, 65usize, 12.0);
        let img = disk(size, r);
        let angles: Vec<f64> = (0..120).map(|k| PI * f64::from(k) / 120.0).collect();
        let mut sino = vec![0.0; angles.len() * detectors];
        radon_forward(&img, size, &angles, detectors, &mut sino);
        let mut recon = vec![0.0; size * size];
        let mut scratch = vec![0.0; detectors];
        filtered_back_projection(
            &mut sino,
            &angles,
            detectors,
            size,
            &mut recon,
            &mut scratch,
        );
        let c = (size as f64 - 1.0) / 2.0;
        let (mut inside, mut outside) = (0usize, 0usize);
        for (k, &value) in recon.iter().enumerate() {
            let (x, y) = ((k % size) as f64 - c, (k / size) as f64 - c);
            let d2 = x * x + y * y;
            if d2 <= (r - 3.0) * (r - 3.0) {
                inside += 1;
                assert!(
                    (value - 1.0).abs() < 0.05,
                    "inside the disk, pixel {k} reconstructs to {value}"
                );
            } else if d2 >= (r + 4.0) * (r + 4.0) {
                outside += 1;
                assert!(
                    value.abs() < 0.05,
                    "outside the disk, pixel {k} reconstructs to {value}"
                );
            }
        }
        assert!(inside > 200 && outside > 3000, "{inside} in, {outside} out");
    }

    /// The anisotropic Gaussian `exp(−x²/(2a²) − y²/(2b²))` of the two gates below, `a = 3`, `b = 4`.
    const GAUSS_A: f64 = 3.0;
    const GAUSS_B: f64 = 4.0;

    /// `σ_θ² = a² cos²θ + b² sin²θ`, the variance of the Gaussian's projection at angle `θ`.
    fn gauss_sig2(theta: f64) -> f64 {
        let (c, s) = (theta.cos(), theta.sin());
        GAUSS_A * GAUSS_A * c * c + GAUSS_B * GAUSS_B * s * s
    }

    /// Its Radon transform, `√(2π)·a·b/σ_θ·exp(−s²/(2σ_θ²))` (`derivations/radon_transform.mac`).
    fn gauss_projection(s: f64, theta: f64) -> f64 {
        let sig2 = gauss_sig2(theta);
        (2.0 * PI).sqrt() * GAUSS_A * GAUSS_B / sig2.sqrt() * (-s * s / (2.0 * sig2)).exp()
    }

    /// (G3) The projections of a sampled anisotropic Gaussian at 0 and 90 degrees are its closed-form
    /// Radon transform, to rounding.
    ///
    /// At these two angles, with an odd image and as many detectors as columns, every ray runs along
    /// a pixel column (row) and its samples fall half-way between rows (columns), so the bilinear
    /// interpolation and the midpoint rule together sum the column at unit spacing. For a Gaussian of
    /// width 3 or 4 pixels that sum is the integral to 3.8e-16 relative (Poisson aliasing and the
    /// image edge, bounded in `derivations/radon_transform.gp`), so the gate can ask for 1e-12 —
    /// where the disk test, with a discontinuous indicator, can only ask for two pixels. The two
    /// angles see different widths (`σ_0 = a`, `σ_90 = b`) and so different heights, `√(2π)·b` and
    /// `√(2π)·a`, which a projection taken along the wrong axis would swap.
    #[test]
    fn the_projection_of_a_sampled_gaussian_is_its_closed_form() {
        let (size, detectors) = (65usize, 65usize);
        let c = (size as f64 - 1.0) / 2.0;
        let img: Vec<f64> = (0..size * size)
            .map(|k| {
                let (x, y) = ((k % size) as f64 - c, (k / size) as f64 - c);
                (-x * x / (2.0 * GAUSS_A * GAUSS_A) - y * y / (2.0 * GAUSS_B * GAUSS_B)).exp()
            })
            .collect();
        let angles = [0.0, PI / 2.0];
        let mut sino = vec![0.0; angles.len() * detectors];
        radon_forward(&img, size, &angles, detectors, &mut sino);
        let det_center = (detectors as f64 - 1.0) / 2.0;
        let mut worst = 0.0_f64;
        for (ai, &theta) in angles.iter().enumerate() {
            let peak = gauss_projection(0.0, theta);
            for d in 0..detectors {
                let s = d as f64 - det_center;
                let (got, want) = (sino[ai * detectors + d], gauss_projection(s, theta));
                let err = (got - want).abs() / peak;
                assert!(err < 1e-12, "theta {theta}, s {s}: {got} vs {want}");
                worst = worst.max(err);
            }
        }
        println!("radon of a sampled Gaussian: worst error {worst:.3e} of the peak");
    }

    /// (G2) Filtered back-projection of the Gaussian's analytic sinogram returns the Gaussian's
    /// centre value, 1, to rounding.
    ///
    /// `derivations/radon_transform.mac` shows `f(0) = ∫₀^π ∫|ν| P_θ(ν) dν dθ = 1` for this
    /// Gaussian: the ramp integral is `ab/(πσ_θ²)` at each angle and the angle integral of that is
    /// one. `filtered_back_projection` discretises both — the Ram-Lak convolution on unit-spaced
    /// detectors, and `π/n` times the sum over `n` equispaced angles. For a projection this smooth
    /// the first is exact to 1e-17 and the second to 4e-51 (`derivations/radon_transform.gp`), and
    /// at the centre pixel every ray lands on a detector exactly, so the back-projection does not
    /// interpolate. What is left is the shipped code: the taps, the sum over angles and the scale.
    /// The sinogram varies with the angle (`σ_θ` runs from 3 to 4), so an angle sum that dropped
    /// or repeated a row, or a scale off by any factor, is a different number.
    #[test]
    fn filtered_back_projection_inverts_the_analytic_sinogram_of_a_gaussian() {
        let (size, detectors, n_angles) = (65usize, 65usize, 60u32);
        let angles: Vec<f64> = (0..n_angles)
            .map(|k| PI * f64::from(k) / f64::from(n_angles))
            .collect();
        let det_center = (detectors as f64 - 1.0) / 2.0;
        let mut sino: Vec<f64> = angles
            .iter()
            .flat_map(|&theta| {
                (0..detectors).map(move |d| gauss_projection(d as f64 - det_center, theta))
            })
            .collect();
        let mut recon = vec![0.0; size * size];
        let mut scratch = vec![0.0; detectors];
        filtered_back_projection(
            &mut sino,
            &angles,
            detectors,
            size,
            &mut recon,
            &mut scratch,
        );
        let centre = recon[(size / 2) * size + size / 2];
        println!("FBP of the analytic Gaussian sinogram: centre {centre}");
        assert!(
            (centre - 1.0).abs() < 1e-12,
            "centre reconstructs to {centre}"
        );
    }
}
