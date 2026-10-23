import java.util.List;
import org.hipparchus.util.FastMath;
import org.orekit.forces.gravity.potential.TideSystem;
import org.orekit.forces.gravity.potential.UnnormalizedSphericalHarmonicsProvider;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.EquinoctialOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.PropagationType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.semianalytical.dsst.forces.DSSTZonal;
import org.orekit.propagation.semianalytical.dsst.forces.ShortPeriodTerms;
import org.orekit.propagation.semianalytical.dsst.utilities.AuxiliaryElements;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.ParameterDriver;

/**
 * Orekit's zonal short-period correction evaluated directly at a given mean state.
 *
 * The fixture `dsst::short_period` used before this differenced two DSSTPropagator runs -- one MEAN,
 * one OSCULATING -- at the epochs both sampled. That difference is Orekit's short-period delta plus
 * Orekit's Hermite interpolation of its own short-period coefficients between integration steps,
 * which the port deliberately does not reproduce. The interpolation residual measured 3.03e-6 m in
 * `a`, fifteen times the 1.97e-7 m effect the surviving mutants of `compute_cij_sij` move, so the
 * comparison could not see them.
 *
 * `ShortPeriodTerms.value(meanOrbit)` is the uninterpolated quantity: `initializeShortPeriodTerms`
 * builds the terms, `updateShortPeriodTerms` computes the Fourier coefficients at exactly this mean
 * state, and `value` sums the series there. No propagation, no interpolation, no step boundaries.
 *
 * No `orekit-data` is needed: the field is supplied here, the frame is EME2000 (a fixed bias from
 * GCRF, which the zonal expansion is invariant to), and the epoch is the J2000 constant.
 *
 * <pre>
 * javac -cp "$JARS/orekit-13.1.7.jar:$JARS/hipparchus-core-4.0.3.jar:..." DsstZonalShortPeriodOracle.java
 * java  -cp "$JARS/*:." DsstZonalShortPeriodOracle
 * </pre>
 */
public final class DsstZonalShortPeriodOracle {

    /** The field the Rust fixture uses: unnormalized C(n,0), n = 2..6, with its own GM and a_e. */
    private static final double MU = 3.98600441500000000e+14;

    private static final double AE = 6.37813646000000000e+06;

    private static final double[] CN0 = {
        0.0,
        0.0,
        -1.08262689216735860e-03,
        2.53209950653743350e-06,
        1.62004577259847830e-06,
        2.27515624171733840e-07,
        -5.41217294926390600e-07,
    };

    /** The two MEAN equinoctial states `(a, k, h, q, p, lambda)` the Rust fixture evaluates at. */
    private static final double[][] MEAN = {
        {
            7.20000000000000000e+06, 3.16395301514028070e-03, 9.47328593348677900e-03,
            8.51839935608425400e-01, 7.73121044973265600e-01, 2.37304247212616130e+01,
        },
        {
            7.20000000000000000e+06, 3.23744043048822130e-03, 9.43531196934242000e-03,
            8.48776222732037000e-01, 7.76483292346584100e-01, 4.60296794537376600e+01,
        },
    };

    private DsstZonalShortPeriodOracle() {
    }

    public static void main(String[] args) {
        Frame frame = FramesFactory.getEME2000();
        System.out.printf("# Orekit 13.1.7 DSSTZonal.ShortPeriodTerms.value at the mean state.%n");
        System.out.printf("# rows: (degree, state) in row-major order over degrees 2..%d.%n",
                CN0.length - 1);
        // The degree sweep matters: `min(2N-1, N+sMax)` is the frequency the C^j/S^j arrays die at,
        // and it is set by `2N-1` only for N <= 5 -- so the window guards of `compute_cij_sij` that
        // are unreachable at N = 6 are live below it.
        for (int maxDegree = 2; maxDegree < CN0.length; maxDegree++) {
            // Orekit's own defaults for a degree-n zonal field, spelled out so Rust can match.
            int maxEccPow = FastMath.min(maxDegree - 1, 4);
            int maxFrequency = 2 * maxDegree + 1;
            double[] cn0 = new double[maxDegree + 1];
            System.arraycopy(CN0, 0, cn0, 0, maxDegree + 1);
            System.out.printf("# maxDegree=%d maxEccPow=%d maxFrequencyShortPeriodics=%d%n",
                    maxDegree, maxEccPow, maxFrequency);

            for (double[] mean : MEAN) {
                EquinoctialOrbit orbit = new EquinoctialOrbit(mean[0], mean[1], mean[2], mean[3],
                        mean[4], mean[5], PositionAngleType.MEAN, frame, AbsoluteDate.J2000_EPOCH,
                        MU);
                DSSTZonal zonal = new DSSTZonal(frame, new ZonalProvider(cn0), maxDegree, maxEccPow,
                        maxFrequency);
                List<ParameterDriver> drivers = zonal.getParametersDrivers();
                double[] parameters = new double[drivers.size()];
                for (int i = 0; i < parameters.length; i++) {
                    parameters[i] = drivers.get(i).getValue(orbit.getDate());
                }
                AuxiliaryElements aux = new AuxiliaryElements(orbit, 1);
                List<ShortPeriodTerms> terms =
                        zonal.initializeShortPeriodTerms(aux, PropagationType.OSCULATING,
                                parameters);
                zonal.updateShortPeriodTerms(parameters, new SpacecraftState(orbit));

                double[] delta = new double[6];
                for (ShortPeriodTerms term : terms) {
                    double[] v = term.value(orbit);
                    for (int i = 0; i < 6; i++) {
                        delta[i] += v[i];
                    }
                }
                StringBuilder row = new StringBuilder("        [");
                for (int i = 0; i < 6; i++) {
                    row.append(String.format("%+.17e", delta[i]));
                    row.append(i == 5 ? "]," : ", ");
                }
                System.out.println(row);
            }
        }
    }

    /** A zonal-only provider over a caller-supplied unnormalized C(n,0), so no data files load. */
    private record ZonalProvider(double[] c) implements UnnormalizedSphericalHarmonicsProvider {

        @Override
        public int getMaxDegree() {
            return c.length - 1;
        }

        @Override
        public int getMaxOrder() {
            return 0;
        }

        @Override
        public double getMu() {
            return MU;
        }

        @Override
        public double getAe() {
            return AE;
        }

        @Override
        public AbsoluteDate getReferenceDate() {
            return AbsoluteDate.J2000_EPOCH;
        }

        @Override
        public TideSystem getTideSystem() {
            return TideSystem.ZERO_TIDE;
        }

        @Override
        public UnnormalizedSphericalHarmonics onDate(AbsoluteDate date) {
            return new UnnormalizedSphericalHarmonics() {

                @Override
                public AbsoluteDate getDate() {
                    return date;
                }

                @Override
                public double getUnnormalizedCnm(int n, int m) {
                    return m == 0 && n < c.length ? c[n] : 0.0;
                }

                @Override
                public double getUnnormalizedSnm(int n, int m) {
                    return 0.0;
                }
            };
        }
    }
}
