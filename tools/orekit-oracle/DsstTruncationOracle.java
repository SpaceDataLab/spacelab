import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.util.List;
import org.hipparchus.util.FastMath;
import org.orekit.forces.gravity.potential.TideSystem;
import org.orekit.forces.gravity.potential.UnnormalizedSphericalHarmonicsProvider;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.EquinoctialOrbit;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.PropagationType;
import org.orekit.propagation.semianalytical.dsst.forces.DSSTZonal;
import org.orekit.propagation.semianalytical.dsst.utilities.AuxiliaryElements;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.ParameterDriver;

/**
 * Golden values for the DSST eccentricity-power truncation order (`maxEccPowMeanElements`).
 *
 * This is the one output of `dsst::coefficients` that never reaches the answer: it decides where the
 * eccentricity series is cut, and a cut that is one term too early costs accuracy without moving any
 * end-to-end comparison above its tolerance. Nothing in `spacelab` pinned it against a number.
 *
 * Orekit computes it in `DSSTZonal.computeMeanElementsTruncations`, which is private and stores its
 * result in the private field `maxEccPowMeanElements`. The public entry point that runs it is
 * `initializeShortPeriodTerms`, so the value is reached by calling that and reading the field
 * reflectively -- both classes live on the classpath in the unnamed module, so no `--add-opens`.
 *
 * The gravity field is `spacelab`'s own embedded XGM2019e zonals, so the Rust side derives the same
 * unnormalized coefficients from `geopotential_coeffs::MODEL_COEFFS` rather than trusting a literal
 * copied across languages; the `cn0` rows exist so that derivation is itself asserted.
 *
 * No `orekit-data` is needed: the field is supplied here, the frame is EME2000 (a fixed bias from
 * GCRF), and the epoch is the J2000 constant.
 */
public final class DsstTruncationOracle {

    /** XGM2019e defining constants, verbatim from `geopotential_coeffs.rs`. */
    private static final double MODEL_GM = 398600441500000.0;

    private static final double MODEL_RE = 6378136.3;

    /**
     * Fully-normalized C-bar(n,0) of XGM2019e for n = 2..12, indexed by degree, verbatim from
     * `geopotential_coeffs::MODEL_COEFFS`. Degrees 0 and 1 are unused by the DSST zonal expansion.
     */
    private static final double[] CBAR_N0 = {
        0.0,
        0.0,
        -0.0004841694947475625,
        9.571725961869197e-07,
        5.399778363124431e-07,
        6.863789456281855e-08,
        -1.499836903807493e-07,
        9.048923326225471e-08,
        4.947703305460833e-08,
        2.802191419160437e-08,
        5.335074308452505e-08,
        -5.076095923890381e-08,
        3.645246193128655e-08,
    };

    private static final int MAX_DEGREE_TOP = CBAR_N0.length - 1;

    private static final int[] DEGREES = {2, 3, 4, 6, 8, 12};

    /** Keplerian test states as (a [m], e, i [deg]); the angles below are common to all of them. */
    private static final double[][] STATES = {
        // LEO, across the inclinations that matter (equatorial, ISS, sun-synchronous, retrograde).
        {7078137.0, 0.0, 0.0}, {7078137.0, 0.0, 28.5}, {7078137.0, 0.0, 51.6},
        {7078137.0, 0.0, 63.4}, {7078137.0, 0.0, 98.0}, {7078137.0, 0.0, 145.0},
        {7078137.0, 0.001, 0.0}, {7078137.0, 0.001, 51.6}, {7078137.0, 0.001, 98.0},
        {7078137.0, 0.01, 0.0}, {7078137.0, 0.01, 51.6}, {7078137.0, 0.01, 98.0},
        {7878137.0, 0.02, 63.4}, {7878137.0, 0.05, 63.4},
        // MEO / GNSS.
        {26560000.0, 0.0, 0.0}, {26560000.0, 0.0, 55.0}, {26560000.0, 0.0, 98.0},
        {26560000.0, 0.01, 55.0}, {26560000.0, 0.1, 55.0},
        // GEO and near-GEO.
        {42164000.0, 0.0, 0.0}, {42164000.0, 0.0001, 0.5}, {42164000.0, 0.0001, 15.0},
        // Eccentric: GTO, Molniya, and two deep-HEO states where the series is cut latest.
        {24396000.0, 0.7306, 7.0}, {24396000.0, 0.7306, 28.5},
        {26554000.0, 0.74, 63.4},
        {42164000.0, 0.5, 10.0}, {42164000.0, 0.5, 63.4},
        {42164000.0, 0.8, 10.0}, {42164000.0, 0.8, 63.4},
        // Very eccentric, high-apogee states. These are where the series is cut latest, and the
        // only place the `min(maxDegree - 2)` clamp on the answer can be reached at all.
        {100000000.0, 0.80, 20.0}, {100000000.0, 0.90, 20.0}, {100000000.0, 0.93, 63.4},
        {384400000.0, 0.95, 28.5}, {384400000.0, 0.97, 5.0},
    };

    /**
     * Two dense eccentricity sweeps, at the two altitudes where the truncation order moves fastest.
     * The step is finer than any transition the coarse grid above can resolve, so the fixture pins
     * *where* the order steps up, not merely that it does. Both stay above the surface at perigee.
     */
    private static final double[][] SWEEPS = {
        // {a, i [deg], e_max, e_step}
        {26560000.0, 55.0, 0.75, 0.0125},
        {7078137.0, 98.0, 0.06, 0.002},
        {100000000.0, 20.0, 0.93, 0.015},
    };

    private static final double PA_DEG = 30.0;

    private static final double RAAN_DEG = 40.0;

    private static final double ANOMALY_DEG = 10.0;

    public static void main(String[] args) throws Exception {
        String out = args.length > 0 ? args[0] : "orekit_dsst_truncation.csv";
        Frame frame = FramesFactory.getEME2000();
        AbsoluteDate date = AbsoluteDate.J2000_EPOCH;
        double[] cn0 = unnormalizedCn0(MAX_DEGREE_TOP);

        Field field = DSSTZonal.class.getDeclaredField("maxEccPowMeanElements");
        field.setAccessible(true);

        try (PrintWriter w = new PrintWriter(out, "UTF-8")) {
            w.println("# Orekit 13.1.7 DSSTZonal.maxEccPowMeanElements over XGM2019e zonals.");
            w.println("# cn0,n,value          -- unnormalized C(n,0) = C-bar(n,0) * sqrt(2n+1).");
            w.println("# trunc,a,k,h,q,p,lambda,max_degree,max_ecc_pow  -- lambda is mean longitude.");
            for (int n = 2; n <= MAX_DEGREE_TOP; n++) {
                w.printf("cn0,%d,%.17e%n", n, cn0[n]);
            }
            for (double[] s : states()) {
                KeplerianOrbit kep = new KeplerianOrbit(s[0], s[1],
                        FastMath.toRadians(s[2]), FastMath.toRadians(PA_DEG),
                        FastMath.toRadians(RAAN_DEG), FastMath.toRadians(ANOMALY_DEG),
                        PositionAngleType.MEAN, frame, date, MODEL_GM);
                EquinoctialOrbit orbit = new EquinoctialOrbit(kep);
                for (int maxDegree : DEGREES) {
                    int pow = truncation(frame, cn0, maxDegree, orbit, field);
                    w.printf("trunc,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%d,%d%n",
                            orbit.getA(), orbit.getEquinoctialEx(), orbit.getEquinoctialEy(),
                            orbit.getHx(), orbit.getHy(), orbit.getLM(), maxDegree, pow);
                }
            }
        }
        System.out.println("wrote " + out);
    }

    /** The coarse grid followed by the two dense eccentricity sweeps, as (a, e, i [deg]). */
    private static List<double[]> states() {
        List<double[]> all = new java.util.ArrayList<>(java.util.Arrays.asList(STATES));
        for (double[] s : SWEEPS) {
            int steps = (int) FastMath.round(s[2] / s[3]);
            for (int i = 0; i <= steps; i++) {
                all.add(new double[] {s[0], i * s[3], s[1]});
            }
        }
        return all;
    }

    /**
     * Run `computeMeanElementsTruncations` for one degree and read what it decided.
     *
     * The short-period constructor arguments are pinned to Orekit's own upper limits so this only
     * ever exercises the mean-element branch; they do not enter the truncation loop.
     */
    private static int truncation(Frame frame, double[] cn0, int maxDegree,
                                  EquinoctialOrbit orbit, Field field) throws Exception {
        double[] c = new double[maxDegree + 1];
        System.arraycopy(cn0, 0, c, 0, maxDegree + 1);
        DSSTZonal zonal = new DSSTZonal(frame, new ZonalProvider(c), maxDegree,
                FastMath.min(maxDegree - 1, 4), 2 * maxDegree + 1);
        List<ParameterDriver> drivers = zonal.getParametersDrivers();
        double[] parameters = new double[drivers.size()];
        for (int i = 0; i < parameters.length; i++) {
            parameters[i] = drivers.get(i).getValue(orbit.getDate());
        }
        AuxiliaryElements aux = new AuxiliaryElements(orbit, 1);
        zonal.initializeShortPeriodTerms(aux, PropagationType.MEAN, parameters);
        return field.getInt(zonal);
    }

    /** C(n,0) = C-bar(n,0) * sqrt(2n+1): the m = 0 case of the Kaula normalization factor. */
    private static double[] unnormalizedCn0(int maxDegree) {
        double[] c = new double[maxDegree + 1];
        for (int n = 2; n <= maxDegree; n++) {
            c[n] = CBAR_N0[n] * FastMath.sqrt(2.0 * n + 1.0);
        }
        return c;
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
            return MODEL_GM;
        }

        @Override
        public double getAe() {
            return MODEL_RE;
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

    private DsstTruncationOracle() {
    }
}
