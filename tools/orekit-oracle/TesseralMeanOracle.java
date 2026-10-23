import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.RotationConvention;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.forces.gravity.potential.TideSystem;
import org.orekit.forces.gravity.potential.UnnormalizedSphericalHarmonicsProvider;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.Transform;
import org.orekit.orbits.EquinoctialOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.PropagationType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.semianalytical.dsst.forces.DSSTTesseral;
import org.orekit.propagation.semianalytical.dsst.utilities.AuxiliaryElements;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.ParameterDriver;

import java.util.List;

/** DSSTTesseral mean-element rates for the spacelab tesseral fixtures. No orekit-data needed. */
public final class TesseralMeanOracle {

    private static final double MU = 3.98600441500000000e+14;
    private static final double AE = 6.37813646000000000e+06;
    private static final double RATE = 7.29211500000000000e-05;
    private static final double GMST = 1.234;

    /** (n, m, Cnm, Snm) unnormalized, n = 2..4. */
    private static final double[][] CS = {
        {2, 0, -1.08262689216735860e-03,  0.00000000000000000e+00},
        {2, 1, -7.74637991057574400e-10,  2.13693885436337300e-09},
        {2, 2,  1.57460625724152070e-06, -9.03896492990687600e-07},
        {3, 0,  2.53209950653743350e-06,  0.00000000000000000e+00},
        {3, 1,  2.19321948061808240e-06,  2.68492359875363950e-07},
        {3, 2,  3.09070220068959660e-07, -2.11471885565559140e-07},
        {3, 3,  1.00581879094907070e-07,  1.97234537055279140e-07},
        {4, 0,  1.62004577259847830e-06,  0.00000000000000000e+00},
        {4, 1, -5.08710241363811900e-07, -4.49279874110026540e-07},
        {4, 2,  7.83670455129086500e-08,  1.48157820133437310e-07},
        {4, 3,  5.92141380826491500e-08, -1.20029237873928450e-08},
        {4, 4, -3.98333574448002600e-09,  6.52507740576946850e-09},
    };

    private static final int MAX_DEGREE = 4;

    /** (name, a, ex=k, ey=h, hx=q, hy=p, lM). */
    private static final Object[][] CASES = {
        {"geo", 4.21641730000000000e+07, 3.09016994374947470e-03, 9.51056516295153600e-03,
         9.78368723432483700e-02, 8.80927155393123200e-02, 1.43116998663535020e+00},
        {"gps", 2.65600000000000000e+07, 3.09016994374947470e-03, 9.51056516295153600e-03,
         3.86856709931559350e-01, 3.48327346186132370e-01, 1.43116998663535020e+00},
        // Same GEO geometry with e = 0.004 (below Orekit's 0.005 maxEccPow step).
        {"geo_low_e", 4.21641730000000000e+07, 1.23606797749978980e-03, 3.80422606518061440e-03,
         9.78368723432483700e-02, 8.80927155393123200e-02, 1.43116998663535020e+00},
    };

    public static void main(String[] args) {
        Frame eme = FramesFactory.getEME2000();
        AbsoluteDate date = AbsoluteDate.J2000_EPOCH;
        Transform t = new Transform(date,
                new Rotation(Vector3D.PLUS_K, GMST, RotationConvention.FRAME_TRANSFORM));
        Frame body = new Frame(eme, t, "bodyZ", false);

        for (Object[] c : CASES) {
            String name = (String) c[0];
            EquinoctialOrbit orbit = new EquinoctialOrbit(
                    (Double) c[1], (Double) c[2], (Double) c[3], (Double) c[4], (Double) c[5],
                    (Double) c[6], PositionAngleType.MEAN, eme, date, MU);
            SpacecraftState state = new SpacecraftState(orbit);
            AuxiliaryElements aux = new AuxiliaryElements(orbit, 1);

            DSSTTesseral tesseral = new DSSTTesseral(body, RATE, new Provider());
            List<ParameterDriver> drivers = tesseral.getParametersDrivers();
            double[] parameters = new double[drivers.size()];
            for (int i = 0; i < parameters.length; i++) {
                parameters[i] = drivers.get(i).getValue(date);
            }
            tesseral.initializeShortPeriodTerms(aux, PropagationType.MEAN, parameters);
            double[] rate = tesseral.getMeanElementRate(state, aux, parameters);

            StringBuilder sb = new StringBuilder(name);
            sb.append(",").append(String.format("%.17e", orbit.getE()));
            for (double r : rate) {
                sb.append(",").append(String.format("%.17e", r));
            }
            System.out.println(sb);
        }
    }

    /** The 4x4 unnormalized field above, served directly. */
    private static final class Provider implements UnnormalizedSphericalHarmonicsProvider {

        @Override
        public int getMaxDegree() {
            return MAX_DEGREE;
        }

        @Override
        public int getMaxOrder() {
            return MAX_DEGREE;
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
                    return lookup(n, m, 2);
                }

                @Override
                public double getUnnormalizedSnm(int n, int m) {
                    return lookup(n, m, 3);
                }
            };
        }

        private static double lookup(int n, int m, int col) {
            for (double[] row : CS) {
                if ((int) row[0] == n && (int) row[1] == m) {
                    return row[col];
                }
            }
            return 0.0;
        }
    }
}
