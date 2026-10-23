import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.optim.nonlinear.vector.leastsquares.LevenbergMarquardtOptimizer;
import org.orekit.bodies.CelestialBody;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.data.DataContext;
import org.orekit.data.DataSource;
import org.orekit.data.DirectoryCrawler;
import org.orekit.estimation.leastsquares.BatchLSEstimator;
import org.orekit.estimation.measurements.ObservableSatellite;
import org.orekit.estimation.measurements.ObservedMeasurement;
import org.orekit.estimation.measurements.Position;
import org.orekit.files.sp3.SP3;
import org.orekit.files.sp3.SP3Parser;
import org.orekit.forces.ForceModel;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.Relativity;
import org.orekit.forces.gravity.ThirdBodyAttraction;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient;
import org.orekit.forces.radiation.SolarRadiationPressure;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.BoundedPropagator;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.conversion.DormandPrince853IntegratorBuilder;
import org.orekit.propagation.conversion.NumericalPropagatorBuilder;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.ParameterDriver;
import org.orekit.utils.PVCoordinates;
import org.orekit.utils.TimeStampedPVCoordinates;

/**
 * Independent-lineage cross-check for the IGS precise-orbit fit of
 * {@code crates/spacelab/tests/igs_orbit_fit.rs}.
 *
 * <p>Same file, same satellite, same seven estimated parameters — six epoch state components and
 * one scale on an isotropic solar-radiation pressure — solved by Orekit's {@code BatchLSEstimator}
 * against Orekit's own SP3 parser, frame chain, DE440 ephemerides and Holmes-Featherstone
 * geopotential. Nothing is shared but the product file, so agreement in the post-fit RMS is
 * evidence about the physics rather than about either implementation.
 *
 * <p>Two model differences are deliberate and are the reason the two numbers are not expected to
 * agree to the last digit: Orekit reads EIGEN-6S from {@code orekit-data} where the Rust side uses
 * XGM2019e (identical to ~1e-9 relative below degree 12), and Orekit's third bodies come from DE440
 * where the Rust side uses the low-precision analytic Sun and Moon.
 *
 * <p>Usage: {@code java -cp "jars/*:." IgsOrbitFitOracle <file.SP3> [satelliteId]}
 */
public final class IgsOrbitFitOracle {

    /** Geopotential truncation, matching the Rust side and IGS analysis-centre practice. */
    private static final int DEGREE = 12;
    /** Nominal solar-radiation acceleration in full sun at 1 AU, m/s². */
    private static final double SRP_NOMINAL = 1.0e-7;
    /** Spacecraft mass, kg. With unit area and unit coefficient it only sets the SRP scale. */
    private static final double MASS = 1000.0;
    /** Assumed per-component uncertainty of an IGS final orbit position, m. */
    private static final double SIGMA = 0.05;

    private IgsOrbitFitOracle() {
    }

    public static void main(final String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: IgsOrbitFitOracle <file.SP3> [satelliteId]");
            System.exit(2);
        }
        final File data = new File(System.getProperty("orekit.data", "orekit-data"));
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(data));

        final SP3 sp3 = new SP3Parser().parse(new DataSource(new File(args[0])));
        final List<String> ids = new ArrayList<>(sp3.getSatellites().keySet());
        final List<String> wanted = args.length > 1 ? List.of(args[1]) : ids;

        final Frame gcrf = FramesFactory.getGCRF();
        final Frame itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, false);
        final CelestialBody sun = CelestialBodyFactory.getSun();
        final CelestialBody moon = CelestialBodyFactory.getMoon();
        final NormalizedSphericalHarmonicsProvider gravity =
                GravityFieldFactory.getNormalizedProvider(DEGREE, DEGREE);
        final double mu = gravity.getMu();
        // A sphere, not an ellipsoid: the Rust side's conical shadow uses one equatorial radius, and
        // the flattening would otherwise be a difference in the penumbra geometry rather than in the
        // physics under test.
        final OneAxisEllipsoid earth =
                new OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 0.0, itrf);

        System.out.println("orekit " + sp3.getSatellites().size() + " satellites, "
                + "gravity " + gravity.getMaxDegree() + "x" + gravity.getMaxOrder()
                + " mu=" + mu);
        System.out.println("sat    radial    along    cross       3D     worst   srp_scale  iters");

        for (final String id : wanted) {
            if (!sp3.getSatellites().containsKey(id)) {
                System.err.println("no such satellite in the file: " + id);
                continue;
            }
            try {
                fitOne(sp3, id, gcrf, itrf, gravity, mu, sun, moon, earth);
            } catch (RuntimeException e) {
                // One satellite that will not converge is a finding about that satellite, not a
                // reason to lose the other thirty-one.
                System.out.println(id + "  failed: " + e.getMessage());
            }
        }
    }

    private static void fitOne(final SP3 sp3, final String id, final Frame gcrf, final Frame itrf,
                               final NormalizedSphericalHarmonicsProvider gravity, final double mu,
                               final CelestialBody sun, final CelestialBody moon,
                               final OneAxisEllipsoid earth) {

        final List<TimeStampedPVCoordinates> samples = new ArrayList<>();
        sp3.getSatellites().get(id).getSegments()
                .forEach(segment -> samples.addAll(segment.getCoordinates()));
        if (samples.size() < 90) {
            System.out.println(id + "  skipped, " + samples.size() + " samples");
            return;
        }
        final Frame fileFrame = sp3.getSatellites().get(id).getFrame();
        final AbsoluteDate epoch = samples.get(0).getDate();

        // Measurements: the product's positions, rotated into the propagation frame.
        final ObservableSatellite satellite = new ObservableSatellite(0);
        final List<ObservedMeasurement<?>> measurements = new ArrayList<>();
        for (final TimeStampedPVCoordinates pv : samples) {
            final Vector3D inertial = fileFrame.getStaticTransformTo(gcrf, pv.getDate())
                    .transformPosition(pv.getPosition());
            measurements.add(new Position(pv.getDate(), inertial, SIGMA, 1.0, satellite));
        }

        // The starting guess comes from the file's own interpolating propagator, so the two
        // implementations do not even share an initial state.
        final BoundedPropagator guess = sp3.getSatellites().get(id).getPropagator();
        final PVCoordinates pv0 = guess.getPVCoordinates(epoch, gcrf);
        final Orbit initial = new CartesianOrbit(pv0, gcrf, epoch, mu);

        final NumericalPropagatorBuilder builder = new NumericalPropagatorBuilder(
                initial,
                new DormandPrince853IntegratorBuilder(1.0, 600.0, 1.0e-3),
                PositionAngleType.TRUE,
                1.0);
        builder.setMass(MASS);

        final SolarRadiationPressure srp = new SolarRadiationPressure(
                Constants.IAU_2012_ASTRONOMICAL_UNIT,
                SRP_NOMINAL * MASS,
                sun,
                earth,
                new IsotropicRadiationSingleCoefficient(1.0, 1.0));

        for (final ForceModel force : List.of(
                new HolmesFeatherstoneAttractionModel(itrf, gravity),
                new ThirdBodyAttraction(sun),
                new ThirdBodyAttraction(moon),
                new Relativity(mu),
                srp)) {
            builder.addForceModel(force);
        }
        for (final ParameterDriver driver : srp.getParametersDrivers()) {
            driver.setSelected(true);
        }

        final BatchLSEstimator estimator =
                new BatchLSEstimator(new LevenbergMarquardtOptimizer(), builder);
        estimator.setParametersConvergenceThreshold(1.0e-4);
        estimator.setMaxIterations(60);
        estimator.setMaxEvaluations(200);
        measurements.forEach(estimator::addMeasurement);

        final Propagator[] estimated = estimator.estimate();

        double srpScale = Double.NaN;
        for (final ParameterDriver driver : srp.getParametersDrivers()) {
            srpScale = driver.getValue();
        }

        // Post-fit residuals in the fitted orbit's radial / along-track / cross-track frame.
        double sumR = 0;
        double sumA = 0;
        double sumC = 0;
        double sum3 = 0;
        double worst = 0;
        for (final TimeStampedPVCoordinates pv : samples) {
            final Vector3D observed = fileFrame.getStaticTransformTo(gcrf, pv.getDate())
                    .transformPosition(pv.getPosition());
            final PVCoordinates fitted = estimated[0].getPVCoordinates(pv.getDate(), gcrf);
            final Vector3D d = fitted.getPosition().subtract(observed);
            final Vector3D radial = fitted.getPosition().normalize();
            final Vector3D cross = Vector3D.crossProduct(fitted.getPosition(),
                    fitted.getVelocity()).normalize();
            final Vector3D along = Vector3D.crossProduct(cross, radial);
            sumR += square(Vector3D.dotProduct(d, radial));
            sumA += square(Vector3D.dotProduct(d, along));
            sumC += square(Vector3D.dotProduct(d, cross));
            sum3 += d.getNormSq();
            worst = Math.max(worst, d.getNorm());
        }
        final int n = samples.size();
        System.out.printf("%s  %8.3f %8.3f %8.3f %8.3f %9.3f   %7.4f  %5d%n",
                id,
                Math.sqrt(sumR / n), Math.sqrt(sumA / n), Math.sqrt(sumC / n),
                Math.sqrt(sum3 / n), worst, srpScale, estimator.getIterationsCount());
    }

    private static double square(final double x) {
        return x * x;
    }
}
