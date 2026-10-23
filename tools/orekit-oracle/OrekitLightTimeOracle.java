import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.util.FastMath;

import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.estimation.measurements.AbstractMeasurement;
import org.orekit.estimation.measurements.EstimatedMeasurementBase;
import org.orekit.estimation.measurements.GroundStation;
import org.orekit.estimation.measurements.ObservableSatellite;
import org.orekit.estimation.measurements.Range;
import org.orekit.estimation.measurements.RangeRate;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.TopocentricFrame;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;
import org.orekit.utils.TimeStampedPVCoordinates;

/**
 * A third-lineage oracle for the light-time epoch, for {@code spacelab_estimation::observed}.
 *
 * <p>The experiment is designed to isolate one variable. The orbit is <em>two-body Keplerian</em>,
 * which both sides compute in closed form from the same Cartesian epoch state, so the trajectories
 * agree to round-off and the only thing left to disagree about is the light-time solution. The
 * comparison runs entirely in EME2000 and the station's inertial position and velocity are emitted
 * per epoch, so the Rust side needs no frame chain at all -- a frame difference would otherwise
 * enter as an offset of the same size as the effect being measured.
 *
 * <p>Two flight times are emitted per epoch. {@code tau_shifted} is what Orekit's own {@code Range}
 * uses internally -- the emitter Keplerian-shifted from the reception state -- and
 * {@code tau_resolved} re-evaluates the propagator at every iterate, which is the true fixed point.
 * Their difference is the price of Orekit's own shortcut, and is reported rather than assumed
 * negligible.
 *
 * <p>The Java source is committed as provenance, not as a dependency: we derive from mathematics and
 * cite Orekit, we do not port it.
 */
public final class OrekitLightTimeOracle {

    /** The same gravitational parameter the Rust side uses, so the two propagations are one orbit. */
    private static final double MU = 3.986004418e14;

    private static final double A = 6_378_137.0 + 550_000.0;
    private static final double E = 0.001;
    private static final double I_DEG = 53.0;
    private static final double PA_DEG = 40.0;
    private static final double RAAN_DEG = 120.0;
    private static final double ANOMALY_DEG = 10.0;

    private static final double STATION_LAT_DEG = 35.0;
    private static final double STATION_LON_DEG = 33.0;
    private static final double STATION_HEIGHT_M = 0.0;

    private static final double SPAN_SECONDS = 90_000.0;
    private static final double STEP_SECONDS = 10.0;

    public static void main(String[] args) throws IOException {
        final File dataDir = new File("/tmp/orekit-oracle/orekit-data");
        if (!dataDir.isDirectory()) {
            throw new IllegalStateException("orekit-data not found at " + dataDir);
        }
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(dataDir));

        final TimeScale utc = TimeScalesFactory.getUTC();
        final Frame eme2000 = FramesFactory.getEME2000();
        final Frame itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, itrf);

        final AbsoluteDate epoch = new AbsoluteDate(2024, 1, 1, 0, 0, 0.0, utc);
        final KeplerianOrbit orbit = new KeplerianOrbit(
                A, E, FastMath.toRadians(I_DEG), FastMath.toRadians(PA_DEG),
                FastMath.toRadians(RAAN_DEG), FastMath.toRadians(ANOMALY_DEG),
                PositionAngleType.MEAN, eme2000, epoch, MU);
        final KeplerianPropagator propagator = new KeplerianPropagator(orbit);

        final TopocentricFrame topo = new TopocentricFrame(earth, new GeodeticPoint(
                FastMath.toRadians(STATION_LAT_DEG), FastMath.toRadians(STATION_LON_DEG),
                STATION_HEIGHT_M), "oracle");
        final GroundStation station = new GroundStation(topo);
        final ObservableSatellite sat = new ObservableSatellite(0);

        // Orekit's `GroundStation` carries estimable Earth-orientation parameters, and their linear
        // models refuse to evaluate without a reference date. They are all left at zero -- this
        // oracle is about light time, not about EOP -- but the dates have to be set for the
        // measurement to evaluate at all.
        station.getPrimeMeridianOffsetDriver().setReferenceDate(epoch);
        station.getPrimeMeridianDriftDriver().setReferenceDate(epoch);
        station.getPolarOffsetXDriver().setReferenceDate(epoch);
        station.getPolarDriftXDriver().setReferenceDate(epoch);
        station.getPolarOffsetYDriver().setReferenceDate(epoch);
        station.getPolarDriftYDriver().setReferenceDate(epoch);

        final PVCoordinates pv0 = orbit.getPVCoordinates(epoch, eme2000);

        final List<String> rows = new ArrayList<>();
        for (double t = 0.0; t <= SPAN_SECONDS; t += STEP_SECONDS) {
            final AbsoluteDate date = epoch.shiftedBy(t);
            final SpacecraftState state = propagator.propagate(date);

            final double elevation = topo.getElevation(state.getPosition(eme2000), eme2000, date);
            if (elevation <= 0.0) {
                continue;
            }

            final PVCoordinates receiverPv = topo.getPVCoordinates(date, eme2000);
            final Vector3D receiver = receiverPv.getPosition();

            final TimeStampedPVCoordinates emitterPv = state.getPVCoordinates(eme2000);
            final double tauShifted = AbstractMeasurement.signalTimeOfFlightAdjustableEmitter(
                    emitterPv, receiver, date, eme2000);
            final double tauResolved = AbstractMeasurement.signalTimeOfFlightAdjustableEmitter(
                    propagator, date, receiver, date, eme2000);

            final SpacecraftState emission = propagator.propagate(date.shiftedBy(-tauResolved));
            final PVCoordinates emissionPv = emission.getPVCoordinates(eme2000);

            final Range oneWay = new Range(station, false, date, 0.0, 1.0, 1.0, sat);
            final Range twoWay = new Range(station, true, date, 0.0, 1.0, 1.0, sat);
            final RangeRate rate = new RangeRate(station, date, 0.0, 1.0, 1.0, false, sat);
            final EstimatedMeasurementBase<Range> oneWayEstimate =
                    oneWay.estimateWithoutDerivatives(new SpacecraftState[] { state });
            final EstimatedMeasurementBase<Range> twoWayEstimate =
                    twoWay.estimateWithoutDerivatives(new SpacecraftState[] { state });
            final EstimatedMeasurementBase<RangeRate> rateEstimate =
                    rate.estimateWithoutDerivatives(new SpacecraftState[] { state });

            final Vector3D instant = state.getPosition(eme2000).subtract(receiver);

            rows.add(String.format(Locale.ROOT,
                    "%.6f,%s,%s,%s,%s,%s,%s,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e",
                    t,
                    vec(receiver), vec(receiverPv.getVelocity()),
                    vec(emitterPv.getPosition()), vec(emitterPv.getVelocity()),
                    vec(emissionPv.getPosition()), vec(emissionPv.getVelocity()),
                    tauShifted, tauResolved,
                    oneWayEstimate.getEstimatedValue()[0],
                    twoWayEstimate.getEstimatedValue()[0],
                    rateEstimate.getEstimatedValue()[0],
                    instant.getNorm()));
        }

        final StringBuilder out = new StringBuilder();
        out.append("# Orekit 13.1.7 light-time oracle. Two-body Keplerian, EME2000 throughout.\n");
        out.append(String.format(Locale.ROOT, "# mu=%.17e a=%.17e e=%.17e i_deg=%.17e%n", MU, A, E, I_DEG));
        out.append(String.format(Locale.ROOT,
                "# pa_deg=%.17e raan_deg=%.17e mean_anomaly_deg=%.17e%n", PA_DEG, RAAN_DEG, ANOMALY_DEG));
        out.append(String.format(Locale.ROOT, "# epoch_state=%s,%s%n",
                vec(pv0.getPosition()), vec(pv0.getVelocity())));
        out.append("# t,rx,ry,rz,rvx,rvy,rvz,sx,sy,sz,svx,svy,svz,ex,ey,ez,evx,evy,evz,"
                + "tau_shifted,tau_resolved,range_one_way,range_two_way,range_rate,range_instantaneous\n");
        for (String row : rows) {
            out.append(row).append('\n');
        }
        Files.write(Paths.get("orekit_light_time.csv"), out.toString().getBytes("UTF-8"));
        System.out.println("rows: " + rows.size());
    }

    private static String vec(Vector3D v) {
        return String.format(Locale.ROOT, "%.17e,%.17e,%.17e", v.getX(), v.getY(), v.getZ());
    }
}
