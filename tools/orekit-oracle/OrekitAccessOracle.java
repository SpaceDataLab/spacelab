import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Locale;

import org.hipparchus.util.FastMath;

import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.TopocentricFrame;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.tle.TLE;
import org.orekit.propagation.analytical.tle.TLEPropagator;
import org.orekit.propagation.events.ElevationDetector;
import org.orekit.propagation.events.EventsLogger;
import org.orekit.propagation.events.handlers.ContinueOnEvent;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;

/**
 * A third-lineage oracle for access-window edges: Orekit's own {@link ElevationDetector} over a
 * ground station, for the ISS element set, emitting the rise/set times a from-scratch access
 * scanner must reproduce. Orekit's SGP4 shares Vallado's lineage with the {@code sgp4} crate, so the
 * propagation is a transcription check; its topocentric-frame and root-finding event machinery are
 * its own, and are what the Rust {@code access_windows} is validated against
 * ({@code crates/spacelab/tests/access_oracle.rs}, plan C2).
 *
 * <p>Earth orientation is left at Orekit's bundled EOP; the Rust side uses zero EOP, so the residual
 * is the frame difference, tens of milliseconds — well inside the plan's one-second tolerance.
 */
public final class OrekitAccessOracle {

    // ISS, epoch 2008-264, matching the ground-track and coverage tests.
    private static final String LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    private static final String LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    private static final double STATION_LAT_DEG = 35.0;
    private static final double STATION_LON_DEG = 33.0;
    private static final double MIN_ELEVATION_DEG = 10.0;
    private static final double SPAN_SECONDS = 86_400.0;

    public static void main(String[] args) throws IOException {
        final File dataDir = new File("/tmp/orekit-oracle/orekit-data");
        if (!dataDir.isDirectory()) {
            throw new IllegalStateException("orekit-data not found at " + dataDir);
        }
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(dataDir));

        final TimeScale utc = TimeScalesFactory.getUTC();
        final Frame itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, itrf);
        final GeodeticPoint stationPoint = new GeodeticPoint(
                FastMath.toRadians(STATION_LAT_DEG), FastMath.toRadians(STATION_LON_DEG), 0.0);
        final TopocentricFrame station = new TopocentricFrame(earth, stationPoint, "station");

        final TLE tle = new TLE(LINE1, LINE2);
        final TLEPropagator propagator = TLEPropagator.selectExtrapolator(tle);

        // Start the scan at 2008-09-20 12:00 UTC, as the Rust coverage tests do.
        final AbsoluteDate start = new AbsoluteDate(2008, 9, 20, 12, 0, 0.0, utc);

        final EventsLogger logger = new EventsLogger();
        final ElevationDetector detector = new ElevationDetector(station)
                .withConstantElevation(FastMath.toRadians(MIN_ELEVATION_DEG))
                .withMaxCheck(60.0)
                .withThreshold(1.0e-6)
                .withHandler(new ContinueOnEvent()); // log every pass, do not stop at the first
        propagator.addEventDetector(logger.monitorDetector(detector));
        propagator.propagate(start, start.shiftedBy(SPAN_SECONDS));

        // Emit rise/set as offsets in seconds from the start; increasing elevation = rise.
        final StringBuilder csv = new StringBuilder("kind,offset_seconds\n");
        int rises = 0;
        int sets = 0;
        for (EventsLogger.LoggedEvent e : logger.getLoggedEvents()) {
            final SpacecraftState s = e.getState();
            final double offset = s.getDate().durationFrom(start);
            final boolean rise = e.isIncreasing();
            if (rise) {
                rises++;
            } else {
                sets++;
            }
            csv.append(String.format(Locale.ROOT, "%s,%.6f%n", rise ? "rise" : "set", offset));
        }

        final File out = new File(
                "crates/spacelab/tests/fixtures/orekit_iss_passes.csv");
        out.getParentFile().mkdirs();
        Files.writeString(out.toPath(), csv.toString());
        System.out.printf(Locale.ROOT, "wrote %d rises, %d sets to %s%n", rises, sets, out.getPath());
    }
}
