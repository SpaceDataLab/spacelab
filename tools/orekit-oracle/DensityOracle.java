import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.data.DataContext;
import org.orekit.data.DataSource;
import org.orekit.data.DirectoryCrawler;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.models.earth.atmosphere.Atmosphere;
import org.orekit.models.earth.atmosphere.DTM2000;
import org.orekit.models.earth.atmosphere.NRLMSISE00;
import org.orekit.models.earth.atmosphere.data.CssiSpaceWeatherData;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.IERSConventions;

/**
 * Independent-lineage baseline for the measured-density comparison of
 * {@code crates/spacelab-propagation/tests/measured_density.rs}.
 *
 * <p>Same TU Delft density file, same days, same three families of empirical thermosphere model,
 * driven from the same published index series — but through Orekit's own file loaders, its own
 * geodetic-to-inertial chain, its own solar position and its own model code. Nothing is shared but
 * the measurement and the index files, so agreement in the bias against measurement is evidence
 * about the models rather than about either implementation.
 *
 * <p>One model differs deliberately and is reported as such: Orekit ships DTM2000, not DTM2020. The
 * Rust side's DTM2020 is a generation newer and was fitted on this very data family, so the pair is
 * a statement about what twenty years of refitting bought, not a parity check.
 *
 * <p>Usage: {@code java -cp "jars/*:." DensityOracle <density.txt> <SW-All.txt> <SOLFSMY.TXT> <DTCFILE.TXT>}
 */
public final class DensityOracle {

    /** GRS80 equatorial radius, m — the ellipsoid the product's geodetic altitude is above. */
    private static final double GRS80_A = 6378137.0;
    /** GRS80 flattening. */
    private static final double GRS80_F = 1.0 / 298.257222101;

    private DensityOracle() {
    }

    /** One row of the TU Delft product. */
    private record Row(AbsoluteDate date, GeodeticPoint point, double measured) {
    }

    public static void main(final String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: DensityOracle <density.txt> [more.txt ...]");
            System.exit(2);
        }
        final File data = new File(System.getProperty("orekit.data", "orekit-data"));
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(data));

        final Frame itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, false);
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(GRS80_A, GRS80_F, itrf);
        final var sun = CelestialBodyFactory.getSun();

        // The indices come from orekit-data's own CSSI and Space Environment Technologies mirrors,
        // in the fixed-width layouts Orekit parses, rather than from the CSV the Rust side reads.
        // Same publishers, separately parsed — a driver disagreement would show up as one.
        final CssiSpaceWeatherData cssi =
                new CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES);

        final Atmosphere[] models = {
            new NRLMSISE00(cssi, sun, earth),
            new DTM2000(cssi, sun, earth),
        };
        final String[] names = {
            "NRLMSISE-00",
            "DTM2000",
        };

        for (final String file : args) {
            final List<Row> rows = read(file);
            System.out.printf(Locale.ROOT, "%s: %d usable rows%n", new File(file).getName(), rows.size());
            for (int m = 0; m < models.length; ++m) {
                int n = 0;
                double sum = 0;
                double sq = 0;
                for (final Row r : rows) {
                    final Vector3D position = earth.transform(r.point());
                    final double rho = models[m].getDensity(r.date(), position, itrf);
                    if (!Double.isFinite(rho) || rho <= 0) {
                        continue;
                    }
                    final double ratio = rho / r.measured();
                    ++n;
                    sum += ratio;
                    sq += (ratio - 1) * (ratio - 1);
                }
                System.out.printf(Locale.ROOT, "  %-12s n %4d  bias %.3f  rms %.3f%n",
                                  names[m], n, sum / n, Math.sqrt(sq / n));
            }
        }
    }

    /**
     * Read the product. Nine whitespace-separated fields are always present: date, time, time
     * system, altitude (m), longitude and latitude (deg), local solar time, argument of latitude,
     * and density (kg/m³). The accelerometer missions add three more — a running orbit mean and two
     * quality flags — which the Swarm files, whose density comes from precise orbit determination,
     * do not carry. Demanding twelve reads the whole Swarm archive as zero rows without complaining,
     * so the width gate is nine and the flag is read only when it is there. Rows flagged non-nominal
     * are dropped, as they are on the Rust side.
     */
    private static List<Row> read(final String path) throws Exception {
        final TimeScale gps = TimeScalesFactory.getGPS();
        final TimeScale utc = TimeScalesFactory.getUTC();
        final List<Row> rows = new ArrayList<>();
        try (var lines = java.nio.file.Files.lines(java.nio.file.Path.of(path))) {
            for (final String line : (Iterable<String>) lines::iterator) {
                if (line.isBlank() || line.startsWith("#")) {
                    continue;
                }
                final String[] t = line.trim().split("\\s+");
                if (t.length < 9) {
                    continue;
                }
                if (t.length > 10 && Double.parseDouble(t[10]) != 0.0) {
                    continue;
                }
                final double rho = Double.parseDouble(t[8]);
                if (rho <= 0) {
                    continue;
                }
                final TimeScale scale = "GPS".equals(t[2]) ? gps : utc;
                final AbsoluteDate date = new AbsoluteDate(t[0] + "T" + t[1], scale);
                final GeodeticPoint point = new GeodeticPoint(Math.toRadians(Double.parseDouble(t[5])),
                                                              Math.toRadians(Double.parseDouble(t[4])),
                                                              Double.parseDouble(t[3]));
                rows.add(new Row(date, point, rho));
            }
        }
        return rows;
    }
}
