import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.util.FastMath;
import org.hipparchus.util.MathUtils;

import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.frames.EOPEntry;
import org.orekit.frames.Frame;
import org.orekit.frames.Frames;
import org.orekit.frames.ITRFVersion;
import org.orekit.frames.Transform;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.DateComponents;
import org.orekit.time.TimeComponents;
import org.orekit.time.TimeScalarFunction;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScales;
import org.orekit.time.TimeScalesFactory;
import org.orekit.time.UT1Scale;
import org.orekit.time.UTCScale;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;

/**
 * Third-lineage numerical oracle for the TEME -> ITRF transformation, computed with Orekit.
 *
 * <p>Orekit implements the IERS Conventions directly and descends from neither SOFA nor
 * Vallado, so its numbers are an independent cross-check for our Rust implementation and for
 * our SOFA-lineage (pyerfa/skyfield) and Vallado-lineage (python-sgp4) oracles.</p>
 *
 * <h2>Design decisions (each one can move the answer, so they are explicit)</h2>
 * <ul>
 *   <li><b>Explicit EOP, no reliance on bundled values.</b> For every epoch we build our own
 *       {@link org.orekit.frames.EOPHistory} from a list of identical daily {@link EOPEntry}
 *       around the epoch. Because the entries are constant, degree-3 interpolation returns our
 *       exact (xp, yp, dut1) at the epoch. We verify this at run time (assertion) and print the
 *       values Orekit actually used, so the inputs are fully reproducible.</li>
 *   <li><b>simpleEOP = true.</b> Disables sub-daily tidal EOP corrections, otherwise the pole
 *       and dut1 Orekit uses would differ from the constants we specify.</li>
 *   <li><b>Nutation corrections = 0.</b> ddPsi/ddEps (equinox) and dx/dy (CIO) are all zero, so
 *       the transform is a pure IAU precession-nutation model with no empirical correction. This
 *       isolates the model, which is what a from-scratch implementation reproduces.</li>
 *   <li><b>lod = 0.</b> {@code TIRFProvider} computes the Earth rotation rate as
 *       {@code AVE * (1 - lod/86400)} with {@code AVE = 7.292115146706979e-5 rad/s}. lod = 0
 *       gives exactly the nominal rate, which drives the omega x r term in the ITRF velocity.</li>
 *   <li><b>ITRF realization.</b> We use {@link Frames#buildUncachedITRF(UT1Scale)}, which chains
 *       GCRF -> CIRF -> TIRF -> ITRF and applies <em>only</em> polar motion plus the secular TIO
 *       locator s'. It performs <em>no</em> ITRF-datum (Helmert) step, so there is no ITRF-2014
 *       vs ITRF-2020 ambiguity here: this is the "raw" CIO-based terrestrial frame, exactly what a
 *       from-scratch TEME->ITRF routine produces. The ITRFVersion tag on each EOPEntry is inert.</li>
 *   <li><b>TEME provider.</b> Orekit defines TEME once, from TOD with IERS-1996 conventions and
 *       EOP corrections switched off ({@code getTOD(IERS_1996, false, true)}); it is
 *       convention-independent. So when we "try both IERS_2010 and IERS_1996" only the ITRF side
 *       changes, and the difference measures the CIRF precession-nutation model difference
 *       (IAU-2006/2000A vs IAU-1976/1980). The TIRF rotation (ERA) is convention-independent.</li>
 *   <li><b>GMST(IAU-1982).</b> Taken from {@code IERSConventions.IERS_1996.getGMSTFunction(ut1)},
 *       whose source uses the classic constants 24110.54841, 8640184.812866, 0.093104, -6.2e-6 —
 *       i.e. the IAU-1982 GMST-UT1 model, our {@code gmst82}. IERS_2003/2010 use ERA-based GMST
 *       and are NOT this oracle. We report the value normalized to [0, 2*pi).</li>
 * </ul>
 */
public final class OrekitTemeItrfOracle {

    /** Radians per arcsecond. */
    private static final double ARCSEC = Constants.ARC_SECONDS_TO_RADIANS;

    /** Nominal mean Earth angular velocity used by Orekit's TIRFProvider (rad/s). */
    private static final double AVE = 7.292115146706979e-5;

    /** simpleEOP flag: true => no sub-daily tidal EOP corrections. */
    private static final boolean SIMPLE_EOP = true;

    /** Half-width (days) of the constant-EOP window built around each epoch. */
    private static final int EOP_WINDOW_DAYS = 10;

    // ---- Fixed TEME test states (metres, metres/second) ----

    /** AIAA-2006-6753 TEME example position (m). */
    private static final Vector3D R_TEME_AIAA = new Vector3D(-9060473.73569, 4658709.52502, 813686.73153);
    /** ISS-like TEME position (m). */
    private static final Vector3D R_TEME_ISS = new Vector3D(-2932912.880, -6041930.794, -509897.582);
    /** TEME velocity (m/s), transformed with the omega x r term included. */
    private static final Vector3D V_TEME = new Vector3D(-2233.348094, -4110.136162, -3157.394074);

    /** One epoch plus the constant EOP we impose at it. xp/yp in arcsec, dut1 in seconds. */
    private record Fixture(int year, int month, int day, int hour, int minute, double second,
                           double xpArcsec, double ypArcsec, double dut1) { }

    /**
     * ~12 epochs spanning 2000-2030. xp/yp/dut1 are hand-chosen constants of realistic magnitude;
     * their exact values are irrelevant as long as our Rust code is fed the identical numbers.
     * Includes leap-second-day instants (2012-06-30 and 2016-12-31, both mid-leap at 23:59:60.5)
     * and a 2005 instant, per the requirement.
     */
    private static final Fixture[] FIXTURES = {
        new Fixture(2000,  1,  1, 12,  0,  0.0,  0.043, 0.377,  0.3554),
        new Fixture(2003,  6, 15,  0,  0,  0.0,  0.100, 0.420, -0.3500),
        new Fixture(2005,  3, 20,  6,  0,  0.0,  0.020, 0.380, -0.5500),
        new Fixture(2008,  9, 22, 18, 30,  0.0,  0.200, 0.330, -0.2000),
        new Fixture(2010, 12, 25,  0,  0,  0.0,  0.120, 0.280,  0.1000),
        new Fixture(2012,  6, 30, 23, 59, 60.5,  0.150, 0.450,  0.4200),  // leap second
        new Fixture(2015,  7,  1,  0,  0,  0.0,  0.050, 0.440, -0.3200),
        new Fixture(2016, 12, 31, 23, 59, 60.5,  0.050, 0.300,  0.3000),  // leap second
        new Fixture(2016, 12, 31, 12,  0,  0.0,  0.060, 0.310,  0.3100),  // same day, safe instant
        new Fixture(2020,  2, 29, 12,  0,  0.0,  0.080, 0.300, -0.1800),  // leap day
        new Fixture(2024, 11,  5, 14, 22, 33.0,  0.140, 0.280,  0.0500),
        new Fixture(2029, 12, 31, 23, 59, 59.0,  0.100, 0.350,  0.0000),
    };

    private static final IERSConventions[] CONVENTIONS = {
        IERSConventions.IERS_2010, IERSConventions.IERS_1996
    };

    private OrekitTemeItrfOracle() { }

    public static void main(String[] args) throws IOException {
        final File dataDir = new File("/tmp/orekit-oracle/orekit-data");
        if (!dataDir.isDirectory()) {
            throw new IllegalStateException("orekit-data not found at " + dataDir);
        }
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(dataDir));

        // Default UTC (from bundled tai-utc.dat) supplies the leap-second model we reuse verbatim.
        final UTCScale defaultUtc = TimeScalesFactory.getUTC();

        final String orekitVersion = "13.1.7";
        final String hipparchusVersion = "4.0.3";

        final StringBuilder json = new StringBuilder();
        final StringBuilder md = new StringBuilder();

        md.append("# Orekit TEME -> ITRF oracle fixtures\n\n");
        md.append("- Orekit version: ").append(orekitVersion).append('\n');
        md.append("- Hipparchus version: ").append(hipparchusVersion).append('\n');
        md.append("- simpleEOP: ").append(SIMPLE_EOP).append('\n');
        md.append("- Nutation corrections (ddPsi, ddEps, dx, dy): 0 (pure IAU model)\n");
        md.append("- LOD: 0 s  => Earth rate = AVE = ").append(fmt(AVE)).append(" rad/s\n");
        md.append("- ITRF: raw CIO-based (buildUncachedITRF); polar motion + secular s' only, no Helmert datum step\n");
        md.append("- Matrix convention: row-major M with r_itrf = M * r_teme\n");
        md.append("- TEME: Orekit's single definition (TOD/IERS-1996, EOP corrections off)\n\n");

        json.append("{\n");
        json.append("  \"orekit_version\": \"").append(orekitVersion).append("\",\n");
        json.append("  \"hipparchus_version\": \"").append(hipparchusVersion).append("\",\n");
        json.append("  \"simple_eop\": ").append(SIMPLE_EOP).append(",\n");
        json.append("  \"nutation_corrections\": \"ddPsi=ddEps=dx=dy=0 (pure IAU model)\",\n");
        json.append("  \"lod_s\": 0.0,\n");
        json.append("  \"ave_earth_rate_rad_s\": ").append(fmt(AVE)).append(",\n");
        json.append("  \"itrf_realization\": \"raw CIO-based buildUncachedITRF; polar motion + secular s'; no Helmert datum\",\n");
        json.append("  \"matrix_convention\": \"row-major, r_itrf = M * r_teme\",\n");
        json.append("  \"test_vectors\": {\n");
        json.append("    \"r_teme_aiaa_m\": ").append(vec(R_TEME_AIAA)).append(",\n");
        json.append("    \"r_teme_iss_m\": ").append(vec(R_TEME_ISS)).append(",\n");
        json.append("    \"v_teme_m_s\": ").append(vec(V_TEME)).append("\n");
        json.append("  },\n");
        json.append("  \"epochs\": [\n");

        for (int fi = 0; fi < FIXTURES.length; fi++) {
            final Fixture f = FIXTURES[fi];

            final double xpRad = f.xpArcsec() * ARCSEC;
            final double ypRad = f.ypArcsec() * ARCSEC;

            // One TimeScales instance per epoch carrying our constant EOP for all conventions.
            final List<EOPEntry> entries = new ArrayList<>();
            final TimeScales ts = TimeScales.of(defaultUtc.getBaseOffsets(),
                    (conv, tss) -> entries);
            final UTCScale utc = ts.getUTC();
            final AbsoluteDate date =
                    new AbsoluteDate(f.year(), f.month(), f.day(), f.hour(), f.minute(), f.second(), utc);

            // Build ~21 identical daily EOP entries bracketing the epoch, so interpolation is exact.
            // Distinct MJDs are generated directly (a shiftedBy scheme can collide across a leap
            // second). Rates are exactly zero: constant EOP has zero derivatives, and passing
            // zeros avoids Orekit's derivative-reconstruction path (which cannot fit constants).
            final int epochMjd = date.getComponents(utc).getDate().getMJD();
            for (int k = -EOP_WINDOW_DAYS; k <= EOP_WINDOW_DAYS; k++) {
                final int mjd = epochMjd + k;
                final DateComponents dc =
                        new DateComponents(DateComponents.MODIFIED_JULIAN_EPOCH, mjd);
                final AbsoluteDate entryDate = new AbsoluteDate(dc, TimeComponents.H00, utc);
                entries.add(new EOPEntry(
                        mjd,
                        f.dut1(),          // UT1-UTC (s)
                        0.0,               // LOD (s) -> nominal Earth rate
                        xpRad, ypRad,      // pole motion (rad)
                        0.0, 0.0,          // pole rates: exactly zero (constant EOP)
                        0.0, 0.0,          // ddPsi, ddEps (rad)
                        0.0, 0.0,          // dx, dy (rad)
                        ITRFVersion.ITRF_2014, // inert (no Helmert in buildUncachedITRF)
                        entryDate));
            }

            final Frames frames = Frames.of(ts, () -> null);
            final Frame teme = frames.getTEME();

            // GMST(IAU-1982): a function of UT1 built from IERS-1996 conventions.
            final UT1Scale ut1_1996 = ts.getUT1(IERSConventions.IERS_1996, SIMPLE_EOP);
            final TimeScalarFunction gmst82fn = IERSConventions.IERS_1996.getGMSTFunction(ut1_1996, ts);
            final double gmstRaw = gmst82fn.value(date);
            final double gmst82 = MathUtils.normalizeAngle(gmstRaw, FastMath.PI); // -> [0, 2*pi)

            // Convention-independent time offsets and the (verified) EOP inputs Orekit used.
            final double datS = -utc.offsetFromTAI(date).toDouble();          // TAI - UTC
            final double ut1MinusTai = ut1_1996.offsetFromTAI(date).toDouble(); // UT1 - TAI

            final String iso = date.toString(utc);

            // Per-convention results.
            final ConvResult[] results = new ConvResult[CONVENTIONS.length];
            for (int ci = 0; ci < CONVENTIONS.length; ci++) {
                final IERSConventions conv = CONVENTIONS[ci];
                final UT1Scale ut1 = ts.getUT1(conv, SIMPLE_EOP);
                final Frame itrf = frames.buildUncachedITRF(ut1);
                final Transform t = teme.getTransformTo(itrf, date);

                // Verify the EOP Orekit used equals what we specified (no interpolation surprise).
                final org.orekit.frames.EOPHistory eop = ut1.getEOPHistory();
                final org.orekit.frames.PoleCorrection pc = eop.getPoleCorrection(date);
                assertClose("xp", pc.getXp(), xpRad, 1e-18);
                assertClose("yp", pc.getYp(), ypRad, 1e-18);
                assertClose("dut1", eop.getUT1MinusUTC(date), f.dut1(), 1e-12);

                final Vector3D rAiaa = t.transformPosition(R_TEME_AIAA);
                final Vector3D rIss = t.transformPosition(R_TEME_ISS);
                final Vector3D vAiaa = t.transformPVCoordinates(
                        new PVCoordinates(R_TEME_AIAA, V_TEME)).getVelocity();

                final double[] m = resolveMatrix(t);

                results[ci] = new ConvResult(conv.name(), m, rAiaa, rIss, vAiaa);
            }

            // IERS_2010 vs IERS_1996 sensitivity, in metres, on the AIAA test position.
            final double diff2010m1996 =
                    results[0].rAiaa().subtract(results[1].rAiaa()).getNorm();

            appendMd(md, fi, iso, f, datS, ut1MinusTai, gmst82, diff2010m1996, results);
            appendJson(json, fi == FIXTURES.length - 1, iso, f, datS, ut1MinusTai,
                       gmst82, diff2010m1996, results);
        }

        json.append("  ]\n}\n");

        Files.writeString(Paths.get("/tmp/orekit-oracle/fixtures.json"), json.toString());
        Files.writeString(Paths.get("/tmp/orekit-oracle/fixtures.md"), md.toString());

        System.out.println(md);
        System.out.println("Wrote /tmp/orekit-oracle/fixtures.json and /tmp/orekit-oracle/fixtures.md");
    }

    /** Per-convention transform result. */
    private record ConvResult(String convention, double[] matrix,
                              Vector3D rAiaa, Vector3D rIss, Vector3D vAiaa) { }

    /**
     * Return the row-major 3x3 matrix M with r_itrf = M * r_teme. Hipparchus'
     * {@link Rotation#getMatrix()} may express either M or M^T depending on convention, so we
     * pick whichever reproduces {@link Transform#transformPosition} on both test vectors and
     * assert the residual is negligible.
     */
    private static double[] resolveMatrix(final Transform t) {
        final Rotation r = t.getRotation();
        final double[][] g = r.getMatrix();
        final double[][] gt = transpose(g);
        final Vector3D want1 = t.transformPosition(R_TEME_AIAA);
        final Vector3D want2 = t.transformPosition(R_TEME_ISS);
        final double errG = apply(g, R_TEME_AIAA).subtract(want1).getNorm()
                          + apply(g, R_TEME_ISS).subtract(want2).getNorm();
        final double errGt = apply(gt, R_TEME_AIAA).subtract(want1).getNorm()
                           + apply(gt, R_TEME_ISS).subtract(want2).getNorm();
        final double[][] chosen = errG <= errGt ? g : gt;
        final double err = Math.min(errG, errGt);
        final double scale = R_TEME_AIAA.getNorm() + R_TEME_ISS.getNorm();
        if (err > 1e-6 * scale) {
            throw new IllegalStateException("matrix orientation unresolved; residual = " + err + " m");
        }
        return new double[] {
            chosen[0][0], chosen[0][1], chosen[0][2],
            chosen[1][0], chosen[1][1], chosen[1][2],
            chosen[2][0], chosen[2][1], chosen[2][2]
        };
    }

    private static Vector3D apply(final double[][] m, final Vector3D v) {
        return new Vector3D(
            m[0][0] * v.getX() + m[0][1] * v.getY() + m[0][2] * v.getZ(),
            m[1][0] * v.getX() + m[1][1] * v.getY() + m[1][2] * v.getZ(),
            m[2][0] * v.getX() + m[2][1] * v.getY() + m[2][2] * v.getZ());
    }

    private static double[][] transpose(final double[][] m) {
        return new double[][] {
            { m[0][0], m[1][0], m[2][0] },
            { m[0][1], m[1][1], m[2][1] },
            { m[0][2], m[1][2], m[2][2] }
        };
    }

    private static void assertClose(final String what, final double got, final double expected,
                                    final double tol) {
        if (Math.abs(got - expected) > tol) {
            throw new IllegalStateException(String.format(Locale.US,
                "EOP reproduction failed for %s: got %.17g expected %.17g", what, got, expected));
        }
    }

    /** 17-significant-digit, locale-stable formatting. */
    private static String fmt(final double x) {
        return String.format(Locale.US, "%.17g", x);
    }

    private static String vec(final Vector3D v) {
        return "[" + fmt(v.getX()) + ", " + fmt(v.getY()) + ", " + fmt(v.getZ()) + "]";
    }

    private static String arr(final double[] a) {
        final StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            if (i > 0) {
                b.append(", ");
            }
            b.append(fmt(a[i]));
        }
        return b.append("]").toString();
    }

    private static void appendMd(final StringBuilder md, final int idx, final String iso,
                                 final Fixture f, final double datS, final double ut1MinusTai,
                                 final double gmst82, final double diff, final ConvResult[] results) {
        md.append("## Epoch ").append(idx).append(": ").append(iso).append("\n\n");
        md.append("- xp = ").append(fmt(f.xpArcsec())).append(" arcsec, yp = ")
          .append(fmt(f.ypArcsec())).append(" arcsec\n");
        md.append("- dut1 (UT1-UTC) = ").append(fmt(f.dut1())).append(" s\n");
        md.append("- dat (TAI-UTC)  = ").append(fmt(datS)).append(" s\n");
        md.append("- UT1-TAI = ").append(fmt(ut1MinusTai)).append(" s\n");
        md.append("- GMST(IAU-1982) = ").append(fmt(gmst82)).append(" rad\n");
        md.append("- |ITRF_2010 - ITRF_1996| on AIAA position = ").append(fmt(diff)).append(" m\n\n");
        for (final ConvResult r : results) {
            md.append("### ").append(r.convention()).append("\n\n");
            md.append("Matrix TEME->ITRF (row-major, r_itrf = M*r_teme):\n\n");
            md.append("```\n");
            md.append(fmt(r.matrix()[0])).append("  ").append(fmt(r.matrix()[1])).append("  ")
              .append(fmt(r.matrix()[2])).append('\n');
            md.append(fmt(r.matrix()[3])).append("  ").append(fmt(r.matrix()[4])).append("  ")
              .append(fmt(r.matrix()[5])).append('\n');
            md.append(fmt(r.matrix()[6])).append("  ").append(fmt(r.matrix()[7])).append("  ")
              .append(fmt(r.matrix()[8])).append('\n');
            md.append("```\n\n");
            md.append("- r_itrf(AIAA) = ").append(vec(r.rAiaa())).append(" m\n");
            md.append("- r_itrf(ISS)  = ").append(vec(r.rIss())).append(" m\n");
            md.append("- v_itrf(AIAA) = ").append(vec(r.vAiaa())).append(" m/s\n\n");
        }
    }

    private static void appendJson(final StringBuilder json, final boolean last, final String iso,
                                   final Fixture f, final double datS, final double ut1MinusTai,
                                   final double gmst82, final double diff, final ConvResult[] results) {
        json.append("    {\n");
        json.append("      \"utc\": \"").append(iso).append("\",\n");
        json.append("      \"xp_arcsec\": ").append(fmt(f.xpArcsec())).append(",\n");
        json.append("      \"yp_arcsec\": ").append(fmt(f.ypArcsec())).append(",\n");
        json.append("      \"dut1_s\": ").append(fmt(f.dut1())).append(",\n");
        json.append("      \"dat_s\": ").append(fmt(datS)).append(",\n");
        json.append("      \"ut1_minus_tai_s\": ").append(fmt(ut1MinusTai)).append(",\n");
        json.append("      \"gmst82_rad\": ").append(fmt(gmst82)).append(",\n");
        json.append("      \"diff_2010_1996_m_aiaa\": ").append(fmt(diff)).append(",\n");
        json.append("      \"conventions\": {\n");
        for (int i = 0; i < results.length; i++) {
            final ConvResult r = results[i];
            json.append("        \"").append(r.convention()).append("\": {\n");
            json.append("          \"matrix_teme_to_itrf_rowmajor\": ").append(arr(r.matrix())).append(",\n");
            json.append("          \"r_itrf_aiaa_m\": ").append(vec(r.rAiaa())).append(",\n");
            json.append("          \"r_itrf_iss_m\": ").append(vec(r.rIss())).append(",\n");
            json.append("          \"v_itrf_aiaa_m_s\": ").append(vec(r.vAiaa())).append("\n");
            json.append("        }").append(i == results.length - 1 ? "\n" : ",\n");
        }
        json.append("      }\n");
        json.append("    }").append(last ? "\n" : ",\n");
    }
}
