import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Locale;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.hipparchus.util.FastMath;

import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.NewtonianAttraction;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.frames.EOPHistory;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.PoleCorrection;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.OrbitType;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.BoundedPropagator;
import org.orekit.propagation.EphemerisGenerator;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;
import org.orekit.bodies.CelestialBody;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.forces.ForceModel;
import org.orekit.forces.gravity.SolidTides;
import org.orekit.forces.gravity.Relativity;
import org.orekit.forces.gravity.LenseThirringRelativity;
import org.orekit.forces.gravity.DeSitterRelativity;
import org.orekit.propagation.analytical.BrouwerLyddanePropagator;
import org.orekit.forces.gravity.potential.TideSystem;
import org.orekit.frames.StaticTransform;
import org.orekit.time.UT1Scale;
import org.orekit.forces.gravity.potential.UnnormalizedSphericalHarmonicsProvider;
import org.orekit.forces.gravity.OceanTides;
import org.orekit.forces.gravity.potential.OceanTidesWave;
import org.orekit.data.BodiesElements;
import org.orekit.data.FundamentalNutationArguments;
import org.orekit.time.TimeScales;
import java.util.ArrayList;
import org.orekit.orbits.EquinoctialOrbit;
import org.orekit.propagation.PropagationType;
import org.orekit.propagation.semianalytical.dsst.DSSTPropagator;
import org.orekit.propagation.semianalytical.dsst.forces.DSSTZonal;
import org.orekit.propagation.semianalytical.dsst.forces.DSSTThirdBody;
import org.orekit.propagation.semianalytical.dsst.forces.DSSTSolarRadiationPressure;
import org.orekit.propagation.semianalytical.dsst.forces.DSSTTesseral;
import org.orekit.propagation.semianalytical.dsst.forces.DSSTAtmosphericDrag;
import org.orekit.models.earth.atmosphere.Atmosphere;
import org.orekit.models.earth.atmosphere.HarrisPriester;
import org.orekit.utils.ExtendedPositionProvider;
import org.hipparchus.CalculusFieldElement;
import org.hipparchus.geometry.euclidean.threed.FieldVector3D;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.frames.Transform;
import org.hipparchus.geometry.euclidean.threed.RotationConvention;
import org.orekit.utils.Constants;
import java.util.List;
import org.orekit.propagation.semianalytical.dsst.utilities.AuxiliaryElements;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.attitudes.FrameAlignedProvider;
import org.orekit.attitudes.Attitude;
import org.orekit.attitudes.AttitudeProvider;
import org.orekit.forces.BoxAndSolarArraySpacecraft;
import org.orekit.forces.Panel;
import org.orekit.forces.radiation.KnockeRediffusedForceModel;
import org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient;
import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.ode.nonstiff.AdaptiveStepsizeIntegrator;
import org.hipparchus.ode.nonstiff.ClassicalRungeKuttaIntegrator;
import org.orekit.forces.maneuvers.ConstantThrustManeuver;
import org.orekit.forces.maneuvers.ImpulseManeuver;
import org.orekit.forces.empirical.ParametricAcceleration;
import org.orekit.forces.empirical.PolynomialAccelerationModel;
import org.orekit.attitudes.LofOffset;
import org.orekit.frames.LOFType;
import org.orekit.orbits.CircularOrbit;
import org.orekit.propagation.events.DateDetector;

/**
 * A propagation benchmark against Orekit's numerical propagator. The same Cartesian initial state is
 * flown for a day under (a) point-mass gravity and (b) point-mass + J2, with a DP853 integrator, and
 * the inertial position sampled every 60 s. Orekit's own degree-2 field constants (GM, Rₑ, J₂) are
 * printed so the {@code spacelab} side flies the identical force model — the comparison is then the
 * integrator and force formulation, not the constants. Emits one CSV per case plus a header line with
 * the constants, the initial state and Orekit's own wall-clock; {@code experiments/orekit_benchmark.py}
 * consumes it.
 */
public final class OrekitPropagationBench {

    private static final double SPAN_SECONDS = 86_400.0;
    private static final double STEP_SECONDS = 60.0;
    private static final int TIMING_REPEATS = 20;

    public static void main(String[] args) throws IOException {
        final File dataDir = new File(System.getProperty("orekit.data", "/tmp/orekit-bench/orekit-data-main"));
        if (!dataDir.isDirectory()) {
            throw new IllegalStateException("orekit-data not found at " + dataDir);
        }
        DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(dataDir));
        final File outDir = new File(System.getProperty("out.dir", "/tmp/orekit-bench/out"));
        outDir.mkdirs();

        final TimeScale utc = TimeScalesFactory.getUTC();
        final Frame gcrf = FramesFactory.getGCRF();
        final Frame itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
        final AbsoluteDate epoch = new AbsoluteDate(System.getProperty("epoch", "2024-01-01T00:00:00.000"), utc);

        // The Earth-orientation parameters Orekit used to build its ITRF, emitted so the spacelab side can
        // fly the identical frame (matched pole and UT1) — otherwise the comparison confounds EOP with the
        // precession-nutation model itself.
        final EOPHistory eopHistory = FramesFactory.getEOPHistory(IERSConventions.IERS_2010, true);
        final PoleCorrection pole = eopHistory.getPoleCorrection(epoch);
        final double xpRad = pole.getXp();
        final double ypRad = pole.getYp();
        final double dut1 = eopHistory.getUT1MinusUTC(epoch);
        final double lod = eopHistory.getLOD(epoch);
        // Celestial-pole offsets (dX, dY) that correct the CIO precession-nutation model, radians.
        final double[] cip = eopHistory.getNonRotatinOriginNutationCorrection(epoch);

        // Orekit's own degree-2 field: use its GM, Rₑ and J₂ everywhere so the force models are identical.
        final NormalizedSphericalHarmonicsProvider provider = GravityFieldFactory.getNormalizedProvider(2, 0);
        final double mu = provider.getMu();
        final double ae = provider.getAe();
        final double c20 = provider.onDate(epoch).getNormalizedCnm(2, 0);
        final double j2 = -c20 * FastMath.sqrt(5.0);

        // Orbit from properties (LEO default), defined Keplerian then taken to Cartesian (what spacelab
        // integrates). MEO/GEO are selected by the caller via -Dsma/-Decc/-Dinc.
        final double sma = Double.parseDouble(System.getProperty("sma", "6928137.0"));
        final double ecc = Double.parseDouble(System.getProperty("ecc", "0.001"));
        final double inc = Double.parseDouble(System.getProperty("inc", "51.6"));
        final Orbit keplerian = new KeplerianOrbit(
                sma, ecc, FastMath.toRadians(inc),
                FastMath.toRadians(40.0), FastMath.toRadians(30.0), FastMath.toRadians(10.0),
                PositionAngleType.MEAN, gcrf, epoch, mu);
        final PVCoordinates pv0 = keplerian.getPVCoordinates();
        final Vector3D r0 = pv0.getPosition();
        final Vector3D v0 = pv0.getVelocity();
        final double posTol = Double.parseDouble(System.getProperty("posTol", "1.0e-9"));

        // Analytic Keplerian propagation — the exact two-body reference the numerical propagators are
        // scored against. The Rust side cross-checks it with spacelab's own universal-variable Kepler.
        writeSamples(new File(outDir, "orekit_kepler.csv"), new KeplerianPropagator(keplerian), gcrf, epoch);

        // Point mass; J2 in the true ITRF (bulge tracks the CIP pole); J2 in GCRF (bulge about the fixed
        // inertial z — the same convention spacelab's inertial J2 uses, for a matched integrator check).
        final long tTwoBody = writeCase(new File(outDir, "orekit_twobody.csv"), pv0, gcrf, epoch, mu, null, posTol);
        final long tJ2 = writeCase(new File(outDir, "orekit_j2.csv"), pv0, gcrf, epoch, mu, itrf, posTol);
        final long tJ2Inertial = writeCase(new File(outDir, "orekit_j2_inertial.csv"), pv0, gcrf, epoch, mu, gcrf, posTol);

        // A machine-readable header for the Python side: constants, IC, tolerance, and Orekit's timing.
        final String meta = String.format(Locale.ROOT,
                "mu,%.17e%nae,%.17e%nj2,%.17e%npos_tol,%.1e%n"
                        + "r0,%.17e,%.17e,%.17e%nv0,%.17e,%.17e,%.17e%n"
                        + "span_s,%.1f%nstep_s,%.1f%n"
                        + "xp_rad,%.17e%nyp_rad,%.17e%ndut1_s,%.17e%nlod_s,%.17e%n"
                        + "dx_rad,%.17e%ndy_rad,%.17e%n"
                        + "orekit_twobody_ms,%.3f%norekit_j2_ms,%.3f%norekit_j2_inertial_ms,%.3f%n",
                mu, ae, j2, posTol,
                r0.getX(), r0.getY(), r0.getZ(), v0.getX(), v0.getY(), v0.getZ(),
                SPAN_SECONDS, STEP_SECONDS,
                xpRad, ypRad, dut1, lod,
                cip[0], cip[1],
                tTwoBody / 1.0e6 / TIMING_REPEATS, tJ2 / 1.0e6 / TIMING_REPEATS, tJ2Inertial / 1.0e6 / TIMING_REPEATS);
        Files.writeString(new File(outDir, "orekit_meta.csv").toPath(), meta);

        // The EOP time series Orekit used across the span, so the spacelab side can fly the identical
        // time-varying frame — feeding constant epoch EOP is the remaining ~mm difference over a day.
        final StringBuilder eopCsv = new StringBuilder("t_s,xp_rad,yp_rad,dut1_s,dx_rad,dy_rad\n");
        for (int k = 0; k * STEP_SECONDS <= SPAN_SECONDS; k++) {
            final double t = k * STEP_SECONDS;
            final AbsoluteDate d = epoch.shiftedBy(t);
            final PoleCorrection p = eopHistory.getPoleCorrection(d);
            final double[] cipT = eopHistory.getNonRotatinOriginNutationCorrection(d);
            eopCsv.append(String.format(Locale.ROOT, "%.1f,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                    t, p.getXp(), p.getYp(), eopHistory.getUT1MinusUTC(d), cipT[0], cipT[1]));
        }
        Files.writeString(new File(outDir, "orekit_eop.csv").toPath(), eopCsv.toString());

        writeSolidTideValidation(new File(outDir, "orekit_solid_tide.csv"), itrf, gcrf, epoch, mu, ae);
        writeTideStep2Validation(new File(outDir, "orekit_tide_step2.csv"), epoch);
        writeOceanTideValidation(new File(outDir, "orekit_ocean_tide.csv"), itrf, gcrf, epoch, mu, ae);
        writeRelativityValidation(new File(outDir, "orekit_relativity.csv"), gcrf, epoch, mu);
        writeBrouwerLyddaneValidation(new File(outDir, "orekit_brouwer_lyddane.csv"), gcrf, epoch, mu);
        writeDsstZonalValidation(new File(outDir, "orekit_dsst_zonal.csv"), gcrf, epoch);
        writeDsstZonalShortPeriodValidation(new File(outDir, "orekit_dsst_zonal_sp.csv"), gcrf, epoch);
        writeDsstThirdBodyValidation(new File(outDir, "orekit_dsst_thirdbody.csv"), gcrf, epoch);
        writeDsstThirdBodyShortPeriodValidation(new File(outDir, "orekit_dsst_thirdbody_sp.csv"), gcrf, epoch);
        writeDsstSrpValidation(new File(outDir, "orekit_dsst_srp.csv"), gcrf, itrf, epoch);
        writeDsstSrpShortPeriodValidation(new File(outDir, "orekit_dsst_srp_sp.csv"), gcrf, itrf, epoch);
        writeDsstTesseralValidation(new File(outDir, "orekit_dsst_tesseral.csv"), gcrf, epoch);
        writeDsstTesseralShortPeriodValidation(new File(outDir, "orekit_dsst_tesseral_sp.csv"), gcrf, epoch);
        writeDsstDragValidation(new File(outDir, "orekit_dsst_drag.csv"), gcrf, epoch);
        writeDsstDragShortPeriodValidation(new File(outDir, "orekit_dsst_drag_sp.csv"), gcrf, epoch);
        writeBoxWingValidation(new File(outDir, "orekit_box_wing.csv"), gcrf, epoch, mu);
        writeEarthRadiationValidation(new File(outDir, "orekit_earth_radiation.csv"), gcrf, epoch, mu, ae);
        writeHarrisPriesterValidation(new File(outDir, "orekit_harris_priester.csv"));
        writeManeuverValidation(new File(outDir, "orekit_maneuver.csv"), gcrf, epoch, mu);

        System.out.print(meta);
    }

    /** Sample any propagator's inertial position to CSV every STEP_SECONDS over the span. */
    private static void writeSamples(File out, Propagator prop, Frame gcrf, AbsoluteDate epoch) throws IOException {
        final StringBuilder csv = new StringBuilder("t_s,x_m,y_m,z_m\n");
        for (int k = 0; k * STEP_SECONDS <= SPAN_SECONDS; k++) {
            final double t = k * STEP_SECONDS;
            final Vector3D p = prop.propagate(epoch.shiftedBy(t)).getPVCoordinates(gcrf).getPosition();
            csv.append(String.format(Locale.ROOT, "%.1f,%.9f,%.9f,%.9f%n", t, p.getX(), p.getY(), p.getZ()));
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /** Propagate one numerical case, sample to CSV, and return the total nanoseconds of TIMING_REPEATS runs. */
    private static long writeCase(File out, PVCoordinates pv0, Frame gcrf, AbsoluteDate epoch,
                                  double mu, Frame fieldFrame, double posTol) throws IOException {
        final Orbit orbit = new CartesianOrbit(pv0, gcrf, epoch, mu);
        writeSamples(out, propagate(orbit, mu, fieldFrame, posTol), gcrf, epoch);

        long total = 0;
        for (int i = 0; i < TIMING_REPEATS; i++) {
            final long t0 = System.nanoTime();
            propagate(orbit, mu, fieldFrame, posTol).propagate(epoch.shiftedBy(SPAN_SECONDS));
            total += System.nanoTime() - t0;
        }
        return total;
    }

    /** `fieldFrame == null` → point-mass gravity; otherwise the degree-2 field applied in that frame. */
    private static BoundedPropagator propagate(Orbit orbit, double mu, Frame fieldFrame, double posTol) {
        final double[][] tol = NumericalPropagator.tolerances(posTol, orbit, OrbitType.CARTESIAN);
        final DormandPrince853Integrator integrator =
                new DormandPrince853Integrator(1.0e-6, 300.0, tol[0], tol[1]);
        final NumericalPropagator propagator = new NumericalPropagator(integrator);
        propagator.setOrbitType(OrbitType.CARTESIAN);
        propagator.setInitialState(new SpacecraftState(orbit));
        if (fieldFrame != null) {
            propagator.addForceModel(new HolmesFeatherstoneAttractionModel(
                    fieldFrame, GravityFieldFactory.getNormalizedProvider(2, 0)));
        } else {
            propagator.addForceModel(new NewtonianAttraction(mu));
        }
        final EphemerisGenerator generator = propagator.getEphemerisGenerator();
        propagator.propagate(orbit.getDate().shiftedBy(SPAN_SECONDS));
        return generator.getGeneratedEphemeris();
    }

    /**
     * Emit the IERS 2010 solid Earth tide acceleration at a fixed geometry, for validating spacelab.
     * The tidal-only acceleration is the underlying model of {@link SolidTides} (the ΔC/ΔS harmonics,
     * no monopole). Everything is written in the ITRF so the spacelab side needs no frame convention.
     */
    private static void writeSolidTideValidation(File out, Frame itrf, Frame gcrf, AbsoluteDate epoch,
                                                 double mu, double ae) throws IOException {
        final UT1Scale ut1 = TimeScalesFactory.getUT1(IERSConventions.IERS_2010, true);
        final CelestialBody sun = CelestialBodyFactory.getSun();
        final CelestialBody moon = CelestialBodyFactory.getMoon();
        final SolidTides tides = new SolidTides(itrf, ae, mu, TideSystem.ZERO_TIDE,
                IERSConventions.IERS_2010, ut1, sun, moon);
        final ForceModel underlying = tides.getUnderlyingModel();

        final Vector3D rGcrf = new Vector3D(6.5e6, 2.0e6, 3.0e6);
        final Vector3D vGcrf = new Vector3D(0.0, 7000.0, 1000.0);
        final SpacecraftState state = new SpacecraftState(
                new CartesianOrbit(new PVCoordinates(rGcrf, vGcrf), gcrf, epoch, mu));
        final Vector3D aGcrf = underlying.acceleration(state, underlying.getParameters(epoch));

        // Same tide WITHOUT the solid Earth pole tide (§6.4), no interpolation (nbPoints < 2), to
        // isolate the pole-tide ΔC21/ΔS21 contribution that spacelab does not (yet) model.
        final SolidTides tidesNoPole = new SolidTides(itrf, ae, mu, TideSystem.ZERO_TIDE, false,
                60.0, 1, IERSConventions.IERS_2010, ut1, sun, moon);
        final ForceModel underlyingNoPole = tidesNoPole.getUnderlyingModel();
        final Vector3D aNoPoleGcrf =
                underlyingNoPole.acceleration(state, underlyingNoPole.getParameters(epoch));

        final StaticTransform g2i = gcrf.getStaticTransformTo(itrf, epoch);
        final Vector3D rItrf = g2i.transformPosition(rGcrf);
        final Vector3D sunItrf = g2i.transformPosition(sun.getPosition(epoch, gcrf));
        final Vector3D moonItrf = g2i.transformPosition(moon.getPosition(epoch, gcrf));
        final Vector3D aItrf = g2i.transformVector(aGcrf);
        final Vector3D aNoPoleItrf = g2i.transformVector(aNoPoleGcrf);

        // Solid Earth pole tide (§6.4) inputs/outputs for the spacelab pole_tide_corrections gate:
        // the polar motion (xp, yp, radians), the seconds since J2000 the mean-pole polynomial uses,
        // and Orekit's [ΔC21, ΔS21].
        final double poleXp = ut1.getEOPHistory().getPoleCorrection(epoch).getXp();
        final double poleYp = ut1.getEOPHistory().getPoleCorrection(epoch).getYp();
        final double poleT = epoch.durationFrom(AbsoluteDate.J2000_EPOCH);
        final double[] poleCS =
                IERSConventions.IERS_2010.getSolidPoleTide(ut1.getEOPHistory()).value(epoch);
        final double[] oceanPoleCS =
                IERSConventions.IERS_2010.getOceanPoleTide(ut1.getEOPHistory()).value(epoch);

        final String csv = String.format(Locale.ROOT,
                "mu_earth,%.17e%nae,%.17e%nmu_sun,%.17e%nmu_moon,%.17e%n"
                        + "sat_itrf,%.9f,%.9f,%.9f%nsun_itrf,%.9f,%.9f,%.9f%nmoon_itrf,%.9f,%.9f,%.9f%n"
                        + "a_tide_itrf,%.17e,%.17e,%.17e%n"
                        + "a_tide_no_pole_itrf,%.17e,%.17e,%.17e%n"
                        + "pole_xp_yp_t,%.17e,%.17e,%.17e%npole_tide_cs,%.17e,%.17e%n"
                        + "ocean_pole_tide_cs,%.17e,%.17e%n",
                mu, ae, sun.getGM(), moon.getGM(),
                rItrf.getX(), rItrf.getY(), rItrf.getZ(),
                sunItrf.getX(), sunItrf.getY(), sunItrf.getZ(),
                moonItrf.getX(), moonItrf.getY(), moonItrf.getZ(),
                aItrf.getX(), aItrf.getY(), aItrf.getZ(),
                aNoPoleItrf.getX(), aNoPoleItrf.getY(), aNoPoleItrf.getZ(),
                poleXp, poleYp, poleT, poleCS[0], poleCS[1],
                oceanPoleCS[0], oceanPoleCS[1]);
        Files.writeString(out.toPath(), csv);
    }

    /**
     * Emit the IERS 2010 ocean-tide (IERS Conventions 2010, §6.3, Eq. 6.15) validation data.
     *
     * <p>Two things are emitted so the spacelab port can be gated at two levels:
     * <ol>
     *   <li>the per-coefficient ΔC̄nm/ΔS̄nm (degree 2..6) -- assembled by calling Orekit's own
     *       {@link OceanTidesWave#addContribution} over the same waves the field uses -- the
     *       diagnostic gate, a pure deterministic sum expected near-bit-exact;</li>
     *   <li>the ITRF tidal acceleration at a fixed GCRF state.</li>
     * </ol>
     *
     * <p>The {@link OceanTides} is built <em>without</em> the ocean pole tide (out of scope, IERS
     * §6.4) and with {@code nbPoints < 2} (no time interpolation -- the raw per-epoch field). Both
     * keep the acceleration a clean gate for spacelab's wave-only, exact-epoch port: the default
     * {@code OceanTides(...)} constructor would instead add pole tide (ΔC21/ΔS21) and a 600 s cached
     * interpolation, neither of which the port models, confounding the comparison.
     *
     * <p>The sat/Sun-independent inputs (GMST and the five Delaunay arguments Orekit used) are emitted
     * so the Rust test drives the identical θ.
     */
    private static void writeOceanTideValidation(File out, Frame itrf, Frame gcrf, AbsoluteDate epoch,
                                                 double mu, double ae) throws IOException {
        final UT1Scale ut1 = TimeScalesFactory.getUT1(IERSConventions.IERS_2010, true);
        final OceanTides tides = new OceanTides(itrf, ae, mu, false,
                OceanTides.DEFAULT_STEP, 1, 6, 6, IERSConventions.IERS_2010, ut1);
        final ForceModel underlying = tides.getUnderlyingModel();

        final Vector3D rGcrf = new Vector3D(6.5e6, 2.0e6, 3.0e6);
        final Vector3D vGcrf = new Vector3D(0.0, 7000.0, 1000.0);
        final SpacecraftState state = new SpacecraftState(
                new CartesianOrbit(new PVCoordinates(rGcrf, vGcrf), gcrf, epoch, mu));
        final Vector3D aGcrf = underlying.acceleration(state, underlying.getParameters(epoch));

        final StaticTransform g2i = gcrf.getStaticTransformTo(itrf, epoch);
        final Vector3D rItrf = g2i.transformPosition(rGcrf);
        final Vector3D aItrf = g2i.transformVector(aGcrf);

        // Fundamental arguments Orekit uses to drive the waves (same call path as OceanTidesField).
        final TimeScales timeScales = ut1.getEOPHistory().getTimeScales();
        final FundamentalNutationArguments args =
                IERSConventions.IERS_2010.getNutationArguments(ut1, timeScales);
        final BodiesElements be = args.evaluateAll(epoch);
        final double gmst = be.getGamma() - FastMath.PI; // γ = GMST + π

        // ΔC̄nm/ΔS̄nm from Orekit's own addContribution, waves sorted by Doodson to match spacelab's
        // accumulation order (removes floating-point summation order as a source of disagreement).
        final List<OceanTidesWave> waves = new ArrayList<>(GravityFieldFactory.getOceanTidesWaves(6, 6));
        waves.sort((a, b) -> Integer.compare(a.getDoodson(), b.getDoodson()));
        final double[][] cnm = new double[7][];
        final double[][] snm = new double[7][];
        for (int i = 0; i <= 6; i++) {
            cnm[i] = new double[i + 1];
            snm[i] = new double[i + 1];
        }
        for (final OceanTidesWave w : waves) {
            w.addContribution(be, cnm, snm);
        }

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu_earth,%.17e%nae,%.17e%nsat_itrf,%.9f,%.9f,%.9f%na_tide_itrf,%.17e,%.17e,%.17e%n",
                mu, ae, rItrf.getX(), rItrf.getY(), rItrf.getZ(),
                aItrf.getX(), aItrf.getY(), aItrf.getZ()));
        csv.append(String.format(Locale.ROOT, "gmst,%.17e%n", gmst));
        csv.append(String.format(Locale.ROOT, "delaunay,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                be.getL(), be.getLPrime(), be.getF(), be.getD(), be.getOmega()));
        for (int n = 2; n <= 6; n++) {
            for (int m = 0; m <= n; m++) {
                csv.append(String.format(Locale.ROOT, "dCS,%d,%d,%.17e,%.17e%n",
                        n, m, cnm[n][m], snm[n][m]));
            }
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Emit a Brouwer-Lyddane analytical propagation for validating spacelab's port. From a fixed
     * initial osculating orbit and the un-normalized zonal coefficients C20..C50 (= −J2..−J5), the
     * propagator is stepped every 3 h to 24 h; the initial state (r0, v0), the coefficients and the
     * osculating (r, v) at each time are written in the GCRF, so the spacelab side reproduces the full
     * osculating→mean→propagate→osculating pipeline from identical inputs.
     */
    /**
     * Emit the IERS 2010 Step-2 (frequency-dependent) solid-tide corrections to the geopotential
     * coefficients [ΔC20, ΔC21, ΔS21, ΔC22, ΔS22] at the epoch, for validating spacelab's Step-2 port.
     */
    private static void writeTideStep2Validation(File out, AbsoluteDate epoch) throws IOException {
        final UT1Scale ut1 = TimeScalesFactory.getUT1(IERSConventions.IERS_2010, true);
        final org.orekit.time.TimeVectorFunction dcs =
                IERSConventions.IERS_2010.getTideFrequencyDependenceFunction(ut1);
        final double[] v = dcs.value(epoch);
        final String csv = String.format(Locale.ROOT,
                "dC20,%.17e%ndC21,%.17e%ndS21,%.17e%ndC22,%.17e%ndS22,%.17e%n",
                v[0], v[1], v[2], v[3], v[4]);
        Files.writeString(out.toPath(), csv);
    }

    private static void writeBrouwerLyddaneValidation(File out, Frame gcrf, AbsoluteDate epoch,
                                                      double mu) throws IOException {
        final double reBL = 6_378_136.3;
        final double c20 = -1.082_626_68e-3;
        final double c30 = 2.532_656_49e-6;
        final double c40 = 1.619_621_59e-6;
        final double c50 = 2.272_960_83e-7;
        final KeplerianOrbit k0 = new KeplerianOrbit(
                7.2e6, 0.01, FastMath.toRadians(98.0),
                FastMath.toRadians(30.0), FastMath.toRadians(42.0), FastMath.toRadians(10.0),
                PositionAngleType.TRUE, gcrf, epoch, mu);
        final BrouwerLyddanePropagator bl =
                new BrouwerLyddanePropagator(k0, 1000.0, reBL, mu, c20, c30, c40, c50, 0.0);
        final Vector3D r0 = k0.getPVCoordinates(gcrf).getPosition();
        final Vector3D v0 = k0.getPVCoordinates(gcrf).getVelocity();

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nre,%.17e%nc20,%.17e%nc30,%.17e%nc40,%.17e%nc50,%.17e%n"
                        + "r0,%.9f,%.9f,%.9f%nv0,%.12f,%.12f,%.12f%nt_s,x,y,z,vx,vy,vz%n",
                mu, reBL, c20, c30, c40, c50,
                r0.getX(), r0.getY(), r0.getZ(), v0.getX(), v0.getY(), v0.getZ()));
        for (int k = 0; k <= 8; k++) {
            final double t = k * 10800.0;
            final org.orekit.utils.PVCoordinates pv =
                    bl.propagate(epoch.shiftedBy(t)).getPVCoordinates(gcrf);
            csv.append(String.format(Locale.ROOT, "%.1f,%.9f,%.9f,%.9f,%.12f,%.12f,%.12f%n",
                    t, pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ(),
                    pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ()));
        }
        Files.writeString(out.toPath(), csv.toString());

        // Same orbit with a non-zero empirical along-track term M2, to validate the spacelab M2 path.
        final double m2 = 1.0e-14;
        final BrouwerLyddanePropagator blM2 =
                new BrouwerLyddanePropagator(k0, 1000.0, reBL, mu, c20, c30, c40, c50, m2);
        final StringBuilder csv2 = new StringBuilder();
        csv2.append(String.format(Locale.ROOT, "m2,%.17e%nt_s,x,y,z,vx,vy,vz%n", m2));
        for (int k = 0; k <= 8; k++) {
            final double t = k * 10800.0;
            final org.orekit.utils.PVCoordinates pv =
                    blM2.propagate(epoch.shiftedBy(t)).getPVCoordinates(gcrf);
            csv2.append(String.format(Locale.ROOT, "%.1f,%.9f,%.9f,%.9f,%.12f,%.12f,%.12f%n",
                    t, pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ(),
                    pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ()));
        }
        Files.writeString(new File(out.getParentFile(), "orekit_brouwer_lyddane_m2.csv").toPath(),
                csv2.toString());
    }

    /**
     * Emit a DSST (Draper Semi-analytical Satellite Theory) central-body zonal validation for the
     * spacelab port. The zonal field is taken to degree 6 (J2..J6) with Orekit's own unnormalized
     * coefficients; from a fixed MEAN equinoctial state we emit (a) the six mean-element rates
     * getMeanElementRate = [da, dk, dh, dq, dp, dM] at that state, and (b) a MEAN semi-analytical
     * propagation sampling the mean equinoctial (a, ex, ey, hx, hy, lM) every 6 h to 24 h. The
     * spacelab side flies the identical field constants, initial mean state and rates.
     *
     * <p>The constructor {@code new DSSTZonal(provider)} uses a null body-fixed frame, so the pole
     * direction cosines reduce to the z-components of the equinoctial f/g/w vectors — pure functions
     * of (p, q) — which is what the port reproduces. The truncation order maxEccPowMeanElements is
     * read reflectively and emitted so the port's identical eccentricity-power truncation is checked.
     */
    private static void writeDsstZonalValidation(File out, Frame gcrf, AbsoluteDate epoch)
            throws IOException {

        final UnnormalizedSphericalHarmonicsProvider provider = GravityFieldFactory.getUnnormalizedProvider(6, 0);
        final double muZ = provider.getMu();
        final double ae = provider.getAe();
        final UnnormalizedSphericalHarmonicsProvider.UnnormalizedSphericalHarmonics harm = provider.onDate(epoch);
        final double[] cn0 = new double[7];
        for (int n = 2; n <= 6; n++) {
            cn0[n] = harm.getUnnormalizedCnm(n, 0);
        }

        // MEAN initial orbit: a typical sun-synchronous-ish LEO (a = 7.2e6 m, e = 0.01, i = 98 deg).
        final KeplerianOrbit kep = new KeplerianOrbit(
                7.2e6, 0.01, FastMath.toRadians(98.0),
                FastMath.toRadians(30.0), FastMath.toRadians(42.0), FastMath.toRadians(10.0),
                PositionAngleType.MEAN, gcrf, epoch, muZ);
        final EquinoctialOrbit eq = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(kep);

        // (a) Fixed-state mean-element rates. initializeShortPeriodTerms sets the truncation and the
        // Hansen objects that getMeanElementRate consumes; AuxiliaryElements is built exactly as
        // DSSTPropagator.computeDerivatives builds it (retrograde factor I = 1).
        final DSSTZonal zonal = new DSSTZonal(provider);
        final AuxiliaryElements aux = new AuxiliaryElements(eq, 1);
        final double[] params = zonal.getParameters(epoch);
        zonal.initializeShortPeriodTerms(aux, PropagationType.MEAN, params);
        final SpacecraftState state = new SpacecraftState(eq);
        final double[] rates = zonal.getMeanElementRate(state, aux, params);

        int maxEccPow = -1;
        try {
            final java.lang.reflect.Field f = DSSTZonal.class.getDeclaredField("maxEccPowMeanElements");
            f.setAccessible(true);
            maxEccPow = f.getInt(zonal);
        } catch (ReflectiveOperationException e) {
            maxEccPow = -1;
        }

        // (b) MEAN semi-analytical propagation. A fresh DSSTZonal avoids reusing the mutated one; the
        // propagator auto-adds a DSSTNewtonianAttraction, so dy/dt = zonal rates + [0,0,0,0,0,n].
        final DSSTZonal zonalProp = new DSSTZonal(provider);
        final double[][] tol = DSSTPropagator.tolerances(1.0e-6, eq);
        final AdaptiveStepsizeIntegrator integ =
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]);
        final DSSTPropagator dsst = new DSSTPropagator(integ, PropagationType.MEAN);
        dsst.setInitialState(new SpacecraftState(eq), PropagationType.MEAN);
        dsst.addForceModel(zonalProp);

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nae,%.17e%nc20,%.17e%nc30,%.17e%nc40,%.17e%nc50,%.17e%nc60,%.17e%n"
                        + "max_ecc_pow,%d%n"
                        + "a0,%.17e%nex0,%.17e%ney0,%.17e%nhx0,%.17e%nhy0,%.17e%nlm0,%.17e%n"
                        + "rate_da,%.17e%nrate_dk,%.17e%nrate_dh,%.17e%nrate_dq,%.17e%nrate_dp,%.17e%nrate_dm,%.17e%n"
                        + "t_s,a,ex,ey,hx,hy,lm%n",
                muZ, ae, cn0[2], cn0[3], cn0[4], cn0[5], cn0[6],
                maxEccPow,
                eq.getA(), eq.getEquinoctialEx(), eq.getEquinoctialEy(), eq.getHx(), eq.getHy(), eq.getLM(),
                rates[0], rates[1], rates[2], rates[3], rates[4], rates[5]));
        for (int k = 0; k <= 4; k++) {
            final double t = k * 21600.0;
            final EquinoctialOrbit o = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(
                    dsst.propagate(epoch.shiftedBy(t)).getOrbit());
            csv.append(String.format(Locale.ROOT, "%.1f,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                    t, o.getA(), o.getEquinoctialEx(), o.getEquinoctialEy(),
                    o.getHx(), o.getHy(), o.getLM()));
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Read the fixed truncation orders (maxAR3Pow, maxEccPow) Orekit's DSSTThirdBodyStaticContext set
     * during initializeShortPeriodTerms, via reflection, so the spacelab port's identical truncation
     * search is checked.
     */
    private static int[] thirdBodyMaxPowers(DSSTThirdBody tb) {
        final int[] out = {-1, -1};
        try {
            final java.lang.reflect.Field scF = DSSTThirdBody.class.getDeclaredField("staticContext");
            scF.setAccessible(true);
            final Object sc = scF.get(tb);
            final java.lang.reflect.Field arF = sc.getClass().getDeclaredField("maxAR3Pow");
            arF.setAccessible(true);
            out[0] = arF.getInt(sc);
            final java.lang.reflect.Field epF = sc.getClass().getDeclaredField("maxEccPow");
            epF.setAccessible(true);
            out[1] = epF.getInt(sc);
        } catch (ReflectiveOperationException e) {
            // leave as -1
        }
        return out;
    }

    /**
     * Emit a DSST third-body (Sun + Moon) mean-element-rate validation for the spacelab port. From the
     * same fixed MEAN equinoctial LEO state as the zonal case, we emit (a) the six mean-element rates
     * getMeanElementRate = [da, dk, dh, dq, dp, dM] for the Sun and for the Moon, (b) the Sun and Moon
     * geocentric positions in the propagation frame (GCRF) at the epoch — so the spacelab side uses
     * identical body geometry — plus each body's GM and the fixed truncation orders (maxAR3Pow,
     * maxEccPow), and (c) a MEAN DSSTPropagator run with BOTH bodies added, sampling the mean
     * equinoctial (a, ex, ey, hx, hy, lM) every 6 h to 24 h together with the body positions at each
     * sample time.
     *
     * <p>The force context is set up exactly as {@code DSSTPropagator.computeDerivatives} does:
     * {@code new AuxiliaryElements(orbit, 1)}, then {@code initializeShortPeriodTerms(aux, MEAN,
     * params)} (which fixes the truncation and the Hansen objects), then
     * {@code getMeanElementRate(state, aux, params)}. The third body's direction cosines α/β/γ are the
     * normalized body direction dotted with the equinoctial f/g/w vectors — which is what the port
     * reproduces from the emitted body positions.
     */
    private static void writeDsstThirdBodyValidation(File out, Frame gcrf, AbsoluteDate epoch)
            throws IOException {

        final double mu = GravityFieldFactory.getNormalizedProvider(2, 0).getMu();

        // MEAN initial orbit: same sun-synchronous-ish LEO as the zonal case.
        final KeplerianOrbit kep = new KeplerianOrbit(
                7.2e6, 0.01, FastMath.toRadians(98.0),
                FastMath.toRadians(30.0), FastMath.toRadians(42.0), FastMath.toRadians(10.0),
                PositionAngleType.MEAN, gcrf, epoch, mu);
        final EquinoctialOrbit eq = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(kep);

        final CelestialBody sunBody = CelestialBodyFactory.getSun();
        final CelestialBody moonBody = CelestialBodyFactory.getMoon();

        // (a) Fixed-state mean-element rates for the Sun and the Moon.
        final DSSTThirdBody sun = new DSSTThirdBody(sunBody, mu);
        final DSSTThirdBody moon = new DSSTThirdBody(moonBody, mu);
        final AuxiliaryElements aux = new AuxiliaryElements(eq, 1);
        final SpacecraftState state = new SpacecraftState(eq);

        final double[] sunParams = sun.getParameters(epoch);
        sun.initializeShortPeriodTerms(aux, PropagationType.MEAN, sunParams);
        final double[] sunRates = sun.getMeanElementRate(state, aux, sunParams);
        final int[] sunPow = thirdBodyMaxPowers(sun);

        final double[] moonParams = moon.getParameters(epoch);
        moon.initializeShortPeriodTerms(aux, PropagationType.MEAN, moonParams);
        final double[] moonRates = moon.getMeanElementRate(state, aux, moonParams);
        final int[] moonPow = thirdBodyMaxPowers(moon);

        // (b) Body positions in the propagation frame at the epoch, and each body's GM
        // (parameters[0] is the 3rd body attraction coefficient).
        final Vector3D sunPos = sunBody.getPosition(epoch, gcrf);
        final Vector3D moonPos = moonBody.getPosition(epoch, gcrf);
        final double sunGm = sunParams[0];
        final double moonGm = moonParams[0];

        // (c) MEAN semi-analytical propagation with BOTH third bodies. Fresh instances avoid reusing
        // the mutated ones; the propagator auto-adds a DSSTNewtonianAttraction, so
        // dy/dt = sun rates + moon rates + [0,0,0,0,0,n].
        final DSSTThirdBody sunProp = new DSSTThirdBody(sunBody, mu);
        final DSSTThirdBody moonProp = new DSSTThirdBody(moonBody, mu);
        final double[][] tol = DSSTPropagator.tolerances(1.0e-6, eq);
        final AdaptiveStepsizeIntegrator integ =
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]);
        final DSSTPropagator dsst = new DSSTPropagator(integ, PropagationType.MEAN);
        dsst.setInitialState(new SpacecraftState(eq), PropagationType.MEAN);
        dsst.addForceModel(sunProp);
        dsst.addForceModel(moonProp);

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nsun_gm,%.17e%nmoon_gm,%.17e%n"
                        + "sun_max_ar3,%d%nsun_max_ecc,%d%nmoon_max_ar3,%d%nmoon_max_ecc,%d%n"
                        + "a0,%.17e%nex0,%.17e%ney0,%.17e%nhx0,%.17e%nhy0,%.17e%nlm0,%.17e%n"
                        + "sun_pos,%.17e,%.17e,%.17e%nmoon_pos,%.17e,%.17e,%.17e%n"
                        + "sun_rate_da,%.17e%nsun_rate_dk,%.17e%nsun_rate_dh,%.17e%n"
                        + "sun_rate_dq,%.17e%nsun_rate_dp,%.17e%nsun_rate_dm,%.17e%n"
                        + "moon_rate_da,%.17e%nmoon_rate_dk,%.17e%nmoon_rate_dh,%.17e%n"
                        + "moon_rate_dq,%.17e%nmoon_rate_dp,%.17e%nmoon_rate_dm,%.17e%n"
                        + "t_s,a,ex,ey,hx,hy,lm,sunx,suny,sunz,moonx,moony,moonz%n",
                mu, sunGm, moonGm,
                sunPow[0], sunPow[1], moonPow[0], moonPow[1],
                eq.getA(), eq.getEquinoctialEx(), eq.getEquinoctialEy(), eq.getHx(), eq.getHy(), eq.getLM(),
                sunPos.getX(), sunPos.getY(), sunPos.getZ(),
                moonPos.getX(), moonPos.getY(), moonPos.getZ(),
                sunRates[0], sunRates[1], sunRates[2], sunRates[3], sunRates[4], sunRates[5],
                moonRates[0], moonRates[1], moonRates[2], moonRates[3], moonRates[4], moonRates[5]));
        for (int k = 0; k <= 4; k++) {
            final double t = k * 21600.0;
            final AbsoluteDate d = epoch.shiftedBy(t);
            final EquinoctialOrbit o = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(
                    dsst.propagate(d).getOrbit());
            final Vector3D sp = sunBody.getPosition(d, gcrf);
            final Vector3D mp = moonBody.getPosition(d, gcrf);
            csv.append(String.format(Locale.ROOT,
                    "%.1f,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                    t, o.getA(), o.getEquinoctialEx(), o.getEquinoctialEy(), o.getHx(), o.getHy(), o.getLM(),
                    sp.getX(), sp.getY(), sp.getZ(), mp.getX(), mp.getY(), mp.getZ()));
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Emit a DSST third-body (Sun / Sun+Moon) SHORT-PERIOD validation for the spacelab port. From the
     * same fixed MEAN equinoctial LEO state as the third-body mean case, we run two pairs of
     * DSSTPropagators — a {@code PropagationType.MEAN} and a {@code PropagationType.OSCULATING} one,
     * each seeded from the MEAN initial state — for (i) the Sun alone and (ii) the Sun plus the Moon.
     * At every t &gt; 0 sample we emit both the mean equinoctial state (from the MEAN run) and the
     * osculating equinoctial state (from the OSCULATING run), plus the Sun (and Moon) geocentric
     * positions in the propagation frame (GCRF) at that date. This lets the Rust side validate ONLY the
     * short-period assembly: osc(t) − mean(t) must equal the port's third-body short-period δ evaluated
     * at mean(t) with the emitted body geometry, isolated from mean-propagation error.
     *
     * <p>t = 0 is excluded on purpose: a DSST OSCULATING propagator returns the raw mean state at the
     * initial epoch (the short-period terms are only added strictly after the first step), so the
     * osculating and mean states coincide there and would spuriously show a zero correction — the same
     * documented behaviour the zonal short-period oracle relies on.
     *
     * <p>Fresh {@link DSSTThirdBody} instances are used for every propagator because the force models
     * are mutated during propagation. The truncation orders (maxAR3Pow, maxEccPow) are read via
     * reflection so the port's identical truncation search is checked (Sun: (3, 2); Moon: (4, 2)).
     */
    private static void writeDsstThirdBodyShortPeriodValidation(File out, Frame gcrf, AbsoluteDate epoch)
            throws IOException {

        final double mu = GravityFieldFactory.getNormalizedProvider(2, 0).getMu();

        // MEAN initial orbit: same LEO as the third-body mean case.
        final KeplerianOrbit kep = new KeplerianOrbit(
                7.2e6, 0.01, FastMath.toRadians(98.0),
                FastMath.toRadians(30.0), FastMath.toRadians(42.0), FastMath.toRadians(10.0),
                PositionAngleType.MEAN, gcrf, epoch, mu);
        final EquinoctialOrbit eq = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(kep);

        final CelestialBody sunBody = CelestialBodyFactory.getSun();
        final CelestialBody moonBody = CelestialBodyFactory.getMoon();

        // Truncation orders (for the Rust test's assertions) and each body's GM.
        final DSSTThirdBody sunTr = new DSSTThirdBody(sunBody, mu);
        final DSSTThirdBody moonTr = new DSSTThirdBody(moonBody, mu);
        final AuxiliaryElements aux = new AuxiliaryElements(eq, 1);
        final double[] sunParams = sunTr.getParameters(epoch);
        final double[] moonParams = moonTr.getParameters(epoch);
        sunTr.initializeShortPeriodTerms(aux, PropagationType.OSCULATING, sunParams);
        moonTr.initializeShortPeriodTerms(aux, PropagationType.OSCULATING, moonParams);
        final int[] sunPow = thirdBodyMaxPowers(sunTr);
        final int[] moonPow = thirdBodyMaxPowers(moonTr);
        final double sunGm = sunParams[0];
        final double moonGm = moonParams[0];

        final double[][] tol = DSSTPropagator.tolerances(1.0e-6, eq);

        // Sun-only: paired MEAN and OSCULATING propagators (fresh force-model instances).
        final DSSTPropagator sunMean = new DSSTPropagator(
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]), PropagationType.MEAN);
        sunMean.setInitialState(new SpacecraftState(eq), PropagationType.MEAN);
        sunMean.addForceModel(new DSSTThirdBody(sunBody, mu));
        final DSSTPropagator sunOsc = new DSSTPropagator(
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]), PropagationType.OSCULATING);
        sunOsc.setInitialState(new SpacecraftState(eq), PropagationType.MEAN);
        sunOsc.addForceModel(new DSSTThirdBody(sunBody, mu));

        // Sun+Moon: paired MEAN and OSCULATING propagators.
        final DSSTPropagator bothMean = new DSSTPropagator(
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]), PropagationType.MEAN);
        bothMean.setInitialState(new SpacecraftState(eq), PropagationType.MEAN);
        bothMean.addForceModel(new DSSTThirdBody(sunBody, mu));
        bothMean.addForceModel(new DSSTThirdBody(moonBody, mu));
        final DSSTPropagator bothOsc = new DSSTPropagator(
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]), PropagationType.OSCULATING);
        bothOsc.setInitialState(new SpacecraftState(eq), PropagationType.MEAN);
        bothOsc.addForceModel(new DSSTThirdBody(sunBody, mu));
        bothOsc.addForceModel(new DSSTThirdBody(moonBody, mu));

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nsun_gm,%.17e%nmoon_gm,%.17e%n"
                        + "sun_max_ar3,%d%nsun_max_ecc,%d%nmoon_max_ar3,%d%nmoon_max_ecc,%d%n"
                        + "a0,%.17e%nex0,%.17e%ney0,%.17e%nhx0,%.17e%nhy0,%.17e%nlm0,%.17e%n",
                mu, sunGm, moonGm, sunPow[0], sunPow[1], moonPow[0], moonPow[1],
                eq.getA(), eq.getEquinoctialEx(), eq.getEquinoctialEy(), eq.getHx(), eq.getHy(), eq.getLM()));

        // Sun-only samples (exclude t=0: OSCULATING returns raw mean there).
        csv.append("sun_t_s,a_mean,ex_mean,ey_mean,hx_mean,hy_mean,lm_mean,"
                + "a_osc,ex_osc,ey_osc,hx_osc,hy_osc,lm_osc,sunx,suny,sunz").append(System.lineSeparator());
        for (int k = 1; k <= 4; k++) {
            final double t = k * 21600.0;
            final AbsoluteDate d = epoch.shiftedBy(t);
            final EquinoctialOrbit om = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(sunMean.propagate(d).getOrbit());
            final EquinoctialOrbit oo = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(sunOsc.propagate(d).getOrbit());
            final Vector3D sp = sunBody.getPosition(d, gcrf);
            csv.append(String.format(Locale.ROOT,
                    "%.1f,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                    t, om.getA(), om.getEquinoctialEx(), om.getEquinoctialEy(), om.getHx(), om.getHy(), om.getLM(),
                    oo.getA(), oo.getEquinoctialEx(), oo.getEquinoctialEy(), oo.getHx(), oo.getHy(), oo.getLM(),
                    sp.getX(), sp.getY(), sp.getZ()));
        }

        // Sun+Moon samples.
        csv.append("both_t_s,a_mean,ex_mean,ey_mean,hx_mean,hy_mean,lm_mean,"
                + "a_osc,ex_osc,ey_osc,hx_osc,hy_osc,lm_osc,sunx,suny,sunz,moonx,moony,moonz").append(System.lineSeparator());
        for (int k = 1; k <= 4; k++) {
            final double t = k * 21600.0;
            final AbsoluteDate d = epoch.shiftedBy(t);
            final EquinoctialOrbit om = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(bothMean.propagate(d).getOrbit());
            final EquinoctialOrbit oo = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(bothOsc.propagate(d).getOrbit());
            final Vector3D sp = sunBody.getPosition(d, gcrf);
            final Vector3D mp = moonBody.getPosition(d, gcrf);
            csv.append(String.format(Locale.ROOT,
                    "%.1f,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                    t, om.getA(), om.getEquinoctialEx(), om.getEquinoctialEy(), om.getHx(), om.getHy(), om.getLM(),
                    oo.getA(), oo.getEquinoctialEx(), oo.getEquinoctialEy(), oo.getHx(), oo.getHy(), oo.getLM(),
                    sp.getX(), sp.getY(), sp.getZ(), mp.getX(), mp.getY(), mp.getZ()));
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Emit a DSST solar-radiation-pressure mean-element-rate validation for the spacelab port. From
     * the same fixed MEAN equinoctial LEO state as the zonal/third-body cases, we build the
     * cannonball {@link DSSTSolarRadiationPressure} (reflection coefficient cr, cross-section area,
     * spacecraft mass) with a SPHERICAL central body (a {@code OneAxisEllipsoid} of flattening 0 and
     * radius ae): a sphere makes the conical umbra/penumbra lighting ratio frame-independent, so the
     * Rust port reproduces it directly from the emitted GCRF Sun position with no Earth-orientation
     * transform. We emit the six fixed-state mean-element rates getMeanElementRate =
     * [da, dk, dh, dq, dp, dM], the Sun position at the epoch (GCRF), the SRP parameters
     * (cr, area, mass, pRef, dRef, ae, mu), plus — via reflection — the integration limits
     * [Lexit, Lentry] getLLimits set and the Gauss quadrature order the isDirty pass selected.
     *
     * <p>The force context is set up exactly as {@code DSSTPropagator.computeDerivatives} does:
     * {@code new AuxiliaryElements(orbit, 1)}, an attitude provider is registered (SRP needs one even
     * though the isotropic model ignores it), then {@code initializeShortPeriodTerms(aux, MEAN,
     * params)} followed by {@code getMeanElementRate(state, aux, params)} — whose returned value is
     * the order-48 quadrature (the isDirty pass only caches a smaller order for later steps).
     */
    private static void writeDsstSrpValidation(File out, Frame gcrf, Frame itrf, AbsoluteDate epoch)
            throws IOException {

        final double mu = GravityFieldFactory.getNormalizedProvider(2, 0).getMu();
        final double ae = GravityFieldFactory.getNormalizedProvider(2, 0).getAe();

        // MEAN initial orbit: same sun-synchronous-ish LEO as the zonal/third-body cases.
        final KeplerianOrbit kep = new KeplerianOrbit(
                7.2e6, 0.01, FastMath.toRadians(98.0),
                FastMath.toRadians(30.0), FastMath.toRadians(42.0), FastMath.toRadians(10.0),
                PositionAngleType.MEAN, gcrf, epoch, mu);
        final EquinoctialOrbit eq = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(kep);

        // Cannonball spacecraft.
        final double cr = 1.5;
        final double area = 10.0;
        final double mass = 1000.0;
        final double pRef = 4.56e-6;
        final double dRef = 149597870000.0;

        // SPHERICAL Earth (flattening 0): the occultation limb radius reduces to asin(ae / r), so the
        // conical lighting ratio is rotation-invariant and reproducible from inertial positions.
        final CelestialBody sunBody = CelestialBodyFactory.getSun();
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(ae, 0.0, itrf);
        final DSSTSolarRadiationPressure srp =
                new DSSTSolarRadiationPressure(cr, area, sunBody, earth, mu);
        srp.registerAttitudeProvider(new FrameAlignedProvider(Rotation.IDENTITY));

        final AuxiliaryElements aux = new AuxiliaryElements(eq, 1);
        final SpacecraftState state = new SpacecraftState(eq).withMass(mass);
        final double[] params = srp.getParameters(epoch);
        srp.initializeShortPeriodTerms(aux, PropagationType.MEAN, params);
        final double[] rates = srp.getMeanElementRate(state, aux, params);

        final Vector3D sunPos = sunBody.getPosition(epoch, gcrf);

        // Integration limits getLLimits actually used (protected — via reflection).
        double llLow = Double.NaN;
        double llHigh = Double.NaN;
        try {
            final java.lang.reflect.Method m = DSSTSolarRadiationPressure.class
                    .getDeclaredMethod("getLLimits", SpacecraftState.class, AuxiliaryElements.class);
            m.setAccessible(true);
            final double[] ll = (double[]) m.invoke(srp, state, aux);
            llLow = ll[0];
            llHigh = ll[1];
        } catch (ReflectiveOperationException e) {
            // leave as NaN
        }

        // Gauss quadrature order the isDirty pass selected (field on AbstractGaussianContribution).
        int gaussOrder = -1;
        try {
            final java.lang.reflect.Field intF =
                    srp.getClass().getSuperclass().getDeclaredField("integrator");
            intF.setAccessible(true);
            final Object gq = intF.get(srp);
            final java.lang.reflect.Field npF = gq.getClass().getDeclaredField("numberOfPoints");
            npF.setAccessible(true);
            gaussOrder = npF.getInt(gq);
        } catch (ReflectiveOperationException e) {
            gaussOrder = -1;
        }

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nae,%.17e%ncr,%.17e%narea,%.17e%nmass,%.17e%np_ref,%.17e%nd_ref,%.17e%n"
                        + "gauss_order,%d%nll_low,%.17e%nll_high,%.17e%n"
                        + "a0,%.17e%nex0,%.17e%ney0,%.17e%nhx0,%.17e%nhy0,%.17e%nlm0,%.17e%n"
                        + "sun_pos,%.17e,%.17e,%.17e%n"
                        + "rate_da,%.17e%nrate_dk,%.17e%nrate_dh,%.17e%n"
                        + "rate_dq,%.17e%nrate_dp,%.17e%nrate_dm,%.17e%n",
                mu, ae, cr, area, mass, pRef, dRef,
                gaussOrder, llLow, llHigh,
                eq.getA(), eq.getEquinoctialEx(), eq.getEquinoctialEy(), eq.getHx(), eq.getHy(), eq.getLM(),
                sunPos.getX(), sunPos.getY(), sunPos.getZ(),
                rates[0], rates[1], rates[2], rates[3], rates[4], rates[5]));
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Emit a DSST ATMOSPHERIC DRAG mean-element-rate validation for the spacelab port. A cannonball
     * IsotropicDrag (cd, area, mass) on a MEAN equinoctial LEO whose apogee lies below the critical
     * atmospheric radius rbar, so getLLimits returns the full circle and the Gauss quadrature averages
     * the whole orbit. The atmosphere is a CONSTANT-density, purely co-rotating model: rho everywhere
     * and vAtm = omega*zHat x r (a fixed spin about +Z, getVelocity overridden to the explicit cross
     * product), driven identically on both the Java and Rust sides so the co-rotation kinematics + the
     * Gauss quadrature are validated with no NRLMSISE-00 or ITRF frame table. Emits getMeanElementRate
     * = [da,dk,dh,dq,dp,dM], the getLLimits window used and the Gauss order the isDirty pass selected.
     */
    private static void writeDsstDragValidation(File out, Frame gcrf, AbsoluteDate epoch)
            throws IOException {

        final double mu = GravityFieldFactory.getNormalizedProvider(2, 0).getMu();
        final double ae = GravityFieldFactory.getNormalizedProvider(2, 0).getAe();

        // MEAN equinoctial LEO, apogee below rbar (whole orbit inside the atmosphere).
        final double a0 = 7.0e6;
        final double ex0 = 3.09016994374947470e-03;
        final double ey0 = 9.51056516295153600e-03;
        final double hx0 = 8.54890329218965000e-01;
        final double hy0 = 7.69746709859868000e-01;
        final double lm0 = 1.43116998663535020e+00;
        final EquinoctialOrbit eq = new EquinoctialOrbit(
                a0, ex0, ey0, hx0, hy0, lm0, PositionAngleType.MEAN, gcrf, epoch, mu);

        // Cannonball spacecraft.
        final double cd = 2.2;
        final double area = 10.0;
        final double mass = 1000.0;

        // Constant-density, purely co-rotating atmosphere. getVelocity is overridden to the explicit
        // omega x r so it is bit-identical to the Rust CoRotatingConstant (no frame-transform velocity).
        final double rho = 1.0e-12;
        final double omega = 7.29211514670697900e-05;
        final double rbar = 7.37813700000000000e+06;
        final Atmosphere atmosphere = new Atmosphere() {
            @Override
            public Frame getFrame() {
                return gcrf;
            }

            @Override
            public double getDensity(final AbsoluteDate date, final Vector3D position, final Frame frame) {
                return rho;
            }

            @Override
            public <T extends CalculusFieldElement<T>> T getDensity(
                    final FieldAbsoluteDate<T> date, final FieldVector3D<T> position, final Frame frame) {
                return position.getX().getField().getZero().add(rho);
            }

            @Override
            public Vector3D getVelocity(final AbsoluteDate date, final Vector3D position, final Frame frame) {
                return Vector3D.crossProduct(new Vector3D(0.0, 0.0, omega), position);
            }
        };

        final DSSTAtmosphericDrag drag = new DSSTAtmosphericDrag(atmosphere, cd, area, mu);
        drag.setRbar(rbar);
        drag.registerAttitudeProvider(new FrameAlignedProvider(Rotation.IDENTITY));

        final AuxiliaryElements aux = new AuxiliaryElements(eq, 1);
        final SpacecraftState state = new SpacecraftState(eq).withMass(mass);
        final double[] params = drag.getParameters(epoch);
        drag.initializeShortPeriodTerms(aux, PropagationType.MEAN, params);
        final double[] rates = drag.getMeanElementRate(state, aux, params);

        // Integration limits getLLimits actually used (protected — via reflection).
        double llLow = Double.NaN;
        double llHigh = Double.NaN;
        try {
            final java.lang.reflect.Method m = DSSTAtmosphericDrag.class
                    .getDeclaredMethod("getLLimits", SpacecraftState.class, AuxiliaryElements.class);
            m.setAccessible(true);
            final double[] ll = (double[]) m.invoke(drag, state, aux);
            llLow = ll[0];
            llHigh = ll[1];
        } catch (ReflectiveOperationException e) {
            // leave as NaN
        }

        // Gauss quadrature order the isDirty pass selected (field on AbstractGaussianContribution).
        int gaussOrder = -1;
        try {
            final java.lang.reflect.Field intF =
                    drag.getClass().getSuperclass().getDeclaredField("integrator");
            intF.setAccessible(true);
            final Object gq = intF.get(drag);
            final java.lang.reflect.Field npF = gq.getClass().getDeclaredField("numberOfPoints");
            npF.setAccessible(true);
            gaussOrder = npF.getInt(gq);
        } catch (ReflectiveOperationException e) {
            gaussOrder = -1;
        }

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nae,%.17e%ncd,%.17e%narea,%.17e%nmass,%.17e%nrho,%.17e%nomega,%.17e%nrbar,%.17e%n"
                        + "gauss_order,%d%nll_low,%.17e%nll_high,%.17e%n"
                        + "a0,%.17e%nex0,%.17e%ney0,%.17e%nhx0,%.17e%nhy0,%.17e%nlm0,%.17e%n"
                        + "rate_da,%.17e%nrate_dk,%.17e%nrate_dh,%.17e%n"
                        + "rate_dq,%.17e%nrate_dp,%.17e%nrate_dm,%.17e%n",
                mu, ae, cd, area, mass, rho, omega, rbar,
                gaussOrder, llLow, llHigh,
                eq.getA(), eq.getEquinoctialEx(), eq.getEquinoctialEy(), eq.getHx(), eq.getHy(), eq.getLM(),
                rates[0], rates[1], rates[2], rates[3], rates[4], rates[5]));
        Files.writeString(out.toPath(), csv.toString());
    }

    /** Build a fresh cannonball {@link DSSTSolarRadiationPressure} over a SPHERICAL Earth (flattening
     * 0, radius ae) — a fresh instance per propagator because the force model is mutated during
     * propagation. The sphere makes the conical umbra/penumbra lighting ratio frame-independent, so the
     * spacelab port reproduces it directly from the emitted GCRF Sun position. */
    private static DSSTSolarRadiationPressure newSrp(final double cr, final double area,
            final CelestialBody sunBody, final double ae, final Frame itrf, final double mu) {
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(ae, 0.0, itrf);
        return new DSSTSolarRadiationPressure(cr, area, sunBody, earth, mu);
    }

    /**
     * Emit a DSST solar-radiation-pressure SHORT-PERIOD validation for the spacelab port. From the same
     * fixed MEAN equinoctial LEO as the SRP mean case (a shadow-crossing orbit, so the sunlit arc is a
     * proper sub-interval), we run a {@code PropagationType.MEAN} and a {@code PropagationType.OSCULATING}
     * DSSTPropagator, each seeded from the MEAN initial state, with ONLY the cannonball SRP force. At
     * every t &gt; 0 sample we emit both the mean equinoctial state (MEAN run) and the osculating one
     * (OSCULATING run), plus the geocentric GCRF Sun position at that date. The Rust side validates ONLY
     * the short-period assembly: osc(t) − mean(t) must equal the port's Gaussian short-period δ evaluated
     * at mean(t) with the emitted Sun geometry, isolated from mean-propagation error.
     *
     * <p>t = 0 is excluded on purpose: a DSST OSCULATING propagator returns the raw mean state at the
     * initial epoch, so the correction would spuriously read zero there — the same documented behaviour
     * the zonal / third-body short-period oracles rely on. Fresh {@link DSSTSolarRadiationPressure}
     * instances are used for each propagator because the force models are mutated during propagation.
     */
    private static void writeDsstSrpShortPeriodValidation(File out, Frame gcrf, Frame itrf, AbsoluteDate epoch)
            throws IOException {

        final double mu = GravityFieldFactory.getNormalizedProvider(2, 0).getMu();
        final double ae = GravityFieldFactory.getNormalizedProvider(2, 0).getAe();

        final KeplerianOrbit kep = new KeplerianOrbit(
                7.2e6, 0.01, FastMath.toRadians(98.0),
                FastMath.toRadians(30.0), FastMath.toRadians(42.0), FastMath.toRadians(10.0),
                PositionAngleType.MEAN, gcrf, epoch, mu);
        final EquinoctialOrbit eq = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(kep);

        final double cr = 1.5;
        final double area = 10.0;
        final double mass = 1000.0;
        final double pRef = 4.56e-6;
        final double dRef = 149597870000.0;
        final CelestialBody sunBody = CelestialBodyFactory.getSun();

        final double[][] tol = DSSTPropagator.tolerances(1.0e-6, eq);
        final DSSTPropagator srpMean = new DSSTPropagator(
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]), PropagationType.MEAN);
        srpMean.setInitialState(new SpacecraftState(eq).withMass(mass), PropagationType.MEAN);
        srpMean.addForceModel(newSrp(cr, area, sunBody, ae, itrf, mu));
        final DSSTPropagator srpOsc = new DSSTPropagator(
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]), PropagationType.OSCULATING);
        srpOsc.setInitialState(new SpacecraftState(eq).withMass(mass), PropagationType.MEAN);
        srpOsc.addForceModel(newSrp(cr, area, sunBody, ae, itrf, mu));

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nae,%.17e%ncr,%.17e%narea,%.17e%nmass,%.17e%np_ref,%.17e%nd_ref,%.17e%n"
                        + "a0,%.17e%nex0,%.17e%ney0,%.17e%nhx0,%.17e%nhy0,%.17e%nlm0,%.17e%n",
                mu, ae, cr, area, mass, pRef, dRef,
                eq.getA(), eq.getEquinoctialEx(), eq.getEquinoctialEy(), eq.getHx(), eq.getHy(), eq.getLM()));
        csv.append("t_s,a_mean,ex_mean,ey_mean,hx_mean,hy_mean,lm_mean,"
                + "a_osc,ex_osc,ey_osc,hx_osc,hy_osc,lm_osc,sunx,suny,sunz").append(System.lineSeparator());
        for (int k = 1; k <= 4; k++) {
            final double t = k * 21600.0;
            final AbsoluteDate d = epoch.shiftedBy(t);
            final EquinoctialOrbit om = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(srpMean.propagate(d).getOrbit());
            final EquinoctialOrbit oo = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(srpOsc.propagate(d).getOrbit());
            final Vector3D sp = sunBody.getPosition(d, gcrf);
            csv.append(String.format(Locale.ROOT,
                    "%.1f,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                    t, om.getA(), om.getEquinoctialEx(), om.getEquinoctialEy(), om.getHx(), om.getHy(), om.getLM(),
                    oo.getA(), oo.getEquinoctialEx(), oo.getEquinoctialEy(), oo.getHx(), oo.getHy(), oo.getLM(),
                    sp.getX(), sp.getY(), sp.getZ()));
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Emit a DSST ATMOSPHERIC DRAG SHORT-PERIOD validation for the spacelab port. From the same fixed
     * MEAN equinoctial LEO as the drag mean case (apogee below rbar, so getLLimits is the full circle),
     * we run a {@code PropagationType.MEAN} and a {@code PropagationType.OSCULATING} DSSTPropagator, each
     * seeded from the MEAN initial state, with ONLY the cannonball drag force over the SAME constant-
     * density, purely co-rotating atmosphere as the mean case (rho everywhere, vAtm = omega*zHat x r).
     * At every t &gt; 0 sample we emit both the mean equinoctial state (MEAN run) and the osculating one
     * (OSCULATING run). The Rust side validates ONLY the short-period assembly: osc(t) − mean(t) must
     * equal the port's Gaussian short-period δ evaluated at mean(t) with the identical atmosphere,
     * isolated from mean-propagation error. t = 0 is excluded (OSCULATING returns raw mean there).
     */
    private static void writeDsstDragShortPeriodValidation(File out, Frame gcrf, AbsoluteDate epoch)
            throws IOException {

        final double mu = GravityFieldFactory.getNormalizedProvider(2, 0).getMu();
        final double ae = GravityFieldFactory.getNormalizedProvider(2, 0).getAe();

        final double a0 = 7.0e6;
        final double ex0 = 3.09016994374947470e-03;
        final double ey0 = 9.51056516295153600e-03;
        final double hx0 = 8.54890329218965000e-01;
        final double hy0 = 7.69746709859868000e-01;
        final double lm0 = 1.43116998663535020e+00;
        final EquinoctialOrbit eq = new EquinoctialOrbit(
                a0, ex0, ey0, hx0, hy0, lm0, PositionAngleType.MEAN, gcrf, epoch, mu);

        final double cd = 2.2;
        final double area = 10.0;
        final double mass = 1000.0;
        final double rho = 1.0e-12;
        final double omega = 7.29211514670697900e-05;
        final double rbar = 7.37813700000000000e+06;
        final Atmosphere atmosphere = new Atmosphere() {
            @Override
            public Frame getFrame() {
                return gcrf;
            }

            @Override
            public double getDensity(final AbsoluteDate date, final Vector3D position, final Frame frame) {
                return rho;
            }

            @Override
            public <T extends CalculusFieldElement<T>> T getDensity(
                    final FieldAbsoluteDate<T> date, final FieldVector3D<T> position, final Frame frame) {
                return position.getX().getField().getZero().add(rho);
            }

            @Override
            public Vector3D getVelocity(final AbsoluteDate date, final Vector3D position, final Frame frame) {
                return Vector3D.crossProduct(new Vector3D(0.0, 0.0, omega), position);
            }
        };

        final double[][] tol = DSSTPropagator.tolerances(1.0e-6, eq);
        final DSSTPropagator dragMean = new DSSTPropagator(
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]), PropagationType.MEAN);
        dragMean.setInitialState(new SpacecraftState(eq).withMass(mass), PropagationType.MEAN);
        final DSSTAtmosphericDrag dm = new DSSTAtmosphericDrag(atmosphere, cd, area, mu);
        dm.setRbar(rbar);
        dragMean.addForceModel(dm);
        final DSSTPropagator dragOsc = new DSSTPropagator(
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]), PropagationType.OSCULATING);
        dragOsc.setInitialState(new SpacecraftState(eq).withMass(mass), PropagationType.MEAN);
        final DSSTAtmosphericDrag doo = new DSSTAtmosphericDrag(atmosphere, cd, area, mu);
        doo.setRbar(rbar);
        dragOsc.addForceModel(doo);

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nae,%.17e%ncd,%.17e%narea,%.17e%nmass,%.17e%nrho,%.17e%nomega,%.17e%nrbar,%.17e%n"
                        + "a0,%.17e%nex0,%.17e%ney0,%.17e%nhx0,%.17e%nhy0,%.17e%nlm0,%.17e%n",
                mu, ae, cd, area, mass, rho, omega, rbar,
                eq.getA(), eq.getEquinoctialEx(), eq.getEquinoctialEy(), eq.getHx(), eq.getHy(), eq.getLM()));
        csv.append("t_s,a_mean,ex_mean,ey_mean,hx_mean,hy_mean,lm_mean,"
                + "a_osc,ex_osc,ey_osc,hx_osc,hy_osc,lm_osc").append(System.lineSeparator());
        for (int k = 1; k <= 4; k++) {
            final double t = k * 21600.0;
            final AbsoluteDate d = epoch.shiftedBy(t);
            final EquinoctialOrbit om = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(dragMean.propagate(d).getOrbit());
            final EquinoctialOrbit oo = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(dragOsc.propagate(d).getOrbit());
            csv.append(String.format(Locale.ROOT,
                    "%.1f,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                    t, om.getA(), om.getEquinoctialEx(), om.getEquinoctialEy(), om.getHx(), om.getHy(), om.getLM(),
                    oo.getA(), oo.getEquinoctialEx(), oo.getEquinoctialEy(), oo.getHx(), oo.getHy(), oo.getLM()));
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Emit a DSST central-body TESSERAL mean-element-rate validation for the spacelab port. The
     * non-zonal field is taken to degree/order 4x4 with Orekit's own unnormalized Cnm/Snm. From a
     * fixed MEAN equinoctial state we emit the six mean-element rates getMeanElementRate =
     * [da, dk, dh, dq, dp, dM].
     *
     * <p>Only resonant tesseral terms contribute to the tesseral MEAN rate (the m-daily and other
     * non-resonant terms are short-period, computed in updateShortPeriodTerms — out of scope here).
     * We therefore emit three cases: a LEO (no resonance -> all rates identically zero), a GEO
     * (1:1 resonance, resonant orders 1..4) and a GPS-like orbit (2:1 resonance, resonant orders 2,4)
     * that exercise the resonant machinery (Hansen tesseral kernels, GHmsj, GammaMns, Jacobi, Vmns).
     *
     * <p>The body-fixed frame is a FIXED rotation of the orbit (GCRF) frame about +Z by a sidereal
     * angle gmst0. A pure-Z rotation keeps the celestial pole aligned with the inertial Z, so the
     * tesseral direction cosines alpha/beta/gamma reduce (to round-off) to the (p,q)-only equinoctial
     * values the port computes, while Orekit's central-body rotation angle theta equals gmst0 exactly.
     * The port is fed gmst = theta and reconstructs the m*theta phase from it.
     */
    private static void writeDsstTesseralValidation(File out, Frame gcrf, AbsoluteDate epoch)
            throws IOException {

        final int deg = 4;
        final UnnormalizedSphericalHarmonicsProvider provider = GravityFieldFactory.getUnnormalizedProvider(deg, deg);
        final double muT = provider.getMu();
        final double aeT = provider.getAe();
        final double rate = Constants.WGS84_EARTH_ANGULAR_VELOCITY;
        final UnnormalizedSphericalHarmonicsProvider.UnnormalizedSphericalHarmonics harm = provider.onDate(epoch);
        final int maxFreq = FastMath.min(12, provider.getMaxDegree() + 4);

        // Body-fixed frame: fixed rotation of GCRF about +Z by gmst0 (idealized non-precessing Earth).
        final double gmst0 = 1.234;
        final Transform zrot = new Transform(epoch,
                new Rotation(Vector3D.PLUS_K, gmst0, RotationConvention.FRAME_TRANSFORM));
        final Frame bodyFrame = new Frame(gcrf, zrot, "ZBODY");

        // Central body rotation angle theta from the same formula as DSSTTesseralContext (I = +1).
        final StaticTransform t = bodyFrame.getStaticTransformTo(gcrf, epoch);
        final Vector3D xB = t.transformVector(Vector3D.PLUS_I);
        final Vector3D yB = t.transformVector(Vector3D.PLUS_J);

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nae,%.17e%nrate,%.17e%ngmst0,%.17e%nmax_degree,%d%nmax_order,%d%nmax_frequency,%d%n",
                muT, aeT, rate, gmst0, provider.getMaxDegree(), provider.getMaxOrder(), maxFreq));
        // Unnormalized Cnm/Snm table.
        for (int n = 2; n <= deg; n++) {
            for (int m = 0; m <= n; m++) {
                csv.append(String.format(Locale.ROOT, "cs,%d,%d,%.17e,%.17e%n",
                        n, m, harm.getUnnormalizedCnm(n, m), harm.getUnnormalizedSnm(n, m)));
            }
        }

        final String[] names = {"leo", "geo", "gps"};
        final double[][] orbits = {
            {7.2e6, 0.01, 98.0},
            {42164173.0, 0.01, 15.0},
            {26560000.0, 0.01, 55.0},
        };
        for (int ci = 0; ci < names.length; ci++) {
            final double[] ob = orbits[ci];
            final KeplerianOrbit kep = new KeplerianOrbit(ob[0], ob[1], FastMath.toRadians(ob[2]),
                    FastMath.toRadians(30.0), FastMath.toRadians(42.0), FastMath.toRadians(10.0),
                    PositionAngleType.MEAN, gcrf, epoch, muT);
            final EquinoctialOrbit eq = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(kep);

            final DSSTTesseral tess = new DSSTTesseral(bodyFrame, rate, provider);
            final AuxiliaryElements aux = new AuxiliaryElements(eq, 1);
            final double[] params = tess.getParameters(epoch);
            tess.initializeShortPeriodTerms(aux, PropagationType.MEAN, params);
            final SpacecraftState state = new SpacecraftState(eq);
            final double[] rates = tess.getMeanElementRate(state, aux, params);

            final int I = 1;
            final double theta = FastMath.atan2(
                    -aux.getVectorF().dotProduct(yB) + I * aux.getVectorG().dotProduct(xB),
                     aux.getVectorF().dotProduct(xB) + I * aux.getVectorG().dotProduct(yB));

            int maxEccPow = -1;
            int maxHansen = -1;
            String res = "";
            try {
                final java.lang.reflect.Field ep = DSSTTesseral.class.getDeclaredField("maxEccPow");
                ep.setAccessible(true);
                maxEccPow = ep.getInt(tess);
                final java.lang.reflect.Field mh = DSSTTesseral.class.getDeclaredField("maxHansen");
                mh.setAccessible(true);
                maxHansen = mh.getInt(tess);
                final java.lang.reflect.Field rf = DSSTTesseral.class.getDeclaredField("resOrders");
                rf.setAccessible(true);
                @SuppressWarnings("unchecked")
                final List<Integer> ro = (List<Integer>) rf.get(tess);
                final StringBuilder rb = new StringBuilder();
                for (int k = 0; k < ro.size(); k++) {
                    if (k > 0) rb.append(' ');
                    rb.append(ro.get(k));
                }
                res = rb.toString();
            } catch (ReflectiveOperationException e) {
                res = "?";
            }

            csv.append(String.format(Locale.ROOT,
                    "case,%s,theta,%.17e,res,%s,maxEccPow,%d,maxHansen,%d%n"
                            + "state,%s,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n"
                            + "rate,%s,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                    names[ci], theta, res, maxEccPow, maxHansen,
                    names[ci], eq.getA(), eq.getEquinoctialEx(), eq.getEquinoctialEy(),
                    eq.getHx(), eq.getHy(), eq.getLM(),
                    names[ci], rates[0], rates[1], rates[2], rates[3], rates[4], rates[5]));
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Emit a DSST tesseral SHORT-PERIOD validation for the spacelab port. Same 4x4 unnormalized field,
     * WGS84 rotation rate and idealized non-precessing body frame (a fixed +Z rotation of GCRF by
     * gmst0) as {@link #writeDsstTesseralValidation}. From a fixed MEAN LEO state we run a
     * {@code PropagationType.MEAN} and a {@code PropagationType.OSCULATING} DSSTPropagator carrying ONLY
     * {@link DSSTTesseral} (built with the default constructor, so the short-period truncations follow
     * Orekit's defaults: maxDegree/OrderTesseralSP = 4, maxEccPowTesseralSP = 4, maxFrequency = 8,
     * maxDegree/OrderMdailyTesseralSP = 4, maxEccPowMdailyTesseralSP = 2). At every t &gt; 0 sample we
     * emit the mean equinoctial state (from a MEAN run) and Orekit's tesseral short-period delta
     * evaluated DIRECTLY via {@code ShortPeriodTerms.value(meanOrbit)} at that mean state (no second
     * propagator, no coefficient interpolation), so the Rust side validates ONLY the tesseral
     * short-period assembly (m-daily + non-resonant): the port's delta at mean(t) with the emitted GMST
     * must equal Orekit's value(mean(t)). On a LEO there is no tesseral resonance, so the mean tesseral
     * drift is identically zero (a, k, h, q, p are frozen) and the delta is the whole tesseral
     * short-period (dominated by the ever-present m-daily terms); the four samples share the same frozen
     * a/k/h/q/p and differ only in the mean longitude, exercising the phase assembly.
     *
     * <p>Evaluating {@code value(...)} directly (as the zonal/third-body mean-rate oracles call
     * getMeanElementRate directly) isolates the closed-form short-period from the propagator's Hermite
     * interpolation of the coefficients between integration-step grid points.
     *
     * <p>The GMST emitted is the central-body rotation angle theta from the same atan2 formula
     * {@code DSSTTesseralContext} / the tesseral {@code value(...)} assembly use (I = +1); with a
     * non-precessing body frame and a tesseral-only LEO (p, q constant) it is time-independent, so the
     * single epoch value drives every sample.
     *
     * <p>The field is FROZEN at the epoch ({@code getConstantUnnormalizedProvider}) so that
     * {@code provider.onDate(t)} returns the same Cnm/Snm at every sample. The default
     * {@code getUnnormalizedProvider} is time-dependent (pole tide / secular drift — e.g. C21 drifts
     * ~3e-4 relative per day), which would make Orekit's short-period coefficients at t &gt; 0 use
     * different harmonics than the static table the spacelab port carries; freezing the field isolates
     * the short-period assembly from the field's time model, exactly as the port's static Cnm/Snm intends.
     */
    private static void writeDsstTesseralShortPeriodValidation(File out, Frame gcrf, AbsoluteDate epoch)
            throws IOException {

        final int deg = 4;
        final UnnormalizedSphericalHarmonicsProvider provider =
                GravityFieldFactory.getConstantUnnormalizedProvider(deg, deg, epoch);
        final double muT = provider.getMu();
        final double aeT = provider.getAe();
        final double rate = Constants.WGS84_EARTH_ANGULAR_VELOCITY;
        final UnnormalizedSphericalHarmonicsProvider.UnnormalizedSphericalHarmonics harm = provider.onDate(epoch);
        final int maxFreq = FastMath.min(12, provider.getMaxDegree() + 4);

        // Body-fixed frame: fixed rotation of GCRF about +Z by gmst0 (idealized non-precessing Earth).
        final double gmst0 = 1.234;
        final Transform zrot = new Transform(epoch,
                new Rotation(Vector3D.PLUS_K, gmst0, RotationConvention.FRAME_TRANSFORM));
        final Frame bodyFrame = new Frame(gcrf, zrot, "ZBODY");

        // Central body rotation angle theta from the same formula as DSSTTesseralContext (I = +1).
        final StaticTransform t = bodyFrame.getStaticTransformTo(gcrf, epoch);
        final Vector3D xB = t.transformVector(Vector3D.PLUS_I);
        final Vector3D yB = t.transformVector(Vector3D.PLUS_J);

        // MEAN initial orbit: same LEO as the tesseral mean / zonal / third-body cases.
        final KeplerianOrbit kep = new KeplerianOrbit(
                7.2e6, 0.01, FastMath.toRadians(98.0),
                FastMath.toRadians(30.0), FastMath.toRadians(42.0), FastMath.toRadians(10.0),
                PositionAngleType.MEAN, gcrf, epoch, muT);
        final EquinoctialOrbit eq = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(kep);

        final int I = 1;
        final AuxiliaryElements aux0 = new AuxiliaryElements(eq, I);
        final double theta = FastMath.atan2(
                -aux0.getVectorF().dotProduct(yB) + I * aux0.getVectorG().dotProduct(xB),
                 aux0.getVectorF().dotProduct(xB) + I * aux0.getVectorG().dotProduct(yB));

        final double[][] tol = DSSTPropagator.tolerances(1.0e-6, eq);

        // MEAN propagator (fresh force-model instance), seeded from MEAN, for realistic mean states.
        final DSSTPropagator meanProp = new DSSTPropagator(
                new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]), PropagationType.MEAN);
        meanProp.setInitialState(new SpacecraftState(eq), PropagationType.MEAN);
        meanProp.addForceModel(new DSSTTesseral(bodyFrame, rate, provider));

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nae,%.17e%nrate,%.17e%ngmst,%.17e%nmax_degree,%d%nmax_order,%d%nmax_frequency,%d%n",
                muT, aeT, rate, theta, provider.getMaxDegree(), provider.getMaxOrder(), maxFreq));
        for (int n = 2; n <= deg; n++) {
            for (int m = 0; m <= n; m++) {
                csv.append(String.format(Locale.ROOT, "cs,%d,%d,%.17e,%.17e%n",
                        n, m, harm.getUnnormalizedCnm(n, m), harm.getUnnormalizedSnm(n, m)));
            }
        }
        csv.append(String.format(Locale.ROOT,
                "a0,%.17e%nex0,%.17e%ney0,%.17e%nhx0,%.17e%nhy0,%.17e%nlm0,%.17e%n",
                eq.getA(), eq.getEquinoctialEx(), eq.getEquinoctialEy(), eq.getHx(), eq.getHy(), eq.getLM()));
        csv.append("t_s,a_mean,ex_mean,ey_mean,hx_mean,hy_mean,lm_mean,"
                + "da,dk,dh,dq,dp,dl").append(System.lineSeparator());
        for (int k = 1; k <= 4; k++) {
            final double tt = k * 21600.0;
            final AbsoluteDate d = epoch.shiftedBy(tt);
            final EquinoctialOrbit om = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(meanProp.propagate(d).getOrbit());

            // Orekit's tesseral short-period delta evaluated DIRECTLY at the mean state via value().
            final DSSTTesseral tessSp = new DSSTTesseral(bodyFrame, rate, provider);
            final AuxiliaryElements auxK = new AuxiliaryElements(om, I);
            final double[] paramsK = tessSp.getParameters(om.getDate());
            final List<org.orekit.propagation.semianalytical.dsst.forces.ShortPeriodTerms> spt =
                    tessSp.initializeShortPeriodTerms(auxK, PropagationType.OSCULATING, paramsK);
            tessSp.updateShortPeriodTerms(paramsK, new SpacecraftState(om));
            final double[] delta = spt.get(0).value(om);

            csv.append(String.format(Locale.ROOT,
                    "%.1f,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                    tt, om.getA(), om.getEquinoctialEx(), om.getEquinoctialEy(), om.getHx(), om.getHy(), om.getLM(),
                    delta[0], delta[1], delta[2], delta[3], delta[4], delta[5]));
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * DSST zonal OSCULATING propagation: mean drift + short-period corrections. Emits the initial MEAN
     * equinoctial and the OSCULATING equinoctial at 3 h steps, for validating spacelab's short-period.
     */
    private static void writeDsstZonalShortPeriodValidation(File out, Frame gcrf, AbsoluteDate epoch)
            throws IOException {
        final UnnormalizedSphericalHarmonicsProvider provider = GravityFieldFactory.getUnnormalizedProvider(6, 0);
        final double muZ = provider.getMu();
        final double ae = provider.getAe();
        final UnnormalizedSphericalHarmonicsProvider.UnnormalizedSphericalHarmonics harm = provider.onDate(epoch);
        final double[] cn0 = new double[7];
        for (int n = 2; n <= 6; n++) {
            cn0[n] = harm.getUnnormalizedCnm(n, 0);
        }
        final KeplerianOrbit kep = new KeplerianOrbit(
                7.2e6, 0.01, FastMath.toRadians(98.0),
                FastMath.toRadians(30.0), FastMath.toRadians(42.0), FastMath.toRadians(10.0),
                PositionAngleType.MEAN, gcrf, epoch, muZ);
        final EquinoctialOrbit eq = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(kep);

        final DSSTZonal zonalProp = new DSSTZonal(provider);
        final double[][] tol = DSSTPropagator.tolerances(1.0e-6, eq);
        final AdaptiveStepsizeIntegrator integ = new DormandPrince853Integrator(1.0, 43200.0, tol[0], tol[1]);
        final DSSTPropagator dsst = new DSSTPropagator(integ, PropagationType.OSCULATING);
        dsst.setInitialState(new SpacecraftState(eq), PropagationType.MEAN);
        dsst.addForceModel(zonalProp);

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "mu,%.17e%nae,%.17e%nc20,%.17e%nc30,%.17e%nc40,%.17e%nc50,%.17e%nc60,%.17e%n"
                        + "a0,%.17e%nex0,%.17e%ney0,%.17e%nhx0,%.17e%nhy0,%.17e%nlm0,%.17e%n"
                        + "t_s,a,ex,ey,hx,hy,lm%n",
                muZ, ae, cn0[2], cn0[3], cn0[4], cn0[5], cn0[6],
                eq.getA(), eq.getEquinoctialEx(), eq.getEquinoctialEy(), eq.getHx(), eq.getHy(), eq.getLM()));
        for (int k = 0; k <= 4; k++) {
            final double t = k * 10800.0;
            final EquinoctialOrbit o = (EquinoctialOrbit) OrbitType.EQUINOCTIAL.convertType(
                    dsst.propagate(epoch.shiftedBy(t)).getOrbit());
            csv.append(String.format(Locale.ROOT, "%.1f,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n",
                    t, o.getA(), o.getEquinoctialEx(), o.getEquinoctialEy(),
                    o.getHx(), o.getHy(), o.getLM()));
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Emit the IERS 2010 relativistic accelerations (Eq. 10.12) at a fixed GCRF geometry, for
     * validating spacelab's Schwarzschild / Lense-Thirring / de Sitter terms. Orekit's
     * {@link LenseThirringRelativity} orients the angular momentum J along the supplied frame's Z
     * axis; the GCRF is passed so J = [0, 0, 9.8e8] in the inertial frame, matching spacelab's
     * convention. The Sun position and velocity are emitted so the spacelab side drives its de Sitter
     * term from the identical ephemeris. Everything is written in the GCRF.
     */
    private static void writeRelativityValidation(File out, Frame gcrf, AbsoluteDate epoch, double mu)
            throws IOException {
        final CelestialBody sun = CelestialBodyFactory.getSun();
        final ForceModel schwarzschild = new Relativity(mu);
        final ForceModel lense = new LenseThirringRelativity(mu, gcrf);
        final ForceModel deSitter = new DeSitterRelativity();

        final Vector3D rGcrf = new Vector3D(7.0e6, 1.2e6, 2.0e6);
        final Vector3D vGcrf = new Vector3D(-1500.0, 6500.0, 3200.0);
        final SpacecraftState state = new SpacecraftState(
                new CartesianOrbit(new PVCoordinates(rGcrf, vGcrf), gcrf, epoch, mu));

        final Vector3D aS = schwarzschild.acceleration(state, schwarzschild.getParameters(epoch));
        final Vector3D aL = lense.acceleration(state, lense.getParameters(epoch));
        final Vector3D aD = deSitter.acceleration(state, deSitter.getParameters(epoch));

        final PVCoordinates sunPv = sun.getPVCoordinates(epoch, gcrf);
        final Vector3D rSun = sunPv.getPosition();
        final Vector3D vSun = sunPv.getVelocity();

        // The exact Earth-relative-to-Sun PV that DeSitterRelativity consumes internally (Earth in the
        // Sun's inertially-oriented frame). spacelab drives its de Sitter term from these to validate
        // the formula against the identical ephemeris inputs, independent of the geocentric-Sun path.
        final CelestialBody earth = CelestialBodyFactory.getEarth();
        final PVCoordinates earthRelSun = earth.getPVCoordinates(epoch, sun.getInertiallyOrientedFrame());
        final Vector3D rEs = earthRelSun.getPosition();
        final Vector3D vEs = earthRelSun.getVelocity();

        final String csv = String.format(Locale.ROOT,
                "mu_earth,%.17e%nmu_sun,%.17e%n"
                        + "sat_gcrf,%.9f,%.9f,%.9f%nvel_gcrf,%.9f,%.9f,%.9f%n"
                        + "sun_gcrf,%.9f,%.9f,%.9f%nsunvel_gcrf,%.9f,%.9f,%.9f%n"
                        + "earth_rel_sun,%.9f,%.9f,%.9f%nearthvel_rel_sun,%.9f,%.9f,%.9f%n"
                        + "a_schwarzschild,%.17e,%.17e,%.17e%n"
                        + "a_lense_thirring,%.17e,%.17e,%.17e%n"
                        + "a_de_sitter,%.17e,%.17e,%.17e%n",
                mu, sun.getGM(),
                rGcrf.getX(), rGcrf.getY(), rGcrf.getZ(),
                vGcrf.getX(), vGcrf.getY(), vGcrf.getZ(),
                rSun.getX(), rSun.getY(), rSun.getZ(),
                vSun.getX(), vSun.getY(), vSun.getZ(),
                rEs.getX(), rEs.getY(), rEs.getZ(),
                vEs.getX(), vEs.getY(), vEs.getZ(),
                aS.getX(), aS.getY(), aS.getZ(),
                aL.getX(), aL.getY(), aL.getZ(),
                aD.getX(), aD.getY(), aD.getZ());
        Files.writeString(out.toPath(), csv);
    }

    /**
     * Emit box-wing (BoxAndSolarArraySpacecraft) drag + SRP accelerations for the spacelab port. A
     * convex box (buildBox) and a box + Sun-tracking solar array (buildPanels, PointingPanel) are
     * evaluated at a fixed attitude (FrameAlignedProvider) over several (relative-velocity, flux)
     * geometries, calling dragAcceleration(state, density, relativeVelocity, params) and
     * radiationPressureAcceleration(state, flux, params) directly. Per case we emit the attitude
     * matrix M (inertial->body, columns = applyTo(basis) so M*v == rotation.applyTo(v) regardless of
     * the quaternion convention), the inputs and Orekit's inertial-frame outputs. A lift-enabled box
     * (liftRatio 0.25) exercises the drag lift term; the box+array cases exercise the Sun-tracking
     * normal and the double-sided panel.
     */
    private static void writeBoxWingValidation(File out, Frame gcrf, AbsoluteDate epoch, double mu)
            throws IOException {

        // Box geometry + coefficients (diffuse reflection = 1 - absorption - reflection = 0.3).
        final double xLen = 1.5, yLen = 2.0, zLen = 3.0;
        final double mass = 1000.0;
        final double cd = 2.2;
        final double absorption = 0.3;
        final double reflection = 0.4;
        final double lift0 = 0.0;
        final double liftL = 0.25;
        final double arrayArea = 20.0;
        final Vector3D arrayAxis = new Vector3D(0.0, 1.0, 0.0);
        final double pRef = 4.56e-6, dRef = 149597870000.0;
        final double kRef = pRef * dRef * dRef;
        final double[] params = new double[] { 1.0 };
        final double rho = 2.0e-12;

        final CelestialBody sunBody = CelestialBodyFactory.getSun();

        // Spacecraft models: bare box, lift-enabled box, box + Sun-tracking solar array.
        final BoxAndSolarArraySpacecraft boxOnly =
                new BoxAndSolarArraySpacecraft(BoxAndSolarArraySpacecraft.buildBox(
                        xLen, yLen, zLen, cd, lift0, absorption, reflection));
        final BoxAndSolarArraySpacecraft boxLift =
                new BoxAndSolarArraySpacecraft(BoxAndSolarArraySpacecraft.buildBox(
                        xLen, yLen, zLen, cd, liftL, absorption, reflection));
        final BoxAndSolarArraySpacecraft boxArray =
                new BoxAndSolarArraySpacecraft(xLen, yLen, zLen, sunBody,
                        arrayArea, arrayAxis, cd, lift0, absorption, reflection);

        // A LEO orbit for the state (the satellite position drives the Sun-tracking array normal).
        final KeplerianOrbit kep = new KeplerianOrbit(
                7.0e6, 0.01, FastMath.toRadians(45.0),
                FastMath.toRadians(20.0), FastMath.toRadians(30.0), FastMath.toRadians(10.0),
                PositionAngleType.MEAN, gcrf, epoch, mu);
        final Orbit orbit = new CartesianOrbit(kep.getPVCoordinates(), gcrf, epoch, mu);
        final Vector3D satPos = orbit.getPVCoordinates().getPosition();
        final Vector3D sunPos = sunBody.getPosition(epoch, gcrf);

        // Two fixed attitudes: identity and a tilt about (1,2,3) by 0.7 rad.
        final Rotation rotId = Rotation.IDENTITY;
        final Rotation rotTilt = new Rotation(new Vector3D(1.0, 2.0, 3.0).normalize(), 0.7,
                RotationConvention.VECTOR_OPERATOR);

        // Box-only geometries (relative velocity + flux chosen freely, independent of the Sun).
        final Vector3D relA = new Vector3D(7400.0, 900.0, -1200.0);
        final Vector3D relB = new Vector3D(-2600.0, 6100.0, 3300.0);
        final double fluxMag = kRef / (dRef * dRef); // ~ pRef
        final Vector3D fluxA = new Vector3D(fluxMag, new Vector3D(0.6, -0.7, 0.4).normalize());
        final Vector3D fluxB = new Vector3D(fluxMag, new Vector3D(-0.3, 0.2, -0.93).normalize());

        // Box+array geometry: flux and Sun direction consistent with the real Sun and orbit position.
        final Vector3D sunSat = satPos.subtract(sunPos); // sat - sun, i.e. the Sun->sat flux direction
        final double sunSatNorm = sunSat.getNorm();
        final Vector3D fluxSun = new Vector3D(kRef / (sunSatNorm * sunSatNorm * sunSatNorm), sunSat);
        final Vector3D sunDir = sunPos.subtract(satPos).normalize(); // sat -> Sun unit
        final Vector3D dummyDir = new Vector3D(1.0, 0.0, 0.0);        // unused by fixed facets

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "xlen,%.17e%nylen,%.17e%nzlen,%.17e%nmass,%.17e%ncd,%.17e%n"
                        + "absorption,%.17e%nreflection,%.17e%nlift0,%.17e%nliftL,%.17e%n"
                        + "array_area,%.17e%narray_axis,%.17e,%.17e,%.17e%nk_ref,%.17e%nrho,%.17e%n",
                xLen, yLen, zLen, mass, cd, absorption, reflection, lift0, liftL,
                arrayArea, arrayAxis.getX(), arrayAxis.getY(), arrayAxis.getZ(), kRef, rho));

        emitBoxWingCase(csv, "box_id", boxOnly, orbit, rotId, mass, gcrf, epoch, rho, relA, fluxA, dummyDir, params);
        emitBoxWingCase(csv, "box_tilt", boxOnly, orbit, rotTilt, mass, gcrf, epoch, rho, relB, fluxB, dummyDir, params);
        emitBoxWingCase(csv, "lift_tilt", boxLift, orbit, rotTilt, mass, gcrf, epoch, rho, relA, fluxA, dummyDir, params);
        emitBoxWingCase(csv, "arr_id", boxArray, orbit, rotId, mass, gcrf, epoch, rho, relA, fluxSun, sunDir, params);
        emitBoxWingCase(csv, "arr_tilt", boxArray, orbit, rotTilt, mass, gcrf, epoch, rho, relB, fluxSun, sunDir, params);

        Files.writeString(out.toPath(), csv.toString());
    }

    /** One box-wing case: build the fixed-attitude state, call Orekit, emit inputs + inertial outputs. */
    private static void emitBoxWingCase(StringBuilder csv, String label,
                                        BoxAndSolarArraySpacecraft sc, Orbit orbit, Rotation rot,
                                        double mass, Frame gcrf, AbsoluteDate epoch, double rho,
                                        Vector3D relVel, Vector3D flux, Vector3D sunDir, double[] params) {
        final AttitudeProvider ap = new FrameAlignedProvider(rot);
        final Attitude att = ap.getAttitude(orbit, epoch, gcrf);
        final SpacecraftState state = new SpacecraftState(orbit, att).withMass(mass);

        // M (inertial->body): columns are the images of the body-axis basis under applyTo, so M*v
        // equals rotation.applyTo(v) exactly. Rust applies M inbound and M^T outbound.
        final Rotation r = state.getAttitude().getRotation();
        final Vector3D c0 = r.applyTo(Vector3D.PLUS_I);
        final Vector3D c1 = r.applyTo(Vector3D.PLUS_J);
        final Vector3D c2 = r.applyTo(Vector3D.PLUS_K);

        final Vector3D drag = sc.dragAcceleration(state, rho, relVel, params);
        final Vector3D srp = sc.radiationPressureAcceleration(state, flux, params);

        csv.append(String.format(Locale.ROOT,
                "%s_att,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n"
                        + "%s_relvel,%.17e,%.17e,%.17e%n%s_flux,%.17e,%.17e,%.17e%n"
                        + "%s_sundir,%.17e,%.17e,%.17e%n"
                        + "%s_drag,%.17e,%.17e,%.17e%n%s_srp,%.17e,%.17e,%.17e%n",
                label, c0.getX(), c1.getX(), c2.getX(), c0.getY(), c1.getY(), c2.getY(),
                c0.getZ(), c1.getZ(), c2.getZ(),
                label, relVel.getX(), relVel.getY(), relVel.getZ(),
                label, flux.getX(), flux.getY(), flux.getZ(),
                label, sunDir.getX(), sunDir.getY(), sunDir.getZ(),
                label, drag.getX(), drag.getY(), drag.getZ(),
                label, srp.getX(), srp.getY(), srp.getZ()));
    }

    /**
     * Emit Orekit's {@link KnockeRediffusedForceModel} acceleration (Earth albedo + thermal-IR
     * radiation pressure) at several LEO geometries, for validating the spacelab port. A cannonball
     * {@link IsotropicRadiationSingleCoefficient} spacecraft (so the response is isotropic and the
     * comparison isolates the Knocke rediffused-flux grid) over a spherical Earth of radius {@code ae}
     * at a fixed angular resolution. Per case we emit the satellite position (GCRF), the Sun position
     * (GCRF at that date), the seconds since Knocke's reference epoch (1981-12-22, which drives the
     * seasonal term), and the resulting inertial acceleration.
     */
    private static void writeEarthRadiationValidation(File out, Frame gcrf, AbsoluteDate epoch,
                                                      double mu, double ae) throws IOException {

        final double cr = 1.5;
        final double area = 10.0;
        final double mass = 1000.0;
        final double angularResolution = 0.1; // rad

        final TimeScale utc = TimeScalesFactory.getUTC();
        final AbsoluteDate knockeRef = new AbsoluteDate(1981, 12, 22, 0, 0, 0.0, utc);
        final CelestialBody sunBody = CelestialBodyFactory.getSun();

        final IsotropicRadiationSingleCoefficient spacecraft =
                new IsotropicRadiationSingleCoefficient(area, cr);
        final KnockeRediffusedForceModel knocke =
                new KnockeRediffusedForceModel(sunBody, spacecraft, ae, angularResolution);

        // Cases: (satellite position in GCRF, days past the base epoch). The epoch offset both moves
        // the Sun and advances the annual albedo/emissivity term. Positions are LEO points chosen so
        // no crown offset sits on the visible-horizon edge.
        final double[][] satPositions = {
            { 6.928137e6, 5.0e5, 8.0e5 },
            { -2.6e6, 6.1e6, 3.3e6 },
            { 1.5e6, -3.0e6, 6.4e6 },
            { 4.5e6, 4.5e6, -3.2e6 },
        };
        final double[] offsetDays = { 0.0, 100.0, 250.0, 30.0 };
        final String[] labels = { "leo_a", "leo_b", "leo_c", "leo_d" };

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT,
                "cr,%.17e%narea,%.17e%nmass,%.17e%nae,%.17e%nangular_resolution,%.17e%n",
                cr, area, mass, ae, angularResolution));

        for (int i = 0; i < satPositions.length; i++) {
            final AbsoluteDate date = epoch.shiftedBy(offsetDays[i] * 86400.0);
            final Vector3D satPos = new Vector3D(satPositions[i]);
            // A plausible velocity (perpendicular-ish to r); Knocke ignores it but SpacecraftState
            // needs a valid orbit.
            final Vector3D vel = new Vector3D(-satPos.getY(), satPos.getX(), 0.0)
                    .normalize().scalarMultiply(7500.0);
            final Orbit orbit = new CartesianOrbit(new PVCoordinates(satPos, vel), gcrf, date, mu);
            final SpacecraftState state = new SpacecraftState(orbit).withMass(mass);

            final double[] params = knocke.getParameters(date);
            final Vector3D acc = knocke.acceleration(state, params);
            final Vector3D sunPos = sunBody.getPosition(date, gcrf);
            final double deltaT = date.durationFrom(knockeRef);

            csv.append(String.format(Locale.ROOT,
                    "%s_sat,%.17e,%.17e,%.17e%n%s_sun,%.17e,%.17e,%.17e%n"
                            + "%s_offset,%.17e%n%s_acc,%.17e,%.17e,%.17e%n",
                    labels[i], satPos.getX(), satPos.getY(), satPos.getZ(),
                    labels[i], sunPos.getX(), sunPos.getY(), sunPos.getZ(),
                    labels[i], deltaT,
                    labels[i], acc.getX(), acc.getY(), acc.getZ()));
        }

        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Emit Modified Harris-Priester densities for validating spacelab's port.
     *
     * <p>Uses the explicit-vector entry point {@link HarrisPriester#getDensity(Vector3D, Vector3D)},
     * so the comparison is the {@literal Montenbruck & Gill} density table and the diurnal-bulge
     * cosine law alone — no frames, ephemeris or date confound it. The Sun direction is fixed
     * (right ascension 20°, declination −15°, at one astronomical unit) and the satellite is sampled
     * over the model's 100–1000 km range, across every altitude-bracket regime, the cosine exponents
     * {@code n ∈ {2, 4, 6}} and bulge angles from the apex to its exact antipode (where the cosine
     * power vanishes and only the minimum density survives). The Sun ephemeris handed to the
     * constructor is never consulted on this path; the WGS84 ellipsoid supplies only its equatorial
     * radius and flattening to the height reduction.
     */
    public static void writeHarrisPriesterValidation(File out) throws IOException {
        final Frame gcrf = FramesFactory.getGCRF();
        final OneAxisEllipsoid earth = new OneAxisEllipsoid(6378137.0, 1.0 / 298.257223563, gcrf);
        final ExtendedPositionProvider sun = CelestialBodyFactory.getSun();

        // Sun direction: right ascension 20°, declination −15°, at one astronomical unit.
        final double au = 1.495978707e11;
        final double sra = FastMath.toRadians(20.0);
        final double sdec = FastMath.toRadians(-15.0);
        final Vector3D sunVec = new Vector3D(au * FastMath.cos(sdec) * FastMath.cos(sra),
                                             au * FastMath.cos(sdec) * FastMath.sin(sra),
                                             au * FastMath.sin(sdec));

        final double a = 6378137.0;
        // {altitude_m, latitude_deg, longitude_deg, cosine exponent n}. The diurnal-bulge apex sits at
        // RA 50°, Dec −15°; the (alt, lat 15°, lon 230°) sample is its exact antipode (night side).
        final double[][] samples = {
            {110000.0,   0.0,  50.0, 4.0},
            {145000.0,  10.0,  50.0, 4.0},
            {200000.0, -15.0,  50.0, 2.0},
            {250000.0,   0.0,  20.0, 4.0},
            {300000.0,  30.0, 140.0, 6.0},
            {350000.0,   0.0,  50.0, 4.0},
            {400000.0, -15.0,  50.0, 4.0},
            {450000.0,  45.0, 230.0, 6.0},
            {500000.0,   0.0, 110.0, 2.0},
            {600000.0, -30.0,  50.0, 4.0},
            {700000.0,  15.0, 230.0, 4.0},
            {800000.0,  20.0,  50.0, 4.0},
            {850000.0,   0.0,  60.0, 6.0},
            {900000.0, -15.0,  50.0, 2.0},
            {950000.0,  10.0, 300.0, 4.0},
            {990000.0,   0.0,  50.0, 4.0},
        };

        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT, "# sun=%.17e,%.17e,%.17e%n",
                sunVec.getX(), sunVec.getY(), sunVec.getZ()));
        csv.append("posx,posy,posz,n,rho\n");
        for (final double[] s : samples) {
            final double lat = FastMath.toRadians(s[1]);
            final double lon = FastMath.toRadians(s[2]);
            final double n = s[3];
            final double r = a + s[0];
            final Vector3D pos = new Vector3D(r * FastMath.cos(lat) * FastMath.cos(lon),
                                              r * FastMath.cos(lat) * FastMath.sin(lon),
                                              r * FastMath.sin(lat));
            final double rho = new HarrisPriester(sun, earth, n).getDensity(sunVec, pos);
            csv.append(String.format(Locale.ROOT, "%.17e,%.17e,%.17e,%.1f,%.17e%n",
                    pos.getX(), pos.getY(), pos.getZ(), n, rho));
        }
        Files.writeString(out.toPath(), csv.toString());
    }

    /**
     * Emit maneuver and empirical-acceleration validation data for the spacelab {@code maneuver} port:
     *
     * <ol>
     *   <li><b>Constant thrust</b> — a LEO arc under point-mass gravity plus an always-firing
     *       {@link ConstantThrustManeuver}, integrated with a <em>fixed-step</em>
     *       {@link ClassicalRungeKuttaIntegrator} so the spacelab RK4 can reproduce the same nodes;
     *       initial and final [r, v, m] are emitted, with mass depleting at the rocket-equation flow
     *       rate. The initial-state thrust acceleration (F/m·d̂) is emitted too as a direct formula
     *       check. The attitude is frame-aligned identity, so the satellite-frame direction is inertial.</li>
     *   <li><b>Impulse</b> — an {@link ImpulseManeuver} at a date trigger, inertial and RTN (QSW), each
     *       compared as pre-impulse (r, v) plus Δv against the post-impulse velocity. A two-body
     *       propagator with the impulse event gives the post state; the same propagator without it gives
     *       the pre state (position is continuous across an impulse). A Java-side self-check prints the
     *       residual v_post − v_pre − R·Δv.</li>
     *   <li><b>RTN frame</b> — the {@link LOFType#QSW} basis (Q, S, W axes in inertial) and the argument
     *       of latitude ({@link CircularOrbit#getAlphaV}) at a fixed state, validating spacelab's RTN
     *       basis and argument-of-latitude geometry against Orekit.</li>
     *   <li><b>Polynomial empirical</b> — a {@link ParametricAcceleration} with an inertial direction and
     *       a {@link PolynomialAccelerationModel}, evaluated at several dates; acceleration = direction ·
     *       Σ cₖ(t−t_ref)ᵏ.</li>
     * </ol>
     */
    private static void writeManeuverValidation(File out, Frame gcrf, AbsoluteDate epoch, double mu)
            throws IOException {
        final double g0 = Constants.G0_STANDARD_GRAVITY;
        final StringBuilder csv = new StringBuilder();
        csv.append(String.format(Locale.ROOT, "mu,%.17e%ng0,%.17e%n", mu, g0));

        // ---- 1. Constant-thrust short arc, matched fixed-step RK4 -------------------------------------
        final double thrust = 400.0;                    // N
        final double isp = 300.0;                        // s
        final double flowRate = thrust / (isp * g0);     // kg/s (positive)
        final Vector3D ctDir = new Vector3D(0.6, 0.5, -0.624_5).normalize(); // inertial (identity attitude)
        final double mass0 = 1200.0;                     // kg
        final double span = 600.0;                       // s
        final int nsteps = 20;                           // fixed steps => step = 30 s
        final double step = span / nsteps;

        final KeplerianOrbit ctKep = new KeplerianOrbit(
                7.0e6, 0.001, FastMath.toRadians(51.6),
                FastMath.toRadians(40.0), FastMath.toRadians(30.0), FastMath.toRadians(10.0),
                PositionAngleType.MEAN, gcrf, epoch, mu);
        final Orbit ctOrbit = new CartesianOrbit(ctKep.getPVCoordinates(), gcrf, epoch, mu);
        final Vector3D r0 = ctOrbit.getPVCoordinates().getPosition();
        final Vector3D v0 = ctOrbit.getPVCoordinates().getVelocity();

        // Fire the whole arc: start before epoch and end after span, so no trigger lands inside [0, span]
        // and the fixed-step RK4 node sequence is uncontaminated by event restarts.
        final ConstantThrustManeuver maneuver = new ConstantThrustManeuver(
                epoch.shiftedBy(-100.0), span + 200.0, thrust, isp, ctDir);

        final ClassicalRungeKuttaIntegrator rk4 = new ClassicalRungeKuttaIntegrator(step);
        final NumericalPropagator ctProp = new NumericalPropagator(rk4);
        ctProp.setOrbitType(OrbitType.CARTESIAN);
        // Align the identity attitude to GCRF (the propagation frame), not the default EME2000 —
        // otherwise the ~8e-8 rad GCRF↔EME2000 frame bias rotates the "inertial" thrust direction.
        ctProp.setAttitudeProvider(new FrameAlignedProvider(Rotation.IDENTITY, gcrf));
        ctProp.setInitialState(new SpacecraftState(ctOrbit).withMass(mass0));
        ctProp.addForceModel(new NewtonianAttraction(mu));
        ctProp.addForceModel(maneuver);
        final SpacecraftState ctFinal = ctProp.propagate(epoch.shiftedBy(span));
        final Vector3D rf = ctFinal.getPVCoordinates(gcrf).getPosition();
        final Vector3D vf = ctFinal.getPVCoordinates(gcrf).getVelocity();
        final double mf = ctFinal.getMass();

        // Direct acceleration formula check at the initial state (firing): a = (F/m0)·d̂ (inertial).
        final SpacecraftState ctState0 = new SpacecraftState(ctOrbit,
                new FrameAlignedProvider(Rotation.IDENTITY, gcrf).getAttitude(ctOrbit, epoch, gcrf)).withMass(mass0);
        final Vector3D ctAccel0 = maneuver.acceleration(ctState0, maneuver.getParameters(epoch));

        csv.append(String.format(Locale.ROOT,
                "ct_thrust,%.17e%nct_isp,%.17e%nct_flow_rate,%.17e%n"
                        + "ct_dir,%.17e,%.17e,%.17e%nct_mass0,%.17e%nct_span,%.17e%nct_nsteps,%d%n"
                        + "ct_r0,%.17e,%.17e,%.17e%nct_v0,%.17e,%.17e,%.17e%n"
                        + "ct_rf,%.17e,%.17e,%.17e%nct_vf,%.17e,%.17e,%.17e%nct_mf,%.17e%n"
                        + "ct_accel0,%.17e,%.17e,%.17e%n",
                thrust, isp, flowRate,
                ctDir.getX(), ctDir.getY(), ctDir.getZ(), mass0, span, nsteps,
                r0.getX(), r0.getY(), r0.getZ(), v0.getX(), v0.getY(), v0.getZ(),
                rf.getX(), rf.getY(), rf.getZ(), vf.getX(), vf.getY(), vf.getZ(), mf,
                ctAccel0.getX(), ctAccel0.getY(), ctAccel0.getZ()));

        // ---- 2. Impulsive maneuver, inertial and RTN (QSW) -------------------------------------------
        final Vector3D rImp = new Vector3D(6.9e6, 8.0e5, 1.3e6);
        final Vector3D vImp = new Vector3D(-1.0e3, 6.7e3, 3.6e3);
        final AbsoluteDate trig = epoch.shiftedBy(300.0);

        emitImpulseCase(csv, "imp_in", rImp, vImp, gcrf, epoch, trig, mu,
                new Vector3D(50.0, -30.0, 20.0), new FrameAlignedProvider(Rotation.IDENTITY, gcrf));
        emitImpulseCase(csv, "imp_rtn", rImp, vImp, gcrf, epoch, trig, mu,
                new Vector3D(15.0, 40.0, -10.0), new LofOffset(gcrf, LOFType.QSW));

        // ---- 3. RTN (QSW) basis and argument of latitude at a fixed state ----------------------------
        final Vector3D rFrame = new Vector3D(6.8e6, 9.0e5, 2.1e6);
        final Vector3D vFrame = new Vector3D(-1.2e3, 6.4e3, 3.2e3);
        final PVCoordinates pvFrame = new PVCoordinates(rFrame, vFrame);
        final Rotation qsw = LOFType.QSW.rotationFromInertial(pvFrame);
        final Vector3D axisR = qsw.applyInverseTo(Vector3D.PLUS_I); // Q = radial, in inertial
        final Vector3D axisT = qsw.applyInverseTo(Vector3D.PLUS_J); // S = transverse
        final Vector3D axisW = qsw.applyInverseTo(Vector3D.PLUS_K); // W = normal
        final double alphaV = new CircularOrbit(
                new CartesianOrbit(pvFrame, gcrf, epoch, mu)).getAlphaV();
        csv.append(String.format(Locale.ROOT,
                "rtn_r,%.17e,%.17e,%.17e%nrtn_v,%.17e,%.17e,%.17e%n"
                        + "rtn_axis_r,%.17e,%.17e,%.17e%nrtn_axis_t,%.17e,%.17e,%.17e%n"
                        + "rtn_axis_w,%.17e,%.17e,%.17e%nrtn_alpha_v,%.17e%n",
                rFrame.getX(), rFrame.getY(), rFrame.getZ(),
                vFrame.getX(), vFrame.getY(), vFrame.getZ(),
                axisR.getX(), axisR.getY(), axisR.getZ(),
                axisT.getX(), axisT.getY(), axisT.getZ(),
                axisW.getX(), axisW.getY(), axisW.getZ(), alphaV));

        // ---- 4. Polynomial empirical acceleration (inertial direction) -------------------------------
        final Vector3D polyDir = new Vector3D(0.3, -0.5, 0.8).normalize();
        final double polyTref = 50.0; // seconds from epoch
        final double[] polyCoeffs = { 2.0e-7, 3.0e-9, -1.0e-11 }; // c0 + c1·τ + c2·τ²
        final PolynomialAccelerationModel polyModel = new PolynomialAccelerationModel(
                "emp", epoch.shiftedBy(polyTref), polyCoeffs.length - 1);
        final ParametricAcceleration parametric = new ParametricAcceleration(polyDir, true, polyModel);
        csv.append(String.format(Locale.ROOT,
                "poly_dir,%.17e,%.17e,%.17e%npoly_tref,%.17e%npoly_coeffs,%.17e,%.17e,%.17e%n",
                polyDir.getX(), polyDir.getY(), polyDir.getZ(), polyTref,
                polyCoeffs[0], polyCoeffs[1], polyCoeffs[2]));
        final double[] polyTimes = { 0.0, 100.0, 250.0, 500.0 };
        for (final double tOff : polyTimes) {
            final AbsoluteDate d = epoch.shiftedBy(tOff);
            final SpacecraftState st = new SpacecraftState(
                    new CartesianOrbit(ctOrbit.shiftedBy(tOff).getPVCoordinates(), gcrf, d, mu));
            final Vector3D a = parametric.acceleration(st, polyCoeffs);
            csv.append(String.format(Locale.ROOT, "poly_case,%.17e,%.17e,%.17e,%.17e%n",
                    tOff, a.getX(), a.getY(), a.getZ()));
        }

        Files.writeString(out.toPath(), csv.toString());
    }

    /** One impulse case: propagate two-body to the trigger with and without the impulse, emit pre (r, v),
     * Δv (in the attitude frame), and post velocity, plus a self-check residual. */
    private static void emitImpulseCase(StringBuilder csv, String label, Vector3D r, Vector3D v,
                                        Frame gcrf, AbsoluteDate epoch, AbsoluteDate trig, double mu,
                                        Vector3D deltaV, AttitudeProvider attitude) {
        final Orbit orbit = new CartesianOrbit(new PVCoordinates(r, v), gcrf, epoch, mu);

        final SpacecraftState pre = twoBody(orbit, mu).propagate(trig);
        final NumericalPropagator withImpulse = twoBody(orbit, mu);
        withImpulse.addEventDetector(new ImpulseManeuver(new DateDetector(trig), attitude, deltaV, 300.0));
        final SpacecraftState post = withImpulse.propagate(trig);

        final Vector3D rPre = pre.getPVCoordinates(gcrf).getPosition();
        final Vector3D vPre = pre.getPVCoordinates(gcrf).getVelocity();
        final Vector3D vPost = post.getPVCoordinates(gcrf).getVelocity();

        // Self-check: rotate Δv (attitude/LOF frame) to inertial and compare v_post − v_pre.
        final Rotation rot = attitude.getAttitude(orbit, trig, gcrf).getRotation();
        final Vector3D dvInertial = rot.applyInverseTo(deltaV);
        final Vector3D residual = vPost.subtract(vPre).subtract(dvInertial);
        System.out.println(label + " impulse self-check residual (m/s): " + residual.getNorm());

        csv.append(String.format(Locale.ROOT,
                "%s_r,%.17e,%.17e,%.17e%n%s_vpre,%.17e,%.17e,%.17e%n"
                        + "%s_dv,%.17e,%.17e,%.17e%n%s_vpost,%.17e,%.17e,%.17e%n",
                label, rPre.getX(), rPre.getY(), rPre.getZ(),
                label, vPre.getX(), vPre.getY(), vPre.getZ(),
                label, deltaV.getX(), deltaV.getY(), deltaV.getZ(),
                label, vPost.getX(), vPost.getY(), vPost.getZ()));
    }

    /** A two-body numerical propagator (adaptive DP853, point-mass gravity) from an initial orbit. */
    private static NumericalPropagator twoBody(Orbit orbit, double mu) {
        final double[][] tol = NumericalPropagator.tolerances(1.0e-11, orbit, OrbitType.CARTESIAN);
        final DormandPrince853Integrator integrator =
                new DormandPrince853Integrator(1.0e-6, 300.0, tol[0], tol[1]);
        final NumericalPropagator propagator = new NumericalPropagator(integrator);
        propagator.setOrbitType(OrbitType.CARTESIAN);
        propagator.setInitialState(new SpacecraftState(orbit));
        propagator.addForceModel(new NewtonianAttraction(mu));
        return propagator;
    }
}
