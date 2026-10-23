// Standalone Orekit 13.1.7 oracle for the Brouwer-Lyddane port (crates/spacelab-elements).
// Emits osculating (r, v) in GCRF from BrouwerLyddanePropagator plus the converged mean elements
// from BrouwerLyddanePropagator.computeMeanOrbit; the goldens embedded in
// brouwer_lyddane.rs::tests are copied from this output. Build/run against the orekit-oracle jars:
//   javac -cp 'jars/*' BlOracle.java && java -cp 'jars/*:.' BlOracle
//
// Two orbits. The first is the near-circular LEO the port was originally validated on. The second
// is eccentric (e = 0.3) and sampled over eight weeks, because the long-period corrections are
// driven by the argument of perigee and scale with powers of e: on the near-circular case the
// third-harmonic (3-omega) terms move the state by under a micrometre over a day, which is below
// the port's own agreement floor with Orekit, and no assertion on that orbit can see them at all.
import java.io.File;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.analytical.BrouwerLyddanePropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;

public class BlOracle {

  static final double MU = 3.986004415e14;
  static final double RE = 6378136.3;
  static final double C20 = -1.08262668e-3;
  static final double C30 = 2.53265649e-6;
  static final double C40 = 1.61962159e-6;
  static final double C50 = 2.27296083e-7;

  // BrouwerLyddanePropagator.computeMeanOrbit defaults (EPSILON_DEFAULT, MAX_ITERATIONS_DEFAULT).
  static final double EPS = 1.0e-13;
  static final int MAX_IT = 200;

  public static void main(String[] args) throws Exception {
    File orekitData = new File(System.getProperty("orekit.data", "orekit-data-main"));
    if (orekitData.isDirectory()) {
      DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(orekitData));
    }

    Frame gcrf = FramesFactory.getGCRF();
    AbsoluteDate t0 = AbsoluteDate.J2000_EPOCH;

    Orbit leo = new CartesianOrbit(
        new PVCoordinates(
            new Vector3D(4485195.611065954, 3180300.412195459, 4537882.996315523),
            new Vector3D(-3045.222561050032, -3821.479786492346, 5708.385836216403)),
        gcrf, t0, MU);
    emit("LEO", leo, 10800.0, 9, 0.0);

    Orbit ecc = new KeplerianOrbit(
        7200000.0, 0.3, Math.toRadians(30.0),
        Math.toRadians(25.0), Math.toRadians(40.0), Math.toRadians(10.0),
        PositionAngleType.TRUE, gcrf, t0, MU);
    emit("ECC", ecc, 8.0 * 86400.0, 8, 0.0);
  }

  static void emit(String tag, Orbit osculating, double step, int samples, double m2) {
    KeplerianOrbit mean =
        BrouwerLyddanePropagator.computeMeanOrbit(
            osculating, RE, MU, C20, C30, C40, C50, m2, EPS, MAX_IT);
    System.out.printf(
        "%s MEAN %.17e %.17e %.17e %.17e %.17e %.17e%n",
        tag, mean.getA(), mean.getE(), mean.getI(),
        mean.getPerigeeArgument(), mean.getRightAscensionOfAscendingNode(),
        mean.getMeanAnomaly());

    BrouwerLyddanePropagator prop =
        new BrouwerLyddanePropagator(osculating, RE, MU, C20, C30, C40, C50, m2);
    for (int k = 0; k < samples; k++) {
      double dt = step * k;
      PVCoordinates pv = prop.getPVCoordinates(osculating.getDate().shiftedBy(dt), osculating.getFrame());
      System.out.printf(
          "%s %.1f %.17e %.17e %.17e %.17e %.17e %.17e%n",
          tag, dt,
          pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ(),
          pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ());
    }
  }
}
