// Standalone Orekit 13.1.7 oracle for the Eckstein-Hechler port (crates/spacelab-elements).
// Emits osculating (r, v) in GCRF from EcksteinHechlerPropagator; the golden embedded in
// eckstein_hechler.rs::tests is copied from this output. Uses the same denormalized XGM2019e
// C_n0 the Rust test uses. Build/run against the orekit-oracle jars + orekit-data, e.g.:
//   javac -cp 'jars/*' EhOracle.java && java -cp 'jars/*:.' EhOracle
// (the orekit-data path below is the local orekit-oracle working copy).
import org.hipparchus.util.FastMath;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.analytical.EcksteinHechlerPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;
import java.io.File;

public class EhOracle {
  public static void main(String[] args) throws Exception {
    File orekitData = new File("/tmp/orekit-bench/orekit-data-main");
    DataContext.getDefault().getDataProvidersManager().addProvider(new DirectoryCrawler(orekitData));

    double mu = 3.98600441500000000e14;
    double re = 6378136.3;
    // denormalized zonal coefficients = embedded XGM2019e Cbar_n0 * sqrt(2n+1)
    double c20 = -0.0004841694947475625 * FastMath.sqrt(5.0);
    double c30 =  9.571725961869197e-07 * FastMath.sqrt(7.0);
    double c40 =  5.399778363124431e-07 * FastMath.sqrt(9.0);
    double c50 =  6.863789456281855e-08 * FastMath.sqrt(11.0);
    double c60 = -1.499836903807493e-07 * FastMath.sqrt(13.0);

    Frame frame = FramesFactory.getGCRF();
    AbsoluteDate epoch = AbsoluteDate.J2000_EPOCH;

    double a = 7200000.0, e = 0.001, inc = FastMath.toRadians(98.0);
    double pa = FastMath.toRadians(40.0), raan = FastMath.toRadians(30.0), m = FastMath.toRadians(50.0);
    KeplerianOrbit orbit = new KeplerianOrbit(a, e, inc, pa, raan, m, PositionAngleType.MEAN, frame, epoch, mu);

    EcksteinHechlerPropagator prop = new EcksteinHechlerPropagator(orbit, re, mu, c20, c30, c40, c50, c60);

    System.out.printf("# c20..c60 = %.17e %.17e %.17e %.17e %.17e%n", c20, c30, c40, c50, c60);
    double[] times = {0, 900, 1800, 2700, 3600, 4500, 5400, 6300};
    for (double t : times) {
      PVCoordinates pv = prop.propagate(epoch.shiftedBy(t)).getPVCoordinates(frame);
      System.out.printf("%.1f,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e%n",
        t, pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ(),
        pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ());
    }
  }
}
