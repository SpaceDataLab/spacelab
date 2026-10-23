import java.io.PrintWriter;
import java.util.Map;
import org.orekit.propagation.semianalytical.dsst.utilities.CoefficientsFactory;
import org.orekit.propagation.semianalytical.dsst.utilities.CoefficientsFactory.NSKey;
import org.orekit.propagation.semianalytical.dsst.utilities.JacobiPolynomials;
import org.orekit.propagation.semianalytical.dsst.utilities.UpperBounds;

/**
 * Golden values for the DSST scalar coefficients and the two Danielson upper bounds.
 *
 * These are the functions `spacelab`'s `dsst::coefficients` ports, and until now nothing compared
 * them against a number: only the propagated orbit was checked, and the eccentricity series has
 * converged well enough by then that a wrong bound changes where it truncates without moving the
 * result above the end-to-end tolerance. A bound that is too small truncates too early and costs
 * accuracy silently, which is the failure this file exists to make visible.
 *
 * Every method used here is `public static` in Orekit 13.1.7 -- no reflection, no forked source.
 */
public final class DsstCoefficientsOracle {

    /** chi = 1/B = 1/sqrt(1 - e^2), so this spans e = 0 to about 0.9. */
    private static final double[] CHI = {1.0, 1.005, 1.05, 1.2, 1.6, 2.3};

    /**
     * gamma is the equinoctial direction cosine, in [-1, 1]. The values next to the ends are there
     * because `getRnml` floors sin(i) at 0.03, and only a gamma inside that floor exercises it.
     */
    private static final double[] GAMMA = {-1.0, -0.9995, -0.8, -0.4, 0.0, 0.4, 0.8, 0.9995, 1.0};

    private static final int[] DEGREE = {2, 3, 4, 6, 9, 12};

    public static void main(String[] args) throws Exception {
        String out = args.length > 0 ? args[0] : "orekit_dsst_coefficients.csv";
        try (PrintWriter w = new PrintWriter(out, "UTF-8")) {
            w.println("# Orekit 13.1.7 DSST scalar coefficients and Danielson upper bounds.");
            w.println("# kind,a,b,c,d,e,value  -- unused columns are 0. Doubles are %.17e.");
            dnl(w);
            rnml(w);
            vns(w);
            qns(w);
            gsHs(w);
            jacobi(w);
        }
        System.out.println("wrote " + out);
    }

    /** Dnl(chi^2, chi*(chi^2/2)^l, n, l): the three branches are n < l+2, n == l+2, n > l+2. */
    private static void dnl(PrintWriter w) {
        for (double chi : CHI) {
            double xx = chi * chi;
            for (int n : DEGREE) {
                for (int l = 0; l <= n; l++) {
                    double xpl = chi * Math.pow(xx / 2.0, l);
                    row(w, "dnl", chi, n, l, 0, 0, UpperBounds.getDnl(xx, xpl, n, l));
                }
            }
        }
    }

    /**
     * Rnml(gamma, n, l, m, eps, I). The caller swaps l and m between its two branches, so both
     * orders are emitted. The retrograde factor is +1 everywhere in this port, and passing -1 here
     * would pin code no caller can reach.
     */
    private static void rnml(PrintWriter w) {
        for (double gamma : GAMMA) {
            for (int n : DEGREE) {
                for (int l = 0; l <= n; l++) {
                    for (int m = 0; m <= Math.min(n, 3); m++) {
                        for (int eps = -1; eps <= 1; eps += 2) {
                            row(w, "rnml", gamma, n, l, m, eps,
                                UpperBounds.getRnml(gamma, n, l, m, eps, 1));
                        }
                    }
                }
            }
        }
    }

    /** Vns, keyed (n, s); Orekit returns a sorted map, we flatten it. */
    private static void vns(PrintWriter w) {
        int order = 14;
        Map<NSKey, Double> map = CoefficientsFactory.computeVns(order);
        for (Map.Entry<NSKey, Double> e : map.entrySet()) {
            row(w, "vns", 0.0, e.getKey().getN(), e.getKey().getS(), 0, 0, e.getValue());
        }
    }

    /**
     * P_l^{v,w}(x) and its derivative, the shared kernel of the tesseral and third-body
     * short-period contributions. `x` is gamma, which is where the DSST evaluates it.
     */
    private static void jacobi(PrintWriter w) {
        for (double x : GAMMA) {
            for (int l = 0; l <= 6; l++) {
                for (int v = 0; v <= 4; v++) {
                    for (int wq = 0; wq <= 4; wq++) {
                        double[] vd = JacobiPolynomials.getValueAndDerivative(l, v, wq, x);
                        row(w, "jacobi_value", x, l, v, wq, 0, vd[0]);
                        row(w, "jacobi_deriv", x, l, v, wq, 0, vd[1]);
                    }
                }
            }
        }
    }

    /** Qns(gamma, n, m): the Jacobi-polynomial table, dense in n and m. */
    private static void qns(PrintWriter w) {
        int n = 12;
        int m = 12;
        for (double gamma : GAMMA) {
            double[][] q = CoefficientsFactory.computeQns(gamma, n, m);
            for (int i = 0; i <= n; i++) {
                for (int j = 0; j < q[i].length; j++) {
                    row(w, "qns", gamma, i, j, 0, 0, q[i][j]);
                }
            }
        }
    }

    /** Gs and Hs from (k, h, alpha, beta), the equinoctial eccentricity and direction pairs. */
    private static void gsHs(PrintWriter w) {
        double[][] kh = {{0.0, 0.0}, {0.05, -0.02}, {-0.3, 0.4}, {0.6, 0.6}};
        double[][] ab = {{1.0, 0.0}, {0.6, -0.8}, {-0.28, 0.96}, {0.0, 1.0}};
        int order = 12;
        for (double[] p : kh) {
            for (double[] q : ab) {
                double[][] gh = CoefficientsFactory.computeGsHs(p[0], p[1], q[0], q[1], order);
                for (int s = 0; s <= order; s++) {
                    // The four inputs are packed into the key columns so a row is self-describing.
                    row(w, "gs", p[0], s, idx(kh, p), idx(ab, q), 0, gh[0][s]);
                    row(w, "hs", p[1], s, idx(kh, p), idx(ab, q), 0, gh[1][s]);
                }
            }
        }
    }

    private static int idx(double[][] table, double[] item) {
        for (int i = 0; i < table.length; i++) {
            if (table[i] == item) {
                return i;
            }
        }
        throw new IllegalStateException("row not in table");
    }

    private static void row(PrintWriter w, String kind, double a, int b, int c, int d, int e,
                            double value) {
        w.printf("%s,%.17e,%d,%d,%d,%d,%.17e%n", kind, a, b, c, d, e, value);
    }

    private DsstCoefficientsOracle() {
    }
}
