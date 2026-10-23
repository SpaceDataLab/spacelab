/* Black-box oracle harness for IAU SOFA.
 *
 * This file is ours. It links against SOFA and calls it; it contains no SOFA
 * code and reproduces none. Executing a program and recording its output does
 * not create a derivative work under SOFA's licence, ERFA's, or anyone else's.
 *
 * The numbers it emits are facts. They are committed as fixtures; SOFA itself
 * never enters the build, never enters CI, and is never linked into a crate.
 *
 * Build (SOFA is not vendored; point at your own copy):
 *
 *   SOFA=/path/to/sofa/20231011/c
 *   gcc -O2 -I "$SOFA/src" tools/sofa-oracle/harness.c \
 *       "$SOFA/src/gmst82.c" "$SOFA/src/anp.c" -lm -o /tmp/sofa-oracle
 *
 * Usage: one "days since J2000 (UT1)" per line on stdin; GMST in radians on stdout.
 */

#include <stdio.h>
#include "sofa.h"

int main(void) {
    double days;
    while (scanf("%lf", &days) == 1) {
        /* Two-part Julian Date: the large part exact, the small part carrying
         * the resolution. This is the calling convention SOFA requires. */
        printf("%.17g\n", iauGmst82(2451545.0, days));
    }
    return 0;
}
