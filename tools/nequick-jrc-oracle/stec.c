/* Slant TEC from the JRC NeQuickG reference implementation, at 17 significant
 * digits.
 *
 * Reads lines "a0 a1 a2 month ut_hours sta_lon sta_lat sta_h sat_lon sat_lat
 * sat_h" (degrees, metres) on stdin and prints one STEC in TECU per line.
 * Usage: stec <modip file> <ccir dir>.
 */
#include "NeQuickG_JRC.h"
#include <stdio.h>

int main(int argc, char **argv) {
  NeQuickG_handle h;
  if (argc < 3 || NeQuickG.init(argv[1], argv[2], &h) != NEQUICK_OK)
    return 1;
  double az[3], ut, slon, slat, sh, tlon, tlat, th;
  int month;
  while (scanf("%lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf", &az[0], &az[1],
               &az[2], &month, &ut, &slon, &slat, &sh, &tlon, &tlat,
               &th) == 11) {
    double tec;
    if (NeQuickG.set_solar_activity_coefficients(h, az, 3) != NEQUICK_OK)
      return 2;
    if (NeQuickG.set_time(h, (uint8_t)month, ut) != NEQUICK_OK)
      return 3;
    if (NeQuickG.set_receiver_position(h, slon, slat, sh) != NEQUICK_OK)
      return 4;
    if (NeQuickG.set_satellite_position(h, tlon, tlat, th) != NEQUICK_OK)
      return 5;
    if (NeQuickG.get_total_electron_content(h, &tec) != NEQUICK_OK)
      return 6;
    printf("%.17g\n", tec);
  }
  NeQuickG.close(h);
  return 0;
}
