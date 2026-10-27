\\ Second engine for derivations/swath_sphere.mac: the swath geometry and the great-circle walk of
\\ crates/spacelab-earthvision/src/swath.rs at 60 digits, computed from the geometry itself rather
\\ than from the closed forms the code uses.
\\
\\ Swath: the slant range is the smaller root of |S + rho u|^2 = re^2 (polroots of the defining
\\ quadratic), the ground point T = S + rho u, the central angle atan2(T_x, T_y) and the elevation the
\\ angle between S - T and T's horizon. Walk: the arrival point is the start rotated about the axis
\\ normal to the great circle, X = cos(d) P + sin(d) (cos(az) N + sin(az) E), read back as
\\ (asin X_z, lon + arg(X_x + i X_y)). Every input is a dyadic double (angles, metre altitudes),
\\ so the values are the answers for the doubles the Rust gate passes. Each closed-form agreement
\\ prints `<name>: 0` when |difference| < 1e-50; the rows are the literals of the gate
\\ `the_swath_is_the_root_of_the_line_of_sight_quadratic` in swath.rs.
\\
\\ Run: timeout 300 gp -q -f derivations/swath_sphere.gp </dev/null
default(realprecision, 60);
chk(name, e) = printf("%s: %s\n", name, if(abs(e) < 1e-50, 0, e));
fmt(v) = concat([c | c <- Vec(Strprintf("%.30e", v)), c != " "]);
re = 6378137;

swath(h, eta, tag) = {
  my(rs = re + h, u = [sin(eta), -cos(eta)], S = [0, rs], q, r, rho, T, lam, eps, k);
  q = x^2 - 2*rs*cos(eta)*x + rs^2 - re^2;
  r = real(polroots(q));
  rho = vecmin(r);
  T = S + rho*u;
  lam = arg(T[2] + I*T[1]);
  eps = asin((S - T)*T~/(re*rho));
  \\ The code's closed forms, at the same precision.
  k = rs/re*sin(eta);
  chk(Str(tag, "_central_angle_closed_form"), (asin(k) - eta) - lam);
  chk(Str(tag, "_elevation_closed_form"), (Pi/2 - asin(k)) - eps);
  chk(Str(tag, "_slant_range_closed_form"), sqrt(re^2 + rs^2 - 2*re*rs*cos(lam)) - rho);
  chk(Str(tag, "_ground_point_on_sphere"), sqrt(T*T~) - re);
  printf("        (%.1f, %.17g, %s, %s, %s),\n", h, eta, fmt(lam), fmt(eps), fmt(rho));
}

walk(lat, lon, d, az, tag) = {
  my(P = [cos(lat), 0, sin(lat)], N = [-sin(lat), 0, cos(lat)], E = [0, 1, 0], X, lat2, lon2);
  X = cos(d)*P + sin(d)*(cos(az)*N + sin(az)*E);
  chk(Str(tag, "_arrival_is_unit"), X*X~ - 1);
  lat2 = asin(X[3]);
  lon2 = lon + arg(X[1] + I*X[2]);
  printf("        (%.17g, %.17g, %.17g, %.17g, %s, %s),\n", lat, lon, d, az, fmt(lat2), fmt(lon2));
}

print("swath rows (altitude, look angle, central angle, elevation, slant range):");
swath(550000, 1/16, "leo550_a");
swath(550000, 7/16, "leo550_b");
swath(700000, 1/4, "leo700_a");
swath(700000, 35/32, "leo700_near_limb");
swath(35786000, 1/8, "geo");
print("walk rows (lat, lon, central angle, azimuth, lat2, lon2):");
walk(19/32, -19/16, 13/32, 11/16, "walk_a");
walk(-11/32, 93/32, 21/16, -61/32, "walk_b");
walk(45/32, 1/8, 1/64, 3, "walk_c");
