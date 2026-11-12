/* Second engine for derivations/event_geometry.mac, numerically at 100 digits.

   (1) The tangent point P of a unit sphere centred at (d, 0, 0), seen from the origin, built from
   cos(phi) = -1/d: (P - V) . (P - O) vanishes and the angle P makes with O is asin(1/d).
   (2) The rim of a disk of angular radius a whose centre is s from e = (1, 0, 0), sampled at 3600
   position angles (both extremes among them): the farthest rim point is s + a from e and the
   nearest |s - a|, the containment tests behind the umbra and penumbra g.
   (3) The elevation atan2(v, h) of a line of sight d against a unit up is asin(v/|d|).
   (4) atan2(|a x b|, a . b) is acos(a . b / (|a| |b|)).

       gp -q -f derivations/event_geometry.gp </dev/null                                        */

default(realprecision, 100);
setrand(20261112);
u01() = random(10^30)/10^30;
cross(x, y) = [x[2]*y[3] - x[3]*y[2], x[3]*y[1] - x[1]*y[3], x[1]*y[2] - x[2]*y[1]];
norm2(x) = sqrt(x*x~);
\\ atan2(y, x), which GP spells as the argument of x + iy
atan2(y, x) = arg(x + I*y);

worst_tangent = 0;
{
for (k = 1, 50,
  my(d = 1 + 20*u01(), c = -1/d, P = [d + c, sqrt(1 - c^2), 0], O = [d, 0, 0]);
  worst_tangent = max(worst_tangent, abs(P*(P - O)~));
  worst_tangent = max(worst_tangent, abs(acos((P*O~)/(norm2(P)*d)) - asin(1/d))));
}

worst_rim = 0;
{
for (k = 1, 50,
  my(s = 2*u01(), a = u01()/2, u = [cos(s), sin(s), 0], n1 = [-sin(s), cos(s), 0],
     n2 = [0, 0, 1], far = 0, near = 10);
  for (j = 0, 3599,
    my(psi = 2*Pi*j/3600, r = cos(a)*u + sin(a)*(cos(psi)*n1 + sin(psi)*n2), D = acos(r[1]));
    far = max(far, D); near = min(near, D));
  worst_rim = max(worst_rim, max(abs(far - (s + a)), abs(near - abs(s - a)))));
}

worst_angle = 0;
{
for (k = 1, 50,
  my(d = [u01() - 1/2, u01() - 1/2, u01() - 1/2], up = [u01() - 1/2, u01() - 1/2, u01() - 1/2], v, h);
  up = up/norm2(up);
  v = d*up~; h = norm2(d - v*up);
  worst_angle = max(worst_angle, abs(atan2(v, h) - asin(v/norm2(d))));
  my(a = [u01(), u01() - 1/2, u01()], b = [u01() - 1/2, u01(), u01() - 1/2]);
  worst_angle = max(worst_angle, abs(atan2(norm2(cross(a, b)), a*b~) - acos(a*b~/(norm2(a)*norm2(b))))));
}

print("tangent cone worst: ", worst_tangent);
print("rim extremes worst: ", worst_rim);
print("elevation and angle forms worst: ", worst_angle);
{
if (worst_tangent < 10^-90 && worst_rim < 10^-90 && worst_angle < 10^-90,
  print("event geometry holds at 100 digits"),
  print("FAIL: event geometry"));
}
