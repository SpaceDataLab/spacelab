/* Second engine for derivations/box_wing_momentum.mac, numerically at 100 digits.

   (1) Lambert's 2/3 by quadrature over the hemisphere.
   (2) SRP: the three fates (absorbed, specular, Lambertian) summed photon by photon against the
       code's cN n + cS F, at random unit normals and Sun directions with n.s > 0.
   (3) Drag: absorbed plus specular molecules against (1 - l)|f| w + 2 l f |d| n, at random
       windward geometries; and the specular force is |d| times Orekit's 2 l f n.
   (4) The Sun-tracking normal: no rotation of the array about its axis, sampled every 0.1 degree,
       sees the Sun better than the projection does, and the projection is attained.

       gp -q -f derivations/box_wing_momentum.gp </dev/null                                       */

default(realprecision, 100);
setrand(20261112);
u01() = random(10^30)/10^30;
unitv() = my(v = [u01() - 1/2, u01() - 1/2, u01() - 1/2]); v/sqrt(v*v~);
cross(x, y) = [x[2]*y[3] - x[3]*y[2], x[3]*y[1] - x[1]*y[3], x[1]*y[2] - x[2]*y[1]];

lambert = intnum(t = 0, Pi/2, cos(t)^2*sin(t))/intnum(t = 0, Pi/2, cos(t)*sin(t));
worst_lambert = abs(lambert - 2/3);

worst_srp = 0;
{
for (k = 1, 50,
  my(n = unitv(), s = unitv(), P = 1 + u01(), A = 1 + 5*u01(), Ca = u01()/2, Cs = u01()/2,
     Cd, c, u, Phi, fates, F, dS, cN, cS);
  c = n*s~;
  if (c < 0, s = -s; c = -c);
  Cd = 1 - Ca - Cs; u = -s; Phi = P*A*c;
  fates = Ca*Phi*u + Cs*Phi*(u - (u - 2*(u*n~)*n)) + Cd*Phi*(u - 2/3*n);
  F = P*u; dS = n*F~;
  cN = 2*A*dS*(Cd/3 - Cs*dS/P); cS = (A*dS/P)*(Cs - 1);
  worst_srp = max(worst_srp, vecmax(abs(cN*n + cS*F - fates))));
}

worst_drag = 0;
{
for (k = 1, 50,
  my(n = unitv(), w = unitv(), V = 7000 + 1000*u01(), rho = u01(), A = 1 + 5*u01(), CD = 2 + u01(),
     l = u01(), d, mflux, momentum, f, shipped, specular);
  d = n*w~;
  if (d > 0, w = -w; d = -d);
  mflux = -rho*V*A*d;
  momentum = (CD/2)*((1 - l)*mflux*V*w + l*mflux*V*(w - (w - 2*d*n)));
  f = (rho*V^2/2)*CD*A*d;
  shipped = (1 - l)*(-f)*w + 2*l*f*(-d)*n;
  specular = (CD/2)*l*mflux*V*2*d*n;
  worst_drag = max(worst_drag, vecmax(abs(shipped - momentum))/vecmax(abs(momentum)));
  worst_drag = max(worst_drag, vecmax(abs(-d*2*l*f*n - specular))/vecmax(abs(momentum))));
}

worst_array = 0;
{
for (k = 1, 20,
  my(a = unitv(), s = unitv(), sa, m, e1, e2, best = -2);
  sa = s*a~;
  m = (s - sa*a)/sqrt(1 - sa^2);
  e1 = m; e2 = cross(a, m);
  for (j = 0, 3599, my(q = cos(2*Pi*j/3600)*e1 + sin(2*Pi*j/3600)*e2); best = max(best, q*s~));
  worst_array = max(worst_array, abs(best - sqrt(1 - sa^2)));
  worst_array = max(worst_array, abs(m*a~) + abs(m*m~ - 1) + abs(m*s~ - sqrt(1 - sa^2))));
}

print("Lambert 2/3: ", worst_lambert);
print("SRP fates against the code form: ", worst_srp);
print("drag momentum against the shipped form and Orekit's term: ", worst_drag);
print("array normal: ", worst_array);
{
if (worst_lambert < 10^-90 && worst_srp < 10^-90 && worst_drag < 10^-90 && worst_array < 10^-90,
  print("box-wing momentum balance holds at 100 digits"),
  print("FAIL: box-wing momentum balance"));
}
