/* Second engine for derivations/ray_ellipsoid_geolocation.mac, by a different route to each value.

   - Each hit is found by bisection-safe root-finding (solve) of the ellipsoid equation along the
     ray, on [0, t_min], where t_min minimises the stretched distance to the centre -- the near root
     -- not by the closed form the code and Maxima use.
   - Each geodetic point is checked against the defining equation of its latitude,
     rho sin(phi) - z cos(phi) = e^2 N(phi) sin(phi) cos(phi), solved for phi by solve(), and its
     height recovered; each subpoint is the point minus h along the geodetic normal.
   - The rotation is the quaternion q = (cos(-theta/2), sin(-theta/2) u) acting as q v q*.

   Every comparison is against the 40-digit values Maxima printed; the summary line counts the ones
   off by more than 1e-30.

       timeout 300 gp -q -f derivations/ray_ellipsoid_geolocation.gp </dev/null                    */

default(realprecision, 60);
A = 6378137/1000; B = 6356752314245/10^9; E2 = (A^2 - B^2)/A^2;
off = 0; checked = 0;
{
chk(x, y) =
  checked++;
  if (abs(x - y) > 1e-30, off++; print("  off: ", x, " vs ", y));
}
chkv(u, v) = for (i = 1, #u, chk(u[i], v[i]));
surface(p) = p[1]^2/A^2 + p[2]^2/A^2 + p[3]^2/B^2 - 1;

{
hits = [[[5120, 3584, 2304], [-3/4, -1/2, -1/4],
         [4890.290815378335094847476252633982658003, 3430.860543585556729898317501755988438668,
          2227.430271792778364949158750877994219334]],
        [[7168, 0, 0], [-1, 1/4, 1/2],
         [6362.12128196360247518565889751358320676, 201.46967950909938120358527562160419831,
          402.9393590181987624071705512432083966199]],
        [[1024, -2048, 6656], [-1/8, 1/4, -1],
         [942.0112164896066729437312161930296918389, -1884.022432979213345887462432386059383678,
          6000.089731916853383549849729544237534712]]];
for (k = 1, #hits,
  my(o = hits[k][1], d = hits[k][2], f, g, tmin, t);
  f = (s -> surface(o + s*d));
  /* the stretched distance is quadratic in t; its minimiser separates the two roots */
  g = [d[1]/A, d[2]/A, d[3]/B]; tmin = -([o[1]/A, o[2]/A, o[3]/B] * g~)/(g * g~);
  t = solve(s = 0, tmin, f(s));
  chkv(o + t*d, hits[k][3]));
}

{
pts = [[11/16, 3/8, 830,
        [5189.774376376714187859954821224595835099, 2042.833117599752086115134390505359496098,
         4552.669169229801625033284288504557110374],
        [4592.897465840407941659330085441662559025, 1807.88650305627562424890351061612326901,
         4025.945292817128108647527059715509814499]],
       [-39/32, -5/2, 0,
        [-1767.17869722729794810423723526096890889, -1320.121890033959005847309249555550562114,
         -5964.497471789410159029018412661337872095],
        [-1767.17869722729794810423723526096890889, -1320.121890033959005847309249555550562114,
         -5964.497471789410159029018412661337872095]],
       [3/2, 1, 35786,
        [1612.3070398063300776602343621570681776, 2511.019438309857403672832230318964789423,
         42037.07724271363723638518156038863368714],
        [244.5849728831760277074734344590675461336, 380.9185261027681394929370689141821087784,
         6340.721652100945370704669000719368356764]],
       [0, 1/4, 500,
        [6664.312377527589183681955311197614020173, 1701.678326095026579408480160226663255675, 0],
        [6179.856166672266791609657586450519420271, 1577.976346467765114610055807801968657728, 0]]];
for (k = 1, #pts,
  my(P = pts[k][4], S = pts[k][5], rho = sqrt(P[1]^2 + P[2]^2), N, phi, h, nrm, lam);
  N = (f -> A/sqrt(1 - E2*sin(f)^2));
  phi = solve(f = -Pi/2 + 1e-9, Pi/2 - 1e-9, rho*sin(f) - P[3]*cos(f) - E2*N(f)*sin(f)*cos(f));
  h = rho*cos(phi) + P[3]*sin(phi) - A*sqrt(1 - E2*sin(phi)^2);
  lam = atan(P[2]/P[1]) + if (P[1] < 0, if (P[2] < 0, -Pi, Pi), 0);
  chk(phi, pts[k][1]); chk(h, pts[k][3]); chk(lam, pts[k][2]);
  nrm = [cos(phi)*cos(lam), cos(phi)*sin(lam), sin(phi)];
  chkv(P - h*nrm, S));
}

{
my(u = [1, 2, 2]/3, v = [1/2, -1, 2], th = -3/4, q, w, qv, cross);
cross = ((a, b) -> [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]]);
/* q v q* for q = (w, qv): v + 2w (qv x v) + 2 qv x (qv x v) */
w = cos(th/2); qv = sin(th/2)*u;
chkv(v + 2*w*cross(qv, v) + 2*cross(qv, cross(qv, v)),
     [-0.9229022158524859142860752937819913053789, -0.3554142093514988786851129648512779786391,
      2.066865317277741835828150611742273631329]);
}
printf("ray_ellipsoid_geolocation: %d values checked, %d off by more than 1e-30\n", checked, off);
