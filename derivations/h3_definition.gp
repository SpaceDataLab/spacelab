/* Second engine for derivations/h3_definition.mac, at 100 digits.

   (1) The 20 face centres of h3_constants.mac pair up with cosines -1, -sqrt5/3, -1/3, 1/3, sqrt5/3
       in counts 10, 30, 60, 60, 30, each within 1e-15.
   (2) Each face's i, j, k azimuths point at its three vertices (the circumcentre of the face centre
       and two neighbours), counter-clockwise 120 degrees apart, within 1e-14.
   (3) An exact icosahedron, vertices (0, +-1, +-phi) cycled, has tan(centroid to vertex) = 3 - sqrt5;
       the data's vertices sit there too, within 1e-15.
   (4) H3's aperture-7 constant is atan(sqrt3/5) to 1e-44, and sin^2 of it is 3/28.
   (5) Every row of crates/spacelab-grid/tests/h3_maxima.rs, rebuilt here: the closest face, the
       gnomonic plane with u = (3 - sqrt5)/2, the nearest point of the lattice (i - j/2, j sqrt3/2),
       back to the sphere; the table's centre within 1e-16 of its value, relative.

       gp -q -f derivations/h3_definition.gp </dev/null                                            */

default(realprecision, 100);
starts(s, p) = #s >= #p && Vecsmall(s)[1 .. #p] == Vecsmall(p);
strip(s) = Strchr(select(c -> c != 91 && c != 93 && c != 32 && c != 36, Vec(Vecsmall(s))));

/* ---- h3_constants.mac: `name : [ ... ]$` blocks of rationals, and `name : p/q$` scalars ---------- */
src = readstr("h3_constants.mac");
block(name) =
{
  my(body = "", inside = 0);
  for (k = 1, #src,
    my(s = src[k]);
    if (inside, body = concat(body, s); if (#s && Vecsmall(s)[#s] == 36, break));
    if (starts(s, Str(name, " : [")), inside = 1));
  apply(t -> eval(t), select(t -> #t, strsplit(strip(body), ",")));
}
scalar(name) =
{
  for (k = 1, #src,
    my(s = src[k], head = Str(name, " : "));
    if (starts(s, head), return(eval(strip(Strchr(Vecsmall(s)[#head + 1 .. #s]))))));
  error("no ", name);
}
cf = block("h3_face_centre");
az = block("h3_face_axes_az");
if (#cf != 60 || #az != 60, print("FAIL: read ", #cf, " and ", #az, " numbers"));
C = vector(20, f, [cf[3*f - 2], cf[3*f - 1], cf[3*f]] * 1.);
AZ = vector(20, f, [az[3*f - 2], az[3*f - 1], az[3*f]] * 1.);
ap7 = scalar("h3_ap7_rot");

dotv(a, b) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
crossv(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
unitv(a) = a / sqrt(dotv(a, a));
posang(a) = a - 2*Pi*floor(a/(2*Pi));
angdiff(a, b) = my(d = posang(a - b)); min(d, 2*Pi - d);
northof(p) = unitv([0, 0, 1] - p[3]*p);
eastof(p) = crossv(northof(p), p);
azimuth(p1, p2) = my(q = unitv(p2 - dotv(p2, p1)*p1)); posang(atan2(dotv(q, eastof(p1)), dotv(q, northof(p1))));
atan2(y, x) = arg(x + I*y);
tol = 1e-14;
ok = 1;

/* (1) */
{
my(values = [-1, -sqrt(5)/3, -1/3, 1/3, sqrt(5)/3], want = [10, 30, 60, 60, 30], got = vector(5), worst = 0);
for (a = 1, 20, for (b = a + 1, 20,
  my(c = dotv(C[a], C[b]), k = 1);
  for (j = 2, 5, if (abs(c - values[j]) < abs(c - values[k]), k = j));
  got[k]++; worst = max(worst, abs(c - values[k]))));
if (got != want || worst > 1e-15, ok = 0; print("FAIL: icosahedron ", got, " ", worst));
}

/* (2) */
neighbours(f) = select(g -> g != f && abs(dotv(C[f], C[g]) - sqrt(5)/3) < tol, vector(20, g, g));
vertices(f) =
{
  my(nb = neighbours(f), out = List());
  for (a = 1, 3, for (b = a + 1, 3,
    my(v = unitv(crossv(C[nb[a]] - C[f], C[nb[b]] - C[f])));
    if (dotv(v, C[f]) < 0, v = -v);
    listput(out, v)));
  Vec(out);
}
{
for (f = 1, 20,
  my(vs = vertices(f), azs);
  if (#vs != 3, ok = 0; print("FAIL: face ", f, " has ", #vs, " vertices"); next);
  azs = apply(v -> azimuth(C[f], v), vs);
  for (k = 1, 3,
    if (vecmin(apply(a -> angdiff(a, AZ[f][k]), azs)) > tol, ok = 0; print("FAIL: face ", f, " axis ", k)));
  if (angdiff(AZ[f][2], AZ[f][1] - 2*Pi/3) > tol || angdiff(AZ[f][3], AZ[f][1] - 4*Pi/3) > tol,
    ok = 0; print("FAIL: face ", f, " axes not 120 degrees apart")));
}

/* (3) */
tangent(c, p) = sqrt(dotv(p, p)*dotv(c, c) - dotv(p, c)^2) / dotv(p, c);
{
my(phi = (1 + sqrt(5))/2, V = List(), F = List(), gap = 0);
foreach ([-1, 1], s1, foreach ([-1, 1], s2,
  listput(V, [0, s1, s2*phi]); listput(V, [s1, s2*phi, 0]); listput(V, [s2*phi, 0, s1])));
V = Vec(V);
adj(a, b) = abs(dotv(a - b, a - b) - 4) < 1e-90;
for (a = 1, 12, for (b = a + 1, 12, for (c = b + 1, 12,
  if (adj(V[a], V[b]) && adj(V[b], V[c]) && adj(V[a], V[c]), listput(F, [V[a], V[b], V[c]])))));
if (#F != 20, ok = 0; print("FAIL: ", #F, " faces"));
foreach (F, t,
  my(c = (t[1] + t[2] + t[3])/3);
  gap = max(gap, abs(tangent(c, t[1]) - (3 - sqrt(5))));
  gap = max(gap, abs(tangent(c, (t[1] + t[2])/2) - (3 - sqrt(5))/2)));
if (gap > 1e-90, ok = 0; print("FAIL: exact icosahedron tangents ", gap));
gap = 0;
for (f = 1, 20, foreach (vertices(f), v, gap = max(gap, abs(tangent(C[f], v) - (3 - sqrt(5))))));
if (gap > 1e-15, ok = 0; print("FAIL: data vertex tangents ", gap));
}

/* (4) */
{
if (abs(ap7 - atan(sqrt(3)/5)) > 1e-44 || abs(sin(atan(sqrt(3)/5))^2 - 3/28) > 1e-90,
  ok = 0; print("FAIL: aperture-7 angle"));
}

/* (5) */
u = (3 - sqrt(5))/2;
closest(p) = my(best = 1); for (f = 2, 20, if (dotv(C[f], p) > dotv(C[best], p), best = f)); best;
centre(lat, lon, res) =
{
  my(d = Pi/180, p = [cos(lat*d)*cos(lon*d), cos(lat*d)*sin(lon*d), sin(lat*d)], f = closest(p), r, th,
     v, j0, i0, best = [10^9, 0, 0], L, c, dir);
  r = atan2(sqrt(dotv(crossv(C[f], p), crossv(C[f], p))), dotv(C[f], p));
  th = AZ[f][1] - azimuth(C[f], p) - if (res % 2, atan(sqrt(3)/5), 0);
  r = tan(r) / u * sqrt(7)^res;
  v = [r*cos(th), r*sin(th)];
  j0 = floor(2*v[2]/sqrt(3)); i0 = floor(v[1] + v[2]/sqrt(3));
  for (i = i0 - 1, i0 + 2, for (j = j0 - 1, j0 + 2,
    my(x = i - j/2, y = j*sqrt(3)/2, e = sqrt((v[1] - x)^2 + (v[2] - y)^2));
    if (e < best[1], best = [e, x, y])));
  L = best[2 .. 3];
  r = sqrt(L[1]^2 + L[2]^2);
  if (r < 1e-30, c = C[f],
    th = atan2(L[2], L[1]) + if (res % 2, atan(sqrt(3)/5), 0);
    r = atan(r * sqrt(7)^(-res) * u);
    th = AZ[f][1] - th;
    dir = cos(th)*northof(C[f]) + sin(th)*eastof(C[f]);
    c = cos(r)*C[f] + sin(r)*dir);
  [asin(c[3])/d, atan2(c[2], c[1])/d];
}
tab = readstr("../crates/spacelab-grid/tests/h3_maxima.rs");
body = ""; inside = 0;
{
for (k = 1, #tab,
  my(s = tab[k]);
  if (inside && starts(s, "];"), break);
  if (inside, body = concat(body, s));
  if (starts(s, "const H3_MAXIMA:"), inside = 1));
}
vals = apply(s -> eval(s), select(s -> #s, strsplit(strip(body), ",")));
worst = 0;
{
for (r = 0, #vals/5 - 1,
  my(row = vals[5*r + 1 .. 5*r + 5], got);
  got = centre(round(row[1]*10^10)/10^10, round(row[2]*10^10)/10^10, round(row[3]));
  for (k = 1, 2, worst = max(worst, abs(got[k] - row[3 + k])/max(abs(got[k]), 1))));
}
print("worst relative difference against the table: ", worst);
{
if (ok && #vals == 120 && worst < 10^-16,
  print("h3 definition holds at 100 digits"),
  print("FAIL: h3 definition"));
}
