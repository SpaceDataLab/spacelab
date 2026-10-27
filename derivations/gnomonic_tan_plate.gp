/* Second engine for derivations/gnomonic_tan_plate.mac, by different routes.

   - The tangent-plane coordinates as vector geometry, with no spherical trigonometry formula: the
     star's unit vector s is scaled onto the plane tangent at c, s/(s.c), and read along the local
     east e = (-sin a0, cos a0, 0) and north n = (-sin d0 cos a0, -sin d0 sin a0, cos d0).
   - The least-squares plate by matsolve on the exact rational normal equations.
   - The SIP plate at the pixel in exact rationals.

   Compared against the values Maxima printed; the summary counts the ones off by more than 1e-30
   (exact rationals must agree exactly, so they are compared with ==).

       timeout 300 gp -q -f derivations/gnomonic_tan_plate.gp </dev/null                          */

default(realprecision, 60);
off = 0; checked = 0;
{
chk(x, y) =
  checked++;
  if (abs(x - y) > 1e-30, off++; print("  off: ", x, " vs ", y));
}
chkv(u, v) = for (i = 1, #u, chk(u[i], v[i]));
{
same(x, y) =
  checked++;
  if (x != y, off++; print("  off: ", x, " vs ", y));
}

a0 = 5/4; d0 = 1/2;
unitv(a, d) = [cos(d)*cos(a), cos(d)*sin(a), sin(d)];
{
my(c = unitv(a0, d0), e = [-sin(a0), cos(a0), 0], n = [-sin(d0)*cos(a0), -sin(d0)*sin(a0), cos(d0)],
   stars = [[161/128, 127/256], [79/64, 33/64], [41/32, 61/128]],
   want = [[0.006870836682979272888933354842053015111573, -0.003893494409455924647699405689189852725542],
           [-0.01359588481762888980947666798792613229059, 0.01567865290752769947818496722356329734449],
           [0.02778172082060422935033400578894062350972, -0.02324259286176584323607780627493715352756]]);
for (k = 1, #stars,
  my(s = unitv(stars[k][1], stars[k][2]), p);
  p = s/(s*c~);
  chkv([p*e~, p*n~], want[k]));
}

{
my(uv = [[-40, -30], [50, -20], [10, 45], [-25, 35], [60, 55], [-55, -50], [5, -5]],
   nxi = [3, -2, 1, 4, -5, 2, -3], neta = [-1, 2, -4, 3, 1, -2, 1], A, X, E, N, cx, ce, rx, re,
   vx, ve, m, q);
A = matrix(7, 3, i, j, if (j == 1, 1, uv[i][j - 1]));
X = vectorv(7, i, (3*uv[i][1] - uv[i][2])/2^18 + 1/2^12 + nxi[i]/2^24);
E = vectorv(7, i, (uv[i][1] + 2*uv[i][2])/2^18 - 1/2^13 + neta[i]/2^24);
N = A~*A;
cx = matsolve(N, A~*X); ce = matsolve(N, A~*E);
rx = X - A*cx; re = E - A*ce;
same(A~*rx, vectorv(3)); same(A~*re, vectorv(3));
vx = (rx~*rx)/4; ve = (re~*re)/4;
chkv(cx, [0.0002441385390936867081698731728288907996561, 0.00001143965513604178820678864342879820093922,
          -0.000003813471110679696607435180898207553409617]);
chkv(ce, [-0.0001220698700350244532481549154485525938664, 0.000003815888464917349595121756288996185814758,
          0.000007629092756206980825382520889829574266376]);
chkv(concat(Vec(N^-1)), [0.1450845514474061335626253940957294353683,
  0.0002178274577242762969332186873029521352823, -0.0005560332473488105474347950702206936084838,
  0.0002178274577242762969332186873029521352823, 0.0001231121987785788301695438410829640408316,
  -0.00007134510659876094097934166721784950503781, -0.0005560332473488105474347950702206936084838,
  -0.00007134510659876094097934166721784950503781, 0.0001416319421478492845646757942544700928192]);
chkv([vx, ve], [0.0000000000000159152596445256146302404712137307697475,
                0.00000000000002871666448176488020695573630691334208681]);
chk(sqrt((rx~*rx + re~*re)/14), 0.0000001129246577227369073029386277288105622296);
m = [1, 30, -12]; q = m*N^-1*m~;
chkv([q*vx, q*ve], [0.000000000000005635011771896418279524154348434269163574,
                    0.00000000000001016752136117400937127312542763445400683]);
}

{
my(CD = [3/2^17, -1/2^19; 1/2^18, 5/2^18], u = 37 - 512, v = -21 - 384, f, g);
f = 1/2^14*u^2 - 3/2^16*u*v + 1/2^15*v^2;
g = -1/2^15*u^2 + 1/2^13*u*v + 3/2^16*v^2;
same(CD*[u + f; v + g], [-340751845/34359738368; -77643725/8589934592]);
}
printf("gnomonic_tan_plate: %d values checked, %d off by more than 1e-30\n", checked, off);
