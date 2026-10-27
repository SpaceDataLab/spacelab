/* Second engine for derivations/hull_delaunay.mac: the two 14-point integer sets rebuilt from the
   same 61-bit LCG, the Delaunay triangulation, the spatial hull and the planar hull found again by
   brute force with PARI's exact integer determinants, and each compared with the table Maxima
   printed (pasted below). Indices are 0-based, triangles start at their smallest index.

       timeout 300 gp -q -f derivations/hull_delaunay.gp </dev/null                              */

lcg = 987654321;
rnext() = lcg = (lcg * 6364136223846793005 + 1442695040888963407) % 2^61;
rint(lo, width) = lo + floor(rnext() * width / 2^61);

n = 14;
P2 = vector(n, i, [rint(0, 1024), rint(0, 1024)]);
P3 = vector(n, i, [rint(-1024, 2048), rint(-1024, 2048), rint(-1024, 2048)]);

o2(a, b, c) = (a[1] - c[1]) * (b[2] - c[2]) - (a[2] - c[2]) * (b[1] - c[1]);
{ic(a, b, c, d) = my(p = a - d, q = b - d, r = c - d);
  matdet([p[1], p[2], p[1]^2 + p[2]^2; q[1], q[2], q[1]^2 + q[2]^2; r[1], r[2], r[1]^2 + r[2]^2]);}
o3(a, b, c, d) = my(p = a - d, q = b - d, r = c - d); matdet([p[1], p[2], p[3]; q[1], q[2], q[3]; r[1], r[2], r[3]]);

rotmin(t) = my(m = vecmin(t)); if (t[1] == m, t, if (t[2] == m, [t[2], t[3], t[1]], [t[3], t[1], t[2]]));

{degenerate = 0;
forvec (q = vector(4, i, [1, n]),
  degenerate += (ic(P2[q[1]], P2[q[2]], P2[q[3]], P2[q[4]]) == 0)
              + (o3(P3[q[1]], P3[q[2]], P3[q[3]], P3[q[4]]) == 0), 2);
forvec (t = vector(3, i, [1, n]), degenerate += (o2(P2[t[1]], P2[t[2]], P2[t[3]]) == 0), 2);}

{delaunay = List(); hull = List();
forvec (t = vector(3, i, [1, n]),
  my(u = if (o2(P2[t[1]], P2[t[2]], P2[t[3]]) > 0, t, [t[1], t[3], t[2]]), empty = 1, back = 1, front = 1);
  for (p = 1, n, if (setsearch(Set(t), p), next);
    if (ic(P2[u[1]], P2[u[2]], P2[u[3]], P2[p]) >= 0, empty = 0);
    my(s = o3(P3[t[1]], P3[t[2]], P3[t[3]], P3[p]));
    if (s <= 0, back = 0);
    if (s >= 0, front = 0));
  if (empty, listput(delaunay, rotmin(u) - [1, 1, 1]));
  if (back, listput(hull, rotmin(t) - [1, 1, 1]));
  if (front, listput(hull, rotmin([t[1], t[3], t[2]]) - [1, 1, 1])), 2);}
delaunay = vecsort(Vec(delaunay)); hull = vecsort(Vec(hull));

lexless(a, b) = a[1] < b[1] || (a[1] == b[1] && a[2] < b[2]);
{nextv(cur) = for (j = 1, n, if (j == cur, next);
  if (prod(p = 1, n, p == cur || p == j || o2(P2[cur], P2[j], P2[p]) > 0), return(j)));}
{start = 1; for (i = 2, n, if (lexless(P2[i], P2[start]), start = i));
cyc = [start]; cur = nextv(start);
while (cur != start, cyc = concat(cyc, cur); cur = nextv(cur));}
cyc = cyc - vector(#cyc, i, 1);

maxima_delaunay = [[0,1,4],[0,4,10],[0,6,8],[0,7,6],[0,8,11],[0,10,7],[0,11,1],[1,11,4],[2,4,11],[2,11,12],[3,6,5],[3,9,6],[3,13,9],[5,6,7],[5,7,10],[6,9,8],[8,9,13],[8,12,11],[8,13,12]];
maxima_planar_hull = [2,4,10,5,3,13,12];
maxima_hull = [[1,3,5],[1,5,9],[1,8,11],[1,9,8],[1,11,13],[1,13,3],[3,13,5],[4,5,13],[4,7,10],[4,10,5],[4,13,7],[5,10,9],[7,11,10],[7,13,11],[8,9,10],[8,10,11]];

printf("degenerate triples and quadruples: %d\n", degenerate);
printf("delaunay: %d triangles, %d unlike Maxima's table\n", #delaunay, #setminus(Set(delaunay), Set(maxima_delaunay)) + #setminus(Set(maxima_delaunay), Set(delaunay)));
printf("spatial hull: %d faces, %d unlike Maxima's table\n", #hull, #setminus(Set(hull), Set(maxima_hull)) + #setminus(Set(maxima_hull), Set(hull)));
printf("planar hull: %d vertices, %d positions unlike Maxima's cycle\n", #cyc, if (#cyc == #maxima_planar_hull, sum(i = 1, #cyc, cyc[i] != maxima_planar_hull[i]), #cyc));
