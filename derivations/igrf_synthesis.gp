/* Second engine for derivations/igrf_synthesis.mac: the IGRF-14 geocentric field at the three
   reference points, from PARI's own Legendre polynomials (pollegendre, not Rodrigues' formula),
       P_n^m(cos t) = sqrt((2 - delta_m0)(n-m)!/(n+m)!) sin(t)^m Q(cos t),  Q = d^m P_n/dx^m,
       dP_n^m/dt    = m sin(t)^(m-1) cos(t) Q(cos t) - sin(t)^(m+1) Q'(cos t),
   the coefficients read as exact decimals from the same IAGA file, and compared component by
   component with the values Maxima printed (pasted below).

   Run from derivations/, as verify.sh does:
       timeout 300 gp -q -f igrf_synthesis.gp </dev/null                            */

default(realprecision, 60);

words(s) = select(w -> w != "", strsplit(s, " "));
{decimal(t) = my(ch = Vec(t), neg = (ch[1] == "-"), u, parts, v);
  u = if (neg, strjoin(ch[2..#ch]), t);
  parts = strsplit(u, ".");
  v = if (#parts == 1, eval(parts[1]), eval(parts[1]) + eval(parts[2]) / 10^#parts[2]);
  if (neg, -v, v);}

G = matrix(14, 14); H = matrix(14, 14); GS = matrix(14, 14); HS = matrix(14, 14);
{lines = readstr("../crates/spacelab-gnss/tests/fixtures/igrf14coeffs.txt"); rowsread = 0;
foreach (lines, l,
  my(t = words(l), n, m);
  if (#t > 3 && t[1] == "g/h",
    for (k = 1, #t, if (t[k] == "2025.0", c25 = k); if (t[k] == "2025-30", csv = k)));
  if (#t > 3 && (t[1] == "g" || t[1] == "h"),
    n = eval(t[2]); m = eval(t[3]); rowsread++;
    if (t[1] == "g", G[n + 1, m + 1] = decimal(t[c25]); GS[n + 1, m + 1] = decimal(t[csv]),
                     H[n + 1, m + 1] = decimal(t[c25]); HS[n + 1, m + 1] = decimal(t[csv]))));}

{field(theta, lam, rad, year) = my(dt = year - 2025, b = [0., 0., 0.], ct = cos(theta), st = sin(theta), a = 63712/10);
  for (n = 1, 13,
    my(Pn = pollegendre(n, 'x));
    for (m = 0, n,
      my(Q = Pn, Q1, norm, pp, dpp, g, h, w);
      for (k = 1, m, Q = deriv(Q, 'x));
      Q1 = deriv(Q, 'x);
      norm = sqrt((if (m == 0, 1, 2)) * (n - m)! / (n + m)!);
      pp = norm * st^m * subst(Q, 'x, ct);
      dpp = norm * (if (m == 0, 0, m * st^(m - 1) * ct * subst(Q, 'x, ct)) - st^(m + 1) * subst(Q1, 'x, ct));
      g = G[n + 1, m + 1] + dt * GS[n + 1, m + 1]; h = H[n + 1, m + 1] + dt * HS[n + 1, m + 1];
      w = (a / rad)^(n + 2);
      b[1] += w * (n + 1) * pp * (g * cos(m * lam) + h * sin(m * lam));
      b[2] -= w * dpp * (g * cos(m * lam) + h * sin(m * lam));
      b[3] += w * m * pp * (g * sin(m * lam) - h * cos(m * lam)) / st));
  b;}

points = [[1, 1/2, 27085/4, 2025], [7/4, 7/2, 13143/2, 2030], [21/8, 21/4, 7000, 4055/2]];
{maxima = [[-2.745691301632609806507634325022553533148e4, -2.437751823657892777948550313259657607389e4, 1.810361034812790661533836797608695878954e3],
          [1.085958398417876417948311373642412659937e4, -2.889611203608967979272386856787747651347e4, 5.757108496833010019237629378447388139413e3],
          [2.221308232454103436252749632446625474901e4, -1.386195549581576320968095115259524339195e4, 1.912815784310201997253132608934219810885e3]];}

{off = 0; worst = 0.;
for (i = 1, #points,
  my(q = points[i], b = field(q[1], q[2], q[3], q[4]));
  for (k = 1, 3, my(d = abs(b[k] - maxima[i][k])); worst = max(worst, d); off += d > 1e-30));}
printf("%d IAGA coefficient rows read\n", rowsread);
printf("%d of 9 field components off by more than 1e-30 nT from Maxima; largest gap %.3e nT\n", off, worst);
