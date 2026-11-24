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

/* Section 6 of igrf_synthesis.mac, the polar axis (plans/18 Ф28.K). From PARI's Legendre
   polynomials: P_n^1/sin(t) = sqrt(2/(n(n+1))) P_n'(cos t) and cos(t) dP_n^1/dt agree on both axes
   and equal c^(n+1) sqrt(n(n+1)/2), c = cos t = +-1; the axis field in closed form and field_ecef's
   reference beside the axis, against the values Maxima printed (pasted below, the module test's). */
{axisbad = 0;
for (n = 1, 13, my(P1 = deriv(pollegendre(n, 'x), 'x), k = sqrt(2 / (n * (n + 1))));
  foreach ([1, -1], c0,
    my(quot = k * subst(P1, 'x, c0), dpdt = c0 * k * subst(P1, 'x, c0));
    if (abs(quot - c0 * dpdt) > 1e-55 || abs(quot - c0^(n + 1) * sqrt(n * (n + 1) / 2)) > 1e-55, axisbad++)));
if (axisbad, printf("FAIL: %d axis limits disagree\n", axisbad),
  printf("the axis limit of P_n^1/sin(t) is cos(t) dP_n^1/dt = c^(n+1) sqrt(n(n+1)/2), n = 1..13, both axes\n"));}
{axis(rad, c0) = my(a = 63712/10, v = [0., 0., 0.]);
  for (n = 1, 13, my(w = (a / rad)^(n + 2), l = c0^(n + 1) * sqrt(n * (n + 1) / 2));
    v[1] -= w * l * G[n + 1, 2]; v[2] -= w * l * H[n + 1, 2]; v[3] += w * c0^(n + 1) * (n + 1) * G[n + 1, 1]);
  v;}
{ecef(px, py, pz) = my(rr = sqrt(px^2 + py^2 + pz^2), th = atan(sqrt(px^2 + py^2) / pz), la, b, st, ct);
  if (pz < 0, th += Pi);
  la = if (px == 0 && py == 0, 0, atan(py / px));
  b = field(th, la, rr / 1000, 2025); st = sin(th); ct = cos(th);
  [b[1] * st * cos(la) + b[2] * ct * cos(la) - b[3] * sin(la),
   b[1] * st * sin(la) + b[2] * ct * sin(la) + b[3] * cos(la), b[1] * ct - b[2] * st];}
{maxima_axis = [[-1.15154760713670661717965673626953237555643474463375577377274e3, 1.03757825779880875617223669690019383007283707789877326954919e2, -4.79223894266417214539678176120203723664077604289038376377983e4],
                [1.07543075864577103624906925167824116240642970066961923057537e4, -7.22412092491878654195030584349991045955679061511209060288797e3, -4.28527693775096540398469162326673208837159792124287691488551e4]];
maxima_near = [[21, 6771200, [-1.15154760714195163768599451093785817054948545075777969428012e3, 1.03757825771662997888085086312061591474254871876661784679906e2, -4.79223894266417813193541868398532443007076810589178360192517e4]],
  [21, -6771200, [1.0754307586462865324476463706593673808505623042374973216572e4, -7.22412092490981736448172671874138037521893621054131380330043e3, -4.2852769377509429918206678583031145239927167274225601989812e4]],
  [14, 6771200, [-1.15154760780806924198895530725651502816904570189662077679578e3, 1.03757824727992526284582119077231243828683682782994199654929e2, -4.79223894266493842233833596088322929906490115685987995891342e4]],
  [14, -6771200, [1.0754307587117545496655157653143650888318154843785521079576e4, -7.22412092377073182597228843821161357441202810197931081457882e3, -4.28527693774809664698415608128208667998358081323246395583348e4]],
  [7, 6771200, [-1.15154769307112256104083873272901742129371769256904656617124e3, 1.03757691138172113478349049404182783489862965416377127038902e2, -4.79223894276225552883983805467026828283868528045559804503653e4]],
  [7, -6771200, [1.07543076709166073021167672810948253148717181907401363988966e4, -7.22412077796778289857555810245511331219388796701362358077126e3, -4.28527693738376441790856347220737654676153528894247231543372e4]],
  [5, 6771200, [-1.15154795087437000887790055146789624883671677525211594880142e3, 1.03757287213045251838762677205580281745312842195284436174061e2, -4.79223894305650489210952832067590899505297334093413494779925e4]],
  [5, -6771200, [1.07543079242932952978334048698723356180597084777297442655903e4, -7.22412033711477199036037414742202131775936055465248904714645e3, -4.28527693628216037108458632335448210309530117157764466759773e4]],
  [0, 6771200, [-1.15155860674142758098781366676994819111085046959133123674735e3, 1.03740591640380088189212464128023129673015885247838522633332e2, -4.79223895521777902281465520258633679919573297775034877281632e4]],
  [0, -6771200, [1.0754318397192694188147425720550204271328689162179814684062e4, -7.22410211519034987509992926231536841967714602065614619176907e3, -4.28527689074776450231896410478908827699381906860856022820663e4]]];}
{gap = 0.;
for (i = 1, 2, my(v = axis(67712/10, if (i == 1, 1, -1))); for (k = 1, 3, gap = max(gap, abs(v[k] - maxima_axis[i][k]))));
for (i = 1, #maxima_near, my([k, z0, want] = maxima_near[i], v = ecef(2^(-k), 2^(1 - k), z0));
  for (j = 1, 3, gap = max(gap, abs(v[j] - want[j]))));
if (gap > 1e-45, printf("FAIL: the axis and near-axis fields are %.3e nT from Maxima\n", gap),
  printf("axis closed form and 10 near-axis ECEF fields agree with Maxima to %.3e nT\n", gap));}
