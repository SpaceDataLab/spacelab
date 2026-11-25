/* Second engine for derivations/nrlmsise00_flux_domain.mac (plans/18 Ф23.AJ, finding F128): the
   NRLMSISE-00 temperature profile's three fitted quantities Tinf, Tlb and g0 at the four flare
   inputs, at 60 digits, from a GLOBE7 transcribed again from NRL's FORTRAN (`nrlmsise00_sub.for`,
   FUNCTION GLOBE7 and GTS7) with its 1-based P indices.

   The coefficient tables are read from `tables.rs` as exact decimals. Each row must have
   Tinf < Tlb with Tlb > 0 and g0 > 0 -- the Bates-Walker profile's hypotheses failing on Tinf alone
   -- and a flux between F10.7a and the row's F10.7 where Tinf = Tlb, found here by bisection on the
   full model rather than Maxima's quadratic formula. The test module's ROW_* literals in `mod.rs`
   must be these values to the digits they show.

   Run from derivations/, as verify.sh does:
       timeout 300 gp -q -f nrlmsise00_flux_domain.gp </dev/null                                  */

default(realprecision, 60);

{decimal(t) = my(ch = Vec(t), neg = (ch[1] == "-"), u, parts, v);
  u = if (neg, strjoin(ch[2..#ch]), t);
  parts = strsplit(u, ".");
  v = if (#parts == 1, eval(parts[1]), eval(parts[1]) + eval(parts[2]) / 10^#parts[2]);
  if (neg, -v, v);}
{literal(t) = my(m = strsplit(t, "e"));
  if (#m == 1, decimal(m[1]), decimal(m[1]) * 10^eval(m[2]));}
/* The values between the last brackets of one Rust array line, as exact numbers. */
{numbers(l) = my(ch = Vec(l), a = 0, b = 0, body);
  for (k = 1, #ch, if (ch[k] == "[", a = k); if (ch[k] == "]", b = k));
  body = strjoin(ch[a + 1 .. b - 1]);
  apply(w -> literal(strjoin(select(c -> c != " ", Vec(w)))), strsplit(body, ","));}

tables = readstr("../crates/spacelab-propagation/src/nrlmsise00/tables.rs");
module = readstr("../crates/spacelab-propagation/src/nrlmsise00/mod.rs");
starts(l, p) = #l >= #p && strjoin(Vec(l)[1 .. #p]) == p;
line_of(lines, p) = { for (k = 1, #lines, if (starts(lines[k], p), return(k))); error("no line ", p); }

PT = numbers(tables[line_of(tables, "pub static PT:")]);
PS = numbers(tables[line_of(tables, "pub static PS:")]);
PTM = numbers(tables[line_of(tables, "pub static PTM:")]);
/* FORTRAN PD(1,4): the fourth row of PD, one array line each. */
PD4 = numbers(tables[line_of(tables, "pub static PD:") + 4]);
print("table sizes 150 150 50 150: ", if ([#PT, #PS, #PTM, #PD4] == [150, 150, 50, 150], "ok", "FAIL"));

DGTR = 174533 / 10^7; DR = 172142 / 10^7; HR = 2618 / 10^4; SR = 72722 / 10^9;

/* GLOBE7 with every switch on; SW(9) = -1 when `storm`. `x` = [day, sec, lat, lon, tloc, f107a,
   f107, ap, aparr, storm]. */
{globe7(P, x) = my(day = x[1], sec = x[2], lat = x[3], lon = x[4], tloc = x[5], f107a = x[6],
    f107 = x[7], ap = x[8], aparr = x[9], storm = x[10], T = vector(14), L = matrix(8, 4),
    c = sin(lat * DGTR), s = cos(lat * DGTR), c2, c4, s2, cd14, cd18, cd32, cd39, df, dfa, f1, f2,
    apdf, apt, p25, ex, G0, sg0, ctl, stl, c2tl, s2tl, c3tl, s3tl);
  c2 = c^2; c4 = c2^2; s2 = s^2;
  L[2,1] = c; L[3,1] = (3*c2 - 1) / 2; L[4,1] = (5*c*c2 - 3*c) / 2;
  L[5,1] = (35*c4 - 30*c2 + 3) / 8; L[6,1] = (63*c2*c2*c - 70*c2*c + 15*c) / 8;
  L[7,1] = (11*c*L[6,1] - 5*L[5,1]) / 6;
  L[2,2] = s; L[3,2] = 3*c*s; L[4,2] = 3/2 * (5*c2 - 1) * s; L[5,2] = 5/2 * (7*c2*c - 3*c) * s;
  L[6,2] = 15/8 * (21*c4 - 14*c2 + 1) * s; L[7,2] = (11*c*L[6,2] - 6*L[5,2]) / 5;
  L[3,3] = 3*s2; L[4,3] = 15*s2*c; L[5,3] = 15/2 * (7*c2 - 1) * s2;
  L[6,3] = 3*c*L[5,3] - 2*L[4,3]; L[7,3] = (11*c*L[6,3] - 7*L[5,3]) / 4;
  L[8,3] = (13*c*L[7,3] - 8*L[6,3]) / 5;
  L[4,4] = 15*s2*s; L[5,4] = 105*s2*s*c; L[6,4] = (9*c*L[5,4] - 7*L[4,4]) / 2;
  L[7,4] = (11*c*L[6,4] - 8*L[5,4]) / 3;
  ctl = cos(HR*tloc); stl = sin(HR*tloc); c2tl = cos(2*HR*tloc); s2tl = sin(2*HR*tloc);
  c3tl = cos(3*HR*tloc); s3tl = sin(3*HR*tloc);
  cd14 = cos(DR*(day - P[14])); cd18 = cos(2*DR*(day - P[18]));
  cd32 = cos(DR*(day - P[32])); cd39 = cos(2*DR*(day - P[39]));
  df = f107 - f107a; dfa = f107a - 150;
  T[1] = P[20]*df*(1 + P[60]*dfa) + P[21]*df^2 + P[22]*dfa + P[30]*dfa^2;
  f1 = 1 + P[48]*dfa + P[20]*df + P[21]*df^2;
  f2 = 1 + P[50]*dfa + P[20]*df + P[21]*df^2;
  T[2] = P[2]*L[3,1] + P[3]*L[5,1] + P[23]*L[7,1] + P[15]*L[3,1]*dfa + P[27]*L[2,1];
  T[3] = P[19]*cd32;
  T[4] = (P[16] + P[17]*L[3,1])*cd18;
  T[5] = f1*(P[10]*L[2,1] + P[11]*L[4,1])*cd14;
  T[6] = P[38]*L[2,1]*cd39;
  T[7] = f2*((P[4]*L[2,2] + P[5]*L[4,2] + P[28]*L[6,2] + P[12]*L[3,2]*cd14)*ctl
           + (P[7]*L[2,2] + P[8]*L[4,2] + P[29]*L[6,2] + P[13]*L[3,2]*cd14)*stl);
  T[8] = f2*((P[6]*L[3,3] + P[42]*L[5,3] + (P[24]*L[4,3] + P[36]*L[6,3])*cd14)*c2tl
           + (P[9]*L[3,3] + P[43]*L[5,3] + (P[34]*L[4,3] + P[37]*L[6,3])*cd14)*s2tl);
  T[14] = f2*((P[40]*L[4,4] + (P[94]*L[5,4] + P[47]*L[7,4])*cd14)*s3tl
            + (P[41]*L[4,4] + (P[95]*L[5,4] + P[49]*L[7,4])*cd14)*c3tl);
  if (!storm,
    my(apd = ap - 4, p44 = P[44], p45 = P[45]);
    if (p44 < 0, p44 = 1/10^5);
    apdf = apd + (p45 - 1)*(apd + (exp(-p44*apd) - 1)/p44);
    T[9] = apdf*(P[33] + P[46]*L[3,1] + P[35]*L[5,1]
                 + (P[101]*L[2,1] + P[102]*L[4,1] + P[103]*L[6,1])*cd14
                 + (P[122]*L[2,2] + P[123]*L[4,2] + P[124]*L[6,2])*cos(HR*(tloc - P[125])));
    T[13] = apdf*(1 + P[121]*L[2,1])*(P[61]*L[3,2] + P[62]*L[5,2] + P[63]*L[7,2])*cos(DGTR*(lon - P[64]))
          + apdf*(P[116]*L[2,2] + P[117]*L[4,2] + P[118]*L[6,2])*cd14*cos(DGTR*(lon - P[119]))
          + apdf*(P[84]*L[2,1] + P[85]*L[4,1] + P[86]*L[6,1])*cos(SR*(sec - P[76])),
  /* else */
    if (P[52] != 0,
      ex = exp(-10800*abs(P[52])/(1 + P[139]*(45 - abs(lat))));
      if (ex > 99999/100000, ex = 99999/100000);
      p25 = max(P[25], 1/10^4);
      G0 = (a -> a - 4 + (P[26] - 1)*(a - 4 + (exp(-p25*(a - 4)) - 1)/p25));
      sg0 = (G0(aparr[2]) + (G0(aparr[3])*ex + G0(aparr[4])*ex^2 + G0(aparr[5])*ex^3
             + (G0(aparr[6])*ex^4 + G0(aparr[7])*ex^12)*(1 - ex^8)/(1 - ex)))
            / (1 + (1 - ex^19)/(1 - ex)*sqrt(ex));
      apt = sg0;
      T[9] = apt*(P[51] + P[97]*L[3,1] + P[55]*L[5,1]
                  + (P[126]*L[2,1] + P[127]*L[4,1] + P[128]*L[6,1])*cd14
                  + (P[129]*L[2,2] + P[130]*L[4,2] + P[131]*L[6,2])*cos(HR*(tloc - P[132])));
      T[13] = apt*(1 + P[133]*L[2,1])*(P[53]*L[3,2] + P[99]*L[5,2] + P[68]*L[7,2])*cos(DGTR*(lon - P[98]))
            + apt*(P[134]*L[2,2] + P[135]*L[4,2] + P[136]*L[6,2])*cd14*cos(DGTR*(lon - P[137]))
            + apt*(P[56]*L[2,1] + P[57]*L[4,1] + P[58]*L[6,1])*cos(SR*(sec - P[59]))));
  T[11] = (1 + P[81]*dfa)*((P[65]*L[3,2] + P[66]*L[5,2] + P[67]*L[7,2] + P[104]*L[2,2]
                            + P[105]*L[4,2] + P[106]*L[6,2]
                            + (P[110]*L[2,2] + P[111]*L[4,2] + P[112]*L[6,2])*cd14)*cos(DGTR*lon)
                           + (P[91]*L[3,2] + P[92]*L[5,2] + P[93]*L[7,2] + P[107]*L[2,2]
                              + P[108]*L[4,2] + P[109]*L[6,2]
                              + (P[113]*L[2,2] + P[114]*L[4,2] + P[115]*L[6,2])*cd14)*sin(DGTR*lon));
  T[12] = (1 + P[96]*L[2,1])*(1 + P[82]*dfa)*(1 + P[120]*L[2,1]*cd14)
          *(P[69]*L[2,1] + P[70]*L[4,1] + P[71]*L[6,1])*cos(SR*(sec - P[72]))
        + (P[77]*L[4,3] + P[78]*L[6,3] + P[79]*L[8,3])*cos(SR*(sec - P[80]) + 2*DGTR*lon)*(1 + P[138]*dfa);
  P[31] + vecsum(T);}

/* GTS7 above ZA: [Tinf, Tlb, g0]. */
profile(x) = [PTM[1]*PT[1]*(1 + globe7(PT, x)), PTM[2]*(1 + globe7(PD4, x))*PD4[1], PTM[4]*PS[1]*(1 + globe7(PS, x))];

/* [name, day, sec, lat, lon, tloc, f107a, f107, ap, aparr, storm]; rows C and D take the local
   time the bridges derive, sec/3600 + lon/15. */
{rows = [
  ["A", 253, 2507, decimal("-54.04366595"), decimal("141.81501274"), decimal("10.199"), decimal("98.8"),
   decimal("707.6"), 33, [33, 9, 18, 32, 32, decimal("8.875"), decimal("7.25")], 1],
  ["B", 341, 1786, decimal("35.51414905"), decimal("59.92934048"), decimal("4.638"), decimal("91.5"),
   decimal("573.4"), 25, [25, 48, 22, 39, 18, decimal("17.625"), 2], 1],
  ["C", 253, 57600, -5, 0, 57600/3600 + 0/15, decimal("98.8"), decimal("707.6"), 33, [], 0],
  ["D", 341, 79200, 35, 105, 79200/3600 + 105/15, decimal("91.5"), decimal("573.4"), 25, [], 0]];}
args(r) = vector(10, k, r[k + 1]);
/* Tinf - Tlb at the inputs x with the daily F10.7 set to F. */
gap(x, F) = my(y = x, w); y[7] = F; w = profile(y); w[1] - w[2];

/* The mod.rs literal `name`, its value and its count of decimals. */
{rust_const(name) = my(p = Str("    const ", name, ": f64 = "), l, t, d);
  l = module[line_of(module, p)];
  t = strjoin(Vec(l)[#p + 1 .. #l - 1]);
  d = strsplit(t, ".");
  [decimal(t), #d[2]];}
/* `name` holds v rounded to the digits it shows. */
holds(name, v) = my(c = rust_const(name)); abs(c[1] - v) <= 10^-c[2] / 2;

{for (i = 1, #rows,
  my(r = rows[i], x = args(r), v = profile(x), boundary, lo, hi);
  printf("row %s: Tinf = %.20f K, Tlb = %.20f K, g0 = %.20f K/km\n", r[1], v[1], v[2], v[3]);
  print("row ", r[1], " Tinf < Tlb, so no profile: ", if (v[1] < v[2], "ok", "FAIL"));
  print("row ", r[1], " Tlb > 0 and g0 > 0, so Tinf alone breaks the profile: ", if (v[2] > 0 && v[3] > 0, "ok", "FAIL"));
  lo = x[6]; hi = x[7];
  print("row ", r[1], " in the domain at F10.7a: ", if (gap(x, lo) > 0, "ok", "FAIL"));
  boundary = solve(F = lo, hi, gap(x, F));
  printf("row %s: Tinf = Tlb at F10.7 = %.20f sfu\n", r[1], boundary);
  foreach ([["T_INF", v[1]], ["T_LB", v[2]], ["BOUNDARY", boundary]], q,
    my(name = Str("ROW_", r[1], "_", q[1]));
    print("mod.rs ", name, " is this value to its digits: ", if (holds(name, q[2]), "ok", "FAIL")));
  if (r[1] == "C", print("row C's Tinf is below 0 K: ", if (v[1] < 0, "ok", "FAIL"))));}
