/* Second engine for derivations/gpt3_evaluation.mac, at 100 digits.

   The same definition by another route: every node of the vendored grid is keyed by its own
   coordinate columns (not by TU Wien's line arithmetic, which the Maxima file uses), the site's cell
   is found from those keys, and each entry of GPT3_MAXIMA in crates/spacelab-gnss/tests/
   gpt3_maxima.rs must equal this evaluation to within its 17 printed digits: rounding to 17 digits
   moves a value by at most 5e-17 of itself, so 1e-16 of the field's scale is the bound.

       gp -q -f derivations/gpt3_evaluation.gp </dev/null                                         */

default(realprecision, 100);

starts(s, p) = #s >= #p && Strchr(Vecsmall(s)[1 .. #p]) == p;
tok(s) = my(v = strsplit(s, " "), out = List()); for (i = 1, #v, if (#v[i], listput(out, v[i]))); Vec(out);

L = readstr("../crates/spacelab-gnss/assets/gpt3_5.grd");
hdr = tok(L[1]); hdr = hdr[2 .. #hdr];
col(lab) = for (i = 1, #hdr, if (hdr[i] == lab, return(i))); error("no column ", lab);

/* nodes keyed by tenths of a degree, longitude folded into [0, 3600) */
nodes = Map();
{
for (k = 2, #L,
  my(t = tok(L[k]));
  if (#t, mapput(nodes, [round(eval(t[1])*10), round(eval(t[2])*10) % 3600], t)));
}
if (#nodes != 36*72, print("FAIL: ", #nodes, " nodes"));
node(lat, lon) = mapget(nodes, [lat*10, (lon*10) % 3600]);

seasonal(t, lab, scale, d) = {
  my(c = col(lab), th = 2*Pi*d/(36525/100));
  scale*(eval(t[c]) + eval(t[c + 1])*cos(th) + eval(t[c + 2])*sin(th)
         + eval(t[c + 3])*cos(2*th) + eval(t[c + 4])*sin(2*th));
}

corner(t, d, h) = {
  my(dh = h - eval(t[col("undu")]) - eval(t[col("Hs")]),
     T0 = seasonal(t, "T:a0", 1, d), Q = seasonal(t, "Q:a0", 1/1000, d),
     dT = seasonal(t, "dT:a0", 1/1000, d), p0 = seasonal(t, "p:a0", 1, d),
     la = seasonal(t, "lambda:a0", 1, d), Tv, f, e0);
  Tv = T0*(1 + 6077/10000*Q);
  f = -dh*(980665/100000)*(28965/1000000)/((83143/10000)*Tv);
  e0 = Q*p0/(622/1000 + 378/1000*Q);
  [seasonal(t, "a_h:a0", 1/1000, d), seasonal(t, "a_w:a0", 1/1000, d), p0*exp(f), T0 + dT*dh,
   e0*exp(f*(1 + la)), seasonal(t, "Tm:a0", 1, d), la,
   seasonal(t, "Gn_h:a0", 1/100000, d), seasonal(t, "Ge_h:a0", 1/100000, d),
   seasonal(t, "Gn_w:a0", 1/100000, d), seasonal(t, "Ge_w:a0", 1/100000, d)];
}

/* the south-west node: the last node at or below the site, the row held to the 35 full cells */
gpt3_at(lat, lon, d, h) = {
  my(lo = lon - 360*floor((lon + 355/2)/360), la0, lo0, t, u);
  la0 = min(165/2, max(-175/2, 5*floor((lat - 5/2)/5) + 5/2));
  lo0 = 5*floor((lo - 5/2)/5) + 5/2;
  t = (lo - lo0)/5; u = (lat - la0)/5;
  (1 - t)*(1 - u)*corner(node(la0, lo0), d, h) + (1 - t)*u*corner(node(la0 + 5, lo0), d, h)
    + t*(1 - u)*corner(node(la0, lo0 + 5), d, h) + t*u*corner(node(la0 + 5, lo0 + 5), d, h);
}

/* the table as the Rust test holds it: the lines from `const GPT3_MAXIMA` to `];`, numbers only */
src = readstr("../crates/spacelab-gnss/tests/gpt3_maxima.rs");
body = ""; inside = 0;
{
for (k = 1, #src,
  my(s = src[k]);
  if (inside && starts(s, "];"), break);
  if (inside, body = concat(body, s));
  if (starts(s, "const GPT3_MAXIMA:"), inside = 1));
}
clean = Strchr(select(c -> c != 91 && c != 93 && c != 32, Vec(Vecsmall(body))));
vals = apply(s -> eval(s), select(s -> #s, strsplit(clean, ",")));
if (#vals != 150, print("FAIL: table has ", #vals, " entries"));
scale = [1e-3, 1e-4, 1e5, 1e2, 1e1, 1e2, 1, 1e-5, 1e-5, 1e-5, 1e-5];

worst = 0;
{
for (r = 0, #vals/15 - 1,
  my(row = vals[15*r + 1 .. 15*r + 15], inp, got);
  inp = apply(x -> bestappr(x, 10^6), row[1 .. 4]);
  got = gpt3_at(inp[1], inp[2], inp[3], inp[4]);
  for (k = 1, 11, worst = max(worst, abs(got[k] - row[4 + k])/max(abs(got[k]), scale[k]))));
}
print("worst relative difference against the table: ", worst);
{
if (#nodes == 36*72 && #vals == 150 && worst < 10^-16,
  print("gpt3 evaluation holds at 100 digits"),
  print("FAIL: gpt3 evaluation"));
}
