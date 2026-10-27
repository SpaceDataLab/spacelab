/* Second engine for derivations/relativistic_aberration.mac: the apparent direction as a Lorentz
   boost of the photon's four-momentum, not the coded closed form.

   A photon from direction n travels along -n with energy 1. In the frame of an observer moving at
   beta, its momentum along beta-hat becomes gamma (p_par - |beta|) and across it is unchanged; the
   apparent direction is minus the boosted momentum, normalised.

   Compared against the 40-digit directions Maxima printed; the summary counts the ones off by more
   than 1e-30.

       timeout 300 gp -q -f derivations/relativistic_aberration.gp </dev/null                     */

default(realprecision, 60);
off = 0; checked = 0;
{
chk(x, y) =
  checked++;
  if (abs(x - y) > 1e-30, off++; print("  off: ", x, " vs ", y));
}
chkv(u, v) = for (i = 1, #u, chk(u[i], v[i]));
{
boost(n, beta) =
  my(nn = n/sqrt(n*n~), b = sqrt(beta*beta~), bh = beta/b, gam = 1/sqrt(1 - b^2), p, ppar, pperp, q);
  p = -nn; ppar = p*bh~; pperp = p - ppar*bh;
  q = pperp + gam*(ppar - b)*bh;
  -q/sqrt(q*q~);
}
{
dirs = [[1, 2, 2]/3, [-3/5, 0, 4/5], [2, -3, 6]/7];
fast = [36/100, 48/100, 0];
earth = [12000, -25000, 3000]/299792458;
wfast = [[0.4962962962962962962962962962962962962963, 0.7851851851851851851851851851851851851852,
          0.3703703703703703703703703703703703703704],
         [-0.208163265306122448979591836734693877551, 0.5387755102040816326530612244897959183673,
          0.816326530612244897959183673469387755102],
         [0.6331210191082802547770700636942675159236, 0.1222929936305732484076433121019108280255,
          0.7643312101910828025477707006369426751593]];
wearth = [[0.333385220772806913559202147607236864455, 0.6666069922311377980383385435637642765582,
           0.6667003918394098146555189730784553921878],
          [-0.5999695761909179721722899270901547019765, -0.00008339169140025920609533872348400794112357,
           0.8000228126066881598930190268833874486706],
          [0.2857383825540365113087487091298228592199, -0.4286309226668546437503102937878463758485,
           0.8571050745790529719295063283790830245722]];
for (k = 1, #dirs, chkv(boost(dirs[k], fast), wfast[k]); chkv(boost(dirs[k], earth), wearth[k]));
}
printf("relativistic_aberration: %d values checked, %d off by more than 1e-30\n", checked, off);
