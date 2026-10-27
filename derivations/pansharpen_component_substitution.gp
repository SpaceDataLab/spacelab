/* Second engine for derivations/pansharpen_component_substitution.mac: the three-band example from
   the closed form of Gram-Schmidt sharpening, out_k = MS_k + g_k (lambda (PAN - mean PAN) - S~),
   g_k = cov(MS_k, S)/var(S), lambda = sigma(S)/sigma(PAN) -- no orthogonalisation is run -- and
   Brovey from its definition in exact rationals.

   The summary counts the values off by more than 1e-30 from Maxima's (Brovey: not equal).

       timeout 300 gp -q -f derivations/pansharpen_component_substitution.gp </dev/null           */

default(realprecision, 60);
off = 0; checked = 0;
{
chk(x, y) =
  checked++;
  if (abs(x - y) > 1e-30, off++; print("  off: ", x, " vs ", y));
}
{
same(x, y) =
  checked++;
  if (x != y, off++; print("  off: ", x, " vs ", y));
}
mean(v) = vecsum(v)/#v;
centre(v) = v - vector(#v, i, mean(v));
cov(u, v) = (centre(u)*centre(v)~)/#u;
{
my(b = [[1/2, 3/4, 5/8, 1, 7/8, 3/8], [1/4, 1/2, 3/4, 5/8, 1/2, 1/4], [3/8, 1/4, 1/2, 7/8, 3/4, 5/8]],
   pan = [5/8, 1/4, 7/8, 3/4, 1, 1/2], s, lam, detail,
   want = [[0.7039941959533993291507205629459281660903, 0.5182676476073877869200496661565293673245,
            0.7286528734440125285583237907749463255121, 0.7629978466253114334416781401631895393791,
            0.9773354231984934401887309092766113685257, 0.4337520131713954817404969306827952331684],
           [0.398446480623365709006392897918398430723, 0.331368475958427732265763841569305408147,
            0.8254281473418870512983107397658060115229, 0.4525336442578557379505169564567811202055,
            0.5744694394167440527194990184876749395375, 0.2927538124017197167595165458020340898643],
           [0.5445163036795853580266551156874614337933, 0.05743367899768844265187225779204553059363,
            0.5861340779323484392245225867003075099326, 0.6780545486041320362402677502764532792024,
            0.8350392953339593376216214598214094470848, 0.6738220954522863862350608297223227993934]],
   brovey = [[5/18, 1/8, 7/24, 3/10, 7/17, 3/20], [5/36, 1/12, 7/20, 3/16, 4/17, 1/10],
             [5/24, 1/24, 7/30, 21/80, 6/17, 1/4]]);
s = (b[1] + b[2] + b[3])/3;
lam = sqrt(cov(s, s)/cov(pan, pan));
detail = lam*centre(pan) - centre(s);
for (k = 1, 3,
  my(o = b[k] + cov(b[k], s)/cov(s, s)*detail);
  for (p = 1, 6, chk(o[p], want[k][p]));
  for (p = 1, 6, same(b[k][p]*pan[p]/(b[1][p] + b[2][p] + b[3][p]), brovey[k][p])));
}
printf("pansharpen_component_substitution: %d values checked, %d off by more than 1e-30\n", checked, off);
