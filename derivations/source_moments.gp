/* Second engine for derivations/source_moments.mac.

   K = 1/Phi^-1(3/4) by root-finding Phi(x) = 1 - erfc(x/sqrt 2)/2 (Maxima used erf and a bigfloat
   bracket), the frame's median and MAD by vecsort, the blob set by the cut, and each blob's moments
   from the frame itself -- flux, first and second moments summed over the pixels above the cut,
   not over the blob matrices Maxima summed.

   Compared against Maxima's values: exact rationals with ==, the rest to 1e-30. The summary counts
   the ones that differ.

       timeout 300 gp -q -f derivations/source_moments.gp </dev/null                              */

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

K = 1/solve(x = 1/2, 1, 1 - erfc(x/sqrt(2))/2 - 3/4);
chk(K, 1.4826022185056018605470765293604234313267032);

PA = [20, 40, 10; 60, 100, 30; 20, 50, 10];
PB = 10*[1; 3; 3; 1]*[1, 3, 3, 1];
{
pix(x, y) =
  if (x >= 3 && x <= 5 && y >= 4 && y <= 6, 10 + PA[y - 3, x - 2],
      x >= 10 && x <= 13 && y >= 9 && y <= 12, 10 + PB[y - 8, x - 9],
      10 + (x + 2*y) % 3 - 1);
}
med(L) = my(s = vecsort(L), n = #L); if (n % 2 == 0, (s[n/2] + s[n/2 + 1])/2, s[(n + 1)/2]);
{
my(frame = vector(256, i, pix((i - 1) % 16, (i - 1) \ 16)), level, mad, cut);
level = med(frame);
mad = med(apply(v -> abs(v - level), frame));
same(level, 10); same(mad, 1);
cut = level + 3*K*mad;
/* each blob: the pixels above the cut inside its bounding box, and nothing above it outside */
my(outside = 0);
for (i = 0, 255, my(x = i % 16, y = i \ 16, inA = x >= 3 && x <= 5 && y >= 4 && y <= 6,
                    inB = x >= 10 && x <= 13 && y >= 9 && y <= 12);
  if ((pix(x, y) > cut) != (inA || inB), outside++));
same(outside, 0);
my(blobs = [[3, 5, 4, 6], [10, 13, 9, 12]],
   want = [[340, 131/34, 171/34, 497/1156, 0.655691082635528043256204619557255867220608971,
            0.00857760929796859229874263641580077520121276613, 76.4421716888892642295857283146814810105158338],
           [640, 23/2, 21/2, 3/4, 0.866025403784438646763723170752936183471402627,
            0.00802481990583136491882037428747667086261233998, 107.918360031373078912356322326609149661904707]]);
for (b = 1, 2,
  my(bx = blobs[b], fl = 0, sx = 0, sy = 0, n = 0, cx, cy, m2 = 0, wd, Q);
  for (y = bx[3], bx[4], for (x = bx[1], bx[2],
    my(v = pix(x, y) - level); fl += v; sx += v*x; sy += v*y; n++));
  cx = sx/fl; cy = sy/fl;
  for (y = bx[3], bx[4], for (x = bx[1], bx[2],
    m2 += (pix(x, y) - level)*((x - cx)^2 + (y - cy)^2)));
  wd = sqrt(m2/(2*fl)); Q = fl/(K*mad*sqrt(n));
  same(fl, want[b][1]); same(cx, want[b][2]); same(cy, want[b][3]); same(m2/(2*fl), want[b][4]);
  chk(wd, want[b][5]); chk(wd/Q, want[b][6]); chk(Q, want[b][7]));
}
printf("source_moments: %d values checked, %d off by more than 1e-30\n", checked, off);
