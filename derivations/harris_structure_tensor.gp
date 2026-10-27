/* Second engine for derivations/harris_structure_tensor.mac: the Harris responses of the 12 x 12
   pattern from the structure tensor's eigenvalues, R = l1 l2 - k (l1 + l2)^2, with the tensor
   built from a replicate-padded copy of the image and the Sobel kernels as 3 x 3 matrices, and
   the eigenvalues taken by polroots of its characteristic polynomial.

   The summary counts the responses off by more than 1e-30 from Maxima's exact values.

       timeout 300 gp -q -f derivations/harris_structure_tensor.gp </dev/null                     */

default(realprecision, 60);
off = 0; checked = 0;
{
chk(x, y) =
  checked++;
  if (abs(x - y) > 1e-30, off++; print("  off: ", x, " vs ", y));
}
H = 12; W = 12; P = 2;
cl(v, n) = max(0, min(n - 1, v));
/* image padded by two pixels on every side; entry [i, j] is image row i - 1 - P, column j - 1 - P */
img = matrix(H + 2*P, W + 2*P, i, j, my(y = cl(i - 1 - P, H), x = cl(j - 1 - P, W)); (7*y + 13*x + 3*x*y) % 19);
KX = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
KY = [-1, -2, -1; 0, 0, 0; 1, 2, 1];
{
grad(K, y, x) =
  my(s = 0, yy = cl(y, H), xx = cl(x, W));
  for (a = -1, 1, for (b = -1, 1, s += K[a + 2, b + 2]*img[yy + 1 + P + a, xx + 1 + P + b]));
  s;
}
{
response(y, x, k) =
  my(T = matrix(2, 2), r, gxv, gyv);
  for (a = -1, 1, for (b = -1, 1,
    gxv = grad(KX, y + a, x + b); gyv = grad(KY, y + a, x + b);
    T += [gxv^2, gxv*gyv; gxv*gyv, gyv^2]));
  r = real(polroots(charpoly(T)));
  r[1]*r[2] - k*(r[1] + r[2])^2;
}
{
my(want = [[0, 0, 81271807/4], [0, 11, 5636567/4], [11, 0, -1469020], [11, 11, 2388343/4],
           [3, 3, 14053503/4], [5, 7, 1302176], [6, 6, 4149183], [8, 4, 29572783/4],
           [2, 9, 36422887/4], [9, 9, 12361671/4]]);
for (i = 1, #want, chk(response(want[i][1], want[i][2], 1/16), want[i][3]));
}
printf("harris_structure_tensor: %d responses checked, %d off by more than 1e-30\n", checked, off);
