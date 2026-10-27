/* Second engine for derivations/detection_box_geometry.mac: the fifteen IoUs of the six-box fixture
   by counting the unit cells each integer box covers, not by the min/max formula, and the decoded
   box from the anchor's corners.

   The summary counts the values that differ from Maxima's exact rationals or, for the decoded box,
   are off by more than 1e-30.

       timeout 300 gp -q -f derivations/detection_box_geometry.gp </dev/null                      */

default(realprecision, 60);
off = 0; checked = 0;
{
same(x, y) =
  checked++;
  if (x != y, off++; print("  off: ", x, " vs ", y));
}
{
chk(x, y) =
  checked++;
  if (abs(x - y) > 1e-30, off++; print("  off: ", x, " vs ", y));
}
cells(b, x, y) = x >= b[1] && x < b[3] && y >= b[2] && y < b[4];
{
iou_cells(a, b) =
  my(i = 0, u = 0);
  for (x = 0, 300, for (y = 0, 300,
    my(p = cells(a, x, y), q = cells(b, x, y)); i += p && q; u += p || q));
  if (u, i/u, 0);
}
{
my(boxes = [[10, 10, 50, 50], [12, 11, 52, 51], [90, 90, 140, 140], [92, 93, 141, 139],
            [30, 30, 70, 70], [200, 200, 260, 255]],
   want = [741/859, 0, 0, 1/7, 0, 0, 0, 231/1369, 0, 1104/1273, 0, 0, 0, 0, 0], n = 0);
for (i = 1, 6, for (j = i + 1, 6, n++; same(iou_cells(boxes[i], boxes[j]), want[n])));
}
{
my(a = [20, 40, 60, 100], d = [3/10, -1/5, 7/10, -2/5], wa = a[3] - a[1], ha = a[4] - a[2],
   cx, cy, w, h);
cx = (a[1] + a[3])/2 + d[1]*wa; cy = (a[2] + a[4])/2 + d[2]*ha;
w = wa*exp(d[3]); h = ha*exp(d[4]);
chk(cx - w/2, 11.72494585059046956750901222833869459965);
chk(cy - h/2, 37.89039861893082097766701224556521784189);
chk(cx + w/2, 92.27505414940953043249098777166130540035);
chk(cy + h/2, 78.10960138106917902233298775443478215811);
}
printf("detection_box_geometry: %d values checked, %d differ\n", checked, off);
