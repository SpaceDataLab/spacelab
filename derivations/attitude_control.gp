/* Second engine for derivations/attitude_control.mac, in exact rationals at random states rather
   than symbolically. Unit quaternions with rational components come from the inverse stereographic
   map q = (1 - t.t, 2 t)/(1 + t.t), t in Q^3, so nothing is rounded:
     - Shepperd: on R(q), candidate k is 4 q_k^2 and the branch numerators are 4 q_i q_k;
     - kinematics: d/dh R(q + h qdot) at h = 0 is R [w]x for qdot = 1/2 q (x) (0, w);
     - the MRP kinematics sigma' = 1/4 B(sigma) w, sigma = v/(1 + q0), from qdot;
     - the Lyapunov derivative of the Schaub-Junkins law, V' = w.I w' + 4K sigma.sigma'/(1 + sigma.sigma),
       equals -P w.w for random symmetric positive inertias and gains.
   The error rotation: Rr^T R has body-rate kinematics and R Rr^T does not, by a random example.

       timeout 300 gp -q -f derivations/attitude_control.gp </dev/null                          */

seed = 7;
rnd() = seed = (seed * 6364136223846793005 + 1442695040888963407) % 2^61; (seed % 2001 - 1000) / 97;
skew(a) = [0, -a[3], a[2]; a[3], 0, -a[1]; -a[2], a[1], 0];
cross3(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
rq(q) = my(q0 = q[1], v = q[2..4]); (q0^2 - v * v~) * matid(3) + 2 * v~ * v + 2 * q0 * skew(v);
unitq() = my(t = [rnd(), rnd(), rnd()], n = 1 + t * t~); concat([(1 - t * t~) / n], 2 * t / n);
qdot(q, w) = my(v = q[2..4]); concat([-(v * w~) / 2], (q[1] * w + cross3(v, w)) / 2);

{off_shepperd = 0; off_kin = 0; off_mrp = 0; off_lyap = 0; off_err = 0; trials = 50;
for (k = 1, trials,
  my(q = unitq(), w = [rnd(), rnd(), rnd()], M, tr, cands, num, h, D, qd, v, sig, sigd, B,
     L, I, K, P, Iw, u, wd, Vd, Rr, R, E1, E2);
  M = rq(q); tr = trace(M);
  cands = [1 + tr, 1 + 2*M[1,1] - tr, 1 + 2*M[2,2] - tr, 1 + 2*M[3,3] - tr];
  for (i = 1, 4, off_shepperd += (cands[i] != 4 * q[i]^2));
  num = [M[3,2] - M[2,3] - 4*q[1]*q[2], M[1,3] - M[3,1] - 4*q[1]*q[3], M[2,1] - M[1,2] - 4*q[1]*q[4],
         M[1,2] + M[2,1] - 4*q[2]*q[3], M[1,3] + M[3,1] - 4*q[2]*q[4], M[2,3] + M[3,2] - 4*q[3]*q[4]];
  off_shepperd += #select(x -> x != 0, num);
  qd = qdot(q, w);
  D = matrix(3, 3, i, j, subst(deriv((rq(q + 'h * qd))[i, j], 'h), 'h, 0));
  off_kin += (D != M * skew(w));
  v = q[2..4]; sig = v / (1 + q[1]);
  sigd = qd[2..4] / (1 + q[1]) - v * qd[1] / (1 + q[1])^2;
  B = (1 - sig * sig~) * matid(3) + 2 * skew(sig) + 2 * sig~ * sig;
  off_mrp += (sigd~ != B * w~ / 4);
  L = matrix(3, 3, i, j, if (i >= j, rnd(), 0)); I = L * L~ + matid(3);
  K = abs(rnd()) + 1; P = abs(rnd()) + 1;
  Iw = (I * w~)~;
  u = -K * sig - P * w + cross3(w, Iw);
  wd = (I^-1 * (u - cross3(w, Iw))~)~;
  Vd = w * I * wd~ + 4 * K * (sig * sigd~) / (1 + sig * sig~);
  off_lyap += (Vd != -P * (w * w~));
  Rr = rq(unitq()); R = M;
  E1 = Rr~ * (R * skew(w)) - (Rr~ * R) * skew(w);
  E2 = (R * skew(w)) * Rr~ - (R * Rr~) * skew(w);
  off_err += (E1 != 0) + (E2 == 0));}
printf("%d random states\n", trials);
printf("Shepperd candidates or numerators unlike 4 q_i q_k: %d\n", off_shepperd);
printf("states where q' = 1/2 q (x) w does not give R' = R [w]x: %d\n", off_kin);
printf("states off the MRP kinematics: %d\n", off_mrp);
printf("states where V' + P w.w is not exactly zero: %d\n", off_lyap);
printf("states where Rr^T R lacks, or R Rr^T has, body-rate kinematics: %d\n", off_err);

/* Item 8's fixture (plans/18 Ф28.K, F119), from Rodrigues' formula here: exp(t [n]x) for t = 10^-k
   rounded to doubles, and the exact angle of those doubles through Shepperd's quaternion, against
   the 40 digits attitude_control.mac prints and attitude.rs's test reads at 25. */
default(realprecision, 100);
default(path, "derivations:."); read("binary64.gp");
\\ Shepperd's quaternion of an exact 3x3 matrix on the branch the code takes, sign-fixed to w >= 0,
\\ and from it [angle, rotation vector]; atan2(s, w) is arg(w + i s), w >= 0.
shep(m) = {
  my(tr = m[1, 1] + m[2, 2] + m[3, 3], b = 1, c, rt, h, q, s, t);
  c = [1 + tr, 1 + 2 * m[1, 1] - tr, 1 + 2 * m[2, 2] - tr, 1 + 2 * m[3, 3] - tr];
  for (i = 2, 4, if (c[i] > c[b], b = i));
  rt = sqrt(c[b]); h = 1 / (2 * rt);
  q = if (b == 1, [rt / 2, (m[3, 2] - m[2, 3]) * h, (m[1, 3] - m[3, 1]) * h, (m[2, 1] - m[1, 2]) * h],
      if (b == 2, [(m[3, 2] - m[2, 3]) * h, rt / 2, (m[1, 2] + m[2, 1]) * h, (m[1, 3] + m[3, 1]) * h],
      if (b == 3, [(m[1, 3] - m[3, 1]) * h, (m[1, 2] + m[2, 1]) * h, rt / 2, (m[2, 3] + m[3, 2]) * h],
                  [(m[2, 1] - m[1, 2]) * h, (m[1, 3] + m[3, 1]) * h, (m[2, 3] + m[3, 2]) * h, rt / 2])));
  if (q[1] < 0, q = -q);
  s = sqrt(q[2]^2 + q[3]^2 + q[4]^2);
  t = 2 * arg(q[1] + s * sqrt(-1));
  [t, t / s * [q[2], q[3], q[4]]];
}
rod(n, t) = { my(K = [0, -n[3], n[2]; n[3], 0, -n[1]; -n[2], n[1], 0]); matid(3) + sin(t) * K + (1 - cos(t)) * K^2; }
axes = [[2, -3, 6] / 7, [6, 2, -3] / 7, [-3, 6, 2] / 7];
{
SA = [9.999999999999999922211909320346093716959e-2,
  9.999999999999999853704149996642222866048e-3,
  1.000000000000000011880823284430709661740e-3,
  1.000000000000000011924442406521109310542e-4,
  1.000000000000000015998175552110046657130e-5,
  9.999999999999999961845499646950247053584e-7,
  1.000000000000000045410046869680871585447e-7,
  9.999999999999999416880321351070552065589e-9,
  1.000000000000000028963322798655232924639e-9,
  1.000000000000000013351547783698234295070e-10];
}
{
my(off = 0, worst = 0.);
for (k = 1, 10,
  my(n = axes[(k - 1) % 3 + 1], m = matrix(3, 3, i, j, f64(rod(n, 10^-k)[i, j])), d);
  d = abs(shep(m)[1] / SA[k] - 1); worst = max(worst, d); if (d > 1e-38, off++));
printf("small-angle error angles off Maxima's by more than 1e-38 relative: %d off (largest %.3g)\n", off, worst);
if (off, print("FAIL: the fixture disagrees with Maxima's"));
}
