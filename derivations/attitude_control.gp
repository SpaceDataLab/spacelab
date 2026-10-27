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
