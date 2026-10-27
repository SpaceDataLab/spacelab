/* The rational linear-covariance scenario of derivations/lincov_recursion.mac, in exact rationals by
   a different route: the augmented covariance of [x; e] propagated as one matrix,

       time:         P+ = A P A' + G Q G',    A = [Phi, 0; D, Phih],  G = [I; T],  D = T Phi - Phih T
       measurement:  P+ = B P B' + g R g',    B = [I, 0; -K Delta, I - K Hh],  g = [0; -K],

   with K the filter's own gain from its own Ph and Rh, against the block-by-block recursion the Maxima
   file transcribes from the code. Agreement is exact equality of rationals.

       timeout 300 gp -q -f derivations/lincov_recursion.gp </dev/null                              */

default(realprecision, 60);

Phi = [1, 1/2, 1/8; 0, 1, 1/4; 0, 0, 99/100];
Phih = [1, 51/100; 0, 1];
T = [1, 0, 0; 0, 1, 1/2];
Q = [1/1000, 0, 1/5000; 0, 2/1000, 0; 1/5000, 0, 3/1000];
Qh = [1/2000, 0; 0, 1/1000];
Cxx = [16, 1, 0; 1, 4, 1/2; 0, 1/2, 1];
P0 = [9, 1/2; 1/2, 2];
H = Mat([1, 1/10, 1/20]);
Hh = Mat([1, 0]);
R = 9/100; Rh = 4/100;

D = T * Phi - Phih * T;
A = matconcat([Phi, matrix(3, 2); D, Phih]);
G = matconcat([matid(3); T]);
P = matconcat([Cxx, matrix(3, 2); matrix(2, 3), P0]);
P = A * P * A~ + G * Q * G~;
Ph = Phih * P0 * Phih~ + Qh;

f = (Hh * Ph * Hh~)[1, 1] + Rh;
K = Ph * Hh~ / f;
Delta = H - Hh * T;
B = matconcat([matid(3), matrix(3, 2); -K * Delta, matid(2) - K * Hh]);
g = matconcat([matrix(3, 1); -K]);
Pt = P;
P = B * P * B~ + g * R * g~;
Ph = (matid(2) - K * Hh) * Ph * (matid(2) - K * Hh)~ + K * Rh * K~;

/* The values the Maxima file prints, exactly. */
cross = P[4..5, 1..3];
err = P[4..5, 4..5];
{
my(bad = 0);
/* The Maxima prints carry 40 significant digits; compare at that resolution. */
for (i = 1, 2, for (j = 1, 3,
  my(w = [-3.331813938703367193938852314139037008351e-1, -4.672649619688800182708252653738071832147e-1,
          -1.234448979216936260637294329093310296206e-1; 5.336296707279533696763879372834063173364e-2,
          1.359685548174406942913600842046729621576e-1, 2.455661210243577904217184505545791255821e-1][i, j]);
  if (abs(cross[i, j] - w) > 10^-38 * abs(w), bad++)));
my(we = [1.422149419746660870423272900927262942843e-1, -1.234850701363867313417009073839060743026e-2;
         -1.234850701363867313417009073839060743026e-2, 1.835866321618709469076320002220917218209e0]);
my(wp = [3.984112325856196689405900284985154954472e-2, 6.037316174645258025757891705641117300684e-3;
         6.037316174645258025757891705641117300684e-3, 1.771581985363480195021200115185637542574e0]);
for (i = 1, 2, for (j = 1, 2,
  if (abs(err[i, j] - we[i, j]) > 10^-38 * abs(we[i, j]), bad++);
  if (abs(Ph[i, j] - wp[i, j]) > 10^-38 * abs(wp[i, j]), bad++)));
if (P[1..3, 1..3] != Pt[1..3, 1..3], bad++);
if (P[1..3, 4..5] != cross~, bad++);
printf("innovation variance %s (exact %s)\n", f, f * 1.);
printf("Cee = %s\n", err);
printf("%d of 16 checks off the Maxima block recursion by more than 1e-38 relative\n", bad);
}
