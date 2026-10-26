\\ Second engine for flower_phasing.mac: the repeating-orbit semi-major axis at 60 digits for the
\\ fixtures of flower.rs, and Kepler's third law at that precision (prints `<name>: 0`).
\\ Run: timeout 120 gp -q -f derivations/flower_phasing.gp </dev/null
default(realprecision, 60);
mu = 3986004418 * 10^5;
w = 72921151465 / 10^15;
{
  foreach([[3, 1], [14, 3], [2, 1], [13, 1]], c,
    my(n = c[1]*w/c[2], a = (mu/n^2)^(1/3));
    printf("Np=%d Nd=%d a %.20e\n", c[1], c[2], a);
    printf("kepler_np%d_nd%d: %s\n", c[1], c[2], if(abs(n^2*a^3 - mu)/mu < 1e-55, 0, n^2*a^3 - mu)));
}
