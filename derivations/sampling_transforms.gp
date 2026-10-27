\\ Second engine for derivations/sampling_transforms.mac: SplitMix64 seeding and xoshiro256++ from
\\ Blackman & Vigna's recurrences in PARI/GP integers, against the stream sampling.rs pins; and the
\\ Box-Muller inverse u1 = exp(-(x^2 + y^2)/2), u2 = arg(x + i y)/(2 pi) mod 1 at 100 digits on a grid.
\\
\\ Run: timeout 300 gp -q -f derivations/sampling_transforms.gp </dev/null
default(realprecision, 100);
W = 2^64;
shr(a, k) = a >> k;
shl(a, k) = (a << k) % W;
rotl(a, k) = (shl(a, k) + shr(a, 64 - k)) % W;
splitmix(seed) =
{
  my(z = seed, s = vector(4), x);
  for (i = 1, 4,
    z = (z + 0x9e3779b97f4a7c15) % W;
    x = z;
    x = (bitxor(x, shr(x, 30)) * 0xbf58476d1ce4e5b9) % W;
    x = (bitxor(x, shr(x, 27)) * 0x94d049bb133111eb) % W;
    s[i] = bitxor(x, shr(x, 31)));
  s;
}
stream(seed, n) =
{
  my(s = splitmix(seed), out = vector(n), t);
  for (i = 1, n,
    out[i] = (rotl((s[1] + s[4]) % W, 23) + s[1]) % W;
    t = shl(s[2], 17);
    s[3] = bitxor(s[3], s[1]);
    s[4] = bitxor(s[4], s[2]);
    s[2] = bitxor(s[2], s[3]);
    s[1] = bitxor(s[1], s[4]);
    s[3] = bitxor(s[3], t);
    s[4] = rotl(s[4], 45));
  out;
}
pinned = [[0, [0x53175d61490b23df, 0x61da6f3dc380d507, 0x5c0fdf91ec9a7bfc, 0x02eebf8c3bbe5e1a]], \
          [1, [0xcfc5d07f6f03c29b, 0xbf424132963fe08d, 0x19a37d5757aaf520, 0xbf08119f05cd56d6]], \
          [42, [0xd0764d4f4476689f, 0x519e4174576f3791, 0xfbe07cfb0c24ed8c, 0xb37d9f600cd835b8]]];
off = 0;
{
  for (c = 1, #pinned,
    my(got = stream(pinned[c][1], 4));
    for (i = 1, 4, if (got[i] != pinned[c][2][i], off++)));
}
printf("xoshiro256++ stream: %d of 12 draws differ from the pinned values\n", off);
bm = 0;
{
  for (a = 1, 19, for (b = 0, 19,
    my(u1 = a / 20, u2 = b / 20, r = sqrt(-2 * log(u1)), x, y, back2);
    x = r * cos(2 * Pi * u2); y = r * sin(2 * Pi * u2);
    back2 = arg(x + I * y) / (2 * Pi); if (back2 < 0, back2 += 1);
    if (abs(exp(-(x^2 + y^2) / 2) - u1) > 1e-90, bm++);
    if (abs(back2 - u2) > 1e-90 && abs(abs(back2 - u2) - 1) > 1e-90, bm++)));
}
printf("Box-Muller inverse: %d of 760 recovered uniforms off by more than 1e-90\n", bm);
quit;
