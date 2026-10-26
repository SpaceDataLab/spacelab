\\ Second engine for shell_surveillance.mac: the characteristic velocity at 60 digits for the
\\ fixtures of shell.rs (Nguyen's constants), and the energy balance re-checked there.
\\ Run: timeout 120 gp -q -f derivations/shell_surveillance.gp </dev/null
default(realprecision, 60);
mu = 3986004415 * 10^5; re = 6371000; w = 72921159 / 10^12;
chv(n, h, ideg) = n*(sqrt(mu/re*(re + 2*h)/(re + h)) - w*re*cos(ideg*Pi/180));
{
  foreach([[12, 1000000, 60], [12, 3000000, 65], [12, 8000000, 65], [1, 500000, 60]], c,
    my(v = chv(c[1], c[2], c[3]), dv = sqrt(2*(mu/(re + c[2])/2 - mu/(re + c[2]) + mu/re)));
    printf("n=%d h=%d i=%d chv %.20e\n", c[1], c[2], c[3], v);
    printf("energy_balance_h%d: %s\n", c[2], if(abs(dv - v/c[1] - w*re*cos(c[3]*Pi/180)) < 1e-50, 0, 1)));
}
