\\ Second engine for draim_tetrahedral.mac: the semi-major axis of the patent's 48-hour example at
\\ 60 digits, and its period re-derived (prints `<name>: 0`).
\\ Run: timeout 120 gp -q -f derivations/draim_tetrahedral.gp </dev/null
default(realprecision, 60);
mu = 3986004418 * 10^5;
P = 48 * 3600;
a = (mu*(P/(2*Pi))^2)^(1/3);
printf("a_48h %.20e\n", a);
printf("period_48h: %s\n", if(abs(2*Pi*sqrt(a^3/mu) - P) < 1e-50, 0, 2*Pi*sqrt(a^3/mu) - P));
printf("perigee_above_earth: %s\n", if(a*(1 - 28/100) > 6378137, 0, 1));
