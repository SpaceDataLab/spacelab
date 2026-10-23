\p 60
R = 6378137;
mask_deg = [0, 5, 10, 25];
alt_km = [500, 800, 2000, 20200, 35786];
for(i = 1, length(alt_km), L = (R + alt_km[i]*1000) / R; for(j = 1, length(mask_deg), s = sin(mask_deg[j] * Pi / 180); u = sqrt(L^2 - 1 + s^2); lam = acos((s*u + 1 - s^2) / L); printf("%d %d %.40g\n", alt_km[i], mask_deg[j], lam)))
quit
