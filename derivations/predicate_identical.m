Re = 6378137.0; mask = 5.0*pi/180; s = sin(mask);
printf("Pointwise identity of the three decision forms, and the width of the fuzzy band.\n\n");
printf("%-8s %-12s %-12s %-14s %-12s\n","alt km","fast==mid","fast==asin","band (rad)","band on ground");
printf("%s\n", repmat('-',1,68));
for alt_km = [500, 800, 2000, 20200, 35786]
  L = (Re + alt_km*1e3)/Re;
  u = sqrt(L^2 - 1 + s^2); clam = (s*u + 1 - s^2)/L; lam0 = acos(clam);
  K = 200001;
  % perturb in units of eps(lam0) so we resolve the true float grid
  h = eps(lam0);
  lam = lam0 + h*linspace(-2000,2000,K);
  sat = (Re*L)*[sin(lam); zeros(1,K); cos(lam)];
  d = sat - [0;0;Re]; t = d(3,:); q = sum(d.*d,1);
  fast = (t>=0) & (t.^2 >= s^2*q);
  mid  = t >= s.*sqrt(q);
  ref  = asin(t./sqrt(q)) >= mask;
  % elevation computed the accurate way, to locate the true crossing
  epsv = asin(t./sqrt(q));
  % fuzzy band = spread of lam where fast flips relative to monotone truth
  flip = find(diff(fast)~=0);
  if isempty(flip), band = 0; else band = (max(flip)-min(flip))*(lam(2)-lam(1)); end
  % convert angular band at Earth centre to ground distance
  printf("%-8d %-12s %-12s %-14.3e %-12.3e m\n", alt_km, ...
     mat2str(isequal(fast,mid)), mat2str(isequal(fast,ref)), band, band*Re);
end
printf("\nAlso: sensitivity of the boundary to elevation error, d(ground)/d(eps):\n");
for alt_km = [500, 35786]
  L = (Re+alt_km*1e3)/Re; u = sqrt(L^2-1+s^2);
  cl = @(ss) (ss.*sqrt(L^2-1+ss.^2) + 1 - ss.^2)/L;
  de = 1e-8; l1 = acos(cl(sin(mask-de))); l2 = acos(cl(sin(mask+de)));
  printf("  alt %6d km : 1 nrad of elevation error moves the footprint edge %.3e m\n", ...
         alt_km, abs(l2-l1)/(2*de)*Re*1e-9);
end
