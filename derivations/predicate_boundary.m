% Exercise the decision boundary itself, using the cos(lam) we derived and Maxima proved:
%     cos(lam*) = (s*u + 1 - s^2)/L,   u = sqrt(L^2 - 1 + s^2),  s = sin(eps_min)
% At lam = lam* the elevation is exactly eps_min. Perturb lam and compare the three forms.

Re = 6378137.0;
mask = 5.0*pi/180;  s = sin(mask);

printf("%-10s %-14s | disagreements out of 2*K, per form (ref = 200-digit-free asin)\n","alt km","perturb rad");
printf("%s\n", repmat('-',1,86));

for alt_km = [500, 800, 2000, 20200, 35786]
  L = (Re + alt_km*1e3)/Re;
  u = sqrt(L^2 - 1 + s^2);
  clam = (s*u + 1 - s^2)/L;
  lam0 = acos(clam);

  for dl = [1e-15 1e-13 1e-11 1e-9]
    K = 20000;
    lam = lam0 + dl*linspace(-1,1,K);
    sat = (Re*L) * [sin(lam); zeros(1,K); cos(lam)];
    gp  = [0;0;Re];  zen = [0;0;1];
    d = sat - gp;
    t = d(3,:);                      % d . zen
    q = sum(d.*d,1);

    % ground truth: which side of lam0 (elevation decreases as lam grows)
    truth = lam <= lam0;

    fast = (t >= 0) & (t.^2 >= s^2*q);
    mid  = t >= s.*sqrt(q);
    ref  = asin(t./sqrt(q)) >= mask;

    printf("%-10d %-14.0e | fast %5d   mid %5d   asin %5d\n", alt_km, dl, ...
           sum(fast~=truth), sum(mid~=truth), sum(ref~=truth));
  end
end
