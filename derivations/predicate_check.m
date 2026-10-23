% Is the sqrt-free coverage predicate exactly equivalent, and is it sharper?
%
%   d = r_sat - r_cell,  t = d.zenith,  q = d.d,  s = sin(eps_min) >= 0
%   exact:      eps >= eps_min   <=>   t >= s*sqrt(q)
%   sqrt-free:                    <=>   t >= 0  &&  t^2 >= s^2*q
%
% Both boundaries coincide for t,s >= 0. The question is rounding behaviour.

rand("seed", 20260710); randn("seed", 20260710);
N = 2e6;  Re = 6378137.0;

% random ground point on the sphere, random satellite at 300..36000 km altitude
v = randn(3,N); v = v ./ sqrt(sum(v.^2,1));
gp = Re * v;  zen = v;                       % sphere: zenith == radial
w  = randn(3,N); w = w ./ sqrt(sum(w.^2,1));
alt = 300e3 + (36000e3-300e3).*rand(1,N);
sat = (Re+alt) .* w;

d  = sat - gp;
t  = sum(d.*zen, 1);
q  = sum(d.*d,   1);
mask_deg = 5.0;  s = sin(mask_deg*pi/180);

ref  = asin(t ./ sqrt(q)) >= (mask_deg*pi/180);      % trig reference
fast = (t >= 0) & (t.^2 >= s^2 .* q);                % sqrt-free
mid  = t >= s.*sqrt(q);                              % one-sqrt form

printf("N = %d,  mask = %.1f deg\n", N, mask_deg);
printf("  covered by reference : %d\n", sum(ref));
printf("  disagreements fast vs reference : %d\n", sum(fast ~= ref));
printf("  disagreements mid  vs reference : %d\n", sum(mid  ~= ref));

% where do disagreements sit? measure angular distance from the boundary
eps_true = asin(t./sqrt(q));
bad = find(fast ~= ref);
if !isempty(bad)
  printf("  worst |eps - mask| among disagreements : %.3e rad (%.3e arcsec)\n", ...
     max(abs(eps_true(bad)-mask_deg*pi/180)), max(abs(eps_true(bad)-mask_deg*pi/180))*206265);
end

% sharpness: relative margin of each form near the boundary
near = abs(eps_true - mask_deg*pi/180) < 1e-9;
printf("  samples within 1e-9 rad of the boundary : %d\n", sum(near));
