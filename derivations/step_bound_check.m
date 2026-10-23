% Empirical falsification test for the bound  |dg/dt| <= (1+s)*(v_perigee + omega*r_apogee).
% If any sampled geometry exceeds it, the bound is wrong and the event detector can skip events.
mu = 3.986004418e14; Re = 6378137.0; omega = 7.292115146706979e-5;
rand("seed",1); randn("seed",1);
mask = 5*pi/180; s = sin(mask);

worst = 0; worst_case = [];
for trial = 1:400
  alt_p = 200e3 + rand*40000e3;
  e     = rand*0.7;
  a     = (Re+alt_p)/(1-e);
  rp = a*(1-e); ra = a*(1+e);
  vp = sqrt(mu*(1+e)/rp);
  bound = (1+s)*(vp + omega*ra);

  % sample true state along the orbit (Keplerian, planar, then rotate frame)
  M = 2*pi*rand(1,4000);
  E = M; for k=1:60, E = E - (E-e*sin(E)-M)./(1-e*cos(E)); end
  nu = 2*atan2(sqrt(1+e).*sin(E/2), sqrt(1-e).*cos(E/2));
  r  = a*(1-e*cos(E));
  n  = sqrt(mu/a^3);
  % perifocal position/velocity
  rp_v = [r.*cos(nu); r.*sin(nu); zeros(1,numel(nu))];
  h  = sqrt(mu*a*(1-e^2));
  vp_v = [-mu/h*sin(nu); mu/h*(e+cos(nu)); zeros(1,numel(nu))];
  % random inclination/RAAN/argp rotation
  i0=rand*pi; O=rand*2*pi; w=rand*2*pi;
  Rz=@(t)[cos(t) -sin(t) 0; sin(t) cos(t) 0; 0 0 1];
  Rx=@(t)[1 0 0; 0 cos(t) -sin(t); 0 sin(t) cos(t)];
  Q = Rz(O)*Rx(i0)*Rz(w);
  ri = Q*rp_v; vi = Q*vp_v;
  % body-fixed velocity: v_bf = v_i - omega x r
  wv = [0;0;omega];
  vbf = vi - cross(repmat(wv,1,size(ri,2)), ri);

  % random ground cells
  m = 200;
  u = randn(3,m); u = u./sqrt(sum(u.^2,1));
  cellp = Re*u; zen = u;
  for j = 1:m
    d  = ri - cellp(:,j);
    nd = sqrt(sum(d.^2,1));
    dg = sum(vbf.*zen(:,j),1) - s*sum(d.*vbf,1)./nd;
    mx = max(abs(dg));
    ratio = mx/bound;
    if ratio > worst, worst = ratio; worst_case = [alt_p/1e3, e, mx, bound]; end
  end
end
printf("orbits sampled: 400,  states x cells: %d\n", 400*4000*200);
printf("worst  max|dg/dt| / bound  = %.6f\n", worst);
printf("  at alt_p=%.0f km, e=%.3f : |dg/dt|=%.1f m/s, bound=%.1f m/s\n", worst_case);
if worst > 1.0
  printf("*** BOUND VIOLATED — the coarse step could skip events ***\n");
else
  printf("bound holds; slack factor at the worst sample = %.2fx\n", 1/worst);
end
