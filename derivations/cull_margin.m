% How much must a spherical-cap cull be dilated to never miss a covered cell on the WGS-84
% ellipsoid? The footprint radius was derived on a SPHERE; the real Earth is an ellipsoid whose
% geodetic zenith is not radial (deflection up to ~0.19 deg), and whose local radius varies.
%
% Conservative cull proposal:
%   L = r_sat / R_polar         (smallest Earth radius -> largest footprint -> conservative)
%   cos(lam_max) = (s*u + 1 - s^2)/L,  u = sqrt(L^2-1+s^2),  s = sin(mask)
%   lam_cap = lam_max + margin
% and cull keeps every cell with geocentric angle(unit(sat), unit(cell)) <= lam_cap.
%
% This measures the WORST (covered-cell geocentric angle - lam_max) over random geometry, to find
% the margin the ellipsoid actually demands. If it exceeds our proposed margin, the cull is unsafe.

a_eq = 6378137.0; f = 1/298.257223563; b_pol = a_eq*(1-f); e2 = f*(2-f);
rand('seed',12345); randn('seed',12345);

worst_excess = -1e9;   % max over covered cells of (geocentric_angle - lam_max_using_Rpolar)
worst_case = [];
Ntot = 0; Ncov = 0;

for mask_deg = [0 5 10]
  s = sin(mask_deg*pi/180);
  for trial = 1:4000
    % random satellite altitude 300..2000 km (LEO, worst case for large caps)
    alt = 300e3 + rand*1700e3;
    subdir = randn(3,1); subdir = subdir/norm(subdir);
    r_sat_mag = a_eq + alt;               % roughly; use geocentric radius
    sat = r_sat_mag * subdir;

    L = r_sat_mag / b_pol;                % conservative L (smallest radius)
    u = sqrt(L^2 - 1 + s^2);
    lam_max = acos((s*u + 1 - s^2)/L);

    % sample cells on the true WGS-84 ellipsoid around the sub-point, out to lam_max+1deg
    for k = 1:300
      % random direction within a cap of lam_max+1.5deg around subdir
      ang = (lam_max + 1.5*pi/180) * sqrt(rand);
      perp = randn(3,1); perp = perp - (perp'*subdir)*subdir; perp = perp/norm(perp);
      celldir = cos(ang)*subdir + sin(ang)*perp;     % geocentric direction of the cell
      % place on ellipsoid: scale celldir to ellipsoid surface
      d = celldir;
      kell = 1/sqrt((d(1)^2+d(2)^2)/a_eq^2 + d(3)^2/b_pol^2);
      cellp = kell * d;
      % true geodetic zenith (ellipsoid normal)
      zen = [cellp(1)/a_eq^2; cellp(2)/a_eq^2; cellp(3)/b_pol^2]; zen = zen/norm(zen);
      % exact elevation of satellite from this cell
      los = sat - cellp;
      sin_el = (los'*zen)/norm(los);
      covered = sin_el >= s;
      Ntot++;
      if covered
        Ncov++;
        geo_ang = acos(max(-1,min(1, celldir'*subdir)));
        excess = geo_ang - lam_max;      % how far beyond the spherical cap this covered cell sits
        if excess > worst_excess
          worst_excess = excess;
          worst_case = [mask_deg, alt/1e3, geo_ang*180/pi, lam_max*180/pi];
        end
      end
    end
  end
end

printf('sampled %d cells, %d covered\n', Ntot, Ncov);
printf('worst (covered geocentric angle - spherical lam_max) = %.6f deg = %.3f km on ground\n', ...
       worst_excess*180/pi, worst_excess*a_eq/1000);
if worst_excess > 0
  printf('  at mask=%.0f deg, alt=%.0f km: covered cell at %.4f deg, cap was %.4f deg\n', worst_case);
  printf('  => cull MUST dilate by at least this much or it drops covered cells\n');
else
  printf('  R_polar cap already covers everything; no dilation needed (margin >= 0)\n');
end
printf('\nmax vertical deflection (geodetic-geocentric latitude), for reference: %.4f deg\n', ...
       max(abs(atan((1-e2)*tan((0:90)*pi/180)) - (0:90)*pi/180))*180/pi);
