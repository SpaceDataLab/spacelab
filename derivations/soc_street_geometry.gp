\\ Second engine for derivations/soc_street_geometry.mac: the Street-of-Coverage reference values
\\ at 60 digits, and the two touch conditions re-measured from plane normals built as vectors,
\\ not from the closed forms. Each check prints `<name>: 0` when |residual| < 1e-50.
\\ Run: timeout 120 gp -q -f derivations/soc_street_geometry.gp </dev/null
default(realprecision, 60);
chk(name, e) = printf("%s: %s\n", name, if(abs(e) < 1e-50, 0, e));
nrm(W, i) = [sin(W)*sin(i), -cos(W)*sin(i), cos(i)];
dihedral(a, b) = my(c = abs(a*b~)); acos(c);
re = 6378137;
soc(alt, incdeg, epsdeg, sover, tag) = {
  my(eps = epsdeg*Pi/180, inc = incdeg*Pi/180, theta, s, h, lam, dco, dct, span, P, dw, du);
  theta = acos(re/(re + alt)*cos(eps)) - eps;
  s = if(sover, sover, ceil(Pi/theta));
  h = Pi/s;
  lam = acos(cos(theta)/cos(h));
  dco = 2*asin(sin((theta + lam)/2)/sin(inc));
  dct = 2*asin(sqrt(sin(lam)^2 - cos(inc)^2)/sin(inc));
  span = Pi - dct;
  P = ceil(span/dco) + 1;
  dw = span/(P - 1);
  du = h - 2*atan(cos(inc)*tan(dco/2));
  \\ cos b is the ratio of the .mac's section 6, so the two forms agree up to the sign of b.
  chk(Str(tag, "_foot_cosine"), cos(atan(cos(inc)*tan(dco/2))) - cos(dco/2)/cos((theta + lam)/2));
  printf("case %s S=%d P=%d\n", tag, s, P);
  printf("  theta %.20e\n  lambda %.20e\n  raan_step_co %.20e\n  raan_step_counter %.20e\n", theta, lam, dco, dct);
  printf("  raan_step %.20e\n  raan_span %.20e\n  inter_plane_phase %.20e\n", dw, span, du);
  \\ Co-rotating neighbours at the maximum spacing touch at theta + lambda; the seam pair (plane
  \\ P-1 ascending against plane 0, whose normal as a point set is the same) touches at 2 lambda.
  chk(Str(tag, "_co_dihedral_is_theta_plus_lambda"), dihedral(nrm(0, inc), nrm(dco, inc)) - (theta + lam));
  chk(Str(tag, "_seam_dihedral_is_two_lambda"), dihedral(nrm(0, inc), nrm((P - 1)*dw, inc)) - 2*lam);
  \\ The filled spacing never exceeds the maximum, and one plane fewer could not span the nodes.
  chk(Str(tag, "_filled_spacing_within_max"), (dw <= dco) - 1);
  chk(Str(tag, "_plane_count_is_minimal"), ((P - 2)*dco < span) - 1);
}
soc(780000, 864/10, 82/10, 0, "iridium");
soc(780000, 864/10, 82/10, 12, "iridium_s12");
soc(600000, 88, 5, 0, "i88");
soc(1200000, 90, 10, 0, "polar");
soc(780000, 98, 82/10, 0, "retro98");
