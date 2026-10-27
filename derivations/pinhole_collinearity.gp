\\ Second engine for derivations/pinhole_collinearity.mac: the camera of the Rust gate
\\ `the_projection_and_its_jacobians_are_the_collinearity_equations_and_their_derivatives` in
\\ crates/spacelab-earthvision/src/camera.rs, rebuilt at 60 digits with the rotation as the power
\\ series of the matrix exponential (not Rodrigues) and the Jacobians as derivatives of the
\\ perturbed projection, then held to the 40 digits the .mac prints. Each check prints `<name>: 0`
\\ when the relative difference is below 1e-38.
\\
\\ Run: timeout 300 gp -q -f derivations/pinhole_collinearity.gp </dev/null
default(realprecision, 60);
chk(name, got, want) = printf("%s: %s\n", name, if(abs(got - want) <= 1e-38*max(1, abs(want)), 0, got - want));
hat(v) = [0, -v[3], v[2]; v[3], 0, -v[1]; -v[2], v[1], 0];
expm(A) = my(S = matid(3), T = matid(3), k = 0); while(k < 200, k++; T = T*A/k; S += T); S;
w0 = [5/16, -3/16, 1/8];
R0 = expm(hat(w0));
C0 = [3/2, -9/4, 1/2]~; X0 = [13/4, 9/2, 60]~;
f0 = 1000; cx0 = 320; cy0 = 240;
proj(R, C, X) = my(xc = R*(X - C)); [f0*xc[1]/xc[3] + cx0, f0*xc[2]/xc[3] + cy0];
\\ The perturbation to second order in the rotation is exact for a first derivative at zero.
E2(w) = matid(3) + hat(w) + hat(w)^2/2;
\\ Column j: only the j-th of [dw, dC, dX] is the series variable e, the rest are zero.
pert(j) = my(d = vector(9, k, if(k == j, e, 0))); proj(E2(d[1..3])*R0, C0 + d[4..6]~, X0 + d[7..9]~);
J = matrix(2, 9);
for(j = 1, 9, my(p = pert(j)); for(i = 1, 2, J[i, j] = subst(deriv(p[i], e), e, 0)));
{
  my(rot = [9.749219018284145320270860560471980652292e-1, -1.508666900165644021627740348222757284052e-1, -1.636047895958829333118761923514087556807e-1;
            9.299415577444409145604954877734818662639e-2, 9.440565502326170329834996634899033762805e-1, -3.164005640871846791648743767085154021452e-1;
            2.021864790906298071163591830480271168666e-1, 2.932515503903365548821845822905443854337e-1, 9.344111278589303145323789158157487859841e-1],
     pix = [1.638355461915735686847878807002294085584e2, 2.783780704457788293607718969421699611154e1],
     jac = [-3.313219298168147280635081037884214605692e1, 1.024387336633284152527717637227888213202e3, 2.121621929554221170639228103057830038885e2, -1.737413168304442149448431993252603243042e1, 1.813738756498192212010265446706195798816, 3.052435939321838686433322372546846237182e-1, 1.737413168304442149448431993252603243042e1, -1.813738756498192212010265446706195798816, -3.052435939321838686433322372546846237182e-1;
            -1.045012796119653766195956356881458447261e3, 3.313219298168147280635081037884214605692e1, -1.561644538084264313152121192997705914416e2, -2.345740637574449039917259002830445337031, -1.737028569961928457229154569496665828023e1, 2.039571001482108515677699776402995340022, 2.345740637574449039917259002830445337031, 1.737028569961928457229154569496665828023e1, -2.039571001482108515677699776402995340022]);
  for(i = 1, 3, for(j = 1, 3, chk(Str("rotation_", i - 1, j - 1), R0[i, j], rot[i, j])));
  my(p = proj(R0, C0, X0));
  for(i = 1, 2, chk(Str("pixel_", i - 1), p[i], pix[i]));
  for(i = 1, 2, for(j = 1, 9, chk(Str("jacobian_", i - 1, "_", j - 1), J[i, j], jac[i, j])));
  chk("series_exponential_is_orthogonal", vecmax(apply(abs, concat(Vec(R0~*R0 - matid(3))))), 0);
}
