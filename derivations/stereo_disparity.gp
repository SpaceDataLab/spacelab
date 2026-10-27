\\ Second engine for derivations/stereo_disparity.mac: the sub-pixel vertices the Rust gate
\\ `the_subpixel_refinement_is_the_vertex_of_the_interpolating_parabola` in
\\ crates/spacelab-earthvision/src/stereo.rs reads, found here by interpolating the three costs with
\\ polinterpolate and taking the root of its derivative -- not from the closed form. Exact rationals;
\\ each row prints `<name>: 0` when it equals the .mac's value, and a rectified pair built from a
\\ rotation checks Z = f B / d numerically.
\\
\\ Run: timeout 300 gp -q -f derivations/stereo_disparity.gp </dev/null
default(realprecision, 60);
chk(name, e) = printf("%s: %s\n", name, if(abs(e) < 1e-50, 0, e));
vtx(d, c0, c1, c2) = my(D = deriv(polinterpolate([-1, 0, 1], [c0, c1, c2], t), t)); d - polcoef(D, 0, t)/polcoef(D, 1, t);
chk("vertex_row_1", vtx(1, 3/2, 1/4, 5/8) - 33/26);
chk("vertex_row_2", vtx(7, 41/8, 9/2, 13/2) - 283/42);
chk("vertex_row_3", vtx(12, 1/1024, 0, 3/512) - 163/14);
\\ A rectified pair with a generic common rotation: R = R1(0.3) R3(-0.7), baseline 5 along camera x.
rot1(a) = [1, 0, 0; 0, cos(a), sin(a); 0, -sin(a), cos(a)];
rot3(a) = [cos(a), sin(a), 0; -sin(a), cos(a), 0; 0, 0, 1];
{
  my(R = rot1(0.3)*rot3(-0.7), f = 1000, B = 5, CL = [1.5, -2, 0.25]~, CR, X = [3, 4, 60]~, xl, xr);
  CR = CL + B*R~*[1, 0, 0]~;
  xl = R*(X - CL); xr = R*(X - CR);
  chk("rectified_rows_agree", f*xl[2]/xl[3] - f*xr[2]/xr[3]);
  chk("depth_from_disparity_is_z", f*B/(f*xl[1]/xl[3] - f*xr[1]/xr[3]) - xl[3]);
}
