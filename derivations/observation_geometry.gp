\\ Second engine for derivations/observation_geometry.mac: the Kasten-Young air mass of
\\ crates/spacelab-earthvision/src/geometry.rs at 60 digits against the 40-digit values the .mac
\\ prints (and the Rust gate `the_air_mass_is_kasten_young_s_formula` reads), and the scattering
\\ angle's law of cosines at sample directions built as vectors. Each check prints `<name>: 0`.
\\
\\ Run: timeout 300 gp -q -f derivations/observation_geometry.gp </dev/null
default(realprecision, 60);
chk(name, e, tol) = printf("%s: %s\n", name, if(abs(e) < tol, 0, e));
ky(z) = 1/(cos(z) + 50572/100000*(9607995/100000 - z*180/Pi)^(-16364/10000));
{
rows = [[0, 9.997119918558381315393438355106013890384e-1],
        [1/2, 1.138826624854773829714342718122949757378],
        [1, 1.846471199582342699878539262176237955795],
        [11/8, 5.017246978829234931212123394087344332953],
        [49/32, 1.810269069650286168852866930954648170836e1],
        [25/16, 3.163415279610748979292860300654418652789e1],
        [884279719003555/2^49, 3.791960837783627306248202055583037736895e1]];
}
for(i = 1, #rows, chk(Str("kasten_young_row_", i), ky(rows[i][1])/rows[i][2] - 1, 1e-38));
\\ E = N x U with N = x, U = z is -y; a direction at zenith z, azimuth a (north toward east).
dir(z, a) = [sin(z)*cos(a), -sin(z)*sin(a), cos(z)];
{
  my(pts = [[0.61, 0.7, 0.35, 4.36], [1.2, -2.0, 0.05, 3.0], [0.3, 1.1, 1.5, -0.4]], zs, as, zv, av, s, v);
  for(i = 1, #pts,
    [zs, as, zv, av] = pts[i];
    s = dir(zs, as); v = dir(zv, av);
    chk(Str("scattering_law_of_cosines_", i),
      -(s*v~) + (cos(zs)*cos(zv) + sin(zs)*sin(zv)*cos(av - as)), 1e-50);
    chk(Str("azimuth_reads_back_", i), arg(s[1] + I*(-s[2])) - as, 1e-50));
}
