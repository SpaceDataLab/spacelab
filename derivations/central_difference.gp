/* Second engine for derivations/central_difference.mac: the nine-node central difference's weights in
   exact rationals, and on sin at 60 digits. Prints `PASS` lines; a failed check prints `FAIL`.

       timeout 60 gp -q -f derivations/central_difference.gp </dev/null                         */

default(realprecision, 60);
check(name, ok) = print(name, if (ok, ": PASS", ": FAIL"));
w = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280];
moment(p) = sum(k = 1, 9, w[k] * (k - 5)^p);
{
  check("the weights reproduce the slope of x^0 ... x^8 exactly",
    vector(9, p, if (p == 1, sum(k = 1, 9, w[k]), moment(p - 1))) == [0, 1, 0, 0, 0, 0, 0, 0, 0]);
}
check("the first missed term is -h^8 f^(9)/630", moment(9)/9! == -1/630);
{
  my(x = 3/10, h = 1/100, d = sum(k = 1, 9, w[k] * sin(x + (k - 5)*h)) / h,
     err = d - cos(x), predicted = -h^8 / 630 * cos(x));
  print("on sin at 0.3, h = 0.01: error ", err, ", -h^8 f^(9)/630 = ", predicted);
  check("the error on sin is the predicted term to 0.1 %", abs(err / predicted - 1) < 1/1000);
}
