% Why a batch least-squares solver must not form H'WH.
%
% GMAT's BatchEstimator accumulates the information matrix H'WH explicitly and inverts it
% (default: LU). Orbit determination Jacobians are ill-conditioned -- a short arc barely
% separates semi-major axis from mean motion -- and cond(H'H) = cond(H)^2.
%
% This measures the digits lost, against a reference computed in extended precision.

rand("seed", 7); randn("seed", 7);
printf("%-12s | %-14s %-14s | %-14s %-14s\n", "cond(H)", "normal eq err", "digits left", "QR err", "digits left");
printf("%s\n", repmat('-', 1, 76));

m = 200; n = 6;                         % 200 observations, 6-element state
for k = [1e2 1e4 1e6 1e8 1e10]
  % build H with a prescribed condition number via SVD
  [U,~,~] = svd(randn(m,n), 0);
  [V,~,~] = svd(randn(n,n));
  s = logspace(0, -log10(k), n);
  H = U * diag(s) * V';
  x_true = randn(n,1);
  b = H * x_true;                       % noiseless: any error is the solver's

  % (a) normal equations, as GMAT does: form H'H, invert, multiply
  N = H' * H;
  x_ne = N \ (H' * b);

  % (b) QR on the Jacobian itself, never forming H'H
  [Q,R] = qr(H, 0);
  x_qr = R \ (Q' * b);

  e_ne = norm(x_ne - x_true) / norm(x_true);
  e_qr = norm(x_qr - x_true) / norm(x_true);
  printf("%-12.0e | %-14.3e %-14.1f | %-14.3e %-14.1f\n", ...
         k, e_ne, max(0,-log10(e_ne)), e_qr, max(0,-log10(e_qr)));
end

printf("\n  f64 carries ~15.95 decimal digits.\n");
printf("  normal equations: cond(H'H) = cond(H)^2, so the error grows as eps*cond(H)^2.\n");
printf("  QR on H:           error grows as eps*cond(H).  Half the digits, kept.\n");
printf("\n  At cond(H)=1e8 -- an ordinary short-arc OD problem -- the normal equations\n");
printf("  have no correct digits left, and the Cholesky may not even complete.\n");
