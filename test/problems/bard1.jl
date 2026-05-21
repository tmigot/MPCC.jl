function bard1(T = Float64)
  x0 = ones(T, 5)
  f(x) = (x[1] - 5)^2 + (2 * x[2] + 1)^2
  # c(x) = - 1.5 * x[1] + 2 * (x[2] - 1) + x[3] - 0.5 * x[4] + x[5] == 0
  A = sparse([-1.5 2 1 -0.5 1])
  lcon = 2 * ones(T, 1)
  ucon = 2 * ones(T, 1)
  G(x) = [
    3 * x[1] - x[2] - 3;
    -x[1] + 0.5 * x[2] + 4;
    -x[1] - x[2] + 7
  ]
  lccG = zeros(T, 3)
  H(x) = [
    x[3];
    x[4];
    x[5]
  ]
  lccH = zeros(T, 3)
  return ADMPCCModel(G, H, lccG, lccH, f, x0, A, lcon, ucon)
end
