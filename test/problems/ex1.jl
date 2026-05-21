function ex1(T = Float64)
  x0 = ones(T, 2)
  f(x) = x[1] - x[2]
  # c(x) = 1 - x[2]
  A = sparse([0 2])
  lcon = zeros(T, 1)
  ucon = Inf * ones(T, 1)
  G(x) = x[1]
  lccG = zeros(T, 1)
  H(x) = x[2]
  lccH = zeros(T, 1)
  return ADMPCCModel(G, H, lccG, lccH, f, x0, A, lcon, ucon)
end
