function ex1bd(T = Float64)
  x0 = ones(T, 2)
  f(x) = x[1] - x[2]
  lvar = -Inf * ones(T, 2)
  uvar = T[Inf; 1]
  G(x) = x[1]
  lccG = zeros(T, 1)
  H(x) = x[2]
  lccH = zeros(T, 1)
  return ADMPCCModel(G, H, lccG, lccH, f, x0, lvar = lvar, uvar = uvar)
end
