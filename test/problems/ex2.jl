function ex2(T = Float64)
  x0 = -ones(T, 2)
  f(x) = 0.5 * ((x[1] - 1)^2 + (x[2] - 1)^2)
  G(x) = x[1]
  lccG = zeros(T, 1)
  H(x) = x[2]
  lccH = zeros(T, 1)
  return ADMPCCModel(G, H, lccG, lccH, f, x0)
end
