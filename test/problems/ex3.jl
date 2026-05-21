# Exemple 3 :
# minimize x1+x2-x3
# s.t.     0<=x1 _|_ x2>=0
#          -4x1+x3<=0
#          -4x2+x3<=0
function ex3(T = Float64)
  x0 = ones(T, 3)
  f(x) = x[1] + x[2] - x[3]
  # c(x) = 1 - x[2]
  A = sparse([
    -4 0 1;
    0 -4 1
  ])
  lcon = -Inf * ones(T, 2)
  ucon = zeros(T, 2)
  G(x) = x[1]
  lccG = zeros(T, 1)
  H(x) = x[2]
  lccH = zeros(T, 1)
  return ADMPCCModel(G, H, lccG, lccH, f, x0, A, lcon, ucon)
end
