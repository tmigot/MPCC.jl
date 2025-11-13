# Exemple 3 :
# minimize x1+x2-x3
# s.t.     0<=x1 _|_ x2>=0
#          -4x1+x3<=0
#          -4x2+x3<=0
function ex3(T = Float64)
  x0 = ones(T, 3)
  f(x) = x[1] + x[2] - x[3]
  # c(x) = 1 - x[2]
  A = sparse([-4  0 1;
               0 -4 1])
  lcon = -Inf * ones(T, 2)
  ucon = zeros(T, 2)
  G(x) = x[1]
  lccG = zeros(T, 1)
  H(x) = x[2]
  lccH = zeros(T, 1)
  return ADMPCCModel(G, H, lccG, lccH, f, x0, A, lcon, ucon)
end

#=
function ex3()

  ex3 = JuMP.Model()
  JuMP.@variable(ex3, x[1:3], start = 1.0)
  JuMP.@NLobjective(ex3, Min, x[1] + x[2] - x[3])
  JuMP.@NLconstraint(ex3, c1, -4 * x[1] + x[3] <= 0)
  JuMP.@NLconstraint(ex3, c2, -4 * x[2] + x[3] <= 0)
  ex3 = MathOptNLPModel(ex3) #MathProgNLPModel(ex3)
  G = JuMP.Model()
  JuMP.@variable(G, x[1:3], start = 1.0)
  JuMP.@constraint(G, x[1] >= 0)
  JuMP.@NLobjective(G, Min, 0.0)
  G = MathOptNLPModel(G) #MathProgNLPModel(G)
  H = JuMP.Model()
  JuMP.@variable(H, x[1:3], start = 1.0)
  JuMP.@constraint(H, x[2] >= 0)
  JuMP.@NLobjective(H, Min, 0.0)
  H = MathOptNLPModel(H) #MathProgNLPModel(H)

  return MPCCNLPs(ex3, G, H)
end
=#
