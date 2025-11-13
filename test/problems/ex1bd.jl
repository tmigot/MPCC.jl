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

#=
function ex1bd()

  ex1bd = JuMP.Model()
  ux(i) = [Inf; 1][i]
  JuMP.@variable(ex1bd, x[i = 1:2], upper_bound = ux(i), start = 1.0)
  JuMP.@NLobjective(ex1bd, Min, x[1] - x[2])
  ex1bd = MathOptNLPModel(ex1bd)#MathProgNLPModel(ex1bd)
  G = JuMP.Model()
  JuMP.@variable(G, x[1:2], start = 1.0)
  JuMP.@constraint(G, x[1] >= 0)
  JuMP.@NLobjective(G, Min, 0.0)
  G = MathOptNLPModel(G) #MathProgNLPModel(G)
  H = JuMP.Model()
  JuMP.@variable(H, x[1:2], start = 1.0)
  JuMP.@constraint(H, x[2] >= 0)
  JuMP.@NLobjective(H, Min, 0.0)
  H = MathOptNLPModel(H) #MathProgNLPModel(H)

  return MPCCNLPs(ex1bd, G, H)
end
=#
