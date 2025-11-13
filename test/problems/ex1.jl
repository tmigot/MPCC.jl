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

#=
function ex1()

  ex1 = JuMP.Model()
  JuMP.@variable(ex1, x[1:2], start = 1.0)
  JuMP.@NLobjective(ex1, Min, x[1] - x[2])
  JuMP.@constraint(ex1, 1 - x[2] >= 0)
  ex1 = MathOptNLPModel(ex1)#MathProgNLPModel(ex1)
  G = JuMP.Model()
  JuMP.@variable(G, x[1:2], start = 1.0)
  JuMP.@constraint(G, x[1] >= 0)
  JuMP.@NLobjective(G, Min, 0.0)
  G = MathOptNLPModel(G)#MathProgNLPModel(G)
  H = JuMP.Model()
  JuMP.@variable(H, x[1:2], start = 1.0)
  JuMP.@constraint(H, x[2] >= 0)
  JuMP.@NLobjective(H, Min, 0.0)
  H = MathOptNLPModel(H)#MathProgNLPModel(H)

  return MPCCNLPs(ex1, G, H)
end
=#
