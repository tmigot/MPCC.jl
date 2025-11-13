function ex2(T = Float64)
  x0 = -ones(T, 2)
  f(x) = 0.5 * ((x[1] - 1)^2 + (x[2] - 1)^2)
  G(x) = x[1]
  lccG = zeros(T, 1)
  H(x) = x[2]
  lccH = zeros(T, 1)
  return ADMPCCModel(G, H, lccG, lccH, f, x0)
end

#=
function ex2()

  ex2 = JuMP.Model()
  JuMP.@variable(ex2, x[1:2], start = -1.0)
  JuMP.@NLobjective(ex2, Min, 0.5 * ((x[1] - 1)^2 + (x[2] - 1)^2))
  ex2 = MathOptNLPModel(ex2) #MathProgNLPModel(ex2)
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

  return MPCCNLPs(ex2, G, H)
end
=#
