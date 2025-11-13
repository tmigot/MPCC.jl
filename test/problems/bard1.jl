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

#=
#Bard1 (MacMPEC)
function bard1()

  ex3 = JuMP.Model()
  JuMP.@variable(ex3, x[1:5], start = 1.0)
  JuMP.@NLobjective(ex3, Min, (x[1] - 5)^2 + (2 * x[2] + 1)^2)
  JuMP.@NLconstraint(ex3, 2 * (x[2] - 1) - 1.5 * x[1] + x[3] - 0.5 * x[4] + x[5] == 0)
  ex3 = MathOptNLPModel(ex3) #MathProgNLPModel(ex3)
  G = JuMP.Model()
  JuMP.@variable(G, x[1:5], start = 1.0)
  JuMP.@constraint(G, 3 * x[1] - x[2] - 3 >= 0)
  JuMP.@constraint(G, -x[1] + 0.5 * x[2] + 4 >= 0)
  JuMP.@constraint(G, -x[1] - x[2] + 7 >= 0)
  JuMP.@NLobjective(G, Min, 0.0)
  G = MathOptNLPModel(G) #MathProgNLPModel(G)
  H = JuMP.Model()
  JuMP.@variable(H, x[1:5], start = 1.0)
  JuMP.@constraint(H, x[3] >= 0)
  JuMP.@constraint(H, x[4] >= 0)
  JuMP.@constraint(H, x[5] >= 0)
  JuMP.@NLobjective(H, Min, 0.0)
  H = MathOptNLPModel(H) #MathProgNLPModel(H)

  return MPCCNLPs(ex3, G, H)
end
=#
