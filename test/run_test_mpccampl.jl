
test1a=JuMP.Model()
ux(i)=[100;100][i]
lx(i)=[-100;-100][i]
JuMP.@variable(test1a,x[i=1:2], upperbound=ux(i), lowerbound=lx(i),start=1.0)
JuMP.@NLobjective(test1a,Min,x[1]-x[2])
JuMP.@constraint(test1a,1-x[2]>=0)

testa = MathProgNLPModel(test1a)
mpcca = MPCCAmpl(testa, [0], 0)

@test mpcca.meta.ncc == 0
@test mpcca.cvar == [0]
@test mpcca.meta.lvar == [-100.;-100.]
@test mpcca.meta.uvar == [ 100.; 100.]
@test mpcca.meta.ncon == 1
