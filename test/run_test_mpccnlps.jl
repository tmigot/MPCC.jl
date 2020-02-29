test1bis=JuMP.Model()
ux(i)=[100;100][i]
lx(i)=[-100;-100][i]
JuMP.@variable(test1bis,x[i=1:2], upperbound=ux(i), lowerbound=lx(i),start=1.0)
JuMP.@NLobjective(test1bis,Min,x[1]-x[2])
JuMP.@constraint(test1bis,1-x[2]>=0)
test1bis = MathProgNLPModel(test1bis)

test1 = ex1()
test1bis = MPCCNLPs(test1bis)

#Test 1: meta check
@test test1.meta.ncc == 1
@test test1bis.meta.ncc == 0
@test test1.meta.nvar == 2
@test test1bis.meta.nvar == 2

#Test 2: data check
#test1 has no bounds constraints, one >= constraint, and complementarity constraints
@test test1.meta.lvar == -Inf*ones(2)
@test test1.meta.uvar == Inf*ones(2)
@test test1.meta.ucon == Inf*ones(1)
@test test1.meta.lccG == zeros(1)
@test test1.meta.lccH == zeros(1)

#Test 3: check obj et grad
#test1 has a linear objective function, so the gradient is constant and hessian vanishes
@test obj(test1, test1.meta.x0) == 0.0
@test grad(test1,test1.meta.x0) == grad(test1,[Inf,Inf])
@test hess(test1,test1.meta.x0) == sparse(zeros(2,2))

#Test 4: check les contraintes et la réalisabilité
#test1 has two stationary points:
Wpoint, Spoint = zeros(2), [0.0, 1.0]
@test norm(viol(test1,Spoint),Inf) == 0.0
@test norm(viol(test1,Wpoint),Inf) == 0.0
@test cons(test1,Wpoint) == zeros(3)

#Test 5: check the jacobian
# Wpoint is an W-stationary point, and Spoint is an S-stationary point
lambdaW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
AW = jacl(test1, Wpoint)
aw = grad(test1, Wpoint)
lambdaS = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
AS = jacl(test1, Spoint)
bs = grad(test1, Spoint)

@test norm(aw + AW*lambdaW,Inf) == 0.0
@test norm(bs + AS*lambdaS,Inf) == 0.0
@test norm(AW\(-aw)-lambdaW,Inf) == 0.0
@test norm(AS\(-bs)-lambdaS,Inf) == 0.0
