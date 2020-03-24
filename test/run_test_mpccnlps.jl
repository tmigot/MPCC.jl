test1bis=JuMP.Model()
ux(i)=[100;100][i]
lx(i)=[-100;-100][i]
JuMP.@variable(test1bis,x[i=1:2], upperbound=ux(i), lowerbound=lx(i),start=1.0)
JuMP.@NLobjective(test1bis,Min,x[1]-x[2])
JuMP.@constraint(test1bis,1-x[2]>=0)
test1bis = MathProgNLPModel(test1bis)

#Test 0: constructor
test1 = ex1()
test1bis = MPCCNLPs(test1bis)
#Creating two NLPs with a different number of constraints would lead to an error
error_G = JuMP.Model()
JuMP.@variable(error_G,x[1:2],start=1.0)
JuMP.@constraint(error_G,x[1]>=0)
JuMP.@NLobjective(error_G,Min,0.0)
error_G = MathProgNLPModel(error_G)
error_H=JuMP.Model()
JuMP.@variable(error_H,x[1:2],start=1.0)
JuMP.@NLobjective(error_H,Min,0.0)
error_H=MathProgNLPModel(error_H)
try
    error_test = MPCCNLPs(ex1,G = error_G, H = error_H)
    @test false
catch
    @test true
end
test1unc = MPCCNLPs(ADNLPModel(rosenbrock, zeros(6)))
test1unbd = ex1bd()

#Test 1: meta check
@test test1.meta.ncc == 1
@test test1bis.meta.ncc == 0
@test test1.meta.nvar == 2
@test test1bis.meta.nvar == 2
@test test1unc.meta.ncon + test1unc.meta.ncc == 0
@test test1unbd.meta.ncc == 1
@test test1unbd.meta.ncon == 0

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
@test hess(test1, zeros(2), zeros(3)) == sparse(zeros(2,2))
@test hprod(test1, zeros(2), zeros(3), ones(2)) == zeros(2)

#Test 4: check cons, consG, consH
@test cons(test1bis, zeros(2)) == [0.0]
@test consG(test1bis, zeros(2)) == []
@test consH(test1bis, zeros(2)) == []

#Test 5: jacobian matrix
@test jacG(test1bis, zeros(2)) == []
@test jacH(test1bis, zeros(2)) == []
@test size(jac(test1unc, zeros(6)),1) == 0
@test size(jac_actif(test1unc, zeros(6), 1e-6)[1],1) == 0
@test size(jac(test1, test1.meta.x0)) == (3,2)
@test size(jac_actif(test1, zeros(2),1e-6)[1]) == (2,8)
@test size(jac(test1unbd, test1unbd.meta.x0)) == (2,2)
@test size(jac_actif(test1unbd, zeros(2), 1e-6)[1]) == (2,6)

@test jprod(test1, zeros(2), ones(2))  == - ones(3)
@test jtprod(test1, zeros(2), ones(3)) == [-1.0, -2.0]

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
