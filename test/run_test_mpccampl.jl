
test1a=JuMP.Model()
ux(i)=[100;100][i]
lx(i)=[-100;-100][i]
JuMP.@variable(test1a,x[i=1:2], upperbound=ux(i), lowerbound=lx(i),start=1.0)
JuMP.@NLobjective(test1a,Min,x[1]-x[2])
JuMP.@constraint(test1a,1-x[2]>=0)

testa = MathProgNLPModel(test1a)
mpcca = MPCCAmpl(testa, [0], 0)

@test mpcca.meta.ncc == 0
@test mpcca.meta.nvar == 2
@test mpcca.cvar == [0]
@test mpcca.meta.lvar == [-100.;-100.]
@test mpcca.meta.uvar == [ 100.; 100.]
@test mpcca.meta.ncon == 1

@test obj(mpcca.mp, mpcca.meta.x0)  == 0.0
@test grad(mpcca.mp, mpcca.meta.x0) == [1.0, -1.0]
@test cons(mpcca.mp, mpcca.meta.x0) == [-1.0]
@test jac(mpcca.mp, mpcca.meta.x0)  == [0.0 -1.0]

@test hessG(mpcca, mpcca.meta.x0, zeros(0)) == zeros(2,2)
@test hGprod(mpcca, mpcca.meta.x0, zeros(0), ones(2)) == zeros(2)
@test issparse(jacG(mpcca, mpcca.meta.x0))
@test consG(mpcca, mpcca.meta.x0) == zeros(0)

@test hessH(mpcca, mpcca.meta.x0, zeros(0)) == zeros(2,2)
@test hHprod(mpcca, mpcca.meta.x0, zeros(0), ones(2)) == zeros(2)
@test sum(jacH(mpcca, mpcca.meta.x0)) == 0
@test consH(mpcca, mpcca.meta.x0) == zeros(0)

@test jprod(mpcca, mpcca.meta.x0, zeros(2)) == zeros(1)
@test jtprod(mpcca, mpcca.meta.x0, zeros(1)) == zeros(2)

@test size(hess(mpcca, mpcca.meta.x0)) == (2,2)
@test hess(mpcca, mpcca.meta.x0, zeros(1), obj_weight = 0.0) == zeros(2,2)
@test hprod(mpcca, mpcca.meta.x0, zeros(2)) == zeros(2)
@test hprod(mpcca, mpcca.meta.x0, zeros(1), ones(2), obj_weight = 0.0) == zeros(2)

@test hessH_structure(mpcca) == hessG_structure(mpcca)
@test length(hessH_coord(mpcca, mpcca.meta.x0, zeros(0))) == mpcca.meta.nvar * (mpcca.meta.nvar + 1) / 2
@test length(hessG_coord(mpcca, mpcca.meta.x0, zeros(0))) == mpcca.meta.nvar * (mpcca.meta.nvar + 1) / 2
