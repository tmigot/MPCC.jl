test1 = ex1()
#nlp = convert(AbstractNLPModel, test1)
nlp = NLMPCC(test1)

@test nlp.mod.meta.ncc == 1

@test nlp.meta.nvar == 2
@test nlp.meta.lvar == [-Inf, -Inf]
@test nlp.meta.uvar == [Inf, Inf]

@test nlp.meta.ncon == 4
@test nlp.meta.lcon == [-1.0, -0.0, -0.0, -Inf]
@test nlp.meta.ucon == [Inf, Inf, Inf, 0.0]

@test obj(nlp, nlp.meta.x0) == 0.0
@test grad(nlp, nlp.meta.x0) == grad(test1,[Inf,Inf])
@test hess(nlp, nlp.meta.x0) == sparse(zeros(2,2))
@test hess(nlp, nlp.meta.x0, obj_weight = 0.0) == sparse(zeros(2,2))

t0 = ones(2)
@test cons(nlp, t0) == [-1.0, 1.0, 1.0, 1.0]
@test jac(nlp, t0) == [0.0 -1.0 ; 1.0 0.0 ; 0.0 1.0 ; 1.0 1.0]
@test jac_coord(nlp, t0) == jac(nlp, t0)[:]
@test jac_structure(nlp) == ([1, 2, 3, 4, 1, 2, 3, 4], [1, 1, 1, 1, 2, 2, 2, 2])
@test jprod(nlp, t0, zeros(2))  == zeros(4)
@test jtprod(nlp, t0, zeros(4)) == zeros(2)
@test hess(nlp, t0, ones(4)) == tril([0.0 1.0 ; 1.0 0.0])
@test length(hess_coord(nlp, t0)) == 3
@test length(hess_coord(nlp, t0, ones(4))) == 3
@test hess_structure(nlp) == ([1, 2, 2], [1, 1, 2])
@test hprod(nlp, t0, ones(4), nlp.meta.x0) == [1.0, 1.0]
@test hprod(nlp, t0, nlp.meta.x0) == [0.0, 0.0]

@test viol(nlp, t0) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
