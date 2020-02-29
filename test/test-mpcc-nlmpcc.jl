test1 = ex1()
#nlp = convert(AbstractNLPModel, test1)
nlp = NLMPCC(test1)

@test nlp.mod.meta.ncc == 1
@test obj(nlp, nlp.meta.x0) == 0.0
@test grad(nlp, nlp.meta.x0) == grad(test1,[Inf,Inf])
@test hess(nlp, nlp.meta.x0) == sparse(zeros(2,2))
