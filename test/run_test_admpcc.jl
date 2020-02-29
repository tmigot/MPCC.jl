x0 = ones(6)
c(x) = [sum(x)]
G(x) = [x[1];x[3]]
H(x) = [x[2];x[4]]
admpcc = ADMPCCModel(rosenbrock,  x0,
                     lvar = fill(-10.0,size(x0)), uvar = fill(10.0,size(x0)),
                     y0 = [0.0], c = c, lcon = [-Inf], ucon = [6.],
                     G = G, H = H, lccG = zeros(2), lccH = zeros(2))

@test obj(admpcc, admpcc.meta.x0)  == 0.0
@test grad(admpcc, admpcc.meta.x0) == zeros(6)
@test issparse(hess(admpcc, admpcc.meta.x0))
@test issparse(jac(admpcc, admpcc.meta.x0))
@test cons(admpcc, admpcc.meta.x0) == [6.0, 1.0, 1.0, 1.0, 1.0]
@test viol(admpcc, admpcc.meta.x0) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

@test hessG(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6,6)
@test issparse(jacG(admpcc, admpcc.meta.x0))
@test consG(admpcc, admpcc.meta.x0) == ones(2)

@test hessH(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6,6)
@test sum(jacH(admpcc, admpcc.meta.x0)) == 2
@test consH(admpcc, admpcc.meta.x0) == ones(2)
