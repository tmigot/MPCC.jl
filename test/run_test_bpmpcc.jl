using Printf, Test, Main.MPCC

x0 = ones(6)
m0 = ones(2)
c(x) = [sum(x)]
f1(x) = sum(x[1:6].^2) - sum(x[7:8].^2)
f2(x) = sum(x[7:8].^2)
g(x) = []
ll_lvar = zeros(2)
ll_uvar = ones(2)
ll_lcon, ll_ucon = zeros(0), zeros(0)

bpmpcc = BPMPCCModel(f1,  x0, m0,
                     lvar = fill(-10.0,size(x0)), uvar = fill(10.0,size(x0)),
                     y0 = [0.0], c = c, lcon = [-Inf], ucon = [6.],
                     f2 = f2, g = g, ll_lvar = ll_lvar, ll_uvar = ll_uvar,
                     ll_lcon = ll_lcon, ll_ucon = ll_ucon, no_third_derivative = true)

@test bpmpcc.n == 6
@test bpmpcc.m == 2
@test bpmpcc.llncon == 0

@test bpmpcc.meta.nvar == 12
@test length(bpmpcc.meta.lvar) == 12
@test length(bpmpcc.meta.uvar) == 12
@test bpmpcc.meta.lvar[1:6] == fill(-10.0,size(x0))
@test bpmpcc.meta.uvar[1:6] == fill( 10.0,size(x0))
@test bpmpcc.meta.lvar[7:8] == -Inf * ones(2)
@test bpmpcc.meta.uvar[7:8] ==  Inf * ones(2)

@test bpmpcc.meta.ncon == 3
@test length(bpmpcc.meta.lcon) == 3
@test length(bpmpcc.meta.ucon) == 3
@test length(bpmpcc.meta.y0)   == 3

@test bpmpcc.meta.ncc  == 2 * 2
@test bpmpcc.meta.lccG == zeros(bpmpcc.meta.ncc)
@test bpmpcc.meta.lccG == bpmpcc.meta.lccH

t0 = rand(12)
x0, y0, l1, l2, m1, m2 = Main.MPCC._var_bp(bpmpcc, t0)
@test (x0 == t0[1:6]) && (y0 == t0[7:8]) && (l1 == zeros(0)) && (l2 == zeros(0)) && (m1 == t0[9:10]) && (m2 == t0[11:12])
@test obj(bpmpcc, t0) == f1(t0[1:8])
@test grad(bpmpcc, t0)[1:6]  ==  2*t0[1:6]
@test grad(bpmpcc, t0)[7:8]  == -2*t0[7:8]
@test grad(bpmpcc, t0)[9:12] == zeros(4)

@test length(cons_nl(bpmpcc, t0)) == 3 #d'autres tests?
@test consG(bpmpcc, t0) == vcat(t0[7:8] - ll_lvar, ll_uvar - t0[7:8])
@test consH(bpmpcc, t0) == t0[9:12]

@test size(jac_nl(bpmpcc, t0)) == (3,12)
@test jnlprod(bpmpcc, t0, zeros(12)) == zeros(3)
@test jnltprod(bpmpcc, t0, zeros(3)) == zeros(12)
@test size(jacG(bpmpcc, t0)) == (4,12)
@test jGprod(bpmpcc, t0, zeros(12)) == zeros(4)
@test jGtprod(bpmpcc, t0, zeros(4)) == zeros(12)
@test size(jacH(bpmpcc, t0)) == (4,12)
@test jHprod(bpmpcc, t0, zeros(12)) == zeros(4)
@test jHtprod(bpmpcc, t0, zeros(4)) == zeros(12)

@test hessG(bpmpcc, t0, zeros(4)) == zeros(12,12)
@test hessH(bpmpcc, t0, zeros(4)) == zeros(12,12)
@test hGprod(bpmpcc, t0, zeros(4), t0) == zeros(12)
@test hHprod(bpmpcc, t0, zeros(4), t0) == zeros(12)

@test hess(bpmpcc, t0, zeros(7), obj_weight = 0.0) == zeros(12,12)
@test hess(bpmpcc, t0, obj_weight = 0.0) == zeros(12,12)
@test hprod(bpmpcc, t0, zeros(7), zeros(12), obj_weight = 0.0) == zeros(12)
@test hprod(bpmpcc, t0, zeros(12), obj_weight = 0.0) == zeros(12)

@test hessH_structure(bpmpcc) == hessG_structure(bpmpcc)
@test length(hessH_coord(bpmpcc, bpmpcc.meta.x0, zeros(2))) == bpmpcc.meta.nvar * (bpmpcc.meta.nvar + 1) / 2
@test length(hessG_coord(bpmpcc, bpmpcc.meta.x0, zeros(2))) == bpmpcc.meta.nvar * (bpmpcc.meta.nvar + 1) / 2
