# on vérifie simplement que le constructeur par défaut fait son travail
n = 2
test_meta = MPCCModelMeta(n)

@test test_meta.nvar == n       # number of variables
@test test_meta.x0   == zeros(n)    # initial guess
@test size(test_meta.lvar) == size(zeros(n))   # vector of lower bounds
@test size(test_meta.uvar) == size(zeros(n))   # vector of upper bounds

@test test_meta.ncon == 0            # number of general constraints
@test test_meta.y0   == zeros(0)    # initial Lagrange multipliers
@test test_meta.lcon == zeros(0)    # vector of constraint lower bounds
@test test_meta.ucon == zeros(0)    # vector of constraint upper bounds

@test test_meta.ncc   == 0             # number of complementarity constraints
@test test_meta.lccG  == zeros(0)   # vector of constraint lower bounds of the complementarity constraint
@test test_meta.lccH  == zeros(0)   # vector of constraint upper bounds of the complementarity constraint

@test test_meta.minimize == true        # true if optimize == minimize
@test test_meta.nlo  == 1             # number of nonlinear objectives
@test test_meta.islp == false             # true if the problem is a linear program

try
    MPCCModelMeta(0)
    @test false
catch
    @test true
end

try
    show(test_meta)
    print(test_meta)
    @test true
catch
    @test false
end

@test MPCC.bound_constrained(test_meta)           == false
@test MPCC.unconstrained(test_meta)               == true
@test MPCC.linearly_constrained(test_meta)        == false
@test MPCC.equality_constrained(test_meta)        == false
@test complementarity_constrained(test_meta) == false
