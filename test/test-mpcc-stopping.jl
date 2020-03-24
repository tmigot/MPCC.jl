# We create a simple function to test
A = rand(5, 5);
Q = A' * A

f(x) = x' * Q * x
nlp = ADNLPModel(f, zeros(5))
mpcc = MPCCNLPs(nlp)
nlp_at_x = MPCCAtX(zeros(5), zeros(0))
stop_nlp = MPCCStopping(mpcc, SStat, nlp_at_x, optimality0 = 0.0)


a = zeros(5)
fill_in!(stop_nlp, a)

# we make sure that the fill_in! function works properly
@test obj(mpcc, a) == stop_nlp.current_state.fx
@test grad(mpcc, a) == stop_nlp.current_state.gx
@test stop_nlp.meta.optimality0 == 0.0

# we make sure the optimality check works properly
@test stop!(stop_nlp)
# we make sure the counter of stop works properly
@test stop_nlp.meta.nb_of_stop == 1

reinit!(stop_nlp, rstate = true, x = ones(5))
@test stop_nlp.current_state.x == ones(5)
@test stop_nlp.current_state.fx == nothing
@test stop_nlp.meta.nb_of_stop == 0

#We know test how to initialize the counter:
test_max_cntrs = _init_max_counters(obj = 2)
stop_nlp_cntrs = MPCCStopping(mpcc, max_cntrs = test_max_cntrs)
@test stop_nlp_cntrs.max_cntrs[:neval_obj] == 2
@test stop_nlp_cntrs.max_cntrs[:neval_grad] == 40000
@test stop_nlp_cntrs.max_cntrs[:neval_sum] == 40000*11

x0 = ones(6)
c(x) = [sum(x)]
mp2 = ADNLPModel(rosenbrock,  x0,
                 lvar = fill(-10.0,size(x0)), uvar = fill(10.0,size(x0)),
                 y0 = [0.0], c = c, lcon = [-Inf], ucon = [6.])
nlp2 = MPCCNLPs(mp2)
nlp_at_x_c = MPCCAtX(x0, NaN*ones(nlp2.meta.ncon))
stop_nlp_c = MPCCStopping(nlp2, SStat, nlp_at_x_c)

a = zeros(6)
fill_in!(stop_nlp_c, a)

@test cons(nlp2, a) == stop_nlp_c.current_state.cx
@test jac(nlp2, a) == stop_nlp_c.current_state.Jx

@test stop!(stop_nlp_c) == false
# we make sure the counter of stop works properly
@test stop_nlp_c.meta.nb_of_stop == 1

sol = ones(6)
fill_in!(stop_nlp_c, sol)

@test stop!(stop_nlp_c) == true

stop_nlp_default = MPCCStopping(nlp2, atol = 1.0)
fill_in!(stop_nlp_default, sol)
@test stop_nlp_default.meta.atol == 1.0
@test stop!(stop_nlp_default) == true

#Keywords in the stop! call
nlp_at_x_kargs = MPCCAtX(x0, NaN*ones(nlp2.meta.ncon))
stop_nlp_kargs = MPCCStopping(nlp2, (x,y; test = 1.0, kwargs...) -> MStat(x,y; kwargs...) + test, nlp_at_x_c)
fill_in!(stop_nlp_kargs, sol)
@test stop!(stop_nlp_kargs) == false
@test stop!(stop_nlp_kargs, test = 0.0) == true

test1 = ex1()
stop_w = MPCCStopping(test1, WStat, MPCCAtX(x0, zeros(nlp2.meta.ncon)))
stop_s = MPCCStopping(test1, SStat, MPCCAtX(x0, zeros(nlp2.meta.ncon)))
Wpoint, Spoint = zeros(2), [0.0, 1.0]
fill_in!(stop_w, Wpoint)
@test stop_w.current_state.lambdaG == [1.0]
@test stop_w.current_state.lambdaH == [-1.0]
#lambdaW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
#update_and_stop!(stop_w, lambdaG = [1.0], lambdaH = [-1.0])
stop!(stop_w)
@test status(stop_w) == :Optimal
fill_in!(stop_s, Spoint)
@test stop_s.current_state.lambdaG == [1.0]
@test stop_s.current_state.lambdaH == [0.0]
#lambdaS = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
#update_and_stop!(stop_s, lambdaG = [1.0], lambdaH = [0.0])
stop!(stop_s)
@test status(stop_s) == :Optimal
reinit!(stop_s)
stop_s.optimality_check = WStat
stop!(stop_s)
@test status(stop_s) == :Optimal

a = MPCC.neval_obj(stop_s.pb)
decrement!(stop_s.pb, :neval_obj)
@test MPCC.neval_obj(stop_s.pb) == a-1
reset!(stop_s.pb)
@test neval_consG(stop_s.pb) == 0

testM = ex3()
stop_m = MPCCStopping(testM, MStat, MPCCAtX(zeros(3), zeros(2)))
stop_c = MPCCStopping(testM, CStat, MPCCAtX(zeros(3), zeros(2)))
fill_in!(stop_m, zeros(3))
#the sign computed by fill_in are not M-stat. optimized
update_and_stop!(stop_m, lambda = [0.75, 0.25], lambdaG = [-2.0], lambdaH = [0.0])
@test status(stop_m) == :Optimal

fill_in!(stop_c, ones(3))
stop!(stop_c)
@test status(stop_c) == :Unknown

#Unconstrained MPCC:
test_unc = MPCCNLPs(ADNLPModel(rosenbrock, zeros(6)))
stop_unc = MPCCStopping(test_unc, MStat, MPCCAtX(zeros(6), zeros(0)))
fill_in!(stop_unc, ones(6))
@test stop_unc.current_state.lambda == zeros(0)
@test stop_unc.current_state.lambdaG == nothing
@test stop_unc.current_state.lambdaH == nothing
@test stop_unc.current_state.mu == nothing
