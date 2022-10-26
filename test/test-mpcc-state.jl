state = MPCCAtX(zeros(10), zeros(0), cGx = [0.0], cHx = [1.0])

@test state.x == zeros(10)
@test state.fx == nothing
@test state.gx == nothing
@test state.Hx == nothing
@test state.mu == nothing
@test state.cx == nothing
@test state.Jx == nothing

@test state.lambda == zeros(0)
@test state.current_time == nothing
@test (!(state.evals == nothing))

# On vérifie que la fonction update! fonctionne
update!(state, x = ones(10), fx = 1.0, gx = ones(10))
update!(state, lambda = ones(10), current_time = 1.0)
update!(state, Hx = ones(10, 10), mu = ones(10), cx = ones(10), Jx = ones(10, 10))

@test (false in (state.x .== 1.0)) == false #assez bizarre comme test...
@test state.fx == 1.0
@test (false in (state.gx .== 1.0)) == false
@test (false in (state.Hx .== 1.0)) == false
@test state.mu == ones(10)
@test state.cx == ones(10)
@test (false in (state.Jx .== 1.0)) == false
@test (false in (state.lambda .== 1.0)) == false
@test state.current_time == 1.0

reinit!(state)
@test state.x == ones(10)
@test state.fx == nothing
reinit!(state, x = zeros(10))
@test state.x == zeros(10)
@test state.fx == nothing
