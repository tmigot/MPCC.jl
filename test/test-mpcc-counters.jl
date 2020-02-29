try_counters = MPCCCounters()

@test getfield(try_counters, :neval_obj)     == 0
@test getfield(try_counters, :neval_grad)    == 0
@test getfield(try_counters, :neval_cons)    == 0
@test getfield(try_counters, :neval_consG)   == 0
@test getfield(try_counters, :neval_consH)   == 0
@test getfield(try_counters, :neval_jcon)    == 0
@test getfield(try_counters, :neval_jgrad)   == 0
@test getfield(try_counters, :neval_jac)     == 0
@test getfield(try_counters, :neval_jacG)    == 0
@test getfield(try_counters, :neval_jacH)    == 0
@test getfield(try_counters, :neval_jprod)   == 0
@test getfield(try_counters, :neval_jGprod)  == 0
@test getfield(try_counters, :neval_jHprod)  == 0
@test getfield(try_counters, :neval_jtprod)  == 0
@test getfield(try_counters, :neval_jGtprod) == 0
@test getfield(try_counters, :neval_jHtprod) == 0
@test getfield(try_counters, :neval_hess)    == 0
@test getfield(try_counters, :neval_hessG)   == 0
@test getfield(try_counters, :neval_hessH)   == 0
@test getfield(try_counters, :neval_hprod)   == 0
@test getfield(try_counters, :neval_hGprod)  == 0
@test getfield(try_counters, :neval_hHprod)  == 0
@test getfield(try_counters, :neval_jhprod)  == 0

@test sum_counters(try_counters) == 0

reset!(try_counters)
@test sum_counters(try_counters) == 0
