mutable struct Bard1 <:AbstractMPCCModel
  meta :: MPCCModelMeta
  counters :: MPCCCounters
end

function Bard1(; x0 = zeros(5))

  nvar = length(x0)
  lvar = -Inf*ones(nvar)
  uvar =  Inf*ones(nvar)
  ncon = 1
  lcon = [0.]
  ucon = [0.]
  y0   = [0.]
  ncc  = 3
  lccG = zeros(3)
  lccH = zeros(3)

  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon

  meta = MPCCModelMeta(nvar, x0 = x0, y0=y0, ncc = ncc, lccG = lccG, lccH = lccH,
                             lvar = lvar, uvar = uvar,
                             ncon = ncon, lcon = lcon, ucon = ucon,
                             nnzj=nnzj, nnzh=nnzh,
                             minimize=true, islp=false, name="Bard1")

  return Bard1(meta, MPCCCounters())
end

function MPCC.obj(nlp :: Bard1, x :: AbstractVector)
  #increment!(nlp, :neval_obj)
  return (x[1]-5)^2+(2*x[2]+1)^2
end

function MPCC.grad!(nlp :: Bard1, x :: AbstractVector, gx :: AbstractVector)
  #increment!(nlp, :neval_grad)
  gx .= vcat(2*(x[1] - 5), 4 * (2 * x[2] + 1), zeros(3))
  return gx
end
function MPCC.cons_nl!(nlp :: Bard1, x :: AbstractVector, c :: AbstractVector)
  #increment!(nlp, :neval_cons)
  c[1:nlp.meta.ncon] .= [2*(x[2]-1)-1.5*x[1]+x[3]-0.5*x[4]+x[5]]
  return c
end

function MPCC.consG!(nlp :: Bard1, x :: AbstractVector, c :: AbstractVector)
  #increment!(nlp, :neval_consG)
  c[1:nlp.meta.ncc] .= vcat(3x[1]-x[2]-3, -x[1]+0.5x[2]+4, -x[1]-x[2]+7 )
  return c
end

function MPCC.consH!(nlp :: Bard1, x :: AbstractVector, c :: AbstractVector)
  #increment!(nlp, :neval_consH)
  c[1:nlp.meta.ncc] .= vcat(x[3], x[4], x[5])
  return c
end

function MPCC.jac_nl_structure!(nlp :: Bard1, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncon, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : nlp.meta.nnzj] .= getindex.(I, 1)[:]
  cols[1 : nlp.meta.nnzj] .= getindex.(I, 2)[:]
  return rows, cols
end

function MPCC.jac_nl_coord!(nlp :: Bard1, x :: AbstractVector, vals :: AbstractVector)
  #increment!(nlp, :neval_jac)
  Jx = [-1.5 2. 1. -0.5 1.]
  vals[1 : nlp.meta.nnzj] .= Jx[:]
  return vals
end

function MPCC.jacG_structure!(nlp :: Bard1, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncc, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : n*m] .= getindex.(I, 1)[:]
  cols[1 : n*m] .= getindex.(I, 2)[:]
  return rows, cols
end

function MPCC.jacG_coord!(nlp :: Bard1, x :: AbstractVector, vals :: AbstractVector)
  m, n = nlp.meta.ncc, nlp.meta.nvar
  #increment!(nlp, :neval_jacG)
  Jx = vcat([3 -1 0 0 0], [-1 0.5 0 0 0], [-1 -1 0 0 0])
  vals[1 : n*m] .= Jx[:]
  return vals
end

function MPCC.jacH_structure!(nlp :: Bard1, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncc, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : n*m] .= getindex.(I, 1)[:]
  cols[1 : n*m] .= getindex.(I, 2)[:]
  return rows, cols
end

function MPCC.jacH_coord!(nlp :: Bard1, x :: AbstractVector, vals :: AbstractVector)
  m, n = nlp.meta.ncc, nlp.meta.nvar
  #increment!(nlp, :neval_jacH)
  Jx = zeros(3, 5)
  Jx[1,1] = 1.
  Jx[2,2] = 1.
  Jx[3,3] = 1.
  vals[1 : n*m] .= Jx[:]
  return vals
end

function MPCC.jnlprod!(nlp :: Bard1, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  #increment!(nlp, :neval_jprod)
  Jv[1:nlp.meta.ncon] .= [-1.5 2. 1. -0.5 1.] * v
  return Jv
end

function MPCC.jnltprod!(nlp :: Bard1, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  #increment!(nlp, :neval_jtprod)
  Jtv[1:nlp.meta.nvar] .= [-1.5 2. 1. -0.5 1.]' * v
  return Jtv
end

function MPCC.jGprod!(nlp :: Bard1, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  #increment!(nlp, :neval_jGprod)
  Jv[1:nlp.meta.ncc] .=  vcat([3 -1 0 0 0], [-1 0.5 0 0 0], [-1 -1 0 0 0]) * v
  return Jv
end

function MPCC.jGtprod!(nlp :: Bard1, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  #increment!(nlp, :neval_jGtprod)
  Jtv[1:nlp.meta.nvar] .=  vcat([3 -1 0 0 0], [-1 0.5 0 0 0], [-1 -1 0 0 0])' * v
  return Jtv
end

function MPCC.jHprod!(nlp :: Bard1, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  #increment!(nlp, :neval_jHprod)
  Jx = zeros(3, 5)
  Jx[1,1] = 1.
  Jx[2,2] = 1.
  Jx[3,3] = 1.
  Jv[1:nlp.meta.ncc] .= Jx * v
  return Jv
end

function MPCC.jHtprod!(nlp :: Bard1, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  #increment!(nlp, :neval_jHtprod)
  Jx = zeros(3, 5)
  Jx[1,1] = 1.
  Jx[2,2] = 1.
  Jx[3,3] = 1.
  Jtv[1:nlp.meta.nvar] .= Jx' * v
  return Jtv
end

#function hess(nlp :: Bard1, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))
#  #increment!(nlp, :neval_hess)
#  ℓ(x) = obj_weight * nlp.f(x)
#  Hx = ForwardDiff.hessian(ℓ, x)
#  return tril(Hx)
#end

#function hess(nlp :: Bard1, x :: AbstractVector, y :: AbstractVector; obj_weight :: Real = one(eltype(x)))
#  #increment!(nlp, :neval_hess)
#  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
#  Hx = ForwardDiff.hessian(ℓ, x)
#  return tril(Hx)
#end

function MPCC.hess_structure!(nlp :: Bard1, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function MPCC.hess_coord!(nlp :: Bard1, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  #increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x)
  Hx = zeros(5,5)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function MPCC.hess_coord!(nlp :: Bard1, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  #increment!(nlp, :neval_hess)
  ncon, ncc = nlp.meta.ncon, nlp.meta.ncc
  # ℓ(x) = obj_weight * f(x) + λ₁ c(x) - λ₂ G(x) - λ₃ H(x)
  #ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y[1:ncon]) - dot(nlp.G(x), y[ncon+1:ncon+ncc]) - dot(nlp.H(x), y[ncon+ncc+1:ncon+2*ncc])
  Hx = zeros(5,5)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function MPCC.hessG(nlp :: Bard1, x :: AbstractVector, y :: AbstractVector)
  #increment!(nlp, :neval_hessG)
  #ℓ(x) = dot(nlp.G(x), y)
  Hx = zeros(5,5)
  return tril(Hx)
end

function MPCC.hessG_structure!(nlp :: Bard1, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function MPCC.hessG_coord!(nlp :: Bard1, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector)
  #increment!(nlp, :neval_hessG)
  #ℓ(x) = dot(nlp.G(x), y)
  Hx = zeros(5,5)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function MPCC.hessH(nlp :: Bard1, x :: AbstractVector, y :: AbstractVector)
  #increment!(nlp, :neval_hessH)
  #ℓ(x) = dot(nlp.H(x), y)
  Hx = zeros(5,5)
  return tril(Hx)
end

function MPCC.hessH_structure!(nlp :: Bard1, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function MPCC.hessH_coord!(nlp :: Bard1, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector)
  #increment!(nlp, :neval_hess)
  #ℓ(x) = dot(nlp.H(x), y)
  Hx = zeros(5,5)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function MPCC.hprod!(nlp :: Bard1, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  #increment!(nlp, :neval_hprod)
  #ℓ(x) = obj_weight * nlp.f(x)
  Hv .= zeros(5)
  return Hv
end

function MPCC.hprod!(nlp :: Bard1, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  #increment!(nlp, :neval_hprod)
  ncon, ncc = nlp.meta.ncon, nlp.meta.ncc
  #ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y[1:ncon]) - dot(nlp.G(x), y[ncon+1:ncon+ncc]) - dot(nlp.H(x), y[ncon+ncc+1:ncon+2*ncc])
  Hv .= zeros(5)
  return Hv
end

function MPCC.hGprod!(nlp :: Bard1, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector)
  #increment!(nlp, :neval_hGprod)
  #ℓ(x) = dot(nlp.G(x), y)
  Hv .= zeros(5)
  return Hv
end

function MPCC.hHprod!(nlp :: Bard1, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector)
  #increment!(nlp, :neval_hHprod)
  #ℓ(x) = dot(nlp.H(x), y)
  Hv .= zeros(5) #ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end
