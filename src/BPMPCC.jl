"""
This type convert a bilevel problem as an MPCCModel.

Bilevel Program:
````
min_{x,y} f1(x,y)
s.t.  lcon    <= c(x)   <= ucon
      lvar    <= x      <= uvar
      y in argmin_{y} f2(x,y)
                 s.t. ll_lcon <= g(x,y) <= ll_ucon (l1,  l2)
                      ll_lvar <= y      <= ll_uvar (mu1, mu2)
````
MPCC reformulation of the Bilevel Program:
       min_{x,y,l1,l2,mu1,mu2} f1(x,y)
       s.t.  lvar <= x      <= uvar
             lcon <= c(x)   <= ucon
             0    <= ∇_y f2(x,y) - ∇_y g(x,y)' * (l1 - l2) - (mu1 - mu2) <= 0
             G(x):=(  g(x,y)  - ll_lcon)
                      ll_ucon - g(x,y) )
                      ll_uvar - y
                      y - ll_lvar )
             H(x):=( l1, l2, mu1, mu2 )
             0 <= G(x) _|_ H(x) >= 0
````
Gradient, Jacobian and Hessian are computed using automatic differentiation

set no_third_derivative as true, to ignore the 3rd derivative of g.
"""
mutable struct BPMPCCModel <: AbstractMPCCModel
  meta     :: MPCCModelMeta
  counters :: MPCCCounters

  # Functions
  f1 :: Function
  f2 :: Function
  c  :: Function
  g  :: Function

  ll_lcon :: AbstractVector
  ll_ucon :: AbstractVector
  ll_lvar :: AbstractVector
  ll_uvar :: AbstractVector

  no_third_derivative :: Bool

  m      :: Int #size of the lower level dimension
  n      :: Int #size of the upper level dimension + m
  llncon :: Int #number of constraints in the lower level problem
end

#m0 is the initial guess for the lower leverl problem
function BPMPCCModel(f1, x0::AbstractVector, m0::AbstractVector;
                     y0::AbstractVector = eltype(x0)[],
                     lvar::AbstractVector = eltype(x0)[],
                     uvar::AbstractVector = eltype(x0)[],
                     lcon::AbstractVector = eltype(x0)[],
                     ucon::AbstractVector = eltype(x0)[],
                     c = (args...)->throw(NotImplementedError("c")),
                     g = (args...)->throw(NotImplementedError("g")),
                     f2 = (args...)->throw(NotImplementedError("f2")),
                     ll_lcon::AbstractVector = eltype(x0)[],
                     ll_ucon::AbstractVector = eltype(x0)[],
                     ll_lvar::AbstractVector = eltype(x0)[],
                     ll_uvar::AbstractVector = eltype(x0)[],
                     name::String = "Generic",
                     lin::AbstractVector{<: Integer}=Int[],
                     no_third_derivative::Bool=false)

  m, n, llncon = length(ll_lvar), length(x0), length(ll_ucon)

  nvar = n + 3 * m + 2 * llncon
  length(lvar) == 0 && (lvar = -Inf*ones(n))
  length(uvar) == 0 && (uvar =  Inf*ones(n))
  lvar = vcat(lvar, -Inf*ones(3 * m + 2 * llncon))
  uvar = vcat(uvar,  Inf*ones(3 * m + 2 * llncon))

  ncon = maximum([length(lcon); length(ucon); length(y0)]) + length(m0)
  lcon = vcat(lcon, zeros(m))
  ucon = vcat(ucon, zeros(m))

  ncc  = 2 * m + 2 * llncon
  lccG = zeros(ncc)
  lccH = zeros(ncc)
  ini0 = vcat(x0, m0, zeros(2 * m + 2 * llncon))

  meta = MPCCModelMeta(nvar, x0 = ini0, lvar = lvar, uvar = uvar,
                                        ncon = ncon, lcon = lcon,
                                        ucon = ucon, ncc = ncc,
                                        lccG = lccG, lccH = lccH)

  return BPMPCCModel(meta, MPCCCounters(), f1, f2, c, g,
                     ll_lcon, ll_ucon, ll_lvar, ll_uvar,
                     no_third_derivative, m, n, llncon)
end

"""
_var_bp(:: BPMPCCModel, :: AbstractVector):
return the details of the different variables in the MPCC corresponding
to the bilevel program.
return: x, y, l1, l2, m1, m2
"""
function _var_bp(nlp :: BPMPCCModel, x :: AbstractVector)

 if length(x) != nlp.meta.nvar throw("Error: wrong vector size") end

 m, n, ncon = nlp.m, nlp.n, nlp.llncon

 x0, y0 = x[1:n], x[n+1:n+m]
 l1, l2 = x[n+m+1:n+m+ncon], x[n+m+ncon+1:n+m+2*ncon]
 m1, m2 = x[n+m+2*ncon+1:n+2*m+2*ncon], x[n+2*m+2*ncon+1:n+3*m+2*ncon]

 return x0, y0, l1, l2, m1, m2
end

"""
JLag: gives the gradient w.r.t. y of the Lagrangian of the lower level problem.

JLag(:: BPMPCCModel, :: Function, :: Function, :: AbstractVector)
"∇_y f2(x,y) - ∇_y g(x,y)' * v - mu"
"""
function JLag(nlp :: BPMPCCModel,
              f   :: Function,
              c   :: Function,
              x   :: AbstractVector)
    Jtv = zeros(nlp.m)
    x0, y0, l1, l2, m1, m2 = _var_bp(nlp, x)
    lag(x) = nlp.llncon > 0 ? f(x) - dot(c(x), l1-l2) : f(x)
    Jtv = ForwardDiff.gradient(lag, x)[nlp.n+1:nlp.n+nlp.m] - (m1 - m2)
    return Jtv
end

"""
HLag: gives the hessian w.r.t. y of the Lagrangian of the lower level problem.

HLag(:: BPMPCCModel, :: Function, :: Function, :: AbstractVector)
"∇^2_y f2(x,y) - sum_i ∇^2_y g_i(x,y)' * v_i"
"""
function HLag(nlp :: BPMPCCModel,
              f   :: Function,
              c   :: Function,
              x   :: AbstractVector)
    Jtv = zeros(nlp.n+nlp.m, nlp.n+nlp.m)
    x0, y0, l1, l2, m1, m2 = _var_bp(nlp, x)
    lag(x) = nlp.llncon > 0 ? f(x) - dot(c(x), l1-l2) : f(x)
    Jtv = ForwardDiff.hessian(lag, x)
    Jtvy = Jtv[nlp.n+1:nlp.n+nlp.m,nlp.n+1:nlp.n+nlp.m]
    Jtvx = Jtv[nlp.n+1:nlp.n+nlp.m,1:nlp.n]
    return Jtvy, Jtvx
end

function obj(nlp :: BPMPCCModel, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return nlp.f1(x)
end

function grad!(nlp :: BPMPCCModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  ForwardDiff.gradient!(view(g, 1:length(x)), nlp.f1, x)
  return g
end

function cons_nl!(nlp :: BPMPCCModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  Jl = JLag(nlp, nlp.f1, nlp.g, x)
  c[1:nlp.meta.ncon] .= vcat(nlp.c(x), Jl)
  return c
end

function consG!(nlp :: BPMPCCModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_consG)
  gxy = nlp.g(x)
  y   = x[nlp.n+1:nlp.n+nlp.m]
  c[1:nlp.meta.ncc] .= vcat( gxy - nlp.ll_lcon,
                            -gxy + nlp.ll_ucon,
                             y - nlp.ll_lvar,
                            -y + nlp.ll_uvar)
  return c
end

function consH!(nlp :: BPMPCCModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_consH)
  c[1:nlp.meta.ncc] .= x[nlp.n+nlp.m+1:nlp.meta.nvar]
  return c
end

function jac_nl_structure!(nlp  :: BPMPCCModel,
                           rows :: AbstractVector{<: Integer},
                           cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncon, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows .= getindex.(I, 1)[:]
  cols .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_nl_coord!(nlp  :: BPMPCCModel,
                       x    :: AbstractVector,
                       vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  m, n, llncon = nlp.m, nlp.n, nlp.llncon

  Jx  = ForwardDiff.jacobian(nlp.c, x)
  Jg  = ForwardDiff.jacobian(nlp.g, x)
  Jyy, Jyx = HLag(nlp, nlp.f1, nlp.g, x)
  Jyl = llncon >0 ? vcat(Jg[:,n+1:n+m], -Jg[:,n+1:n+m]) : zeros(m, 0)
  Jym = hcat(diagm(0 => ones(m)),-diagm(0 => ones(m)))
  Jl  = hcat(Jyx, Jyy, Jyl, Jym)
  J   = vcat(Jx, Jl)

  vals .= J[:]
  return vals
end

function jnlprod!(nlp :: BPMPCCModel,
                  x   :: AbstractVector,
                  v   :: AbstractVector,
                  Jv  :: AbstractVector)
  increment!(nlp, :neval_jprod)
  #Jv[1:nlp.meta.ncon] = ForwardDiff.derivative(t -> nlp.c(x + t * v), 0)
  Jv .= jac_nl(nlp, x) * v
  return Jv
end

function jnltprod!(nlp :: BPMPCCModel,
                   x   :: AbstractVector,
                   v   :: AbstractVector,
                   Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  #Jtv[1:nlp.meta.nvar] = ForwardDiff.gradient(x -> dot(nlp.c(x), v), x)
  Jtv .= jac_nl(nlp, x)' * v
  return Jtv
end

function jacG_structure!(nlp  :: BPMPCCModel,
                         rows :: AbstractVector{<: Integer},
                         cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncc, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
   rows[1 : n*m] .= getindex.(I, 1)[:]
   cols[1 : n*m] .= getindex.(I, 2)[:]
  return rows, cols
end

function jacG_coord!(nlp  :: BPMPCCModel,
                     x    :: AbstractVector,
                     vals :: AbstractVector)
  increment!(nlp, :neval_jacG)
  ncc, nvar = nlp.meta.ncc, nlp.meta.nvar

  Jg  = ForwardDiff.jacobian(nlp.g, x)
  Idy = hcat(zeros(nlp.m,nlp.n),diagm(0 => ones(nlp.m)),zeros(nlp.m,ncc))
  Jx  = vcat(Jg, -Jg, Idy, - Idy)

  vals[1 : nvar * ncc] .= Jx[:]
  return vals
end

function jGprod!(nlp :: BPMPCCModel,
                 x   :: AbstractVector,
                 v   :: AbstractVector,
                 Jv  :: AbstractVector)
  increment!(nlp, :neval_jGprod)
  #Jv[1:nlp.meta.ncc] = ForwardDiff.derivative(t -> nlp.G(x + t * v), 0)
  Jv[1:nlp.meta.ncc] .= jacG(nlp, x) * v
  return Jv
end

function jGtprod!(nlp :: BPMPCCModel,
                  x   :: AbstractVector,
                  v   :: AbstractVector,
                  Jtv :: AbstractVector)
  increment!(nlp, :neval_jGtprod)
  #Jtv[1:nlp.meta.nvar] = ForwardDiff.gradient(x -> dot(nlp.G(x), v), x)
  Jtv[1:nlp.meta.nvar] .= jacG(nlp, x)' * v
  return Jtv
end

function jacH_structure!(nlp  :: BPMPCCModel,
                         rows :: AbstractVector{<: Integer},
                         cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncc, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : n*m] .= getindex.(I, 1)[:]
  cols[1 : n*m] .= getindex.(I, 2)[:]
  return rows, cols
end

function jacH_coord!(nlp  :: BPMPCCModel,
                     x    :: AbstractVector,
                     vals :: AbstractVector)
  m, n = nlp.meta.ncc, nlp.meta.nvar
  increment!(nlp, :neval_jacH)
  Jx = hcat(zeros(m, nlp.n + nlp.m), diagm(0 => ones(m)))

  vals[1 : n*m] .= Jx[:]
  return vals
end

function jHprod!(nlp :: BPMPCCModel,
                 x   :: AbstractVector,
                 v   :: AbstractVector,
                 Jv  :: AbstractVector)
  increment!(nlp, :neval_jHprod)
  #Jv[1:nlp.meta.ncc] = ForwardDiff.derivative(t -> nlp.G(x + t * v), 0)
  Jv[1:nlp.meta.ncc] .= jacH(nlp, x) * v
  return Jv
end

function jHtprod!(nlp :: BPMPCCModel,
                  x   :: AbstractVector,
                  v   :: AbstractVector,
                  Jtv :: AbstractVector)
  increment!(nlp, :neval_jHtprod)
  #Jtv[1:nlp.meta.nvar] = ForwardDiff.gradient(x -> dot(nlp.G(x), v), x)
  Jtv[1:nlp.meta.nvar] .= jacH(nlp, x)' * v
  return Jtv
end

function hess_structure!(nlp :: BPMPCCModel,
                        rows :: AbstractVector{<: Integer},
                        cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp        :: BPMPCCModel,
                     x          :: AbstractVector,
                     vals       :: AbstractVector;
                     obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f1(x)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(nlp        :: BPMPCCModel,
                     x          :: AbstractVector,
                     y          :: AbstractVector,
                     vals       :: AbstractVector;
                     obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  ncon, ncc, m, nt = nlp.meta.ncon, nlp.meta.ncc, nlp.m, nlp.no_third_derivative
  llncon = nlp.llncon

  temp(x) = nt ? obj_weight * nlp.f1(x) + dot(nlp.c(x),y[1:ncon - m]) : obj_weight * nlp.f1(x) + dot(cons_nl(nlp,x), y[1:ncon])
  ℓ(x) = llncon > 0 ? temp(x) + dot(nlp.g(x), y[ncon+1:ncon+llncon]) - dot(nlp.g(x), y[ncon+llncon+1:ncon+2*llncon]) : temp(x)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hessG(nlp :: BPMPCCModel, x :: AbstractVector, y :: AbstractVector)
  increment!(nlp, :neval_hessG)
  lln = nlp.llncon
  ℓ(x) = lln > 0 ? dot(nlp.g(x), y[1:lln]) - dot(nlp.g(x), y[lln+1:2*lln]) : 0
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hessG_structure!(nlp :: BPMPCCModel,
                         rows :: AbstractVector{<: Integer},
                         cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hessG_coord!(nlp  :: BPMPCCModel,
                      x    :: AbstractVector,
                      y    :: AbstractVector,
                      vals :: AbstractVector)
  increment!(nlp, :neval_hessG)
  lln = nlp.llncon
  ℓ(x) = lln > 0 ? dot(nlp.g(x), y[1:lln]) - dot(nlp.g(x), y[lln+1:2*lln]) : 0
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end


function hessH(nlp :: BPMPCCModel, x :: AbstractVector, y :: AbstractVector)
  increment!(nlp, :neval_hessH)
  Hx = zeros(nlp.meta.nvar, nlp.meta.nvar)
  return tril(Hx)
end

function hessH_structure!(nlp  :: BPMPCCModel,
                          rows :: AbstractVector{<: Integer},
                          cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hessH_coord!(nlp  :: BPMPCCModel,
                      x    :: AbstractVector,
                      y    :: AbstractVector,
                      vals :: AbstractVector)
  increment!(nlp, :neval_hess)
  ℓ(x) = dot(nlp.H(x), y)
  Hx = zeros(nlp.meta.nvar, nlp.meta.nvar)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hprod!(nlp        :: BPMPCCModel,
                x          :: AbstractVector,
                v          :: AbstractVector,
                Hv         :: AbstractVector;
                obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  ℓ(x) = obj_weight * nlp.f1(x)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hprod!(nlp        :: BPMPCCModel,
                x          :: AbstractVector,
                y          :: AbstractVector,
                v          :: AbstractVector,
                Hv         :: AbstractVector;
                obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  ncon, ncc, m, nt = nlp.meta.ncon, nlp.meta.ncc, nlp.m, nlp.no_third_derivative
  llncon = nlp.llncon

  temp(x) = nt ? obj_weight * nlp.f1(x) + dot(nlp.c(x),y[1:ncon - m]) : obj_weight * nlp.f1(x) + dot(cons_nl(nlp,x), y[1:ncon])
  ℓ(x) = llncon > 0 ? temp(x) + dot(nlp.g(x), y[ncon+1:ncon+llncon]) - dot(nlp.g(x), y[ncon+llncon+1:ncon+2*llncon]) : temp(x)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hGprod!(nlp :: BPMPCCModel,
                 x   :: AbstractVector,
                 y   :: AbstractVector,
                 v   :: AbstractVector,
                 Hv  :: AbstractVector)
  increment!(nlp, :neval_hGprod)
  lln = nlp.llncon
  ℓ(x) = lln > 0 ? dot(nlp.g(x), y[1:lln]) - dot(nlp.g(x), y[lln+1:2*lln]) : 0
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hHprod!(nlp :: BPMPCCModel,
                 x   :: AbstractVector,
                 y   :: AbstractVector,
                 v   :: AbstractVector,
                 Hv  :: AbstractVector)
  increment!(nlp, :neval_hHprod)
  Hv .= zeros(nlp.meta.nvar)
  return Hv
end
