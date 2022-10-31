mutable struct MPCCNLPs{T,S} <: AbstractMPCCModel{T,S}

    mp::AbstractNLPModel{T,S}
    G::AbstractNLPModel{T,S}
    H::AbstractNLPModel{T,S}

    meta::NLPModelMeta{T,S}
    cc_meta::MPCCModelMeta{T,S}
    counters::Counters
    cc_counters::MPCCCounters

end

function MPCCNLPs(
    mp::AbstractNLPModel{T,S},
    G::AbstractNLPModel{T,S},
    H::AbstractNLPModel{T,S};
    kwargs...,
) where {T,S}

    ncc = G.meta.ncon
    @lencheck ncc G.meta.y0 H.meta.y0
    nvar = mp.meta.nvar
    @lencheck nvar G.meta.x0 H.meta.x0
    cc_meta = MPCCModelMeta(
        nvar,
        ncc,
        lccG = G.meta.lcon,
        lccH = H.meta.lcon,
        yG = G.meta.y0,
        yH = H.meta.y0,
        nnzjG = G.meta.nnzj,
        nnzjH = H.meta.nnzj,
    )
    meta = NLPModelMeta(
        nvar;
        x0 = mp.meta.x0,
        lvar = mp.meta.lvar,
        uvar = mp.meta.uvar ,
        ncon = mp.meta.ncon,
        y0 = mp.meta.y0,
        lcon = mp.meta.lcon,
        ucon = mp.meta.ucon,
        nnzj = mp.meta.nnzj,
        lin_nnzj = mp.meta.lin_nnzj,
        nln_nnzj = mp.meta.nln_nnzj,
        nnzh = nvar * (nvar + 1) / 2, # mp.meta.nnzh + G.meta.nnzh + H.meta.nnzh
        lin = mp.meta.lin,
        minimize = mp.meta.minimize,
        islp = mp.meta.islp,
        name = mp.meta.name,
      )

    return MPCCNLPs(mp, G, H, meta, cc_meta, Counters(), MPCCCounters())
end

for meth in (:obj, :grad!, :objgrad!, :objcons!, :jac_op!, :ghjvprod!, :jth_hprod!)
    @eval begin
        $meth(nlp::MPCCNLPs, args...; kwargs...) = $meth(nlp.mp, args...; kwargs...)
    end
end
cons!(nlp::MPCCNLPs, x::AbstractVector, cx::AbstractVector) = cons!(nlp.mp, x, cx)
jac_coord!(nlp::MPCCNLPs, x::AbstractVector, vals::AbstractVector) =
    jac_coord!(nlp.mp, x, vals)
jac_structure!(nlp::MPCCNLPs, rows::AbstractVector, cols::AbstractVector) =
    jac_structure!(nlp.mp, rows, cols)
jprod!(nlp::MPCCNLPs, x::AbstractVector, v::AbstractVector, Jv::AbstractVector) =
    jprod!(nlp.mp, x, v, Jv)
jtprod!(nlp::MPCCNLPs, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) =
    jtprod!(nlp.mp, x, v, Jtv)

"""
Evaluate ``G(x)``, the constraints at `x`.
"""
function consG!(mod::MPCCNLPs, x::Vector, c::AbstractVector)
    increment!(mod, :neval_consG)
    cons!(mod.G, x, c)
end

"""
Evaluate ``H(x)``, the constraints at `x`.
"""
function consH!(mod::MPCCNLPs, x::Vector, c::AbstractVector)
    increment!(mod, :neval_consH)
    cons!(mod.H, x, c)
end

function jacG_structure!(
    mod::MPCCNLPs,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_structure!(mod.G, rows, cols)
end

function jacH_structure!(
    mod::MPCCNLPs,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_structure!(mod.H, rows, cols)
end

function jacG_coord!(mod::MPCCNLPs, x::AbstractVector, vals::AbstractVector)
    jac_coord!(mod.G, x, vals)
end

function jacH_coord!(mod::MPCCNLPs, x::AbstractVector, vals::AbstractVector)
    jac_coord!(mod.H, x, vals)
end

"""
Jacobienne des contraintes actives à precmpcc près

(Tangi: copie de colonne si contrainte d'égalité?)
"""
function jacl(mod::MPCCNLPs, x::Vector) #bad name
    A, Il, Iu, Ig, Ih, IG, IH = jac_actif(mod, x, 0.0)
    return A
end

function jac_actif(mod::MPCCNLPs, x::Vector, prec::Float64)

    n = mod.meta.nvar
    ncc = mod.meta.ncc

    Il = findall(z -> z <= prec, abs.(x - mod.meta.lvar))
    Iu = findall(z -> z <= prec, abs.(x - mod.meta.uvar))
    jl = zeros(n)
    jl[Il] .= 1.0
    Jl = diagm(0 => jl)
    ju = zeros(n)
    ju[Iu] .= 1.0
    Ju = diagm(0 => ju)
    IG = Int64[]
    IH = Int64[]
    Ig = Int64[]
    Ih = Int64[]

    if mod.meta.ncon + ncc == 0

        A = []

    else

        if mod.meta.ncon > 0
            c = cons_nl(mod, x)
            J = jac_nl(mod, x)
        else
            c = Float64[]
            J = sparse(zeros(0, 2))
        end

        Ig = findall(z -> z <= prec, abs.(c - mod.meta.lcon))
        Ih = findall(z -> z <= prec, abs.(c - mod.meta.ucon))

        Jg, Jh = zeros(mod.meta.ncon, n), zeros(mod.meta.ncon, n)

        Jg[Ig, 1:n] = J[Ig, 1:n]
        Jh[Ih, 1:n] = J[Ih, 1:n]

        if ncc > 0

            IG = findall(z -> z <= prec, abs.(consG(mod, x) - mod.meta.lccG))
            IH = findall(z -> z <= prec, abs.(consH(mod, x) - mod.meta.lccH))

            JG, JH = zeros(ncc, n), zeros(ncc, n)
            JG[IG, 1:n] = jacG(mod, x)[IG, 1:n]
            JH[IH, 1:n] = jacH(mod, x)[IH, 1:n]

            A = [Jl; Ju; -Jg; Jh; -JG; -JH]'
        else
            A = [Jl; Ju; -Jg; Jh]'
        end

    end

    return A, Il, Iu, Ig, Ih, IG, IH
end

"""
  JGv = jGprod!(nlp, x, v, Jv)
Evaluate ``∇G(x)v``, the Jacobian-vector product at `x` in place.
"""
function jGprod!(mod::MPCCNLPs, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    increment!(mod, :neval_jGprod)
    jprod!(mod.G, x, v, Jv)
end

"""
  JHv = jHprod!(nlp, x, v, Jv)
Evaluate ``∇H(x)v``, the Jacobian-vector product at `x` in place.
"""
function jHprod!(mod::MPCCNLPs, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    increment!(mod, :neval_jHprod)
    jprod!(mod.H, x, v, Jv)
end

"""
    Jtv = jtprodG(nlp, x, v, Jtv)
Evaluate ``∇G(x)^Tv``, the transposed-Jacobian-vector product at `x`
"""
function jGtprod!(mod::MPCCNLPs, x::Vector, v::Vector, Jtv::Vector)
    increment!(mod, :neval_jGtprod)
    jtprod!(mod.G, x, v, Jtv)
end

"""
    Jtv = jtprodH(nlp, x, v, Jtv)
Evaluate ``∇H(x)^Tv``, the transposed-Jacobian-vector product at `x`
"""
function jHtprod!(mod::MPCCNLPs, x::Vector, v::Vector, Jtv::Vector)
    increment!(mod, :neval_jHtprod)
    jtprod!(mod.H, x, v, Jtv)
end

function hGprod!(
    mod::MPCCNLPs,
    x::AbstractVector,
    y::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector,
)
    increment!(mod, :neval_hGprod)
    hprod!(mod.G, x, y, v, Hv, obj_weight = 0)
end

function hHprod!(
    mod::MPCCNLPs,
    x::AbstractVector,
    y::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector,
)
    increment!(mod, :neval_hHprod)
    hprod!(mod.H, x, y, v, Hv, obj_weight = 0)
end

function hessG_structure!(
    mod::MPCCNLPs,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return hess_structure!(mod.G, rows, cols)
end

function hessG_coord!(
    mod::MPCCNLPs,
    x::AbstractVector,
    y::AbstractVector,
    vals::AbstractVector,
)
    increment!(mod, :neval_hess)
    return hess_coord!(mod.G, x, y, vals)
end

function hessH_structure!(
    mod::MPCCNLPs,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return hess_structure!(mod.H, rows, cols)
end

function hessH_coord!(
    mod::MPCCNLPs,
    x::AbstractVector,
    y::AbstractVector,
    vals::AbstractVector,
)
    increment!(mod, :neval_hess)
    return hess_coord!(mod.H, x, y, vals)
end

function hprod!(
    mod::MPCCNLPs,
    x::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight::Real = one(eltype(x)),
)
    increment!(mod, :neval_hprod)
    hprod!(mod.mp, x, v, Hv, obj_weight = obj_weight)
end

function hprod!(
    mod::MPCCNLPs,
    x::AbstractVector,
    y::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight::Real = one(eltype(x)),
)
    increment!(mod, :neval_hprod)
    ncon, ncc = mod.meta.ncon, mod.cc_meta.ncc
    Hv .=
        hprod(mod.mp, x, y[1:ncon], v, obj_weight = obj_weight) -
        hprod(mod.G, x, y[ncon+1:ncon+ncc], v, obj_weight = 0) -
        hprod(mod.H, x, y[ncon+ncc+1:ncon+2*ncc], v, obj_weight = 0)
    return Hv
end

function hess_structure!(
    mod::MPCCNLPs,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    n = mod.meta.nvar
    I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
    rows[1:mod.meta.nnzh] .= getindex.(I, 1)
    cols[1:mod.meta.nnzh] .= getindex.(I, 2)
    return rows, cols
end

function hess_coord!(
    mod::MPCCNLPs,
    x::AbstractVector,
    vals::AbstractVector;
    obj_weight::Real = one(eltype(x)),
)
    increment!(mod, :neval_hess)
    hess_coord!(mod.mp, x, vals, obj_weight = obj_weight)
end

function hess_coord!(
    mod::MPCCNLPs,
    x::AbstractVector,
    y::AbstractVector,
    vals::AbstractVector;
    obj_weight::Real = one(eltype(x)),
)
    increment!(mod, :neval_hess)
    ncon, ncc = mod.meta.ncon, mod.cc_meta.ncc
    yG, yH = y[ncon+1:ncon+ncc], y[ncon+ncc+1:ncon+2*ncc]
    Hx =
        hess(mod.mp, x, y[1:ncon], obj_weight = obj_weight) -
        hess(mod.G, x, yG, obj_weight = 0) - hess(mod.H, x, yH, obj_weight = 0)
    k = 1
    for j = 1:mod.meta.nvar
        for i = j:mod.meta.nvar
            vals[k] = Hx[i, j]
            k += 1
        end
    end
    return vals
end
