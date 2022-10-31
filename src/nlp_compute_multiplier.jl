"""
_compute_mutliplier: Additional function to estimate Lagrange multiplier of the problems
    (guarantee if LICQ holds)
"""
function _compute_mutliplier(
    pb::AbstractNLPModel,
    x::T,
    gx::T,
    cx::T,
    Jx::AbstractMatrix;
    active_prec_c::Float64 = 1e-6,
    active_prec_b::Float64 = 1e-6,
) where {T}

    n = length(x)
    nc = cx == nothing ? 0 : length(cx)

    #active res_bounds
    Ib = findall(
        x -> (norm(x) <= active_prec_b),
        min(abs.(x - pb.meta.lvar), abs.(x - pb.meta.uvar)),
    )
    if nc != 0
        #active constraints
        Ic = findall(
            x -> (norm(x) <= active_prec_c),
            min(abs.(cx - pb.meta.ucon), abs.(cx - pb.meta.lcon)),
        )

        Jc = hcat(Matrix(1.0I, n, n)[:, Ib], Jx'[:, Ic])
    else
        Ic = []
        Jc = hcat(Matrix(1.0I, n, n)[:, Ib])
    end

    mu, lambda = zeros(n), zeros(nc)
    if (Ib != []) || (Ic != [])
        l = Jc \ (-gx)
        mu[Ib], lambda[Ic] = l[1:length(Ib)], l[length(Ib)+1:length(l)]
    end

    return mu, lambda
end

function _compute_mutliplier(
    pb::AbstractMPCCModel,
    x::AbstractVector,
    gx::AbstractVector,
    cx::AbstractVector,
    Jx::AbstractMatrix,
    Gx::AbstractVector,
    JGx::AbstractMatrix,
    Hx::AbstractVector,
    JHx::AbstractMatrix;
    active_prec_c::Float64 = 1e-6,
    active_prec_cc::Float64 = 1e-3,
    active_prec_b::Float64 = 1e-6,
)

    n, ncc = length(x), pb.cc_meta.ncc
    nc = cx == nothing ? 0 : length(cx)

    #active res_bounds
    Ib = findall(
        x -> (norm(x) <= active_prec_b),
        min(abs.(x - pb.meta.lvar), abs.(x - pb.meta.uvar)),
    )
    if nc != 0
        #active constraints
        Ic = findall(
            x -> (norm(x) <= active_prec_c),
            min(abs.(cx - pb.meta.ucon), abs.(cx - pb.meta.lcon)),
        )

        Jc = hcat(Matrix(1.0I, n, n)[:, Ib], Jx'[:, Ic])
    else
        Ic = []
        Jc = hcat(Matrix(1.0I, n, n)[:, Ib])
    end

    if ncc != 0
        #active constraints
        IG = findall(x -> (norm(x) <= active_prec_cc), abs.(Gx))
        IH = findall(x -> (norm(x) <= active_prec_cc), abs.(Hx))
        Jc =
            (length(Ic) + length(Ib)) != 0 ? hcat(Jc, -JGx'[:, IG], -JHx'[:, IH]) :
            hcat(-JGx'[:, IG], -JHx'[:, IH])
    else
        IG, IH = [], []
    end

    mu, lambda, etaG, etaH = zeros(n), zeros(nc), zeros(ncc), zeros(ncc)
    if (Ib != []) || (Ic != []) || (IG != []) || (IH != [])
        l = Jc \ (-gx)
        mu[Ib], lambda[Ic], etaG[IG], etaH[IH] = l[1:length(Ib)],
        l[length(Ib)+1:length(Ib)+length(Ic)],
        l[length(Ib)+length(Ic)+1:length(Ib)+length(Ic)+length(IG)],
        l[length(Ib)+length(Ic)+length(IG)+1:length(Ib)+length(Ic)+length(IG)+length(IH)]
    end

    return mu, lambda, etaG, etaH
end
