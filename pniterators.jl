struct ForwardIterator{S <: PNSolver}
    solver::S
    g_idx::NTuple{3, Int64}
end

function forward(pn_solv, g_idx)
    return ForwardIterator(pn_solv, g_idx)
end

function Base.iterate(pn_it::ForwardIterator)
    pn_solv = pn_it.solver
    initialize!(pn_solv)
    pn_equ = get_pn_equ(pn_solv)
    ϵ = _energy_interval(pn_equ)[2]
    return ϵ, (ϵ, 1)
end

function Base.iterate(pn_it::ForwardIterator, (ϵ, i))
    pn_equ = get_pn_equ(pn_it.solver)
    ϵ_cutoff = _energy_interval(pn_equ)[1]
    if isapprox(ϵ - ϵ_cutoff, 0.0, atol=1e-8)
        return nothing
    else
        pn_solv = pn_it.solver
        Δϵ = energy_step(pn_solv)
        # here we update the solver state from i+1 to i !!! NOTE: forward means from higher to lower energies
        ϵip1 = ϵ
        ϵi = ϵ - Δϵ
        if (ϵi < ϵ_cutoff) ϵi = ϵ_cutoff end
        step_forward!(pn_solv, ϵi, ϵip1, pn_it.g_idx)
        return ϵi, (ϵi, i+1)
    end
end

struct BackwardIterator{S <: PNSolver}
    solver::S
    μ_idx::NTuple{3, Int64}
end

function backward(pn_solv, μ_idx)
    return BackwardIterator(pn_solv, μ_idx)
end

function Base.iterate(pn_it::BackwardIterator)
    pn_solv = pn_it.solver
    initialize!(pn_solv)
    pn_equ = get_pn_equ(pn_solv)
    ϵ = _energy_interval(pn_equ)[1]
    return ϵ, (ϵ, 1)
end

function Base.iterate(pn_it::BackwardIterator, (ϵ, i))
    pn_equ = get_pn_equ(pn_it.solver)
    ϵ_initial = _energy_interval(pn_equ)[2]
    if isapprox(ϵ - ϵ_initial, 0.0, atol=1e-8)
        return nothing
    else
        pn_solv = pn_it.solver
        Δϵ = energy_step(pn_solv)
        ϵi = ϵ
        ϵip1 = ϵ + Δϵ
        if (ϵip1 > ϵ_initial) ϵip1 = ϵ_initial end
        step_backward!(pn_solv, ϵi, ϵip1, pn_it.μ_idx)
        return ϵip1, (ϵip1, i+1)
    end
end

struct AugmentedForwardIterator{S}
    solver::S
    g_factors::Array{3, Float64}
end

