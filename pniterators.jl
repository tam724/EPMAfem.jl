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
    ϵ = pn_solv.ϵ_interval[2]
    return ϵ, (ϵ, 1)
end

function Base.iterate(pn_it::ForwardIterator, (ϵ, i))
    if isapprox(ϵ - pn_it.solver.ϵ_interval[1], 0.0, atol=1e-8)
        return nothing
    else
        pn_solv = pn_it.solver
        Δϵ = energy_step(pn_solv)
        # here we update the solver state from i+1 to i !!! NOTE: forward means from higher to lower energies
        ϵip1 = ϵ
        ϵi = ϵ - Δϵ
        if (ϵi < pn_solv.ϵ_interval[1]) ϵi = pn_solv.ϵ_interval[1] end
        # update_rhs_forward!(pn_solv, ϵi,  ϵip1, pn_it.g_idx)
        # update_mat_forward!(pn_solv, ϵi, ϵip1)
        step_forward!(pn_solv, ϵi, ϵip1, pn_it.g_idx)
        # Krylov.solve!(pn_solv.lin_solver, pn_solv.A, pn_solv.b, rtol=1e-8, atol=1e-8)
        # new_state = (solution(pn_solv), ϵi, k-1)
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
    ϵ = pn_solv.ϵ_interval[1]
    return ϵ, (ϵ, 1)
end

function Base.iterate(pn_it::BackwardIterator, (ϵ, i))
    if isapprox(ϵ - pn_it.solver.ϵ_interval[2], 0.0, atol=1e-8)
        return nothing
    else
        pn_solv = pn_it.solver
        Δϵ = energy_step(pn_solv)
        ϵi = ϵ
        ϵip1 = ϵ + Δϵ
        if (ϵip1 > pn_solv.ϵ_interval[2]) ϵip1 = pn_solv.ϵ_interval[2] end
        step_backward!(pn_solv, ϵi, ϵip1, pn_it.μ_idx)
        return ϵip1, (ϵip1, i+1)
    end
end

struct AugmentedForwardIterator{S}
    solver::S
    g_factors::Array{3, Float64}
end

