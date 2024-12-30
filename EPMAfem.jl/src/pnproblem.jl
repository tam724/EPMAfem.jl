@concrete struct DiscretePNProblem <: AbstractDiscretePNProblem
    model
    arch

    # energy (these will always live on the cpu)
    s
    τ
    σ

    # space (might be moved to gpu)
    ρp
    ρp_tens
    ρm
    ρm_tens

    ∂p
    ∇pm

    # direction (might be moved to gpu)
    Ip
    Im
    kp
    km
    absΩp
    Ωpm
end

architecture(pnproblem::DiscretePNProblem) = pnproblem.arch
n_basis(pnproblem::DiscretePNProblem) = n_basis(pnproblem.model)
n_sums(pnproblem::DiscretePNProblem) = (nd = length(pnproblem.∇pm), ne = size(pnproblem.s, 1), nσ = size(pnproblem.σ, 2))

@concrete struct NonAdjointIterator
    system
    rhs

    reverse
    state
end

function iterator(system::AbstractDiscretePNSystem, rhs::AbstractDiscretePNVector{false})
    return NonAdjointIterator(system, rhs, false, nothing)
end

function reverse_iterator(system::AbstractDiscretePNSystem, rhs::AbstractDiscretePNVector{false}, state)
    return NonAdjointIterator(system, rhs, true, state)
end

function Base.iterate(it::NonAdjointIterator)
    if it.reverse
        initialize_from_state!(it.solver, it.state)
        ϵs = energy_model(it.system.problem.model)
        ϵ = ϵs[1]
        return (ϵ, 1), 1
    end

    initialize!(it.system)
    ϵs = energy_model(it.system.problem.model)
    ϵ = ϵs[end]
    return (ϵ, length(ϵs)), length(ϵs)
end

function Base.iterate(it::NonAdjointIterator, i)
    if it.reverse
        ϵs = energy_model(it.system.problem.model)
        if i >= length(ϵs)
            return nothing
        else
            # here we update the solver state from i to i+1! 
            i = i+1
            ϵi, ϵip1 = ϵs[i-1], ϵs[i]
            Δϵ = ϵip1-ϵi
            step_nonadjoint!(it.system, it.rhs, i, -Δϵ)
            return (ϵip1, i), i
        end
    end

    if i <= 1
        return nothing
    else
        ϵs = energy_model(it.system.problem.model)
        # here we update the solver state from i+1 to i! NOTE: NonAdjoint means from higher to lower energies/times
        ϵi, ϵip1 = ϵs[i-1], ϵs[i]
        Δϵ = ϵip1-ϵi
        step_nonadjoint!(it.system, it.rhs, i, Δϵ)
        return (ϵi, i-1), i-1
    end
end

@concrete struct AdjointIterator
    system
    rhs
end

function iterator(system::AbstractDiscretePNSystem, rhs::AbstractDiscretePNVector{true})
    return AdjointIterator(system, rhs)
end

function Base.iterate(it::AdjointIterator)
    initialize!(it.system)
    ϵs = energy_model(it.system.problem.model)
    ϵ = ϵs[1]
    return (ϵ, 1), 1
end

function Base.iterate(it::AdjointIterator, i)
    ϵs = energy_model(it.system.problem.model)
    if i >= length(ϵs)
        return nothing
    else
        # here we update the solver state from i to i+1! NOTE: Adjoint means from lower to higher energies/times
        ϵi, ϵip1 = ϵs[i], ϵs[i+1]
        Δϵ = ϵip1-ϵi
        step_adjoint!(it.system, it.rhs, i, Δϵ)
        return (ϵip1, i+1), i+1
    end
end

@concrete struct SaveAll
    it
    cache
end

function saveall(it)
    cache = Dict{Int, typeof(current_solution(it.system))}()
    for (ϵ, i) in it
        cache[i] = current_solution(it.system) |> copy
    end
    return SaveAll(it, cache)
end

function Base.getindex(it::SaveAll, i)
    return it.cache[i]
end
