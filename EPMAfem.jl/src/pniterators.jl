
@concrete struct DiscretePNIterator <: AbstractDiscretePNSolution
    system
    rhs

    reverse
    initial_state
end

function DiscretePNIterator(system::AbstractDiscretePNSystem, rhs::AbstractDiscretePNVector)
    if system.adjoint != _is_adjoint_vector(rhs)
        @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs))}"
    end
    return DiscretePNIterator(system, rhs, false, nothing)
end

function _is_adjoint_solution(ψ::DiscretePNIterator)
    return ψ.system.adjoint
end

function _iterate_nonadjoint(it::DiscretePNIterator)
    if it.reverse
        initialize_or_fillzero!(it.system, it.initial_state)
        ϵs = energy_model(it.system.problem.model)
        idx = last_index_nonadjoint(ϵs)
        return idx, 1
    end

    initialize_or_fillzero!(it.system, it.initial_state)
    ϵs = energy_model(it.system.problem.model)
    idx = first_index_nonadjoint(ϵs)
    return idx, idx
end

function _iterate_nonadjoint(it::DiscretePNIterator, idx::ϵidx)
    ϵs = energy_model(it.system.problem.model)
    Δϵ = step(ϵs)

    if it.reverse
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

    idx_minus1 = minus1(idx)
    if isnothing(idx_minus1) return nothing end
    #### here we update the solver state from i+1 to i! NOTE: NonAdjoint means from higher to lower energies/times
    #### ϵi, ϵip1 = ϵs[i-1], ϵs[i]
    # update the system state from i to i-1
    step_nonadjoint!(it.system, it.rhs, idx, Δϵ)
    return idx_minus1, idx_minus1
end

Base.iterate(it::DiscretePNIterator) = if _is_adjoint_solution(it) _iterate_adjoint(it) else _iterate_nonadjoint(it) end
Base.iterate(it::DiscretePNIterator, idx::ϵidx) = if _is_adjoint_solution(it) _iterate_adjoint(it, idx) else _iterate_nonadjoint(it, idx) end

# @concrete struct AdjointIterator
#     system
#     rhs

#     state
# end

# function iterator(system::AbstractDiscretePNSystem, rhs::AbstractDiscretePNVector{true})
#     return AdjointIterator(system, rhs, nothing)
# end

function _iterate_adjoint(it::DiscretePNIterator)
    if it.reverse throw(ArgumentError("Reverse not supported for adjoint iterator")) end
    initialize_or_fillzero!(it.system, it.initial_state)
    ϵs = energy_model(it.system.problem.model)
    idx = first_index_adjoint(ϵs)
    return idx, idx
end

function _iterate_adjoint(it::DiscretePNIterator, idx::ϵidx)
    if it.reverse throw(ArgumentError("Reverse not supported for adjoint iterator")) end
    ϵs = energy_model(it.system.problem.model)
    Δϵ = step(ϵs)
    idx_plus1 = plus1(idx)
    if isnothing(idx_plus1) return nothing end

    # here we update the solver state from i to i+1! NOTE: Adjoint means from lower to higher energies/times
    # ϵi, ϵip1 = ϵs[i]-Δϵ/2, ϵs[i+1]-Δϵ/2
    # Δϵ = ϵip1-ϵi
    # update the system state from i to i-1
    step_adjoint!(it.system, it.rhs, idx, Δϵ)
    return idx_plus1, idx_plus1
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
