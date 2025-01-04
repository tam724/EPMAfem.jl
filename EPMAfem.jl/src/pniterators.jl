
@concrete struct DiscretePNIterator <: AbstractDiscretePNSolution
    system
    rhs
    state

    reverse
    initial_state
end

function initialize_or_fillzero!(it::DiscretePNIterator, ::Nothing)
    arch = architecture(it.system.problem)
    T = base_type(arch)
    fill!(it.state, zero(T))
end

function initialize_or_fillzero!(it::DiscretePNIterator, initial_state)
    copy!(it.state, initial_state)
end

function DiscretePNIterator(system::AbstractDiscretePNSystem, rhs::AbstractDiscretePNVector)
    if system.adjoint != _is_adjoint_vector(rhs)
        @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs))}"
    end
    state = allocate_solution_vector(system)
    return DiscretePNIterator(system, rhs, state, false, nothing)
end

function _is_adjoint_solution(ψ::DiscretePNIterator)
    return ψ.system.adjoint
end

Base.length(it::DiscretePNIterator) = length(energy_model(it.system.problem.model))

function Base.iterate(it::DiscretePNIterator)
    initialize_or_fillzero!(it, it.initial_state)
    ϵs = energy_model(it.system.problem.model)
    if !it.reverse
        idx = first_index(ϵs, _is_adjoint_solution(it))
    else
        @info "iterating in reverse"
        idx = last_index(ϵs, _is_adjoint_solution(it))
    end
    return idx => it.state, idx
end

function _iterate_reverse(it::DiscretePNIterator, idx::ϵidx)
    ## THIS SHOULD BE TESTED IN ONLYENERGYMODEL! (should basically give the same result in normal and reverse)
    ϵs = energy_model(it.system.problem.model)
    Δϵ = step(ϵs)
    idx_prev = previous(idx)
    if isnothing(idx_prev) return nothing end
    if _is_adjoint_solution(it)
        # update the system to idx.i - 1/2 from idx.i + 1/2
        step_adjoint!(it.state, it.system, it.rhs, idx_prev, Δϵ)
    else
        # update the system to idx.i from idx.i - 1
        step_nonadjoint!(it.state, it.system, it.rhs, idx_prev, Δϵ)
    end
    return idx_prev => it.state, idx_prev
end

function Base.iterate(it::DiscretePNIterator, idx::ϵidx)
    ϵs = energy_model(it.system.problem.model)
    Δϵ = step(ϵs)
    if it.reverse return _iterate_reverse(it, idx) end
    idx_next = next(idx)
    if isnothing(idx_next) return nothing end
    if _is_adjoint_solution(it)
        # update the system from idx.i - 1/2 -> idx.i + 1/2
        step_adjoint!(it.state, it.system, it.rhs, idx, Δϵ)
    else
        # update the system from idx.i -> idx.i - 1
        step_nonadjoint!(it.state, it.system, it.rhs, idx, Δϵ)
    end
    return idx_next => it.state, idx_next
end

@concrete struct CachedDiscreteSolution <: AbstractDiscretePNSolution
    it
    # reverse
    # adjoint
    cache
end

function saveall(it)
    CachedDiscreteSolution(it, Dict(idx => copy(sol) for (idx, sol) in it))
end

function _is_adjoint_solution(it::CachedDiscreteSolution)
    return _is_adjoint_solution(it.it)
end

function Base.length(it::CachedDiscreteSolution)
    return length(it.cache)
end

function Base.iterate(it::CachedDiscreteSolution)
    (idx, _), _ = iterate(it.it)
    return idx => it.cache[idx], idx
end

function Base.iterate(it::CachedDiscreteSolution, idx::ϵidx)
    idx_next = next(idx)
    if isnothing(idx_next) return nothing end
    return idx_next => it.cache[idx_next], idx_next
end

function Base.getindex(it::CachedDiscreteSolution, idx::ϵidx)
    return it.cache[idx]
end