
@concrete struct IterableDiscretePNSolution <: AbstractDiscretePNSolution
    system
    b_assembler
    current_solution

    reverse
    initial_solution
end

function initialize_or_fillzero!(it::IterableDiscretePNSolution, ::Nothing)
    arch = architecture(it.system.problem)
    T = base_type(arch)
    fill!(it.current_solution, zero(T))
end

function initialize_or_fillzero!(it::IterableDiscretePNSolution, initial_solution)
    copy!(it.current_solution, initial_solution)
end

function IterableDiscretePNSolution(system::AbstractDiscretePNSystem, b::AbstractDiscretePNVector; initial_solution=nothing)
    if system.adjoint != _is_adjoint_vector(b)
        @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(b))}"
    end
    current_solution = allocate_solution_vector(system)
    b_assembler = initialize_assembly(b)
    return IterableDiscretePNSolution(system, b_assembler, current_solution, false, initial_solution)
end

function _is_adjoint_solution(ψ::IterableDiscretePNSolution)
    return ψ.system.adjoint
end

Base.length(it::IterableDiscretePNSolution) = length(energy_model(it.system.problem.model))

function Base.iterate(it::IterableDiscretePNSolution)
    initialize_or_fillzero!(it, it.initial_solution)
    ϵs = energy_model(it.system.problem.model)
    if !it.reverse
        idx = first_index(ϵs, _is_adjoint_solution(it))
    else
        @info "iterating in reverse"
        idx = last_index(ϵs, _is_adjoint_solution(it))
    end
    return idx => it.current_solution, idx
end

function _iterate_reverse(it::IterableDiscretePNSolution, idx::ϵidx)
    ## THIS SHOULD BE TESTED IN ONLYENERGYMODEL! (should basically give the same result in normal and reverse)
    ϵs = energy_model(it.system.problem.model)
    T = base_type(architecture(it.system.problem))
    Δϵ = T(step(ϵs))
    idx_prev = previous(idx)
    if isnothing(idx_prev) return nothing end
    if _is_adjoint_solution(it)
        # update the system to idx.i - 1/2 from idx.i + 1/2
        step_adjoint!(it.current_solution, it.system, it.b_assembler, idx_prev, Δϵ)
    else
        # update the system to idx.i from idx.i - 1
        step_nonadjoint!(it.current_solution, it.system, it.b_assembler, idx_prev, Δϵ)
    end
    return idx_prev => it.current_solution, idx_prev
end

function Base.iterate(it::IterableDiscretePNSolution, idx::ϵidx)
    ϵs = energy_model(it.system.problem.model)
    T = base_type(architecture(it.system.problem))
    Δϵ = T(step(ϵs))
    if it.reverse return _iterate_reverse(it, idx) end
    idx_next = next(idx)
    if isnothing(idx_next) return nothing end
    if _is_adjoint_solution(it)
        # update the system from idx.i - 1/2 -> idx.i + 1/2
        step_adjoint!(it.current_solution, it.system, it.b_assembler, idx, Δϵ)
    else
        # update the system from idx.i -> idx.i - 1
        step_nonadjoint!(it.current_solution, it.system, it.b_assembler, idx, Δϵ)
    end
    return idx_next => it.current_solution, idx_next
end

@concrete struct CachedDiscretePNSolution <: AbstractDiscretePNSolution
    it
    # reverse
    # adjoint
    solution_cache
end

function saveall(it)
    # for not the cached solution remains in the "architecture" device.. 
    CachedDiscretePNSolution(it, Dict(idx => copy(sol) for (idx, sol) in it))
end

function _is_adjoint_solution(it::CachedDiscretePNSolution)
    return _is_adjoint_solution(it.it)
end

function Base.length(it::CachedDiscretePNSolution)
    return length(it.solution_cache)
end

function Base.iterate(it::CachedDiscretePNSolution)
    (idx, _), _ = iterate(it.it)
    return idx => it.solution_cache[idx], idx
end

function Base.iterate(it::CachedDiscretePNSolution, idx::ϵidx)
    idx_next = next(idx)
    if isnothing(idx_next) return nothing end
    return idx_next => it.solution_cache[idx_next], idx_next
end

function Base.getindex(it::CachedDiscretePNSolution, idx::ϵidx)
    return it.solution_cache[idx]
end

@concrete struct DiscreteIntervalPNIterator{IT}
    it::IT
    cached_solution
end

_is_adjoint_solution(it::DiscreteIntervalPNIterator) = _is_adjoint_solution(it.it)
Base.length(it::DiscreteIntervalPNIterator) = length(it.it) - 1

function taketwo(it::IterableDiscretePNSolution)
    DiscreteIntervalPNIterator(it, allocate_solution_vector(it.system))
end

function taketwo(it::CachedDiscretePNSolution)
    DiscreteIntervalPNIterator(it, nothing)
end

function Base.iterate(it::DiscreteIntervalPNIterator{<:IterableDiscretePNSolution})
    ret1 = iterate(it.it)
    if isnothing(ret1) return nothing end
    (idx1, sol1), idx = ret1
    copy!(it.cached_solution, sol1)

    ret2 = iterate(it.it, idx)
    if isnothing(ret2) return nothing end
    (idx2, sol2), idx = ret2
    if idx1 < idx2
        return (idx1 => it.cached_solution, idx2 => sol2), idx
    elseif idx2 < idx1
        return (idx2 => sol2, idx1 => it.cached_solution), idx
    else
        throw(ArgumentError("idx1 == idx2"))
    end
end

function Base.iterate(it::DiscreteIntervalPNIterator{<:IterableDiscretePNSolution}, idx::ϵidx)
    idx1 = idx
    copy!(it.cached_solution, it.it.current_solution)

    ret = iterate(it.it, idx)
    if isnothing(ret) return nothing end
    (idx2, sol2), idx = ret
    
    if idx1 < idx2
        return (idx1 => it.cached_solution, idx2 => sol2), idx
    elseif idx2 < idx1
        return (idx2 => sol2, idx1 => it.cached_solution), idx
    else
        throw(ArgumentError("idx1 == idx2"))
    end
end

# cached iterator (we do not use the internal state (=nothing), but instead just return from the cache)
function Base.iterate(it::DiscreteIntervalPNIterator{<:CachedDiscretePNSolution})
    ret1 = iterate(it.it)
    if isnothing(ret1) return nothing end
    (idx1, sol1), idx = ret1
    ret2 = iterate(it.it, idx)
    if isnothing(ret2) return nothing end
    (idx2, sol2), idx = ret2
    if idx1 < idx2
        return (idx1 => sol1, idx2 => sol2), idx
    elseif idx2 < idx1
        return (idx2 => sol2, idx1 => sol1), idx
    else
        throw(ArgumentError("idx1 == idx2"))
    end
end

function Base.iterate(it::DiscreteIntervalPNIterator{<:CachedDiscretePNSolution}, idx::ϵidx)
    idx1 = idx
    ret = iterate(it.it, idx)
    if isnothing(ret) return nothing end
    (idx2, sol2), idx = ret
    
    if idx1 < idx2
        return (idx1 => it.it[idx1], idx2 => sol2), idx
    elseif idx2 < idx1
        return (idx2 => sol2, idx1 => it.it[idx1]), idx
    else
        throw(ArgumentError("idx1 == idx2"))
    end
end
