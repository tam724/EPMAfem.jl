abstract type AbstractPNEquations end

# a "solvable" PNProblem, it holds the problem and defines the solver
abstract type AbstractDiscretePNSystem end

abstract type AbstractDiscretePNVector end
# there is no structural difference between a vector and its adjoint (but during construction we can specify if we want to use it with with the adjoint or nonadjoint system)
_is_adjoint_vector(b::AbstractArray{<:AbstractDiscretePNVector}) = all(bi -> _is_adjoint_vector(bi) == _is_adjoint_vector(first(b)), b) ? _is_adjoint_vector(first(b)) : throw(ArgumentError("Vectors have different adjoint properties"))

@concrete terse struct PNVectorIntegrator{V}
    b::V
    cache
end
_is_adjoint_vector((; b)::PNVectorIntegrator) = _is_adjoint_vector(b)

@concrete terse struct PNVectorAssembler{V}
    b::V
    cache
end
_is_adjoint_vector((; b)::PNVectorAssembler) = _is_adjoint_vector(b)


# this is "almost" a AbstractDiscretePNVector. But we have to differentiate the two, because the adjoint-solution has the shifted energy grid. (typically iterators)
abstract type AbstractDiscretePNSolution end

## some syntactic sugar
Base.:*(A::AbstractDiscretePNSystem, g::AbstractDiscretePNVector) = IterableDiscretePNSolution(A, g)
Base.:*(g::AbstractDiscretePNVector, A::AbstractDiscretePNSystem) = IterableDiscretePNSolution(adjoint(A), g)
Base.:*(A::AbstractDiscretePNSystem, g::AbstractArray{<:AbstractDiscretePNVector}) = [IterableDiscretePNSolution(A, g_) for g_ in g]
Base.:*(g::AbstractArray{<:AbstractDiscretePNVector}, A::AbstractDiscretePNSystem) = [IterableDiscretePNSolution(adjoint(A), g_) for g_ in g]

function solve_and_integrate(b::Union{AbstractDiscretePNVector, AbstractArray{<:AbstractDiscretePNVector}}, it::AbstractDiscretePNSolution)
    b_integrator! = initialize_integration(b)
    for (idx, ψ) in it
        if is_first(idx) continue end # (where ψ is initialized to 0 anyways..)
        b_integrator!(idx, ψ)
    end
    return finalize_integration(b_integrator!)
end

function Base.:*(h::Union{AbstractDiscretePNVector, AbstractArray{<:AbstractDiscretePNVector}}, ψ::AbstractDiscretePNSolution)
    if _is_adjoint_vector(h) == _is_adjoint_solution(ψ) @warn "Vector and solution are not compatible" end
    return solve_and_integrate(h, ψ)
end

function Base.:*(h::Union{AbstractDiscretePNVector, AbstractArray{<:AbstractDiscretePNVector}}, ψ::AbstractArray{<:AbstractDiscretePNSolution})
    full_size = (size(h)..., size(ψ)...)
    result = zeros(full_size)
    for i in eachindex(IndexCartesian(), ψ)
        result[axes(h)..., i] .= h * ψ[i]
    end
    return result
end

function Base.:*(ψ::AbstractDiscretePNSolution, g::Union{AbstractDiscretePNVector, AbstractArray{<:AbstractDiscretePNVector}})
    if _is_adjoint_vector(g) == _is_adjoint_solution(ψ) @warn "Vector and solution are not compatible" end
    return solve_and_integrate(g, ψ)
end

function Base.:*(ψ::AbstractArray{<:AbstractDiscretePNSolution}, g::Union{AbstractDiscretePNVector, AbstractArray{<:AbstractDiscretePNVector}})
    full_size = (size(ψ)..., size(g)...)
    result = zeros(full_size)
    for i in eachindex(IndexCartesian(), ψ)
        result[i, axes(g)...] .= ψ[i] * g
    end
    return result
end

function Base.:*(h::AbstractArray{<:AbstractDiscretePNVector}, A::AbstractDiscretePNSystem, g::AbstractArray{<:AbstractDiscretePNVector})
    full_size = (size(h)..., size(g)...)
    result = zeros(full_size)
    if length(g) < length(h)
        return h * (A * g)
    else
        return (h * A) * g
    end
    return result
end

Base.:*(h::AbstractArray{<:AbstractDiscretePNVector}, A::AbstractDiscretePNSystem, g::AbstractDiscretePNVector) = h * (A * g)
Base.:*(h::AbstractDiscretePNVector, A::AbstractDiscretePNSystem, g::AbstractArray{<:AbstractDiscretePNVector}) = (h * A) * g