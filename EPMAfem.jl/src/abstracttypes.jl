abstract type AbstractPNEquations end

# basic grid definitions, number of basis functions, etc.. 
abstract type AbstractPNModel end

# holds the basis matrices for the PNProblem
abstract type AbstractDiscretePNProblem end

# a "solvable" PNProblem, it holds the problem and defines the solver
abstract type AbstractDiscretePNSystem end

# implicit midpoint systems
abstract type AbstractDiscretePNSystemIM <: AbstractDiscretePNSystem end

abstract type AbstractDiscretePNVector end
# there is no structural difference between a vector and its adjoint (but during construction we can specify if we want to use it with with the adjoint or nonadjoint system)
is_adjoint(b::AbstractDiscretePNVector) = _is_adjoint_vector(b)

# this is "almost" a AbstractDiscretePNVector. But we have to differentiate the two, because the adjoint-solution has the shifted energy grid. (typically iterators)
abstract type AbstractDiscretePNSolution end
is_adjoint(ψ::AbstractDiscretePNSolution) = _is_adjoint_solution(ψ)

## some syntactic sugar
Base.:*(A::AbstractDiscretePNSystem, g::AbstractDiscretePNVector) = DiscretePNIterator(A, g)
Base.:*(g::AbstractDiscretePNVector, A::AbstractDiscretePNSystem) = DiscretePNIterator(adjoint(A), g)
Base.:*(A::AbstractDiscretePNSystem, g::AbstractArray{<:AbstractDiscretePNVector}) = [DiscretePNIterator(A, g_) for g_ in g]
Base.:*(g::AbstractArray{<:AbstractDiscretePNVector}, A::AbstractDiscretePNSystem) = [DiscretePNIterator(adjoint(A), g_) for g_ in g]

function solve_and_integrate(b::Union{AbstractDiscretePNVector, AbstractArray{<:AbstractDiscretePNVector}}, it::AbstractDiscretePNSolution)
    cache = initialize_integration(b)
    for (idx, ψ) in it
        if is_first(idx) continue end # (where ψ is initialized to 0 anyways..)
        integrate_at!(cache, idx, b, ψ)
    end
    return finalize_integration(cache, b)
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