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
function Base.:*(A::AbstractDiscretePNSystem, g::AbstractDiscretePNVector)
    return DiscretePNIterator(A, g)
end

function Base.:*(h::AbstractDiscretePNVector, ψ::AbstractDiscretePNSolution)
    if _is_adjoint_vector(h) == _is_adjoint_solution(ψ) @warn "Adjoint vector and solution are not compatible" end
    if _is_adjoint_solution(ψ)
        return solve_and_integrate_adjoint(h, ψ)
    else
        return solve_and_integrate_nonadjoint(h, ψ)
    end
end

function Base.:*(h::AbstractArray{<:AbstractDiscretePNVector}, ψ::AbstractDiscretePNSolution)
    if any(_h -> _is_adjoint_vector(_h) == _is_adjoint_solution(ψ), h) @warn "Adjoint vector and solution are not compatible" end
    res = zeros(size(h))
    if _is_adjoint_solution(ψ)
        return solve_and_integrate_adjoint!(res, h, ψ)
    else
        return solve_and_integrate_nonadjoint!(res, h, ψ)
    end
end

function Base.:*(h::AbstractArray{<:AbstractDiscretePNVector}, A::AbstractDiscretePNSystem, g::AbstractArray{<:AbstractDiscretePNVector})
    full_size = (size(h)..., size(g)...)
    result = zeros(full_size)
    if length(g) < length(h)
        for i in eachindex(IndexCartesian(), g)
            solution = iterator(A, g[i])
            result[axes(h)..., i] = h(solution)
        end
    else
        for i in eachindex(IndexCartesian(), h)
            solution = iterator(adjoint(A), h[i])
            result[i, axes(g)...] = g(solution)
        end
    end
    return result
end