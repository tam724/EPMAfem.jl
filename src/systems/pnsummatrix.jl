const TSumMatrix{T} = LazyOpMatrix{T, typeof(+), <:Tuple{Vararg{<:AbstractMatrix{T}}}}
const VSumMatrix{T} = LazyOpMatrix{T, typeof(+), <:AbstractVector{<:AbstractMatrix{T}}}
const SumMatrix{T} = Union{TSumMatrix{T}, VSumMatrix{T}}
@inline As(S::SumMatrix) = S.args
Base.size(S::SumMatrix) = only_unique(size(A) for A in As(S))
max_size(S::SumMatrix) = only_unique(max_size(A) for A in As(S))
lazy_getindex(S::SumMatrix, idx::Vararg{<:Integer}) = +(getindex.(As(S), idx...)...)
isdiagonal(S::SumMatrix) = all(isdiagonal, As(S))

broadcast_materialize(T::TSumMatrix) = broadcast_materialize(As(T)...)

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, S::VSumMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    for A in As(S)
        mul_with!(ws, Y, A, X, α, β)
        β = true
    end
    return
end
# TODO: worth doing for all mul_with!(⋅) ?
mul_with!(ws::Workspace, Y::AbstractVecOrMat, S::TSumMatrix, X::AbstractVecOrMat, α::Number, β::Number) = _sum_mul_with!(ws, Y, As(S), X, α, β)
function _sum_mul_with!(ws::Workspace, Y::AbstractVecOrMat, (A, Ss...)::Tuple{<:AbstractMatrix, Vararg{<:AbstractMatrix}}, X::AbstractVecOrMat, α::Number, β::Number)
    _sum_mul_with!(ws, Y, Ss, X, α, β)
    mul_with!(ws, Y, A, X, α, true)
end
_sum_mul_with!(ws::Workspace, Y::AbstractVecOrMat, A::Tuple{<:AbstractMatrix}, X::AbstractVecOrMat, α::Number, β::Number) = mul_with!(ws, Y, only(A), X, α, β)

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, St::Transpose{T, <:SumMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T
    for A in parent(St).args
        mul_with!(ws, Y, transpose(A), X, α, β)
        β = true
    end
    return
end

function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, S::SumMatrix, α::Number, β::Number)
    for A in As(S)
        mul_with!(ws, Y, X, A, α, β)
        β = true
    end
    return
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, St::Transpose{T, <:SumMatrix{T}}, α::Number, β::Number) where T
    for A in parent(St).args
        mul_with!(ws, Y, X, transpose(A), α, β)
        β = true
    end
    return
end
required_workspace(::typeof(mul_with!), S::SumMatrix) = maximum(required_workspace(mul_with!, A) for A in As(S))

function materialize_with(ws::Workspace, S::SumMatrix, from_cache=nothing)
    # what we do here is to wrap every component into a lazy(materialize, ) and then materialize the full matrix
    S_mat, rem = structured_from_ws(ws, S, from_cache)
    S_mat .= zero(eltype(S_mat))
    for A in As(S)
        A_mat, _ = materialize_with(rem, materialize(A), nothing)
        S_mat .+= A_mat
    end
    return S_mat, rem
end

materialize_broadcasted(ws::Workspace, S::SumMatrix) = Base.Broadcast.broadcasted(+, materialize_broadcasted.(Ref(ws), As(S))...)

function required_workspace(::typeof(materialize_with), S::SumMatrix)
    max_workspace = 0
    for A in As(S)
        max_workspace = max(max_workspace, required_workspace(materialize_with, materialize(A)))
    end
    return max_workspace
end
