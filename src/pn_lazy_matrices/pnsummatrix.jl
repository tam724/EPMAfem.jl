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
required_workspace(::typeof(mul_with!), S::SumMatrix, cache_notifier) = maximum(required_workspace(mul_with!, A, cache_notifier) for A in As(S))

_fillzero!(A::AbstractArray) = fill!(A, zero(eltype(A)))
_fillzero!(D::Diagonal) = fill!(D.diag, zero(eltype(D)))

_add!(A::AbstractArray, B::AbstractArray) = axpy!(true, B, A)
_add!(A::Diagonal, B::Diagonal) = axpy!(true, B.diag, A.diag)

materialize_with(ws::Workspace, S::SumMatrix, skeleton::AbstractMatrix) = materialize_with(ws, S, skeleton, true, false)
function materialize_with(ws::Workspace, S::SumMatrix, skeleton::AbstractMatrix, α::Number, β::Number)
    A_mat, _ = materialize_with(ws, first(As(S)), skeleton, α, β)
    for A in As(S)[2:end]
        A_mat, _ = materialize_with(ws, A, skeleton, α, true)
    end
    return A_mat, ws
end

materialize_broadcasted(ws::Workspace, S::SumMatrix) = Base.Broadcast.broadcasted(+, materialize_broadcasted.(Ref(ws), As(S))...)

required_workspace(::typeof(materialize_with), S::SumMatrix, cache_notifier) = maximum(required_workspace(materialize_with, A, cache_notifier) for A in As(S))
