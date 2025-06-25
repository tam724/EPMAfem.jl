const ScaleMatrixL{T, M <: AbstractMatrix{T}} = LazyOpMatrix{T, typeof(*), <:Tuple{T, M}}
const ScaleMatrixR{T, M <: AbstractMatrix{T}} = LazyOpMatrix{T, typeof(*), <:Tuple{M, T}}
const ScaleMatrix{T, M<:AbstractMatrix{T}} = Union{ScaleMatrixL{T, M}, ScaleMatrixR{T, M}}
const NotFusedScaleMatrix{T} = ScaleMatrix{T, <:AbstractLazyMatrix{T}}
@inline a(S::ScaleMatrixL) = S.args[1]
@inline A(S::ScaleMatrixL) = S.args[2]
@inline a(S::ScaleMatrixR) = S.args[2]
@inline A(S::ScaleMatrixR) = S.args[1]
Base.size(S::ScaleMatrix) = Base.size(A(S))
max_size(S::ScaleMatrix) = max_size(A(S))
Base.getindex(S::ScaleMatrix, idx::Vararg{<:Integer}) = *(a(S), getindex(A(S), idx...))
@inline isdiagonal(S::ScaleMatrix) = isdiagonal(A(S))


mul_with!(ws::Workspace, Y::AbstractVecOrMat, S::NotFusedScaleMatrix, X::AbstractVecOrMat, α::Number, β::Number) = mul_with!(ws, Y, A(S), X, a(S)*α, β)
mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, S::NotFusedScaleMatrix, α::Number, β::Number) = mul_with!(ws, Y, X, A(S), a(S)*α, β)
mul_with!(ws::Workspace, Y::AbstractVecOrMat, St::Transpose{T, <:NotFusedScaleMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T = mul_with!(ws, Y, transpose(A(parent(St))), X, a(parent(St))*α, β)
mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, St::Transpose{T, <:NotFusedScaleMatrix{T}}, α::Number, β::Number) where T = mul_with!(ws, Y, X, transpose(A(parent(St))), a(parent(St))*α, β)
required_workspace(::typeof(mul_with!), S::NotFusedScaleMatrix) = required_workspace(mul_with!, A(S))

# for ScaleMatrix (fused) the mul_with! simply calls back into mul!
mul_with!(::Workspace, Y::AbstractVecOrMat, S::ScaleMatrix, X::AbstractVecOrMat, α::Number, β::Number) = mul!(Y, S, X, α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, S::ScaleMatrix, α::Number, β::Number) = mul!(Y, X, S, α, β)
LinearAlgebra.mul!(Y::AbstractVector, S::ScaleMatrix, X::AbstractVector, α::Number, β::Number) = mul!(Y, A(S), X, a(S)*α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, S::ScaleMatrix, X::AbstractMatrix, α::Number, β::Number) = mul!(Y, A(S), X, a(S)*α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, S::ScaleMatrix, α::Number, β::Number) = mul!(Y, X, A(S), a(S)*α, β)
LinearAlgebra.mul!(Y::AbstractVector, St::Transpose{T, <:ScaleMatrix{T}}, X::AbstractVector, α::Number, β::Number) where T = mul!(Y, transpose(A(parent(St))), X, a(parent(St))*α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, St::Transpose{T, <:ScaleMatrix{T}}, X::AbstractMatrix, α::Number, β::Number) where T = mul!(Y, transpose(A(parent(St))), X, a(parent(St))*α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, St::Transpose{T, <:ScaleMatrix{T}}, α::Number, β::Number) where T = mul!(Y, X, transpose(A(parent(St))), a(parent(St))*α, β)
required_workspace(::typeof(mul_with!), S::ScaleMatrix) = 0

materialize_with(ws::Workspace, S::ScaleMatrix) = materialize_with(broadcast_materialize(S), ws, S)
function materialize_with(::ShouldNotBroadcastMaterialize, ws::Workspace, S::ScaleMatrix)
    S_mat, rem = structured_from_ws(ws, S)
    A_mat, _ = materialize_with(rem, materialize(A(S)))
    S_mat .= a(S) .* A_mat
    return S_mat, rem
end
# function materialized_with(::ShouldBroadcastMaterialize, ws, _, S::ScaleMatrix)

broadcast_materialize(S::ScaleMatrix) = broadcast_materialize(A(S))
materialize_broadcasted(S::ScaleMatrix) = Base.Broadcast.broadcasted(*, a(S), materialize_broadcasted(A(S)))
required_workspace(::typeof(materialize_with), S::ScaleMatrix) = required_workspace(broadcast_materialize(S), materialize_with, S)
required_workspace(::ShouldBroadcastMaterialize, ::typeof(materialize_with), S::ScaleMatrix) = 0
required_workspace(::ShouldNotBroadcastMaterialize, ::typeof(materialize_with), S::ScaleMatrix) = required_workspace(materialize_with, A(S)) # todo..

# materialize_with(::Workspace, ::AbstractVector, S::ScaleMatrix) = S
# function materialize_with(ws::Workspace, materialized::AbstractVector, S::NotFusedScaleMatrix)
#     materialize_with(ws, materialized, A(S))
#     materialized .*= a(S)
#     return reshape(materialized, size(S))
# end
# required_workspace(::typeof(materialize_with), S::ScaleMatrix) = 0
# required_workspace(::typeof(materialize_with), S::NotFusedScaleMatrix) = required_workspace(materialize_with, A(S))


const MulMatrix{T, N} = LazyOpMatrix{T, typeof(*), <:NTuple{N, AbstractMatrix{T}}}
Base.size(M::MulMatrix) = (size(first(M.args), 1), size(last(M.args), 2)) # implement this safer
max_size(M::MulMatrix) = (max_size(first(M.args), 1), max_size(last(M.args), 2)) # implement this safer
function mul_with!(ws::Workspace, y::AbstractVector, A::MulMatrix, x::AbstractVector, α::Number, β::Number)
    error("Todo")
end
required_workspace(::typeof(mul_with!), M::MulMatrix) = 0


# @concrete struct ProductMatrix{T} <: AbstractPNMatrix{T}
#     A
#     B
#     C
# end

