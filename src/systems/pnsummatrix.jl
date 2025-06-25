const TSumMatrix{T} = LazyOpMatrix{T, typeof(+), <:Tuple{Vararg{<:AbstractMatrix{T}}}}
const VSumMatrix{T} = LazyOpMatrix{T, typeof(+), <:AbstractVector{<:AbstractMatrix{T}}}
const SumMatrix{T} = Union{TSumMatrix{T}, VSumMatrix{T}}
@inline As(S::SumMatrix) = S.args
Base.size(S::SumMatrix) = only_unique(size(A) for A in As(S))
max_size(S::SumMatrix) = only_unique(max_size(A) for A in As(S))
Base.getindex(S::SumMatrix, idx::Vararg{<:Integer}) = +(getindex.(As(S), idx...)...)
@inline isdiagonal(S::SumMatrix) = all(isdiagonal, As(S))

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

function reshape_into(ws::AbstractVector, materialized::Matrix)
    return reshape(@view(ws[1:length(materialized)]), size(materialized))
end

function reshape_into(ws::AbstractVector, materialized::Diagonal)
    return Diagonal(@view(ws[1:length(materialized.diag)]))
end

function materialize_with(ws::Workspace, S::SumMatrix)
    # what we do here is to wrap every component into a lazy(materialize, ) and then materialize the full matrix
    S_mat, rem = structured_from_ws(ws, S)
    S_mat .= zero(eltype(S_mat))
    for A in As(S)
        A_mat, _ = materialize_with(rem, materialize(A))
        S_mat .+= A_mat
    end
    return S_mat, rem
end

materialize_broadcasted(S::SumMatrix) = Base.Broadcast.broadcasted(+, materialize_broadcasted.(As(S))...)

required_workspace(::typeof(materialize_with), S::SumMatrix) = required_workspace(broadcast_materialize(S), materialize_with, S)
required_workspace(::ShouldBroadcastMaterialize, ::typeof(materialize_with), S::SumMatrix) = 0
function required_workspace(::ShouldNotBroadcastMaterialize, ::typeof(materialize_with), S::SumMatrix)
    # we report the maximal workspace to materialize an inner matrix
    max_workspace = 0
    for A in As(S)
        @show typeof(A)
        max_workspace = max(max_workspace, required_workspace(materialize_with, materialize(A)))
    end
    return max_workspace
end
