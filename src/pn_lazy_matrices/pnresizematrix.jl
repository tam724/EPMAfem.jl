
@concrete struct LazyResizeMatrix{T, MT<:AbstractMatrix{T}} <: AbstractLazyMatrix{T}
    A::MT# not a AbstractLazyArray
    size::Tuple{Base.RefValue{Int}, Base.RefValue{Int}}
end

_reshape_view(R::LazyResizeMatrix) = reshape(@view(R.A[1:prod(size(R))]), size(R))
function resize!(R, (m, n)::Tuple{<:Integer, <:Integer})
    if size(R.A, 1) < m || size(R.A, 2) < n
        error("size too big!")
    end
    R.size[1][] = m
    R.size[2][] = n
    return R
end

function resize!(R, (m, _)::Tuple{<:Integer, Colon})
    if size(R.A, 1) < m
        error("size too big!")
    end
    R.size[1][] = m
    return R
end

function resize!(R, (_, n)::Tuple{Colon, <:Integer})
    if size(R.A, 2) < n
        error("size too big!")
    end
    R.size[2][] = n
    return R
end

Base.copyto!(R::LazyResizeMatrix, vals::AbstractArray) = copyto!(_reshape_view(R), vals)
do_copyto!(R::LazyResizeMatrix, vals) = copyto!(_reshape_view(R), vals)
@inline A(R::LazyResizeMatrix) = _reshape_view(R)
Base.size(R::LazyResizeMatrix) = (R.size[1][], R.size[2][])
max_size(R::LazyResizeMatrix) = size(R.A)
lazy_getindex(R::LazyResizeMatrix, i::Integer, j::Integer) = getindex(_reshape_view(R), i, j)
@inline isdiagonal(R::LazyResizeMatrix) = false

mul_with!(::Workspace, Y::AbstractVecOrMat, S::LazyResizeMatrix, X::AbstractVecOrMat, α::Number, β::Number) = mul_with!(nothing, Y, A(S), X, α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, S::LazyResizeMatrix, α::Number, β::Number) = mul_with!(nothing, Y, X, A(S), α, β)
mul_with!(::Workspace, Y::AbstractVecOrMat, St::Transpose{T, <:LazyResizeMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T= mul_with!(nothing, Y, transpose(A(parent(St))), X, α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, St::Transpose{T, <:LazyResizeMatrix{T}}, α::Number, β::Number) where T = mul_with!(nothing, Y, X, transpose(A(parent(St))), α, β)
required_workspace(::typeof(mul_with!), S::LazyResizeMatrix) = 0

materialize(R::LazyResizeMatrix) = R
materialize(Rt::Transpose{T, <:LazyResizeMatrix{T}}) where T = Rt
function materialize_with(ws::Workspace, R::LazyResizeMatrix, ::Nothing)
    return _reshape_view(R), ws
end
function materialize_with(ws::Workspace, Rt::Transpose{T, <:LazyResizeMatrix{T}}, ::Nothing) where T
    return transpose(_reshape_view(parent(Rt))), ws
end

broadcast_materialize(::LazyResizeMatrix) = ShouldBroadcastMaterialize()
materialize_broadcasted(::Workspace, R::LazyResizeMatrix) = _reshape_view(R)
required_workspace(::typeof(materialize_with), S::LazyResizeMatrix) = 0
