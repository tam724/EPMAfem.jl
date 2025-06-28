
@concrete struct LazyReshapeMatrix{T, MT<:AbstractMatrix{T}} <: AbstractLazyMatrix{T}
    A::MT# not a AbstractLazyArray
    size::Tuple{Base.RefValue{Int}, Base.RefValue{Int}}
end

_reshape_view(R::LazyReshapeMatrix) = reshape(@view(R.A[1:prod(size(R))]), size(R))
function reshape!(R, (m, n)::Tuple{<:Integer, <:Integer})
    if size(R.A, 1) < m || size(R.A, 2) < n
        error("size too big!")
    end
    R.size[1][] = m
    R.size[2][] = n
    return R
end

Base.copyto!(R::LazyReshapeMatrix, vals::AbstractArray) = copyto!(_reshape_view(R), vals)
@inline A(R::LazyReshapeMatrix) = _reshape_view(R)
Base.size(R::LazyReshapeMatrix) = (R.size[1][], R.size[2][])
max_size(R::LazyReshapeMatrix) = size(R.A)
lazy_getindex(R::LazyReshapeMatrix, i::Integer, j::Integer) = getindex(_reshape_view(R), i, j)
@inline isdiagonal(R::LazyReshapeMatrix) = false

# for ScaleMatrix (fused) the mul_with! simply calls back into mul!
mul_with!(::Workspace, Y::AbstractVecOrMat, S::LazyReshapeMatrix, X::AbstractVecOrMat, α::Number, β::Number) = mul!(Y, A(S), X, α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, S::LazyReshapeMatrix, α::Number, β::Number) = mul!(Y, X, A(S), α, β)
mul_with!(::Workspace, Y::AbstractVecOrMat, St::Transpose{T, <:LazyReshapeMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T= mul!(Y, transpose(A(parent(St))), X, α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, St::Transpose{T, <:LazyReshapeMatrix{T}}, α::Number, β::Number) where T = mul!(Y, X, transpose(A(parent(St))), α, β)
required_workspace(::typeof(mul_with!), S::LazyReshapeMatrix) = 0

required_workspace(::typeof(materialize_with), S::LazyReshapeMatrix) = 0

function materialize_with(ws::Workspace, R::LazyReshapeMatrix, ::Nothing)
    return _reshape_view(R), ws
end

function materialize_with(ws::Workspace, R::LazyReshapeMatrix, skeleton::AbstractMatrix)
    skeleton .= _reshape_view(R)
    return skeleton, ws
end

broadcast_materialize(::ReshapeableMatrix) = ShouldBroadcastMaterialize()
materialize_broadcasted(::Workspace, R::ScaleMatrix) = _reshape_view(R)
required_workspace(::typeof(materialize_with), S::ScaleMatrix) = r0
