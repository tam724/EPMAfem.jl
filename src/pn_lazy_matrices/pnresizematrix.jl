
@concrete struct LazyResizeMatrix{T, MT<:AbstractMatrix{T}} <: AbstractLazyMatrix{T}
    A::MT # not a AbstractLazyArray
    size::Tuple{Base.RefValue{Int}, Base.RefValue{Int}}
end

_reshape_view(R::LazyResizeMatrix) = reshape(@view(R.A[1:prod(size(R))]), size(R))

function quiet_resize!(R::LazyResizeMatrix, (m, n)::Tuple{<:Integer, <:Integer})
    if size(R.A, 1) < m || size(R.A, 2) < n
        error("size too big!")
    end
    R.size[1][] = m
    R.size[2][] = n
end

function resize!(R::LazyResizeMatrix, (m, n)::Tuple{<:Integer, <:Integer}, ws::Workspace)
    quiet_resize!(R, (m, n))
    notify_cache(ws, R)
end

resize!(R::LazyResizeMatrix, (m, _)::Tuple{<:Integer, Colon}, ws::Workspace) = resize!(R, (m, size(R, 2)), ws)
resize!(R::LazyResizeMatrix, (_, n)::Tuple{Colon, <:Integer}, ws::Workspace) = resize!(R, (size(R, 1), :), ws)

@inline A(R::LazyResizeMatrix) = _reshape_view(R)
Base.size(R::LazyResizeMatrix) = (R.size[1][], R.size[2][])
max_size(R::LazyResizeMatrix) = size(R.A)
max_size(R::LazyResizeMatrix, n::Integer) = size(R.A, n)
lazy_getindex(R::LazyResizeMatrix, i::Integer, j::Integer) = getindex(_reshape_view(R), i, j)
@inline isdiagonal(R::LazyResizeMatrix) = false

function Base.setindex!(R::LazyResizeMatrix, val, ws::Workspace, i::Integer, j::Integer)
    A(R)[i, j] = val
    notify_cache(ws, R)
end

function Base.copyto!(ws::Workspace, R::LazyResizeMatrix, A_)
    @assert size(R) == size(A_)
    copyto!(A(R), A_)
    notify_cache(ws, R)
end

function resize_copyto!(ws::Workspace, R::LazyResizeMatrix, A_)
    quiet_resize!(R, size(A_))
    copyto!(ws, R, A_)
end



mul_with!(::Workspace, Y::AbstractVecOrMat, S::LazyResizeMatrix, X::AbstractVecOrMat, α::Number, β::Number) = mul_with!(nothing, Y, A(S), X, α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, S::LazyResizeMatrix, α::Number, β::Number) = mul_with!(nothing, Y, X, A(S), α, β)
mul_with!(::Workspace, Y::AbstractVecOrMat, St::Transpose{T, <:LazyResizeMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T= mul_with!(nothing, Y, transpose(A(parent(St))), X, α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, St::Transpose{T, <:LazyResizeMatrix{T}}, α::Number, β::Number) where T = mul_with!(nothing, Y, X, transpose(A(parent(St))), α, β)
function required_workspace(::typeof(mul_with!), S::LazyResizeMatrix, cache_notifier)
    return 0 + register_cache_notifier(S, cache_notifier)
end

materialize(R::LazyResizeMatrix) = R
materialize(Rt::Transpose{T, <:LazyResizeMatrix{T}}) where T = Rt
function materialize_with(ws::Workspace, R::LazyResizeMatrix)
    return _reshape_view(R), ws
end
function materialize_with(ws::Workspace, Rt::Transpose{T, <:LazyResizeMatrix{T}}) where T
    return transpose(_reshape_view(parent(Rt))), ws
end

should_broadcast_materialize(::LazyResizeMatrix) = ShouldBroadcastMaterialize()
materialize_broadcasted(::Workspace, R::LazyResizeMatrix) = _reshape_view(R)
function required_workspace(::typeof(materialize_with), S::LazyResizeMatrix, cache_notifier)
    return 0 + register_cache_notifier(S, cache_notifier)
end

function register_cache_notifier(s::LazyResizeMatrix, cache_notifier)
    return WorkspaceSize(0, CacheStructure(nothing, Dict(lazy_objectid(s)=>cache_notifier)))
end
