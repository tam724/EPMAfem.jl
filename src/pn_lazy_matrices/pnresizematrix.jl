# TODO: combine LazyResizeMatrix with LazyMatrix to unify the interface. (but think twice about Diagonal, Sparse etc..)
@concrete struct LazyResizeMatrix{T, A<:AbstractArray{T}} <: AbstractLazyMatrix{T}
    R_mem::Base.RefValue{A} # not Lazy (probably dense)
    max_size::Tuple{Int64, Int64}
    size::Tuple{Base.RefValue{Int64}, Base.RefValue{Int64}}
end

function LazyResizeMatrix(A::AbstractMatrix{T}) where T
    v_A = A[:]
    return LazyResizeMatrix{T, typeof(v_A)}(Ref(v_A), size(A), Ref.(size(A)))
end

function LazyResizeMatrix(A::AbstractMatrix{T}, size_R) where T
    v_A = A[:]
    return LazyResizeMatrix{T, typeof(v_A)}(Ref(v_A), size(A), Ref.(size_R))
end


_reshape_view(R::LazyResizeMatrix) = reshape(@view(R.R_mem[][1:prod(size(R))]), size(R))

function quiet_resize!(R::LazyResizeMatrix, (m, n)::Tuple{<:Integer, <:Integer})
    if R.max_size[1] < m || R.max_size[2] < n
        @show R.max_size, (m, n)
        error("size too big!")
    end
    R.size[1][] = m
    R.size[2][] = n
end

function lazy_resize!(ws::Workspace, R::LazyResizeMatrix, (m, n)::Tuple{<:Integer, <:Integer})
    if size(R, 1) == m && size(R, 2) == n return end # return without notifying the cache
    quiet_resize!(R, (m, n))
    notify_cache(ws, R)
end

lazy_resize!(ws::Workspace, R::LazyResizeMatrix, (m, _)::Tuple{<:Integer, Colon}) = lazy_resize!(ws, R, (m, size(R, 2)))
lazy_resize!(ws::Workspace, R::LazyResizeMatrix, (_, n)::Tuple{Colon, <:Integer}) = lazy_resize!(ws, R, (size(R, 1), :))

@inline A(R::LazyResizeMatrix) = _reshape_view(R)
Base.size(R::LazyResizeMatrix) = (R.size[1][], R.size[2][])
max_size(R::LazyResizeMatrix) = R.max_size
max_size(R::LazyResizeMatrix, n::Integer) = R.max_size[n]
lazy_getindex(R::LazyResizeMatrix, i::Integer, j::Integer) = getindex(_reshape_view(R), i, j)
@inline isdiagonal(R::LazyResizeMatrix) = false

function lazy_setindex!(ws::Workspace, R::LazyResizeMatrix, val, i::Integer, j::Integer)
    A(R)[i, j] = val
    notify_cache(ws, R)
end

function quiet_copyto!(R::LazyResizeMatrix, A_)
    @assert size(R) == size(A_)
    copyto!(A(R), A_)
end

function lazy_copyto!(ws::Workspace, R::LazyResizeMatrix, A_)
    quiet_copyto!(R, A_)
    notify_cache(ws, R)
end

function lazy_resize_copyto!(ws::Workspace, R::LazyResizeMatrix, A_)
    quiet_resize!(R, size(A_))
    quiet_copyto!(R, A_)
    notify_cache(ws, R)
end

function quiet_set_memory!(R::LazyResizeMatrix, R_mem_new)
    R.R_mem[] = R_mem_new
end

function lazy_set_memory!(ws::Workspace, R::LazyResizeMatrix, R_mem_new)
    if R.R_mem[] === R_mem_new return end # if the underlying memory and the size did not change (this is happening in every dlr step..)
    quiet_set_memory!(R, R_mem_new)
    notify_cache(ws, R)
end

function lazy_set!(ws::Workspace, R::LazyResizeMatrix, R_mem_new, (m, n)::Tuple{<:Integer, <:Integer})
    if size(R, 1) == m && size(R, 2) == n && R.R_mem[] === R_mem_new return end # return without notifying the cache
    quiet_set_memory!(R, R_mem_new)
    quiet_resize!(R, (m, n))
    notify_cache(ws, R)
end

mul_with!(::Workspace, Y::AbstractVecOrMat, S::LazyResizeMatrix, X::AbstractVecOrMat, α::Number, β::Number) = mul_with!(nothing, Y, A(S), X, α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, S::LazyResizeMatrix, α::Number, β::Number) = mul_with!(nothing, Y, X, A(S), α, β)
mul_with!(::Workspace, Y::AbstractVecOrMat, St::Transpose{T, <:LazyResizeMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T= mul_with!(nothing, Y, transpose(A(parent(St))), X, α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, St::Transpose{T, <:LazyResizeMatrix{T}}, α::Number, β::Number) where T = mul_with!(nothing, Y, X, transpose(A(parent(St))), α, β)
function required_workspace(::typeof(mul_with!), S::LazyResizeMatrix, n, cache_notifier)
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

function required_workspace(::typeof(materialize_with), S::LazyResizeMatrix, cache_notifier)
    return 0 + register_cache_notifier(S, cache_notifier)
end

function register_cache_notifier(s::LazyResizeMatrix, cache_notifier)
    return WorkspaceSize(0, CacheStructure(nothing, Dict(lazy_objectid(s)=>cache_notifier)))
end
