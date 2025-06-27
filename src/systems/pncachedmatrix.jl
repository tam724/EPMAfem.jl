const MaterializedMatrix{T} = LazyOpMatrix{T, typeof(materialize), <:Tuple{<:AbstractLazyMatrixOrTranspose{T}}}

@inline A(M::MaterializedMatrix) = only(M.args)
Base.size(M::MaterializedMatrix) = size(A(M))
max_size(M::MaterializedMatrix) = max_size(A(M))
Base.getindex(M::MaterializedMatrix{T}, idx::Vararg{<:Integer}) where T = getindex(A(M), idx...)
@inline isdiagonal(M::MaterializedMatrix) = isdiagonal(A(M))

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, M::MaterializedMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    materialized_M, _ = materialize_with(ws, M)
    mul!(Y, materialized_M, X, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractVecOrMat, Mt::Transpose{T, <:MaterializedMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T
    materialized_M, _ = materialize_with(ws, parent(Mt))
    mul!(Y, transpose(materialized_M), X, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, M::MaterializedMatrix, α::Number, β::Number)
    materialized_M, _ = materialize_with(ws, M)
    mul!(Y, X, materialized_M, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, Mt::Transpose{T, <:MaterializedMatrix{T}}, α::Number, β::Number) where T
    materialized_M, _ = materialize_with(ws, parent(Mt))
    mul!(Y, X, transpose(materialized_M), α, β)
end
# this may be extended to multiplications of multiple materialized matrices.. (we are good with only one now..)
required_workspace(::typeof(mul_with!), M::MaterializedMatrix) = required_workspace(materialize_with, M)

materialize_with(ws::Workspace, M::MaterializedMatrix) = materialize_with(broadcast_materialize(A(M)), ws, M)
function materialize_with(::ShouldBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix)
    bcd = Base.Broadcast.instantiate(materialize_broadcasted(A(M)))
    if isdiagonal(M) #ideally the compiler can proof this, we could also implement a trait for that.. if the compiler can proof, this is type stable..
        ws_M, rem = take_ws(ws, only_unique(size(M)))
        materialized_M = Base.Broadcast.materialize!(Diagonal(ws_M), bcd)
        return materialized_M, rem
    end
    ws_M, rem = take_ws(ws, size(M))
    materialized_M = Base.Broadcast.materialize!(ws_M, bcd)
    return materialized_M, rem
end
materialize_with(::ShouldNotBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix) = materialize_with(ws, A(M))

# simply pass through the broadcast materialize calls.. 
broadcast_materialize(S::MaterializedMatrix) = broadcast_materialize(A(S))
materialize_broadcasted(S::MaterializedMatrix) = materialize_broadcasted(A(S))

function required_workspace(::typeof(materialize_with), M::MaterializedMatrix)
    if isdiagonal(A(M)) #  we only track diagonal (thats the only thing we will need this for, not general though..)
        return only_unique(max_size(M)) + required_workspace(materialize_with, A(M))
    else
        return prod(max_size(M)) + required_workspace(materialize_with, A(M))
    end
end

materialize(M::Union{MaterializedMatrix{T}, Transpose{T, <:MaterializedMatrix{T}}}) where T = M


# struct Cached{T, PT} <: AbstractPNMatrix{T}
#     P::PT
#     o
# end

# function Cached(P::AbstractMatrix)
#     o = Observable(-1)
#     if is_observable(P)
#         on(_ -> o[] = -1, get_observable(P))
#     end
#     return Cached{eltype(P), typeof(P)}(P, o)
# end

# size_string(C::Cached{T}) where T = "$(size(C, 1))x$(size(C, 2)) Cached{$(typeof(C.P).name.wrapper){$(typeof(C.P).parameters[1])}}"
# size_string(C::Transpose{T, <:Cached{T}}) where T = "$(size(C, 1))x$(size(C, 2)) transpose(::Cached{$(typeof(parent(C).P).name.wrapper){$(typeof(parent(C).P).parameters[1])})}"
# content_string(C::Cached{T}) where T = content_string(C.P)
# content_string(C::Transpose{T, <:Cached{T}}) where T = content_string(transpose(parent(C).P))

# function cache_with!(ws::WorkspaceCache, outer_cached, C::Cached, α::Number, β::Number)
#     inner_cached = get_cached!(ws, C)
#     outer_cached .= α .* inner_cached .+ β .* outer_cached
# end

# Base.size(C::Cached) = size(C.P)
# max_size(C::Cached) = max_size(C.P)

# function required_workspace_cache(C::Cached)
#     P_wsch = required_workspace_cache(C.P)

#     # P_mul_with_ws = mul_with_ws(P_wsch) # this will never mul_with, it is cached. therefore mul_with_ws = cache_with_ws, but will only recompute the cache if it is invalid
#     P_cache_with_ws = cache_with_ws(P_wsch)

#     return WorkspaceCache((prod(size(C.P)), ch(P_wsch)), (mul_with=P_cache_with_ws, cache_with=P_cache_with_ws))
# end

# function invalidate_cache!(C::Cached)
#     invalidate_cache!(C.P)
#     C.o[] = -1
# end

# function get_cached!(ws::WorkspaceCache, C::Cached)
#     cache_and_id, rem = take_ch(ws)
#     cache, cache_id = cache_and_id
#     cached = reshape(@view(cache[:]), size(C.P))
#     if !(C.o[] == cache_id) # check_cache_invalidity
#         @show "caching"
#         cache_with!(rem, cached, C.P, true, false)
#         C.o[] = cache_id
#     end
#     return cached
# end

# LinearAlgebra.isdiag(C::Cached) = isdiag(C.P)
# LinearAlgebra.issymmetric(C::Cached) = issymmetric(C.P)

# function mul_with!(ws::WorkspaceCache, Y::AbstractVecOrMat, C::Cached, X::AbstractVecOrMat, α::Number, β::Number)
#     cached = get_cached!(ws, C)
#     mul!(Y, cached, X, α, β)
# end

# function mul_with!(ws::WorkspaceCache, Y::AbstractVecOrMat, X::AbstractVecOrMat, C::Cached, α::Number, β::Number)
#     cached = get_cached!(ws, C)
#     mul!(Y, X, cached, α, β)
# end

# function mul_with!(ws::WorkspaceCache, Y::AbstractVecOrMat, Ct::Transpose{T, <:Cached{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T
#     cached = get_cached!(ws, parent(Ct))
#     mul!(Y, transpose(cached), X, α, β)
# end

# function mul_with!(ws::WorkspaceCache, Y::AbstractVecOrMat, X::AbstractVecOrMat, Ct::Transpose{T, <:Cached{T}}, α::Number, β::Number) where T
#     cached = get_cached!(ws, parent(Ct))
#     mul!(Y, X, transpose(cached), α, β)
# end

# # wrapped
# @concrete struct Wrapped{T} <: AbstractPNMatrix{T}
#     A
#     workspace_cache
#     o
# end

# function Wrapped(A)
#     T = eltype(A)
#     o = Observable(nothing)
#     if is_observable(A)
#         on(_ -> notify(o), get_observable(A))
#     end
#     ws = allocate_workspace_cache(mul_with!, required_workspace_cache(A))
#     return Wrapped{T}(A, ws, o)
# end

# size_string(W::Wrapped{T}) where T = "$(size(W, 1))x$(size(W, 2)) Wrapped{$(typeof(W.A).name.wrapper){$(typeof(W.A).parameters[1])}}"
# size_string(Wt::Transpose{T, <:Wrapped{T}}) where T = "$(size(Wt, 1))x$(size(Wt, 2)) transpose(::Wrapped{$(typeof(parent(Wt).A).name.wrapper){$(typeof(parent(Wt).A).parameters[1])})}"
# content_string(W::Wrapped{T}) where T = content_string(W.A)
# content_string(Wt::Transpose{T, <:Wrapped{T}}) where T = content_string(transpose(parent(Wt).A))

# Base.size(W::Wrapped) = size(W.A)
# max_size(W::Wrapped) = max_size(W.A)

# function invalidate_cache!(W::Wrapped)
#     invalidate_cache!(W.A)
# end

# LinearAlgebra.isdiag(W::Wrapped) = isdiag(W.A)
# LinearAlgebra.issymmetric(W::Wrapped) = issymmetric(W.A)

# function LinearAlgebra.mul!(y::AbstractVector, W::Wrapped, x::AbstractVector, α::Number, β::Number)
#     mul_with!(W.workspace_cache, y, W.A, x, α, β)
#     return y
# end

# function LinearAlgebra.mul!(y::AbstractVector, Wt::Transpose{T, <:Wrapped{T}}, x::AbstractVector, α::Number, β::Number) where T
#     mul_with!(parent(Wt).workspace_cache, y, transpose(parent(Wt).A), x, α, β)
#     return y
# end
