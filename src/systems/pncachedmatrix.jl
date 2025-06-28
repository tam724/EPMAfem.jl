const MaterializedMatrix{T} = LazyOpMatrix{T, typeof(materialize), <:Tuple{<:AbstractLazyMatrixOrTranspose{T}}}
const CachedMatrix{T} = LazyOpMatrix{T, typeof(cache), <:Tuple{<:AbstractLazyMatrixOrTranspose{T}}}
const MaterializedOrCachedMatrix{T} = Union{MaterializedMatrix{T}, CachedMatrix{T}}

@inline A(M::MaterializedOrCachedMatrix) = only(M.args)
Base.size(M::MaterializedOrCachedMatrix) = size(A(M))
max_size(M::MaterializedOrCachedMatrix) = max_size(A(M))
lazy_getindex(M::MaterializedOrCachedMatrix{T}, idx::Vararg{<:Integer}) where T = lazy_getindex(A(M), idx...)
@inline isdiagonal(M::MaterializedOrCachedMatrix) = isdiagonal(A(M))
lazy_objectid(M::MaterializedOrCachedMatrix) = lazy_objectid(A(M))

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, M::MaterializedOrCachedMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    materialized_M, _ = materialize_with(ws, M)
    mul!(Y, materialized_M, X, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractVecOrMat, Mt::Transpose{T, <:MaterializedOrCachedMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T
    materialized_M, _ = materialize_with(ws, parent(Mt))
    mul!(Y, transpose(materialized_M), X, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, M::MaterializedOrCachedMatrix, α::Number, β::Number)
    materialized_M, _ = materialize_with(ws, M)
    mul!(Y, X, materialized_M, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, Mt::Transpose{T, <:MaterializedOrCachedMatrix{T}}, α::Number, β::Number) where T
    materialized_M, _ = materialize_with(ws, parent(Mt))
    mul!(Y, X, transpose(materialized_M), α, β)
end
# this may be extended to multiplications of multiple materialized matrices.. (we are good with only one now..)
required_workspace(::typeof(mul_with!), M::MaterializedOrCachedMatrix) = required_workspace(materialize_with, M)


# the materialize_with is different for MaterializeMatrix and CachedMatrix though...
##### MaterializedMatrix
materialize_with(ws::Workspace, M::MaterializedMatrix, from_cache=nothing) = materialize_with(broadcast_materialize(A(M)), ws, M, from_cache)
function materialize_with(::ShouldBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix, from_cache)
    bcd = Base.Broadcast.instantiate(materialize_broadcasted(ws, A(M)))
    A_, rem = structured_from_ws(ws, A(M), from_cache)
    M_ = Base.Broadcast.materialize!(A_, bcd)
    return M_, rem
    # if isdiagonal(M) #ideally the compiler can proof this, we could also implement a trait for that.. if the compiler can proof, this is type stable..
    #     ws_M, rem = take_ws(ws, only_unique(size(M)))
    #     materialized_M = Base.Broadcast.materialize!(Diagonal(ws_M), bcd)
    #     return materialized_M, rem
    # end
    # ws_M, rem = take_ws(ws, size(M))

    # materialized_M = Base.Broadcast.materialize!(ws_M, bcd)
    # return materialized_M, rem
end
materialize_with(::ShouldNotBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix, from_cache) = materialize_with(ws, A(M), from_cache)

# simply pass through the broadcast materialize calls.. 
broadcast_materialize(S::MaterializedMatrix) = broadcast_materialize(A(S))
materialize_broadcasted(ws::Workspace, S::MaterializedMatrix) = materialize_broadcasted(ws, A(S))

function required_workspace(::typeof(materialize_with), M::MaterializedMatrix)
    if isdiagonal(A(M)) #  we only track diagonal (thats the only thing we will need this for, not general though..)
        return only_unique(max_size(M)) + required_workspace(materialize_with, A(M))
    else
        return prod(max_size(M)) + required_workspace(materialize_with, A(M))
    end
end
materialize(M::Union{MaterializedMatrix{T}, Transpose{T, <:MaterializedMatrix{T}}}) where T = M

##### CachedMatrix
function materialize_with(ws::Workspace, C::CachedMatrix, from_cache=nothing)
    valid, _ = ws.cache[lazy_objectid(A(C))]
    if !valid[]
        materialize_with(ws, materialize(A(C)), lazy_objectid(A(C)))
        valid[] = true
    end
    return structured_from_ws(ws, A(C), lazy_objectid(A(C)))
end

# simply pass through the broadcast materialize calls.. 
broadcast_materialize(::CachedMatrix) = ShouldBroadcastMaterialize()
materialize_broadcasted(ws::Workspace, C::CachedMatrix) = first(materialize_with(ws, C))

function required_workspace(::typeof(materialize_with), C::CachedMatrix)
    if isdiagonal(A(C)) #  we only track diagonal (thats the only thing we will need this for, not general though..)
        return WorkspaceSize(0, Dict(lazy_objectid(C) => only_unique(max_size(C)))) + required_workspace(materialize_with, A(C))
    else
        return WorkspaceSize(0, Dict(lazy_objectid(C) =>  prod(max_size(C)))) + required_workspace(materialize_with, A(C))
    end
end
materialize(M::Union{CachedMatrix{T}, Transpose{T, <:CachedMatrix{T}}}) where T = M
