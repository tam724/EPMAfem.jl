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
    materialized_M, _ = materialize_with(ws, M, nothing)
    mul!(Y, materialized_M, X, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractVecOrMat, Mt::Transpose{T, <:MaterializedOrCachedMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T
    materialized_M, _ = materialize_with(ws, parent(Mt), nothing)
    mul!(Y, transpose(materialized_M), X, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, M::MaterializedOrCachedMatrix, α::Number, β::Number)
    materialized_M, _ = materialize_with(ws, M, nothing)
    mul!(Y, X, materialized_M, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, Mt::Transpose{T, <:MaterializedOrCachedMatrix{T}}, α::Number, β::Number) where T
    materialized_M, _ = materialize_with(ws, parent(Mt), nothing)
    mul!(Y, X, transpose(materialized_M), α, β)
end
# this may be extended to multiplications of multiple materialized matrices.. (we are good with only one now..)
required_workspace(::typeof(mul_with!), M::MaterializedOrCachedMatrix) = required_workspace(materialize_with, M)


# the materialize_with is different for MaterializeMatrix and CachedMatrix though...
##### MaterializedMatrix
materialize_with(ws::Workspace, M::MaterializedMatrix, skeleton::Nothing) = materialize_with(broadcast_materialize(A(M)), ws, M, skeleton)
materialize_with(ws::Workspace, M::MaterializedMatrix, skeleton::AbstractMatrix) = materialize_with(broadcast_materialize(A(M)), ws, M, skeleton)
function materialize_with(::ShouldBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix, ::Nothing)
    bcd = Base.Broadcast.instantiate(materialize_broadcasted(ws, A(M)))
    A_, rem = structured_from_ws(ws, A(M))
    M_ = Base.Broadcast.materialize!(A_, bcd)
    return M_, rem
end
function materialize_with(::ShouldBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix, skeleton::AbstractMatrix)
    bcd = Base.Broadcast.instantiate(materialize_broadcasted(ws, A(M)))
    Base.Broadcast.materialize!(skeleton, bcd)
    return skeleton, ws
end
materialize_with(::ShouldNotBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix, skeleton) = materialize_with(ws, A(M), skeleton)

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
function materialize_with(ws::Workspace, C::CachedMatrix, ::Nothing)
    valid, memory = ws.cache[lazy_objectid(A(C))]
    C_cached = structured_mat_view(memory, C)
    if !valid[]
        C_cached, _ = materialize_with(ws, materialize(A(C)), C_cached)
        valid[] = true
    end
    return C_cached, ws
end
materialize_with(ws::Workspace, C::CachedMatrix, skeleton::AbstractMatrix) = error("TODO")

# simply pass through the broadcast materialize calls.. 
broadcast_materialize(::CachedMatrix) = ShouldBroadcastMaterialize()
materialize_broadcasted(ws::Workspace, C::CachedMatrix) = first(materialize_with(ws, C, nothing))

function required_workspace(::typeof(materialize_with), C::CachedMatrix)
    if isdiagonal(A(C)) #  we only track diagonal (thats the only thing we will need this for, not general though..)
        return WorkspaceSize(0, Dict(lazy_objectid(C) => only_unique(max_size(C)))) + required_workspace(materialize_with, A(C))
    else
        return WorkspaceSize(0, Dict(lazy_objectid(C) =>  prod(max_size(C)))) + required_workspace(materialize_with, A(C))
    end
end
materialize(M::Union{CachedMatrix{T}, Transpose{T, <:CachedMatrix{T}}}) where T = M
