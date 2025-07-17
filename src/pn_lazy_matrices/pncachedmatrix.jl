const MaterializedMatrix{T} = LazyOpMatrix{T, typeof(materialize), <:Tuple{<:AbstractMatrix{T}}}
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
    materialized_Mt, _ = materialize_with(ws, parent(Mt), nothing)
    mul!(Y, transpose(materialized_Mt), X, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, M::MaterializedOrCachedMatrix, α::Number, β::Number)
    materialized_M, _ = materialize_with(ws, M, nothing)
    mul!(Y, X, materialized_M, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, Mt::Transpose{T, <:MaterializedOrCachedMatrix{T}}, α::Number, β::Number) where T
    materialized_Mt, _ = materialize_with(ws, parent(Mt), nothing)
    mul!(Y, X, transpose(materialized_Mt), α, β)
end
# this may be extended to multiplications of multiple materialized matrices.. (we are good with only one now..)
required_workspace(::typeof(mul_with!), M::MaterializedOrCachedMatrix) = required_workspace(materialize_with, M)


# the materialize_with is different for MaterializeMatrix and CachedMatrix though...
##### MaterializedMatrix
materialize_with(ws::Workspace, M::MaterializedMatrix, ::Nothing) = materialize_with(broadcast_materialize(A(M)), ws, M, nothing)
function materialize_with(::ShouldBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix, ::Nothing)
    bcd = Base.Broadcast.instantiate(materialize_broadcasted(ws, A(M)))
    A_, rem = structured_from_ws(ws, A(M))
    M_ = Base.Broadcast.materialize!(A_, bcd)
    return M_, rem
end
function materialize_with(::ShouldNotBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix, ::Nothing)
    A_, rem = structured_from_ws(ws, A(M))
    strategy = materialize_strategy(M)
    if strategy == :mat
        materialize_with(rem, A(M), A_)
    elseif strategy == :x_mul
        x_i, rem_ = take_ws(rem, size(A(M), 1))
        y, rem_ = take_ws(rem_, size(A(M), 2))
        _fillzero!(x_i)
        for i in 1:size(A(M), 1)
            x_i[i] = one(eltype(A(M)))
            mul_with!(rem_, y, transpose(A(M)), x_i, true, false)
            copyto!(@view(A_[i, :]), y)
            x_i[i] = zero(eltype(M))
        end
    elseif strategy == :mul_x
        x_i, rem_ = take_ws(rem, size(A(M), 2))
        _fillzero!(x_i)
        for i in 1:size(A(M), 2)
            x_i[i] = one(eltype(A(M)))
            mul_with!(rem_, @view(A_[:, i]), A(M), x_i, true, false)
            x_i[i] = zero(eltype(A(M)))
        end
    end
    return A_, rem
end

materialize_with(ws::Workspace, M::MaterializedMatrix, skeleton::AbstractMatrix) = materialize_with(broadcast_materialize(A(M)), ws, M, skeleton)
function materialize_with(::ShouldBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix, skeleton::AbstractMatrix)
    @warn "Materializing a materialized matrix..."
    bcd = Base.Broadcast.instantiate(materialize_broadcasted(ws, A(M)))
    Base.Broadcast.materialize!(skeleton, bcd)
    return skeleton, ws
end
function materialize_with(::ShouldNotBroadcastMaterialize, ws::Workspace, M::MaterializedMatrix, skeleton::AbstractMatrix)
    @warn "Materializing a materialized matrix..."
    A_, rem = structured_from_ws(ws, A(M))
    materialize_with(rem, A(M), A_)
    skeleton .= A_
    return skeleton, ws
end

# simply pass through the broadcast materialize calls.. 
broadcast_materialize(S::MaterializedMatrix) = broadcast_materialize(A(S))
materialize_broadcasted(ws::Workspace, S::MaterializedMatrix) = materialize_broadcasted(ws, A(S))

function materialize_strategy(M::MaterializedMatrix)
    # this is a crude heuristic! (if it is "cheaper" to multiply with the matrix than to materialize, then materialize by multiplication)
    mat = workspace_size(required_workspace(materialize_with, A(M)))
    mA, nA = max_size(A(M))
    mul = min(mA, nA) * workspace_size(required_workspace(mul_with!, A(M)))
    if mat < mul
        return :mat
    elseif nA <= mA
        return :mul_x
    else
        return :x_mul
    end
end

function required_workspace(::typeof(materialize_with), M::MaterializedMatrix)
    strategy = materialize_strategy(M)
    if strategy == :mat
        return required_workspace(structured_from_ws, A(M)) + required_workspace(materialize_with, A(M))
    elseif strategy == :x_mul
        return required_workspace(structured_from_ws, A(M)) + required_workspace(mul_with!, A(M)) + max_size(A(M))[1] + max_size(A(M))[2] # because we cannot directly write into the memory for A
    else # strategy == :mul_x
        return required_workspace(structured_from_ws, A(M)) + required_workspace(mul_with!, A(M)) + max_size(A(M))[2]
    end
    # @show required_workspace(materialize_with, A(M)), required_workspace(mul_with!, A(M)), size(A(M))
    return 
end
materialize(M::Union{MaterializedMatrix{T}, Transpose{T, <:MaterializedMatrix{T}}}) where T = M

##### CachedMatrix
function materialize_with(ws::Workspace, C::CachedMatrix, ::Nothing)
    if !haskey(ws.cache, lazy_objectid(A(C))) @warn "$(typeof(A(C))) key not found" end
    valid, memory = ws.cache[lazy_objectid(A(C))]
    C_cached = structured_mat_view(memory, C)
    if !valid[]
        C_cached, _ = materialize_with(ws, A(C), C_cached)
        valid[] = true
    end
    return C_cached, ws
end
materialize_with(ws::Workspace, C::CachedMatrix, skeleton::AbstractMatrix) = error("TODO")

# simply pass through the broadcast materialize calls.. 
broadcast_materialize(::CachedMatrix) = ShouldBroadcastMaterialize()
materialize_broadcasted(ws::Workspace, C::CachedMatrix) = first(materialize_with(ws, C, nothing))

function required_workspace(::typeof(materialize_with), C::CachedMatrix)
    cache_size = required_workspace(structured_from_ws, A(C))
    return WorkspaceSize(0, Dict(lazy_objectid(C) => cache_size)) + required_workspace(materialize_with, A(C))
end

# TODO: does this make sense?
materialize(M::Union{CachedMatrix{T}, Transpose{T, <:CachedMatrix{T}}}) where T = M
