function broadcast_materialize() end
function mat_with_materialize() end
function mul_materialize() end
function cache() end

const BMaterializedMatrix{T} = LazyOpMatrix{T, typeof(broadcast_materialize), <:Tuple{<:AbstractMatrix{T}}}
const MMaterializedMatrix{T} = LazyOpMatrix{T, typeof(mat_with_materialize), <:Tuple{<:AbstractMatrix{T}}}
const XMaterializedMatrix{T} = LazyOpMatrix{T, typeof(mul_materialize), <:Tuple{<:AbstractMatrix{T}}}
const MaterializedMatrix{T} = Union{BMaterializedMatrix{T}, MMaterializedMatrix{T}, XMaterializedMatrix{T}}

const CachedMatrix{T} = LazyOpMatrix{T, typeof(cache), <:Tuple{<:MaterializedMatrix{T}}}
const MaterializedOrCachedMatrix{T} = Union{MaterializedMatrix{T}, CachedMatrix{T}}

@inline A(M::MaterializedMatrix) = only(M.args)
@inline A(C::CachedMatrix) = A(M(C))
@inline M(C::CachedMatrix) = only(C.args)

Base.size(M::MaterializedOrCachedMatrix) = size(A(M))
max_size(M::MaterializedOrCachedMatrix) = max_size(A(M))
lazy_getindex(M::MaterializedOrCachedMatrix{T}, idx::Vararg{<:Integer}) where T = lazy_getindex(A(M), idx...)
@inline isdiagonal(M::MaterializedOrCachedMatrix) = isdiagonal(A(M))
lazy_objectid(M::MaterializedOrCachedMatrix) = lazy_objectid(A(M))

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, M::MaterializedOrCachedMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    materialized_M, _ = materialize_with(ws, M)
    mul_with!(nothing, Y, materialized_M, X, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractVecOrMat, Mt::Transpose{T, <:MaterializedOrCachedMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T
    materialized_Mt, _ = materialize_with(ws, parent(Mt))
    mul_with!(nothing, Y, transpose(materialized_Mt), X, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, M::MaterializedOrCachedMatrix, α::Number, β::Number)
    materialized_M, _ = materialize_with(ws, M)
    mul_with!(nothing, Y, X, materialized_M, α, β)
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, Mt::Transpose{T, <:MaterializedOrCachedMatrix{T}}, α::Number, β::Number) where T
    materialized_Mt, _ = materialize_with(ws, parent(Mt))
    mul_with!(nothing, Y, X, transpose(materialized_Mt), α, β)
end
required_workspace(::typeof(mul_with!), M::MaterializedOrCachedMatrix, n, cache_notifier) = required_workspace(materialize_with, M, cache_notifier)

# the materialize_with is different for MaterializeMatrix and CachedMatrix though...
##### MaterializedMatrix
function materialize_with(ws::Workspace, M::MaterializedMatrix)
    A_, rem = structured_from_ws(ws, A(M))
    return materialize_with(rem, M, A_; warn=false)
end
function materialize_with(ws::Workspace, M::BMaterializedMatrix, skeleton::AbstractMatrix; warn=true)
    if warn @warn "Materializing a materialized matrix" end
    bcd_, _ = materialize_broadcasted(ws, A(M))
    bcd = Base.Broadcast.instantiate(bcd_)
    M_ = Base.Broadcast.materialize!(skeleton, bcd)
    return M_, ws
end
function materialize_with(ws::Workspace, M::BMaterializedMatrix, skeleton::AbstractMatrix, α::Number, β::Number; warn=true)
    if warn @warn "Materializing a materialized matrix" end
    bcd_, _ = materialize_broadcasted(ws, A(M))
    bcd_αβ = Base.Broadcast.broadcasted(+, Base.Broadcast.broadcasted(*, α, bcd_), Base.Broadcast.broadcasted(*, β, skeleton))
    bcd = Base.Broadcast.instantiate(bcd_αβ)
    M_ = Base.Broadcast.materialize!(skeleton, bcd)
    return M_, ws
end

function materialize_with(ws::Workspace, M::MMaterializedMatrix, skeleton::AbstractMatrix; warn=true)
    if warn @warn "Materializing a materialized matrix" end
    return materialize_with(ws, A(M), skeleton)
end
function materialize_with(ws::Workspace, M::MMaterializedMatrix, skeleton::AbstractMatrix, α::Number, β::Number; warn=true)
    if warn @warn "Materializing a materialized matrix" end
    return materialize_with(ws, A(M), skeleton, α, β)
end

function materialize_with(ws::Workspace, M::XMaterializedMatrix, skeleton::AbstractMatrix; warn=true)
    if warn @warn "Materializing a materialized matrix" end
    mM, nM = size(M)
    if mM < nM
        x_i, rem_ = take_ws(ws, size(A(M), 1))
        y, rem_ = take_ws(rem_, size(A(M), 2))
        _fillzero!(x_i)
        for i in 1:size(A(M), 1)
            x_i[i:i] .= one(eltype(A(M)))
            mul_with!(rem_, y, transpose(A(M)), x_i, true, false)
            copyto!(@view(skeleton[i, :]), y)
            x_i[i:i] .= zero(eltype(M))
        end
    else #mM >= nM
        x_i, rem_ = take_ws(ws, size(A(M), 2))
        _fillzero!(x_i)
        for i in 1:size(A(M), 2)
            x_i[i:i] .= one(eltype(A(M)))
            mul_with!(rem_, @view(skeleton[:, i]), A(M), x_i, true, false)
            x_i[i:i] .= zero(eltype(A(M)))
        end
    end
    return skeleton, ws
end
function materialize_with(ws::Workspace, M::XMaterializedMatrix, skeleton::AbstractMatrix, α::Number, β::Number; warn=true)
    if warn @warn "Materializing a materialized matrix" end
    mM, nM = size(M)
    if mM < nM
        x_i, rem_ = take_ws(ws, size(A(M), 1))
        y, rem_ = take_ws(rem_, size(A(M), 2))
        _fillzero!(x_i)
        for i in 1:size(A(M), 1)
            x_i[i:i] .= one(eltype(A(M)))
            mul_with!(rem_, y, transpose(A(M)), x_i, true, false)
            @view(skeleton[i, :]) .= α.*y .+ β.*@view(skeleton[i, :])
            x_i[i:i] .= zero(eltype(M))
        end
    else #mM >= nM
        x_i, rem_ = take_ws(ws, size(A(M), 2))
        _fillzero!(x_i)
        for i in 1:size(A(M), 2)
            x_i[i:i] .= one(eltype(A(M)))
            mul_with!(rem_, @view(skeleton[:, i]), A(M), x_i, α, β)
            x_i[i:i] .= zero(eltype(A(M)))
        end
    end
    return skeleton, ws
end

function required_workspace(::typeof(materialize_with), M::MaterializedMatrix, cache_notifier)
    return required_workspace(structured_from_ws, A(M)) + _required_workspace(materialize_with, M, cache_notifier)
end

function _required_workspace(::typeof(materialize_with), M::BMaterializedMatrix, cache_notifier)
    return required_workspace(materialize_with, A(M), cache_notifier)
end
function _required_workspace(::typeof(materialize_with), M::MMaterializedMatrix, cache_notifier)
    return required_workspace(materialize_with, A(M), cache_notifier)
end
function _required_workspace(::typeof(materialize_with), M::XMaterializedMatrix, cache_notifier)
    # TODO: generalize n = 1 to size(M) by multyplying with I
    return required_workspace(mul_with!, A(M), 1, cache_notifier) + sum(max_size(A(M))) # because we cannot directly write into the memory for A
end

##### CachedMatrix
function materialize_with(ws::Workspace, C::CachedMatrix)
    if !haskey(ws.cache.cache, lazy_objectid(A(C))) @warn "$(typeof(A(C))) key not found" end
    valid, memory = ws.cache.cache[lazy_objectid(A(C))]
    C_cached = structured_mat_view(memory, C)
    if !valid[]
        CUDA.NVTX.@range "precompute cache" begin
            materialize_with(ws, M(C), C_cached; warn=false)
        end
        valid[] = true
    end
    return C_cached, ws
end
function materialize_with(ws::Workspace, C::CachedMatrix, skeleton::AbstractMatrix; warn=true)
    if warn @warn "Materializing a cached matrix" end
    A_, _ = materialize_with(ws, C)
    skeleton .= A_
    return skeleton, ws
end

function materialize_with(ws::Workspace, C::CachedMatrix, skeleton::AbstractMatrix, α::Number, β::Number; warn=true)
    if warn @warn "Materializing a cached matrix" end
    A_, _ = materialize_with(ws, C)
    skeleton .= α.*A_ .+ β.*skeleton
    return skeleton, ws
end

function required_workspace(::typeof(materialize_with), C::CachedMatrix, cache_notifier)
    cache_size = required_workspace(structured_from_ws, A(C))
    valid = Ref(false)
    cache_ws = WorkspaceSize(0, CacheStructure(
        Dict(lazy_objectid(C) => (valid, cache_size)),
        nothing,
    ))
    return cache_ws + _required_workspace(materialize_with, M(C), (cache_notifier..., valid))
end

# broadcast materialize logic
# generally opt out of broadcasting materialize..
should_broadcast_materialize(::AbstractLazyMatrixOrTranspose) = false

should_broadcast_materialize(::AbstractMatrix) = true
materialize_broadcasted(ws::Workspace, A::AbstractMatrix) = A, ws

should_broadcast_materialize(S::ScaleMatrix) = should_broadcast_materialize(A(S))
function materialize_broadcasted(ws::Workspace, S::ScaleMatrix)
    A_bcd, rem = materialize_broadcasted(ws, A(S))
    return Base.Broadcast.broadcasted(*, a(S), A_bcd), rem
end

should_broadcast_materialize(::LazyResizeMatrix) = true
materialize_broadcasted(ws::Workspace, R::LazyResizeMatrix) = _reshape_view(R), ws

should_broadcast_materialize(S::SumMatrix) = all(should_broadcast_materialize, As(S))
materialize_broadcasted(ws::Workspace, S::SumMatrix) = Base.Broadcast.broadcasted(+, first.(materialize_broadcasted.(Ref(ws), As(S)))...), ws

should_broadcast_materialize(::CachedMatrix) = true
materialize_broadcasted(ws::Workspace, C::CachedMatrix) = first(materialize_with(ws, C)), ws

should_broadcast_materialize(M::MaterializedMatrix) = should_broadcast_materialize(A(M))
function materialize_broadcasted(ws::Workspace, M::MaterializedMatrix)
    @assert should_broadcast_materialize(A(M))
    return materialize_broadcasted(ws, A(M))
end
