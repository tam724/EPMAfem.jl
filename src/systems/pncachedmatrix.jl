struct Cached{T, PT} <: AbstractPNMatrix{T}
    P::PT
    valid
end

function Cached{T}(P) where T
    return Cached{T, typeof(P)}(P, Ref(-1))
end

size_string(C::Cached{T}) where T = "$(size(C, 1))x$(size(C, 2)) Cached{$(typeof(C.P).name.wrapper){$(typeof(C.P).parameters[1])}}"
size_string(C::Transpose{T, <:Cached{T}}) where T = "$(size(C, 1))x$(size(C, 2)) transpose(::Cached{$(typeof(parent(C).P).name.wrapper){$(typeof(parent(C).P).parameters[1])})}"
content_string(C::Cached{T}) where T = content_string(C.P)
content_string(C::Transpose{T, <:Cached{T}}) where T = content_string(transpose(parent(C).P))

function cache_with!(ws::WorkspaceCache, outer_cached, C::Cached, α::Number, β::Number)
    inner_cached = get_cached!(ws, C)
    outer_cached .= α .* inner_cached .+ β .* outer_cached
end

Base.size(C::Cached) = size(C.P)
max_size(C::Cached) = max_size(C.P)

function required_workspace_cache(C::Cached)
    P_wsch = required_workspace_cache(C.P)

    # P_mul_with_ws = mul_with_ws(P_wsch) # this will never mul_with, it is cached. therefore mul_with_ws = cache_with_ws, but will only recompute the cache if it is invalid
    P_cache_with_ws = cache_with_ws(P_wsch)

    return WorkspaceCache((prod(size(C.P)), ch(P_wsch)), (mul_with=P_cache_with_ws, cache_with=P_cache_with_ws))
end

function invalidate_cache!(C::Cached)
    invalidate_cache!(C.P)
    C.valid[] = -1
end

function get_cached!(ws::WorkspaceCache, C::Cached)
    cache_and_id, rem = take_ch(ws)
    cache, cache_id = cache_and_id
    cached = reshape(@view(cache[:]), size(C.P))
    if !(C.valid[] == cache_id) # check_cache_invalidity
        cache_with!(rem, cached, C.P, true, false)
        C.valid[] = cache_id
    end
    return cached
end

LinearAlgebra.isdiag(C::Cached) = isdiag(C.P)
LinearAlgebra.issymmetric(C::Cached) = issymmetric(C.P)

function mul_with!(ws::WorkspaceCache, Y::AbstractVecOrMat, C::Cached, X::AbstractVecOrMat, α::Number, β::Number)
    cached = get_cached!(ws, C)
    mul!(Y, cached, X, α, β)
end

function mul_with!(ws::WorkspaceCache, Y::AbstractVecOrMat, X::AbstractVecOrMat, C::Cached, α::Number, β::Number)
    cached = get_cached!(ws, C)
    mul!(Y, X, cached, α, β)
end

function mul_with!(ws::WorkspaceCache, Y::AbstractVecOrMat, Ct::Transpose{T, <:Cached{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T
    cached = get_cached!(ws, parent(Ct))
    mul!(Y, transpose(cached), X, α, β)
end

function mul_with!(ws::WorkspaceCache, Y::AbstractVecOrMat, X::AbstractVecOrMat, Ct::Transpose{T, <:Cached{T}}, α::Number, β::Number) where T
    cached = get_cached!(ws, parent(Ct))
    mul!(Y, X, transpose(cached), α, β)
end

# wrapped
@concrete struct Wrapped{T} <: AbstractPNMatrix{T}
    A
    workspace_cache
end

function Wrapped{T}(A) where T
    ws = allocate_workspace_cache(mul_with!, required_workspace_cache(A))
    return Wrapped{T}(A, ws)
end

size_string(W::Wrapped{T}) where T = "$(size(W, 1))x$(size(W, 2)) Wrapped{$(typeof(W.A).name.wrapper){$(typeof(W.A).parameters[1])}}"
size_string(Wt::Transpose{T, <:Wrapped{T}}) where T = "$(size(Wt, 1))x$(size(Wt, 2)) transpose(::Wrapped{$(typeof(parent(Wt).A).name.wrapper){$(typeof(parent(Wt).A).parameters[1])})}"
content_string(W::Wrapped{T}) where T = content_string(W.A)
content_string(Wt::Transpose{T, <:Wrapped{T}}) where T = content_string(transpose(parent(Wt).A))

Base.size(W::Wrapped) = size(W.A)
max_size(W::Wrapped) = max_size(W.A)

function invalidate_cache!(W::Wrapped)
    invalidate_cache!(W.A)
end

LinearAlgebra.isdiag(W::Wrapped) = isdiag(W.A)
LinearAlgebra.issymmetric(W::Wrapped) = issymmetric(W.A)

function LinearAlgebra.mul!(y::AbstractVector, W::Wrapped, x::AbstractVector, α::Number, β::Number)
    mul_with!(W.workspace_cache, y, W.A, x, α, β)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, Wt::Transpose{T, <:Wrapped{T}}, x::AbstractVector, α::Number, β::Number) where T
    mul_with!(parent(Wt).workspace_cache, y, transpose(parent(Wt).A), x, α, β)
    return y
end
