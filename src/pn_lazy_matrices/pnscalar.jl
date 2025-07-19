mutable struct LazyScalar{T}
    val::T
end

lazy_objectid(s::LazyScalar) = objectid(s)

Base.getindex(s::LazyScalar) = s.val
Base.eltype(::LazyScalar{T}) where T = T

function quiet_setindex!(s::LazyScalar, x)
    s.val = x
end

function Base.setindex!(s::LazyScalar, x, ws::Workspace)
    quiet_setindex!(s, x)
    notify_cache(ws, s)
end

function register_cache_notifier(s::LazyScalar, cache_notifier)
    return WorkspaceSize(0, CacheStructure(nothing, Dict(lazy_objectid(s)=>cache_notifier)))
end

@concrete struct NotSoLazyScalar{T}
    scalar::LazyScalar{T}
    ws
end

Base.getindex(s::NotSoLazyScalar) = getindex(s.scalar)
Base.eltype(s::NotSoLazyScalar) = eltype(s.scalar)
Base.setindex!(s::NotSoLazyScalar, x) = setindex!(s.scalar, x, s.ws)
