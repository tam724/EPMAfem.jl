abstract type AbstractLazyScalar{T} end

mutable struct LazyScalar{T} <: AbstractLazyScalar{T}
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

@concrete terse struct LazyOpScalar{T} <: AbstractLazyScalar{T}
    op
    args
end

function lazy(func, args::Vararg{<:AbstractLazyScalar})
    T = promote_type(eltype.(args)...)
    return LazyOpScalar{T}(func, args)
end

lazy_objectid(s::LazyOpScalar) = objectid(s)
Base.eltype(::LazyOpScalar{T}) where T = T

function register_cache_notifier(s::LazyOpScalar, cache_notifier)
    return sum(register_cache_notifier.(s.args, Ref(cache_notifier)))
end

const SumScalar{T} = LazyOpScalar{T, typeof(+), <:Tuple{Vararg{<:AbstractLazyScalar{T}}}}
Base.getindex(s::SumScalar) = sum(getindex(a) for a in s.args)
Base.:+(a::AbstractLazyScalar, b::AbstractLazyScalar) = lazy(+, a, b)
Base.:+(a::AbstractLazyScalar, b::Number) = lazy(+, a, lazy(b))
Base.:+(a::Number, b::AbstractLazyScalar) = lazy(+, lazy(a), b)
lazy(::typeof(+), a::LazyScalar, b::SumScalar) = lazy(+, a, b.args...)
lazy(::typeof(+), a::SumScalar, b::LazyScalar) = lazy(+, a.args..., b)
lazy(::typeof(+), a::SumScalar, b::SumScalar) = lazy(+, a.args..., b.args...)

const ProdScalar{T} = LazyOpScalar{T, typeof(*), <:Tuple{Vararg{<:AbstractLazyScalar{T}}}}
Base.getindex(s::ProdScalar) = prod(getindex(a) for a in s.args)
Base.:*(a::AbstractLazyScalar, b::AbstractLazyScalar) = lazy(*, a, b)
Base.:*(a::AbstractLazyScalar, b::Number) = lazy(*, a, lazy(b))
Base.:*(a::Number, b::AbstractLazyScalar) = lazy(*, lazy(a), b)
Base.:*(a::LazyScalar, b::ProdScalar) = lazy(*, a, b.args...)
Base.:*(a::ProdScalar, b::LazyScalar) = lazy(*, a.args..., b)
Base.:*(a::ProdScalar, b::ProdScalar) = lazy(*, a.args..., b.args...)

Base.:/(a::AbstractLazyScalar, b::Number) = lazy(*, a, lazy(one(b)/b))


@concrete struct NotSoLazyScalar{T}
    scalar::LazyScalar{T}
    ws
end

Base.getindex(s::NotSoLazyScalar) = getindex(s.scalar)
Base.eltype(s::NotSoLazyScalar) = eltype(s.scalar)
Base.setindex!(s::NotSoLazyScalar, x) = setindex!(s.scalar, x, s.ws)
