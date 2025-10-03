# utils
# generally useful for size checking
function only_unique(iter)
    a, rest... = iter
    if !all(x -> x == a, rest)  error("Collection has multiple elements, must containt exactly 1 element") end
    return a
end

abstract type Workspace{VT} end

struct Cache{VT<:AbstractVector}
    cache::Dict{UInt64, Tuple{Base.RefValue{Bool}, VT}}
    cache_notifier::Dict{UInt64, NTuple{N, Base.RefValue{Bool}} where N}
end

struct CacheStructure
    cache::Dict{UInt64, Tuple{Base.RefValue{Bool}, Int64}}
    cache_notifier::Dict{UInt64, NTuple{N, Base.RefValue{Bool}} where N}
end

function CacheStructure(::Nothing, ::Nothing)
    CacheStructure(
        Dict{UInt64, Tuple{Base.RefValue{Bool}, Int64}}(),
        Dict{UInt64, Vector{Base.RefValue{Bool}}}()
    )
end

function CacheStructure(cache, ::Nothing)
    CacheStructure(
        cache,
        Dict{UInt64, Vector{Base.RefValue{Bool}}}()
    )
end

function CacheStructure(::Nothing, cache_notifier)
    CacheStructure(
        Dict{UInt64, Tuple{Base.RefValue{Bool}, Int64}}(),
        cache_notifier
    )
end

@concrete struct PreallWorkspace{VT<:AbstractVector, CT<:Cache} <: Workspace{VT}
    workspace::VT
    cache::CT
end

@concrete struct WorkspaceSize{ST<:Integer}
    workspace::ST
    cache::CacheStructure
end

abstract type AbstractLazyMatrix{T} <: AbstractMatrix{T} end
const AbstractLazyMatrixOrTranspose{T} = Union{<:AbstractLazyMatrix{T}, Transpose{T, <:AbstractLazyMatrix{T}}}
# interface:
function mul_with!() end
function materialize_with() end

function mul_with!(::Nothing, Y::AbstractVecOrMat, A::AbstractMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    # CUDA.NVTX.@range "mul!(.., :$(typeof(A)), $(typeof(X)))" begin
        mul!(Y, A, X, α, β)    
    # end 
end

# abstract implementations
Base.getindex(L::AbstractLazyMatrix{T}, I::CartesianIndex) where T = getindex(L, I.I...)
Base.getindex(L::AbstractLazyMatrix{T}, i::Int, j::Int) where T = CUDA.@allowscalar lazy_getindex(L, i, j)

lazy_objectid(L::AbstractLazyMatrix) = objectid(L) # we give each matrix an objectid for caching
lazy_objectid(::AbstractMatrix) = error("oh ohh.. ") 

max_size(A::AbstractLazyMatrix, n::Integer) = max_size(A)[n]
LinearAlgebra.transpose(A::AbstractLazyMatrix) = isdiagonal(A) ? A : Transpose(A)
required_workspace(::typeof(mul_with!), L::AbstractLazyMatrix, cache_notifier) = required_workspace(mul_with!, L, 1, cache_notifier) # TODO: remove usage, deprecated

@concrete struct LazyOpMatrix{T} <: AbstractLazyMatrix{T}
    op
    args
end

function lazy(func, args...)
	T = promote_type(eltype.(args)...)
	return LazyOpMatrix{T}(func, args)
end

function lazy(func, arg::AbstractVector)
	T = promote_type(eltype.(arg)...)
	return LazyOpMatrix{T}(func, arg)
end
