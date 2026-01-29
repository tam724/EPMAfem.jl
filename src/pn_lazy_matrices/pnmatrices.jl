# utils
# generally useful for size checking
only_unique(elm...) = only_unique(elm)
function only_unique(iter)
    a, rest... = iter
    if !all(x -> x == a, rest)  error("Collection has multiple elements, must contain exactly 1 element") end
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

# indirection to catch mul!(::Transpose, ...)
_mul!(C::Transpose, A, B, α, β) = mul!(parent(C), transpose(B), transpose(A), α, β)
_mul!(C, A, B, α, β) = mul!(C, A, B, α, β)

function mul_with!(::Nothing, Y::AbstractVecOrMat, A::AbstractMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    # CUDA.NVTX.@range "mul!(.., :$(typeof(A)), $(typeof(X)))" begin
    try
        _mul!(Y, A, X, α, β)
    catch e
        @show "mul! error with $(typeof(Y)), $(typeof(A)), $(typeof(X))"
        throw(e)
    end
    # end 
end

# abstract implementations
Base.getindex(L::AbstractLazyMatrix{T}, I::CartesianIndex) where T = getindex(L, I.I...)
Base.getindex(L::AbstractLazyMatrix{T}, i::Int, j::Int) where T = CUDA.@allowscalar lazy_getindex(L, i, j)

lazy_objectid(L::AbstractLazyMatrix) = objectid(L) # we give each matrix an objectid for caching
lazy_objectid(::AbstractMatrix) = error("oh ohh.. ") 

max_size(A::AbstractLazyMatrix, n::Integer) = max_size(A)[n]
LinearAlgebra.transpose(A::AbstractLazyMatrix) = isdiagonal(A) ? A : Transpose(A)
LinearAlgebra.adjoint(A::AbstractLazyMatrix{<:Real}) = transpose(A)
required_workspace(::typeof(mul_with!), L::AbstractLazyMatrix, cache_notifier) = required_workspace(mul_with!, L, 1, cache_notifier) # TODO: remove usage, deprecated

"""
    LazyOpMatrix{T, ...}

The core matrix expression type. Stores the computational graph for matrix-vector and matrix-matrix multiplication in the type. (is this good? TBD..)

Fields:
 - op:OP (the operation e.g. Base.:+ for the sum of two matrices or Base.kron for the kronecker product)
 - args::ARGS (the arguments of the op, a tuple of one or more AbstractLazyMatrix'es)
 - kwargs::KWARG (optional keyword arguments to op)

Construct a LazyOpMatrices via 'lazy()' (e.g. 'lazy(+, A, B)'). Or via the overloads of the high level operators '+'.
"""
struct LazyOpMatrix{T<:Number, OP, ARGS<:Tuple, KWARGS<:NamedTuple} <: AbstractLazyMatrix{T}
    op::OP
    args::ARGS
    kwargs::KWARGS
end

const _NO_KWARGS = @NamedTuple{}

# omit printing the inner types.. print only the eltype T and the op OP
function Base.show(io::IO, ::Type{<:LazyOpMatrix{_T, _OP}}) where {_T, _OP}
    print(io, "LazyOpMatrix{$_T, $_OP, (...)}")
end

# "constructor" 
function lazy(op, args...; kwargs...)
	T = promote_type(eltype.(args)...)
    _kwargs = (; kwargs...)
	return LazyOpMatrix{T, typeof(op), typeof(args), typeof(_kwargs)}(op, args, _kwargs)
end

# eagerly convert a vector of arguments to a tuple (not type stable!)
lazy(op, arg::AbstractVector) = lazy(op, arg...)

lazy(::typeof(+), A::AbstractLazyMatrixOrTranspose) = A
lazy(::typeof(*), A::AbstractLazyMatrixOrTranspose) = A

