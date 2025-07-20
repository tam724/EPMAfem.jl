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
Base.getindex(L::AbstractLazyMatrix{T}, I::CartesianIndex) where T = getindex(L, I.I...)
Base.getindex(L::AbstractLazyMatrix{T}, i::Int, j::Int) where T = CUDA.@allowscalar lazy_getindex(L, i, j)
lazy_getindex(Lt::Transpose{T, <:AbstractLazyMatrix{<:T}}, i::Int, j::Int) where T = CUDA.@allowscalar lazy_getindex(parent(Lt), j, i)
# Base.getindex(::AbstractLazyMatrix{T}, args...) where T = error("should be defined")

lazy_objectid(L::AbstractLazyMatrix) = objectid(L) # we give each matrix an objectid for caching
lazy_objectid(::AbstractMatrix) = error("why cache base matrices?") 
lazy_objectid(Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T = lazy_objectid(parent(Lt)) # transpose gets the same as the parent

max_size(A::AbstractLazyMatrix, n::Integer) = max_size(A)[n]
max_size(At::Transpose{T, <:AbstractLazyMatrix}) where T = reverse(max_size(parent(At)))
function max_size(At::Transpose{T, <:AbstractLazyMatrix}, n::Integer) where T
    if n == 1
        return max_size(parent(At), 2)
    elseif n == 2
        return max_size(parent(At), 1)
    else
        error("Dimension $n out of bounds")
    end
end

isdiagonal(::AbstractLazyMatrix) = error("should be defined")
isdiagonal(Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T = isdiagonal(parent(Lt))
# indirect the isdiagonal
isdiagonal(A::AbstractMatrix) = false
isdiagonal(A::Diagonal) = true

LinearAlgebra.transpose(A::AbstractLazyMatrix) = isdiagonal(A) ? A : Transpose(A)
mul_with!(::Workspace, Y::AbstractVecOrMat, A::AbstractLazyMatrix, X::AbstractVecOrMat, ::Number, ::Number) = error("mul_with!(::Workspace, ::$(typeof(Y)), ::$(typeof(A)), ::$(typeof(X)), ...) should be defined")
mul_with!(::Workspace, Y::AbstractMatrix, A::AbstractMatrix, X::AbstractLazyMatrix, ::Number, ::Number) = error("mul_with!(::Workspace, ::$(typeof(Y)), ::$(typeof(A)), ::$(typeof(X)), ...) should be defined")
# this function should return the workspace size that is needed to mul_with! the AbstractLazyMatrix
required_workspace(::typeof(mul_with!), L::AbstractLazyMatrix, cache_notifier) = error("should be defined: $(typeof(L))")
required_workspace(::typeof(mul_with!), Lt::Transpose{T, <:AbstractLazyMatrix{T}}, cache_notifier) where T = required_workspace(mul_with!, parent(Lt), cache_notifier)

# returns the matrix in materialized form (if provided, uses skeleton::AbstractMatrix to materialize AND if provided αβs to skeleton)
materialize_with(::Workspace, ::AbstractLazyMatrix) = error("should be defined")
materialize_with(::Workspace, ::AbstractLazyMatrix, ::AbstractMatrix) = error("should be defined")
materialize_with(::Workspace, ::AbstractLazyMatrix, ::AbstractMatrix, α::Number, β::Number) = error("should be defined")
function materialize_with(ws::Workspace, Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T
    L, rem = materialize_with(ws, parent(Lt))
    return transpose(L), rem
end
function materialize_with(ws::Workspace, Lt::Transpose{T, <:AbstractLazyMatrix{T}}, skeleton::AbstractMatrix) where T
    L, rem = materialize_with(ws, parent(Lt), transpose(skeleton))
    return transpose(L), rem
end
function materialize_with(ws::Workspace, Lt::Transpose{T, <:AbstractLazyMatrix{T}}, skeleton::AbstractMatrix, α::Number, β::Number) where T
    L, rem = materialize_with(ws, parent(Lt), transpose(skeleton), α, β)
    return transpose(L), rem
end

required_workspace(::typeof(materialize_with), ::AbstractLazyMatrix, cache_notifier) = error("should be defined")
required_workspace(::typeof(materialize_with), Lt::Transpose{T, <:AbstractLazyMatrix{T}}, cache_notifier) where T = required_workspace(materialize_with, parent(Lt), cache_notifier)

# default dispatch for AbstractMatrices, they should not be AbstractLazyMatrices though..
function max_size(A::AbstractMatrix)
    if A isa AbstractLazyMatrixOrTranspose
        @warn "max size default for $(typeof(A))"
    end 
    return size(A)
end
function max_size(A::AbstractMatrix, n::Integer)
    if A isa AbstractLazyMatrixOrTranspose
        @warn "max size default for $(typeof(A))"
    end
    return size(A, n)
end

#use nothing here to explcitly say that the lazy path should end
function mul_with!(::Union{Workspace, Nothing}, Y::AbstractVecOrMat, A::AbstractMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    if Y isa AbstractLazyMatrixOrTranspose error("should not happen! mul_with!(.., ::$(typeof(Y)), ::$(typeof(A)), ::$(typeof(X)), ...)") end
    if A isa AbstractLazyMatrixOrTranspose error("should not happen! mul_with!(.., ::$(typeof(Y)), ::$(typeof(A)), ::$(typeof(X)), ...)") end
    if X isa AbstractLazyMatrixOrTranspose error("should not happen! mul_with!(.., ::$(typeof(Y)), ::$(typeof(A)), ::$(typeof(X)), ...)") end
    if A isa Diagonal
        CUDA.NVTX.@range "Diag mul" begin
            mul!(Y, A, X, α, β)            
        end
    else
        CUDA.NVTX.@range "normal mul" begin
            mul!(Y, A, X, α, β)            
        end
    end
end

required_workspace(::typeof(mul_with!), A::AbstractMatrix, cache_notifier) = 0 # TODO: see below

materialize_with(ws::Workspace, A::AbstractMatrix) = A, ws
function materialize_with(ws::Workspace, A::AbstractMatrix, skeleton::AbstractMatrix)
    @warn "Materializing a Matrix into skeleton"
    skeleton .= A
    return skeleton, ws
end
function materialize_with(ws::Workspace, A::AbstractMatrix, skeleton::AbstractMatrix, α::Number, β::Number)
    @warn "Did you consider broadcasting?"
    skeleton .= α.*A .+ β.*skeleton
    return skeleton, ws
end
required_workspace(::typeof(materialize_with), A::AbstractMatrix, cache_notifier) = 0 # TODO: make AbstractMatrix Lazy + add cache Notifier

@concrete struct LazyOpMatrix{T} <: AbstractLazyMatrix{T}
    op
    args
end

# so normal * (mul) works
function Base.similar(L::LazyOpMatrix{T}, TS::Type, (m, n)::Tuple{Int, Int}) where T
    # TODO: HACK! so scalemat works (kinda, only if the internal type is not a lazymatrix,)
    if first(L.args) isa Base.Ref{T} 
        return similar(L.args[2], TS, (m, n))
    else
        return similar(L.args[1], TS, (m, n))
    end
end

function lazy(func, args...)
	T = promote_type(eltype.(args)...)
	return LazyOpMatrix{T}(func, unwrap.(args))
end

# single argument
function lazy(func, arg::AbstractVector)
	T = promote_type(eltype.(arg)...)
	return LazyOpMatrix{T}(func, unwrap.(arg))
end
