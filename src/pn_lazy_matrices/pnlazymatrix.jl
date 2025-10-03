# a wrapper around "concrete" matrices (TODO: combine with LazyResizeMatrix)
struct LazyMatrix{T, AT<:AbstractMatrix{T}} <: AbstractLazyMatrix{T}
    A::AT
end

A(L::LazyMatrix) = L.A

Base.size(L::LazyMatrix) = size(A(L))
max_size(L::LazyMatrix) = size(L)
lazy_getindex(L::LazyMatrix, i::Int, j::Int) = getindex(A(L), i, j)

isdiagonal(::LazyMatrix{<:Number, <:Diagonal}) = true
isdiagonal(::LazyMatrix{<:Number, <:AbstractArray}) = false

_label(A::AbstractMatrix) = "$(size(A))Matrix"
_label(A::CUDA.CUSPARSE.SparseArrays.AbstractSparseMatrix) = "$(size(A))Sparse[nnz=$(CUDA.CUSPARSE.nnz(A))]"
_label(A::Diagonal) = "$(size(A))Diagonal"

function mul_with!(::Workspace, Y::AbstractVecOrMat, L::LazyMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    CUDA.NVTX.@range "mul $(_label(A(L)))" begin
        mul_with!(nothing, Y, A(L), X, α, β)
    end
end
function mul_with!(::Workspace, Y::AbstractVecOrMat, Lt::Transpose{T, <:LazyMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T
    CUDA.NVTX.@range "mul trans $(_label(A(parent(Lt))))" begin
        mul_with!(nothing, Y, transpose(A(parent(Lt))), X, α, β)
    end
end
function mul_with!(::Workspace, Y::AbstractVecOrMat, X::AbstractVecOrMat, L::LazyMatrix, α::Number, β::Number)
    CUDA.NVTX.@range "mul $(_label(A(L)))" begin
        mul_with!(nothing, Y, X, A(L), α, β)    
    end
end        
function mul_with!(::Workspace, Y::AbstractVecOrMat, X::AbstractVecOrMat, Lt::Transpose{T, <:LazyMatrix{T}}, α::Number, β::Number) where T
    CUDA.NVTX.@range "mul trans $(_label(A(parent(Lt))))" begin
        mul_with!(nothing, Y, X, transpose(A(parent(Lt))), α, β)
    end
end    

required_workspace(::typeof(mul_with!), ::LazyMatrix, n, cache_notifier) = 0 # TODO: add cache Notifier

materialize_with(ws::Workspace, L::LazyMatrix) = A(L), ws
function materialize_with(ws::Workspace, L::LazyMatrix, skeleton::AbstractMatrix)
    @warn "Materializing a Matrix into skeleton"
    skeleton .= A(L)
    return skeleton, ws
end
function materialize_with(ws::Workspace, L::LazyMatrix, skeleton::AbstractMatrix, α::Number, β::Number)
    @warn "Did you consider broadcasting?"
    skeleton .= α.*A(L) .+ β.*skeleton
    return skeleton, ws
end
required_workspace(::typeof(materialize_with), ::LazyMatrix, cache_notifier) = 0 # TODO: add cache Notifier
