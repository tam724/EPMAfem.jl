
## convert to Diagonal type if A is diagonal
diag_if_diag(A::Diagonal) = A
function diag_if_diag(A::AbstractMatrix)
    if isdiag(A)
        return Diagonal(Vector(diag(A)))
    else
        return A
    end
end

function dot_buf(x::AbstractVector, A::AbstractMatrix, y::AbstractVector, buf::AbstractVector)
    mul!(transpose(buf), transpose(x), A)
    return dot(y, buf)
end

# this feels hacky and should be done in CUDA.jl, avoids NaN * false = NaN (sould be 0.0)
# see: https://github.com/JuliaGPU/CUDA.jl/issues/2607
function my_rmul!(A::AbstractArray{T}, β::Bool) where T
    if !β fill!(A, zero(T)) end
    # if β == true, we do nothing
    return A
end
my_rmul!(A::AbstractArray, β::Number) = rmul!(A, β)
