# this feels hacky and should be done in CUDA.jl, e.g. 

function my_rmul!(A::AbstractArray{T}, β::Bool) where T
    if !β fill!(A, zero(T)) end
    # if β == true, we do nothing
    return A
end
my_rmul!(A::AbstractArray, β::Number) = rmul!(A, β)
