using GPUArrays
using LinearAlgebra

function LinearAlgebra.mul!(B::GPUArrays.AbstractGPUVecOrMat,
    A::GPUArrays.AbstractGPUVecOrMat,
    D::Diagonal{<:Any, <:GPUArrays.AbstractGPUArray},
    α::Number,
    β::Number)
dd = D.diag
d = length(dd)
m, n = size(A, 1), size(A, 2)
m′, n′ = size(B, 1), size(B, 2)
n == d || throw(DimensionMismatch("left hand side has $n columns but D is $d by $d"))
(m, n) == (m′, n′) || throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
B .= α .* A .* transpose(dd) .+ β .* B

B
end