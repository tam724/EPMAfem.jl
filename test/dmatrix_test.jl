module DMatrixTest

using Test
using LinearAlgebra
using Random

using EPMAfem

# test for DMatrix2 (and its transpose)
D = EPMAfem.DMatrix2{Float64}([rand(10, 11) for i in 1:2], [rand(12, 13) for i in 1:2], 10, 11, 12, 13, 2, zeros(10, 12))
A = kron(transpose(D.B[1]), D.A[1]) + kron(transpose(D.B[2]), D.A[2])

x = rand(11*12)
y = zeros(10*13)
y2 = zeros(10*13)

mul!(y, D, x, true, false)
y2 = A*x
@test y ≈ y2 

x = rand(10*13)
y = zeros(11*12)
y2 = zeros(11*12)

DT = transpose(D);
AT = transpose(A);
mul!(y, DT, x, true, false)
y2 = AT*x
@test y ≈ y2 

end
