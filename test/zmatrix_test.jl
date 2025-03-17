module ZMatrixTest

using Test
using LinearAlgebra
using Random

using EPMAfem

# test for ZMatrix2 (and its transpose)
function test_matmul()
    Z = EPMAfem.ZMatrix2{Float64}([rand(10, 11) for i in 1:2], rand(12, 13), [[rand(12, 13) for j in 1:3] for i in 1:2], rand(2), [rand(3) for i in 1:2], 10, 11, 12, 13, 2, 3, zeros(10, 12), zeros(12, 13))
    EPMAfem.size_check(Z)
    A = sum(kron(transpose(γ_i * Z.B + sum((δ_ij * C_ij for (δ_ij, C_ij) in zip(δ_i, C_i)))), A_i) for (γ_i, δ_i, C_i, A_i) in zip(Z.γ, Z.δ, Z.C, Z.A))

    x = rand(11*12)
    y = zeros(10*13)
    y2 = zeros(10*13)

    mul!(y, Z, x, true, false)
    mul!(y2, A, x, true, false)
    @test y ≈ y2

    x = rand(10*13)
    y = zeros(11*12)
    y2 = zeros(11*12)

    ZT = transpose(Z);
    AT = transpose(A);
    mul!(y, ZT, x, true, false)
    y2 = AT*x
    @test y ≈ y2
end

test_matmul()

function test_diagonal_assembly()
    # test ZMatrix2 (diagonal assembly)
    Z = EPMAfem.ZMatrix2{Float64}([Diagonal(rand(10)) for i in 1:2], Diagonal(rand(11)), [[Diagonal(rand(11)) for j in 1:3] for i in 1:2], rand(2), [rand(3) for i in 1:2], 10, 10, 11, 11, 2, 3, zeros(10, 11), Diagonal(zeros(11)))
    EPMAfem.size_check(Z)

    C_ass = Diagonal(zeros(10*11))

    α = randn()

    EPMAfem.assemble_diag(C_ass, Z, α)

    x = rand(10*11)
    y = zeros(10*11)
    y2 = zeros(10*11)

    mul!(y, Z, x, α, false)
    mul!(y2, C_ass, x, true, false)

    @test y ≈ y2
end

test_diagonal_assembly()
end
