module TwoDiagMatrixTest

using Test
using LinearAlgebra
using EPMAfem

A = EPMAfem.TwoDiagonalMatrix(rand(), rand(), 3)

@testset "$(β), $(op)" for β in [rand(), false], op in [identity, transpose]
    A_op = op(A)

    X = rand(size(A_op, 2), 5)
    Y1 = rand(size(A_op, 1), 5)
    Y2 = copy(Y1)

    α = rand()
    mul!(Y1, A_op, X, α, β)
    mul!(Y2, Matrix(A_op), X, α, β)

    @test Y1 ≈ Y2

    X = rand(5, size(A_op, 1))
    Y1 = rand(5, size(A_op, 2))
    Y2 = copy(Y1)

    mul!(Y1, X, A_op, α, β)
    mul!(Y2, X, Matrix(A_op), α, β)

    @test Y1 ≈ Y2
end

end
