module RowSparseMatricesTest
using Test
using EPMAfem.RowSparseMatrices
using LinearAlgebra

A = RowSparseMatrix([1, 3, 6], 10, rand(3, 5))
A_ = Matrix(A)

function test_multiplication(A, A_)
    y = rand(size(A, 1))
    y_ = copy(y)
    x = rand(size(A, 2))
    α = rand()
    β = rand()
    mul!(y, A, x, α, β)
    mul!(y_, A_, x, α, β)
    @test y ≈ y_

    Y = rand(size(A, 1), 4)
    Y_ = copy(Y)
    X = rand(size(A, 2), 4)
    mul!(Y, A, X, α, β)
    mul!(Y_, A_, X, α, β)
    @test Y ≈ Y_

    Y = rand(4, size(A, 2))
    Y_ = copy(Y)
    X = rand(4, size(A, 1))
    mul!(Y, X, A, α, β)
    mul!(Y_, X, A_, α, β)
    @test Y ≈ Y_
end

test_multiplication(A, A_)
test_multiplication(transpose(A), transpose(A_))

svd_ = svd(A)
@test svd_.U * Diagonal(svd_.S) * svd_.Vt ≈ A

svd_ = svd(transpose(A))
@test svd_.U * Diagonal(svd_.S) * svd_.Vt ≈ transpose(A)


end
