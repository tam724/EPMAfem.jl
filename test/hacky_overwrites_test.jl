module HackyOverwritesTest

using CUDA
using SparseArrays
using EPMAfem
using Test
using LinearAlgebra

for p in 0:0.1:1.0
    for n in [0, 1, 2, 10, 20, 50]
        for m in [0, 1, 2, 10, 20, 50]
            for k in [-5, -2, -1, 0, 1, 2, 5]
                if abs(k) > n || abs(k) > m continue end
                A = sprand(m, n, p)
                A_csc = cu(A)
                A_csr = CUDA.CUSPARSE.CuSparseMatrixCSR(A_csc)
                @test collect(diag(A, k)) ≈ collect(diag(A_csc, k))
                @test collect(diag(A, k)) ≈ collect(diag(A_csr, k))
                @test collect(diag(transpose(A), k)) ≈ collect(diag(transpose(A_csc), k))
                @test collect(diag(transpose(A), k)) ≈ collect(diag(transpose(A_csr), k))
            end
        end
    end
end

end
