module BlockedMatricesTest

using Test
using LinearAlgebra
using Random

import EPMAfem.BlockedMatrices as BM

rand_without_repl(vec) = vec[randperm(length(vec))]
rand_without_repl(vec, n) = rand_without_repl(vec)[1:n]

function random_blocked_matrix(m, n, n_blocks)
    i = [0, m, rand_without_repl(2:m-1, n_blocks-1)...] |> sort
    j = [0, n, rand_without_repl(2:n-1, n_blocks-1)...] |> sort

    is = rand_without_repl([i[k]+1:i[k+1] for k in 1:length(i)-1])
    js = rand_without_repl([j[k]+1:j[k+1] for k in 1:length(j)-1])
    # js = [j_:j_+1 for j_ in j]

    indices = tuple(((i, j) for (i, j) in zip(is, js))...)
    return BM.blocked_from_mat(rand(m, n), indices, false)
end

function test_blocked_matrix_construction(m, n, n_blocks)
    A = random_blocked_matrix(m, n, n_blocks)
    A_full = A |> collect

    @test A ≈ A_full

    A_new = BM.blocked_from_mat(A_full, A.indices)
    
    @test A_new ≈ A
    @test A_new ≈ A_full

    AT = transpose(A)
    AT_full = transpose(A_full)

    @test AT |> collect ≈ AT_full
end

test_blocked_matrix_construction(20, 30, 5)
test_blocked_matrix_construction(100, 100, 40)

function test_blocked_matrix_matmul(m, n, k, n_blocks)
    A = random_blocked_matrix(m, n, n_blocks)
    B = rand(k, m)
    C1 = zeros(k, n)
    C2 = zeros(k, n)

    mul!(C1, B, A, true, false)
    mul!(C2, B, A |> collect, true, false)

    @test C1 ≈ C2

    # also test transpose
    AT = transpose(A)
    B = rand(k, n)
    C1 = zeros(k, m)
    C2 = zeros(k, m)

    mul!(C1, B, AT, true, false)
    mul!(C2, B, AT |> collect, true, false)

    @test C1 ≈ C2
end

test_blocked_matrix_matmul(20, 30, 40, 5)
test_blocked_matrix_matmul(100, 100, 100, 40)

function test_blocked_matrix_matmul_noncontig(m, n, k, n_blocks)
    A = random_blocked_matrix(m, n, n_blocks)
    B = rand(n, k)
    C1 = zeros(m, k)
    C2 = zeros(m, k)

    mul!(C1, B, A, true, false)
    mul!(C2, B, A |> collect, true, false)

    @test C1 ≈ C2

    # also test transpose
    AT = transpose(A)
    B = rand(m, k)
    C1 = zeros(n, k)
    C2 = zeros(n, k)

    mul!(C1, B, AT, true, false)
    mul!(C2, B, AT |> collect, true, false)

    @test C1 ≈ C2
end

test_blocked_matrix_matmul(20, 30, 40, 5)
test_blocked_matrix_matmul(100, 100, 100, 40)

end
