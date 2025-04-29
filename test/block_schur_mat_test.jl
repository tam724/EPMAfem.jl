module BlockSchurMatrixTest
using Test
using LinearAlgebra
using Random

using EPMAfem

function block_matmul_test()
    n1 = 10
    n2 = 11
    B = EPMAfem.BlockMat2{Float64}(rand(n1, n1), rand(n1, n2), rand(n2, n2), rand(n1, n1), n1, n2, Ref(rand()), Ref(rand()), Ref(rand()), Ref(rand()), false)
    B_sym = EPMAfem.BlockMat2{Float64}(B.A, B.B, B.C, B.D, B.n1, B.n2, B.γ, B.δ, B.δT, B.Δ, true)
    A = B.Δ[]*[B.A + B.γ[]*B.D B.δ[]*B.B
        B.δT[]*transpose(B.B) B.C]
    A_sym = B.Δ[]*[B.A + B.γ[]*B.D B.δ[]*B.B
        -B.δT[]*transpose(B.B) -B.C]

    x = rand(n1 + n2)
    y = rand(n1 + n2)
    y2 = rand(n1 + n2)

    mul!(y, B, x, true, false)
    mul!(y2, A, x, true, false)
    @test y ≈ y2

    mul!(y, B_sym, x, true, false)
    mul!(y2, A_sym, x, true, false)
    @test y ≈ y2

    BT = transpose(B);
    AT = transpose(A)
    x = rand(n1 + n2)
    y = rand(n1 + n2)
    y2 = rand(n1 + n2)

    mul!(y, BT, x, true, false)
    mul!(y2, AT, x, true, false)
    @test y ≈ y2
end

block_matmul_test()

## test for schur matrix
function schur_matmul_test()
    n1 = 10
    n2 = 11
    A = rand(n1, n1)
    B = rand(n1, n2)
    C = Diagonal(rand(n2))
    D = rand(n1, n1)

    M = EPMAfem.BlockMat2{Float64}(A, B, C, D, n1, n2, Ref(rand()), Ref(rand()), Ref(rand()), Ref(rand()), Ref(true))
    N = EPMAfem.SchurBlockMat2{Float64}(M, Diagonal(zeros(n2)), zeros(n2))

    M_full = EPMAfem.assemble_from_op(M)
    UL = M_full[1:10, 1:10]
    UR = M_full[1:10, 11:21]
    LL = M_full[11:21, 1:10]
    LR = M_full[11:21, 11:21]

    temp = UL .- UR * inv(Diagonal(diag(LR))) * LL

    EPMAfem.update_cache!(N)

    N_full = EPMAfem.assemble_from_op(N)
    @test N_full ≈ temp
end

schur_matmul_test()

function full_schur_solve_test()
    ## test solving of block mat and schur mat
    n1 = 10
    n2 = 11
    A__ = randn(n1, n1)
    A = (A__ + transpose(A__)) / 2
    B = randn(n1, n2)
    C = Diagonal(randn(n2))
    D = Diagonal(randn(n1))

    δ = randn()
    M = EPMAfem.BlockMat2{Float64}(A, B, C, D, n1, n2, Ref(randn()), Ref(δ), Ref(-δ), Ref(0.1), Ref(true))

    solver = EPMAfem.PNKrylovMinresSolver(Vector{Float64}, M)
    direct_solver = EPMAfem.PNDirectSolver(Vector{Float64}, M)
    schur_solver = EPMAfem.PNSchurSolver(Vector{Float64}, M)

    b = rand(n1+n2)
    x = zeros(n1+n2)
    x2 = zeros(n1+n2)
    x3 = zeros(n1+n2)

    EPMAfem.pn_linsolve!(solver, x, M, b)
    EPMAfem.pn_linsolve!(direct_solver, x2, M, b)
    EPMAfem.pn_linsolve!(schur_solver, x3, M, b)

    @test x ≈ x2
    @test x ≈ x3

    A = EPMAfem.assemble_from_op(M)
    
    @test A*x ≈ b
    @test A*x2 ≈ b
    @test A*x3 ≈ b

    ## transpose system
    EPMAfem.pn_linsolve!(solver, x, transpose(M), b)
    EPMAfem.pn_linsolve!(direct_solver, x2, transpose(M), b)
    EPMAfem.pn_linsolve!(schur_solver, x3, transpose(M), b)


    @test x ≈ x2
    @test x ≈ x3

    AT = transpose(A)
    
    @test AT*x ≈ b
    @test AT*x2 ≈ b
    @test AT*x3 ≈ b
end

full_schur_solve_test()

end
