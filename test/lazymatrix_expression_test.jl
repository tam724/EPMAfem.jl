using Revise
using EPMAfem.PNLazyMatrices
using Test
using LinearAlgebra
using CUDA
using SparseArrays

function build_matrices(vars, expr)
    lazy_vars = Dict(key=>lazy(val) for (key, val) in vars)

    assignments = [:($key = $(val)) for (key, val) in vars]
    M = eval(Expr(:block, assignments..., expr))

    lazy_assignments = [:($key = $(val)) for (key, val) in lazy_vars]
    lazy_M = eval(Expr(:block, lazy_assignments..., expr))
    return M, lazy_M
end

function test_random_multiplication(AT, M, lazy_M, N = 5)
    unlazy_M = unlazy(lazy_M, size->similar(dense_array_type(AT), size))
    y, lazy_y = similar(dense_array_type(AT), size(M, 1)), similar(dense_array_type(AT), size(M, 1))
    fill!(y, zero(eltype(M))); 
    fill!(lazy_y, zero(eltype(M)));
    x = similar(dense_array_type(AT), size(M, 2))
    for _ in 1:N
        x .= rand(size(x))
        mul!(y, M, x)
        mul!(lazy_y, unlazy_M, x)
        @test y ≈ lazy_y
    end
end

function test_random_multiplication_mat(AT, M, lazy_M, N = 5, n = 5)
    unlazy_M = unlazy(lazy_M, size->similar(dense_array_type(AT), size); n=n)
    Y, lazy_Y = similar(dense_array_type(AT), size(M, 1), n), similar(dense_array_type(AT), size(M, 1), n)
    fill!(Y, zero(eltype(M))); 
    fill!(lazy_Y, zero(eltype(M)));
    X = similar(dense_array_type(AT), size(M, 2), n)
    for _ in 1:N
        X .= rand(size(X))
        mul!(Y, M, X)
        mul!(lazy_Y, unlazy_M, X)
        @test Y ≈ lazy_Y
    end
end

function test_materialization(AT, M, lazy_M)
    if AT <: CUDA.CUSPARSE.CuSparseMatrixCSC || AT <: CuArray return end
    unlazy_M = unlazy(materialize(lazy_M), size->similar(dense_array_type(AT), size))
    materialized_M, _ = materialize_with(unlazy_M.ws, unlazy_M.A)
    @test materialized_M ≈ M
end

function test_diag_materialization(AT, M, lazy_M)
    if AT <: CUDA.CUSPARSE.CuSparseMatrixCSC || AT <: CuArray return end
    if size(M, 1) != size(M, 2) return end

    unlazy_M = unlazy(materialize(lazy_M), size->similar(dense_array_type(AT), size))
    skel = Diagonal(similar(dense_array_type(AT), size(M, 1)))
    diag_M, _ = PNLazyMatrices.materialize_diag_with(unlazy_M.ws, unlazy_M.A.args[1], skel, true, false)
    @test diag(diag_M) ≈ diag(M)
end

function test_expression(AT, vars, expr)
    M, lazy_M = build_matrices(vars, expr)
    @testset "type sanity" begin
        @test M isa AT
        @test lazy_M isa AbstractLazyMatrix
    end
    @testset "matrix size" begin
        @test size(M) == size(lazy_M)
    end

    @testset "multiplication" begin
        test_random_multiplication(AT, M, lazy_M)
        test_random_multiplication_mat(AT, M, lazy_M)
    end

    @testset "transpose multiplication" begin
        test_random_multiplication(AT, transpose(M), transpose(lazy_M))
        test_random_multiplication_mat(AT, transpose(M), transpose(lazy_M))
    end

    @testset "materialization" begin
        test_materialization(AT, M, lazy_M)
    end
    
    @testset "diag_materialization" begin
        test_diag_materialization(AT, M, lazy_M)
    end
end

dense_array_type(::Type{Array{T}}) where T = Array{T}
dense_array_type(::Type{SparseMatrixCSC{T}})  where T = Array{T}
dense_array_type(::Type{CuArray{T}})  where T = CuArray{T}
dense_array_type(::Type{CUDA.CUSPARSE.CuSparseMatrixCSC{T}}) where T = CuArray{T}

CUDA.CUSPARSE.CuSparseMatrixCSC{T}(M::Matrix) where T = CUDA.CUSPARSE.CuSparseMatrixCSC{T, Int32}(sparse(M))

let
    @testset "$AT{$ST}" for (AT, ST) in [(Array, Float64), (SparseMatrixCSC, Float64), (CuArray, Float32), (CUDA.CUSPARSE.CuSparseMatrixCSC, Float32)]
        @testset "A + B" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST})
            expr = :(A + B)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "A - B" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST})
            expr = :(A - B)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "A * B" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST})
            expr = :(A * B)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*A + b*B" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST},
                :a => rand() |> ST,
                :b => rand() |> ST)
            expr = :(a*A + b*B)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "A + B + A" begin
            vars = Dict(
                :A => [1.0 2.0
                    3.0 4.0] |> AT{ST},
                :B => [1.0 0.0
                    0.0 1.0] |> AT{ST})
            expr = :(A + B + A)
            test_expression(AT{ST}, vars, expr)

            expr = :(A + B - A)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*A + b*A" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST},
                :a => rand() |> ST,
                :b => rand() |> ST)
            expr = :(a*A + b*A)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*A + b*B + c*C + A + B" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST},
                :C => rand(2, 2) |> AT{ST},
                :a => rand() |> ST,
                :b => rand() |> ST,
                :c => rand() |> ST)
            expr = :(a*A + b*B + c*C + A + B)
            test_expression(AT{ST}, vars, expr)

            expr = :(a*A + b*B + (c*C + A + B))
            test_expression(AT{ST}, vars, expr)

            expr = :((a*A + b*B) + (c*C + (A + B)))
            test_expression(AT{ST}, vars, expr)

            expr = :(a*A + ((b*B + c*C) + A) + B)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A, B)" begin
            vars = Dict(
                :A => [1.0 2.0
                    3.0 4.0] |> AT{ST},
                :B => [1.0 0.0
                    0.0 1.0] |> AT{ST})
            expr = :(kron(A, B))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A, a*B) + kron(A, a*B) + kron(c*A, C)" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST},
                :C => rand(2, 2) |> AT{ST},
                :a => rand() |> ST,
                :b => rand() |> ST,
                :c => rand() |> ST)
            expr = :(kron(A, a*B) + kron(A, a*B) + kron(c*A, C))
            test_expression(AT{ST}, vars, expr)
        end

         @testset "transpose(A) + B" begin
            vars = Dict(
                :A => rand(3, 2) |> AT{ST},
                :B => rand(2, 3) |> AT{ST})
            expr = :(transpose(A) + B)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "transpose(A) * B" begin
            vars = Dict(
                :A => rand(3, 2) |> AT{ST},
                :B => rand(3, 4) |> AT{ST})
            expr = :(transpose(A) * B)
            test_expression(AT{ST}, vars, expr)
        end

        if !(AT <: CUDA.CUSPARSE.CuSparseMatrixCSC)
            @testset "a*transpose(A) + b*B" begin
                vars = Dict(
                    :A => rand(2, 3) |> AT{ST},
                    :B => rand(3, 2) |> AT{ST},
                    :a => rand() |> ST,
                    :b => rand() |> ST)
                expr = :(a*transpose(A) + b*B)
                test_expression(AT{ST}, vars, expr)
            end
        end

        @testset "A * B * C" begin
            vars = Dict(
                :A => rand(2, 3) |> AT{ST},
                :B => rand(3, 4) |> AT{ST},
                :C => rand(4, 2) |> AT{ST})
            expr = :(A * B * C)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*A * b*B" begin
            vars = Dict(
                :A => rand(3, 2) |> AT{ST},
                :B => rand(2, 3) |> AT{ST},
                :a => rand() |> ST,
                :b => rand() |> ST)
            expr = :(a*A * b*B)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A, B) + kron(B, A)" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST})
            expr = :(kron(A, B) + kron(B, A))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A, B, C) + kron(A, D, C)" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST},
                :C => rand(2, 2) |> AT{ST},
                :D => rand(2, 2) |> AT{ST},
                :a => rand() |> ST,
                :b => rand() |> ST,
                :c => rand() |> ST)
            expr = :(kron(A, B, C) + kron(A, D, C))
            test_expression(AT{ST}, vars, expr)

            expr = :(kron(A, a*B, C) + kron(A, b*D, C))
            test_expression(AT{ST}, vars, expr)

            expr = :(kron(A, B, C) + kron(A, b*D, C))
            test_expression(AT{ST}, vars, expr)

            expr = :(kron(A, a*B, C) + kron(A, D, C))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A + B, C)" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST},
                :C => rand(2, 2) |> AT{ST})
            expr = :(kron(A + B, C))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A, B + C)" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST},
                :C => rand(2, 2) |> AT{ST})
            expr = :(kron(A, B + C))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*(A + B) * C" begin
            vars = Dict(
                :A => rand(2, 3) |> AT{ST},
                :B => rand(2, 3) |> AT{ST},
                :C => rand(3, 2) |> AT{ST},
                :a => rand() |> ST)
            expr = :(a*(A + B) * C)
            test_expression(AT{ST}, vars, expr)
        end

        if !(AT <: CUDA.CUSPARSE.CuSparseMatrixCSC)
            @testset "a*transpose(A * B) + b*C" begin
                vars = Dict(
                    :A => rand(2, 3) |> AT{ST},
                    :B => rand(3, 2) |> AT{ST},
                    :C => rand(2, 2) |> AT{ST},
                    :a => rand() |> ST,
                    :b => rand() |> ST)
                expr = :(a*transpose(A * B) + b*C)
                test_expression(AT{ST}, vars, expr)
            end
        end

        @testset "A * B + C * D" begin
            vars = Dict(
                :A => rand(2, 3) |> AT{ST},
                :B => rand(3, 2) |> AT{ST},
                :C => rand(2, 3) |> AT{ST},
                :D => rand(3, 2) |> AT{ST})
            expr = :(A * B + C * D)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A, B) * C" begin
            vars = Dict(
                :A => rand(2, 2) |> AT{ST},
                :B => rand(2, 2) |> AT{ST},
                :C => rand(4, 3) |> AT{ST})
            expr = :(kron(A, B) * C)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "C * kron(A, B)" begin
            vars = Dict(
            :A => rand(2, 2) |> AT{ST},
            :B => rand(2, 2) |> AT{ST},
            :C => rand(3, 4) |> AT{ST})
            expr = :(C * kron(A, B))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A, B) + kron(A, B)" begin
            vars = Dict(
            :A => rand(2, 2) |> AT{ST},
            :B => rand(2, 2) |> AT{ST})
            expr = :(kron(A, B) + kron(A, B))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*kron(A, B) + b*kron(C, D)" begin
            vars = Dict(
            :A => rand(2, 2) |> AT{ST},
            :B => rand(2, 2) |> AT{ST},
            :C => rand(2, 2) |> AT{ST},
            :D => rand(2, 2) |> AT{ST},
            :a => rand() |> ST,
            :b => rand() |> ST)
            expr = :(a*kron(A, B) + b*kron(C, D))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "A * transpose(B) + C" begin
            vars = Dict(
            :A => rand(2, 3) |> AT{ST},
            :B => rand(2, 3) |> AT{ST},
            :C => rand(2, 2) |> AT{ST})
            expr = :(A * transpose(B) + C)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "transpose(A) * transpose(B) + C" begin
            vars = Dict(
            :A => rand(3, 2) |> AT{ST},
            :B => rand(2, 3) |> AT{ST},
            :C => rand(2, 2) |> AT{ST})
            expr = :(transpose(A) * transpose(B) + C)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*kron(A, B) * C + b*D" begin
            vars = Dict(
            :A => rand(2, 2) |> AT{ST},
            :B => rand(2, 2) |> AT{ST},
            :C => rand(4, 3) |> AT{ST},
            :D => rand(4, 3) |> AT{ST},
            :a => rand() |> ST,
            :b => rand() |> ST)
            expr = :(a*kron(A, B) * C + b*D)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A * B, C * D)" begin
            vars = Dict(
            :A => rand(2, 2) |> AT{ST},
            :B => rand(2, 2) |> AT{ST},
            :C => rand(2, 2) |> AT{ST},
            :D => rand(2, 2) |> AT{ST})
            expr = :(kron(A * B, C * D))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*kron(A, B) + b*kron(C, D) + c*kron(E, F)" begin
            vars = Dict(
            :A => rand(2, 2) |> AT{ST},
            :B => rand(2, 2) |> AT{ST},
            :C => rand(2, 2) |> AT{ST},
            :D => rand(2, 2) |> AT{ST},
            :E => rand(2, 2) |> AT{ST},
            :F => rand(2, 2) |> AT{ST},
            :a => rand() |> ST,
            :b => rand() |> ST,
            :c => rand() |> ST)
            expr = :(a*kron(A, B) + b*kron(C, D) + c*kron(E, F))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "A * B * transpose(C) + D" begin
            vars = Dict(
            :A => rand(2, 3) |> AT{ST},
            :B => rand(3, 4) |> AT{ST},
            :C => rand(2, 4) |> AT{ST},
            :D => rand(2, 2) |> AT{ST})
            expr = :(A * B * transpose(C) + D)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A, B) + kron(A, C) + kron(A, D)" begin
            vars = Dict(
            :A => rand(2, 2) |> AT{ST},
            :B => rand(2, 2) |> AT{ST},
            :C => rand(2, 2) |> AT{ST},
            :D => rand(2, 2) |> AT{ST})
            expr = :(kron(A, B) + kron(A, C) + kron(A, D))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*A * B + b*transpose(C) * D" begin
            vars = Dict(
            :A => rand(2, 3) |> AT{ST},
            :B => rand(3, 2) |> AT{ST},
            :C => rand(4, 2) |> AT{ST},
            :D => rand(4, 2) |> AT{ST},
            :a => rand() |> ST,
            :b => rand() |> ST)
            expr = :(a*A * B + b*transpose(C) * D)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*kron(A, B) * C + b*kron(D, E) * F + c*G" begin
            vars = Dict(
            :A => rand(2, 2) |> AT{ST},
            :B => rand(2, 2) |> AT{ST},
            :C => rand(4, 3) |> AT{ST},
            :D => rand(2, 2) |> AT{ST},
            :E => rand(2, 2) |> AT{ST},
            :F => rand(4, 3) |> AT{ST},
            :G => rand(4, 3) |> AT{ST},
            :a => rand() |> ST,
            :b => rand() |> ST,
            :c => rand() |> ST)
            expr = :(a*kron(A, B) * C + b*kron(D, E) * F + c*G)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*(A * B + C) * D + b*transpose(E) * F" begin
            vars = Dict(
            :A => rand(2, 3) |> AT{ST},
            :B => rand(3, 2) |> AT{ST},
            :C => rand(2, 2) |> AT{ST},
            :D => rand(2, 3) |> AT{ST},
            :E => rand(3, 2) |> AT{ST},
            :F => rand(3, 3) |> AT{ST},
            :a => rand() |> ST,
            :b => rand() |> ST)
            expr = :(a*(A * B + C) * D + b*transpose(E) * F)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "kron(A * B, C) + kron(D, E * F) + G" begin
            vars = Dict(
            :A => rand(2, 2) |> AT{ST},
            :B => rand(2, 2) |> AT{ST},
            :C => rand(2, 2) |> AT{ST},
            :D => rand(2, 2) |> AT{ST},
            :E => rand(2, 2) |> AT{ST},
            :F => rand(2, 2) |> AT{ST},
            :G => rand(4, 4) |> AT{ST})
            expr = :(kron(A * B, C) + kron(D, E * F) + G)
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*kron(A + B, C * D) + transpose(kron(E, F))" begin
            vars = Dict(
            :A => rand(2, 2) |> AT{ST},
            :B => rand(2, 2) |> AT{ST},
            :C => rand(2, 2) |> AT{ST},
            :D => rand(2, 2) |> AT{ST},
            :E => rand(2, 2) |> AT{ST},
            :F => rand(2, 2) |> AT{ST},
            :a => rand() |> ST,
            :b => rand() |> ST)
            expr = :(a*kron(A + B, C * D) + transpose(kron(E, F)))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "A * B * C + transpose(D) * E * F + kron(G, H)" begin
            vars = Dict(
            :A => rand(2, 3) |> AT{ST},
            :B => rand(3, 4) |> AT{ST},
            :C => rand(4, 2) |> AT{ST},
            :D => rand(3, 2) |> AT{ST},
            :E => rand(3, 4) |> AT{ST},
            :F => rand(4, 2) |> AT{ST},
            :G => rand(2, 2) |> AT{ST},
            :H => rand(1, 1) |> AT{ST})
            expr = :(A * B * C + transpose(D) * E * F + kron(G, H))
            test_expression(AT{ST}, vars, expr)
        end

        @testset "a*(A * transpose(B) + C * D) + b*kron(E, F * transpose(G))" begin
            vars = Dict(
            :A => rand(2, 3) |> AT{ST},
            :B => rand(2, 3) |> AT{ST},
            :C => rand(2, 3) |> AT{ST},
            :D => rand(3, 2) |> AT{ST},
            :E => rand(1, 1) |> AT{ST},
            :F => rand(2, 2) |> AT{ST},
            :G => rand(2, 2) |> AT{ST},
            :a => rand() |> ST,
            :b => rand() |> ST)
            expr = :(a*(A * transpose(B) + C * D) + b*kron(E, F * transpose(G)))
            test_expression(AT{ST}, vars, expr)
        end
    end
end
