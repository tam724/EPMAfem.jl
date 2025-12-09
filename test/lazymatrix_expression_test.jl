using Revise
using EPMAfem.PNLazyMatrices
using Test
using LinearAlgebra
using CUDA

function build_matrices(vars, expr)
    lazy_vars = Dict(key=>lazy(val) for (key, val) in vars)

    assignments = [:($key = $(val)) for (key, val) in vars]
    M = eval(Expr(:block, assignments..., expr))

    lazy_assignments = [:($key = $(val)) for (key, val) in lazy_vars]
    lazy_M = eval(Expr(:block, lazy_assignments..., expr))
    return M, lazy_M
end

function test_random_multiplication(AT, M, lazy_M, N = 5)
    unlazy_M = unlazy(lazy_M, size->similar(AT, size))
    y, lazy_y = similar(M, size(M, 1)), similar(M, size(M, 1))
    fill!(y, zero(eltype(M))); 
    fill!(lazy_y, zero(eltype(M)));
    x = similar(M, size(M, 2))
    for _ in 1:N
        x .= rand(size(x))
        mul!(y, M, x)
        mul!(lazy_y, unlazy_M, x)
        @test y ≈ lazy_y
    end
end

function test_random_multiplication_mat(AT, M, lazy_M, N = 5, n=5)
    unlazy_M = unlazy(lazy_M, size->similar(AT, size); n=n)
    Y, lazy_Y = similar(AT, size(M, 1), n), similar(AT, size(M, 1), n)
    fill!(Y, zero(eltype(M))); 
    fill!(lazy_Y, zero(eltype(M)));
    X = similar(AT, size(M, 2), n)
    for _ in 1:N
        X .= rand(size(X))
        mul!(Y, M, X)
        mul!(lazy_Y, unlazy_M, X)
        @test Y ≈ lazy_Y
    end
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
end


let
    @testset "$AT{$ST}" for (AT, ST) in [(Array, Float64), (CuArray, Float32)]
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

        # @testset "kron(A, a*B) + kron(A, a*B) + kron(c*A, C)" begin
        #     vars = Dict(
        #         :A => rand(2, 2) |> AT{ST},
        #         :B => rand(2, 2) |> AT{ST},
        #         :C => rand(2, 2) |> AT{ST},
        #         :a => rand() |> ST,
        #         :b => rand() |> ST,
        #         :c => rand() |> ST)
        #     expr = :(kron(A, a*B) + kron(A, a*B) + kron(c*A, C))
        #     test_expression(AT{ST}, vars, expr)
        # end

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

        @testset "a*transpose(A) + b*B" begin
            vars = Dict(
                :A => rand(2, 3) |> AT{ST},
                :B => rand(3, 2) |> AT{ST},
                :a => rand() |> ST,
                :b => rand() |> ST)
            expr = :(a*transpose(A) + b*B)
            test_expression(AT{ST}, vars, expr)
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

        # @testset "kron(A, B) + kron(A, B)" begin
        #     vars = Dict(
        #     :A => rand(2, 2) |> AT{ST},
        #     :B => rand(2, 2) |> AT{ST})
        #     expr = :(kron(A, B) + kron(A, B))
        #     test_expression(AT{ST}, vars, expr)
        # end

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
    end
end
