module Sparse3TensorTest

using Test
using LinearAlgebra
using Random
using Gridap

import EPMAfem.Sparse3Tensor as ST
import EPMAfem.SpaceModels as SM

function rand_tensor((n1, n2, n3), nvals)
    I = rand(1:n1, nvals)
    J = rand(1:n2, nvals)
    K = rand(1:n3, nvals)

    V = rand(nvals)

    return ST.sparse3tensor(I, J, K, V, (n1, n2, n3))
end

function test_sparse3tensor_construction(; nvals, n1, n2, n3)    
    I = rand(1:n1, nvals)
    J = rand(1:n2, nvals)
    K = rand(1:n3, nvals)

    V = rand(nvals)
    
    A = zeros(n1, n2, n3)
    for i in 1:nvals
        A[I[i], J[i], K[i]] += V[i]
    end

    u = rand(n1)
    v = rand(n2)
    w = rand(n3)

    A_tensor = ST.sparse3tensor(I, J, K, V, (n1, n2, n3))
    A_tensor_from_array = ST.sparse3tensor(A)

    @test Array(A_tensor) ≈ A
    @test Array(A_tensor_from_array) ≈ A

    @test ST.tensordot(A, u, v, w) ≈ ST.tensordot(A_tensor, u, v, w)
    @test ST.tensordot(A, u, v, w) ≈ ST.tensordot(A_tensor_from_array, u, v, w)
end

# make sure we have duplicates
test_sparse3tensor_construction(; n1 = 5, n2 = 6, n3 = 7, nvals = 1000)

# probably no duplicates
test_sparse3tensor_construction(; n1 = 5, n2 = 6, n3 = 7, nvals = 5)

function test_simple_dense_tensor()
    A = rand(5, 10, 20)

    A_sparse = ST.sparse3tensor(A)
    A_ssm = ST.convert_to_SSM(A_sparse)

    u = rand(5)
    v = rand(10)
    w = rand(20)

    @test ST.tensordot(A, u, v, w) ≈ ST.tensordot(A_sparse, u, v, w)
    @test ST.tensordot(A, u, v, w) ≈ ST.tensordot(A_ssm, u, v, w)
end

test_simple_dense_tensor()

function test_ssm_conversion(; n1, n2, n3, nvals)
    A_sparse = rand_tensor((n1, n2, n3), nvals)
    A_ssm = ST.convert_to_SSM(A_sparse)
    A_ssm_kij = ST.convert_to_SSM(A_sparse, :kij)

    u = rand(n1)
    v = rand(n2)
    w = rand(n3)

    @test ST.tensordot(A_sparse, u, v, w) ≈ ST.tensordot(A_ssm, u, v, w)
    @test ST.tensordot(A_sparse, u, v, w) ≈ ST.tensordot(A_ssm_kij, w, u, v)

    a_sparse = ST.project!(A_ssm, w)
    @test ST.tensordot(A_ssm, u, v, w) ≈ dot(u, a_sparse, v)

    y = zeros(n3)
    ST.contract!(y, A_ssm, u, v, true, false)
    @test ST.tensordot(A_ssm, u, v, w) ≈ dot(y, w)
end

test_ssm_conversion(; n1=5, n2=6, n3=7, nvals=1000)
test_ssm_conversion(; n1=5, n2=6, n3=7, nvals=5)

function test_diagonal(; n, nvals)
    I = rand(1:n, nvals)
    V = rand(nvals)


    u = rand(n)
    v = rand(n)
    w = rand(n)

    A_tensor = ST.sparse3tensor(I, I, I, V, (n, n, n))
    A_ssm = ST.convert_to_SSM(A_tensor)
    A = Array(A_tensor)

    @test A_ssm.skeleton isa Diagonal
    @test all([proj isa Diagonal for proj in A_ssm.projector])

    @test ST.tensordot(A, u, v, w) ≈ ST.tensordot(A_tensor, u, v, w)
    @test ST.tensordot(A, u, v, w) ≈ ST.tensordot(A_ssm, u, v, w)
end

test_diagonal(; n=5, nvals=1000)
test_diagonal(; n=5, nvals=2)

function test_sparse3tensor_gridap_assembly()
    gmodel = CartesianDiscreteModel((0, 1, 0, 1), (30, 30))
    model = SM.GridapSpaceModel(gmodel)

    gR = Triangulation(gmodel)
    gdx = Measure(gR, 5)
    g∂R = BoundaryTriangulation(gmodel)
    gdΓ = Measure(g∂R, 5)

    ga(u, v, w) = ∫(u*v*w)gdx
    a(u, v, (dims, dx, ∂R, dΓ, n)) = ∫(u*v)dx

    tensor = SM.assemble_trilinear(a, model, SM.even(model), SM.even(model))

    reffe = ReferenceFE(lagrangian, Float64, 1)
    reffe0 = ReferenceFE(lagrangian, Float64, 0)
    V = TestFESpace(gmodel, reffe, conformity=:H1)
    U = TrialFESpace(V)
    W = TestFESpace(gmodel, reffe0, conformity=:L2)

    u = FEFunction(U, rand(num_free_dofs(U)))
    v = FEFunction(V, rand(num_free_dofs(V)))
    w = FEFunction(W, rand(num_free_dofs(W)))

    gval = ga(u, v, w) |> sum
    val = ST.tensordot(tensor, u.free_values, v.free_values, w.free_values)

    @test gval ≈ val

    tensor_ssm = ST.convert_to_SSM(tensor)

    mat = ST.project!(tensor_ssm, w.free_values)
    gmat = Gridap.assemble_matrix((u, v) -> ga(u, v, w), U, V)

    @test gmat ≈ mat

    #tensor_ssm_kij = ST.convert_to_SSM(tensor, :kij)
    y = zeros(num_free_dofs(W))
    qy = zeros(num_free_dofs(W))
    ST.contract!(y, tensor_ssm, u.free_values, v.free_values, 1.0, 0.0)

    w_ = FEFunction(W, zeros(num_free_dofs(W)))
    for i in 1:num_free_dofs(W)
        w_.free_values[i] = 1.0
        qy[i] = ga(u, v, w_) |> sum
        w_.free_values[i] = 0.0
    end
    @test all(y ≈ qy)
end

test_sparse3tensor_gridap_assembly()

function test_tensor_matrix_multiplication(; n1, n2, n3, nu, nvals)
    A_sparse = rand_tensor((n1, n2, n3), nvals)
    A = Array(A_sparse)
    A_ssm = ST.convert_to_SSM(A_sparse)


    u = rand(n1, nu)
    u2 = rand(n1, nu)
    v = rand(n2, nu)
    v2 = rand(n2, nu)
    w = rand(n3)

    @test ST.tensordot(A_sparse, u, v, w) ≈ ST.tensordot(A, u, v, w)
    @test ST.tensordot(A_ssm, u, v, w) ≈ ST.tensordot(A, u, v, w)

    y = rand(length(w)) #initialize with random number to test the zero out
    ST.contract!(y, A_ssm, u, v, true, false)
    @test ST.tensordot(A_ssm, u, v, w) ≈ dot(y, w)

    is, js = ST.get_ijs(A_ssm)
    nz_vals = rand(length(is)) # init with random
    y_2 = rand(length(w)) #init with random
    ST.special_matmul!(nz_vals, is, js, u, v, true, false)
    ST.contract!(y_2, A_ssm, nz_vals, true, false)
    @test y ≈ y_2
    @test ST.tensordot(A_ssm, u, v, w) ≈ dot(y_2, w)

    # now test accumulation
    α = rand()
    ST.contract!(y, A_ssm, u2, v2, α, true)
    @test ST.tensordot(A_ssm, u, v, w) + α * ST.tensordot(A_ssm, u2, v2, w) ≈ dot(y, w)

    # also for the searated accumulation
    ST.special_matmul!(nz_vals, is, js, u2, v2, α, true)
    ST.contract!(y_2, A_ssm, nz_vals, true, false)
    @test y ≈ y_2
    @test ST.tensordot(A_ssm, u, v, w) + α * ST.tensordot(A_ssm, u2, v2, w) ≈ dot(y_2, w)
end

test_tensor_matrix_multiplication(; n1=5, n2=6, n3=7, nu=5, nvals=1000)
test_tensor_matrix_multiplication(; n1=5, n2=6, n3=7, nu=5, nvals=25)

end
