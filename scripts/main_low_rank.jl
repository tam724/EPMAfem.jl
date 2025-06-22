using Revise
using EPMAfem
using LinearAlgebra
using Test
using CUDA
# include("plot_overloads.jl")

create_rand_mat(n::Int, m::Int) = rand(n, m) |> cu
create_rand_vec(n::Int) = rand(n) |> cu
T() = Float32
T(x) = T()(x)

# test the caching system to some extent ...
@testset let
    nS1 = rand(1:40)
    mS1 = rand(1:40)
    A1 = create_rand_mat(nS1, mS1)
    A2 = create_rand_mat(nS1, mS1)
    S1 = EPMAfem.SumMatrix{T()}((A1, A2), (rand(T()), rand(T())))
    C1 = EPMAfem.Cached{T()}(S1)

    nS2 = rand(1:40)
    mS2 = rand(1:40)
    B1 = create_rand_mat(nS2, mS2)
    B2 = create_rand_mat(nS2, mS2)
    S2 = EPMAfem.SumMatrix{T()}([B1, B2], rand(2))
    C2 = EPMAfem.Cached{T()}(S2)

    KS = EPMAfem.KronMatrix{T()}(S1, S2)
    KC = EPMAfem.KronMatrix{T()}(C1, C2)

    KS_ref = Matrix(KS)
    KC_ref = Matrix(KC)
    @test KS_ref ≈ KC_ref

    KS_wsch = EPMAfem.required_workspace_cache(KS)
    KC_wsch = EPMAfem.required_workspace_cache(KC)

    @test EPMAfem.mul_with_ws(KS_wsch) == min(nS1*nS2, mS1*mS2)# the kronecker workspace
    @test EPMAfem.cache_with_ws(KS_wsch) == nS1*mS1 + nS2*mS2 # the cached sums

    @test EPMAfem.mul_with_ws(KC_wsch) == min(nS1*nS2, mS1*mS2)# the kronecker workspace
    @test EPMAfem.cache_with_ws(KC_wsch) == nS1*mS1 + nS2*mS2 # the cached sums

    WS = EPMAfem.Wrapped{T()}(KS)
    WC = EPMAfem.Wrapped{T()}(KC)

    # WS.workspace_cache
    # WC.workspace_cache

    x = create_rand_vec(size(WS, 2))
    yS = create_rand_vec(size(WS, 1))
    yC = create_rand_vec(size(WC, 1))

    mul!(yS, WS, x)
    mul!(yC, WC, x)

    # @benchmark mul!($yS, $WS, $x)
    # @benchmark mul!($yC, $WC, $x)

    @test yS ≈ yC

    # WS.workspace_cache
    # WC.workspace_cache
end

# Cached and Wrapped
@testset let
    A = create_rand_mat(10, 11)
    C = EPMAfem.Cached{T()}(A)
    W = EPMAfem.Wrapped{T()}(C)

    x = create_rand_vec(size(C, 2))
    @test A*x ≈ C*x
    @test A*x ≈ W*x

    At = transpose(A)
    Ct = transpose(C)
    Wt = transpose(W)

    x = create_rand_vec(size(Ct, 2))
    @test At*x ≈ Ct*x
    @test At*x ≈ Wt*x
end

# BlockMatrix
@testset let
    a = create_rand_mat(10, 10)
    b = create_rand_mat(10, 15)
    c = create_rand_mat(15, 15)

    B = EPMAfem.BlockMatrix{T()}(a, b, c, rand(T()), rand(T()), rand(T()), rand(T()), true, T(1.0))
    B_ref_ = B.Δ .* [
        B.α .* B.A      B.β .* B.B
        B.δ .* B.βt .* transpose(B.B)   B.δ .* B.γ .* B.C
    ]
    B_ref = Matrix(B)
    @test B_ref_ ≈ B_ref

    x = create_rand_vec(size(B, 2))
    @test B_ref*x ≈ B*x

    Bt = transpose(B)
    Bt_ref = transpose(B_ref)
    x = create_rand_vec(size(Bt, 2))
    @test Bt_ref*x ≈ Bt*x
end

# stich KronMatrix and BlockMatrix together
@testset let
    KA = EPMAfem.KronMatrix{T()}(create_rand_mat(10, 10), create_rand_mat(11, 11))
    KA_ref = Matrix(KA)

    KB = EPMAfem.KronMatrix{T()}(create_rand_mat(10, 9), create_rand_mat(12, 11))
    KB_ref = Matrix(KB)

    KC = EPMAfem.KronMatrix{T()}(create_rand_mat(9, 9), create_rand_mat(12, 12))
    KC_ref = Matrix(KC)

    B = EPMAfem.BlockMatrix{T()}(KA, KB, KC, rand(T()), rand(T()), rand(T()), rand(T()), true, T(1.0))
    B_ref = Matrix(B)

    x = create_rand_vec(size(B, 2))
    @test B*x ≈ B_ref*x

    Bt = transpose(B)
    x = create_rand_vec(size(Bt, 2))
    @test Bt*x ≈ transpose(B_ref)*x
end

## SumMatrix
@testset let
    A1 = create_rand_mat(10, 11)
    A2 = create_rand_mat(10, 11)
    A3 = create_rand_mat(10, 11)
    αs = rand(3)
    S = EPMAfem.SumMatrix{T()}([A1, A2, A3], αs)
    S_tuple = EPMAfem.SumMatrix{T()}((A1, A2, A3), tuple(αs...))

    S_ref_ = αs[1] .* A1 .+ αs[2] .* A2 .+ αs[3] .* A3 
    S_ref = Matrix(S)
    @test S_ref_ ≈ S_ref
    S_ref = Matrix(S_tuple)
    @test S_ref_ ≈ S_ref

    x = create_rand_vec(size(S)[2])
    @test S_ref * x ≈ S * x
    @test S_ref * x ≈ S_tuple * x

    St = transpose(S)
    St_tuple = transpose(S_tuple)
    St_ref = transpose(S_ref)

    x = create_rand_vec(size(St)[2])
    @test St_ref * x ≈ St * x
    @test St_ref * x ≈ St_tuple * x
end

# stich KronMatrix and SumMatrix together
@testset let
    K1 = EPMAfem.KronMatrix{T()}(create_rand_mat(10, 11), create_rand_mat(12, 13))
    K1_ref = kron(transpose(K1.B), K1.A)
    K2 = EPMAfem.KronMatrix{T()}(create_rand_mat(10, 11), create_rand_mat(12, 13))
    K2_ref = kron(transpose(K2.B), K2.A)
    S1 = EPMAfem.SumMatrix{T()}([K1, K2], rand(2))
    S1_ref = K1_ref .* S1.αs[1] .+ K2_ref .* S1.αs[2]

    x = create_rand_vec(size(S1)[2])
    @test S1_ref * x ≈ S1 * x

    S1t = transpose(S1)
    x = create_rand_vec(size(S1t)[2])
    @test transpose(S1_ref) * x ≈ S1t * x

    # stich the two together
    S1 = EPMAfem.SumMatrix{T()}([create_rand_mat(10, 11), create_rand_mat(10, 11)], rand(2))
    S1_ref = S1.As[1] .* S1.αs[1] .+ S1.As[2] .* S1.αs[2]

    S2 = EPMAfem.SumMatrix{T()}([create_rand_mat(12, 13), create_rand_mat(12, 13)], rand(2))
    S2_ref = S2.As[1] .* S2.αs[1] .+ S2.As[2] .* S2.αs[2]

    K1 = EPMAfem.KronMatrix{T()}(S1, S2)
    K1_ref = kron(transpose(S2_ref), S1_ref)

    x = create_rand_vec(size(K1)[2])
    @test K1*x ≈ K1_ref*x

    S3 = EPMAfem.SumMatrix{T()}([create_rand_mat(10, 11), create_rand_mat(10, 11)], rand(2))
    S3_ref = S3.As[1] .* S3.αs[1] .+ S3.As[2] .* S3.αs[2]

    S4 = EPMAfem.SumMatrix{T()}([create_rand_mat(12, 13), create_rand_mat(12, 13)], rand(2))
    S4_ref = S4.As[1] .* S4.αs[1] .+ S4.As[2] .* S4.αs[2]

    K2 = EPMAfem.KronMatrix{T()}(S3, S4)
    K2_ref = kron(transpose(S4_ref), S3_ref)

    K = EPMAfem.SumMatrix{T()}([K1, K2], rand(2))
    K_ref = K1_ref .* K.αs[1] .+ K2_ref .* K.αs[2]

    x = create_rand_vec(size(K)[2])
    @test K*x ≈ K_ref*x

    x = create_rand_vec(size(K)[1])
    @test transpose(K)*x ≈ transpose(K_ref)*x
end

## KronMatrix
@testset let
    A = create_rand_mat(10, 11)
    B = create_rand_mat(12, 13)

    K = EPMAfem.KronMatrix{T()}(A, B)
    K_ref_ = kron(transpose(B), A)
    K_ref = Matrix(K)
    @test K_ref_ ≈ K_ref

    @test isdiag(K) == false

    @test size(K) == size(K_ref)

    x = create_rand_vec(size(K)[2])
    @test K * x ≈ K_ref * x

    y = create_rand_vec(size(K)[1])
    y_ref = copy(y)

    α = rand(T())
    β = rand(T())
    mul!(y, K, x, α, β)
    mul!(y_ref, K_ref, x, α, β)
    @test y ≈ y_ref

    Kt = transpose(K)
    Kt_ref = transpose(K)
    isdiag(Kt) == false

    @test size(Kt) == size(Kt_ref)

    x = create_rand_vec(size(Kt)[2])
    @test Kt * x ≈ Kt_ref * x

    y = create_rand_vec(size(Kt)[1])
    y_ref = copy(y)
    α = rand(T())
    β = rand(T())
    mul!(y, Kt, x, α, β)
    mul!(y_ref, Kt_ref, x, α, β)
    @test y ≈ y_ref

    A = Diagonal(create_rand_vec(10))
    B = Diagonal(create_rand_vec(11))

    K_diag = EPMAfem.KronMatrix{T()}(A, B)
    @test isdiag(K_diag) == true
end

nothing

# space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1, 1), (50, 100)))
# direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(13, 2)

# eq = EPMAfem.PNEquations()
# beam = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [1.8], [VectorValue(-1.0, 0.0, 0.0) |> normalize]; beam_position_σ=0.05, beam_energy_σ=0.05)

# model = EPMAfem.DiscretePNModel(space_model, 0.0:0.02:2.0, direction_model)

# problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cuda(), updatable=false)
# system = EPMAfem.implicit_midpoint(problem, EPMAfem.PNSchurSolver)

# rhs = EPMAfem.discretize_rhs(beam, model, EPMAfem.cuda())[1]

# @gif for (ϵ, ψ) in system * rhs
#     @show ϵ
#     ψp, ψm = EPMAfem.pmview(ψ, model)
#     plot(svd(ψp).S |> collect)
#     plot!(svd(ψm).S |> collect)
# end

# probe = EPMAfem.PNProbe(model, EPMAfem.cuda(); Ω = Ω -> 1.0)
# res = probe(system * rhs)

# @gif for i in reverse(1:size(res.p, 2))
#     func = EPMAfem.SpaceModels.interpolable((p=res.p[:, i], m=res.m[:, i]), EPMAfem.space_model(model))
#     heatmap(-1:0.01:1, -1:0.01:0, (x, z) -> func(Gridap.VectorValue(z, x)), aspect_ratio=:equal)
# end
