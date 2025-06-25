using Revise
using EPMAfem
using LinearAlgebra
using Test
using CUDA
using ConcreteStructs
using BenchmarkTools
using SparseArrays
# include("plot_overloads.jl")


lazy(A) = EPMAfem.lazy(A)

using BenchmarkTools

rand_mat(m, n) = rand(m, n)
rand_vec(n) = rand(n)


# materialize kron stays diagonal (we need this for schur)
@testset let
    A_ = Diagonal(rand(5))
    B_ = Diagonal(rand(5))

    A = lazy(A_)
    B = lazy(B_)

    D = EPMAfem.materialize(kron(A, B))
    @test EPMAfem.isdiagonal(D)

    ws = EPMAfem.Workspace(zeros(EPMAfem.required_workspace(EPMAfem.materialize_with, D)))
    @test ws.workspace |> length == 25
    D_mat, rem_ws = EPMAfem.materialize_with(ws, D)
    @test D_mat isa Diagonal
    @test D_mat ≈ kron(A_, B_)

    D_big = EPMAfem.materialize(kron(kron(A, B), 3.0 * kron(A, B)))
    ws = EPMAfem.Workspace(zeros(EPMAfem.required_workspace(EPMAfem.materialize_with, D_big)))
    ws.workspace |> length

    D_big_mat, rem_ws = EPMAfem.materialize_with(ws, D_big)
    EPMAfem.materialize_with(ws, D_big)

    @test D_big_mat isa Diagonal
    @test D_big_mat ≈ kron(kron(A, B), 3.0 .* kron(A, B))
end

@testset let
    a_ = rand_mat(3, 4)
    b_ = rand_mat(5, 6)

    c_ = rand_mat(3, 4)
    d_ = rand_mat(5, 6)

    a, b, c, d = lazy.((a_, b_, c_, d_))

    k1 = kron(3.0 * a, b)
    k2 = kron(3.0 *c, EPMAfem.materialize(2.0 * d))

    s = EPMAfem.materialize(k1 + k2)

    EPMAfem.required_workspace(EPMAfem.mul_with!, s)
    ws = EPMAfem.Workspace(zeros(EPMAfem.required_workspace(EPMAfem.mul_with!, s)))

    x = rand_vec(size(s, 2))
    y = rand_vec(size(s, 1))

    EPMAfem.mul_with!(ws, y, s, x, true, false)

    test = kron(3.0*a_, b_) + kron(3.0*c, 2.0*d)
    @test Matrix(s) ≈ test
end

# benchmark a small example
@testset let
    a_ = rand_mat(50, 60)
    b_ = rand_mat(50, 60)

    c_ = rand_mat(70, 80)
    d_ = rand_mat(70, 80)
    e_ = rand_mat(70, 80)
    f_ = rand_mat(70, 80)
    g_ = rand_mat(70, 80)

    a, b, c, d, e, f, g = lazy.((a_, b_, c_, d_, e_, f_, g_))

    K_ = kron(a_ + b_, c_ + d_ + e_ + f_ + g_)
    K = kron(a + b, c + d + e + f + g)
    KM = kron(EPMAfem.materialize(a + b), EPMAfem.materialize(c + d + e + f + g))

    x = rand_vec(size(K, 2))
    y_ = zeros(size(K, 1))
    y = zeros(size(K, 1))
    yM = zeros(size(K, 1))


    ws = EPMAfem.Workspace(EPMAfem.required_workspace(EPMAfem.mul_with!, K) |> zeros)
    wsM = EPMAfem.Workspace(EPMAfem.required_workspace(EPMAfem.mul_with!, KM) |> zeros)

    ws.workspace |> size
    wsM.workspace |> size

    EPMAfem.mul_with!(ws, y, K, x, true, false)
    EPMAfem.mul_with!(wsM, yM, KM, x, true, false)
    LinearAlgebra.mul!(y_, K_, x, true, false)
    @test y ≈ y_
    @test yM ≈ y_

    temp1 = zeros(size(a_))
    temp2 = zeros(size(c_))
    temp3 = zeros(size(temp2, 1), size(temp1, 2))

    function fair_comparison(y, x, temp1, temp2, temp3, a, b, c, d, e, f, g)
        temp1 .= a .+ b
        temp2 .= c .+ d .+ e .+ f .+ g
        mul!(temp3, temp2, reshape(x, (size(temp2, 2), size(temp1, 2))), true, false)
        mul!(reshape(y, (size(temp2, 1), size(temp1, 1))), temp3, transpose(temp1), true, false)
        return y
    end

    yfair = zeros(size(K, 1))
    fair_comparison(yfair, x, temp1, temp2, temp3, a_, b_, c_, d_, e_, f_, g_)
    @test yfair ≈ y_

    # @profview @benchmark EPMAfem.mul_with!($ws, $y, $K, $x, $true, $false)
    # @profview @benchmark EPMAfem.mul_with!($wsM, $yM, $KM, $x, $true, $false)
    # @profview @benchmark fair_comparison($yfair, $x, $temp1, $temp2, $temp3, $a_, $b_, $c_, $d_, $e_, $f_, $g_)
    # @profview @benchmark LinearAlgebra.mul!($y_, $K_, $x, $true, $false)

    baseline_lazy = @benchmark EPMAfem.mul_with!($ws, $y, $K, $x, $true, $false)
    speedy_lazy = @benchmark EPMAfem.mul_with!($wsM, $yM, $KM, $x, $true, $false)
    speedy = @benchmark fair_comparison($yfair, $x, $temp1, $temp2, $temp3, $a_, $b_, $c_, $d_, $e_, $f_, $g_)
    slow = @benchmark LinearAlgebra.mul!($y_, $K_, $x, $true, $false)

    @test time(slow) > time(baseline_lazy) > time(speedy_lazy) 
    @test time(baseline_lazy) > time(speedy) 
    @show "Lazy"
    display(speedy_lazy)
    @show "fair comparison"
    display(speedy)

    # EPMAfem.mul_with!(wsM, yM, KM, x, true, false)
end


# check ReshapeMatrix + KronMatrix
@testset begin
    A = rand(10, 11)
    B = rand(12, 13)
    R = EPMAfem.ReshapeableMatrix(B)

    K = EPMAfem.KronMatrix(A, R)
    K_ref = kron(transpose(B), A)

    @test size(K) == (size(B, 2)*size(A, 1), size(B, 1)*size(A, 2))
    x = create_rand_vec(size(K, 2))
    @test K * x ≈ K_ref * x

    C = rand(3, 4)
    EPMAfem.set!(R, C)
    K_ref = kron(transpose(C), A)

    @test size(K) == (size(C, 2)*size(A, 1), size(C, 1)*size(A, 2))
    x = create_rand_vec(size(K, 2))
    @test K * x ≈ K_ref * x
end

# check aliasing
@testset begin
    A = rand(10, 11)
    R = EPMAfem.ReshapeableMatrix(A)

    K = EPMAfem.SumMatrix((R, R), (1.0, 2.0))
    K_ref = 1.0 .* A .+ 2.0 .* A


    @test size(K) == (10, 11)
    x = create_rand_vec(size(K, 2))
    @test K*x ≈ K_ref * x

    B = rand(2, 3)
    EPMAfem.set!(R, B)

    K_ref = 1.0 .* B .+ 2.0 .* B
    @test size(K) == (2, 3)
    x = create_rand_vec(size(K, 2))
    @test K*x ≈ K_ref*x
end

# check cache invalidity bubbling
@testset let
    A = create_rand_mat(10, 10)
    α1 = rand()
    S1 = EPMAfem.SumMatrix((A, ), (α1, ))
    C1 = EPMAfem.Cached(S1)
    α2 = rand()
    S2 = EPMAfem.SumMatrix((C1, ), (α2, ))
    C2 = EPMAfem.Cached(S2)
    W = EPMAfem.Wrapped(C2)

    W_ref = α1*α2.*A
    x = create_rand_vec(10)
    @test W * x ≈ W_ref * x
    @test C1.o[] != -1
    @test C2.o[] != -1

    # manually destroy the cache to check that it is not recomputed
    W.workspace_cache.cache[1][1] .= 0
    @test all(W*x .== 0.0)

    S1.αs[1][] = 1.0
    @test C1.o[] == -1
    @test C2.o[] == -1
    W_ref = α2.*A
    x = create_rand_vec(10)
    @test W * x ≈ W_ref * x

    # manually destroy the inner cache to check that it is not recomputed
    W.workspace_cache.cache[2][1][1][1] .= 0
    @test W*x ≈ W_ref*x

    EPMAfem.invalidate_cache!(W)
    @test W*x ≈ W_ref*x

    S2.αs[1][] = 1.0
    @test C1.o[] != -1
    @test C2.o[] == -1
    W_ref = A
    x = create_rand_vec(10)
    @test W*x ≈ W_ref*x
end

# test the caching system to some extent ...
@testset let
    nS1 = rand(1:40)
    mS1 = rand(1:40)
    A1 = create_rand_mat(nS1, mS1)
    A2 = create_rand_mat(nS1, mS1)
    S1 = EPMAfem.SumMatrix((A1, A2), (rand(T()), rand(T())))
    C1 = EPMAfem.Cached(S1)

    nS2 = rand(1:40)
    mS2 = rand(1:40)
    B1 = create_rand_mat(nS2, mS2)
    B2 = create_rand_mat(nS2, mS2)
    S2 = EPMAfem.SumMatrix([B1, B2], rand(2))
    C2 = EPMAfem.Cached(S2)

    KS = EPMAfem.KronMatrix(S1, S2)
    KC = EPMAfem.KronMatrix(C1, C2)

    KS_ref = Matrix(KS)
    KC_ref = Matrix(KC)
    @test KS_ref ≈ KC_ref

    KS_wsch = EPMAfem.required_workspace_cache(KS)
    KC_wsch = EPMAfem.required_workspace_cache(KC)

    @test EPMAfem.mul_with_ws(KS_wsch) == min(nS1*nS2, mS1*mS2)# the kronecker workspace
    @test EPMAfem.cache_with_ws(KS_wsch) == nS1*mS1 + nS2*mS2 # the cached sums

    @test EPMAfem.mul_with_ws(KC_wsch) == min(nS1*nS2, mS1*mS2)# the kronecker workspace
    @test EPMAfem.cache_with_ws(KC_wsch) == nS1*mS1 + nS2*mS2 # the cached sums

    WS = EPMAfem.Wrapped(KS)
    WC = EPMAfem.Wrapped(KC)

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
    C = EPMAfem.Cached(A)
    W = EPMAfem.Wrapped(C)

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
    a = rand_mat(10, 10)
    b = rand_mat(10, 15)
    c = rand_mat(15, 15)

    B = EPMAfem.lazy(EPMAfem.blockmatrix, a, b, c)
    B_ref_ = [
        a b
        transpose(b) c
    ]
    B_ref = Matrix(B)
    @test B_ref_ ≈ B_ref

    x = rand_vec(size(B, 2))
    @test B_ref*x ≈ B*x

    Bt = transpose(B)
    Bt_ref = transpose(B_ref)
    x = rand_vec(size(Bt, 2))
    @test Bt_ref*x ≈ Bt*x
end

# stich KronMatrix and BlockMatrix together
@testset let
    KA = EPMAfem.lazy(kron, rand_mat(10, 10), rand_mat(11, 11))
    KA_ref = Matrix(KA)

    KB = EPMAfem.lazy(kron, rand_mat(10, 9), rand_mat(12, 11))
    KB_ref = Matrix(KB)

    KC = EPMAfem.lazy(kron, rand_mat(9, 9), rand_mat(12, 12))
    KC_ref = Matrix(KC)

    B = EPMAfem.BlockMatrix(KA, KB, KC, rand(), rand(), rand(), rand(), 1.0)
    B_ref = Matrix(B)

    x = create_rand_vec(size(B, 2))
    @test B*x ≈ B_ref*x

    Bt = transpose(B)
    x = create_rand_vec(size(Bt, 2))
    @test Bt*x ≈ transpose(B_ref)*x
end

# stich KronMatrix and SumMatrix together
@testset let
    K1 = EPMAfem.lazy(kron, rand_mat(10, 11), rand_mat(12, 13))
    K1_ref = Matrix(K1)
    K2 = EPMAfem.lazy(kron, rand_mat(10, 11), rand_mat(12, 13))
    K2_ref = Matrix(K2)

    S1 = EPMAfem.lazy(+, K1, K2)
    S1_ref = K1_ref .+ K2_ref
    @test S1_ref ≈ Matrix(S1)

    x = rand_vec(size(S1, 2))
    @test S1_ref * x ≈ S1 * x

    S1t = transpose(S1)
    x = rand_vec(size(S1t, 2))
    @test transpose(S1_ref) * x ≈ S1t * x

    # stich the two together
    S1 = EPMAfem.lazy(+, rand_mat(10, 11), rand_mat(10, 11))
    S1_ref = Matrix(S1)

    S2 = EPMAfem.lazy(+, rand_mat(12, 13), rand_mat(12, 13))
    S2_ref = Matrix(S2)

    K1 = EPMAfem.lazy(kron, S1, S2)
    K1_ref = kron(transpose(S2_ref), S1_ref)
    @test Matrix(K1) ≈ K1_ref

    x = rand_vec(size(K1, 2))
    @test K1*x ≈ K1_ref*x

    S3 = EPMAfem.lazy(+, rand_mat(10, 11), rand_mat(10, 11))
    S3_ref = Matrix(S3)

    S4 = EPMAfem.lazy(+, rand_mat(12, 13), rand_mat(12, 13))
    S4_ref = Matrix(S4)

    K2 = EPMAfem.lazy(kron, S3, S4)
    K2_ref = kron(transpose(S4_ref), S3_ref)

    K = EPMAfem.lazy(+, K1, K2)
    K_ref = K1_ref .+ K2_ref 
    @test Matrix(K) ≈ K_ref

    x = rand_vec(size(K, 2))
    @test K*x ≈ K_ref*x

    x = rand_vec(size(K, 1))
    @test transpose(K)*x ≈ transpose(K_ref)*x
end

## ScaleMatrix
@testset let
    l = rand()
    r = rand()
    L = rand_mat(2, 3)
    R = rand_mat(2, 3)

    SL = EPMAfem.lazy(*, l, L)
    SR = EPMAfem.lazy(*, R, r)

    SL_ref = l*L
    SR_ref = R*r

    x = rand(size(SL, 2))
    @test SL * x ≈ SL_ref * x
    @test SR * x ≈ SR_ref * x

    X = rand_mat(size(SL, 2), 4)
    @test SL * X ≈ SL_ref * X
    @test SR * X ≈ SR_ref * X

    X = rand_mat(4, size(SL, 1))
    @test X * SL ≈ X * SL_ref
    @test X * SR ≈ X * SR_ref

    SLt = transpose(SL)
    SRt = transpose(SR)
    SLt_ref = transpose(SL_ref)
    SRt_ref = transpose(SR_ref)

    x = rand_vec(size(SLt, 2))
    @test SLt * x ≈ SLt_ref * x
    @test SRt * x ≈ SRt_ref * x

    X = rand_mat(size(SLt, 2), 12)
    @test SLt * X ≈ SLt_ref * X
    @test SRt * X ≈ SRt_ref * X

    X = rand_mat(12, size(SLt, 1))
    @test X * SLt ≈ X * SLt_ref
    @test X * SRt ≈ X * SRt_ref
end

## SumMatrix
@testset let
    A1 = rand_mat(10, 11)
    A2 = rand_mat(10, 11)
    A3 = rand_mat(10, 11)
    S = EPMAfem.LazyOpMatrix{eltype(A1)}(+, [A1, A2, A3])
    S_tuple = EPMAfem.lazy(+, A1, A2, A3)

    S_ref_ = A1 .+ A2 .+ A3 
    S_ref = Matrix(S)
    @test S_ref_ ≈ S_ref
    S_ref = Matrix(S_tuple)
    @test S_ref_ ≈ S_ref

    x = rand_vec(size(S, 2))
    @test S_ref * x ≈ S * x
    @test S_ref * x ≈ S_tuple * x

    X = rand_mat(size(S, 2), 12)
    @test S_ref * X ≈ S * X
    @test S_ref * X ≈ S_tuple * X

    X = rand_mat(12, size(S, 1))
    @test X * S_ref ≈ X * S
    @test X * S_ref ≈ X * S_tuple

    St = transpose(S)
    St_tuple = transpose(S_tuple)
    St_ref = transpose(S_ref)

    x = rand_vec(size(St, 2))
    @test St_ref * x ≈ St * x
    @test St_ref * x ≈ St_tuple * x

    X = rand_mat(size(St, 2), 12)
    @test St_ref * X ≈ St * X
    @test St_ref * X ≈ St_tuple * X

    X = rand_mat(12, size(St, 1))
    @test X * St_ref ≈ X * St
    @test X * St_ref ≈ X * St_tuple
end

## KronMatrix
@testset let
    A = rand_mat(10, 11)
    B = rand_mat(12, 13)

    K = EPMAfem.lazy(kron, A, B)
    K_ref_ = kron(transpose(B), A)
    K_ref = Matrix(K)
    @test K_ref_ ≈ K_ref

    # @test isdiag(K) == false

    @test size(K) == size(K_ref)

    x = rand_vec(size(K)[2])
    @test K * x ≈ K_ref * x

    y = rand_vec(size(K)[1])
    y_ref = copy(y)

    α = rand()
    β = rand()
    mul!(y, K, x, α, β)
    mul!(y_ref, K_ref, x, α, β)
    @test y ≈ y_ref

    Kt = transpose(K)
    Kt_ref = transpose(K_ref)

    @test size(Kt) == size(Kt_ref)

    x = rand_vec(size(Kt)[2])
    @test Kt * x ≈ Kt_ref * x

    y = rand_vec(size(Kt)[1])
    y_ref = copy(y)
    α = rand()
    β = rand()
    mul!(y, Kt, x, α, β)
    mul!(y_ref, Kt_ref, x, α, β)
    @test y ≈ y_ref

    # A = Diagonal(create_rand_vec(10))
    # B = Diagonal(create_rand_vec(11))

    # K_diag = EPMAfem.KronMatrix(A, B)
    # @test isdiag(K_diag) == true
end
