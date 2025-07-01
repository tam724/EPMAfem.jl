using Revise
using EPMAfem
using LinearAlgebra
using Test
using CUDA
using ConcreteStructs
using BenchmarkTools
using SparseArrays
# include("plot_overloads.jl")


# from julia 1.12
 # these make eachrow(A') produce simpler views
@inline Base.unsafe_view(A::Transpose{<:Number, <:AbstractMatrix}, i::Integer, j::AbstractArray) =
    Base.unsafe_view(parent(A), j, i)
@inline Base.unsafe_view(A::Transpose{<:Number, <:AbstractMatrix}, i::AbstractArray, j::Integer) =
    Base.unsafe_view(parent(A), j, i)

@inline Base.unsafe_view(A::Adjoint{<:Real, <:AbstractMatrix}, i::Integer, j::AbstractArray) =
    Base.unsafe_view(parent(A), j, i)
@inline Base.unsafe_view(A::Adjoint{<:Real, <:AbstractMatrix}, i::AbstractArray, j::Integer) =
    Base.unsafe_view(parent(A), j, i)


function LinearAlgebra.kron!(C::Diagonal, A::Diagonal, B::Diagonal)
    kron!(C.diag, A.diag, B.diag)
    return C
end

function LinearAlgebra.inv!(D::Diagonal{T})  where T
    D.diag .= inv.(D.diag)
    return D
end

lazy(A) = EPMAfem.lazy(A)

###### CPU TESTING
begin
    cpu_T = Float64
    rand_mat(m, n) = rand(cpu_T, m, n)
    rand_spmat(m, n, d) = sprand(cpu_T, m, n, d)
    rand_vec(n) = rand(cpu_T, n)
    ones_vec(n) = ones(cpu_T, n)
    rand_scal() = rand(cpu_T)
    scal(v) = cpu_T(v)
    cpu = true
end

### CUDA TESTING
begin
    rand_mat(m, n) = rand(m, n) |> cu
    rand_spmat(m, n, d) = sprand(m, n, d) |> cu
    rand_vec(n) = rand(n) |> cu
    ones_vec(n) = ones(n) |> cu
    rand_scal() = rand(Float32)
    scal(v) = Float32(v)
    # switches off getindex tests
    cpu = false

    function LinearAlgebra.mul!(y::AbstractVector, A::EPMAfem.AbstractLazyMatrix{T}, x::AbstractVector, α::Number, β::Number) where T
        ws = EPAMfem.create_workspace(EPMAfem.mul_with!, A, cu ∘ zeros)
        if ws_size > 0 @warn("mul!(::$(typeof(A))) allocates zeros($(T), $(ws_size))!") end
        EPMAfem.mul_with!(ws, y, A, x, α, β)
        return y
    end

    function LinearAlgebra.mul!(Y::AbstractMatrix, A::EPMAfem.AbstractLazyMatrix{T}, X::AbstractMatrix, α::Number, β::Number) where T
        @warn "Not build for this, but we try anyways..."
        ws = EPAMfem.create_workspace(EPMAfem.mul_with!, A, cu ∘ zeros)
        if ws_size > 0 @warn("mul!(::$(typeof(A))) allocates zeros($(T), $(ws_size))!") end
        EPMAfem.mul_with!(ws, Y, A, X, α, β)
        return Y
    end

    function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, A::EPMAfem.AbstractLazyMatrix{T}, α::Number, β::Number) where T
        @warn "Not build for this, but we try anyways..."
        ws = EPAMfem.create_workspace(EPMAfem.mul_with!, A, cu ∘ zeros)
        if ws_size > 0 @warn("mul!(::$(typeof(A))) allocates zeros($(T), $(ws_size))!") end
        EPMAfem.mul_with!(ws, Y, X, A, α, β)
        return Y
    end

    function LinearAlgebra.mul!(y::AbstractVector, At::Transpose{T, <:EPMAfem.AbstractLazyMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
        ws = EPAMfem.create_workspace(EPMAfem.mul_with!, At, cu ∘ zeros)
        if ws_size > 0 @warn("mul!(::$(typeof(At))) allocates zeros($(T), $(ws_size))!") end
        EPMAfem.mul_with!(ws, y, At, x, α, β)
        return y
    end

    function LinearAlgebra.mul!(Y::AbstractMatrix, At::Transpose{T, <:EPMAfem.AbstractLazyMatrix{T}}, X::AbstractMatrix, α::Number, β::Number) where T
        @warn "Not build for this, but we try anyways..."
        ws = EPAMfem.create_workspace(EPMAfem.mul_with!, At, cu ∘ zeros)
        if ws_size > 0 @warn("mul!(::$(typeof(At))) allocates zeros($(T), $(ws_size))!") end
        EPMAfem.mul_with!(ws, Y, At, X, α, β)
        return Y
    end

    function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, At::Transpose{T, <:EPMAfem.AbstractLazyMatrix{T}}, α::Number, β::Number) where T
        @warn "Not build for this, but we try anyways..."
        ws = EPAMfem.create_workspace(EPMAfem.mul_with!, At, cu ∘ zeros)
        if ws_size > 0 @warn("mul!(::$(typeof(At))) allocates zeros($(T), $(ws_size))!") end
        EPMAfem.mul_with!(ws, Y, X, At, α, β)
        return Y
    end
end

do_materialize(A::EPMAfem.AbstractLazyMatrixOrTranspose) = do_materialize(EPMAfem.materialize(A))
function do_materialize(M::EPMAfem.MaterializedMatrix)
    ws = EPMAfem.create_workspace(EPMAfem.materialize_with, M, rand_vec)
    M_mat, _ = EPMAfem.materialize_with(ws, M, nothing)
    return M_mat
end

# first call should cache, the second should use the cache (check result)
# before the last call, we modify the cache without invalidation -> wrong result
function test_cached_LK_K(LK, K)
    x = rand(size(LK, 2))
    y = rand(size(LK, 1))
    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, LK, rand_vec)
    EPMAfem.mul_with!(ws, y, LK, x, true, false)
    @test y ≈ K * x
    EPMAfem.mul_with!(ws, y, LK, x, true, false)
    @test y ≈ K * x
    for (id, (valid, mem)) in ws.cache
        @test valid[]
        mem .= rand(size(mem))
    end
    EPMAfem.mul_with!(ws, y, LK, x, true, false)
    @test !(y ≈ K * x)
end


V = EPMAfem.LazyResizeMatrix(rand(30, 30), (Ref(30), Ref(30)))
A = kron(EPMAfem.cache(transpose(V)*lazy(sprand(30, 30, 0.2) + Diagonal(rand(30)))*V), lazy(Diagonal(rand(20))))
B = kron(EPMAfem.cache(transpose(V)*lazy(sprand(30, 35, 0.2))), lazy(sprand(20, 25, 0.2)))
C = kron(lazy(Diagonal(rand(35))), lazy(Diagonal(rand(25))))

BM = EPMAfem.blockmatrix(A, B, C)



# SC = EPMAfem.schur_complement(BM);

# EPMAfem.resize!(V, (30, 30))

BMd = sparse(BM)

x = rand(size(BM, 2))
y = rand(size(BM, 1))
y2 = rand(size(BM, 1))

ws = EPMAfem.create_workspace(EPMAfem.mul_with!, BM, zeros)
@time EPMAfem.mul_with!(ws, y, BM, x, true, false);
@time mul!(y2, BMd, x, true, false);

@benchmark EPMAfem.mul_with!($ws, $y, $BM, $x, $true, $false)

@profview @benchmark EPMAfem.mul_with!($ws, $y, $BM, $x, $true, $false)
@benchmark mul!($y2, $BMd, $x, $true, $false)
@profview @benchmark mul!($y2, $BMd, $x, $true, $false)

Base.copyto_unaliased!


y ≈ y2


A = EPMAfem.lazy(*, (lazy(rand(4, 4)) for _ in 1:5)...)

A

A = Diagonal(rand(5))
B = rand(5, 6)
C = Diagonal(rand(6))

AL = lazy(A)
BL = lazy(B)
CL = lazy(C)

BM = EPMAfem.blockmatrix(AL, BL, CL)

EPMAfem.A(BM)

KL = EPMAfem.schur_complement(BM)
K = A - B * inv(C) * transpose(B)

x = rand(size(K, 2))
KL * x ≈ K * x

@enter KL * x


@testset "InplaceInverseMatrix" begin
    D = Diagonal(rand(5))
    E = Diagonal(rand(5))

    DL = lazy(D)
    EL = lazy(E)

    K = LinearAlgebra.inv!(kron(D + E, E))
    KL = LinearAlgebra.inv!(kron(DL + EL, EL))

    x = rand(size(KL, 2))

    KL * x ≈ K * x
    KL * x ≈ kron(D + E, E) \ x

    X = rand(size(KL, 2), 4)
    KL * X ≈ K * X
    KL * X ≈ kron(D + E, E) \ X

    X = rand(4, size(KL, 1))
    X * KL ≈ X * K
    X * KL ≈ X * inv(kron(D + E, E))
end

@testset "ResizeMatrix" begin
    A = rand(10, 3)
    B = rand(11, 6)
    C = rand(10, 10)
    D = rand(11, 11)

    AL = EPMAfem.LazyResizeMatrix(rand(15, 15), (Ref(10), Ref(5)))
    BL = EPMAfem.LazyResizeMatrix(rand(15, 15), (Ref(11), Ref(5)))
    CL = C |> lazy
    DL = D |> lazy
    XL = EPMAfem.materialize(transpose(AL) * CL * AL)
    YL = EPMAfem.cache(transpose(BL) * DL * BL)
    KL = kron(XL, YL)

    EPMAfem.resize!(AL, (10, 3))
    EPMAfem.resize!(BL, (11, 6))

    K = kron(transpose(A)*C*A, transpose(B)*D*B)

    AL .= A
    BL .= B

    x = rand(size(KL, 2))
    test_cached_LK_K(KL, K)
end

let
    using LinearMaps
    Ap = sprand(500, 500, 0.01)
    Am = rand(200, 200)

    km1 = Diagonal(rand(200))
    km2 = Diagonal(rand(200))

    Bpm = sprand(500, 180, 0.1)
    Qpm = sprand(200, 150, 0.1)

    Cp = Diagonal(rand(180))
    Cm = Diagonal(rand(150))

    kc1 = Diagonal(rand(150))

    LM(X) = LinearMap(X)
    lz(X) = lazy(X)

    A = kron(Ap, Am + km1 + km2)
    ALM = kron(LM(Ap), LM(Am) + LM(km1) + LM(km2))
    Alz = kron(lz(Ap), lz(Am) + lz(km1) + lz(km2))

    B = kron(Bpm, Qpm)
    BLM = kron(LM(Bpm), LM(Qpm))
    Blz = kron(lz(Bpm), lz(Qpm))

    C = kron(Cp, Cm + kc1)
    CLM = kron(LM(Cp), LM(Cm) + LM(kc1))
    Clz = kron(lz(Cp), lz(Cm) + lz(kc1))

    BM = EPMAfem.blockmatrix(A, B, C)
    BMLM = EPMAfem.blockmatrix(ALM, BLM, CLM)
    BMlz = EPMAfem.blockmatrix(Alz, Blz, Clz)

    x = rand(size(BM, 2))
    y = rand(size(BM, 1))
    yLM = rand(size(BM, 1))
    ylz = rand(size(BM, 1))

    mul!(y, BM, x)
    mul!(yLM, BMLM, x)
    mul!(ylz, BMlz, x)

    @test y ≈ yLM
    @test y ≈ ylz
    # end

    res1 = @benchmark mul!($y, $BM, $x)
    res2 = @benchmark mul!($yLM, $BMLM, $x)

    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, BMlz, zeros)
    res3 = @benchmark EPMAfem.mul_with!($ws, $ylz, $BMlz, $x, true, false)

    display(res1)
    display(res2)
    display(res3)
end

@testset "Complex Computational Graph" begin
    # Random base matrices
    A = rand_mat(4, 4)
    B = rand_mat(4, 4)
    C = rand_mat(4, 4)
    D = rand_mat(4, 4)
    E = rand_mat(4, 4)
    F = rand_mat(4, 4)
    G = rand_mat(4, 4)
    H = rand_mat(4, 4)
    J = rand_mat(16, 4)
    α = rand_scal()
    β = rand_scal()
    γ = rand_scal()

    # Lazy wrappers
    LA, LB, LC, LD, LE, LF, LG, LH, LJ = lazy.((A, B, C, D, E, F, G, H, J))

    # Compose a graph:
    M1 = (A + α*B) * (C + D)
    LM1 = (LA + α * LB) * (LC + LD)

    M2 = kron(E, F) + β * kron(G, H)
    LM2 = kron(LE, LF) + β * kron(LG, LH)

    M3 = transpose(kron(M1, M1)) * M2
    LM3 = transpose(kron(LM1, LM1)) * LM2

    M4 = γ * M3 + kron(M1, transpose(transpose(J)*M2*J))
    LM4 = γ * LM3 + kron(LM1, transpose(transpose(LJ) * LM2 * LJ));

    M5 = M4 * transpose(M4)
    LM5 = LM4 * transpose(LM4);

    x = rand_vec(size(LM5, 2))
    ws_size = EPMAfem.required_workspace(EPMAfem.mul_with!, LM5)
    ws = EPMAfem.create_workspace(ws_size, rand_vec)
    y = rand_vec(size(LM5, 1))
    EPMAfem.mul_with!(ws, y, LM5, x, true, false)
    @test y ≈ M5 * x
end

@testset "Complex Computational Graph Materialized" begin
    # Random base matrices
    A = rand_mat(4, 4)
    B = rand_mat(4, 4)
    C = rand_mat(4, 4)
    D = rand_mat(4, 4)
    E = rand_mat(4, 4)
    F = rand_mat(4, 4)
    G = rand_mat(4, 4)
    H = rand_mat(4, 4)
    J = rand_mat(16, 4)
    α = rand_scal()
    β = rand_scal()
    γ = rand_scal()

    M1 = (A + α*B) * (C + D)
    M2 = kron(E, F) + β * kron(G, H)
    M3 = transpose(kron(M1, M1)) * M2
    M4 = γ * M3 + kron(M1, transpose(transpose(J)*M2*J))
    M5 = M4 * transpose(M4)

    x = rand_vec(size(M5, 2))
    y_ref = M5 * x

    # Lazy wrappers
    LA, LB, LC, LD, LE, LF, LG, LH, LJ = lazy.((A, B, C, D, E, F, G, H, J))

    for m_fac in 0:0.1:1
        may_m(M) = rand() < m_fac ? EPMAfem.materialize(M) : M

        # Compose a graph:
        LM1 = may_m(LA + may_m(α * LB)) * may_m(LC + LD)
        LM2 = may_m(kron(LE, LF)) + may_m(β * kron(LG, LH))
        LM3 = may_m(transpose(kron(LM1, LM1)) * LM2)
        LM4 = may_m(γ * LM3 + kron(LM1, transpose(may_m(transpose(LJ) * LM2 * LJ))));
        LM5 = LM4 * transpose(LM4);

        ws_size = EPMAfem.required_workspace(EPMAfem.mul_with!, LM5)
        ws = EPMAfem.create_workspace(ws_size, rand_vec)
        y = rand_vec(size(LM5, 1))
        EPMAfem.mul_with!(ws, y, LM5, x, true, false)
        @test y ≈ y_ref
    end
end

@testset "Cached Matrix" begin
    A = rand_mat(5, 5)
    B = rand_mat(5, 5)

    LA = lazy(A)
    LB = lazy(B)

    K = A + B
    LK = LA + LB
    x = rand(size(LK, 2))
    @test LK * x ≈ K * x

    K = 2.0 * (A + B)
    LK = 2.0 * (LA + LB)
    x = rand(size(LK, 2))
    @test LK * x ≈ K * x

    K = transpose(A + B)
    LK = transpose(LA + LB)
    x = rand(size(LK, 2))
    @test LK * x ≈ K * x

    K = transpose(2.0 * (A + B))
    LK = transpose(2.0 * (LA + LB))
    x = rand(size(LK, 2))
    @test LK * x ≈ K * x

    K = 2.0 * transpose(A + B)
    LK = 2.0 * transpose(LA + LB)
    x = rand(size(LK, 2))
    @test LK * x ≈ K * x

    # now cached
    K = A + B
    LK = EPMAfem.cache(LA + LB)
    x = rand(size(LK, 2))
    test_cached_LK_K(LK, K)

    K = 2.0 * (A + B)
    LK =  EPMAfem.cache(2.0 * (LA + LB))
    x = rand(size(LK, 2))
    test_cached_LK_K(LK, K)

    K = transpose(A + B)
    LK =  EPMAfem.cache(transpose(LA + LB))
    x = rand(size(LK, 2))
    test_cached_LK_K(LK, K)

    K = transpose(2.0 * (A + B))
    LK =  EPMAfem.cache(transpose(2.0 * (LA + LB)))
    x = rand(size(LK, 2))
    test_cached_LK_K(LK, K)

    K = 2.0 * transpose(A + B)
    LK =  EPMAfem.cache(2.0 * transpose(LA + LB))
    x = rand(size(LK, 2))
    test_cached_LK_K(LK, K)
end


@testset "Cached Matrix" begin
    A = rand_mat(5, 5)
    B = rand_mat(5, 5)
    C = rand_mat(5, 5)
    D = rand_mat(5, 5)

    LA = lazy(A)
    LB = lazy(B)
    LC = lazy(C)
    LD = lazy(D)

    K = A + B
    LK = EPMAfem.cache(LA + LB)
    test_cached_LK_K(LK, K)

    K = 2.0 * transpose(A)
    LK = EPMAfem.cache(2.0 * transpose(LA))
    test_cached_LK_K(LK, K)

    K = transpose(transpose(A) + B)
    LK = EPMAfem.cache(transpose(transpose(LA) + LB))
    test_cached_LK_K(LK, K)

    K = A * B
    LK = EPMAfem.cache(LA * LB)
    test_cached_LK_K(LK, K)

    K = kron(A, B)
    LK = EPMAfem.cache(kron(LA, LB))
    test_cached_LK_K(LK, K)

    K = 2.0 * kron(A, B)
    LK = EPMAfem.cache(2.0 * kron(LA, LB))
    test_cached_LK_K(LK, K)

    K = 2.0 * transpose(kron(A, B))
    LK = EPMAfem.cache(2.0 * transpose(kron(LA, LB)))
    test_cached_LK_K(LK, K)

    K = 1.0 * A + 2.0 * B
    LK = EPMAfem.cache(1.0 * LA + 2.0 * LB)
    test_cached_LK_K(LK, K)

    K = 1.0 * A * 2.0 * B * C
    LK = EPMAfem.cache(1.0 * LA * 2.0 * LB * LC)
    test_cached_LK_K(LK, K)

    K = A * B * C * D  * A * B * C * D 
    LK = EPMAfem.cache(LA * LB * LC * LD) * EPMAfem.cache(LA * LB * LC * LD)
    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, LK, rand_vec)
    @test length(ws.cache) == 1
    test_cached_LK_K(LK, K)
end

@testset "Complex Computational Graph Cached" begin
    # Random base matrices
    A = rand_mat(4, 4)
    B = rand_mat(4, 4)
    C = rand_mat(4, 4)
    D = rand_mat(4, 4)
    E = rand_mat(4, 4)
    F = rand_mat(4, 4)
    G = rand_mat(4, 4)
    H = rand_mat(4, 4)
    J = rand_mat(16, 4)
    α = rand_scal()
    β = rand_scal()
    γ = rand_scal()

    M1 = (A + α*B) * (C + D)
    M2 = kron(E, F) + β * kron(G, H)
    M3 = transpose(kron(M1, M1)) * M2
    M4 = γ * M3 + kron(M1, transpose(transpose(J)*M2*J))
    M5 = M4 * transpose(M4)

    x = rand_vec(size(M5, 2))
    y_ref = M5 * x

    # Lazy wrappers
    LA, LB, LC, LD, LE, LF, LG, LH, LJ = lazy.((A, B, C, D, E, F, G, H, J))

    for m_fac in 0:0.1:1
        may_m(M) = rand() < m_fac ? EPMAfem.cache(M) : M

        # Compose a graph:
        LM1 = may_m(LA + α * LB) * may_m(LC + LD)
        LM2 = may_m(kron(LE, LF)) + may_m(β * kron(LG, LH))
        LM3 = may_m(transpose(kron(LM1, LM1)) * LM2)
        LM4 = may_m(γ * LM3 + kron(LM1, transpose(may_m(transpose(LJ) * LM2 * LJ))));
        LM5 = LM4 * transpose(LM4);

        ws_size = EPMAfem.required_workspace(EPMAfem.mul_with!, LM5)
        ws = EPMAfem.create_workspace(ws_size, rand_vec)
        y = rand_vec(size(LM5, 1))
        EPMAfem.mul_with!(ws, y, LM5, x, true, false)
        @test y ≈ y_ref
    end
end

@testset "Another Complex Computational Graph" begin
    # Random base matrices
    A = rand_mat(3, 3)
    B = rand_mat(3, 3)
    C = rand_mat(3, 3)
    D = rand_mat(3, 3)
    E = rand_mat(3, 3)
    F = rand_mat(3, 3)
    G = rand_mat(3, 3)
    H = rand_mat(3, 3)
    J = rand_mat(9, 9)
    α = rand_scal()
    β = rand_scal()
    γ = rand_scal()
    δ = rand_scal()

    # Lazy wrappers
    LA, LB, LC, LD, LE, LF, LG, LH, LJ = lazy.((A, B, C, D, E, F, G, H, J))

    M1 = kron(A, (A + α*B) * (C - D)) + β * kron(E, F)
    M2 = kron(G + γ*H, E*F) + δ * kron(A, B)
    M3 = transpose(M1) * M2 + J * M1 * transpose(J)
    M4 = EPMAfem.blockmatrix(M1, M2, M3)


    for m_fac in 0:0.1:1
        may_m(M) = rand() < m_fac ? EPMAfem.materialize(M) : M

        LM1 = kron(LA, may_m((LA + α*LB) * may_m(LC - LD))) + β * kron(LE, LF)
        LM2 = may_m(kron(LG + γ*LH, LE*LF) + δ * kron(LA, LB))
        LM3 = may_m(transpose(LM1) * LM2) + may_m(LJ * LM1 * transpose(LJ))
        LM4 = EPMAfem.blockmatrix(LM1, LM2, LM3)

        x = rand_vec(size(LM4, 2))
        ws_size = EPMAfem.required_workspace(EPMAfem.mul_with!, LM4)
        ws = EPMAfem.create_workspace(ws_size, rand_vec)
        y = rand_vec(size(LM4, 1))
        EPMAfem.mul_with!(ws, y, LM4, x, true, false)
        @test y ≈ M4 * x

        x = rand_vec(size(LM4, 1))
        ws_size = EPMAfem.required_workspace(EPMAfem.mul_with!, LM4)
        ws = EPMAfem.create_workspace(ws_size, rand_vec)
        y = rand_vec(size(LM4, 2))
        EPMAfem.mul_with!(ws, y, transpose(LM4), x, true, false)
        @test y ≈ transpose(M4) * x

        @test Matrix(LM4) ≈ Matrix(M4)
    end
end

@testset "Another Complex Computational Graph Cached" begin
    # Random base matrices
    A = rand_mat(3, 3)
    B = rand_mat(3, 3)
    C = rand_mat(3, 3)
    D = rand_mat(3, 3)
    E = rand_mat(3, 3)
    F = rand_mat(3, 3)
    G = rand_mat(3, 3)
    H = rand_mat(3, 3)
    J = rand_mat(9, 9)
    α = rand_scal()
    β = rand_scal()
    γ = rand_scal()
    δ = rand_scal()

    # Lazy wrappers
    LA, LB, LC, LD, LE, LF, LG, LH, LJ = lazy.((A, B, C, D, E, F, G, H, J))

    M1 = kron(A, (A + α*B) * (C - D)) + β * kron(E, F)
    M2 = kron(G + γ*H, E*F) + δ * kron(A, B)
    M3 = transpose(M1) * M2 + J * M1 * transpose(J)
    M4 = EPMAfem.blockmatrix(M1, M2, M3)


    for m_fac in 0:0.1:1
        may_m(M) = rand() < m_fac ? EPMAfem.cache(M) : M

        LM1 = kron(LA, may_m((LA + α*LB) * may_m(LC  + scal(-1)*LD))) + β * kron(LE, LF)
        LM2 = may_m(kron(LG + γ*LH, LE*LF) + δ * kron(LA, LB))
        LM3 = may_m(transpose(LM1) * LM2) + may_m(LJ * LM1 * transpose(LJ))
        LM4 = EPMAfem.blockmatrix(LM1, LM2, LM3)

        x = rand_vec(size(LM4, 2))
        ws_size = EPMAfem.required_workspace(EPMAfem.mul_with!, LM4)
        ws = EPMAfem.create_workspace(ws_size, rand_vec)
        y = rand_vec(size(LM4, 1))
        EPMAfem.mul_with!(ws, y, LM4, x, true, false)
        @test y ≈ M4 * x

        x = rand_vec(size(LM4, 1))
        ws_size = EPMAfem.required_workspace(EPMAfem.mul_with!, LM4)
        ws = EPMAfem.create_workspace(ws_size, rand_vec)
        y = rand_vec(size(LM4, 2))
        EPMAfem.mul_with!(ws, y, transpose(LM4), x, true, false)
        @test y ≈ transpose(M4) * x

        @test Matrix(LM4) ≈ Matrix(M4)
    end
end

@testset "ProductChain Matrix" begin
    A_ = rand_mat(10, 11)
    B_ = rand_mat(11, 12)
    C_ = rand_mat(12, 13)
    D_ = rand_mat(13, 14)
    E_ = rand_mat(14, 15)
    A, B, C, D, E = lazy.((A_, B_, C_, D_, E_))

    M2_ = A_ * B_
    M2 = A * B

    M3_ = A_ * B_ * C_
    M3 = A * B * C

    M4_ = A_ * B_ * C_ * D_
    M4 = A * B * C * D

    M5_ = A_ * B_ * C_ * D_ * E_
    M5 = A * B * C * D * E

    for (M, M_) in [(M2, M2_), (M3, M3_), (M4, M4_), (M5, M5_)]
        # we have no entry point for (Matrix) * (LazyMatrix) multiplication (this is not how it was designed..) however, internally we do at least try..
        ws_size = EPMAfem.required_workspace(EPMAfem.mul_with!, M)
        ws = EPMAfem.create_workspace(ws_size, rand_vec)

        # Y = X * M
        X = rand_mat(rand(3:30), size(M, 1))
        Y = rand_mat(size(X, 1), size(M, 2))
        EPMAfem.mul_with!(ws, Y, X, M, true, false)
        @test Y ≈ X * M_

        # Y = X * transpose(M)
        Mt = transpose(M)
        X = rand_mat(rand(3:30), size(Mt, 1))
        Y = rand_mat(size(X, 1), size(Mt, 2))
        EPMAfem.mul_with!(ws, Y, X, Mt, true, false)
        @test Y ≈ X * transpose(M_)
    end
end

# Product Matrix * Matrix
@testset "ProductChain Matrix2" begin
    A_ = rand_mat(10, 13)
    B_ = rand_mat(13, 15)
    C_ = rand_mat(15, 9)
    D_ = rand_mat(9, 16)
    E_ = rand_mat(16, 20)

    A, B, C, D, E = lazy.((A_, B_, C_, D_, E_))

    P2_ = A_ * B_
    P3_ = A_ * B_ * C_
    P4_ = A_ * B_ * C_ * D_
    P5_ = A_ * B_ * C_ * D_ * E_ 
    
    P2 = A * B
    P3 = A * B * C
    P4 = A * B * C * D
    P5 = A * B * C * D * E

    x = rand_vec(size(P2, 2))
    @test P2 * x ≈ P2_ * x

    X = rand_mat(size(P2, 2), rand(3:30))
    @test P2 * X ≈ P2_ * X
    X = rand_mat(size(P2, 2), 1) # also test with single dimension matrix
    @test P2 * X ≈ P2_ * X
    
    X = rand_mat(size(P3, 2), rand(3:30))
    @test P3 * X ≈ P3_ * X
    X = rand_mat(size(P3, 2), 1)
    @test P3 * X ≈ P3_ * X

    X = rand_mat(size(P4, 2), rand(3:30))
    @test P4 * X ≈ P4_ * X
    X = rand_mat(size(P4, 2), 1)
    @test P4 * X ≈ P4_ * X

    X = rand_mat(size(P5, 2), rand(3:30))
    @test P5 * X ≈ P5_ * X
    X = rand_mat(size(P5, 2), 1)
    @test P5 * X ≈ P5_ * X

    for (P, P_) in [(P2, P2_), (P3, P3_), (P4, P4_), (P5, P5_)]
        X = rand_mat(size(P, 2), rand(3:10))
        Y = rand_mat(size(P, 1), size(X, 2))
        ws = EPMAfem.create_workspace(EPMAfem.mul_with!, P, rand_vec)
        @show ws.workspace |> length
        EPMAfem.mul_with!(ws, Y, P, X, true, false)
        @test Y ≈ P_ * X

        X = rand_mat(size(P, 2), 1)
        Y = rand_mat(size(P, 1), size(X, 2))
        ws = EPMAfem.create_workspace(EPMAfem.mul_with!, P, rand_vec)
        @show ws.workspace |> length
        EPMAfem.mul_with!(ws, Y, P, X, true, false)
        @test Y ≈ P_ * X
    end
    
    P2t, P3t, P4t, P5t = (P2, P3, P4, P5) .|> transpose

    for (Pt, P_) in [(P2t, P2_), (P3t, P3_), (P4t, P4_), (P5t, P5_)]
        X = rand_mat(size(Pt, 2), rand(3:30))
        @test Pt*X ≈ transpose(P_)*X
        X = rand_mat(size(Pt, 2), 1)
        @test Pt*X ≈ transpose(P_)*X
    end
end

@testset "robin1" begin # ROBIN 1
    A_ = rand_mat(2, 2)
    B_ = rand_mat(2, 4)
    C_ = rand_mat(4, 2)
    D_ = rand_mat(2, 3)
    E_ = rand_mat(3, 2)
    F_ = rand_mat(2, 2)
    G_ = rand_mat(2, 2)

    A, B, C, D, E, F, G = lazy.((A_, B_, C_, D_, E_, F_, G_))

    K_ = kron(A_*B_*C_ + D_*E_*A_, F_ + G_)
    K = kron(A*B*C + D*E*A, F + G)

    x = rand_vec(4)
    @test K * x ≈ K_ * x
    x = rand_vec(size(K, 1))
    @test transpose(K) * x ≈ transpose(K_) * x
end

@testset "robin2" begin # ROBIN 2
    A_ = rand_mat(3, 2)
    B_ = rand_mat(3, 4)
    C_ = rand_mat(2, 2)
    D_ = rand_mat(5, 2)
    E_ = rand_mat(3, 4)
    
    A, B, C, D, E = lazy.((A_, B_, C_, D_, E_))

    M_ = kron(E_, transpose(transpose(A_)*B_)*C_*transpose(D_))
    M = kron(E, transpose(transpose(A)*B)*C*transpose(D))

    x = rand_vec(size(M, 2))
    @test M * x ≈ M_ * x
    x = rand_vec(size(M, 1))
    @test transpose(M) * x ≈ transpose(M_) * x
end

@testset "Diagonal bubbling" begin # lets test how far Diagonal bubbles..
    A_ = Diagonal(rand_vec(10))
    A = A_ |> lazy
    # no need to test the wrapper

    B_ = 10.0 * A_
    B = 10.0 * A
    @test B_ ≈ B
    @test do_materialize(B) isa Diagonal
    @test B_ ≈ do_materialize(B)
    x = rand_vec(size(B, 2))
    @test B_*x ≈ B*x

    D_ = Diagonal(rand_vec(5))
    D = D_ |> lazy
    # no need to test

    E_ = Diagonal(rand_vec(2))
    E = E_ |> lazy
    # no need to test

    F_ = kron(D_, E_)
    F = kron(D, E)
    @test F_ ≈ F
    @test F_ ≈ do_materialize(F)
    @test do_materialize(F) isa Diagonal
    x = rand_vec(size(F, 2))
    @test F_*x ≈ F*x

    G_ = A_ * B_ + F_
    G = A * B + F
    @test G_ ≈ G
    @test G_ ≈ do_materialize(G)
    @test do_materialize(G) isa Diagonal
    x = rand_vec(size(G, 2))
    @test G_*x ≈ G*x

    H_ = Diagonal(rand_vec(10))
    H = H_ |> lazy
    # no need to test

    I_ = H_ * G_
    I = H * G
    @test I_ ≈ I
    @test I_ ≈ do_materialize(I)
    @test do_materialize(I) isa Diagonal
    x = rand_vec(size(I, 2))
    @test I_*x ≈ I*x

    J_ = Diagonal(rand_vec(10))
    J = J_ |> lazy
    # no need to test

    K_ = J_ * I_ * transpose(J_)
    K = J * I * transpose(J)
    @test K_ ≈ K
    @test K_ ≈ do_materialize(K)
    @test do_materialize(K) isa Diagonal
    x = rand_vec(size(K, 2))
    @test K_*x ≈ K*x

    L_ = rand_mat(10, 10)
    L = L_ |> lazy

    M_ = EPMAfem.blockmatrix(K_, L_, I_)
    M = EPMAfem.blockmatrix(K, L, I);
    @test M_ ≈ M
    # currently not implemented, but multiplication is!
    # @test M_ ≈ do_materialize(M)
    x = rand_vec(size(M, 2))
    @test M_*x ≈ M*x

    N_ = kron(M_, M_)
    N = kron(M, M);
    # TODO: skip this test for now (it only calls getindex which has tons of output..)
    # @test N_ ≈ N
    # not implemented, ...
    # @test N_ ≈ do_materialize(N)
    # x = rand(size(N, 2))
    # @test N_*x ≈ N*x

    O_ = rand_mat(5, 4)
    P_ = rand_mat(4, 5)
    O, P = lazy.((O_, P_))

    Q_ = M_ + 10.0*kron(O_, P_)
    Q = M + 10.0*kron(O, P)
    @test Q_ ≈ Q
    # currently not implemented, but multiplication is
    # @test Q_ ≈ do_materialize(Q)
    x = rand_vec(size(Q, 2))
    @test Q_*x ≈ Q*x
end

# i just wanted to build a weird structure...
@testset "Weird structure" begin
    A_ = rand_mat(10, 10)
    A = A_ |> lazy
    # no need to test the wrapper

    B_ = 10.0 * A_
    B = 10.0 * A
    @test B isa EPMAfem.AbstractLazyMatrix
    @test B_ ≈ B
    @test B_ ≈ do_materialize(B)
    x = rand_vec(size(B, 2))
    @test B_*x ≈ B*x

    D_ = rand_mat(5, 5)
    D = D_ |> lazy
    # no need to test

    E_ = rand_mat(2, 2)
    E = E_ |> lazy
    # no need to test

    F_ = kron(D_, E_)
    F = kron(D, E)
    @test F_ ≈ F
    @test F_ ≈ do_materialize(F)
    x = rand_vec(size(F, 2))
    @test F_*x ≈ F*x

    G_ = A_ * B_ + F_
    G = A * B + F
    @test G_ ≈ G
    @test G_ ≈ do_materialize(G)
    x = rand_vec(size(G, 2))
    @test G_*x ≈ G*x

    H_ = rand_mat(10, 10)
    H = H_ |> lazy
    # no need to test

    I_ = H_ * G_
    I = H * G
    @test I_ ≈ I
    @test I_ ≈ do_materialize(I)
    x = rand_vec(size(I, 2))
    @test I_*x ≈ I*x

    J_ = rand_mat(2, 10)
    J = J_ |> lazy
    # no need to test

    K_ = J_ * I_ * transpose(J_)
    K = J * I * transpose(J)
    @test K_ ≈ K
    @test K_ ≈ do_materialize(K)
    x = rand_vec(size(K, 2))
    @test K_*x ≈ K*x

    L_ = rand_mat(2, 10)
    L = L_ |> lazy

    M_ = EPMAfem.blockmatrix(K_, L_, I_)
    M = EPMAfem.blockmatrix(K, L, I)
    @test M_ ≈ M
    # currently not implemented, but multiplication is!
    # @test M_ ≈ do_materialize(M)
    x = rand_vec(size(M, 2))
    @test M_*x ≈ M*x

    N_ = kron(M_, M_)
    N = kron(M, M)
    # TODO: skip this test for now (it only calls getindex,  which has too much output)
    # @test N_ ≈ N
    # not implemented, ...
    # @test N_ ≈ do_materialize(N)
    # x = rand(size(N, 2))
    # @test N_*x ≈ N*x

    O_ = rand_mat(3, 4)
    P_ = rand_mat(4, 3)
    O, P = lazy.((O_, P_))

    Q_ = M_ + 10.0*kron(O_, P_)
    Q = M + 10.0*kron(O, P)
    @test Q_ ≈ Q
    # currently not implemented, but multiplication is
    # @test Q_ ≈ do_materialize(Q)
    x = rand_vec(size(Q, 2))
    @test Q_*x ≈ Q*x
end

@testset "Inner Allocating ProdMatrix" begin # check inner allocating promatrix
    A_ = rand_mat(3, 10)
    B_ = rand_mat(11, 8)
    C_ = rand_mat(4, 16)
    D_ = rand_mat(20, 2)
    E_ = rand_mat(4, 100)
    F_ = rand_mat(8, 3)
    G_ = rand_mat(50, 5)
    H_ = rand_mat(6, 5)
    IJ_ = rand_mat(25, 500)

    A, B, C, D, E, F, G, H, IJ = lazy.((A_, B_, C_, D_, E_, F_, G_, H_, IJ_))

    M2 = EPMAfem.lazy(*, kron(A, B), kron(C, D))
    M3 = EPMAfem.lazy(*, kron(A, B), kron(C, D),  kron(E, F));
    M4 = EPMAfem.lazy(*, kron(A, B), kron(C, D), kron(E, F), kron(G, H));
    M5 = EPMAfem.lazy(*, kron(A, B), kron(C, D), kron(E, F), kron(G, H), IJ);
    M2_ref = kron(A_, B_) * kron(C_, D_)
    M3_ref = kron(A_, B_) * kron(C_, D_) * kron(E_, F_)
    M4_ref = kron(A_, B_) * kron(C_, D_) * kron(E_, F_) * kron(G_, H_)
    M5_ref = kron(A_, B_) * kron(C_, D_) * kron(E_, F_) * kron(G_, H_) * IJ_

    x = rand_vec(size(M2, 2))
    @test M2*x ≈ M2_ref * x
    x = rand_vec(size(M3, 2))
    @test M3*x ≈ M3_ref * x
    x = rand_vec(size(M4, 2))
    @test M4*x ≈ M4_ref * x
    x = rand_vec(size(M5, 2))
    @test M5*x ≈ M5_ref * x

    M2t = transpose(M2)
    M3t = transpose(M3);
    M4t = transpose(M4);
    M5t = transpose(M5);

    x = rand_vec(size(M2t, 2))
    @test M2t*x ≈ transpose(M2_ref) * x
    x = rand_vec(size(M3t, 2))
    @test M3t*x ≈ transpose(M3_ref) * x
    x = rand_vec(size(M4t, 2))
    @test M4t*x ≈ transpose(M4_ref) * x
    x = rand_vec(size(M5t, 2))
    @test M5t*x ≈ transpose(M5_ref) * x

    MM2 = EPMAfem.materialize(M2)
    MM3 = EPMAfem.materialize(M3)
    MM4 = EPMAfem.materialize(M4)
    MM5 = EPMAfem.materialize(M5)

    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, MM2, rand_vec)
    @show length(ws.workspace)
    MM2_mat, _ = EPMAfem.materialize_with(ws, MM2, nothing)
    @test MM2_mat ≈ M2_ref

    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, MM3, rand_vec)
    @show length(ws.workspace)
    MM3_mat, _ = EPMAfem.materialize_with(ws, MM3, nothing)
    @test MM3_mat ≈ M3_ref

    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, MM4, rand_vec)
    @show length(ws.workspace)
    MM4_mat, _ = EPMAfem.materialize_with(ws, MM4, nothing)
    @test MM4_mat ≈ M4_ref

    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, MM5, rand_vec)
    @show length(ws.workspace)
    MM5_mat, _ = EPMAfem.materialize_with(ws, MM5, nothing)
    @test MM5_mat ≈ M5_ref
end

@testset "Inner Allocating TwoProdMatrix" begin # check inner allocating TwoProdMatrix
    A_ = rand_mat(5, 5)
    B_ = rand_mat(5, 5)
    C_ = rand_mat(5, 5)
    D_ = rand_mat(5, 5)

    A, B, C, D = lazy.((A_, B_, C_, D_))

    M = EPMAfem.lazy(*, kron(A, B), kron(C, D));
    M_ref = kron(A_, B_) * kron(C_, D_)

    x = rand_vec(size(M, 2))
    @test M*x ≈ M_ref * x

    Mt = transpose(M)

    x = rand_vec(size(Mt, 2))
    @test Mt*x ≈ transpose(M_ref)*x

    MM = EPMAfem.materialize(M)

    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, MM, rand_vec)
    MM_mat, rem = EPMAfem.materialize_with(ws, MM, nothing)
    @test MM_mat ≈ M_ref
end

# matrix product chain
@testset "Product Chain" begin
    A_ = rand_mat(4, 5)
    B_ = rand_mat(5, 6)
    C_ = rand_mat(6, 7)
    D_ = rand_mat(7, 8)
    E_ = rand_mat(8, 9)
    F_ = rand_mat(9, 10)

    A, B, C, D, E, F = lazy.((A_, B_, C_, D_, E_, F_))

    M = EPMAfem.lazy(*, A, B, C, D, E, F) 
    M_ref = A_ * B_ * C_ * D_ * E_ * F_

    x = rand_vec(size(M, 2))
    @test M*x  ≈ M_ref * x

    Mt = transpose(M)
    x = rand_vec(size(Mt, 2))
    @test Mt * x ≈ transpose(M_ref) * x

    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, M, rand_vec)

    x = rand_vec(size(M, 2))
    y = rand_vec(size(M, 1))
    EPMAfem.mul_with!(ws, y, M, x, true, false)
    @test y ≈ M_ref * x
    # @profview @benchmark EPMAfem.mul_with!($ws, $y, $M, $x, $true, $false)

    x = rand_vec(size(Mt, 2))
    y = rand_vec(size(Mt, 1))
    EPMAfem.mul_with!(ws, y, Mt, x, true, false)
    # @profview @benchmark EPMAfem.mul_with!($ws, $y, $Mt, $x, $true, $false)
    @test y ≈ transpose(M_ref) * x

    MM = EPMAfem.materialize(M)
    ws = EPMAfem.create_workspace(EPMAfem.materialize_with, MM, rand_vec)

    MM_mat, rem = EPMAfem.materialize_with(ws, MM, nothing)

    @test MM_mat ≈ M_ref
end

# TwoProdMatrix
@testset "TwoProdMatrix" begin
    A_ = rand_mat(4, 5)
    B_ = rand_mat(5, 7)

    A = A_ |> lazy
    B = B_ |> lazy

    C = A*B
    x = rand_vec(size(C, 2))
    @test A_*B_*x ≈ C*x

    Ct = transpose(C)
    x = rand_vec(size(Ct, 2))
    @test transpose(B_)*transpose(A_)*x ≈ Ct*x

    C = EPMAfem.materialize(A*B)
    x = rand_vec(size(C, 2))
    A_*B_*x ≈ C*x

    Ct = EPMAfem.materialize(transpose(C))
    x = rand_vec(size(Ct, 2))
    @test transpose(B_)*transpose(A_)*x ≈ Ct*x
end

# lets implement something that somewhat resembles EPMA
@testset "Small EPMA" begin
    nxp = 500
    nxm = 300
    nΩp = 120
    nΩm = 150

    ne = 3
    nd = 2

    ρp_ = [sprand(nxp, nxp, 0.01) for i in 1:ne]
    ρm_ = [Diagonal(rand(nxm)) for i in 1:ne]


    Ip_temp = Diagonal(ones(nΩp))
    Ip_ = [Ip_temp for i in 1:ne]
    Im_temp = Diagonal(ones(nΩm))
    Im_ = [Im_temp for i in 1:ne]

    ∇d_ = [sprand(nxp, nxm, 0.01) for i in 1:nd]
    Ωd_ = [sprand(nΩm, nΩp, 0.01) for i in 1:nd]

    kp_ = [Diagonal(rand(nΩp)) for i in 1:ne]
    km_ = [Diagonal(rand(nΩm)) for i in 1:ne]

    ρp, ρm, Ip, Im, ∇d, Ωd, kp, km = lazy.(ρp_), lazy.(ρm_), lazy.(Ip_), lazy.(Im_), lazy.(∇d_), lazy.(Ωd_), lazy.(kp_), lazy.(km_)

    A = sum(kron(ρp[i], Ip[i] + kp[i]) for i in 1:ne)
    C = sum(kron(ρm[i], Im[i] + km[i]) for i in 1:ne)
    B = sum(kron(∇d[i], transpose(Ωd[i])) for i in 1:nd)

    BM = EPMAfem.blockmatrix(A, B, C)

    x = rand_vec(size(BM, 2))
    y = rand_vec(size(BM, 1))
    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, BM, rand_vec)
    EPMAfem.mul_with!(ws, y, BM, x, true, false)
    # @profview EPMAfem.mul_with!(ws, y, BM, x, true, false)
    # @profview @benchmark EPMAfem.mul_with!($ws, $y, $BM, $x, $true, $false)

    @time A_ = sum(kron(ρp_[i], Ip_[i] + kp_[i]) for i in 1:ne)
    @time C_ = sum(kron(ρm_[i], Im_[i] + km_[i]) for i in 1:ne)
    @time B_ = sum(kron(∇d_[i], transpose(Ωd_[i])) for i in 1:nd)

    BM_ = EPMAfem.blockmatrix(A_, B_, C_)
    @show (BM_.nzval |> length)/prod(size(BM_))
    @time BM_ = EPMAfem.blockmatrix(A_, B_, C_)
    @test y ≈ BM_ * x

    y_ = rand_vec(size(BM_, 2))

    bench_sparse = @benchmark mul!($y_, $BM_, $x)
    bench_lazy = @benchmark EPMAfem.mul_with!($ws, $y, $BM, $x, $true, $false)

    @show "Lazy: (900μs)"
    display(bench_lazy)
    @show "Sparse (2.7ms)"
    display(bench_sparse)
end

# materialize kron stays diagonal (we need this for schur)
@testset "Diagonal + Kron" begin
    A_ = Diagonal(rand_vec(5))
    B_ = Diagonal(rand_vec(5))

    A = lazy(A_)
    B = lazy(B_)

    D = EPMAfem.materialize(kron(A, B))
    @test EPMAfem.isdiagonal(D)

    ws = EPMAfem.create_workspace(EPMAfem.materialize_with, D, rand_vec)

    @test ws.workspace |> length == 25
    D_mat, rem_ws = EPMAfem.materialize_with(ws, D, nothing)
    @test D_mat isa Diagonal
    @test D_mat.diag ≈ kron(A_, B_).diag

    D_big = EPMAfem.materialize(kron(kron(A, B), scal(3.0) * kron(A, B)))
    ws = EPMAfem.create_workspace(EPMAfem.materialize_with, D_big, rand_vec)

    ws.workspace |> length

    D_big_mat, rem_ws = EPMAfem.materialize_with(ws, D_big, nothing)
    EPMAfem.materialize_with(ws, D_big, nothing)

    @test D_big_mat isa Diagonal
    @test D_big_mat.diag ≈ kron(kron(A_, B_), scal(3.0) * kron(A_, B_)).diag
end

# test materialize
@testset "Simple Materialize" begin
    a_ = rand_mat(3, 4)
    b_ = rand_mat(5, 6)

    c_ = rand_mat(3, 4)
    d_ = rand_mat(5, 6)

    a, b, c, d = lazy.((a_, b_, c_, d_))

    k1 = kron(scal(3.0) * a, b)
    k2 = kron(scal(3.0) *c, EPMAfem.materialize(scal(2.0) * d))

    s = EPMAfem.materialize(k1 + k2)

    EPMAfem.required_workspace(EPMAfem.mul_with!, s)
    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, s, rand_vec)

    x = rand_vec(size(s, 2))
    y = rand_vec(size(s, 1))

    EPMAfem.mul_with!(ws, y, s, x, true, false)

    test = kron(3.0*a_, b_) + kron(3.0*c_, 2.0*d_)
    @test do_materialize(s) ≈ test
end

function fair_comparison(y, x, temp1, temp2, temp3, a, b, c, d, e, f, g)
    temp1 .= a .+ b
    temp2 .= c .+ d .+ e .+ f .+ g
    mul!(temp3, temp2, reshape(x, (size(temp2, 2), size(temp1, 2))), true, false)
    mul!(reshape(y, (size(temp2, 1), size(temp1, 1))), temp3, transpose(temp1), true, false)
    return y
end

# benchmark a small example
# @testset "Small Benchmark" begin
let
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
    y_ = rand_vec(size(K, 1))
    y = rand_vec(size(K, 1))
    yM = rand_vec(size(K, 1))


    ws = EPMAfem.create_workspace(EPMAfem.mul_with!, K, rand_vec)
    wsM = EPMAfem.create_workspace(EPMAfem.mul_with!, KM, rand_vec)

    ws.workspace |> size
    wsM.workspace |> size

    EPMAfem.mul_with!(ws, y, K, x, true, false)
    EPMAfem.mul_with!(wsM, yM, KM, x, true, false)
    LinearAlgebra.mul!(y_, K_, x, true, false)
    @test y ≈ y_
    @test yM ≈ y_

    temp1 = rand_mat(size(a_)...)
    temp2 = rand_mat(size(c_)...)
    temp3 = rand_mat(size(temp2, 1), size(temp1, 2))

    yfair = rand_vec(size(K, 1))
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

    # on cuda this is not true...
    @test time(slow) > time(baseline_lazy) > time(speedy_lazy) 
    @test time(baseline_lazy) > time(speedy) 
    @show "Lazy (should be equally fast ~ 40μs)"
    display(speedy_lazy)
    @show "fair comparison"
    display(speedy)

    # EPMAfem.mul_with!(wsM, yM, KM, x, true, false)
end

# BlockMatrix
@testset "BlockMatrix" begin
    a = rand_mat(10, 10)
    b = rand_mat(10, 15)
    c = rand_mat(15, 15)

    B = EPMAfem.lazy(EPMAfem.blockmatrix, a, b, c)
    B_ref = [
        a b
        transpose(b) c
    ]

    if cpu @test B_ref ≈ Matrix(B) end

    x = rand_vec(size(B, 2))
    @test B_ref*x ≈ B*x

    Bt = transpose(B)
    Bt_ref = transpose(B_ref)
    x = rand_vec(size(Bt, 2))
    @test Bt_ref*x ≈ Bt*x
end

# stich KronMatrix and BlockMatrix together
@testset "KronMatrix + BlockMatrix" begin
    KA = EPMAfem.lazy(kron, rand_mat(10, 10), rand_mat(11, 11))
    KA_ref = do_materialize(KA)

    KB = EPMAfem.lazy(kron, rand_mat(10, 9), rand_mat(12, 11))
    KB_ref = do_materialize(KB)

    KC = EPMAfem.lazy(kron, rand_mat(9, 9), rand_mat(12, 12))
    KC_ref = do_materialize(KC)

    B = EPMAfem.blockmatrix(KA, KB, KC)
    B_ref = EPMAfem.blockmatrix(KA_ref, KB_ref, KC_ref)

    x = rand_vec(size(B, 2))
    @test B*x ≈ B_ref*x

    Bt = transpose(B)
    x = rand_vec(size(Bt, 2))
    @test Bt*x ≈ transpose(B_ref)*x
end

# stich KronMatrix and SumMatrix together
@testset "KronMatrix + SumMatrix" begin
    K1 = EPMAfem.lazy(kron, rand_mat(10, 11), rand_mat(12, 13))
    K1_ref = do_materialize(K1)
    K2 = EPMAfem.lazy(kron, rand_mat(10, 11), rand_mat(12, 13))
    K2_ref = do_materialize(K2)

    S1 = EPMAfem.lazy(+, K1, K2)
    S1_ref = K1_ref .+ K2_ref
    @test S1_ref ≈ do_materialize(S1)

    x = rand_vec(size(S1, 2))
    @test S1_ref * x ≈ S1 * x

    S1t = transpose(S1)
    x = rand_vec(size(S1t, 2))
    @test transpose(S1_ref) * x ≈ S1t * x

    # stich the two together
    S1 = EPMAfem.lazy(+, rand_mat(10, 11), rand_mat(10, 11))
    S1_ref = do_materialize(S1)

    S2 = EPMAfem.lazy(+, rand_mat(12, 13), rand_mat(12, 13))
    S2_ref = do_materialize(S2)

    K1 = EPMAfem.lazy(kron, S1, S2)
    K1_ref = kron(transpose(S2_ref), S1_ref)
    @test do_materialize(K1) ≈ K1_ref

    x = rand_vec(size(K1, 2))
    @test K1*x ≈ K1_ref*x

    S3 = EPMAfem.lazy(+, rand_mat(10, 11), rand_mat(10, 11))
    S3_ref = do_materialize(S3)

    S4 = EPMAfem.lazy(+, rand_mat(12, 13), rand_mat(12, 13))
    S4_ref = do_materialize(S4)

    K2 = EPMAfem.lazy(kron, S3, S4)
    K2_ref = kron(transpose(S4_ref), S3_ref)

    K = EPMAfem.lazy(+, K1, K2)
    K_ref = K1_ref .+ K2_ref 
    @test do_materialize(K) ≈ K_ref

    x = rand_vec(size(K, 2))
    @test K*x ≈ K_ref*x

    x = rand_vec(size(K, 1))
    @test transpose(K)*x ≈ transpose(K_ref)*x
end

## ScaleMatrix
@testset "ScaleMatrix" begin
    l = rand_scal()
    r = rand_scal()
    L = rand_mat(2, 3)
    R = rand_mat(2, 3)

    SL = EPMAfem.lazy(*, l, L)
    SR = EPMAfem.lazy(*, R, r)

    SL_ref = l*L
    SR_ref = R*r

    x = rand_vec(size(SL, 2))
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
@testset "SumMatrix" begin
    A1 = rand_mat(10, 11)
    A2 = rand_mat(10, 11)
    A3 = rand_mat(10, 11)
    S = EPMAfem.LazyOpMatrix{eltype(A1)}(+, [A1, A2, A3])
    S_tuple = EPMAfem.lazy(+, A1, A2, A3)

    S_ref = A1 .+ A2 .+ A3
    if cpu
        S_ref_ = Matrix(S)
        @test S_ref_ ≈ S_ref
        S_ref = Matrix(S_tuple)
        @test S_ref_ ≈ S_ref
    end

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
@testset "KronMatrix" begin
    A = rand_mat(10, 11)
    B = rand_mat(12, 13)

    K = EPMAfem.lazy(kron, A, B)
    K_ref = kron(transpose(B), A)
    if cpu @test Matrix(K) ≈ K_ref end
    # @test K_ref_ ≈ K_ref

    # @test isdiag(K) == false

    @test size(K) == size(K_ref)

    x = rand_vec(size(K)[2])
    @test K * x ≈ K_ref * x

    y = rand_vec(size(K)[1])
    y_ref = copy(y)

    α = rand_scal()
    β = rand_scal()
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
    α = rand_scal()
    β = rand_scal()
    mul!(y, Kt, x, α, β)
    mul!(y_ref, Kt_ref, x, α, β)
    @test y ≈ y_ref

    # A = Diagonal(create_rand_vec(10))
    # B = Diagonal(create_rand_vec(11))

    # K_diag = EPMAfem.KronMatrix(A, B)
    # @test isdiag(K_diag) == true
end

# Additional testsets for edge cases and less common scenarios (thanks AI :D )
@testset "Empty and singleton matrices" begin
    # Empty matrices
    A_ = rand_mat(0, 5)
    B_ = rand_mat(5, 0)
    C_ = rand_mat(0, 0)
    A, B, C = lazy.((A_, B_, C_))
    @test size(A * B) == (0, 0)
    @test size(B * C) == (5, 0)
    @test size(C * A) == (0, 5)
    if cpu
        @test Matrix(A) == A_
        @test Matrix(B) == B_
        @test Matrix(C) == C_
    end
    # Singleton matrices
    D_ = rand_mat(1, 1)
    D = lazy(D_)
    @test D_ * D_ ≈ do_materialize(D * D)
    if cpu @test Matrix(D) ≈ D_ end
    @test transpose(D_) * transpose(D_) ≈ do_materialize(transpose(D) * transpose(D))
end

# TODO: maybe ? did not think about this yet
# @testset "Type stability and promotion" begin
#     A_ = rand(Float32, 3, 3)
#     B_ = rand(Int, 3, 3)
#     A, B = lazy.((A_, B_))
#     C = A + B
#     @test eltype(Matrix(C)) == promote_type(Float32, Int)
#     D = A * B
#     @test eltype(Matrix(D)) == promote_type(Float32, Int)
# end

@testset "Sparse and dense interop" begin
    A_ = rand_spmat(5, 5, 0.2)
    B_ = rand_mat(5, 5)
    A, B = lazy.((A_, B_))
    if cpu
        @test Matrix(A + B) ≈ A_ + B_
        @test Matrix(A * B) ≈ A_ * B_
        @test Matrix(B * A) ≈ B_ * A_
    end
    # Test multiplication with a vector x
    x = rand_vec(5)
    @test (A + B) * x ≈ (A_ + B_) * x
    @test (A * B) * x ≈ (A_ * B_) * x
    @test (B * A) * x ≈ (B_ * A_) * x
end

@testset "Chained lazy operations" begin
    A_ = rand_mat(4, 4)
    B_ = rand_mat(4, 4)
    C_ = rand_mat(4, 4)
    A, B, C = lazy.((A_, B_, C_))
    # Chain of operations
    M = (A + B) * (C + scal(-1)*A)
    M_ref = (A_ + B_) * (C_ + (scal(-1)* A_))
    if cpu @test Matrix(M) ≈ M_ref end
    x = rand_vec(4)
    @test M * x ≈ M_ref * x
end

@testset "Diagonal and block diagonal special cases" begin
    d = rand_vec(6)
    D = lazy(Diagonal(d))
    # Block diagonal
    B = EPMAfem.blockmatrix(D, D, D)
    B_ref = EPMAfem.blockmatrix(Diagonal(d), Diagonal(d), Diagonal(d))
    x = rand_vec(size(B, 2))
    @test B * x ≈ B_ref * x
    # Diagonal + scalar
    α = rand_scal()
    S = D + α * Diagonal(ones_vec(6));
    S_ref = Diagonal(d .+ α)
    x = rand_vec(size(S, 2))
    @test S * x ≈ S_ref * x
end

@testset "Lazy matrix with view and submatrix" begin
    A_ = rand(6, 6)
    A = lazy(A_)
    v = view(A, 2:4, 3:6)
    v_ref = view(A_, 2:4, 3:6)
    @test Matrix(v) ≈ Matrix(v_ref)
    x = rand(4)
    @test v * x ≈ Matrix(v_ref) * x
end

