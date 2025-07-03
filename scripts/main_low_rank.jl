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

