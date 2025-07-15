using Revise
using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using BenchmarkTools
using LinearAlgebra

using SparseArrays

A = lazy(sprand(70*70, 70*70, 0.01))
B = lazy(sprand(200, 200, 0.01))
C = lazy(sprand(200, 200, 0.01))

D = lazy(Diagonal(rand(70*70 * 200)))

# C = A + 3.0 * B

D = unlazy(D + kron(A, 3.0 * B) + kron(0.5 * A, 2.0 * C))
D_notlazy = kron(A.A, 3.0*B.A) + kron(0.5*A.A, 2.0*C.A)

x = rand(size(D, 2))
y1 = rand(size(D, 1))
y2 = rand(size(D, 1))

@profview mul!(y1, D, x, true, false)


@btime mul!($y1, $D, $x, $true, $false);
@btime mul!($y2, $D_notlazy, $x, $true, $false);

y1 ≈ y2
