using Revise
using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using LinearAlgebra
using EPMAfem.CUDA
using MethodAnalysis

N = 5
A = lazy(randn(N, N) + 3*I |> A -> transpose(A) * A)
B = lazy(randn(N, N) + 3*I |> A -> transpose(A) * A)
C = lazy(Diagonal(rand(N)))

A_dense = kron(2.0 * A.A + B.A, C.A + 2.0*A.A)

A_ = unlazy(kron(2.0 * A + B, C + 2.0 * A))

A_ * rand(size(A_, 2))

methods(PNLazyMatrices.mul_with!)
methodinstances(PNLazyMatrices.mul_with!)


skel = Diagonal(zeros(size(A_, 1)))
test, ws = PNLazyMatrices.materialize_diag_with(A_.ws, A_.A, skel, true, false)

@assert diag(A_dense) ≈ diag(test)







M = unlazy(kron(A, B))

M_ = kron(A.A, B.A)
diag(M_)



test, ws = PNLazyMatrices
