using Revise
using EPMAfem
using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using BenchmarkTools
using LinearAlgebra
using SparseArrays

Lazy = EPMAfem.PNLazyMatrices

A_ = rand(3, 3)
B_ = rand(3, 3)
C_ = rand(3, 3)
D_ = Diagonal(rand(3))

A, B, C, D = lazy.((A_, B_, C_, D_))


BM = [A+B C
transpose(C) D]

BM_inv = EPMAfem.lazy((Lazy.schur_complement, Lazy.minres ∘ cache), BM)

X, Y, Z, K = Lazy._schur_components(BM_inv)

test = unlazy(BM_inv)

keys(test.ws.cache)

x = rand(size(test, 2))
test * x

Lazy.lazy_objectid(Lazy._schur_components(BM_inv)[4])
Lazy.lazy_objectid(Lazy._schur_components(BM_inv)[2])

Lazy.lazy_objectid(cache(BM))



M = materialize(A * B * C)
ws = PNLazyMatrices.create_workspace(PNLazyMatrices.required_workspace(materialize_with, M), zeros)
PNLazyMatrices.materialize_strategy(materialize(M))
M_, _ = PNLazyMatrices.materialize_with(ws, M, nothing)
A_*B_*C_ ≈ M_

M = D + materialize(kron(transpose(materialize(transpose(A + A))), B)) + materialize(transpose(kron(B, C)))
x = rand(size(M, 2))
(D_ + kron(A_ + A_, B_) + transpose(kron(B_, C_))) * x ≈ unlazy(M)*x




M = materialize(kron(A + A, B + B) + transpose(kron(B + B, C + C)))
ws = PNLazyMatrices.create_workspace(PNLazyMatrices.required_workspace(materialize_with, M), zeros)
PNLazyMatrices.materialize_strategy(materialize(M))
M_, _ = PNLazyMatrices.materialize_with(ws, M, nothing)
kron(A_ + A_, B_ + B_) + transpose(kron(B_ + B_, C_ + C_)) ≈ M_


using EPMAfem

K1 = EPMAfem.lazy(kron, rand_mat(10, 11), rand_mat(12, 13))
K1_ref = do_materialize(K1)
K2 = EPMAfem.lazy(kron, rand_mat(10, 11), rand_mat(12, 13))
K2_ref = do_materialize(K2)

S1 = EPMAfem.lazy(+, K1, K2)
S1_ref = K1_ref .+ K2_ref
@test S1_ref ≈ do_materialize(S1)

x = rand_vec(size(S1, 2))
@test S1_ref * x ≈ unlazy(S1) * x

S1t = transpose(S1)
x = rand_vec(size(S1t, 2))
@test transpose(S1_ref) * x ≈ unlazy(S1t) * x

# stich the two together
S1 = EPMAfem.lazy(+, rand(10, 11), rand(10, 11))
# S1_ref = do_materialize(S1)

S2 = EPMAfem.lazy(+, rand(12, 13), rand(12, 13))
# S2_ref = do_materialize(S2)

K1 = EPMAfem.lazy(kron, S1, S2)
K1_ref = kron(Matrix(S1), Matrix(S2))
materialize(K1) ≈ K1_ref

x = rand_vec(size(K1, 2))
@test unlazy(K1)*x ≈ K1_ref*x

S3 = EPMAfem.lazy(+, rand_mat(10, 11), rand_mat(10, 11))
S3_ref = do_materialize(S3)

S4 = EPMAfem.lazy(+, rand_mat(12, 13), rand_mat(12, 13))
S4_ref = do_materialize(S4)

K2 = EPMAfem.lazy(kron, S3, S4)
K2_ref = kron(S3_ref, S4_ref)

K = EPMAfem.lazy(+, K1, K2)
K_ref = K1_ref .+ K2_ref 
@test do_materialize(K) ≈ K_ref

x = rand_vec(size(K, 2))
@test unlazy(K)*x ≈ K_ref*x

x = rand_vec(size(K, 1))
@test unlazy(transpose(K))*x ≈ transpose(K_ref)*x
