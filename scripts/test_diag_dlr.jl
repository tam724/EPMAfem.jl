using LinearAlgebra
using EPMAfem.PNLazyMatrices

kron_AXB = PNLazyMatrices.kron_AXB

nxp = 3
nΩp = 4
nxm = 3
nΩm = 4
Al = lazy(rand(nxp, nxp) |> x -> x * transpose(x))
Ar = lazy(rand(nΩp, nΩp) |> x -> x * transpose(x))
Bl = lazy(rand(nxp, nxm))
Br = lazy(rand(nΩm, nΩp))
Cl = lazy(Diagonal(rand(nxm)))
Cr = lazy(Diagonal(rand(nΩm)))

A = kron_AXB(Al, Ar)
B = kron_AXB(Bl, Br)
C = kron_AXB(Cl, Cr)

BM = [A B
transpose(B) C]

U, S, V = svd(rand(4, 4))
Vr = lazy(transpose(V)[1:2, :])

Vr * transpose(Vr)
transpose(Vr) * Vr


Av = kron_AXB(Al, Vr * Ar * transpose(Vr))
Bv = kron_AXB(Bl, Vr * Br * transpose(Vr)) 
Cv = kron_AXB(Cl, Vr * Cr * transpose(Vr))

BMv = [Av Bv
transpose(Bv) Cv] 

test1 = Bv * inv(Matrix(Cv)) * transpose(Bv)
test2 = Bv * kron_AXB(I(nxm), transpose(Vr)) * inv(Matrix(C)) * kron_AXB(I(nxm), Vr) * transpose(Bv)

M = Vr.A * sqrt(Cr)
M_pinv = pinv(M)
Bv * kron_AXB(inv(sqrt(Cl)), M_pinv) * kron_AXB(inv(sqrt(Cl)), transpose(M_pinv)) * transpose(Bv)
Bv * kron_AXB(inv(Cl), transpose(M_pinv) * M_pinv) * transpose(Bv)

MU, MS, MV = svd(Vr.A * sqrt(Cr))
MU * Diagonal(1 ./ MS) * Diagonal(1 ./ MS) * transpose(MU)


MU, MS, MV = svd(Vr.A)




B = rand(4, 4)

D = Diagonal(rand(4, 4))
V = Vr.A
V * transpose(V) ≈ I # true
transpose(V) * V ≈ I # false

M = V * D

pinv(M)
inv(D) * transpose(V) * inv(V * inv(D*D) * transpose(V))
