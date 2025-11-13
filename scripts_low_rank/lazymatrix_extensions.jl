using Revise
using EPMAfem.PNLazyMatrices

A = lazy(rand(2, 2))
B = lazy(rand(2, 2))
C = lazy(rand(2, 2))
D = lazy(rand(2, 2))
E = lazy(rand(2, 2))
F = lazy(rand(2, 2))
G = lazy(rand(2, 2))

A + B + C

A + B + A

1.0*A + 2.0*A

(A + B + C)

2.0*A + 2.0*B + 2.0*C

2.0*kron(A, B) + kron(A, B)

2*A + A

2.0*(A + B)

A + 3.0*A

@edit PNLazyMatrices.lazy_simplify(+, A , B)


1.0*A + 2.0*A

2.0*A + A

A + 2.0*A

kron(A, B, C, E) + kron(A, B, D, E)


1.0*kron(A, B, C) + 2.0*kron(A, B, 2.0*D)
(A + B) + (C + D)

T = kron(A, 1.0*B + 2.0*D)
U = kron(A, 4.0*C)

T + U

T = kron(A, B, C, E, F, G)
U = kron(A, B, D, E, F, G)

T + U

X = unlazy(T + U)
x = rand(size(X, 2))
y = zeros(size(X, 1))

mul!(y, X, x)



PNLazyMatrices.can_simplify(+, T, U)

a, b, c = PNLazyMatrices.split_common_prefix(T.args, U.args)

nxp, nxm, nΩp, nΩm = 2, 3, 4, 5

Ap = rand(nxp, nxp) |> lazy
Am = rand(nxm, nxm) |> lazy
Bp = [rand(nΩp, nΩp) |> lazy for i in 1:2]
Bm = [rand(nΩm, nΩm) |> lazy for i in 1:2]

Cp = [rand(nxp, nxm) |> lazy for i in 1:2]
Dp = [rand(nΩp, nΩm) |> lazy for i in 1:2]

B = sum(kron(Cp[i], Dp[i]) for i in 1:2)

α1 = [LazyScalar(rand()) for i in 1:2]
α2 = [LazyScalar(rand()) for i in 1:2]

BM1 = [sum(kron(Ap, α1[i]*Bp[i]) for i in 1:2) B
    transpose(B) sum(kron(Am, α1[i]*Bm[i]) for i in 1:2)]

BM2 = [sum(kron(Ap, α2[i]*Bp[i]) for i in 1:2) B*0.5
    transpose(B)*0.5 sum(kron(Am, α2[i]*Bm[i]) for i in 1:2)]

test = BM1 + 2.0* BM2


(1.0*A + 2.0*B)

A, B, C, D

using EPMAfem
using EPMAfem.Gridap

energy_model = 0:0.01:1
equations = EPMAfem.PNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5, -1, 1, -1, 1), (2, 2, 2)))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(3, 3, :OE)
model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())

A_i, Acoeffs_i = EPMAfem.system_matrix(problem)
M_i, Mcoeffs_i = EPMAfem.mass_matrix(problem)
coeffs_ = (; Acoeffs_i..., Mcoeffs_i...)

BM, coeffs = unlazy(((1/0.1)*A_i + 0.5*M_i, coeffs_))



(A+B)+(C+D+E)

for τ in coeffs.τ
    τ[] = 1.0
end
for s in coeffs.s
    s[] = 1.0
end
for σ in coeffs.σ
    σ[] = 1.0
end


BM.A

A_ = Matrix(BM.A)
using LinearAlgebra

plot(eigen(A_).values)


A = EPMAfem.PNLazyMatrices.A(A_i) + EPMAfem.PNLazyMatrices.A(M_i)

BM.A
