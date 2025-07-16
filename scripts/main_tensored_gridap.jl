using Revise
using EPMAfem
using EPMAfem.Gridap
using LinearAlgebra
using BenchmarkTools
using CUDA
using KernelAbstractions

using Symbolics, SparseArrays, BenchmarkTools, CUDA

using Test


nx, ny, nz = 5, 5, 5

model2D = CartesianDiscreteModel((0, 1, 0, 2), (nx, ny))
model3D = CartesianDiscreteModel((0, 1, 0, 2, 0, 3), (nx, ny, nz))

model1Dx = CartesianDiscreteModel((0, 1), nx)
model1Dy = CartesianDiscreteModel((0, 2), ny)
model1Dz = CartesianDiscreteModel((0, 3), nz)

function assemble3D(model3D)
    Vp = TestFESpace(model3D, ReferenceFE(lagrangian, Float64, 1); conformity=:H1)
    Vm = TestFESpace(model3D, ReferenceFE(lagrangian, Float64, 0); conformity=:L2)
    Ω = Triangulation(model3D)
    dx = Measure(Ω, 4)
    a(u, v) = ∫(dot(∇(u), VectorValue(1.0, 0.0, 0.0)) * v)dx
    return assemble_matrix(a, TrialFESpace(Vp), Vm)
end

function assemble2D(model2D)
    Vp = TestFESpace(model2D, ReferenceFE(lagrangian, Float64, 1); conformity=:H1)
    Vm = TestFESpace(model2D, ReferenceFE(lagrangian, Float64, 0); conformity=:L2)
    Ω = Triangulation(model2D)
    dx = Measure(Ω, 4)
    a(u, v) = ∫(dot(∇(u), VectorValue(1.0, 0.0)) * v)dx
    return assemble_matrix(a, TrialFESpace(Vp), Vm)
end

function assemble1D(model1D, grad)
    Vp = TestFESpace(model1D, ReferenceFE(lagrangian, Float64, 1); conformity=:H1)
    Vm = TestFESpace(model1D, ReferenceFE(lagrangian, Float64, 0); conformity=:L2)
    Ω = Triangulation(model1D)
    dx = Measure(Ω, 4)
    if grad
        return assemble_matrix(
            (u, v) -> ∫(dot(∇(u), VectorValue(1.0)) * v)dx, TrialFESpace(Vp), Vm)
    else
        return assemble_matrix(
            (u, v) -> ∫(u * v)dx, TrialFESpace(Vp), Vm)
    end
end

A3D = assemble3D(model3D) |> cu
# A2D = assemble2D(model2D)

# A2D = assemble2D(model2D)
A1Dx = assemble1D(model1Dx, true)
A1Dy = assemble1D(model1Dy, false)
A1Dz = assemble1D(model1Dz, false)

Tx = EPMAfem.TwoDiagonalMatrix{Float64}(A1Dx[1, 1], A1Dx[1, 2], size(A1Dx, 2))
Ty = EPMAfem.TwoDiagonalMatrix{Float64}(A1Dy[1, 1], A1Dy[1, 2], size(A1Dy, 2))
Tz = EPMAfem.TwoDiagonalMatrix{Float64}(A1Dz[1, 1], A1Dz[1, 2], size(A1Dz, 2))
Ω = EPMAfem.TwoDiagonalMatrix{Float32}(rand(), rand(), 105)
Ω = sprand(5, 5, 0.5)

TL = EPMAfem.lazy(kron, Tz, EPMAfem.lazy(kron, Ty, EPMAfem.lazy(kron, Tx, Ω)))

TL = EPMAfem.lazy(kron, Tz, EPMAfem.lazy(kron, Ty, Tx))
# TLdense = sparse(TL)

x = rand(size(TL, 2)) |> cu
y1 = rand(size(TL, 1)) |> cu
y2 = rand(size(TL, 1)) |> cu
ws = EPMAfem.create_workspace(EPMAfem.required_workspace(EPMAfem.mul_with!, TL), CUDA.zeros)

EPMAfem.mul_with!(ws, y1, TL, x, true, false)


TLS = EPMAfem.lazy(kron, Ω, A3D)
wsS = EPMAfem.create_workspace(EPMAfem.required_workspace(EPMAfem.mul_with!, TLS), CUDA.zeros)

CUDA.@time EPMAfem.mul_with!(ws, y1, TLS, x, true, false)


mul!(y2, A3D, x, true, false)

@benchmark CUDA.@sync EPMAfem.mul_with!($ws, $y1, $TL, $x, $true, $false)
@benchmark CUDA.@sync mul!($y2, $A3D, $x, $true, $false)

y1 ≈ y2


AΩ = rand(14, 15) |> cu
# testmat = kron(A1Dz, A1Dy, A1Dx)
# res = testmat .- A3D
# res.nzval

A2DL = EPMAfem.lazy(kron, AΩ, EPMAfem.lazy(kron, A1Dz, EPMAfem.lazy(kron, A1Dy, A1Dx)))

X = rand(size(A2DL, 2)) |> cu
Y1 = rand(size(A2DL, 1)) |> cu
Y2 = rand(size(A2DL, 1)) |> cu


ws = EPMAfem.create_workspace(EPMAfem.required_workspace(EPMAfem.mul_with!, A2DL), CUDA.zeros)
EPMAfem.mul_with!(ws, Y1, A2DL, X, true, false);

CUDA.@time EPMAfem.mul_with!(ws, Y1, A2DL, X, true, false);
CUDA.@time mul!(Y2, A3D, X, true, false);


@benchmark CUDA.@sync EPMAfem.mul_with!($ws, $Y1, $A2DL, $X, $true, $false)
@benchmark CUDA.@sync mul!($Y2, $A3D, $X, $true, $false)



A1D_CSR_ = CUDA.CUSPARSE.CuSparseMatrixCSR(A1Dx)
A1D_CSR = reshape(A1D_CSR_, :, :, 1)

A = CUDA.rand(101, 50, 1000)
A2 = CUDA.rand(101, 50, 1000)
A3 = CUDA.rand(101, 50, 1000)
B = CUDA.rand(102, 50, 1000)

@benchmark CUDA.@sync CUDA.CUSPARSE.bmm!('N', 'N', true, A1D_CSR, B, false, A, 'O')

function naive(A, CSR, B)
    for i in 1:size(A, 3)
        mul!(@view(A[:, :, i]), CSR, @view(B[:, :, i]), true, false)
    end
end

@benchmark CUDA.@sync naive(A2, A1D_CSR_, B)

B_res = reshape(B, 102, :)
A3_res = reshape(A3, 101, :)

@benchmark CUDA.@sync mul!(A3_res, A1D_CSR_, B_res)

A ≈ A2
A ≈ A3

A = rand(20, 30)
B = rand(40, 50)
C = rand(60, 70)

X = rand(30, 50, 70)
Y = zeros(20, 40, 60)

using TensorOperations

@tensor Y[i, j, k] := A[i, l] * B[j, m] * C[k, n] * X[l, m, n]


@view(X[:])
@view(Y[:])

kron(A, B, C) * @view(X[:])

T1 = A * reshape(X, 3, 5 * 7)
T2 = reshape(T1, :, 7) * transpose(C)


reshape(transpose(T1), 5, 7, 2) |> Base.iscontiguous

reshape(T1, 2, 5, 7)

cpu_A3D = assemble3D(model3D)

# A2D = assemble2D(model2D)

cpu_A1Dx = assemble1D(model1Dx, true)
cpu_A1Dy = assemble1D(model1Dy, false)
cpu_A1Dz = assemble1D(model1Dz, false)

# testmat = kron(A1Dz, A1Dy, A1Dx)
# res = testmat .- A3D
# res.nzval

cpu_A2DL = EPMAfem.lazy(kron, cpu_A1Dz, EPMAfem.lazy(kron, cpu_A1Dy, cpu_A1Dx))

cpu_X = rand(size(cpu_A2DL, 2))
cpu_Y1 = rand(size(cpu_A2DL, 1))
cpu_Y2 = rand(size(cpu_A2DL, 1))


cpu_ws = EPMAfem.create_workspace(EPMAfem.required_workspace(EPMAfem.mul_with!, cpu_A2DL), zeros)
@enter EPMAfem.mul_with!(cpu_ws, cpu_Y1, cpu_A2DL, cpu_X, true, false)

@benchmark EPMAfem.mul_with!(cpu_ws, cpu_Y1, cpu_A2DL, cpu_X, true, false)
@benchmark mul!(cpu_Y2, cpu_A3D, cpu_X, true, false)


