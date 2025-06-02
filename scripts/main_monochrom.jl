using Revise
using EPMAfem
using Gridap
using LinearAlgebra
using Plots
using Distributions
include("plot_overloads.jl")
Makie.inline!(false)

heatmap(-0.5:0.01:0.5, -0.5:0.01:0.5, (x, z) -> EPMAfem.mass_concentrations(eq, 1, Gridap.Point(z, x)))
plot(-1:0.01:1, μ -> EPMAfem.monochrom_scattering_kernel_func(eq, μ))

function scat_coeff(x)
    return sum(EPMAfem.mass_concentrations(eq, e, x)*EPMAfem.scattering_coefficient(eq, e) for e in 1:EPMAfem.number_of_elements(eq))
end

## 2D
eq = EPMAfem.MonochromPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.5, 0.5, -0.5, 0.5), (30, 30)))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(7, 2)
model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)

function fx(; z)
    return exp(-200*(z)^2)
end
function fΩ_left(Ω)
    return pdf(VonMisesFisher([0.0, 1.0, 0.0] |> normalize, 50.0), Ω)
end
function fΩ_right(Ω)
    return pdf(VonMisesFisher([0.0, -1.0, 0.0] |> normalize, 50.0), Ω)
end

bc_left = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.X(), EPMAfem.Dimensions.LeftBoundary(), fx, fΩ_left)
bc_right = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.X(), EPMAfem.Dimensions.RightBoundary(), fx, fΩ_right)

function qx((z, x))
    return exp(-100*(x^2 + z^2))
end

function qΩ(Ω)
    return 1.0
end

bc_source = EPMAfem.PNXΩSource(qx, qΩ)

rhs_source = EPMAfem.discretize_rhs(bc_source, model, EPMAfem.cpu())

problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu())
rhs_left = EPMAfem.discretize_rhs(bc_left, model, EPMAfem.cpu())
rhs_right = EPMAfem.discretize_rhs(bc_right, model, EPMAfem.cpu())

system = EPMAfem.system(problem, EPMAfem.PNSchurSolver)

x_left = EPMAfem.allocate_solution_vector(system)
x_right = EPMAfem.allocate_solution_vector(system)
x_source = EPMAfem.allocate_solution_vector(system)
EPMAfem.solve(x_left, system, rhs_left)
EPMAfem.solve(x_right, system, rhs_right)
EPMAfem.solve(x_source, system, rhs_source)

Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> problem.arch

xp_left, xm_left = EPMAfem.pmview(x_left, model)


xp_right, xm_right = EPMAfem.pmview(x_right, model)
xp_source, xm_source = EPMAfem.pmview(x_source, model)

func_left = EPMAfem.SpaceModels.interpolable((p=xp_left*Ωp|> collect, m=xm_left*Ωm |> collect), space_model)

func_right = EPMAfem.SpaceModels.interpolable((p=xp_right*Ωp|> collect, m=xm_right*Ωm |> collect), space_model)
func_source = EPMAfem.SpaceModels.interpolable((p=xp_source*Ωp|> collect, m=xm_source*Ωm |> collect), space_model)
    
p1 = Plots.contourf(-0.5:0.001:0.5, -0.5:0.001:0.5, (z, x) -> func_left(VectorValue(x, z)), cmap=:grays, aspect_ratio=:equal)

p2 = Plots.contourf(-0.5:0.001:0.5, -0.5:0.001:0.5, (z, x) -> func_right(VectorValue(x, z)), cmap=:grays, aspect_ratio=:equal)
p3 = Plots.contourf(-0.5:0.001:0.5, -0.5:0.001:0.5, (z, x) -> func_source(VectorValue(x, z)), cmap=:grays, aspect_ratio=:equal)
plot(p1, p2, size=(1500, 1000))

Plots.contourf(-0.5:0.001:0.5, -0.5:0.001:0.5, (z, x) -> qx(VectorValue(x, z)), cmap=:grays, aspect_ratio=:equal)



## 1D
eq = EPMAfem.MonochromPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.5, 0.5), (50)))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(7, 1)
model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)

function fx((; ))
    return 1.0
end
function fΩ_left(Ω)
    return 1.0 # pdf(VonMisesFisher([0.0, 1.0, 0.0] |> normalize, 50.0), Ω)
end
function fΩ_right(Ω)
    return 1.0  #pdf(VonMisesFisher([0.0, -1.0, 0.0] |> normalize, 50.0), Ω)
end
bc_left = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.Z(), EPMAfem.Dimensions.LeftBoundary(), fx, fΩ_left)
bc_right = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.Z(), EPMAfem.Dimensions.RightBoundary(), fx, fΩ_right)

problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu(Float64))
rhs_left = EPMAfem.discretize_rhs(bc_left, model, EPMAfem.cpu(Float64))
rhs_right = EPMAfem.discretize_rhs(bc_right, model, EPMAfem.cpu(Float64))

system = EPMAfem.system(problem, EPMAfem.PNSchurSolver; solver=EPMAfem.PNKrylovGMRESSolver)
system = EPMAfem.system(problem, EPMAfem.PNSchurSolver)
system = EPMAfem.system(problem, EPMAfem.PNKrylovMinres2Solver)
# system = EPMAfem.system(problem, EPMAfem.PNKrylovMinresSolver)
x_left = EPMAfem.allocate_solution_vector(system)
x_right = EPMAfem.allocate_solution_vector(system)
EPMAfem.solve(x_left, system, rhs_left)
EPMAfem.solve(x_right, system, rhs_right)

y = zeros(1010)
mul!(y, system.A, x_left, true, false)
EPMAfem.assemble!(y, rhs_left, true, system.A.sym[], 1)
maximum(abs.(y))

system.A.sym[] = true
mat = EPMAfem.assemble_from_op(system.A)

b_left = zeros(1010)
EPMAfem.assemble!(b_left, rhs_left, -1, system.A.sym[])
b_left = b_left

E = mat[1:510, 1:510]
A = mat[1:510, 511:end]

AT = mat[511:end, 1:510]
F = Diagonal(diag(mat[511:end, 511:end]))

b = b_left[1:510]
c = b_left[511:end]

using EPMAfem.Krylov

x, y = trimr(A, b, c, M=opInverse(E; symm=true), N=inv(-F), τ=1.0, ν=-1.0)
#x, y = trimr(A, b, c, N=inv(-F), ν=-1.0)

[E A
transpose(A) -F] * [x; y] .- [b; c]

maximum(abs.([I A
transpose(A) F] * [x; y] .- [b; c]))


x_left .- x_left_new

tricg(A, b, c)


















nothing
schur_mat = UL - UR*inv(LR)*LL

N = EPMAfem.SchurBlockMat2{Float64}(system.A, system.lin_solver.C_ass, system.lin_solver.cache)
EPMAfem.update_cache!(N)

matN = EPMAfem.assemble_from_op(N)
valsN, vecsN = eigen(Matrix(matN))

Plots.scatter(valsN)

schur_mat2 = EPMAfem.assemble_from_op(N)
maximum(abs.(schur_mat2 .- transpose(schur_mat2)))



Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> problem.arch

xp_left, xm_left = EPMAfem.pmview(x_left, model)
xp_right, xm_right = EPMAfem.pmview(x_right, model)
func_left = EPMAfem.SpaceModels.interpolable((p=xp_left*Ωp|> collect, m=xm_left*Ωm |> collect), space_model)
func_right = EPMAfem.SpaceModels.interpolable((p=xp_right*Ωp|> collect, m=xm_right*Ωm |> collect), space_model)
    
p1 = Plots.plot(-0.5:0.001:0.5, (z) -> func_left(VectorValue(z)))
p2 = Plots.plot(-0.5:0.001:0.5, (z) -> func_right(VectorValue(z)))
plot(p1, p2, size=(1500, 1000))

mat = EPMAfem.assemble_from_op(system.A)
