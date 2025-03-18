using Revise
using EPMAfem
using Gridap
using LinearAlgebra
using Plots
include("plot_overloads.jl")


eq = EPMAfem.MonochromPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((0, 1, 0, 1), (50, 50)))
# direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(13, 1)

function f((; z))
    # @show x
    return exp(-100*(z-0.5)^2)
end

bc = EPMAfem.PNSpaceBoundaryCondition(EPMAfem.Dimensions.X(), EPMAfem.Dimensions.RightBoundary(), f)
bc_disc = EPMAfem.discretize(bc, space_model, EPMAfem.cpu())

func = FEFunction(EPMAfem.SpaceModels.even(space_model), bc_disc)
surface(0:0.01:1, 0:0.01:1, func)
xlabel!("z")
ylabel!("x")

plot!()



eq = EPMAfem.MonochromPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.0, 0.0), 50))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(13, 1)

function f(_)
    return 1
end

bc = EPMAfem.PNSpaceBoundaryCondition(EPMAfem.Dimensions.Z(), EPMAfem.Dimensions.RightBoundary(), f)
EPMAfem.discretize(bc, space_model, EPMAfem.cpu())


model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)
problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu())

nb = EPMAfem.n_basis(problem)
ns = EPMAfem.n_sums(problem)

arch = EPMAfem.architecture(problem)
T = EPMAfem.base_type(arch)

cache = EPMAfem.allocate_vec(arch, max(nb.nx.p, nb.nx.m)*max(nb.nΩ.p, nb.nΩ.m))
cache2 = EPMAfem.allocate_vec(arch, max(nb.nΩ.p, nb.nΩ.m))

coeffs = (a = problem.τ, c = [[T(problem.σ[i])] for i in 1:ns.ne])

A = EPMAfem.ZMatrix2{T}(problem.space_discretization.ρp, problem.direction_discretization.Ip, problem.direction_discretization.kp, coeffs.a, coeffs.c, nb.nx.p, nb.nx.p, nb.nΩ.p, nb.nΩ.p, ns.ne, ns.nσ, EPMAfem.mat_view(cache, nb.nx.p, nb.nΩ.p), Diagonal(@view(cache2[1:nb.nΩ.p])))
B = EPMAfem.DMatrix2{T}(problem.space_discretization.∇pm, problem.direction_discretization.Ωpm, nb.nx.p, nb.nx.m, nb.nΩ.m, nb.nΩ.p, ns.nd, EPMAfem.mat_view(cache, nb.nx.p, nb.nΩ.m))
C = EPMAfem.ZMatrix2{T}(problem.space_discretization.ρm, problem.direction_discretization.Im, problem.direction_discretization.km, coeffs.a, coeffs.c, nb.nx.m, nb.nx.m, nb.nΩ.m, nb.nΩ.m, ns.ne, ns.nσ, EPMAfem.mat_view(cache, nb.nx.m, nb.nΩ.m), Diagonal(@view(cache2[1:nb.nΩ.m])))
D = EPMAfem.DMatrix2{T}(problem.space_discretization.∂p, problem.direction_discretization.absΩp, nb.nx.p, nb.nx.p, nb.nΩ.p, nb.nΩ.p, ns.nd, EPMAfem.mat_view(cache, nb.nx.p, nb.nΩ.p))

BM = EPMAfem.BlockMat2{T}(A, B, C, D, nb.nx.p*nb.nΩ.p, nb.nx.m*nb.nΩ.m, Ref(1.0), Ref(-1.0), Ref(1.0), Ref(1.0), Ref(false))
lin_solver = EPMAfem.PNSchurSolver(EPMAfem.vec_type(arch), BM)
x = zeros(2828)
b = zeros(2828)
bp, bm = EPMAfem.pmview(b, model)
init(x) = exp(-150*(x-0.5)^2)
plot(range(0, 1, length=51), init)

bp[:, 1] .= init.(range(0, 1, length=51))
bm[:, 1] .= 1.0.*init.(range(0, 1, length=50))
EPMAfem.pn_linsolve!(lin_solver, x, BM, b)
xp, xm = EPMAfem.pmview(x, model)
plot!(range(0, 1, length=51), xp[:, 1])
