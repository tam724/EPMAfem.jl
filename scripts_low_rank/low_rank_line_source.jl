using Revise
using EPMAfem

using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using EPMAfem.Gridap
using LinearAlgebra
using Plots

struct LineSourceEquations <: EPMAfem.AbstractPNEquations end
EPMAfem.number_of_elements(::LineSourceEquations) = 1
EPMAfem.number_of_scatterings(::LineSourceEquations) = 1
EPMAfem.stopping_power(::LineSourceEquations, e, ϵ) = 1.0
EPMAfem.absorption_coefficient(eq::LineSourceEquations, e, ϵ) = 1.0
EPMAfem.scattering_coefficient(eq::LineSourceEquations, e, i, ϵ) = 1.0
EPMAfem.mass_concentrations(::LineSourceEquations, e, x) = 1.0
EPMAfem.scattering_kernel(::LineSourceEquations, e, i) = μ -> 1/(4π)
equations = LineSourceEquations()

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5, -1.5, 1.5), (200, 200)))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(47, 2)
model = EPMAfem.DiscretePNModel(space_model, 0:0.01:1.0, direction_model)
problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda())

# source / boundary condition (here: zero)
source = EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cuda(), zeros(EPMAfem.n_basis(model).nϵ), zeros(EPMAfem.n_basis(model).nx.p), zeros(EPMAfem.n_basis(model).nΩ.p))

# initial condition
σ = 0.03
# σ = 0.2
using EPMAfem.HCubature
# init_x(x) = 1/(2π*σ^2)*exp(-(x[1]*x[1]+x[2]*x[2])/(2*σ^2)) #normal gaussian
init_x(x) = 1/(8π*σ^2)*exp(-(x[1]*x[1]+x[2]*x[2])/(2*σ^2)) # from (https://doi.org/10.1080/00411450.2014.910226)
init_x(x) = 1/(4π*σ^2)*exp(-(x[1]*x[1]+x[2]*x[2])/(4*σ^2)) # from (https://doi.org/10.1051/m2an/2022090)
init_Ω(_) = 1.0
Mp_cpu = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)))
# Mp = Mp_cpu |> EPMAfem.architecture(problem)
Mm_cpu = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model))) |> EPMAfem.diag_if_diag
# Mm = Mm_cpu |> EPMAfem.architecture(problem)

bxp = Mp_cpu \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)))
bxm = Mm_cpu \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)))
bΩp = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.even(EPMAfem.direction_model(model)))
bΩm = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.odd(EPMAfem.direction_model(model)))

nb = EPMAfem.n_basis(problem)
initial_condition = EPMAfem.allocate_vec(EPMAfem.cuda(), nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m)
ψ0p, ψ0m = EPMAfem.pmview(initial_condition, model)
copy!(ψ0p, bxp .* bΩp')
copy!(ψ0m, bxm .* bΩm')


# #### GIF
# Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> EPMAfem.architecture(problem)
# @gif for (ϵ, ψ) in sol
#     ψp, ψm = EPMAfem.pmview(ψ, model)
#     func = EPMAfem.SpaceModels.interpolable((p=collect(ψp*Ωp), m=collect(ψm*Ωm)), EPMAfem.space_model(model))
#     heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func(VectorValue(x, y)), aspect_ratio=:equal, cmap=:jet)
# end

using Serialization
figpath = mkpath(joinpath(dirname(@__FILE__), "figures/2D_linesource/"))
mkpath(joinpath(dirname(@__FILE__), "figures/2D_linesource/solutions"))

probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), ϵ=0.0, Ω=Ω -> 1.0)


system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
sol = EPMAfem.IterableDiscretePNSolution(system, source, initial_solution=initial_condition);
func = EPMAfem.interpolable(probe, sol)
serialize(joinpath(figpath, "solutions/full.jls"), func)

system_lr10 = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=10, m=10));
sol_lr10 = EPMAfem.IterableDiscretePNSolution(system_lr10, source, initial_solution=initial_condition);
func_lr10 = EPMAfem.interpolable(probe, sol_lr10)
serialize(joinpath(figpath, "solutions/lr10.jls"), func_lr10)

system_lr20 = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=20, m=20));
sol_lr20 = EPMAfem.IterableDiscretePNSolution(system_lr20, source, initial_solution=initial_condition);
func_lr20 = EPMAfem.interpolable(probe, sol_lr20)
serialize(joinpath(figpath, "solutions/lr20.jls"), func_lr20)

system_lr50 = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=50, m=50));
sol_lr50 = EPMAfem.IterableDiscretePNSolution(system_lr50, source, initial_solution=initial_condition);
func_lr50 = EPMAfem.interpolable(probe, sol_lr50)
serialize(joinpath(figpath, "solutions/lr50.jls"), func_lr50)

system_lr100 = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=100, m=100));
sol_lr100 = EPMAfem.IterableDiscretePNSolution(system_lr100, source, initial_solution=initial_condition);
func_lr100 = EPMAfem.interpolable(probe, sol_lr100)
serialize(joinpath(figpath, "solutions/lr100.jls"), func_lr100)

func = deserialize(joinpath(figpath, "solutions/full.jls"))
func_lr10 = deserialize(joinpath(figpath, "solutions/lr10.jls"))
func_lr20 = deserialize(joinpath(figpath, "solutions/lr20.jls"))
func_lr50 = deserialize(joinpath(figpath, "solutions/lr50.jls"))
func_lr100 = deserialize(joinpath(figpath, "solutions/lr100.jls"))

clims=(-1, 2)
plot(
heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func(VectorValue(x, y))/π, aspect_ratio=:equal, cmap=:jet1, clims=clims),
heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func_lr10(VectorValue(x, y))/π, aspect_ratio=:equal, cmap=:jet1, clims=clims),
heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func_lr20(VectorValue(x, y))/π, aspect_ratio=:equal, cmap=:jet1, clims=clims),
heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func_lr50(VectorValue(x, y))/π, aspect_ratio=:equal, cmap=:jet1, clims=clims),
heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func_lr100(VectorValue(x, y))/π, aspect_ratio=:equal, cmap=:jet1, clims=clims)
)

plot(0:0.01:1.2, x -> func(VectorValue(x, 0.0))/π, label="full")
plot!(0:0.01:1.2, x -> func_lr10(VectorValue(x, 0.0))/π, label="r10")
plot!(0:0.01:1.2, x -> func_lr20(VectorValue(x, 0.0))/π, label="r20")
plot!(0:0.01:1.2, x -> func_lr50(VectorValue(x, 0.0))/π, label="r50")
plot!(0:0.01:1.2, x -> func_lr100(VectorValue(x, 0.0))/π, label="r100")

ylims!(-0.1, 0.7)
### energy/mass plot
Mp = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)), EPMAfem.SpaceModels.even(EPMAfem.space_model(model))) |> EPMAfem.architecture(problem)
Mm = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model))) |> EPMAfem.architecture(problem)

Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> EPMAfem.architecture(problem)
xp, xm = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(model), x -> 1.0) |> EPMAfem.architecture(problem)

energy, energy_lr = zeros(length(sol)), zeros(length(sol))
mass, mass_lr = zeros(length(sol)), zeros(length(sol))

dot(xp, ψ0p*Ωp)
dot(xm, ψ0m*Ωm) 


for (i, ((ϵ, ψ), (ϵ_lr, ψ_lr))) in enumerate(zip(sol, sol_lr))
    ψp, ψm = EPMAfem.pmview(ψ, model)
    ψp_lr, ψm_lr = EPMAfem.pmview(ψ_lr, model)
    energy[i] = dot(ψp, Mp*ψp) + dot(ψm, Mm*ψm)
    energy_lr[i] = dot(ψp_lr, Mp*ψp_lr) + dot(ψm_lr, Mm*ψm_lr)

    mass[i] = dot(xp, ψp*Ωp) + dot(xm, ψm*Ωm)
    mass_lr[i] = dot(xp, ψp_lr*Ωp) + dot(xm, ψm_lr*Ωm)
end

plot(energy)
plot!(energy_lr)


plot(mass)
plot!(mass_lr)
#### ϵ = 0 PLOT

probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), ϵ=0.0, Ω=Ω -> 1.0)
func = EPMAfem.interpolable(probe, sol)
func_lr = EPMAfem.interpolable(probe, sol_lr)

p1 = heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func(VectorValue(x, y)), aspect_ratio=:equal)
p2 = plot(0.0:0.01:1.2, x -> func(VectorValue(x, 0.0)))
plot(p1, p2)

p_lr1 = heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func_lr(VectorValue(x, y)), aspect_ratio=:equal)
p_lr2 = plot(0.0:0.01:1.2, x -> func_lr(VectorValue(x, 0.0)))
plot(p_lr1, p_lr2)
