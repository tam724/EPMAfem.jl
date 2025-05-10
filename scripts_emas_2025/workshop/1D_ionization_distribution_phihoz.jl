# activate local environment and install required packages
using Pkg
Pkg.activate("./scripts_emas_2025/workshop")

Pkg.add("Plots")
# we need NeXLCore#master as of now.. (there's a typo in the current released version of NeXLCore that errors EPMAfem.jl)
Pkg.add(url="https://github.com/usnistgov/NeXLCore.jl", rev="master")
Pkg.add("NeXLMatrixCorrection")
Pkg.add("Unitful")
Pkg.add("Dimensionless")
Pkg.add(url="https://github.com/tam724/EPMAfem.jl", rev="a63052f6473865223501c0947ec830220de42136")

# include the required packages
using Plots
using EPMAfem
using NeXLCore
using NeXLMatrixCorrection
using Unitful
using Dimensionless
using EPMAfem.Gridap

NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)

# material components: Copper, Aluminium (per default the material is homogeneous (first element, here: Copper))
mat = [n"Cu", n"Al"]
# define the energy range (and the steps in energy)
ϵ_range = range(3u"keV", 17u"keV", length=20)
# prepare the "equations": stopping power, cross sections, etc...
eq = NExt.epma_equations(mat, ϵ_range, 27)

# plot the stopping power (thereby reintroduce units)
plot(ϵ_range, ϵ -> dimful(EPMAfem.stopping_power(eq, 1, dimless(ϵ, eq.dim_basis)), u"keV"/u"nm", eq.dim_basis))
plot!(ϵ_range, ϵ -> dimful(EPMAfem.stopping_power(eq, 2, dimless(ϵ, eq.dim_basis)), u"keV"/u"nm", eq.dim_basis))

# define the model (spatial extents, number of grid nodes in space, number of spherical harmonics)
model = NExt.epma_model(eq, (-800u"nm", 0u"nm"), (20), 1)

# discretize the problem -> build the "matrix"
pnproblem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu(), updatable=true)
# define the "linear solver" (system can be thought of as "matrix"^{-1})
system = EPMAfem.implicit_midpoint(pnproblem.problem, EPMAfem.PNSchurSolver);

# define a beam at (0, 0) with energy 15keV in direction -z
beam = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [dimless(15.0u"keV", eq.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)]; beam_energy_σ=0.05)
# discretize the "rhs of the equation"
pnsource = EPMAfem.discretize_rhs(beam, model, EPMAfem.cpu())[1]

# define what we want to do with the high dimensional solution. here : marginalize over energies and direction -> results in a function in space
probe = EPMAfem.PNProbe(model, EPMAfem.cpu(), ϵ = ϵ -> 1.0, Ω = Ω -> 1.0)

# compute the solution and build an "interpolable" function
f = EPMAfem.interpolable(probe, system * pnsource)
# plot the function over the extent in space
plot(range(-800u"nm", 0u"nm", 100), x -> f(VectorValue(dimless(x, eq.dim_basis))))
plot!(range(-800u"nm", 0u"nm", 100), x -> f(VectorValue(dimless(x, eq.dim_basis))))

###### vizualize the fluence over energies
cached_solution = EPMAfem.saveall(system * pnsource)
@gif for energy in reverse(eq.energy_model_dimless)
    probe = EPMAfem.PNProbe(model, EPMAfem.cpu(), ϵ = energy, Ω = Ω -> 1.0)
    f = EPMAfem.interpolable(probe, cached_solution)
    plot(range(-800u"nm", 0u"nm", 100), x -> f(VectorValue(dimless(x, eq.dim_basis))))
    # ylims!(0, 0.02)
    title!("ϵ = $(dimful(energy, u"keV", eq.dim_basis))")
end

###### change the material to inhomogeneous
function mass_concentrations(elm, x_)
    z = dimful(x_[1], u"nm", eq.dim_basis)
    if z < -100u"nm"
        return elm == n"Cu" ? dimless(n"Cu".density, eq.dim_basis) : 0.0
    else
        return elm == n"Al" ? dimless(n"Al".density, eq.dim_basis) : 0.0
    end
end

ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(elm, x) for elm in mat], model)

# vizualize the inhomogeneous material
func = FEFunction(EPMAfem.SpaceModels.material(model.space_mdl), ρs[2, :])
plot(range(-800u"nm", 0u"nm", 100), x -> func(VectorValue(dimless(x, eq.dim_basis))))

# update the problem, such that it works with the updated material
EPMAfem.update_problem!(pnproblem, ρs)

######### same plots as before
# compute the solution and build an "interpolable" function
f = EPMAfem.interpolable(probe, system * pnsource)
# plot the function over the extent in space
plot(range(-800u"nm", 0u"nm", 100), x -> f(VectorValue(dimless(x, eq.dim_basis))))

cached_solution = EPMAfem.saveall(system * pnsource)
@gif for energy in reverse(eq.energy_model_dimless)
    probe = EPMAfem.PNProbe(model, EPMAfem.cpu(), ϵ = energy, Ω = Ω -> 1.0)
    f = EPMAfem.interpolable(probe, cached_solution)
    plot(range(-800u"nm", 0u"nm", 100), x -> f(VectorValue(dimless(x, eq.dim_basis))))
    ylims!(0, 0.01)
    title!("ϵ = $(dimful(energy, u"keV", eq.dim_basis))")
end fps=4

############## MORE EXAMPLES
# compare EPMAfem.jl to traditional phi-rho-z curves
include("phi_rho_z_nexl.jl")

ϕ_z_nexl = compute_phi_z_nexl(n"Cu K", 15.0u"keV")
ϕ_z_nexl2 = compute_phi_z_nexl(n"Cu K", 15.0u"keV", NeXLMatrixCorrection.CitZAF)
ϕ_z_epmafem = compute_phi_z_epmafem(n"Cu K", 15.0u"keV", 700.0u"nm")

plot(range(-700.0u"nm", 0u"nm", 100), ϕ_z_nexl, label="ϕ([ρ]z) XPP Cu 15keV")
plot!(range(-700.0u"nm", 0u"nm", 100), ϕ_z_nexl2, label="ϕ([ρ]z) CitZAF Cu 15keV")
plot!(range(-700.0u"nm", 0u"nm", 100), ϕ_z_epmafem, label="ϕ([ρ]z) EPMAfem Cu 15keV")
