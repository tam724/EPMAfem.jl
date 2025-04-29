# activate local environment and install required packages
using Pkg
Pkg.activate("./scripts_emas_2025/workshop")

Pkg.add("Plots")
# we need NeXLCore#master as of now.. (there's a typo in the current released version of NeXLCore that errors EPMAfem.jl)
Pkg.add(url="https://github.com/usnistgov/NeXLCore.jl", rev="master")
Pkg.add("NeXLMatrixCorrection")
Pkg.add("Unitful")
Pkg.add("Dimensionless")
Pkg.add(url="https://github.com/tam724/EPMAfem.jl", rev="main")

# include the required packages
using Plots
using EPMAfem
using NeXLCore
using NeXLMatrixCorrection
using Unitful
using Dimensionless
using EPMAfem.Gridap

NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)

# homogeneous material: Copper
mat = [n"Cu"]
# define the energy range (and the steps in energy)
ϵ_range = range(3u"keV", 17u"keV", length=50)
# prepare the "equations": stopping power, cross sections, etc...
eq = NExt.epma_equations(mat, ϵ_range, 23)
# define the model (spatial extents, number of grid nodes in space, number of spherical harmonics)
model = NExt.epma_model(eq, (-800u"nm", 0u"nm"), (100), 23)

# discretize the problem -> build the "matrix"
pnproblem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu(), updatable=true)
# define the "linear solver" (system can be thought of as "matrix"^{-1})
system = EPMAfem.implicit_midpoint(pnproblem.problem, EPMAfem.PNSchurSolver);

# define a beam at (0, 0) with energy 15keV in direction -z
beam = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [dimless(15u"keV", eq.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)])
# discretize the "rhs of the equation"
pnsource = EPMAfem.discretize_rhs(beam, model, EPMAfem.cpu())[1]

# define what we want to do with the high dimensional solution. here : marginalize over energies and direction -> results in a function in space
probe = EPMAfem.PNProbe(model, EPMAfem.cpu(), ϵ = ϵ -> 1.0, Ω = Ω -> 1.0)
# compute the solution and build an "interpolable" function
f = EPMAfem.interpolable(probe, system * pnsource)
# plot the function over the extent in space
plot(range(-800u"nm", 0u"nm", 100), x -> f(VectorValue(dimless(x, eq.dim_basis))))



############## MORE EXAMPLES
# compare EPMAfem.jl to traditional phi-rho-z curves

include("phi_rho_z_nexl.jl")

ϕ_z_nexl = compute_phi_z_nexl(n"Cu K", 15.0u"keV")
ϕ_z_nexl2 = compute_phi_z_nexl(n"Cu K", 15.0u"keV", NeXLMatrixCorrection.CitZAF)
ϕ_z_epmafem = compute_phi_z_epmafem(n"Cu K", 15.0u"keV", 800.0u"nm")

plot(range(-800.0u"nm", 0u"nm", 100), ϕ_z_nexl, label="ϕ([ρ]z) XPP Cu 15keV")
plot!(range(-800.0u"nm", 0u"nm", 100), ϕ_z_nexl2, label="ϕ([ρ]z) CitZAF Cu 15keV")
plot!(range(-800.0u"nm", 0u"nm", 100), ϕ_z_epmafem, label="ϕ([ρ]z) EPMAfem Cu 15keV")

savefig("1D_phirhoz.png")
