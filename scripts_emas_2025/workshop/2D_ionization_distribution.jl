# activate local environment and install required packages
using Pkg
Pkg.activate("./scripts_emas_2025/workshop")

Pkg.add("Plots")
Pkg.add(url="https://github.com/usnistgov/NeXLCore.jl", rev="master")
Pkg.add("NeXLMatrixCorrection")
Pkg.add("Unitful")
Pkg.add("Dimensionless")
Pkg.add(url="https://github.com/tam724/EPMAfem.jl", rev="main")

# include the required packages
using Plots
using EPMAfem
using NeXLCore
using Unitful
using Dimensionless
using EPMAfem.Gridap

NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)

mat = [n"Cu", n"Al"]
ϵ_range = range(3u"keV", 17u"keV", length=80)
eq = NExt.epma_equations(mat, ϵ_range, 23)
model = NExt.epma_model(eq, (-800u"nm", 0u"nm", -500u"nm", 500u"nm"), (80, 100), 23)

pnproblem = EPMAfem.discretize_problem(eq, model, EPMAfem.cuda(), updatable=true)
system = EPMAfem.implicit_midpoint(pnproblem.problem, EPMAfem.PNSchurSolver);

beam = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [dimless(15u"keV", eq.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)])
pnsource = EPMAfem.discretize_rhs(beam, model, EPMAfem.cuda())[1]

function mass_concentrations(elm, x_)
    z = dimful(x_[1], u"nm", eq.dim_basis)
    x = dimful(x_[2], u"nm", eq.dim_basis)
    if x < 0u"nm"
        return elm == n"Cu" ? dimless(n"Cu".density, eq.dim_basis) : 0.0
    else
        return elm == n"Al" ? dimless(n"Al".density, eq.dim_basis) : 0.0
    end
end

ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(elm, x) for elm in mat], model)
EPMAfem.update_problem!(pnproblem, ρs)

probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), ϵ = ϵ -> ionizationcrosssection(n"Cu K", dimful(ϵ, u"eV", eq.dim_basis) |> ustrip), Ω = Ω -> 1.0)
f = EPMAfem.interpolable(probe, system * pnsource)

heatmap(-500u"nm":10u"nm":500u"nm", -800u"nm":10u"nm":0u"nm", (x, z) -> f(VectorValue(dimless(z, eq.dim_basis), dimless(x, eq.dim_basis))), aspect_ratio=:equal, colorbar=false)
savefig("CuAl.png")
