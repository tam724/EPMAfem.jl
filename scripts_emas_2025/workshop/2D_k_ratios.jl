using Revise
using Plots
using EPMAfem
using NeXLCore
using NeXLMatrixCorrection
using Unitful
using EPMAfem.Gridap
using LinearAlgebra

include("../../scripts/plot_overloads.jl")

NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)
dimless, dimful = NExt.dimless, NExt.dimful

mat = [n"Al", n"Fe"]
dets = NExt.EPMADetector.([n"Al K-L2", n"Fe K-L2"], Ref(VectorValue(1.0, 0.3) |> normalize))

ϵ_range = range(3u"keV", 13u"keV", length=100)
eq = NExt.epma_equations(mat, dets, ϵ_range, 27);

model = NExt.epma_model(eq, (-1500u"nm", 0u"nm", -1300u"nm", 1300u"nm"), (150, 260), 21)
# model = NExt.epma_model(eq, (-1500u"nm", 0u"nm", -1300u"nm", 1300u"nm"), (100, 100), 21)
pnproblem = EPMAfem.discretize_problem(eq, model, EPMAfem.cuda(), updatable=true)
system = EPMAfem.implicit_midpoint(pnproblem.problem, EPMAfem.PNSchurSolver);

detectors = NExt.discretize_detectors(eq, model, EPMAfem.cuda(), updatable=true)

# @time detectors = NExt.discretize_detectors(eq, model, EPMAfem.cuda(), updatable=true)
# @profview detectors = NExt.discretize_detectors(eq, model, EPMAfem.cuda(), updatable=true)

###### change the material to inhomogeneous
function mass_concentrations(elm, x_)
    z = dimful(x_[1], u"nm", eq.dim_basis)
    x = dimful(x_[2], u"nm", eq.dim_basis)
    if z > -100u"nm" && x > -100u"nm" && x < 100u"nm"
        return elm == n"Fe" ? dimless(n"Fe".density, eq.dim_basis) : 0.0
    else
        return elm == n"Al" ? dimless(n"Al".density, eq.dim_basis) : 0.0
    end
end

ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(elm, x) for elm in mat], model)

EPMAfem.update_vector!(detectors[1], ρs)
EPMAfem.update_vector!(detectors[2], ρs)
EPMAfem.update_problem!(pnproblem, ρs)

func = EPMAfem.absorption_approximation(detectors[1].bxp_updater, ρs)
heatmap(dimless.(-1500u"nm":5u"nm":0u"nm", eq.dim_basis), dimless.(-1300u"nm":5u"nm":1300u"nm", eq.dim_basis), func, aspect_ratio=:equal, swapxy=true)

# quicky normalize the vectors
for i in 1:2
    detectors[i].vector.bϵ .= detectors[i].vector.bϵ / maximum(abs.(detectors[i].vector.bϵ))
    detectors[i].vector.bxp .= detectors[i].vector.bxp / maximum(abs.(detectors[i].vector.bxp))
    detectors[i].vector.bΩp .= detectors[i].vector.bΩp / maximum(abs.(detectors[i].vector.bΩp))
end

probe = EPMAfem.PNProbe(model, EPMAfem.cuda(); Ω = Ω -> 1.0, ϵ = ϵ -> 1.0)

# f = EPMAfem.interpolable(probe, detectors[1].vector * system)
# contourf(dimless.(-800u"nm":5u"nm":800u"nm", eq.dim_basis), dimless.(-800u"nm":5u"nm":0u"nm", eq.dim_basis), (x, z) -> -f(VectorValue(z, x)), aspect_ratio=:equal, flipxy=true)

beams_cont = EPMAfem.pn_excitation([(x=dimless(x_, eq.dim_basis), y=0.0) for x_ in -400u"nm":10u"nm":400u"nm"], [dimless(11.0u"keV", eq.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)]; beam_energy_σ=0.05, beam_position_σ=dimless(30u"nm", eq.dim_basis))
beams = EPMAfem.discretize_rhs(beams_cont, model, EPMAfem.cuda())

meas = [det.vector for det in detectors] * system * beams

plot(-400u"nm":10u"nm":400u"nm", meas[1, 1, :, 1])
plot!(-400u"nm":10u"nm":400u"nm", meas[2, 1, :, 1])

f = EPMAfem.interpolable(probe, system * beams[1, 1, 1])
contourf(dimless.(-800u"nm":5u"nm":800u"nm", eq.dim_basis), dimless.(-800u"nm":5u"nm":0u"nm", eq.dim_basis), (x, z) -> f(VectorValue(z, x)), aspect_ratio=:equal)

anim = @animate for i in [1, 10, 20, 30, 40, 50, 60, 70, 80]
    @show i
    f = EPMAfem.interpolable(probe, system * beams[1, i, 1])
    contourf(dimless.(-1300u"nm":30u"nm":1300u"nm", eq.dim_basis), dimless.(-1500u"nm":30u"nm":0u"nm", eq.dim_basis) , (x, z) -> f(VectorValue(z, x)), aspect_ratio=:equal)
end

gif(anim, fps=1)
