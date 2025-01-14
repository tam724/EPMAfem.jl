using Revise
using EPMAfem
using EPMAfem.Gridap
using Optim
using Lux
using Zygote
using LinearAlgebra
using ComponentArrays
using LaTeXStrings
include("../scripts/plot_overloads.jl")

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1.5, 1.5), (40, 120)))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(21, 2)
model = EPMAfem.DiscretePNModel(space_model, 0.0:0.05:1.0, direction_model)

equations = EPMAfem.PNEquations()
excitation = EPMAfem.PNExcitation([(x=x_, y=0.0) for x_ in -0.7:0.02:0.7], [0.8, 0.7, 0.6], normalize.([VectorValue(-1.0, 0.75, 0.0), VectorValue(-1.0, 0.5, 0.0), VectorValue(-1.0, 0.25, 0.0), VectorValue(-1.0, 0.0, 0.0), VectorValue(-1.0, -0.25, 0.0), VectorValue(-1.0, -0.5, 0.0), VectorValue(-1.0, -0.75, 0.0)]))
extraction = EPMAfem.PNExtraction([0.1, 0.2], equations)

updatable_pnproblem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu(), updatable=true)
discrete_system = EPMAfem.schurimplicitmidpointsystem(updatable_pnproblem.problem)

discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cpu())
discrete_ext = EPMAfem.discretize_extraction(extraction, model, EPMAfem.cpu(), updatable=true)

prob = EPMAfem.EPMAProblem(updatable_pnproblem, discrete_rhs, discrete_ext)
EPMAfem.update_standard_intensities!(prob)

# compute the "true measurements"
function mass_concentrations(e, x)
    if sqrt((x[1] + 0.05)^2 + (x[2] - 0.3)^2) < 0.2
        return e==1 ? 1.0 : 0.0
    else
        return e==1 ? 0.0 : 1.0
    end
end

true_ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:2], model)
true_meas = prob(true_ρs)

# taylor remainder (should be run on CPU..)
function finite_difference_grad(f, p, h)
    val = f(p)
    grad = similar(p)
    for i in eachindex(p)
        @show i, length(p)
        p[i] += h
        grad[i] = (f(p) - val)/h
        p[i] -= h
    end
    return grad
end

objective_function(ρs) = sum((true_meas .- prob(ρs)).^2) / length(true_meas)

ρs = similar(true_ρs)
ρs .= 0.5
δρs = randn(size(ρs)...)
hs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
taylor_1st = zeros(size(hs))
taylor_2nd_ad = zeros(size(hs))
grad = Zygote.gradient(objective_function, ρs)
grad_fd_01 = finite_difference_grad(objective_function, ρs, 1e-1)
grad_fd_03 = finite_difference_grad(objective_function, ρs, 1e-3)
for (i, h) in enumerate(hs)
    Cδρs = objective_function(ρs + h*δρs)
    C = objective_function(ρs)
    taylor_1st[i] = abs(Cδρs - C)
    taylor_2nd_ad[i] = abs(Cδρs - C - h*dot(grad[1], δρs))
end
scatter(hs, taylor_1st, xaxis=:log, yaxis=:log)
scatter!(hs, taylor_2nd_ad, xaxis=:log, yaxis=:log)
plot!(hs, hs, xaxis=:log, yaxis=:log)
plot!(hs, hs.^2, xaxis=:log, yaxis=:log)
