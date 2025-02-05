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

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -0.5, 0.5), (40, 40)))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(21, 2)
model = EPMAfem.DiscretePNModel(space_model, 0.0:0.05:1.0, direction_model)

equations = EPMAfem.PNEquations()
excitation = EPMAfem.PNExcitation([(x=x_, y=0.0) for x_ in -0.2:0.05:0.2], [0.8, 0.7], normalize.([VectorValue(-1.0, 0.5, 0.0), VectorValue(-1.0, 0.0, 0.0), VectorValue(-1.0, -0.5, 0.0)]))
extraction = EPMAfem.PNExtraction([0.1, 0.2], equations)

updatable_pnproblem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda(), updatable=true)
discrete_system = EPMAfem.schurimplicitmidpointsystem(updatable_pnproblem.problem)

discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())
discrete_ext = EPMAfem.discretize_extraction(extraction, model, EPMAfem.cuda(), updatable=true)

prob = EPMAfem.EPMAProblem(updatable_pnproblem, discrete_rhs, discrete_ext)
EPMAfem.update_standard_intensities!(prob)

# compute the "true measurements"
function mass_concentrations(e, x)
    if x[2] > -0.1 && x[2] < 0.25
    # if sqrt((x[1] + 0.05)^2 + (x[2] - 0.3)^2) < 0.2
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

objective_function(ρs) = sum((true_meas .- prob(ρs)).^2)# / length(true_meas)

Nδs = 20
ρs = similar(true_ρs)
ρs .= 0.5
hs = [1/2^i for i in 3:12]
taylor_1st = zeros(size(hs)..., Nδs)
taylor_2nd_ad = zeros(size(hs)..., Nδs)
taylor_2nd_fd_01 = zeros(size(hs)..., Nδs)
grad = Zygote.gradient(objective_function, ρs)
grad_fd_01 = finite_difference_grad(objective_function, ρs, 1e-1)
# grad_fd_03 = finite_difference_grad(objective_function, ρs, 1e-3)
C = objective_function(ρs)

for n in 1:Nδs
    δρs = randn(size(ρs)...)
    for (i, h) in enumerate(hs)
        Cδρs = objective_function(ρs + h*δρs)
        taylor_1st[i, n] = abs(Cδρs - C)
        taylor_2nd_ad[i, n] = abs(Cδρs - C - h*dot(grad[1], δρs))
        taylor_2nd_fd_01[i, n] = abs(Cδρs - C - h*dot(grad_fd_01, δρs))
    end
end

scatter(hs, sum(taylor_1st; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="1st rem.")
scatter!(hs, sum(taylor_2nd_ad; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (Adjoint)")
scatter!(hs, sum(taylor_2nd_fd_01; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (FD, Δ=0.1)")
plot!(hs, hs, xaxis=:log, yaxis=:log, color=:gray, ls=:dash, label="1st order")
plot!(hs, 0.1*hs.^2, xaxis=:log, yaxis=:log, color=:gray, ls=:dashdot, label="2nd order")
plot!(size=(400, 300), dpi=1000, legend=:bottomright)
xlabel!(L"h")
ylabel!("taylor remainder")
savefig("figures/epma_taylor_remainder.png")