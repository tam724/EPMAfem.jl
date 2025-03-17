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

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0), (30)))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(5, 1)
model = EPMAfem.DiscretePNModel(space_model, 0.0:0.1:1.0, direction_model)

equations = EPMAfem.PNEquations()
excitation = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [0.8, 0.7, 0.6], normalize.([VectorValue(-1.0, 0.0, 0.0)]))
extraction = EPMAfem.PNExtraction([0.1, 0.2], equations)

updatable_pnproblem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu(), updatable=true)
discrete_system = EPMAfem.implicit_midpoint(updatable_pnproblem.problem, EPMAfem.PNDirectSolver)

discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cpu())
discrete_ext = EPMAfem.discretize_extraction(extraction, model, EPMAfem.cpu(), updatable=true)

prob = EPMAfem.EPMAProblem(updatable_pnproblem, discrete_rhs, discrete_ext)

EPMAfem.update_standard_intensities!(prob)

# compute the "true measurements"
function mass_concentrations(e, x)
    return e==1 ? 0.4 : 0.6
end
true_ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:2], model)
@time true_meas = prob(true_ρs)

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

Nδs = 20
ρs = similar(true_ρs)
ρs .= 0.5
hs = [1/3^i for i in 0:8]
taylor_1st = zeros(size(hs)..., Nδs)
taylor_2nd_ad = zeros(size(hs)..., Nδs)
taylor_2nd_fd_02 = zeros(size(hs)..., Nδs)
taylor_2nd_fd_04 = zeros(size(hs)..., Nδs)
taylor_2nd_fd_05 = zeros(size(hs)..., Nδs)

grad = Zygote.gradient(objective_function, ρs)

grad_fd_02 = finite_difference_grad(objective_function, ρs, 1e-2)
grad_fd_04 = finite_difference_grad(objective_function, ρs, 1e-4)
grad_fd_05 = finite_difference_grad(objective_function, ρs, 1e-5)
C = objective_function(ρs)

for n in 1:Nδs
    δρs = randn(size(ρs)...) |> normalize
    for (i, h) in enumerate(hs)
        @show n, i
        Cδρs = objective_function(ρs + h*δρs)
        taylor_1st[i, n] = abs(Cδρs - C)
        taylor_2nd_ad[i, n] = abs(Cδρs - C - h*dot(grad[1], δρs))
        taylor_2nd_fd_02[i, n] = abs(Cδρs - C - h*dot(grad_fd_02, δρs))
        taylor_2nd_fd_04[i, n] = abs(Cδρs - C - h*dot(grad_fd_04, δρs))
        taylor_2nd_fd_05[i, n] = abs(Cδρs - C - h*dot(grad_fd_05, δρs))
    end
end

gr()
plot(hs, sum(taylor_1st; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="1st rem.", marker=:x, color=1)
plot!(hs, sum(taylor_2nd_fd_02; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (FD, 1e-2)", marker=:x, color=3)
plot!(hs, sum(taylor_2nd_fd_04; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (FD, 1e-4)", marker=:x, color=4)
plot!(hs, sum(taylor_2nd_fd_05; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (FD, 1e-5)", marker=:x, color=5)
plot!(hs, sum(taylor_2nd_ad; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (Adjoint)", marker=:x, color=2)

plot!(hs, 2e-3*hs, xaxis=:log, yaxis=:log, color=:gray, ls=:dash, label="1st order")
plot!(hs, 2e-4*hs.^2, xaxis=:log, yaxis=:log, color=:gray, ls=:dashdot, label="2nd order")
plot!(size=(400, 300), dpi=1000, legend=:bottomright, fontfamily="Computer Modern")
xlabel!(L"h")
ylabel!("Taylor remainder")
savefig("figures/epma_taylor_remainder_direct_1D.png")
