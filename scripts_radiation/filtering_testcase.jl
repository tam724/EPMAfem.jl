using Revise
using EPMAfem
using EPMAfem.Gridap
using LinearAlgebra
using Plots
using Distributions
using ConcreteStructs
using QuadGK
# include("../scripts/plot_overloads.jl")
# Makie.inline!(false)

plotly()
## 1D exact solution
function interval_ray_intersection(R, x, Ω)
    if iszero(Ω) return (nothing, nothing) end

    if norm(x) < R
        if Ω > 0
            return (nothing, (x + R) / Ω)
        else #  Ω < 0
            return (nothing, (x - R) / Ω)
        end
    else # norm(x) >= R
        if x > 0 && Ω < 0
            return (nothing, nothing)
        elseif x > 0 && Ω > 0
            return (-(R - x)/Ω, 2*R/abs(Ω))
        elseif x < 0 && Ω > 0
            return (nothing, nothing)
        else # x < 0 && Ω < 0
            return (-(-R-x)/Ω, 2*R/abs(Ω))
        end
    end
end

function distr_func(x, Ω; R, b, μ_in, μ_out)
    if iszero(Ω)
        if norm(x) < R
            return b / μ_in
        else
            return 0.0
        end
    end
    path_before, path_in = interval_ray_intersection(R, x, Ω)
    contrib = 0.0
    if !isnothing(path_in)
        contrib += b*(-exp(-μ_in*path_in) + 1.0)/μ_in
    end
    if !isnothing(path_before)
        contrib *= exp(-μ_out*path_before)
    end
    return contrib
end

function intensity(x; R, kwargs...)
    if norm(x) <= R
        return quadgk(Ω -> 0.5*distr_func(x, Ω; R=R, kwargs...), -1.0, 1.0)[1]
    elseif x > R
        return quadgk(Ω -> 0.5*distr_func(x, Ω; R=R, kwargs...), 0.0, 1.0)[1]
    elseif x < R
        return quadgk(Ω -> 0.5*distr_func(x, Ω; R=R, kwargs...), -1.0, 0.0)[1]
    end
end

function flux(x; R, kwargs...)
    if norm(x) <= R
        return quadgk(Ω -> Ω*0.5*distr_func(x, Ω; R=R, kwargs...), -1.0, 1.0)[1]
    elseif x > R
        return quadgk(Ω -> Ω*0.5*distr_func(x, Ω; R=R, kwargs...), 0.0, 1.0)[1]
    elseif x < R
        return quadgk(Ω -> Ω*0.5*distr_func(x, Ω; R=R, kwargs...), -1.0, 0.0)[1]
    end
end

kwargs_analytic = (R=0.15, b=10.0, μ_in=50.0, μ_out=0.01)
I_0 = intensity(0; kwargs_analytic...)


### 1D numerical solution

@concrete struct DummyPNEquations <: EPMAfem.AbstractMonochromPNEquations end
EPMAfem.number_of_elements(::DummyPNEquations) = 1
EPMAfem.scattering_coefficient(::DummyPNEquations, e) = 0.0
EPMAfem.scattering_kernel(::DummyPNEquations, e) = μ -> 0.0
EPMAfem.absorption_coefficient(::DummyPNEquations, e) = 1.0
function EPMAfem.mass_concentrations(::DummyPNEquations, e, x)
    return (norm(x) < 0.15) ? 50.0 : 0.01
end

function distr_func_numeric(x, Ω, N, σ_f, α)
    direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1)
    xp, xm = EPMAfem.SpaceModels.eval_basis(space_model, VectorValue(x))
    θ = acos(Ω)
    Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(direction_model, VectorValue(Ω, sin(θ)))

    solp, solm = sol["N"*string(N)*","*"σ_f"*string(σ_f)*","*"α"*string(α)]
    return dot(xp, solp * Ωp) + dot(xm, solm * Ωm)
end

eq = DummyPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.5, 0.5), (300)))

sol = Dict()
Ω0th = Dict()
Ω1st = Dict()
for N in [1, 3, 7, 21, 27]
    for σ_f in [0.0, 0.01, 0.1, 1.0, 10.0]
        for α in [1.0, 5.0, 15.0]
            println(string(N)*","*string(σ_f)*","*string(α))
            direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1)
            model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)

            function q(x)
                return (norm(x) < 0.15) ? 10.0 : 0.0
            end

            source = EPMAfem.PNXΩSource(q, Ω -> 1.0)
            rhs = EPMAfem.discretize_rhs(source, model, EPMAfem.cpu())
            eq_filtered = EPMAfem.filter_exp(eq, σ_f, α)
            problem = EPMAfem.discretize_problem(eq_filtered, model, EPMAfem.cpu())
            system = EPMAfem.system2(problem, EPMAfem.Krylov.minres)

            solution = EPMAfem.allocate_solution_vector(system)
            EPMAfem.solve(solution, system, rhs)
            solp, solm = EPMAfem.pmview(solution, model)
            
            Ωp_0th, Ωm_0th = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0)
            Ωp_1st, Ωm_1st = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), EPMAfem.Dimensions.Ωz)

            sol["N"*string(N)*","*"σ_f"*string(σ_f)*","*"α"*string(α)] = [solp, solm]
            Ω0th["N"*string(N)*","*"σ_f"*string(σ_f)*","*"α"*string(α)] = [Ωp_0th, Ωm_0th]
            Ω1st["N"*string(N)*","*"σ_f"*string(σ_f)*","*"α"*string(α)] = [Ωp_1st, Ωm_1st]
        end
    end
end

Ω0th["N3,σ_f0.0,α1.0"][1][2]

# Plot 0th moment with and without filtering
no_filter_keys = ["N1,σ_f0.0,α1.0","N3,σ_f0.0,α1.0","N7,σ_f0.0,α1.0","N21,σ_f0.0,α1.0","N27,σ_f0.0,α1.0"]
no_filter = plot()
for key in no_filter_keys
        solp, solm = sol[key]
        Ωp_0th, Ωm_0th = Ω0th[key]
        func = EPMAfem.SpaceModels.interpolable((p=solp*Ωp_0th|> collect, m=solm*Ωm_0th |> collect), space_model)
        plot!(-0.5:0.0001:0.5  , (z -> func(Gridap.VectorValue(z))), label=key)
end
plot!(-0.5:0.0001:0.5, x -> 4*pi*intensity(x; kwargs_analytic...), label="analytical")

filter_keys = ["N1,σ_f0.1,α1.0","N3,σ_f0.1,α1.0","N7,σ_f0.1,α1.0","N21,σ_f0.1,α1.0","N27,σ_f0.1,α1.0"]
filter = plot()
for key in filter_keys
        solp, solm = sol[key]
        Ωp_0th, Ωm_0th = Ω0th[key]
        func = EPMAfem.SpaceModels.interpolable((p=solp*Ωp_0th|> collect, m=solm*Ωm_0th |> collect), space_model)
        plot!(-0.5:0.0001:0.5  , (z -> func(Gridap.VectorValue(z))), label=key)
end
plot!(-0.5:0.0001:0.5, x -> 4*pi*intensity(x; kwargs_analytic...), label="analytical")

filter2_keys = ["N1,σ_f0.01,α15.0","N3,σ_f0.01,α15.0","N7,σ_f0.01,α15.0","N21,σ_f0.01,α15.0","N27,σ_f0.01,α15.0"]
filter2 = plot()
for key in filter2_keys
        solp, solm = sol[key]
        Ωp_0th, Ωm_0th = Ω0th[key]
        func = EPMAfem.SpaceModels.interpolable((p=solp*Ωp_0th|> collect, m=solm*Ωm_0th |> collect), space_model)
        plot!(-0.5:0.0001:0.5  , (z -> func(Gridap.VectorValue(z))), label=key)
end
plot!(-0.5:0.0001:0.5, x -> 4*pi*intensity(x; kwargs_analytic...), label="analytical")

plot(no_filter, filter)

custom_cmap = cgrad([:green, :white, :red], [-1.0, 0.0, 1.0])

# Plot Heatmaps aswell
heatmap_sigma0_alpha1 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 0.0, 1.0), clims=(-0.1,0.3), title="σ_f=0, α=1")
heatmap_sigma0_alpha5 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 0.0, 5.0), clims=(-0.1,0.3), title="σ_f=0, α=5")
heatmap_sigma0_alpha15 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 0.0, 15.0), clims=(-0.1,0.3), title="σ_f=0, α=15")
heatmap_sigma01_alpha1 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 0.1, 1.0), clims=(-0.1,0.3), title="σ_f=0.1, α=1")
heatmap_sigma01_alpha5 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 0.1, 5.0), clims=(-0.1,0.3), title="σ_f=0.1, α=5")
heatmap_sigma01_alpha15 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 0.1, 15.0), clims=(-0.1,0.3), title="σ_f=0.1, α=15")
heatmap_sigma1_alpha1 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 1.0, 1.0), clims=(-0.1,0.3), title="σ_f=1, α=1")
heatmap_sigma1_alpha5 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 1.0, 5.0), clims=(-0.1,0.3), title="σ_f=1, α=5")
heatmap_sigma1_alpha15 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 1.0, 15.0), clims=(-0.1,0.3), title="σ_f=1, α=15", color=custom_cmap)
heatmap_sigma10_alpha1 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 10.0, 1.0), clims=(-0.1,0.3), title="σ_f=10, α=1")
heatmap_sigma10_alpha5 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 10.0, 5.0), clims=(-0.1,0.3), title="σ_f=10, α=5")
heatmap_sigma10_alpha15 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 27, 10.0, 15.0), clims=(-0.1,0.3), title="σ_f=10, α=15")

plot(heatmap_sigma0_alpha1,heatmap_sigma01_alpha1,heatmap_sigma1_alpha1,heatmap_sigma10_alpha1,
heatmap_sigma0_alpha5,heatmap_sigma01_alpha5,heatmap_sigma1_alpha5,heatmap_sigma10_alpha5,
heatmap_sigma0_alpha15,heatmap_sigma01_alpha15,heatmap_sigma1_alpha15,heatmap_sigma10_alpha15, layout=(3,4), size=(1500, 900))



# Plot 1st Moment 
no_filter_1st_keys = ["N1,σ_f0.0,α1.0","N3,σ_f0.0,α1.0","N7,σ_f0.0,α1.0","N21,σ_f0.0,α1.0","N27,σ_f0.0,α1.0"]
no_filter_1st = plot()
for key in no_filter_1st_keys
        solp, solm = sol[key]
        Ωp_1st, Ωm_1st = Ω1st[key]
        func = EPMAfem.SpaceModels.interpolable((p=solp*Ωp_1st|> collect, m=solm*Ωm_1st |> collect), space_model)
        plot!(-0.5:0.0001:0.5  , (z -> func(Gridap.VectorValue(z))), label=key)
end
plot!(-0.5:0.0001:0.5, x -> 4*pi*flux(x; kwargs_analytic...), label="analytical")

filter_1st_keys = ["N1,σ_f0.1,α1.0","N3,σ_f0.1,α1.0","N7,σ_f0.1,α1.0","N21,σ_f0.1,α1.0","N27,σ_f0.1,α1.0"]
filter_1st = plot()
for key in filter_1st_keys
        solp, solm = sol[key]
        Ωp_1st, Ωm_1st = Ω1st[key]
        func = EPMAfem.SpaceModels.interpolable((p=solp*Ωp_1st|> collect, m=solm*Ωm_1st |> collect), space_model)
        plot!(-0.5:0.0001:0.5  , (z -> func(Gridap.VectorValue(z))), label=key)
end
plot!(-0.5:0.0001:0.5, x -> 4*pi*flux(x; kwargs_analytic...), label="analytical")

filter2_1st_keys = ["N1,σ_f0.1,α15.0","N3,σ_f0.1,α15.0","N7,σ_f0.1,α15.0","N21,σ_f0.1,α15.0","N27,σ_f0.1,α15.0"]
filter2_1st = plot()
for key in filter2_1st_keys
        solp, solm = sol[key]
        Ωp_1st, Ωm_1st = Ω1st[key]
        func = EPMAfem.SpaceModels.interpolable((p=solp*Ωp_1st|> collect, m=solm*Ωm_1st |> collect), space_model)
        plot!(-0.5:0.0001:0.5  , (z -> func(Gridap.VectorValue(z))), label=key)
end
plot!(-0.5:0.0001:0.5, x -> 4*pi*flux(x; kwargs_analytic...), label="analytical")

plot(no_filter_1st, filter2_1st)
