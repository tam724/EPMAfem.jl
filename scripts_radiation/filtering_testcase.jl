using Revise
using EPMAfem
using Gridap
using LinearAlgebra
using Plots
using Distributions
using ConcreteStructs
using QuadGK
include("../scripts/plot_overloads.jl")
Makie.inline!(false)


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

### 1D numerical solution

@concrete struct DummyPNEquations <: EPMAfem.AbstractMonochromPNEquations end
EPMAfem.number_of_elements(::DummyPNEquations) = 1
EPMAfem.scattering_coefficient(::DummyPNEquations, e) = 0.0
EPMAfem.scattering_kernel(::DummyPNEquations, e) = μ -> 0.0
EPMAfem.absorption_coefficient(::DummyPNEquations, e) = 1.0
function EPMAfem.mass_concentrations(::DummyPNEquations, e, x)
    return (norm(x) < 0.15) ? 50.0 : 0.01
end

eq = DummyPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.5, 0.5), (300)))

sol = Dict()
for N in [1,3,7,15,21,27]
    for σ_f in [0.0,0.01,0.1,1.0]
        for α in [1.0,5.0,15.0]
            println(string(N)*","*string(σ_f)*","*string(α))
            direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1, σ_f, α)
            model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)

            function q(x)
                return (norm(x) < 0.15) ? 10.0 : 0.0
            end

            source = EPMAfem.PNXΩSource(q, Ω -> 1.0)
            rhs = EPMAfem.discretize_rhs(source, model, EPMAfem.cpu())
            problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu())
            system = EPMAfem.system2(problem, EPMAfem.Krylov.minres)

            solution = EPMAfem.allocate_solution_vector(system)
            EPMAfem.solve(solution, system, rhs)
            solp, solm = EPMAfem.pmview(solution, model)
            sol["N"*string(N)*","*"σ_f"*string(σ_f)*","*"α"*string(α)] = [solp, solm]
        end
    end
end


plot()
for key in collect(keys(sol))
    if startswith(key, "27,0.01")
        solp, solm = sol[key]
        func = EPMAfem.SpaceModels.interpolable((p=solp[:, 1] |> collect, ), space_model)
        p = plot!(-0.5:0.0001:0.5  , (z -> func(Gridap.VectorValue(z))), label=key)
        display(p)
    end
end
# p2 = plot()
# for key in keys(sol)
#     if startswith(key, "27") && occursin(key, "0.1")
#         solp, solm = sol[key]
#         func = EPMAfem.SpaceModels.interpolable((p=solp[:, 1] |> collect, ), space_model)
#         plot!(-0.5:0.0001:0.5  , (z -> func(Gridap.VectorValue(z))), label=key)
#     end
# end

# plot(p1, p2, size=(700, 300))
