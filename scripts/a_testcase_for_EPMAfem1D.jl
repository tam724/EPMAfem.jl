using Revise
using EPMAfem
using Gridap
using GridapGmsh
using LinearAlgebra
using Plots
using Distributions
using ConcreteStructs
using QuadGK
include("plot_overloads.jl")
include("grid_gen.jl")
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

# plots for debugging interval_ray_intersection
# nothing_to_zero(x) = isnothing(x) ? 0.0 : x

# plot(-1:0.01:1, x -> nothing_to_zero(interval_ray_intersection(0.1, x, -0.1)[1]))
# plot!(-1:0.01:1, x -> nothing_to_zero(interval_ray_intersection(0.1, x, -0.1)[2]))

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

# distribution function plots
# plot(-1:0.01:1, x -> distr_func(x, 0.01; R=0.1, b=1.0, μ_in=10.0, μ_out=0.0))
# plot(-1:0.01:1, Ω -> distr_func(0.1, Ω; R=0.1, b=1.0, μ_in=10.0, μ_out=0.0))

heatmap(-1:0.001:1, -1:0.001:1, (x, Ω) -> distr_func(x, Ω; R=0.1, b=1.0, μ_in=10.0, μ_out=0.1))
Plots.xlabel!("x")
Plots.ylabel!("Ω")

# Plots.xlabel!("x")
# Plots.ylabel!("Ω")

function quad_dist_func(x, func; R, kwargs...)
    if norm(x) <= R
        return quadgk(Ω -> 0.5*func(Ω)*distr_func(x, Ω; R=R, kwargs...), -1.0, 1.0)[1]
    elseif x > R
        return quadgk(Ω -> 0.5*func(Ω)*distr_func(x, Ω; R=R, kwargs...), 0.0, 1.0)[1]
    elseif x < R
        return quadgk(Ω -> 0.5*func(Ω)*distr_func(x, Ω; R=R, kwargs...), -1.0, 0.0)[1]
    end
end
intensity(x; R, kwargs...) = quad_dist_func(x, Ω -> 1.0; R=R, kwargs...)

plot(-1:0.01:1, x -> intensity(x; R=0.1, b=10.0, μ_in=100.0, μ_out=1.0))

plot(-1:0.01:1, x -> intensity(x; R=0.1, b=10.0, μ_in=1000.0, μ_out=1.0))
hline!([10.0/1000.0])

plot()
for R in [0.0001, 0.001, 0.01, 0.1]
    plot!(-1:0.01:1, x -> intensity(x; R=R, b=1.0/(2.0*R), μ_in=1.0, μ_out=0.0))
end
plot!()



### 1D numerical solution

@concrete struct DummyPNEquations <: EPMAfem.AbstractMonochromPNEquations end
EPMAfem.number_of_elements(::DummyPNEquations) = 1
EPMAfem.scattering_coefficient(::DummyPNEquations, e) = 0.0
EPMAfem.scattering_kernel(::DummyPNEquations, e) = μ -> 0.0
EPMAfem.absorption_coefficient(::DummyPNEquations, e) = 1.0
function EPMAfem.mass_concentrations(::DummyPNEquations, e, x)
    return (norm(x) < 0.15) ? 50.0 : 1.0
end

eq = DummyPNEquations()


# cartesian space model
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.5, 0.5), (400)))

grid_gen_1D((-0.5, 0.5); res=0.01, highres=(dist=0.3, res=0.0001, points=[-0.15, 0.15]))
space_model = EPMAfem.SpaceModels.GridapSpaceModel(DiscreteModelFromFile("/tmp/tmp_msh.msh"))

EPMAfem.SpaceModels.n_basis(space_model)
plot()
vline!(getindex.(space_model.discrete_model.grid.node_coordinates, 1), color=:gray, α=0.1)
vline!(getindex.(space_model.discrete_model.grid.node_coords, 1), color=:gray, α=0.1)

plotly()

gr()
plot()
sol = Dict()
for N in [1, 3, 13, 19, 21, 27]
    direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1)
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

    Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(direction_model, Ω -> 1.0)

    func = EPMAfem.SpaceModels.interpolable((p=solp*Ωp, m=solm*Ωm), space_model)

    vals = -0.5:0.0001:0.5 .|> (z -> func(Gridap.VectorValue(z)))

    p = plot!(-0.5:0.0001:0.5, vals, label="P$(N)")
    @show "$N, max=$(maximum(vals))"
    display(p)
end

kwargs_analytic = (R=0.15, b=10.0, μ_in=50.0, μ_out=1.0)
# I_0 = intensity(0; kwargs_analytic...)
plot!(-0.5:0.0001:0.5, x -> 4*π*intensity(x; kwargs_analytic...), label="analytic_solution")
plot!(-0.5:0.0001:0.5, x -> 4*π*intensity(x; kwargs_analytic...), label="analytic_solution")
plot!(-0.5:0.0001:0.5, x -> 4*π*quad_dist_func(x, Ω->Ω; kwargs_analytic...), label="analytic_solution")

savefig("plot_with_refined.png")

p1 = heatmap(-0.5:0.001:0.5, -1:0.001:1, (x, Ω) -> distr_func(x, Ω; kwargs_analytic...), clims=(-0.1, 0.3))

function distr_func_numeric(x, Ω, N)
    direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1)
    xp, xm = EPMAfem.SpaceModels.eval_basis(space_model, VectorValue(x))
    θ = acos(Ω)
    Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(direction_model, VectorValue(Ω, sin(θ)))

    solp, solm = sol[N]
    return dot(xp, solp * Ωp) + dot(xm, solm * Ωm)
end

p2 = heatmap(-0.5:0.01:0.5, -1:0.01:1, (x, Ω) -> distr_func_numeric(x, Ω, 3), clims=(-0.1, 0.3))

plot(p1, p2, size=(700, 300))

nothing
