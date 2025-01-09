using Revise
using EPMAfem
using EPMAfem.Gridap
using Optim
using Lux
using Zygote
using LinearAlgebra
using ComponentArrays
include("../scripts/plot_overloads.jl")

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1.5, 1.5), (40, 120)))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(21, 2)
model = EPMAfem.DiscretePNModel(space_model, 0.0:0.05:1.0, direction_model)

equations = EPMAfem.PNEquations()
excitation = EPMAfem.PNExcitation([(x=x_, y=0.0) for x_ in -0.7:0.02:0.7], [0.8, 0.7], [VectorValue(-1.0, 0.5, 0.0) |> normalize, VectorValue(-1.0, 0.0, 0.0), VectorValue(-1.0, -0.5, 0.0) |> normalize])
extraction = EPMAfem.PNExtraction([0.1, 0.2], equations)

updatable_pnproblem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda(), updatable=true)
discrete_system = EPMAfem.schurimplicitmidpointsystem(updatable_pnproblem.problem)

discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())
discrete_ext = EPMAfem.discretize_extraction(extraction, model, EPMAfem.cuda(), updatable=true)

prob = EPMAfem.EPMAProblem(updatable_pnproblem, discrete_rhs, discrete_ext)
EPMAfem.update_standard_intensities!(prob)

probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω = Ω -> 1.0, ϵ = ϵ -> EPMAfem.extraction_energy_distribution(extraction, 1, ϵ))
@time averaged_solution = probe(prob.system * prob.excitations[1, 10, 2])

@time @gif for i in 20:50
    averaged_solution = probe(prob.system * prob.excitations[1, i, 2])
    func = EPMAfem.SpaceModels.interpolable(averaged_solution, space_model)
    contourf(range(-1, 0, 40), range(-1.5, 1.5, 120), func, swapxy=true, aspect_ratio=:equal)
    vline!([-0.2, 0.2])
end

# compute the "true measurements"
function mass_concentrations(e, x)
    if x[2] < -0.2 || x[2] > 0.2
        return e==1 ? 1.0 : 0.0
    else
        return e==1 ? 0.0 : 1.0
    end
end

true_ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:2], model)
true_meas = prob(true_ρs)

function compute_cell_midpoints(space_model)
    trian = Gridap.get_triangulation(space_model.discrete_model) 
    cell_node_ids = trian.grid.cell_node_ids
    midpoints = [mean(trian.grid.node_coords[cell_node_id]) for cell_node_id in cell_node_ids]
    return midpoints
end

x = transpose(getindex.(compute_cell_midpoints(space_model)[:], Ref(2)) ./ 3.0 .+ 0.5)
xy = zeros(2, 4800)
xy[1, :] .= -getindex.(compute_cell_midpoints(space_model)[:], Ref(1))
xy[2, :] .= getindex.(compute_cell_midpoints(space_model)[:], Ref(2)) ./ 3.0 .+ 0.5

lux_model = Chain(Dense(2, 10, relu), Dense(10, 2), Lux.softmax)
ps, st = Lux.setup(Lux.Random.default_rng(), lux_model)
p0 = ComponentArray(ps)
ρs = Lux.apply(lux_model, xy, p0, st)[1]
plt1 = heatmap(reshape(ρs[1, :], (40, 120)), aspect_ratio=:equal)
plt2 = heatmap(reshape(ρs[2, :], (40, 120)), aspect_ratio=:equal)
display(plot(plt1, plt2))

plt1 = heatmap(reshape(true_ρs[1, :], (40, 120)), aspect_ratio=:equal)
plt2 = heatmap(reshape(true_ρs[2, :], (40, 120)), aspect_ratio=:equal)
display(plot(plt1, plt2))

cached_intensities = zeros(size(true_meas))
# function objective_function2(p)
#     ρs = Lux.apply(lux_model, x, p, st)[1]
#     return sum((ρs .- true_ρs).^2)*1e-8
# end

function objective_function(p)
    intensities = prob(Lux.apply(lux_model, xy, p, st)[1])
    Zygote.ignore() do 
        cached_intensities .= intensities
    end
    return sum((true_meas .- intensities).^2)/length(true_meas)
end

val_and_grad = Zygote.withgradient(objective_function, p0)

function fg!(F, G, p)
    ρs = Lux.apply(lux_model, xy, p, st)[1]
    plt1 = heatmap(reshape(ρs[1, :], (40, 120)), aspect_ratio=:equal)
    plt2 = heatmap(reshape(ρs[2, :], (40, 120)), aspect_ratio=:equal)
    plt3 = heatmap(reshape(true_ρs[1, :], (40, 120)), aspect_ratio=:equal)
    plt4 = heatmap(reshape(true_ρs[2, :], (40, 120)), aspect_ratio=:equal)
    display(plot(plt1, plt2, plt3, plt4))
    if G !== nothing
        val_and_grad = Zygote.withgradient(objective_function, p)
        G .= val_and_grad.grad[1]
        return val_and_grad.val
    end
    if F !== nothing
        return objective_function(p)
    end
end

int_store = []
function cb(tr)
    push!(int_store, copy(cached_intensities))
    # @show tr[end].x
    p = plot()
    color_i = 1
    for i in 1:2, j in 1:2, k in 1:3
        plot!(cached_intensities[i, j, :, k], color=color_i, ls=:dash)
        plot!(true_meas[i, j, :, k], color=color_i)
        color_i += 1
    end
    display(p)
    return false
end

# p0 = Optim.minimizer(res)

res = optimize(Optim.only_fg!(fg!), p0, Optim.LBFGS(), Optim.Options(callback=cb, store_trace=true, extended_trace=true, iterations=50, time_limit=1000, g_abstol=1e-10, g_reltol=1e-10))

@gif for (s_i, state) in enumerate(res.trace)
    p = state.metadata["x"]
    ρs = Lux.apply(lux_model, xy, p, st)[1]
    plt1 = heatmap(reshape(ρs[1, :], (40, 120)), aspect_ratio=:equal, clim=(0, 1), cb=nothing)
    plt2 = heatmap(reshape(ρs[2, :], (40, 120)), aspect_ratio=:equal, clim=(0, 1), cb=nothing)
    plt3 = heatmap(reshape(true_ρs[1, :], (40, 120)), aspect_ratio=:equal, clim=(0, 1), cb=nothing)
    plt4 = heatmap(reshape(true_ρs[2, :], (40, 120)), aspect_ratio=:equal, clim=(0, 1), cb=nothing)
    color_i = 1
    plt5 = plot()
    for i in 1:2, j in 1:2, k in 1:3
        plot!(int_store[s_i][i, j, :, k], color=color_i)
        plot!(true_meas[i, j, :, k], color=color_i, legend=false, ls=:dash)
        color_i += 1
    end
    l = @layout [
    grid(2, 2) grid(1, 1)]
    plot(plt1, plt2, plt3, plt4, plt5, layout=l, size=(1500, 400))
    title!("iteration: $s_i")
end fps=5

plot(plot(rand(10)), plot(rand(10)), plot(rand(10)), plot(rand(10)), plot(rand(10)), layout=l)