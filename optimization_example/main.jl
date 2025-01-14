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
model = EPMAfem.DiscretePNModel(space_model, 0.0:0.01:1.0, direction_model)

equations = EPMAfem.PNEquations()
excitation = EPMAfem.PNExcitation([(x=x_, y=0.0) for x_ in -0.7:0.02:0.7], [0.8, 0.7, 0.6], normalize.([VectorValue(-1.0, 0.75, 0.0), VectorValue(-1.0, 0.5, 0.0), VectorValue(-1.0, 0.25, 0.0), VectorValue(-1.0, 0.0, 0.0), VectorValue(-1.0, -0.25, 0.0), VectorValue(-1.0, -0.5, 0.0), VectorValue(-1.0, -0.75, 0.0)]))
extraction = EPMAfem.PNExtraction([0.1, 0.2], equations)

updatable_pnproblem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda(), updatable=true)
discrete_system = EPMAfem.schurimplicitmidpointsystem(updatable_pnproblem.problem)

discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())
discrete_ext = EPMAfem.discretize_extraction(extraction, model, EPMAfem.cuda(), updatable=true)

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

# plots of forward problem
let
    probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω = Ω -> 1.0, ϵ = ϵ -> EPMAfem.extraction_energy_distribution(extraction, 1, ϵ))

    averaged_sol = probe(discrete_system * discrete_rhs[1, 36, 4])
    averaged_sol2 = probe(discrete_system * discrete_rhs[1, 36, 1])
    func = EPMAfem.SpaceModels.interpolable(averaged_sol, space_model)
    func2 = EPMAfem.SpaceModels.interpolable(averaged_sol2, space_model)
    plot()
    for i in 36:1:40
        plot!(-0.8:0.01:0.8, x -> EPMAfem.beam_space_distribution(excitation, i, Point(0.0, x, 0.0))*0.2, color=:gray, ls=:dash, label=nothing)
    end
    plot!(-0.8:0.01:0.8, x -> EPMAfem.beam_space_distribution(excitation, 36, Point(0.0, x, 0.0))*0.2, color=:black, ls=:dash, label=nothing)
    contourf!(range(-1, 0, 40), range(-0.8, 0.8, 120), func, swapxy=true, aspect_ratio=:equal, linewidth=0, cmap=reverse(cgrad(:roma)))
    contour!(range(-1, 0, 40), range(-0.8, 0.8, 120), func2, swapxy=true, aspect_ratio=:equal, color=:white)
    scatter!([pos.x for (i, pos) in enumerate(excitation.beam_positions) if i != 36], zeros(length(excitation.beam_positions)), color=:white, label=nothing, markersize=2)
    scatter!([excitation.beam_positions[36].x], [0.0], color=:black, label=nothing, markersize=2)
    xlims!(-0.82, 0.82)
    ylims!(-1.02, 0.2)
    xlabel!(L"x")
    ylabel!(L"z")
    plot!(size=(400, 300), dpi=1000)
    savefig("figures/epma_forward_main.png")
    # savefig("figures/")
end

# plots of adjoint forward problem
let
    probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω = Ω -> 1.0, ϵ = ϵ -> EPMAfem.extraction_energy_distribution(extraction, 1, ϵ))

    averaged_sol = probe(discrete_ext[1].vector * discrete_system)
    func = EPMAfem.SpaceModels.interpolable(averaged_sol, space_model)
    extr_func = FEFunction(EPMAfem.SpaceModels.material(space_model), true_ρs[1, :])
    contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), func, swapxy=true, aspect_ratio=:equal, linewidth=0, cmap=:roma)
    contour!(range(-1, 0, 100), range(-0.8, 0.8, 200), -extr_func, swapxy=true, aspect_ratio=:equal, linewidth=1, color=:black, levels=[-0.5])
    xlims!(-0.82, 0.82)
    ylims!(-1.02, 0.2)
    xlabel!(L"x")
    ylabel!(L"z")
    plot!(size=(400, 300), dpi=1000)
    savefig("figures/epma_forward_adjoint.png")
end

# plots of riesz representation forward
let
    probe_beam = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω = Ω -> EPMAfem.beam_direction_distribution(excitation, 4, Ω))

    averaged_sol2 = probe_beam(discrete_ext[1].vector * discrete_system)

    x = range(-0.8, 0.8, 120)
    boundary_evals = zeros(length(x), length(EPMAfem.energy_model(model)))
    for i in 1:length(EPMAfem.energy_model(model))
        # b_func = interpret()
        boundary_evals[:, i] = EPMAfem.SpaceModels.interpolable(averaged_sol2.p[:, i], space_model)(Point.(0.0, x))
    end

    heatmap(x, EPMAfem.energy_model(model), -boundary_evals', cmap=reverse(cgrad(:roma)), aspect_ratio=:equal, clim=(0, 0.125))
    # the 0.12 is only for visualization
    contour!(x, EPMAfem.energy_model(model), (x, ϵ) -> 0.12*EPMAfem.beam_space_distribution(excitation, 36, Point(0.0, x, 0.0))*EPMAfem.beam_energy_distribution(excitation, 1, ϵ), color=:black, levels=range(0, 0.12, 10), colorbar_entry=false)

    scatter!([pos.x for (i, pos) in enumerate(excitation.beam_positions) if i != 36], [excitation.beam_energies[1]], color=:white, label=nothing, markersize=3)
    scatter!([excitation.beam_positions[36].x], [excitation.beam_energies[1]], color=:black, label=nothing, markersize=3)
    
    scatter!([pos.x for pos in excitation.beam_positions], [excitation.beam_energies[2]], color=:white, label=nothing, markersize=3)
    scatter!([pos.x for pos in excitation.beam_positions], [excitation.beam_energies[3]], color=:white, label=nothing, markersize=3)
    # for j in 10:5:36
    #     contour!(x, EPMAfem.energy_model(model), (x, ϵ) -> EPMAfem.beam_space_distribution(excitation, j, Point(0.0, x, 0.0))*EPMAfem.beam_energy_distribution(excitation, 1, ϵ), color=:gray, ls=:dash)
    # end
    # for i in 2:3
    #     contour!(x, EPMAfem.energy_model(model), (x, ϵ) -> EPMAfem.beam_space_distribution(excitation, 36, Point(0.0, x, 0.0))*EPMAfem.beam_energy_distribution(excitation, i, ϵ), color=:gray, ls=:dash)
    # end
    plot!()
    xlabel!(L"x")
    ylabel!(L"\epsilon")
    plot!(size=(400, 300), dpi=1000)
    savefig("figures/epma_riesz_forward.png")
end


let
    plot()
    for i in 1:size(true_meas, 1), j in 1:size(true_meas, 2), k in 1:size(true_meas, 4)
        plot!([pos.x for pos in excitation.beam_positions], true_meas[i, j, :, k], color=:lightgray, ls=:dashdot, label=nothing)
    end

    plot!([pos.x for (i, pos) in enumerate(excitation.beam_positions)], true_meas[1, 1, :, 4], color=1, label="A")
    plot!([pos.x for (i, pos) in enumerate(excitation.beam_positions)], true_meas[1, 3, :, 4], color=1, ls=:dash, label=nothing)
    plot!([pos.x for (i, pos) in enumerate(excitation.beam_positions)], true_meas[1, 1, :, 1], color=1, ls=:dot, label=nothing)
    plot!([pos.x for (i, pos) in enumerate(excitation.beam_positions)], true_meas[2, 1, :, 4], color=2, label="B")
    plot!([pos.x for (i, pos) in enumerate(excitation.beam_positions)], true_meas[2, 3, :, 4], color=2, ls=:dash, label=nothing)
    plot!([pos.x for (i, pos) in enumerate(excitation.beam_positions)], true_meas[2, 1, :, 1], color=2, ls=:dot, label=nothing)
    plot!([0.0], [0.0], color=:gray, ls=:dash, label="different energy")
    plot!([0.0], [0.0], color=:gray, ls=:dot, label="different direction")

    scatter!([excitation.beam_positions[36].x], [true_meas[1, 1, 36, 4]], color=1, label=nothing)
    scatter!([excitation.beam_positions[36].x], [true_meas[2, 1, 36, 4]], color=2, label=nothing)
    plot!(legend=:left)
    plot!(size=(400, 300), dpi=1000)
    savefig("figures/epma_measurements.png")
end


# plots of derivative of adjoint forward
ρs = similar(true_ρs)
ρs .= 0.5
EPMAfem.update_problem_and_vectors!(prob, ρs)
λ = EPMAfem.saveall(discrete_ext[1].vector * discrete_system)
probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω = Ω -> 1.0, ϵ = ϵ -> EPMAfem.beam_energy_distribution(excitation, 1, ϵ))
averaged_sol = probe((EPMAfem.tangent(updatable_pnproblem, λ)[1, 2465]) * discrete_system)
func = EPMAfem.SpaceModels.interpolable(averaged_sol, space_model)
contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), func, swapxy=true, aspect_ratio=:equal, linewidth=0, cmap=reverse(cgrad(:roma)))

# plots of adjoint derivative of adjoint forward
ρs = similar(true_ρs)
ρs .= 0.5
meas2 = prob(ρs)
Σ_bar = 2*(meas2[1, :, :, :] .- true_meas[1, :, :, :])
augmented_primal = EPMAfem.weighted(Σ_bar, discrete_rhs)
probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω = Ω -> 1.0, ϵ = ϵ -> 1.0)
averaged_sol = probe(discrete_system * augmented_primal)
func = EPMAfem.SpaceModels.interpolable(averaged_sol, space_model)
contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), func, swapxy=true, aspect_ratio=:equal, linewidth=0, cmap=reverse(cgrad(:roma)))

# plots of gradient
ρs = similar(true_ρs)
ρs .= 0.5
objective_function(ρs) = sum((true_meas .- prob(ρs)).^2) / length(true_meas)
grad = Zygote.gradient(objective_function, ρs)
grad_func = FEFunction(EPMAfem.SpaceModels.odd(space_model), grad[1][2, :])
heatmap(range(-1, 0, 40), range(-0.8, 0.8, 120), grad_func, swapxy=true)

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

@time averaged_solution = probe(prob.system * prob.excitations[1, 10, 2])



@time @gif for i in 20:50
    averaged_solution = probe(prob.system * prob.excitations[1, i, 2])
    func = EPMAfem.SpaceModels.interpolable(averaged_solution, space_model)
    contourf(range(-1, 0, 40), range(-1.5, 1.5, 120), func, swapxy=true, aspect_ratio=:equal)
    vline!([-0.2, 0.2])
end


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

lux_model = Chain(Dense(2, 30, relu), Dense(30, 30, relu), Dense(30, 2), Lux.softmax)
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


EPMAfem.CUDA.@profile val_and_grad = Zygote.withgradient(objective_function, p0)

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

res = optimize(Optim.only_fg!(fg!), p0, Optim.LBFGS(), Optim.Options(callback=cb, store_trace=true, extended_trace=true, iterations=100, time_limit=3000, g_abstol=1e-10, g_reltol=1e-10))

@gif for (s_i, state) in enumerate(res.trace)
    p = state.metadata["x"]
    ρs = Lux.apply(lux_model, xy, p, st)[1]
    plt1 = heatmap(reshape(ρs[1, :], (40, 120)), aspect_ratio=:equal, clim=(0, 1), cb=nothing)
    plt2 = heatmap(reshape(ρs[2, :], (40, 120)), aspect_ratio=:equal, clim=(0, 1), cb=nothing)
    plt3 = heatmap(reshape(true_ρs[1, :], (40, 120)), aspect_ratio=:equal, clim=(0, 1), cb=nothing)
    plt4 = heatmap(reshape(true_ρs[2, :], (40, 120)), aspect_ratio=:equal, clim=(0, 1), cb=nothing)
    color_i = 1
    plt5 = plot()
    for i in 1:2, j in 1:2, k in 1:7
        plot!(int_store[end-84:end][s_i][i, j, :, k], color=color_i)
        plot!(true_meas[i, j, :, k], color=color_i, legend=false, ls=:dash)
        color_i += 1
    end
    l = @layout [
    grid(2, 2) grid(1, 1)]
    plot(plt1, plt2, plt3, plt4, plt5, layout=l, size=(1500, 400))
    title!("iteration: $s_i")
end fps=5

plot(plot(rand(10)), plot(rand(10)), plot(rand(10)), plot(rand(10)), plot(rand(10)), layout=l)