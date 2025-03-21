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
figpath = mkpath(joinpath(dirname(@__FILE__), "figures"))

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1.5, 1.5), (40, 120)))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(21, 2)
model = EPMAfem.DiscretePNModel(space_model, 0.0:0.05:1.0, direction_model)

equations = EPMAfem.PNEquations()
# excitation = EPMAfem.PNExcitation([(x=x_, y=0.0) for x_ in -0.7:0.02:0.7], [0.8, 0.7, 0.6], normalize.([VectorValue(-1.0, 0.75, 0.0), VectorValue(-1.0, 0.5, 0.0), VectorValue(-1.0, 0.25, 0.0), VectorValue(-1.0, 0.0, 0.0), VectorValue(-1.0, -0.25, 0.0), VectorValue(-1.0, -0.5, 0.0), VectorValue(-1.0, -0.75, 0.0)]))
excitation = EPMAfem.pn_excitation([(x=x_, y=0.0) for x_ in -0.7:0.02:0.7], [0.8, 0.6], normalize.([VectorValue(-1.0, 0.5, 0.0), VectorValue(-1.0, 0.0, 0.0), VectorValue(-1.0, -0.5, 0.0)]))
extraction = EPMAfem.PNExtraction([0.1, 0.2], equations)

updatable_pnproblem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda(), updatable=true)
discrete_system = EPMAfem.implicit_midpoint(updatable_pnproblem.problem, EPMAfem.PNSchurSolver)

discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())
discrete_ext = EPMAfem.discretize_extraction(extraction, model, EPMAfem.cuda(), updatable=true)

prob = EPMAfem.EPMAProblem(updatable_pnproblem, discrete_rhs, discrete_ext)
EPMAfem.update_standard_intensities!(prob)

# compute the "true measurements"
function mass_concentrations(e, x)
    z_, x_ = x
    if z_*-0.4 + x_ + 0.1 > 0.0 &&  z_*-0.1 + x_ - 0.2 < 0.0
    # if x[2] > -0.1 && x[2] < 0.25
        # if sqrt((x[1] + 0.05)^2 + (x[2] - 0.3)^2) < 0.2
        return e == 1 ? 1.0 : 0.0
    else
        return e == 1 ? 0.0 : 1.0
    end
end

true_ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:2], model)
true_meas = prob(true_ρs)

# plt1 = heatmap(reshape(true_ρs[1, :], (40, 120)), aspect_ratio=:equal)
# plt2 = heatmap(reshape(true_ρs[2, :], (40, 120)), aspect_ratio=:equal)
# display(plot(plt1, plt2))

# plots of forward problem
let
    probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω=Ω -> 1.0, ϵ=ϵ -> EPMAfem.extraction_energy_distribution(extraction, 1, ϵ))

    averaged_sol = probe(discrete_system * discrete_rhs[1, 36, 2])
    averaged_sol2 = probe(discrete_system * discrete_rhs[1, 36, 1])
    func = EPMAfem.SpaceModels.uncached_interpolable(averaged_sol, space_model)
    func2 = EPMAfem.SpaceModels.uncached_interpolable(averaged_sol2, space_model)
    Plots.plot(fontfamily="Computer Modern")
    for i in 36:1:40
        Plots.plot!(-0.8:0.01:0.8, x -> EPMAfem.beam_space_distribution(excitation, i, VectorValue(0.0, x, 0.0)) * 0.2, color=:gray, ls=:dash, label=nothing)
    end
    Plots.plot!(-0.8:0.01:0.8, x -> EPMAfem.beam_space_distribution(excitation, 36, VectorValue(0.0, x, 0.0)) * 0.2, color=:black, ls=:dash, label=nothing)
    Plots.contourf!(range(-1, 0, 40), range(-0.8, 0.8, 120), func, swapxy=true, aspect_ratio=:equal, linewidth=0, cmap=reverse(cgrad(:roma)))
    Plots.contour!(range(-1, 0, 40), range(-0.8, 0.8, 120), func2, swapxy=true, aspect_ratio=:equal, color=:white)
    Plots.scatter!([pos.x for (i, pos) in enumerate(excitation.beam_positions) if i != 36], zeros(length(excitation.beam_positions)), color=:white, label=nothing, markersize=2)
    Plots.scatter!([excitation.beam_positions[36].x], [0.0], color=:black, label=nothing, markersize=2)
    Plots.xlims!(-0.82, 0.82)
    Plots.ylims!(-1.02, 0.2)
    Plots.xlabel!(L"x")
    Plots.ylabel!(L"z")
    Plots.plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm)
    Plots.savefig(joinpath(figpath, "epma_forward_main.png"))
    # savefig("figures/")
end

# plots of adjoint forward problem
let
    probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω=Ω->1.0, ϵ=ϵ->1.0)

    averaged_sol = probe(discrete_ext[1].vector * discrete_system)
    func = EPMAfem.SpaceModels.uncached_interpolable(averaged_sol, space_model)
    extr_func = FEFunction(EPMAfem.SpaceModels.material(space_model), true_ρs[1, :])
    Plots.contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), func, swapxy=true, aspect_ratio=:equal, linewidth=0, cmap=:roma)
    Plots.contour!(range(-1, 0, 100), range(-0.8, 0.8, 200), -extr_func, swapxy=true, aspect_ratio=:equal, linewidth=1, color=:black, levels=[-0.5])
    Plots.xlims!(-0.82, 0.82)
    Plots.ylims!(-1.02, 0.2)
    Plots.xlabel!(L"x")
    Plots.ylabel!(L"z")
    Plots.plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    Plots.savefig(joinpath(figpath, "epma_forward_adjoint.png"))
end

# plots of riesz representation forward
let
    probe_beam = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω=Ω -> EPMAfem.beam_direction_distribution(excitation, 2, Ω))

    averaged_sol2 = probe_beam(discrete_ext[1].vector * discrete_system)

    x = range(-0.8, 0.8, 120)
    boundary_evals = zeros(length(x), length(EPMAfem.energy_model(model)))
    for i in 1:length(EPMAfem.energy_model(model))
        # b_func = interpret()
        boundary_evals[:, i] = EPMAfem.SpaceModels.interpolable(averaged_sol2.p[:, i], space_model)(VectorValue.(0.0, x))
    end

    Plots.contourf(x, EPMAfem.energy_model(model), -boundary_evals', cmap=reverse(cgrad(:roma)), aspect_ratio=:equal, linewidth=0)
    # the 0.12 is only for visualization
    Plots.contour!(x, EPMAfem.energy_model(model), (x, ϵ) -> 0.12 * EPMAfem.beam_space_distribution(excitation, 36, VectorValue(0.0, x, 0.0)) * EPMAfem.beam_energy_distribution(excitation, 1, ϵ), color=:black, levels=range(0, 0.12, 10), colorbar_entry=false)

    Plots.scatter!([pos.x for (i, pos) in enumerate(excitation.beam_positions) if i != 36], [excitation.beam_energies[1]], color=:white, label=nothing, markersize=2)

    Plots.scatter!([pos.x for pos in excitation.beam_positions[1:35]], [excitation.beam_energies[1]], color=:white, label=nothing, markersize=2)
    Plots.scatter!([excitation.beam_positions[36].x], [excitation.beam_energies[1]], color=:black, label=nothing, markersize=2)
    Plots.scatter!([pos.x for pos in excitation.beam_positions[37:end]], [excitation.beam_energies[1]], color=:white, label=nothing, markersize=2)

    Plots.scatter!([pos.x for pos in excitation.beam_positions], [excitation.beam_energies[2]], color=:white, label=nothing, markersize=2)


    # for j in 10:5:36
    #     contour!(x, EPMAfem.energy_model(model), (x, ϵ) -> EPMAfem.beam_space_distribution(excitation, j, VectorValue(0.0, x, 0.0))*EPMAfem.beam_energy_distribution(excitation, 1, ϵ), color=:gray, ls=:dash)
    # end
    # for i in 2:3
    #     contour!(x, EPMAfem.energy_model(model), (x, ϵ) -> EPMAfem.beam_space_distribution(excitation, 36, VectorValue(0.0, x, 0.0))*EPMAfem.beam_energy_distribution(excitation, i, ϵ), color=:gray, ls=:dash)
    # end
    Plots.xlims!(-0.82, 0.82)

    Plots.plot!()
    Plots.xlabel!(L"x")
    Plots.ylabel!(L"\epsilon")
    Plots.plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    Plots.savefig(joinpath(figpath, "epma_riesz_forward.png"))

end

let
    beam_pos = [pos.x for (i, pos) in enumerate(excitation.beam_positions)]
    Plots.plot(beam_pos, true_meas[1, 1, :, 2], color=cgrad(:tab20)[1], label="A, high energy")
    Plots.plot!(beam_pos, true_meas[1, 2, :, 2], color=cgrad(:tab20)[2], label="A, low energy")
    Plots.plot!(beam_pos, true_meas[2, 1, :, 2], color=cgrad(:tab20)[3], label="B, high energy")
    Plots.plot!(beam_pos, true_meas[2, 2, :, 2], color=cgrad(:tab20)[4], label="B, low energy")

    Plots.plot!(beam_pos, true_meas[1, 1, :, 3], color=cgrad(:tab20)[1], ls=:dash, label=nothing)
    Plots.plot!(beam_pos, true_meas[1, 2, :, 3], color=cgrad(:tab20)[2], ls=:dash, label=nothing)
    Plots.plot!(beam_pos, true_meas[2, 1, :, 3], color=cgrad(:tab20)[3], ls=:dash, label=nothing)
    Plots.plot!(beam_pos, true_meas[2, 2, :, 3], color=cgrad(:tab20)[4], ls=:dash, label=nothing)
    Plots.plot!([], [], ls=:dash, color=:gray, label="tilted beam")

    i_beam = 36
    Plots.scatter!(fill(beam_pos[i_beam], 8), [true_meas[i, j, i_beam, k] for i in 1:2 for j in 1:2 for k in 2:3], color=:gray, label=nothing, alpha=0.5)
    Plots.plot!(legend=:left)
    Plots.plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    Plots.xlabel!("beam position")
    Plots.ylabel!("k-ratio")
    Plots.savefig(joinpath(figpath, "epma_measurements.png"))

end

## taylor remainder
let
    println("Computing taylor remainder...")
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
    true_ρs = EPMAfem.discretize_mass_concentrations([x -> 0.4, x -> 0.6], model)
    true_meas = prob(true_ρs)
    objective_function(ρs) = sum((true_meas .- prob(ρs)).^2) / length(true_meas)
    ρs = similar(true_ρs)
    ρs .= 0.5
    grad = Zygote.gradient(objective_function, ρs)
    selected_indx = reshape(1:4800, (40, 120))[35:end, 55:65][:]
    grad_selected = grad[1][:, selected_indx]

    function objective_function_selected(ρs_selected)
        full_ρs = similar(true_ρs)
        full_ρs .= 0.5
        full_ρs[:, selected_indx] .= ρs_selected[:, :]
        return objective_function(full_ρs)
    end

    ρs_selected = zeros(size(ρs, 1), length(selected_indx))
    ρs_selected .= 0.5

    grad_fd_01 = finite_difference_grad(objective_function_selected, ρs_selected, 1e-1)
    grad_fd_015 = finite_difference_grad(objective_function_selected, ρs_selected, 5e-2)
    grad_fd_02 = finite_difference_grad(objective_function_selected, ρs_selected, 1e-2)

    Nδs = 7
    hs = [1/2^(i/2) for i in 0:8]
    taylor_1st = zeros(size(hs)..., Nδs)
    taylor_2nd_ad = zeros(size(hs)..., Nδs)
    taylor_2nd_fd_01 = zeros(size(hs)..., Nδs)
    taylor_2nd_fd_015 = zeros(size(hs)..., Nδs)
    taylor_2nd_fd_02 = zeros(size(hs)..., Nδs)

    C = objective_function_selected(ρs_selected)

    for n in 1:Nδs
        δρs = randn(size(ρs_selected)...) |> normalize
        for (i, h) in enumerate(hs)
            @show n, i
            Cδρs = objective_function_selected(ρs_selected + h*δρs)
            taylor_1st[i, n] = abs(Cδρs - C)
            taylor_2nd_ad[i, n] = abs(Cδρs - C - h*dot(grad_selected, δρs))
            taylor_2nd_fd_01[i, n] = abs(Cδρs - C - h*dot(grad_fd_01, δρs))
            taylor_2nd_fd_015[i, n] = abs(Cδρs - C - h*dot(grad_fd_015, δρs))
            taylor_2nd_fd_02[i, n] = abs(Cδρs - C - h*dot(grad_fd_02, δρs))
            # taylor_2nd_fd_04[i, n] = abs(Cδρs - C - h*dot(grad_fd_04, δρs))
            # taylor_2nd_fd_05[i, n] = abs(Cδρs - C - h*dot(grad_fd_05, δρs))
        end
    end

    gr()
    plot(hs, sum(taylor_1st; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="1st rem.", marker=:x, color=1)
    plot!(hs, sum(taylor_2nd_fd_01; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (FD, 1e-1)", marker=:x, color=3)
    plot!(hs, sum(taylor_2nd_fd_015; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (FD, 5e-2)", marker=:x, color=4)
    plot!(hs, sum(taylor_2nd_fd_02; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (FD, 1e-2)", marker=:x, color=5)
    # plot!(hs, sum(taylor_2nd_fd_04; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (FD, 1e-4)", marker=:x, color=4)
    # plot!(hs, sum(taylor_2nd_fd_05; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (FD, 1e-5)", marker=:x, color=5)
    plot!(hs, sum(taylor_2nd_ad; dims=2)/Nδs, xaxis=:log, yaxis=:log, label="2nd rem. (Adjoint)", marker=:x, color=2)

    plot!(hs, 1e-4*hs, xaxis=:log, yaxis=:log, color=:gray, ls=:dash, label="1st order")
    plot!(hs, 1e-6*hs.^2, xaxis=:log, yaxis=:log, color=:gray, ls=:dashdot, label="2nd order")
    ylims!(1e-10, 2e-4)
    xlabel!(L"h")
    ylabel!("Taylor remainder")
    plot!(size=(400, 300), dpi=1000, legend=:bottomright, fontfamily="Computer Modern")
    fig = savefig(joinpath(figpath, "epma_taylor_remainder.png"))
    println("Saved $(fig)")

end

# plots of derivative of adjoint forward
let
    ρs = similar(true_ρs)
    ρs .= 0.5
    EPMAfem.update_problem_and_vectors!(prob, ρs)
    λ = EPMAfem.saveall(discrete_ext[1].vector * discrete_system)
    probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω=Ω -> 1.0, ϵ=ϵ -> EPMAfem.beam_energy_distribution(excitation, 1, ϵ))
    averaged_sol = probe((EPMAfem.tangent(updatable_pnproblem, λ)[1, 2465]) * discrete_system)
    # averaged_sol = probe(((EPMAfem.tangent(updatable_pnproblem, λ)[1, 2465]) + EPMAfem.tangent(discrete_ext[1])[1, 2465]) * discrete_system)
    func = EPMAfem.SpaceModels.uncached_interpolable(averaged_sol, space_model)
    Plots.contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), func, swapxy=true, aspect_ratio=:equal, linewidth=0, cmap=reverse(cgrad(:roma)))
    Plots.xlabel!(L"x")
    Plots.ylabel!(L"z")
    Plots.xlims!(-0.82, 0.82)

    Plots.plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    fig = Plots.savefig(joinpath(figpath, "epma_tangent_nonadjoint.png"))
    println("Saved $(fig)")

end

# plots of adjoint derivative of adjoint forward
let
    ρs = similar(true_ρs)
    ρs .= 0.5
    meas2 = prob(ρs)
    Σ_bar = 2 * (meas2[1, :, :, :] .- true_meas[1, :, :, :])
    augmented_primal = EPMAfem.weighted(Σ_bar, discrete_rhs)
    probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω=Ω -> 1.0, ϵ=ϵ -> 1.0)
    averaged_sol = probe(discrete_system * augmented_primal)
    func = EPMAfem.SpaceModels.uncached_interpolable(averaged_sol, space_model)
    Plots.contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), func, swapxy=true, aspect_ratio=:equal, linewidth=0, cmap=reverse(cgrad(:roma)))
    Plots.xlabel!(L"x")
    Plots.ylabel!(L"z")
    Plots.xlims!(-0.82, 0.82)

    Plots.plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    Plots.savefig(joinpath(figpath, "epma_tangent_adjoint.png"))
end

# plots of gradient
let
    ρs = similar(true_ρs)
    ρs .= 0.5
    objective_function(ρs) = sum((true_meas .- prob(ρs)) .^ 2) / length(true_meas)
    grad = Zygote.gradient(objective_function, ρs)
    grad_func = FEFunction(EPMAfem.SpaceModels.odd(space_model), grad[1][1, :] * 1e3)
    Plots.contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), grad_func, swapxy=true, cmap=:roma, linewidth=0, aspect_ratio=:equal)
    Plots.xlabel!(L"x")
    Plots.ylabel!(L"z")
    Plots.xlims!(-0.82, 0.82)
    Plots.plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    Plots.savefig(joinpath(figpath, "epma_gradient1.png"))


    grad_func = FEFunction(EPMAfem.SpaceModels.odd(space_model), grad[1][2, :] * 1e3)
    Plots.contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), grad_func, swapxy=true, cmap=:roma, linewidth=0, aspect_ratio=:equal)
    Plots.xlabel!(L"x")
    Plots.ylabel!(L"z")
    Plots.xlims!(-0.82, 0.82)
    Plots.plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    Plots.savefig(joinpath(figpath, "epma_gradient2.png"))

end

# @time averaged_solution = probe(prob.system * prob.excitations[1, 10, 2])

# @time @gif for i in 20:50
#     averaged_solution = probe(prob.system * prob.excitations[1, i, 2])
#     func = EPMAfem.SpaceModels.interpolable(averaged_solution, space_model)
#     contourf(range(-1, 0, 40), range(-1.5, 1.5, 120), func, swapxy=true, aspect_ratio=:equal)
#     vline!([-0.15, 0.25])
# end

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

# plt1 = heatmap(reshape(ρs[1, :], (40, 120)), aspect_ratio=:equal)
# plt2 = heatmap(reshape(ρs[2, :], (40, 120)), aspect_ratio=:equal)
# display(plot(plt1, plt2))

# plt1 = heatmap(reshape(true_ρs[1, :], (40, 120)), aspect_ratio=:equal)
# plt2 = heatmap(reshape(true_ρs[2, :], (40, 120)), aspect_ratio=:equal)
# display(plot(plt1, plt2))

cached_intensities = zeros(size(true_meas))
# function objective_function2(p)
#     ρs = Lux.apply(lux_model, x, p, st)[1]
#     return sum((ρs .- true_ρs).^2)*1e-8
# end

noisy_meas = true_meas .+ randn(size(true_meas)) * 0.01

function objective_function(p)
    intensities = prob(Lux.apply(lux_model, xy, p, st)[1])
    Zygote.ignore() do
        cached_intensities .= intensities
    end
    return sum((noisy_meas .- intensities) .^ 2) / length(true_meas)
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

res = optimize(Optim.only_fg!(fg!), p0, Optim.LBFGS(), Optim.Options(callback=cb, store_trace=true, extended_trace=true, iterations=250, time_limit=4000, g_abstol=1e-7, g_reltol=1e-7))
using Serialization
Serialization.serialize(joinpath(figpath, "opti_trace.jls"), (res, int_store))



p = res.trace[1].metadata["x"]
ρs = Lux.apply(lux_model, xy, p, st)[1]

# plot final iteration
ρs = Lux.apply(lux_model, xy, res.minimizer, st)[1]
for (ρs_, figname) in [(ρs, "epma_opti_material_noisy"), (true_ρs, "epma_opti_material_true")]
    mat_func = FEFunction(EPMAfem.SpaceModels.odd(space_model), ρs_[1, :])
    mat_func_true = FEFunction(EPMAfem.SpaceModels.odd(space_model), true_ρs[1, :])

    plt1 = contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), mat_func, linewidth=0, swapxy=true, aspect_ratio=:equal, clim=(0, 1), cmap=cgrad([get_color_palette(:auto, 1)[2], :black, get_color_palette(:auto, 1)[1]]))
    plt1 = contour!(range(-1, 0, 40), range(-0.8, 0.8, 120), mat_func_true, linewidth=1, swapxy=true, aspect_ratio=:equal, clim=(0, 1), color=:lightgray, levels=1)
    scatter!([pos.x for pos in excitation.beam_positions], [0.0], color=:white, label=nothing, markersize=2)
    xlabel!(L"x")
    ylabel!(L"z")
    xlims!(-0.82, 0.82)

    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    savefig(joinpath(figpath, "$figname.png"))

end


# plot final measurements
plot()
color_i = 1
for i in 1:2, j in 1:2, k in 1:3
    plot!([pos.x for pos in excitation.beam_positions], noisy_meas[i, j, :, k], color=color_i, legend=false, ls=:solid)
    scatter!([pos.x for pos in excitation.beam_positions], int_store[end][i, j, :, k], color=color_i, markersize=1)
    color_i += 1
end
xlabel!(L"beam center $x$")
ylabel!(L"observations $\mathcal{Y}^{(ji)}$")
plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
savefig(joinpath(figpath, "epma_opti_measurements.png"))


# plot animated results
probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω=Ω -> 1.0, ϵ=ϵ -> EPMAfem.extraction_energy_distribution(extraction, 1, ϵ))
averaged_sol = probe(discrete_system * discrete_rhs[1, 36, 2])
func = EPMAfem.SpaceModels.interpolable(averaged_sol, space_model)

anim = @animate for (s_i, state) in enumerate(res.trace)
    p = state.metadata["x"]
    ρs = Lux.apply(lux_model, xy, p, st)[1]
    grad_func = FEFunction(EPMAfem.SpaceModels.odd(space_model), ρs[1, :] - ρs[2, :])

    plt1 = contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), grad_func, linewidth=0, swapxy=true, aspect_ratio=:equal, clim=(-1, 1), cmap=cgrad([get_color_palette(:auto, 1)[2], :black, get_color_palette(:auto, 1)[1]]), cb=nothing)
    scatter!([pos.x for pos in excitation.beam_positions], [0.0], color=:white, label=nothing, markersize=2)

    grad_func_true = FEFunction(EPMAfem.SpaceModels.odd(space_model), true_ρs[1, :] - true_ρs[2, :])
    plt3 = contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), grad_func_true, linewidth=0, swapxy=true, aspect_ratio=:equal, clim=(-1.5, 1.5), cmap=cgrad([get_color_palette(:auto, 1)[2], :black, get_color_palette(:auto, 1)[1]]), cb=nothing, levels=[0.0])

    plt3 = contour!(range(-1, 0, 40), range(-0.8, 0.8, 120), func, swapxy=true, aspect_ratio=:equal, color=:black, linewidth=1, clim=(0, 0.1))

    scatter!([pos.x for (i, pos) in enumerate(excitation.beam_positions) if i != 36], zeros(length(excitation.beam_positions)), color=:white, label=nothing, markersize=2)
    scatter!([excitation.beam_positions[36].x], [0.0], color=:black, label=nothing, markersize=2)

    plot(plt1, plt3, size=(800, 300))

end fps = 5

fig = gif(anim, joinpath(figpath, "epma_opti_material_noisy.mp4"))
println("Saved $(fig.filename)")
gif(anim, joinpath(figpath, "epma_opti_material_noisy.gif"))
println("Saved $(fig.filename)")

using Printf

anim2 = @animate for (s_i, state) in enumerate(res.trace)
    color_i = 1
    plt5 = plot()
    for i in 1:2, j in 1:2, k in 1:3
        plot!([pos.x for pos in excitation.beam_positions], int_store[s_i][i, j, :, k], color=i, markersize=2)
        plot!([pos.x for pos in excitation.beam_positions], noisy_meas[i, j, :, k], color=i, legend=false, ls=:dash)
        color_i += 1
    end
    annotate!(-0.5, 0.5, text("MSE\n = $(@sprintf "%.2e" state.value)", :black, :center, 10))
    plot!(size=(400, 300))
end fps = 5

fig = gif(anim2, joinpath(figpath, "epma_opti_measurements_noisy.mp4"))
println("Saved $(fig.filename)")

fig = gif(anim2, joinpath(figpath, "epma_opti_measurements_noisy.gif"))
println("Saved $(fig.filename)")

anim3 = @animate for (s_i, state) in enumerate(res.trace)
    p = state.metadata["x"]
    ρs = Lux.apply(lux_model, xy, p, st)[1]
    grad_func = FEFunction(EPMAfem.SpaceModels.odd(space_model), ρs[1, :] - ρs[2, :])

    plt1 = contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), grad_func, linewidth=0, swapxy=true, aspect_ratio=:equal, clim=(-1, 1), cmap=cgrad([get_color_palette(:auto, 1)[2], :black, get_color_palette(:auto, 1)[1]]), cb=nothing)
    scatter!([pos.x for pos in excitation.beam_positions], [0.0], color=:white, label=nothing, markersize=2)

    grad_func_true = FEFunction(EPMAfem.SpaceModels.odd(space_model), true_ρs[1, :] - true_ρs[2, :])
    plt3 = contourf(range(-1, 0, 40), range(-0.8, 0.8, 120), grad_func_true, linewidth=0, swapxy=true, aspect_ratio=:equal, clim=(-1.5, 1.5), cmap=cgrad([get_color_palette(:auto, 1)[2], :black, get_color_palette(:auto, 1)[1]]), cb=nothing, levels=[0.0])

    plt3 = contour!(range(-1, 0, 40), range(-0.8, 0.8, 120), func, swapxy=true, aspect_ratio=:equal, color=:black, linewidth=1, clim=(0, 0.1))

    scatter!([pos.x for (i, pos) in enumerate(excitation.beam_positions) if i != 36], zeros(length(excitation.beam_positions)), color=:white, label=nothing, markersize=2)
    scatter!([excitation.beam_positions[36].x], [0.0], color=:black, label=nothing, markersize=2)

    plt5 = plot()
    color_i = 1
    for i in 1:2, j in 1:2, k in 1:3
        plot!([pos.x for pos in excitation.beam_positions], int_store[s_i][i, j, :, k], color=i, markersize=2)
        plot!([pos.x for pos in excitation.beam_positions], noisy_meas[i, j, :, k], color=i, legend=false, ls=:dash)
        color_i += 1
    end
    annotate!(-0.5, 0.5, text("MSE\n = $(@sprintf "%.2e" state.value)", :black, :center, 10))
    plot!(size=(400, 300))

    plt_mse = plot([t.value for t in res.trace], label=nothing, yaxis=:log)
    vline!(s_i, label=nothing)
    scatter!([s_i], [state.value], color=1, label=nothing)

    plot(plt1, plt3, plt5, plt_mse, size=(800, 300))

end fps = 5

fig = gif(anim3, joinpath(figpath, "epma_opti_all_noisy.mp4"))
println("Saved $(fig.filename)")

fig = gif(anim3, joinpath(figpath, "epma_opti_all_noisy.gif"))
println("Saved $(fig.filename)")
