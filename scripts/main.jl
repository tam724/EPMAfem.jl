using Revise

using EPMAfem
using EPMAfem.CUDA
include("plot_overloads.jl")

using Zygote
#using GLMakie

#import EPMAfem.SphericalHarmonicsModels as SH
#import EPMAfem.SpaceModels as SM
using LinearAlgebra
using Gridap
#using StaticArrays

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -0.5, 0.5), (100, 100)))
# direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(21, 3)
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(27, 2)

equations = EPMAfem.PNEquations()
excitation = EPMAfem.PNExcitation([(x=x_, y=0.0) for x_ in -0.7:0.02:0.7], [0.8, 0.7], [VectorValue(-1.0, 0.5, 0.0) |> normalize, VectorValue(-1.0, 0.0, 0.0), VectorValue(-1.0, -0.5, 0.0) |> normalize])
extraction = EPMAfem.PNExtraction([0.1, 0.2], equations)

model = EPMAfem.DiscretePNModel(space_model, 0.0:0.01:1.0, direction_model)

EPMAfem.compute_influx(excitation, model)
# SM.dimensionality(space_model)
# SH.dimensionality(direction_model)

# model_cuda = EPMAfem.PNGridapModel(space_model, 0.0:0.01:1.0, direction_model, EPMAfem.cuda())
# model_cuda = EPMAfem.PNGridapModel(space_model, 0.0:0.05:1.0, direction_model, EPMAfem.cuda())

# discrete_problem_cpu = EPMAfem.discretize_problem(equations, model_cpu)
# discrete_problem_cuda = EPMAfem.discretize_problem(equations, model_cuda)

# abs.((discrete_problem_cpu.ρm[2] .- (discrete_problem_cuda.ρm[2] |> collect))) |> maximum

#model = model_cuda
#discrete_problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda())
updatable_pnproblem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda(), updatable=true)
# updatable_pnproblem.problem.kp[1][1].diag .= 1.0
# updatable_pnproblem.problem.kp[2][1].diag .= 1.0
# updatable_pnproblem.problem.km[1][1].diag .= 1.0
# updatable_pnproblem.problem.km[2][1].diag .= 1.0

updatable_pnproblem.problem.kp[1][1].diag |> collect |> plot

discrete_system = EPMAfem.schurimplicitmidpointsystem(updatable_pnproblem.problem)

discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())
discrete_ext = EPMAfem.discretize_extraction(extraction, model, EPMAfem.cuda(), updatable=true)

discrete_outflux = EPMAfem.discretize_outflux(model, EPMAfem.cuda())


sol = discrete_system * discrete_rhs[1, 35, 2]

@gif for (ϵ, ψ) in sol
    ψp, ψm = EPMAfem.pmview(ψ, model)
    func = EPMAfem.SpaceModels.interpolable(ψp[:, 1] |> collect, EPMAfem.space_model(model))
    p = heatmap(-1.0:0.01:0, -0.5:0.01:0.5, func.interp, swapxy=true, aspect_ratio=:equal)
    # display(p)
end

cached_sol = EPMAfem.saveall(sol)

anim = @animate for ϵ in 1.0:-0.03:0.0
    @show ϵ
    mean_probe = EPMAfem.PNProbe(model, EPMAfem.cuda(); ϵ = ϵ, Ω = Ω -> 1.0)
    q_mean = EPMAfem.interpolable(mean_probe, cached_sol)

    zx = [Gridap.Point(z_, x_) for x_ in range(-0.3, 0.3, length=20) for z_ in range(-0.5, -0.01, length=20)]
    point_probes = [EPMAfem.PNProbe(model, EPMAfem.cuda(), ϵ = ϵ, x = zx_) for zx_ in zx]
    evals = [EPMAfem.interpolable(pr, cached_sol) for pr in point_probes]

    heatmap(-0.5:0.01:0.0, -0.5:0.01:0.5, q_mean.interp, swapxy=true, aspect_ratio=:equal)
    for i in 1:length(zx)
        # circle_lines!(evals[i], zx[i][1], zx[i][2], 0.03/q_mean.interp(zx[i]); color=:white, label=nothing)
        circle_lines!(evals[i], zx[i][1], zx[i][2], 0.1; color=:white, label=nothing, linewidth=1)
    end
    Plots.scatter!(Dimensions.Ωx.(zx), Dimensions.Ωz.(zx), marker=:dot, color=:white, markersize=2, label=nothing)
    Plots.xlims!(-0.5, 0.5)
    Plots.ylims!(-0.5, 0.0)
end

gif(anim, fps=4)

cached_sol = EPMAfem.saveall(sol)

q_mean

@time q_mean(zx[1])
mean_cache = Gridap.Arrays.return_cache(q_mean, zx[1])
@time Gridap.Arrays.evaluate!(mean_cache, q_mean, zx[1])

function streamline_plot(sol, ϵ_func)
    arch = sol.system.problem.arch
    model = sol.system.problem.model
    cached_sol = EPMAfem.saveall(sol)
    zprobe = EPMAfem.PNProbe(model, arch; ϵ = ϵ_func, Ω = Ω -> Dimensions.Ωz(Ω))
    qz = EPMAfem.interpolable(zprobe, cached_sol)
    xprobe = EPMAfem.PNProbe(model, arch; ϵ = ϵ_func, Ω = Ω -> Dimensions.Ωx(Ω))
    qx = EPMAfem.interpolable(xprobe, cached_sol)

    mean_probe = EPMAfem.PNProbe(model, arch; ϵ = ϵ_func, Ω = Ω -> 1.0)
    q_mean = EPMAfem.interpolable(mean_probe, cached_sol)
    zx = [Gridap.Point(z, x) for z in -0.7:0.01:0.0, x in -0.5:0.01:0.5]
    q_mean_eval = q_mean.(zx)

    function f(x_::Point2)
        z = x_[2]
        x = x_[1]
        return Point2(qx(Gridap.Point(z, x)), qz(Gridap.Point(z, x)))
    end

    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    CairoMakie.contourf!(ax, -0.5:0.01:0.5, -0.7:0.01:0.0, q_mean_eval')
    splt = CairoMakie.streamplot!(ax, f, -0.5..0.5, -0.7..0.0, color=p -> RGBA(1.0, 1.0, 1.0, norm(p)/0.0025), maxsteps=9)
    fig
end

streamline_plot(sol, 0.5)


using CairoMakie



# temp = probe(sol)

xx_probe = EPMAfem.PNProbe(model, EPMAfem.cuda(); ϵ = ϵ -> 1.0, x=Point(-0.1, 0.0))
hΩ = EPMAfem.interpolable(xx_probe, cached_sol)
circle_viz(hΩ)

EPMAfem.SphericalHarmonicsModels.eval_basis_functions!(direction_model, Point(-1.0, 0.0, 0.0))

temp_func = EPMAfem.SpaceModels.interpolable(temp.p, EPMAfem.space_model(model))
plot(-1.5:0.01:1.5, x -> temp_func(Point(0.0, x)))
plot!(-1.5:0.01:1.5, x -> EPMAfem.beam_space_distribution(excitation, 1, Point(0.0, x, 0.0)))

outflux = 2*discrete_outflux * discrete_system * discrete_rhs + influx

#plot([pos.x for pos in excitation.beam_positions], meas[2, :, 2])
plot!([pos.x for pos in excitation.beam_positions], outflux[1, :, 2] ./ -influx[1, :, 2])
#plot!([pos.x for pos in excitation.beam_positions], outflux[2, :, 2] ./ -influx[2, :, 2])

#plot!([pos.x for pos in excitation.beam_positions], )


prob = EPMAfem.EPMAProblem(updatable_pnproblem, discrete_rhs, discrete_ext)
EPMAfem.update_standard_intensities!(prob)

plot(prob.standard_intensities[1, 1, :, 1])
plot!(prob.standard_intensities[2, 1, :, 1])
plot!(prob.standard_intensities[1, 1, :, 2])
plot!(prob.standard_intensities[2, 1, :, 2])
plot!(prob.standard_intensities[1, 1, :, 3])
plot!(prob.standard_intensities[2, 1, :, 3])


function mass_concentrations(e, x)
    # homogeneous material
    return e == 1 ? 0.0 : 3.0
    # return 0.5

    # if x[2] < 0.0 #|| x[2] > 0.1
    #     return e==1 ? 3.0 : 0.0
    # else
    #     return e==1 ? 0.0 : 2.0
    # end
end
ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:2], model)
EPMAfem.update_problem!(updatable_pnproblem, ρs)


meas = prob(ρs)
# CUDA.@profile meas = prob(ρs)

plot(-0.7:0.02:0.7, meas[1, 1, :, 2], legend=nothing)
plot!(-0.7:0.02:0.7, meas[1, 2, :, 2], legend=nothing)
plot!(-0.7:0.02:0.7, meas[1, 1, :, 1], legend=nothing)
plot!(-0.7:0.02:0.7, meas[1, 1, :, 3], legend=nothing)
plot!(-0.7:0.02:0.7, meas[2, 1, :, 2])


f(ρs) = sum((prob(ρs) - meas).^2)

ρs .= 0.5

#@time grad = Zygote.gradient(f, ρs)
grad = Zygote.gradient(f, ρs)
CUDA.@profile @time grad = Zygote.gradient(f, ρs)

@time grad = Zygote.gradient(f, ρs)

GC.gc()

surface(reshape(grad[1][2, :], (40, 120)), aspect_ratio=:equal)




y, pb = Zygote.pullback(f, ρs);
a = pb(1.0)[1]

a[2, 2840]
abs.(a) |> argmax

Δρs = copy(ρs)
Δρs[2, 2840] += 1e-1

(f(Δρs) - f(ρs)) / (1e-1)

discrete_ext[1].vector * discrete_system * discrete_rhs
discrete_ext[1:2] * (discrete_system * discrete_rhs[1:2])
(discrete_ext[1:2] * discrete_system) * discrete_rhs[1:2]
discrete_ext[1] * discrete_system * discrete_rhs[1, 14, 1]
discrete_ext[1] * (discrete_system * (discrete_rhs[1, 14, 1]))
(discrete_ext[1] * discrete_system) * discrete_rhs[1, 14, 1]

c_dot = EPMAfem.tangent(discrete_ext[2])



der = c_dot * (discrete_system * discrete_rhs[1, 15, 2])

c_dot[4001] * (discrete_system * discrete_rhs[1, 15, 2])
der[4001]



der = c_dot * ψ

discrete_ext[1].element_index
reshape(der[:, 1], (40, 80)) |> heatmap

der[:, 1] |> argmax
der[1630, 1]

ρspΔ = copy(ρs)
ρspΔ[1, 1630] += 1e-1
(f(ρspΔ) - f(ρs)) / (1e-1)

function f(ρs)
    EPMAfem.update_vector!(discrete_ext[1], ρs)
    res = discrete_ext[1].vector * (discrete_system * discrete_rhs[1, 15, 2])
    return res[1]
end

f(ρs)


c1_dot = EPMAfem.tangent(discrete_ext[1])
c2_dot = EPMAfem.tangent(discrete_ext[2])

c_dot = hcat(c1_dot, c2_dot)

allunique(c_dot)

[vs.updatable_problem_or_vector.vector for vs in c_dot] |> EPMAfem.ideal_index_order


using BenchmarkTools
@benchmark idx_order = discrete_rhs |> EPMAfem.ideal_index_order

function f(ρs)
    # EPMAfem.update_problem!(updatable_pnproblem, ρs)
    EPMAfem.update_vector!(discrete_ext[1], ρs)
    res = (discrete_ext[1].vector * discrete_system) * sum(discrete_rhs)
    return res[1]
end

f(ρs)

(discrete_ext[1] * discrete_system) * sum(discrete_rhs)

ψ = discrete_ext[1].vector * discrete_system
ψ_cached = EPMAfem.saveall(ψ)
a_dot = EPMAfem.tangent(updatable_pnproblem, ψ_cached);
ϕ = discrete_system * sum(discrete_rhs);
der1 = (a_dot) * ϕ
der2 = (c_dot) * ϕ

ans = (der1 + der2) .- (a_dot + c_dot) * ϕ

EPMAfem.weighted([1.0, 1.0], [a_dot, c_dot])

ρs = copy(updatable_pnproblem.ρs)



f(ρs)

ρspΔ = copy(ρs)
ρspΔ[1, 1634] += 1e-1
(f(ρspΔ) - f(ρs)) / (1e-1)
der[1][1634]

heatmap(-reshape(der[1, :], (40, 80)))

ψ_dot = a_dot[1, 10050] * discrete_system
for (idx, ψ_i) in ψ_dot
end

@gif for (idx, ψ_i) in ψ_dot
    @show idx
    solp = EPMAfem.pview(ψ_i, model)
    heatmap(reshape(@view(solp[:, 1]), (101, 201))|> collect)
end

# cached_solution = EPMAfem.saveall(discrete_system * discrete_rhs[1, 15, 1])
probe = EPMAfem.PNProbe(model, EPMAfem.cuda(); Ω = Ω -> 1.0, ϵ=ϵ->1.0)

res = probe(discrete_system * (discrete_rhs[1, 19, 1] + discrete_rhs[1, 11, 3]))
heatmap(-1:0.01:0, -1:0.01:1, EPMAfem.SpaceModels.interpolable(res, space_model), aspect_ratio=:equal, size=(400, 800))

@gif for i in 101:-1:1
    interp = EPMAfem.SpaceModels.interpolable((p=res.p[:, i], m=res.m[:, i]), space_model)
    heatmap(-1:0.01:0, -1:0.01:1, interp; aspect_ratio=:equal)
end

θ = range(0, 2π, 100)
z = sin.(θ)
x = cos.(θ)

func = EPMAfem.SphericalHarmonicsModels.interpolable(res, direction_model)
plot(z, x, func.(VectorValue.(z, x)))


@gif for i in 1:101
    func = EPMAfem.SphericalHarmonicsModels.interpolable((p=@view(res.p[:, i]), m=@view(res.m[:, i])), direction_model)
    plot(z, x, func.(VectorValue.(z, x)))
    title!("$i")
    zlims!(-0.01, 0.7)
end fps=3


using BenchmarkTools

for ((idx1, ψ1), (idx2, ψ2)) in EPMAfem.taketwo(ψ_cached)
    @show maximum(abs.(ψ1 .- ψ2))
    @show idx1, idx2
end

for (idx, sol) in discrete_system * discrete_rhs[1, 14, 1]
    @show idx
end

@gif for (idx, sol) in cahced
    solp = EPMAfem.pview(sol, model)
    heatmap(reshape(solp[:, 1], (41, 81)))
end

discrete_ext * discrete_system * discrete_rhs



discrete_problem(ψ_cached, ϕ_cached) + discrete_rhs[1, 14, 1]*ϕ_cached

ψ = discrete_system * discrete_rhs[1, 14, 1]

cahced = EPMAfem.saveall(ψ)

(discrete_ext[1] * ψ)
(discrete_ext[2] * ψ)

EPMAfem.weighted([1.0, 1.0], discrete_ext) * ψ_cached

discrete_rhs[1, 14, 1] * (adjoint(discrete_system) * EPMAfem.weighted([1.0, 1.0], discrete_ext))

ϕ = adjoint(discrete_system) * discrete_ext[2]
test = sum([1.0, 1.0, 3.0] .* discrete_ext[[1, 2, 1]])

discrete_rhs * ϕ

ψ_cached = EPMAfem.saveall(ψ)

ϕ = adjoint(discrete_system)*discrete_ext[1]
ϕ_cached = EPMAfem.saveall(ϕ)

res = discrete_rhs[1, 14, 1]*ϕ

discrete_rhs*ϕ_cached

discrete_ext[1]*ψ


discrete_ext[1]*ψ
using BenchmarkTools
@profview @benchmark discrete_ext[1]*ψ_cached

discrete_ext[1]*ψ_2cached




ϕ_cached = EPMAfem.saveall(ϕ)

res = discrete_rhs*ϕ_cached

discrete_rhs[1].bxp == discrete_rhs[2].bxp

eachindex(discrete_rhs)


compute_low_rank_indices(discrete_rhs)


res[1, 14, 1]


for i in ψ
    @show i
end

it = adjoint(discrete_system) * discrete_ext[1];

for i in it
    @show i
end

#test_rhs = EPMAfem.discretize_stange_rhs(excitation, model)
# discrete_ext_old = EPMAfem.discretize_extraction_old(extraction, model, EPMAfem.cuda())

discrete_rhs * discrete_system * discrete_ext

# solver_schur = EPMAfem.pn_schurimplicitmidpointsolver(equations, model, EPMAfem.cuda(), sqrt(eps(Float64)))

#solver_full = EPMAfem.pn_fullimplicitmidpointsolver(equations, model)
# solution = EPMAfem.iterator(discrete_problem, discrete_rhs[1, 14, 1], solver)

g = discrete_rhs
A_gi1 = EPMAfem.iterator(discrete_system, g[1, 20, 1])
A_gi2 = EPMAfem.iterator(discrete_system, g[1, 13, 1])
h = discrete_ext

@profview hh1 = h(A_gi1)

#hh1 = discrete_ext_old(A_gi1)

CUDA.@profile hh1 = h(A_gi1)

hh2 = h(A_gi2)

@gif for (ϵ, i) in A_gi1
    @show i
    sol = EPMAfem.current_solution(A_gi1.system)
    sol_p = EPMAfem.pview(sol, model)
    cpu_vec = collect(@view(sol_p[:, 1]))
    # @show sol_p |> size
    heatmap(reshape(cpu_vec, (101, 201)))
    #display(p)
    #sleep(0.01)
end

cv(x) = EPMAfem.convert_to_architecture(EPMAfem.architecture(model), x)
ρs = [EPMAfem.SpaceModels.L2_projection(x -> EPMAfem.mass_concentrations(equations, e, x), EPMAfem.space_model(model)) for e in 1:EPMAfem.number_of_elements(equations)] |> cv
EPMAfem.update_problem!(discrete_problem, ρs)

Astar_hi1 = EPMAfem.iterator(discrete_system, h[1])
Astar_hi2 = EPMAfem.iterator(discrete_system, h[2])
@profview gg1 = g(Astar_hi1)
gg2 = g(Astar_hi2)

plot(gg1[1, :, 1])
plot!(gg2[1, :, 1])

scatter!([20, 20], [hh1[1], hh1[2]])
scatter!([13, 13], [hh2[1], hh2[2]])

cache = EPMAfem.saveall(solution)

forward_store = Dict()
@gif for (ϵ, i) in solution
    @show i
    sol = EPMAfem.current_solution(solver)
    forward_store[i] = copy(sol)
    sol_p = EPMAfem.pview(sol, model)
    cpu_vec = collect(@view(sol_p[:, 1]))
    # @show sol_p |> size
    heatmap(reshape(cpu_vec, (31, 61)))
end

final_state = copy(EPMAfem.current_solution(solver))

# to_copy = discrete_rhs[1, 14, 1]
# zero_rhs = EPMAfem.Rank1DiscretePNVector{false}(model, zero(to_copy.bϵ), zero(to_copy.bxp), zero(to_copy.bΩp))

rev_solution = EPMAfem.reverse_iterator(discrete_problem, discrete_rhs[1, 14, 1], solver, final_state)

@animate for (ϵ, i) in rev_solution
    @show (ϵ, i)
    sol = EPMAfem.current_solution(solver)
    sol_p = EPMAfem.pview(sol, model)
    cpu_vec = collect(@view(sol_p[:, 1]))
    cpu_vec_fwd = collect(@view(EPMAfem.pview(forward_store[i], model)[:, 1]))
    # @show sol_p |> size
    p1 = heatmap(reshape(cpu_vec, (31, 61)))
    p2 = heatmap(reshape(cpu_vec_fwd, (31, 61)))
    p3 = heatmap(reshape(cpu_vec .- cpu_vec_fwd, (31, 61)))
    # correct the current solution
    # sol .= forward_store[i]
    plot(p1, p2, p3)
    title!("ϵ = $ϵ")
end fps = 5


solution_schur = EPMAfem.iterator(discrete_system, discrete_ext[1])

solution_full = EPMAfem.iterator(discrete_problem, discrete_ext[1], solver_full)
meas_schur = discrete_rhs(solution_schur)
meas_full = discrete_rhs(solution_full)

plot(meas_schur[1, :, 1])
plot!(meas_full[1, :, 1])
plot!(meas_schur[2, :, 1])
plot!(meas_full[2, :, 1])

cv(x) = EPMAfem.convert_to_architecture(EPMAfem.architecture(model), x)

ρs = [EPMAfem.SpaceModels.L2_projection(x -> EPMAfem.mass_concentrations(equations, e, x), EPMAfem.space_model(model)) for e in 1:EPMAfem.number_of_elements(equations)] |> cv

ρs_err = deepcopy(ρs)
EPMAfem.CUDA.@allowscalar begin ρs_err[1][400] += 5e-1 end
EPMAfem.update_problem!(discrete_problem, ρs_err)
meas_err_schur = discrete_rhs(solution_schur)
meas_err_full = discrete_rhs(solution_full)

@gif for (i, ϵ) in solution_schur
    sol = EPMAfem.current_solution(solver_schur)
    sol_p = EPMAfem.pview(sol, model)
    cpu_vec = collect(@view(sol_p[:, 1]))
    # @show sol_p |> size
    heatmap(reshape(cpu_vec, (21, 41)))
end

meas_grad_schur = (meas_err_schur .- meas_schur) ./ 5e-1
meas_grad_full = (meas_err_full .- meas_full) ./ 5e-1

plot(meas_grad_schur[1, :, 1])
plot!(meas_grad_full[1, :, 1])

plot!(meas_grad_schur[2, :, 1])
plot!(meas_grad_full[2, :, 1])

#savesol = EPMAfem.saveall(solution)
ρs = [EPMAfem.SpaceModels.L2_projection(x -> EPMAfem.mass_concentrations(equations, e, x), EPMAfem.space_model(model)) for e in 1:EPMAfem.number_of_elements(equations)] |> cv

EPMAfem.update_problem!(discrete_problem, ρs)

tangent_rhs_schur = EPMAfem.tangent(solution_schur)
tangent_rhs_full = EPMAfem.tangent(solution_full)

new_rhs_schur = tangent_rhs_schur[1, 400, 1e6];
new_rhs_full = tangent_rhs_full[1, 400];

der_sol_schur = EPMAfem.iterator(discrete_system, new_rhs_schur);
der_sol_full = EPMAfem.iterator(discrete_problem, new_rhs_full, solver_full);

meas_tang_schur = discrete_rhs(der_sol_schur)

weights = zeros(size(discrete_rhs))
weights[1, 15, 1] = 1.0
adjoint_rhs_schur = EPMAfem.weighted(weights, discrete_rhs)
adjoint_adjoint_solution_schur = EPMAfem.iterator(discrete_system, adjoint_rhs_schur)
ρs_adjoint = tangent_rhs_schur(adjoint_adjoint_solution_schur)

@profview ρs_adjoint = tangent_rhs_schur(adjoint_adjoint_solution_schur)

Ag = EPMAfem.iterator(discrete_problem, discrete_rhs[1, 15, 1], solver_schur)
EPMAfem.CUDA.@profile discrete_ext(Ag)

heatmap(reshape(ρs_adjoint[1], (100, 200)))

tensor = discrete_problem.ρp_tens2
using CUDA
using BenchmarkTools
y = CUDA.zeros(800)
y2 = CUDA.zeros(800)
u = CUDA.rand(861)
v = CUDA.rand(861)
@benchmark CUDA.@sync EPMAfem.Sparse3Tensor.contract!(y, tensor, u, v, true, false)

@gif for _ in adjoint_adjoint_solution_schur
    sol = EPMAfem.current_solution(solver_schur)
    sol_p = EPMAfem.pview(sol, model)
    cpu_vec = collect(@view(sol_p[:, 1]))
    # @show sol_p |> size
    heatmap(reshape(cpu_vec, (21, 41)))
end

ρs_adjoint[1][400]
meas_tang_schur[1, 15, 1] 

m = FEFunction(EPMAfem.SpaceModels.material(EPMAfem.space_model(model)), ρs_adjoint[1])

trian = EPMAfem.SpaceModels.get_args(EPMAfem.space_model(model))[2]
Gridap.writevtk(trian, "output", cellfields=Dict("m" => m))

heatmap(-1:0.01:0, -1:0.01:1, (x, y) -> m(Point(x, y)))

plot(ρs_adjoint[1])


meas_tang_full = discrete_rhs(der_sol_full)

plot!(meas_tang_schur[1, :, 1])
plot!(meas_tang_schur[2, :, 1])

plot!(meas_tang_full[1, :, 1])
plot!(meas_tang_full[2, :, 1])


# end
meas_schur = discrete_rhs(solution_schur)
meas_full = discrete_rhs(solution_full)

ρs_err = deepcopy(ρs)
ρs_err[1][400] += 1e-4

EPMAfem.update_problem!(discrete_problem, ρs_err)
meas_err_schur = discrete_rhs(solution_schur)
meas_err_full = discrete_rhs(solution_full)

meas_grad_schur = (meas_err_schur .- meas_schur) / 1e-4
meas_grad_full = (meas_err_full .- meas_full) / 1e-4
plot!(meas_grad_schur[1, :, 1])
plot!(meas_grad_schur[2, :, 1])
plot!(meas_grad_full[1, :, 1])
plot!(meas_grad_full[1, :, 1])

@gif for i in der_sol
    sol = EPMAfem.current_solution(solver)
    sol_p = EPMAfem.pview(sol, model)
    cpu_vec = collect(@view(sol_p[:, 1]))
    # @show sol_p |> size
    heatmap(reshape(cpu_vec, (21, 41)))
end

@gif for i in solution2
    sol = EPMAfem.current_solution(solver)
    sol_p = EPMAfem.pview(sol, model)
    cpu_vec = collect(@view(sol_p[:, 1]))
    # @show sol_p |> size
    heatmap(reshape(cpu_vec, (101, 201)))
end

discrete_rhs[1, 1, 1](solution)

measurements = discrete_rhs(solution)

plot(measurements[1, :, 1])
plot!(measurements[2, :, 1])
plot!(measurements[1, :, 2])
plot!(measurements[2, :, 2])

discrete_ext(solution2)
weights = rand(size(discrete_rhs)...)
new_rhs = EPMAfem.weight_array_of_r1(weights, discrete_rhs)
solution3 = EPMAfem.iterator(discrete_problem, new_rhs, solver)

@gif for i in solution3
    sol = EPMAfem.current_solution(solver)
    sol_p = EPMAfem.pview(sol, model)
    cpu_vec = collect(@view(sol_p[:, 1]))
    # @show sol_p |> size
    heatmap(reshape(cpu_vec, (51, 101)))
end


# new_dict = filter(p -> p[2][2] == 0.0, SH.boundary_matrix_dict)
# new_dict = Dict(((key, val[1]) for (key, val) in new_dict))

using Serialization
serialize("boundary_matrix_dict2.jls", new_dict)

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscrereModel((0, 1), 10))

sing 

model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(11, 1)

n = 100
θ = [0;(0.5:n-0.5)/n;1]
ϕ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(ϕ)*sinpi(θ) for θ in θ, ϕ in ϕ]
y = [sinpi(ϕ)*sinpi(θ) for θ in θ, ϕ in ϕ]
z = [cospi(θ) for θ in θ, ϕ in ϕ]

for i in 1:SH.num_dofs(model)
    vec = zeros(SH.num_dofs(model))
    vec[i] = 1.0

    color = [dot(vec, SH._eval_basis_functions!(Y, model, SH.VectorValue(x_, y_, z_))) for (x_, y_, z_) in zip(x, y, z)]

    s = surface(x, y, z, color=color)
    display(s)
    sleep(1)
end

using BenchmarkTools
A1 = SH.assemble_bilinear(SH.∫S²_Ωzuv, model, SH.even(model), SH.odd(model), SH.exact_quadrature())
A2 = SH.assemble_bilinear(SH.∫S²_Ωzuv, model, SH.even(model), SH.odd(model), SH.hcubature_quadrature(1e-5, 1e-5))
A3 = SH.assemble_bilinear(SH.∫S²_Ωzuv, model, SH.even(model), SH.odd(model))

maximum(abs.(A1 .- A2))
maximum(abs.(A1 .- A3))


A1 = SH.assemble_bilinear(SH.∫S²_absΩyuv, model, SH.even(model), SH.even(model), SH.exact_quadrature())
A1 = SH.assemble_bilinear(SH.∫S²_absΩyuv, model, SH.even(model), SH.even(model), SH.hcubature_quadrature(1e-5, 1e-5, 1000))
A1 = SH.assemble_bilinear(SH.∫S²_absΩyuv, model, SH.even(model), SH.even(model), SH.lebedev_quadrature(SH.guess_lebedev_order_from_model(model, 1000)))

#A1x = SH.assemble_bilinear(SH.∫S²_absΩxuv, model, SH.even(model), SH.even(model), SH.lebedev_quadrature)
A1y = SH.assemble_bilinear(SH.∫S²_absΩyuv, model, SH.even(model), SH.even(model), SH.exact_quadrature)

abs.(A1 .- A1y) |> maximum

Plots.spy(A1x)
Plots.spy(A1)

isapprox.(A1 .- A1y, 0.0, atol=1e-13) |> all

A1 .- A1y

nothing

A1 
A1y
A1 = SH.assemble_bilinear(SH.∫uv, model, SH.even(model), SH.even(model), SH.hcubature_quadrature)
A1x = SH.assemble_bilinear(SH.∫uv, model, SH.even(model), SH.even(model), SH.lebedev_quadrature)
A1y = SH.assemble_bilinear(SH.∫uv, model, SH.even(model), SH.even(model), SH.exact_quadrature)


A1 = SH.assemble_bilinear(SH.∫uv, model, SH.odd(model), SH.odd(model), SH.exact_quadrature)

Ax = SH.assemble_bilinear(SH.∫Ωxuv, model, SH.odd(model), SH.even(model))
Ay = SH.assemble_bilinear(SH.∫Ωyuv, model, SH.odd(model), SH.even(model))
Az = SH.assemble_bilinear(SH.∫Ωzuv, model, SH.odd(model), SH.even(model))

A2x = SH.assemble_bilinear_analytic(SH.∫Ωxuv, model)
A2y = SH.assemble_bilinear_analytic(SH.∫Ωyuv, model)
A2z = SH.assemble_bilinear_analytic(SH.∫Ωzuv, model)

Plots.spy(round.(Ay, digits=14))

Makie.spy(Az)

maximum(abs.(Az .- A2z))
maximum(abs.(Ay .- A2y))
maximum(abs.(Ax .- A2x))
# A1 = SH.assemble_bilinear(SH.∫Ωxuv, model)
# A1 = SH.assemble_bilinear(SH.∫Ωyuv, model)

Plots.spy(A1)

A2 = SH.assemble_bilinear(SH.∫uv, model, SH.hcubature_quadrature)

isapprox.(A1 .- A2, 0.0, atol=1e-10) |> all