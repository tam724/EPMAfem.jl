using Gridap
using Serialization
using SparseArrays
using HCubature
using LinearAlgebra
using Enzyme
using Distributions
using Plots
using IterativeSolvers
using Zygote
using Lux
using Optim, Lux, Random, Optimisers


include("spherical_harmonics.jl")
using .SphericalHarmonicsMatrices
include("epma-fem.jl")


#some fixed physics:
scattering_kernel_(μ) = exp(-4.0*(μ-(1))^2)
scattering_norm_factor = 2*π*hquadrature(x -> scattering_kernel_(x), -1.0, 1.0, rtol=1e-8, atol=1e-8, maxevals=100000)[1]
scattering_kernel(μ) = scattering_kernel_(μ) / scattering_norm_factor

ss(ϵ) = [1.0 + 0.1*exp(-ϵ), 1.0 + 0.2*exp(-ϵ)]
∂ss(ϵ) = Enzyme.autodiff(Forward, ss, Duplicated(ϵ, 1.0))[1]
σ(ϵ) = [0.5 + 0.1*exp(-3*ϵ), 0.5 + 0.1*exp(-4*ϵ)]
ττ(ϵ) = σ(ϵ)

s(ϵ) = 0.5 .* ss(ϵ)
τ(ϵ) = ττ(ϵ) .- 0.5 .* ∂ss(ϵ)

physics = (s=s, τ=τ, σ=σ)

nhx = 120
model = CartesianDiscreteModel((0.0, 1.0, -1.0, 1.0), (60, nhx))

M = material_space(model)

p_ellipse = (μ1=0.9, μ2=0.0, r=-π/2.5, a=0.35, b=0.15)

is_in_ellipse(x, p) = ((x[1] - p.μ1)*cos(p.r) + (x[2] - p.μ2)*sin(p.r))^2/p.a^2 + ((x[1] - p.μ1)*sin(p.r) - (x[2] - p.μ2)*cos(p.r))^2/p.b^2 < 1.0

function ellipse_points(p, thick=false)
    z = [p.μ1 + p.a*cos(θ)*cos(p.r) - p.b*sin(θ)*sin(p.r) for θ ∈ 0:0.01:2π]
    x = [p.μ2 + p.a*cos(θ)*sin(p.r) + p.b*sin(θ)*cos(p.r) for θ ∈ 0:0.01:2π]
    limit = thick ? 0.99 : 1.0
    for i in 1:length(z)
        if z[i] > limit
            z[i] = NaN
            x[i] = NaN
        end
    end
    return x, z
end

function ρ(x)
    if is_in_ellipse(x, p_ellipse)
        return [0.8, 0.0]
    else
        return [0.0, 1.2]
    end
end

# function ρ(x)
#     if (x[2] > -0.15 && x[2] < 0.15) && (x[1] > 0.85)
#         return [0.8, 0.0]
#     else
#         return [0.0, 1.2]
#     end
# end

mass_concentrations = [interpolate(x -> ρ(x)[1], M), interpolate(x -> ρ(x)[2], M)]
true_ρ_vals = [mass_concentrations[1].free_values mass_concentrations[2].free_values]

# build solver:

contourf(-1:0.01:1, 0:0.01:1, (x, z) -> mass_concentrations[2](Point(z, x)))
plot!(ellipse_points(p_ellipse)..., color=:gray, linestyle=:dash, label=nothing)

solver = build_solver(model, 11)
X = assemble_space_matrices(solver, mass_concentrations)
Ω = assemble_direction_matrices(solver, scattering_kernel)

g = (ϵ = [(ϵ -> pdf(Normal(ϵpos, 0.04), ϵ[1])) for ϵpos ∈ [0.85, 0.75, 0.65, 0.55, 0.45, 0.35]],
    x = [(x -> isapprox(x[1], 1.0) ? (pdf(MultivariateNormal([xpos, 0.0], [0.05, 0.05]), [(length(x)>1) ? x[2] : 0.0, (length(x)>2) ? x[3] : 0.0])) : 0.0) for xpos ∈ range(-0.5, 0.5, length=40)], 
    Ω = [(Ω -> pdf(VonMisesFisher(normalize(Ωpos), 10.0), [Ω...])) for Ωpos ∈ [[-0.5, 0.0, -1.0], [0.0, 0.0, -1.0], [0.5, 0.0, -1.0]]])

# reduced number of experiments
# g = (ϵ = [(ϵ -> pdf(Normal(ϵpos, 0.04), ϵ[1])) for ϵpos ∈ [0.85]],
#     x = [(x -> isapprox(x[1], 1.0) ? (pdf(MultivariateNormal([xpos, 0.0], [0.05, 0.05]), [(length(x)>1) ? x[2] : 0.0, (length(x)>2) ? x[3] : 0.0])) : 0.0) for xpos ∈ range(-0.5, 0.5, length=5)], 
#     Ω = [(Ω -> pdf(VonMisesFisher(normalize(Ωpos), 10.0), [Ω...])) for Ωpos ∈ [[0.0, 0.0, -1.0]]])

plot()
for gϵ ∈ g.ϵ
    plot!(range(0, 1, 100), gϵ)
end
plot!()

plot()
for gx ∈ g.x
    plot!(range(-1, 1, nhx), x -> gx(Point(1.0, x)))
end
plot!()

# function μ_x(nd)
#     if nd == Val(1)
#         return x -> (x[1] < 0.9 && x[1] > 0.8) ? 1.0 : 0.0
#     elseif nd == Val(2)
#         return x -> (x[2] > -0.2 && x[2] < 0.2) ? 1.0 : 0.0
#     end
# end

μ = (ϵ = [(ϵ -> (ϵ[1]-0.1 > 0) ? sqrt(ϵ[1]-0.1) : 0.0), (ϵ -> (ϵ[1]-0.2 > 0) ? sqrt(ϵ[1]-0.2) : 0.0)],
    x = [(ρ -> ρ), (ρ -> ρ)],
    Ω = [(Ω -> 1.0)])

plot()
for μϵ ∈ μ.ϵ
    plot!(range(0, 1, 100), μϵ)
end
plot!()

gh = semidiscretize_boundary(solver, g)
μh = semidiscretize_source(solver, μ, mass_concentrations)

# plotly()
# mm = measure_forward(solver, X, Ω, physics, (0, 1), 100, gh, μh)
# mmx = measure_adjoint(solver, X, Ω, physics, (0, 1), 100, gh, μh)

# plot(mm[:, 1, 1, 2])
# plot!(mmx[:, 1, 1, 2])

# ϵs, ψs = solve_forward(solver, X, Ω, physics, (0, 1), 100, ϵ -> 2*gh.ϵ[1](ϵ)*vcat(gh.x[1].p⊗gh.Ω[2].p, gh.x[1].m⊗gh.Ω[2].m))


param = (mass_concentrations=mass_concentrations, solver=solver, X=X, Ω=Ω, physics=physics, ϵ=(0.0, 1.0), N=100, gh=gh, μh=μh, proj=projection_matrix(solver), gram=gram_matrix(solver))

std_ρ_vals1 = zeros(size(true_ρ_vals))
std_ρ_vals2 = zeros(size(true_ρ_vals))
std_ρ_vals1[:, 1] .= 0.8
std_ρ_vals2[:, 2] .= 1.2
std_measurement = measure_std(std_ρ_vals1, param)
std_measurement[:, :, :, 2] = measure_std(std_ρ_vals2, param)[:, :, :, 2]

function measure_std(ρ_vals, params)
    for i ∈ 1:length(params.mass_concentrations)
        params.mass_concentrations[i].free_values .= ρ_vals[:, i]
    end
    update_space_matrices!(params.X, params.solver, params.mass_concentrations)
    update_extractions!(params.μh, params.solver, params.mass_concentrations)
    std_measurements = measure_adjoint(params.solver, params.X, params.Ω, params.physics, params.ϵ, params.N, params.gh, params.μh)
    return std_measurements
end

function measure_raw(ρ_vals, params)
    for i ∈ 1:length(params.mass_concentrations)
        params.mass_concentrations[i].free_values .= ρ_vals[:, i]
    end
    update_space_matrices!(params.X, params.solver, params.mass_concentrations)
    update_extractions!(params.μh, params.solver, params.mass_concentrations)
    measurements = measure_adjoint(params.solver, params.X, params.Ω, params.physics, params.ϵ, params.N, params.gh, params.μh)
    measurements .= measurements./std_measurement
    return measurements
end

Zygote.@adjoint function measure_raw(ρ_vals, params)
    for i ∈ 1:length(params.mass_concentrations)
        params.mass_concentrations[i].free_values .= ρ_vals[:, i]
    end
    update_space_matrices!(params.X, params.solver, params.mass_concentrations)
    update_extractions!(params.μh, params.solver, params.mass_concentrations)
    measurements, ψxss = measure_adjoint(params.solver, params.X, params.Ω, params.physics, params.ϵ, params.N, params.gh, params.μh, true)
    measurements .= measurements./std_measurement
    function measure_raw_adjoint(measurements_)
        measurements_ .= measurements_ ./ std_measurement
        function gh(ϵ, l)
            n_basis = number_of_basis_functions(params.solver)
            res = zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
            for (i, ghx) ∈ enumerate(params.gh.x)
                for (j, ghΩ) ∈ enumerate(params.gh.Ω)
                    for (k, ghϵ) ∈ enumerate(params.gh.ϵ)
                        res .+= measurements_[i, j, k, l] * 2.0*ghϵ(ϵ) * vcat(ghx.p⊗ghΩ.p, ghx.m⊗ghΩ.m)
                    end
                end
            end
            return res
        end
        ## hacked ... (first we project all the x.p basis to L2 -> orthogonal basis and we dont need space derivatives from now on...)
        ψxss_proj = [[project_disassemble_solution(params.solver, ψx, params.proj) for ψx ∈ ψxs] for ψxs ∈ ψxss]
        ρ_vals_ = zeros(size(ρ_vals))
        ϵs = range(params.ϵ[1], params.ϵ[2], params.N)
        Δϵ = ϵs[2] - ϵs[1]
        for (l, (μhx, μhϵ)) ∈ enumerate(zip(params.μh.x, params.μh.ϵ))
            _, ψs_ = solve_forward(params.solver, params.X, params.Ω, params.physics, params.ϵ, params.N, ϵ -> gh(ϵ, l))
            ψs_proj = [project_disassemble_solution(params.solver, ψ, params.proj) for  ψ ∈ ψs_]
            K = Diagonal([diag(params.Ω.Kpp); diag(param.Ω.Kmm)])
            for i ∈ 1:2 # number of elements
                for k ∈ 1:params.N-1
                    s_2 = params.physics.s((ϵs[k] + ϵs[k+1])/2)
                    ρ_vals_[:, i] .+= params.gram * (s_2[i] .* sum(ψs_proj[k] .* ψxss_proj[l][k+1] .- ψs_proj[k+1] .* ψxss_proj[l][k], dims=1)[:])
                end
                
                # initial and final are 0.0 (because initial and final conditions of ψ and λ)
                for k ∈ 2:params.N-1
                    τ = params.physics.τ(ϵs[k])
                    σ = params.physics.σ(ϵs[k])
                    ρ_vals_[:, i] .+= params.gram * ((Δϵ*τ[i]).*sum(ψs_proj[k] .* ψxss_proj[l][k], dims=1)[:])
                    ρ_vals_[:, i] .-= params.gram * ((Δϵ*σ[i]).*sum(K * (ψs_proj[k] .* ψxss_proj[l][k]), dims=1)[:])
                end
            end
            #for i ∈ 1:2 # number of elements
                # final is 0.0 (forward solution)
                μ_ϵ = μhϵ(ϵs[1])
                μ_Ω = [params.μh.Ω[1].p; params.μh.Ω[1].m]
                ρ_vals_[:, l] .+= params.gram * ((Δϵ * μ_ϵ / 2) .* (μ_Ω' * ψs_proj[1]))[:]
                for k ∈ 2:params.N-1
                    μ_ϵ = μhϵ(ϵs[k])
                    ρ_vals_[:, l] .+= params.gram * ((Δϵ*μ_ϵ) .* (μ_Ω' * ψs_proj[k]))[:]
                end
            # add dot_c 
            # project to ρ_vals
            # end
        end
        # for i in 1:2
        #     ρ_vals_[:, i] .= params.gram * ρ_vals_[:, i]
        # end
        return (ρ_vals_, nothing)
    end
    return measurements, measure_raw_adjoint
end

function project_disassemble_solution(solver, ψ, projection)
    n_basis = number_of_basis_functions(solver)
    ψp = reshape(@view(ψ[1:n_basis.x.p*n_basis.Ω.p]), n_basis.Ω.p, n_basis.x.p)
    ψm = reshape(@view(ψ[1 + n_basis.x.p*n_basis.Ω.p:n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m]), n_basis.Ω.m, n_basis.x.m)
    return vcat(ψp*projection, ψm)
end

true_measurements = measure_raw(true_ρ_vals, param)

plot(true_measurements[:, 2, 1, 2])
plot!(true_measurements[:, 1, 1, 2])
plot!(true_measurements[:, 3, 1, 2])
p = plot(measurements[:, 2, 1, 1], color=1)
plot()
p = plot!(true_measurements[:, 2, 1, 1], color=1, linestyle=:dash)
p = plot!(measurements[:, 2, 1, 2], color=2)
p = plot!(true_measurements[:, 2, 1, 2], color=2, linestyle=:dash)
p = plot!(measurements[:, 1, 1, 1], color=3)
p = plot!(true_measurements[:, 1, 1, 1], color=3, linestyle=:dash)
counter = [0]

function objective(ρ_vals)
    n = length(true_measurements)
    measurements = measure_raw(ρ_vals, param)
    Zygote.ignore() do
        p = plot(measurements[:, 2, 1, 1], color=1)
        p = plot!(true_measurements[:, 2, 1, 1], color=1, linestyle=:dash)
        p = plot!(measurements[:, 2, 1, 2], color=2)
        p = plot!(true_measurements[:, 2, 1, 2], color=2, linestyle=:dash)
        p = plot!(measurements[:, 1, 1, 1], color=3)
        p = plot!(true_measurements[:, 1, 1, 1], color=3, linestyle=:dash)
        display(p)
        savefig(p, "measurements/plot$(counter[1]).pdf")
        counter[1] += 1
    end
    return 1/n * sum((true_measurements .- measurements).^2)
end

xys = mean.(Gridap.get_cell_coordinates(M.fe_basis.trian.grid))[:]
xys_mat = [xys[i][j] for j in 1:2, i in 1:length(xys)]

# model 1
NN = Chain(Dense(2, 10, tanh), Dense(10, 10, tanh), Dense(10, 2), Lux.softmax)

#model 2
NN = Chain(Dense(2, 10, tanh), Dense(10, 2), Lux.softmax)
ps, st = Lux.setup(Random.default_rng(), NN)
p0, restruct = Optimisers.destructure(ps)

function p_trans(p)
    NN_params = restruct(p)
    p_ = Lux.apply(NN, xys_mat, NN_params, st)[1]
    # pxs = choose_p.(xy, Ref(p))
    # p_ = Lux.NNlib.σ.(pxs)
    ρ_vals = p_[1, :] .* [0.0, 1.2]' .+ p_[2, :] .* [0.8, 0.0]'
    @show sum(ρ_vals, dims=2)
    @assert all(sum(ρ_vals, dims=2) .> 0.75)
    return ρ_vals
end

function fg!(F, G, x)
    if G !== nothing
        res = Zygote.withgradient(objective ∘ p_trans, x)
        G .= res.grad[1]
        return res.val
    end
    if F !== nothing
        return objective(p_trans(x))
    end
end

counter2 = [0]

function optim_callback(trace)
    m_temp = FEFunction(M, p_trans(trace[end].metadata["x"])[:, 2])
    p = contourf(-1:0.05:1, 0:0.05:1, (x, z) -> m_temp(Point(z, x)))
    display(p)
    savefig(p, "material/plot$(counter2[1]).pdf")
    serialize("traces/trace$(counter2[1]).jls", trace)
    counter2[1] += 1
    return false
end

res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.LBFGS(), Optim.Options(store_trace=true, extended_trace=true, iterations=1000, time_limit=60*60*30, g_abstol=1e-5, g_reltol=1e-5, callback=optim_callback))


# ### run only the following line:
# res_cont = Optim.optimize(Optim.only_fg!(fg!), res.trace[end].metadata["x"], res.method, Optim.Options(store_trace=true, extended_trace=true, iterations=1000, time_limit=60*60*10, g_abstol=1e-6, g_reltol=1e-6, callback=optim_callback))

# res_cont2 = Optim.optimize(Optim.only_fg!(fg!), res_cont.trace[end].metadata["x"], res_cont.method, Optim.Options(store_trace=true, extended_trace=true, iterations=1000, time_limit=60*60*10, g_abstol=1e-6, g_reltol=1e-6, callback=optim_callback))


nothing


#### end
@gif for i in 1:length(res.trace)
    m_temp = FEFunction(M, p_trans(res.trace[i].metadata["x"])[:, 1])
    p = contourf(-1:0.05:1, 0:0.05:1, (x, z) -> m_temp(Point(z, x)), clims=(0, 0.8))
    hline!([0.85])
    vline!([-0.15, 0.15])
    title!("iteration: $(i)")
end fps=5

trace_measurements = [measure_raw(p_trans(res.trace[i].metadata["x"]), param) for i in 1:length(res.trace)]

@gif for i in 1:length(res.trace)
    plot(true_measurements[:, 2, 1, 1], color=1)
    plot!(true_measurements[:, 2, 1, 2], color=2)
    plot!(true_measurements[:, 1, 1, 1], color=3)
    # measurements = measure_raw(p_trans(res.trace[i].metadata["x"]), param)
    plot!(trace_measurements[i][:, 2, 1, 1], color=1, linestyle=:dash)
    plot!(trace_measurements[i][:, 2, 1, 2], color=2, linestyle=:dash)
    plot!(trace_measurements[i][:, 1, 1, 1], color=3, linestyle=:dash)
    title!("interation: $(i)")
end fps=5


function finite_diff(f, x, h, index)
    val0 = f(x)
    x_ = copy(x)
    x_[index...] += h
    return (objective(x_) - val0)/h
end

finite_diff(objective, ρ_vals, 0.01, (875, 1))
grad[1][875, 1]

argmax(grad[1])

m = FEFunction(M, p_trans(grad.grad[1])[:, 1])
m = FEFunction(M, p_trans(p0)[:, 1])
m = FEFunction(M, p_trans(res.minimizer)[:, 2])
contourf(0:0.05:1, -1:0.05:1, (z, x) -> m(Point(z, x)))

@gif for i in 1:16
    m_i = FEFunction(M, p_trans(res.trace[i].metadata["x"])[:, 2])
    contourf(0:0.05:1, -1:0.05:1, (z, x) -> m_i(Point(z, x)), clims=(0, 1))
end fps=2
p_trans(p0)

# test_vec = reshape(ψs[50][1:n_basis.x.p*n_basis.Ω.p], n_basis.Ω.p, n_basis.x.p)
# test_vec_proj = test_vec * projection_matrix
# u = FEFunction(solver.U[1], test_vec[1, :])
# m = FEFunction(M, test_vec_proj[1, :])

# contourf(0:0.1:1, -1:0.1:1, (z, x) -> u(Point(z, x)))
# contourf(0:0.1:1, -1:0.1:1, (z, x) -> m(Point(z, x)))


# function integrate_direction(solver, ψ, ϕ)
#     n_basis = number_of_basis_functions(solver)
#     ψp = reshape(@view(ψ[1:n_basis.x.p*n_basis.Ω.p]), n_basis.Ω.p, n_basis.x.p)
#     ψm = reshape(@view(ψ[1 + n_basis.x.p*n_basis.Ω.p:n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m]), n_basis.Ω.m, n_basis.x.m)
#     ϕp = reshape(@view(ϕ[1:n_basis.x.p*n_basis.Ω.p]), n_basis.Ω.p, n_basis.x.p)
#     ϕm = reshape(@view(ϕ[1 + n_basis.x.p*n_basis.Ω.p:n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m]), n_basis.Ω.m, n_basis.x.m)
#     return (p=sum(ψp .* ϕp, dims=1)[:], m=sum(ψm .* ϕm, dims=1)[:])
# end

@time res = integrate_direction(solver, ψs[10], ψs[20])
res.p

plot(measurementsx2[:, 2, 1, 1])
plot!(measurementsx2[:, 2, 1, 2])

plot!(measurementsx2[:, 2, end, 1])
plot!(measurementsx2[:, 2, end, 2])

plot!(measurementsx2[:, 1, 1, 1])
plot!(measurementsx2[:, 1, 1, 2])

plot!(measurementsx2[:, 3, 1, 1])
plot!(measurementsx2[:, 3, 1, 2])

plot(measurementsx2[:, 2, 1, 1])
plot!(measurementsx2[:, 1, 1, 2])


z_coords = range(0, 1, length=100)
x_coords = range(-0.7, 0.7, length=150)

e_x = eval_space(solver.U, Point.(z_coords, x_coords'))
sol = zeros(length(z_coords), length(x_coords))
n_basis = number_of_basis_functions(solver)
e_Ω_p = spzeros(n_basis.Ω.p)
# e_Ω_p = gh.Ω[1].p
e_Ω_m = spzeros(n_basis.Ω.m)
e_Ω_p[1] = 1.0

cgrad_electron = cgrad([:black, :lightblue, :white])
cgrad_phase1 = cgrad([:black, :lightgreen, :white])
cgrad_phase2 = cgrad([:black, :orange, :white])
cgrad_phases = cgrad([:green, :orange])

#### VISUALIZATION OF THE EXPERIMENT

## VISUALIZATION OF THE MATERIAL
using LaTeXStrings

let
    z_coords_phase = range(0, 1, length=300)
    x_coords_phase = range(-0.7, 0.7, length=500)
    phases = reshape(getindex.(ρ.(Point.(z_coords_phase, x_coords_phase')[:]), 2), length(z_coords_phase), length(x_coords_phase))

    # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
    p1 = scatter(xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1))
    p1 = ylabel!("Intensity")

    # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
    # p1 = xlims!(-0.75, 0.75)

    #p2 = contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, aspect_ratio=:equal, c=cgrad_electron, colorbar=false, xticks=-0.7:0.2:0.7)
    p2 = contourf(x_coords_phase, z_coords_phase, phases, linewidth=0.0, levels=10, c=cgrad_phases, colorbar=false, xticks=-0.7:0.2:0.7)
    p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
    p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
    p2 = plot!([-0.3, -0.3], [1.07, 1.01], arrow=true, color=:white, label=nothing)
    p2 = xlabel!("Position x")
    p2 = ylabel!("Depth z")
    # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
    # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

    plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
    savefig("full_visualization/01_material.pdf")
end


## VISUALIZATION OF ONE FULL FORWARD SOLUTION
I_fwrd = 10
forward_sol = solve_forward(solver, X, Ω, physics, (0, 1), 100, ϵ -> 2*gh.ϵ[1](ϵ)*vcat(gh.x[I_fwrd].p⊗gh.Ω[2].p, gh.x[I_fwrd].m⊗gh.Ω[2].m))
let
    counter = [0]
    for k in 100:-1:1
        @show k
        for (i, x_) in enumerate(z_coords)
            for (j, y_) in enumerate(x_coords)
                e_x_p, e_x_m = e_x[i, j]
                full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
                sol[i, j] = dot(forward_sol[2][k], full_basis)
                #proj_basis = vcat((μh.x[2].p .* e_x_p)⊗μh.Ω[1].p, (μh.x[2].m .* e_x_m)⊗μh.Ω[1].m)
                #sol[i, j] = μh.ϵ[2](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis)
            end
        end
        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.7, 0.7, length=200), x -> 1/650*g.ϵ[1](forward_sol[1][k])*g.x[I_fwrd]([1.0, x]), xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=1)
        p1 = ylabel!("Beam Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, c=cgrad_electron, colorbar=false, clims=(0, 140), xticks=-0.7:0.2:0.7)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[I_fwrd], range(-0.5, 0.5, length=40)[I_fwrd]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        savefig("full_visualization/02_forward/$(string(counter[1], pad=3))_fluence.pdf")
        counter[1] += 1
    end 
    # savefig("material.pdf")
end

## VISUALIZATION OF THE IONIZATION
let
    counter = [0]
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))

    for k in 100:-1:1
        @show k
        for (i, x_) in enumerate(z_coords)
            for (j, y_) in enumerate(x_coords)
                e_x_p, e_x_m = e_x[i, j]
                #full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
                # sol[i, j] = dot(forward_sol[2][k], full_basis)
                if is_in_ellipse([x_, y_], p_ellipse)
                    proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
                    sol1[i, j] = μh.ϵ[1](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis1)
                    sol2[i, j] = NaN
                else
                    proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
                    sol1[i, j] = 0.0
                    sol2[i, j] = μh.ϵ[2](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis2)
                end
            end
        end
        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.7, 0.7, length=200), x -> 1/650*g.ϵ[1](forward_sol[1][k])*g.x[I_fwrd]([1.0, x]), xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=1)
        p1 = ylabel!("Beam Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, max.(sol1, 0.0), linewidth=0.0, levels=10, clims=(0, 0.03), c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, clims=(0, 0.03), c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[I_fwrd], range(-0.5, 0.5, length=40)[I_fwrd]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        savefig("full_visualization/03_ionization/$(string(counter[1], pad=3))_ionization.pdf")
        counter[1] += 1
    end
end

## VISUALIZATION OF THE INTERACTION VOLUME
let
    Ωint = [Diagonal(ones(n_basis.x.p)) ⊗ e_Ω_p spzeros(n_basis.x.p*n_basis.Ω.p, n_basis.x.m)
            spzeros(n_basis.x.m*n_basis.Ω.m, n_basis.x.p) Diagonal(ones(n_basis.x.m)) ⊗ e_Ω_m]
    temp1 = zeros(n_basis.x.p + n_basis.x.m)
    temp2 = zeros(n_basis.x.p + n_basis.x.m)
    # temp2 = zeros(forward_sol[2][1] |> size)
    Δϵ = forward_sol[1][2] - forward_sol[1][1]

    for k in 1:100
        temp1 += (Δϵ* μh.ϵ[1](forward_sol[1][k])).*(Ωint'*forward_sol[2][k])
        temp2 += (Δϵ* μh.ϵ[2](forward_sol[1][k])).*(Ωint'*forward_sol[2][k])
    end

    temp1 .*= vcat(μh.x[1].p, μh.x[1].m)
    temp2 .*= vcat(μh.x[2].p, μh.x[2].m)
    
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))
    for (i, x_) in enumerate(z_coords)
        for (j, y_) in enumerate(x_coords)
            e_x_p, e_x_m = e_x[i, j]
            full_basis = vcat(e_x_p, e_x_m)
            if is_in_ellipse([x_, y_], p_ellipse)
                # proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
                sol1[i, j] = dot(temp1, full_basis)
                sol2[i, j] = NaN
            else
                # proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
                sol1[i, j] = 0.0
                sol2[i, j] = dot(temp2, full_basis)
            end
        end
    end
    # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
    p1 = scatter([range(-0.5, 0.5, length=40)[I_fwrd]], [true_measurements[I_fwrd, 1, 1, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, markerstrokewidth=0.5)
    p1 = scatter!([range(-0.5, 0.5, length=40)[I_fwrd]], [true_measurements[I_fwrd, 1, 1, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, markerstrokewidth=0.5)
    p1 = ylabel!("Photon Intensity")

    # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
    # p1 = xlims!(-0.75, 0.75)

    p2 = contourf(x_coords, z_coords, max.(sol1, 0.0), linewidth=0.0, levels=10, c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
    p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
    p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
    p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
    p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
    p2 = plot!([range(-0.5, 0.5, length=40)[I_fwrd], range(-0.5, 0.5, length=40)[I_fwrd]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
    p2 = xlabel!("Position x")
    p2 = ylabel!("Depth z")
    # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
    # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

    plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
    savefig("full_visualization/04_interaction_volume.pdf")
end

## VISUALIZATION OF A LINESCAN

function compute_and_integrate_not_space(i_x, i_ϵ, i_Ω)
    forward_sol = solve_forward(solver, X, Ω, physics, (0, 1), 100, ϵ -> 2*gh.ϵ[i_ϵ](ϵ)*vcat(gh.x[i_x].p⊗gh.Ω[i_Ω].p, gh.x[i_x].m⊗gh.Ω[i_Ω].m))
    Ωint = [Diagonal(ones(n_basis.x.p)) ⊗ e_Ω_p spzeros(n_basis.x.p*n_basis.Ω.p, n_basis.x.m)
            spzeros(n_basis.x.m*n_basis.Ω.m, n_basis.x.p) Diagonal(ones(n_basis.x.m)) ⊗ e_Ω_m]
    temp1 = zeros(n_basis.x.p + n_basis.x.m)
    temp2 = zeros(n_basis.x.p + n_basis.x.m)
    # temp2 = zeros(forward_sol[2][1] |> size)
    Δϵ = forward_sol[1][2] - forward_sol[1][1]

    for k in 1:100
        temp1 += (Δϵ* μh.ϵ[1](forward_sol[1][k])).*(Ωint'*forward_sol[2][k])
        temp2 += (Δϵ* μh.ϵ[2](forward_sol[1][k])).*(Ωint'*forward_sol[2][k])
    end

    temp1 .*= vcat(μh.x[1].p, μh.x[1].m)
    temp2 .*= vcat(μh.x[2].p, μh.x[2].m)
    return temp1, temp2
end

temps = [compute_and_integrate_not_space(i, 1, 2) for i in 1:5]
serialize("full_visualization/temps.jls", temps)

let
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))
    counter = [0]
    for k in 1:length(temps)
        for (i, x_) in enumerate(z_coords)
            for (j, y_) in enumerate(x_coords)
                e_x_p, e_x_m = e_x[i, j]
                full_basis = vcat(e_x_p, e_x_m)
                if is_in_ellipse([x_, y_], p_ellipse)
                    # proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
                    sol1[i, j] = dot(temps[k][1], full_basis)
                    sol2[i, j] = NaN
                else
                    # proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
                    sol1[i, j] = 0.0
                    sol2[i, j] = dot(temps[k][2], full_basis)
                end
            end
        end

        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.5, 0.5, length=40)[1:k], [true_measurements[1:k, 2, 1, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green)
        p1 = plot!(range(-0.5, 0.5, length=40)[1:k], [true_measurements[1:k, 2, 1, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 2, 1, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, markerstrokewidth=0.5)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 2, 1, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, markerstrokewidth=0.5)
        p1 = ylabel!("Photon Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, max.(sol1, 0.0), linewidth=0.0, levels=10, c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[k], range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        savefig("full_visualization/05_linescan1/$(string(counter[1], pad=3))_linescan1.pdf")
        counter[1] += 1
    end
end

## VISUALIZATION OF LINESCAN 2
temps2 = [compute_and_integrate_not_space(i, 1, 3) for i in 1:5]
serialize("full_visualization/temps2.jls", temps2)

let
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))
    counter = [0]
    @gif for k in 1:40
        for (i, x_) in enumerate(z_coords)
            for (j, y_) in enumerate(x_coords)
                e_x_p, e_x_m = e_x[i, j]
                full_basis = vcat(e_x_p, e_x_m)
                if is_in_ellipse([x_, y_], p_ellipse)
                    # proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
                    sol1[i, j] = dot(temps2[1][1], full_basis)
                    sol2[i, j] = NaN
                else
                    # proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
                    sol1[i, j] = 0.0
                    sol2[i, j] = dot(temps2[1][2], full_basis)
                end
            end
        end

        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange)
        p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, 3, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dash)
        p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, 3, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dash)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 3, 1, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, markerstrokewidth=0.5)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 3, 1, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, markerstrokewidth=0.5)
        p1 = ylabel!("Photon Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, max.(sol1, 0.0), linewidth=0.0, levels=10, c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[k] + 0.03, range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        #savefig("full_visualization/06_linescan2/$(string(counter[1], pad=3))_linescan2.pdf")
        counter[1] += 1
    end fps=5
end

## VISUALIZATION OF LINESCAN 2
temps3 = [compute_and_integrate_not_space(i, 5, 2) for i in 1:3]
serialize("full_visualization/temps3.jls", temps3)

let
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))
    counter = [0]
    @gif for k in 1:3
        for (i, x_) in enumerate(z_coords)
            for (j, y_) in enumerate(x_coords)
                e_x_p, e_x_m = e_x[i, j]
                full_basis = vcat(e_x_p, e_x_m)
                if is_in_ellipse([x_, y_], p_ellipse)
                    # proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
                    sol1[i, j] = dot(temps3[k][1], full_basis)
                    sol2[i, j] = NaN
                else
                    # proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
                    sol1[i, j] = 0.0
                    sol2[i, j] = dot(temps3[k][2], full_basis)
                end
            end
        end

        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 3, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dash)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 3, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dash)
        p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, 2, 5, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dot)
        p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, 2, 5, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dot)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 2, 5, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, markerstrokewidth=0.5)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 2, 5, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, markerstrokewidth=0.5)
        p1 = ylabel!("Photon Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, max.(sol1, 0.0), linewidth=0.0, levels=10, c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[k], range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        #savefig("full_visualization/07_linescan3/$(string(counter[1], pad=3))_linescan3.pdf")
        counter[1] += 1
    end fps=5
end

let
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))
    counter = [0]
    for k = 1:40
        # for (i, x_) in enumerate(z_coords)
        #     for (j, y_) in enumerate(x_coords)
        #         e_x_p, e_x_m = e_x[i, j]
        #         full_basis = vcat(e_x_p, e_x_m)
        #         if is_in_ellipse([x_, y_], p_ellipse)
        #             # proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
        #             sol1[i, j] = dot(temps3[k][1], full_basis)
        #             sol2[i, j] = NaN
        #         else
        #             # proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
        #             sol1[i, j] = 0.0
        #             sol2[i, j] = dot(temps3[k][2], full_basis)
        #         end
        #     end
        # end

        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 3, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dash)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 3, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dash)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 2, 5, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dot)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 2, 5, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dot)
        for (i, j) ∈ [(1, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (1, 5), (3, 5), (1, 6), (2, 6), (3, 6)]
            p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, i, j, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dashdot)
            p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, i, j, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dashdot)
            p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, i, j, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, markerstrokewidth=0.5)
            p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, i, j, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, markerstrokewidth=0.5)
        end

        p1 = ylabel!("Photon Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, fill(NaN, size(sol1)), linewidth=0.0, levels=10, clims=(0, 1), c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
        # p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
        # p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[k], range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = plot!([range(-0.5, 0.5, length=40)[k]-0.03, range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = plot!([range(-0.5, 0.5, length=40)[k]+0.03, range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = annotate!(0.0, 0.5, "too expensive to compute", color=:green)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        savefig("full_visualization/08_full_measurements/$(string(counter[1], pad=3))_full_measurements.pdf")
        counter[1] += 1
    end
end

plot(range(0, 1, length=100), ϵ -> g.ϵ[1](ϵ))

plot(range(-0.7, 0.7, length=200), x -> 1/65 *g.x[I_fwrd]([1.0, x]), xticks=-0.7:0.2:0.7, label=nothing)

contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, aspect_ratio=:equal, clims=(0, 200), c=cgrad_electron, colorbar=false)
plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing)
#contourf(x_coords, z_coords, reverse(sol, dims=2), linewidth=0.0, levels=10, aspect_ratio=:equal)
#plot(x_coords, -sol[1, :])
title!("energy: $(round(forward_sol[1][k], digits=2))")

@gif for k in 100:-1:1
    @show k
    for (i, x_) in enumerate(z_coords)
        for (j, y_) in enumerate(x_coords)
            e_x_p, e_x_m = e_x[i, j]
            full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
            sol[i, j] = dot(forward_sol[2][k], full_basis)
            #proj_basis = vcat((μh.x[2].p .* e_x_p)⊗μh.Ω[1].p, (μh.x[2].m .* e_x_m)⊗μh.Ω[1].m)
            #sol[i, j] = μh.ϵ[2](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis)
        end
    end
    contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, aspect_ratio=:equal, clims=(0, 200), c=cgrad_electron, colorbar=false)
    plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing)
    #contourf(x_coords, z_coords, reverse(sol, dims=2), linewidth=0.0, levels=10, aspect_ratio=:equal)
    #plot(x_coords, -sol[1, :])
    title!("energy: $(round(forward_sol[1][k], digits=2))")
end fps=10


for k in 100:-1:1
    @show k
    for (i, x_) in enumerate(z_coords)
        for (j, y_) in enumerate(x_coords)
            e_x_p, e_x_m = e_x[i, j]
            #full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
            #sol[i, j] = dot(forward_sol[2][k], full_basis)
            proj_basis = vcat((μh.x[2].p .* e_x_p)⊗μh.Ω[1].p, (μh.x[2].m .* e_x_m)⊗μh.Ω[1].m)
            sol[i, j] += μh.ϵ[2](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis)
        end
    end
end


p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 2.1), xlims=(-0.71, 0.71), aspect_ratio=:equal)
p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
p1 = xlims!(-0.75, 0.75)

#p2 = contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, aspect_ratio=:equal, c=cgrad_electron, colorbar=false, xticks=-0.7:0.2:0.7)
p2 = contourf(x_coords, z_coords, phases, linewidth=0.0, levels=10, aspect_ratio=:equal, c=cgrad_phases, colorbar=false, xticks=-0.7:0.2:0.7, xlims=(-0.71, 0.71))
p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=3)
# p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

plot(p1, p2, layout=(2, 1), link=:x)

@gif for k in 100:-1:1
    @show k
    for (i, x_) in enumerate(z_coords)
        for (j, y_) in enumerate(x_coords)
            e_x_p, e_x_m = e_x[i, j]
            full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
            sol[i, j] = dot(forward_sol[2][k], full_basis)
            #proj_basis = vcat((μh.x[2].p .* e_x_p)⊗μh.Ω[1].p, (μh.x[2].m .* e_x_m)⊗μh.Ω[1].m)
            #sol[i, j] = μh.ϵ[2](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis)
        end
    end
    contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, aspect_ratio=:equal, clims=(0, 200), c=cgrad_electron, colorbar=false)
    plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing)
    #contourf(x_coords, z_coords, reverse(sol, dims=2), linewidth=0.0, levels=10, aspect_ratio=:equal)
    #plot(x_coords, -sol[1, :])
    title!("energy: $(round(forward_sol[1][k], digits=2))")
end fps=10
