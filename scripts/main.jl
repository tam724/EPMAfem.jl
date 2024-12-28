using Revise

using EPMAfem
using Plots
#using GLMakie

#import EPMAfem.SphericalHarmonicsModels as SH
#import EPMAfem.SpaceModels as SM
using LinearAlgebra
using Gridap

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1, 1), (20, 40)))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(7, 2)
equations = EPMAfem.PNEquations()
excitation = EPMAfem.PNExcitation([(x=x_, y=0.0) for x_ in -0.7:0.05:0.7], [0.8, 0.7], [VectorValue(-1.0, 0.0, 0.0), VectorValue(-1.0, -1.0, 0.0) |> normalize])
extraction = EPMAfem.PNExtraction()

# SM.dimensionality(space_model)
# SH.dimensionality(direction_model)

model = EPMAfem.PNGridapModel(space_model, 0.0:0.01:1.0, direction_model, EPMAfem.cuda())
model_cuda = EPMAfem.PNGridapModel(space_model, 0.0:0.01:1.0, direction_model, EPMAfem.cuda())
# model_cuda = EPMAfem.PNGridapModel(space_model, 0.0:0.05:1.0, direction_model, EPMAfem.cuda())

# discrete_problem_cpu = EPMAfem.discretize_problem(equations, model_cpu)
# discrete_problem_cuda = EPMAfem.discretize_problem(equations, model_cuda)

# abs.((discrete_problem_cpu.ρm[2] .- (discrete_problem_cuda.ρm[2] |> collect))) |> maximum


#model = model_cuda
discrete_problem = EPMAfem.discretize_problem(equations, model)
discrete_problem2 = EPMAfem.discretize_problem(equations, model_cuda)

for (ρp, ρp2) in zip(discrete_problem.ρp, discrete_problem2.ρp)
    @assert collect(ρp) ≈ collect(ρp2)
end
for (ρm, ρm2) in zip(discrete_problem.ρm, discrete_problem2.ρm)
    @assert collect(ρm.diag) ≈ collect(ρm2.diag)
end
for (∂p, ∂p2) in zip(discrete_problem.∂p, discrete_problem2.∂p)
    @assert collect(∂p) ≈ collect(∂p2)
end
for (∇pm, ∇pm2) in zip(discrete_problem.∇pm, discrete_problem2.∇pm)
    @assert collect(∇pm) ≈ collect(∇pm2)
end
@assert collect(discrete_problem.Ip.diag) ≈ collect(discrete_problem2.Ip.diag)
@assert collect(discrete_problem.Im.diag) ≈ collect(discrete_problem2.Im.diag)
for (kp, kp2) in zip(discrete_problem.kp, discrete_problem2.kp)
    for (k, k2) in zip(kp, kp2)
        @assert collect(k.diag) ≈ collect(k2.diag)
    end
end
for (km, km2) in zip(discrete_problem.km, discrete_problem2.km)
    for (k, k2) in zip(km, km2)
        @assert collect(k.diag) ≈ collect(k2.diag)
    end
end
for (absΩp, absΩp2) in zip(discrete_problem.absΩp, discrete_problem2.absΩp)
    @assert collect(absΩp) ≈ collect(absΩp2)
end
for (Ωpm, Ωpm2) in zip(discrete_problem.Ωpm, discrete_problem2.Ωpm)
    @assert collect(Ωpm) ≈ collect(Ωpm2)
end


# test = EPMAfem.SpaceModels.assemble_bilinear(SM.∫R_uv, space_model, EPMAfem.SpaceModels.even(space_model), EPMAfem.SpaceModels.even(space_model))

discrete_rhs = EPMAfem.discretize_rhs(excitation, model)
#test_rhs = EPMAfem.discretize_stange_rhs(excitation, model)
discrete_ext = EPMAfem.discretize_extraction(extraction, model)

solver_schur = EPMAfem.pn_schurimplicitmidpointsolver(equations, model, sqrt(eps(Float64)))
solver_schur.tmp .= NaN
solver_schur.tmp2 .= NaN
solver_schur.tmp3 .= NaN
solver_schur.D .= NaN
solver_schur.rhs .= NaN
solver_schur.sol .= NaN


#solver_full = EPMAfem.pn_fullimplicitmidpointsolver(equations, model)

# solution = EPMAfem.iterator(discrete_problem, discrete_rhs[1, 14, 1], solver)

g = discrete_rhs
A_gi = EPMAfem.iterator(discrete_problem, g[1, 14, 1], solver_schur)
h = discrete_ext

hh = h(A_gi)

for (ϵ, i) in A_gi
    @show i
    sol = EPMAfem.current_solution(A_gi.solver)
    sol_p = EPMAfem.pview(sol, model)
    cpu_vec = collect(@view(sol_p[:, 1]))
    # @show sol_p |> size
    p = heatmap(reshape(cpu_vec, (21, 41)))
    display(p)
    sleep(0.01)
end

cv(x) = EPMAfem.convert_to_architecture(EPMAfem.architecture(model), x)
ρs = [EPMAfem.SpaceModels.L2_projection(x -> EPMAfem.mass_concentrations(equations, e, x), EPMAfem.space(model)) for e in 1:EPMAfem.number_of_elements(equations)] |> cv
EPMAfem.update_problem!(discrete_problem, ρs)


Astar_hi = EPMAfem.iterator(discrete_problem, h[1], solver_schur)
gg = g(Astar_hi)

hh[1]
gg[1, 14, 1]

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


solution_schur = EPMAfem.iterator(discrete_problem, discrete_ext[1], solver_schur)
solution_full = EPMAfem.iterator(discrete_problem, discrete_ext[1], solver_full)
meas_schur = discrete_rhs(solution_schur)
meas_full = discrete_rhs(solution_full)

plot(meas_schur[1, :, 1])
plot!(meas_full[1, :, 1])
plot!(meas_schur[2, :, 1])
plot!(meas_full[2, :, 1])

cv(x) = EPMAfem.convert_to_architecture(EPMAfem.architecture(model), x)

ρs = [EPMAfem.SpaceModels.L2_projection(x -> EPMAfem.mass_concentrations(equations, e, x), EPMAfem.space(model)) for e in 1:EPMAfem.number_of_elements(equations)] |> cv

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
ρs = [EPMAfem.SpaceModels.L2_projection(x -> EPMAfem.mass_concentrations(equations, e, x), EPMAfem.space(model)) for e in 1:EPMAfem.number_of_elements(equations)] |> cv

EPMAfem.update_problem!(discrete_problem, ρs)

tangent_rhs_schur = EPMAfem.tangent(solution_schur)
tangent_rhs_full = EPMAfem.tangent(solution_full)

new_rhs_schur = tangent_rhs_schur[1, 400];
new_rhs_full = tangent_rhs_full[1, 400];

der_sol_schur = EPMAfem.iterator(discrete_problem, new_rhs_schur, solver_schur);
der_sol_full = EPMAfem.iterator(discrete_problem, new_rhs_full, solver_full);

meas_tang_schur = discrete_rhs(der_sol_schur)

weights = zeros(size(discrete_rhs))
weights[1, 15, 1] = 1.0
adjoint_rhs_schur = EPMAfem.weight_array_of_r1(weights, discrete_rhs)
adjoint_adjoint_solution_schur = EPMAfem.iterator(discrete_problem, adjoint_rhs_schur, solver_schur) 
ρs_adjoint = tangent_rhs_schur(adjoint_adjoint_solution_schur)

@gif for _ in adjoint_adjoint_solution_schur
    sol = EPMAfem.current_solution(solver_schur)
    sol_p = EPMAfem.pview(sol, model)
    cpu_vec = collect(@view(sol_p[:, 1]))
    # @show sol_p |> size
    heatmap(reshape(cpu_vec, (21, 41)))
end

Vector(ρs_adjoint[1])[400]
meas_tang_schur[1, 15, 1]

m = FEFunction(EPMAfem.SpaceModels.material(EPMAfem.space(model)), ρs_adjoint[1])

trian = EPMAfem.SpaceModels.get_args(EPMAfem.space(model))[2]
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