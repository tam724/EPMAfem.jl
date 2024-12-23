using Revise

using EPMAfem
using Plots
#using GLMakie

import EPMAfem.SphericalHarmonicsModels as SH
import EPMAfem.SpaceModels as SM
using LinearAlgebra
using Gridap

space_model = SM.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1, 1), (100, 200)))
direction_model = SH.EEEOSphericalHarmonicsModel(21, 2)

SM.dimensionality(space_model)
SH.dimensionality(direction_model)

model = EPMAfem.PNGridapModel(space_model, 0:0.01:1, direction_model, EPMAfem.cuda())
equations = EPMAfem.PNEquations()
excitation = EPMAfem.PNExcitation([(x=x_, y=0.0) for x_ in -0.7:0.05:0.7], [0.8, 0.7], [VectorValue(-1.0, 0.0, 0.0), VectorValue(-1.0, -1.0, 0.0) |> normalize])
extraction = EPMAfem.PNExtraction()

discrete_problem = EPMAfem.discretize_problem(equations, model)
discrete_rhs = EPMAfem.discretize_rhs(excitation, model)
discrete_ext = EPMAfem.discretize_extraction(extraction, model)

solver = EPMAfem.pn_schurimplicitmidpointsolver(equations, model)
solution2 = EPMAfem.iterator(discrete_problem, discrete_rhs[1, 1, 1], solver)
solution = EPMAfem.iterator(discrete_problem, discrete_ext[1], solver)

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