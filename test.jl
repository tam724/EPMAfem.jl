using Gridap
using SparseArrays
using HCubature
using LinearAlgebra
using Enzyme
using Distributions
using Plots
using IterativeSolvers

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

nhx = 100
model = CartesianDiscreteModel((0.0, 1.0, -1.0, 1.0), (60, nhx))

M = material_space(model)

function ρ(x)
    if x[2] < -0.1 || x[2] > 0.1
        return [0.0, 1.0]
    else
        return [0.8, 0.0]
    end
end

mass_concentrations = [interpolate(x -> ρ(x)[1], M), interpolate(x -> ρ(x)[2], M)]
# build solver:

solver = build_solver(model, 11)
X = assemble_space_matrices(solver, mass_concentrations)
Ω = assemble_direction_matrices(solver, scattering_kernel)

g = (ϵ = [(ϵ -> pdf(Normal(ϵpos, 0.04), ϵ[1])) for ϵpos ∈ [0.85, 0.75, 0.65, 0.55, 0.45, 0.35]],
    x = [(x -> isapprox(x[1], 1.0) ? (pdf(MultivariateNormal([xpos, 0.0], [0.05, 0.05]), [(length(x)>1) ? x[2] : 0.0, (length(x)>2) ? x[3] : 0.0])) : 0.0) for xpos ∈ range(-0.5, 0.5, length=40)], 
    Ω = [(Ω -> pdf(VonMisesFisher(normalize(Ωpos), 10.0), [Ω...])) for Ωpos ∈ [[-0.5, 0.0, -1.0], [0.0, 0.0, -1.0], [0.5, 0.0, -1.0]]])

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

function μ_x(nd)
    if nd == Val(1)
        return x -> (x[1] < 0.9 && x[1] > 0.8) ? 1.0 : 0.0
    elseif nd == Val(2)
        return x -> (x[2] > -0.2 && x[2] < 0.2) ? 1.0 : 0.0
    end
end

μ = (ϵ = [(ϵ -> (ϵ[1]-0.1 > 0) ? sqrt(ϵ[1]-0.1) : 0.0), (ϵ -> (ϵ[1]-0.2 > 0) ? sqrt(ϵ[1]-0.2) : 0.0)],
    x = [(x -> ρ(x)[1]), (x -> ρ(x)[2])],
    Ω = [(Ω -> 1.0)])

plot()
for μϵ ∈ μ.ϵ
    plot!(range(0, 1, 100), μϵ)
end
plot!()

gh = semidiscretize_boundary(solver, g)
μh = semidiscretize_source(solver, μ)

# measurements2 = measure_forward(solver, (0, 1), 100, gh, μh) #dont use!!
measurementsx2 = measure_adjoint(solver, X, Ω, physics, (0, 1), 100, gh, μh)

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


z_coords = range(0, 1, length=20)
x_coords = range(-1, 1, length=40)

e_x = eval_space(solver.U, Point.(z_coords, x_coords'))
sol = zeros(length(z_coords), length(x_coords))
n_basis = number_of_basis_functions(solver)
e_Ω_p = spzeros(n_basis.Ω.p)
e_Ω_p = gh.Ω[1].p
e_Ω_m = spzeros(n_basis.Ω.m)
e_Ω_p[1] = 1.0

@gif for k in 100:-1:1
    @show k
    for (i, x_) in enumerate(z_coords)
        for (j, y_) in enumerate(x_coords)
            e_x_p, e_x_m = e_x[i, j]
            full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
            sol[i, j] = dot(ψs[k], full_basis)
        end
    end
    contourf(x_coords, z_coords, reverse(sol, dims=2), linewidth=0.0, levels=10, clims=(-0.2, max(2, maximum(sol))), aspect_ratio=:equal)
    #contourf(x_coords, z_coords, reverse(sol, dims=2), linewidth=0.0, levels=10, aspect_ratio=:equal)
    #plot(x_coords, -sol[1, :])
    title!("energy: $(round(ϵs[k], digits=2))")
end fps=3
