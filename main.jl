using Revise
using EPMAfem
using Distributions
using SpecialFunctions
using EPMAfem.Gridap
using Plots
using LinearAlgebra
using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using LaTeXStrings

datapath = joinpath(@__DIR__, "scripts_2D_comparison_convergence/figures/")

struct TestEquations <: EPMAfem.AbstractPNEquations end
EPMAfem.number_of_elements(::TestEquations) = 2
EPMAfem.number_of_scatterings(::TestEquations) = 1
EPMAfem.stopping_power(::TestEquations, e, ϵ) = exp(-ϵ)
EPMAfem.absorption_coefficient(eq::TestEquations, e, ϵ) = e == 1 ? 10.0 : 40.0
EPMAfem.scattering_coefficient(eq::TestEquations, e, i, ϵ) = e == 1 ? 10.0 : 40.0
function EPMAfem.mass_concentrations(::TestEquations, e, x)
    if x[2] < 0.0
        return e == 1 ? 1.0 : 0.0
    else
        return e == 2 ? 1.0 : 0.0
    end
end
# function EPMAfem.mass_concentrations(::TestEquations, e, x)
#     if sqrt((x[1] + 0.3)^2 + (x[2] - 0.1)^2) < 0.2
#         return e == 1 ? 1.0 : 0.0
#     else
#         return e == 2 ? 1.0 : 0.0
#     end
# end

vmf_normalization(p, κ) = κ^(p/2-1)/((2π)^(p/2)*besseli(p/2-1, κ))
EPMAfem.scattering_kernel(::TestEquations, e, i) = μ -> vmf_normalization(2, 5.0)*exp(5.0*μ)

struct TestExcitations end
EPMAfem.number_of_beam_energies(eq::TestExcitations) = 1
EPMAfem.number_of_beam_positions(eq::TestExcitations) = 20
EPMAfem.number_of_beam_directions(eq::TestExcitations) = 1

function EPMAfem.beam_space_distribution(::TestExcitations, i, (z, x, y)) # y is unused, z is 0
    μ_x = range(-0.5, 0.5, length=20)[i]
    # return isapprox(0.0, z, atol=1e-12)*pdf(Uniform(μ_x-0.025/2, μ_x+0.025/2), x)
    return isapprox(0.0, z, atol=1e-12)*pdf(Normal(μ_x, 0.025), x)
end

function EPMAfem.beam_direction_distribution(::TestExcitations, i, Ω)
    # HACK! smuggle the 2 in here
    # HACK! divide by n*Omega to compensate
    return 2.0*pdf(VonMisesFisher([-1.0, 0.0], 20.0), [Ω...]) / abs(dot([-1.0, 0.0], Ω))
end

function EPMAfem.beam_energy_distribution(::TestExcitations, i, ϵ)
    return pdf(Uniform(0.8, 1.0), ϵ)
end

struct TestExtractions end
EPMAfem.number_of_extractions(eq::TestExtractions) = 2
function EPMAfem.extraction_space_distribution(eq::TestExtractions, i, x)
    # if i == 1
    #     return sqrt((x[1] + 0.3)^2 + (x[2] - 0.1)^2) < 0.2 ? 1.0/(π*0.2^2) : 0.0
    # else
    #     @assert i == 2
    #     return sqrt((x[1] + 0.3)^2 + (x[2] - 0.1)^2) < 0.2 ? 0.0 : 1.0/(4.0 - (π*0.2^2))
    # end
    if i == 1
        return x[2] < 0.0 ? 1.0/2.0 : 0.0
    else
        @assert i == 2
        return x[2] < 0.0 ? 0.0 : 1.0/2.0
    end
end
function EPMAfem.extraction_direction_distribution(eq::TestExtractions, i, Ω)
    return 1/(2π)
end
function EPMAfem.extraction_energy_distribution(eq::TestExtractions, i, ϵ)
    return 1.0
end

# forward_plots
energy_model = range(0, 1, 1200)
energy_model = range(0, 1, 100)
equations = TestEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.0, 0.0, -2.0, 2.0), (160, 640)))
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.0, 0.0, -2.0, 2.0), (80, 320)))
direction_model = EPMAfem.SphericalHarmonicsModels.EOCircularHarmonicsModel(27, 2, :EO)
direction_model = EPMAfem.SphericalHarmonicsModels.EOCircularHarmonicsModel(15, 2, :EO)
model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)

problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())
rhs = EPMAfem.discretize_rhs(TestExcitations(), model, EPMAfem.cpu())
extr = EPMAfem.discretize_extraction(TestExtractions(), model, EPMAfem.cpu(); updatable=false)
system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
probe = EPMAfem.PNProbe(model, EPMAfem.cpu(); ϵ=ϵ->1.0, Ω=Ω->1/(2π))
viz_interpolable = EPMAfem.interpolable(probe, system*(rhs[1]+rhs[end]))

begin
    heatmap(-2.0:0.005:2, -1:0.005:0, (x, z) -> viz_interpolable(VectorValue(z, x)), aspect_ratio=:equal, colorbar=true, cmap=reverse(cgrad(:roma)))
    contour!(-2.0:0.005:2, -1:0.005:0, (x, z) -> viz_interpolable(VectorValue(z, x)), aspect_ratio=:equal, colorbar=false, color=:white, linewidth=0.5, levels=[0.2, 0.4, 0.8, 1.6, 3.2])
    vline!([0], color=:gray, ls=:dash, label=nothing)
    for (i, μ_x) in enumerate(range(-0.5, 0.5, 20))
        if i != 1 && i != 20
            plot!([μ_x, μ_x], [-0.02, 0.02], color=:gray, label=nothing)
        else
            plot!([μ_x, μ_x], [-0.02, 0.02], color=:black, label=nothing)
            plot!([μ_x - 2*0.025, μ_x + 2*0.025], [0, 0], linewidth=1, color=:black, label=nothing)
        end
    end
    plot!([], [], color=:black, label=L"beam $\mu \pm 2\sigma$", marker=:vline, legend=:bottomright)
    xlims!(-2.1, 2.1) #interesting region
    ylims!(-1.1, 0.1)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(datapath, "epma_forward_det.png"))
    plot!(colorbar=true)
    savefig(joinpath(datapath, "epma_forward_det_colorbar.png"))
end

adjoint_probe = EPMAfem.PNProbe(model, EPMAfem.cpu(); ϵ=ϵ->pdf(Uniform(0.8, 1.0), ϵ), Ω=Ω->1.0)
adjoint_viz_interpolable = EPMAfem.interpolable(adjoint_probe, extr[1]*system)
begin
    heatmap(-2.0:0.005:2, -1:0.005:0, (x, z) -> -adjoint_viz_interpolable(VectorValue(z, x)), aspect_ratio=:equal, colorbar=true, cmap=reverse(cgrad(:roma)))
    contour!(-2.0:0.005:2, -1:0.005:0, (x, z) -> -adjoint_viz_interpolable(VectorValue(z, x)), aspect_ratio=:equal, colorbar=false, color=:white, linewidth=0.5, levels=10)
    vline!([0], color=:gray, ls=:dash, label=nothing)
    for (i, μ_x) in enumerate(range(-0.5, 0.5, 20))
        if i != 1 && i != 20
            plot!([μ_x, μ_x], [-0.02, 0.02], color=:gray, label=nothing)
        else
            plot!([μ_x, μ_x], [-0.02, 0.02], color=:black, label=nothing)
            plot!([μ_x - 2*0.025, μ_x + 2*0.025], [0, 0], linewidth=1, color=:black, label=nothing)
        end
    end
    plot!([], [], color=:black, label=L"beam $\mu \pm 2\sigma$", marker=:vline, legend=:bottomright)
    xlims!(-2.1, 2.1) #interesting region
    ylims!(-1.1, 0.1)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(datapath, "epma_adjoint_det.png"))
end


# dig into boundary 
probe = EPMAfem.PNProbe(model, EPMAfem.cpu(); ϵ = ϵ -> 1.0)
test = probe(extr[1]*system)
serialize(joinpath(datapath, "adjoint_solution.jls"), test)

begin
    # plot([-2, -2, 2, 2, -2], [-1, 0, 0, -1, -1], aspect_ratio=:equal)
    plot(aspect_ratio=:equal)
    test_mean = EPMAfem.SphericalHarmonicsModels.eval_basis(direction_model, Ω -> 1.0)
    interp = EPMAfem.SpaceModels.interpolable((p=test.p*test_mean.p, m=test.m*test_mean.m), space_model)
    heatmap!(-0.28:0.001:0.28, -0.25:0.001:0, (x, z) -> interp(VectorValue(z, x)), colorbar=false, cmap=:roma)
    hline!([0], color=:black, label=nothing)
    plot!([0, 0], [-0.25, 0], color=:gray, ls=:dash, label=nothing)
    for (x_, x_pos, sb) in [(-0.25, 0.0436Plots.w, 2), (-0.125, 0.218Plots.w, 3), (0.0, 0.391Plots.w, 4), (0.125, 0.564Plots.w, 5), (0.25, 0.737Plots.w, 6)]
        x_val = VectorValue(0.0, x_)
        x_eval = EPMAfem.SpaceModels.eval_basis(space_model, x_val)

        θs = range(-π, π, 500)
        evals = zeros(length(θs))
        for (i, θ) in enumerate(θs)
            Ω_eval = EPMAfem.SphericalHarmonicsModels.eval_basis(direction_model, VectorValue(cos(θ), sin(-θ)))
            evals[i] = -(dot(x_eval.p, test.p, Ω_eval.p) + dot(x_eval.m, test.m, Ω_eval.m))
        end
        plot!(θs .+ π/2, evals, proj = :polar, label=nothing, inset=bbox(x_pos, 0.126, 0.3, 0.3), subplot=sb, xaxis=false, yticks=[], background_color = :transparent, color=sb, ylims=(0, 0.05))
        scatter!([0], [0], subplot=sb, label=nothing, marker=:x, color=:black)
    end
    xticks!([-0.25, -0.125, 0, 0.125, 0.25], subplot=1)
    xlims!(-0.31, 0.31, subplot=1)
    ylims!(-0.3, 0.1, subplot=1)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(datapath, "epma_adjoint_det_detector_angle_distribution.png"))
end

let
    eval_x_pos = range(-0.250, 0.250, 250)
    eval_θ_pos = range(-π/2, 3π/2, 250)
    evals = zeros(length(eval_x_pos), length(eval_θ_pos))
    for (j, x_) in enumerate(eval_x_pos)
        x_val = VectorValue(0.0, x_)
        x_eval = EPMAfem.SpaceModels.eval_basis(space_model, x_val)
        for (i, θ) in enumerate(eval_θ_pos)
            θ = θ - π/2
            Ω_eval = EPMAfem.SphericalHarmonicsModels.eval_basis(direction_model, VectorValue(cos(θ), sin(-θ)))
            evals[j, i] = -(dot(x_eval.p, test.p, Ω_eval.p) + dot(x_eval.m, test.m, Ω_eval.m))
        end
    end
    heatmap(eval_x_pos, eval_θ_pos, transpose(evals), colorbar=false)
    xlabel!(L"beam $x$")
    ylabel!(L"beam $\theta$")
    yticks!([-π/2, 0, π/2, π, 3π/2], [L"\frac{-\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi", L"\frac{3\pi}{2}"])
    xticks!([-0.25, -0.125, 0, 0.125, 0.25])
    xlims!(-0.31, 0.31)
    ylims!(-π/2-0.3, 3π/2+0.3)
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    savefig(joinpath(datapath, "epma_adjoint_det_detector_angle_distribution_full.png"))
end



upper_boundary = (p = zeros(321, 15), m = zeros(320, 16))
for i in 1:15
    upper_boundary.p[:, i] = reshape(test.p[:, i], (81, 321))[1, :]
end
for i in 1:16
    upper_boundary.m[:, i] = reshape(test.m[:, i], (80, 320))[1, :]
end
heatmap(upper_boundary)


# backscattered electrons computation
bse_extr = let
    T = EPMAfem.base_type(EPMAfem.cpu())
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels
    μϵ = Vector{T}([1.0 for ϵ ∈ EPMAfem.energy_model(model)])
    n = VectorValue(1.0, 0.0)
    # uniform over outgoing half-sphere -> 1/π
    μΩ = (p=SH.assemble_linear(SH.∫S²_hv(Ω -> 1/π*abs(dot(Ω, n))), direction_model, SH.plus(direction_model)), m=zeros(EPMAfem.n_basis(model).nΩ.m)) |> EPMAfem.cpu()
    # uniform over (-2, 2) -> 1/4 probability
    μx = (p=SM.assemble_linear(SM.∫∂R_ngv{EPMAfem.Dimensions.Z}(x -> 1/4*isapprox(x[1], 0.0, atol=1e-12)), space_model, SM.plus(space_model)), m=zeros(EPMAfem.n_basis(model).nx.m)) |> EPMAfem.cpu()
    [EPMAfem.Rank1DiscretePNVector(true, model, EPMAfem.cpu(), μϵ, μx, μΩ)]
end

test = (bse_extr*system)*rhs
# the probability density function of the bse "detector" is uniform -> the correction factor is easy: 1/(4π)
bse_corrected = test .- (1/(4π))
bse_corrected = [0.011964804866893636;;; 0.011998057495017478;;; 0.012070082198462465;;; 0.012201271965407767;;; 0.012420764714362709;;; 0.012772270393880028;;; 0.013326228370498558;;; 0.014210756963143834;;; 0.01572778103164313;;; 0.019513716398069786;;; 0.02862243095247441;;; 0.033740457865038445;;; 0.03558438322056208;;; 0.036440623102791614;;; 0.03686759603630527;;; 0.03708316021522083;;; 0.03718799137792457;;; 0.03723303610410625;;; 0.037245923644862386;;; 0.037242387053672846;;;;]


# backscattered electron visualizations
probe = EPMAfem.PNProbe(model, EPMAfem.cpu(); ϵ=ϵ->EPMAfem.beam_energy_distribution(TestExcitations(), 1, ϵ), Ω=Ω->1/(2π))
viz_bse = EPMAfem.interpolable(probe, bse_extr[1]*system)
begin
    heatmap(-2.0:0.005:2, -1:0.005:0, (x, z) -> -viz_bse(VectorValue(z, x)), aspect_ratio=:equal, colorbar=true, cmap=reverse(cgrad(:roma)))
    contour!(-2.0:0.005:2, -1:0.005:0, (x, z) -> -viz_bse(VectorValue(z, x)), aspect_ratio=:equal, colorbar=false, color=:white, linewidth=0.5, levels=10)
    vline!([0], color=:gray, ls=:dash, label=nothing)
    for (i, μ_x) in enumerate(range(-0.5, 0.5, 20))
        if i != 1 && i != 20
            plot!([μ_x, μ_x], [-0.02, 0.02], color=:gray, label=nothing)
        else
            plot!([μ_x, μ_x], [-0.02, 0.02], color=:black, label=nothing)
            plot!([μ_x - 2*0.025, μ_x + 2*0.025], [0, 0], linewidth=1, color=:black, label=nothing)
        end
    end
    plot!([], [], color=:black, label=L"beam $\mu \pm 2\sigma$", marker=:vline, legend=:bottomright)
    xlims!(-2.1, 2.1) #interesting region
    ylims!(-1.1, 0.1)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(datapath, "epma_adjoint_det_bse.png"))
end


# m = SH.assemble_linear(SH.∫S²_hv(Ω->extraction_direction_distribution(pn_ex, i, Ω)), direction_mdl, SH.minus(direction_mdl))) |> arch for i in 1:number_of_extractions(pn_ex)]

#     if updatable
#         ρ_proj = SM.assemble_bilinear(SM.∫R_uv, space_mdl, SM.minus(space_mdl), SM.plus(space_mdl))
#         ρs = discretize_mass_concentrations(pn_ex.pn_eq, mdl)
#         n_parameters = (number_of_elements(pn_ex.pn_eq), n_basis(mdl).nx.m)
#         return [UpdatableRank1DiscretePNVector(Rank1DiscretePNVector(true, mdl, arch, μϵs[i], ρ_proj*@view(ρs[i, :]) |> arch, μΩps[i]), EPMAfem.PNNoAbsorption(mdl, arch, ρ_proj, i), n_parameters) for i in 1:number_of_extractions(pn_ex)]
#     else
#         μxps = [(p=SM.assemble_linear(SM.∫R_μv(x -> extraction_space_distribution(pn_ex, i, x)), space_mdl, SM.plus(space_mdl)),
#                     m = SM.assemble_linear(SM.∫R_μv(x -> extraction_space_distribution(pn_ex, i, x)), space_mdl, SM.minus(space_mdl))) |> arch for i in 1:number_of_extractions(pn_ex)]
#         return [Rank1DiscretePNVector(true, mdl, arch, μϵs[i], μxps[i], μΩps[i]) for i in 1:number_of_extractions(pn_ex)]
#     end
# end

# measurement computations
using ProgressLogging
measurements = Dict()
@progress for n_ϵ in [38, 75, 150, 300, 600], n_x in [(10, 40), (20, 80), (40, 160), (80, 320), (160, 640)], n_Ω in [3, 9, 15, 21, 27]
    @show n_ϵ, n_x, n_Ω
    equations = TestEquations()
    energy_model = range(0, 1, n_ϵ)
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.0, 0.0, -2.0, 2.0), n_x))
    direction_model = EPMAfem.SphericalHarmonicsModels.EOCircularHarmonicsModel(n_Ω, 2, :EO)
    model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)

    problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())
    rhs = EPMAfem.discretize_rhs(TestExcitations(), model, EPMAfem.cpu())
    extr = EPMAfem.discretize_extraction(TestExtractions(), model, EPMAfem.cpu(); updatable=false)
    system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
    measurements[(n_ϵ, n_x, n_Ω)] = copy(extr*system*rhs)
end

using Serialization
serialize(joinpath(datapath, "measurements.jls"), measurements)
measurements = deserialize(joinpath(datapath, "measurements.jls"))

# solution computations
using ProgressLogging
solutions = Dict()
# @progress for (n_ϵ, n_x, n_Ω) in zip([38, 75, 150, 300, 600], [(10, 40), (20, 80), (40, 160), (80, 320), (160, 640)], [3, 9, 15, 21, 27])
@progress for (n_ϵ, n_x, n_Ω) in zip([300], [(80, 320)], [9])
    @show n_ϵ, n_x, n_Ω
    equations = TestEquations()
    energy_model = range(0, 1, n_ϵ)
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.0, 0.0, -2.0, 2.0), n_x))
    direction_model = EPMAfem.SphericalHarmonicsModels.EOCircularHarmonicsModel(n_Ω, 2, :EO)
    model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)

    problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())
    rhs = EPMAfem.discretize_rhs(TestExcitations(), model, EPMAfem.cpu())
    # extr = EPMAfem.discretize_extraction(TestExtractions(), model, EPMAfem.cpu(); updatable=false)
    system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
    sol = system * rhs[1, 10, 1]
    probe = EPMAfem.PNProbe(model, EPMAfem.cpu(); ϵ=ϵ->1.0, Ω=Ω->1/(2π))
    solutions[(n_ϵ, n_x, n_Ω)] = EPMAfem.interpolable(probe, sol)
end

let
    n_ϵ, n_x, n_Ω = 300, (80, 320), 9
    p = heatmap(-0.5:0.001:0.5, -0.5:0.001:0, (x, z) -> solutions[(n_ϵ, n_x, n_Ω)](VectorValue(z, x)) |> negtonan, aspect_ratio=:equal, clims=(-0.0, 2.5), cmap=reverse(cgrad(:roma)))
    contour!(-0.5:0.001:0.5, -0.5:0.001:0, (x, z) -> solutions[(n_ϵ, n_x, n_Ω)](VectorValue(z, x)), aspect_ratio=:equal, clims=(-0.5, 2.5), color=:white, levels=10)
    plot!(colorbar=false)
    μ_x = range(-0.5, 0.5, 20)[10]
    plot!([μ_x, μ_x], [-0.02, 0.02], color=:black, label=nothing)
    plot!([μ_x - 2*0.025, μ_x + 2*0.025], [0, 0], linewidth=1, color=:black, label=nothing)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    xlims!(-0.51, 0.51)
    ylims!(-0.51, 0.01)
end

serialize(joinpath(datapath, "solutions.jls"), solutions)
solutions = deserialize(joinpath(datapath, "measurements.jls"))

begin
    negtonan(x) = x < 0.0 ? NaN : x
    for (i, (n_ϵ, n_x, n_Ω)) in enumerate(zip([38, 75, 150, 300, 600], [(10, 40), (20, 80), (40, 160), (80, 320), (160, 640)], [3, 9, 15, 21, 27]))
        p = heatmap(-0.5:0.001:0.5, -0.5:0.001:0, (x, z) -> solutions[(n_ϵ, n_x, n_Ω)](VectorValue(z, x)) |> negtonan, aspect_ratio=:equal, clims=(-0.0, 2.5), cmap=reverse(cgrad(:roma)))
        contour!(-0.5:0.001:0.5, -0.5:0.001:0, (x, z) -> solutions[(n_ϵ, n_x, n_Ω)](VectorValue(z, x)), aspect_ratio=:equal, clims=(-0.5, 2.5), color=:white, levels=10)
        plot!(colorbar=false)
        μ_x = range(-0.5, 0.5, 20)[10]
        plot!([μ_x, μ_x], [-0.02, 0.02], color=:black, label=nothing)
        plot!([μ_x - 2*0.025, μ_x + 2*0.025], [0, 0], linewidth=1, color=:black, label=nothing)
        plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
        xlims!(-0.51, 0.51)
        ylims!(-0.51, 0.01)
        savefig(joinpath(datapath, "sol_$i.png"))
    end
end

using Serialization
serialize(joinpath(datapath, "measurements.jls"), measurements)
measurements = deserialize(joinpath(datapath, "measurements.jls"))

function n_dof(n_ϵ, n_x, n_Ω)
    energy_model = range(0, 1, n_ϵ)
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.0, 0.0, -2.0, 2.0), n_x))
    direction_model = EPMAfem.SphericalHarmonicsModels.EOCircularHarmonicsModel(n_Ω, 2, :EO)
    model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
    n_b = model.number_of_basis_functions
    return n_b.nϵ * (n_b.nx.p * n_b.nΩ.p + n_b.nx.m * n_b.nΩ.m)
end

m_ref = measurements[(600, (160, 640), 27)]

function compute_errors(comp, ref, e=:L1)
    ns = zeros(length(comp))
    err = zeros(length(comp))
    for (i, (n, m)) in enumerate(comp)
        ns[i] = n_dof(n...)
        if e == :L1
            err[i] = sum(abs.(
                (m .- ref) ./ ref))/length(m)
        elseif e == :L2
            err[i] = sqrt(sum(((m .- ref)./ref).^2))
        elseif e == :Linf
            err[i] = maximum(abs.(m .- ref) ./ ref)
        end
    end
    perm = sortperm(ns)

    return ns[perm], err[perm]
end

comp_Ω = [(n, m) for (n, m) in measurements if n[1] == 600 && n[2] == (160, 640) && n[3] != 27]
comp_ϵ = [(n, m) for (n, m) in measurements if n[1] != 600 && n[2] == (160, 640) && n[3] == 27]
comp_x = [(n, m) for (n, m) in measurements if n[1] == 600 && n[2] != (160, 640) && n[3] == 27]
comp_all = filter(((n, m), ) -> n ∈ [(38, (10, 40), 3), (75, (20, 80), 9), (150, (40, 160), 15), (300, (80, 320), 21)], measurements)
comp_noref = filter(((n, m), ) -> !(n ∈ [(600, (160, 640), 27)]), measurements)

begin
    gr()
    e = :L1
    plot()
    # scatter!(compute_errors(comp_noref, m_ref, e)..., xaxis=:log, yaxis=:log, color=:lightgray, label=nothing, alpha=0.1)
    plot!(compute_errors(comp_all, m_ref, e)..., xaxis=:log, yaxis=:log, label=L"var. $(n_\epsilon, n_x, n_\Omega)$", marker=:o, color=1)
    plot!(compute_errors(comp_ϵ, m_ref, e)..., xaxis=:log, yaxis=:log, label=L"var. $n_\epsilon$", marker=:o, color=2)
    plot!(compute_errors(comp_x, m_ref, e)..., xaxis=:log, yaxis=:log, label=L"var. $n_x$", marker=:o, color=3)
    plot!(compute_errors(comp_Ω, m_ref, e)..., xaxis=:log, yaxis=:log, label=L"var. $n_\Omega$", marker=:o, color=4)
    plot!([8e6, 1e9], 1e5.*[1/1e7, 1/1e9], color=:lightgray, ls=:dash, label=nothing)
    annotate!(1e8, 4e-4, Plots.text(L"\mathcal{O}(1/N)", 7))
    plot!([5e8, 5e9], 1e16.*([1/5e8, 1/5e9]).^2, color=:lightgray, ls=:dashdot, label=nothing)
    annotate!(3e9, 1e-2, Plots.text(L"\mathcal{O}(1/N^2)", 7))
    yticks!([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    xlabel!("DOF")
    ylabel!("mean rel. error")
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000, legend=:bottomleft)
    blue = palette(:default)[1]
    annotate!(2.9e5, 3.2e-2, Plots.text(L"38, (10, 40), P_3", 5, color=blue))
    annotate!(1e6, 1.1e-2, Plots.text(L"75, (20, 80), P_9", 5, color=blue))
    annotate!(1.2e7, 4.5e-3, Plots.text(L"150, (40, 160), P_{15}", 5, color=blue))
    annotate!(1.3e8, 1.4e-3, Plots.text(L"300, (80, 320), P_{21}", 5, color=blue))
    orange = palette(:default)[2]
    annotate!(1.6e8, 2.2e-2, Plots.text(L"38", 5, color=orange))
    annotate!(3.1e8, 1.2e-2, Plots.text(L"75", 5, color=orange))
    annotate!(6.1e8, 5e-3, Plots.text(L"150", 5, color=orange))
    annotate!(1.2e9, 1.7e-3, Plots.text(L"300", 5, color=orange))

    green = palette(:default)[3]
    annotate!(2.5e7, 1.7e-2, Plots.text(L"(10, 40)", 5, color=green))
    annotate!(9.9e7, 4.5e-3, Plots.text(L"(20, 80)", 5, color=green))
    annotate!(4e8, 1e-3, Plots.text(L"(40, 160)", 5, color=green))
    annotate!(4.5e8, 2e-4, Plots.text(L"(80, 320)", 5, color=green))

    purple = palette(:default)[4]
    annotate!(6e8, 1.7e-2, Plots.text(L"P_3", 5, color=purple))
    annotate!(1.6e9, 2.3e-4, Plots.text(L"P_9", 5, color=purple))
    annotate!(2.7e9, 4.5e-5, Plots.text(L"P_{15}", 5, color=purple))
    annotate!(3.6e9, 1.3e-5, Plots.text(L"P_{21}", 5, color=purple))
    ylims!(5e-6, 5e-2)
    xlims!(9e4, 5e9)
    savefig(joinpath(datapath, "rel_L1_error.png"))
end

function compute_error(comp, ref)
    ns = zeros(length(comp))
    err = zeros(length(comp))
    for (i, (n, m)) in enumerate(comp)
        ns[i] = n_dof(n...)
        err[i] = abs((m[1, 1, 10, 1] - ref[1, 1, 10, 1])/ref[1, 1, 10, 1])
    end
    perm = sortperm(ns)
    return ns[perm], err[perm]
end

begin
    gr()
    e = :L1
    plot()
    # scatter!(compute_error(comp_noref, m_ref)..., xaxis=:log, yaxis=:log, color=:lightgray, label=nothing, alpha=0.1)
    plot!(compute_error(comp_all, m_ref)..., xaxis=:log, yaxis=:log, label=L"var. $(n_\epsilon, n_x, n_\Omega)$", marker=:o, color=1)
    plot!(compute_error(comp_Ω, m_ref)..., xaxis=:log, yaxis=:log, label=L"var. $n_\Omega$", marker=:o, color=2)
    plot!(compute_error(comp_ϵ, m_ref)..., xaxis=:log, yaxis=:log, label=L"var. $n_\epsilon$", marker=:o, color=3)
    plot!(compute_error(comp_x, m_ref)..., xaxis=:log, yaxis=:log, label=L"var. $n_x$", marker=:o, color=4)
    # plot!(N_trajectories, abs.((Y_means_adj .- m_ref[1, 1, 10, 1])./m_ref[1, 1, 10, 1]), marker=:o, color=6)
    plot!([8e6, 1e9], 1e5.*[1/1e7, 1/1e9], color=:lightgray, ls=:dash, label=nothing)
    annotate!(1e8, 4e-4, Plots.text(L"\mathcal{O}(1/N)", 7))
    plot!([5e8, 5e9], 1e16.*([1/5e8, 1/5e9]).^2, color=:lightgray, ls=:dashdot, label=nothing)
    annotate!(3e9, 1e-2, Plots.text(L"\mathcal{O}(1/N^2)", 7))
    yticks!([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    xlabel!("DOF")
    ylabel!("rel. abs error")
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000, legend=:bottomleft)
    blue = palette(:default)[1]
    annotate!(3e5, 3.7e-2, Plots.text(L"38, (10, 40), P_3", 5, color=blue))
    annotate!(1e6, 2e-2, Plots.text(L"75, (20, 80), P_9", 5, color=blue))
    annotate!(1.2e7, 7e-3, Plots.text(L"150, (40, 160), P_{15}", 5, color=blue))
    annotate!(2e8, 1.5e-3, Plots.text(L"300, (80, 320), P_{21}", 5, color=blue))
    orange = palette(:default)[2]
    annotate!(3e8, 1e-2, Plots.text(L"P_3", 5, color=orange))
    annotate!(1.6e9, 2.5e-4, Plots.text(L"P_9", 5, color=orange))
    annotate!(2.8e9, 5.5e-5, Plots.text(L"P_{15}", 5, color=orange))
    annotate!(3.9e9, 1.9e-5, Plots.text(L"P_{21}", 5, color=orange))
    green = palette(:default)[3]
    annotate!(1.6e8, 2.2e-2, Plots.text(L"38", 5, color=green))
    annotate!(5.5e8, 1.7e-2, Plots.text(L"75", 5, color=green))
    annotate!(6.8e8, 4.5e-3, Plots.text(L"150", 5, color=green))
    annotate!(1.2e9, 1.7e-3, Plots.text(L"300", 5, color=green))
    purple = palette(:default)[4]
    annotate!(0.8e7, 5.5e-2, Plots.text(L"(10, 40)", 5, color=purple))
    annotate!(3e7, 1.5e-2, Plots.text(L"(20, 80)", 5, color=purple))
    annotate!(1.1e8, 2.5e-3, Plots.text(L"(40, 160)", 5, color=purple))
    annotate!(4.5e8, 3.5e-4, Plots.text(L"(80, 320)", 5, color=purple))
    ylims!(1.5e-5, 10e-2)
    xlims!(9e4, 5e9)
    # savefig("")
    # savefig(joinpath(@__DIR__, "scripts_2D_comparison_convergence/figures/rel_abs_error.png"))
end


rel1 = measurements[(600, (160, 640), 27)][1, 1, :, 1]
rel2 = measurements[(600, (160, 640), 27)][2, 1, :, 1]
process1(m) = abs.((m .- rel1)./rel1)
process2(m) = abs.((m .- rel2)./rel2)

# ϵ absolute
begin
    gr()
    plot()
    for (i, n_ϵ) in enumerate([38, 75, 150, 300, 600])
        ref = n_ϵ == 600
        ls = ref ? :solid : [:dashdot, :dot, :dash, :solid][i]
        plot!(range(-0.5, 0.5, 20), measurements[(n_ϵ, (160, 640), 27)][1, 1, :, 1], color= ref ? :black : 1, label=nothing, marker= ref ? false : false, ls=ls)
        plot!(range(-0.5, 0.5, 20), measurements[(n_ϵ, (160, 640), 27)][2, 1, :, 1], color= ref ? :black : 2, label=nothing, marker= ref ? false : false, ls=ls)
        if !ref
            plot!([], [], marker=false, color=:gray, ls=ls, label=L"n_ϵ = %$n_ϵ")
        else
            plot!([], [], color=:black, label="reference")
        end
    end
    plot!(legend=:left, size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    xlabel!("beam position")
    savefig(joinpath(datapath, "absolute_measurements_energy.png"))
end
# ϵ relative
begin
    gr()
    plot()
    for (i, n_ϵ) in enumerate([38, 75, 150, 300, 600])
        ref = n_ϵ == 600
        ls = ref ? :solid : [:dashdot, :dot, :dash, :solid][i]
        plot!(range(-0.5, 0.5, 20), measurements[(n_ϵ, (160, 640), 27)][1, 1, :, 1] |> process1, color= ref ? :black : 1, label=nothing, marker= ref ? false : false, ls=ls)
        plot!(range(-0.5, 0.5, 20), measurements[(n_ϵ, (160, 640), 27)][2, 1, :, 1] |> process2, color= ref ? :black : 2, label=nothing, marker= ref ? false : false, ls=ls)
        if !ref
            plot!([], [], marker=false, color=:gray, ls=ls, label=L"n_ϵ = %$n_ϵ")
        else
            # plot!([], [], color=:black, label="reference")
        end
    end
    plot!(legend=:topright, size=(400, 300), fontfamily="Computer Modern", dpi=1000, yaxis=:log)
    xlabel!("beam position")
    ylabel!("rel. abs. error")
    ylims!(2e-4, 5e-2)
    savefig(joinpath(datapath, "relative_measurements_energy.png"))
end

# x absolute
begin
    gr()
    plot()
    # plot!([], [], inset=(1, bbox(0.1, 0.3, 0.35, 0.35)), subplot=2, label=nothing)
    for (i, n_x) in enumerate([(10, 40), (20, 80), (40, 160), (80, 320), (160, 640)])
        ref = n_x == (160, 640)
        ls = ref ? :solid : [:dashdot, :dot, :dash, :solid][i]
        plot!(range(-0.5, 0.5, 20), measurements[(600, n_x, 27)][1, 1, :, 1], color= ref ? :black : 1, label=nothing, marker= ref ? false : false, ls=ls)
        # plot!(range(-0.5, 0.5, 20), measurements[(600, n_x, 27)][1, 1, :, 1], color= ref ? :black : 1, label=nothing, marker= ref ? false : false, ls=ls, subplot=2)
        plot!(range(-0.5, 0.5, 20), measurements[(600, n_x, 27)][2, 1, :, 1], color= ref ? :black : 2, label=nothing, marker= ref ? false : false, ls=ls)
        # plot!(range(-0.5, 0.5, 20), measurements[(600, n_x, 27)][2, 1, :, 1], color= ref ? :black : 2, label=nothing, marker= ref ? false : false, ls=ls, subplot=2)
        if !ref
            plot!([], [], marker=false, color=:gray, ls=ls, label=L"n_x = %$n_x")
        else
            plot!([], [], color=:black, label="reference")
        end
    end
    plot!(legend=:left, size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    # xlims!(subplot=2, -0.2, 0.0)
    # ylims!(subplot=2, 0.05, 0.08)
    xlabel!("beam position", subplot=1)
    savefig(joinpath(datapath, "absolute_measurements_position.png"))
end
# x relative
begin
    gr()
    plot()
    for (i, n_x) in enumerate([(10, 40), (20, 80), (40, 160), (80, 320), (160, 640)])
        ref = n_x == (160, 640)
        ls = ref ? :solid : [:dashdot, :dot, :dash, :solid][i]
        plot!(range(-0.5, 0.5, 20), measurements[(600, n_x, 27)][1, 1, :, 1] |> process1, color= ref ? :black : 1, label=nothing, marker= ref ? false : false, ls=ls)
        plot!(range(-0.5, 0.5, 20), measurements[(600, n_x, 27)][2, 1, :, 1] |> process2, color= ref ? :black : 2, label=nothing, marker= ref ? false : false, ls=ls)
        if !ref
            plot!([], [], marker=false, color=:gray, ls=ls, label=L"n_x = %$n_x")
        else
            plot!([], [], color=:black, label="reference")
        end
    end
    plot!(legend=:topright, size=(400, 300), fontfamily="Computer Modern", dpi=1000, yaxis=:log)
    xlabel!("beam position")
    ylims!(1e-6, 1e-1)
    ylabel!("rel. abs. error")
    savefig(joinpath(datapath, "relative_measurements_position.png"))
end

# Ω absolute
begin
    gr()
    plot()
    # plot!([], [], inset=(1, bbox(0.1, 0.3, 0.35, 0.35)), subplot=2, label=nothing)
    for (i, n_Ω) in enumerate([3, 9, 15, 21, 27])
        ref = n_Ω == 27
        ls = ref ? :solid : [:dashdot, :dot, :dash, :solid][i]
        plot!(range(-0.5, 0.5, 20), measurements[(600, (160, 640), n_Ω)][1, 1, :, 1], color= ref ? :black : 1, label=nothing, marker= ref ? false : false, ls=ls)
        # plot!(range(-0.5, 0.5, 20), measurements[(600, (160, 640), n_Ω)][1, 1, :, 1], color= ref ? :black : 1, label=nothing, marker= ref ? false : false, ls=ls, subplot=2)
        plot!(range(-0.5, 0.5, 20), measurements[(600, (160, 640), n_Ω)][2, 1, :, 1], color= ref ? :black : 2, label=nothing, marker= ref ? false : false, ls=ls)
        # plot!(range(-0.5, 0.5, 20), measurements[(600, (160, 640), n_Ω)][2, 1, :, 1], color= ref ? :black : 2, label=nothing, marker= ref ? false : false, ls=ls, subplot=2)
        if !ref
            plot!([], [], marker=false, color=:gray, ls=ls, label=L"n_\Omega = %$n_Ω")
        else
            plot!([], [], color=:black, label="reference")
        end
    end
    xlabel!("beam position", subplot=1)
    # xlims!(subplot=2, -0.027, -0.025)
    # ylims!(subplot=2, 0.02, 0.04)
    plot!(legend=:left, size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(datapath, "absolute_measurements_direction.png"))
end
# Ω relative
begin
    gr()
    plot()
    for (i, n_Ω) in enumerate([3, 9, 15, 21, 27])
        ref = n_Ω == 27
        ls = ref ? :solid : [:dashdot, :dot, :dash, :solid][i]
        plot!(range(-0.5, 0.5, 20), measurements[(600, (160, 640), n_Ω)][1, 1, :, 1] |> process1, color= ref ? :black : 1, label=nothing, marker= ref ? false : false, ls=ls)
        plot!(range(-0.5, 0.5, 20), measurements[(600, (160, 640), n_Ω)][2, 1, :, 1] |> process2, color= ref ? :black : 2, label=nothing, marker= ref ? false : false, ls=ls)
        if !ref
            plot!([], [], marker=false, color=:gray, ls=ls, label=L"n_\Omega = %$n_Ω")
        else
            plot!([], [], color=:black, label="reference")
        end
    end
    plot!(legend=:topright, size=(400, 300), fontfamily="Computer Modern", dpi=1000, yaxis=:log)
    xlabel!("beam position")
    ylims!(1e-6, 1e-1)
    ylabel!("rel. abs. error")
    savefig(joinpath(datapath, "relative_measurements_direction.png"))
end
