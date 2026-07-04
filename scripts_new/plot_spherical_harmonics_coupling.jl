using Revise
using Plots
using EPMAfem
using EPMAfem.SphericalHarmonicsModels
SH = EPMAfem.SphericalHarmonicsModels
using EPMAfem.Gridap
using LinearAlgebra
using SparseArrays
using LaTeXStrings
include("../scripts/grid_gen.jl")
include("../scripts_new/2D_spherical_harmonics.jl")
# sh1 = SH.SphericalHarmonic(1, -1)
using HCubature
using StaticArrays


moms = SH.spherical_harmonics(3, EPMAfem.Dimensions._3D())

even_moments = [m for m in moms if SH.is_even(m)]
odd_moments = [m for m in moms if SH.is_odd(m)]


A = [SH.get_transport_coefficient(m1, m2, EPMAfem.Dimensions.X()) for m1 in [even_moments..., odd_moments...], m2 in [even_moments..., odd_moments...]]


nothing


quad = SH.lebedev_quadrature_max()
quad = f -> hquadrature(θ -> f(Ω2D(θ)), 0, 2π; atol=1e-13, rtol=1e-13)[1]

moms = SH.spherical_harmonics(4)

θ = 0.3
R = TensorValue([cos(θ) sin(θ)
sin(θ) -cos(θ)])

D = [quad(Ω -> m1(R ⋅ Ω)*m2(Ω)) for m1 in moms, m2 in moms]

l = 4
TensorValue([cos(l*θ) sin(l*θ)
sin(l*θ) -cos(l*θ)])

plot()
plot_spherical_harmonic(SphericalHarmonic2D(1, 0); aspect_ratio=:equal)

@gif for θ in 0:0.05:2π
    n = VectorValue(cos(θ), sin(θ))
    l = 1
    sh10 = SphericalHarmonic2D(l, 0)
    sh11 = SphericalHarmonic2D(l, 1)
    plot()
    # plot_spherical_harmonic(sh10; aspect_ratio=:equal)
    w = [sh10(n), sh11(n)] |> normalize
    plot_spherical_harmonic(Ω -> w[1]*sh10(Ω) + w[2]*sh11(Ω); aspect_ratio=:equal)
    plot!([0, n[1]], [0, n[2]], label=nothing)
    xlims!(-1, 1)
    ylims!(-1, 1)
end

plot()
plot_spherical_harmonic(SphericalHarmonic2D(3, 0))
plot(0:0.01:2π, θ -> SphericalHarmonic2D(2, 1)(Ω2D(θ)))


# As = []
# for i in -2:1:2
#     sh = SH.SphericalHarmonic(2, i)
#     A = zeros(3, 3)
#     for i in 1:3, j in 1:3
#         shi = SH.SphericalHarmonic(1, i-2)
#         A[i, j] = quad(Ω -> sh(Ω)*shi(Ω)*Ω[j])
#     end
#     push!(As, A)
# end

# round.(vcat(As...), digits=15) |> unique

# As_ = [round.(A, digits=15) for A in As]

x_position(sh::SH.SphericalHarmonic) = sh.order
y_position(sh::SH.SphericalHarmonic) = -sh.degree
x_position(sh::SphericalHarmonic2D) = sh.order
y_position(sh::SphericalHarmonic2D) = -sh.degree

function flip(Ω::VectorValue, dim)
    if dim == 1
        return VectorValue(-Ω[1], Ω[2], Ω[3])
    elseif dim == 2
        return VectorValue(Ω[1], -Ω[2], Ω[3])
    elseif dim == 3
        return VectorValue(Ω[1], Ω[2], -Ω[3])
    else
        error("dim !∈ (1, 2, 3)")
    end
end

# unstable!
function is_even(sh)
    Ωs = [VectorValue(randn(3)) |> normalize]
    res = sh.(Ωs) .- sh.(.-Ωs)
    res_absmax = maximum(abs.(res))
    return isapprox(res_absmax, 0.0, atol=1e-12, rtol=1e-12)
end

# function is_eveneven(sh::SH.SphericalHarmonic)
#     Ωs = [VectorValue(randn(3)) |> normalize]
#     res1 = sh.(Ωs) .- sh.(flip.(Ωs, 1))
#     res2 = sh.(Ωs) .- sh.(flip.(Ωs, 2))
#     res3 = sh.(Ωs) .- sh.(flip.(Ωs, 3))
#     res_absmax = max(maximum(abs.(res1)), maximum(abs.(res2)), maximum(abs.(res3)))
#     return isapprox(res_absmax, 0.0, atol=1e-12, rtol=1e-12)
# end

function is_odd(sh)
    Ωs = [VectorValue(randn(3)) |> normalize]
    return all(sh.(Ωs) .≈ .-sh.(.-Ωs))
end

# function is_oddodd(sh::SH.SphericalHarmonic)
#     Ωs = [VectorValue(randn(3)) |> normalize]
#     return all(sh.(Ωs) .≈ .-sh.(flip.(Ωs, 1))) && all(sh.(Ωs) .≈ .-sh.(flip.(Ωs, 2))) && all(sh.(Ωs) .≈ .-sh.(flip.(Ωs, 3)))
# end

struct Path
    x0
    x1
end

function Plots.plot!(p::Path; kwargs...)
    plot!([p.x0[1], p.x1[1]], [p.x0[2], p.x1[2]]; kwargs...)
end

function path(sh0, sh1)
    return Path(
        (x_position(sh0), y_position(sh0)),
        (x_position(sh1), y_position(sh1)),
    )
end

N = 4
moms = SH.spherical_harmonics(N, EPMAfem.Dimensions._3D())

m1 = [m for m in moms if SH.is_even(m) && (SH.is_even_in(m, EPMAfem.Dimensions.Z())&& SH.is_even_in(m, EPMAfem.Dimensions.X())&& SH.is_even_in(m, EPMAfem.Dimensions.Y()))]
m3 = [m for m in moms if SH.is_even(m) && !(SH.is_even_in(m, EPMAfem.Dimensions.Z())&& SH.is_even_in(m, EPMAfem.Dimensions.X())&& SH.is_even_in(m, EPMAfem.Dimensions.Y()))]
m4 = [m for m in moms if SH.is_odd(m) && (SH.is_odd_in(m, EPMAfem.Dimensions.Z())&& SH.is_odd_in(m, EPMAfem.Dimensions.X())&& SH.is_odd_in(m, EPMAfem.Dimensions.Y()))]
m2 = [m for m in moms if SH.is_odd(m) && !(SH.is_odd_in(m, EPMAfem.Dimensions.Z())&& SH.is_odd_in(m, EPMAfem.Dimensions.X())&& SH.is_odd_in(m, EPMAfem.Dimensions.Y()))]

moms2 = [m1..., m2..., m3..., m4...]


moms = SH.spherical_harmonics(N)

nx = VectorValue(1.0, 0.0)
ny = VectorValue(0.0, 1.0)
[quad(Ω -> moms[1](Ω)*dot(n, Ω)) for n in [nx, ny]]
[quad(Ω -> moms[1](Ω)*dot(n, Ω)*moms[2](Ω)) for n in [nx, ny]]

[quad(Ω -> *dot(n, Ω)) for n in [nx, ny]]
[quad(Ω -> moms[3](Ω)*dot(n, Ω)) for n in [nx, ny]]
[quad(Ω -> moms[4](Ω)*dot(n, Ω)) for n in [nx, ny]]
[quad(Ω -> moms[5](Ω)*dot(n, Ω)) for n in [nx, ny]]
[quad(Ω -> moms[6](Ω)*dot(n, Ω)) for n in [nx, ny]]
[quad(Ω -> moms[7](Ω)*dot(n, Ω)) for n in [nx, ny]]




quad(Ω -> moms[1](Ω)*moms[2](Ω)*dot(VectorValue(0.0, 1.0), Ω))
quad(Ω -> moms[1](Ω)*moms[3](Ω)*dot(VectorValue(0.0, 1.0), Ω))

[quad(Ω -> m1(Ω)*m2(Ω)*abs(dot(VectorValue(1.0, 0.0), Ω))) for m1 in moms, m2 in moms]

Bx = [SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in [m for m in moms if SH.degree(m) == 3], sh2 in [m for m in moms if SH.degree(m) == 4]]
rank(Bx)


By = [SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in moms, sh2 in moms]
Bz = [SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Y()) for sh1 in moms, sh2 in moms]

Bx2 = [SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in moms2, sh2 in moms2]

eigen(Bx).values



confs = Dict(mom => Dict(EPMAfem.Dimensions.X()=>false, EPMAfem.Dimensions.Y()=>false, EPMAfem.Dimensions.Z()=>false) for mom in moms)
for deg in 1:N
    for sh in [m for m in moms if SH.degree(m) == deg]
        for sh2 in [m for m in moms if SH.degree(m) == deg - 1]
            for dim in [EPMAfem.Dimensions.X(), EPMAfem.Dimensions.Y(), EPMAfem.Dimensions.Z()]
                if !iszero(SH.get_transport_coefficient(sh, sh2, dim)) # if one of the moments required x-conformity
                    if !confs[sh2][dim] # check the lower order moment for conformity
                        confs[sh][dim] = true # set sh's conf to true, if the lower order moment does not have the required conformity
                    end
                end
            end
        end
    end
end

function Base.sum(a::Dict{EPMAfem.Dimensions.SpaceDimension, Bool})
    Base.sum(a[dim] for dim in [EPMAfem.Dimensions.X(), EPMAfem.Dimensions.Y(), EPMAfem.Dimensions.Z()])
end

function Base.:<(a::Dict{EPMAfem.Dimensions.SpaceDimension, Bool}, b::Dict{EPMAfem.Dimensions.SpaceDimension, Bool})
    s_a = sum(a)
    s_b = sum(b)
    return Base.:<(s_a, s_b)
end

moms0 = [m for m in moms if sum(confs[m]) == 0]
moms1 = [m for m in moms if sum(confs[m]) == 1]
moms2 = [m for m in moms if sum(confs[m]) == 2]
moms3 = [m for m in moms if sum(confs[m]) == 3]

mom_system = (L2 = moms0, Hdiv = [(moms1[i], moms1[i+1]) for i in 1:2:length(moms1)], H1=moms2)
mom_system = (L2 = [moms[1], moms[5]], Hdiv = [(moms1[i], moms1[i+1]) for i in 1:2:length(moms1)], H1=[moms[4]])

Bz = [SH.get_transport_coefficient(m1, m2, EPMAfem.Dimensions.Z()) for m1 in [m for m in moms], m2 in [m for m in moms]]
Bx = [SH.get_transport_coefficient(m1, m2, EPMAfem.Dimensions.X()) for m1 in [m for m in moms], m2 in [m for m in moms]]
By = [SH.get_transport_coefficient(m1, m2, EPMAfem.Dimensions.Y()) for m1 in [m for m in moms], m2 in [m for m in moms]]

for sh in moms
    for sh2 in moms
        for dim in [EPMAfem.Dimensions.X(), EPMAfem.Dimensions.Y(), EPMAfem.Dimensions.Z()]
            if !iszero(SH.get_transport_coefficient(sh, sh2, dim))
                # check if ONLY one of the two participating functions in the "transport coupling have the required regularity
                @show xor(confs[sh][dim], confs[sh2][dim])
                @assert xor(confs[sh][dim], confs[sh2][dim])
                # check if it is always the higher reg. function (according to L2 < Hdiv < Hcurl < H1) that holds the derivatives
                if confs[sh][dim]
                    @show confs[sh2] < confs[sh]
                    @assert confs[sh2] < confs[sh]
                else
                    @show confs[sh] < confs[sh2]
                    @assert confs[sh] < confs[sh2]
                end
            end
        end
    end
end

function conf_to_string(conf)
    conf_string = "("
    if conf[EPMAfem.Dimensions.X()]
        conf_string *= "x,"
    end
    if conf[EPMAfem.Dimensions.Y()]
        conf_string *= "y,"
    end
    if conf[EPMAfem.Dimensions.Z()]
        conf_string *= "z,"
    end
    conf_string *= ")"
    return conf_string
end

function dim_name(dim)
    if dim == EPMAfem.Dimensions.X()
        return "x"
    elseif dim == EPMAfem.Dimensions.Y()
        return "y"
    elseif dim == EPMAfem.Dimensions.Z()
        return "z"
    end
end


function x_position(sh::SH.SphericalHarmonic)
    if sum(confs[sh]) == 0
        return 0
    elseif sum(confs[sh]) == 1
        return 1
    elseif sum(confs[sh]) == 2
        return 0
    elseif sum(confs[sh]) == 3
        return 1
    end
end

function y_position(sh::SH.SphericalHarmonic)
    if sum(confs[sh]) == 0
        return 0
    elseif sum(confs[sh]) == 1
        return 0
    elseif sum(confs[sh]) == 2
        return 1
    elseif sum(confs[sh]) == 3
        return 1
    end
end

begin
    plotly()
    plot()
    for sh in moms
        annotate!([x_position(sh)], [y_position(sh)], Plots.text("Y($(sh.degree), $(sh.order))<br> $(conf_to_string(confs[sh]))", 8))
    end
    paths = Dict(EPMAfem.Dimensions.X() => (x=[], y=[]), EPMAfem.Dimensions.Y() => (x=[], y=[]), EPMAfem.Dimensions.Z() => (x=[], y=[]))
    for sh in moms
        for sh2 in [m for m in moms if (SH.degree(m) == SH.degree(sh) + 1)]
            for (i, dim) in enumerate([EPMAfem.Dimensions.X(), EPMAfem.Dimensions.Y(), EPMAfem.Dimensions.Z()])
                if !iszero(SH.get_transport_coefficient(sh, sh2, dim))
                    plot!([x_position(sh), x_position(sh2)], [y_position(sh), y_position(sh2)], color=i, label=SH.get_transport_coefficient(sh, sh2, dim) |> x -> round(x, digits=10))
                    # push!(paths[dim].x, )
                    # push!(paths[dim].y, )
                end
            end
        end
    end
    # for (i, dim) in enumerate([EPMAfem.Dimensions.X(), EPMAfem.Dimensions.Y(), EPMAfem.Dimensions.Z()])
    #     x_pos = vcat(transpose.(paths[dim].x)...)
    #     y_pos = vcat(transpose.(paths[dim].y)...)
    #     plot!(x_pos', y_pos', label=dim_name(dim), color=i)
    # end
    
    xlims!(-N-1, N+1)
    ylims!(-N-1, 1)
    gr()
    plot!(size=(1500, 800))
end

[m for m in moms if (SH.is_even_in(m, EPMAfem.Dimensions.X()) && SH.is_even_in(m, EPMAfem.Dimensions.Y()) && SH.is_even_in(m, EPMAfem.Dimensions.Z()))]
[m for m in moms if (SH.is_odd_in(m, EPMAfem.Dimensions.X()) && SH.is_odd_in(m, EPMAfem.Dimensions.Y()) && SH.is_odd_in(m, EPMAfem.Dimensions.Z()))]

function interpolable(f::CellField)
    interp = Gridap.CellData.Interpolable(f; searchmethod=Gridap.CellData.KDTreeSearch(; num_nearest_vertices=5))
    rand_point = VectorValue(0.0, 0.0)
    cache = Gridap.Arrays.return_cache(interp, rand_point)
    return x -> Gridap.Arrays.evaluate!(cache, interp, x)
end

model = CartesianDiscreteModel((0, 1, 0, 1), (2, 2))
V = TestFESpace(model, ReferenceFE(bubble, VectorValue{2, Float64}))
f = interpolable(FEFunction(V, [0.0, 0, 0, 0, 0, 1, 0, 0]))
heatmap(0:0.01:1, 0:0.01:1, (x, y) -> f(VectorValue(x, y))[2])
# grid_gen_2D((0, 1, 0, 1); min_res=4, max_res=4, filepath="/tmp/tmp_msh_coarse.msh")
# model = DiscreteModelFromFile("/tmp/tmp_msh_coarse.msh")
# V_hdiv = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, 0), conformity=:Hdiv)
# V_l2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
# V_h1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)

# Ω = Triangulation(model)
# dx = Measure(Ω, 5)

# Dx1 = assemble_matrix((u, v) -> ∫(dot(VectorValue(1.0, 0.0), ∇(u))* v)dx, TrialFESpace(V_h1), V_l2) * P_Hdivx1_to_H1
# Dx2 = assemble_matrix((u, v) -> ∫(dot(VectorValue(0.0, 1.0), ∇(u))* v)dx, TrialFESpace(V_h1), V_l2) * P_Hdivx2_to_H1

# D_div = Dx1 + Dx2
# D_div_ref = assemble_matrix((u, v) -> ∫(divergence(u)*v)dx, TrialFESpace(V_hdiv), V_l2)

# f_Hdiv = interpolable(FEFunction(V_hdiv, rand(V_hdiv.nfree)))
# f_H1 = interpolable(FEFunction(V_h1, P_Hdivx1_to_H1 * f_Hdiv.interp.uh.free_values))

# p1 = heatmap(0:0.001:1, 0:0.001:1, (x, y) -> f_Hdiv(VectorValue(x, y))[1])
# p2 = heatmap(0:0.001:1, 0:0.001:1, (x, y) -> f_H1(VectorValue(x, y)))
# plot(p1, p2)

function blockdiag(Ms, Bs)
    Σm = 0
    for M in Ms
        m, n = size(M)
        @assert m==n
        Σm += m
    end
    MM = spzeros(Σm, Σm)
    MB = spzeros(Σm, Σm)
    start = 0
    for i in 1:length(Ms)
        M = Ms[i]
        MM[start+1:start+size(M, 1), start+1:start+size(M, 1)] .= M
        if i != 1
            B = Bs[i-1]
            MB[start+1:start+size(M, 1), start-size(B, 1)+1:start] = transpose(B)
        end
        if i != length(Ms)
            B = Bs[i]
            MB[start+1:start+size(M, 1), start+size(M, 1)+1:start+size(M, 1)+size(B, 2)] = -B
        end
        start += size(M, 1)
    end
    return MM, MB
end

# function exact_solution(params, N)
#     moms = SH.spherical_harmonics(N, EPMAfem.Dimensions._2D())
#     A_x = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in moms, sh2 in moms] .|> x -> round(x, digits=14)
#     A_x = eigen(A_x)
#     λ = A_x.values
#     B = A_x.vectors

#     w_0 = function(r)
#         u_0_ = zeros(length(moms))
#         u_0_[1] = params.C*exp(-params.σ*r^2)
#         return transpose(B)*u_0_
#     end

#     return function(t, r)
#         w_ = zeros(length(moms))
#         for i in 1:length(moms)
#             w_[i] = w_0(r - λ[i]*t)[i] * (r - λ[i]*t) / r
#         end
#         return B*w_
#     end
# end

using SpecialFunctions

function init_fourier(kx, ky)
    return π/params.σ * exp(-(kx^2+ky^2)/(4*params.σ))
end

function evolve_fourier(params, N)
    moms = SH.spherical_harmonics(N, EPMAfem.Dimensions._2D())
    A_x = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in moms, sh2 in moms] .|> x -> round(x, digits=14)
    A_y = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in moms, sh2 in moms] .|> x -> round(x, digits=14)
    
    u0_cache = zeros(length(moms))
    return function(t, kx, ky)
        u0_cache[1] = init_fourier(kx, ky)
        return (exp(-1im*(A_x*kx + A_y*ky)*t) * u0_cache)[1]
    end
end

function exact_solution2(params, N)
    f = evolve_fourier(params, N)
    return (t, x, y) -> hcubature(k -> 1/(2π)^2*f(t, k[1], k[2])*exp(1im*(k[1]*x + k[2]*y)), [-100, -100], [100, 100])[1]
end


function exact_solution(params, N)
    moms = SH.spherical_harmonics(N, EPMAfem.Dimensions._2D())
    A_x = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in moms, sh2 in moms] .|> x -> round(x, digits=14)
    A_x = eigen(A_x)
    λ = A_x.values

    return (t, r) -> exp(-params.σ*(r^2+maximum(λ)^2*t^2))*besseli(0, 2*params.σ*r*t*maximum(λ))
end

# okay, lets try to build the P_3 system
# mass matrix
grid_gen_2D((-1.5, 1.5, -1.5, 1.5); min_res=0.1, max_res=0.1, filepath="/tmp/tmp_msh_coarse.msh")
# grid_gen_2D((-1, 1, -1, 1); min_res=1, max_res=1, filepath="/tmp/tmp_msh_coarse.msh")
model = DiscreteModelFromFile("/tmp/tmp_msh_coarse.msh")
# model = CartesianDiscreteModel((-1.5, 1.5, -1.5, 1.5), (60, 60))
V_l2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
V_h1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
dx = Measure(Triangulation(model), 50)

mom_0 = [sh for sh in SH.spherical_harmonics(2, EPMAfem.Dimensions._2D()) if sh.degree==0]
mom_1 = [sh for sh in SH.spherical_harmonics(2, EPMAfem.Dimensions._2D()) if sh.degree==1]
mom_2 = [sh for sh in SH.spherical_harmonics(2, EPMAfem.Dimensions._2D()) if sh.degree==2]
mom_3 = [sh for sh in SH.spherical_harmonics(2, EPMAfem.Dimensions._2D()) if sh.degree==3]

quad = SH.lebedev_quadrature_max()

M_Ω0 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_0, sh2 in mom_0] .|> x -> round(x, digits=14)
M_Ω1 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_1] .|> x -> round(x, digits=14)
M_Ω2 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_2, sh2 in mom_2] .|> x -> round(x, digits=14)
M_Ω3 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_3, sh2 in mom_3] .|> x -> round(x, digits=14)

M_X0 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_l2), V_l2)
M_X1 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_h1), V_h1)
M_X2 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_l2), V_l2)
M_X3 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_h1), V_h1)

∂x(u) = dot(VectorValue(1.0, 0.0), ∇(u))
∂y(u) = dot(VectorValue(0.0, 1.0), ∇(u))

Bx_X0X1 = assemble_matrix((u, v) -> ∫(∂x(u) * v)dx, TrialFESpace(V_h1), V_l2)
Bx_X1X2 = assemble_matrix((u, v) -> ∫(u * ∂x(v))dx, TrialFESpace(V_l2), V_h1)
Bx_X2X3 = assemble_matrix((u, v) -> ∫(∂x(u) * v)dx, TrialFESpace(V_h1), V_l2)

By_X0X1 = assemble_matrix((u, v) -> ∫(∂y(u) * v)dx, TrialFESpace(V_h1), V_l2)
By_X1X2 = assemble_matrix((u, v) -> ∫(u * ∂y(v))dx, TrialFESpace(V_l2), V_h1)
By_X2X3 = assemble_matrix((u, v) -> ∫(∂y(u) * v)dx, TrialFESpace(V_h1), V_l2)

Bx_Ω0Ω1 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_0, sh2 in mom_1] .|> x -> round(x, digits=14)
Bx_Ω1Ω2 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2] .|> x -> round(x, digits=14)
Bx_Ω2Ω3 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2, sh2 in mom_3] .|> x -> round(x, digits=14)

By_Ω0Ω1 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_0, sh2 in mom_1] .|> x -> round(x, digits=14)
By_Ω1Ω2 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2] .|> x -> round(x, digits=14)
By_Ω2Ω3 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2, sh2 in mom_3] .|> x -> round(x, digits=14)

Bz_Ω1Ω2 = [quad(Ω -> Ω[3]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2] .|> x -> round(x, digits=14)

N = 1
Bx = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D()), sh2 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D())] .|> x -> round(x, digits=14)
By = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D()), sh2 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D())] .|> x -> round(x, digits=14)

# P3
M, B = blockdiag([kron(M_X0, M_Ω0), kron(M_X1, M_Ω1), kron(M_X2, M_Ω2), kron(M_X3, M_Ω3)], [kron(Bx_X0X1, Bx_Ω0Ω1) + kron(By_X0X1, By_Ω0Ω1), kron(Bx_X1X2, Bx_Ω1Ω2) + kron(By_X1X2, By_Ω1Ω2), kron(Bx_X2X3, Bx_Ω2Ω3) + kron(By_X2X3, By_Ω2Ω3)])
# P2
M, B = blockdiag([kron(M_X0, M_Ω0), kron(M_X1, M_Ω1), kron(M_X2, M_Ω2)], [kron(Bx_X0X1, Bx_Ω0Ω1) + kron(By_X0X1, By_Ω0Ω1), kron(Bx_X1X2, Bx_Ω1Ω2) + kron(By_X1X2, By_Ω1Ω2)])
# P1
M, B = blockdiag([kron(M_X0, M_Ω0), kron(M_X1, M_Ω1)], [kron(Bx_X0X1, Bx_Ω0Ω1) + kron(By_X0X1, By_Ω0Ω1)])

params = (C=1.0, σ=100.0)
radius(x) = sqrt(x[1]^2 + x[2]^2)
init_f(params) = x -> params.C*exp(-params.σ*radius(x)^2) #heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> init_f(VectorValue(x, y)))
b_x0 = assemble_vector(v -> ∫(init_f(params)*v)dx, V_l2)
b0 = zeros(size(M, 1)); b0[1:length(b_x0)] .= M_X0 \b_x0
mass_v = assemble_vector(v -> ∫(1*v)dx, V_l2)

sol = copy(b0)
Δt = 0.01
A = (M/Δt + 0.5*B)
A_LU = lu(A)
f_exact = exact_solution(params, 1)
eigs = eigen(Bx).values

anim = @animate for i in 1:100
    @show i
    rhs = ((M*sol)./Δt - 0.5.*B*sol)
    ldiv!(sol, A_LU, rhs)

    f = interpolable(FEFunction(V_l2, sol[1:V_l2.nfree]))
    @show dot(mass_v, sol[1:V_l2.nfree])
    p1 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f(VectorValue(x, y)), aspect_ratio=:equal)
    for λ ∈ eigs
        plot!(i.*Δt.*λ .* sin.(0:0.01:2π), i.*Δt.*λ.*cos.(0:0.01:2π), label=nothing)
    end


    # p2 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f_exact(i*Δt, norm((x, y))), aspect_ratio=:equal)
    # for λ ∈ eigs
    #     plot!(i.*Δt.*λ .* sin.(0:0.01:2π), i.*Δt.*λ.*cos.(0:0.01:2π), label=nothing)
    # end

    # plot(p1, p2)
    # # plot(-1:0.01:1, x -> f(VectorValue(x, 0.0)))
end
gif(anim)

# let # Hdiv
    V_hdiv = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, 0), conformity=:Hdiv)
    V_h1_conf_l2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)

    Mdiv_X1 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_hdiv), V_hdiv)

    Bdiv_X0X1 = assemble_matrix((u, v) -> ∫(divergence(u)*v)dx, TrialFESpace(V_hdiv), V_l2)

    # mass matrix of H1/L2 space
    
    M_h1_l2 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_h1_conf_l2), V_h1_conf_l2) |> EPMAfem.BlockDiagonals.BlockDiagonal{6}
    # projector of Hdiv first component into H1
    Bx1 = assemble_matrix((u, v) -> ∫(dot(VectorValue(1.0, 0.0), u)*v)dx, TrialFESpace(V_hdiv), V_h1_conf_l2)
    Bx2 = assemble_matrix((u, v) -> ∫(dot(VectorValue(0.0, 1.0), u)*v)dx, TrialFESpace(V_hdiv), V_h1_conf_l2)
    
    inv_M_h1_l2 = sparse(LinearAlgebra.inv!(copy(M_h1_l2)))
    
    P_Hdivx1_to_H1 = (inv_M_h1_l2 * Bx1 |> sparse .|> x -> round(x, digits=13)) |> dropzeros!
    P_Hdivx2_to_H1 = (inv_M_h1_l2 * Bx2 |> sparse .|> x -> round(x, digits=13)) |> dropzeros!

    Dx1_Ω1 = assemble_matrix((u, v) -> ∫(dot(VectorValue(1.0, 0.0), ∇(u))* v)dx, TrialFESpace(V_h1_conf_l2), V_l2) * P_Hdivx1_to_H1 |> transpose
    Dx2_Ω1 = assemble_matrix((u, v) -> ∫(dot(VectorValue(0.0, 1.0), ∇(u))* v)dx, TrialFESpace(V_h1_conf_l2), V_l2) * P_Hdivx1_to_H1 |> transpose
    
    Dx1_Ω2 = assemble_matrix((u, v) -> ∫(dot(VectorValue(1.0, 0.0), ∇(u))* v)dx, TrialFESpace(V_h1_conf_l2), V_l2) * P_Hdivx2_to_H1 |> transpose 
    Dx2_Ω2 = assemble_matrix((u, v) -> ∫(dot(VectorValue(0.0, 1.0), ∇(u))* v)dx, TrialFESpace(V_h1_conf_l2), V_l2) * P_Hdivx2_to_H1 |> transpose

    # test (should be 0!)
    Div_x1x2 = assemble_matrix((u, v) -> ∫(divergence(u)*v)dx, TrialFESpace(V_hdiv), V_l2)
    maximum(Dx1_Ω1 + Dx2_Ω2 - transpose(Div_x1x2) .|> abs) 

    BB_XΩ1XΩ2 = kron(Dx1_Ω1, Bx_Ω1Ω2[1:1, :]) + kron(Dx2_Ω1, By_Ω1Ω2[1:1, :]) + kron(Dx1_Ω2, Bx_Ω1Ω2[2:2, :]) + kron(Dx2_Ω2, By_Ω1Ω2[2:2, :])
    
    # P3
    M, B = blockdiag([kron(M_X0, M_Ω0), Mdiv_X1, kron(M_X2, M_Ω2), kron(M_X3, M_Ω3)], [Bdiv_X0X1/sqrt(3), BB_XΩ1XΩ2, kron(Bx_X2X3, Bx_Ω2Ω3) + kron(By_X2X3, By_Ω2Ω3)])
    # P1
    M, B = blockdiag([kron(M_X0, M_Ω0), Mdiv_X1], [Bdiv_X0X1/sqrt(3)])

# end




begin
    l = 3
    m = -2
    sh = SH.SphericalHarmonic(l, m)
    fac = sqrt((2*l+1)/(2π)*factorial(l-abs(m))/factorial(l+abs(m)))
    if m == 0
        fac = fac * 1/sqrt(2)
    end
    test((z, x, y)) = fac*30*x*y*z

    Ωs = [normalize(VectorValue(randn(3))) for i in 1:100]
    ratios = sh.(Ωs) ./ test.(Ωs)
    @show ratios
    @assert all(ratios .≈ 1.0)
end

mom_1
mom_2_l2 = [mom_2[3], mom_2[5]]
mom_2_hcurl = [mom_2[1], mom_2[2], mom_2[4]]

Bx_Ω1Ω2 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2_l2] .|> x -> round(x, digits=14)
Bx_Ω1Ω2 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2_l2] .|> x -> round(x, digits=14)
Bx_Ω1Ω2 = [quad(Ω -> Ω[3]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2_l2] .|> x -> round(x, digits=14)

Bx_Ω1Ω2 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2_hcurl] .|> x -> round(x, digits=14)
Bx_Ω1Ω2 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2_hcurl] .|> x -> round(x, digits=14)
Bx_Ω1Ω2 = [quad(Ω -> Ω[3]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2_hcurl] .|> x -> round(x, digits=14)
