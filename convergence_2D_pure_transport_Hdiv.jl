# this is completely unrelated to the library: We compare the L2 - Hdiv - H1 - L2 - hierarchy to the previous test cases:

using Gridap
using LinearAlgebra
using SparseArrays
using EPMAfem.SphericalHarmonicsModels
using EPMAfem.SphericalHarmonicsModels: CircularHarmonic, get_transport_coefficient
using EPMAfem.Dimensions
using Plots
using SpecialFunctions
using Serialization
using LaTeXStrings
using DataFrames
using Pardiso

include("scripts/grid_gen.jl")

function interpolable(f::CellField)
    interp = Gridap.CellData.Interpolable(f; searchmethod=Gridap.CellData.KDTreeSearch(; num_nearest_vertices=5))
    rand_point = VectorValue(0.0, 0.0)
    cache = Gridap.Arrays.return_cache(interp, rand_point)
    return x -> Gridap.Arrays.evaluate!(cache, interp, x)
end

function Gridap.TrialFESpace(V::MultiFieldFESpace)
    return MultiFieldFESpace(TrialFESpace.(V))
end

struct Multi{V0_, V1_}
    V0::V0_
    V1::V1_
end

abstract type HierarchyVariant end
struct L2_H1 <: HierarchyVariant end # classic
name(::L2_H1) = "L2-H1"
Lname(::L2_H1) = L"L^2\!-\!H^1"
struct H1_L2 <: HierarchyVariant end # classic
name(::H1_L2) = "H1-L2"
Lname(::H1_L2) = L"H^1\!-\!L^2"
struct L2_Hdiv_H1_{V} <: HierarchyVariant end # L2 Hdiv H1 with arbitrary next 
name(::L2_Hdiv_H1_{V}) where V = "L2-Hdiv-H1_$(V)"
Lname(::L2_Hdiv_H1_{:H1H1}) = L"L^2\!-\!H(\textrm{div})\!-\!H^1"
struct L2_H1_H1H1 <: HierarchyVariant end # L2 following only H1 (to substantiate Hdiv)
name(::L2_H1_H1H1) = "L2-H1-H1H1"
Lname(::L2_H1_H1H1) = L"L^2\!-\!H^1\!-\!H^1H^1"
struct L2_H1_L2H1 <: HierarchyVariant end # mäh
name(::L2_H1_L2H1) = "L2-H1-L2H1"
Lname(::L2_H1_L2H1) = L"L^2\!-\!H^1\!-\!L^2H^1"

deg_to_space(v::HierarchyVariant, deg) = deg_to_spaces(v, deg)[1]

function deg_to_spaces(::L2_H1_H1H1, deg)
    if deg == 0
        return :L2_, :H1
    else
        if deg % 2 == 0
            return :H1H1, :H1
        else
            return :H1, :H1H1
        end
    end
end

function deg_to_spaces(::L2_H1_L2H1, deg)
    if deg == 0
        return :L2_, :H1
    else
        if deg % 2 == 0
            return :L2H1, :H1
        else
            return :H1, :L2H1
        end
    end
end

function deg_to_spaces(::L2_Hdiv_H1_{V}, deg) where V
    if deg == 0
        return :L2_, :Hdiv
    elseif deg == 1
        return :Hdiv, :H1
    else
        if deg % 2 == 0
            return :H1, V
        else
            return V, :H1
        end
    end
end

function deg_to_spaces(::L2_H1, deg)
    if deg == 0
        return :L2_, :H1
    else
        if deg % 2 == 0
            return :L2, :H1
        else
            return :H1, :L2
        end
    end
end

function deg_to_spaces(::H1_L2, deg)
    if deg == 0
        return :H1_, :L2
    else
        if deg % 2 == 0
            return :H1, :L2
        else
            return :L2, :H1
        end
    end
end

for v in [H1_L2(), L2_H1(), L2_Hdiv_H1_{:H1}(), L2_Hdiv_H1_{:L2}(), L2_Hdiv_H1_{:H1L2}(), L2_Hdiv_H1_{:L2H1}(), L2_Hdiv_H1_{:H1H1}(), L2_H1_H1H1(), L2_H1_L2H1()]
    @show "testing $(name(v))"
    for deg in 0:21
        # test validity
        s0, s1 = deg_to_spaces(v, deg)
        @assert deg_to_space(v, deg) == s0 
        @assert deg_to_space(v, deg+1) == s1
    end
    @show "success!"
end

function assemble_mass_matrix_part(space, (Ω, dx))
    return assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(space), space)
end

function assemble_mass_matrix_part(space::Multi, (Ω, dx))
    M0 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(space.V0), space.V0)
    M1 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(space.V1), space.V1)
    return blockdiag(M0, M1)
end

function assemble_mass_matrix(v::HierarchyVariant, N, spaces, gridap_args) # N is the P_N
    @assert N >= 1
    M = map(s -> assemble_mass_matrix_part(s, gridap_args), spaces)
    return blockdiag((getproperty(M, deg_to_space(v, i)) for i in 0:N)...)
end

val0(f) = dot(VectorValue(1.0, 0.0), f)
val1(f) = dot(VectorValue(0.0, 1.0), f) 

val0(f::Gridap.MultiField.MultiFieldCellField) = f[1]
val1(f::Gridap.MultiField.MultiFieldCellField) = f[2]

∂z(f) = dot(VectorValue(1.0, 0.0), ∇(f))
∂x(f) = dot(VectorValue(0.0, 1.0), ∇(f))

function assemble_transport_matrix_part(v::HierarchyVariant, deg, spaces, (Ω, dx))
    @assert deg >= 0
    deg0 = deg
    deg1 = deg + 1

    sp0, sp1 = deg_to_spaces(v, deg)
    space0 = getproperty(spaces, sp0)
    space1 = getproperty(spaces, sp1)

    z_00 = get_transport_coefficient(CircularHarmonic(deg0, 0), CircularHarmonic(deg1, 0), Z())
    z_01 = get_transport_coefficient(CircularHarmonic(deg0, 0), CircularHarmonic(deg1, 1), Z())
    x_00 = get_transport_coefficient(CircularHarmonic(deg0, 0), CircularHarmonic(deg1, 0), X())
    x_01 = get_transport_coefficient(CircularHarmonic(deg0, 0), CircularHarmonic(deg1, 1), X())

    if deg == 0 && deg_to_space(v, deg) == :L2_
        return assemble_matrix((u, v) -> ∫(- u*z_00*∂z(val0(v)) - u*z_01*∂z(val1(v)) - u*x_00*∂x(val0(v)) - u*x_01*∂x(val1(v)))dx, TrialFESpace(space0), space1)
    elseif deg == 0 && deg_to_space(v, deg) == :H1_
        return assemble_matrix((u, v) -> ∫(∂z(u)*z_00*val0(v) + ∂z(u)*z_01*val1(v) + ∂x(u)*x_00*val0(v) + ∂x(u)*x_01*val1(v))dx, TrialFESpace(space0), space1)
    elseif deg == 0
        error("deg 0 must be :L2_ or :H1_")
    end
    
    z_10 = get_transport_coefficient(CircularHarmonic(deg0, 1), CircularHarmonic(deg1, 0), Z())
    z_11 = get_transport_coefficient(CircularHarmonic(deg0, 1), CircularHarmonic(deg1, 1), Z())
    x_10 = get_transport_coefficient(CircularHarmonic(deg0, 1), CircularHarmonic(deg1, 0), X())
    x_11 = get_transport_coefficient(CircularHarmonic(deg0, 1), CircularHarmonic(deg1, 1), X())
        
    if (deg_to_space(v, deg) == :Hdiv) || (deg_to_space(v, deg) == :L2) || (deg_to_space(v, deg) == :H1H1)
        # do not realize the derivatives on the u0 or u1 (do realize on v0 or v1)
        return assemble_matrix((u, v) -> ∫( - val0(u)*z_00*∂z(val0(v)) - val0(u)*z_01*∂z(val1(v)) - val0(u)*x_00*∂x(val0(v)) - val0(u)*x_01*∂x(val1(v))
                                            - val1(u)*z_10*∂z(val0(v)) - val1(u)*z_11*∂z(val1(v)) - val1(u)*x_10*∂x(val0(v)) - val1(u)*x_11*∂x(val1(v)))dx, TrialFESpace(space0), space1)
    elseif (deg_to_space(v, deg) == :H1L2) || (deg_to_space(v, deg) == :L2H1)
        # do not realize the derivatives on the u0 or u1 (do realize on v0 or v1)
        # here the space is a Multi
        B0 = assemble_matrix((u0, v) -> ∫( - u0*z_00*∂z(val0(v)) - u0*z_01*∂z(val1(v)) - u0*x_00*∂x(val0(v)) - u0*x_01*∂x(val1(v)))dx, TrialFESpace(space0.V0), space1)
        B1 = assemble_matrix((u1, v) -> ∫( - u1*z_10*∂z(val0(v)) - u1*z_11*∂z(val1(v)) - u1*x_10*∂x(val0(v)) - u1*x_11*∂x(val1(v)))dx, TrialFESpace(space0.V1), space1)
        return [B0 B1]
    elseif deg_to_space(v, deg) == :H1
        if space1 isa Multi
            B0 = assemble_matrix((u, v0) -> ∫( ∂z(val0(u))*z_00*v0 + ∂x(val0(u))*x_00*v0 + ∂z(val1(u))*z_10*v0 + ∂x(val1(u))*x_10*v0)dx, TrialFESpace(space0), space1.V0)
            B1 = assemble_matrix((u, v1) -> ∫( ∂z(val0(u))*z_01*v1 + ∂x(val0(u))*x_01*v1 + ∂z(val1(u))*z_11*v1 + ∂x(val1(u))*x_11*v1)dx, TrialFESpace(space0), space1.V1)
            return [B0; B1]
        end
        # do realize the derivatives on the u0 or u1 (do not realize on v0 or v1)
        return assemble_matrix((u, v) -> ∫( ∂z(val0(u))*z_00*val0(v) + ∂z(val0(u))*z_01*val1(v) + ∂x(val0(u))*x_00*val0(v) + ∂x(val0(u))*x_01*val1(v)
                                          + ∂z(val1(u))*z_10*val0(v) + ∂z(val1(u))*z_11*val1(v) + ∂x(val1(u))*x_10*val0(v) + ∂x(val1(u))*x_11*val1(v))dx, TrialFESpace(space0), space1)
    end
    error("ohh no")
end

function n_variables_part(v::HierarchyVariant, deg, spaces)
    space = getproperty(spaces, deg_to_space(v, deg))
    if space isa Multi
        return space.V0.nfree + space.V1.nfree
    else
        return space.nfree
    end
end

function n_variables(v::HierarchyVariant, N, spaces)
    @assert N >= 1
    return sum(n_variables_part(v, i, spaces) for i in 0:N)
end

function assemble_transport_matrix(v::HierarchyVariant, N, spaces, gridap_args)
    Bs = [assemble_transport_matrix_part(v, i, spaces, gridap_args) for i in 0:N-1]
    n_Bs = [n_variables_part(v, i, spaces) for i in 0:N]
    n_B = n_variables(v, N, spaces)
    B = spzeros(n_B, n_B)

    offsets = cumsum([1; n_Bs])
    for i in 1:length(Bs)
        # block between variables i and i+1
        r = offsets[i]:offsets[i+1]-1
        c = offsets[i+1]:offsets[i+2]-1

        # lower block
        B[c, r] = Bs[i]

        # upper block is always the transpose
        B[r, c] = -Bs[i]'
    end
    return B
end

function interpolable_deg(v, spaces, deg, u)
    index_start = 0
    for i in 0:deg-1
        index_start += n_variables_part(v, i, spaces)
    end
    u_ = @view(u[index_start+1:index_start+n_variables_part(v, deg, spaces)])
    space = getproperty(spaces, deg_to_space(v, deg))
    if space isa Multi
        V0_interp = interpolable(FEFunction(space.V0, u_[1:space.V0.nfree]))
        V1_interp = interpolable(FEFunction(space.V1, u_[space.V0.nfree+1:end]))
        return x -> (V0_interp(x), V1_interp(x))
    end
    return interpolable(FEFunction(space, u_)) 
end

function gridap_setup(mdl, order=0)
    # trian and measure
    Ω = Triangulation(mdl)
    dx = Measure(Ω, 30)
    
    # spaces
    V_L2 = TestFESpace(mdl, ReferenceFE(lagrangian, VectorValue{2, Float64}, order), conformity=:L2)
    V_H1 = TestFESpace(mdl, ReferenceFE(lagrangian, VectorValue{2, Float64}, order+1), conformity=:H1)
    V_Hdiv = TestFESpace(mdl, ReferenceFE(raviart_thomas, Float64, order), conformity=:Hdiv)
    V_H1H1 = TestFESpace(mdl, ReferenceFE(lagrangian, VectorValue{2, Float64}, order+1), conformity=:H1) # no derivatives realized on this space V_H1H1 = V_H1
    # the zeroth moment is special.
    V_L2_ = TestFESpace(mdl, ReferenceFE(lagrangian, Float64, order), conformity=:L2)
    V_H1_ = TestFESpace(mdl, ReferenceFE(lagrangian, Float64, order+1), conformity=:H1)

    # weird combination
    V_H1L2 = Multi(V_H1_, V_L2_)
    V_L2H1 = Multi(V_L2_, V_H1_)

    spaces = (L2_ = V_L2_, H1_ = V_H1_, Hdiv = V_Hdiv, H1 = V_H1, L2 = V_L2, H1L2=V_H1L2, L2H1=V_L2H1, H1H1=V_H1H1)    
    gridap_args = (Ω, dx)
    return spaces, gridap_args
end



# A = sprand(10, 10, 0.8)

# solver = MKLPardisoSolver()
# b = rand(10)
# x = zeros(10)

# set_phase!(solver, 12) # analysis/numerical_factorization
# fix_iparm!(solver, :N)
# Pardiso.pardiso(solver, x, A, b)

# set_phase!(solver, 33) # solve/iterative_refinement
# fix_iparm!(solver, :N)
# Pardiso.pardiso(solver, x, A, b)

# A*x-b

# solver

# A = sprand(2, 2, 1.0)


# df = DataFrame(variant = String[], order = Int[], grid_res = Float64[], N = Int[], L2 = Float64[], L1 = Float64[], Linf = Float64[], L2_input = Float64[], L1_input = Float64[], Linf_input = Float64[], n_cells = Int[], n_dof0 = Int[], n_dof = Int[])
folder_name = "results_l2hdiv_cartesian/"
# df = serialize(joinpath(folder_name, "data.jls"), df)
df = deserialize(joinpath(folder_name, "data.jls"))

# initial condition (the sqrt(2π) is the integrated zeroth moment)
gaussian(x; σ = 0.1) = sqrt(2π)/(2π*σ^2)*exp(-1/(2σ^2)*(x[1]^2 + x[2]^2))

# analytic solution of the integrated flux
function analytic_solution(x, t; σ=0.1)
    r2 = x[1]^2 + x[2]^2
    return 2π/(σ^2*2π) *
           exp(-(r2+t^2)/(2σ^2)) *
           besseli(0, sqrt(r2)*t/σ^2)
end

# create grids
# begin
#     res_unstructured = [0.25, 0.18, 0.12, 0.09, 0.06, 0.045, 0.03, 0.02, 0.015]
#     for grid_res in 1:9 # 1:5 # 1:7
#         grid_gen_2D((-1.5, 1.5, -1.5, 1.5); min_res=res_unstructured[grid_res], max_res=res_unstructured[grid_res], filepath="results_l2hdiv/grids/msh_$(res_unstructured[grid_res]).msh")
#     end
# end

mesh = :cartesian

begin
    for v in [L2_H1(), H1_L2(), L2_Hdiv_H1_{:L2}(), L2_Hdiv_H1_{:H1H1}(), L2_H1_H1H1()]
        for order in [0, 1]
            N = 39
            res_unstructured = [0.25, 0.18, 0.12, 0.09, 0.06, 0.045, 0.03, 0.02]
            res_structured = [20, 25, 40, 50, 75, 100, 150, 200]
            res_range = if order == 0
                1:8
            elseif order == 1
                1:5
            end
            for grid_res in res_range
                # skip if already computed
                if !isempty(df[(df.variant .== name(v)) .& (df.order .== order) .& (df.grid_res .== grid_res) .& (df.N .== N), :]) @show ("skip", name(v), order, grid_res, N) continue end
                GC.gc()
                @show ("computing", name(v), order, grid_res, N)

                if mesh == :triangular
                    grid_gen_2D((-1.5, 1.5, -1.5, 1.5); min_res=res_unstructured[grid_res], max_res=res_unstructured[grid_res], filepath="/tmp/tmp_msh.msh")
                    model = DiscreteModelFromFile("/tmp/tmp_msh.msh")
                else
                    @assert mesh == :cartesian
                    model = CartesianDiscreteModel((-1.5, 1.5, -1.5, 1.5), (res_structured[grid_res], res_structured[grid_res]))
                end
                spaces, gridap_args = gridap_setup(model, order)

                M = assemble_mass_matrix(v, N, spaces, gridap_args)
                B = assemble_transport_matrix(v, N, spaces, gridap_args)

                N_t = 299
                Δt = 1.0 / N_t
                # A = lu((M/Δt + B/2));
                A = (M/Δt + B/2);
                C = (M/Δt - B/2);

                dx = gridap_args[2]

                u, input_errors_ = let
                    u = zeros(n_variables(v, N, spaces))
                    
                    M0 = assemble_mass_matrix_part(getproperty(spaces, deg_to_space(v, 0)), gridap_args)
                    u0 = M0 \ assemble_vector(v -> ∫(gaussian * v)dx, getproperty(spaces, deg_to_space(v, 0)))
                    u[1:n_variables_part(v, 0, spaces)] .= u0
                    
                    f_u0 = FEFunction(getproperty(spaces, deg_to_space(v, 0)), u0)

                    L2_input_ = sqrt(sum(∫((f_u0 - gaussian)*(f_u0 - gaussian))*dx))
                    L1_input_ = sum(∫(abs(f_u0 - gaussian))*dx)
                    interp_f_u0 = interpolable(f_u0)
                    Linf_input_ = maximum(abs(interp_f_u0(VectorValue(x, y)) - gaussian(VectorValue(x, y))) for x in -1.5:0.001:1.5, y in -1.5:0.001:1.5)
                    input_errors_ = (L2 = L2_input_, L1 = L1_input_, Linf = Linf_input_)
                    u, input_errors_
                end

                solver = MKLPardisoSolver()

                set_phase!(solver, 12) # analysis/numerical_factorization
                fix_iparm!(solver, :N)
                Pardiso.pardiso(solver, u, A, C*u)

                for i in 1:N_t
                    @show i
                    rhs = C*u

                    set_phase!(solver, 33) # solve/iterative_refinement
                    fix_iparm!(solver, :N)
                    Pardiso.pardiso(solver, u, A, rhs)
                    # Pardiso.solve!(ps, u, A, rhs)
                    # u = ldiv!(u, A, C*u)
                end

                set_phase!(solver, -1) # release internal memory
                Pardiso.pardiso(solver, u, A, C*u)

                f = interpolable_deg(v, spaces, 0, u)
                # multiply f by sqrt(2π) to get the angular integral
                begin
                    heatmap(-1.5:0.005:1.5, -1.5:0.005:1.5, (x, y) -> sqrt(2π)*f(VectorValue(x, y)), aspect_ratio=:equal, clims=(-4, 4), cmap=:jet)
                    plot!(size=(315, 300), dpi=1000, fontfamily="Computer Modern")
                    savefig(joinpath(folder_name, "$(name(v))_$(order)_$(grid_res).png"))
                end

                L2_ = sqrt(sum(∫((sqrt(2π)*f.interp.uh - (x -> analytic_solution(x, 1.0)))*(sqrt(2π)*f.interp.uh - (x -> analytic_solution(x, 1.0))))*dx))
                L1_ = sum(∫(abs(sqrt(2π)*f.interp.uh - (x -> analytic_solution(x, 1.0))))*dx)
                Linf_ = maximum(abs(sqrt(2π)*f(VectorValue(x, y)) - analytic_solution(VectorValue(x, y), 1.0)) for x in -1.5:0.001:1.5, y in -1.5:0.001:1.5)
                
                if grid_res == 5 && order == 0
                    mkdir(joinpath(folder_name, "all_moments_$(name(v))"))
                    mkdir(joinpath(folder_name, "all_moments_$(name(v))_noclim"))
                    for i in 0:N
                        f_viz = interpolable_deg(v, spaces, i, u)
                        heatmap(-1.5:0.005:1.5, -1.5:0.005:1.5, (x, y) -> f_viz(VectorValue(x, y))[1], aspect_ratio=:equal, clims=(-4/sqrt(2π), 4/sqrt(2π)), cmap=:jet)
                        plot!(size=(315, 300), dpi=1000, fontfamily="Computer Modern")
                        savefig(joinpath(folder_name, "all_moments_$(name(v))/ch_$(i)_0.png"))

                        heatmap(-1.5:0.005:1.5, -1.5:0.005:1.5, (x, y) -> f_viz(VectorValue(x, y))[1], aspect_ratio=:equal, cmap=:jet)
                        plot!(size=(315, 300), dpi=1000, fontfamily="Computer Modern")
                        savefig(joinpath(folder_name, "all_moments_$(name(v))_noclim/ch_$(i)_0.png"))
                        if i != 0
                            heatmap(-1.5:0.005:1.5, -1.5:0.005:1.5, (x, y) -> f_viz(VectorValue(x, y))[2], aspect_ratio=:equal, clims=(-4/sqrt(2π), 4/sqrt(2π)), cmap=:jet)
                            plot!(size=(315, 300), dpi=1000, fontfamily="Computer Modern")
                            savefig(joinpath(folder_name, "all_moments_$(name(v))/ch_$(i)_1.png"))

                            heatmap(-1.5:0.005:1.5, -1.5:0.005:1.5, (x, y) -> f_viz(VectorValue(x, y))[2], aspect_ratio=:equal, cmap=:jet)
                            plot!(size=(315, 300), dpi=1000, fontfamily="Computer Modern")
                            savefig(joinpath(folder_name, "all_moments_$(name(v))_noclim/ch_$(i)_1.png"))
                        end
                    end
                end

                # data_collection
                data_ = (
                    variant = name(v),
                    order = order,
                    grid_res = grid_res,
                    N = N,
                    L2 = L2_,
                    L1 = L1_,
                    Linf = Linf_,
                    L2_input = input_errors_.L2,
                    L1_input = input_errors_.L1,
                    Linf_input = input_errors_.Linf,
                    n_cells = Gridap.num_cells(model.grid),
                    n_dof0 = n_variables_part(v, 0, spaces),
                    n_dof = n_variables(v, N, spaces)
                    )

                push!(df, data_)
                serialize(joinpath(folder_name, "data.jls"), df)
            end
        end
    end
end

# heatmap(-1.5:0.005:1.5, -1.5:0.005:1.5, (x, y) -> analytic_solution(VectorValue(x, y), 1), aspect_ratio=:equal, clims=(-4, 4), cmap=:jet)
# plot!(size=(315, 300), dpi=1000, fontfamily="Computer Modern")
# savefig("results_l2hdiv/reference.png")

# using EPMAfem.HCubature
# σ = 0.1
# gaussian(x) = 2π/(2π*σ^2)*exp(-1/(2σ^2)*(x[1]^2 + x[2]^2))
# L2_ref = sqrt(hcubature(x -> analytic_solution(VectorValue(x[1], x[2]), 1; σ=σ)^2, (-1.5, -1.5), (1.5, 1.5))[1])
# L1_ref = hcubature(x -> abs(analytic_solution(VectorValue(x[1], x[2]), 1; σ=σ)), (-1.5, -1.5), (1.5, 1.5))[1]
# Linf_ref = maximum(abs(analytic_solution(VectorValue(x, y), 1; σ=σ)) for x in -1.5:0.001:1.5, y in -1.5:0.001:1.5)

# L2_input_ref = sqrt(hcubature(x -> gaussian(VectorValue(x[1], x[2]))^2, (-1.5, -1.5), (1.5, 1.5))[1])
# L1_input_ref = hcubature(x -> abs(gaussian(VectorValue(x[1], x[2]))), (-1.5, -1.5), (1.5, 1.5))[1]
# Linf_input_ref = maximum(abs(gaussian(VectorValue(x, y))) for x in -1.5:0.001:1.5, y in -1.5:0.001:1.5)

# begin
#     plot(xaxis=:log, yaxis=:log)
#     vars_cols = [(L2_H1(), 1), (H1_L2(), 2), (L2_Hdiv_H1_{:H1H1}(), 3)]
#     for (v, c) in vars_cols
#         y_axis_val = :n_cells
#         df_v = sort(df[(df.variant .== name(v)) .& (df.order .== 0), :], y_axis_val)
#         plot!(df_v.:($y_axis_val), df_v.L2 ./ L2_ref, color=c, ls=:solid, label=nothing, marker=:o)
#         plot!(df_v.:($y_axis_val), df_v.Linf ./ Linf_ref, color=c, ls=:dash, label=nothing, marker=:o)
#         plot!(df_v.:($y_axis_val), df_v.L1 ./ L1_ref, color=c, ls=:dot, label=nothing, marker=:o)
#     end

#     plot!([], [], color=vars_cols[1][2], ls=:solid, label=Lname(vars_cols[1][1]), marker=:o)
#     plot!([], [], color=:gray, ls=:solid, label=L"rel. $L^2$ err.")
#     plot!([], [], color=vars_cols[2][2], ls=:solid, label=Lname(vars_cols[2][1]), marker=:o)
#     plot!([], [], color=:gray, ls=:dash, label=L"rel. $L^\infty$ err.")
#     plot!([], [], color=vars_cols[3][2], ls=:solid, label=Lname(vars_cols[3][1]), marker=:o)
#     plot!([], [], color=:gray, ls=:dot, label=L"rel. $L^1$ err.")

#     plot!(3e2:5e4, x->200/x, ls=:dash, color=:gray, label=nothing)
#     annotate!(2e3, 5.2e-2, Plots.text(L"\mathcal{O}(1/N)", 9, :gray), color=:gray)
#     plot!(3e2:5e4, x->120/(x)^(1/2), ls=:dash, color=:gray, label=nothing)
#     # plot!(3e2:5e4, x->50/(x)^(1/2), ls=:dash, color=:gray, label=nothing)
#     annotate!(14e3, 1.9, Plots.text(L"\mathcal{O}(1/\sqrt{N})", 9, :gray), color=:gray)
#     plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", legend=:bottomleft, legend_columns=2)
#     xlabel!("number of grid cells")
#     ylabel!("relative errors")
#     # savefig("results_l2hdiv/convergence_order0_variants.png")
# end

# begin
#     plot(xaxis=:log, yaxis=:log)
#     vars_order_cols = [(L2_H1(), 0, 1, :o), (H1_L2(), 0, 2, :o), (L2_Hdiv_H1_{:H1H1}(), 0, 3, :o), (L2_H1(), 1, 1, :dtriangle), (H1_L2(), 1, 2, :dtriangle), (L2_Hdiv_H1_{:H1H1}(), 1, 3, :dtriangle)]
#     for (v, o, c, marker) in vars_order_cols
#         y_axis_val = :n_dof
#         df_v = sort(df[(df.variant .== name(v)) .& (df.order .== o), :], y_axis_val)
#         plot!(df_v.:($y_axis_val), df_v.L2 ./ L2_ref, color=c, ls=:solid, label=nothing, marker=marker)
#         plot!(df_v.:($y_axis_val), df_v.Linf ./ Linf_ref, color=c, ls=:dash, label=nothing, marker=marker)
#         # plot!(df_v.:($y_axis_val), df_v.L1 ./ L1_ref, color=c, ls=:dot, label=nothing, marker=marker)
#     end

#     plot!([], [], color=vars_order_cols[1][3], ls=:solid, label=Lname(vars_order_cols[1][1]))
#     plot!([], [], color=vars_order_cols[2][3], ls=:solid, label=Lname(vars_order_cols[2][1]))
#     plot!([], [], color=vars_order_cols[3][3], ls=:solid, label=Lname(vars_order_cols[3][1]))
#     plot!([], [], color=:gray, ls=:solid, label=L"rel. $L^2$ err.")
#     plot!([], [], color=:gray, ls=:dash, label=L"rel. $L^\infty$ err.")
#     plot!([], [], color=:transparent, label=" ")
#     plot!([], [], color=:gray, ls=:solid, label="order: 0", marker=:o)
#     plot!([], [], color=:gray, ls=:solid, label="order: 1", marker=:dtriangle)

#     plot!([1e4, 5e6], x->1e3/sqrt(x), ls=:dash, color=:gray, label=nothing)
#     plot!([1e4, 5e6], x->5e4/x, ls=:dash, color=:gray, label=nothing)
#     plot!([1e4, 5e6], x->2e8/x^2, ls=:dash, color=:gray, label=nothing)

#     ylims!(5e-4, 11)
#     yticks!([10^0, 10^-1, 10^-2, 10^-3])


#     plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", legend=:bottomleft, legend_columns=2)
#     xlabel!("number of dof")
#     ylabel!("relative errors")
#     # savefig("results_l2hdiv/convergence_ndof_variants.png")
#     # savefig("results_l2hdiv/convergence_ndof_variants.pdf")
# end


# begin
#     plot(xaxis=:log, yaxis=:log)
#     vars_cols = [(L2_H1(), 1), (H1_L2(), 2), (L2_Hdiv_H1_{:H1H1}(), 3)]
#     for (v, c) in vars_cols
#         y_axis_val = :n_cells
#         df_v = sort(df[(df.variant .== name(v)) .& (df.order .== 1), :], y_axis_val)
#         plot!(df_v.:($y_axis_val), df_v.L2 ./ L2_ref, color=c, ls=:solid, label=nothing, marker=:o)
#         plot!(df_v.:($y_axis_val), df_v.Linf ./ Linf_ref, color=c, ls=:dash, label=nothing, marker=:o)
#         plot!(df_v.:($y_axis_val), df_v.L1 ./ L1_ref, color=c, ls=:dot, label=nothing, marker=:o)
#     end

#     plot!([], [], color=vars_cols[1][2], ls=:solid, label=Lname(vars_cols[1][1]), marker=:o)
#     plot!([], [], color=:gray, ls=:solid, label=L"rel. $L^2$ err.")
#     plot!([], [], color=vars_cols[2][2], ls=:solid, label=Lname(vars_cols[2][1]), marker=:o)
#     plot!([], [], color=:gray, ls=:dash, label=L"rel. $L^\infty$ err.")
#     plot!([], [], color=vars_cols[3][2], ls=:solid, label=Lname(vars_cols[3][1]), marker=:o)
#     plot!([], [], color=:gray, ls=:dot, label=L"rel. $L^1$ err.")
#     plot!(3e2:7e3, x->2000/x, ls=:dash, color=:gray, label=nothing)
#     annotate!(3e3, 1.9, Plots.text(L"\mathcal{O}(1/N)", 9, :gray), color=:gray)
#     plot!(3e2:7e3, x->20000/(x)^(2), ls=:dash, color=:gray, label=nothing)
#     annotate!(5e2, 3e-2, Plots.text(L"\mathcal{O}(1/N^2)", 9, :gray), color=:gray)
#     plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", legend=:bottomleft, legend_columns=2)
#     xlabel!("number of grid cells")
#     ylabel!("relative errors")
#     yticks!([10^0, 10^-1, 10^-2, 10^-3])
#     xticks!([10^2.5, 10^3, 10^3.5])
#     savefig("results_l2hdiv/convergence_order1_variants.png")
# end


# begin
#     plot(xaxis=:log, yaxis=:log)
#     for (v, c, o, marker) in [(L2_H1(), 1, 0, :o), (H1_L2(), 2, 0, :o), (L2_H1(), 1, 1, :dtriangle), (H1_L2(), 2, 1, :dtriangle)]
#         df_v = sort(df[(df.variant .== name(v)) .& (df.order .== o), :], :grid_res)
#         plot!(df_v.n_dof0, df_v.L2_input ./ L2_input_ref, color=c, label=nothing, ls=:solid, marker=marker)
#         plot!(df_v.n_dof0, df_v.Linf_input ./ Linf_input_ref, color=c, label=nothing, ls=:dash, marker=marker)
#         plot!(df_v.n_dof0, df_v.L1_input ./ L1_input_ref, color=c, label=nothing, ls=:dot, marker=marker)
#     end
#     plot!([], [], color=1, label=L"L^2")
#     plot!([], [], color=2, label=L"H^1")
#     plot!([], [], color=:gray, ls=:solid, label=L"rel. $L^2$ err.")
#     plot!([], [], color=:gray, ls=:dash, label=L"rel. $L^\infty$ err.")
#     plot!([], [], color=:gray, ls=:dot, label=L"rel. $L^1$ err.")
#     plot!([], [], color=:transparent, label=" ")
#     plot!([], [], color=:gray, ls=:solid, label="order: 0", marker=:o)
#     plot!([], [], color=:gray, ls=:solid, label="order: 1", marker=:dtriangle)
    
#     # plot!([1e4, 5e6], x->1e2/sqrt(x), ls=:dash, color=:gray, label=nothing)
#     # plot!([1e4, 5e6], x->5e3/x, ls=:dash, color=:gray, label=nothing)
#     # plot!([1e4, 5e6], x->2e7/x^2, ls=:dash, color=:gray, label=nothing)

#     ylims!(5e-4, 1)
    
#     plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", legend=:bottomleft, legend_columns=2)
#     xlabel!("number of dof")
#     ylabel!("relative error (input)")
#     savefig("results_l2hdiv/convergence_ndof_input.png")
# end


# nothing



# # compute a single solution for visualization
# N = 39
# v = H1_L2()
# grid_gen_2D((-1.5, 1.5, -1.5, 1.5); min_res=0.06, max_res=0.06, filepath="/tmp/tmp_msh.msh")
# model = DiscreteModelFromFile("/tmp/tmp_msh.msh")
# spaces, gridap_args = gridap_setup(model, 0)

# M = assemble_mass_matrix(v, N, spaces, gridap_args)
# B = assemble_transport_matrix(v, N, spaces, gridap_args)

# N_t = 299
# Δt = 1.0 / N_t
# σ = 0.0
# # try with absorption # implicit midpoint
# A = (M/Δt + (σ/2)*M + B/2);
# C = (M/Δt - (σ/2)*M - B/2);

# # implicit euler
# A = (M/Δt + σ*M + B)
# C = M/Δt

# dx = gridap_args[2]

# u = let
#     u = zeros(n_variables(v, N, spaces))
    
#     M0 = assemble_mass_matrix_part(getproperty(spaces, deg_to_space(v, 0)), gridap_args)
#     u0 = M0 \ assemble_vector(v -> ∫(gaussian * v)dx, getproperty(spaces, deg_to_space(v, 0)))
#     u[1:n_variables_part(v, 0, spaces)] .= u0
#     u
# end

# solver = MKLPardisoSolver()

# set_phase!(solver, 12) # analysis/numerical_factorization
# fix_iparm!(solver, :N)
# Pardiso.pardiso(solver, u, A, C*u)

# for i in 1:N_t
#     @show i
#     rhs = C*u

#     set_phase!(solver, 33) # solve/iterative_refinement
#     fix_iparm!(solver, :N)
#     Pardiso.pardiso(solver, u, A, rhs)
#     # Pardiso.solve!(ps, u, A, rhs)
#     # u = ldiv!(u, A, C*u)
# end

# set_phase!(solver, -1) # release internal memory
# Pardiso.pardiso(solver, u, A, C*u)

# f = interpolable_deg(v, spaces, 0, u)
# # multiply f by sqrt(2π) to get the angular integral
# begin
#     heatmap(-1.5:0.005:1.5, -1.5:0.005:1.5, (x, y) -> sqrt(2π)*f(VectorValue(x, y)), aspect_ratio=:equal, cmap=:jet)
#     plot!(size=(315, 300), dpi=1000, fontfamily="Computer Modern")
# end

