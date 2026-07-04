using Revise
using Plots
using EPMAfem
using EPMAfem.SphericalHarmonicsModels
SH = EPMAfem.SphericalHarmonicsModels
using EPMAfem.Gridap
using LinearAlgebra
using SparseArrays
include("../scripts/grid_gen.jl")

function interpolable(f::CellField)
    interp = Gridap.CellData.Interpolable(f; searchmethod=Gridap.CellData.KDTreeSearch(; num_nearest_vertices=5))
    rand_point = VectorValue(0.0, 0.0)
    cache = Gridap.Arrays.return_cache(interp, rand_point)
    return x -> Gridap.Arrays.evaluate!(cache, interp, x)
end

function _blockdiag(Ms)
    Σm = 0
    for M in Ms
        m, n = size(M)
        @assert m==n
        Σm += m
    end
    MM = spzeros(Σm, Σm)
    start = 0
    for i in 1:length(Ms)
        M = Ms[i]
        MM[start+1:start+size(M, 1), start+1:start+size(M, 1)] .= M
        start += size(M, 1)
    end
    return MM
end

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
            MB[start+1:start+size(M, 1), start-size(B, 1)+1:start] = -transpose(B)
        end
        if i != length(Ms)
            B = Bs[i]
            # compared to the old code, the minus is flipped
            MB[start+1:start+size(M, 1), start+size(M, 1)+1:start+size(M, 1)+size(B, 2)] = B
        end
        start += size(M, 1)
    end
    return MM, MB
end


# building a mixed L2-Hdiv-H1 hierarchy
grid_gen_2D((-1, 1, -1, 1); min_res=0.1, max_res=0.1, filepath="/tmp/tmp_msh_coarse.msh")
# grid_gen_2D((-1, 1, -1, 1); min_res=1, max_res=1, filepath="/tmp/tmp_msh_coarse.msh")
model = DiscreteModelFromFile("/tmp/tmp_msh_coarse.msh")
# model = CartesianDiscreteModel((-1, 1, -1, 1), (40, 30))
V_l2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
V_hdiv = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, 0), conformity=:Hdiv)
V_h1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
dx = Measure(Triangulation(model), 50)

mom_0_l2 = [sh for sh in SH.spherical_harmonics(3, EPMAfem.Dimensions._2D()) if sh.degree==0]
mom_1_hdiv = [sh for sh in SH.spherical_harmonics(3, EPMAfem.Dimensions._2D()) if sh.degree==1]
mom_2_l2 = [sh for sh in SH.spherical_harmonics(3, EPMAfem.Dimensions._2D()) if sh.degree==2][[1, 3]]
mom_2_h1 = [sh for sh in SH.spherical_harmonics(3, EPMAfem.Dimensions._2D()) if sh.degree==2][2:2]
mom_3_hdiv1 = [sh for sh in SH.spherical_harmonics(3, EPMAfem.Dimensions._2D()) if sh.degree==3][1:2]
mom_3_hdiv2 = [sh for sh in SH.spherical_harmonics(3, EPMAfem.Dimensions._2D()) if sh.degree==3][3:4]

quad = SH.lebedev_quadrature_max()

M_Ω0_l2 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_0_l2, sh2 in mom_0_l2] .|> x -> round(x, digits=14)
M_Ω1_hdiv = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_1_hdiv, sh2 in mom_1_hdiv] .|> x -> round(x, digits=14)
M_Ω2_l2 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_2_l2, sh2 in mom_2_l2] .|> x -> round(x, digits=14)
M_Ω2_h1 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_2_h1, sh2 in mom_2_h1] .|> x -> round(x, digits=14)
M_Ω3_hdiv1 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_3_hdiv1, sh2 in mom_3_hdiv1] .|> x -> round(x, digits=14)
M_Ω3_hdiv2 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_3_hdiv2, sh2 in mom_3_hdiv2] .|> x -> round(x, digits=14)

M_l2 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_l2), V_l2)
M_hdiv = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_hdiv), V_hdiv)
M_h1 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_h1), V_h1)

Bx_Ω0_l2_Ω1_hdiv = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_0_l2, sh2 in mom_1_hdiv] .|> x -> round(x, digits=14)
Bx_Ω1_hdiv_Ω2_l2 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1_hdiv, sh2 in mom_2_l2] .|> x -> round(x, digits=14)
Bx_Ω1_hdiv_Ω2_h1 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1_hdiv, sh2 in mom_2_h1] .|> x -> round(x, digits=14)
Bx_Ω2_l2_Ω3_hdiv1 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2_l2, sh2 in mom_3_hdiv1] .|> x -> round(x, digits=14)
Bx_Ω2_l2_Ω3_hdiv2 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2_l2, sh2 in mom_3_hdiv2] .|> x -> round(x, digits=14)
Bx_Ω2_h1_Ω3_hdiv1 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2_h1, sh2 in mom_3_hdiv1] .|> x -> round(x, digits=14)
Bx_Ω2_h1_Ω3_hdiv2 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2_h1, sh2 in mom_3_hdiv2] .|> x -> round(x, digits=14)

By_Ω0_l2_Ω1_hdiv = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_0_l2, sh2 in mom_1_hdiv] .|> x -> round(x, digits=14)
By_Ω1_hdiv_Ω2_l2 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1_hdiv, sh2 in mom_2_l2] .|> x -> round(x, digits=14)
By_Ω1_hdiv_Ω2_h1 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1_hdiv, sh2 in mom_2_h1] .|> x -> round(x, digits=14)
By_Ω2_l2_Ω3_hdiv1 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2_l2, sh2 in mom_3_hdiv1] .|> x -> round(x, digits=14)
By_Ω2_l2_Ω3_hdiv2 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2_l2, sh2 in mom_3_hdiv2] .|> x -> round(x, digits=14)
By_Ω2_h1_Ω3_hdiv1 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2_h1, sh2 in mom_3_hdiv1] .|> x -> round(x, digits=14)
By_Ω2_h1_Ω3_hdiv2 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2_h1, sh2 in mom_3_hdiv2] .|> x -> round(x, digits=14)

∂x(u) = dot(VectorValue(1.0, 0.0), ∇(u))
∂y(u) = dot(VectorValue(0.0, 1.0), ∇(u))

x(u) = dot(VectorValue(1.0, 0.0), u)
y(u) = dot(VectorValue(0.0, 1.0), u)

# a helper space to project the components of the hdiv to (order must be at least [order of hdiv]+1, conformity must be L2)
V_helper = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)
M_helper = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_helper), V_helper) |> EPMAfem.BlockDiagonals.BlockDiagonal{3}
inv_M_helper = sparse(LinearAlgebra.inv!(copy(M_helper)))

# assemble projectors from Hdiv into the components of the Hdiv to the helper space
Bx1 = assemble_matrix((u, v) -> ∫(x(u)*v)dx, TrialFESpace(V_hdiv), V_helper)
Bx2 = assemble_matrix((u, v) -> ∫(y(u)*v)dx, TrialFESpace(V_hdiv), V_helper)
P_Hdivx1_to_helper = (inv_M_helper * Bx1 |> sparse .|> x -> round(x, digits=13)) |> dropzeros!
P_Hdivx2_to_helper = (inv_M_helper * Bx2 |> sparse .|> x -> round(x, digits=13)) |> dropzeros!

D_hdivx1_l2 = assemble_matrix((u, v) -> ∫(∂x(u)* v)dx, TrialFESpace(V_helper), V_l2) * P_Hdivx1_to_helper |> transpose
D_hdivx2_l2 = assemble_matrix((u, v) -> ∫(∂y(u)* v)dx, TrialFESpace(V_helper), V_l2) * P_Hdivx2_to_helper |> transpose

D_hdivx1_h1x = assemble_matrix((u, v) -> ∫(x(u)*∂x(v))dx, TrialFESpace(V_hdiv), V_h1)
D_hdivx1_h1y = assemble_matrix((u, v) -> ∫(x(u)*∂y(v))dx, TrialFESpace(V_hdiv), V_h1)

D_hdivx2_h1x = assemble_matrix((u, v) -> ∫(y(u)*∂x(v))dx, TrialFESpace(V_hdiv), V_h1)
D_hdivx2_h1y = assemble_matrix((u, v) -> ∫(y(u)*∂y(v))dx, TrialFESpace(V_hdiv), V_h1)

# test (should be 0!)
Div_x1x2 = assemble_matrix((u, v) -> ∫(divergence(u)*v)dx, TrialFESpace(V_hdiv), V_l2)
maximum(D_hdivx1_l2 + D_hdivx2_l2 - transpose(Div_x1x2) .|> abs) 

# assemble the P1 system
M_P1, B_P1 = blockdiag([    kron(M_l2, M_Ω0_l2),    # Y_00
                            M_hdiv,                 # Y_10, Y_11
                       ],
                       [
                            - (kron(transpose(D_hdivx1_l2), Bx_Ω0_l2_Ω1_hdiv[:, 1:1]) + kron(transpose(D_hdivx2_l2), By_Ω0_l2_Ω1_hdiv[:, 2:2]))
                       ])


# assemble the P2 system
M_P2, B_P2 = blockdiag([    kron(M_l2, M_Ω0_l2),                # Y_00
                            M_hdiv,                             # Y_10, Y_11
                            _blockdiag([kron(M_l2, M_Ω2_l2),    # Y_20, Y_22
                                        kron(M_h1, M_Ω2_h1)]),      # Y_21
                       ],
                       [
                            -(kron(transpose(D_hdivx1_l2), Bx_Ω0_l2_Ω1_hdiv[:, 1:1]) + kron(transpose(D_hdivx2_l2), By_Ω0_l2_Ω1_hdiv[:, 2:2])),
                            hcat(
                                kron(D_hdivx1_l2, Bx_Ω1_hdiv_Ω2_l2[1:1, :]) + kron(D_hdivx2_l2, By_Ω1_hdiv_Ω2_l2[2:2, :]),
                                -kron(transpose(D_hdivx2_h1y), Bx_Ω1_hdiv_Ω2_h1[2:2, :]) - kron(transpose(D_hdivx1_h1x), By_Ω1_hdiv_Ω2_h1[1:1, :])
                            )
                       ])
# in the last line (my current reasoning would be to have D_hdivx1_h1x coupled with the x derivative and D_hdivx2_h1y coupled with the y derivative)
# the implementation is switched (because this resembles P2 better..)
# INVESTIGATE! maybe the reason is the transpose (is the D matrix the matrix that i want here? maybe the construction with hdiv -> hhelper -> derivative is not correct with transpose..)

# assemble the P3 system
M_P3, B_P3 = blockdiag([    kron(M_l2, M_Ω0_l2),                # Y_00
                            M_hdiv,                             # Y_10, Y_11
                            _blockdiag([kron(M_l2, M_Ω2_l2),    # Y_20, Y_22
                                        kron(M_h1, M_Ω2_h1)]),      # Y_21
                            _blockdiag([M_hdiv, M_hdiv])        # Y_30, Y_31, Y_32, Y_33

                       ],
                       [
                            -(kron(transpose(D_hdivx1_l2), Bx_Ω0_l2_Ω1_hdiv[:, 1:1]) + kron(transpose(D_hdivx2_l2), By_Ω0_l2_Ω1_hdiv[:, 2:2])),
                            hcat(
                                kron(D_hdivx1_l2, Bx_Ω1_hdiv_Ω2_l2[1:1, :]) + kron(D_hdivx2_l2, By_Ω1_hdiv_Ω2_l2[2:2, :]),
                                -kron(transpose(D_hdivx2_h1y), Bx_Ω1_hdiv_Ω2_h1[2:2, :]) - kron(transpose(D_hdivx1_h1x), By_Ω1_hdiv_Ω2_h1[1:1, :])
                            ),
                            hvcat((2, 2),
                                -kron(transpose(D_hdivx1_l2), Bx_Ω2_l2_Ω3_hdiv1[:, 1:1]) - kron(transpose(D_hdivx2_l2), By_Ω2_l2_Ω3_hdiv1[:, 2:2]),
                                -kron(transpose(D_hdivx1_l2), Bx_Ω2_l2_Ω3_hdiv2[:, 1:1]) - kron(transpose(D_hdivx2_l2), By_Ω2_l2_Ω3_hdiv2[:, 2:2]),
                                kron(nothing, Bx_Ω2_h1_Ω3_hdiv1) + kron(nothing, By_Ω2_h1_Ω3_hdiv1),
                                kron(nothing, Bx_Ω2_h1_Ω3_hdiv1) + kron(nothing, By_Ω2_h1_Ω3_hdiv1)

                            )
                       ])

M, _ = blockdiag([  kron(M_l2, M_Ω0_l2),                # Y_00
                    M_hdiv,                             # Y_10, Y_11
                    _blockdiag([kron(M_l2, M_Ω2_l2),    # Y_20, Y_22
                        kron(M_h1, M_Ω2_h1)]),              # Y_21
                    _blockdiag([M_hdiv, M_hdiv])     # Y_30, Y_31, Y_32, Y_33
                 ], 
                 [ 

                 ])


          
N = 2
M, B = M_P2, B_P2
Bx = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D()), sh2 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D())] .|> x -> round(x, digits=14)
By = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D()), sh2 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D())] .|> x -> round(x, digits=14)

init_f(x) = exp(-100*(x[1]^2+x[2]^2))#; heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> init_f(VectorValue(x, y)))
b_x0 = assemble_vector(v -> ∫(init_f*v)dx, V_l2)
b0 = zeros(size(M, 1)); b0[1:length(b_x0)] .= M_l2 \ b_x0
mass_v = assemble_vector(v -> ∫(1*v)dx, V_l2)

sol = copy(b0)
Δt = 0.01
A = (M/Δt + 0.5*B)
A_LU = lu(A)
eigs = eigen(Bx).values

anim = @animate for i in 1:100
    @show i
    rhs = ((M*sol)./Δt - 0.5.*B*sol)
    ldiv!(sol, A_LU, rhs)

    f = interpolable(FEFunction(V_l2, sol[1:V_l2.nfree]))
    @show dot(mass_v, sol[1:V_l2.nfree])
    heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f(VectorValue(x, y)), aspect_ratio=:equal)
    for λ ∈ eigs
        plot!(i.*Δt.*λ .* sin.(0:0.01:2π), i.*Δt.*λ.*cos.(0:0.01:2π), label=nothing)
    end
end
gif(anim)
