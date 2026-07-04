
grid_gen_2D((-1, 1, -1, 1); min_res=0.05, max_res=0.05, filepath="/tmp/tmp_msh_coarse.msh")
model = DiscreteModelFromFile("/tmp/tmp_msh_coarse.msh")
# model = CartesianDiscreteModel((-1, 1, -1, 1), (60, 60))
V_l2_e = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
V_l2_o = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
V_hdiv = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, 0), conformity=:Hdiv)
V_h1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
V_helper = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)

dx = Measure(Triangulation(model), 50)
quad = SH.lebedev_quadrature_max()

mom_system2 = (
    L2_e = moms[1:1],
    H1 = [m for m in moms[2:end] if SH.is_even(m)],
    Hdiv = [(moms[2], moms[3])],
    L2_o = [m for m in moms[4:end] if SH.is_odd(m)]
)

# build the mass matrices
MΩ0_e = Diagonal(ones(length(mom_system2.L2_e)))
MΩ1 = Diagonal(ones(length(mom_system2.Hdiv)))
MΩ2 = Diagonal(ones(length(mom_system2.H1)))
MΩ3 = Diagonal(ones(length(mom_system2.L2_o)))

# build the coupling matrices
# Hdiv(1): trial function, L2(0): test function
# d/dx needs the first element of the Hdiv, is tested with L2
BΩx_1_0 = sparse([SH.get_transport_coefficient(sh1, sh2[1], EPMAfem.Dimensions.Z()) for sh1 in mom_system2.L2_e, sh2 in mom_system2.Hdiv])

# check validity
maximum(abs.(sparse([SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in mom_system2.L2_e, sh2 in mom_system2.L2_o]))) < 1e-15

# d/dy needs the second element of the Hdiv, is tested with L2
BΩy_1_0 = sparse([SH.get_transport_coefficient(sh1, sh2[2], EPMAfem.Dimensions.X()) for sh1 in mom_system2.L2_e, sh2 in mom_system2.Hdiv])

# check validity
maximum(abs.(sparse([SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in mom_system2.L2_e, sh2 in mom_system2.L2_o]))) < 1e-15

# Hdiv(1): trial function, H1(2): test function
# d/dx needs both elements of the Hdiv (is tested with the x-derivative of the H1)
BΩx_1x_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2[1], EPMAfem.Dimensions.Z()) for sh1 in mom_system2.H1, sh2 in mom_system2.Hdiv])
BΩx_1y_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2[2], EPMAfem.Dimensions.Z()) for sh1 in mom_system2.H1, sh2 in mom_system2.Hdiv])

# d/dy needs both element of the Hdiv (is tested with the y-derivative of the H1)
BΩy_1x_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2[1], EPMAfem.Dimensions.X()) for sh1 in mom_system2.H1, sh2 in mom_system2.Hdiv])
BΩy_1y_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2[2], EPMAfem.Dimensions.X()) for sh1 in mom_system2.H1, sh2 in mom_system2.Hdiv])

BΩx_3_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in mom_system2.H1, sh2 in mom_system2.L2_o])
BΩy_3_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in mom_system2.H1, sh2 in mom_system2.L2_o])

# same for space (mass matrices)
MX0_e = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_l2_e), V_l2_e)
MX1 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_hdiv), V_hdiv)
MX2 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_h1), V_h1)
MX3 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_l2_o), V_l2_o)

# coupling matrices
∂x(u) = dot(VectorValue(1.0, 0.0), ∇(u))
∂y(u) = dot(VectorValue(0.0, 1.0), ∇(u))

x(u) = dot(VectorValue(1.0, 0.0), u)
y(u) = dot(VectorValue(0.0, 1.0), u)

# umweg: 
M_helper = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_helper), V_helper) |> EPMAfem.BlockDiagonals.BlockDiagonal{3}
inv_M_helper = sparse(LinearAlgebra.inv!(copy(M_helper)))

# project the first component of the Hdiv into an higher order L2 space (V_helper)
Bx1 = assemble_matrix((u, v) -> ∫(x(u)*v)dx, TrialFESpace(V_hdiv), V_helper)
Bx2 = assemble_matrix((u, v) -> ∫(y(u)*v)dx, TrialFESpace(V_hdiv), V_helper)
BXx_1_0_0 = assemble_matrix((u, v) -> ∫(∂x(u)*v)dx, TrialFESpace(V_helper), V_l2_e) * inv_M_helper * Bx1
BXy_1_0_0 = assemble_matrix((u, v) -> ∫(∂y(u)*v)dx, TrialFESpace(V_helper), V_l2_e) * inv_M_helper * Bx2
# check validity
Bdiv_1_0_0 = assemble_matrix((u, v) -> ∫(divergence(u)*v)dx, TrialFESpace(V_hdiv), V_l2_e)
maximum(abs.(Bdiv_1_0 - BXx_1_0 - BXy_1_0))

BXx_1x_2 = assemble_matrix((u, v) -> ∫(x(u)*∂x(v))dx, TrialFESpace(V_hdiv), V_h1)
BXx_1y_2 = assemble_matrix((u, v) -> ∫(y(u)*∂x(v))dx, TrialFESpace(V_hdiv), V_h1)
BXy_1x_2 = assemble_matrix((u, v) -> ∫(x(u)*∂y(v))dx, TrialFESpace(V_hdiv), V_h1)
BXy_1y_2 = assemble_matrix((u, v) -> ∫(y(u)*∂y(v))dx, TrialFESpace(V_hdiv), V_h1)

BXx_3_2 = assemble_matrix((u, v) -> ∫(u*∂x(v))dx, TrialFESpace(V_l2_e), V_h1)
BXy_3_2 = assemble_matrix((u, v) -> ∫(u*∂y(v))dx, TrialFESpace(V_l2_e), V_h1)

M0_e = kron(MΩ0_e, MX0_e)
M2 = kron(MΩ2, MX2)
M1 = kron(MΩ1, MX1)
M3 = kron(MΩ3, MX3)
M = blockdiag(M0_e, M2, M1, M3)

_B1 = hcat(kron(BΩx_1_0, BXx_1_0_0) + kron(BΩy_1_0, BXy_1_0_0), zeros(size(M0_e, 1), size(M3, 1)))
_B2 = hcat(-kron(BΩx_1x_2, BXx_1x_2)-kron(BΩx_1y_2, BXx_1y_2)-kron(BΩy_1x_2, BXy_1x_2)-kron(BΩy_1y_2, BXy_1y_2), -kron(BΩx_3_2, BXx_3_2)-kron(BΩy_3_2, BXy_3_2))
_B = vcat(_B1, _B2) 
B = [spzeros(size(M0_e, 1) + size(M2, 1), size(M0_e, 1) + size(M2, 1)) _B
-transpose(_B) spzeros(size(M1, 1) + size(M3, 1), size(M1, 1) + size(M3, 1))]

init_f(x) = exp(-100.0*(x[1]^2 + x[2]^2)) #heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> init_f(VectorValue(x, y)))
b_x0 = assemble_vector(v -> ∫(init_f*v)dx, V_l2_e)
b0 = zeros(size(M, 1)); b0[1:length(b_x0)] .= M0_e \b_x0
mass_v = assemble_vector(v -> ∫(1*v)dx, V_l2_e)

sol = copy(b0)
Δt = 0.01
A = (M/Δt + 0.5*B)
A_LU = lu(A)
eigs = eigen([quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D()), sh2 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D())] .|> x -> round(x, digits=14)).values

anim = @animate for i in 1:100
    @show i
    rhs = ((M*sol)./Δt - 0.5.*B*sol)
    ldiv!(sol, A_LU, rhs)

    f = interpolable(FEFunction(V_l2_e, sol[1:V_l2_e.nfree]))
    @show dot(mass_v, sol[1:V_l2_e.nfree])
    p1 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f(VectorValue(x, y)), aspect_ratio=:equal)
    # for λ ∈ eigs
    #     plot!(i.*Δt.*λ .* sin.(0:0.01:2π), i.*Δt.*λ.*cos.(0:0.01:2π), label=nothing)
    # end
end
gif(anim)

