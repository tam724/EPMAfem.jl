
grid_gen_2D((-1, 1, -1, 1); min_res=0.05, max_res=0.05, filepath="/tmp/tmp_msh_coarse.msh")
model = DiscreteModelFromFile("/tmp/tmp_msh_coarse.msh")
# model = CartesianDiscreteModel((-1, 1, -1, 1), (60, 60))
V_l2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
V_hdiv = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, 0), conformity=:Hdiv)
# V_hdiv = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2, Float64}, 1), conformity=:H1)
V_h1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
# V_helper = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)

dx = Measure(Triangulation(model), 50)
quad = SH.lebedev_quadrature_max()

# build the mass matrices
MΩ_l2 = Diagonal(ones(1))
MΩ_hdiv = Diagonal(ones(1))
MΩ_e_1 = Diagonal(ones(1))
MΩ_e_2 = Diagonal(ones(1))

# build the coupling matrices
# Hdiv(1): trial function, L2(0): test function
# d/dz needs the first element of the Hdiv, is tested with L2
BΩz_hdiv1_l2 = sparse([SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in moms[1:1], sh2 in moms[2:2]])
# BΩz_hdiv2_l2 = sparse([SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in moms[1:1], sh2 in moms[3:3]])

# d/dx needs the second element of the Hdiv, is tested with L2
# BΩx_hdiv1_l2 = sparse([SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in moms[1:1], sh2 in moms[2:2]])
BΩx_hdiv2_l2 = sparse([SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in moms[1:1], sh2 in moms[3:3]])

# Hdiv(1): trial function, H1(2): test function
# d/dz needs both elements of the Hdiv (is tested with the x-derivative of the H1)
BΩz_hdiv1_e_1 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in moms[4:4], sh2 in moms[2:2]])
BΩz_hdiv2_e_1 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in moms[4:4], sh2 in moms[3:3]])
BΩz_hdiv1_e_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in moms[5:5], sh2 in moms[2:2]])
BΩz_hdiv2_e_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in moms[5:5], sh2 in moms[3:3]])

# d/dx needs both element of the Hdiv (is tested with the y-derivative of the H1)
BΩx_hdiv1_e_1 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in moms[4:4], sh2 in moms[2:2]])
BΩx_hdiv2_e_1 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in moms[4:4], sh2 in moms[3:3]])
BΩx_hdiv1_e_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in moms[5:5], sh2 in moms[2:2]])
BΩx_hdiv2_e_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in moms[5:5], sh2 in moms[3:3]])

# same for space (mass matrices)
V_e_1 = V_h1
V_e_2 = V_h1
MX_l2 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_l2), V_l2)
MX_hdiv = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_hdiv), V_hdiv)
MX_e_1 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_e_1), V_e_1)
MX_e_2 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_e_2), V_e_2)

# coupling matrices
∂z(u) = dot(VectorValue(1.0, 0.0), ∇(u))
∂x(u) = dot(VectorValue(0.0, 1.0), ∇(u))

z(u) = dot(VectorValue(1.0, 0.0), u)
x(u) = dot(VectorValue(0.0, 1.0), u)

# project the first component of the Hdiv into an higher order L2 space (V_helper)
BXz_hdiv1_l2 = assemble_matrix((u, v) -> ∫(∂z(z(u))*v)dx, TrialFESpace(V_hdiv), V_l2)
BXx_hdiv2_l2 = assemble_matrix((u, v) -> ∫(∂x(x(u))*v)dx, TrialFESpace(V_hdiv), V_l2)

if V_e_1 == V_h1
    BXz_hdiv1_e_1 = assemble_matrix((u, v) -> ∫(z(u)*∂z(v))dx, TrialFESpace(V_hdiv), V_e_1)
    BXz_hdiv2_e_1 = assemble_matrix((u, v) -> ∫(x(u)*∂z(v))dx, TrialFESpace(V_hdiv), V_e_1)
    BXx_hdiv1_e_1 = assemble_matrix((u, v) -> ∫(z(u)*∂x(v))dx, TrialFESpace(V_hdiv), V_e_1)
    BXx_hdiv2_e_1 = assemble_matrix((u, v) -> ∫(x(u)*∂x(v))dx, TrialFESpace(V_hdiv), V_e_1)
else
    BXz_hdiv1_e_1 = -assemble_matrix((u, v) -> ∫(∂z(z(u))*v)dx, TrialFESpace(V_hdiv), V_e_1)
    BXz_hdiv2_e_1 = -assemble_matrix((u, v) -> ∫(∂z(x(u))*v)dx, TrialFESpace(V_hdiv), V_e_1)
    BXx_hdiv1_e_1 = -assemble_matrix((u, v) -> ∫(∂x(z(u))*v)dx, TrialFESpace(V_hdiv), V_e_1)
    BXx_hdiv2_e_1 = -assemble_matrix((u, v) -> ∫(∂x(x(u))*v)dx, TrialFESpace(V_hdiv), V_e_1)
end
if V_e_2 == V_h1
    BXz_hdiv1_e_2 = assemble_matrix((u, v) -> ∫(z(u)*∂z(v))dx, TrialFESpace(V_hdiv), V_e_2)
    BXz_hdiv2_e_2 = assemble_matrix((u, v) -> ∫(x(u)*∂z(v))dx, TrialFESpace(V_hdiv), V_e_2)
    BXx_hdiv1_e_2 = assemble_matrix((u, v) -> ∫(z(u)*∂x(v))dx, TrialFESpace(V_hdiv), V_e_2)
    BXx_hdiv2_e_2 = assemble_matrix((u, v) -> ∫(x(u)*∂x(v))dx, TrialFESpace(V_hdiv), V_e_2)
else
    BXz_hdiv1_e_2 = -assemble_matrix((u, v) -> ∫(∂z(z(u))*v)dx, TrialFESpace(V_hdiv), V_e_2)
    BXz_hdiv2_e_2 = -assemble_matrix((u, v) -> ∫(∂z(x(u))*v)dx, TrialFESpace(V_hdiv), V_e_2)
    BXx_hdiv1_e_2 = -assemble_matrix((u, v) -> ∫(∂x(z(u))*v)dx, TrialFESpace(V_hdiv), V_e_2)
    BXx_hdiv2_e_2 = -assemble_matrix((u, v) -> ∫(∂x(x(u))*v)dx, TrialFESpace(V_hdiv), V_e_2)
end
# check validity
BXdiv_hdiv_l2 = assemble_matrix((u, v) -> ∫(divergence(u)*v)dx, TrialFESpace(V_hdiv), V_l2)
maximum(abs.(BXdiv_hdiv_l2 - BXz_hdiv1_l2 - BXx_hdiv2_l2))

M_l2 = kron(MΩ_l2, MX_l2)
M_e_1 = kron(MΩ_e_1, MX_e_1)
M_e_2 = kron(MΩ_e_2, MX_e_2)
M_hdiv = kron(MΩ_hdiv, MX_hdiv)
M = blockdiag(M_l2, M_e_1, M_e_2, M_hdiv)

_B1 = kron(BΩz_hdiv1_l2, BXz_hdiv1_l2) + kron(BΩx_hdiv2_l2, BXx_hdiv2_l2)
_B2 = kron(BΩz_hdiv1_e_1, BXz_hdiv1_e_1) + kron(BΩx_hdiv1_e_1, BXx_hdiv1_e_1) + kron(BΩz_hdiv2_e_1, BXz_hdiv2_e_1) + kron(BΩx_hdiv2_e_1, BXx_hdiv2_e_1)
_B3 = kron(BΩz_hdiv1_e_2, BXz_hdiv1_e_2) + kron(BΩx_hdiv1_e_2, BXx_hdiv1_e_2) + kron(BΩz_hdiv2_e_2, BXz_hdiv2_e_2) + kron(BΩx_hdiv2_e_2, BXx_hdiv2_e_2)
_B = vcat(_B1, _B2, _B3) 
dupl(x) = (x, x)
B = [spzeros(dupl(size(M_l2, 1) + size(M_e_1, 1)+ size(M_e_2, 1))...) _B
    -transpose(_B) spzeros(dupl(size(M_hdiv, 1))...)]

init_f(x) = exp(-100.0*(x[1]^2 + x[2]^2)) #heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> init_f(VectorValue(x, y)))
b_x0 = assemble_vector(v -> ∫(init_f*v)dx, V_l2)
b0 = zeros(size(M, 1)); b0[1:length(b_x0)] .= M_l2 \b_x0
mass_v = assemble_vector(v -> ∫(1*v)dx, V_l2)

sol = copy(b0)
Δt = 0.01
A = (M/Δt + 0.5*B)
A_LU = lu(A)
eigs = eigen([quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D()), sh2 in SH.spherical_harmonics(N, EPMAfem.Dimensions._2D())] .|> x -> round(x, digits=14)).values

anim = @animate for i in 1:100
    @show i
    rhs = ((M*sol)./Δt - 0.5.*B*sol)
    ldiv!(sol, A_LU, rhs)

    f = interpolable(FEFunction(V_l2, sol[1:V_l2.nfree]))
    @show dot(mass_v, sol[1:V_l2.nfree])
    p1 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f(VectorValue(x, y)), aspect_ratio=:equal)
    # f_e_1 = interpolable(FEFunction(V_e_1, sol[V_l2.nfree+1:V_l2.nfree+V_e_1.nfree]))
    # p2 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f_e_1(VectorValue(x, y)), aspect_ratio=:equal)
    # f_e_2 = interpolable(FEFunction(V_e_2, sol[V_l2.nfree+V_e_1.nfree+1:V_l2.nfree+V_e_1.nfree+V_e_2.nfree]))
    # p3 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f_e_2(VectorValue(x, y)), aspect_ratio=:equal)
    # f_hdiv = interpolable(FEFunction(V_hdiv, sol[V_l2.nfree+V_e_1.nfree+V_e_2.nfree+1:V_l2.nfree+V_e_1.nfree+V_e_2.nfree+V_hdiv.nfree]))
    # p4 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f_hdiv(VectorValue(x, y))[1], aspect_ratio=:equal)
    # p5 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f_hdiv(VectorValue(x, y))[2], aspect_ratio=:equal)

    # plot(p1, p4, p5, p2, p3)
    # for λ ∈ eigs
    #     plot!(i.*Δt.*λ .* sin.(0:0.01:2π), i.*Δt.*λ.*cos.(0:0.01:2π), label=nothing)
    # end
end
gif(anim)
