

# func(x) = VectorValue(sinpi(4*x[1]), cospi(3*x[2]))

# A = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_hdiv), V_hdiv)
# b = assemble_vector((v) -> ∫(dot(func, v))dx, V_hdiv)

# f_hdiv = FEFunction(V_hdiv, A\b)

# M = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_helper_conf), V_helper_conf)
# b = assemble_vector(v -> ∫(x(f_hdiv)*v)dx, V_helper_conf)
# M_lumped = Diagonal(sum(M; dims=1)[:])

# f_h1 = FEFunction(V_h1, M\b) |> ∂x |> interpolable
# f_h1_lumped = FEFunction(V_h1, M_lumped\b) |> ∂x |> interpolable

# p1 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f_h1(VectorValue(x, y)), aspect_ratio=:equal)
# p2 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f_h1_lumped(VectorValue(x, y)), aspect_ratio=:equal)

# plot(p1, p2)

grid_gen_2D((-1, 1, -1, 1); min_res=0.5, max_res=0.5, filepath="/tmp/tmp_msh_coarse.msh")
model = DiscreteModelFromFile("/tmp/tmp_msh_coarse.msh")


V = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, 0), conformity=:Hdiv)
F = FEFunction(V, rand(V.nfree))

@gif for θ in 0:0.05:2π
    @show θ
    n = VectorValue(cos(θ), sin(θ))
    f = interpolable(dot(F, n))
    heatmap(-1:0.02:1, -1:0.02:1, (x, y) -> f(VectorValue(x, y)), aspect_ratio=:equal)
    plot!([0, n[1]], [0, n[2]], color=:black)
end

#start
grid_gen_2D((-1, 1, -1, 1); min_res=0.05, max_res=0.05, filepath="/tmp/tmp_msh_coarse.msh")
model = DiscreteModelFromFile("/tmp/tmp_msh_coarse.msh")

# model = CartesianDiscreteModel((-1, 1, -1, 1), (60, 60))
V_l2_0 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
V_l2_1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
# V_hdiv = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2, Float64}, 1), conformity=:H1)
V_hdiv = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, 0), conformity=:Hdiv)
V_h1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
V_helper = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)
V_helper_conf = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)

dx = Measure(Triangulation(model), 50)
quad = SH.lebedev_quadrature_max()

# build the mass matrices
mom_system_L2_0 = mom_system.L2[1:1]
mom_system_L2_1 = mom_system.L2[2:end]
MΩ0_0 = Diagonal(ones(length(mom_system_L2_0)))
MΩ0_1 = Diagonal(ones(length(mom_system_L2_1)))
MΩ1 = Diagonal(ones(length(mom_system.Hdiv)))
MΩ2 = Diagonal(ones(length(mom_system.H1)))

# build the coupling matrices
# Hdiv(1): trial function, L2(0): test function
# d/dx needs the first element of the Hdiv, is tested with L2
BΩx_1_0_0 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2[1], EPMAfem.Dimensions.Z()) for sh1 in mom_system_L2_0, sh2 in mom_system.Hdiv])
BΩx_1_0_1 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2[1], EPMAfem.Dimensions.Z()) for sh1 in mom_system_L2_1, sh2 in mom_system.Hdiv])

# check validity
maximum(abs.([SH.get_transport_coefficient(sh1, sh2[2], EPMAfem.Dimensions.Z()) for sh1 in mom_system.L2, sh2 in mom_system.Hdiv])) < 1e-15

# d/dy needs the second element of the Hdiv, is tested with L2
BΩy_1_0_0 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2[2], EPMAfem.Dimensions.X()) for sh1 in mom_system_L2_0, sh2 in mom_system.Hdiv])
BΩy_1_0_1 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2[2], EPMAfem.Dimensions.X()) for sh1 in mom_system_L2_1, sh2 in mom_system.Hdiv])

# check validity
maximum(abs.([SH.get_transport_coefficient(sh1, sh2[1], EPMAfem.Dimensions.X()) for sh1 in mom_system.L2, sh2 in mom_system.Hdiv])) < 1e-15

# Hdiv(1): trial function, H1(2): test function
# d/dx needs the second element of the Hdiv (is tested with the x-derivative of the H1)
BΩx_1_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2[2], EPMAfem.Dimensions.Z()) for sh1 in mom_system.H1, sh2 in mom_system.Hdiv])

maximum(abs.(Float64[SH.get_transport_coefficient(sh1, sh2[1], EPMAfem.Dimensions.Z()) for sh1 in mom_system.H1, sh2 in mom_system.Hdiv])) < 1e-14

# d/dy needs the first element of the Hdiv (is tested with the y-derivative of the H1)
BΩy_1_2 = sparse(Float64[SH.get_transport_coefficient(sh1, sh2[1], EPMAfem.Dimensions.X()) for sh1 in mom_system.H1, sh2 in mom_system.Hdiv])
maximum(abs.(Float64[SH.get_transport_coefficient(sh1, sh2[2], EPMAfem.Dimensions.X()) for sh1 in mom_system.H1, sh2 in mom_system.Hdiv])) < 1e-14

# same for space (mass matrices)
MX0_0 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_l2_0), V_l2_0)
MX0_1 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_l2_1), V_l2_1)
MX1 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_hdiv), V_hdiv)
MX2 = assemble_matrix((u, v) -> ∫(dot(u, v))dx, TrialFESpace(V_h1), V_h1)

# coupling matrices
∂x(u) = dot(VectorValue(1.0, 0.0), ∇(u))
∂y(u) = dot(VectorValue(0.0, 1.0), ∇(u))

x(u) = dot(VectorValue(1.0, 0.0), u)
y(u) = dot(VectorValue(0.0, 1.0), u)

# umweg: 
M_helper = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_helper), V_helper) |> EPMAfem.BlockDiagonals.BlockDiagonal{3}
M_helper_conf = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_helper_conf), V_helper_conf) 
# M_helper_conf = Diagonal(sum(M_helper_conf; dims=[1])[:])
inv_M_helper = sparse(LinearAlgebra.inv!(copy(M_helper)))

# project the first component of the Hdiv into an higher order L2 space (V_helper)
BXx_1_0_0 = assemble_matrix((u, v) -> ∫(∂x(x(u))*v)dx, TrialFESpace(V_hdiv), V_l2_0) 
BXy_1_0_0 = assemble_matrix((u, v) -> ∫(∂y(y(u))*v)dx, TrialFESpace(V_hdiv), V_l2_0)
# check validity
Bdiv_1_0_0 = assemble_matrix((u, v) -> ∫(divergence(u)*v)dx, TrialFESpace(V_hdiv), V_l2_0)
maximum(abs.(Bdiv_1_0_0 - BXx_1_0_0 - BXy_1_0_0))

# Bx1 = assemble_matrix((u, v) -> ∫(x(u)*v)dx, TrialFESpace(V_hdiv), V_helper_conf)
# Bx2 = assemble_matrix((u, v) -> ∫(y(u)*v)dx, TrialFESpace(V_hdiv), V_helper_conf)
# BXx_1_0_1 = assemble_matrix((u, v) -> ∫(∂x(u)*v)dx, TrialFESpace(V_helper_conf), V_l2_1) * (M_helper_conf \ Matrix(Bx1))
# BXx_1_0_1 = (BXx_1_0_1 .|> x -> round(x; digits=15))|> sparse
# BXy_1_0_1 = assemble_matrix((u, v) -> ∫(∂y(u)*v)dx, TrialFESpace(V_helper_conf), V_l2_1) * (M_helper_conf \ Matrix(Bx2))
# BXy_1_0_1 = (BXy_1_0_1 .|> x -> round(x; digits=15))|> sparse
BXx_1_0_1 = assemble_matrix((u, v) -> ∫(∂x(x(u))*v)dx, TrialFESpace(V_hdiv), V_l2_1) 
BXy_1_0_1 = assemble_matrix((u, v) -> ∫(∂y(y(u))*v)dx, TrialFESpace(V_hdiv), V_l2_1) 

# check validity
Bdiv_1_0_1 = assemble_matrix((u, v) -> ∫(divergence(u)*v)dx, TrialFESpace(V_hdiv), V_l2_1)
maximum(abs.(Bdiv_1_0_1 - BXx_1_0_1 - BXy_1_0_1))

BXx_1_2 = assemble_matrix((u, v) -> ∫(y(u)*∂x(v))dx, TrialFESpace(V_hdiv), V_h1)
BXy_1_2 = assemble_matrix((u, v) -> ∫(x(u)*∂y(v))dx, TrialFESpace(V_hdiv), V_h1)

M0_0 = kron(MΩ0_0, MX0_0)
M0_1 = kron(MΩ0_1, MX0_1)
M2 = kron(MΩ2, MX2)
M1 = kron(MΩ1, MX1)
M = blockdiag(M0_0, M0_1, M2, M1)
_B1_0 = kron(BΩx_1_0_0, BXx_1_0_0) + kron(BΩy_1_0_0, BXy_1_0_0)
_B1_1 = kron(BΩx_1_0_1, BXx_1_0_1) + kron(BΩy_1_0_1, BXy_1_0_1)
_B2 = -kron(BΩx_1_2, BXx_1_2) - kron(BΩy_1_2, BXy_1_2)
_B = vcat(_B1_0, _B1_1, _B2) 
B = [spzeros(size(M0_0, 1) + size(M0_1, 1) + size(M2, 1), size(M0_0, 1) + size(M0_1, 1) + size(M2, 1)) _B
-transpose(_B) spzeros(size(M1, 1), size(M1, 1))]

init_f(x) = exp(-100.0*(x[1]^2 + x[2]^2)) #heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> init_f(VectorValue(x, y)))
b_x0 = assemble_vector(v -> ∫(init_f*v)dx, V_l2_0)
b0 = zeros(size(M, 1)); b0[1:length(b_x0)] .= M0_0 \b_x0
mass_v = assemble_vector(v -> ∫(1*v)dx, V_l2_0)

sol = copy(b0)
Δt = 0.01
A = (M/Δt + 0.5*B)
A_LU = lu(A);
eigs = eigen([get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in spherical_harmonics(N), sh2 in spherical_harmonics(N)] .|> x -> round(x, digits=14)).values

anim = @animate for i in 1:100
    @show i
    rhs = ((M*sol)./Δt - 0.5.*B*sol)
    ldiv!(sol, A_LU, rhs)

    f = interpolable(FEFunction(V_l2_0, sol[1:V_l2_0.nfree]))
    @show dot(mass_v, sol[1:V_l2_0.nfree])
    p1 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f(VectorValue(x, y)), aspect_ratio=:equal)
    # for λ ∈ eigs
    #     plot!(i.*Δt.*λ .* sin.(0:0.01:2π), i.*Δt.*λ.*cos.(0:0.01:2π), label=nothing)
    # end
end
gif(anim)
