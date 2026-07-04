
function interpolable(f::CellField)
    interp = Gridap.CellData.Interpolable(f; searchmethod=Gridap.CellData.KDTreeSearch(; num_nearest_vertices=5))
    rand_point = VectorValue(0.0)
    cache = Gridap.Arrays.return_cache(interp, rand_point)
    return x -> Gridap.Arrays.evaluate!(cache, interp, x)
end


model = CartesianDiscreteModel((-1.5, 1.5), (60))
V_l2 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 5), conformity=:L2)
V_h1 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 6), conformity=:H1)
dx = Measure(Triangulation(model), 50)

mom_0 = [sh for sh in SH.spherical_harmonics(3, EPMAfem.Dimensions._1D()) if sh.degree==0]
mom_1 = [sh for sh in SH.spherical_harmonics(3, EPMAfem.Dimensions._1D()) if sh.degree==1]
mom_2 = [sh for sh in SH.spherical_harmonics(3, EPMAfem.Dimensions._1D()) if sh.degree==2]
mom_3 = [sh for sh in SH.spherical_harmonics(3, EPMAfem.Dimensions._1D()) if sh.degree==3]

quad = SH.lebedev_quadrature_max()

M_Ω0 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_0, sh2 in mom_0] .|> x -> round(x, digits=14)
M_Ω1 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_1] .|> x -> round(x, digits=14)
M_Ω2 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_2, sh2 in mom_2] .|> x -> round(x, digits=14)
M_Ω3 = [quad(Ω -> sh1(Ω)*sh2(Ω)) for sh1 in mom_3, sh2 in mom_3] .|> x -> round(x, digits=14)

M_X0 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_l2), V_l2)
M_X1 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_h1), V_h1)
M_X2 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_l2), V_l2)
M_X3 = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V_h1), V_h1)

∂x(u) = dot(VectorValue(1.0), ∇(u))
∂y(u) = dot(VectorValue(0.0, 1.0), ∇(u))

Bx_X0X1 = assemble_matrix((u, v) -> ∫(∂x(u) * v)dx, TrialFESpace(V_h1), V_l2)
Bx_X1X2 = assemble_matrix((u, v) -> ∫(u * ∂x(v))dx, TrialFESpace(V_l2), V_h1)
Bx_X2X3 = assemble_matrix((u, v) -> ∫(∂x(u) * v)dx, TrialFESpace(V_h1), V_l2)

# By_X0X1 = assemble_matrix((u, v) -> ∫(∂y(u) * v)dx, TrialFESpace(V_h1), V_l2)
# By_X1X2 = assemble_matrix((u, v) -> ∫(u * ∂y(v))dx, TrialFESpace(V_l2), V_h1)
# By_X2X3 = assemble_matrix((u, v) -> ∫(∂y(u) * v)dx, TrialFESpace(V_h1), V_l2)

Bx_Ω0Ω1 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_0, sh2 in mom_1] .|> x -> round(x, digits=14)
Bx_Ω1Ω2 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2] .|> x -> round(x, digits=14)
Bx_Ω2Ω3 = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2, sh2 in mom_3] .|> x -> round(x, digits=14)

By_Ω0Ω1 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_0, sh2 in mom_1] .|> x -> round(x, digits=14)
By_Ω1Ω2 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_1, sh2 in mom_2] .|> x -> round(x, digits=14)
By_Ω2Ω3 = [quad(Ω -> Ω[2]*sh1(Ω)*sh2(Ω)) for sh1 in mom_2, sh2 in mom_3] .|> x -> round(x, digits=14)

Bx = [quad(Ω -> Ω[1]*sh1(Ω)*sh2(Ω)) for sh1 in SH.spherical_harmonics(1, EPMAfem.Dimensions._1D()), sh2 in SH.spherical_harmonics(1, EPMAfem.Dimensions._1D())] .|> x -> round(x, digits=14)

# P3
M, B = blockdiag([kron(M_X0, M_Ω0), kron(M_X1, M_Ω1), kron(M_X2, M_Ω2), kron(M_X3, M_Ω3)], [kron(Bx_X0X1, Bx_Ω0Ω1) + kron(By_X0X1, By_Ω0Ω1), kron(Bx_X1X2, Bx_Ω1Ω2) + kron(By_X1X2, By_Ω1Ω2), kron(Bx_X2X3, Bx_Ω2Ω3) + kron(By_X2X3, By_Ω2Ω3)])
# P1
M, B = blockdiag([kron(M_X0, M_Ω0), kron(M_X1, M_Ω1)], [kron(Bx_X0X1, Bx_Ω0Ω1)])

params = (C=1.0, σ=100.0)
radius(x) = sqrt(x[1]^2)
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
    plot(-1.5:0.001:1.5, x -> f(VectorValue(x)))
    # for λ ∈ eigs
    #     vline!([i.*Δt.*λ])
    #     # plot!(i.*Δt.*λ .* sin.(0:0.01:2π), i.*Δt.*λ.*cos.(0:0.01:2π), label=nothing)
    # end

    # plot!()

    # p2 = heatmap(-1:0.01:1, -1:0.01:1, (x, y) -> f_exact(i*Δt, norm((x, y))), aspect_ratio=:equal)
    # for λ ∈ eigs
    #     plot!(i.*Δt.*λ .* sin.(0:0.01:2π), i.*Δt.*λ.*cos.(0:0.01:2π), label=nothing)
    # end

    # plot(p1, p2)
    # plot(-1:0.01:1, x -> f(VectorValue(x, 0.0)))
end
gif(anim)





c = 1

heaviside(ξ) = ξ > 0 ? 1.0 : 0.0
delta(ξ) = -1e-2 <= ξ <= 1e-2  ? 1.0 : 0.0
safe_sqrt(x) = x < 0 ? NaN : sqrt(x)

@gif for t in 0:0.01:1
    plot(-1:0.001:1, x -> 1/(2*c) * heaviside(c*t - x), label="1D")
    plot!(0:0.001:1, x -> 1/(2*c*π * safe_sqrt(c^2*t^2 - x^2)) * heaviside(c*t - x), label="2D")
    plot!(-1:0.001:1, x -> 1/(4π)*delta(t - x/c), label="3D")
    ylims!(-0.1, 0.7)
end
