using Gridap
using Plots
using SparseArrays
include("../scripts/grid_gen.jl")

grid_gen_2D((-1.5, 1.5, -1.5, 1.5); min_res=0.1, max_res=0.1, filepath="/tmp/tmp_msh1.msh")

model = let
    # model = CartesianDiscreteModel((-1.5, 1.5, -1.5, 1.5), (20, 20))
    model = DiscreteModelFromFile("/tmp/tmp_msh1.msh")
    Ω = Triangulation(model)
    dx = Measure(Ω, 20)
    V0 = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
    (model=model, Ω=Ω, dx=dx, V0=V0)
end

function assemble_problem_L2H1(model)
    V0 = model.V0
    V1x = TestFESpace(model.model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
    V1y = TestFESpace(model.model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)

    V = MultiFieldFESpace([V0, V1x, V1y])

    dx = model.dx

    m((u0, u1x, u1y), (v0, v1x, v1y)) = ∫(u0*v0 + u1x*v1x + u1y*v1y)dx
    b((u0, u1x, u1y), (v0, v1x, v1y)) = ∫(-u0*(dot(VectorValue(1/sqrt(3), 0), ∇(v1x))
                                            + dot(VectorValue(0, 1/sqrt(3)), ∇(v1y)))
                                        +v0*(dot(VectorValue(1/sqrt(3), 0), ∇(u1x)) 
                                            + dot(VectorValue(0, 1/sqrt(3)), ∇(u1y))))dx

    M = assemble_matrix(m, MultiFieldFESpace(TrialFESpace.(V)), V)
    B = assemble_matrix(b, MultiFieldFESpace(TrialFESpace.(V)), V)
    return (M=M, B=B), (V0, V1x, V1y)
end

function assemble_problem_L2Hdiv(model)
    V0 = model.V0
    V1 = TestFESpace(model.model, ReferenceFE(raviart_thomas, Float64, 0), conformity=:Hdiv)

    V = MultiFieldFESpace([V0, V1])

    dx = model.dx


    m((u0, u1), (v0, v1)) = ∫(u0*v0 + dot(u1, v1))dx
    b((u0, u1), (v0, v1)) = ∫(-u0*(1/sqrt(3))*divergence(v1)
                              +v0*(1/sqrt(3))*divergence(u1))dx

    M = assemble_matrix(m, MultiFieldFESpace(TrialFESpace.(V)), V)
    B = assemble_matrix(b, MultiFieldFESpace(TrialFESpace.(V)), V)
    return (M=M, B=B), (V0, V1)
end

function discretize_initial_condition(f, model)
    V0 = model.V0
    dx = model.dx
    A = assemble_matrix((u, v) -> ∫(u*v)dx, TrialFESpace(V0), V0)
    b_init = assemble_vector(v -> ∫(f*v)dx, V0)
    return A\b_init
end

σ = 0.1 #0.03
init_f(x) = 1/(4π*σ^2)*exp(-(x[1]*x[1]+x[2]*x[2])/(4*σ^2)) # from (https://doi.org/10.1051/m2an/2022090)

problem, (V0, V1x, V1y) = assemble_problem_L2H1(model)
problem2, (V0, V1) = assemble_problem_L2Hdiv(model)
initial_value = [discretize_initial_condition(init_f, model)..., zeros(V1x.nfree)..., zeros(V1y.nfree)...]
initial_value2 = [discretize_initial_condition(init_f, model)..., zeros(V1.nfree)...]

function step(problem, Δt, u)
    M, B = problem.M, problem.B
    rhs = (M*u) ./ Δt .- (B*u) .* 0.5
    return (M/Δt + 0.5*B)\rhs
end

function interpolable(values, V_zero)
    f = FEFunction(V_zero, values[1:V_zero.nfree])
    cache = Gridap.Arrays.return_cache(f, VectorValue(0.0, 0.0))
    x -> Gridap.evaluate!(cache, f, x)
end


sol = copy(initial_value)
sol2 = copy(initial_value2)
Δt = 0.01
@gif for i in 1:100
    @show i
    sol .= step(problem, Δt, sol)
    sol2 .= step(problem2, Δt, sol2)
    f = interpolable(sol, model.V0)
    f2 = interpolable(sol2, model.V0)
    p = heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> f(VectorValue(x, y)), aspect_ratio=:equal)
    p2 = heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> f2(VectorValue(x, y)), aspect_ratio=:equal)
    plot(p, p2)

    # plot(-1:0.005:1, x -> f(VectorValue(0.0, x)))
    # plot!(-1:0.005:1, x -> f2(VectorValue(0.0, x)))
end


plot(-1:0.01:1, x -> f(VectorValue(0.0, x)))
plot!(-1:0.01:1, x -> f2(VectorValue(0.0, x)))
