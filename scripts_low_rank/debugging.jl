using Revise
using EPMAfem
using EPMAfem.Gridap
using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using LinearAlgebra

function OnlyEnergyModel(energy_model)
    nϵ = length(energy_model)
    return EPMAfem.DiscretePNModel(
        nothing,
        energy_model,
        nothing,
        (nϵ = nϵ, nx=(p=1, m=1), nΩ=(p=1, m=1))
    )
end
EPMAfem.Dimensions.dimensionality(::Nothing) = EPMAfem.Dimensions._1D()

EPMAfem.@concrete struct DummyDiscretePNSolution <: EPMAfem.AbstractDiscretePNSolution
    model
    adjoint
    cache
end

_is_adjoint_solution(sol::DummyDiscretePNSolution) = sol.adjoint

function Base.iterate(sol::DummyDiscretePNSolution)
    idx = EPMAfem.first_index(sol.model.energy_mdl, sol.adjoint)
    return idx => [sin(EPMAfem.ϵ(idx)), 0.0], idx
end

function Base.iterate(sol::DummyDiscretePNSolution, idx)
    idx_next = EPMAfem.next(idx)
    if isnothing(idx_next) return nothing end
    return idx_next => [sin(EPMAfem.ϵ(idx_next)), 0.0], idx_next
end 

EPMAfem.SpaceModels.eval_basis(::Nothing, μ::Function) = (p=[1.0], m=[1.0])
EPMAfem.SphericalHarmonicsModels.eval_basis(::Nothing, μ::Function) = (p=[1.0], m=[1.0])

model = OnlyEnergyModel(0:0.000001:π);
sol = DummyDiscretePNSolution(model, true, nothing);
ϵ0 = rand()
probe = EPMAfem.PNProbe(model, EPMAfem.cpu(), x=x->1.0, Ω=Ω->1.0, ϵ=ϵ0);

a = EPMAfem.interpolable(probe, sol)
b = sin(ϵ0)

a - b

probe2 = EPMAfem.PNProbe(model, EPMAfem.cpu(), x=x->1.0, Ω=Ω->1.0, ϵ=ϵ->1.0);
a = probe2(sol)
b = EPMAfem.hquadrature(x -> sin(x), 0.0, π)[1]
a - b 



space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5, -1.5, 1.5), (10, 10)))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(11, 2)
model = EPMAfem.DiscretePNModel(space_model, 0:0.01:1.0, direction_model)
problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())

system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
system_lr = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=4, m=4));

EPMAfem.n_basis(problem)

nb = EPMAfem.n_basis(problem)
initial_condition = EPMAfem.allocate_vec(EPMAfem.cpu(), nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m)

ψ0p_test = rand(nb.nx.p) .* rand(nb.nΩ.p)' + rand(nb.nx.p) .* rand(nb.nΩ.p)' + rand(nb.nx.p) .* rand(nb.nΩ.p)'
ψ0m_test = rand(nb.nx.m) .* rand(nb.nΩ.m)' + rand(nb.nx.m) .* rand(nb.nΩ.m)' + rand(nb.nx.m) .* rand(nb.nΩ.m)'
ψ0p, ψ0m = EPMAfem.pmview(initial_condition, model)
copy!(ψ0p, ψ0p_test)
copy!(ψ0m, ψ0m_test)

initial_solution = EPMAfem.allocate_solution_vector(system_lr)
EPMAfem.initialize!(initial_solution, model, initial_condition)

ψ0p_tt, ψ0m_tt = EPMAfem.pmview(initial_solution, model)
heatmap((collect(ψ0p_tt) .- ψ0p_test) .|> abs |> maximum)
heatmap(collect(ψ0m_tt) .- ψ0m_test )
