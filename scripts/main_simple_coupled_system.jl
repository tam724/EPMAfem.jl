using Revise
using EPMAfem
using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using LinearAlgebra
using Plots

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

function construct_problem(mdl::EPMAfem.DiscretePNModel, arch::EPMAfem.PNArchitecture)
    T = EPMAfem.base_type(arch)
    ϵs = EPMAfem.energy_model(mdl)

    n_elem = 1
    n_scat = 1

    s = Matrix{T}([one(T) for e in 1:n_elem, ϵ ∈ ϵs])
    τ = Matrix{T}([zero(T) for e in 1:n_elem, ϵ ∈ ϵs])
    σ = Array{T}([zero(T) for e in 1:n_elem, i in 1:n_scat, ϵ ∈ ϵs])

    ρp = [ones(T, 1, 1) for e in 1:n_elem] |> arch
    ρm = [Diagonal(ones(T, 1)) for e in 1:n_elem] |> arch

    ∂p = [zeros(1, 1) for _ in 1:1] |> arch
    ∇pm = [ones(1, 1) for _ in 1:1] |> arch 

    kp = [[Diagonal(zeros(1)) for i in 1:n_scat] for e in 1:n_elem] |> arch
    km = [[Diagonal(zeros(1)) for i in 1:n_scat] for e in 1:n_elem] |> arch

    Ip = Diagonal(ones(1)) |> arch
    Im = Diagonal(ones(1)) |> arch

    absΩp = [zeros(1, 1) for _ in 1:1] |> arch
    Ωpm = [ones(1, 1) for _ in 1:1] |> arch

    space_discretization = EPMAfem.SpaceDiscretization(EPMAfem.space_model(mdl), arch, ρp, ρm, ∂p, ∇pm)
    direction_discretization = EPMAfem.DirectionDiscretization(EPMAfem.direction_model(mdl), arch, Ip, Im, kp, km, absΩp, Ωpm)

    EPMAfem.DiscretePNProblem(mdl, arch, s, τ, σ, space_discretization, direction_discretization)
end

function construct_rhs(mdl, arch::EPMAfem.PNArchitecture)
    T = EPMAfem.base_type(arch)
    ϵs = EPMAfem.energy_model(mdl)

    ## assemble excitation 
    gϵ = Vector{T}([exp(-10*(ϵ - 0.8)^2) for ϵ ∈ ϵs])
    gxp = ones(1, 1) |> arch
    gΩp = ones(1, 1) |> arch
    return EPMAfem.Rank1DiscretePNVector(false, mdl, arch, gϵ, gxp, gΩp)
end


function compute(model, system, _rhs)
    sol = system * _rhs

    u = zeros(length(EPMAfem.energy_model(model)))
    v = zeros(length(EPMAfem.energy_model(model)))

    for (i, ψ) in sol
        ψp, ψm = EPMAfem.pmview(ψ, model)
        u[i] = only(ψp)
        v[i] = only(ψm)
    end
    return u, v
end

mdl = OnlyEnergyModel(-5:0.01:1);
pbl = construct_problem(mdl, EPMAfem.cpu())
rhs = construct_rhs(mdl, EPMAfem.cpu())

u, v = compute(mdl, EPMAfem.implicit_midpoint2(pbl, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!)), rhs)
u2, v2 = compute(mdl, EPMAfem.implicit_midpoint_dlr(pbl; max_rank=1), rhs)
u3, v3 = compute(mdl, EPMAfem.implicit_midpoint_dlr2(pbl; max_rank=1), rhs)

plotly()
plot(EPMAfem.energy_model(mdl), u)
plot!(EPMAfem.energy_model(mdl), v)
plot!(EPMAfem.energy_model(mdl), u2)
plot!(EPMAfem.energy_model(mdl), v2)
plot!(EPMAfem.energy_model(mdl), u3)
plot!(EPMAfem.energy_model(mdl), v3)
