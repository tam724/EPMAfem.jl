@concrete struct OnlyEnergyModel <: AbstractPNModel
    energy_mdl
    nϵ::Int64
end

function OnlyEnergyModel(energy_model)
    nϵ = length(energy_model)
    return OnlyEnergyModel(
        energy_model,
        nϵ
    )
end

function EPMAfem.n_basis(model::OnlyEnergyModel)
    return (nϵ = model.nϵ, nx=(p=1, m=1), nΩ=(p=1, m=1))
end

function EPMAfem.energy_model(model::OnlyEnergyModel)
    return model.energy_mdl
end

@concrete struct OnlyEnergyEquations
    params
end

# function OnlyEnergyEquations()
#     return OnlyEnergyEquations((α = 100.0, β = 0.7, s=1.0, τ=1.0, τ2=1.0))
# end

function EPMAfem.mass_concentrations(eq::OnlyEnergyEquations, e)
    return eq.params.ρ[e]
end

function EPMAfem.stopping_power(eq::OnlyEnergyEquations, e, ϵ)
    return exp(eq.params.s[e]*ϵ)
end

function ∂stopping_power(eq::OnlyEnergyEquations, e, ϵ)
    return eq.params.s[e]*exp(eq.params.s[e]*ϵ)
end

function EPMAfem.absorption_coefficient(eq::OnlyEnergyEquations, e, ϵ)
    return eq.params.τ2[e]*cos(eq.params.τ[e]*ϵ)
end

function EPMAfem.scattering_coefficient(eq::OnlyEnergyEquations, e, i, ϵ)
    return eq.params.σ2[e, i]*sin(eq.params.σ[e, i]*ϵ)
end

function scattering_coefficient2(eq::OnlyEnergyEquations, e, i)
    return eq.params.k[e, i]
end

function EPMAfem.number_of_scatterings(eq::OnlyEnergyEquations)
    @assert size(eq.params.σ, 2) == size(eq.params.σ2, 2) == size(eq.params.k, 2)
    return size(eq.params.σ, 2)
end

function EPMAfem.number_of_elements(eq::OnlyEnergyEquations)
    @assert length(eq.params.ρ) == length(eq.params.s) == length(eq.params.τ) == size(eq.params.σ, 1) == size(eq.params.σ2, 1) == size(eq.params.k, 1)
    return length(eq.params.ρ)
end

function source(eq::OnlyEnergyEquations, ϵ)
    # negative because on the left side of the equation
    α = eq.params.α
    β = eq.params.β
    return -exp(-α*(ϵ-β)^2)
end

function EPMAfem.discretize_problem(eq::OnlyEnergyEquations, mdl::OnlyEnergyModel, arch::PNArchitecture)
    T = base_type(arch)

    ϵs = energy_model(mdl)

    n_elem = number_of_elements(eq)
    n_scat = number_of_scatterings(eq)

    s = Matrix{T}([stopping_power(eq, e, ϵ) for e in 1:n_elem, ϵ ∈ ϵs])
    τ = Matrix{T}([absorption_coefficient(eq, e, ϵ) for e in 1:n_elem, ϵ ∈ ϵs])
    σ = Array{T}([scattering_coefficient(eq, e, i, ϵ) for e in 1:n_elem, i in 1:n_scat, ϵ ∈ ϵs])

    ρp_tens = nothing
    ρp = [mass_concentrations(eq, e)*ones(1, 1) for e in 1:n_elem] |> arch
    ρm_tens = nothing
    ρm = [Diagonal(mass_concentrations(eq, e)*ones(1)) for e in 1:n_elem] |> arch

    ∂p = [zeros(1, 1) for _ in 1:1] |> arch
    ∇pm = [zeros(1, 1) for _ in 1:1] |> arch 

    kp = [[Diagonal(ones(1))*scattering_coefficient2(eq, e, i) for i in 1:n_scat] for e in 1:n_elem] |> arch
    km = [[Diagonal(ones(1))*scattering_coefficient2(eq, e, i) for i in 1:n_scat] for e in 1:n_elem] |> arch

    Ip = Diagonal(ones(1)) |> arch
    Im = Diagonal(ones(1)) |> arch

    absΩp = [zeros(1, 1) for _ in 1:1] |> arch
    Ωpm = [zeros(1, 1) for _ in 1:1] |> arch

    DiscretePNProblem(mdl, arch, s, τ, σ, ρp, ρp_tens, ρm, ρm_tens, ∂p, ∇pm, Ip, Im, kp, km, absΩp, Ωpm)
end

function EPMAfem.discretize_rhs(eq::OnlyEnergyEquations, mdl::OnlyEnergyModel, arch::PNArchitecture)
    T = base_type(arch)

    ϵs = energy_model(mdl)

    ## assemble excitation 
    gϵ = Vector{T}([source(eq, ϵ) for ϵ ∈ ϵs])
    gxp = ones(1, 1) |> arch
    gΩp = ones(1, 1) |> arch
    return Rank1DiscretePNVector(false, mdl, arch, gϵ, gxp, gΩp)
end

function discretize_adjoint_rhs(eq::OnlyEnergyEquations, mdl::OnlyEnergyModel, arch::PNArchitecture)
    T = base_type(arch)

    ϵs = energy_model(mdl)

    ## assemble excitation 
    gϵ = Vector{T}([source(eq, ϵ) for ϵ ∈ ϵs])
    gxp = ones(1, 1) |> arch
    gΩp = ones(1, 1) |> arch
    return Rank1DiscretePNVector(true, mdl, arch, gϵ, gxp, gΩp)
end

## some exact solutions (computed via mathematica)
function exact_solution(eq::OnlyEnergyEquations, use_adjoint)
    if number_of_elements(eq) != 1
        throw(ArgumentError("Exact solution is not implemented for this case"))
    end
    if number_of_scatterings(eq) != 1
        throw(ArgumentError("Exact solution is not implemented for this case"))
    end
    α = eq.params.α
    β = eq.params.β
    τ = eq.params.τ[1]
    ρ = eq.params.ρ[1]
    τ2 = eq.params.τ2[1]
    s = eq.params.s[1]
    σ = eq.params.σ[1, 1]
    σ2 = eq.params.σ2[1, 1]
    k = eq.params.k[1, 1]
    if !isone(k)
        throw(ArgumentError("Exact solution is not implemented for this case"))
    end
    if !use_adjoint && iszero(s) && iszero(τ) && iszero(σ2) && iszero(σ)
        #there is a closed form solution
        return ϵ -> (exp(ϵ * τ2 - β * τ2 + τ2^2 / (4 * α)) * sqrt(π) * (erf((2 * α * (1 - β) + τ2) / (2 * sqrt(α))) - erf((2 * α * (ϵ - β) + τ2) / (2 * sqrt(α))))) / (2 * sqrt(α) * ρ)
    elseif !use_adjoint
        @warn "using numerical quadrature"
        return function(ϵ)
            integrand1(x) = exp(-(s * x) + (σ * σ2 * cos(x * σ)) / (exp(s * x) * (s^2 + σ^2)) - (s * τ2 * cos(x * τ)) / (exp(s * x) * (s^2 + τ^2)) + (s * σ2 * sin(x * σ)) / (exp(s * x) * (s^2 + σ^2)) + (τ * τ2 * sin(x * τ)) / (exp(s * x) * (s^2 + τ^2)))
            integrand2(x) = -exp(-(s * x) - α * (-β + x)^2 + (-((σ * σ2 * cos(σ * x)) / (s^2 + σ^2)) - (s * σ2 * sin(σ * x)) / (s^2 + σ^2) + (s * τ2 * cos(τ * x) + exp(s * x) * s * (s^2 + τ^2) * x - τ * τ2 * sin(τ * x)) / (s^2 + τ^2)) / exp(s * x)) / ρ
            integral_result, error = hquadrature(integrand2, 1, ϵ)
            if error > 1e-8
                @warn "error in quadrature is $error"
            end
            return integrand1(ϵ) * integral_result
        end
    elseif use_adjoint && iszero(s) && iszero(τ) && iszero(σ2)
        return ϵ -> (exp(-(ϵ * τ2) + β * τ2 + τ2^2 / (4 * α)) * sqrt(π) * (erf((2 * α * (ϵ - β) - τ2) / (2 * sqrt(α))) - erf((-2 * α * β - τ2) / (2 * sqrt(α))))) / (2 * sqrt(α) * ρ)
    elseif use_adjoint
        @warn "using numerical quadrature"
        return function(ϵ)
            integrand1(x) = exp(-(σ * σ2 * cos(x * σ)) / (exp(s * x) * (s^2 + σ^2)) - (s * σ2 * sin(x * σ)) / (exp(s * x) * (s^2 + σ^2)) + (τ2 * (s * cos(x * τ) - τ * sin(x * τ))) / (exp(s * x) * (s^2 + τ^2)))
            integrand2(x) = exp(-(s * x) - α * (-β + x)^2 - (-(σ * σ2 * cos(σ * x)) / (s^2 + σ^2) - (s * σ2 * sin(σ * x)) / (s^2 + σ^2) + (τ2 * (s * cos(τ * x) - τ * sin(τ * x))) / (s^2 + τ^2)) / exp(s * x)) / ρ
            integral_result, error = hquadrature(integrand2, 0, ϵ)
            if error > 1e-8
                @warn "error in quadrature is $error"
            end
            return integrand1(ϵ) * integral_result
        end
    else
        throw(ArgumentError("Exact solution is not implemented for this case"))
    end
end

function diffeq_solution(eq::OnlyEnergyEquations, use_adjoint)
    ne = number_of_elements(eq)
    ns = number_of_scatterings(eq)

    if use_adjoint
        function fa(u, p, ϵ)
            ρ_tot = sum(e->mass_concentrations(eq, e)*stopping_power(eq, e, ϵ), 1:ne)
            du = (-sum(e->mass_concentrations(eq, e)*(absorption_coefficient(eq, e, ϵ) - sum(i -> scattering_coefficient(eq, e, i, ϵ)*scattering_coefficient2(eq, e, i), 1:ns)),1:ne)*u-source(eq, ϵ))/ρ_tot
            return du
        end
        problem = ODEProblem(fa, 0.0, (0.0, 1.0))
        sol = OrdinaryDiffEq.solve(problem, Tsit5(), saveat=range(0, 1, length=500), reltol=1e-10, abstol=1e-10, dtmax=0.01)
        return ϵ -> sol(ϵ)
    else
        function fna(u, p, t)
            ϵ = one(eltype(t)) - t # variable transormation
            ρ_tot = sum(e->mass_concentrations(eq, e)*stopping_power(eq, e, ϵ), 1:ne)
            du = (-sum(e->mass_concentrations(eq, e)*(-∂stopping_power(eq, e, ϵ) + absorption_coefficient(eq, e, ϵ) - sum(i -> scattering_coefficient(eq, e, i, ϵ)*scattering_coefficient2(eq, e, i), 1:ns)),1:ne)*u-source(eq, ϵ))/ρ_tot
            return du
        end
        problem = ODEProblem(fna, 0.0, (0.0, 1.0))
        sol = OrdinaryDiffEq.solve(problem, Tsit5(), saveat=range(0, 1, length=500), reltol=1e-10, abstol=1e-10, dtmax=0.01)
        # variable transformation
        return ϵ -> sol(one(eltype(ϵ)) - ϵ)
    end
end
