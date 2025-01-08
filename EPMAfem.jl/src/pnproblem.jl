@concrete struct DiscretePNProblem <: AbstractDiscretePNProblem
    model
    arch

    # energy (these will always live on the cpu)
    s
    τ
    σ

    # space (might be moved to gpu)
    ρp
    ρm

    ∂p
    ∇pm

    # direction (might be moved to gpu)
    Ip
    Im
    kp
    km
    absΩp
    Ωpm
end

architecture(pnproblem::DiscretePNProblem) = pnproblem.arch
n_basis(pnproblem::DiscretePNProblem) = n_basis(pnproblem.model)
n_sums(pnproblem::DiscretePNProblem) = (nd = length(pnproblem.∇pm), ne = size(pnproblem.s, 1), nσ = size(pnproblem.σ, 2))

Base.show(io::IO, p::DiscretePNProblem) = print(io, "PNProblem [$(n_basis(p)) and $(n_sums(p))]")
Base.show(io::IO, ::MIME"text/plain", p::DiscretePNProblem) = show(io, p)


# this is more or less "activity tracking" for the derivative
@concrete struct UpdatableDiscretePNProblem
    problem
    ρp_tens
    ρm_tens

    n_parameters
end

n_parameters(upd_problem::UpdatableDiscretePNProblem) = upd_problem.n_parameters

Base.show(io::IO, p::UpdatableDiscretePNProblem) = print(io, "UpdateablePNProblem [$(n_basis(p.problem)) and $(n_sums(p.problem))]")
Base.show(io::IO, ::MIME"text/plain", p::UpdatableDiscretePNProblem) = show(io, p)

function update_problem!(upd_problem::UpdatableDiscretePNProblem, ρs)
    problem = upd_problem.problem
    arch = problem.arch
    @assert size(ρs) == n_parameters(upd_problem)
    for i in 1:n_parameters(upd_problem)[1]
        Sparse3Tensor.project!(upd_problem.ρp_tens, @view(ρs[i, :]))
        nonzeros(problem.ρp[i]) .= nonzeros(upd_problem.ρp_tens.skeleton) |> arch
        Sparse3Tensor.project!(upd_problem.ρm_tens, @view(ρs[i, :]))
        nonzeros(problem.ρm[i]) .= nonzeros(upd_problem.ρm_tens.skeleton) |> arch
    end
end

# mainly used this for debugging.. 
function (problem::DiscretePNProblem)(ψ::AbstractDiscretePNSolution, ϕ::AbstractDiscretePNSolution)
    @assert !_is_adjoint_solution(ψ) && _is_adjoint_solution(ϕ)
    arch = problem.arch
    T = base_type(arch)
    nd, ne, nσ = n_sums(problem)

    Δϵ = step(energy_model(problem.model))
    a = zero(T)

    for (idx⁺½, ϕ⁺½) in ϕ
        if is_first(idx⁺½) continue end # where ϕ is initialized to zero anyways
        ϕp⁺½ = pview(ϕ⁺½, problem.model)
        ϕm⁺½ = mview(ϕ⁺½, problem.model)
        
        i = minus½(idx⁺½)
        ψpi = pview(ψ[i], problem.model)
        ψmi = mview(ψ[i], problem.model)

        i⁺1 = plus½(idx⁺½)
        ψpi⁺1 = pview(ψ[i⁺1], problem.model)
        ψmi⁺1 = mview(ψ[i⁺1], problem.model)

        for ie in 1:ne
            a -= dot(problem.ρp[ie] * (problem.s[ie, i⁺1].*ψpi⁺1 .- problem.s[ie, i].*ψpi) * problem.Ip, ϕp⁺½)
            a -= dot(problem.ρm[ie] * (problem.s[ie, i⁺1].*ψmi⁺1 .- problem.s[ie, i].*ψmi) * problem.Im, ϕm⁺½)

            a += Δϵ*dot(problem.ρp[ie] * (problem.τ[ie, i].*ψpi .+ problem.τ[ie, i⁺1].*ψpi⁺1) * problem.Ip, ϕp⁺½) / 2
            a += Δϵ*dot(problem.ρm[ie] * (problem.τ[ie, i].*ψmi .+ problem.τ[ie, i⁺1].*ψmi⁺1) * problem.Im, ϕm⁺½) / 2
            
            kp = fill!(similar(problem.kp[ie][1]), zero(T))
            km = fill!(similar(problem.km[ie][1]), zero(T))
            for iσ in 1:nσ
                kp .+= problem.σ[ie, iσ, i] .* problem.kp[ie][iσ]
                km .+= problem.σ[ie, iσ, i] .* problem.km[ie][iσ]
            end
            a -= Δϵ*dot(problem.ρp[ie] * ψpi * kp, ϕp⁺½) / 2
            a -= Δϵ*dot(problem.ρm[ie] * ψmi * km, ϕm⁺½) / 2

            fill!(kp, zero(T))
            fill!(km, zero(T))
            for iσ in 1:nσ
                kp .+= problem.σ[ie, iσ, i⁺1] .* problem.kp[ie][iσ]
                km .+= problem.σ[ie, iσ, i⁺1] .* problem.km[ie][iσ]
            end
            a -= Δϵ*dot(problem.ρp[ie] * ψpi⁺1 * kp, ϕp⁺½) / 2
            a -= Δϵ*dot(problem.ρm[ie] * ψmi⁺1 * km, ϕm⁺½) / 2
        end

        for id in 1:nd
            a -= Δϵ*dot(problem.∇pm[id] * ((ψmi .+ ψmi⁺1) * problem.Ωpm[id]), ϕp⁺½) / 2
            a += Δϵ*dot(transpose(problem.∇pm[id]) * ((ψpi .+ ψpi⁺1) * transpose(problem.Ωpm[id])), ϕm⁺½) / 2
            a += Δϵ*dot(problem.∂p[id] * ((ψpi .+ ψpi⁺1) * problem.absΩp[id]), ϕp⁺½) / 2
        end
    end
    return a
end