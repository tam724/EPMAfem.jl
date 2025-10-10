@concrete struct DiscretePNProblem
    model
    arch

    # energy (these will always live on the cpu)
    s
    τ
    σ

    # (might be moved to gpu)
    space_discretization
    direction_discretization
end

architecture(problem::DiscretePNProblem) = problem.arch
n_basis(problem::DiscretePNProblem) = n_basis(problem.model)
n_sums(problem::DiscretePNProblem) = (nd = length(dimensions(problem.model)), ne = size(problem.s, 1), nσ = size(problem.σ, 2))

Base.show(io::IO, p::DiscretePNProblem) = print(io, "PNProblem [$(n_basis(p)) and $(n_sums(p))]")
Base.show(io::IO, ::MIME"text/plain", p::DiscretePNProblem) = show(io, p)

space_matrices(problem::DiscretePNProblem) = problem.space_discretization.ρp, problem.space_discretization.ρm, problem.space_discretization.∂p, problem.space_discretization.∇pm
lazy_space_matrices(problem::DiscretePNProblem) = lazy_space_matrices(problem.space_discretization)

direction_matrices(problem::DiscretePNProblem) = problem.direction_discretization.Ip, problem.direction_discretization.Im, problem.direction_discretization.kp, problem.direction_discretization.km, problem.direction_discretization.absΩp, problem.direction_discretization.Ωpm
lazy_direction_matrices(problem::DiscretePNProblem) = lazy_direction_matrices(problem.direction_discretization)

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
        nonzeros(problem.space_discretization.ρp[i]) .= nonzeros(upd_problem.ρp_tens.skeleton) |> arch
        Sparse3Tensor.project!(upd_problem.ρm_tens, @view(ρs[i, :]))
        nonzeros(problem.space_discretization.ρm[i]) .= nonzeros(upd_problem.ρm_tens.skeleton) |> arch
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

    ρp, ρm, ∂p, ∇pm = space_matrices(problem)
    Ip, Im, kp, km, absΩp, Ωpm = direction_matrices(problem)

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
            a -= dot(ρp[ie] * (problem.s[ie, i⁺1].*ψpi⁺1 .- problem.s[ie, i].*ψpi) * Ip, ϕp⁺½)
            a -= dot(ρm[ie] * (problem.s[ie, i⁺1].*ψmi⁺1 .- problem.s[ie, i].*ψmi) * Im, ϕm⁺½)

            a += Δϵ*dot(ρp[ie] * (problem.τ[ie, i].*ψpi .+ problem.τ[ie, i⁺1].*ψpi⁺1) * Ip, ϕp⁺½) / 2
            a += Δϵ*dot(ρm[ie] * (problem.τ[ie, i].*ψmi .+ problem.τ[ie, i⁺1].*ψmi⁺1) * Im, ϕm⁺½) / 2
            
            Wp = fill!(similar(kp[ie][1]), zero(T))
            Wm = fill!(similar(km[ie][1]), zero(T))
            for iσ in 1:nσ
                Wp .+= problem.σ[ie, iσ, i] .* kp[ie][iσ]
                Wm .+= problem.σ[ie, iσ, i] .* km[ie][iσ]
            end
            a -= Δϵ*dot(ρp[ie] * ψpi * Wp, ϕp⁺½) / 2
            a -= Δϵ*dot(ρm[ie] * ψmi * Wm, ϕm⁺½) / 2

            fill!(Wp, zero(T))
            fill!(Wm, zero(T))
            for iσ in 1:nσ
                Wp .+= problem.σ[ie, iσ, i⁺1] .* kp[ie][iσ]
                Wm .+= problem.σ[ie, iσ, i⁺1] .* km[ie][iσ]
            end
            a -= Δϵ*dot(ρp[ie] * ψpi⁺1 * Wp, ϕp⁺½) / 2
            a -= Δϵ*dot(ρm[ie] * ψmi⁺1 * Wm, ϕm⁺½) / 2
        end

        for id in 1:nd
            a -= Δϵ*dot(∇pm[id] * ((ψmi .+ ψmi⁺1) * Ωpm[id]), ϕp⁺½) / 2
            a += Δϵ*dot(transpose(∇pm[id]) * ((ψpi .+ ψpi⁺1) * transpose(Ωpm[id])), ϕm⁺½) / 2
            a += Δϵ*dot(∂p[id] * ((ψpi .+ ψpi⁺1) * absΩp[id]), ϕp⁺½) / 2
        end
    end
    return a
end


function system_matrix(problem::DiscretePNProblem, ϵ_idx)
    ns = EPMAfem.n_sums(problem)
    ρp, ρm, ∂p, ∇pm = lazy_space_matrices(problem)
    Ip, Im, kp, km, absΩp, Ωpm = lazy_direction_matrices(problem)

    Ikp(i) = materialize(LazyScalar(problem.τ[i, ϵ_idx])*Ip - sum(LazyScalar(problem.σ[i, j, ϵ_idx])*kp[i][j] for j in 1:ns.nσ))
    Ikm(i) = materialize(LazyScalar(problem.τ[i, ϵ_idx])*Im - sum(LazyScalar(problem.σ[i, j, ϵ_idx])*km[i][j] for j in 1:ns.nσ))

    A = sum(kron_AXB(ρp[i], Ikp(i)) for i in 1:ns.ne) + sum(kron_AXB(∂p[i], absΩp[i]) for i in 1:ns.nd)
    C = sum(kron_AXB(ρm[i], Ikm(i)) for i in 1:ns.ne)
    B = sum(kron_AXB(∇pm[i], Ωpm[i]) for i in 1:ns.nd)
    return [A B
    -transpose(B) C]
end

function mass_matrix(problem::DiscretePNProblem, ϵ_idx)
    ns = EPMAfem.n_sums(problem)
    nb = EPMAfem.n_basis(problem)
    ρp, ρm, ∂p, ∇pm = lazy_space_matrices(problem)
    Ip, Im, kp, km, absΩp, Ωpm = lazy_direction_matrices(problem)

    A = sum(LazyScalar(problem.s[i, ϵ_idx])*kron_AXB(ρp[i], Ip) for i in 1:ns.ne)
    C = sum(LazyScalar(problem.s[i, ϵ_idx])*kron_AXB(ρm[i], Im) for i in 1:ns.ne)
    B = kron_AXB(lazy(zeros(nb.nx.p, nb.nx.m)), lazy(zeros(nb.nΩ.m, nb.nΩ.p))) # TODO for PNLazyMatrices (allow zero blocks)
    return [A B
    transpose(B) C] 
end
