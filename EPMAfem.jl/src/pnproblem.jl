@concrete struct DiscretePNProblem <: AbstractDiscretePNProblem
    model
    arch

    # energy (these will always live on the cpu)
    s
    τ
    σ

    # space (might be moved to gpu)
    ρp
    ρp_tens
    ρm
    ρm_tens

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
