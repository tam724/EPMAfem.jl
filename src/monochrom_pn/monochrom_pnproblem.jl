@concrete struct SpaceDiscretization
    space_model
    arch

    ρp
    ρm

    ∂p
    ∇pm
end

@concrete struct DirectionDiscretization
    direction_model
    arch

    Ip
    Im
    kp
    km
    absΩp
    Ωpm
end

@concrete struct DiscreteMonochromPNProblem
    model
    arch

    # coefficients
    τ
    σ

    # discretizations
    space_discretization
    direction_discretization
end

architecture(problem::DiscreteMonochromPNProblem) = problem.arch
n_basis(problem::DiscreteMonochromPNProblem) = n_basis(problem.model)
n_sums(problem::DiscreteMonochromPNProblem) = (nd = length(dimensions(problem.model)), ne = size(problem.τ, 1))

Base.show(io::IO, p::DiscreteMonochromPNProblem) = print(io, "MonochromPNProblem [$(n_basis(p)) and $(n_sums(p))]")
Base.show(io::IO, ::MIME"text/plain", p::DiscreteMonochromPNProblem) = show(io, p)

# function schurimplicitmidpointsystem(pnproblem::DiscretePNProblem, tol=nothing; use_direct_solver=false)
#     (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(pnproblem)
#     (nd, ne, nσ) = n_sums(pnproblem)

#     arch = architecture(pnproblem)
#     T = base_type(arch)

#     if isnothing(tol)
#         tol = sqrt(eps(Float64))
#     end

#     np = nxp*nΩp
#     n_tot = nxp*nΩp + nxm*nΩm
#     return DiscretePNSystem_IMS(
#         use_direct_solver=use_direct_solver,
#         problem = pnproblem,
#         a = Vector{T}(undef, ne),
#         c = [Vector{T}(undef, nσ) for _ in 1:ne],
#         tmp = allocate_vec(arch, max(nxp, nxm)*max(nΩp, nΩm)),
#         tmp2 = allocate_vec(arch, max(nΩp, nΩm)),
#         tmp3 = allocate_vec(arch, nxm*nΩm),
#         D = allocate_vec(arch, nxm*nΩm),
#         rhs_schur = allocate_vec(arch, np),
#         rhs = allocate_vec(arch, n_tot),
#         sol = allocate_vec(arch, n_tot),
#         lin_solver = allocate_minres_krylov_buf(vec_type(arch), np, np),
#         rtol = T(tol),
#         atol = T(0)
#     )
# end