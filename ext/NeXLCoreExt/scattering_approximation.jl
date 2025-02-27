
# computes the svd of the matrices As and a rank r, such that for all rank-r approximation A_r we have a relative error of ||A_r - A|| <= ϵ_rel ||A||
function compute_svd_approx(As, ϵ_rel)
    @assert all((size(As[1]) == size(A) for A in As))
    A_svds = [svd(A) for A in As]

    max_rank = 0
    for (A_svd, A) in zip(A_svds, As)
        max_rank_A = minimum(size(A))
        @assert max_rank_A == length(A_svd.S)
        abs_ϵ = 0.0
        while sqrt(abs_ϵ += A_svd.S[max_rank_A]^2) < ϵ_rel*norm(A)
            max_rank_A -= 1
        end
        max_rank = max(max_rank, max_rank_A)
    end
    return max_rank, A_svds
end

function truncate_svd_and_normalize(svd_, rank)
    U_r = svd_.U[:, 1:rank]
    S_r = Diagonal(svd_.S[1:rank])
    Vt_r = svd_.Vt[1:rank, :]
    E = U_r*S_r*Diagonal(Vt_r[:, 1])
    K = inv(Diagonal(Vt_r[:, 1])) * Vt_r
    return E, K
end

function build_scattering_approximation(E, K, ϵ_model, rank)
    return [(scale(interpolate(E[:, r], BSpline(Linear())), ϵ_model), EPMAfem.SphericalHarmonicsModels.LegendreBasisExp(K[r, :])) for r in 1:rank]
end