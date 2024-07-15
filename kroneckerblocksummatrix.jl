"""
This matrix type implements the action of a matrix A when multiplied with a vector B
where A = ∑_i [Dpp_i ⊗ ∑_j(ep_ij * Epp_ij)   0
               0                              Dmm_i ⊗ ∑_j(em_ij * Emm_ij)]
"""
struct KroneckerBlockSumMatrix{N, M, Dpp_, Dmm_, Epp_, Emm_, T_}
    size::Tuple{Tuple{Int64, Int64}, Tuple{Int64, Int64}}
    D::Tuple{NTuple{N, Dpp_}, NTuple{N, Dmm_}}
    E::Tuple{NTuple{N, NTuple{M, Epp_}}, NTuple{N, NTuple{M, Emm_}}}
    e::Tuple{ep, em}
    tmp::T_
end