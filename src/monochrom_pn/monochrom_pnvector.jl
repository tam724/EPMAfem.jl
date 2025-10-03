@concrete struct DiscreteMonochromPNVector
    adjoint::Bool
    model
    arch

    # might be moded to gpu
    bxp
    # bxm
    bΩp
    # bΩm
end

_is_adjoint_vector(b::DiscreteMonochromPNVector) = b.adjoint

function assemble!(rhs, B::Vector, Δ, sym, β=false)
    for i in eachindex(B)
        assemble!(rhs, B[i], Δ, sym, β)
        β = true
    end
end

function assemble!(rhs, b::DiscreteMonochromPNVector, Δ, sym, β=false)
    rhs_p, rhs_m = pmview(rhs, b.model)
    mul!(rhs_p, b.bxp, transpose(b.bΩp), Δ, β)
    my_rmul!(rhs_m, β) # *-1 if sym
end

function integrate(b::DiscreteMonochromPNVector, ψ)
    ψp = pview(ψ, b.model)
    return dot(transpose(ψp) * b.bxp, b.bΩp)
end
