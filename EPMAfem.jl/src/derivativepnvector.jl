function update_pnsystem!(discrete_problem::DiscretePNSystem, mass_concentrations)
    # mass_concentrations should be a vector (number of elements) of vectors (number of grid cells)
    SM.project_matrices(discrete_problem.ρp, discrete_problem.ρp_proj, mass_concentrations)
    SM.project_matrices(discrete_problem.ρm, discrete_problem.ρm_proj, mass_concentrations)
end

@concrete struct VecOfDerivativePNVector <: AbstractDiscretePNVector{true}

end

function (b::VecOfDerivativePNVector)(it::NonAdjointIterator)
    Δϵ = step(energy_model(b.model))
    integral = 0.0
    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)
        integral += Δϵ * b.bϵ[i_ϵ]*dot(b.bxp, ψp * b.bΩp)   
    end
    return integral
end