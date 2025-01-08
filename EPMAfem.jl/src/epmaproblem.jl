@concrete terse struct EPMAProblem
    discrete_problem
    discrete_system

    discrete_excitations
    discrete_extractions
end

function EPMAProblem(problem::UpdatableDiscretePNProblem, excitations::Array{<:Rank1DiscretePNVector}, extractions::Array{<:UpdatableRank1DiscretePNVector})
    system = schurimplicitmidpointsystem(problem.problem)
    return EPMAProblem(problem, system, excitations, extractions)
end

function (epma_problem::EPMAProblem)(ρs::Array)
    update_problem!(epma_problem.discrete_problem, ρs)
    for extraction in epma_problem.discrete_extractions
        update_vector!(extraction, ρs)
    end
    return [extraction.vector for extraction in epma_problem.discrete_extractions] * epma_problem.discrete_system * epma_problem.discrete_excitations
end

function ChainRulesCore.rrule(epma_problem::EPMAProblem, ρs::Array)
    @show "calling rrule"
    update_problem!(epma_problem.discrete_problem, ρs)
    for extraction in epma_problem.discrete_extractions
        update_vector!(extraction, ρs)
    end
    @assert length(epma_problem.discrete_extractions) < length(epma_problem.discrete_excitations)
    res = [extraction.vector for extraction in epma_problem.discrete_extractions] * epma_problem.discrete_system * epma_problem.discrete_excitations
    function epma_problem_pullback(res_bar)
        ρ_bar = zeros(n_parameters(epma_problem.discrete_problem))    
        for (j, cj) in pairs(IndexCartesian(), epma_problem.discrete_extractions)
            # recompute and cache the forward pass
            λ = saveall(cj.vector * epma_problem.discrete_system)
            a_dot = tangent(epma_problem.discrete_problem, λ)
            cj_dot = tangent(cj)
            λ_bar = epma_problem.discrete_system * dot(@view(res_bar[j, axes(epma_problem.discrete_excitations)...]), epma_problem.discrete_excitations)
            ρ_bar += (a_dot + cj_dot) * λ_bar
        end
        return ZeroTangent(), ρ_bar
    end
    return res, epma_problem_pullback
end