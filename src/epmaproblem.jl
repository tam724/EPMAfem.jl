@concrete terse struct EPMAProblem
    upd_problem
    system

    excitations
    upd_extractions
    standard_intensities
end

function EPMAProblem(problem::UpdatableDiscretePNProblem, excitations::Array{<:Rank1DiscretePNVector}, extractions::Array{<:UpdatableRank1DiscretePNVector})
    system = implicit_midpoint(problem.problem, PNSchurSolver)
    standard_intensities = ones(size(extractions)..., size(excitations)...)
    return EPMAProblem(problem, system, excitations, extractions, standard_intensities)
end

function update_problem_and_vectors!(epma_problem::EPMAProblem, ρs::Array)
    update_problem!(epma_problem.upd_problem, ρs)
    for extraction in epma_problem.upd_extractions
        update_vector!(extraction, ρs)
    end
end

function update_standard_intensities!(epma_problem)
    (_, ne, _) = n_sums(epma_problem.upd_problem.problem)
    for i in 1:ne
        ρs = discretize_mass_concentrations([x -> i == e ? 1.0 : 0.0 for e in 1:ne], epma_problem.upd_problem.problem.model)
        update_problem_and_vectors!(epma_problem, ρs)
        # collect all extractions with element_index i
        idxs = [idx for (idx, c) in pairs(IndexCartesian(), epma_problem.upd_extractions) if c.element_index == i]
        cs = [c.vector for (_, c) in pairs(IndexCartesian(), epma_problem.upd_extractions) if c.element_index == i]
        intensities = cs * epma_problem.system * epma_problem.excitations
        for (idx1, idx2) in pairs(IndexLinear(), idxs)
            epma_problem.standard_intensities[idx2, axes(epma_problem.excitations)...] .= intensities[idx1, axes(epma_problem.excitations)...]
        end
    end
end

function (epma_problem::EPMAProblem)(ρs::Array)
    update_problem_and_vectors!(epma_problem, ρs)
    intensities = [extraction.vector for extraction in epma_problem.upd_extractions] * epma_problem.system * epma_problem.excitations
    k_ratios = intensities ./ epma_problem.standard_intensities
    return k_ratios
end

function ChainRulesCore.rrule(epma_problem::EPMAProblem, ρs::Array)
    update_problem_and_vectors!(epma_problem, ρs)
    @assert length(epma_problem.upd_extractions) < length(epma_problem.excitations)
    intensities = [extraction.vector for extraction in epma_problem.upd_extractions] * epma_problem.system * epma_problem.excitations
    k_ratios = intensities ./ epma_problem.standard_intensities
    function epma_problem_pullback(k_ratios_bar)
        intensities_bar = k_ratios_bar ./ epma_problem.standard_intensities
        ρ_bar = zeros(n_parameters(epma_problem.upd_problem))    
        for (j, cj) in pairs(IndexCartesian(), epma_problem.upd_extractions)
            # recompute and cache the forward pass
            λ = saveall(cj.vector * epma_problem.system)
            a_dot = tangent(epma_problem.upd_problem, λ)
            cj_dot = tangent(cj)
            λ_bar = epma_problem.system * dot(@view(intensities_bar[j, axes(epma_problem.excitations)...]), epma_problem.excitations)
            ρ_bar += (a_dot + cj_dot) * λ_bar
        end
        return ZeroTangent(), ρ_bar
    end
    return k_ratios, epma_problem_pullback
end
