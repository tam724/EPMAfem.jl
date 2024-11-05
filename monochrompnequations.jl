abstract type AbstractMonoChromPNEquations end
struct DummyAbstractMonoChromPNEquations <: AbstractMonoChromPNEquations end

function number_of_elements(::AbstractMonoChromPNEquations)
    n_elem = 2
    @info "using default number_of_elements: $(n_elem)"
    return n_elem
end

function number_of_scatterings(::AbstractMonoChromPNEquations)
    return 1
end

function mass_concentrations(::AbstractMonoChromPNEquations, i)
    @info "using default mass_concentrations: 1.0"
    return function(x)
        return (0.5, 0.5)[i]
        # if x[2] < 0.0 && x[1] < -0.1 && x[1] > -0.2
        #     return (1.0, 0.0)[i]
        # elseif x[2] > 0.0 && x[1] < -0.4 && x[1] > -0.5
        #     return (1.0, 0.0)[i]
        # else
        #     return (0.0, 1.0)[i]
        # end
    end
end

function specific_total_scattering_cross_section(::AbstractMonoChromPNEquations)
    @info "using default specific_scattering_cross_section: 1.0"
    n_elem = number_of_elements(DummyAbstractMonoChromPNEquations())
    n_scat = number_of_scatterings(DummyAbstractMonoChromPNEquations())
    μs = zeros(n_elem, n_scat)
    #μs[1, 1] = 0.0
    return μs
end

function specific_absorption_cross_section(::AbstractMonoChromPNEquations)
    @info "using default specific_absorption_cross_section: 1.0"
    n_elem = number_of_elements(DummyAbstractMonoChromPNEquations())
    return [0.1, 0.1]
end

function specific_scattering_kernel(::AbstractMonoChromPNEquations)
    @info "using default specific_absorption_cross_section: 1.0"
    return μ -> exp(-5.0*(μ-1.0)^2) / 2.4902319837537594
end

function excitation_spatial_distribution(::AbstractMonoChromPNEquations)
    return function(x)
        return exp(-2*sqrt((x[1]-0.0)^2))
        #return expm2(x, [-0.5, 0.0], [0.01, 0.01])

        # if length(x) == 2
        #     if x[2] > -0.1 && x[2] < 0.1
        #         return isapprox(0.0, x[1])*1.0
        #     else
        #         return isapprox(0.0, x[1])*0.0
        #     end
        #     # return  #* expm2(x[2], 0.0, 0.1)
        # elseif length(x) == 1
        #     return isapprox(0.0, x[1]) 
        # end
    end
end

function excitation_direction_distribution(::AbstractMonoChromPNEquations)
    return IsotropicExtraction()
    #gΩp = [1.0, 0.0, -1.0]
    #return VMFBeam(normalize(gΩp), 70.0)
end