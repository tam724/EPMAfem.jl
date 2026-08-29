@concrete struct ∫S²_hv
    h
end

@concrete struct ∫S²_nΩgv
    n
    g
    weighting
    influx_outflux
end

∫S²_nΩgv(n, g) = ∫S²_nΩgv(n, g, true, true)
∫S²_nΩgv(n, g, weighting) = ∫S²_nΩgv(n, g, weighting, true)
∫S²_2nΩgv(n, g, weighting=true) = ∫S²_nΩgv(n, Ω -> 2*g(Ω), weighting)

int_func(int::∫S²_hv, Ω) = int.h(Ω)
function int_func(int::∫S²_nΩgv, Ω)
    g = int.g(Ω)
    dot_n_Ω = dot(int.n, Ω)
    if (int.influx_outflux && dot_n_Ω <= 0) || (!int.influx_outflux && dot_n_Ω >= 0)
        return int.weighting ? dot_n_Ω*g : g
    else
        return zero(g)
    end
end

function assemble_linear(int::Union{∫S²_hv, ∫S²_nΩgv}, model::AbstractHarmonicsModel{D}, V, quad::NSphericalQuadrature{D}=LebedevQuadrature{D}()) where {D}
    quad isa HCubatureQuadrature && @warn("hcubature quadrature does not perform very well here!")
    cache = zeros(length(V))
    function f!(cache, Ω)
        Y_V = _eval_basis_functions!(model, Ω, V)
        cache .= int_func(int, Ω) .* Y_V
    end
    return quad(f!, cache)
end
