@concrete struct ∫S²_nΩgv
    n
    g
end

@concrete struct ∫S²_hv
    h
end

function assemble_linear(int::∫S²_nΩgv, model, V, quad::SphericalQuadrature=lebedev_quadrature(guess_lebedev_order_from_model(model)))
    cache = zeros(length(V))
    function f!(cache, Ω)
        Y_V = _eval_basis_functions!(model, Ω, V)
        dot_n_Ω = dot(int.n, Ω)
        if dot_n_Ω <= 0
            # TODO: maybe add the two here.
            cache .= (dot_n_Ω * int.g(Ω)) .* Y_V
        else
            cache .= zero(cache)
        end
    end
    return quad(f!, cache)
end

function assemble_linear(int::∫S²_hv, model, V, quad::SphericalQuadrature=lebedev_quadrature(guess_lebedev_order_from_model(model)))
    cache = zeros(length(V))
    function f!(cache, Ω)
        Y_V = _eval_basis_functions!(model, Ω, V)
        cache .= int.h(Ω) .* Y_V
    end
    return quad(f!, cache)
end