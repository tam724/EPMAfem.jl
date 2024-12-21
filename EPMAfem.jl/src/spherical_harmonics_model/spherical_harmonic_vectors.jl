const ∫S²_nΩgv = Val{:∫S²_nΩgv}()
const ∫S²_hv = Val{:∫S²_hv}()

function assemble_linear(::Val{:∫S²_nΩgv}, n, g, model, V, quad::Quadrature=lebedev_quadrature)
    cache = zeros(length(V))
    function f!(cache, Ω)
        Y_V = _eval_basis_functions!(model, Ω, V)
        dot_n_Ω = dot(n, Ω)
        if dot_n_Ω <= 0
            # TODO: maybe add the two here.
            cache .= (dot_n_Ω * g_Ω(Ω)) .* Y_V
        else
            cache .= zero(cache)
        end
    end
    return quad(f!, cache, model)
end

function assemble_linear(::Val{:∫S²_hv}, h, model, V, quad::Quadrature=lebedev_quadrature)
    cache = zeros(length(V))
    function f!(cache, Ω)
        Y_V = _eval_basis_functions!(model, Ω, V)
        cache .= h(Ω) .* Y_V
    end
    return quad(f!, cache, model)
end