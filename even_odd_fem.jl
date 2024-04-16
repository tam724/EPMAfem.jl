module EvenOddFEM
    using Gridap

    function center(R)
        return sum(R)/2.0
    end

    function ⁺(f)
        Ω0 = 0.0
        return x -> 1.0/2.0*(f(x) + f(2.0*Ω0-x))
    end

    function ⁻(f)
        Ω0 = 0.0
        return x -> 1.0/2.0*(f(x) - f(2.0*Ω0-x))
    end

    function ⁺(u::Gridap.MultiField.MultiFieldCellField)
        return u[1]
    end

    function ⁻(u::Gridap.MultiField.MultiFieldCellField)
        return u[2]
    end

    function ∂(u::Gridap.MultiField.MultiFieldCellField)
        return Gridap.MultiField.MultiFieldCellField([dot(∇(⁻(u)), VectorValue(1.0)), dot(∇(⁺(u)), VectorValue(1.0))])
    end

    import Base.:*
    function *(u::Gridap.MultiField.MultiFieldCellField, v::Gridap.MultiField.MultiFieldCellField)
        return Gridap.MultiField.MultiFieldCellField([⁺(u)* ⁺(v) + ⁻(u)* ⁻(v), ⁺(u)* ⁻(v) + ⁻(u)* ⁺(v)])
    end

    function *(f::Function, u::Gridap.MultiField.MultiFieldCellField)
        return Gridap.MultiField.MultiFieldCellField([⁺(f) * ⁺(u) + ⁻(f) * ⁻(u), ⁺(f) * ⁻(u) + ⁻(f) * ⁺(u)])
    end

    export center, ⁺, ⁻, ∂, *
end