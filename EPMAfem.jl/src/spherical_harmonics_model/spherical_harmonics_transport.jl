## advection matrices
Θ(k) = k < 0 ? -1 : 1
plus(k) = k == 0 ? 1 : k + sign(k)
minus(k) = k == 0 ? -1 : k - sign(k)

a(k, l) = sqrt((l - k + 1) * (l + k + 1) / ((2 * l + 3) * (2 * l + 1)))
b(k, l) = sqrt((l - k) * (l + k) / ((2 * l + 1) * (2 * l - 1)))

function c(k, l)
    c = sqrt((l + k + 1) * (l + k + 2) / ((2 * l + 3) * (2 * l + 1)))
    if k < 0
        return 0
    elseif k > 0
        return c
    else
        return c * sqrt(2)
    end
end

function d(k, l)
    d = sqrt((l - k) * (l - k - 1) / ((2 * l + 1) * (2 * l - 1)))
    if k < 0
        return 0
    elseif k > 0
        return d
    else
        return d * sqrt(2)
    end
end

function e(k, l)
    e = sqrt((l - k + 1) * (l - k + 2) / ((2 * l + 3) * (2 * l + 1)))
    if k == 1
        return e * sqrt(2)
    elseif k > 1
        return e
    else
        error("k < 1")
    end
end

function f(k, l)
    f = sqrt((l + k) * (l + k - 1) / ((2 * l + 1) * (2 * l - 1)))
    if k == 1
        return f * sqrt(2)
    elseif k > 1
        return f
    else
        error("k < 1")
    end
end

function A_minus(l, k, k´, ::X)
    if k´ == minus(k) && k != -1
        return 1 / 2 * c(abs(k) - 1, l - 1)
    elseif k´ == plus(k)
        return -1 / 2 * e(abs(k) + 1, l - 1)
    else
        return 0
    end
end

function A_minus(l, k, k´, ::Y)
    if k´ == -minus(k) && k != 1
        return -Θ(k) / 2 * c(abs(k) - 1, l - 1)
    elseif k´ == -plus(k)
        return -Θ(k) / 2 * e(abs(k) + 1, l - 1)
    else
        return 0
    end
end

function A_minus(l, k, k´, ::Z)
    if k´ == k
        return a(k, l - 1)
    else
        return 0
    end
end

function A_plus(l, k, k´, ::X)
    if k´ == minus(k) && k != -1
        return -1 / 2 * d(abs(k) - 1, l + 1)
    elseif k´ == plus(k)
        return 1 / 2 * f(abs(k) + 1, l + 1)
    else
        return 0
    end
end

function A_plus(l, k, k´, ::Y)
    if k´ == -minus(k) && k != 1
        return Θ(k) / 2 * d(abs(k) - 1, l + 1)
    elseif k´ == -plus(k)
        return Θ(k) / 2 * f(abs(k) + 1, l + 1)
    else
        return 0
    end
end

function A_plus(l, k, k´, ::Z)
    if k´ == k
        return b(k, l + 1)
    else
        return 0
    end
end

function get_transport_coefficient(m1, m2, dim)
    l, k = degree(m1), order(m1)
    l_, k_ = degree(m2), order(m2)
    val = 0.0
    if l == l_- 1
        # take Aplus
        val = A_plus(l, k, k_, dim)
    elseif l == l_ + 1
        # take Aminus
        val = A_minus(l, k, k_, dim)
    else
        val = 0.0
    end
    if val == -0.0
        val = 0.0
    end
    return val
end