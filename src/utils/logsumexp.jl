function logsumexp(a)
    T = eltype(a)
    m = first(a)
    se = zero(T)
    for x in a
        if x < m
            se += exp(x - m)
        elseif x == m
            se += one(T)
        else
            se = muladd(se, exp(m - x), one(T))
            m = x
        end
    end
    lse = m + log(se)
    return lse
end

function logsumexp_dumb(a)
    m = maximum(a)
    se = sum(x -> exp(x - m), a)
    return m + log(se)
end
