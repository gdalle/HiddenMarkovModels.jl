function check_nan(x::AbstractArray)
    if any(isnan, x)
        throw(OverflowError("Some values are NaNs"))
    end
end

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
end;
