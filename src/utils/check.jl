function check_no_nan(a)
    if any(isnan, a)
        throw(OverflowError("Some values are NaN"))
    end
end

function check_positive(a)
    if any(!>(zero(eltype(a))), a)
        throw(OverflowError("Some values are not positive"))
    end
end
