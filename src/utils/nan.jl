function check_nan(a)
    if any(isnan, a)
        throw(OverflowError("Some values are NaN"))
    end
end
