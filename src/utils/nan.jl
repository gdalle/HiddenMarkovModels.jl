function check_nan(x::AbstractArray)
    if any(isnan, x)
        throw(OverflowError("Some values are NaNs"))
    end
end
