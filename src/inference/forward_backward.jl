function forward!(α, c, p, A, B)
    T = size(α, 2)
    @views α[:, 1] .= p .* B[:, 1]
    @views c[1] = inv(sum(α[:, 1]))
    @views α[:, 1] .*= c[1]
    @views for t in 1:(T - 1)
        mul!(α[:, t + 1], A', α[:, t])
        α[:, t + 1] .*= B[:, t + 1]
        c[t + 1] = inv(sum(α[:, t + 1]))
        α[:, t + 1] .*= c[t + 1]
    end
    check_nan(α)
    return nothing
end

function backward!(β, Bβ, c, A, B)
    T = size(β, 2)
    β[:, T] .= one(eltype(β))
    @views for t in (T - 1):-1:1
        Bβ[:, t + 1] .= B[:, t + 1] .* β[:, t + 1]
        mul!(β[:, t], A, Bβ[:, t + 1])
        β[:, t] .*= c[t]
    end
    check_nan(β)
    return nothing
end

function marginals!(γ, ξ, α, β, Bβ, A)
    T = size(γ, 2)
    @views for t in 1:T
        γ[:, t] .= α[:, t] .* β[:, t]
        γt_sum_inv = inv(sum(γ[:, t]))
        γ[:, t] .*= γt_sum_inv
    end
    check_nan(γ)
    @views for t in 1:(T - 1)
        ξ[:, :, t] .= α[:, t] .* A .* Bβ[:, t + 1]'
        ξt_sum_inv = inv(sum(ξ[:, :, t]))
        ξ[:, :, t] .*= ξt_sum_inv
    end
    check_nan(ξ)
    return nothing
end

function forward_backward!(α, c, β, Bβ, γ, ξ, p, A, B)
    forward!(α, c, p, A, B)
    backward!(β, Bβ, c, A, B)
    marginals!(γ, ξ, α, β, Bβ, A)
    return nothing
end
