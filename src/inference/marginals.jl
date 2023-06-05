"""
    marginals!(γ, ξ, α, β, Bβ, A)

Compute state and transition marginals by mutating `γ` and `ξ` (after forward and backward passes).
"""
function marginals!(γ::Matrix, ξ::Array, α::Matrix, β::Matrix, Bβ::Matrix, A)
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
