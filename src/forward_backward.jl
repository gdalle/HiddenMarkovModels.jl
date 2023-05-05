Base.@kwdef struct ForwardBackward{R}
    α::Matrix{R}
    c::Vector{R}
    β::Matrix{R}
    bβ::Matrix{R}
    γ::Matrix{R}
    ξ::Array{R,3}
end

"""
    forward!(α, c, p, A, B)

Perform a forward pass by mutating `α` and `c`.
"""
function forward!(α::Matrix, c::Vector, p, A, B)
    T = size(α, 2)
    @views begin
        α[:, 1] .= p .* B[:, 1]
        c[1] = inv(sum(α[:, 1]))
        α[:, 1] .*= c[1]
        for t in 1:(T - 1)
            mul!(α[:, t + 1], A', α[:, t])
            α[:, t + 1] .*= B[:, t + 1]
            c[t + 1] = inv(sum(α[:, t + 1]))
            α[:, t + 1] .*= c[t + 1]
        end
    end
    check_nan(α)
    return nothing
end

"""
    backward!(β, bβ, A, B)

Perform a backward pass by mutating `β` and `bβ` (after forward pass).
"""
function backward!(β::Matrix, bβ::Matrix, c::Vector, A, B)
    T = size(β, 2)
    @views begin
        β[:, T] .= one(eltype(β))
        for t in (T - 1):-1:1
            bβ[:, t + 1] .= B[:, t + 1] .* β[:, t + 1]
            mul!(β[:, t], A, bβ[:, t + 1])
            β[:, t] .*= c[t]
        end
    end
    check_nan(β)
    return nothing
end

"""
    marginals!(γ, ξ, α, β, bβ, A)

Compute state and transition marginals by mutating `γ` and `ξ` (after backward pass).
"""
function marginals!(γ::Matrix, ξ::Array, α::Matrix, β::Matrix, bβ::Matrix, A)
    S, T = size(γ)
    γ .= α .* β
    @views for t in 1:T
        γ_sum_inv = inv(sum(γ[:, t]))
        γ[:, t] .*= γ_sum_inv
    end
    check_nan(γ)
    @views for t in 1:(T - 1)
        for j in 1:S
            for i in 1:S
                ξ[i, j, t] = α[i, t] * A[i, j] * bβ[j, t + 1]
            end
        end
        ξ_sum_inv = inv(sum(ξ[:, :, t]))
        ξ[:, :, t] .*= ξ_sum_inv
    end
    check_nan(ξ)
    return nothing
end

"""
    forward_backward!(α, c, β, bβ, γ, ξ, p, A, B)

Apply the full forward-backward algorithm by mutating `α`, `c`, `β`, `bβ`, `γ` and `ξ`.
"""
function forward_backward!(
    α::Matrix, c::Vector, β::Matrix, bβ::Matrix, γ::Matrix, ξ::Array{<:Any,3}, p, A, B
)
    forward!(α, c, p, A, B)
    backward!(β, bβ, c, A, B)
    marginals!(γ, ξ, α, β, bβ, A)
    logL = -sum(log, c)
    return logL
end

function forward_backward!(forback::ForwardBackward, p, A, B)
    (; α, c, β, bβ, γ, ξ) = forback
    return forward_backward!(α, c, β, bβ, γ, ξ, p, A, B)
end

function initialize_forward_backward(p, A, B)
    S, T = size(B)
    R = promote_type(eltype(p), eltype(A), eltype(B))
    α = Matrix{R}(undef, S, T)
    c = Vector{R}(undef, T)
    β = Matrix{R}(undef, S, T)
    bβ = Matrix{R}(undef, S, T)
    γ = Matrix{R}(undef, S, T)
    ξ = Array{R,3}(undef, S, S, T - 1)
    return ForwardBackward(α, c, β, bβ, γ, ξ)
end

function forward_backward(hmm::AbstractHMM, θ, obs_seq)
    p = initial_distribution(hmm, θ)
    A = transition_matrix(hmm, θ)
    B = likelihoods(hmm, θ, obs_seq)
    forback = initialize_forward_backward(p, A, B)
    return forward_backward!(forback, p, A, B)
end
