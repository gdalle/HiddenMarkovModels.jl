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
