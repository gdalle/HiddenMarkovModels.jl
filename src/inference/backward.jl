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
