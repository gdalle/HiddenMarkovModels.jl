"""
    backward!(β, Bβ, c, A, B)

Perform a backward pass by mutating `β` and `Bβ` (after forward pass).
"""
function backward!(β::Matrix, Bβ::Matrix, c::Vector, A, B)
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
