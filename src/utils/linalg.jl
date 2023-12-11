sum_to_one!(x) = ldiv!(sum(x), x)

mysimilar_mutable(x::AbstractArray, ::Type{R}) where {R} = similar(x, R)

mynonzeros(x::AbstractArray) = x
mynnz(x) = length(mynonzeros(x))

function mul_rows_cols!(
    B::AbstractMatrix, l::AbstractVector, A::AbstractMatrix, r::AbstractVector
)
    B .= l .* A .* r'
    return nothing
end
