function mul_rows_cols!(
    B::AbstractMatrix, l::AbstractVector, A::AbstractMatrix, r::AbstractVector
)
    B .= l .* A .* r'
    return nothing
end

function mul_rows_cols!(
    B::SparseMatrixCSC, l::AbstractVector, A::SparseMatrixCSC, r::AbstractVector
)
    @assert size(B) == size(A) == (length(l), length(r))
    B .= A
    for j in axes(B, 2)
        for k in nzrange(B, j)
            i = B.rowval[k]
            B.nzval[k] *= l[i] * r[j]
        end
    end
    return nothing
end
