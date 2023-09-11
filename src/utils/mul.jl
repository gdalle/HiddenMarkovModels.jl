function mul_rows!(A::AbstractMatrix, v::AbstractVector)
    return A .*= v
end

function mul_cols!(A::AbstractMatrix, v::AbstractVector)
    return A .*= v'
end

function mul_rows!(A::SparseMatrixCSC, v::AbstractVector)
    for (k, i) in enumerate(rowvals(A))
        A.nzval[k] *= v[i]
    end
end

function mul_cols!(A::SparseMatrixCSC, v::AbstractVector)
    for j in eachindex(v)
        for k in nzrange(A, j)
            A.nzval[k] *= v[j]
        end
    end
end

mynonzeros(x::AbstractArray) = x
mynonzeros(x::AbstractSparseArray) = nonzeros(x)
