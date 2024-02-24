sum_to_one!(x) = ldiv!(sum(x), x)

mynonzeros(x::AbstractArray) = x
mynonzeros(x::AbstractSparseArray) = nonzeros(x)

mynnz(x::AbstractArray) = length(mynonzeros(x))

elementwise_log(x::AbstractArray) = log.(x)

function elementwise_log(A::SparseMatrixCSC)
    return SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, log.(A.nzval))
end

"""
    mul_rows_cols!(B, l, A, r)

Perform the in-place operation `B .= l .* A .* transpose(r)`.
"""
function mul_rows_cols!(
    B::AbstractMatrix, l::AbstractVector, A::AbstractMatrix, r::AbstractVector
)
    B .= l .* A .* transpose(r)
    return B
end

function mul_rows_cols!(
    B::SparseMatrixCSC, l::AbstractVector, A::SparseMatrixCSC, r::AbstractVector
)
    @argcheck axes(A, 1) == eachindex(r)
    @argcheck axes(A, 2) == eachindex(l)
    @argcheck size(A) == size(B)
    @argcheck nnz(B) == nnz(A)
    Brv = rowvals(B)
    Bnz = nonzeros(B)
    Anz = nonzeros(A)
    for j in axes(B, 2)
        @argcheck nzrange(B, j) == nzrange(A, j)
        for k in nzrange(B, j)
            i = Brv[k]
            Bnz[k] = l[i] * Anz[k] * r[j]
        end
    end
    return B
end

"""
    argmaxplus_mul!(y, ind, A, x)

Perform the in-place multiplication `A * x` _in the sense of max-plus algebra_, store the result in `y`, and store the index of the maximum for each row in `ind`.
"""
function argmaxplus_mul!(
    y::AbstractVector{R},
    ind::AbstractVector{<:Integer},
    A::AbstractMatrix,
    x::AbstractVector,
) where {R}
    @argcheck axes(A, 1) == eachindex(y)
    @argcheck axes(A, 2) == eachindex(x)
    y .= typemin(R)
    ind .= 0
    for j in axes(A, 2)
        for i in axes(A, 1)
            z = A[i, j] + x[j]
            if z > y[i]
                y[i] = z
                ind[i] = j
            end
        end
    end
    return y
end

function argmaxplus_mul!(
    y::AbstractVector{R},
    ind::AbstractVector{<:Integer},
    A::SparseMatrixCSC,
    x::AbstractVector,
) where {R}
    @argcheck axes(A, 1) == eachindex(y)
    @argcheck axes(A, 2) == eachindex(x)
    y .= typemin(R)
    ind .= 0
    Anz = nonzeros(A)
    Arv = rowvals(A)
    for j in axes(A, 2)
        for k in nzrange(A, j)
            i, a = Arv[k], Anz[k]
            z = a + x[j]
            if z > y[i]
                y[i] = z
                ind[i] = j
            end
        end
    end
    return y
end
