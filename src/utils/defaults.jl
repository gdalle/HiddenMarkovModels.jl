duration(seq::AbstractVector) = length(seq)
duration(seq::AbstractMatrix) = size(seq, 2)

at_time(seq::AbstractVector, t::Integer) = seq[t]
at_time(seq::AbstractMatrix, t::Integer) = view(seq, :, t)
