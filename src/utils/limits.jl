"""
$(SIGNATURES)

Return a tuple `(t1, t2)` giving the begin and end indices of subsequence `k` within a set of sequences ending at `seq_ends`.
"""
function seq_limits(seq_ends::AbstractVectorOrNTuple{Int}, k::Integer)
    if k == 1
        return 1, seq_ends[k]
    else
        return seq_ends[k - 1] + 1, seq_ends[k]
    end
end
