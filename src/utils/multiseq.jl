"""
$(TYPEDEF)

A wrapper for several sequences to enable seamless dispatch.

Amenable to `length` and `getindex` only.

# Fields

$(TYPEDFIELDS)
"""
struct MultiSeq{O,V<:AbstractVector{O}} <: AbstractVector{V}
    "underlying sequences"
    seqs::Vector{V}
end

sequences(m::MultiSeq) = m.seqs

# Mandatory
Base.size(m::MultiSeq) = size(sequences(m))
Base.getindex(m::MultiSeq, k::Integer) = sequences(m)[k]

# Optional
Base.IndexStyle(::Type{<:MultiSeq}) = IndexLinear()
