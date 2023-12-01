"""
$(TYPEDEF)

A wrapper for several sequences to enable seamless dispatch.

Amenable to `length` and `getindex` only.

# Fields

$(TYPEDFIELDS)
"""
struct MultiSeq{O} <: AbstractVector{Vector{O}}
    "underlying sequences"
    seqs::Vector{Vector{O}}
end

sequences(m::MultiSeq) = m.seqs

# Mandatory
Base.size(m::MultiSeq) = size(sequences(m))
Base.getindex(m::MultiSeq, k::Integer) = sequences(m)[k]
Base.eachindex(m::MultiSeq, a::AbstractArray) = eachindex(sequences(m), a)

# Optional
Base.IndexStyle(::Type{<:MultiSeq}) = IndexLinear()
