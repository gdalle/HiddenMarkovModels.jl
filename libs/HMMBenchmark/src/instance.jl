Base.@kwdef struct Instance
    nb_states::Int
    obs_dim::Int
    seq_length::Int = 100
    nb_seqs::Int = 20
    bw_iter::Int = 1
    sparse::Bool = false
    custom_dist::Bool = false
end

function Base.string(c::Instance)
    return reduce(*, "$n $(Int(getfield(c, n))) " for n in fieldnames(typeof(c)))[1:(end - 1)]
end

function Instance(s::String)
    vals = parse.(Int, split(s, " ")[2:2:end])
    return Instance(vals...)
end

function build_data(rng::AbstractRNG, instance::Instance)
    (; nb_seqs, seq_length, obs_dim) = instance
    return randn(rng, nb_seqs, seq_length, obs_dim)
end
