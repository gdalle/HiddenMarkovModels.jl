Base.@kwdef struct Configuration
    sparse::Bool
    nb_states::Int
    obs_dim::Int
    seq_length::Int
    nb_seqs::Int
    bw_iter::Int
end

function to_tuple(c::Configuration)
    return Base.Tuple(getfield(c, n) for n in fieldnames(typeof(c)))
end

function to_namedtuple(c::Configuration)
    return NamedTuple(n => getfield(c, n) for n in fieldnames(typeof(c)))
end
