Base.@kwdef struct Configuration
    nb_states::Int
    obs_dim::Int
    seq_length::Int = 100
    nb_seqs::Int = 20
    bw_iter::Int = 1
    sparse::Bool = false
    custom_dist::Bool = false
end

function Base.string(c::Configuration)
    return reduce(*, "$n $(Int(getfield(c, n))) " for n in fieldnames(typeof(c)))[1:(end - 1)]
end

function Configuration(s::String)
    vals = parse.(Int, split(s, " ")[2:2:end])
    return Configuration(vals...)
end

function to_namedtuple(c::Configuration)
    return NamedTuple(n => getfield(c, n) for n in fieldnames(typeof(c)))
end
