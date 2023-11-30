struct ControlledHMM{R,C}
    init::Vector{R}
    trans::Matrix{R}
    means::Vector{R}
    control_seq::Vector{C}
end
