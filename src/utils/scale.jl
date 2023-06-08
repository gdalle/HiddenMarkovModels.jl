"""
    Scale

Abstract type for dispatch-based choice of numerical robustness setting.
"""
abstract type Scale end

"""
    NormalScale <: Scale

Tell algorithms to use no logarithmic scaling.
"""
struct NormalScale <: Scale end

"""
    SemiLogScale <: Scale

Tell algorithms to use partial logarithmic scaling.
"""
struct SemiLogScale <: Scale end

"""
    LogScale <: Scale

Tell algorithms to use full logarithmic scaling.
"""
struct LogScale <: Scale end
