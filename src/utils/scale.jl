"""
    Scale

Abstract type for dispatch-based choice of numerical robustness setting.
"""
abstract type Scale end

"""
    NormalScale <: Scale

Tell algorithms to perform computation without logarithmic scaling.
"""
struct NormalScale <: Scale end

"""
    LogScale <: Scale

Tell algorithms to perform computation with logarithmic scaling.
"""
struct LogScale <: Scale end
