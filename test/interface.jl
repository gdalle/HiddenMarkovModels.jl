using Pkg
Pkg.add(;
    url="https://github.com/Seelengrab/RequiredInterfaces.jl", rev="feat/multifunc_required"
)

using RequiredInterfaces
using HiddenMarkovModels
using Test

RequiredInterfaces.check_interface_implemented(AbstractHMM, HMM)

Pkg.rm("RequiredInterfaces")
