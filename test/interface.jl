using HiddenMarkovModels
using RequiredInterfaces: check_interface_implemented
using Test

struct EmptyHMM end

@test check_interface_implemented(AbstractHMM, HMM)

@test check_interface_implemented(AbstractHMM, EmptyHMM) != true
