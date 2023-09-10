using HiddenMarkovModels
using RequiredInterfaces: check_interface_implemented
using Test

struct EmptyMC end
struct EmptyHMM end

@test check_interface_implemented(AbstractMC, MarkovChain)
@test check_interface_implemented(AbstractHMM, HMM)

@test check_interface_implemented(AbstractMC, EmptyMC) != true
@test check_interface_implemented(AbstractHMM, EmptyHMM) != true
