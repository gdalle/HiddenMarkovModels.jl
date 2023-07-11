using HiddenMarkovModels
using RequiredInterfaces
using Test

@test RequiredInterfaces.check_interface_implemented(AbstractHMM, HMM)
