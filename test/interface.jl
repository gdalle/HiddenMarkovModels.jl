using HiddenMarkovModels
using RequiredInterfaces: check_interface_implemented
using Suppressor
using Test

struct Empty end

@test check_interface_implemented(AbstractMC, MarkovChain)
@test check_interface_implemented(AbstractHMM, HMM)
@test check_interface_implemented(AbstractMC, HMM)
@test check_interface_implemented(AbstractHMM, PermutedHMM)

@suppress begin
    @test check_interface_implemented(AbstractMC, Empty) != true
    @test check_interface_implemented(AbstractHMM, Empty) != true
    @test check_interface_implemented(AbstractHMM, MarkovChain) != true
end
