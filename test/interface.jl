using HiddenMarkovModels
using RequiredInterfaces: check_interface_implemented
using Suppressor
using Test

struct EmptyMC end
struct EmptyHMM end

@test check_interface_implemented(AbstractMC, MarkovChain)
@test check_interface_implemented(AbstractHMM, HMM)

@suppress begin
    @test check_interface_implemented(AbstractMC, EmptyMC) != true
    @test check_interface_implemented(AbstractHMM, MarkovChain) != true
    @test check_interface_implemented(AbstractHMM, EmptyHMM) != true
end
