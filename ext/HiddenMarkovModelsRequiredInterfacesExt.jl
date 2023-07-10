module HiddenMarkovModelsRequiredInterfacesExt

using HiddenMarkovModels
using RequiredInterfaces: @required

@required AbstractHMM begin
    Base.length(::AbstractHMM)
    HiddenMarkovModels.initial_distribution(::AbstractHMM)
    HiddenMarkovModels.transition_matrix(::AbstractHMM)
    HiddenMarkovModels.obs_distribution(::AbstractHMM, ::Integer)
end

end
