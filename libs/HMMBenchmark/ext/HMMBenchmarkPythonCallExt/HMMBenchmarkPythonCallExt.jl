module HMMBenchmarkPythonCallExt

using BenchmarkTools
using HMMBenchmark
using PythonCall
using Random: AbstractRNG
using SimpleUnPack

const torch = pyimport("torch")

const np = pyimport("numpy")

const jax = pyimport("jax")
pyimport("jax.numpy")
pyimport("jax.random")

const hmmlearn = pyimport("hmmlearn")
pyimport("hmmlearn.hmm")

const pomegranate = pyimport("pomegranate")
pyimport("pomegranate.hmm")
pyimport("pomegranate.distributions")

const dynamax = pyimport("dynamax")
pyimport("dynamax.hidden_markov_model")

function HMMBenchmark.print_python_setup(; path)
    open(path, "w") do file
        redirect_stdout(file) do
            println("Pytorch threads = $(torch.get_num_threads())")
            println("\n# Python packages\n")
        end
        redirect_stderr(file) do
            PythonCall.CondaPkg.status()
        end
    end
end

include("hmmlearn.jl")
include("pomegranate.jl")
include("dynamax.jl")

end
