module HMMBenchmarkPythonCallExt

using BenchmarkTools
using PythonCall
import PythonCall.CondaPkg as CondaPkg
using SimpleUnPack

const torch = pyimport("torch")
const np = pyimport("numpy")
const jnp = pyimport("jax.numpy")

const hmmlearn = pyimport("hmmlearn")
const pomegranate = pyimport("pomegranate")
const dynamax = pyimport("dynamax")

pyimport("pomegranate.hmm")
pyimport("pomegranate.distributions")

pyimport("hmmlearn.hmm")

function print_python_setup(; path)
    open(path, "w") do file
        redirect_stdout(file) do
            println("Pytorch threads = $(torch.get_num_threads())")
            println("\n# Python packages\n")
        end
        redirect_stderr(file) do
            CondaPkg.status()
        end
    end
end

# include("hmmlearn.jl")
# include("pomegranate.jl")
# include("dynamax.jl")

end
