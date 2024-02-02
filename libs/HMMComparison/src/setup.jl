
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
