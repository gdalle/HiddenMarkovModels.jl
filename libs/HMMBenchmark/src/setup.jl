function print_julia_setup(path)
    open(path, "w") do file
        redirect_stdout(file) do
            InteractiveUtils.versioninfo()
            println("\n# Multithreading\n")
            println("Julia threads = $(Threads.nthreads())")
            println("OpenBLAS threads = $(BLAS.get_num_threads())")
            println("\n# Julia packages\n")
            Pkg.status()
        end
    end
end
