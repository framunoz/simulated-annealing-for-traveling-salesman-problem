using Test

function run_tests(directory::String)
    println("🔍 Scanning for tests in: $directory")
    
    if !isdir(directory)
        println("❌ Directory not found: $directory")
        return
    end

    test_files = String[]
    
    for (root, dirs, files) in walkdir(directory)
        for file in files
            if startswith(file, "test_") && endswith(file, ".jl")
                push!(test_files, joinpath(root, file))
            end
        end
    end
    
    if isempty(test_files)
        println("⚠️  No tests found matching 'test_*.jl' in $directory")
        return
    end

    println("✓ Found $(length(test_files)) test files:")
    for tf in test_files
        println("  - $tf")
    end
    println("-"^40)
    
    # Run them
    @testset "Test Suite: $directory" begin
        for tf in test_files
            println("\n▶️  Running $tf ...")
            try
                include(tf)
            catch e
                println("❌ Error loading $tf")
                showerror(stdout, e, catch_backtrace())
                println()
            end
        end
    end
end

# Handle command line arguments
if !isempty(ARGS)
    for arg in ARGS
        run_tests(arg)
    end
else
    # Default behavior if no args
    run_tests("tests/srcjl")
end
