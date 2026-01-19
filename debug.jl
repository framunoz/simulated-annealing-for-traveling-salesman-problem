# Debugging script for Main.jl

println("--- Starting Debug Session ---")

try
    include("Main.jl")
    # Call the main function explicitly
    if isdefined(Main, :main)
        Main.main()
    else
        println("Warning: main() function not found in Main.jl")
    end
catch e
    println("\n--- Error during execution ---")
    @error "An error occurred" exception = (e, catch_backtrace())
end

println("\n--- Debug Session Finished ---")
