# This script is for compiling SGD optimization results into a single text file
# Loads the parameter vectors from individual result files and saves them in a 
# formatted text file for easy reference.

using Printf

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

N = 6  # Number of states - 1
M = 7  # Number of cells
type_of_boundary_condition = "boundary_2" #same boundary condition as in the three cell system, as in the paper.

# List of h_errors to process
h_errors = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

println("=== SGD Results Compiler ===")
println("N = $N, M = $M")
println("Processing h_errors: $h_errors")

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Minor helper functionss

function load_theta_from_file(filename)
    """Load θ_star from a theta_star.txt file"""
    if !isfile(filename)
        return nothing
    end
    content = read(filename, String)
    theta_str = strip(content)
    theta_values = [parse(Float64, strip(s)) for s in split(theta_str, ",")]
    return theta_values
end

function convert_theta_to_full_params(θ_star, N)
    """Convert θ_star (compressed) to full parameter vector"""
    f_p_0_ours = θ_star[1:N-1]
    f_p_0_true = vcat(0.0, f_p_0_ours)
    f_p_1_ours = θ_star[N:2*N-2]
    f_p_1_true = vcat(0.0, f_p_1_ours)
    f_m_0_ours = θ_star[2*N-1:3*N-3]
    f_m_0_true = vcat(f_m_0_ours, 0.0)
    f_m_1_ours = θ_star[3*N-2:4*N-4]
    f_m_1_true = vcat(f_m_1_ours, 0.0)
    g_ours = θ_star[4*N-3:5*N-3]
    k_off_ours = θ_star[5*N-2]
    θ_full = vcat(f_p_0_true, f_p_1_true, f_m_0_true, f_m_1_true, g_ours, k_off_ours)
    return θ_full
end

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Load parameter vectors from files

strategy_results = Vector{Dict}()

for h_error in h_errors
    he_str = replace(string(h_error), "." => "p")
    base_folder = joinpath(dirname(dirname(@__DIR__)), "results", "seven_cell_results", "SGD_results_"*type_of_boundary_condition)
    base_name = "sgd_N$(N)_M$(M)_he$(he_str)"
    theta_filename = "$(base_name).txt"
    
    if !isfile(theta_filename)
        println("  Skipping h_error=$h_error (file not found: $theta_filename)")
        continue
    end
    
    θ_star = load_theta_from_file(theta_filename)
    if θ_star === nothing
        println("  Skipping h_error=$h_error (failed to load)")
        continue
    end
    
    θ_full = convert_theta_to_full_params(θ_star, N)
    
    push!(strategy_results, Dict(
        "h_error" => h_error,
        "theta_star_file" => theta_filename,
        "theta_star" => θ_star,
        "theta_full" => θ_full
    ))
    
    println("  Loaded h_error=$h_error from $theta_filename")
end

println("\nLoaded $(length(strategy_results)) strategies")

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Save parameter vectors to text file
base_folder = joinpath(dirname(dirname(@__DIR__)), "results", "seven_cell_results", "SGD_results_combined_"*type_of_boundary_condition)
strategies_txt_filename = joinpath(base_folder, "all_N$(N)_M$(M)_strategies_per_h_error.txt")

open(strategies_txt_filename, "w") do io
    println(io, "Summary of inferred control strategies (parameter vectors) for each h_error.")
    println(io, "N = $N (internal states), M = $M (cells)")
    println(io, "="^98)
    
    for (k, strat) in enumerate(strategy_results)
        println(io, "\n" * "-"^98)
        println(io, "Strategy #$k for h_error = $(strat["h_error"]):")
        println(io, "  Parameter file: $(strat["theta_star_file"])")
        println(io, "")
        println(io, "  θ_star (compressed vector from file):")
        println(io, "    " * join(strat["theta_star"], ", "))
        println(io, "")
        println(io, "  θ_full (expanded parameter vector for simulation):")
        println(io, "    [")
        for (i, x) in enumerate(strat["theta_full"])
            println(io, "      ", @sprintf("%.8f", x), i < length(strat["theta_full"]) ? "," : "")
        end
        println(io, "    ]")
    end
    
    println(io, "\n" * "="^98)
    println(io, "")
    println(io, "Parameter vector structure (θ_full):")
    println(io, "  f_p_0[1:N]     - forward rates when s=0")
    println(io, "  f_p_1[1:N]     - forward rates when s=1")
    println(io, "  f_m_0[1:N]     - backward rates when s=0")
    println(io, "  f_m_1[1:N]     - backward rates when s=1")
    println(io, "  g[1:N+1]       - signaling rates")
    println(io, "  k_off          - signal off rate")
end

println("\nSaved: $strategies_txt_filename")
println("Done!")
