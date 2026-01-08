using Random
using LinearAlgebra
using Statistics
using Plots
using Printf
using Distributions
using LightGraphs
using FileIO
using VideoIO
using LaTeXStrings
using Measures
using Revise
using CellularDecisions
using Distributed
using ExponentialUtilities
using Dates
using JSON

# Include helper files
include("../utils/utils.jl")
include("../mult_cell/mult_cell.jl")

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
N = 6 # Number of states - 1
M = 3  # Number of cells
initial_state_array = ((1,0),(1,0),(1,0))  # Initial state for simulations
# initial_state_array = ((1,0),(1,0),(1,0), (1,0), (1,0), (1,0))  # Initial state for simulations

type_of_boundary_condition = "boundary_2"  # for saving the results

# List of h_errors to process
h_errors = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# Setup logging to file
logfile_name = "sgd_plot_N$(N)_M$(M)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log"
logfile = open(logfile_name, "w")

# Helper function to log to both console and file
function log_msg(msg)
    println(msg)
    println(logfile, msg)
    flush(stdout)
    flush(stderr)
    flush(logfile)
end

log_msg("=== Starting parameter loading and validation ===")
log_msg("N = $N, M = $M")
log_msg("h_errors to process: $h_errors")
log_msg("Log file: $logfile_name")

# Adjacency Matrix
AdjMat = zeros(Int, M, M)
for i in 1:M
    for j in 1:M
        if i != j
            AdjMat[i, j] = 1
        end
    end
end

T_horizon = 300.0
log_msg("AdjMat: $AdjMat")
log_msg("T_horizon: $T_horizon")

# Target states
TG_proper = [(N,0,0), (0,N,0), (0,0,N)]
TB_proper = []
for u_1 in [0,N]
    for u_2 in [0,N]
        for u_3 in [0,N]
            curr = (u_1, u_2, u_3)
            if curr ∈ TG_proper
                continue
            end
            if curr ∈ TB_proper
                continue
            end
            push!(TB_proper, curr)
        end
    end
end
log_msg("TG_proper: $TG_proper")
log_msg("TB_proper: $(length(TB_proper)) bad states")

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Simulation functions
#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

function outgoing_rates(state_array_curr, state_array_next, AdjMat, P_opt_dict, N, M)
    flags_found = zeros(Int, M)
    changed_cell = 0

    # Single pass to find changed cell
    for i in 1:M
        if state_array_curr[i] != state_array_next[i]
            flags_found[i] = 1
            changed_cell = i
        end
    end

    # Early return if no valid transition
    if sum(flags_found) != 1 || changed_cell == 0
        return 0.0
    end

    curr_cell_now = state_array_curr[changed_cell]
    curr_cell_next = state_array_next[changed_cell]

    # Case A: change in internal state, no change in signalling state
    if curr_cell_now[1] != curr_cell_next[1] && curr_cell_now[2] == curr_cell_next[2]
        if curr_cell_next[1] == curr_cell_now[1] + 1
            return P_opt_dict[changed_cell][curr_cell_now[2]*N + curr_cell_now[1] + 1]
        elseif curr_cell_next[1] == curr_cell_now[1] - 1
            return P_opt_dict[changed_cell][curr_cell_now[2]*N + curr_cell_now[1] + 2N]
        end
    end
    if curr_cell_now[1] == curr_cell_next[1] && curr_cell_now[2] != curr_cell_next[2]
        if curr_cell_next[2]==0 && curr_cell_now[2]==1
            return P_opt_dict[changed_cell][5*N + 2]
        elseif curr_cell_next[2]==1 && curr_cell_now[2]==0
            all_cells_neighbors = findall(AdjMat[changed_cell, :] .== 1)
            rate_to_give = 0.0
            for cell_neighbor in all_cells_neighbors
                rate_to_give += P_opt_dict[cell_neighbor][4*N + state_array_curr[cell_neighbor][1] + 1]
            end
            return rate_to_give
        end
    end
end

function all_states_from_curr_state(state_array_curr, AdjMat, N, M)
    all_states = []
    for i in 1:M
        # Handle internal state transitions
        if state_array_curr[i][1] != N && state_array_curr[i][1] != 0
            # Create new state array with deepcopy to avoid modifying original
            new_state = [(j == i ? (state_array_curr[j][1] + 1, state_array_curr[j][2]) : 
                         deepcopy(state_array_curr[j])) for j in 1:M]
            new_state = tuple(new_state...)
            push!(all_states, new_state)
            new_state = [(j == i ? (state_array_curr[j][1] - 1, state_array_curr[j][2]) : 
                         deepcopy(state_array_curr[j])) for j in 1:M]
            new_state = tuple(new_state...)
            push!(all_states, new_state)
        end

        # Handle signaling state transitions
        if state_array_curr[i][2] == 0
            new_state = [(j == i ? (state_array_curr[j][1], 1) :
                         deepcopy(state_array_curr[j])) for j in 1:M]
            new_state = tuple(new_state...)
            push!(all_states, new_state)
        elseif state_array_curr[i][2] == 1
            new_state = [(j == i ? (state_array_curr[j][1], 0) :
                         deepcopy(state_array_curr[j])) for j in 1:M]
            new_state = tuple(new_state...)
            push!(all_states, new_state)
        end
    end
    return all_states
end

function simulate_ctmc_faster(N, M, AdjMat, P_opt_dict, initial_state, T, TG, TB)
    t = 0.0
    times = [t]
    states_till_now = [initial_state]
    is_bad = 0
    terminal_time = T
    while t < T
        curr_state = states_till_now[end]
        all_states = all_states_from_curr_state(curr_state, AdjMat, N, M)
        all_outgoing_rates = [outgoing_rates(curr_state, all_states[i], AdjMat, P_opt_dict, N, M) for i in 1:length(all_states)]
        sum_of_outgoing_rates = sum(all_outgoing_rates)
        if sum_of_outgoing_rates <= 1e-9
            break
        end
        Δt = rand(Exponential(1/sum_of_outgoing_rates))
        t += Δt
        if t >= T
            break
        end
        probs = all_outgoing_rates ./ sum_of_outgoing_rates
        dist = Categorical(probs)
        s = rand(dist)
        push!(states_till_now, all_states[s])
        push!(times, t)
        # Check if the state is terminal or not, and if it is bad or not
        curr_state_full = all_states[s]
        current_internal_states = tuple([curr_state_full[i][1] for i in 1:M]...)
        if current_internal_states ∈ TB
            is_bad = 1
        end
        if current_internal_states ∈ TG || current_internal_states ∈ TB
            terminal_time = min(terminal_time, t)
            break
        end
    end
    return times, states_till_now, terminal_time, is_bad
end

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Helper functions for loading parameters
#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

function load_theta_from_file(filename)
    """Load θ_star from a theta_star.txt file"""
    if !isfile(filename)
        error("File not found: $filename")
    end
    content = read(filename, String)
    # Parse comma-separated values
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
# Main processing loop
#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

# Storage for summary results across all h_errors
summary_results = Dict(
    "h_errors" => Float64[],
    "mean_terminal_times" => Float64[],
    "std_terminal_times" => Float64[],
    "mean_error_rates" => Float64[],
    "std_error_rates" => Float64[],
    "mean_time_plus_1std" => Float64[],
    "mean_time_minus_1std" => Float64[],
    "mean_time_err_bar" => Float64[],
    "mean_error_plus_1std" => Float64[],
    "mean_error_minus_1std" => Float64[],
    "mean_error_err_bar" => Float64[]
)

# Collect strategies (full θ vector) for each h_error
strategy_results = Vector{Dict}()

for h_error in h_errors
    log_msg("\n" * "="^60)
    log_msg("Processing h_error = $h_error")
    log_msg("="^60)
    
    # Construct filename for parameters
    he_str = replace(string(h_error), "." => "p")
    base_name = "sgd_N$(N)_M$(M)_he$(he_str)"
    theta_filename = "$(base_name)_theta_star.txt"
    
    # Check if file exists
    if !isfile(theta_filename)
        log_msg("WARNING: File not found: $theta_filename - skipping")
        continue
    end
    
    # Load parameters
    log_msg("Loading parameters from: $theta_filename")
    θ_star = load_theta_from_file(theta_filename)
    log_msg("Loaded θ_star (length=$(length(θ_star))): $θ_star")
    
    # Convert to full parameter vector
    θ_full = convert_theta_to_full_params(θ_star, N)
    log_msg("Full parameter vector: $θ_full")
    
    # Store this strategy's info for summary text generation later
    push!(strategy_results, Dict(
        "h_error" => h_error,
        "theta_star_file" => theta_filename,
        "theta_star" => θ_star,
        "theta_full" => θ_full
    ))
    
    # Create parameter dictionary for all cells
    param_dict_now = Dict()
    for i in 1:M
        param_dict_now[i] = θ_full
    end
    
    #--------------------------------
    # Run validation simulations
    #--------------------------------
    log_msg("\nStarting validation simulations...")
    N_simulations = 5000
    terminal_times = zeros(N_simulations)
    is_bad = zeros(N_simulations)
    
    num_workers = Threads.nthreads()
    log_msg("Number of threads: $num_workers")
    
    Threads.@threads for i in 1:N_simulations
        if i % 1000 == 0
            log_msg("  Simulation $i / $N_simulations")
        end
        times, states, terminal_time, is_bad_temp = simulate_ctmc_faster(
            N, M, AdjMat, param_dict_now, initial_state_array, 
            T_horizon, TG_proper, TB_proper
        )
        terminal_times[i] = terminal_time
        is_bad[i] = is_bad_temp
    end
    
    # Compute statistics
    mean_terminal_time = mean(terminal_times)
    std_terminal_time = std(terminal_times)/sqrt(N_simulations)
    mean_is_bad = mean(is_bad)
    std_is_bad = std(is_bad)/sqrt(N_simulations)
    err_bar_term_time = 1.96 * std_terminal_time
    err_bar_is_bad = 1.96 * std_is_bad
    
    # For text reproduction: add ± 1σ error bars too
    mean_time_plus_1std = mean_terminal_time + std_terminal_time
    mean_time_minus_1std = mean_terminal_time - std_terminal_time
    mean_error_plus_1std = mean_is_bad + std_is_bad
    mean_error_minus_1std = mean_is_bad - std_is_bad

    log_msg("\n" * "-"^60)
    log_msg("VALIDATION RESULTS for h_error = $h_error")
    log_msg("-"^60)
    log_msg("Mean terminal time: $(mean_terminal_time) ± $(std_terminal_time)")
    log_msg("Mean error rate: $(mean_is_bad) ± $(std_is_bad)")
    log_msg("Target error rate: $(h_error)")
    log_msg("-"^60)
    
    #--------------------------------
    # Generate and save plots
    #--------------------------------
    log_msg("\nGenerating plots...")
    
    plot_width = 1200
    plot_height = 800
    
    # Histogram of terminal times
    p1 = histogram(terminal_times, 
        xlabel="Terminal Time", 
        ylabel="Frequency",
        title="Distribution of Terminal Times (h_error=$h_error)\nMean=$(round(mean_terminal_time, digits=2)) ± $(round(std_terminal_time, digits=2))",
        label="Terminal Times",
        size=(plot_width, plot_height),
        bins=50)
    savefig(p1, "$(base_name)_validation_terminal_times_hist.png")
    
    # Bar plot of error rate
    p2 = bar(["Target", "Achieved"], 
        [h_error, mean_is_bad],
        xlabel="",
        ylabel="Error Rate",
        title="Error Rate Comparison (h_error=$h_error)",
        label="",
        size=(plot_width, plot_height),
        ylims=(0, max(h_error, mean_is_bad) * 1.2),
        color=[:blue, :red])
    hline!([h_error], label="Target", linestyle=:dash, linewidth=2, color=:black)
    savefig(p2, "$(base_name)_validation_error_rate_bar.png")
    
    # Scatter plot of terminal time vs outcome
    outcome_labels = ["Good" for _ in 1:N_simulations]
    outcome_labels[is_bad .== 1] .= "Bad"
    colors = [b == 0 ? :green : :red for b in is_bad]
    
    p3 = scatter(1:N_simulations, terminal_times,
        xlabel="Simulation Index",
        ylabel="Terminal Time",
        title="Terminal Times by Outcome (h_error=$h_error)",
        color=colors,
        label="",
        size=(plot_width, plot_height),
        alpha=0.6,
        markersize=2)
    savefig(p3, "$(base_name)_validation_terminal_times_scatter.png")
    
    # CDF of terminal times
    sorted_times = sort(terminal_times)
    cdf_values = (1:N_simulations) ./ N_simulations
    p4 = plot(sorted_times, cdf_values,
        xlabel="Terminal Time",
        ylabel="CDF",
        title="CDF of Terminal Times (h_error=$h_error)",
        label="CDF",
        size=(plot_width, plot_height),
        linewidth=2)
    savefig(p4, "$(base_name)_validation_terminal_times_cdf.png")
    
    log_msg("Plots saved with prefix: $(base_name)_validation_*")
    
    #--------------------------------
    # Save validation results (with extra error bar info)
    #--------------------------------
    results_filename = "$(base_name)_validation_results.txt"
    open(results_filename, "w") do io
        println(io, "Validation Results for h_error = $h_error")
        println(io, "="^60)
        println(io, "")
        println(io, "System Configuration:")
        println(io, "  N (states-1) = $N")
        println(io, "  M (cells) = $M")
        println(io, "  T_horizon = $T_horizon")
        println(io, "  Initial state = $(initial_state_array)")
        println(io, "")
        println(io, "Loaded Parameters (θ_star, vector from file):")
        println(io, "  File: $theta_filename")
        println(io, "  θ_star: $(join(θ_star, ", "))")
        println(io, "")
        println(io, "Expanded Full Parameter Vector (used by each cell):")
        println(io, "  θ_full = [")
        for x in θ_full
            println(io, "    ", @sprintf("%.8f", x))
        end
        println(io, "  ]")
        println(io, "")
        println(io, "Validation Statistics   [N_simulations = $N_simulations]:")
        println(io, "  Mean terminal time:   $(@sprintf("%.6f ± %.6f", mean_terminal_time, std_terminal_time))")
        println(io, "    +1σ: $(@sprintf("%.6f", mean_time_plus_1std)), -1σ: $(@sprintf("%.6f", mean_time_minus_1std)), ±(1.96σ): $(@sprintf("%.6f", err_bar_term_time))")
        println(io, "  Mean error rate:      $(@sprintf("%.6f ± %.6f", mean_is_bad, std_is_bad))")
        println(io, "    +1σ: $(@sprintf("%.6f", mean_error_plus_1std)), -1σ: $(@sprintf("%.6f", mean_error_minus_1std)), ±(1.96σ): $(@sprintf("%.6f", err_bar_is_bad))")
        println(io, "  Target error rate:    $h_error")
        println(io, "  Constraint satisfied: $(mean_is_bad <= h_error ? "YES" : "NO")")
        println(io, "")
        println(io, "Detailed Statistics:")
        println(io, "  Min terminal time:    $(@sprintf("%.6f", minimum(terminal_times)))")
        println(io, "  Max terminal time:    $(@sprintf("%.6f", maximum(terminal_times)))")
        println(io, "  Median terminal time: $(@sprintf("%.6f", median(terminal_times)))")
        println(io, "  Number of bad outcomes: $(Int(sum(is_bad)))")
        println(io, "  Number of good outcomes: $(Int(N_simulations - sum(is_bad)))")
    end
    log_msg("Validation results saved to: $results_filename")

    # Save raw data as JSON (including error bars)
    data_filename = "$(base_name)_validation_data.json"
    datadict = Dict(
        "h_error" => h_error,
        "N" => N,
        "M" => M,
        "T_horizon" => T_horizon,
        "N_simulations" => N_simulations,
        "theta_star" => θ_star,
        "theta_full" => θ_full,
        "terminal_times" => terminal_times,
        "is_bad" => is_bad,
        "mean_terminal_time" => mean_terminal_time,
        "std_terminal_time" => std_terminal_time,
        "mean_is_bad" => mean_is_bad,
        "std_is_bad" => std_is_bad,
        "mean_time_plus_1std" => mean_time_plus_1std,
        "mean_time_minus_1std" => mean_time_minus_1std,
        "mean_time_err_bar" => err_bar_term_time,
        "mean_error_plus_1std" => mean_error_plus_1std,
        "mean_error_minus_1std" => mean_error_minus_1std,
        "mean_error_err_bar" => err_bar_is_bad
    )
    open(data_filename, "w") do io
        JSON.print(io, datadict)
    end
    log_msg("Raw validation data saved to: $data_filename")
    
    # Store results for summary plots & minimal summary text file
    push!(summary_results["h_errors"], h_error)
    push!(summary_results["mean_terminal_times"], mean_terminal_time)
    push!(summary_results["std_terminal_times"], std_terminal_time)
    push!(summary_results["mean_error_rates"], mean_is_bad)
    push!(summary_results["std_error_rates"], std_is_bad)
    push!(summary_results["mean_time_plus_1std"], mean_time_plus_1std)
    push!(summary_results["mean_time_minus_1std"], mean_time_minus_1std)
    push!(summary_results["mean_time_err_bar"], err_bar_term_time)
    push!(summary_results["mean_error_plus_1std"], mean_error_plus_1std)
    push!(summary_results["mean_error_minus_1std"], mean_error_minus_1std)
    push!(summary_results["mean_error_err_bar"], err_bar_is_bad)

    log_msg("\nCompleted processing h_error = $h_error\n")
end

#--------------------------------
# Generate summary plots across all h_errors
#--------------------------------
if length(summary_results["h_errors"]) > 0
    log_msg("\n" * "="^60)
    log_msg("Generating summary plots across all h_errors...")
    log_msg("="^60)
    
    h_err_vals = summary_results["h_errors"]
    mean_times = summary_results["mean_terminal_times"]
    std_times = summary_results["std_terminal_times"]
    mean_errs = summary_results["mean_error_rates"]
    std_errs = summary_results["std_error_rates"]
    mean_times_p1s = summary_results["mean_time_plus_1std"]
    mean_times_m1s = summary_results["mean_time_minus_1std"]
    mean_times_errbar = summary_results["mean_time_err_bar"]
    mean_errs_p1s = summary_results["mean_error_plus_1std"]
    mean_errs_m1s = summary_results["mean_error_minus_1std"]
    mean_errs_errbar = summary_results["mean_error_err_bar"]
    
    # Load exact data from data_list.txt
    exact_h_global = [0.02000000820526077, 0.030000007867891625, 0.04000000789532438, 0.04999995446888241, 0.06000000976753563, 0.06999990982533871, 0.08000001026212479, 0.09000001143763728, 0.10000001079320482, 0.11000001033056739, 0.12000001031881162, 0.1300000120775603, 0.14000001010639238, 0.1500000045174575, 0.1600000086151831, 0.17000000901129347, 0.18000000855544337, 0.19000000814348675, 0.20000000775343746, 0.21000000522964263, 0.22000000633680958, 0.23000000122443112, 0.2400000001934378, 0.24999999925390246, 0.2599999983835102, 0.2699999975658789, 0.28000000414936677, 0.2899999781340802, 0.29999999901357194, 0.3500000000000158, 0.4000000000000003, 0.4500000000000002, 0.5000000000000016, 0.5500000000001175, 0.5999999780179028, 0.6499999772105834, 0.6999999765063272, 0.7499999758851068, 0.799999975332272, 0.8499999748371072, 0.8999999743927704, 0.9500000000000276, 1.0]
    exact_hitting_times = [10.942808719586106, 9.457855138791126, 8.509344854078703, 7.867885859329496, 7.407423735759681, 7.069411443698219, 6.798416454648042, 6.546413460342658, 6.322758142371081, 6.091079793101303, 5.878482438583745, 5.686306255398104, 5.508709946437765, 5.40240996113116, 5.216187091757594, 5.0426713734905295, 4.8799883633265555, 4.7269035401603, 4.5821004113664285, 4.452161448446196, 4.342139148582988, 4.258683621648738, 4.196066640229892, 4.137860908338273, 4.0830793631304525, 4.031058942044064, 3.9698710914357753, 3.9163370898533034, 3.8696047080190157, 3.674824216510539, 3.5061473509474563, 3.349074268838642, 3.1981738309575256, 3.0479189701300866, 2.8882286187195003, 2.747926564345681, 2.610819629626121, 2.476399328726689, 2.344275119306643, 2.2141396229576475, 2.0857460514538215, 1.9588921107591937, 1.8333333333333335]

    plot_width = 1200
    plot_height = 1000

    # Sort all data by mean error rates
    sort_idx = sortperm(mean_errs)
    mean_errs_sorted = mean_errs[sort_idx]
    mean_times_sorted = mean_times[sort_idx]
    std_errs_sorted = std_errs[sort_idx]
    std_times_sorted = std_times[sort_idx]

    # Single plot: Mean Terminal Time vs Mean Error Rate with error bars and ribbon
    # Ribbon shows 1.96*stddev (95% CI), error bars show 1*stddev
    p_summary = plot(mean_errs_sorted, mean_times_sorted,
        xlabel="Error",
        ylabel="Mean First Passage Time",
        title="Tradeoff between accuracy and speed",
        legend=:topright,
        label="Approximate (SGD)",
        color="#807dba",
        linewidth=1,
        size=(plot_width, plot_height),
        ribbon=1.96 .* std_times_sorted,
        fillalpha=0.5)

    scatter!(mean_errs_sorted, mean_times_sorted,
        xlabel="Error",
        ylabel="Mean First Passage Time",
        color="#54278f",
        label="",
        markersize=3,
        xerror=1.96 .* std_errs_sorted,
        yerror=1.96 .* std_times_sorted)

    # Add exact data if available
    if length(exact_h_global) > 0
        log_msg("Adding exact data to plot ($(length(exact_h_global)) points)")
        plot!(exact_h_global, exact_hitting_times,
            label="Exact",
            color="#2ca02c",
            linewidth=2,
            linestyle=:dash)
        scatter!(exact_h_global, exact_hitting_times,
            color="#2ca02c",
            label="",
            markersize=3)
    else
        log_msg("WARNING: No exact data to plot")
    end

    savefig(p_summary, "summary_N$(N)_M$(M)_terminal_time_vs_error_rate.png")
    savefig(p_summary, "summary_N$(N)_M$(M)_terminal_time_vs_error_rate.svg")
    log_msg("Saved: summary_N$(N)_M$(M)_terminal_time_vs_error_rate.png and .svg")

    # Save summary results as JSON (with 1σ/1.96σ bars)
    summary_filename = "summary_N$(N)_M$(M)_results_all_h_errors.json"
    open(summary_filename, "w") do io
        JSON.print(io, summary_results)
    end
    log_msg("Saved: $summary_filename")
    
    # Save summary results as text table including all ±std and 1.96*std error bars
    summary_txt_filename = "summary_N$(N)_M$(M)_results_all_h_errors.txt"
    open(summary_txt_filename, "w") do io
        println(io, "Summary Results Across All h_error Values")
        println(io, "="^120)
        println(io, "")
        println(io, @sprintf("%-9s %-22s %-22s %-22s %-22s %-16s %-16s", 
            "h_error", 
            "MeanTime ± Std", 
            "+1σ", 
            "-1σ",
            "±1.96σ (bar)", 
            "MeanErr ± Std", 
            "±1.96σ"))
        println(io, "-"^120)
        for i in 1:length(h_err_vals)
            println(io, @sprintf(
                "%-9.4f %-22s %-22.4f %-22.4f %-22.4f %-16s %-16.4f",
                h_err_vals[i],
                @sprintf("%.5f ± %.5f", mean_times[i], std_times[i]),
                mean_times_p1s[i], mean_times_m1s[i], mean_times_errbar[i],
                @sprintf("%.5f ± %.5f", mean_errs[i], std_errs[i]),
                mean_errs_errbar[i]
            ))
        end
        println(io, "="^120)
        println(io, "")
        println(io, "This table includes the mean, stddev, mean±1σ, and mean±1.96σ (approx 95% conf. error bars) for both quantity and error rate.")
    end
    log_msg("Saved: $summary_txt_filename")
end

#------------------------------------------
# Output all strategies (θ vectors/params) as a nice text file
#------------------------------------------
strategies_txt_filename = "all_N$(N)_M$(M)_strategies_per_h_error.txt"
open(strategies_txt_filename, "w") do io
    println(io, "Summary of inferred control strategies (parameter vectors) for each h_error.")
    println(io, "="^98)
    for (k, strat) in enumerate(strategy_results)
        println(io, "\n" * "-"^98)
        println(io, "Strategy #$k for h_error = $(strat["h_error"]):")
        println(io, "  Parameter file: $(strat["theta_star_file"])")
        println(io, "  θ_star (file vector, comma-delimited):")
        println(io, "    " * join(strat["theta_star"], ", "))
        println(io, "  θ_full (used for all M cells in simulation):")
        println(io, "    [")
        for x in strat["theta_full"]
            println(io, "      ", @sprintf("%.8f", x))
        end
        println(io, "    ]")
    end
    println(io, "\n" * "-"^98)
    println(io, "Done. Each strategy block above lists:")
    println(io, "  * the error constraint used (h_error value)")
    println(io, "  * the parameter file loaded")
    println(io, "  * the θ_star vector as read from file (compressed)")
    println(io, "  * the full θ parameter vector as used in simulation (one line per value, 8 decimals)")
end
log_msg("Saved: $strategies_txt_filename")

#--------------------------------
# Final summary
#--------------------------------
log_msg("\n" * "="^60)
log_msg("All processing completed!")
log_msg("="^60)

# Close the log file
close(logfile)

println("\nAll done! Check log file: $logfile_name")
