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

include("utils/utils.jl") 
include("mult_cell/mult_cell.jl")
include("sampling_based_optimization/sampling_based_opti.jl")

#==============================================================================#
#                           CONFIGURATION                                      #
#==============================================================================#

N = 6  # Number of states - 1
M = 7  # Number of cells
h_error = 0.02  # Target error rate
initial_state_array = ((1,0), (1,0), (1,0), (1,0), (1,0), (1,0), (1,0))
type_of_boundary_condition = "boundary_2"

#==============================================================================#
#                           LOGGING SETUP                                      #
#==============================================================================#

logfile_name = "only_center_sgd_N$(N)_M$(M)_he$(replace(string(h_error), "." => "p"))_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log"
logfile = open(logfile_name, "w")

function log_msg(msg)
    println(msg)
    println(logfile, msg)
    flush(stdout)
    flush(stderr)
    flush(logfile)
end

log_msg("=== Starting SGD optimization ===")
log_msg("N = $N, M = $M, h_error = $h_error")
log_msg("Log file: $logfile_name")

#==============================================================================#
#                           ADJACENCY MATRIX                                   #
#==============================================================================#

AdjMat = zeros(Int, M, M)

AdjMat[1,2] = 1; AdjMat[1,3] = 1; AdjMat[1,4] = 1
AdjMat[1,5] = 1; AdjMat[1,6] = 1; AdjMat[1,7] = 1
AdjMat[2,1] = 1; AdjMat[2,3] = 1; AdjMat[2,7] = 1
AdjMat[3,1] = 1; AdjMat[3,2] = 1; AdjMat[3,4] = 1
AdjMat[4,1] = 1; AdjMat[4,3] = 1; AdjMat[4,5] = 1
AdjMat[5,1] = 1; AdjMat[5,4] = 1; AdjMat[5,6] = 1
AdjMat[6,1] = 1; AdjMat[6,5] = 1; AdjMat[6,7] = 1
AdjMat[7,1] = 1; AdjMat[7,2] = 1; AdjMat[7,6] = 1

T_horizon = 300.0
log_msg("AdjMat: $AdjMat")
log_msg("T_horizon: $T_horizon")

#==============================================================================#
#                           TERMINAL STATES                                    #
#==============================================================================#

TG_proper = [
    (N, 0, 0, 0, 0, 0, 0),
    (0, N, 0, 0, N, 0, 0),
    (0, 0, N, 0, 0, N, 0),
    (0, 0, 0, N, 0, 0, N),
    (0, N, 0, N, 0, N, 0),
    (0, 0, N, 0, N, 0, N)
]

TB_proper = []
for u_1 in [0, N]
    for u_2 in [0, N]
        for u_3 in [0, N]
            for u_4 in [0, N]
                for u_5 in [0, N]
                    for u_6 in [0, N]
                        for u_7 in [0, N]
                            curr = (u_1, u_2, u_3, u_4, u_5, u_6, u_7)
                            if curr ∉ TG_proper && curr ∉ TB_proper
                                push!(TB_proper, curr)
                            end
                        end
                    end
                end
            end
        end
    end
end

log_msg("TG_proper: $TG_proper")
log_msg("TB_proper: $(length(TB_proper)) bad states")

#==============================================================================#
#                           INITIAL PARAMETERS                                 #
#==============================================================================#

# initial parameter vector for optimization
θ1_initial = [
    0.03267535067141225, 0.03005574245984719, 0.7424934427333019, 0.999998504619645, 0.997224474458932,
    0.0, 0.0, 0.009709612682650784, 0.016265159707732284, 0.017488701382889396,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.8608396516039294, 0.9997260908704086, 1.0, 1.0, 0.05722003157483944,
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.12566278559435934
]

log_msg("θ0 length: $(length(θ1_initial))")
log_msg("Expected length (5N-2): $(5*N - 2)")
log_msg("θ1_initial: $θ1_initial")

#==============================================================================#
#                           INITIAL VALIDATION                                 #
#==============================================================================#

log_msg("\n================================================")
log_msg("Starting INITIAL VALIDATION")
log_msg("================================================")

mean_T_init, mean_bad_init = validate_params(
    θ1_initial, N, M, AdjMat, initial_state_array, T_horizon, TG_proper, TB_proper, 10000;
    log_fn=log_msg
)
log_msg("Initial validation - mean terminal time: $mean_T_init")
log_msg("Initial validation - mean is_bad: $mean_bad_init")

#==============================================================================#
#                           WARM START PHASE                                   #
#==============================================================================#

log_msg("\n================================================")
log_msg("Starting WARM START phase")
log_msg("================================================")

θ_star_warm, lambda_star_warm, logs_warm = run_ctmc_projected_sgd!(
    θ1_initial;
    iters=3000,
    N_sim=500,
    η_max=1e-3,
    η_min=1e-5,
    lr_T0=500,
    lr_Tmult=1.0,
    ρ=100.0,
    ε_tol=h_error,
    N=N,
    M=M,
    AdjMat=AdjMat,
    initial_state=initial_state_array,
    T=T_horizon,
    TG=TG_proper,
    TB=TB_proper,
    window_size=100,
    min_iters=40,
    tol_feas=0.02,
    tol_T=1e-1,
    tol_λ=1e-1,
    do_early_stopping=true,
    λ_start=0.0,
    log_fn=log_msg
)

log_msg("\n================================================")
log_msg("WARM START completed")
log_msg("================================================")

#==============================================================================#
#                           WARM START VALIDATION                              #
#==============================================================================#

log_msg("\n================================================")
log_msg("Starting WARMUP VALIDATION")
log_msg("================================================")

mean_T_warm, mean_bad_warm = validate_params(
    θ_star_warm, N, M, AdjMat, initial_state_array, T_horizon, TG_proper, TB_proper, 1000;
    log_fn=log_msg
)
log_msg("WARMUP validation - mean terminal time: $mean_T_warm")
log_msg("WARMUP validation - mean is_bad: $mean_bad_warm")
log_msg("θ_star_warm: $θ_star_warm")

#==============================================================================#
#                           MAIN OPTIMIZATION                                  #
#==============================================================================#

log_msg("\n================================================")
log_msg("Starting MAIN SGD")
log_msg("================================================")

θ_star, lambda_star, logs = run_ctmc_projected_sgd!(
    θ_star_warm;
    iters=100000,
    N_sim=1000,
    η_max=1e-4,
    η_min=1e-7,
    lr_T0=500,
    lr_Tmult=1.0,
    ρ=100.0,
    ε_tol=h_error,
    N=N,
    M=M,
    AdjMat=AdjMat,
    initial_state=initial_state_array,
    T=T_horizon,
    TG=TG_proper,
    TB=TB_proper,
    window_size=200,
    min_iters=40,
    tol_feas=0.07,
    tol_T=5e-2,
    tol_λ=1e-2,
    do_early_stopping=true,
    λ_start=lambda_star_warm,
    log_fn=log_msg
)

#==============================================================================#
#                           FINAL VALIDATION                                   #
#==============================================================================#

log_msg("\n================================================")
log_msg("Starting FINAL VALIDATION")
log_msg("================================================")

mean_T_final, mean_bad_final = validate_params(
    θ_star, N, M, AdjMat, initial_state_array, T_horizon, TG_proper, TB_proper, 1000;
    log_fn=log_msg
)

θ_full = params_to_full_vector(θ_star, N)
log_msg("Final θ_star (optimized): $θ_star")
log_msg("Final θ_full (with boundaries): $θ_full")
log_msg("Final lambda_star: $lambda_star")
log_msg("Final validation - mean terminal time: $mean_T_final")
log_msg("Final validation - mean is_bad: $mean_bad_final")

log_msg("\n================================================")
log_msg("Optimization completed successfully!")
log_msg("================================================")

close(logfile)

#==============================================================================#
#                           PLOTTING & SAVING RESULTS                          #
#==============================================================================#

iters = 1:length(logs.T_hist)

plot_width = 2400
plot_height = 600

function idx_last(n, arrlen)
    n_last = min(n, arrlen)
    return (arrlen - n_last + 1):arrlen
end

ranges_and_labels = [
    (1:length(iters), "full"),
    (idx_last(10000, length(iters)), "last10k"),
    (idx_last(1000, length(iters)), "last1k"),
]

he_str = replace(string(h_error), "." => "p")
base_name = "proper_sgd_N$(N)_M$(M)_he$(he_str)"

for (idx_range, label) in ranges_and_labels
    # Terminal Time
    p1 = plot(iters[idx_range], logs.T_hist[idx_range], xlabel="Iteration", ylabel="T̄", label="T̄",
        title="Mean Terminal Time ($label)", size=(plot_width, plot_height))
    savefig(p1, "$(base_name)_logs_Tmean_$label.png")

    # Bad Probability
    p2 = plot(iters[idx_range], logs.b_hist[idx_range], xlabel="Iteration", ylabel="b̄", label="b̄",
        title="Mean Bad Probability ($label)", size=(plot_width, plot_height), yrange=(0,1))
    savefig(p2, "$(base_name)_logs_BadProb_$label.png")

    # Lambda
    p3 = plot(iters[idx_range], logs.λ_hist[idx_range], xlabel="Iteration", ylabel="λ", label="λ",
        title="Dual Variable (λ) ($label)", size=(plot_width, plot_height))
    savefig(p3, "$(base_name)_logs_Lambda_$label.png")

    # Gradient Norm
    p4 = plot(iters[idx_range], logs.gnorm[idx_range], xlabel="Iteration", ylabel="‖gθ‖₂", label="‖gθ‖₂",
        title="Gradient Norm ($label)", size=(plot_width, plot_height))
    savefig(p4, "$(base_name)_logs_GNorm_$label.png")

    # Augmented Loss
    p5 = plot(iters[idx_range], logs.Loss_hist[idx_range], xlabel="Iteration", ylabel="Loss", label="Loss",
        title="Augmented Loss ($label)", size=(plot_width, plot_height))
    savefig(p5, "$(base_name)_logs_Loss_$label.png")
end

#==============================================================================#

println("Saving optimization results...")

open("$(base_name)_final_results.txt", "w") do io
    println(io, "CTMC Optimization Results")
    println(io, "========================")
    println(io, "")
    println(io, "θ_star (final optimized parameters):")
    println(io, join(θ_star, ", "))
    println(io, "")
    println(io, "λ_star (final dual variable):")
    println(io, @sprintf("%.6f", lambda_star))
    println(io, "")
    println(io, "mean(terminal_times): $(@sprintf("%.6f", mean_T_final))")
    println(io, "mean(is_bad): $(@sprintf("%.6f", mean_bad_final))")
    println(io, "")
    println(io, "SGD optimization configuration:")
    println(io, "N = $N, M = $M, T_horizon = $T_horizon, h_error = $h_error, boundary_condition = $type_of_boundary_condition")
    println(io, "SGD iters = $(length(logs.T_hist))")
    println(io, "Initial state: $(initial_state_array)")
end

logdict = Dict(
    "T_hist"    => logs.T_hist,
    "b_hist"    => logs.b_hist,
    "λ_hist"    => logs.λ_hist,
    "gnorm"     => logs.gnorm,
    "Loss_hist" => logs.Loss_hist,
    "stopped_early" => get(logs, :stopped_early, false),
    "iter_last"     => get(logs, :iter_last, length(logs.T_hist))
)
open("$(base_name)_logs.json", "w") do io
    JSON.print(io, logdict)
end

open("$(base_name)_theta_star.txt", "w") do io
    println(io, join(θ_star, ", "))
end
open("$(base_name)_lambda_star.txt", "w") do io
    println(io, @sprintf("%.12f", lambda_star))
end

println("Results saved: $(base_name)_final_results.txt, $(base_name)_theta_star.txt, $(base_name)_lambda_star.txt, $(base_name)_logs.json")

