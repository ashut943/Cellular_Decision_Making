# This script is for calculating and plotting instantaneous information measures
# for the three cell system. Computes I(u₁;u₂), I(u₁;u₃), I((u₁,u₂);u₃), 
# and correlational information CI(u₁;u₂;u₃) using probability evolution.

using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, LaTeXStrings
using Measures
using Revise
using CellularDecisions
using ExponentialUtilities
using Statistics

include("../../../mult_cell/mult_cell.jl")
include("../../../utils/utils.jl")

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# CONFIGURATION
#--------------------------------

N = 6
M_cell = 3
K = 3
initial_state_array = ((1,0),(1,0),(1,0))
type_of_boundary_condition = "boundary_2"
h_error = 0.02

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# OUTPUT DIRECTORY SETUP
#--------------------------------

println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("Now running for error rate: ", h_error)

error_str = replace(string(round(h_error*100, digits=1)), "." => "_")
base_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "results", "three_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_error_fix_%s", N, error_str))

folder_name_for_plots = joinpath(dirname(dirname(dirname(@__DIR__))), "plots", "three_cell_results", type_of_boundary_condition, "N_$(N)_error_fix_$(error_str)"*"_for_static_vs_dynamic_info_plots")
mkpath(folder_name_for_plots)

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Get state matrices, sizes, and target states

statedict, statedictinv, terminal_states, TG, TB, Tc = CellularDecisions.statematrices(N, M_cell, type_of_boundary_condition)
ni, np = CellularDecisions.varioussizes(N, M_cell)
ns = length(Tc)
targetstates_good = [target_state+1 for target_state ∈ TG]
targetstates_bad = [target_state+1 for target_state ∈ TB]
targetstates = [targetstates_good; targetstates_bad]
startstates = [start_state+1 for start_state ∈ Tc]
allstates = [startstates; targetstates_good; targetstates_bad]
all_targetstates = vcat(targetstates_good, targetstates_bad)
initial_state = statedictinv[initial_state_array] + 1

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Load optimized system

threecell_system_filename = generate_filename(folder_name, "threecell_system_global")
threecell_system = CellularDecisions.load(threecell_system_filename)
Q_opt_absorbing = threecell_system.Q_matrix

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Display optimal values

hitting_values = hitting_time_mod(Q_opt_absorbing, targetstates_good, targetstates_bad, startstates, 0.0)
h_global = hitting_prob_mod_huh(Q_opt_absorbing, targetstates_good, targetstates_bad, startstates, 0.0)
hitting_values_expected = hitting_values
h_global_expected = h_global

println("Expected hitting time: ", hitting_values_expected[initial_state])
println("Expected hitting prob: ", h_global_expected[initial_state])
println("h_error: ", h_error)
println("Target states good: ", length(targetstates_good))
println("Target states bad: ", length(targetstates_bad))
println("------------------------------------------------")

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Probability evolution setup

T = 50.0
pi_initial = zeros(1, ni)
pi_initial[initial_state] = 1
time_points = collect(range(0, T, length=1000))

pi_in_time = Array{Float64}(undef, length(time_points), ni)
pi_in_time[1, :] = reshape(pi_initial, 1, ni)

Q_transpose = Matrix(Q_opt_absorbing')
transpose_vector = vec(pi_initial')

states_u_1 = [(i, j) for i in 0:N for j in 0:1]
states_u_2 = [(i, j) for i in 0:N for j in 0:1]
states_u_3 = [(i, j) for i in 0:N for j in 0:1]

distributions_u1 = []
distributions_u2 = []
distributions_u3 = []

mutual_information_u1_u2 = zeros(Float64, length(time_points))
mutual_information_u1_u3 = zeros(Float64, length(time_points))
mutual_information_u1u2_u3 = zeros(Float64, length(time_points))
correlational_information_u1_u2_u3 = zeros(Float64, length(time_points))

println("Q matrix valid: diag <= 0: ", all(diag(Q_opt_absorbing) .<= 0))
println("Q matrix valid: rows sum to 0: ", all(isapprox.(sum(Q_opt_absorbing, dims=2), 0, atol=1e-10)))

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Main loop: probability evolution and mutual information calculation

for (i, t) in enumerate(time_points)
    delta_t = (i == 1) ? 0.0 : t - time_points[i-1]
    transpose_vector = (i == 1) ? vec(pi_initial') : vec(pi_in_time[i-1, :]')
    
    # Compute matrix exponential for probability evolution
    exp_result = ExponentialUtilities.expv(delta_t, Q_transpose, transpose_vector)
    curr_pi_in_time = reshape(Matrix(exp_result'), ni)
    pi_in_time[i, :] = curr_pi_in_time
    
    # Renormalize if necessary
    if sum(pi_in_time[i, :]) > 1.0
        println("Warning: sum of probabilities > 1.0 at t=$t, renormalizing")
        pi_in_time[i, :] ./= sum(pi_in_time[i, :])
    end
    
    curr_pi_in_time = pi_in_time[i, :]
    
    # Initialize probability dictionaries
    P_u1_dict = Dict(state => 0.0 for state in 0:N)
    P_u2_dict = Dict(state => 0.0 for state in 0:N)
    P_u3_dict = Dict(state => 0.0 for state in 0:N)
    P_u1u2_dict = Dict((s1, s2) => 0.0 for s1 in 0:N for s2 in 0:N)
    P_u1u3_dict = Dict((s1, s3) => 0.0 for s1 in 0:N for s3 in 0:N)
    P_u2u3_dict = Dict((s2, s3) => 0.0 for s2 in 0:N for s3 in 0:N)
    P_u1u2u3_dict = Dict((s1, s2, s3) => 0.0 for s1 in 0:N for s2 in 0:N for s3 in 0:N)
    
    P_000, P_100, P_011, P_111 = 0.0, 0.0, 0.0, 0.0
    
    # Compute marginal and joint distributions
    for state_1 in states_u_1
        for state_2 in states_u_2
            for state_3 in states_u_3
                curr_full_state = (state_1, state_2, state_3)
                curr_full_state_index = Int(statedictinv[curr_full_state] + 1)
                prob = curr_pi_in_time[curr_full_state_index]
                
                P_u1_dict[state_1[1]] += prob
                P_u2_dict[state_2[1]] += prob
                P_u3_dict[state_3[1]] += prob
                P_u1u2_dict[(state_1[1], state_2[1])] += prob
                P_u1u3_dict[(state_1[1], state_3[1])] += prob
                P_u2u3_dict[(state_2[1], state_3[1])] += prob
                P_u1u2u3_dict[(state_1[1], state_2[1], state_3[1])] += prob
                
                # Track terminal state probabilities
                if state_1[1] == 0 && state_2[1] == 0 && state_3[1] == 0
                    P_000 += prob
                end
                if (state_1[1] == N && state_2[1] == 0 && state_3[1] == 0) || 
                   (state_1[1] == 0 && state_2[1] == N && state_3[1] == 0) || 
                   (state_1[1] == 0 && state_2[1] == 0 && state_3[1] == N)
                    P_100 += prob
                end
                if (state_1[1] == 0 && state_2[1] == N && state_3[1] == N) || 
                   (state_1[1] == N && state_2[1] == 0 && state_3[1] == N) || 
                   (state_1[1] == N && state_2[1] == N && state_3[1] == 0)
                    P_011 += prob
                end
                if state_1[1] == N && state_2[1] == N && state_3[1] == N
                    P_111 += prob
                end
            end
        end
    end
    
    # Compute mutual information I(u₁;u₂)
    mi_u1_u2 = 0.0
    for s1 in 0:N, s2 in 0:N
        p_joint = P_u1u2_dict[(s1, s2)]
        p1, p2 = P_u1_dict[s1], P_u2_dict[s2]
        if p_joint > 0 && p1 > 0 && p2 > 0
            mi_u1_u2 += p_joint * log(p_joint / (p1 * p2))
        end
    end
    
    # Compute mutual information I((u₁,u₂);u₃)
    mi_u1u2_u3 = 0.0
    for s1 in 0:N, s2 in 0:N, s3 in 0:N
        p_joint = P_u1u2u3_dict[(s1, s2, s3)]
        p12, p3 = P_u1u2_dict[(s1, s2)], P_u3_dict[s3]
        if p_joint > 0 && p12 > 0 && p3 > 0
            mi_u1u2_u3 += p_joint * log(p_joint / (p12 * p3))
        end
    end
    
    # Compute mutual information I(u₁;u₃)
    mi_u1_u3 = 0.0
    for s1 in 0:N, s3 in 0:N
        p_joint = P_u1u3_dict[(s1, s3)]
        p1, p3 = P_u1_dict[s1], P_u3_dict[s3]
        if p_joint > 0 && p1 > 0 && p3 > 0
            mi_u1_u3 += p_joint * log(p_joint / (p1 * p3))
        end
    end
    
    # Compute correlational information CI(u₁;u₂;u₃)
    ci_u1_u2_u3 = 0.0
    for s1 in 0:N, s2 in 0:N, s3 in 0:N
        p_joint = P_u1u2u3_dict[(s1, s2, s3)]
        p1, p2, p3 = P_u1_dict[s1], P_u2_dict[s2], P_u3_dict[s3]
        if p_joint > 0 && p1 > 0 && p2 > 0 && p3 > 0
            ci_u1_u2_u3 += p_joint * log(p_joint / (p1 * p2 * p3))
        end
    end
    
    mutual_information_u1_u2[i] = mi_u1_u2
    mutual_information_u1u2_u3[i] = mi_u1u2_u3
    mutual_information_u1_u3[i] = mi_u1_u3
    correlational_information_u1_u2_u3[i] = ci_u1_u2_u3 / 3.0
    
    println("t=$t: I(u₁;u₂)=$(round(mi_u1_u2, digits=4)), I((u₁,u₂);u₃)=$(round(mi_u1u2_u3, digits=4)), I(u₁;u₃)=$(round(mi_u1_u3, digits=4)), CI=$(round(ci_u1_u2_u3, digits=4))")
    println("P_000=$P_000, P_100=$P_100, P_011=$P_011, P_111=$P_111")
    println("------------------------------------------------")
end

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Plotting instantaneous information metrics over time

# I((u₁,u₂);u₃) over time
p = plot(time_points, mutual_information_u1u2_u3,
    xlabel="Time", ylabel="Mutual Information I((u₁,u₂);u₃)",
    title="Mutual Information between (u₁,u₂) and u₃ over Time",
    label="I((u₁,u₂);u₃)", linewidth=2, color=:blue,
    size=(800, 600), margin=5mm, grid=false, framestyle=:box, legend=:topright)
savefig(p, generate_filename(folder_name_for_plots, "mutual_information_u1_u2_u3_over_time.png"))
savefig(p, generate_filename(folder_name_for_plots, "mutual_information_u1_u2_u3_over_time.svg"))

# I(u₁;u₂) over time
p = plot(time_points, mutual_information_u1_u2,
    xlabel="Time", ylabel="Mutual Information I(u₁;u₂)",
    title="Mutual Information between u₁ and u₂ over Time",
    label="I(u₁;u₂)", linewidth=2, color=:blue,
    size=(800, 600), margin=5mm, grid=false, framestyle=:box, legend=:topright)
savefig(p, generate_filename(folder_name_for_plots, "mutual_information_u1_u2_over_time.png"))
savefig(p, generate_filename(folder_name_for_plots, "mutual_information_u1_u2_over_time.svg"))

# I(u₁;u₃) over time
p = plot(time_points, mutual_information_u1_u3,
    xlabel="Time", ylabel="Mutual Information I(u₁;u₃)",
    title="Mutual Information between u₁ and u₃ over Time",
    label="I(u₁;u₃)", linewidth=2, color=:blue,
    size=(800, 600), margin=5mm, grid=false, framestyle=:box, legend=:topright)
savefig(p, generate_filename(folder_name_for_plots, "mutual_information_u1_u3_over_time.png"))
savefig(p, generate_filename(folder_name_for_plots, "mutual_information_u1_u3_over_time.svg"))

# CI(u₁;u₂;u₃) over time
p = plot(time_points, correlational_information_u1_u2_u3,
    xlabel="Time", ylabel="Correlational Information CI(u₁;u₂;u₃)",
    title="Correlational Information over Time",
    label="CI(t)", linewidth=2, color=:blue,
    size=(800, 600), margin=5mm, grid=false, framestyle=:box, legend=:topright)
savefig(p, generate_filename(folder_name_for_plots, "correlational_information_u1_u2_u3_over_time.png"))
savefig(p, generate_filename(folder_name_for_plots, "correlational_information_u1_u2_u3_over_time.svg"))

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Plotting combined information measures

p_combined = plot(time_points, correlational_information_u1_u2_u3,
    xlabel="Time", ylabel="Information Measures",
    title="Information Measures over Time",
    label="CI(t)", linewidth=3, color="#6a3d9a",
    size=(800, 600), margin=5mm, grid=false, framestyle=:box, legend=:topright)
plot!(time_points, mutual_information_u1_u2, label="I(u₁;u₂)", linewidth=3, color="#1f78b4")
plot!(time_points, mutual_information_u1u2_u3, label="I((u₁,u₂);u₃)", linewidth=3, color="#33a02c")

savefig(p_combined, generate_filename(folder_name_for_plots, "all_information_measures_over_time.png"))
savefig(p_combined, generate_filename(folder_name_for_plots, "all_information_measures_over_time.svg"))

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Hitting probability calculations for terminal states

h_global = hitting_prob_mod(Q_opt_absorbing, targetstates_good, targetstates_bad, startstates, 0.0)
println("h_global: ", h_global[initial_state])

# Probability of hitting (0,0,0)
target_states_000_array = vec([((0, x1), (0, x2), (0, x3)) for x1 in 0:1, x2 in 0:1, x3 in 0:1])
target_states_000 = [statedictinv[state_huh] + 1 for state_huh in target_states_000_array]
target_states_000_left = [s for s in all_targetstates if !(s in target_states_000)]
h_000 = hitting_prob_mod(Q_opt_absorbing, target_states_000_left, target_states_000, startstates, 0.0)
p_000 = h_000[initial_state]
println("p_000: ", p_000)

# Probability of hitting (N,0,0) or permutations
target_states_100_array = vec(vcat(
    [((N, x1), (0, x2), (0, x3)) for x1 in 0:1, x2 in 0:1, x3 in 0:1],
    [((0, x1), (N, x2), (0, x3)) for x1 in 0:1, x2 in 0:1, x3 in 0:1],
    [((0, x1), (0, x2), (N, x3)) for x1 in 0:1, x2 in 0:1, x3 in 0:1]))
target_states_100 = [statedictinv[state_huh] + 1 for state_huh in target_states_100_array]
target_states_100_left = [s for s in all_targetstates if !(s in target_states_100)]
h_100 = hitting_prob_mod(Q_opt_absorbing, target_states_100_left, target_states_100, startstates, 0.0)
p_100 = h_100[initial_state]
println("p_100: ", p_100)

# Probability of hitting (0,N,N) or permutations
target_states_011_array = vec(vcat(
    [((0, x1), (N, x2), (N, x3)) for x1 in 0:1, x2 in 0:1, x3 in 0:1],
    [((N, x1), (0, x2), (N, x3)) for x1 in 0:1, x2 in 0:1, x3 in 0:1],
    [((N, x1), (N, x2), (0, x3)) for x1 in 0:1, x2 in 0:1, x3 in 0:1]))
target_states_011 = [statedictinv[state_huh] + 1 for state_huh in target_states_011_array]
target_states_011_left = [s for s in all_targetstates if !(s in target_states_011)]
h_011 = hitting_prob_mod(Q_opt_absorbing, target_states_011_left, target_states_011, startstates, 0.0)
p_011 = h_011[initial_state]
println("p_011: ", p_011)

# Probability of hitting (N,N,N)
target_states_111_array = vec([((N, x1), (N, x2), (N, x3)) for x1 in 0:1, x2 in 0:1, x3 in 0:1])
target_states_111 = [statedictinv[state_huh] + 1 for state_huh in target_states_111_array]
target_states_111_left = [s for s in all_targetstates if !(s in target_states_111)]
h_111 = hitting_prob_mod(Q_opt_absorbing, target_states_111_left, target_states_111, startstates, 0.0)
p_111 = h_111[initial_state]
println("p_111: ", p_111)

# Verify probabilities sum to 1
total_sum = p_111 + p_000 + p_011 + p_100
if !isapprox(total_sum, 1.0, atol=1e-10)
    println("Warning: Probabilities do not sum to 1. Sum is: $total_sum. Renormalizing...")
    p_000 /= total_sum
    p_100 /= total_sum
    p_011 /= total_sum
    p_111 /= total_sum
    println("Renormalized sum: ", p_111 + p_000 + p_011 + p_100)
end
println("------------------------------------------------")

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Final state mutual information (analytical)

a = p_000
b = p_100 / 3
c = p_011 / 3
d = p_111
q = a + 2*b + c

final_state_mi = (a+b)*log((a+b)/(q*q)) + 2*(b+c)*log((b+c)/(q*(1-q))) + (c+d)*log((c+d)/((1-q)*(1-q)))
final_state_mi_two = a*log(a/((a+b)*(a+2*b+c))) + b*log(b/((a+b)*(b+2*c+d))) + 
                     2*b*log(b/((b+c)*(a+2*b+c))) + 2*c*log(c/((b+c)*(b+2*c+d))) + 
                     c*log(c/((c+d)*(a+2*b+c))) + d*log(d/((c+d)*(b+2*c+d)))
final_state_ci = (a*log(a/(q*q*q)) + 3*b*log(b/(q*q*(1-q))) + 
                  3*c*log(c/((1-q)*(1-q)*q)) + d*log(d/((1-q)*(1-q)*(1-q)))) / 3.0

println("Final state MI (pairwise): ", final_state_mi)
println("Final state MI (three-way): ", final_state_mi_two)
println("Final state CI: ", final_state_ci)

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Summary of instantaneous information measures

final_mi_u1_u2 = mutual_information_u1_u2[end]
final_mi_u1_u3 = mutual_information_u1_u3[end]
final_mi_u1u2_u3 = mutual_information_u1u2_u3[end]
final_ci_u1_u2_u3 = correlational_information_u1_u2_u3[end]

peak_mi_u1_u2 = maximum(mutual_information_u1_u2)
peak_mi_u1_u3 = maximum(mutual_information_u1_u3)
peak_mi_u1u2_u3 = maximum(mutual_information_u1u2_u3)
peak_ci_u1_u2_u3 = maximum(correlational_information_u1_u2_u3)

time_peak_mi_u1_u2 = time_points[argmax(mutual_information_u1_u2)]
time_peak_mi_u1_u3 = time_points[argmax(mutual_information_u1_u3)]
time_peak_mi_u1u2_u3 = time_points[argmax(mutual_information_u1u2_u3)]
time_peak_ci_u1_u2_u3 = time_points[argmax(correlational_information_u1_u2_u3)]

println("I(u₁;u₂): peak=$(round(peak_mi_u1_u2, digits=4)), final=$(round(final_mi_u1_u2, digits=4)), diff=$(round(peak_mi_u1_u2-final_mi_u1_u2, digits=4)), t_peak=$(round(time_peak_mi_u1_u2, digits=2))")
println("I(u₁;u₃): peak=$(round(peak_mi_u1_u3, digits=4)), final=$(round(final_mi_u1_u3, digits=4)), diff=$(round(peak_mi_u1_u3-final_mi_u1_u3, digits=4)), t_peak=$(round(time_peak_mi_u1_u3, digits=2))")
println("I((u₁,u₂);u₃): peak=$(round(peak_mi_u1u2_u3, digits=4)), final=$(round(final_mi_u1u2_u3, digits=4)), diff=$(round(peak_mi_u1u2_u3-final_mi_u1u2_u3, digits=4)), t_peak=$(round(time_peak_mi_u1u2_u3, digits=2))")
println("CI(u₁;u₂;u₃): peak=$(round(peak_ci_u1_u2_u3, digits=4)), final=$(round(final_ci_u1_u2_u3, digits=4)), diff=$(round(peak_ci_u1_u2_u3-final_ci_u1_u2_u3, digits=4)), t_peak=$(round(time_peak_ci_u1_u2_u3, digits=2))")
println("================================")
