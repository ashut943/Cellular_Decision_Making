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

include("../../utils/utils.jl")
include("../../mult_cell/mult_cell.jl")

M = 7
N = 6

#--------------------------------
# Output directories
#--------------------------------
folder_name_for_plots = joinpath(dirname(dirname(@__DIR__)), "plots", "seven_cell_results", "arrhenius_landscape")
mkpath(folder_name_for_plots)

#--------------------------------
# Parameter sets for different h_error values
#--------------------------------
parameter_sets = Dict(
    0.02 => Dict(
        :k_off =>  [0.16672773604385013],
        :f_plus_0 =>  [0.0, 0.03295253588963976, 0.029389852058424044, 0.7030231251614495, 0.9808367795978125, 0.9795684377588251],
        :f_plus_1 =>  [0.0, 0.001738214156646322, 0.0, 0.040132518419628525, 0.04818356155078206, 0.006481623816163985],
        :f_minus_0 =>  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        :f_minus_1 =>  [0.8331518414863534, 1.0, 1.0, 0.9999925017306047, 0.09335175912092401, 0.0],
        :g =>  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],         
    ),
    0.10 => Dict(
        :k_off =>  [7.533745440929073e-5],
        :f_plus_0 =>  [0.0, 0.093302274624027, 0.11843185725126298, 0.9761468934321484, 0.9952473255771057, 1.0],
        :f_plus_1 =>  [0.0, 0.016745981117983002, 6.82224843288691e-6, 0.03488871856494207, 0.6578944998371575, 0.8934179979365698],
        :f_minus_0 =>  [0.0, 0.023343739073633813, 0.0, 0.0, 0.0, 0.0],
        :f_minus_1 =>  [1.0, 0.9999846591733771, 1.0, 0.9999954972230595, 0.09890088258840768, 0.0],
        :g =>  [0.0, 0.0, 2.7006761716804e-7, 1.0, 1.0, 1.0, 1.0],  
    ),
    0.20 => Dict(
        :k_off =>  [0.01054550330391466],
        :f_plus_0 =>  [0.0, 0.16836270474596915, 0.2753352416767114, 0.9687001662254024, 1.0, 1.0],
        :f_plus_1 =>  [0.0, 0.0, 2.1473627891133393e-5, 0.00714360769246155, 0.43537400430146267, 0.9642559371687264],
        :f_minus_0 =>  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        :f_minus_1 =>  [1.0, 1.0, 1.0, 1.0, 0.09248859339371171, 0.0],
        :g =>  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    ),
    0.30 => Dict(
        :k_off =>  [0.0120371516830899],
        :f_plus_0 =>  [0.0, 0.22872106786677274, 0.6493203274778565, 1.0, 1.0, 1.0],
        :f_plus_1 =>  [0.0, 0.0, 8.992381370551024e-5, 0.0009127773534195543, 0.8655391021841581, 1.0],
        :f_minus_0 =>  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        :f_minus_1 =>  [1.0, 1.0, 1.0, 1.0, 0.005075255387805302, 0.0],
        :g =>  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    ),
    0.40 => Dict(
        :k_off =>  [0.000311606459694454],
        :f_plus_0 =>  [0.0, 0.3632173816914747, 0.8302311601428864, 1.0, 1.0, 1.0],
        :f_plus_1 =>  [0.0, 0.0, 8.40175045251951e-5, 0.013366098258732403, 1.0, 1.0],
        :f_minus_0 =>  [0.00020870775280686741, 0.0, 0.0, 0.0, 0.0, 0.0],
        :f_minus_1 =>  [1.0, 1.0, 1.0, 0.9592969010770217, 0.0024771068983254355, 0.0],
        :g =>  [0.0010060164555596, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    ),
    0.50 => Dict(
        :k_off =>  [2.2390907298802878e-5],
        :f_plus_0 =>  [0.0, 0.5504078567707597, 0.9969490754426136, 1.0, 1.0, 1.0],
        :f_plus_1 =>  [0.0, 0.0, 4.2468400881961674e-5, 0.07980330403597179, 1.0, 1.0],
        :f_minus_0 =>  [0.022337579944871032, 0.0, 0.0, 0.0, 0.0, 0.0],
        :f_minus_1 =>  [1.0, 1.0, 1.0, 0.7959486617223295, 0.005442165783440641, 0.0],
        :g =>  [0.0022583424589338844, 0.0, 0.0011449087396559628, 1.0, 1.0, 1.0, 1.0]    
    )
)

h_error_list = [0.02, 0.10, 0.20, 0.30, 0.40, 0.50]

#--------------------------------
# Energy landscape computations
#--------------------------------

function compute_energy_landscape(f_plus_0, f_minus_0, f_plus_1, f_minus_1, N, epsilon=1e-20)
    # Compute E for s=0
    E = zeros(N+1)
    for u in 1:N-1
        f_p_curr = f_plus_0[u] == 0.0 ? epsilon : f_plus_0[u]
        f_m_curr = f_minus_0[u] == 0.0 ? epsilon : f_minus_0[u]
        
        if u != N && u != 1
            E[u+1] = E[u] - log(f_p_curr) + log(f_m_curr) 
        end
    end
    
    if abs(f_minus_0[1]) == 0.0
        E[1] = E[2] - log(f_minus_0[1] + epsilon) + log(epsilon) 
    else
        E[1] = E[2] + log(epsilon) - log(f_minus_0[1] + epsilon) 
    end
    
    if abs(f_plus_0[N]) == 0.0
        E[N+1] = E[N] - log(f_plus_0[N] + epsilon) + log(epsilon) 
    else
        E[N+1] = E[N] - log(f_plus_0[N]) + log(epsilon) 
    end
    
    # Compute barriers and forces for s=0
    B = zeros(N+1, N+1)
    F = zeros(N+1, N+1)
    for u in 0:N-1
        curr_ind_1 = u + 1
        B[curr_ind_1, curr_ind_1+1] = 0.5 * (E[curr_ind_1] + E[curr_ind_1+1] - 
                                              log(f_plus_0[curr_ind_1] + epsilon) - 
                                              log(f_minus_0[curr_ind_1] + epsilon))
        B[curr_ind_1+1, curr_ind_1] = B[curr_ind_1, curr_ind_1+1]
        
        F[curr_ind_1, curr_ind_1+1] = -(E[curr_ind_1] - E[curr_ind_1+1]) + 
                                       log(f_plus_0[curr_ind_1] + epsilon) - 
                                       log(f_minus_0[curr_ind_1] + epsilon)
        F[curr_ind_1+1, curr_ind_1] = -F[curr_ind_1, curr_ind_1+1]
    end
    
    # Compute E1 for s=1
    E1 = zeros(N+1)
    for u in 1:N-1
        f_p_curr = f_plus_1[u] == 0.0 ? epsilon : f_plus_1[u]
        f_m_curr = f_minus_1[u] == 0.0 ? epsilon : f_minus_1[u]
        
        if u != N && u != 1
            E1[u+1] = E1[u] - log(f_p_curr) + log(f_m_curr) 
        end
    end
    
    if abs(f_minus_1[1]) == 0.0
        E1[1] = E1[2] - log(f_minus_1[1] + epsilon) + log(epsilon) 
    else
        E1[1] = E1[2] + log(epsilon) - log(f_minus_1[1] + epsilon) 
    end
    
    if abs(f_plus_1[N]) == 0.0
        E1[N+1] = E1[N] - log(f_plus_1[N] + epsilon) + log(epsilon) 
    else
        E1[N+1] = E1[N] + log(epsilon) - log(f_plus_1[N]) 
    end
    
    # Compute barriers and forces for s=1
    B1 = zeros(N+1, N+1)
    F1 = zeros(N+1, N+1)
    for u in 0:N-1
        curr_ind_1 = u + 1
        B1[curr_ind_1, curr_ind_1+1] = 0.5 * (E1[curr_ind_1] + E1[curr_ind_1+1] - 
                                               log(f_plus_1[curr_ind_1] + epsilon) - 
                                               log(f_minus_1[curr_ind_1] + epsilon))
        B1[curr_ind_1+1, curr_ind_1] = B1[curr_ind_1, curr_ind_1+1]
        
        F1[curr_ind_1, curr_ind_1+1] = -(E1[curr_ind_1] - E1[curr_ind_1+1]) + 
                                        log(f_plus_1[curr_ind_1] + epsilon) - 
                                        log(f_minus_1[curr_ind_1] + epsilon)
        F1[curr_ind_1+1, curr_ind_1] = -F1[curr_ind_1, curr_ind_1+1]
    end
    
    return E, B, F, E1, B1, F1
end

epsilon = 1e-20
all_energies = Dict()
y_min_global, y_max_global = Inf, -Inf

println("Computing all energy landscapes...")
for h_error in h_error_list
    params = parameter_sets[h_error]
    local E, B, F, E1, B1, F1
    E, B, F, E1, B1, F1 = compute_energy_landscape(
        params[:f_plus_0], params[:f_minus_0],
        params[:f_plus_1], params[:f_minus_1],
        N, epsilon
    )
    
    all_energies[h_error] = (E=E, B=B, F=F, E1=E1, B1=B1, F1=F1)
    
    global y_min_global = min(y_min_global, minimum(E), minimum(E1), minimum(B), minimum(B1))
    global y_max_global = max(y_max_global, maximum(E), maximum(E1), maximum(B), maximum(B1))
    
    println("h_error = $h_error: E range [$(minimum(E)), $(maximum(E))], E1 range [$(minimum(E1)), $(maximum(E1))]")
end

y_padding_global = 0.1 * (y_max_global - y_min_global)
y_min_global -= y_padding_global
y_max_global += y_padding_global

println("Global y-axis range: [$y_min_global, $y_max_global]")

using Plots
default(linewidth=2, framestyle=:box)

println("\nCreating plots...")
for h_error in h_error_list
    println("Plotting h_error = $h_error")
    
    local E, B, F, E1, B1, F1
    E = all_energies[h_error].E
    B = all_energies[h_error].B
    F = all_energies[h_error].F
    E1 = all_energies[h_error].E1
    B1 = all_energies[h_error].B1
    F1 = all_energies[h_error].F1
    
    xs0 = 0:N
    xs1 = 0:N
    step_width = 0.9
    barrier_edge = (1 - step_width) / 2
    
    p1 = plot(; xlabel="Internal state", ylabel=L"E_i", title="s = 0", 
              legend=false, ylims=(y_min_global, y_max_global))
    scatter!(p1, xs0, E, marker=:circle, ms=5, color=:auto, lw=3)
    
    for i in 1:N
        j = i + 1
        x_start = i - 1
        x_left_edge = x_start + barrier_edge
        x_right_edge = x_start + 1 - barrier_edge
        x_end = j - 1
        barrier_height = B[i, j]
        plot!(p1, [x_start, x_left_edge, x_right_edge, x_end],
              [E[i], barrier_height, barrier_height, E[j]],
              color=:gray, lw=2, label=false)
    end
    
    p2 = plot(; xlabel="Internal state", ylabel=L"E_i", title="s = 1", 
              legend=false, ylims=(y_min_global, y_max_global))
    scatter!(p2, xs1, E1, marker=:circle, ms=5, color=:auto, lw=3)
    
    for i in 1:N
        j = i + 1
        x_start = i - 1
        x_left_edge = x_start + barrier_edge
        x_right_edge = x_start + 1 - barrier_edge
        x_end = j - 1
        barrier_height = B1[i, j]
        plot!(p2, [x_start, x_left_edge, x_right_edge, x_end],
              [E1[i], barrier_height, barrier_height, E1[j]],
              color=:gray, lw=2, label=false)
    end
    
    final_plot = plot(p1, p2, layout=(1, 2), size=(1100, 420), margin=5mm)
    error_folder = joinpath(folder_name_for_plots, "h_error_$(replace(string(h_error), "." => "_"))")
    mkpath(error_folder)
    savefig(final_plot, joinpath(error_folder, "energy_landscapes_s0_s1.png"))
    savefig(final_plot, joinpath(error_folder, "energy_landscapes_s0_s1.svg"))
end

println("Individual plots generated.")

println("Creating combined figures...")
all_plots_full = []
for h_error in h_error_list
    local E, B, F, E1, B1, F1
    E = all_energies[h_error].E
    B = all_energies[h_error].B
    F = all_energies[h_error].F
    E1 = all_energies[h_error].E1
    B1 = all_energies[h_error].B1
    F1 = all_energies[h_error].F1
    
    xs0 = 0:N
    xs1 = 0:N
    step_width = 0.9
    barrier_edge = (1 - step_width) / 2
    
    p1 = plot(; xlabel="Internal state", ylabel=L"E_i", 
              title="h_error = $h_error, s = 0", 
              legend=false, ylims=(y_min_global, y_max_global),
              titlefontsize=10)
    scatter!(p1, xs0, E, marker=:circle, ms=4, color=:auto, lw=2)
    
    for i in 1:N
        j = i + 1
        x_start = i - 1
        x_left_edge = x_start + barrier_edge
        x_right_edge = x_start + 1 - barrier_edge
        x_end = j - 1
        barrier_height = B[i, j]
        plot!(p1, [x_start, x_left_edge, x_right_edge, x_end],
              [E[i], barrier_height, barrier_height, E[j]],
              color=:gray, lw=1.5, label=false)
    end
    
    p2 = plot(; xlabel="Internal state", ylabel=L"E_i", 
              title="h_error = $h_error, s = 1", 
              legend=false, ylims=(y_min_global, y_max_global),
              titlefontsize=10)
    scatter!(p2, xs1, E1, marker=:circle, ms=4, color=:auto, lw=2)
    
    for i in 1:N
        j = i + 1
        x_start = i - 1
        x_left_edge = x_start + barrier_edge
        x_right_edge = x_start + 1 - barrier_edge
        x_end = j - 1
        barrier_height = B1[i, j]
        plot!(p2, [x_start, x_left_edge, x_right_edge, x_end],
              [E1[i], barrier_height, barrier_height, E1[j]],
              color=:gray, lw=1.5, label=false)
    end
    
    push!(all_plots_full, p1)
    push!(all_plots_full, p2)
end

combined_full = plot(all_plots_full..., 
                     layout=(length(h_error_list), 2), 
                     size=(1200, 600 * length(h_error_list)), 
                     margin=3mm)
savefig(combined_full, joinpath(folder_name_for_plots, "energy_landscapes_s0_s1_all_h_errors.png"))
savefig(combined_full, joinpath(folder_name_for_plots, "energy_landscapes_s0_s1_all_h_errors.svg"))

println("Combined figures created.")

println("Creating overlay plots...")
colors_palette = palette(:YlGnBu, 9)
color_map = Dict(
    0.02 => colors_palette[3],
    0.10 => colors_palette[4],
    0.20 => colors_palette[5],
    0.30 => colors_palette[6],
    0.40 => colors_palette[7],
    0.50 => colors_palette[8],
)
xs_states = 0:N
step_width = 0.9
barrier_edge = (1 - step_width) / 2

p_s0_overlay = plot(; xlabel="Internal state", ylabel=L"E_i", 
                    title="s = 0 (All h_error values)", 
                    legend=:outertopright, ylims=(y_min_global, y_max_global),
                    size=(600, 500), legendfontsize=8)

for h_error in h_error_list
    local E, B
    E = all_energies[h_error].E
    B = all_energies[h_error].B
    
    color = color_map[h_error]
    
    scatter!(p_s0_overlay, xs_states, E, marker=:circle, ms=5, 
             color=color, lw=2, label="h_error = $h_error")
    
    for i in 1:N
        j = i + 1
        x_start = i - 1
        x_left_edge = x_start + barrier_edge
        x_right_edge = x_start + 1 - barrier_edge
        x_end = j - 1
        barrier_height = B[i, j]
        plot!(p_s0_overlay, [x_start, x_left_edge, x_right_edge, x_end],
              [E[i], barrier_height, barrier_height, E[j]],
              color=color, lw=3, label=false)
    end
end

p_s1_overlay = plot(; xlabel="Internal state", ylabel=L"E_i", 
                    title="s = 1 (All h_error values)", 
                    legend=:outertopright, ylims=(y_min_global, y_max_global),
                    size=(600, 500), legendfontsize=8)

for h_error in h_error_list
    local E1, B1
    E1 = all_energies[h_error].E1
    B1 = all_energies[h_error].B1
    
    color = color_map[h_error]
    
    scatter!(p_s1_overlay, xs_states, E1, marker=:circle, ms=5, 
             color=color, lw=2, label="h_error = $h_error")
    
    for i in 1:N
        j = i + 1
        x_start = i - 1
        x_left_edge = x_start + barrier_edge
        x_right_edge = x_start + 1 - barrier_edge
        x_end = j - 1
        barrier_height = B1[i, j]
        plot!(p_s1_overlay, [x_start, x_left_edge, x_right_edge, x_end],
              [E1[i], barrier_height, barrier_height, E1[j]],
              color=color, lw=3, label=false)
    end
end

overlay_combined = plot(p_s0_overlay, p_s1_overlay, 
                        layout=(1, 2), 
                        size=(1400, 500), 
                        margin=5mm)
savefig(overlay_combined, joinpath(folder_name_for_plots, "energy_landscapes_overlay_all_h_errors.png"))
savefig(overlay_combined, joinpath(folder_name_for_plots, "energy_landscapes_overlay_all_h_errors.svg"))

println("All plots saved to: $folder_name_for_plots")

display(overlay_combined)