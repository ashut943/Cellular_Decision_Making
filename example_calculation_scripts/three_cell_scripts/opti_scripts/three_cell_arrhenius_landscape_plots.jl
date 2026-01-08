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

include("../../../utils/utils.jl")
include("../../../mult_cell/mult_cell.jl")

M = 3
N = 6

#--------------------------------
# Output directories
#--------------------------------
folder_name_for_plots = joinpath(dirname(dirname(dirname(@__DIR__))), "plots", "three_cell_results", "arrhenius_landscape")
mkpath(folder_name_for_plots)

#--------------------------------
# Parameter sets for different h_error values
#--------------------------------
parameter_sets = Dict(
    0.02 => Dict(
        :k_off => [0.3423641318866556],
        :f_plus_0 => [0.0, 0.17629326970507203, 0.999999989026727, 0.4769223612336185, 0.6446712533120912, 0.8569258893993309],
        :f_plus_1 => [0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
        :f_minus_0 => [0.0, 1.0, -0.0, -0.0, -0.0, 0.0],
        :f_minus_1 => [1.0, 0.4563432292332899, 1.0, 1.0, 0.999999998920317, 0.0],
        :g => [-0.0, -0.0, -0.0, 1.0, 1.0, 1.0, 1.0]
    ),
    0.05 => Dict(
        :k_off => [0.2453080676045083],
        :f_plus_0 => [-0.0, 0.28856290353564384, 0.9999995766683331, 0.9267752006614016, 0.9999995875721831, 0.9999997238089335],
        :f_plus_1 => [-0.0, 0.0, 0.0, 1.9699021388923456e-7, 3.875408668775059e-7, 2.2503734197361696e-7],
        :f_minus_0 => [0.010556139708503472, 0.9999994492041374, 0.0, 2.5384881340164237e-7, 1.887203248558727e-7, 0.0],
        :f_minus_1 => [0.9999999259790472, 0.9999995500079888, 0.9999998742695807, 0.9999996126755645, 0.9999990911013611, 0.0],
        :g => [0.0, -0.0, 0.0, 0.9999999046655611, 0.9999999080679266, 0.9999998833337426, 0.9999999689231271]
    ),
    0.10 => Dict(
        :k_off => [0.14827235396393476],
        :f_plus_0 => [-0.0, 0.5971138256117456, 0.9999999976622214, 1.0, 1.0, 1.0],
        :f_plus_1 => [0.0, -0.0, -0.0, -0.0, 6.915170148050551e-6, 0.9999998570103659],
        :f_minus_0 => [0.0875015639039488, 0.9999999852899016, -0.0, -0.0, -0.0, -0.0],
        :f_minus_1 => [1.0, 1.0, 1.0, 0.9999999962256939, 0.9999997849567083, -0.0],
        :g => [-0.0, -0.0, -0.0, 1.0, 1.0, 1.0, 1.0]
    ),
    0.15 => Dict(
        :k_off => [2.3826835420864667e-6],
        :f_plus_0 => [0.0, 0.9999999968996617, 0.5101661558247754, 1.0, 1.0, 1.0],
        :f_plus_1 => [0.0, -0.0, -0.0, 0.0, 0.8291360232358246, 0.9999999709612025],
        :f_minus_0 => [0.5350435716244849, 0.0, -0.0, -0.0, -0.0, -0.0],
        :f_minus_1 => [1.0, 1.0, 1.0, 0.9999999751848484, 0.00022269713040848335, -0.0],
        :g => [-0.0, -0.0, -0.0, 1.0, 1.0, 1.0, 1.0]
    ),
    0.20 => Dict(
        :k_off => [4.959040223543948e-6],
        :f_plus_0 => [-0.0, 0.9999999929159187, 0.9748969400611982, 1.0, 1.0, 1.0],
        :f_plus_1 => [-0.0, -0.0, -0.0, 0.0, 0.9999998477688145, 0.9999999763625324],
        :f_minus_0 => [0.757811174197377, 1.787666703327605e-7, 0.0, -0.0, -0.0, 0.0],
        :f_minus_1 => [1.0, 1.0, 1.0, 0.9999999571306338, 2.8667654669813788e-5, 0.0],
        :g => [-0.0, -0.0, -0.0, 1.0, 1.0, 1.0, 0.9999999992805424]
    ),
    0.25=> Dict(
        :k_off => [1.0439918645631172e-6],
        :f_plus_0 => [0.0, 0.999999954087904, 1.0, 1.0, 1.0, 1.0],
        :f_plus_1 => [0.0, -0.0, -0.0, 0.03894950978064692, 0.999999934589256, 0.9999999658872942],
        :f_minus_0 => [1.0, 0.20955553190898535, -0.0, -0.0, -0.0, 0.0],
        :f_minus_1 => [1.0, 1.0, 0.999999999173893, 0.9999996030046693, 1.3319562359618286e-5, -0.0],
        :g => [0.0, -0.0, 0.0, 0.9999999936086836, 0.9999999939001695, 0.9999999837652187, 0.9999999733343988]
    ),
    0.30 => Dict(
        :k_off => [1.3579124167291824e-6],
        :f_plus_0 => [0.0, 0.7169568918537759, 1.0, 1.0, 1.0, 1.0],
        :f_plus_1 => [-0.0, -0.0, 1.3561414976712252e-6, 0.9999998302887482, 0.999999978388217, 0.999999983336807],
        :f_minus_0 => [1.0, 1.3500611380118173e-7, -0.0, -0.0, -0.0, -0.0],
        :f_minus_1 => [1.0, 0.9999999984565304, 0.9999999168726775, 5.425171137603429e-5, 0.0, 0.0],
        :g => [0.0, -0.0, 0.9999999647182821, 0.9999999872855637, 0.9999999765296815, 0.9999999314913955, 0.9999998884928465]
    ),
    0.35 => Dict(
        :k_off => [5.7975112684084475e-5],
        :f_plus_0 => [-0.0, 0.5297594946314349, 0.9999998478649836, 0.9999998575864298, 0.9999998569574671, 0.9999998542191492],
        :f_plus_1 => [0.0, 2.2248106586040928e-7, 8.893845047790688e-5, 0.9999909139319015, 0.9999983602227306, 0.999998622161244],
        :f_minus_0 => [0.9999998727884507, 1.6046348350809453e-5, 1.4892172343451182e-7, 1.4328874205554336e-7, 1.4599977322606752e-7, -0.0],
        :f_minus_1 => [0.9999995452328263, 0.9999993506647751, 0.9999948682363826, 0.004514983534970316, 0.00035793114471197485, 0.0],
        :g => [7.002686203479624e-7, 1.649936401045244e-7, 0.9999980531519567, 0.9999987017020125, 0.9999980463206369, 0.9999955118016313, 0.999993186402054]
    ),
    0.40 => Dict(
        :k_off => [7.226492551865321e-5],
        :f_plus_0 => [0.0, 0.4215444371296066, 0.9999998392081535, 0.9999998469727222, 0.9999998462648039, 0.9999998439295039],
        :f_plus_1 => [0.0, 2.7266228834386636e-7, 0.00013091777339666375, 0.9999876970256542, 0.9999977740282553, 0.999998132483708],
        :f_minus_0 => [0.9999998967870243, 4.680939571062543e-5, 1.5774530391248178e-7, 1.532945303157162e-7, 1.5567625066505314e-7, -0.0],
        :f_minus_1 => [0.9999994389518099, 0.9999990822205255, 0.9999929176066503, 0.006117997470754325, 0.0005083212478482855, 0.0],
        :g => [8.937895169822984e-7, 2.0778287592682244e-7, 0.9999975425703143, 0.9999981884251204, 0.9999972576153208, 0.9999937887994805, 0.9999906973327118]
    ),
    0.50 => Dict(
        :k_off => [0.00011065948172912752],
        :f_plus_0 => [-0.0, 0.3398644943655177, 0.9999998096133109, 0.9999998160199115, 0.9999998150664, 0.9999998128634492],
        :f_plus_1 => [0.0, 3.3832721221309055e-7, 0.00023926390522611336, 0.9999831700677204, 0.9999966329920473, 0.9999971764332599],
        :f_minus_0 => [0.999999918552377, 0.15118967143253179, 2.4200234443207273e-7, 1.8362799839118975e-7, 1.859157305709691e-7, 0.0],
        :f_minus_1 => [0.9999993215853402, 0.9999985635284062, 0.9999884847839862, 0.008742403041800009, 0.0008247137174167732, 0.0],
        :g => [1.297631909643007e-6, 2.6259256279010937e-7, 0.9999954850231616, 0.9999971301982857, 0.9999955534158879, 0.9999899193320442, 0.9999848064253503]
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