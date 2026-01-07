using Plots, Printf, FileIO

function plot_ctmc(S::StochasticPath)
    p  = plot(
        xlabel="Time (t)", 
        ylabel="State (s)", 
        title="CTMC Simulation",
        yticks=collect(1:maximum(S.states)),
        xlim=(0, S.final_time), 
        ylim=(minimum(S.states)-0.5, maximum(S.states)+0.5),
        grid=:both,
        size=(1500, 1000),left_margin=10Plots.mm, right_margin=10Plots.mm,bottom_margin=10Plots.mm)
    plot_ctmc!(p,S.times,S.states,S.final_time)
    
    return p
end

function plot_ctmc!(p,times, states, final_time;c=:gray,linewidth=2)
    # Core plotting functionality for a set of discrete jumps between "states" and "times"
    for i in 1:(length(times)-1)
        plot!(p,[times[i], times[i+1]], [states[i], states[i]], linewidth=linewidth, color=c, label=false)
        plot!([times[i+1], times[i+1]], [states[i], states[i+1]], linewidth=linewidth, color=c, label=false)
    end
    plot!(p,[times[end], final_time], [states[end], states[end]], linewidth=linewidth, color=c, label=false)
end

function plot_ctmc_multi_traj_heatmap(N::Int, trajectories_array::Vector{StochasticPath}, filename::String; save_plots::Bool=true)
    visit_counts = zeros(Int, N+1, N+1)

    for i in 1:length(trajectories_array)
        curr_path = unpack(trajectories_array[i])
        # Increment counts for each corresponding pair of states simultaneously
        for (state_a, state_b) in zip(curr_path.u1, curr_path.u2)
            visit_counts[state_a + 1, state_b + 1] += 1
        end
    end    
    # Create tick labels as fractions
    tick_vals = 0:1:N
    tick_labels = ["$i/$N" for i in tick_vals]
    
    p = heatmap(
        0:N, 0:N, log.(visit_counts .+ 1)', 
        xlabel="Cell a", ylabel="Cell b", 
        color=:matter,
        size=(1500, 1000),
        xticks=(tick_vals, tick_labels),  # Use fractional labels for x-axis
        yticks=(tick_vals, tick_labels),  # Use fractional labels for y-axis
        left_margin=10Plots.mm, 
        right_margin=10Plots.mm, 
        bottom_margin=10Plots.mm
    )
    
    display(p)
    if save_plots
        savefig(filename * ".png")
        savefig(filename * ".svg")
    end
end

function plot_ctmc_invar_distn_heatmap(Q::Array{Float64,2}, N::Int, filename::String, λ::Float64; save_plots::Bool=true)
    ni=size(Q,1)
    statedict, _, _, _, _, _ = CellularDecisions.statematrices(N)
    tolerance = 1e-8

    eig = eigen(Q')
    eigenvalues = eig.values
    eigenvectors = eig.vectors
    
    zero_indices = findall(x -> isapprox(x, 0.0, atol=tolerance), eigenvalues)
    pi_vector = real(eigenvectors[:, zero_indices])
    threshold = 1e-8
    pi_vector = abs.(map(x -> abs(x) < threshold ? 0.0 : x, pi_vector))
    pi_vector /= sum(pi_vector)
    
    pi_matrix = zeros(Float64, N+1, N+1)
    for state_index in 1:ni
        curr_state = statedict[state_index - 1] 
        ua, ub = curr_state[1][1], curr_state[2][1]
        pi_matrix[ua + 1, ub + 1] += pi_vector[state_index]
    end

    # Create tick labels as fractions
    tick_vals = 0:1:N
    tick_labels = ["$i/$N" for i in tick_vals]

    p = heatmap(
        0:N, 0:N, pi_matrix', 
        xlabel="Cell a", ylabel="Cell b", 
        color=:matter,
        size=(1500, 1000),
        xticks=(tick_vals, tick_labels),  # Use fractional labels for x-axis
        yticks=(tick_vals, tick_labels),  # Use fractional labels for y-axis
        left_margin=10Plots.mm, 
        right_margin=10Plots.mm, 
        bottom_margin=10Plots.mm
    )

    display(p)
    if save_plots
        savefig(filename * ".png")
        savefig(filename * ".svg")
    end
end

function plot_Q_with_colored_yticks(Q::Matrix, N::Int, special_ticks::Vector{Int}, filename, λ::Float64=missing; save_plots::Bool=true)
    title_text = "Transition Rate Matrix Heatmap"
    title_text = @sprintf("Transition Rate Matrix Heatmap, N=%d", N)
    if λ !== missing
        title_text = @sprintf("Transition Rate Matrix Heatmap, N=%d, λ=%s", N, string(λ))
    end
    p = heatmap(
        Q,
        c = :matter,
        title = title_text,
        xlabel = "States",
        ylabel = "States",
        size = (1600, 800),
        left_margin = 10Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 10Plots.mm,
        yflip = true
    )
    
    y_ticks = collect(1:size(Q, 1))
    yticks!(p, (y_ticks, string.(y_ticks)))
    
    for i in special_ticks
        annotate!(p, 0.5, i, text("T", :red, 12, :right))
    end
    display(p)
    if save_plots
        savefig(p, filename * ".png")
        savefig(p, filename * ".svg")
    end
    
end