using Plots, Printf, LinearAlgebra

function plot_ctmc_with_index_number(times::Vector{Float64}, states::Vector{Int}, T::Float64, N::Int, filename; save_plots::Bool=true)
    ni,np,ns,nt=varioussizes(N)

    p = plot(
        xlabel="Time (t)", 
        ylabel="State (s)", 
        title="CTMC Simulation",
        yticks=collect(1:ni),
        xlim=(0, T), 
        ylim=(0.5, ni+0.5),
        grid=:both,
        size=(1500, 1000),
        left_margin=10Plots.mm, 
        right_margin=10Plots.mm,
        bottom_margin=10Plots.mm
    )
    
    for i in 1:(length(times)-1)
        plot!(p, [times[i], times[i+1]], [states[i], states[i]], linewidth=2, label=false)
        if i < length(times)-1
            plot!(p, [times[i+1], times[i+1]], [states[i], states[i+1]], linestyle=:dash, color=:gray, label=false)
        end
    end
    
    display(p)
    if save_plots
        savefig(filename * ".png")
        savefig(filename * ".svg")
    end
end

function plot_ctmc_our_problem(times::Vector{Float64}, states::Vector{Int}, T::Float64, N::Int, filename::String, λ::Float64; save_plots::Bool=true)
    ni, np, ns, nt = varioussizes(N)
    S, Skeyer, T_, TG, TB, Tc = statematrices(N)

    u_a_vals = [S[states[i] - 1][1][1] for i in 1:length(states)]
    u_b_vals = [S[states[i] - 1][2][1] for i in 1:length(states)]
    
    all_vals = vcat(u_a_vals, u_b_vals)
    y_min = minimum(all_vals)
    y_max = maximum(all_vals)

    # Create tick labels as fractions
    tick_vals = 0:1:N
    tick_labels = ["$i/$N" for i in tick_vals]

    p = plot(
        xlabel="Time (t)", 
        ylabel="State Values", 
        title=@sprintf("CTMC Simulation - Trajectories of Cells a and b; N=%d, λ=%s",N,string(λ)),
        xlim=(0, T),
        ylim=(y_min - 0.1, y_max + 0.1),
        yticks=(tick_vals, tick_labels),  # Use fractional labels
        grid=:both,
        size=(1500, 1000),
        left_margin=10Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=10Plots.mm,
        legend=:topright
    )
    
    for i in 1:(length(times) - 1)
        plot!(p, [times[i], times[i + 1]], [u_a_vals[i], u_a_vals[i]], linewidth=2, color=:blue, label=(i == 1 ? "Cell a" : ""))
        if i < length(times) - 1
            plot!(p, [times[i + 1], times[i + 1]], [u_a_vals[i], u_a_vals[i + 1]], linestyle=:dash, color=:blue, label=false)
        end
    end

    for i in 1:(length(times) - 1)
        plot!(p, [times[i], times[i + 1]], [u_b_vals[i], u_b_vals[i]], linewidth=2, color=:red, label=(i == 1 ? "Cell b" : ""))
        if i < length(times) - 1
            plot!(p, [times[i + 1], times[i + 1]], [u_b_vals[i], u_b_vals[i + 1]], linestyle=:dash, color=:red, label=false)
        end
    end

    display(p)
    if save_plots
        savefig(filename * ".png")
        savefig(filename * ".svg")
    end
end

function plot_ctmc_heatmap_mod(time_points::AbstractVector{Float64}, state_probs::Array{Float64,2}, N::Int, filename::String, λ::Float64; save_plots::Bool=true)
    ni, np, ns, nt = varioussizes(N)
    S, Skeyer, T_, TG, TB, Tc = statematrices(N)
    
    num_states = size(state_probs, 1)
    println(num_states)

    yticks_vals = collect(1:ni)
    # Update labels to show fractions
    yticks_labels = [@sprintf("(%s/%d, %s/%d)", S[i-1][1], N, S[i-1][2], N) for i in 1:ni]

    p = heatmap(
        time_points,
        yticks_vals,
        state_probs,
        xlabel = "Time",
        ylabel = "(uₐ, u_b)",
        title = @sprintf("CTMC Heatmap Trajectories (Empirical), N=%d, λ=%s",N,string(λ)),
        aspect_ratio = :auto,
        xlims = (time_points[1], time_points[end]),
        ylims = (0.5, num_states + 0.5),
        yticks = (yticks_vals, yticks_labels),
        colorbar = true,
        c = :viridis,
        clim = (0, 1),
        legend = false,
        size = (1500, 1000),
        left_margin = 10Plots.mm, 
        right_margin = 10Plots.mm, 
        bottom_margin = 10Plots.mm
    )
    
    display(p)
    if save_plots
        savefig(filename * ".png")
        savefig(filename * ".svg")
    end
end

function plot_ctmc_our_problem_multi(Q::Array{Float64,2}, initial_state::Int, T::Float64, N::Int, num_simulations::Int, filename::String, λ::Float64; save_plots::Bool=true)
    ni, np, ns, nt = varioussizes(N)
    S, Skeyer, T_, TG, TB, Tc = statematrices(N)

    visit_counts = zeros(Int, N+1, N+1)

    for _ in 1:num_simulations
        times, states = simulate_ctmc(Q, initial_state, T)
        for state_index in states
            curr_state = S[state_index - 1]
            ua, ub = curr_state[1][1], curr_state[2][1]
            visit_counts[ua + 1, ub + 1] += 1
        end
    end
    
    # Create tick labels as fractions
    tick_vals = 0:1:N
    tick_labels = ["$i/$N" for i in tick_vals]
    
    p = heatmap(
        0:N, 0:N, log.(visit_counts .+ 1)', 
        xlabel="Cell a", ylabel="Cell b", 
        title=@sprintf("Aggregated Frequency of Visits to States across Multiple Simulations N=%d, λ=%s",N,string(λ)),
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

function plot_ctmc_invar_distn_our_problem(Q::Array{Float64,2}, N::Int, filename::String, λ::Float64; save_plots::Bool=true)
    ni, np, ns, nt = varioussizes(N)
    S, Skeyer, T_, TG, TB, Tc = statematrices(N)
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
        curr_state = S[state_index - 1] 
        ua, ub = curr_state[1][1], curr_state[2][1]
        pi_matrix[ua + 1, ub + 1] += pi_vector[state_index]
    end

    # Create tick labels as fractions
    tick_vals = 0:1:N
    tick_labels = ["$i/$N" for i in tick_vals]

    p = heatmap(
        0:N, 0:N, pi_matrix', 
        xlabel="Cell a", ylabel="Cell b", 
        title=@sprintf("Invariant Distribution N=%d, λ=%s",N,string(λ)),
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