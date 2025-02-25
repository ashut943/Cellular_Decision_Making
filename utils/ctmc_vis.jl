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
    plot_ctmc!(p,S)
    
    return p
end

function plot_ctmc!(p,S::StochasticPath;c=:gray)
    times = S.times
    states = S.states
    final_time = S.final_time
    for i in 1:(length(times)-1)
        plot!(p,[times[i], times[i+1]], [states[i], states[i]], linewidth=2, color=c, label=false)
        plot!([times[i+1], times[i+1]], [states[i], states[i+1]], linestyle=:dash, color=c, label=false)
    end
    plot!(p,[times[end], final_time], [states[end], states[end]], linewidth=2, color=c, label=false)
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