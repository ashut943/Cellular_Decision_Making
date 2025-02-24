using Plots, Printf, FileIO

function plot_ctmc(times::Vector{Float64}, states::Vector{Int}, T::Float64)
    plot(
        xlabel="Time (t)", 
        ylabel="State (s)", 
        title="CTMC Simulation",
        yticks=collect(1:maximum(states)),
        xlim=(0, T), 
        ylim=(minimum(states)-0.5, maximum(states)+0.5),
        grid=:both,
        size=(1500, 1000),left_margin=10Plots.mm, right_margin=10Plots.mm,bottom_margin=10Plots.mm)
    
    for i in 1:(length(times)-1)
        plot!([times[i], times[i+1]], [states[i], states[i]], linewidth=2, label=false)
        if i < length(times)-1
            plot!([times[i+1], times[i+1]], [states[i], states[i+1]], linestyle=:dash, color=:gray, label=false)
        end
    end
    display(current())
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
    
    if save_plots
        savefig(p, filename * ".png")
        savefig(p, filename * ".svg")
    end
end