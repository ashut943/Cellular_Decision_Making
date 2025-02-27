using Plots, Printf, FileIO, StatsBase
using Revise
using CellularDecisions

function average_trajectory(path_array, time_points)
    ua_avg = zeros(length(time_points))
    ub_avg = zeros(length(time_points))
    ua_std = zeros(length(time_points))
    ub_std = zeros(length(time_points))

    for (i,t) in enumerate(time_points)
        ua_pts = [path.u1[find_state_at_time(path,t)] for path in path_array]
        ub_pts = [path.u2[find_state_at_time(path,t)] for path in path_array]

        ua_avg[i] = mean(ua_pts)
        ub_avg[i] = mean(ub_pts)

        ua_std[i] = std(ua_pts)
        ub_std[i] = std(ub_pts)
    end
    return ua_avg, ub_avg, ua_std, ub_std
end


function collect_transitions(path_array)
    N = path_array[1].internal_states
    all_transitions = vcat([[(path.states[i],path.states[i+1]) for i in 1:length(path.states)-1] for path in path_array]...)

    statedict, _, _, _, _, _ = CellularDecisions.statematrices(N)
    ua_transitions = []
    ub_transitions = []
    for transition in all_transitions
        u_a_0 = statedict[transition[1] - 1][1][1]
        s_a_0 = statedict[transition[1] - 1][1][2]
        u_b_0 = statedict[transition[1] - 1][2][1]
        s_b_0 = statedict[transition[1] - 1][2][2]

        u_a_1 = statedict[transition[2] - 1][1][1]
        s_a_1 = statedict[transition[2] - 1][1][2]
        u_b_1 = statedict[transition[2] - 1][2][1]
        s_b_1 = statedict[transition[2] - 1][2][2]

        if (s_a_0 == s_a_1) && (u_a_0 == u_a_1)
            push!(ub_transitions, ([u_b_0,s_b_0], [u_b_1,s_b_1]))
        else
            push!(ua_transitions, ([u_a_0,s_a_0], [u_a_1,s_a_1]))
        end
    end

    transition_stats_a = countmap(ua_transitions)
    transition_stats_b = countmap(ub_transitions)
    return transition_stats_a, transition_stats_b, length(all_transitions)
end

# load the data
N = 3  # Number of states - 1
λ = 20.0  # Lambda parameter
initial_state = 1  # Initial state for simulations
T = 1000.0  # Time for simulations
num_simulations = 4000  # Number of simulations

# Get state matrices, sizes, and target states
statedict,statedictinv,terminal_states,TG,TB,Tc=CellularDecisions.statematrices(N);
ni,np,ns,nt=CellularDecisions.varioussizes(N)
targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states
targetstates=[targetstates_good;targetstates_bad]          # All target states
startstates=[start_state+1 for start_state ∈ Tc];         # Starting states
allstates=[startstates;targetstates_good; targetstates_bad]
all_targetstates = vcat(targetstates_good, targetstates_bad)

lambda_str = replace(string(λ), "." => "_")
base_folder = joinpath(dirname(@__DIR__), "experiments", "results", "Interior_point_method_results")
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_lambda_%s", N, lambda_str))
twocell_system_filename = generate_filename(folder_name,"twocell_system")
twocell_system = CellularDecisions.load(twocell_system_filename)

N=twocell_system.internal_states
Q = twocell_system.Q_matrix
P = twocell_system.parameters

Q_opt_absorbing=Q_absorbing_states_maker(Q, all_targetstates)

S_arr  = [CellularDecisions.simulate_ctmc(Q_opt_absorbing, initial_state, T,N) for i = 1:num_simulations]
#CellularDecisions.save(S_arr,"tmp.h5")
#S2_arr = CellularDecisions.load("tmp.h5")

targetstates_good_a = targetstates_good[[statedict[t-1][2][1] ==0 for t in targetstates_good]]
targetstates_good_b = setdiff(targetstates_good, targetstates_good_a)

terminal_classes = [CellularDecisions.terminal_class(path, [targetstates_good_a,targetstates_good_b], targetstates_bad) for path in S_arr]

failed_trajectories = unpack.(S_arr[terminal_classes .== -1])

success_trajectories = unpack.(S_arr[terminal_classes .== 1])

t_plot = 0:0.5:50

ua_avg_fail, ub_avg_fail, ua_std_fail, ub_std_fail = average_trajectory(failed_trajectories, t_plot)
ua_avg_succ, ub_avg_succ, ua_std_succ, ub_std_succ = average_trajectory(success_trajectories, t_plot)

# plot the average
p1 = plot(t_plot,ua_avg_fail,ribbon = ua_std_fail,label="ua",xlabel="Time",ylabel="State",title="Unsuccesful trajectories")
plot!(p1,t_plot,ub_avg_fail,ribbon = ub_std_fail,label="ub")


p2 = plot(t_plot,ua_avg_succ,ribbon = ua_std_succ,label="ua",xlabel="Time",ylabel="State",title="Succesful trajectories")
plot!(p2, t_plot,ub_avg_succ,ribbon = ub_std_succ,label="ub")

plot(p1,p2,layout=(2,1))
savefig("plots/Average_traj_q_absorb.pdf")

# look at some individual trajectories
n_plots = min(length(failed_trajectories), 50) # number of trajectroeies to plot, need this as there might not be enough failed trajectories
p3 = plot()
for i = 1:n_plots
    plot_ctmc!(p3,failed_trajectories[i].times, failed_trajectories[i].u1, failed_trajectories[i].final_time,c=:blue,linewidth=0.2)
    plot_ctmc!(p3,failed_trajectories[i].times, failed_trajectories[i].u2, failed_trajectories[i].final_time,c=:red,linewidth=0.2)
end

p4 = plot()
for i = 1:n_plots
    plot_ctmc!(p4,success_trajectories[i].times, success_trajectories[i].u1, success_trajectories[i].final_time,c=:blue,linewidth=0.2)
    plot_ctmc!(p4,success_trajectories[i].times, success_trajectories[i].u2, success_trajectories[i].final_time,c=:red,linewidth=0.2)
end

plot(p3,p4,layout=(2,1))



success_trajectories_unpacked = S_arr[terminal_classes .== 1]
transitions_a,transitions_b,num_transitions = collect_transitions(success_trajectories_unpacked)

p = plot()

Δ = 0.02 # for offset
scale = 80 # for line thickness
for (i,tstats) in enumerate([transitions_a, transitions_b])
    Δx = 2*i-1
    for k1 in collect(keys(tstats))
        weight = scale*tstats[k1]/num_transitions
        
        u0,s0 =k1[1]
        u1,s1 =k1[2]
        
        color = (u0 <= u1)*(s0 <= s1) ? :red : :blue
        
        xoffset = (u0 != u1) ? ((u0 < u1) ? Δ*(1+0.1*weight) : -Δ*(1+0.1*weight)) : 0
        yoffset = (s0 != s1) ? ((s0 < s1) ? Δ*(1+0.1*weight) : -Δ*(1+0.1*weight)) : 0
        plot!(p,[Δx + s0+xoffset,Δx + s1+xoffset],[u0+yoffset,u1+yoffset],c=color,lw = weight,label=false)

    end
end

xticks!(p, [1.5, 3.5], ["cell 1", "cell 2"])
plot!(p,size=(300,400),grid=false,ylabel="Internal state")
savefig(p,"plots/successful_transitions.pdf")