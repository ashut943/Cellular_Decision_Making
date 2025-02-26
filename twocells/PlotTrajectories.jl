using Plots
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



# This script is half complete, I'm assuming we can load Q_opt_absorbing from somewhere
S_arr  = [CellularDecisions.simulate_ctmc(Q_opt_absorbing, initial_state, T,N) for i = 1:4000]
CellularDecisions.save(S_arr,"tmp.h5")
S2_arr = CellularDecisions.load("tmp.h5")

targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states

targetstates_good_a = targetstates_good[[S[t-1][2][1] ==0 for t in targetstates_good]]
targetstates_good_b = setdiff(targetstates_good, targetstates_good_a)

terminal_classes = [CellularDecisions.terminal_class(path, [targetstates_good_a,targetstates_good_b], targetstates_bad) for path in S_arr]

failed_trajectories = unpack.(S_arr[terminal_classes .== -1])

success_trajectories = unpack.(S_arr[terminal_classes .== 1])

t_plot = 0:0.5:50

ua_avg_fail, ub_avg_fail, ua_std_fail, ub_std_fail = average_trajectory(failed_trajectories, t_plot)
ua_avg_succ, ub_avg_succ, ua_std_succ, ub_std_succ = average_trajectory(success_trajectories, t_plot)

# plot the average
p1 = plot(t_plot,ua_avg_fail,ribbon = ua_std_fail,label="ua",xlabel="Time",ylabel="Probability",title="Unsuccesful trajectories")
plot!(p1,t_plot,ub_avg_fail,ribbon = ub_std_fail,label="ub")


p2 = plot(t_plot,ua_avg_succ,ribbon = ua_std_succ,label="ua",xlabel="Time",ylabel="Probability",title="Succesful trajectories")
plot!(p2, t_plot,ub_avg_succ,ribbon = ub_std_succ,label="ub")

plot(p1,p2,layout=(2,1))
savefig("plots/Average_traj_q_absorb.pdf")

# look at some individual trajectories
p3 = plot()
for i = 1:50
    plot_ctmc!(p3,failed_trajectories[i].times, failed_trajectories[i].u1, failed_trajectories[i].final_time,c=:blue,linewidth=0.2)
    plot_ctmc!(p3,failed_trajectories[i].times, failed_trajectories[i].u2, failed_trajectories[i].final_time,c=:red,linewidth=0.2)
end

p4 = plot()
for i = 1:50
    plot_ctmc!(p4,success_trajectories[i].times, success_trajectories[i].u1, success_trajectories[i].final_time,c=:blue,linewidth=0.2)
    plot_ctmc!(p4,success_trajectories[i].times, success_trajectories[i].u2, success_trajectories[i].final_time,c=:red,linewidth=0.2)
end

plot(p3,p4,layout=(2,1))