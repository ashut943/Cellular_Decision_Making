# assume we can load Q_opt_absorbing from somewhere

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
ua_avg_fail = zeros(length(t_plot))
ub_avg_fail = zeros(length(t_plot))
ua_avg_succ = zeros(length(t_plot))
ub_avg_succ = zeros(length(t_plot))

ua_std_fail = zeros(length(t_plot))
ub_std_fail = zeros(length(t_plot))
ua_std_succ = zeros(length(t_plot))
ub_std_succ = zeros(length(t_plot))

for j = 1:length(t_plot)
    ua_pts_fail = [f.u1[find_state_at_time(f,t_plot[j])]/length(failed_trajectories) for f in failed_trajectories]
    ub_pts_fail = [f.u2[find_state_at_time(f,t_plot[j])]/length(failed_trajectories) for f in failed_trajectories]
    ua_pts_succ = [f.u1[find_state_at_time(f,t_plot[j])]/length(success_trajectories) for f in success_trajectories]
    ub_pts_succ = [f.u2[find_state_at_time(f,t_plot[j])]/length(success_trajectories) for f in success_trajectories]
        
    ua_avg_fail[j] = mean(ua_pts_fail) 
    ub_avg_fail[j] = mean(ub_pts_fail)
    ua_avg_succ[j] = mean(ua_pts_succ)
    ub_avg_succ[j] = mean(ub_pts_succ)

    ua_std_fail[j] = std(ua_pts_fail)
    ub_std_fail[j] = std(ub_pts_fail)
    ua_std_succ[j] = std(ua_pts_succ)
    ub_std_succ[j] = std(ub_pts_succ)
end

using Plots
p1 = plot(t_plot,ua_avg_fail,ribbon = ua_std_fail,label="ua",xlabel="Time",ylabel="Probability",title="Unsuccesful trajectories")
plot!(p1,t_plot,ub_avg_fail,ribbon = ub_std_fail,label="ub")

p2 = plot(t_plot,ua_avg_succ,ribbon = ua_std_succ,label="ua",xlabel="Time",ylabel="Probability",title="Succesful trajectories")
plot!(p2, t_plot,ub_avg_succ,ribbon = ub_std_succ,label="ub")

plot(p1,p2,layout=(2,1))