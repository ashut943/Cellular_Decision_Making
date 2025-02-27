# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using CellularDecisions

# Include helper files
include("../twocells/twocells_vis.jl")
include("../twocells/twocells_setup.jl")
include("../twocells/twocells_nonlinearsolver.jl")
include("../twocells/twocells_hittingtime.jl")
include("../utils/ctmc_vis.jl")
include("../utils/ctmc_core.jl")
include("../utils/file_utils.jl")
include("../utils/video_utils.jl")

N = 3  # Number of states - 1
λ = 20.0  # Lambda parameter
initial_tau_val=5.0  # Initial tau value for optimization
initial_P_val=1.0  # Initial P value for optimization
initial_state = 1  # Initial state for simulations

# Create output directory
lambda_str = replace(string(λ), "." => "_")
base_folder = joinpath(dirname(@__DIR__), "experiments", "results", "Interior_point_method_results")
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_lambda_%s", N, lambda_str))
mkpath(folder_name)

# Get state matrices, sizes, and target states
statedict,statedictinv,terminal_states,TG,TB,Tc=CellularDecisions.statematrices(N);
ni,np,ns,nt=CellularDecisions.varioussizes(N)
targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states
targetstates=[targetstates_good;targetstates_bad]          # All target states
startstates=[start_state+1 for start_state ∈ Tc];         # Starting states
allstates=[startstates;targetstates_good; targetstates_bad]
all_targetstates = vcat(targetstates_good, targetstates_bad)

# Run nonlinear solver to get optimal solution
upper_bound_tau_0,upper_bound,tau_opt,P_opt,terminationstatus=run_nonlinear_solver(N, λ, initial_P_val, initial_tau_val,false,false)

# Extract results
tau_opt_tilde = tau_opt[startstates]
lower_bound=minimum(tau_opt_tilde)

# Clean up P_opt values
P_opt_ = P_opt .* (abs.(P_opt) .>= 1e-8)  # Zero out small values
P_opt_ .= min.(P_opt_, 1.0)                # Cap at 1.0
Q_opt = Q_maker_using_M(P_opt_, N, λ, statedict, statedictinv)  # Generate Q matrix

#save the twocell_system 
parameters_opt = CellularDecisions.parameter_vector_to_parameters(P_opt_, N)

twocell_system = CellularDecisions.build_two_cell_system(N, Q_opt, parameters_opt)
twocell_system_filename = generate_filename(folder_name,"twocell_system")
CellularDecisions.save(twocell_system,twocell_system_filename)

# Print to console
println("================================================")
println("Optimal solution")
println("P_opt: ", P_opt_)
println("tau_opt_0: ", tau_opt[1])
println("------------------------------------------------")
println("F^+: ", parameters_opt.fp)
println("F^-: ", parameters_opt.fn)
println("G: ", parameters_opt.gp)
println("k_off: ", parameters_opt.koff)

# Save to txt file
optimal_values_filename = generate_filename(folder_name, "optimal_values_w_o_fixed_values.txt")
open(optimal_values_filename, "w") do io
    println(io, "tau_opt_0 = ", tau_opt[1])
    println(io, "k_off = ", P_opt_[end])
    println(io, "F+(.,0) = ", P_opt_[1:N])
    println(io, "F+(.,1) = ", P_opt_[N+1:2*N])
    println(io, "F-(.,0) = ", P_opt_[2*N+1:3*N])
    println(io, "F-(.,1) = ", P_opt_[3*N+1:4*N])
    println(io, "G = ", P_opt_[4*N+1:end-1])
end