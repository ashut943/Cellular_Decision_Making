using JuMP, Ipopt, Gurobi,  Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, DataStructures, MathOptInterface, MosekTools, SparseArrays
using Revise
using HSL_jll
import CellularDecisions
using AppleAccelerate

function run_nonlinear_solver(N::Int, M::Int, K::Int, rho::Vector{Int}, h_error::Float64, initial_state::Int, initial_P_values_dict::Dict{Int, Vector{Float64}}, initial_tauval_array::Vector{Float64}, fix_P_dict::Dict{Int, Bool}, boundary_type::String)
    #Function to run the non-linear solver using interior point method
    #N: The number of internal states, integer (or more accurately, the number of internal states is 0,...,N so N+1 internal states)
    #M: The number of cells, integer
    #K: The number of independent strategies, integer, K<=N
    #rho: The vector of length N, telling which strategy each cell uses. For e.g, rho=[1,1,2] implies that cells 1 and 2 use the same strategy, strategy 1, but cell 3 follows strategy 2 which is independent of strategy 1
    #h_error: epsilon_tol, the error tolerance for patterning, float, h_error<=1.0
    #initial state: the initial state of the system, is a tuple that contains pairs to specify the starting internal and receiving state for each cell. For e.g for three cells, could use ((1,0), (1,0), (1,0)) to specify that all cells start in internal state 1, and signal receiving state 0 (i.e. 'off')
    #initial_P_value: a dictionary containing the K initial strategy vectors p
    #initial_tauval_array: 

    # Create model
    #using ipopt, but can use Gurobi, Mosek, and other compatible solvers
    model = Model(Ipopt.Optimizer)

    #choosing optimier attributes/properties
    set_optimizer_attribute(model, "max_iter", 10000)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_attribute(model, "linear_solver", "ma97")

    # Get state matrices and sizes
    state_dict, state_dict_inv, _, TG, TB, Tc = CellularDecisions.statematrices(N, M, boundary_type)
    ni, np = CellularDecisions.varioussizes(N, M)

    # Set up target and starting states
    targetstates_good = [target_state + 1 for target_state in TG]
    targetstates_bad = [target_state + 1 for target_state in TB]
    startstates = [start_state + 1 for start_state in Tc]
    allstates = [startstates; targetstates_good; targetstates_bad]

    # Create variables for K different P vectors
    @variable(model, 0 <= P_[1:K, 1:np] <= 1)
    P_vars = Dict{Int, Vector{VariableRef}}(j => [P_[j, i] for i in 1:np] for j in 1:K)

    # Create variables for hitting times and probabilities
    @variable(model, τ[1:ni])
    @variable(model, 0.0 <= h[1:ni] <= 1.0)

    # Set objective
    @objective(model, Min, τ[initial_state])

    # Create the rate matrix Q
    # Map the K P vectors to M cells using rho
    P_vars_mapped = Dict{Int, Vector{VariableRef}}(m => P_vars[rho[m]] for m in 1:M)
    Q = Q_maker_original_mod(P_vars_mapped, N, M, model, state_dict, state_dict_inv)

    # Set up hitting time constraints
    @expression(model, A, hitting_time_mod_give_A(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @expression(model, b, hitting_time_mod_give_b(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @constraint(model, A * τ == b)

    # Set up hitting probability constraints
    @expression(model, Ah, hitting_prob_mod_give_A(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @expression(model, bh, hitting_prob_mod_give_b(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @constraint(model, Ah * h == bh)
    @constraint(model, h[initial_state] <= h_error)

    # Set initial values
    for k in 1:K
        for i in 1:np
            set_start_value(P_vars[k][i], initial_P_values_dict[1][i])
        end
    end
    
    for i in 1:ni
        set_start_value(τ[i], initial_tauval_array[i])
    end

    # Fix P values if specified
    for m in 1:M
        if fix_P_dict[m]
            k = rho[m]
            for i in 1:np
                @constraint(model, P_vars[k][i] == initial_P_values_dict[m][i])
            end
        end
    end

    # Set boundary conditions
    if boundary_type == "boundary_2"
        for k in 1:K
            @constraint(model, P_vars[k][1] == 0.0)
            @constraint(model, P_vars[k][N+1] == 0.0)
            @constraint(model, P_vars[k][2*N+N] == 0.0)
            @constraint(model, P_vars[k][3*N+N] == 0.0)
        end
    end

    # Optimize
    JuMP.optimize!(model)

    # Extract results
    tau_opt = value.(τ)
    P_opt_ = value.(P_)
    
    # Create P_opt_dict mapping each cell to its assigned P vector
    P_opt_dict = Dict{Int, Vector{Float64}}(m => P_opt_[rho[m], :] for m in 1:M)

    tau_opt_tilde = tau_opt[startstates]
    upper_bound = maximum(tau_opt_tilde)
    upper_bound_tau_0 = tau_opt[1]

    return upper_bound_tau_0, upper_bound, tau_opt, P_opt_dict, termination_status(model)
end

function run_nonlinear_solver_upper_bound_speed(N::Int, M::Int, K::Int, max_time::Float64, rho::Vector{Int}, h_error::Float64, initial_state::Int, initial_P_values_dict::Dict{Int, Vector{Float64}}, initial_tauval_array::Vector{Float64}, fix_P_dict::Dict{Int, Bool}, boundary_type::String)
    #function to run the non-linear solver with the additional constraint of hitting time being less than max_time
    #Arguments:
    #N: the number of internal states
    #M: the number of cells
    #K: the number of independent strategies
    #max_time: the minimum hitting time
    #rho: the vector of length M, telling which strategy each cell uses
    #h_error: the error tolerance for hitting probability
    #initial_state: the initial state of the system
    #initial_P_values_dict: a dictionary containing the K initial strategy vectors p
    #initial_tauval_array: the initial hitting times
    #fix_P_dict: a dictionary containing the K boolean values, telling whether to fix the P values for each cell
    #boundary_type: the type of boundary condition
    #Returns:
    #upper_bound_tau_0: the upper bound on the hitting time
    #upper_bound: the upper bound on the hitting time
    #tau_opt: the optimal hitting times
    #P_opt_dict: the optimal P vectors
    #termination_status: the termination status of the solver
    
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 10000)
    state_dict, state_dict_inv, _, TG, TB, Tc = CellularDecisions.statematrices(N, M, boundary_type)
    ni, np = CellularDecisions.varioussizes(N, M)

    targetstates_good = [target_state + 1 for target_state in TG]
    targetstates_bad = [target_state + 1 for target_state in TB]
    startstates = [start_state + 1 for start_state in Tc]
    allstates = [startstates; targetstates_good; targetstates_bad]
    all_targetstates = vcat(targetstates_good, targetstates_bad)

    @variable(model, 0.0 <= P_[1:K, 1:np] <= 1)
    P_vars = Dict{Int, Vector{VariableRef}}(j => [P_[j, i] for i in 1:np] for j in 1:K)

    @variable(model, 0.0<=τ[1:ni])
    @variable(model, 0.0 <= h[1:ni] <= 1.0)

    @objective(model, Min, τ[initial_state])

    P_vars_mapped = Dict{Int, Vector{VariableRef}}(m => P_vars[rho[m]] for m in 1:M)
    Q = Q_maker_original_mod(P_vars_mapped, N, M, model, state_dict, state_dict_inv)

    @expression(model, A, hitting_time_mod_give_A(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @expression(model, b, hitting_time_mod_give_b(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @constraint(model, A * τ == b)

    @expression(model, Ah, hitting_prob_mod_give_A(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @expression(model, bh, hitting_prob_mod_give_b(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @constraint(model, Ah * h == bh)
    @constraint(model, h[initial_state] <= h_error)
    @constraint(model, τ[initial_state] <= max_time) #the additional constraint

    for k in 1:K
        for i in 1:np
            set_start_value(P_vars[k][i], initial_P_values_dict[1][i])
        end
    end
    
    for i in 1:ni
        set_start_value(τ[i], initial_tauval_array[i])
    end

    for m in 1:M
        if fix_P_dict[m]
            k = rho[m]
            for i in 1:np
                @constraint(model, P_vars[k][i] == initial_P_values_dict[m][i])
            end
        end
    end

    if boundary_type == "boundary_2"
        for k in 1:K
            @constraint(model, P_vars[k][1] == 0.0)
            @constraint(model, P_vars[k][N+1] == 0.0)
            @constraint(model, P_vars[k][2*N+N] == 0.0)
            @constraint(model, P_vars[k][3*N+N] == 0.0)
        end
    end

    JuMP.optimize!(model)

    tau_opt = value.(τ)
    P_opt_ = value.(P_)
    
    P_opt_dict = Dict{Int, Vector{Float64}}(m => P_opt_[rho[m], :] for m in 1:M)

    tau_opt_tilde = tau_opt[startstates]
    upper_bound = maximum(tau_opt_tilde)
    upper_bound_tau_0 = tau_opt[1]

    return upper_bound_tau_0, upper_bound, tau_opt, P_opt_dict, termination_status(model)
end

function run_nonlinear_solver_for_search_in_phase_space(N::Int, M::Int, K::Int, rho::Vector{Int}, max_time::Float64, initial_state::Int, initial_P_values_dict::Dict{Int, Vector{Float64}}, initial_tauval_array::Vector{Float64}, fix_P_dict::Dict{Int, Bool}, boundary_type::String, max_iter_number::Int=1000, verbose_flag::Bool=true)
    # Function to do phase space sampling for other sub-optimal solutions to the problem (i.e. sampling on the relevant manifold)
    # This does so by trying to solve an analgous problem (minimizing hitting probability subject to hitting time constraint), but only for a fixed number of iterations

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", max_iter_number)
    set_optimizer_attribute(model, "acceptable_tol",             1e-4)  
    set_optimizer_attribute(model, "acceptable_iter",            1)
    
    set_attribute(model, "linear_solver", "ma97")
    set_optimizer_attribute(model, "linear_system_scaling", "mc19") 

    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    if !verbose_flag
        set_optimizer_attribute(model, "print_level", 0)
    end

    state_dict, state_dict_inv, _, TG, TB, Tc = CellularDecisions.statematrices(N, M, boundary_type)
    ni, np = CellularDecisions.varioussizes(N, M)

    targetstates_good = [target_state + 1 for target_state in TG]
    targetstates_bad = [target_state + 1 for target_state in TB]
    startstates = [start_state + 1 for start_state in Tc]
    allstates = [startstates; targetstates_good; targetstates_bad]
    all_targetstates = vcat(targetstates_good, targetstates_bad)

    @variable(model, 0 <= P_[1:np] <= 1)
    P_vars = Dict{Int, Vector{VariableRef}}(j => [P_[i] for i in 1:np] for j in 1:K)

    @variable(model, 0.0 <= τ[1:ni])
    @variable(model, 0.0 <= h[1:ni] <= 1.0)

    @objective(model, Min, h[initial_state]) #new objective function
    
    P_vars_mapped = Dict{Int, Vector{VariableRef}}(m => P_vars[rho[m]] for m in 1:M)
    Q = Q_maker_original_mod(P_vars_mapped, N, M, model, state_dict, state_dict_inv)

    @expression(model, A, hitting_time_mod_give_A(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @expression(model, b, hitting_time_mod_give_b(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @constraint(model, A * τ == b)
    @constraint(model, τ[initial_state] <= max_time)

    @objective(model, Min, h[initial_state])
    @expression(model, Ah, hitting_prob_mod_give_A(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @expression(model, bh, hitting_prob_mod_give_b(Q, targetstates_good, targetstates_bad, allstates, 0.0, model))
    @constraint(model, Ah * h == bh)

    for k in 1:K
        for i in 1:np
            set_start_value(P_[i], initial_P_values_dict[1][i])
        end
    end
    

    if boundary_type == "boundary_2"
        #this is the boundary condition considered in the main text
        for k in 1:K
            @constraint(model, P_[1] == 0.0)
            @constraint(model, P_[N+1] == 0.0)
            @constraint(model, P_[2*N+N] == 0.0)
            @constraint(model, P_[3*N+N] == 0.0)
        end
    end

    JuMP.optimize!(model)

    P_opt_ = value.(P_)
    
    P_opt_dict = Dict{Int, Vector{Float64}}(m => P_opt_ for m in 1:M)
    h_opt = value.(h)

    return h_opt, P_opt_dict, termination_status(model)
end
