#Sampling-based optimization for the collective cellular self-organization problem (CTMC)

using Random
using LinearAlgebra
using Statistics
using Distributions
using Printf

#==============================================================================#
#                           CTMC TRANSITION FUNCTIONS                          #
#==============================================================================#
#these are functions written to handle the particular case of the self-organization problem considered in the paper

function outgoing_rates(state_array_curr, state_array_next, AdjMat, P_opt_dict, N, M)
    #function to calculate the outgoing transition rates from the current state to the next state
    #this is necessary as we can't save the full rate matrix due to memory constraints
    #written to handle the particular case of the self-organization problem considered in the paper
    #Arguments:
    #state_array_curr: the current state of the system as a tuple of (internal_state, signal_state) pairs
    #state_array_next: the next state of the system as a tuple of (internal_state, signal_state) pairs
    #AdjMat: the adjacency matrix of the cell connectivity
    #P_opt_dict: the dictionary mapping the cell index to the parameter vector
    #N: the number of internal states - 1
    #M: the number of cells
    #Returns:
    #the outgoing transition rate from the current state to the next state

    flags_found = zeros(Int, M)
    changed_cell = 0
    
    # need to find the cell that changed
    for i in 1:M
        if state_array_curr[i] != state_array_next[i]
            flags_found[i] = 1
            changed_cell = i
        end
    end
    
    # early return if no valid transition (i.e. must change exactly one cell)
    if sum(flags_found) != 1 || changed_cell == 0
        return 0.0
    end

    curr_cell_now = state_array_curr[changed_cell]
    curr_cell_next = state_array_next[changed_cell]

    #if there is a transition, it is one of two possible types:
    
    # Case A: change in internal state, no change in signal receiving state
    if curr_cell_now[1] != curr_cell_next[1] && curr_cell_now[2] == curr_cell_next[2]
        if curr_cell_next[1] == curr_cell_now[1] + 1
            return P_opt_dict[changed_cell][curr_cell_now[2]*N + curr_cell_now[1] + 1]
        elseif curr_cell_next[1] == curr_cell_now[1] - 1
            return P_opt_dict[changed_cell][curr_cell_now[2]*N + curr_cell_now[1] + 2N]
        end
    end
    
    # Case B: change in signal receiving state, no change in internal state
    if curr_cell_now[1] == curr_cell_next[1] && curr_cell_now[2] != curr_cell_next[2]
        if curr_cell_next[2] == 0 && curr_cell_now[2] == 1
            # Signal turns off (k_off)
            return P_opt_dict[changed_cell][5*N + 2]
        elseif curr_cell_next[2] == 1 && curr_cell_now[2] == 0
            # Signal turns on (depends on neighboring cells' internal states)
            all_cells_neighbors = findall(AdjMat[changed_cell, :] .== 1)
            rate_to_give = 0.0
            for cell_neighbor in all_cells_neighbors
                rate_to_give += P_opt_dict[cell_neighbor][4*N + state_array_curr[cell_neighbor][1] + 1]
            end
            return rate_to_give
        end
    end
    
    return 0.0
end

function all_states_from_curr_state(state_array_curr, AdjMat, N, M)
    #function to find all possible next states reachable from the current state
    #Arguments:
    #state_array_curr: the current state of the system as a tuple of (internal_state, signal_state) pairs
    #AdjMat: the adjacency matrix of the cell connectivity
    #N: the number of internal states - 1
    #M: the number of cells
    #Returns:
    #a vector of all possible next states

    all_states = []

    for i in 1:M
        # need to handle the internal state transitions (increase/decrease)
        if state_array_curr[i][1] != N && state_array_curr[i][1] != 0
            # internal state increases
            new_state = [(j == i ? (state_array_curr[j][1] + 1, state_array_curr[j][2]) : 
                         deepcopy(state_array_curr[j])) for j in 1:M]
            push!(all_states, tuple(new_state...))
            
            # internal state decreases
            new_state = [(j == i ? (state_array_curr[j][1] - 1, state_array_curr[j][2]) : 
                         deepcopy(state_array_curr[j])) for j in 1:M]
            push!(all_states, tuple(new_state...))
        end

        # need to handle the signal receiving state transitions
        if state_array_curr[i][2] == 0
            new_state = [(j == i ? (state_array_curr[j][1], 1) :
                         deepcopy(state_array_curr[j])) for j in 1:M]
            push!(all_states, tuple(new_state...))
        elseif state_array_curr[i][2] == 1
            new_state = [(j == i ? (state_array_curr[j][1], 0) :
                         deepcopy(state_array_curr[j])) for j in 1:M]
            push!(all_states, tuple(new_state...))
        end
    end
    return all_states
end

#==============================================================================#
#                           CTMC SIMULATION                                    #
#==============================================================================#
#function to simulate a CTMC trajectory until terminal time or reaching a terminal state
#written to handle the particular case of the self-organization problem considered in the paper

function simulate_ctmc_faster(N, M, AdjMat, P_opt_dict, initial_state, T, TG, TB)
    #function to simulate a CTMC trajectory until terminal time or reaching a terminal state (good or bad)
    #written to handle the particular case of the self-organization problem considered in the paper
    #Arguments:
    #N: the number of internal states - 1
    #M: the number of cells
    #AdjMat: the adjacency matrix of the cell connectivity
    #P_opt_dict: the dictionary mapping the cell index to the parameter vector
    #initial_state: the initial state of the system as a tuple of (internal_state, signal_state) pairs
    #T: the maximum simulation time
    #TG: the set of good terminal states (internal states only)
    #TB: the set of bad terminal states (internal states only)
    #Returns:
    #times: the vector of transition times
    #states_till_now: the vector of states visited
    #terminal_time: the time at which simulation ended
    #is_bad: 1 if ended in bad state, 0 otherwise

    t = 0.0
    times = [t]
    states_till_now = [initial_state]
    is_bad = 0
    terminal_time = T
    
    while t < T
        curr_state = states_till_now[end]
        all_states = all_states_from_curr_state(curr_state, AdjMat, N, M)
        all_outgoing_rates = [outgoing_rates(curr_state, all_states[i], AdjMat, P_opt_dict, N, M) 
                             for i in 1:length(all_states)]
        sum_of_outgoing_rates = sum(all_outgoing_rates)
        
        if sum_of_outgoing_rates <= 1e-9
            #if the sum of outgoing rates is very small, we can't simulate any more transitions
            break
        end
        
        Δt = rand(Exponential(1/sum_of_outgoing_rates))
        t += Δt
        
        if t >= T
            #if the simulation time is greater than the terminal time, we can't simulate any more transitions
            break
        end
        
        probs = all_outgoing_rates ./ sum_of_outgoing_rates
        dist = Categorical(probs)
        s = rand(dist)
        
        push!(states_till_now, all_states[s])
        push!(times, t)
        
        # need to check if the terminal state has been reached
        curr_state_full = all_states[s]
        current_internal_states = tuple([curr_state_full[i][1] for i in 1:M]...)
        
        if current_internal_states ∈ TB
            #if the current state is a bad terminal state, set the is_bad flag to 1
            is_bad = 1
        end
        
        if current_internal_states ∈ TG || current_internal_states ∈ TB
            #if the current state is a good or bad terminal state, set the terminal time to the current time
            terminal_time = min(terminal_time, t)
            break
        end
    end

    #now check if the final state is a terminal state (good or bad, doesn't matter)
    final_state = states_till_now[end]
    final_internal_states = tuple([final_state[i][1] for i in 1:M]...)

    if !(final_internal_states ∈ TG || final_internal_states ∈ TB)
        #Explicitly handle the case where the final state is not a terminal state (and so ended due to right censoring due to max time for simulation)
        terminal_time = T
    end

    return times, states_till_now, terminal_time, is_bad
end

#==============================================================================#
#                           SCORE FUNCTION (POLICY GRADIENT)                   #
#==============================================================================#
#function to calculate the score function for a given path
#written to handle the particular case of the self-organization problem considered in the paper

function score_function(N, M, AdjMat, parameter_opt, times, states)
    #function to calculate the score function for a given path
    #written to handle the particular case of the self-organization problem considered in the paper
    #Arguments:
    #N: the number of internal states - 1
    #M: the number of cells
    #AdjMat: the adjacency matrix of the cell connectivity
    #parameter_opt: the parameter vector
    #times: the vector of transition times
    #states: the vector of states
    #Returns:
    #the score function for the given path, a vector of length 5N-2

    score_path = zeros(5*N - 2)

    #epsilon is a small positive number to avoid division by zero
    epsilon = 1e-8
    
    for i in 1:length(times)-1
        curr_state = states[i]
        next_state = states[i+1]
        delta_t = times[i+1] - times[i]
        
        # need to find the cell that changed
        changed_cell = findall(curr_state .!= next_state)
        if length(changed_cell) != 1
            error("More than one cell changed at the same time")
        end
        changed_cell = changed_cell[1]
        all_cells_neighbors = findall(AdjMat[changed_cell, :] .== 1)
        
        curr_cell_u = curr_state[changed_cell][1] #current internal state of the cell that changed
        curr_cell_s = curr_state[changed_cell][2] #current signal receiving state of the cell that changed
        next_cell_u = next_state[changed_cell][1] #next internal state of the cell that changed
        next_cell_s = next_state[changed_cell][2] #next signal receiving state of the cell that changed

        first_term = zeros(5*N - 2) #first term of the score function
        second_term = zeros(5*N - 2) #second term of the score function

        # Compute total rate for signalling increase
        total_rate_change_for_signalling_increase = 0.0
        for cell_neighbor in all_cells_neighbors
            total_rate_change_for_signalling_increase += parameter_opt[4*N-4+curr_state[cell_neighbor][1]+1]
        end

        #first term: derivative of log(rate) for the transition that occurred
        if curr_cell_u + 1 == next_cell_u
            # f_plus transition
            first_term[curr_cell_u+curr_cell_s*(N-1)] += 1/(parameter_opt[curr_cell_u+curr_cell_s*(N-1)] + epsilon)
        elseif curr_cell_u - 1 == next_cell_u
            # f_minus transition
            first_term[curr_cell_u+curr_cell_s*(N-1)+2*N-2] += 1/(parameter_opt[curr_cell_u+curr_cell_s*(N-1)+2*N-2] + epsilon)
        elseif curr_cell_s == 0 && next_cell_s == 1
            # signal received
            for u_temp in 0:N
                numerator = 0.0
                denominator = total_rate_change_for_signalling_increase
                for cell_neighbor in all_cells_neighbors
                    if curr_state[cell_neighbor][1] == u_temp
                        numerator += 1
                    end
                end
                first_term[4*N-4+u_temp+1] += numerator/(denominator + epsilon)
            end
        elseif curr_cell_s == 1 && next_cell_s == 0
            # signal turns off (k_off)
            first_term[5*N-2] += 1/(parameter_opt[5*N-2] + epsilon)
        end
        
        #second term: integral of rates over time
        #which can be evaluated as a sum
        for cell_number in 1:M
            curr_cell_u_temp = curr_state[cell_number][1]
            curr_cell_s_temp = curr_state[cell_number][2]
            cell_neighbours_temp = findall(AdjMat[cell_number, :] .== 1)
            temp_vector = zeros(5*N - 2)
            
            if curr_cell_u_temp < N && curr_cell_u_temp > 0
                temp_vector[curr_cell_u_temp+curr_cell_s_temp*(N-1)] += 1
                temp_vector[curr_cell_u_temp+curr_cell_s_temp*(N-1)+2*N-2] += 1
            end
            if curr_cell_s_temp == 1
                temp_vector[5*N-2] += 1
            end
            if curr_cell_s_temp == 0
                for cell_neighbour in cell_neighbours_temp
                    u_prime = curr_state[cell_neighbour][1]
                    temp_vector[4*N-4+u_prime+1] += 1
                end
            end
            second_term += temp_vector
        end
        
        score_path .+= first_term .- second_term .* delta_t
    end
    
    return score_path
end

#==============================================================================#
#                           BATCH SIMULATION                                   #
#==============================================================================#

function simulate_batch(N, M, AdjMat, parameter_opt, initial_state, T, TG, TB, N_simulations; log_fn=nothing)
    #run a batch of CTMC simulations in parallel

    terminal_times = zeros(N_simulations)
    is_bad = zeros(N_simulations)
    Score_functions = zeros(N_simulations, 5*N - 2)
    
    # Convert parameter_opt to full parameter vector
    # this is just a pre-processing step, adding 0.0 to ensure that internal state can't increase/decrease if the current internal state is 0/N
    f_p_0_ours = parameter_opt[1:N-1]
    f_p_0_true = vcat(0.0, f_p_0_ours) 
    f_p_1_ours = parameter_opt[N:2*N-2]
    f_p_1_true = vcat(0.0, f_p_1_ours)
    f_m_0_ours = parameter_opt[2*N-1:3*N-3]
    f_m_0_true = vcat(f_m_0_ours, 0.0)
    f_m_1_ours = parameter_opt[3*N-2:4*N-4]
    f_m_1_true = vcat(f_m_1_ours, 0.0)
    g_ours = parameter_opt[4*N-3 : 5*N-3]
    k_off_ours = parameter_opt[5*N-2]


    param_vector_ours = vcat(f_p_0_true, f_p_1_true, f_m_0_true, f_m_1_true, g_ours, k_off_ours)
    
    param_dict_now = Dict(i => param_vector_ours for i in 1:M)

    num_workers = Threads.nthreads()
    if log_fn !== nothing
        log_fn("Number of workers: $num_workers")
    end
    
    Threads.@threads for i in 1:N_simulations
        times, states, terminal_time, is_bad_temp = simulate_ctmc_faster(N, M, AdjMat, param_dict_now, initial_state, T, TG, TB)
        terminal_times[i] = terminal_time
        is_bad[i] = is_bad_temp
        score_function_temp = score_function(N, M, AdjMat, parameter_opt, times, states)
        Score_functions[i, :] = score_function_temp
    end
    
    return Score_functions, terminal_times, is_bad
end

#==============================================================================#
#                           SGD ESTIMATORS                                     #
#==============================================================================#
function estimators_for_sgd(Scores, terminal_times, is_bad; log_fn=nothing)
    #function to calculate the estimators for SGD from batch simulation results
    #written to handle the particular case of the self-organization problem considered in the paper
    #Arguments:
    #Scores: the matrix of score functions
    #terminal_times: the vector of terminal times
    #is_bad: the vector of bad state indicators
    #log_fn: an optional logging function
    #Returns:
    #T_bar: the mean terminal time
    #is_bad_bar: the mean bad probability
    #grad_J: the gradient of the expected terminal time
    #grad_error: the gradient of the expected error probability

    n_sim, D_param = size(Scores)
    #n_sim is the number of simulations in the batch
    #D_param is the number of parameters in the parameter vector
    
    if log_fn !== nothing
        log_fn("n_sim, D_param: $n_sim, $D_param")
    end
    
    T_bar = mean(terminal_times)
    is_bad_bar = mean(is_bad)
    grad_J = zeros(D_param)
    grad_error = zeros(D_param)
    
    # Compute baseline using norm-squared weighting
    # this is a control variate
    norms_sq = sum(Scores.^2, dims=2)    
    b_T = sum(terminal_times .* vec(norms_sq)) / sum(norms_sq) 
    b_err = sum(is_bad .* vec(norms_sq)) / sum(norms_sq)
    
    for i in 1:n_sim
        curr_T_i = terminal_times[i] #terminal time for the i-th simulation
        curr_is_bad_i = is_bad[i] #bad state indicator for the i-th simulation
        curr_score_function_i = Scores[i, :] #score function for the i-th simulation
        grad_J .+= curr_score_function_i .* (curr_T_i - b_T) #gradient of the expected terminal time
        grad_error .+= curr_score_function_i .* (curr_is_bad_i - b_err) #gradient of the expected error probability
    end
    
    grad_J ./= n_sim #montecarlo estimate of the gradients
    grad_error ./= n_sim #montecarlo estimate of the gradients
    
    return T_bar, is_bad_bar, grad_J, grad_error
end

#==============================================================================#
#                           LEARNING RATE SCHEDULE                             #
#==============================================================================#

function cosine_lr(t; T0=500, T_mult=1.0, η_max=1e-3, η_min=1e-5)
    #function to calculate the learning rate for the SGD step
    #written to handle the particular case of the self-organization problem considered in the paper
    #Arguments:
    #t: the iteration index
    #T0: the length of the first cycle
    #T_mult: the factor to grow the period after each restart
    #η_max: the maximum learning rate
    #η_min: the minimum learning rate
    #Returns:
    #the learning rate for the SGD step
    #follows the paper: https://arxiv.org/abs/1608.03983

    t_rem = t - 1  # for 0-based indexing
    cycle = 0
    Ti = T0
    
    while t_rem >= Ti
        t_rem -= Ti
        Ti = Int(floor(Ti * T_mult))
        cycle += 1
    end
    
    cos_inner = π * t_rem / Ti
    η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(cos_inner))
    
    return η_t
end

#==============================================================================#
#                           PROJECTION                                         #
#==============================================================================#

function project!(θ)
    #function to project the parameters onto the [0, 1] box constraint
    #Arguments:
    #θ: the parameter vector
    #Returns:
    #the projected parameter vector

    @inbounds for j in eachindex(θ)
        θ[j] = min(1.0, max(0.0, θ[j]))
    end
    return θ
end

#==============================================================================#
#                           SGD STEP                                           #
#==============================================================================#

function sgd_step!(θ, λ, ρ, N, M, AdjMat, init_state, T, TG, TB, ε_tol; N_sim=1000, log_fn=nothing)
    #function to perform one SGD step with augmented Lagrangian gradient
    #Arguments:
    #θ: the parameter vector
    #λ: the dual variable
    #ρ: the augmented Lagrangian penalty
    #N: the number of internal states - 1
    #M: the number of cells
    #AdjMat: the adjacency matrix of the cell connectivity
    #init_state: the initial state
    #T: the time horizon
    #TG: the good terminal states
    #TB: the bad terminal states
    #ε_tol: the target error tolerance
    #N_sim: the number of simulations to run
    #log_fn: an optional logging function
    #Returns:
    #Tbar: the mean terminal time
    #bbar: the mean bad probability
    #gθ: the combined gradient for the primal update

    Scores, Ts, bad = simulate_batch(N, M, AdjMat, θ, init_state, T, TG, TB, N_sim; log_fn=log_fn)
    Tbar, bbar, gJ, gE = estimators_for_sgd(Scores, Ts, bad; log_fn=log_fn)

    # Augmented-Lagrangian gradient
    γ = λ + ρ * (bbar - ε_tol)
    gθ = gJ .+ γ .* gE

    if log_fn !== nothing
        log_fn("||gJ||2 = $(norm(gJ))")
        log_fn("||gE||2 = $(norm(gE))")
        log_fn("γ = $(γ)")
        log_fn("||gθ||2 = $(norm(gθ))")
    end

    return Tbar, bbar, gθ
end

#==============================================================================#
#                           MAIN SGD OPTIMIZATION                              #
#==============================================================================#

function run_ctmc_projected_sgd!(θ0::Vector{Float64};
    iters::Int=200,
    N_sim::Int=1000,
    η_max::Float64=3e-3,
    η_min::Float64=3e-4,
    lr_T0::Int=500,
    lr_Tmult::Float64=1.0,
    ρ::Float64=1e-2,
    ε_tol::Float64=0.05,
    N::Int,
    M::Int,
    AdjMat,
    initial_state,
    T::Float64,
    TG,
    TB,
    window_size::Int=50,
    min_iters::Int=200,
    tol_feas::Float64=0.01,
    tol_T::Float64=1e-2,
    tol_λ::Float64=1e-2,
    do_early_stopping::Bool=true,
    λ_start::Float64=0.0,
    log_fn=nothing
)
    #function to run the projected sampling based optimization for the CTMC with augmented Lagrangian constraints
    #Arguments:
    #θ0: the initial parameter vector
    #iters: the number of iterations to run
    #N_sim: the number of simulations to run
    #η_max: the maximum learning rate
    #η_min: the minimum learning rate
    #lr_T0: the length of the first cycle
    #lr_Tmult: the factor to grow the period after each restart
    #ρ: the augmented Lagrangian penalty
    #ε_tol: the target error tolerance
    #N: the number of internal states - 1
    #M: the number of cells
    #AdjMat: the adjacency matrix of the cell connectivity
    #initial_state: the initial state
    #T: the time horizon
    #TG: the good terminal states
    #TB: the bad terminal states
    #window_size: the size of the window for early stopping
    #min_iters: the minimum number of iterations to run
    #tol_feas: the tolerance for feasibility
    #tol_T: the tolerance for terminal time
    #tol_λ: the tolerance for the dual variable
    #do_early_stopping: whether to perform early stopping
    #λ_start: the initial value of the dual variable
    #log_fn: an optional logging function
    #Returns:
    #θ: the optimized parameter vector
    #λ: the final value of the dual variable
    #(
    #    T_hist: the history of terminal times
    #    b_hist: the history of bad probabilities
    #    λ_hist: the history of dual variables
    #    gnorm: the history of gradient norms
    #    Loss_hist: the history of augmented Lagrangian values
    #    stopped_early: whether the optimization stopped early
    #    iter_last: the last iteration index
    #)

    # Use @info as default logger if none provided
    _log = log_fn !== nothing ? log_fn : (msg -> @info msg)
    
    λ = λ_start
    φ = copy(θ0)

    # Adam optimizer state
    m = zeros(length(θ0))
    v = zeros(length(θ0))
    vmax = zeros(length(θ0))  # AMSGrad
    β1 = 0.9
    β2 = 0.999
    eps = 1e-8

    # History tracking
    T_hist = zeros(Float64, iters)
    b_hist = zeros(Float64, iters)
    λ_hist = zeros(Float64, iters)
    gnorm = zeros(Float64, iters)
    Loss_hist = zeros(Float64, iters)
    
    for t in 1:iters
        θ = φ
        
        # SGD step
        T_bar, b_bar, gθ = sgd_step!(θ, λ, ρ, N, M, AdjMat, initial_state, T, TG, TB, ε_tol;
            N_sim=N_sim, log_fn=_log)
        
        if b_bar == 0.0
            _log("b_bar is 0.0, terminating SGD")
            break
        end

        gφ = gθ

        # AMSGrad update
        m .= β1 .* m .+ (1 - β1) .* gφ
        v .= β2 .* v .+ (1 - β2) .* (gφ.^2)
        m̂ = m ./ (1 - β1^t)
        v̂ = v ./ (1 - β2^t)
        vmax .= max.(vmax, v̂)

        # Cosine learning rate
        η_t = cosine_lr(t; T0=lr_T0, T_mult=lr_Tmult, η_max=η_max, η_min=η_min)

        # Primal update with projection
        φ .-= η_t .* m̂ ./ (sqrt.(vmax) .+ eps)
        φ = project!(φ)

        # Dual ascent
        λ = λ + ρ * (b_bar - ε_tol)

        # Compute augmented Lagrangian value
        Loss_now = T_bar + λ * (b_bar - ε_tol) + 0.5 * ρ * (b_bar - ε_tol)^2

        # Record history
        T_hist[t] = T_bar
        b_hist[t] = b_bar
        λ_hist[t] = λ
        gnorm[t] = norm(gθ)
        Loss_hist[t] = Loss_now

        _log("iter=$t  T̄=$(T_bar)  b̄=$(b_bar)  λ=$(λ), Lρ=$(Loss_now)")

        # Early stopping check
        if t >= max(min_iters, 2*window_size)
            s1 = t - window_size + 1
            e1 = t
            s0 = t - 2*window_size + 1
            e0 = t - window_size

            b_mean_curr = mean(b_hist[s1:e1])
            T_mean_curr = mean(T_hist[s1:e1])
            λ_curr = λ_hist[e1]

            b_mean_prev = mean(b_hist[s0:e0])
            T_mean_prev = mean(T_hist[s0:e0])
            λ_prev = λ_hist[e0]

            feasible_curr = (abs(b_mean_curr - ε_tol) <= tol_feas)
            stable_T = abs(T_mean_curr - T_mean_prev) <= tol_T
            stable_λ = abs(λ_curr - λ_prev) <= tol_λ
            
            _log(">>>> b_mean_curr-tol_feas: $(abs(b_mean_curr-ε_tol)), abs(T_mean_curr - T_mean_prev): $(abs(T_mean_curr - T_mean_prev)), abs(λ_curr - λ_prev): $(abs(λ_curr - λ_prev))")
            _log(">>>> feasible_curr: $(feasible_curr), stable_T: $(stable_T)")

            if do_early_stopping && feasible_curr
                θ = φ
                _log("Early stop at iter=$t: feasible on average (b̄_W=$(b_mean_curr)), stable T (ΔT=$(abs(T_mean_curr - T_mean_prev))).")
                return θ, λ, (
                    T_hist = T_hist[1:t],
                    b_hist = b_hist[1:t],
                    λ_hist = λ_hist[1:t],
                    gnorm = gnorm[1:t],
                    Loss_hist = Loss_hist[1:t],
                    stopped_early = true,
                    iter_last = t
                )
            end
        end
    end

    θ = φ
    return θ, λ, (
        T_hist = T_hist,
        b_hist = b_hist,
        λ_hist = λ_hist,
        gnorm = gnorm,
        Loss_hist = Loss_hist,
        stopped_early = false,
        iter_last = iters
    )
end

#==============================================================================#
#                           UTILITY FUNCTIONS                                  #
#==============================================================================#

function params_to_full_vector(θ_opt, N)
    #convert optimized parameter vector (size 5N-2) to full parameter vector (size 5N+2)
    #this is just a pre-processing step, adding 0.0 to ensure that internal state can't increase/decrease if the current internal state is 0/N
    f_p_0_ours = θ_opt[1:N-1]
    f_p_0_true = vcat(0.0, f_p_0_ours)
    f_p_1_ours = θ_opt[N:2*N-2]
    f_p_1_true = vcat(0.0, f_p_1_ours)
    f_m_0_ours = θ_opt[2*N-1:3*N-3]
    f_m_0_true = vcat(f_m_0_ours, 0.0)
    f_m_1_ours = θ_opt[3*N-2:4*N-4]
    f_m_1_true = vcat(f_m_1_ours, 0.0)
    g_ours = θ_opt[4*N-3 : 5*N-3]
    k_off_ours = θ_opt[5*N-2]
    return vcat(f_p_0_true, f_p_1_true, f_m_0_true, f_m_1_true, g_ours, k_off_ours)
end

function validate_params(θ_opt, N, M, AdjMat, initial_state, T, TG, TB, N_simulations; log_fn=nothing)
    #run validation simulations with given parameters by running the CTMC multiple times with the given parameters
    #this is just a sanity check to ensure that the optimized parameters are valid

    terminal_times = zeros(N_simulations)
    is_bad = zeros(N_simulations)
    
    param_vector = params_to_full_vector(θ_opt, N)
    param_dict = Dict(i => param_vector for i in 1:M)
    
    for i in 1:N_simulations
        if log_fn !== nothing && i % 1000 == 0
            log_fn("Validation simulation $i / $N_simulations")
        end
        times, states, terminal_time, is_bad_temp = simulate_ctmc_faster(N, M, AdjMat, param_dict, initial_state, T, TG, TB)
        terminal_times[i] = terminal_time
        is_bad[i] = is_bad_temp
    end
    
    return mean(terminal_times), mean(is_bad)
end
