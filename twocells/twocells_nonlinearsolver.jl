using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, DataStructures, MathOptInterface, Printf, Gurobi, MosekTools

function run_nonlinear_solver(N::Int, λ::Float64, initial_Pval::Float64, initial_tauval::Float64, g_fixed::Bool, f_fixed::Bool)
    #Find upper bound
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", 1e-8)
    set_silent(model) 
    # set_optimizer_attribute(model, "print_level", 0)

    S,Skeyer,T,TG,TB,Tc=statematrices(N);
    ni,np,ns,nt=varioussizes(N)

    # set_optimizer_attribute(model, "print_level", 0)

    targetstates_good=[target_state+1 for target_state ∈ TG];
    targetstates_bad=[target_state+1 for target_state ∈ TB];
    targetstates=[targetstates_good;targetstates_bad]
    startstates=[start_state+1 for start_state ∈ Tc];
    allstates=[startstates;targetstates_good; targetstates_bad]
    all_targetstates = vcat(targetstates_good, targetstates_bad)


    @variable(model, 0<=P_[1:np] <= 1) 
    # @variable(model, 0<=P_[1:np]) 
    @variable(model, τ[1:ni]) 
    @objective(model, Min, τ[1])
    @expression(model, A, hitting_time_mod_give_A(Q_maker_original_mod(P_, N, λ, model, S, Skeyer), 
                targetstates_good, targetstates_bad, allstates, λ, model))
    @expression(model, b, hitting_time_mod_give_b(Q_maker_original_mod(P_, N, λ, model, S, Skeyer), 
                targetstates_good, targetstates_bad, allstates, λ, model))
    @constraint(model, A * τ == b)
    #Let's fix g:
    if(g_fixed==true)
        @constraint(model, P_[4*N+1]==0.0)
        for i in 2:N+1
            @constraint(model, P_[4*N+i] == 1.0)
        end
    end

    #let's fix some values of f^+(.,0)
    if(f_fixed==true)   
        for i in 2:N
            @constraint(model, P_[i] == 1.0)
        end
    end
    # # #let's fix some values of f^+(.,1)
    # # for i in 1:N-1
    # #     @constraint(model, P_[N+i] == 0.0)
    # # end

    #let's fix some values of f^-(.,0)
    if(f_fixed==true)   
        for i in 1:N-1
            @constraint(model, P_[2*N+i] == 0.0)
        end
    end
    # #let's fix some values of f^-(.,1)
    # for i in 1:N-1
    #     @constraint(model, P_[3*N+i] == 1.0)
    # end
    # #Let's fix g
    # @constraint(model,P_[4*N+1]==0.0)
    # for i in 2:N+1
    #     @constraint(model, P_[4*N+i] == 1.0)
    # end

    #let's fix f^+(,1)
    # for i in 1:N
    #     @constraint(model, P_[N+i] == 0.0)
    # end
    #let's fix f^-(,1)
    # for i in 1:N
    #     @constraint(model, P_[3*N+i] == 1.0)
    # end
    for i in 1:np
        set_start_value(P_[i], initial_Pval)  # Initial guess for P_
    end

    for i in 1:ns
        set_start_value(τ[i], initial_tauval)   # Initial guess for τ
    end  

    JuMP.optimize!(model)
    P_opt=value.(P_);
    tau_opt=value.(τ);
    tau_opt_tilde = tau_opt[startstates]

    upper_bound=maximum(tau_opt_tilde)
    upper_bound_tau_0=tau_opt[1]  
    # output_text=@sprintf("Termination Status: %s, Upper bound τ₀: %s, Upper Bound on the τ̃:%s", termination_status(model),string(upper_bound_tau_0),string(upper_bound))
    # println(output_text)
    return upper_bound_tau_0,upper_bound,tau_opt,P_opt,termination_status(model)
end

function run_nonlinear_solver_bounded(N::Int, λ::Float64, P_bounds_L::Vector{Float64}, P_bounds_B::Vector{Float64}, tau_bounds_L::Vector{Float64}, tau_bounds_B::Vector{Float64}, initial_Pval::Float64, initial_tauval::Float64)
    #Find upper bound
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", 1e-8)
    set_silent(model) 
    # set_optimizer_attribute(model, "print_level", 0)

    S,Skeyer,T,TG,TB,Tc=statematrices(N);
    ni,np,ns,nt=varioussizes(N)

    # set_optimizer_attribute(model, "print_level", 0)

    targetstates_good=[target_state+1 for target_state ∈ TG];
    targetstates_bad=[target_state+1 for target_state ∈ TB];
    targetstates=[targetstates_good;targetstates_bad]
    startstates=[start_state+1 for start_state ∈ Tc];
    allstates=[startstates;targetstates_good; targetstates_bad]
    all_targetstates = vcat(targetstates_good, targetstates_bad)


    @variable(model, 0<=P_[1:np] <= 1) 
    @variable(model, τ[1:ni]) 
    @objective(model, Min, τ[1])
    @expression(model, A, hitting_time_mod_give_A(Q_maker_original_mod(P_, N, λ, model, S, Skeyer), 
                targetstates_good, targetstates_bad, allstates, λ, model))
    @expression(model, b, hitting_time_mod_give_b(Q_maker_original_mod(P_, N, λ, model, S, Skeyer), 
                targetstates_good, targetstates_bad, allstates, λ, model))
    @constraint(model, A * τ == b)

    for i in 1:np
        @constraint(model, P_bounds_L[i]<= P_[i] <= P_bounds_B[i] )
    end
    
    for i in 1:ni
        @constraint(model, tau_bounds_L[i]<= τ[i] <= tau_bounds_B[i] )
    end

    for i in 1:np
        set_start_value(P_[i], initial_Pval)  # Initial guess for P_
    end

    for i in 1:ns
        set_start_value(τ[i], initial_tauval)   # Initial guess for τ
    end  

    JuMP.optimize!(model)
    P_opt=value.(P_);
    tau_opt=value.(τ);
    tau_opt_tilde = tau_opt[startstates]

    upper_bound=maximum(tau_opt_tilde)
    upper_bound_tau_0=tau_opt[1]  
    output_text=@sprintf("Termination Status: %s, Upper bound τ₀: %s, Upper Bound on the τ̃:%s", termination_status(model),string(upper_bound_tau_0),string(upper_bound))
    println(output_text)
    return upper_bound_tau_0,upper_bound,tau_opt,P_opt,termination_status(model)
end

function run_pipeline_for_various_lambda(N::Int, lambda_values::Vector{Float64}, num_simulations::Int, initialPval::Float64, initialtauval::Float64, main_folder)
    lambda_start = minimum(lambda_values)
    lambda_start_str = replace(string(lambda_start), "." => "_")
    lambda_end = maximum(lambda_values)
    lambda_end_str=lambda_str = replace(string(lambda_end), "." => "_")
    sub_folder = @sprintf("simulation_parameter_results_N_%d_lambdastart_%s_lambdaend_%s", N, lambda_start_str,lambda_end_str)
    overall_folder=joinpath(main_folder,sub_folder)
    if !isdir(overall_folder)
        mkpath(overall_folder)
    end
    movie_folder = @sprintf("movie_folder")
    overall_movie_folder=joinpath(overall_folder,movie_folder)
    if !isdir(overall_movie_folder)
        mkpath(overall_movie_folder)
    end

    movie_folder_2 = @sprintf("movie_folder_2")
    overall_movie_folder_2=joinpath(overall_folder,movie_folder_2)
    if !isdir(overall_movie_folder_2)
        mkpath(overall_movie_folder_2)
    end

    λ_vals_to_plot=[]
    τ_0_values = [] 
    τ_0_values_Simp = [] 
    τ_tilde_bounds=[]   
    isirreducible_values=[]
    λ_transition=300
    for λ in lambda_values
        currlambdaforfilename=round(Int,λ*100)
        # set_optimizer_attribute(model, "tol", 1e-8)
        S, Skeyer, T, TG, TB, Tc = statematrices(N)
        ni, np, ns, nt = varioussizes(N)
        targetstates_good = [target_state + 1 for target_state ∈ TG]
        targetstates_bad = [target_state + 1 for target_state ∈ TB]
        targetstates = [targetstates_good; targetstates_bad]
        startstates = [start_state + 1 for start_state ∈ Tc]
        allstates = [startstates; targetstates_good; targetstates_bad]
        all_targetstates = vcat(targetstates_good, targetstates_bad)

        upper_bound_tau_0,upper_bound,tau_opt,P_opt,termination_status=run_nonlinear_solver(N, λ, initialPval, initialtauval)
        if termination_status!=MOI.OPTIMAL && termination_status!=MOI.LOCALLY_SOLVED
            println(N,"; SKIPPED Upper Bound->",λ," ", termination_status)
            initialtauval=0.0
            initialPval=1.0
            continue
        end
        # if termination_status!=MOI.OPTIMAL && termination_status!=MOI.LOCALLY_SOLVED
        #     upper_bound_tau_0,upper_bound,tau_opt,P_opt,termination_status=run_nonlinear_solver(N, λ, initialPval, upper_bound_tau_0)
        #     if termination_status!=MOI.OPTIMAL && termination_status!=MOI.LOCALLY_SOLVED
        #         println("SKIPPED Upper Bound->",λ," ", termination_status)
        #         continue
        #     end
        # end
        P_opt_ = P_opt .* (abs.(P_opt) .>= 1e-8)
        P_opt_ .= min.(P_opt_, 1.0)
        Q_opt = Q_maker_using_M(P_opt_, N, λ, S, Skeyer)

        # println(is_irreducible(Q_opt))

        if (!is_irreducible(Q_opt))
            println(N,"; SKIPPED Upper Bound->",λ, ",",tau_opt[1]," : The solution found leads to a redducible rate matrix")
            initialtauval=0.0
            initialPval=1.0
            continue
        end

        upper_bound_tau_0_Simp,upper_bound_Simp,tau_opt_Simp,P_opt_Simp,termination_status_Simp=run_nonlinear_solver_simplified(N, λ, initialPval, initialtauval)
        if termination_status_Simp!=MOI.OPTIMAL && termination_status_Simp!=MOI.LOCALLY_SOLVED
            upper_bound_tau_0_Simp,upper_bound_Simp,tau_opt_Simp,P_opt_Simp,termination_status_Simp=run_nonlinear_solver_simplified(N, λ, initialPval, upper_bound_tau_0_Simp)
            if termination_status_Simp!=MOI.OPTIMAL && termination_status_Simp!=MOI.LOCALLY_SOLVED
                println(N,"; SKIPPED Lower Bound->",λ," ", termination_status_Simp)
                initialtauval=0.0
                initialPval=1.0
                continue
            end
        end
        lambda_str = replace(string(λ), "." => "_")
        sub_sub_folder_ = @sprintf("lambda_%s", lambda_str)
        overall_sub_sub_folder=joinpath(overall_folder,sub_sub_folder_)
        if !isdir(overall_sub_sub_folder)
            mkpath(overall_sub_sub_folder)
        end
        initialPval=1.0
        initialtauval=upper_bound#tau_opt[1]
        tau_opt_tilde = tau_opt[startstates]
        upper_bound=maximum(tau_opt_tilde)
        push!(λ_vals_to_plot,λ)
        push!(τ_0_values, tau_opt[1])
        push!(τ_tilde_bounds, upper_bound)
        push!(isirreducible_values,is_irreducible(Q_opt))
        push!(τ_0_values_Simp, tau_opt_Simp[1])
        println(λ," ",termination_status," ", tau_opt[1],"; ",termination_status_Simp," ", tau_opt_Simp[1])

        Q_filename = generate_filename(overall_sub_sub_folder,"Q_matrix_heatmap")
        plot_Q_with_colored_yticks(Q_opt, N, all_targetstates, Q_filename,λ)

        initial_state = 1
        T = 100.0
        times, states = simulate_ctmc(Q_opt, initial_state, T)
        
        ctmc_simulation_filename = generate_filename(overall_sub_sub_folder,"single_ctmc_simulation")
        plot_ctmc_our_problem(times, states, T, N, ctmc_simulation_filename,λ)

        initial_state = 1
        T = 7000.0

        longtime_heatmap_simulation_filename = generate_filename(overall_sub_sub_folder,"multiple_ctmc_simulation_heatmap_longtime")
        plot_ctmc_our_problem_multi(Q_opt, initial_state, T, N, num_simulations, longtime_heatmap_simulation_filename,λ)


        invariant_heatmap_simulation_filename = generate_filename(overall_sub_sub_folder,"invariant_ctmc_heatmap")
        what_is_the_current_status=plot_ctmc_invar_distn_our_problem(Q_opt,  N, invariant_heatmap_simulation_filename, λ)

        if(what_is_the_current_status)
            λ_transition=minimum([λ,λ_transition])
        end

        frame_filename = generate_filename(overall_movie_folder, @sprintf("final_frame_lambda_%s",lambda_str))
        source_path = longtime_heatmap_simulation_filename*".png"
        cp(source_path, frame_filename*".png"; force=true)

        frame_filename = generate_filename(overall_movie_folder_2, @sprintf("final_frame_lambda_%s",lambda_str))
        source_path = invariant_heatmap_simulation_filename*".png"
        cp(source_path, frame_filename*".png"; force=true)
    end
    transition_column = [λ == λ_transition for λ in λ_vals_to_plot]
    title_name=@sprintf("Upper Bound τ₀, N=%d",N)
    display(plot(λ_vals_to_plot, τ_0_values, xlabel="λ", ylabel="Upper Bound τ₀", lw=2, legend=false, title=title_name))
    plot_file_name = generate_filename(overall_folder,"plot_of_optimal_tau_0_vs_lambda")
    savefig(plot_file_name*".png")
    savefig(plot_file_name*".svg")

    title_name=@sprintf("Lower Bound τ₀, N=%d",N)
    display(plot(λ_vals_to_plot, τ_0_values_Simp, xlabel="λ", ylabel="Lower Bound τ₀", lw=2, legend=false, title=title_name))
    plot_file_name = generate_filename(overall_folder,"plot_of_optimal_simplified_tau_0_vs_lambda")
    savefig(plot_file_name*".png")
    savefig(plot_file_name*".svg")

    title_name=@sprintf("Upper Bound on τ̃ , N=%d",N)
    display(plot(λ_vals_to_plot, τ_tilde_bounds, xlabel="λ", ylabel="Upper Bound on τ̃", lw=2, legend=false, title=title_name))
    plot_file_name = generate_filename(overall_folder,"plot_of_tau_tilde_upper_bound_vs_lambda")
    savefig(plot_file_name*".png")
    savefig(plot_file_name*".svg")

    df = DataFrame(Lambda_Values = λ_vals_to_plot, Tau_0_Values = τ_0_values, Tau_0_Simplified_Values=τ_0_values_Simp,IsIrreducible = isirreducible_values,Tau_tilde_bounds=τ_tilde_bounds,IsTransitionPoint = transition_column)
    csv_filename=generate_filename(overall_folder, @sprintf("lambda_tau_values"))
    CSV.write(csv_filename*".csv", df)

    # movie1 = generate_filename(overall_folder, @sprintf("longtimeruns_movie_N_%d",N))
    # movie2 = generate_filename(overall_folder, @sprintf("invariantdistr_movie_N_%d",N))

    return λ_vals_to_plot, τ_0_values,τ_0_values_Simp, isirreducible_values
end

function run_nonlinear_solver_simplified(N::Int, λ::Float64, initial_Pval::Float64, initial_tauval::Float64)
    #Find upper bound
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", 1e-8)
    set_silent(model) 

    S,Skeyer,T,TG,TB,Tc=statematrices(N);
    ni,np,ns,nt=varioussizes(N)

    targetstates_good=[target_state+1 for target_state ∈ TG];
    targetstates_bad=[target_state+1 for target_state ∈ TB];
    targetstates=[targetstates_good;targetstates_bad]
    startstates=[start_state+1 for start_state ∈ Tc];
    allstates=[startstates;targetstates_good; targetstates_bad]
    all_targetstates = vcat(targetstates_good, targetstates_bad)

    nP=2*np
    @variable(model, 0<=P_[1:nP] <= 1) 
    @variable(model, τ[1:ni]) 
    @objective(model, Min, τ[1])
    @expression(model, A, hitting_time_mod_give_A(Q_maker_simplified_mod(P_, N, λ, model, S, Skeyer), 
                targetstates_good, targetstates_bad, allstates, λ, model))
    @expression(model, b, hitting_time_mod_give_b(Q_maker_simplified_mod(P_, N, λ, model, S, Skeyer), 
                targetstates_good, targetstates_bad, allstates, λ, model))
    @constraint(model, A * τ == b)
    for i in 1:nP
        set_start_value(P_[i], initial_Pval)  # Initial guess for P_
    end

    for i in 1:ns
        set_start_value(τ[i], initial_tauval)   # Initial guess for τ
    end  

    JuMP.optimize!(model)
    P_opt=value.(P_);
    tau_opt=value.(τ);
    tau_opt_tilde = tau_opt[startstates]

    upper_bound=maximum(tau_opt_tilde)
    upper_bound_tau_0=tau_opt[1]  
    # local termination_status=termination_status(model)
    # output_text=@sprintf("Simplified Problem: Termination Status: %s, Upper bound τ₀: %s, Upper Bound on the τ̃:%s", termination_status(model),string(upper_bound_tau_0),string(upper_bound))
    # println(output_text)
    return upper_bound_tau_0,upper_bound,tau_opt,P_opt,termination_status(model)
end