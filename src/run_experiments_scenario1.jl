include("new_main_2d_action_space_pomdp.jl")
include("main_hybrid_1d_pomdp_path_planner_reuse_old_path.jl")

discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
actions(m::POMDP_Planner_2D_action_space,b) = get_available_actions(b)

function run_experiment_for_given_world_and_noise_with_2D_POMDP_planner(world, rand_noise_generator, iteration_num)

    #Create POMDP for env_right_now
    env_right_now = deepcopy(world)

    golfcart_2D_action_space_pomdp = POMDP_Planner_2D_action_space(0.97,1.0,-100.0,1.0,-100.0,0.0,1.0,1000.0,5.0,env_right_now)

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space),max_depth=100),
                            calculate_upper_bound_value_pomdp_planning_2D_action_space, check_terminal=true),K=50,D=100,T_max=0.5, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_2D_action_space_pomdp);
    #display_env(golfcart_2D_action_space_pomdp().world)

    just_2D_pomdp_all_gif_environments, just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data,
            just_2D_pomdp_all_generated_beliefs, just_2D_pomdp_all_generated_trees, just_2D_pomdp_all_risky_scenarios,
            just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,just_2D_pomdp_time_taken_by_cart,
            just_2D_pomdp_cart_reached_goal_flag = run_one_simulation_2D_POMDP_planner(env_right_now, rand_noise_generator,
                                                                                        golfcart_2D_action_space_pomdp, planner)

    # anim = @animate for i ∈ 1:length(just_2D_pomdp_all_observed_environments)
    #     display_env(just_2D_pomdp_all_observed_environments[i]);
    #     savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*string(i)*".png")
    # end
    #
    # gif_name = "./scenario_1_gifs/just_2D_action_space_pomdp_planner_run_"*string(iteration_num)*"_"*string(just_2D_pomdp_cart_reached_goal_flag)
    # gif_name = gif_name*"_"*string(just_2D_pomdp_time_taken_by_cart)*"_"*string(just_2D_pomdp_number_risks)*".gif"
    # gif(anim, gif_name, fps = 2)
    return just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,just_2D_pomdp_time_taken_by_cart, just_2D_pomdp_cart_reached_goal_flag
end

discount(p::POMDP_Planner_1D_action_space) = p.discount_factor
isterminal(::POMDP_Planner_1D_action_space, s::POMDP_state_1D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
actions(::POMDP_Planner_1D_action_space) = Float64[-1.0, 0.0, 1.0, -10.0]

function run_experiment_for_given_world_and_noise_with_1D_POMDP_planner(world, rand_noise_generator, iteration_num)

    #Create POMDP for env_right_now
    env_right_now = deepcopy(world)

    golfcart_1D_action_space_pomdp = POMDP_Planner_1D_action_space(0.97,1.0,-100.0,1.0,1.0,1000.0,5.0,env_right_now,1)

    #actions(::POMDP_Planner_1D_action_space) = Float64[-0.5, 0.0, 0.5, -10.0]

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_1D_action_space)),
            calculate_upper_bound_value_pomdp_planning_1D_action_space, check_terminal=true),K=50,D=100,T_max=0.3, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_1D_action_space_pomdp);
    #m = golfcart_1D_action_space_pomdp()

    astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs,
        astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_number_risks,
        astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
        astar_1D_cart_reached_goal_flag = run_one_simulation_1D_POMDP_planner(env_right_now, rand_noise_generator,
                                                                                    golfcart_1D_action_space_pomdp, planner)

    # anim = @animate for i ∈ 1:length(just_2D_pomdp_all_observed_environments)
    #     display_env(just_2D_pomdp_all_observed_environments[i]);
    #     savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*string(i)*".png")
    # end
    #
    # gif_name = "./scenario_1_gifs/just_2D_action_space_pomdp_planner_run_"*string(iteration_num)*"_"*string(just_2D_pomdp_cart_reached_goal_flag)
    # gif_name = gif_name*"_"*string(just_2D_pomdp_time_taken_by_cart)*"_"*string(just_2D_pomdp_number_risks)*".gif"
    # gif(anim, gif_name, fps = 2)
    return astar_1D_number_risks,astar_1D_number_of_sudden_stops,astar_1D_time_taken_by_cart, astar_1D_cart_reached_goal_flag
end

function run_experiment_pipeline(num_humans, num_simulations)

    total_time_taken_2D_POMDP_planner = 0.0
    total_safe_paths_2D_POMDP_planner = 0
    total_sudden_stops_2D_POMDP_planner = 0
    total_time_taken_1D_POMDP_planner = 0.0
    total_safe_paths_1D_POMDP_planner = 0
    total_sudden_stops_1D_POMDP_planner = 0
    rand_rng = MersenneTwister(100)

    for iteration_num in 1:num_simulations
	println("\n Running Simulation #", string(iteration_num), "\n")
        rand_noise_generator_seed = Int(ceil(100*rand(rand_rng)))
        rand_noise_generator = MersenneTwister(rand_noise_generator_seed)
        experiment_env = generate_environment_no_obstacles(num_humans, rand_noise_generator)
        just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,just_2D_pomdp_time_taken_by_cart,
            just_2D_pomdp_cart_reached_goal_flag = run_experiment_for_given_world_and_noise_with_2D_POMDP_planner(experiment_env, MersenneTwister(7), iteration_num)
        if(just_2D_pomdp_cart_reached_goal_flag && just_2D_pomdp_number_risks==0)
            total_time_taken_2D_POMDP_planner += just_2D_pomdp_time_taken_by_cart
            total_safe_paths_2D_POMDP_planner += 1
            total_sudden_stops_2D_POMDP_planner += just_2D_pomdp_number_of_sudden_stops
        end
        astar_1D_number_risks,astar_1D_number_of_sudden_stops,astar_1D_time_taken_by_cart,
            astar_1D_cart_reached_goal_flag = run_experiment_for_given_world_and_noise_with_1D_POMDP_planner(experiment_env, MersenneTwister(7), iteration_num)
        if(astar_1D_cart_reached_goal_flag && astar_1D_number_risks==0)
            total_time_taken_1D_POMDP_planner += astar_1D_time_taken_by_cart
            total_safe_paths_1D_POMDP_planner += 1
            total_sudden_stops_1D_POMDP_planner += astar_1D_number_of_sudden_stops
        end
    end

    average_time_taken_2D_POMDP_planner = total_time_taken_2D_POMDP_planner/total_safe_paths_2D_POMDP_planner
    average_sudden_stops_2D_POMDP_planner = total_sudden_stops_2D_POMDP_planner/total_safe_paths_2D_POMDP_planner
    average_time_taken_1D_POMDP_planner = total_time_taken_1D_POMDP_planner/total_safe_paths_1D_POMDP_planner
    average_sudden_stops_1D_POMDP_planner = total_sudden_stops_1D_POMDP_planner/total_safe_paths_1D_POMDP_planner

    println("\n\n")
    println("For 2D action space POMDP planner")
    println("   Number of safe trajectories executed - ", string(total_safe_paths_2D_POMDP_planner),
                                            " (out of ", string(num_simulations), " )" )
    println("   Average time taken to reach the goal - ", string(average_time_taken_2D_POMDP_planner), " seconds")
    println("   Average number of sudden stop action executed - ", string(average_sudden_stops_2D_POMDP_planner))

    println("\n\n")
    println("For Hybrid A* + 1D action space POMDP planner")
    println("   Number of safe trajectories executed - ", string(total_safe_paths_1D_POMDP_planner),
                                            " (out of ", string(num_simulations), " )" )
    println("   Average time taken to reach the goal - ", string(average_time_taken_1D_POMDP_planner), " seconds")
    println("   Average number of sudden stop action executed - ", string(average_sudden_stops_1D_POMDP_planner))


    return average_time_taken_2D_POMDP_planner, total_safe_paths_2D_POMDP_planner, average_sudden_stops_2D_POMDP_planner,
            average_time_taken_1D_POMDP_planner, total_safe_paths_1D_POMDP_planner, average_sudden_stops_1D_POMDP_planner
end

average_time_taken_2D_POMDP_planner, total_safe_paths_2D_POMDP_planner, average_sudden_stops_2D_POMDP_planner,
average_time_taken_1D_POMDP_planner, total_safe_paths_1D_POMDP_planner, average_sudden_stops_1D_POMDP_planner = run_experiment_pipeline(200,100)

#=
average_time_taken_2D_POMDP_planner, total_safe_paths_2D_POMDP_planner, average_sudden_stops_2D_POMDP_planner,
average_time_taken_1D_POMDP_planner, total_safe_paths_1D_POMDP_planner, average_sudden_stops_1D_POMDP_planner = run_experiment_pipeline(300,20)
=#
