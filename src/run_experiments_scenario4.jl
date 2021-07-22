include("new_main_2d_action_space_pomdp.jl")
include("main_hybrid_1d_pomdp_path_planner_reuse_old_path.jl")
using FileIO
using JLD2

discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
actions(m::POMDP_Planner_2D_action_space,b) = get_available_actions_non_holonomic(m,b)

function run_experiment_for_given_world_and_noise_with_2D_POMDP_planner(world, lookup_table, rand_noise_generator, iteration_num, filename)

    #Create POMDP for env_right_now
    env_right_now = deepcopy(world)

    golfcart_2D_action_space_pomdp = POMDP_Planner_2D_action_space(0.97,1.0,-100.0,1.0,-100.0,0.0,1.0,1000.0,2.0,env_right_now, lookup_table)

	solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(b->calculate_lower_bound_policy_pomdp_planning_2D_action_space(golfcart_2D_action_space_pomdp, b)),
                            max_depth=100),calculate_upper_bound_value_pomdp_planning_2D_action_space, check_terminal=true),K=50,D=100,T_max=0.5, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_2D_action_space_pomdp);

	just_2D_pomdp_all_gif_environments, just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data,
            just_2D_pomdp_all_generated_beliefs, just_2D_pomdp_all_generated_trees, just_2D_pomdp_all_risky_scenarios, just_2D_pomdp_all_actions,
            just_2D_pomdp_cart_throughout_path, just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,just_2D_pomdp_time_taken_by_cart,
            just_2D_pomdp_cart_reached_goal_flag, just_2D_pomdp_cart_ran_into_static_obstacle_flag,
            just_2D_pomdp_cart_ran_into_boundary_wall_flag = run_one_simulation_2D_POMDP_planner(env_right_now, rand_noise_generator,
                                                                                        golfcart_2D_action_space_pomdp, planner, filename)

	#=
    anim = @animate for i ∈ 1:length(just_2D_pomdp_all_observed_environments)
        display_env(just_2D_pomdp_all_observed_environments[i]);
        savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*string(i)*".png")
    end
    gif_name = "./scenario_1_gifs/just_2D_action_space_pomdp_planner_run_"*string(iteration_num)*"_"*string(just_2D_pomdp_cart_reached_goal_flag)
    gif_name = gif_name*"_"*string(just_2D_pomdp_time_taken_by_cart)*"_"*string(just_2D_pomdp_number_risks)*".gif"
    gif(anim, gif_name, fps = 2)
	=#
    return just_2D_pomdp_all_gif_environments, just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data,
            just_2D_pomdp_all_generated_beliefs, just_2D_pomdp_all_generated_trees, just_2D_pomdp_all_risky_scenarios, just_2D_pomdp_all_actions,
            just_2D_pomdp_cart_throughout_path, just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,just_2D_pomdp_time_taken_by_cart,
            just_2D_pomdp_cart_reached_goal_flag, just_2D_pomdp_cart_ran_into_static_obstacle_flag,
            just_2D_pomdp_cart_ran_into_boundary_wall_flag
end

discount(p::POMDP_Planner_1D_action_space) = p.discount_factor
isterminal(::POMDP_Planner_1D_action_space, s::POMDP_state_1D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
actions(::POMDP_Planner_1D_action_space) = Float64[-1.0, 0.0, 1.0]

function run_experiment_for_given_world_and_noise_with_1D_POMDP_planner(world, rand_noise_generator, iteration_num, filename)

    #Create POMDP for env_right_now
    env_right_now = deepcopy(world)

    golfcart_1D_action_space_pomdp = POMDP_Planner_1D_action_space(0.97,1.0,-100.0,1.0,1.0,1000.0,2.0,env_right_now,1)

    #actions(::POMDP_Planner_1D_action_space) = Float64[-0.5, 0.0, 0.5, -10.0]

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_1D_action_space)),
            calculate_upper_bound_value_pomdp_planning_1D_action_space, check_terminal=true),K=50,D=100,T_max=0.3, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_1D_action_space_pomdp);
    #m = golfcart_1D_action_space_pomdp()

	astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs_using_complete_lidar_data,
    astar_1D_all_generated_beliefs,astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_all_actions,
    astar_1D_cart_throughout_path, astar_1D_number_risks, astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
    astar_1D_cart_reached_goal_flag, astar_1D_cart_ran_into_static_obstacle_flag,
    astar_1D_cart_ran_into_boundary_wall_flag = run_one_simulation_1D_POMDP_planner(env_right_now, rand_noise_generator,
                                                                                    golfcart_1D_action_space_pomdp, planner, filename)

	#=
    anim = @animate for i ∈ 1:length(just_2D_pomdp_all_observed_environments)
        display_env(just_2D_pomdp_all_observed_environments[i]);
        savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*string(i)*".png")
    end
    gif_name = "./scenario_1_gifs/just_2D_action_space_pomdp_planner_run_"*string(iteration_num)*"_"*string(just_2D_pomdp_cart_reached_goal_flag)
    gif_name = gif_name*"_"*string(just_2D_pomdp_time_taken_by_cart)*"_"*string(just_2D_pomdp_number_risks)*".gif"
    gif(anim, gif_name, fps = 2)
	=#
    return astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs_using_complete_lidar_data,
    astar_1D_all_generated_beliefs,astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_all_actions,
    astar_1D_cart_throughout_path, astar_1D_number_risks, astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
    astar_1D_cart_reached_goal_flag, astar_1D_cart_ran_into_static_obstacle_flag,
    astar_1D_cart_ran_into_boundary_wall_flag
end

function run_experiment_pipeline(num_humans, num_simulations)

    total_time_taken_2D_POMDP_planner = 0.0
	total_safe_paths_2D_POMDP_planner = 0
    num_times_cart_reached_goal_2D_POMDP_planner = 0
    total_sudden_stops_2D_POMDP_planner = 0

    total_time_taken_1D_POMDP_planner = 0.0
    total_safe_paths_1D_POMDP_planner = 0
	num_times_cart_reached_goal_1D_POMDP_planner = 0
    total_sudden_stops_1D_POMDP_planner = 0
    rand_rng = MersenneTwister(100)

	graph = nothing
	lookup_table = nothing

    for iteration_num in 1:num_simulations
        rand_noise_generator_seed_for_env = Int(ceil(100*rand(rand_rng)))
        rand_noise_generator_for_env = MersenneTwister(rand_noise_generator_seed_for_env)
		rand_noise_generator_seed_for_sim = 7
        experiment_env = generate_environment_L_shaped_corridor(num_humans, rand_noise_generator_for_env)
		#Generate PRM and Lookup Table for the first time
		if(graph == nothing && lookup_table==nothing)
			graph = generate_prm_vertices(500, MersenneTwister(11), experiment_env)
	        d = generate_prm_edges(experiment_env, graph, 10)
	        lookup_table = generate_prm_points_lookup_table_non_holonomic(experiment_env,graph)
		end

		println("\n Running Simulation #", string(iteration_num), "\n")

		#Run experiment for 2D action space POMDP planner
		filename_2D_AS_planner = "./scenario_4/2D/expt_" * string(iteration_num) * ".txt"
		just_2D_pomdp_all_gif_environments, just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data,
        just_2D_pomdp_all_generated_beliefs, just_2D_pomdp_all_generated_trees, just_2D_pomdp_all_risky_scenarios, just_2D_pomdp_all_actions,
        just_2D_pomdp_cart_throughout_path, just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,just_2D_pomdp_time_taken_by_cart,
        just_2D_pomdp_cart_reached_goal_flag, just_2D_pomdp_cart_ran_into_static_obstacle_flag,
        just_2D_pomdp_cart_ran_into_boundary_wall_flag = run_experiment_for_given_world_and_noise_with_2D_POMDP_planner(experiment_env, lookup_table,
													MersenneTwister(rand_noise_generator_seed_for_sim), iteration_num, filename_2D_AS_planner)

		#If this experiment lead to a risky scenario, then store those scenarios for debugging.
		if(just_2D_pomdp_number_risks != 0)
			risk_data_dict = OrderedDict()
			risk_data_dict["rng_seed_for_env_generation"] = rand_noise_generator_seed_for_env
			risk_data_dict["rng_seed_for_simulator"] = rand_noise_generator_seed_for_sim
			risk_data_dict["risky_scenarios"] = Dict()
			for k in keys(just_2D_pomdp_all_risky_scenarios)
				time_stamp = split(k,"=")[2]
				time_stamp_in_seconds = parse(Int, split(time_stamp,"_")[1])
				time_stamp_in_tenth_of_seconds = parse(Int, split(time_stamp,"_")[2])
				#index_in_env = floor( Int(time_stamp_in_seconds) + 1 + (0.1*Int(time_stamp_in_tenth_of_seconds)) )
				#Note that index_in_env here is different from what I described in simulator's documentation
				#because if in a_b, let's say a in 5 and b is 10, then the tree that lead to this is at index 6
				#but using the definition from simulator's documentation will make it 7 (which is the tree at next time step)
				required_dict_key = "t="*string(time_stamp_in_seconds)
				if(required_dict_key ∉ keys(risk_data_dict["risky_scenarios"]))
					new_risky_scenario_dict = Dict()
					new_risky_scenario_dict["start_env"] = just_2D_pomdp_all_observed_environments[required_dict_key]
					new_risky_scenario_dict["current_belief"] = just_2D_pomdp_all_generated_beliefs[required_dict_key]
					new_risky_scenario_dict["current_belief_over_all_lidar_data"] = just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data[required_dict_key]
					new_risky_scenario_dict["tree"] = just_2D_pomdp_all_generated_trees[required_dict_key][:tree]
					new_risky_scenario_dict["action"] = just_2D_pomdp_all_actions[required_dict_key]
					temp_dict = Dict()
					temp_dict[k] = just_2D_pomdp_all_risky_scenarios[k]
					new_risky_scenario_dict["collision_gif_environments"] = temp_dict
					risk_data_dict["risky_scenarios"][required_dict_key] = new_risky_scenario_dict
				else
					risk_data_dict["risky_scenarios"][required_dict_key]["collision_gif_environments"][k] = just_2D_pomdp_all_risky_scenarios[k]
				end
			end
			risky_expt_filename_2D_AS_planner = "./scenario_4/2D/risky_scenarios/expt_" * string(iteration_num) * ".jld2"
			save(risky_expt_filename_2D_AS_planner, risk_data_dict)
		end

		#Find in how many experiments cart actually reached the goal without colliding into boundary wall or static obstacles
		if(just_2D_pomdp_cart_reached_goal_flag)
			num_times_cart_reached_goal_2D_POMDP_planner += 1
		end

		#Find in how many experiments cart reached the goal without encountering a risky scenario.
        if(just_2D_pomdp_cart_reached_goal_flag && just_2D_pomdp_number_risks==0)
            total_time_taken_2D_POMDP_planner += just_2D_pomdp_time_taken_by_cart
            total_safe_paths_2D_POMDP_planner += 1
            total_sudden_stops_2D_POMDP_planner += just_2D_pomdp_number_of_sudden_stops
        end

		#Run experiment for 1D action space POMDP planner
		filename_1D_AS_planner = "./scenario_4/1D/expt_" * string(iteration_num) * ".txt"
		astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs_using_complete_lidar_data,
	    astar_1D_all_generated_beliefs,astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_all_actions,
	    astar_1D_cart_throughout_path, astar_1D_number_risks, astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
	    astar_1D_cart_reached_goal_flag, astar_1D_cart_ran_into_static_obstacle_flag,
	    astar_1D_cart_ran_into_boundary_wall_flag = run_experiment_for_given_world_and_noise_with_1D_POMDP_planner(experiment_env,
												MersenneTwister(rand_noise_generator_seed_for_sim), iteration_num, filename_1D_AS_planner)

		#If this experiment lead to a risky scenario, then store those scenarios for debugging.
		@show(astar_1D_number_risks)
		if(astar_1D_number_risks != 0)
			println("I am in")
			risk_data_dict = Dict()
			risk_data_dict["rng_seed_for_env_generation"] = rand_noise_generator_seed_for_env
			risk_data_dict["rng_seed_for_simulator"] = rand_noise_generator_seed_for_sim
			risk_data_dict["risky_scenarios"] = Dict()
			for k in keys(astar_1D_all_risky_scenarios)
				time_stamp = split(k,"=")[2]
				time_stamp_in_seconds = parse(Int, split(time_stamp,"_")[1])
				time_stamp_in_tenth_of_seconds = parse(Int, split(time_stamp,"_")[2])
				#index_in_env = floor( Int(time_stamp_in_seconds) + 1 + (0.1*Int(time_stamp_in_tenth_of_seconds)) )
				#Note that index_in_env here is different from what I described in simulator's documentation
				#because if in a_b, let's say a in 5 and b is 10, then the tree that lead to this is at index 6
				#but using the definition from simulator's documentation will make it 7 (which is the tree at next time step)
				required_dict_key = "t="*string(time_stamp_in_seconds)
				if(required_dict_key ∉ keys(risk_data_dict["risky_scenarios"]))
					new_risky_scenario_dict = Dict()
					new_risky_scenario_dict["start_env"] = astar_1D_all_observed_environments[required_dict_key]
					new_risky_scenario_dict["current_belief"] = astar_1D_all_generated_beliefs[required_dict_key]
					new_risky_scenario_dict["current_belief_over_all_lidar_data"] = astar_1D_all_generated_beliefs_using_complete_lidar_data[required_dict_key]
					new_risky_scenario_dict["tree"] = astar_1D_all_generated_trees[required_dict_key][:tree]
					new_risky_scenario_dict["action"] = astar_1D_all_actions[required_dict_key]
					temp_dict = Dict()
					temp_dict[k] = astar_1D_all_risky_scenarios[k]
					new_risky_scenario_dict["collision_gif_environments"] = temp_dict
					risk_data_dict["risky_scenarios"][required_dict_key] = new_risky_scenario_dict
				else
					risk_data_dict["risky_scenarios"][required_dict_key]["collision_gif_environments"][k] = astar_1D_all_risky_scenarios[k]
				end
			end
			risky_expt_filename_1D_AS_planner = "./scenario_4/1D/risky_scenarios/expt_" * string(iteration_num) * ".jld2"
			save(risky_expt_filename_1D_AS_planner, risk_data_dict)
		end

		#Find in how many experiments cart actually reached the goal without colliding into boundary wall or static obstacles
		if(astar_1D_cart_reached_goal_flag)
			num_times_cart_reached_goal_1D_POMDP_planner += 1
		end

		#Find in how many experiments cart reached the goal without encountering a risky scenario.
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
	println("   Number of experiments in which vehicle reached its goal - ", string(num_times_cart_reached_goal_2D_POMDP_planner))
    println("   Number of safe trajectories executed - ", string(total_safe_paths_2D_POMDP_planner),
                                            " (out of ", string(num_times_cart_reached_goal_2D_POMDP_planner), " )" )
    println("   Average time taken to reach the goal - ", string(average_time_taken_2D_POMDP_planner), " seconds")
    println("   Average number of sudden stop action executed - ", string(average_sudden_stops_2D_POMDP_planner))

    println("\n\n")
    println("For Hybrid A* + 1D action space POMDP planner")
	println("   Number of experiments in which vehicle reached its goal - ", string(num_times_cart_reached_goal_1D_POMDP_planner))
    println("   Number of safe trajectories executed - ", string(total_safe_paths_1D_POMDP_planner),
                                            " (out of ", string(num_times_cart_reached_goal_1D_POMDP_planner), " )" )
    println("   Average time taken to reach the goal - ", string(average_time_taken_1D_POMDP_planner), " seconds")
    println("   Average number of sudden stop action executed - ", string(average_sudden_stops_1D_POMDP_planner))


    return average_time_taken_2D_POMDP_planner, total_safe_paths_2D_POMDP_planner, average_sudden_stops_2D_POMDP_planner,
            average_time_taken_1D_POMDP_planner, total_safe_paths_1D_POMDP_planner, average_sudden_stops_1D_POMDP_planner
end

average_time_taken_2D_POMDP_planner, total_safe_paths_2D_POMDP_planner, average_sudden_stops_2D_POMDP_planner,
average_time_taken_1D_POMDP_planner, total_safe_paths_1D_POMDP_planner, average_sudden_stops_1D_POMDP_planner = run_experiment_pipeline(300,100)

#=
average_time_taken_2D_POMDP_planner, total_safe_paths_2D_POMDP_planner, average_sudden_stops_2D_POMDP_planner,
average_time_taken_1D_POMDP_planner, total_safe_paths_1D_POMDP_planner, average_sudden_stops_1D_POMDP_planner = run_experiment_pipeline(300,20)
=#

#=
How to load data from saved .jld2 file?
	1)	It was saved using this command --> save(filename, dictname)
	2)	It can be loaded using this command --> h = load(filename)["data"]
=#
