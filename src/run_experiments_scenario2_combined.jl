include("new_main_2d_action_space_pomdp.jl")
include("new_main_2d_action_space_pomdp_for_prm.jl")
include("main_hybrid_1d_pomdp_path_planner_reuse_old_path.jl")
using FileIO
using JLD2

discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
actions(m::POMDP_Planner_2D_action_space,b) = get_actions_holonomic_fmm(m,b)

function run_experiment_for_given_world_and_noise_with_2D_POMDP_planner_fmm(world, gradient_info_matrix, rand_noise_generator_for_sim, iteration_num, output_filename, create_gif_flag=false)

    #Create POMDP for env_right_now
    env_right_now = deepcopy(world)

    golfcart_2D_action_space_pomdp = POMDP_Planner_2D_action_space(0.97,1.0,-100.0,2.0,-100.0,0.0,1.0,1000.0,2.0,env_right_now,gradient_info_matrix)
	solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(b->calculate_lower_bound_policy_pomdp_planning_2D_action_space(golfcart_2D_action_space_pomdp, b)),
                            max_depth=100),calculate_upper_bound_value_pomdp_planning_2D_action_space, check_terminal=true),K=50,D=100,T_max=0.5,max_trials=50, tree_in_info=true)

	io = open(output_filename,"a")
	write_and_print( io, "RNG seed for Solver -> " * string(solver.rng.seed[1]) * "\n")
    close(io)

    planner = POMDPs.solve(solver, golfcart_2D_action_space_pomdp);

	just_2D_pomdp_all_gif_environments, just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data,
    just_2D_pomdp_all_generated_beliefs, just_2D_pomdp_all_generated_trees, just_2D_pomdp_all_risky_scenarios, just_2D_pomdp_all_actions,
    just_2D_pomdp_all_planners,just_2D_pomdp_cart_throughout_path, just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,
    just_2D_pomdp_time_taken_by_cart,just_2D_pomdp_cart_reached_goal_flag, just_2D_pomdp_cart_ran_into_static_obstacle_flag,
    just_2D_pomdp_cart_ran_into_boundary_wall_flag,just_2D_pomdp_experiment_success_flag = run_one_simulation_2D_POMDP_planner(env_right_now, rand_noise_generator_for_sim,
                                                                                 		golfcart_2D_action_space_pomdp, planner, output_filename)

	#=
	if( create_gif_flag )
	    anim = @animate for i ∈ 1:length(just_2D_pomdp_all_observed_environments)
	        display_env(just_2D_pomdp_all_observed_environments[i]);
	        savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*string(i)*".png")
	    end
	    gif_name = "./scenario_1_gifs/just_2D_action_space_pomdp_planner_run_"*string(iteration_num)*"_"*string(just_2D_pomdp_cart_reached_goal_flag)
	    gif_name = gif_name*"_"*string(just_2D_pomdp_time_taken_by_cart)*"_"*string(just_2D_pomdp_number_risks)*".gif"
	    gif(anim, gif_name, fps = 2)
	end
	=#
    return just_2D_pomdp_all_gif_environments, just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data,
    just_2D_pomdp_all_generated_beliefs, just_2D_pomdp_all_generated_trees, just_2D_pomdp_all_risky_scenarios, just_2D_pomdp_all_actions,
    just_2D_pomdp_all_planners,just_2D_pomdp_cart_throughout_path, just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,
    just_2D_pomdp_time_taken_by_cart,just_2D_pomdp_cart_reached_goal_flag, just_2D_pomdp_cart_ran_into_static_obstacle_flag,
    just_2D_pomdp_cart_ran_into_boundary_wall_flag,just_2D_pomdp_experiment_success_flag,solver.rng.seed[1]
end

discount(p::POMDP_Planner_2D_action_space_prm) = p.discount_factor
isterminal(::POMDP_Planner_2D_action_space_prm, s::POMDP_state_2D_action_space_prm) = is_terminal_state_pomdp_planning_prm(s,location(-100.0,-100.0));
actions(m::POMDP_Planner_2D_action_space_prm,b) = get_actions_holonomic_prm(m,b)

function run_experiment_for_given_world_and_noise_with_2D_POMDP_planner_prm(world, prm_details, lookup_table, rand_noise_generator_for_sim, iteration_num, output_filename, create_gif_flag=false)

    #Create POMDP for env_right_now
    env_right_now = deepcopy(world)

	golfcart_2D_action_space_pomdp_prm = POMDP_Planner_2D_action_space_prm(0.97,1.0,-100.0,2.0,-100.0,0.0,1.0,1000.0,2.0,env_right_now,prm_details,lookup_table,-1,-1,false)
	solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(b->calculate_lower_bound_policy_pomdp_planning_2D_action_space_prm(golfcart_2D_action_space_pomdp_prm, b)),
                            max_depth=100),calculate_upper_bound_value_pomdp_planning_2D_action_space_prm, check_terminal=true),K=50,D=100,T_max=0.5, tree_in_info=true)

	io = open(output_filename,"a")
	write_and_print( io, "RNG seed for Solver -> " * string(solver.rng.seed[1]) * "\n")
    close(io)

    planner = POMDPs.solve(solver, golfcart_2D_action_space_pomdp_prm);

	just_2D_pomdp_all_gif_environments, just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data,
    just_2D_pomdp_all_generated_beliefs, just_2D_pomdp_all_generated_trees, just_2D_pomdp_all_risky_scenarios, just_2D_pomdp_all_actions,
    just_2D_pomdp_all_planners,just_2D_pomdp_cart_throughout_path, just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,
    just_2D_pomdp_time_taken_by_cart,just_2D_pomdp_cart_reached_goal_flag, just_2D_pomdp_cart_ran_into_static_obstacle_flag,
    just_2D_pomdp_cart_ran_into_boundary_wall_flag,just_2D_pomdp_experiment_success_flag = run_one_simulation_2D_POMDP_planner_prm(env_right_now, rand_noise_generator_for_sim,
                                                                                 		golfcart_2D_action_space_pomdp_prm, planner, output_filename)

	#=
	if( create_gif_flag )
	    anim = @animate for i ∈ 1:length(just_2D_pomdp_all_observed_environments)
	        display_env(just_2D_pomdp_all_observed_environments[i]);
	        savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*string(i)*".png")
	    end
	    gif_name = "./scenario_1_gifs/just_2D_action_space_pomdp_planner_run_"*string(iteration_num)*"_"*string(just_2D_pomdp_cart_reached_goal_flag)
	    gif_name = gif_name*"_"*string(just_2D_pomdp_time_taken_by_cart)*"_"*string(just_2D_pomdp_number_risks)*".gif"
	    gif(anim, gif_name, fps = 2)
	end
	=#
    return just_2D_pomdp_all_gif_environments, just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data,
    just_2D_pomdp_all_generated_beliefs, just_2D_pomdp_all_generated_trees, just_2D_pomdp_all_risky_scenarios, just_2D_pomdp_all_actions,
    just_2D_pomdp_all_planners,just_2D_pomdp_cart_throughout_path, just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,
    just_2D_pomdp_time_taken_by_cart,just_2D_pomdp_cart_reached_goal_flag, just_2D_pomdp_cart_ran_into_static_obstacle_flag,
    just_2D_pomdp_cart_ran_into_boundary_wall_flag,just_2D_pomdp_experiment_success_flag,solver.rng.seed[1]
end

discount(p::POMDP_Planner_1D_action_space) = p.discount_factor
isterminal(::POMDP_Planner_1D_action_space, s::POMDP_state_1D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
actions(::POMDP_Planner_1D_action_space) = Float64[-1.0, 0.0, 1.0, -10.0]

function run_experiment_for_given_world_and_noise_with_1D_POMDP_planner(world, rand_noise_generator_for_sim, iteration_num, output_filename, create_gif_flag=false)

    #Create POMDP for env_right_now
    env_right_now = deepcopy(world)

	golfcart_1D_action_space_pomdp = POMDP_Planner_1D_action_space(0.97,1.0,-100.0,1.0,1.0,1000.0,2.0,env_right_now,1)
	solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_1D_action_space)),
            calculate_upper_bound_value_pomdp_planning_1D_action_space, check_terminal=true),K=50,D=100,T_max=0.3, tree_in_info=true)

	io = open(output_filename,"a")
	write_and_print( io, "RNG seed for Solver -> " * string(solver.rng.seed[1]) * "\n")
    close(io)

    planner = POMDPs.solve(solver, golfcart_1D_action_space_pomdp);
    #m = golfcart_1D_action_space_pomdp()

	astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs_using_complete_lidar_data,
    astar_1D_all_generated_beliefs,astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_all_actions, astar_1D_all_planners,
    astar_1D_cart_throughout_path, astar_1D_number_risks, astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
    astar_1D_cart_reached_goal_flag, astar_1D_cart_ran_into_static_obstacle_flag, astar_1D_cart_ran_into_boundary_wall_flag,
    astar_1D_experiment_success_flag = run_one_simulation_1D_POMDP_planner(env_right_now, rand_noise_generator_for_sim,
                                                                                golfcart_1D_action_space_pomdp, planner, output_filename)

	#=
	if(create_gif_flag)
	    anim = @animate for i ∈ 1:length(just_2D_pomdp_all_observed_environments)
	        display_env(just_2D_pomdp_all_observed_environments[i]);
	        savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*string(i)*".png")
	    end
	    gif_name = "./scenario_1_gifs/just_2D_action_space_pomdp_planner_run_"*string(iteration_num)*"_"*string(just_2D_pomdp_cart_reached_goal_flag)
	    gif_name = gif_name*"_"*string(just_2D_pomdp_time_taken_by_cart)*"_"*string(just_2D_pomdp_number_risks)*".gif"
	    gif(anim, gif_name, fps = 2)
	end
	=#
    return astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs_using_complete_lidar_data,
    astar_1D_all_generated_beliefs,astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_all_actions, astar_1D_all_planners,
    astar_1D_cart_throughout_path, astar_1D_number_risks, astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
    astar_1D_cart_reached_goal_flag, astar_1D_cart_ran_into_static_obstacle_flag, astar_1D_cart_ran_into_boundary_wall_flag,
    astar_1D_experiment_success_flag, solver.rng.seed[1]
end

function run_experiment_pipeline(num_humans, num_simulations, write_to_file_flag = false)

	total_time_taken_2D_POMDP_planner_dict_fmm = OrderedDict()
	total_time_taken_2D_POMDP_planner_fmm = 0.0
	total_safe_paths_2D_POMDP_planner_fmm = 0
	num_times_cart_reached_goal_2D_POMDP_planner_fmm = 0
	total_sudden_stops_2D_POMDP_planner_dict_fmm = OrderedDict()
	total_sudden_stops_2D_POMDP_planner_fmm = 0

	total_time_taken_2D_POMDP_planner_dict_prm = OrderedDict()
	total_time_taken_2D_POMDP_planner_prm = 0.0
	total_safe_paths_2D_POMDP_planner_prm = 0
	num_times_cart_reached_goal_2D_POMDP_planner_prm = 0
	total_sudden_stops_2D_POMDP_planner_dict_prm = OrderedDict()
	total_sudden_stops_2D_POMDP_planner_prm = 0

	total_time_taken_1D_POMDP_planner_dict = OrderedDict()
	total_time_taken_1D_POMDP_planner = 0.0
	total_safe_paths_1D_POMDP_planner = 0
	num_times_cart_reached_goal_1D_POMDP_planner = 0
	total_sudden_stops_1D_POMDP_planner_dict = OrderedDict()
	total_sudden_stops_1D_POMDP_planner = 0

	planning_time_improvement_dict_fmm = OrderedDict()
	sudden_stops_increment_dict_fmm = OrderedDict()
	planning_time_improvement_dict_prm = OrderedDict()
	sudden_stops_increment_dict_prm = OrderedDict()

	cart_path_2D_POMDP_planner_dict_fmm = OrderedDict()
	cart_path_2D_POMDP_planner_dict_prm = OrderedDict()
	cart_path_1D_POMDP_planner_dict = OrderedDict()

	first_simulation_flag = true
	gradient_info_matrix = nothing
	lookup_table = nothing
	prm_details = nothing

    for iteration_num in 1:num_simulations

		#Set seed for different RNGs
		rand_noise_generator_seed_for_env = rand(UInt32)
	    rand_noise_generator_seed_for_sim = rand(UInt32)
		rand_noise_generator_seed_for_prm = 11
	    rand_noise_generator_for_env = MersenneTwister(rand_noise_generator_seed_for_env)
	    rand_noise_generator_for_sim = MersenneTwister(rand_noise_generator_seed_for_sim)

		#Generate Environemnt
        experiment_env = generate_environment_small_circular_obstacles(num_humans, rand_noise_generator_for_env)

		if(first_simulation_flag)
			#Generate FMM slowness map and the gradient lookup Table
			k = generate_slowness_map_from_given_environment(experiment_env,2.0)
	    	discretization = [0.1,0.1]
	    	source = CartesianIndex(250,1000)
	    	t = solve_eikonal_equation_on_given_map(k, discretization, source)
	    	gradient_info_matrix = calculate_gradients(t)

			#Load prm_details and lookup_table
			prm_info_dict = load("prm_hash_table_scenario2.jld2")
			lookup_table = prm_info_dict["lookup_table"]
			prm_details = prm_info_dict["prm_details"]
			if(lookup_table == nothing)
				graph = generate_prm_vertices(1000, rand_noise_generator_for_prm, experiment_env)
				d = generate_prm_edges(experiment_env, graph, 10)
				prm_details = Array{prm_info_struct,1}(undef,nv(graph))
				for i in 1:nv(graph)
				   st = prm_info_struct(get_prop(graph,i,:x) , get_prop(graph,i,:y), get_prop(graph,i,:dist_to_goal), get_prop(graph,i,:path_to_goal))
				   prm_details[i] = st
				end
				lookup_table = generate_prm_points_coordinates_lookup_table_holonomic_using_x_y_theta(experiment_env,graph)
			end
			first_simulation_flag = false
		end

		#Run experiment for 2D action space POMDP planner with FMM
		output_filename_2D_AS_planner_fmm = "./scenario_2/humans_"*string(num_humans)*"/2D/output_expt_fmm_" * string(iteration_num) * ".txt"
		io = open(output_filename_2D_AS_planner_fmm,"w")
		write_and_print( io, "\n Running Simulation #" * string(iteration_num))
	    write_and_print( io, "RNG seed for generating environemnt -> " * string(rand_noise_generator_seed_for_env))
	    write_and_print( io, "RNG seed for simulating pedestrians -> " * string(rand_noise_generator_seed_for_sim))
		close(io)

		just_2D_pomdp_all_gif_environments_fmm, just_2D_pomdp_all_observed_environments_fmm, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data_fmm,
	    just_2D_pomdp_all_generated_beliefs_fmm, just_2D_pomdp_all_generated_trees_fmm, just_2D_pomdp_all_risky_scenarios_fmm, just_2D_pomdp_all_actions_fmm,
	    just_2D_pomdp_all_planners_fmm,just_2D_pomdp_cart_throughout_path_fmm, just_2D_pomdp_number_risks_fmm,just_2D_pomdp_number_of_sudden_stops_fmm,
	    just_2D_pomdp_time_taken_by_cart_fmm,just_2D_pomdp_cart_reached_goal_flag_fmm, just_2D_pomdp_cart_ran_into_static_obstacle_flag_fmm,
	    just_2D_pomdp_cart_ran_into_boundary_wall_flag_fmm,just_2D_pomdp_experiment_success_flag_fmm,
		just_2D_pomdp_solver_rng_fmm = run_experiment_for_given_world_and_noise_with_2D_POMDP_planner_fmm(experiment_env, gradient_info_matrix,
													rand_noise_generator_for_sim, iteration_num, output_filename_2D_AS_planner_fmm)

		if(write_to_file_flag)
			expt_details_filename_2D_AS_planner_fmm = "./scenario_2/humans_"*string(num_humans)*"/2D/details_expt_fmm_" * string(iteration_num) * ".jld2"
	        write_experiment_details_to_file(rand_noise_generator_seed_for_env,rand_noise_generator_seed_for_sim,
	                just_2D_pomdp_solver_rng_fmm,just_2D_pomdp_all_gif_environments_fmm, just_2D_pomdp_all_observed_environments_fmm,
	                just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data_fmm,just_2D_pomdp_all_generated_beliefs_fmm, just_2D_pomdp_all_generated_trees_fmm,
	                just_2D_pomdp_all_risky_scenarios_fmm, just_2D_pomdp_all_actions_fmm,just_2D_pomdp_all_planners_fmm,just_2D_pomdp_cart_throughout_path_fmm,
	                just_2D_pomdp_number_risks_fmm,just_2D_pomdp_number_of_sudden_stops_fmm,just_2D_pomdp_time_taken_by_cart_fmm,just_2D_pomdp_cart_reached_goal_flag_fmm,
	                just_2D_pomdp_cart_ran_into_static_obstacle_flag_fmm,just_2D_pomdp_cart_ran_into_boundary_wall_flag_fmm,just_2D_pomdp_experiment_success_flag_fmm,
	                expt_details_filename_2D_AS_planner_fmm)
	    end

		#If this experiment lead to a risky scenario, then store those scenarios for debugging.
		if(just_2D_pomdp_number_risks_fmm != 0 || just_2D_pomdp_cart_ran_into_boundary_wall_flag_fmm || just_2D_pomdp_cart_ran_into_static_obstacle_flag_fmm
										|| !just_2D_pomdp_experiment_success_flag_fmm || !just_2D_pomdp_cart_reached_goal_flag_fmm)
			risky_expt_filename_2D_AS_planner_fmm = "./scenario_2/humans_"*string(num_humans)*"/2D/risky_scenarios/expt_fmm_" * string(iteration_num) * ".jld2"
			write_experiment_details_to_file(rand_noise_generator_seed_for_env,rand_noise_generator_seed_for_sim,
	                just_2D_pomdp_solver_rng_fmm,just_2D_pomdp_all_gif_environments_fmm, just_2D_pomdp_all_observed_environments_fmm,
	                just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data_fmm,just_2D_pomdp_all_generated_beliefs_fmm, just_2D_pomdp_all_generated_trees_fmm,
	                just_2D_pomdp_all_risky_scenarios_fmm, just_2D_pomdp_all_actions_fmm,just_2D_pomdp_all_planners_fmm,just_2D_pomdp_cart_throughout_path_fmm,
	                just_2D_pomdp_number_risks_fmm,just_2D_pomdp_number_of_sudden_stops_fmm,just_2D_pomdp_time_taken_by_cart_fmm,just_2D_pomdp_cart_reached_goal_flag_fmm,
	                just_2D_pomdp_cart_ran_into_static_obstacle_flag_fmm,just_2D_pomdp_cart_ran_into_boundary_wall_flag_fmm,just_2D_pomdp_experiment_success_flag_fmm,
	                risky_expt_filename_2D_AS_planner_fmm)
		end

		#Find in how many experiments cart actually reached the goal without colliding into boundary wall or static obstacles
		if(just_2D_pomdp_cart_reached_goal_flag_fmm)
			num_times_cart_reached_goal_2D_POMDP_planner_fmm += 1
		end

		#Find in how many experiments cart reached the goal without encountering a risky scenario.
		if(just_2D_pomdp_cart_reached_goal_flag_fmm && just_2D_pomdp_number_risks_fmm==0)
			total_time_taken_2D_POMDP_planner_dict_fmm[iteration_num] = just_2D_pomdp_time_taken_by_cart_fmm
			total_time_taken_2D_POMDP_planner_fmm += just_2D_pomdp_time_taken_by_cart_fmm
			total_safe_paths_2D_POMDP_planner_fmm += 1
			total_sudden_stops_2D_POMDP_planner_dict_fmm[iteration_num] = just_2D_pomdp_number_of_sudden_stops_fmm
			total_sudden_stops_2D_POMDP_planner_fmm += just_2D_pomdp_number_of_sudden_stops_fmm
			cart_path_2D_POMDP_planner_dict_fmm[iteration_num] = just_2D_pomdp_cart_throughout_path_fmm
		else
			total_time_taken_2D_POMDP_planner_dict_fmm[iteration_num] = nothing
			total_sudden_stops_2D_POMDP_planner_dict_fmm[iteration_num] = nothing
			cart_path_2D_POMDP_planner_dict_fmm[iteration_num] = just_2D_pomdp_cart_throughout_path_fmm
		end

		#Run experiment for 2D action space POMDP planner with PRM
		rand_noise_generator_for_sim = MersenneTwister(rand_noise_generator_seed_for_sim)
		output_filename_2D_AS_planner_prm = "./scenario_2/humans_"*string(num_humans)*"/2D/output_expt_prm_" * string(iteration_num) * ".txt"
		io = open(output_filename_2D_AS_planner_prm,"w")
		write_and_print( io, "\n Running Simulation #" * string(iteration_num))
	    write_and_print( io, "RNG seed for generating environemnt -> " * string(rand_noise_generator_seed_for_env))
	    write_and_print( io, "RNG seed for simulating pedestrians -> " * string(rand_noise_generator_seed_for_sim))
		close(io)

		just_2D_pomdp_all_gif_environments_prm, just_2D_pomdp_all_observed_environments_prm, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data_prm,
	    just_2D_pomdp_all_generated_beliefs_prm, just_2D_pomdp_all_generated_trees_prm, just_2D_pomdp_all_risky_scenarios_prm, just_2D_pomdp_all_actions_prm,
	    just_2D_pomdp_all_planners_prm,just_2D_pomdp_cart_throughout_path_prm, just_2D_pomdp_number_risks_prm,just_2D_pomdp_number_of_sudden_stops_prm,
	    just_2D_pomdp_time_taken_by_cart_prm,just_2D_pomdp_cart_reached_goal_flag_prm, just_2D_pomdp_cart_ran_into_static_obstacle_flag_prm,
	    just_2D_pomdp_cart_ran_into_boundary_wall_flag_prm,just_2D_pomdp_experiment_success_flag_prm,
		just_2D_pomdp_solver_rng_prm = run_experiment_for_given_world_and_noise_with_2D_POMDP_planner_prm(experiment_env, prm_details, lookup_table,
													rand_noise_generator_for_sim, iteration_num, output_filename_2D_AS_planner_prm)

		if(write_to_file_flag)
			expt_details_filename_2D_AS_planner_prm = "./scenario_2/humans_"*string(num_humans)*"/2D/details_expt_prm_" * string(iteration_num) * ".jld2"
	        write_experiment_details_to_file_for_prm(rand_noise_generator_seed_for_env,rand_noise_generator_seed_for_sim,rand_noise_generator_seed_for_prm,
	                just_2D_pomdp_solver_rng_prm,just_2D_pomdp_all_gif_environments_prm, just_2D_pomdp_all_observed_environments_prm,
	                just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data_prm,just_2D_pomdp_all_generated_beliefs_prm, just_2D_pomdp_all_generated_trees_prm,
	                just_2D_pomdp_all_risky_scenarios_prm, just_2D_pomdp_all_actions_prm,just_2D_pomdp_all_planners_prm,just_2D_pomdp_cart_throughout_path_prm,
	                just_2D_pomdp_number_risks_prm,just_2D_pomdp_number_of_sudden_stops_prm,just_2D_pomdp_time_taken_by_cart_prm,just_2D_pomdp_cart_reached_goal_flag_prm,
	                just_2D_pomdp_cart_ran_into_static_obstacle_flag_prm,just_2D_pomdp_cart_ran_into_boundary_wall_flag_prm,just_2D_pomdp_experiment_success_flag_prm,
	                expt_details_filename_2D_AS_planner_prm)
	    end

		#If this experiment lead to a risky scenario, then store those scenarios for debugging.
		if(just_2D_pomdp_number_risks_prm != 0 || just_2D_pomdp_cart_ran_into_boundary_wall_flag_prm || just_2D_pomdp_cart_ran_into_static_obstacle_flag_prm
										|| !just_2D_pomdp_experiment_success_flag_prm || !just_2D_pomdp_cart_reached_goal_flag_prm)
			risky_expt_filename_2D_AS_planner_prm = "./scenario_2/humans_"*string(num_humans)*"/2D/risky_scenarios/expt_prm_" * string(iteration_num) * ".jld2"
			write_experiment_details_to_file_for_prm(rand_noise_generator_seed_for_env,rand_noise_generator_seed_for_sim,rand_noise_generator_seed_for_prm,
	                just_2D_pomdp_solver_rng_prm,just_2D_pomdp_all_gif_environments_prm, just_2D_pomdp_all_observed_environments_prm,
	                just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data_prm,just_2D_pomdp_all_generated_beliefs_prm, just_2D_pomdp_all_generated_trees_prm,
	                just_2D_pomdp_all_risky_scenarios_prm, just_2D_pomdp_all_actions_prm,just_2D_pomdp_all_planners_prm,just_2D_pomdp_cart_throughout_path_prm,
	                just_2D_pomdp_number_risks_prm,just_2D_pomdp_number_of_sudden_stops_prm,just_2D_pomdp_time_taken_by_cart_prm,just_2D_pomdp_cart_reached_goal_flag_prm,
	                just_2D_pomdp_cart_ran_into_static_obstacle_flag_prm,just_2D_pomdp_cart_ran_into_boundary_wall_flag_prm,just_2D_pomdp_experiment_success_flag_prm,
	                risky_expt_filename_2D_AS_planner_prm)
		end

		#Find in how many experiments cart actually reached the goal without colliding into boundary wall or static obstacles
		if(just_2D_pomdp_cart_reached_goal_flag_prm)
			num_times_cart_reached_goal_2D_POMDP_planner_prm += 1
		end

		#Find in how many experiments cart reached the goal without encountering a risky scenario.
		if(just_2D_pomdp_cart_reached_goal_flag_prm && just_2D_pomdp_number_risks_prm==0)
			total_time_taken_2D_POMDP_planner_dict_prm[iteration_num] = just_2D_pomdp_time_taken_by_cart_prm
			total_time_taken_2D_POMDP_planner_prm += just_2D_pomdp_time_taken_by_cart_prm
			total_safe_paths_2D_POMDP_planner_prm += 1
			total_sudden_stops_2D_POMDP_planner_dict_prm[iteration_num] = just_2D_pomdp_number_of_sudden_stops_prm
			total_sudden_stops_2D_POMDP_planner_prm += just_2D_pomdp_number_of_sudden_stops_prm
			cart_path_2D_POMDP_planner_dict_prm[iteration_num] = just_2D_pomdp_cart_throughout_path_prm
		else
			total_time_taken_2D_POMDP_planner_dict_prm[iteration_num] = nothing
			total_sudden_stops_2D_POMDP_planner_dict_prm[iteration_num] = nothing
			cart_path_2D_POMDP_planner_dict_prm[iteration_num] = just_2D_pomdp_cart_throughout_path_prm
		end

		#Run experiment for 1D action space POMDP planner
		rand_noise_generator_for_sim = MersenneTwister(rand_noise_generator_seed_for_sim)
		output_filename_1D_AS_planner = "./scenario_2/humans_"*string(num_humans)*"/1D/output_expt_" * string(iteration_num) * ".txt"
		io = open(output_filename_1D_AS_planner,"w")
		write_and_print( io, "\n Running Simulation #" * string(iteration_num))
	    write_and_print( io, "RNG seed for generating environemnt -> " * string(rand_noise_generator_seed_for_env))
	    write_and_print( io, "RNG seed for simulating pedestrians -> " * string(rand_noise_generator_seed_for_sim))
		close(io)

		astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs_using_complete_lidar_data,
	    astar_1D_all_generated_beliefs,astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_all_actions, astar_1D_all_planners,
	    astar_1D_cart_throughout_path, astar_1D_number_risks, astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
	    astar_1D_cart_reached_goal_flag, astar_1D_cart_ran_into_static_obstacle_flag, astar_1D_cart_ran_into_boundary_wall_flag,
	    astar_1D_experiment_success_flag, astar_1D_solver_rng = run_experiment_for_given_world_and_noise_with_1D_POMDP_planner(experiment_env,
												rand_noise_generator_for_sim, iteration_num, output_filename_1D_AS_planner)

		if(write_to_file_flag)
			expt_details_filename_1D_AS_planner = "./scenario_2/humans_"*string(num_humans)*"/1D/details_expt_" * string(iteration_num) * ".jld2"
			write_experiment_details_to_file(rand_noise_generator_seed_for_env,rand_noise_generator_seed_for_sim,
					astar_1D_solver_rng,astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs_using_complete_lidar_data,
					astar_1D_all_generated_beliefs,astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_all_actions, astar_1D_all_planners,
					astar_1D_cart_throughout_path, astar_1D_number_risks, astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
					astar_1D_cart_reached_goal_flag, astar_1D_cart_ran_into_static_obstacle_flag, astar_1D_cart_ran_into_boundary_wall_flag,
					astar_1D_experiment_success_flag,expt_details_filename_1D_AS_planner)
		end

		#If this experiment lead to a risky scenario, then store those scenarios for debugging.
		if(astar_1D_number_risks != 0 || astar_1D_cart_ran_into_boundary_wall_flag || astar_1D_cart_ran_into_static_obstacle_flag
								|| !astar_1D_experiment_success_flag || !astar_1D_cart_reached_goal_flag)
			risky_expt_filename_1D_AS_planner = "./scenario_2/humans_"*string(num_humans)*"/1D/risky_scenarios/expt_" * string(iteration_num) * ".jld2"
			write_experiment_details_to_file(rand_noise_generator_seed_for_env,rand_noise_generator_seed_for_sim,
					astar_1D_solver_rng,astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs_using_complete_lidar_data,
					astar_1D_all_generated_beliefs,astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_all_actions, astar_1D_all_planners,
					astar_1D_cart_throughout_path, astar_1D_number_risks, astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
					astar_1D_cart_reached_goal_flag, astar_1D_cart_ran_into_static_obstacle_flag, astar_1D_cart_ran_into_boundary_wall_flag,
					astar_1D_experiment_success_flag,risky_expt_filename_1D_AS_planner)
		end

		#Find in how many experiments cart actually reached the goal without colliding into boundary wall or static obstacles
		if(astar_1D_cart_reached_goal_flag)
			num_times_cart_reached_goal_1D_POMDP_planner += 1
		end

		#Find in how many experiments cart reached the goal without encountering a risky scenario.
		if(astar_1D_cart_reached_goal_flag && astar_1D_number_risks==0)
			total_time_taken_1D_POMDP_planner_dict[iteration_num] = astar_1D_time_taken_by_cart
			total_time_taken_1D_POMDP_planner += astar_1D_time_taken_by_cart
			total_safe_paths_1D_POMDP_planner += 1
			total_sudden_stops_1D_POMDP_planner_dict[iteration_num] = astar_1D_number_of_sudden_stops
			total_sudden_stops_1D_POMDP_planner += astar_1D_number_of_sudden_stops
			cart_path_1D_POMDP_planner_dict[iteration_num] = astar_1D_cart_throughout_path
		else
			total_time_taken_1D_POMDP_planner_dict[iteration_num] = nothing
			total_sudden_stops_1D_POMDP_planner_dict[iteration_num] = nothing
			cart_path_1D_POMDP_planner_dict[iteration_num] = astar_1D_cart_throughout_path
		end

		#Find how much was the time performance improvement and how much was the increment in sudden stop action for FMM
		if( total_time_taken_2D_POMDP_planner_dict_fmm[iteration_num] != nothing && total_time_taken_1D_POMDP_planner_dict[iteration_num] != nothing)
			planning_time_improvement_dict_fmm[iteration_num] = total_time_taken_1D_POMDP_planner_dict[iteration_num] - total_time_taken_2D_POMDP_planner_dict_fmm[iteration_num]
			sudden_stops_increment_dict_fmm[iteration_num] = total_sudden_stops_2D_POMDP_planner_dict_fmm[iteration_num] - total_sudden_stops_1D_POMDP_planner_dict[iteration_num]
		else
			planning_time_improvement_dict_fmm[iteration_num] = nothing
			sudden_stops_increment_dict_fmm[iteration_num] = nothing
		end

		#Find how much was the time performance improvement and how much was the increment in sudden stop action for PRM
		if( total_time_taken_2D_POMDP_planner_dict_prm[iteration_num] != nothing && total_time_taken_1D_POMDP_planner_dict[iteration_num] != nothing)
			planning_time_improvement_dict_prm[iteration_num] = total_time_taken_1D_POMDP_planner_dict[iteration_num] - total_time_taken_2D_POMDP_planner_dict_prm[iteration_num]
			sudden_stops_increment_dict_prm[iteration_num] = total_sudden_stops_2D_POMDP_planner_dict_prm[iteration_num] - total_sudden_stops_1D_POMDP_planner_dict[iteration_num]
		else
			planning_time_improvement_dict_prm[iteration_num] = nothing
			sudden_stops_increment_dict_prm[iteration_num] = nothing
		end
    end

    average_time_taken_2D_POMDP_planner_fmm = total_time_taken_2D_POMDP_planner_fmm/total_safe_paths_2D_POMDP_planner_fmm
    average_sudden_stops_2D_POMDP_planner_fmm = total_sudden_stops_2D_POMDP_planner_fmm/total_safe_paths_2D_POMDP_planner_fmm
	average_time_taken_2D_POMDP_planner_prm = total_time_taken_2D_POMDP_planner_prm/total_safe_paths_2D_POMDP_planner_prm
	average_sudden_stops_2D_POMDP_planner_prm = total_sudden_stops_2D_POMDP_planner_prm/total_safe_paths_2D_POMDP_planner_prm
    average_time_taken_1D_POMDP_planner = total_time_taken_1D_POMDP_planner/total_safe_paths_1D_POMDP_planner
    average_sudden_stops_1D_POMDP_planner = total_sudden_stops_1D_POMDP_planner/total_safe_paths_1D_POMDP_planner

	mean_time_taken_2D_POMDP_planner_fmm, variance_time_taken_2D_POMDP_planner_fmm = calculate_mean_and_variance_from_given_dict(total_time_taken_2D_POMDP_planner_dict_fmm)
	mean_sudden_stops_2D_POMDP_planner_fmm, variance_sudden_stops_2D_POMDP_planner_fmm = calculate_mean_and_variance_from_given_dict(total_sudden_stops_2D_POMDP_planner_dict_fmm)
	mean_time_taken_2D_POMDP_planner_prm, variance_time_taken_2D_POMDP_planner_prm = calculate_mean_and_variance_from_given_dict(total_time_taken_2D_POMDP_planner_dict_prm)
	mean_sudden_stops_2D_POMDP_planner_prm, variance_sudden_stops_2D_POMDP_planner_prm = calculate_mean_and_variance_from_given_dict(total_sudden_stops_2D_POMDP_planner_dict_prm)
	mean_time_taken_1D_POMDP_planner, variance_time_taken_1D_POMDP_planner = calculate_mean_and_variance_from_given_dict(total_time_taken_1D_POMDP_planner_dict)
	mean_sudden_stops_1D_POMDP_planner, variance_sudden_stops_1D_POMDP_planner = calculate_mean_and_variance_from_given_dict(total_sudden_stops_1D_POMDP_planner_dict)

	mean_planning_time_improvement_fmm, variance_planning_time_improvement_fmm = calculate_mean_and_variance_from_given_dict(planning_time_improvement_dict_fmm)
	mean_sudden_stops_increment_fmm, variance_sudden_stops_increment_fmm = calculate_mean_and_variance_from_given_dict(sudden_stops_increment_dict_fmm)
	mean_planning_time_improvement_prm, variance_planning_time_improvement_prm = calculate_mean_and_variance_from_given_dict(planning_time_improvement_dict_prm)
	mean_sudden_stops_increment_prm, variance_sudden_stops_increment_prm = calculate_mean_and_variance_from_given_dict(sudden_stops_increment_dict_prm)

	println("\n\n")
	println("For 2D action space POMDP planner with FMM")
	println("   Number of experiments in which vehicle reached its goal - ", string(num_times_cart_reached_goal_2D_POMDP_planner_fmm))
	println("   Number of safe trajectories executed - ", string(total_safe_paths_2D_POMDP_planner_fmm),
											" (out of ", string(num_times_cart_reached_goal_2D_POMDP_planner_fmm), " )" )
	println("   Average time taken to reach the goal - ", string(average_time_taken_2D_POMDP_planner_fmm), " seconds")
	println("   Standard Deviation in time taken to reach the goal - ", string(sqrt(variance_time_taken_2D_POMDP_planner_fmm)), " seconds")
	println("   Average number of times sudden stop action is executed - ", string(average_sudden_stops_2D_POMDP_planner_fmm))
	println("   Standard Deviation in the number of times sudden stop action is executed - ", string(variance_sudden_stops_2D_POMDP_planner_fmm))

	println("\n\n")
	println("For 2D action space POMDP planner with PRM")
	println("   Number of experiments in which vehicle reached its goal - ", string(num_times_cart_reached_goal_2D_POMDP_planner_prm))
	println("   Number of safe trajectories executed - ", string(total_safe_paths_2D_POMDP_planner_prm),
											" (out of ", string(num_times_cart_reached_goal_2D_POMDP_planner_prm), " )" )
	println("   Average time taken to reach the goal - ", string(average_time_taken_2D_POMDP_planner_prm), " seconds")
	println("   Standard Deviation in time taken to reach the goal - ", string(sqrt(variance_time_taken_2D_POMDP_planner_prm)), " seconds")
	println("   Average number of times sudden stop action is executed - ", string(average_sudden_stops_2D_POMDP_planner_prm))
	println("   Standard Deviation in the number of times sudden stop action is executed - ", string(variance_sudden_stops_2D_POMDP_planner_prm))

	println("\n\n")
	println("For Hybrid A* + 1D action space POMDP planner")
	println("   Number of experiments in which vehicle reached its goal - ", string(num_times_cart_reached_goal_1D_POMDP_planner))
	println("   Number of safe trajectories executed - ", string(total_safe_paths_1D_POMDP_planner),
											" (out of ", string(num_times_cart_reached_goal_1D_POMDP_planner), " )" )
	println("   Average time taken to reach the goal - ", string(average_time_taken_1D_POMDP_planner), " seconds")
	println("   Standard Deviation in time taken to reach the goal - ", string(sqrt(variance_time_taken_1D_POMDP_planner)), " seconds")
	println("   Average number of times sudden stop action is executed - ", string(average_sudden_stops_1D_POMDP_planner))
	println("   Standard Deviation in the number of times sudden stop action is executed - ", string(variance_sudden_stops_1D_POMDP_planner))

	println("\n\n")
	println("Mean planning time improvement for FMM is - ", string(mean_planning_time_improvement_fmm), " seconds")
	println("Standard Deviation in planning time improvement for FMM is - ", string(sqrt(variance_planning_time_improvement_fmm)), " seconds")

	println("\n\n")
	println("Mean planning time improvement for PRM is - ", string(mean_planning_time_improvement_prm), " seconds")
	println("Standard Deviation in planning time improvement for PRM is - ", string(sqrt(variance_planning_time_improvement_prm)), " seconds")

	println("\n\n")
	println("Mean increment in number of sudden stops for FMM is - ", string(mean_sudden_stops_increment_fmm))
	println("Standard Deviation in increment of number of sudden stops for FMM is - ", string(sqrt(variance_sudden_stops_increment_fmm)))

	println("\n\n")
	println("Mean increment in number of sudden stops for PRM is - ", string(mean_sudden_stops_increment_prm))
	println("Standard Deviation in increment of number of sudden stops for PRM is - ", string(sqrt(variance_sudden_stops_increment_prm)))

	results_dict = OrderedDict()
	results_filename = "./NEW_RESULTS/scenario2_combined_humans_"*string(num_humans)*"_experiments_"*string(num_simulations) * ".jld2"

	results_dict["total_safe_paths_2D_POMDP_planner_fmm"] = total_safe_paths_2D_POMDP_planner_fmm
	results_dict["total_time_taken_2D_POMDP_planner_fmm"] = total_time_taken_2D_POMDP_planner_fmm
	results_dict["mean_time_taken_2D_POMDP_planner_fmm"] = mean_time_taken_2D_POMDP_planner_fmm
	results_dict["variance_time_taken_2D_POMDP_planner_fmm"] = variance_time_taken_2D_POMDP_planner_fmm
	results_dict["total_time_taken_2D_POMDP_planner_dict_fmm"] = total_time_taken_2D_POMDP_planner_dict_fmm
	results_dict["total_sudden_stops_2D_POMDP_planner_fmm"] = total_sudden_stops_2D_POMDP_planner_fmm
	results_dict["mean_sudden_stops_2D_POMDP_planner_fmm"] = mean_sudden_stops_2D_POMDP_planner_fmm
	results_dict["variance_sudden_stops_2D_POMDP_planner_fmm"] = variance_sudden_stops_2D_POMDP_planner_fmm
	results_dict["total_sudden_stops_2D_POMDP_planner_dict_fmm"] = total_sudden_stops_2D_POMDP_planner_dict_fmm

	results_dict["total_safe_paths_2D_POMDP_planner_prm"] = total_safe_paths_2D_POMDP_planner_prm
	results_dict["total_time_taken_2D_POMDP_planner_prm"] = total_time_taken_2D_POMDP_planner_prm
	results_dict["mean_time_taken_2D_POMDP_planner_prm"] = mean_time_taken_2D_POMDP_planner_prm
	results_dict["variance_time_taken_2D_POMDP_planner_prm"] = variance_time_taken_2D_POMDP_planner_prm
	results_dict["total_time_taken_2D_POMDP_planner_dict_prm"] = total_time_taken_2D_POMDP_planner_dict_prm
	results_dict["total_sudden_stops_2D_POMDP_planner_prm"] = total_sudden_stops_2D_POMDP_planner_prm
	results_dict["mean_sudden_stops_2D_POMDP_planner_prm"] = mean_sudden_stops_2D_POMDP_planner_prm
	results_dict["variance_sudden_stops_2D_POMDP_planner_prm"] = variance_sudden_stops_2D_POMDP_planner_prm
	results_dict["total_sudden_stops_2D_POMDP_planner_dict_prm"] = total_sudden_stops_2D_POMDP_planner_dict_prm

	results_dict["total_safe_paths_1D_POMDP_planner"] = total_safe_paths_1D_POMDP_planner
	results_dict["total_time_taken_1D_POMDP_planner"] = total_time_taken_1D_POMDP_planner
	results_dict["mean_time_taken_1D_POMDP_planner"] = mean_time_taken_1D_POMDP_planner
	results_dict["variance_time_taken_1D_POMDP_planner"] = variance_time_taken_1D_POMDP_planner
	results_dict["total_time_taken_1D_POMDP_planner_dict"] = total_time_taken_1D_POMDP_planner_dict
	results_dict["total_sudden_stops_1D_POMDP_planner"] = total_sudden_stops_1D_POMDP_planner
	results_dict["mean_sudden_stops_1D_POMDP_planner"] = mean_sudden_stops_1D_POMDP_planner
	results_dict["variance_sudden_stops_1D_POMDP_planner"] = variance_sudden_stops_1D_POMDP_planner
	results_dict["total_sudden_stops_1D_POMDP_planner_dict"] = total_sudden_stops_1D_POMDP_planner_dict

	results_dict["mean_planning_time_improvement_fmm"] = mean_planning_time_improvement_fmm
	results_dict["variance_planning_time_improvement_fmm"] = variance_planning_time_improvement_fmm
	results_dict["mean_sudden_stops_increment_fmm"] = mean_sudden_stops_increment_fmm
	results_dict["variance_sudden_stops_increment_fmm"] = variance_sudden_stops_increment_fmm

	results_dict["mean_planning_time_improvement_prm"] = mean_planning_time_improvement_prm
	results_dict["variance_planning_time_improvement_prm"] = variance_planning_time_improvement_prm
	results_dict["mean_sudden_stops_increment_prm"] = mean_sudden_stops_increment_prm
	results_dict["variance_sudden_stops_increment_prm"] = variance_sudden_stops_increment_prm

	results_dict["cart_path_2D_POMDP_planner_dict_fmm"] = cart_path_2D_POMDP_planner_dict_fmm
	results_dict["cart_path_2D_POMDP_planner_dict_prm"] = cart_path_2D_POMDP_planner_dict_prm
	results_dict["cart_path_1D_POMDP_planner_dict"] = cart_path_1D_POMDP_planner_dict
	#=
	results_dict[""] =
	results_dict[""] =
	=#
	save(results_filename, results_dict)
	return
end

num_pedestrians = parse(Int,ARGS[1])
num_experiments = parse(Int,ARGS[2])

delete_old_txt_and_jld2_files_flag = true
if(delete_old_txt_and_jld2_files_flag == true)
	nh = num_pedestrians
	folder_location = "./scenario_1/humans_"*string(nh)*"/"
	#Delete output txt files
	#1D planner
	foreach(rm, filter(endswith(".txt"), readdir(folder_location*"1D",join=true)))
	#2D planner
	foreach(rm, filter(endswith(".txt"), readdir(folder_location*"2D",join=true)))

	#Delete risky scenarios jld2 files
	#1D planner
	foreach(rm, filter(endswith(".jld2"), readdir(folder_location*"1D",join=true)))
	#2D planner
	foreach(rm, filter(endswith(".jld2"), readdir(folder_location*"2D",join=true)))

	#Delete risky scenarios jld2 files
	#1D planner
	foreach(rm, filter(endswith(".jld2"), readdir(folder_location*"1D/risky_scenarios",join=true)))
	#2D planner
	foreach(rm, filter(endswith(".jld2"), readdir(folder_location*"2D/risky_scenarios",join=true)))
end

run_experiment_pipeline(num_pedestrians,num_experiments,false)

#=
average_time_taken_2D_POMDP_planner, total_safe_paths_2D_POMDP_planner, average_sudden_stops_2D_POMDP_planner,
average_time_taken_1D_POMDP_planner, total_safe_paths_1D_POMDP_planner, average_sudden_stops_1D_POMDP_planner = run_experiment_pipeline(300,20)
=#

#=
How to load data from saved .jld2 file?
	1)	It was saved using this command --> save(filename, dictname)
	2)	It can be loaded using this command --> h = load(filename)["data"]
=#
