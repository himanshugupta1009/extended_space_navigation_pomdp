include("environment.jl")
include("utils.jl")
include("hybrid_a_star.jl")
include("one_d_action_space_close_waypoint_pomdp.jl")
include("belief_tracker.jl")
include("simulator.jl")

Base.copy(s::cart_state) = cart_state(s.x, s.y,s.theta,s.v,s.L,s.goal)

function run_one_simulation_1D_POMDP_planner(env_right_now,user_defined_rng, m,
                        planner, filename = "output_resusing_old_hybrid_astar_path_1D_action_space_speed_pomdp_planner.txt")

    time_taken_by_cart = 0
    number_risks = 0
    one_time_step = 0.5
    lidar_range = 30
    num_humans_to_care_about_while_generating_hybrid_astar_path = 6
    num_humans_to_care_about_while_pomdp_planning = 6
    cone_half_angle::Float64 = (2/3)*pi
    number_of_sudden_stops = 0
    cart_ran_into_boundary_wall_flag = false
    cart_ran_into_static_obstacle_flag = false
    cart_reached_goal_flag = true
    experiment_success_flag = true
    cart_throughout_path = OrderedDict()
    all_gif_environments = OrderedDict()
    all_observed_environments = OrderedDict()
    all_generated_beliefs = OrderedDict()
    all_generated_beliefs_using_complete_lidar_data = OrderedDict()
    all_generated_trees = OrderedDict()
    all_risky_scenarios = OrderedDict()
    all_actions = OrderedDict()
    all_planners = OrderedDict()
    MAX_TIME_LIMIT = 300

    #Generate the initial Hybrid A* path without considering humans
    env_right_now.cart_hybrid_astar_path = @time hybrid_a_star_search(env_right_now.cart.x, env_right_now.cart.y,
                                            env_right_now.cart.theta, env_right_now.cart.goal.x, env_right_now.cart.goal.y, env_right_now,
                                            Array{Tuple{human_state,human_probability_over_goals},1}(),100.0);

    #Sense humans near cart before moving
    #Generate Initial Lidar Data and Belief for humans near cart
    env_right_now.complete_cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
    env_right_now.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(env_right_now.cart,
                                                        env_right_now.complete_cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                        m.pedestrian_distance_threshold, cone_half_angle)

    initial_belief_over_complete_cart_lidar_data = update_belief([],env_right_now.goals,[],env_right_now.complete_cart_lidar_data)
    initial_belief = get_belief_for_selected_humans_from_belief_over_complete_lidar_data(initial_belief_over_complete_cart_lidar_data,
                                                            env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)

    dict_key = "t="*string(time_taken_by_cart)
    all_gif_environments[dict_key] = deepcopy(env_right_now)
    all_observed_environments[dict_key] = deepcopy(env_right_now)
    all_generated_beliefs_using_complete_lidar_data[dict_key] = initial_belief_over_complete_cart_lidar_data
    all_generated_beliefs[dict_key] = initial_belief
    cart_throughout_path[dict_key] = copy(env_right_now.cart)
    #Try to generate the Hybrid A* path
    humans_to_avoid = get_nearest_n_pedestrians_hybrid_astar_search(env_right_now,initial_belief,
                                                        num_humans_to_care_about_while_generating_hybrid_astar_path,m.pedestrian_distance_threshold)
    hybrid_a_star_path = @time hybrid_a_star_search(env_right_now.cart.x, env_right_now.cart.y,
        env_right_now.cart.theta, env_right_now.cart.goal.x, env_right_now.cart.goal.y, env_right_now, humans_to_avoid,100.0);
    if(length(hybrid_a_star_path)!= 0)
        env_right_now.cart_hybrid_astar_path = hybrid_a_star_path
    end

    #Simulate for t=0 to t=1
    io = open(filename,"a")
    write_and_print( io, "Simulating for time interval - (" * string(time_taken_by_cart) * " , " * string(time_taken_by_cart+1) * ")" )
    write_and_print( io, "Current cart state = " * string(env_right_now.cart) )


    #Update human positions in environment for two time steps and cart's belief accordingly
    current_belief_over_complete_cart_lidar_data, risks_in_simulation = hybrid_astar_1D_pomdp_simulate_pedestrians_and_generate_gif_environments_when_cart_stationary(
                                                        env_right_now,initial_belief_over_complete_cart_lidar_data,all_gif_environments, all_risky_scenarios, time_taken_by_cart,
                                                        num_humans_to_care_about_while_pomdp_planning,cone_half_angle, lidar_range, m.pedestrian_distance_threshold,
                                                        user_defined_rng ,io )
    current_belief =  get_belief_for_selected_humans_from_belief_over_complete_lidar_data(current_belief_over_complete_cart_lidar_data,
                                                            env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)

    number_risks += risks_in_simulation
    time_taken_by_cart += 1
    dict_key = "t="*string(time_taken_by_cart)
    all_observed_environments[dict_key] = deepcopy(env_right_now)
    all_generated_beliefs_using_complete_lidar_data[dict_key] = current_belief_over_complete_cart_lidar_data
    all_generated_beliefs[dict_key] = current_belief
    cart_throughout_path[dict_key] = copy(env_right_now.cart)
    write_and_print( io, "Modified cart state = " * string(env_right_now.cart) )
    close(io)

    # try
        #Start Simulating for t>1
        while(!is_within_range(location(env_right_now.cart.x,env_right_now.cart.y), env_right_now.cart.goal, 1.0))
            io = open(filename,"a")
            cart_ran_into_boundary_wall_flag = check_if_cart_collided_with_boundary_wall(env_right_now)
            cart_ran_into_static_obstacle_flag = check_if_cart_collided_with_static_obstacles(env_right_now)

            if( !cart_ran_into_boundary_wall_flag && !cart_ran_into_static_obstacle_flag )

                write_and_print( io, "Simulating for time interval - (" * string(time_taken_by_cart) * " , " * string(time_taken_by_cart+1) * ")" )
                write_and_print( io, "Current cart state = " * string(env_right_now.cart) )

                #Try to generate the Hybrid A* path
                humans_to_avoid = get_nearest_n_pedestrians_hybrid_astar_search(env_right_now,current_belief,
                                                                    num_humans_to_care_about_while_generating_hybrid_astar_path,m.pedestrian_distance_threshold)
                hybrid_a_star_path = @time hybrid_a_star_search(env_right_now.cart.x, env_right_now.cart.y,
                    env_right_now.cart.theta, env_right_now.cart.goal.x, env_right_now.cart.goal.y, env_right_now, humans_to_avoid,100.0);

                #If couldn't generate the path and no old path exists
                if( (length(hybrid_a_star_path) == 0) && (length(env_right_now.cart_hybrid_astar_path) == 0) )
                    write_and_print( io, "**********Hybrid A Star Path Not found. No old path exists either**********" )
                    env_right_now.cart.v = 0.0
                    #That means the cart is stationary and we now just have to simulate the pedestrians.
                    current_belief_over_complete_cart_lidar_data, risks_in_simulation = hybrid_astar_1D_pomdp_simulate_pedestrians_and_generate_gif_environments_when_cart_stationary(
                                                                        env_right_now,current_belief_over_complete_cart_lidar_data,all_gif_environments, all_risky_scenarios,
                                                                        time_taken_by_cart,num_humans_to_care_about_while_pomdp_planning, cone_half_angle, lidar_range,
                                                                        m.pedestrian_distance_threshold, user_defined_rng,io )

                    current_belief =  get_belief_for_selected_humans_from_belief_over_complete_lidar_data(current_belief_over_complete_cart_lidar_data,
                                                                        env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)
                    number_risks += risks_in_simulation
                    dict_key = "t="*string(time_taken_by_cart)
                    all_generated_trees[dict_key] = nothing
                    all_actions[dict_key] = nothing
                    all_planners[dict_key] = nothing

                    write_and_print( io, "Modified cart state = " * string(env_right_now.cart) )
                    write_and_print( io, "************************************************************************" )

                else
                    #If new path was found, use it else reuse the old one
                    if(length(hybrid_a_star_path)!= 0)
                        env_right_now.cart_hybrid_astar_path = hybrid_a_star_path
                        write_and_print( io, "**********Hybrid A Star Path found**********" )
                    else
                        write_and_print( io, "**********Hybrid A Star Path Not found. Reusing old path**********" )
                    end

                    dict_key = "t="*string(time_taken_by_cart)
                    all_planners[dict_key] = deepcopy(planner)
                    b = POMDP_1D_action_space_state_distribution(m.world,current_belief,m.start_path_index)
                    a, info = action_info(planner, b)
                    check_consistency_personal_copy(io,planner.rs)
                    if(is_there_immediate_collision_with_pedestrians(m.world, m.pedestrian_distance_threshold+m.world.cart.L))
                        if(env_right_now.cart.v == 1.0)
                            a = -1.0
                        elseif(env_right_now.cart.v == m.max_cart_speed)
                            a = -10.0
                        end
                    end
                    all_generated_trees[dict_key] = deepcopy(info)
                    all_actions[dict_key] = a
                    write_and_print( io, "Action chosen by 1D action space speed POMDP planner: " * string(a) )

                    if(env_right_now.cart.v!=0 && a ==-10.0)
                        number_of_sudden_stops += 1
                    end

                    env_right_now.cart.v = clamp(env_right_now.cart.v + a, 0, m.max_cart_speed)

                    if(env_right_now.cart.v != 0.0)
                        #That means the cart is not stationary and we now have to simulate both cart and the pedestrians.
                        current_belief_over_complete_cart_lidar_data, risks_in_simulation = hybrid_astar_1D_pomdp_simulate_cart_and_pedestrians_and_generate_gif_environments_when_cart_moving(
                                                                            env_right_now,current_belief_over_complete_cart_lidar_data, all_gif_environments, all_risky_scenarios, time_taken_by_cart,
                                                                            num_humans_to_care_about_while_pomdp_planning, cone_half_angle, lidar_range, m.pedestrian_distance_threshold,
                                                                            user_defined_rng,io )

                        current_belief =  get_belief_for_selected_humans_from_belief_over_complete_lidar_data(current_belief_over_complete_cart_lidar_data,
                                                                            env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)
                        number_risks += risks_in_simulation
                    else
                        #That means the cart is stationary and we now just have to simulate the pedestrians.
                        current_belief_over_complete_cart_lidar_data, risks_in_simulation = hybrid_astar_1D_pomdp_simulate_pedestrians_and_generate_gif_environments_when_cart_stationary(
                                                                            env_right_now,current_belief_over_complete_cart_lidar_data,all_gif_environments, all_risky_scenarios,
                                                                            time_taken_by_cart, num_humans_to_care_about_while_pomdp_planning, cone_half_angle, lidar_range,
                                                                            m.pedestrian_distance_threshold, user_defined_rng,io )

                        current_belief =  get_belief_for_selected_humans_from_belief_over_complete_lidar_data(current_belief_over_complete_cart_lidar_data,
                                                                            env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)
                        number_risks += risks_in_simulation
                    end

                    write_and_print( io, "Modified cart state = " * string(env_right_now.cart) )
                    write_and_print( io, "************************************************************************" )
                end

                time_taken_by_cart += 1
                dict_key = "t="*string(time_taken_by_cart)
                all_observed_environments[dict_key] = deepcopy(env_right_now)
                all_generated_beliefs_using_complete_lidar_data[dict_key] = current_belief_over_complete_cart_lidar_data
                all_generated_beliefs[dict_key] = current_belief
                cart_throughout_path[dict_key] = copy(env_right_now.cart)

                if(time_taken_by_cart>MAX_TIME_LIMIT)
                    cart_reached_goal_flag = false
                    break
                end
            else
                if(cart_ran_into_static_obstacle_flag)
                    write_and_print( io, "Cart ran into a static obstacle in the environment")
                elseif (cart_ran_into_boundary_wall_flag)
                    write_and_print( io, "Cart ran into a boundary wall in the environment")
                end
                cart_reached_goal_flag = false
                break
            end
            close(io)
        end
    # catch e
    #     println("\n Things failed during the simulation. \n The error message is : \n ")
    #     println(e)
    #     experiment_success_flag = false
    #     return all_gif_environments, all_observed_environments, all_generated_beliefs_using_complete_lidar_data, all_generated_beliefs,
    #             all_generated_trees,all_risky_scenarios,all_actions,all_planners,cart_throughout_path, number_risks, number_of_sudden_stops,
    #             time_taken_by_cart, cart_reached_goal_flag, cart_ran_into_static_obstacle_flag, cart_ran_into_boundary_wall_flag, experiment_success_flag
    # end
    io = open(filename,"a")

    if(cart_reached_goal_flag == true)
        write_and_print( io, "Goal Reached! :D" )
        write_and_print( io, "Time Taken by cart to reach goal : " * string(time_taken_by_cart) )
    else
        if(cart_ran_into_boundary_wall_flag)
            write_and_print( io, "Cart ran into a wall :(" )
            write_and_print( io, "Time elapsed before this happened : " * string(time_taken_by_cart) )
        elseif cart_ran_into_static_obstacle_flag
            write_and_print( io, "Cart ran into a static obstacle :(" )
            write_and_print( io, "Time elapsed before this happened : " * string(time_taken_by_cart) )
        else
            write_and_print( io, "Cart ran out of time :(" )
            write_and_print( io, "Time Taken by cart when it didn't reach the goal : " * string(time_taken_by_cart) )
        end
    end
    write_and_print( io, "Number of risky scenarios encountered by the cart : " * string(number_risks) )
    write_and_print( io, "Number of sudden stops taken by the cart : " * string(number_of_sudden_stops) )
    close(io)

    return all_gif_environments, all_observed_environments, all_generated_beliefs_using_complete_lidar_data, all_generated_beliefs,
        all_generated_trees,all_risky_scenarios,all_actions,all_planners,cart_throughout_path, number_risks, number_of_sudden_stops,
        time_taken_by_cart, cart_reached_goal_flag, cart_ran_into_static_obstacle_flag, cart_ran_into_boundary_wall_flag, experiment_success_flag
end

gr()
run_simulation_flag = false
write_to_file_flag = false
create_gif_flag = true

if(run_simulation_flag)

    #Set seeds for different random number generators randomly
    # rand_noise_generator_seed_for_env = rand(UInt32)
    # rand_noise_generator_seed_for_sim = rand(UInt32)
    # rand_noise_generator_seed_for_prm = 11
    # rand_noise_generator_for_env = MersenneTwister(rand_noise_generator_seed_for_env)
    # rand_noise_generator_for_sim = MersenneTwister(rand_noise_generator_seed_for_sim)

    #Set seeds for different random number generators manually
    rand_noise_generator_seed_for_env = 4258915202
    rand_noise_generator_seed_for_sim = 946026168
    rand_noise_generator_seed_for_solver = 2162167893
    rand_noise_generator_for_env = MersenneTwister(rand_noise_generator_seed_for_env)
    rand_noise_generator_for_sim = MersenneTwister(rand_noise_generator_seed_for_sim)
    rand_noise_generator_for_solver = MersenneTwister(rand_noise_generator_seed_for_solver)

    #Initialize environment
    # env = generate_environment_no_obstacles(300, rand_noise_generator_for_env)
    # env = generate_environment_small_circular_obstacles(300, rand_noise_generator_for_env)
    # env = generate_environment_large_circular_obstacles(300, rand_noise_generator_for_env)
    env = generate_environment_L_shaped_corridor(300, rand_noise_generator_for_env)
    env_right_now = deepcopy(env)

    filename = "output_resusing_old_hybrid_astar_path_1D_action_space_speed_pomdp_planner.txt"
    io = open(filename,"w")
    write_and_print( io, "RNG seed for generating environemnt -> " * string(rand_noise_generator_seed_for_env))
    write_and_print( io, "RNG seed for simulating pedestrians -> " * string(rand_noise_generator_seed_for_sim))

    #Create POMDP for hybrid_a_star + POMDP speed planners at every time step
    golfcart_1D_action_space_pomdp = POMDP_Planner_1D_action_space(0.97,1.0,-100.0,1.0,1.0,1000.0,2.0,env_right_now,1)
    discount(p::POMDP_Planner_1D_action_space) = p.discount_factor
    isterminal(::POMDP_Planner_1D_action_space, s::POMDP_state_1D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
    actions(::POMDP_Planner_1D_action_space) = Float64[-1.0, 0.0, 1.0]
    #actions(::POMDP_Planner_1D_action_space) = Float64[-0.5, 0.0, 0.5, -10.0]

    # solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_1D_action_space)),
    #         calculate_upper_bound_value_pomdp_planning_1D_action_space, check_terminal=true),K=50,D=100,T_max=Inf, max_trials=50, tree_in_info=true)
    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_1D_action_space)),
            calculate_upper_bound_value_pomdp_planning_1D_action_space, check_terminal=true),K=50,D=100,T_max=Inf, max_trials=50, tree_in_info=true,
            rng=rand_noise_generator_for_solver)

    write_and_print( io, "RNG seed for Solver -> " * string(solver.rng.seed[1]) * "\n")
    close(io)

    planner = POMDPs.solve(solver, golfcart_1D_action_space_pomdp);
    #m = golfcart_1D_action_space_pomdp()

    astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs_using_complete_lidar_data,
    astar_1D_all_generated_beliefs,astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_all_actions, astar_1D_all_planners,
    astar_1D_cart_throughout_path, astar_1D_number_risks, astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
    astar_1D_cart_reached_goal_flag, astar_1D_cart_ran_into_static_obstacle_flag, astar_1D_cart_ran_into_boundary_wall_flag,
    astar_1D_experiment_success_flag = run_one_simulation_1D_POMDP_planner(env_right_now, rand_noise_generator_for_sim,
                                                                            golfcart_1D_action_space_pomdp, planner)
    if(create_gif_flag)
        anim = @animate for k ∈ keys(astar_1D_all_observed_environments)
            display_env(astar_1D_all_observed_environments[k],k);
            #savefig("./plots_reusing_hybrid_astar_path_1d_action_space_speed_pomdp_planner/plot_"*string(i)*".png")
        end
        gif(anim, "resusing_old_hybrid_astar_path_1D_action_space_speed_pomdp_planner_run.gif", fps = 2)
    end

    if(write_to_file_flag)
        expt_file_name = "expt_details_resusing_old_hybrid_astar_path_1D_action_space_speed_pomdp_planner.jld2"
        write_experiment_details_to_file(rand_noise_generator_seed_for_env,rand_noise_generator_seed_for_sim,
                solver.rng.seed[1],astar_1D_all_gif_environments, astar_1D_all_observed_environments, astar_1D_all_generated_beliefs_using_complete_lidar_data,
                astar_1D_all_generated_beliefs,astar_1D_all_generated_trees, astar_1D_all_risky_scenarios, astar_1D_all_actions, astar_1D_all_planners,
                astar_1D_cart_throughout_path, astar_1D_number_risks, astar_1D_number_of_sudden_stops, astar_1D_time_taken_by_cart,
                astar_1D_cart_reached_goal_flag, astar_1D_cart_ran_into_static_obstacle_flag, astar_1D_cart_ran_into_boundary_wall_flag,
                astar_1D_experiment_success_flag,expt_file_name)
    end

end

#=
anim = @animate for k ∈ keys(astar_1D_all_gif_environments)
    display_env(astar_1D_all_gif_environments[k], k, split(k,"=")[2]);
    #println(astar_1D_all_gif_environments[i][1])
    #savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*all_gif_environments[i][1]*".png")
end
gif(anim, "resusing_old_hybrid_astar_path_1D_action_space_speed_pomdp_planner_run.gif", fps = 20)
=#
