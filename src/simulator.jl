#Functions for simulating the cart and pedestrians for the 2D planner

#Returns the updated belief over humans and number of risks encountered
function simulate_pedestrians_and_generate_gif_environments_when_cart_stationary(env_right_now, current_belief,
                                                                        all_gif_environments, all_risky_scenarios, time_stamp,
                                                                        num_humans_to_care_about_while_pomdp_planning, cone_half_angle,
                                                                        lidar_range, closest_ped_dist_threshold, user_defined_rng)

    number_risks = 0
    env_before_humans_simulated_for_first_half_second = deepcopy(env_right_now)

    #Simulate for 0 to 0.5 seconds
    for i in 1:5
        env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,0.1,user_defined_rng)
        env_right_now.complete_cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
        env_right_now.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(env_right_now.cart,
                                                            env_right_now.complete_cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                            closest_ped_dist_threshold, cone_half_angle)
        dict_key = "t="*string(time_stamp)*"_"*string(i)
        all_gif_environments[dict_key] =  deepcopy(env_right_now)
        if(get_count_number_of_risks(env_right_now) != 0)
            number_risks += get_count_number_of_risks(env_right_now)
            all_risky_scenarios[dict_key] =  deepcopy(env_right_now)
        end
    end

    #Update your belief after first 0.5 seconds
    updated_belief = update_belief_from_old_world_and_new_world(current_belief,
                                                    env_before_humans_simulated_for_first_half_second, env_right_now)

    #Simulate for 0.5 to 1 second
    env_before_humans_simulated_for_second_half_second = deepcopy(env_right_now)
    for i in 6:10
        env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,0.1,user_defined_rng)
        if(i==10)
            respawn_humans(env_right_now, user_defined_rng)
        end
        env_right_now.complete_cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
        env_right_now.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(env_right_now.cart,
                                                            env_right_now.complete_cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                            closest_ped_dist_threshold, cone_half_angle)
        dict_key = "t="*string(time_stamp)*"_"*string(i)
        all_gif_environments[dict_key] =  deepcopy(env_right_now)
        if(get_count_number_of_risks(env_right_now) != 0)
            number_risks += get_count_number_of_risks(env_right_now)
            all_risky_scenarios[dict_key] =  deepcopy(env_right_now)
        end
    end

    #Update your belief after second 0.5 seconds
    final_updated_belief = update_belief_from_old_world_and_new_world(updated_belief,
                                                    env_before_humans_simulated_for_second_half_second, env_right_now)

    return final_updated_belief, number_risks
end

#Returns the updated belief over humans and number of risks encountered
function simulate_cart_and_pedestrians_and_generate_gif_environments_when_cart_moving(env_right_now, current_belief,
                                                            all_gif_environments, all_risky_scenarios, time_stamp,
                                                            num_humans_to_care_about_while_pomdp_planning, cone_half_angle,
                                                            lidar_range, closest_ped_dist_threshold, user_defined_rng, steering_angle)

    number_risks = 0

    #Simulate for 0 to 0.5 seconds
    env_before_humans_and_cart_simulated_for_first_half_second = deepcopy(env_right_now)
    initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    for i in 1:5
        extra_parameters = [env_right_now.cart.v, env_right_now.cart.L, steering_angle]
        x,y,theta = get_intermediate_points(initial_state, 0.1, extra_parameters);
        env_right_now.cart.x, env_right_now.cart.y, env_right_now.cart.theta = last(x), last(y), last(theta)
        env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,0.1,user_defined_rng)
        env_right_now.complete_cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
        env_right_now.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(env_right_now.cart,
                                                            env_right_now.complete_cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                            closest_ped_dist_threshold, cone_half_angle)

        dict_key = "t="*string(time_stamp)*"_"*string(i)
        all_gif_environments[dict_key] =  deepcopy(env_right_now)
        if(get_count_number_of_risks(env_right_now) != 0)
            number_risks += get_count_number_of_risks(env_right_now)
            all_risky_scenarios[dict_key] =  deepcopy(env_right_now)
        end
        initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    end

    #Update your belief after first 0.5 seconds
    updated_belief = update_belief_from_old_world_and_new_world(current_belief,
                                                    env_before_humans_and_cart_simulated_for_first_half_second, env_right_now)

    #Simulate for 0.5 to 1 second
    env_before_humans_and_cart_simulated_for_second_half_second = deepcopy(env_right_now)
    initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    for i in 6:10
        extra_parameters = [env_right_now.cart.v, env_right_now.cart.L, steering_angle]
        x,y,theta = get_intermediate_points(initial_state, 0.1, extra_parameters);
        env_right_now.cart.x, env_right_now.cart.y, env_right_now.cart.theta = last(x), last(y), last(theta)
        env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,0.1,user_defined_rng)
        if(i==10)
            respawn_humans(env_right_now, user_defined_rng)
        end
        env_right_now.complete_cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
        env_right_now.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(env_right_now.cart,
                                                            env_right_now.complete_cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                            closest_ped_dist_threshold, cone_half_angle)

        dict_key = "t="*string(time_stamp)*"_"*string(i)
        all_gif_environments[dict_key] =  deepcopy(env_right_now)
        if(get_count_number_of_risks(env_right_now) != 0)
            number_risks += get_count_number_of_risks(env_right_now)
            all_risky_scenarios[dict_key] =  deepcopy(env_right_now)
        end
        initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    end

    #Update your belief after second 0.5 seconds
    final_updated_belief = update_belief_from_old_world_and_new_world(updated_belief,
                                                    env_before_humans_and_cart_simulated_for_second_half_second, env_right_now)

    return final_updated_belief, number_risks
end

#Functions for simulating the cart and pedestrians for the 1D planner

#Returns the updated belief over humans and number of risks encountered
function hybrid_astar_1D_pomdp_simulate_pedestrians_and_generate_gif_environments_when_cart_stationary(env_right_now, current_belief,
                                                                        all_gif_environments, all_risky_scenarios, time_stamp,
                                                                        num_humans_to_care_about_while_pomdp_planning, cone_half_angle,
                                                                        lidar_range, closest_ped_dist_threshold, user_defined_rng)

    number_risks = 0
    env_before_humans_and_cart_simulated_for_first_half_second = deepcopy(env_right_now)

    #Simulate for 0 to 0.5 seconds
    for i in 1:5
        env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,0.1,user_defined_rng)
        env_right_now.complete_cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
        env_right_now.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(env_right_now.cart,
                                                            env_right_now.complete_cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                            closest_ped_dist_threshold, cone_half_angle)
        dict_key = "t="*string(time_stamp)*"_"*string(i)
        all_gif_environments[dict_key] =  deepcopy(env_right_now)
        if(get_count_number_of_risks(env_right_now) != 0)
            number_risks += get_count_number_of_risks(env_right_now)
            all_risky_scenarios[dict_key] =  deepcopy(env_right_now)
        end
    end

    #Update your belief after first 0.5 seconds
    updated_belief = update_belief_from_old_world_and_new_world(current_belief,
                                                    env_before_humans_and_cart_simulated_for_first_half_second, env_right_now)

    #Simulate for 0.5 to 1 second
    env_before_humans_and_cart_simulated_for_second_half_second = deepcopy(env_right_now)
    for i in 6:10
        env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,0.1,user_defined_rng)
        if(i==10)
            respawn_humans(env_right_now, user_defined_rng)
        end
        env_right_now.complete_cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
        env_right_now.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(env_right_now.cart,
                                                            env_right_now.complete_cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                            closest_ped_dist_threshold, cone_half_angle)
        dict_key = "t="*string(time_stamp)*"_"*string(i)
        all_gif_environments[dict_key] =  deepcopy(env_right_now)
        if(get_count_number_of_risks(env_right_now) != 0)
            number_risks += get_count_number_of_risks(env_right_now)
            all_risky_scenarios[dict_key] =  deepcopy(env_right_now)
        end
    end

    #Update your belief after second 0.5 seconds
    final_updated_belief = update_belief_from_old_world_and_new_world(updated_belief,
                                                    env_before_humans_and_cart_simulated_for_second_half_second, env_right_now)

    return final_updated_belief, number_risks
end

#Returns the updated belief over humans and number of risks encountered
function hybrid_astar_1D_pomdp_simulate_cart_and_pedestrians_and_generate_gif_environments_when_cart_moving(env_right_now, current_belief,
                                                            all_gif_environments, all_risky_scenarios, time_stamp,
                                                            num_humans_to_care_about_while_pomdp_planning, cone_half_angle,
                                                            lidar_range, closest_ped_dist_threshold, user_defined_rng)

    #First simulate only the cart and get its path
    goal_reached_in_this_time_step_flag = false
    if(env_right_now.cart.v > length(env_right_now.cart_hybrid_astar_path))
        steering_angles = env_right_now.cart_hybrid_astar_path
        goal_reached_in_this_time_step_flag = true
    else
        steering_angles = env_right_now.cart_hybrid_astar_path[1:Int(env_right_now.cart.v)]
    end
    cart_path_x = Float64[]; cart_path_y = Float64[]; cart_path_theta = Float64[]
    initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    for i in 1:length(steering_angles)
        steering_angle = steering_angles[i]
        extra_parameters = [env_right_now.cart.v, env_right_now.cart.L, steering_angle]
        x,y,theta = get_intermediate_points(initial_state, 1.0/env_right_now.cart.v, extra_parameters, 0.1/env_right_now.cart.v )
        append!(cart_path_x, x[2:end])
        append!(cart_path_y, y[2:end])
        append!(cart_path_theta, theta[2:end])
        initial_state = [last(cart_path_x),last(cart_path_y),last(cart_path_theta)]
    end

    number_risks = 0

    #Simulate for 0 to 0.5 seconds
    env_before_humans_and_cart_simulated_for_first_half_second = deepcopy(env_right_now)
    initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    curr_hybrid_astar_path_index = 0

    for i in 1:5
        cart_path_index = clamp(Int(i*env_right_now.cart.v),1,10*length(steering_angles))
        env_right_now.cart.x, env_right_now.cart.y, env_right_now.cart.theta = cart_path_x[cart_path_index], cart_path_y[cart_path_index], cart_path_theta[cart_path_index]
        env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,0.1,user_defined_rng)
        env_right_now.complete_cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
        env_right_now.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(env_right_now.cart,
                                                            env_right_now.complete_cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                            closest_ped_dist_threshold, cone_half_angle)
        if( floor( (0.1*i) / (1/env_right_now.cart.v) ) > curr_hybrid_astar_path_index)
            curr_hybrid_astar_path_index += 1
            env_right_now.cart_hybrid_astar_path = env_right_now.cart_hybrid_astar_path[2 : end]
        end
        dict_key = "t="*string(time_stamp)*"_"*string(i)
        all_gif_environments[dict_key] =  deepcopy(env_right_now)
        if(get_count_number_of_risks(env_right_now) != 0)
            number_risks += get_count_number_of_risks(env_right_now)
            all_risky_scenarios[dict_key] =  deepcopy(env_right_now)
        end
        initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    end

    #Update your belief after first 0.5 seconds
    updated_belief = update_belief_from_old_world_and_new_world(current_belief,
                                                    env_before_humans_and_cart_simulated_for_first_half_second, env_right_now)

    #Simulate for 0.5 to 1 second
    env_before_humans_and_cart_simulated_for_second_half_second = deepcopy(env_right_now)
    initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    for i in 6:10
        cart_path_index = clamp(Int(i*env_right_now.cart.v),1,10*length(steering_angles))
        env_right_now.cart.x, env_right_now.cart.y, env_right_now.cart.theta = cart_path_x[cart_path_index], cart_path_y[cart_path_index], cart_path_theta[cart_path_index]
        env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,0.1,user_defined_rng)
        if(i==10)
            respawn_humans(env_right_now, user_defined_rng)
        end
        env_right_now.complete_cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
        env_right_now.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(env_right_now.cart,
                                                            env_right_now.complete_cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                            closest_ped_dist_threshold, cone_half_angle)
        if( floor( (0.1*i) / (1/env_right_now.cart.v) ) > curr_hybrid_astar_path_index)
            curr_hybrid_astar_path_index += 1
            env_right_now.cart_hybrid_astar_path = env_right_now.cart_hybrid_astar_path[2 : end]
        end
        dict_key = "t="*string(time_stamp)*"_"*string(i)
        all_gif_environments[dict_key] =  deepcopy(env_right_now)
        if(get_count_number_of_risks(env_right_now) != 0)
            number_risks += get_count_number_of_risks(env_right_now)
            all_risky_scenarios[dict_key] =  deepcopy(env_right_now)
        end
        initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    end

    #Update your belief after second 0.5 seconds
    final_updated_belief = update_belief_from_old_world_and_new_world(updated_belief,
                                                    env_before_humans_and_cart_simulated_for_second_half_second, env_right_now)

    # if(goal_reached_in_this_time_step_flag)
    #     env_right_now.cart_hybrid_astar_path = []
    # else
    #     env_right_now.cart_hybrid_astar_path = env_right_now.cart_hybrid_astar_path[Int(env_right_now.cart.v)+1:end]
    # end
    return final_updated_belief, number_risks
end
