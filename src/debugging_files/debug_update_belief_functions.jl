function debug_update_belief(old_env,old_belief)
    new_env = deepcopy(old_env)
    new_env.humans = move_human_for_one_time_step_in_actual_environment(old_env,0.0,MersenneTwister(7))
    new_env.cart_lidar_data = get_lidar_data(new_env,30)
    new_env.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(new_env.cart,
                                                        new_env.cart_lidar_data, 6, pi/3.0)
    new_belief = update_belief_from_old_world_and_new_world(old_belief, old_env, new_env)
    #@show(current_belief)
    return new_env,new_belief
end

function debug_update_belief_using_temp_world(old_env,old_belief)
    new_env = deepcopy(old_env)
    new_env.humans = move_human_for_one_time_step_in_actual_environment(old_env,1.0,MersenneTwister(7))
    new_env.cart_lidar_data = get_lidar_data(new_env,30)
    new_env.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(new_env.cart,
                                                        new_env.cart_lidar_data, 10, pi/3.0)

    new_belief = update_current_belief_by_creating_temp_world(old_env, new_env, old_belief, 30, 6, pi/3.0)

    #@show(current_belief)
    return new_env,new_belief
end
