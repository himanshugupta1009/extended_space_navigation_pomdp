include("environment.jl")
include("prm.jl")
include("utils.jl")
include("two_d_action_space_pomdp.jl")
include("belief_tracker.jl")
include("simulator.jl")
using DataStructures

Base.copy(s::cart_state) = cart_state(s.x, s.y,s.theta,s.v,s.L,s.goal)

function run_one_simulation_2D_POMDP_planner(env_right_now, user_defined_rng, m,
                            planner, filename = "output_just_2d_action_space_pomdp_planner.txt")

    time_taken_by_cart = 0
    number_risks = 0
    one_time_step = 0.5
    lidar_range = 30
    num_humans_to_care_about_while_pomdp_planning = 6
    cone_half_angle::Float64 = (2/3)*pi
    number_of_sudden_stops = 0
    cart_ran_into_boundary_wall_flag = false
    cart_ran_into_static_obstacle_flag = false
    cart_reached_goal_flag = true
    cart_throughout_path = OrderedDict()
    all_gif_environments = OrderedDict()
    all_observed_environments = OrderedDict()
    all_generated_beliefs = OrderedDict()
    all_generated_beliefs_using_complete_lidar_data = OrderedDict()
    all_generated_trees = OrderedDict()
    all_risky_scenarios = OrderedDict()
    all_actions = OrderedDict()
    MAX_TIME_LIMIT = 300

    #Sense humans near cart before moving
    #Generate Initial Lidar Data and Belief for humans near cart
    env_right_now.complete_cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
    env_right_now.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(env_right_now.cart,
                                                        env_right_now.complete_cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                        m.pedestrian_distance_threshold, cone_half_angle)

    initial_belief_over_complete_cart_lidar_data = update_belief([],env_right_now.goals,[],env_right_now.complete_cart_lidar_data)
    initial_belief = get_belief_for_selected_humans_from_belief_over_complete_lidar_data(initial_belief_over_complete_cart_lidar_data,
                                                            env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)
    #initial_belief = update_belief([],env_right_now.goals,[],env_right_now.complete_cart_lidar_data)

    dict_key = "t="*string(time_taken_by_cart)
    all_gif_environments[dict_key] = deepcopy(env_right_now)
    all_observed_environments[dict_key] = deepcopy(env_right_now)
    all_generated_beliefs_using_complete_lidar_data[dict_key] = initial_belief_over_complete_cart_lidar_data
    all_generated_beliefs[dict_key] = initial_belief
    cart_throughout_path[dict_key] = copy(env_right_now.cart)

    #Simulate for t=0 to t=1
    io = open(filename,"w")
    write_and_print( io, "Simulating for time interval - (" * string(time_taken_by_cart) * " , " * string(time_taken_by_cart+1) * ")" )
    write_and_print( io, "Current cart state = " * string(env_right_now.cart) )

    #Update human positions in environment for two time steps and cart's belief accordingly
    current_belief_over_complete_cart_lidar_data, risks_in_simulation = simulate_pedestrians_and_generate_gif_environments_when_cart_stationary(env_right_now,
                                                        initial_belief_over_complete_cart_lidar_data,all_gif_environments, all_risky_scenarios, time_taken_by_cart,
                                                        num_humans_to_care_about_while_pomdp_planning,cone_half_angle, lidar_range, m.pedestrian_distance_threshold,
                                                        MersenneTwister( Int64( floor( 100*rand(user_defined_rng) ) ) ) )
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

    #Start Simulating for t>1
    while(!is_within_range(location(env_right_now.cart.x,env_right_now.cart.y), env_right_now.cart.goal, 1.0))
        #display_env(env_right_now)
        io = open(filename,"a")
        cart_ran_into_boundary_wall_flag = check_if_cart_collided_with_boundary_wall(env_right_now)
        cart_ran_into_static_obstacle_flag = check_if_cart_collided_with_static_obstacles(env_right_now)
        if( !cart_ran_into_boundary_wall_flag && !cart_ran_into_static_obstacle_flag )

            write_and_print( io, "Simulating for time interval - (" * string(time_taken_by_cart) * " , " * string(time_taken_by_cart+1) * ")" )
            write_and_print( io, "Current cart state = " * string(env_right_now.cart) )

            #Solve POMDP to get the best action
            # m = golfcart_2D_action_space_pomdp()
            @show(m.world.cart.x)
            b = POMDP_2D_action_space_state_distribution(m.world,current_belief)
            a, info = action_info(planner, b)
            write_and_print( io, "Action chosen by 2D action space POMDP planner: " * string((a[1]*180/pi, a[2])) )
            dict_key = "t="*string(time_taken_by_cart)
            all_generated_trees[dict_key] = deepcopy(info)
            all_actions[dict_key] = (a[1]*180/pi, a[2])

            if(env_right_now.cart.v!=0 && a[2] == -10.0)
                number_of_sudden_stops += 1
            end
            env_right_now.cart.v = clamp(env_right_now.cart.v + a[2],0,m.max_cart_speed)

            if(env_right_now.cart.v != 0.0)
                #That means the cart is not stationary and we now have to simulate both cart and the pedestrians.
                # steering_angle = atan((env_right_now.cart.L*a[1])/env_right_now.cart.v)
                current_belief_over_complete_cart_lidar_data, risks_in_simulation = simulate_cart_and_pedestrians_and_generate_gif_environments_when_cart_moving(
                                                                    env_right_now,current_belief_over_complete_cart_lidar_data, all_gif_environments,
                                                                    all_risky_scenarios, time_taken_by_cart,num_humans_to_care_about_while_pomdp_planning,
                                                                    cone_half_angle, lidar_range, m.pedestrian_distance_threshold,
                                                                    MersenneTwister( Int64( floor( 100*rand(user_defined_rng) ) ) ), a[1])
                current_belief =  get_belief_for_selected_humans_from_belief_over_complete_lidar_data(current_belief_over_complete_cart_lidar_data,
                                                                    env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)
                number_risks += risks_in_simulation
            else
                #That means the cart is stationary and we now just have to simulate the pedestrians.
                current_belief_over_complete_cart_lidar_data, risks_in_simulation = simulate_pedestrians_and_generate_gif_environments_when_cart_stationary(
                                                                    env_right_now,current_belief_over_complete_cart_lidar_data,all_gif_environments,
                                                                    all_risky_scenarios, time_taken_by_cart,num_humans_to_care_about_while_pomdp_planning,
                                                                    cone_half_angle, lidar_range, m.pedestrian_distance_threshold,
                                                                    MersenneTwister( Int64( floor( 100*rand(user_defined_rng) ) ) ) )
                current_belief =  get_belief_for_selected_humans_from_belief_over_complete_lidar_data(current_belief_over_complete_cart_lidar_data,
                                                                    env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)


                number_risks += risks_in_simulation
            end

            time_taken_by_cart += 1
            dict_key = "t="*string(time_taken_by_cart)
            all_observed_environments[dict_key] = deepcopy(env_right_now)
            all_generated_beliefs_using_complete_lidar_data[dict_key] = current_belief_over_complete_cart_lidar_data
            all_generated_beliefs[dict_key] = current_belief
            cart_throughout_path[dict_key] = copy(env_right_now.cart)

            write_and_print( io, "Modified cart state = " * string(env_right_now.cart) )
            write_and_print( io, "************************************************************************" )


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
                all_generated_trees,all_risky_scenarios,all_actions,cart_throughout_path, number_risks, number_of_sudden_stops,
                time_taken_by_cart, cart_reached_goal_flag, cart_ran_into_static_obstacle_flag, cart_ran_into_boundary_wall_flag
end

#function get_available_actions(m::POMDP_Planner_2D_action_space,pomdp_state)
function get_available_actions_non_holonomic(m::POMDP_Planner_2D_action_space,b)
    pomdp_state = first(particles(b))
    x_point =  floor(Int64,pomdp_state.cart.x/ 1.0)+1
    y_point =  floor(Int64,pomdp_state.cart.y/ 1.0)+1
    theta_point = clamp(floor(Int64,pomdp_state.cart.theta/(pi/18))+1,1,36)
    # nearest_prm_point -> Format (vertex_num, x_coordinate, y_coordinate, prm_dist_to_goal)
    nearest_prm_point = m.lookup_table[x_point,y_point,theta_point]
    required_orientation = get_heading_angle( nearest_prm_point[2], nearest_prm_point[3], pomdp_state.cart.x, pomdp_state.cart.y)
    delta_angle = required_orientation - pomdp_state.cart.theta
    abs_delta_angle = abs(delta_angle)
    if(abs_delta_angle<=pi)
        delta_angle = clamp(delta_angle, -pi/4, pi/4)
    else
        if(delta_angle>=0.0)
            delta_angle = clamp(delta_angle-2*pi, -pi/4, pi/4)
        else
            delta_angle = clamp(delta_angle+2*pi, -pi/4, pi/4)
        end
    end
    if(pomdp_state.cart.v == 0.0)
        if(delta_angle==pi/4 || delta_angle==-pi/4)
            return [(-pi/4,1.0),(-pi/6,1.0),(-pi/12,1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,1.0),(pi/4,1.0)]
        else
            return [(delta_angle, 1.0),(-pi/4,1.0),(-pi/6,1.0),(-pi/12,1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,1.0),(pi/4,1.0)]
        end
    else
        if(delta_angle==pi/4 || delta_angle==-pi/4)
            # return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0)]
        else
            # return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0)]
        end
    end
end
#@code_warntype get_available_actions(POMDP_state_2D_action_space(env.cart,env.humans))

function get_available_actions_holonomic(m::POMDP_Planner_2D_action_space,b)
    pomdp_state = first(particles(b))
    x_point =  floor(Int64,pomdp_state.cart.x/ 1.0)+1
    y_point =  floor(Int64,pomdp_state.cart.y/ 1.0)+1
    # nearest_prm_point -> Format (vertex_num, x_coordinate, y_coordinate, prm_dist_to_goal)
    nearest_prm_point = m.lookup_table[x_point,y_point]
    required_orientation = get_heading_angle( nearest_prm_point[2], nearest_prm_point[3], pomdp_state.cart.x, pomdp_state.cart.y)
    delta_angle = required_orientation - pomdp_state.cart.theta
    if(pomdp_state.cart.v == 0.0)
        return [(delta_angle, 1.0),(-pi/4,1.0),(-pi/6,1.0),(-pi/12,1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,1.0),(pi/4,1.0)]
    else
        if(delta_angle==pi/4 || delta_angle==-pi/4)
            # return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0)]
        else
            # return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0)]
        end
    end
end

lookup_table = nothing
run_simulation_flag = true
if(run_simulation_flag)
    gr()
    rand_noise_generator_seed_for_env = 69
    rand_noise_generator_for_env = MersenneTwister(rand_noise_generator_seed_for_env)
    rand_noise_generator_seed_for_sim = 7
    # env = generate_environment_no_obstacles(300, rand_noise_generator_for_env)
    # env = generate_environment_small_circular_obstacles(300, rand_noise_generator_for_env)
    # env = generate_environment_large_circular_obstacles(300, rand_noise_generator_for_env)
    env = generate_environment_L_shaped_corridor(300, rand_noise_generator_for_env)
    if(lookup_table == nothing)
        graph = generate_prm_vertices(500, MersenneTwister(11), env)
        d = generate_prm_edges(env, graph, 10)
        lookup_table = generate_prm_points_lookup_table_non_holonomic(env,graph)
    end
    env_right_now = deepcopy(env)

    #Create POMDP for env_right_now
    #POMDP_Planner_2D_action_space <: POMDPs.POMDP{POMDP_state_2D_action_space,Int,Array{location,1}}
    # discount_factor::Float64; pedestrian_distance_threshold::Float64; pedestrian_collision_penalty::Float64;
    # obstacle_distance_threshold::Float64; obstacle_collision_penalty::Float64; goal_reward_distance_threshold::Float64;
    # cart_goal_reached_distance_threshold::Float64; goal_reward::Float64; max_cart_speed::Float64; world::experiment_environment
    golfcart_2D_action_space_pomdp = POMDP_Planner_2D_action_space(0.97,1.0,-100.0,1.0,-100.0,0.0,1.0,1000.0,2.0,env_right_now,lookup_table)
    discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
    isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
    actions(m::POMDP_Planner_2D_action_space,b) = get_available_actions_non_holonomic(m,b)

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(b->calculate_lower_bound_policy_pomdp_planning_2D_action_space(golfcart_2D_action_space_pomdp, b)),
                            max_depth=100),calculate_upper_bound_value_pomdp_planning_2D_action_space, check_terminal=true),K=50,D=100,T_max=Inf,max_trials=50, tree_in_info=true,
                            rng=MersenneTwister(100))
    # solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space),max_depth=100,
    #                         final_value=reward_to_be_awarded_at_max_depth_in_lower_bound_policy_rollout),
    #                         calculate_upper_bound_value_pomdp_planning_2D_action_space, check_terminal=true),K=100,D=100,T_max=0.5, tree_in_info=true, default_action=(-10.0,-10.0))

    planner = POMDPs.solve(solver, golfcart_2D_action_space_pomdp);

    just_2D_pomdp_all_gif_environments, just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs_using_complete_lidar_data,
            just_2D_pomdp_all_generated_beliefs, just_2D_pomdp_all_generated_trees, just_2D_pomdp_all_risky_scenarios, just_2D_pomdp_all_actions,
            just_2D_pomdp_cart_throughout_path, just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,just_2D_pomdp_time_taken_by_cart,
            just_2D_pomdp_cart_reached_goal_flag, just_2D_pomdp_cart_ran_into_static_obstacle_flag,
            just_2D_pomdp_cart_ran_into_boundary_wall_flag = run_one_simulation_2D_POMDP_planner(env_right_now, MersenneTwister(rand_noise_generator_seed_for_sim),
                                                                                        golfcart_2D_action_space_pomdp, planner)

    anim = @animate for k ∈ keys(just_2D_pomdp_all_observed_environments)
        #display_env(just_2D_pomdp_all_observed_environments[k], k);
        #savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*string(i)*".png")
    end

    gif(anim, "just_2D_action_space_pomdp_planner_run.gif", fps = 2)
end

#=
anim = @animate for k ∈ keys(just_2D_pomdp_all_gif_environments)
    display_env(just_2D_pomdp_all_gif_environments[k],k);
    #savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*all_gif_environments[i][1]*".png")
end
gif(anim, "just_2D_action_space_pomdp_planner_run.gif", fps = 20)
=#

#inchrome(D3Tree(just_2D_pomdp_all_generated_trees[9][:tree]))

#Notes on the simulator!
#=
1) just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs and just_2D_pomdp_all_generated_trees have data for
        Initial environment at index 1
        Env from time step 0 to the end from index 2 onwards
        Env at index 2 is obtained by just waiting and updating the belief over pedestrians in lidar data
        Also, ith index in just_2D_pomdp_all_observed_environments implies the environment at time stamp (i-1).
        i.e. just_2D_pomdp_all_observed_environments[23] is actually the environment at time stamp 22 seconds.
2) just_2D_pomdp_all_gif_environments has the simulator data for every 0.1 second.
        just_2D_pomdp_all_gif_environments[i][1] gives the simulator time stamp.
        If just_2D_pomdp_all_gif_environments[i][1] is 22_6, then it means that env is for time step 22 sec to 23 seconds and 0.1*6=0.6 seconds after 22 seconds.
        If it is 22_10, then it means that env is for 0.1*10=1 second after 22 seconds. So, essentially env at 23 seconds
        The next one would be 23_1. There is nothing of the format 23_0 (equivalent of that is 22_10). 23_1 means that env is for 0.1 second after 23 seconds.
        So, 22_6 essentially implies that this corresponds to the simulator starting from environment at index 23 in just_2D_pomdp_all_observed_environments.
        i.e. a_b corresponds to environment at index a+1 in just_2D_pomdp_all_observed_environments when b is not equal to 10
             When b = 10, it corresponds to environment at index a+2 in just_2D_pomdp_all_observed_environments
             In short, we can say a_b corresponds to environment at index floor(a+1+(0.1*b)) in just_2D_pomdp_all_observed_environments
        If, just_2D_pomdp_all_gif_environments[i][1] is 22_6, then i is 227. So, a_b is at index (10*a)+b+1
        Also, just_2D_pomdp_all_observed_environments[i] === just_2D_pomdp_all_gif_environments [(i-1)*10 + 1]
        The index for just_2D_pomdp_all_generated_trees is same as the index for just_2D_pomdp_all_observed_environments.
=#
