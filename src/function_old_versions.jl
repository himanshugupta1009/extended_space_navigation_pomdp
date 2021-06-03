#Function used in Hybrid A* Path for evaluating edge cost

function get_action_cost(environment, current_belief, final_x::Float64, final_y::Float64, obs_thresold::Float64, hum_threshold::Float64, action::Float64)
    total_cost::Float64 = 0.0

    #Cost from going out of bounds
    if(final_x<=0.0 || final_x>=env.length)
        return Inf
    end
    if(final_y<=0.0 || final_y>=env.breadth)
        return Inf
    end

    padding_radius = 5.0
    lidar_range = 20.0

    #Cost from obstacles
    for obstacle in environment.obstacles
        euclidean_distance::Float64 = ( (final_x - obstacle.x)^2 + (final_y - obstacle.y)^2 )^ 0.5
        if(euclidean_distance >= obstacle.r + padding_radius)
            continue
        elseif(euclidean_distance <= obstacle.r)
            total_cost = Inf
        else
            distance_between_cart_and_obstacle = euclidean_distance - obstacle.r
            total_cost += obs_thresold* (padding_radius - distance_between_cart_and_obstacle)
        end
    end

    #Cost from humans seen by lidar
    for human_index in 1:length(environment.cart_lidar_data)
        human_dist_threshold = 4.0
        human = environment.cart_lidar_data[human_index]
        euclidean_distance::Float64 = ( (final_x - human.x)^2 + (final_y - human.y)^2 )^ 0.5
        if(euclidean_distance >= lidar_range)
            continue
        elseif(euclidean_distance <= human_dist_threshold)
            #@show("A")
            total_cost = Inf
        else
            max_ele, max_ele_index = find_max_ele_and_index(current_belief[human_index].distribution,
                                        length(current_belief[human_index].distribution))
            if(max_ele < 0.5)
                #@show("B")
                total_cost += hum_threshold * hum_threshold * (1/euclidean_distance)
                #total_cost = Inf
            else
                estimated_human_goal = environment.goals[max_ele_index]
                direction_human = (estimated_human_goal.x - human.x , estimated_human_goal.y - human.y)
                #direction_human = direction_human/sum(direction_human)
                direction_cart = (final_x - human.x, final_y - human.y)
                #direction_cart = direction_cart/sum(direction_cart)
                t0 = dot(direction_cart,direction_human)/ dot(direction_human,direction_human)
                if(t0 <= 0)
                    #@show("C")
                    continue
                else
                    #@show("D")
                    distance_of_cart_from_line = (direction_cart[1] - t0*direction_human[1],
                                                        direction_cart[2] - t0*direction_human[2])
                    distance_of_cart_from_line = sqrt(dot(distance_of_cart_from_line,distance_of_cart_from_line))
                    if(distance_of_cart_from_line<3.0)
                        total_cost = Inf
                    else
                    #total_cost += hum_threshold * (1/distance_of_cart_from_line)
                    total_cost += 0.0
                    end
                end
            end
        end
    end

    #Cost from no change in steering angle
    if(action == 0.0)
       total_cost += -0.1
    end

    #Cost from Long Paths
    total_cost += 1

    return total_cost
end

#Function used in Hybrid A* Path for evaluating edge cost.
#This version was removed on October 5, 2020 )
function get_action_cost(environment, current_belief, final_x::Float64, final_y::Float64, obs_cost::Float64, pedestrian_cost::Float64, action::Float64, current_node_time_step::Float64)
    total_cost::Float64 = 0.0

    #Cost from going out of bounds
    if(final_x<=0.0 || final_x>=env.length)
        return Inf
    end
    if(final_y<=0.0 || final_y>=env.breadth)
        return Inf
    end

    #Cost from obstacles
    padding_radius = 5.0
    for obstacle in environment.obstacles
        euclidean_distance::Float64 = ( (final_x - obstacle.x)^2 + (final_y - obstacle.y)^2 )^ 0.5
        if(euclidean_distance >= obstacle.r + padding_radius)
            continue
        elseif(euclidean_distance <= obstacle.r)
            total_cost = Inf
        else
            distance_between_cart_and_obstacle = euclidean_distance - obstacle.r
            total_cost += obs_cost* (padding_radius - distance_between_cart_and_obstacle)
        end
    end

    #Cost from humans seen by lidar
    lidar_data_pedestrian_cost_threshold = 20.0
    for human_index in 1:length(environment.cart_lidar_data)

        human_dist_threshold = 4.0
        human = environment.cart_lidar_data[human_index]
        max_ele, max_ele_index = find_max_ele_and_index(current_belief[human_index].distribution,
                                    length(current_belief[human_index].distribution))

        if(max_ele < 0.5)
            #@show("B")
            euclidean_distance::Float64 = sqrt( (final_x - human.x)^2 + (final_y - human.y)^2)
            if(euclidean_distance >= lidar_data_pedestrian_cost_threshold)
                continue
            elseif(euclidean_distance <= human_dist_threshold)
                total_cost = Inf
            else
                total_cost += pedestrian_cost * (1/euclidean_distance)
            end
        else
            inferred_human_goal = environment.goals[max_ele_index]
            inferred_human_state = human_state(human.x,human.y,human.v,inferred_human_goal,human.id)
            expected_human_position = get_expected_human_position_at_time_t_hybrid_astar_planning(inferred_human_state,
                                                                    environment,current_node_time_step)
            distance_of_cart_from_expected_human_position::Float64 = sqrt( (final_x - expected_human_position[1])^2
                                                + (final_y - expected_human_position[2])^2 )
            if(distance_of_cart_from_expected_human_position >= lidar_data_pedestrian_cost_threshold)
                continue
            elseif(distance_of_cart_from_expected_human_position <= human_dist_threshold)
                total_cost = Inf
            else
                total_cost += pedestrian_cost * (1/distance_of_cart_from_expected_human_position)
            end
        end
    end

    #Cost from no change in steering angle
    if(action == 0.0)
       total_cost += -0.1
    end

    #Cost from Long Paths
    total_cost += 1

    return total_cost
end

#Function used in Hybrid A* Path for evaluating edge cost
#This version was removed on October 5, 2020
function get_expected_human_position_at_time_t_hybrid_astar_planning(human, world, time_step)

    #First Quadrant
    if(human.goal.x >= human.x && human.goal.y >= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y + (human.v)*time_step
        elseif(human.goal.y == human.y)
            new_x = human.x + (human.v)*time_step
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x + ((human.v)*time_step)*cos(heading_angle)
            new_y = human.y + ((human.v)*time_step)*sin(heading_angle)
        end
    #Second Quadrant
    elseif(human.goal.x <= human.x && human.goal.y >= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y + (human.v)*time_step
        elseif(human.goal.y == human.y)
            new_x = human.x - (human.v)*time_step
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x - ((human.v)*time_step)*cos(heading_angle)
            new_y = human.y - ((human.v)*time_step)*sin(heading_angle)
        end
    #Third Quadrant
    elseif(human.goal.x <= human.x && human.goal.y <= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y - (human.v)*time_step
        elseif(human.goal.y == human.y)
            new_x = human.x - (human.v)*time_step
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x - ((human.v)*time_step)*cos(heading_angle)
            new_y = human.y - ((human.v)*time_step)*sin(heading_angle)
        end
    #Fourth Quadrant
    else(human.goal.x >= human.x && human.goal.y <= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y - (human.v)*time_step
        elseif(human.goal.y == human.y)
            new_x = human.x + (human.v)*time_step
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x + ((human.v)*time_step)*cos(heading_angle)
            new_y = human.y + ((human.v)*time_step)*sin(heading_angle)
        end
    end

    new_x = clamp(new_x,0,world.length)
    new_y = clamp(new_y,0,world.breadth)
    #@show(new_x,new_y)
    return (new_x,new_y)
end

#Generate Initial POMDP state SparseCat distribution/ Generate the Belief from which POMDP states will be sampled.
function generate_pomdp_states_and_probability(curr_human_index::Int64,num_humans::Int64,
        initial_human_states::Array{human_state,1},all_human_goal_locations::Array{location,1},
        num_goals::Int64,current_belief::Array{human_probability_over_goals,1},
        current_cart_state::cart_state, start_path_index::Int64,
        all_possible_POMDP_states::Array{SP_POMDP_state,1},
        all_probability_values::Array{Float64,1})::Tuple{Array{SP_POMDP_state,1},Array{Float64,1}}

    new_array_all_possible_POMDP_states = Array{SP_POMDP_state,1}()
    new_array_all_probability_values = Array{Float64,1}()

    if(length(all_possible_POMDP_states)==0)
        for goal_index in 1:num_goals
            sampled_human_state = human_state(initial_human_states[curr_human_index].x,
            initial_human_states[curr_human_index].y,initial_human_states[curr_human_index].v,
            all_human_goal_locations[goal_index],initial_human_states[curr_human_index].id)
            #sampled_human_state_array = Array{human_state,1}([sampled_human_state])
            generated_state = SP_POMDP_state(current_cart_state,[sampled_human_state],start_path_index)
            push!(new_array_all_possible_POMDP_states,generated_state)
            push!(new_array_all_probability_values,current_belief[curr_human_index].distribution[goal_index])
        end
    else
        for pomdp_state_index in 1:length(all_possible_POMDP_states)
            for goal_index in 1:num_goals
                sampled_human_state = human_state(initial_human_states[curr_human_index].x,
                initial_human_states[curr_human_index].y,initial_human_states[curr_human_index].v,
                all_human_goal_locations[goal_index],initial_human_states[curr_human_index].id)
                generated_state = SP_POMDP_state(current_cart_state,
                    vcat(all_possible_POMDP_states[pomdp_state_index].pedestrians,sampled_human_state),
                    start_path_index)
                push!(new_array_all_possible_POMDP_states,generated_state)
                push!(new_array_all_probability_values,
                    all_probability_values[pomdp_state_index]*(current_belief[curr_human_index].distribution[goal_index]))
            end
        end
    end
    if(curr_human_index == num_humans)
        return (new_array_all_possible_POMDP_states::Array{SP_POMDP_state,1},
                new_array_all_probability_values::Array{Float64,1})
    else
        returned_tuple::Tuple{Array{SP_POMDP_state,1},Array{Float64,1}} =
        generate_pomdp_states_and_probability(curr_human_index+1,num_humans,initial_human_states,
                all_human_goal_locations,num_goals,current_belief,current_cart_state, start_path_index,
                new_array_all_possible_POMDP_states,
                new_array_all_probability_values)::Tuple{Array{SP_POMDP_state,1},Array{Float64,1}}
        return returned_tuple
    end
end
#@code_warntype generate_pomdp_states_and_probability(1,6,env.humans,env.goals,length(env.goals),current_belief,env.cart,1,SP_POMDP_state[],Float64[])

function initialstate_distribution(m::POMDP_Planner_1D_action_space,current_human_goal_probability)
    initial_cart_state = m.world.cart
    all_human_goal_locations = m.world.goals
    initial_human_states = m.world.cart_lidar_data
    initial_path_start_index = m.start_path_index
    num_goals::Int64 = length(all_human_goal_locations)
    num_humans::Int64 = length(initial_human_states)

    all_possible_states = SP_POMDP_state[]
    all_probability_values = Float64[]

    if(num_humans == 0)
        generated_state = SP_POMDP_state(initial_cart_state,human_state[],initial_path_start_index)
        push!(all_possible_states,generated_state)
        push!(all_probability_values,1.0)
    else
        (all_possible_states,all_probability_values) = generate_pomdp_states_and_probability(1,
            num_humans,initial_human_states,all_human_goal_locations,num_goals,
            current_human_goal_probability,initial_cart_state,initial_path_start_index,
            all_possible_states,all_probability_values);
    end

    d = SparseCat(all_possible_states, all_probability_values)
    return d
end
#@code_warntype initialstate_distribution(golfcart_pomdp(),initial_human_dis_list)

#Function used in main_2d_action_space_pomdp.jl
#This version was removed on October 8, 2020
#Code I had to create a gif. It was not a function. I created a function later for the actual code.
function main_2d_action_space_pomdp_for_animation()
    # anim = @animate for j ∈ 1:10000
    #     io = open("output.txt","a")
    #     @show(j)
    #     if(!is_within_range(location(env_right_now.cart.x,env_right_now.cart.y), env_right_now.cart.goal, 1.0))
    #         global time_taken_by_cart += 1
    #         if( (env_right_now.cart.x<=100.0 && env_right_now.cart.y<=100.0 && env_right_now.cart.x>=0.0 && env_right_now.cart.y>=0.0) )
    #
    #             push!(all_observed_environments, (deepcopy(env_right_now),current_belief))
    #             write(io,"Current Iteration Number - $j\n")
    #
    #             m = golfcart_2D_action_space_pomdp()
    #             b = POMDP_2D_action_space_state_distribution(m.world,current_belief)
    #             a, info = action_info(planner, b)
    #             println("Action chosen " , a)
    #             write(io,"Action chosen: $a\n")
    #             old_env = deepcopy(env_right_now)
    #             env_right_now.cart.v = clamp(env_right_now.cart.v + a[2],0,m.max_cart_speed)
    #             if(env_right_now.cart.v != 0.0)
    #                 println("Current cart state = " , string(env_right_now.cart))
    #                 write(io,"Current cart state = $(env_right_now.cart)\n")
    #
    #                 initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    #                 steering_angle = atan((env_right_now.cart.L*a[1])/env_right_now.cart.v)
    #                 for i in 1:Int64(env_right_now.cart.v)
    #                     extra_parameters = [env_right_now.cart.v, env_right_now.cart.L, steering_angle]
    #                     x,y,theta = get_intermediate_points(initial_state, 1.0/env_right_now.cart.v, extra_parameters);
    #                     env_right_now.cart.x, env_right_now.cart.y, env_right_now.cart.theta = last(x), last(y), last(theta)
    #                     env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,1.0/env_right_now.cart.v)
    #                     display_env(env_right_now);
    #                     savefig("./plots/plot_"*string(j)*"_"*string(i)*".png")
    #                     initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
    #                 end
    #             else
    #                 println("Current cart state = " , string(env_right_now.cart))
    #                 write(io,"Current cart state = $(env_right_now.cart)\n")
    #                 display_env(env_right_now);
    #                 savefig("./plots/plot_"*string(j)*"_"*string(i)*".png")
    #             end
    #
    #             global current_belief = update_current_belief(old_env, env_right_now, current_belief, lidar_range, one_time_step)
    #             global number_risks += get_count_number_of_risks(env_right_now)
    #             #display_env(env_right_now);
    #             #savefig("./plots/plot_$j.png")
    #             push!(cart_throughout_path,(copy(env_right_now.cart)))
    #         else
    #             @show("Cart ran into Boundary Walls")
    #             write(io,"Cart ran into Boundary Walls")
    #             close(io)
    #             break
    #         end
    #     else
    #         @show("Goal reached")
    #         write(io,"Goal Reached!")
    #         close(io)
    #         break
    #     end
    #     close(io)
    # end

    # println("Time Taken by cart to reach goal : ", time_taken_by_cart)
    # println("Number of risky scenarios encountered by the cart : ", number_risks)
    # gif(anim, "og_pipeline_new.gif", fps = 2)
end

#Code I had to create a gif. It was not a function. I created a function later for the actual code.
#Function used in main_hybrid_1d_pomdp_path_planner_reuse_old_path.jl
#This version was removed on October 8, 2020
function main_hybrid_1d_pomdp_path_planner_reuse_old_path_for_animation()
    anim = @animate for j ∈ 1:10000
        io = open("output.txt","a")
        @show(j)
        if(!is_within_range(location(env_right_now.cart.x,env_right_now.cart.y), env_right_now.cart.goal, 1.0))
            global time_taken_by_cart += 1
            humans_to_avoid = get_nearest_six_pedestrians(env_right_now,current_belief)
            hybrid_a_star_path = @time hybrid_a_star_search(env_right_now.cart.x, env_right_now.cart.y,
                env_right_now.cart.theta, env_right_now.cart.goal.x, env_right_now.cart.goal.y, env_right_now, humans_to_avoid);
            write(io,"Current Iteration Number - $j\n")
            if( (length(hybrid_a_star_path) == 0) && (length(env_right_now.cart_hybrid_astar_path) == 0) )
                println("**********Hybrid A Star Path Not found. No old path exists either**********")
                write(io,"**********Hybrid A Star Path Not found. No old path exists either**********\n")
                env_right_now.cart.v = 0.0
                push!(all_observed_environments, (deepcopy(env_right_now),current_belief))
                display_env(env_right_now);
                savefig("./plots_hybrid_with_reusing_path_1d_action_space_pomdp_planner/plot_$j.png")
            else
                if(length(hybrid_a_star_path)!= 0)
                    env_right_now.cart_hybrid_astar_path = hybrid_a_star_path
                    println("**********Hybrid A Star Path found**********")
                    write(io,"**********Hybrid A Star Path found**********\n")
                else
                    println("**********Hybrid A Star Path Not found. Reusing old path**********")
                    write(io,"**********Hybrid A Star Path Not found. Reusing old path**********\n")
                end
                push!(all_observed_environments, (deepcopy(env_right_now),current_belief))
                display_env(env_right_now);
                savefig("./plots_hybrid_with_reusing_path_1d_action_space_pomdp_planner/plot_$j.png")
                m = golfcart_1D_action_space_pomdp()
                b = POMDP_1D_action_space_state_distribution(m.world,current_belief,m.start_path_index)
                a = action(planner, b)
                println("Action chosen " , a)
                write(io,"Action chosen: $a\n")
                env_right_now.cart.v = clamp(env_right_now.cart.v + a,0,m.max_cart_speed)

                if(env_right_now.cart.v != 0.0)
                    println("Current cart state = " , string(env_right_now.cart))
                    write(io,"Current cart state = $(env_right_now.cart)\n")

                    initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
                    for i in 1:Int64(env_right_now.cart.v)
                        if(i>length(env_right_now.cart_hybrid_astar_path))
                            break
                        end
                        steering_angle = env_right_now.cart_hybrid_astar_path[i]
                        extra_parameters = [env_right_now.cart.v, env_right_now.cart.L, steering_angle]
                        x,y,theta = get_intermediate_points(initial_state, 1.0/env_right_now.cart.v, extra_parameters);
                        env_right_now.cart.x, env_right_now.cart.y, env_right_now.cart.theta = last(x), last(y), last(theta)
                        initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
                    end
                    env_right_now.cart_hybrid_astar_path = env_right_now.cart_hybrid_astar_path[Int64(env_right_now.cart.v+1) : end]
                else
                    println("Current cart state = " , string(env_right_now.cart))
                    write(io,"Current cart state = $(env_right_now.cart)\n")
                end
            end
            global current_belief = update_human_position_and_current_belief(env_right_now, current_belief, lidar_range, one_time_step)
            global number_risks += get_count_number_of_risks(env_right_now)
            push!(cart_throughout_path,(copy(env_right_now.cart)))
        else
            @show("Goal reached")
            write(io,"Goal Reached!")
            close(io)
            display_env(env_right_now);
            savefig("./plots_hybrid_with_reusing_path_1d_action_space_pomdp_planner/plot_$j.png")
            break
        end
        close(io)
    end

    println("Time Taken by cart to reach goal : ", time_taken_by_cart)
    println("Number of risky scenarios encountered by the cart : ", number_risks)
    gif(anim, "original_paper_implementation.gif", fps = 2)
end


#Function used for simulating the cart one step forward in POMDP planning according to its new speed
#Function used available in two_d_action_space_pomdp.jl
#This version was removed on October 12, 2020
function update_cart_position_pomdp_planning_2D_action_space(current_cart_position, delta_angle, new_cart_speed, world)
    current_x, current_y, current_theta = current_cart_position.x, current_cart_position.y, current_cart_position.theta
    cart_path = Tuple{Float64,Float64,Float64}[]
    if(new_cart_speed == 0.0)
         push!(cart_path,(Float64(current_x), Float64(current_y), Float64(current_theta)))
    else
        arc_length = new_cart_speed
        time_interval = 1.0/new_cart_speed
        steering_angle = atan((world.cart.L*delta_angle)/arc_length)
        for i in (1:new_cart_speed)
            #@show(starting_index, length(world.cart_hybrid_astar_path))
            if(steering_angle == 0.0)
                new_theta = current_theta
                new_x = current_x + arc_length*cos(current_theta)*time_interval
                new_y = current_y + arc_length*sin(current_theta)*time_interval
            else
                new_theta = current_theta + (arc_length * tan(steering_angle) * time_interval / world.cart.L)
                new_theta = wrap_between_0_and_2Pi(new_theta)
                new_x = current_x + ((world.cart.L / tan(steering_angle)) * (sin(new_theta) - sin(current_theta)))
                new_y = current_y + ((world.cart.L / tan(steering_angle)) * (cos(current_theta) - cos(new_theta)))
            end
            new_x = clamp(new_x,0,world.length)
            new_y = clamp(new_y,0,world.breadth)
            push!(cart_path,(Float64(new_x), Float64(new_y), Float64(new_theta)))
            current_x, current_y,current_theta = new_x,new_y,new_theta
        end
    end
    #@show(current_cart_position,steering_angle, new_cart_speed, cart_path)
    return cart_path
end


#Function used for getting cart_lidar_data such that those that are inside the obstacles are not visisble.
#Function was used in main_hybrid_1d_pomdp_path_planner_reuse_old_path.jl
#This version was removed on October 12, 2020
function get_lidar_data(world,lidar_range)
    initial_cart_lidar_data = Array{human_state,1}()
    for human in world.humans
        if(is_within_range(location(world.cart.x,world.cart.y), location(human.x,human.y), lidar_range))
            inside_obstacle_flag = false
            for obstacle in world.obstacles
                if(is_within_range(location(obstacle.x,obstacle.y), location(human.x,human.y), obstacle.r))
                    inside_obstacle_flag = true
                    break
                end
            end
            if(inside_obstacle_flag == false)
                push!(initial_cart_lidar_data,human)
            end
        end
    end
    return initial_cart_lidar_data
end

#Function used to move human and update belief
#Function was used in main_hybrid_1d_pomdp_path_planner_reuse_old_path.jl
#This version was removed on October 12, 2020
function update_human_position_and_current_belief(world, old_belief, lidar_range, one_time_step)
    #Propogate humans for one time step
    world.humans = move_human_for_one_time_step_in_actual_environment(world,one_time_step)
    #Sense humans near cart after the first time step
    new_lidar_data = get_lidar_data(world,lidar_range)
    #Update belief
    updated_belief = update_belief(old_belief, world.goals,
        world.cart_lidar_data, new_lidar_data)
    world.cart_lidar_data = new_lidar_data
    # @show(env_right_now.cart_lidar_data)
    # @show(updated_belief)
    #Propogate humans for one more time step
    world.humans = move_human_for_one_time_step_in_actual_environment(world,one_time_step)
    #Sense humans near cart after the first time step
    new_lidar_data = get_lidar_data(world,lidar_range)
    #Update belief
    updated_belief = update_belief(updated_belief, world.goals,
        world.cart_lidar_data, new_lidar_data)
    world.cart_lidar_data = new_lidar_data
    # @show(env_right_now.cart_lidar_data)
    # @show(updated_belief)
    return updated_belief
end

#Function used to generate belief from old_world and new_world where new_world and old_world are 1 second apart
#As a result, we have to manually generate a temporary world that is 0.5 seconds apart from old_world to get proper belief updates.
#Function was used in main_2d_action_space_pomdp.jl
#This version was removed on October 12, 2020
function update_current_belief(old_world, new_world, old_belief, lidar_range, max_cart_lidar_size)

    #Propogate humans for one time step
    temp_world = deepcopy(old_world)
    for human_index in 1:length(old_world.humans)
        temp_world.humans[human_index].x = 0.5*(old_world.humans[human_index].x + new_world.humans[human_index].x)
        temp_world.humans[human_index].y = 0.5*(old_world.humans[human_index].y + new_world.humans[human_index].y)
    end
    #Sense humans near cart after the first time step
    new_lidar_data = get_lidar_data(temp_world,lidar_range)
    new_lidar_data = get_nearest_n_pedestrians_pomdp_planning_2D_action_space(temp_world.cart, new_lidar_data, max_cart_lidar_size)
    #Update belief
    updated_belief = update_belief(old_belief, temp_world.goals,
        temp_world.cart_lidar_data, new_lidar_data)
    temp_world.cart_lidar_data = new_lidar_data
    # @show(env_right_now.cart_lidar_data)
    # @show(updated_beltempief)

    #Propogate humans for one more time step
    #Sense humans near cart after the first time step
    new_lidar_data = get_lidar_data(new_world,lidar_range)
    new_lidar_data = get_nearest_n_pedestrians_pomdp_planning_2D_action_space(world.cart, new_lidar_data, max_cart_lidar_size)
    #Update belief
    updated_belief = update_belief(updated_belief, new_world.goals,
        temp_world.cart_lidar_data, new_lidar_data)
    new_world.cart_lidar_data = new_lidar_data
    # @show(env_right_now.cart_lidar_data)
    # @show(updated_belief)
    return updated_belief
end


# Code in main_2d_action_space_pomdp.jl  for moving humans and updating belief.
# This was inside the animate loop
function original_code_to_move_humans_and_update_belief_2d_action_space_pomdp()
    moved_human_positions = Array{human_state,1}()
    for human in env_copy.humans
        push!(moved_human_positions,get_new_human_position_actual_environemnt(human,env_copy,one_time_step))
    end
    env_copy.humans = moved_human_positions

    new_lidar_data = Array{human_state,1}()
    for human in env_copy.humans
        if(is_within_range(location(env_copy.cart.x,env_copy.cart.y), location(human.x,human.y), lidar_range))
            push!(new_lidar_data,human)
        end
    end

    global current_belief = update_belief(current_belief, env_copy.goals,
        env_copy.cart_lidar_data, new_lidar_data)
    env_copy.cart_lidar_data = new_lidar_data

    #Propogate humans for one more time step
    moved_human_positions = Array{human_state,1}()
    for human in env_copy.humans
        push!(moved_human_positions,get_new_human_position_actual_environemnt(human,env_copy,one_time_step))
    end
    env_copy.humans = moved_human_positions

    #Sense humans near cart after the first time step
    new_lidar_data = Array{human_state,1}()
    for human in env_copy.humans
        if(is_within_range(location(env_copy.cart.x,env_copy.cart.y), location(human.x,human.y), lidar_range))
            push!(new_lidar_data,human)
        end
    end

    global current_belief = update_belief(current_belief, env_copy.goals,
        env_copy.cart_lidar_data, new_lidar_data)
    env_copy.cart_lidar_data = new_lidar_data
end


# Code in main_hybrid_pomdp+_path_planner.jl  for moving humans and updating belief.
# This was outside the animate loop
function original_code_to_move_humans_and_update_belief_astar_plus_1d_pomdp()

    #Sense humans near cart before moving
    initial_cart_lidar_data = Array{human_state,1}()
    for human in env_copy.humans
        if(is_within_range(location(env_copy.cart.x,env_copy.cart.y), location(human.x,human.y), lidar_range))
            push!(initial_cart_lidar_data,human)
        end
    end
    env_copy.cart_lidar_data = initial_cart_lidar_data

    #Generate Initial Belief for humans near cart
    initial_belief = update_belief([],env_copy.goals,[],env_copy.cart_lidar_data)

    #Propogate humans for one time step
    moved_human_positions = Array{human_state,1}()
    for human in env_copy.humans
        push!(moved_human_positions,get_new_human_position_actual_environemnt(human,env_copy,one_time_step))
    end
    env_copy.humans = moved_human_positions

    #Sense humans near cart after the first time step
    new_lidar_data = Array{human_state,1}()
    for human in env_copy.humans
        if(is_within_range(location(env_copy.cart.x,env_copy.cart.y), location(human.x,human.y), lidar_range))
            push!(new_lidar_data,human)
        end
    end

    global current_belief = update_belief(initial_belief, env_copy.goals,
        env_copy.cart_lidar_data, new_lidar_data)
    env_copy.cart_lidar_data = new_lidar_data

    #Propogate humans for one more time step
    moved_human_positions = Array{human_state,1}()
    for human in env_copy.humans
        push!(moved_human_positions,get_new_human_position_actual_environemnt(human,env_copy,one_time_step))
    end
    env_copy.humans = moved_human_positions

    #Sense humans near cart after the first time step
    new_lidar_data = Array{human_state,1}()
    for human in env_copy.humans
        if(is_within_range(location(env_copy.cart.x,env_copy.cart.y), location(human.x,human.y), lidar_range))
            push!(new_lidar_data,human)
        end
    end

    global current_belief = update_belief(current_belief, env_copy.goals,
        env_copy.cart_lidar_data, new_lidar_data)
    env_copy.cart_lidar_data = new_lidar_data
end


#Function was used to run simulation in main_hybrid_1d_pomdp_path_planner_reuse_old_path.jl
#This version was removed on October 12, 2020
function run_one_simulation_old(env_right_now)

    time_taken_by_cart = 0
    number_risks = 0
    one_time_step = 0.5
    lidar_range = 30
    num_humans_to_care_about_while_pomdp_planning = 10
    cone_half_angle = pi/3.0
    number_of_sudden_stops = 0
    cart_ran_into_boundary_wall_near_goal_flag = false
    filename = "output_resusing_old_hybrid_astar_path_1D_action_space_speed_pomdp_planner.txt"
    user_defined_rng = MersenneTwister(7)
    cart_throughout_path = []
    all_gif_environments = []
    all_observed_environments = []
    all_generated_beliefs = []
    all_generated_trees = []
    all_risky_scenarios = []

    push!(all_observed_environments, ("0",deepcopy(env_right_now)))
    push!(all_generated_beliefs, [])
    push!(all_generated_trees, nothing)

    #Sense humans near cart before moving
    #Generate Initial Lidar Data and Belief for humans near cart
    env_right_now.cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
    initial_belief = update_belief([],env_right_now.goals,[],env_right_now.cart_lidar_data)
    #Update human positions in environment and cart's belief
    current_belief = update_human_position_and_current_belief(env_right_now, initial_belief, lidar_range, one_time_step)
    if(get_count_number_of_risks(env_right_now) != 0)
        number_risks += get_count_number_of_risks(env_right_now)
        push!(all_risky_scenarios,deepcopy(env_right_now))
    end
    push!(all_observed_environments, ("1",deepcopy(env_right_now)))
    push!(all_generated_beliefs, current_belief)
    push!(all_generated_trees, nothing)

    io = open(filename,"a")
    write(io, "Simulating for time interval - (" * string(time_taken_by_cart-1) * " , " * string(time_taken_by_cart) * ")")
    write(io,"Current cart state = $(env_right_now.cart)\n")
    write(io,"Modified cart state = $(env_right_now.cart)\n")
    write(io,"************************************************************************\n")
    println("Simulating for time interval - (" * string(time_taken_by_cart-1) * " , " * string(time_taken_by_cart) * ")")
    println("Current cart state = " , string(env_right_now.cart))
    println("Modified cart state = " , string(env_right_now.cart))
    println("************************************************************************")
    close(io)

    while(!is_within_range(location(env_right_now.cart.x,env_right_now.cart.y), env_right_now.cart.goal, 1.0))
        io = open(filename,"a")
        time_taken_by_cart += 1
        write(io, "Simulating for time interval - (" * string(time_taken_by_cart-1) * " , " * string(time_taken_by_cart) * ")")
        write(io,"Current cart state = $(env_right_now.cart)\n")
        println("Simulating for time interval - (" * string(time_taken_by_cart-1) * " , " * string(time_taken_by_cart) * ")")
        println("Current cart state = " , string(env_right_now.cart))

        humans_to_avoid = get_nearest_n_pedestrians_hybrid_astar_search(env_right_now,current_belief,6)
        hybrid_a_star_path = @time hybrid_a_star_search(env_right_now.cart.x, env_right_now.cart.y,
            env_right_now.cart.theta, env_right_now.cart.goal.x, env_right_now.cart.goal.y, env_right_now, humans_to_avoid);

        if( (length(hybrid_a_star_path) == 0) && (length(env_right_now.cart_hybrid_astar_path) == 0) )
            println("**********Hybrid A Star Path Not found. No old path exists either**********")
            write(io,"**********Hybrid A Star Path Not found. No old path exists either**********\n")
            env_right_now.cart.v = 0.0
            current_belief = update_human_position_and_current_belief(env_right_now, current_belief, lidar_range, one_time_step)
            push!( all_observed_environments, (string(time_taken_by_cart)*"_"*string(0),deepcopy(env_right_now)) )
            push!(all_generated_beliefs, current_belief)
            push!(all_generated_trees, nothing)
            if(get_count_number_of_risks(env_right_now) != 0)
                number_risks += get_count_number_of_risks(env_right_now)
                push!(all_risky_scenarios,deepcopy(env_right_now))
            end
        else
            if(length(hybrid_a_star_path)!= 0)
                env_right_now.cart_hybrid_astar_path = hybrid_a_star_path
                println("**********Hybrid A Star Path found**********")
                write(io,"**********Hybrid A Star Path found**********\n")
            else
                println("**********Hybrid A Star Path Not found. Reusing old path**********")
                write(io,"**********Hybrid A Star Path Not found. Reusing old path**********\n")
            end
            m = golfcart_1D_action_space_pomdp()
            b = POMDP_1D_action_space_state_distribution(m.world,current_belief,m.start_path_index)
            a, info = action_info(planner, b)
            push!(all_generated_trees, info)

            println("Action chosen by 1D action space speed POMDP planner: " , a)
            write(io,"Action chosen by 1D action space speed POMDP planner: $a\n")
            if(env_right_now.cart.v!=0 && a==-10.0)
                number_of_sudden_stops += 1
            end
            old_env = deepcopy(env_right_now)
            env_right_now.cart.v = clamp(env_right_now.cart.v + a,0,m.max_cart_speed)

            if(env_right_now.cart.v != 0.0)
                initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
                for i in 1:Int64(env_right_now.cart.v)
                    if(length(env_right_now.cart_hybrid_astar_path) == 0)
                        break
                    end
                    steering_angle = env_right_now.cart_hybrid_astar_path[1]
                    extra_parameters = [env_right_now.cart.v, env_right_now.cart.L, steering_angle]
                    x,y,theta = get_intermediate_points(initial_state, 1.0/env_right_now.cart.v, extra_parameters);
                    env_right_now.cart.x, env_right_now.cart.y, env_right_now.cart.theta = last(x), last(y), last(theta)
                    env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,1.0/env_right_now.cart.v)
                    env_right_now.cart_hybrid_astar_path = env_right_now.cart_hybrid_astar_path[2 : end]
                    initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
                    push!( all_observed_environments, (string(time_taken_by_cart)*"_"*string(i),deepcopy(env_right_now)) )
                    if(get_count_number_of_risks(env_right_now) != 0)
                        number_risks += get_count_number_of_risks(env_right_now)
                        push!(all_risky_scenarios,deepcopy(env_right_now))
                    end
                end
                current_belief = update_current_belief(old_env, env_right_now, current_belief, lidar_range)
                push!(all_generated_beliefs, current_belief)
            else
                current_belief = update_human_position_and_current_belief(env_right_now, current_belief, lidar_range, one_time_step)
                push!( all_observed_environments, (string(time_taken_by_cart)*"_"*string(0),deepcopy(env_right_now)) )
                push!(all_generated_beliefs, current_belief)
                if(get_count_number_of_risks(env_right_now) != 0)
                    number_risks += get_count_number_of_risks(env_right_now)
                    push!(all_risky_scenarios,deepcopy(env_right_now))
                end
            end
        end
        push!(cart_throughout_path,(copy(env_right_now.cart)))
        write(io,"Modified cart state = $(env_right_now.cart)\n")
        write(io,"************************************************************************\n")
        println("Modified cart state = " , string(env_right_now.cart))
        println("************************************************************************")
        close(io)
    end
    io = open(filename,"a")
    println("Goal reached")
    println("Time Taken by cart to reach goal : ", time_taken_by_cart)
    println("Number of risky scenarios encountered by the cart : ", number_risks)
    println("Number of sudden stops taken by the cart : ", number_of_sudden_stops)
    write(io,"Goal Reached!")
    write(io,"Time Taken by cart to reach goal :  $time_taken_by_cart\n")
    write(io,"Number of risky scenarios encountered by the cart : $number_risks\n")
    write(io,"Number of sudden stops taken by the cart : $number_of_sudden_stops\n")
    close(io)

    return all_observed_environments, all_generated_beliefs, all_generated_trees,all_risky_scenarios,number_risks,number_of_sudden_stops,time_taken_by_cart
end


function closeness_to_walls_penalty(x,y,dist_threshold,close_to_wall_penalty)
    if( (abs(x-100.0) < dist_threshold) || (abs(y-100.0) < dist_threshold) || (x < dist_threshold) || (y < dist_threshold) )
        return close_to_wall_penalty
    else
        return 0.0
    end
end

# Function not needed because it uses obstacle_collision_flag to check if wall collision occured or not and the
# obstacle_collision_penalty_pomdp_planning_2D_action_space function already does that for you.
function wall_collision_penalty_pomdp_planning_2D_action_space(wall_collision_flag, penalty)
    if( wall_collision_flag )
        return penalty
    else
        return 0.0
    end
end


# Function to calculate the lower bound policy for DESPOT.
# This function found the best possile action out of existing ones and executed that.
#In the new approach, we don't need that anymore. We have modified the action space to have a delta_angle that aligns you with the orientaton to goal directly.
function calculate_lower_bound_policy_pomdp_planning_2D_action_space(b)
    #Implement a reactive controller for your lower bound
    speed_change_to_be_returned = 1.0
    best_delta_angle = 0.0
    d_far_threshold = 5.0
    d_near_threshold = 2.0
    #This bool is also used to check if all the states in the belief are terminal or not.
    first_execution_flag = true

    for (s, w) in weighted_particles(b)
        if(s.cart.x == -100.0 && s.cart.y == -100.0)
            continue
        else
            if(first_execution_flag)
                direct_line_to_goal_angle = wrap_between_0_and_2Pi(atan(s.cart.goal.y-s.cart.y,s.cart.goal.x-s.cart.x))
                delta_angles = Float64[-pi/4, -pi/6, -pi/12, 0.0, pi/12, pi/6 , pi/4]
                best_delta_angle = delta_angles[1]
                best_dot_product_value_so_far = dot( ( cos(direct_line_to_goal_angle), sin(direct_line_to_goal_angle) )
                                   , ( cos(s.cart.theta+delta_angles[1]), sin(s.cart.theta+delta_angles[1]) ) )
                for i in 2:length(delta_angles)
                     dot_prodcut = dot( ( cos(direct_line_to_goal_angle), sin(direct_line_to_goal_angle) )
                                        , ( cos(s.cart.theta+delta_angles[i]), sin(s.cart.theta+delta_angles[i]) ) )
                     if(dot_prodcut > best_dot_product_value_so_far)
                         best_dot_product_value_so_far = dot_prodcut
                         best_delta_angle = delta_angles[i]
                     end
                end
                first_execution_flag = false
            else
                dist_to_closest_human = 200.0  #Some really big infeasible number (not Inf because avoid the type mismatch error)
                for human in s.pedestrians
                    euclidean_distance = sqrt((s.cart.x - human.x)^2 + (s.cart.y - human.y)^2)
                    if(euclidean_distance < dist_to_closest_human)
                        dist_to_closest_human = euclidean_distance
                    end
                    if(dist_to_closest_human < d_near_threshold)
                        return (0.0,-1.0)
                    end
                end
                if(dist_to_closest_human > d_far_threshold)
                    chosen_acceleration = 1.0
                else
                    chosen_acceleration = 0.0
                end
                if(chosen_acceleration < speed_change_to_be_returned)
                    speed_change_to_be_returned = chosen_acceleration
                end
            end
        end
    end

    #This condition is true only when all the states in the belief are terminal. In that case, just return (0.0,0.0)
    if(first_execution_flag == true)
        #@show(0.0,0.0)
        return (0.0,0.0)
    end

    #This means all humans are away and you can accelerate.
    if(speed_change_to_be_returned == 1.0)
        #@show(0.0,speed_change_to_be_returned)
        return (0.0,speed_change_to_be_returned)
    end

    #If code has reached this point, then the best action is to maintain your current speed.
    #We have already found the best steering angle to take.
    #@show(best_delta_angle,0.0)
    return (best_delta_angle,0.0)
end
