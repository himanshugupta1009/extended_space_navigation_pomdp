#Various different miscellaneous functions that are needed by different components and are common to multiple files

function is_within_range_check_with_points(p1_x,p1_y, p2_x, p2_y, threshold_distance)
    euclidean_distance = ((p1_x - p2_x)^2 + (p1_y - p2_y)^2)^0.5
    if(euclidean_distance<=threshold_distance)
        return true
    else
        return false
    end
end

function is_within_range(location1, location2, threshold_distance)
    euclidean_distance = ((location1.x - location2.x)^2 + (location1.y - location2.y)^2)^0.5
    if(euclidean_distance<=threshold_distance)
        return true
    else
        return false
    end
end
#@code_warntype is_within_range(location(0,0,-1), location(3,4,-1), 1)

function wrap_between_0_and_2Pi(theta)
   return mod(theta,2*pi)
end

function travel!(du,u,p,t)
    x,y,theta = u
    v,L,alpha = p

    du[1] = v*cos(theta)
    du[2] = v*sin(theta)
    du[3] = (v/L)*tan(alpha)
end

function get_intermediate_points(initial_state, time_interval, extra_parameters)
    prob = ODEProblem(travel!,initial_state,time_interval,extra_parameters)
    sol = DifferentialEquations.solve(prob,saveat=0.1)
    x = []
    y = []
    theta = []

    for i in 1:length(sol.u)
        push!(x,sol.u[i][1])
        push!(y,sol.u[i][2])
        push!(theta,wrap_between_0_and_2Pi(sol.u[i][3]))
    end

    return x,y,theta
end

function get_heading_angle(human_x, human_y, cart_x, cart_y)

    #First Quadrant
    if(human_x >= cart_x && human_y >= cart_y)
        if(human_x == cart_x)
            heading_angle = pi/2.0
        elseif(human_y == cart_y)
            heading_angle = 0.0
        else
            heading_angle = atan((human_y - cart_y) / (human_x - cart_x))
        end
    #Second Quadrant
    elseif(human_x <= cart_x && human_y >= cart_y)
        if(human_x == cart_x)
            heading_angle = pi/2.0
        elseif(human_y == cart_y)
            heading_angle = pi/1.0
        else
            heading_angle = atan((human_y - cart_y) / (human_x - cart_x)) + pi
        end
    #Third Quadrant
    elseif(human_x <= cart_x && human_y <= cart_y)
        if(human_x == cart_x)
            heading_angle = 3*pi/2.0
        elseif(human_y == cart_y)
            heading_angle = pi/1.0
        else
            heading_angle = atan((human_y - cart_y) / (human_x - cart_x)) + pi
        end
    #Fourth Quadrant
    else(human_x >= cart_x && human_y <= cart_y)
        if(human_x == cart_x)
            heading_angle = 3*pi/2.0
        elseif(human_y == cart_y)
            heading_angle = 0.0
        else
            heading_angle = 2.0*pi + atan((human_y - cart_y) / (human_x - cart_x))
        end
    end

    return heading_angle
end

function get_lidar_data(world,lidar_range)
    initial_cart_lidar_data = Array{human_state,1}()
    for human in world.humans
        if(is_within_range(location(world.cart.x,world.cart.y), location(human.x,human.y), lidar_range))
            if(human.x!= human.goal.x || human.y!= human.goal.y)
                push!(initial_cart_lidar_data,human)
            end
        end
    end
    return initial_cart_lidar_data
end

#Function for actually moving human in the environment
function get_new_human_position_actual_environemnt(human, world, time_step, user_defined_rng)

    rand_num = (rand(user_defined_rng) - 0.5)*0.1
    #rand_num = 0.0
    #First Quadrant
    if(human.goal.x >= human.x && human.goal.y >= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y + (human.v)*time_step + rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x + (human.v)*time_step + rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x + ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y + ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    #Second Quadrant
    elseif(human.goal.x <= human.x && human.goal.y >= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y + (human.v)*time_step + rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x - (human.v)*time_step - rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x - ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y - ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    #Third Quadrant
    elseif(human.goal.x <= human.x && human.goal.y <= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y - (human.v)*time_step - rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x - (human.v)*time_step - rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x - ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y - ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    #Fourth Quadrant
    else(human.goal.x >= human.x && human.goal.y <= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y - (human.v)*time_step - rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x + (human.v)*time_step + rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x + ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y + ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    end

    new_x = clamp(new_x,0,world.length)
    new_y = clamp(new_y,0,world.breadth)
    #@show(new_x,new_y)
    new_human_state = human_state(new_x, new_y, human.v, human.goal,human.id)
    return new_human_state
end

function move_human_for_one_time_step_in_actual_environment(world,time_step,user_defined_rng)
    moved_human_positions = Array{human_state,1}()
    for human in world.humans
        if(human.x == human.goal.x && human.y==human.goal.y)
            new_human_state = human_state(human.x,human.y,human.v,human.goal,human.id)
        elseif (is_within_range(location(human.x,human.y), human.goal, 1.0))
            new_human_state = human_state(human.goal.x,human.goal.y,human.v,human.goal,human.id)
        else
            new_human_state = get_new_human_position_actual_environemnt(human,world,time_step,user_defined_rng)
        end
        push!(moved_human_positions,new_human_state)
    end
    return moved_human_positions
end

function get_count_number_of_risks(world)
    risks = 0
    if(world.cart.v>=1.0)
        for human in world.cart_lidar_data
            euclidean_distance = sqrt( (human.x - world.cart.x)^2 + (human.y - world.cart.y)^2 )
            if(euclidean_distance<=0.3)
                println( "A risky scenario encountered and the distance is : ", euclidean_distance )
                risks += 1
            end
        end
    else
        return 0;
    end
    return risks;
end

function write_and_print(io::IOStream, string_to_be_written_and_printed::String)
    write(io, string_to_be_written_and_printed * "\n")
    println(string_to_be_written_and_printed)
end

function get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(cart, cart_lidar_data, n, cone_half_angle::Float64=pi/3.0)
    nearest_n_pedestrians = Array{human_state,1}()
    priority_queue_nearest_n_pedestrians = PriorityQueue{human_state,Float64}(Base.Order.Forward)
    for i in 1:length(cart_lidar_data)
        human = cart_lidar_data[i]
        angle_between_cart_and_human = get_heading_angle(human.x, human.y, cart.x, cart.y)
        difference_in_angles = abs(cart.theta - angle_between_cart_and_human)
        if(difference_in_angles <= cone_half_angle)
            euclidean_distance = sqrt( (cart.x - human.x)^2 + (cart.y - human.y)^2 )
            priority_queue_nearest_n_pedestrians[human] = euclidean_distance
        elseif ( (2*pi - difference_in_angles) <= cone_half_angle )
            euclidean_distance = sqrt( (cart.x - human.x)^2 + (cart.y - human.y)^2 )
            priority_queue_nearest_n_pedestrians[human] = euclidean_distance
        else
            continue
        end
    end

    priority_queue_nearest_n_pedestrians_ordered_by_id = PriorityQueue{human_state,Float64}(Base.Order.Forward)
    for i in 1:n
        if(length(priority_queue_nearest_n_pedestrians) != 0)
            pedestrian = dequeue!(priority_queue_nearest_n_pedestrians)
            priority_queue_nearest_n_pedestrians_ordered_by_id[pedestrian] = pedestrian.id
        else
            break
        end
    end

    for pedestrian in priority_queue_nearest_n_pedestrians_ordered_by_id
        push!(nearest_n_pedestrians,dequeue!(priority_queue_nearest_n_pedestrians_ordered_by_id))
    end

    return nearest_n_pedestrians
end

function get_nearest_n_pedestrians_hybrid_astar_search(world,current_belief,n,cone_half_angle::Float64=pi/3.0)
    nearest_n_pedestrians = Array{Tuple{human_state,human_probability_over_goals},1}()
    priority_queue_nearest_n_pedestrians = PriorityQueue{Tuple{human_state,human_probability_over_goals},Float64}(Base.Order.Forward)
    cone_half_angle = pi/3.0
    for i in 1:length(world.cart_lidar_data)
        human = world.cart_lidar_data[i]
        angle_between_cart_and_human = get_heading_angle(human.x, human.y, world.cart.x, world.cart.y)
        difference_in_angles = abs(world.cart.theta - angle_between_cart_and_human)
        if(difference_in_angles <= cone_half_angle)
            euclidean_distance = sqrt( (world.cart.x - human.x)^2 + (world.cart.y - human.y)^2 )
            priority_queue_nearest_n_pedestrians[(human,current_belief[i])] = euclidean_distance
        elseif ( (2*pi - difference_in_angles) <= cone_half_angle )
            euclidean_distance = sqrt( (world.cart.x - human.x)^2 + (world.cart.y - human.y)^2 )
            priority_queue_nearest_n_pedestrians[(human,current_belief[i])] = euclidean_distance
        else
            continue
        end
    end
    for i in 1:n
        if(length(priority_queue_nearest_n_pedestrians) != 0)
            push!(nearest_n_pedestrians,dequeue!(priority_queue_nearest_n_pedestrians))
        else
            break
        end
    end
    return nearest_n_pedestrians
end

# Function that updates the belief based on cart_lidar_data of old world and
# cart_lidar_data of the new world
function update_belief_from_old_world_and_new_world(current_belief, old_world, new_world)
    updated_belief = update_belief(current_belief, old_world.goals,
        old_world.cart_lidar_data, new_world.cart_lidar_data)
    return updated_belief
end

function update_current_belief_by_creating_temp_world(old_world, new_world, old_belief, lidar_range,
                                                        num_humans_to_care_about, cone_half_angle)

    #Create the temp world at t = 0.5 second
    temp_world = deepcopy(old_world)
    for human_index in 1:length(old_world.humans)
        temp_world.humans[human_index].x = 0.5*(old_world.humans[human_index].x + new_world.humans[human_index].x)
        temp_world.humans[human_index].y = 0.5*(old_world.humans[human_index].y + new_world.humans[human_index].y)
    end

    new_lidar_data = get_lidar_data(temp_world,lidar_range)
    new_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(temp_world.cart, new_lidar_data,
                                                                                num_humans_to_care_about, cone_half_angle )
    #Update belief
    temp_world.cart_lidar_data = new_lidar_data
    updated_belief = update_belief(old_belief, temp_world.goals,
        old_world.cart_lidar_data, temp_world.cart_lidar_data)

    final_updated_belief = update_belief(updated_belief, new_world.goals,
        temp_world.cart_lidar_data, new_world.cart_lidar_data)

    return final_updated_belief
end

function respawn_humans_in_environment(world, lidar_range, num_humans_to_care_about_while_pomdp_planning,
                                                        cone_half_angle, current_belief, user_defined_rng_for_env)

    old_world = deepcopy(world)
    num_humans_already_at_goal = 0
    for human in world.humans
        if( (human.x == human.goal.x) && (human.y == human.goal.y) )
            num_humans_already_at_goal += 1
        end
    end

    total_number_of_humans = length(world.humans)
    total_humans_not_at_goal = total_number_of_humans - num_humans_already_at_goal

    if( total_humans_not_at_goal < world.num_humans)
        for i in 1:total_humans_not_at_goal
            rand_num = rand(user_defined_rng_for_env)
            if(rand_num > 0.5)
                x = world.length
                y = floor(100*rand_num)
                while(is_within_range_check(x,y,world.cart.goal.x,world.cart.goal.y,5.0))
                    y = floor(100*rand(user_defined_rng))
                end
                new_human_state = human_state(x,y,1.0,world.goals[Int(ceil(rand(user_defined_rng)*4))], float(length(world.humans) + 1))
            else
                x = 0.0
                y = floor(100*rand_num)
                while(is_within_range_check(x,y,world.cart.goal.x,world.cart.goal.y,5.0))
                    y = floor(100*rand(user_defined_rng))
                end
                new_human_state = human_state(x,y,1.0,world.goals[Int(ceil(rand(user_defined_rng)*4))], float(length(world.humans) + 1))
            end
            push!(world.humans,new_human_state)
        end
        world.cart_lidar_data = get_lidar_data(world,lidar_range)
        world.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(world.cart,
                                                            world.cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                            cone_half_angle)

        updated_belief = update_belief_from_old_world_and_new_world(current_belief, old_world, env_right_now)

    else
        updated_belief = current_belief
    end

    return current_belief
end
