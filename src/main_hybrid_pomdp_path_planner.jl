include("environment.jl")
include("utils.jl")
include("hybrid_a_star.jl")
include("one_d_action_space_close_waypoint_pomdp.jl")
include("belief_tracker.jl")

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

function get_nearest_six_pedestrians(world,current_belief)
    nearest_six_pedestrians = Array{Tuple{human_state,human_probability_over_goals},1}()
    priority_queue_nearest_six_pedestrians = PriorityQueue{Tuple{human_state,human_probability_over_goals},Float64}(Base.Order.Forward)
    cone_angle = pi/3.0
    for i in 1:length(world.cart_lidar_data)
        human = world.cart_lidar_data[i]
        angle_between_cart_and_human = get_heading_angle(human.x, human.y, world.cart.x, world.cart.y)
        difference_in_angles = abs(world.cart.theta - angle_between_cart_and_human)
        if(difference_in_angles <= cone_angle)
            euclidean_distance = sqrt( (world.cart.x - human.x)^2 + (world.cart.y - human.y)^2 )
            priority_queue_nearest_six_pedestrians[(human,current_belief[i])] = euclidean_distance
        elseif ( (2*pi - difference_in_angles) <= cone_angle )
            euclidean_distance = sqrt( (world.cart.x - human.x)^2 + (world.cart.y - human.y)^2 )
            priority_queue_nearest_six_pedestrians[(human,current_belief[i])] = euclidean_distance
        else
            continue
        end
    end
    for i in 1:6
        if(length(priority_queue_nearest_six_pedestrians) != 0)
            push!(nearest_six_pedestrians,dequeue!(priority_queue_nearest_six_pedestrians))
        else
            break
        end
    end
    return nearest_six_pedestrians
end

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

function move_human_for_one_time_step_in_actual_environment(world,time_step)
    moved_human_positions = Array{human_state,1}()
    for human in world.humans
        push!(moved_human_positions,get_new_human_position_actual_environemnt(human,world,time_step))
    end
    return moved_human_positions
end

function update_human_position_and_current_belief(world, old_belief, lidar_range, one_time_step)
    #Propogate humans for one time step
    world.humans = move_human_for_one_time_step_in_actual_environment(world,one_time_step)
    #Sense humans near cart after the first time step
    new_lidar_data = get_lidar_data(world,lidar_range)
    #Update belief
    updated_belief = update_belief(old_belief, world.goals,
        world.cart_lidar_data, new_lidar_data)
    world.cart_lidar_data = new_lidar_data
    # @show(env_copy.cart_lidar_data)
    # @show(updated_belief)
    #Propogate humans for one more time step
    world.humans = move_human_for_one_time_step_in_actual_environment(world,one_time_step)
    #Sense humans near cart after the first time step
    new_lidar_data = get_lidar_data(world,lidar_range)
    #Update belief
    updated_belief = update_belief(updated_belief, world.goals,
        world.cart_lidar_data, new_lidar_data)
    world.cart_lidar_data = new_lidar_data
    # @show(env_copy.cart_lidar_data)
    # @show(updated_belief)
    return updated_belief
end

function get_count_number_of_risks(world)
    risks = 0
    if(world.cart.v>1.0)
        for human in world.cart_lidar_data
            euclidean_distance = sqrt( (human.x - world.cart.x)^2 + (human.y - world.cart.y)^2 )
            if(euclidean_distance<=0.5)
                risks += 1
            end
        end
    else
        return 0;
    end
    return risks;
end


Base.copy(s::cart_state) = cart_state(s.x, s.y,s.theta,s.v,s.L,s.goal)
global i = 1
global time_taken_by_cart = 1
global number_risks = 0
cart_throughout_path = []
one_time_step = 0.5
env_copy = deepcopy(env)
all_observed_environments = []
lidar_range = 30

#Create new POMDP for handling hybrid_a_star + POMDP speed planner at every time step
golfcart_1D_action_space_pomdp() = POMDP_Planner_1D_action_space(0.9,2.0,-1000.0,1.0,1.0,100.0,7.0,env_copy,1)
discount(p::POMDP_Planner_1D_action_space) = p.discount_factor
isterminal(::POMDP_Planner_1D_action_space, s::POMDP_state_1D_action_space) = isgoalstate_pomdp_planning(s,terminal_cart_state);
actions(::POMDP_Planner_1D_action_space) = [-1.0, 0.0, 1.0, -10.0]

solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning)),
        calculate_upper_bound_value_pomdp_planning, check_terminal=true),K=100,D=50,T_max=0.5, tree_in_info=true)
planner = POMDPs.solve(solver, golfcart_1D_action_space_pomdp());
m = golfcart_1D_action_space_pomdp()

#Sense humans near cart before moving
#Generate Initial Lidar Data and Belief for humans near cart
env_copy.cart_lidar_data = get_lidar_data(env_copy,lidar_range)
initial_belief = update_belief([],env_copy.goals,[],env_copy.cart_lidar_data)
#Update human positions in environment and cart's belief
global current_belief = update_human_position_and_current_belief(env_copy, initial_belief, lidar_range, one_time_step)
global number_risks += get_count_number_of_risks(env_copy)


anim = @animate for j ∈ 1:10000
    io = open("output.txt","a")
    @show(j)
    if(!is_within_range(location(env_copy.cart.x,env_copy.cart.y), env_copy.cart.goal, 1.0))
        global time_taken_by_cart += 1
        humans_to_avoid = get_nearest_six_pedestrians(env_copy,current_belief)
        env_copy.cart_hybrid_astar_path = @time hybrid_a_star_search(env_copy.cart.x, env_copy.cart.y,
            env_copy.cart.theta, env_copy.cart.goal.x, env_copy.cart.goal.y, env_copy, humans_to_avoid);
        push!(all_observed_environments, (deepcopy(env_copy),current_belief))
        display_env(env_copy);
        savefig("./plots/plot_$j.png")
        write(io,"Current Iteration Number - $j\n")
        if(length(env_copy.cart_hybrid_astar_path) == 0)
            println("**********Hybrid A Star PAth Not found. Vehicle Stopped**********")
            write(io,"**********Hybrid A Star PAth Not found. Vehicle Stopped**********\n")
            env_copy.cart.v = 0.0
        else
            m = golfcart_1D_action_space_pomdp()
            b = POMDP_1D_action_space_state_distribution(m.world,current_belief,m.start_path_index)
            a = action(planner, b)
            println("Action chosen " , a)
            write(io,"Action chosen: $a\n")
            env_copy.cart.v = clamp(env_copy.cart.v + a,0,m.max_cart_speed)

            if(env_copy.cart.v != 0.0)
                println("Current cart state = " , string(env_copy.cart))
                write(io,"Current cart state = $(env_copy.cart)\n")

                initial_state = [env_copy.cart.x,env_copy.cart.y,env_copy.cart.theta]
                for i in 1:Int64(env_copy.cart.v)
                    if(i>length(env_copy.cart_hybrid_astar_path))
                        break
                    end
                    steering_angle = env_copy.cart_hybrid_astar_path[i]
                    extra_parameters = [env_copy.cart.v, env_copy.cart.L, steering_angle]
                    x,y,theta = get_intermediate_points(initial_state, 1.0/env_copy.cart.v, extra_parameters);
                    env_copy.cart.x, env_copy.cart.y, env_copy.cart.theta = last(x), last(y), last(theta)
                    initial_state = [env_copy.cart.x,env_copy.cart.y,env_copy.cart.theta]
                end
            else
                println("Current cart state = " , string(env_copy.cart))
                write(io,"Current cart state = $(env_copy.cart)\n")
            end
        end
        global current_belief = update_human_position_and_current_belief(env_copy, current_belief, lidar_range, one_time_step)
        global number_risks += get_count_number_of_risks(env_copy)
        push!(cart_throughout_path,(copy(env_copy.cart)))
    else
        @show("Goal reached")
        write(io,"Goal Reached!")
        close(io)
        display_env(env_copy);
        savefig("./plots/plot_$j.png")
        break
    end
    close(io)
end

println("Time Taken by cart to reach goal : ", time_taken_by_cart)
println("Number of risky scenarios encountered by the cart : ", number_risks)
gif(anim, "original_paper_implementation.gif", fps = 2)
