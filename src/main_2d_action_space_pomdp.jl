include("environment.jl")
include("utils.jl")
include("two_d_action_space_pomdp.jl")
include("belief_tracker.jl")
using DataStructures

Base.copy(s::cart_state) = cart_state(s.x, s.y,s.theta,s.v,s.L,s.goal)

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

function get_nearest_n_pedestrians_pomdp_planning_2D_action_space(cart, cart_lidar_data, n)
    nearest_n_pedestrians = Array{human_state,1}()
    priority_queue_nearest_n_pedestrians = PriorityQueue{human_state,Float64}(Base.Order.Forward)
    cone_angle = pi/3.0
    for i in 1:length(cart_lidar_data)
        human = cart_lidar_data[i]
        angle_between_cart_and_human = get_heading_angle(human.x, human.y, cart.x, cart.y)
        difference_in_angles = abs(cart.theta - angle_between_cart_and_human)
        if(difference_in_angles <= cone_angle)
            euclidean_distance = sqrt( (cart.x - human.x)^2 + (cart.y - human.y)^2 )
            priority_queue_nearest_n_pedestrians[human] = euclidean_distance
        elseif ( (2*pi - difference_in_angles) <= cone_angle )
            euclidean_distance = sqrt( (cart.x - human.x)^2 + (cart.y - human.y)^2 )
            priority_queue_nearest_n_pedestrians[human] = euclidean_distance
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

function get_lidar_data(world,lidar_range)
    initial_cart_lidar_data = Array{human_state,1}()
    for human in world.humans
        if(is_within_range(location(world.cart.x,world.cart.y), location(human.x,human.y), lidar_range))
            push!(initial_cart_lidar_data,human)
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

function update_human_position_and_current_belief(world, old_belief, lidar_range, max_cart_lidar_size, one_time_step)
    #Propogate humans for one time step
    world.humans = move_human_for_one_time_step_in_actual_environment(world,one_time_step)
    #Sense humans near cart after the first time step
    new_lidar_data = get_lidar_data(world,lidar_range)
    new_lidar_data = get_nearest_n_pedestrians_pomdp_planning_2D_action_space(world.cart, new_lidar_data, max_cart_lidar_size)
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
    new_lidar_data = get_nearest_n_pedestrians_pomdp_planning_2D_action_space(world.cart, new_lidar_data, max_cart_lidar_size)
    #Update belief
    updated_belief = update_belief(updated_belief, world.goals,
        world.cart_lidar_data, new_lidar_data)
    world.cart_lidar_data = new_lidar_data
    world.cart_lidar_data = get_nearest_n_pedestrians(world,10)
    # @show(env_right_now.cart_lidar_data)
    # @show(updated_belief)
    return updated_belief
end

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

function get_count_number_of_risks(world)
    risks = 0
    if(world.cart.v>=1.0)
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

function run_one_simulation(env_right_now)

    time_taken_by_cart = 1
    number_risks = 0
    one_time_step = 0.5
    lidar_range = 30
    cart_throughout_path = []
    all_gif_environments = []
    all_observed_environments = []
    all_generated_beliefs = []
    all_generated_trees = []
    all_risky_scenarios = []
    filename = "output_just_2d_action_space_pomdp_planner.txt"

    #Sense humans near cart before moving
    #Generate Initial Lidar Data and Belief for humans near cart
    env_right_now.cart_lidar_data = get_lidar_data(env_right_now,lidar_range)
    env_right_now.cart_lidar_data = get_nearest_n_pedestrians(env_right_now,10)
    initial_belief = update_belief([],env_right_now.goals,[],env_right_now.cart_lidar_data)
    push!(all_gif_environments, ("0",deepcopy(env_right_now)))
    push!(all_observed_environments,deepcopy(env_right_now))
    push!(all_generated_beliefs, initial_belief)
    push!(all_generated_trees, nothing)

    #Update human positions in environment for two time steps and cart's belief accordingly
    current_belief = update_human_position_and_current_belief(env_right_now, initial_belief, lidar_range, one_time_step)
    if(get_count_number_of_risks(env_right_now) != 0)
        number_risks += get_count_number_of_risks(env_right_now)
        push!(all_risky_scenarios,deepcopy(env_right_now))
    end
    push!(all_gif_environments, ("1",deepcopy(env_right_now)))
    push!(all_observed_environments,deepcopy(env_right_now))
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
        if( (env_right_now.cart.x<=100.0 && env_right_now.cart.y<=100.0 && env_right_now.cart.x>=0.0 && env_right_now.cart.y>=0.0) )

            write(io, "Simulating for time interval - (" * string(time_taken_by_cart-1) * " , " * string(time_taken_by_cart) * ")")
            write(io,"Current cart state = $(env_right_now.cart)\n")
            println("Simulating for time interval - (" * string(time_taken_by_cart-1) * " , " * string(time_taken_by_cart) * ")")
            println("Current cart state = " , string(env_right_now.cart))

            m = golfcart_2D_action_space_pomdp()
            b = POMDP_2D_action_space_state_distribution(m.world,current_belief)
            a, info = action_info(planner, b)
            push!(all_generated_trees, info)

            println("Action chosen by 2D action space POMDP planner: " , a)
            write(io,"Action chosen by 2D action space POMDP planner: $a\n")

            old_env = deepcopy(env_right_now)
            env_right_now.cart.v = clamp(env_right_now.cart.v + a[2],0,m.max_cart_speed)

            if(env_right_now.cart.v != 0.0)
                initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
                steering_angle = atan((env_right_now.cart.L*a[1])/env_right_now.cart.v)
                for i in 1:Int64(env_right_now.cart.v)
                    extra_parameters = [env_right_now.cart.v, env_right_now.cart.L, steering_angle]
                    x,y,theta = get_intermediate_points(initial_state, 1.0/env_right_now.cart.v, extra_parameters);
                    env_right_now.cart.x, env_right_now.cart.y, env_right_now.cart.theta = last(x), last(y), last(theta)
                    env_right_now.humans = move_human_for_one_time_step_in_actual_environment(env_right_now,1.0/env_right_now.cart.v)
                    initial_state = [env_right_now.cart.x,env_right_now.cart.y,env_right_now.cart.theta]
                    push!( all_gif_environments, (string(time_taken_by_cart)*"_"*string(i),deepcopy(env_right_now)) )
                    if(get_count_number_of_risks(env_right_now) != 0)
                        number_risks += get_count_number_of_risks(env_right_now)
                        push!(all_risky_scenarios,deepcopy(env_right_now))
                    end
                end
                current_belief = update_current_belief(old_env, env_right_now, current_belief, lidar_range)
                push!(all_observed_environments,deepcopy(env_right_now))
                push!(all_generated_beliefs, current_belief)
            else
                current_belief = update_human_position_and_current_belief(env_right_now, current_belief, lidar_range, one_time_step)
                push!( all_gif_environments, (string(time_taken_by_cart)*"_"*string(0),deepcopy(env_right_now)) )
                push!(all_observed_environments,deepcopy(env_right_now))
                push!(all_generated_beliefs, current_belief)
                if(get_count_number_of_risks(env_right_now) != 0)
                    number_risks += get_count_number_of_risks(env_right_now)
                    push!(all_risky_scenarios,deepcopy(env_right_now))
                end
            end

            push!(cart_throughout_path,(copy(env_right_now.cart)))
            write(io,"Modified cart state = $(env_right_now.cart)\n")
            write(io,"************************************************************************\n")
            println("Modified cart state = " , string(env_right_now.cart))
            println("************************************************************************")

        else
            push!( all_gif_environments, (string(time_taken_by_cart)*"_"*string(0),deepcopy(env_right_now)) )
            push!(all_observed_environments,deepcopy(env_right_now))
            push!(all_generated_beliefs, current_belief)
            push!(all_generated_trees, nothing)
            if(get_count_number_of_risks(env_right_now) != 0)
                number_risks += get_count_number_of_risks(env_right_now)
                push!(all_risky_scenarios,deepcopy(env_right_now))
            end
            println("Cart ran into Boundary Walls")
            write(io,"Cart ran into Boundary Walls")
            break
        end
        close(io)
        if(time_taken_by_cart>50)
            break
        end
    end

    io = open(filename,"a")
    write(io,"Goal Reached!")
    write(io,"Time Taken by cart to reach goal :  $time_taken_by_cart\n")
    write(io,"Number of risky scenarios encountered by the cart : $number_risks\n")
    println("Goal reached")
    println("Time Taken by cart to reach goal : ", time_taken_by_cart)
    println("Number of risky scenarios encountered by the cart : ", number_risks)
    close(io)

    return all_gif_environments, all_observed_environments, all_generated_beliefs, all_generated_trees,all_risky_scenarios,number_risks,time_taken_by_cart
end


env_right_now = deepcopy(env)
#Create POMDP for env_right_now
golfcart_2D_action_space_pomdp() = POMDP_Planner_2D_action_space(0.9,2.0,-100.0,2.0,-100.0,1.0,1.0,1000.0,7.0,env_right_now,1)
discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = isgoalstate_pomdp_planning(s,terminal_cart_state);
#actions(::POMDP_Planner_2D_action_space) = [(-pi/36,-1.0),(-pi/36,0.0),(-pi/36,1.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/36,-1.0),(pi/36,0.0),(pi/36,1.0)]
actions(::POMDP_Planner_2D_action_space) = [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
# actions(::POMDP_Planner_2D_action_space) = [(-pi/4,0.0),(-pi/4,1.0),(-pi/4,-1.0),
#                                             (-pi/6,0.0),(-pi/6,1.0),(-pi/6,-1.0),
#                                             (-pi/12,0.0),(-pi/12,1.0),(-pi/12,-1.0),
#                                             (0.0,0.0),(0.0,1.0),(0.0,-1.0),
#                                             (pi/12,0.0),(pi/12,1.0),(pi/12,-1.0),
#                                             (pi/6,0.0),(pi/6,1.0),(pi/6,-1.0),
#                                             (pi/4,0.0),(pi/4,1.0),(pi/4,-1.0),
#                                             (-10.0,-10.0)]

solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space)),
        calculate_upper_bound_value_pomdp_planning_2D_action_space, check_terminal=true),K=100,D=50,T_max=0.8, tree_in_info=true)
planner = POMDPs.solve(solver, golfcart_2D_action_space_pomdp());

all_gif_environments, all_observed_environments, all_generated_beliefs, all_generated_trees,
        all_risky_scenarios,number_risks,time_taken_by_cart = run_one_simulation(env_right_now)

anim = @animate for i ∈ 1:length(all_gif_environments)
    display_env(all_gif_environments[i][2]);
    savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*all_gif_environments[i][1]*".png")
end
gif(anim, "just_2D_action_space_pomdp_planner_run.gif", fps = 10)
