#Load Required Packages
using ProfileView
using POMDPs
using Distributions
using Random
import POMDPs: initialstate_distribution, actions, gen, discount, isterminal
using POMDPModels, POMDPSimulators, ARDESPOT, POMDPModelTools, POMDPPolicies
using ParticleFilters
using BenchmarkTools
using Debugger
using LinearAlgebra
using DifferentialEquations


#Define POMDP State Struct
struct POMDP_state_2D_action_space
    cart:: cart_state
    pedestrians::Array{human_state,1}
end

#POMDP struct
mutable struct POMDP_Planner_2D_action_space <: POMDPs.POMDP{POMDP_state_2D_action_space,Tuple{Float64,Float64},Array{location,1}}
    discount_factor::Float64
    pedestrian_distance_threshold::Float64
    pedestrian_collision_penalty::Float64
    obstacle_distance_threshold::Float64
    obstacle_collision_penalty::Float64
    goal_reward_distance_threshold::Float64
    cart_goal_reached_distance_threshold::Float64
    goal_reward::Float64
    max_cart_speed::Float64
    world::experiment_environment
    start_path_index::Int64
end

#Function to check terminal state
function is_terminal_state_pomdp_planning(s,terminal_state)
    if(terminal_state.x == s.cart.x && terminal_state.y == s.cart.y)
        return true
    else
        return false
    end
end



#************************************************************************************************
#Generate Initial POMDP state based on the scenario provided by random operator.

struct POMDP_2D_action_space_state_distribution
    world::experiment_environment
    current_belief::Array{human_probability_over_goals,1}
end

function Base.rand(rng::AbstractRNG, state_distribution::POMDP_2D_action_space_state_distribution)
    pedestrians = Array{human_state,1}()
    for i in 1:length(state_distribution.world.cart_lidar_data)
        sampled_goal = Distributions.rand(rng, SparseCat(state_distribution.world.goals,state_distribution.current_belief[i].distribution))
        new_human = human_state(state_distribution.world.cart_lidar_data[i].x,state_distribution.world.cart_lidar_data[i].y,
                            state_distribution.world.cart_lidar_data[i].v,sampled_goal,
                            state_distribution.world.cart_lidar_data[i].id)
        push!(pedestrians, new_human)
    end
    return POMDP_state_2D_action_space(state_distribution.world.cart,pedestrians)
end



#************************************************************************************************
#Simulating Human One step forward in POMDP planning

function get_pedestrian_discrete_position_pomdp_planning(new_x,new_y,world)
    discretization_step_length = 1.0
    discrete_x = floor(new_x/discretization_step_length) * discretization_step_length
    discrete_y = floor(new_y/discretization_step_length) * discretization_step_length
    discrete_x = clamp(discrete_x,0,world.length)
    discrete_y = clamp(discrete_y,0,world.breadth)
    return discrete_x, discrete_y
end

function update_human_position_pomdp_planning(human, world, time_step, rng)

    rand_num = (rand(rng) - 0.5)*0.2

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
    discrete_new_x, discrete_new_y = get_pedestrian_discrete_position_pomdp_planning(new_x,new_y,world)
    #@show(discrete_new_x,discrete_new_y)
    new_human_state = human_state(new_x, new_y, human.v, human.goal,human.id)
    observed_location = location(discrete_new_x, discrete_new_y)

    return new_human_state,observed_location
end
#@code_warntype update_human_position_pomdp_planning(env.humans[2], env, 1.0, MersenneTwister(1234))



#************************************************************************************************
#Simulating the cart one step forward in POMDP planning according to its new speed

function update_cart_position_pomdp_planning_2D_action_space(current_cart_position, delta_angle, new_cart_speed, world)
    current_x, current_y, current_theta = current_cart_position.x, current_cart_position.y, current_cart_position.theta
    cart_path = Tuple{Float64,Float64,Float64}[]
    if(new_cart_speed == 0.0)
         push!(cart_path,(Float64(current_x), Float64(current_y), Float64(current_theta)))
    else
        push!(cart_path,(Float64(current_x), Float64(current_y), Float64(current_theta)))
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
            push!(cart_path,(Float64(new_x), Float64(new_y), Float64(new_theta)))
            current_x, current_y,current_theta = new_x,new_y,new_theta
            if(current_x>world.length || current_y>world.breadth || current_x<0.0 || current_y<0.0)
                return cart_path
            end
        end
    end
    #@show(current_cart_position,steering_angle, new_cart_speed, cart_path)
    return cart_path
end
#@code_warntype update_cart_position_pomdp_planning(env.cart,pi/18, 5.0, env)
#update_cart_position_pomdp_planning(env.cart, pi/18, 5.0, env)



#************************************************************************************************
# Reward functions for the POMDP model

# Pedestrian Collision Penalty
function pedestrian_collision_penalty_pomdp_planning_2D_action_space(pedestrian_collision_flag, penalty)
    if(pedestrian_collision_flag)
        #@show(penalty)
        return penalty
    else
        return 0.0
    end
end

# Obstacle Collision Penalty
function obstacle_collision_penalty_pomdp_planning_2D_action_space(obstacle_collision_flag, penalty)
    if(obstacle_collision_flag)
        #@show(penalty)
        return penalty
    else
        return 0.0
    end
end

# Goal Reward
function goal_reward_pomdp_planning_2D_action_space(s, distance_threshold, goal_reached_flag, goal_reward)
    total_reward = 0.0
    if(goal_reached_flag)
        total_reward = goal_reward
    else
        euclidean_distance = ((s.cart.x - s.cart.goal.x)^2 + (s.cart.y - s.cart.goal.y)^2)^0.5
        if(euclidean_distance<distance_threshold && euclidean_distance!=0.0)
            total_reward = goal_reward/euclidean_distance
        end
    end
    #@show(total_reward)
    return total_reward
end

# Low Speed Penalty
function speed_reward_pomdp_planning_2D_action_space(s, max_speed)
    t = (s.cart.v - max_speed)/max_speed
    return t
end

function immediate_stop_penalty_pomdp_planning_2D_action_space(immediate_stop_flag, penalty)
    if(immediate_stop_flag)
        return penalty/2.0
    else
        return 0.0
    end
end


#************************************************************************************************
#POMDP Generative Model

#parent = Dict()
function POMDPs.gen(m::POMDP_Planner_2D_action_space, s, a, rng)

    cart_reached_goal_flag = false
    collision_with_pedestrian_flag = false
    collision_with_obstacle_flag = false
    immediate_stop_flag = false
    #Length of one time step
    one_time_step = 1.0

    new_human_states = human_state[]
    observed_positions = location[]

    if(is_within_range_check_with_points(s.cart.x,s.cart.y, s.cart.goal.x, s.cart.goal.y, m.cart_goal_reached_distance_threshold))
        #println("Goal reached")
        new_cart_position = (-100.0, -100.0, -100.0)
        cart_reached_goal_flag = true
        new_cart_velocity = clamp(s.cart.v + a[2], 0.0, m.max_cart_speed)
    elseif( (s.cart.x>m.world.length) || (s.cart.y>m.world.breadth) || (s.cart.x<0.0) || (s.cart.y<0.0) )
        #print("Running into wall")
        new_cart_position = (-100.0, -100.0, -100.0)
        collision_with_obstacle_flag = true
        new_cart_velocity = clamp(s.cart.v + a[2], 0.0, m.max_cart_speed)
    # elseif(a[2] == -10.0)
    #     new_cart_position = (-100.0, -100.0, -100.0)
    #     immediate_stop_flag = true
    #     new_cart_velocity = clamp(s.cart.v + a[2], 0.0, m.max_cart_speed)
    else
        if(a[2] == -10.0)
            immediate_stop_flag = true
        end
        new_cart_velocity = clamp(s.cart.v + a[2], 0.0, m.max_cart_speed)
        cart_path::Vector{Tuple{Float64,Float64,Float64}} = update_cart_position_pomdp_planning_2D_action_space(s.cart, a[1], new_cart_velocity, m.world)
        new_cart_position = cart_path[end]
        if( (new_cart_position[1]>m.world.length) || (new_cart_position[2]>m.world.breadth) || (new_cart_position[1]<0.0) || (new_cart_position[2]<0.0) )
            new_cart_position = (-100.0, -100.0, -100.0)
            collision_with_obstacle_flag = true
            new_cart_velocity = clamp(s.cart.v + a[2], 0.0, m.max_cart_speed)
        else
            if(new_cart_velocity != 0.0 || new_cart_velocity == 0.0 )
                for position in cart_path
                    for human in s.pedestrians
                        if( is_within_range_check_with_points( position[1], position[2], human.x, human.y, m.pedestrian_distance_threshold ) )
                            new_cart_position = (-100.0, -100.0, -100.0)
                            collision_with_pedestrian_flag = true
                            break
                        end
                    end
                    for obstacle in m.world.obstacles
                        if( is_within_range_check_with_points( position[1], position[2], obstacle.x, obstacle.y, obstacle.r + m.obstacle_distance_threshold ) )
                            new_cart_position = (-100.0, -100.0, -100.0)
                            collision_with_obstacle_flag = true
                            break
                        end
                    end
                    if(collision_with_pedestrian_flag || collision_with_obstacle_flag)
                        break
                    end
                end
            end
        end

    end
    cart_new_state = cart_state(new_cart_position[1], new_cart_position[2], new_cart_position[3], new_cart_velocity, s.cart.L, s.cart.goal)

    # Simulate all the pedestrians
    for human in s.pedestrians
        modified_human_state,observed_location = update_human_position_pomdp_planning(human, m.world, one_time_step, rng)
        push!(new_human_states, modified_human_state)
        push!(observed_positions, observed_location)
    end

    # if(length(observed_positionspush!(cart_path,(Float64(current_x), Float64(current_y), Float64(current_theta))))!=0)
    #     println(observed_positions)
    # end
    # Next POMDP State
    sp = POMDP_state_2D_action_space(cart_new_state, new_human_states)
    # Generated Observation
    o = observed_positions

    # R(s,a): Reward for being at state s and taking action a
    #Penalize if collision with pedestrian
    r = pedestrian_collision_penalty_pomdp_planning_2D_action_space(collision_with_pedestrian_flag, m.pedestrian_collision_penalty)
    #println("Reward from collision with pedestrian", r)
    #Penalize if collision with obstacle
    r += obstacle_collision_penalty_pomdp_planning_2D_action_space(collision_with_obstacle_flag, m.obstacle_collision_penalty)
    #println("Reward from collision with obstacle", r)
    #Reward if reached goal
    r += goal_reward_pomdp_planning_2D_action_space(s, m.goal_reward_distance_threshold, cart_reached_goal_flag, m.goal_reward)
    #println("Reward from goal ", r)
    #Penalize if going slow when it can go fast
    r += speed_reward_pomdp_planning_2D_action_space(s, m.max_cart_speed)
    #println("Reward from not traveling at max_speed ", r)
    #Penalize if had to apply sudden brakes
    r += immediate_stop_penalty_pomdp_planning_2D_action_space(immediate_stop_flag, m.pedestrian_collision_penalty)
    #println("Reward if you had to apply immediate brakes", r)
    #Penalty for longer duration paths
    r -= 1.0

    #parent[sp] = s
    # if(a[2] == 1.0 )
    #     println(a)
    #     println(s)
    #     println(sp)
    #     println(r)
    # end
    # if(sp.cart.x!=-100.0)
    #     println(s.cart, a , sp.cart, r)
    # end
    return (sp=sp, o=o, r=r)
end
#@code_warntype POMDPs.gen(golfcart_pomdp(), SP_POMDP_state(env.cart,env.humans), 1, MersenneTwister(1234))



#************************************************************************************************
#Upper bound value function for DESPOT

function is_collision_state_pomdp_planning_2D_action_space(s,m)
    if((s.cart.x>m.world.length) || (s.cart.y>m.world.breadth) || (s.cart.x<0.0) || (s.cart.y<0.0))
        return true
    else
        for human in s.pedestrians
            if(is_within_range(location(s.cart.x,s.cart.y),location(human.x,human.y),m.pedestrian_distance_threshold))
                return true
            end
        end
        for obstacle in m.world.obstacles
            if(is_within_range(location(s.cart.x,s.cart.y),location(obstacle.x,obstacle.y),obstacle.r + m.obstacle_distance_threshold))
                return true
            end
        end
        return false
    end
end

function time_to_goal_pomdp_planning_2D_action_space(s,m)
    cart_distance_to_goal = sqrt( (s.cart.x-s.cart.goal.x)^2 + (s.cart.y-s.cart.goal.y)^2 )
    return floor(cart_distance_to_goal/m.max_cart_speed)
end

function calculate_upper_bound_value_pomdp_planning_2D_action_space(m, b)

    #@show lbound(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space), max_depth=100, final_value=reward_to_be_awarded_at_max_depth_in_lower_bound_policy_rollout),m , b)
    value_sum = 0.0
    for (s, w) in weighted_particles(b)
        if(s.cart.x == -100.0 && s.cart.y == -100.0)
            value_sum += 0.0
        elseif(is_within_range(location(s.cart.x,s.cart.y), s.cart.goal, m.cart_goal_reached_distance_threshold))
            value_sum += w*m.goal_reward
        elseif(is_collision_state_pomdp_planning_2D_action_space(s,m))
            value_sum += w*m.pedestrian_collision_penalty
        else
            value_sum += w*((discount(m)^time_to_goal_pomdp_planning_2D_action_space(s,m))*m.goal_reward)
        end
    end
    return value_sum
end
#@code_warntype calculate_upper_bound_value(golfcart_pomdp(), initialstate_distribution(golfcart_pomdp()))

#Lower bound policy function for DESPOT
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
                dist_to_closest_human = 200.0  #Some really big infeasible negative number (not Inf because avoid the tpe mismatch error)
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
        return (0.0,0.0)
    end

    #This means all humans are away and you can accelerate.
    if(speed_change_to_be_returned == 1.0)
        return (0.0,speed_change_to_be_returned)
    end

    #If code has reached this point, then the best action is to maintain your current speed.
    #We have already found the best steering angle to take.
    return (best_delta_angle,0.0)
end

function reward_to_be_awarded_at_max_depth_in_lower_bound_policy_rollout(m,b)
    #print("HI")
    # #sleep(5)
    # print(b.depth)
    value_sum = 0.0
    for (s, w) in weighted_particles(b)
        cart_distance_to_goal = sqrt( (s.cart.x - s.cart.goal.x)^2 + (s.cart.y - s.cart.goal.y)^2 )
        if(cart_distance_to_goal > 1.0)
            value_sum += w*(1/cart_distance_to_goal)*m.goal_reward
        end
    end
    #println(value_sum)
    return value_sum
end



#************************************************************************************************
#Functions for debugging lb>ub error
function debug_is_collision_state_pomdp_planning_2D_action_space(s,m)

    if((s.cart.x>100.0) || (s.cart.y>100.0) || (s.cart.x<0.0) || (s.cart.y<0.0))
        println("Stepped Outside")
        return true
    else
        for human in s.pedestrians
            if(is_within_range(location(s.cart.x,s.cart.y),location(human.x,human.y),m.pedestrian_distance_threshold))
                println("Collision with human ", human)
                return true
            end
        end
        for obstacle in m.world.obstacles
            if(is_within_range(location(s.cart.x,s.cart.y),location(obstacle.x,obstacle.y),obstacle.r + m.obstacle_distance_threshold))
                println("Collision with obstacle ", obstacle)
                return true
            end
        end
        return false
    end
end

function debug_golfcart_upper_bound_2D_action_space(m,b)

    # lower = lbound(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space), max_depth=100),m , b)
    lower = lbound(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space), max_depth=100, final_value=reward_to_be_awarded_at_max_depth_in_lower_bound_policy_rollout),m , b)
    #@show(lower)
    value_sum = 0.0
    for (s, w) in weighted_particles(b)
        if (s.cart.x == -100.0 && s.cart.y == -100.0)
            value_sum += 0.0
        elseif (is_within_range(location(s.cart.x,s.cart.y), s.cart.goal, m.cart_goal_reached_distance_threshold))
            value_sum += w*m.goal_reward
        elseif (is_collision_state_pomdp_planning_2D_action_space(s,m))
            value_sum += w*m.pedestrian_collision_penalty
        else
            value_sum += w*((discount(m)^time_to_goal_pomdp_planning_2D_action_space(s,m))*m.goal_reward)
        end
    end
    #@show("*********************************************************************")
    #@show(value_sum)
    u = (value_sum)/weight_sum(b)
    if lower > u
        push!(bad, (lower,u,b))
        @show("IN DEBUG",lower,u)
    end
    return u
end
