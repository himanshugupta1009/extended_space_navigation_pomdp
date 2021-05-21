
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
    curr_vertex_num:: Int
    parent_vertex_num:: Int
end

#POMDP struct
mutable struct POMDP_Planner_2D_action_space <: POMDPs.POMDP{POMDP_state_2D_action_space,Int,Array{location,1}}
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
    cv_num::Int64
    pv_num::Int64
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
    #sub_env_graph = state_distribution.world.graph[ filter_vertices(state_distribution.world.graph,:x,state_distribution.world.cart.x)]
    #vertex_iterator = filter_vertices(sub_env_graph,:y,state_distribution.world.cart.y)
    #collect(vertex_iterator)[1] - gives you the vertex number of the vertex in the graph that is at position (world.cart.x, world.cart.y)
    return POMDP_state_2D_action_space(state_distribution.world.cart,pedestrians,state_distribution.cv_num,state_distribution.pv_num)
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

    rand_num = (rand(rng) - 0.5)*0.1
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
    discrete_new_x, discrete_new_y = get_pedestrian_discrete_position_pomdp_planning(new_x,new_y,world)
    #@show(discrete_new_x,discrete_new_y)
    new_human_state = human_state(new_x, new_y, human.v, human.goal,human.id)
    observed_location = location(discrete_new_x, discrete_new_y)

    return new_human_state,observed_location
end
#@code_warntype update_human_position_pomdp_planning(env.humans[2], env, 1.0, MersenneTwister(1234))

function get_pedestrian_intermediate_trajectory_point(start_x, start_y, end_x, end_y, discrete_time)
    tbr_x = start_x + discrete_time*(end_x - start_x)
    tbr_y = start_y + discrete_time*(end_y - start_y)
    return tbr_x,tbr_y
end

#************************************************************************************************
#Simulating the cart one step forward in POMDP planning according to its new speed

function update_cart_position_pomdp_planning_2D_action_space(current_cart, delta_angle, new_cart_speed, world_length, world_breadth,
                                                                                            goal_distance_threshold, num_time_intervals = 10)
    current_x, current_y, current_theta = current_cart.x, current_cart.y, current_cart.theta
    if(new_cart_speed == 0.0)
        cart_path = Tuple{Float64,Float64,Float64}[ ( Float64(current_x), Float64(current_y), Float64(current_theta) ) ]
        cart_path = repeat(cart_path, num_time_intervals+1)
    else
        cart_path = Tuple{Float64,Float64,Float64}[]
        push!(cart_path,(Float64(current_x), Float64(current_y), Float64(current_theta)))
        arc_length = new_cart_speed
        steering_angle = atan((current_cart.L*delta_angle)/arc_length)
        for i in (1:num_time_intervals)
            if(steering_angle == 0.0)
                new_theta = current_theta
                new_x = current_x + arc_length*cos(current_theta)*(1/num_time_intervals)
                new_y = current_y + arc_length*sin(current_theta)*(1/num_time_intervals)
            else
                new_theta = current_theta + (arc_length * tan(steering_angle) * (1/num_time_intervals) / current_cart.L)
                new_theta = wrap_between_0_and_2Pi(new_theta)
                new_x = current_x + ((current_cart.L / tan(steering_angle)) * (sin(new_theta) - sin(current_theta)))
                new_y = current_y + ((current_cart.L / tan(steering_angle)) * (cos(current_theta) - cos(new_theta)))
            end
            push!(cart_path,(Float64(new_x), Float64(new_y), Float64(new_theta)))
            current_x, current_y,current_theta = new_x,new_y,new_theta
            if(current_x>world_length || current_y>world_breadth || current_x<0.0 || current_y<0.0)
                for j in i+1:num_time_intervals
                    push!(cart_path,(current_x, current_y, current_theta))
                end
                if(is_within_range_check_with_points(current_x, current_y, current_cart.goal.x, current_cart.goal.y, goal_distance_threshold))
                    return cart_path, true
                else
                    return cart_path, false
                end
            end
        end
    end
    #@show(current_cart_position,steering_angle, new_cart_speed, cart_path)
    if(is_within_range_check_with_points(cart_path[end][1], cart_path[end][2], current_cart.goal.x, current_cart.goal.y, goal_distance_threshold))
        return cart_path, true
    else
        return cart_path, false
    end
end
#@code_warntype update_cart_position_pomdp_planning_2D_action_space(env.cart, pi/12, 5.0, 100.0,100.0)
#update_cart_position_pomdp_planning_2D_action_space(env.cart, pi/12, 5.0, 100.0,100.0,1.0)



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
        #return penalty/2.0
        return -50.0
    else
        return 0.0
    end
end


#************************************************************************************************
#POMDP Generative Model

#parent = Dict()
function POMDPs.gen(m::POMDP_Planner_2D_action_space, s, a, rng)
    #@show("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", s, a, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    cart_reached_goal_flag = false
    collision_with_pedestrian_flag = false
    collision_with_obstacle_flag = false
    immediate_stop_flag = false
    new_human_states = human_state[]
    observed_positions = location[]
    new_cart_position::Tuple{Float64,Float64,Float64} = (0.0,0.0,0.0)
    cart_final_position::Tuple{Float64,Float64,Float64} = (0.0,0.0,0.0)
    new_cart_velocity::Float64 = s.cart.v

    if(s.curr_vertex_num == 2)
        new_cart_position = (-100.0, -100.0, -100.0)
        cart_reached_goal_flag = true
        push!(observed_positions, location(-25.0,-25.0))
    else
        num_time_intervals = 10
        #Length of one time step
        one_time_step = 1/num_time_intervals
        if(a == 3)
            new_cart_velocity = 0.0
            immediate_stop_flag = false
            new_cart_position = (s.cart.x, s.cart.y, s.cart.theta)
            new_human_states = deepcopy(s.pedestrians)
            for i in 1:num_time_intervals+1
                #Propogate humans
                for human_index in 1:length(new_human_states)
                    new_human_states[human_index],observed_location = update_human_position_pomdp_planning(new_human_states[human_index],
                                                                                m.world,one_time_step, rng)
                    push!(observed_positions, observed_location)
                end
            end
        else
            #println(s.curr_vertex_num, a)
            new_cart_velocity = get_prop(m.world.graph,s.curr_vertex_num,a,:weight)
            new_cart_position = (s.cart.x, s.cart.y, s.cart.theta)
            cart_final_position = (get_prop(m.world.graph,a,:x), get_prop(m.world.graph,a,:y), s.cart.theta)
            new_human_states = deepcopy(s.pedestrians)
            for i in 1:num_time_intervals+1
                #Check collision with human and propogate humans
                for human_index in 1:length(new_human_states)
                    if( find_if_two_circles_intersect(new_cart_position[1], new_cart_position[2], s.cart.L+0.5,new_human_states[human_index].x,
                                                        new_human_states[human_index].y, m.pedestrian_distance_threshold) )
                        new_cart_position = (-100.0, -100.0, -100.0)
                        collision_with_pedestrian_flag = true
                        observed_positions = location[ location(-50.0,-50.0) ]
                        break
                    end
                    new_human_states[human_index],observed_location = update_human_position_pomdp_planning(new_human_states[human_index],
                                                                                m.world,one_time_step, rng)
                    push!(observed_positions, observed_location)
                end
                if(collision_with_pedestrian_flag)
                    break
                end
                #If there was no collision with any pedestrian then propogate the cart forward
                new_cart_position = ( new_cart_position[1] + one_time_step*(cart_final_position[1] - new_cart_position[1]) ,
                                        new_cart_position[2] + one_time_step*(cart_final_position[2] - new_cart_position[2]),
                                        new_cart_position[3] )
            end
        end
    end

    cart_new_state = cart_state(new_cart_position[1], new_cart_position[2], new_cart_position[3], new_cart_velocity, s.cart.L, s.cart.goal)

    # Next POMDP State
    if(a!=3)
        sp = POMDP_state_2D_action_space(cart_new_state, new_human_states,a,s.curr_vertex_num)
    else
        sp = POMDP_state_2D_action_space(cart_new_state, new_human_states,s.curr_vertex_num,s.parent_vertex_num)
    end
    # Generated Observation
    o = observed_positions

    # R(s,a): Reward for being at state s and taking action a
    #Penalize if collision with pedestrian
    r = pedestrian_collision_penalty_pomdp_planning_2D_action_space(collision_with_pedestrian_flag, m.pedestrian_collision_penalty)
    #println("Reward from collision with pedestrian", r)
    #Reward if reached goal
    r += goal_reward_pomdp_planning_2D_action_space(s, m.goal_reward_distance_threshold, cart_reached_goal_flag, m.goal_reward)
    #println("Reward from goal ", r)
    #Penalize if had to apply sudden brakes
    r += immediate_stop_penalty_pomdp_planning_2D_action_space(immediate_stop_flag, m.pedestrian_collision_penalty)
    #println("Reward if you had to apply immediate brakes", r)
    #Penalty for longer duration paths
    r -= 1.0

    return (sp=sp, o=o, r=r)
end
#@code_warntype POMDPs.gen(golfcart_2D_action_space_pomdp, POMDP_state_2D_action_space(env.cart,env.cart_lidar_data,1,0), neighbors(env.graph,1)[1] , MersenneTwister(1234))
#@code_warntype POMDPs.gen(golfcart_2D_action_space_pomdp, POMDP_state_2D_action_space(env.cart,env.cart_lidar_data,2,0), neighbors(env.graph,2)[1] , MersenneTwister(1234))
#@profview POMDPs.gen(golfcart_2D_action_space_pomdp, POMDP_state_2D_action_space(env.cart,env.cart_lidar_data,2,0), neighbors(env.graph,2)[1] , MersenneTwister(1234))


#************************************************************************************************
#Upper bound value function for DESPOT

function is_collision_state_pomdp_planning_2D_action_space(s,m)
    for human in s.pedestrians
        if(is_within_range(location(s.cart.x,s.cart.y),location(human.x,human.y),m.pedestrian_distance_threshold))
            return true
        end
    end
    return false
end

function time_to_goal_pomdp_planning_2D_action_space(s,max_cart_speed)
    path = get_props()
    cart_distance_to_goal = sqrt( (s.cart.x-s.cart.goal.x)^2 + (s.cart.y-s.cart.goal.y)^2 )
    #@show(cart_distance_to_goal)
    return floor(cart_distance_to_goal/max_cart_speed)
end

function calculate_upper_bound_value_pomdp_planning_2D_action_space(m, b)

    #@show lbound(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space), max_depth=100, final_value=reward_to_be_awarded_at_max_depth_in_lower_bound_policy_rollout),m , b)
    value_sum = 0.0
    for (s, w) in weighted_particles(b)
        if(s.cart.x == -100.0 && s.cart.y == -100.0)
            value_sum += 0.0
        elseif(is_within_range_check_with_points(s.cart.x,s.cart.y, s.cart.goal.x, s.cart.goal.y, m.cart_goal_reached_distance_threshold))
            value_sum += w*m.goal_reward
        elseif(is_collision_state_pomdp_planning_2D_action_space(s,m))
            value_sum += w*m.pedestrian_collision_penalty
        else
            path_len = length( get_prop(m.world.graph,s.curr_vertex_num,:path_to_goal) ) - 1
            value_sum += w*((discount(m)^floor(path_len/m.max_cart_speed))*m.goal_reward)
        end
    end
    return value_sum
end
#@code_warntype calculate_upper_bound_value(golfcart_pomdp(), initialstate_distribution(golfcart_pomdp()))


#************************************************************************************************
#Lower bound policy function for DESPOT
function calculate_lower_bound_policy_pomdp_planning_2D_action_space(m,b)
    pomdp_state = first(particles(b))
    possible_next_states = actions(m,b)
    # d_far_threshold = 5.0
    d_near_threshold = 2.0
    # state_to_avoid = pomdp_state.parent_vertex_num
    path = get_prop(m.world.graph, pomdp_state.curr_vertex_num, :path_to_goal)
    if(length(path) == 0)
        return 3
    elseif(length(path) == 1)
        return 2
    end

    for (s,w) in weighted_particles(b)
        if(s.cart.x == -100.0 && s.cart.y == -100.0)
            continue
        else
            for human in s.pedestrians
                euclidean_distance = sqrt((s.cart.x - human.x)^2 + (s.cart.y - human.y)^2)
                if(euclidean_distance < d_near_threshold)
                    return 3
                end
            end
        end
    end
    return path[2]
end

function reward_to_be_awarded_at_max_depth_in_lower_bound_policy_rollout(m,b)
    value_sum = 0.0
    for (s, w) in weighted_particles(b)
        value_sum += w*((discount(m)^time_to_goal_pomdp_planning_2D_action_space(s,m.max_cart_speed))*m.goal_reward)
    end
    #println("HG rules")
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
            value_sum += w*((discount(m)^time_to_goal_pomdp_planning_2D_action_space(s,m.max_cart_speed))*m.goal_reward)
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
