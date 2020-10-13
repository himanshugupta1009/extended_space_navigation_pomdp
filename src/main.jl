include("environment.jl")
include("utils.jl")
include("hybrid_a_star.jl")
include("one_d_action_space_close_waypoint_pomdp.jl")
include("belief_tracker.jl")

Base.copy(s::cart_state) = cart_state(s.x, s.y,s.theta,s.v,s.L,s.goal)
global i = 1
cart_throughout_path = []
one_time_step = 0.5
env_copy = deepcopy(env)
all_observed_environments = []
lidar_range = 30

#Create new POMDP for env_copy
golfcart_pomdp_copy() = Speed_Planner_POMDP(0.9,4.0,-100.0,1.0,1.0,100.0,7.0,env_copy,1)
discount(p::Speed_Planner_POMDP) = p.discount_factor
isterminal(::Speed_Planner_POMDP, s::SP_POMDP_state) = isgoalstate_pomdp_planning(s,terminal_cart_state);
actions(::Speed_Planner_POMDP) = [-1.0, 0.0, 1.0]

solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning)),
        calculate_upper_bound_value_pomdp_planning, check_terminal=true),K=100,D=50,T_max=0.5, tree_in_info=true)
planner = POMDPs.solve(solver, golfcart_pomdp_copy());
m = golfcart_pomdp_copy()

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

# @show(env_copy.cart_lidar_data)
# @show(current_belief)
#break

anim = @animate for j ∈ 1:10000
    io = open("output.txt","a")
    @show(j)
    if(!is_within_range(location(env_copy.cart.x,env_copy.cart.y), env_copy.cart.goal, 1.0))
        env_copy.cart_hybrid_astar_path = @time hybrid_a_star_search(env_copy.cart.x, env_copy.cart.y,
            env_copy.cart.theta, env_copy.cart.goal.x, env_copy.cart.goal.y, env_copy, current_belief);

        push!(all_observed_environments, (deepcopy(env_copy),current_belief))

        write(io,"Current Iteration Number - $j\n")
        if(length(env_copy.cart_hybrid_astar_path) == 0)
            println("**********Hybrid A Star PAth Not found. Vehicle Stopped**********")
            write(io,"**********Hybrid A Star PAth Not found. Vehicle Stopped**********\n")
            env_copy.cart.v = 0.0
        else
            m = golfcart_pomdp_copy()
            b = @time initialstate_distribution(m,current_belief)
            a = action(planner, b)
            println("Action chosen " , a)
            write(io,"Action chosen: $a\n")
            env_copy.cart.v = clamp(env_copy.cart.v + a,0,m.max_cart_speed)

            if(env_copy.cart.v != 0.0)
                println("Current cart velocity = " , string(env_copy.cart.v))
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
                println("Current cart velocity = " , string(env_copy.cart.v))
                write(io,"Current cart state = $(env_copy.cart)\n")
            end
        end
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

        global current_belief = @time update_belief(current_belief, env_copy.goals,
            env_copy.cart_lidar_data, new_lidar_data)
        env_copy.cart_lidar_data = new_lidar_data

        display_env(env_copy);
        push!(cart_throughout_path,(copy(env_copy.cart)))
    else
        @show("Goal reached")
        write(io,"Goal Reached!")
        close(io)
        break
    end
    close(io)
end

gif(anim, "og_pipeline_new.gif", fps = 2)
