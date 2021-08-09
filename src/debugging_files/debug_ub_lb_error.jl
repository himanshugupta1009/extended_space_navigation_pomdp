#bad = []
#rollout = []

function debug_ub_lb_error_2D_action_space(all_observed_environments,all_generated_beliefs,lt,which_env)
    pomdp_ub_debugging_env = deepcopy(all_observed_environments[which_env])
    current_belief_debugging = all_generated_beliefs[which_env]

    golfcart_pomdp_debug = POMDP_Planner_2D_action_space(0.97,1.0,-100.0,1.0,-100.0,0.0,1.0,1000.0,2.0,pomdp_ub_debugging_env,lt)
    discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
    isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
    #actions(::POMDP_Planner_2D_action_space) = [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0)]
    actions(m::POMDP_Planner_2D_action_space,b) = get_available_actions_holonomic(m,b)

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(b->calculate_lower_bound_policy_pomdp_planning_2D_action_space(golfcart_2D_action_space_pomdp, b)),
                            max_depth=100),debug_golfcart_upper_bound_2D_action_space, check_terminal=true),K=50,D=100,T_max=0.5, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_pomdp_debug);

    m_ub_debugging = golfcart_pomdp_debug
    #b_ub_debugging = initialstate_distribution(m_ub_debugging,current_belief_debugging)
    b_ub_debugging = POMDP_2D_action_space_state_distribution(m_ub_debugging.world,current_belief_debugging)
    a, info = action_info(planner, b_ub_debugging);
    @show(a)
    return info
end

function print_belief_states(all_observed_environments,all_generated_beliefs,which_env,bad,loc)

    pomdp_ub_debugging_env = deepcopy(all_observed_environments[which_env])
    current_belief_debugging = all_generated_beliefs[which_env]

    golfcart_pomdp_debug =  POMDP_Planner_2D_action_space(0.99,2.0,-1000.0,1.0,-1000.0,0.0,1.0,1000000.0,2.0,pomdp_ub_debugging_env)
    discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
    isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
    actions(::POMDP_Planner_2D_action_space) = [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space)),
            debug_golfcart_upper_bound_2D_action_space, check_terminal=true),K=100,D=50,T_max=0.5, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_pomdp_debug);

    m_ub_debugging = golfcart_pomdp_debug

    lower = lbound(DefaultPolicyLB(FunctionPolicy(debug_calculate_lower_bound_policy_pomdp_planning_2D_action_space), max_depth=100, final_value=reward_to_be_awarded_at_max_depth_in_lower_bound_policy_rollout),m_ub_debugging , bad[loc][3])

    @show(debug_golfcart_upper_bound_2D_action_space(m_ub_debugging, bad[loc][3]))
    return

    value_sum = 0.0
    for (s,w) in weighted_particles(bad[loc][3])
        @show(s,w)
        #@show(s.cart)
        if(s.cart.x == -100.0 && s.cart.y == -100.0)
            value_sum += 0.0
        elseif(is_within_range(location(s.cart.x,s.cart.y), s.cart.goal, m_ub_debugging.cart_goal_reached_distance_threshold))
            @show(w*m_ub_debugging.goal_reward)
            value_sum += w*m_ub_debugging.goal_reward
            @show("A")
        elseif(debug_is_collision_state_pomdp_planning_2D_action_space(s,m_ub_debugging))
            @show("B")
            @show(w*m_ub_debugging.pedestrian_collision_penalty)
            value_sum += w*m_ub_debugging.pedestrian_collision_penalty
        else
            @show("C")
            @show( w*((discount(m_ub_debugging)^time_to_goal_pomdp_planning_2D_action_space(s,m_ub_debugging.max_cart_speed))*m_ub_debugging.goal_reward) )
            value_sum += w*((discount(m_ub_debugging)^time_to_goal_pomdp_planning_2D_action_space(s,m_ub_debugging.max_cart_speed))*m_ub_debugging.goal_reward)
            @show(value_sum)
        end
        temp = POMDPs.gen(m_ub_debugging, s, (0.0,1.0), MersenneTwister(1234))
        @show(POMDPs.isterminal(m_ub_debugging,s))
        println(temp)
        println()
        #search_state = s
    end
    @show(value_sum)
end

function ARDESPOT_function_get_rng(s::MemorizingSource, scenario::Int, depth::Int)
    rng = s.rngs[depth+1, scenario]
    if rng.finish == 0
        rng.start = s.furthest+1
        rng.idx = rng.start - 1
        rng.finish = s.furthest
        reserve(rng, s.min_reserve)
    end
    rng.idx = rng.start - 1
    return rng
end

function debug_do_the_rollout_policy_on_beliefs_and_accumulate_rewards(all_observed_environments,all_generated_beliefs,which_env,rolled_out_beliefs)

    pomdp_ub_debugging_env = deepcopy(all_observed_environments[which_env])
    current_belief_debugging = all_generated_beliefs[which_env]

    golfcart_pomdp_debug =  POMDP_Planner_2D_action_space(0.99,2.0,-1000.0,1.0,-1000.0,0.0,1.0,1000000.0,2.0,pomdp_ub_debugging_env)
    discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
    isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
    actions(::POMDP_Planner_2D_action_space) = [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space)),
            debug_golfcart_upper_bound_2D_action_space, check_terminal=true),K=100,D=50,T_max=0.5, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_pomdp_debug);
    m_ub_debugging = golfcart_pomdp_debug

    tr = 0.0
    i = 1
    for belief in rolled_out_beliefs
        cart_state = belief.scenarios[1][2]
        action_decided = calculate_lower_bound_policy_pomdp_planning_2D_action_space(belief)
        sp, o ,r =  POMDPs.gen(m_ub_debugging,cart_state, action_decided, MersenneTwister(1234))
        r_sum = 0.0
        for (k,s) in belief.scenarios
            if !isterminal(m_ub_debugging, s)
                rng = ARDESPOT_function_get_rng(belief.random_source, k, belief.depth)
                sp, o, r = @gen(:sp, :o, :r)(m_ub_debugging, s, action_decided, rng)
                r_sum += r
            end
        end
        @show(cart_state)
        @show(action_decided)
        @show(sp)
        @show(o)
        @show(r, r_sum)
        @show(i, belief.depth - rolled_out_beliefs[1].depth, "***************************************************************************")
        tr = tr + (discount(m_ub_debugging)^(belief.depth - rolled_out_beliefs[1].depth))*r_sum
        i = i+1
    end
    return tr/length(rolled_out_beliefs[1].scenarios)
end



#Rollout at any given belief
function find_the_rollout_path(planner_at_that_moment,env_at_that_moment,belief_at_that_moment, scenario_num = 1)
    copy_of_planner = deepcopy(planner_at_that_moment)
    b = POMDP_2D_action_space_state_distribution(env_at_that_moment,belief_at_that_moment);
    supposed_a, supposed_info = action_info(copy_of_planner, b);
    despot_tree = supposed_info[:tree]
    # desired_scenario = despot_tree.scenarios[scenario_num]
    # println(copy_of_planner.rng)
    # println(copy_of_planner.rs.rng)
    scenario_belief = ARDESPOT.get_belief(despot_tree, scenario_num, copy_of_planner.rs)
    #scenario_belief = ScenarioBelief(desired_scenario, copy_of_planner.rs, 0, b)
    L_0, U_0 = bounds(copy_of_planner.bounds, copy_of_planner.pomdp, scenario_belief)
    println(L_0," ",U_0)

    lbound_rollout_path(copy_of_planner.bounds.lower, copy_of_planner.pomdp, scenario_belief)
end


function lbound_rollout_path(lb::DefaultPolicyLB, pomdp::POMDP, b::ScenarioBelief)
    rsum = branching_sim_rollout_path(pomdp, lb.policy, b, lb.max_depth-b.depth, lb.final_value)
    return rsum/length(b.scenarios)
end


function branching_sim_rollout_path(pomdp::POMDP, policy::Policy, b::ScenarioBelief, steps::Integer, fval)
    S = statetype(pomdp)
    O = obstype(pomdp)
    odict = Dict{O, Vector{Pair{Int, S}}}()

    if steps <= 0
        return length(b.scenarios)*fval(pomdp, b)
    end

    a = action(policy, b)

    r_sum = 0.0
    for (k, s) in b.scenarios
        if !isterminal(pomdp, s)
            rng = ARDESPOT.get_rng(b.random_source, k, b.depth)
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
            if haskey(odict, o)
                push!(odict[o], k=>sp)
            else
                odict[o] = [k=>sp]
            end

            r_sum += r
        end
    end

    next_r = 0.0
    for (o, scenarios) in odict
        bp = ScenarioBelief(scenarios, b.random_source, b.depth+1, o)
            @show(first(scenarios)[2].cart)
        if length(scenarios) == 1
            next_r += rollout_for_one_scenario(pomdp, policy, bp, steps-1, fval)
        else
            next_r += branching_sim_rollout_path(pomdp, policy, bp, steps-1, fval)
        end
    end

    return r_sum + discount(pomdp)*next_r
end


function rollout_for_one_scenario(pomdp::POMDP, policy::Policy, b0::ScenarioBelief, steps::Integer, fval)
    @assert length(b0.scenarios) == 1
    disc = 1.0
    r_total = 0.0
    scenario_mem = copy(b0.scenarios)
    (k, s) = first(b0.scenarios)
    b = ScenarioBelief(scenario_mem, b0.random_source, b0.depth, b0._obs)

    while !isterminal(pomdp, s) && steps > 0
        a = action(policy, b)

        rng = ARDESPOT.get_rng(b.random_source, k, b.depth)
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
        if(sp.cart.x == -100.0)
            println("\nTerminal state reached in rollout")
            println(s.cart)
            println(s.pedestrians)
            println(a)
            println(sp)
            # @show("In Rollout ", sp.cart)
        end
        r_total += disc*r

        s = sp
        scenario_mem[1] = k=>s
        b = ScenarioBelief(scenario_mem, b.random_source, b.depth+1, o)

        disc *= discount(pomdp)
        steps -= 1
    end

    if steps == 0 && !isterminal(pomdp, s)
        r_total += disc*fval(pomdp, b)
    end

    return r_total
end



function calculate_lower_bound_policy_pomdp_planning_2D_action_space_debug(m,b,io=nothing)
    #Implement a reactive controller for your lower bound
    speed_change_to_be_returned = 1.0
    delta_angle = 0.0
    d_far_threshold = 6.0
    d_near_threshold = 4.0
    #This bool is also used to check if all the states in the belief are terminal or not.
    first_execution_flag = true
    nearest_prm_point = lookup_table_struct(-1, 0.0, 0.0, -1, 0.0, 0.0)
    if(io!=nothing)
        write_and_print(io,"Current Depth -> " * string(b.depth) )
        write_and_print(io,"Number of scenarios in this belief -> " * string(length(b.scenarios)) )
    else
        println("Current Depth -> ", b.depth)
        println("Number of scenarios in this belief -> " , length(b.scenarios))
    end
    for (s, w) in weighted_particles(b)
        if(s.cart.x == -100.0 && s.cart.y == -100.0)
            if(io!=nothing)
                write_and_print( io,"Terminal :(" * string(s) )
            else
                println("Terminal :(" , s)
            end
            continue
        else
            if(first_execution_flag)
                if(io!=nothing)
                    write_and_print( io,"State -> " * string(s) )
                else
                    println("State -> " , s)
                end

                if(is_within_range_check_with_points(s.cart.x,s.cart.y, s.cart.goal.x, s.cart.goal.y, m.cart_goal_reached_distance_threshold))
                    if(io!=nothing)
                        write_and_print( io,"WOW! Goal reched!" )
                    else
                        println("WOW! Goal reched!")
                    end
                end
                x_point =  clamp(floor(Int64,s.cart.x/ 1.0)+1,1,100)
                y_point =  clamp(floor(Int64,s.cart.y/ 1.0)+1,1,100)
                # theta_point = clamp(floor(Int64,s.cart.theta/(pi/18))+1,1,36)
                # nearest_prm_point -> Format (vertex_num, x_coordinate, y_coordinate, prm_dist_to_goal)
                nearest_prm_point = m.lookup_table[x_point,y_point]
                first_execution_flag = false
            end
            dist_to_closest_human = 200.0  #Some really big infeasible number (not Inf because avoid the type mismatch error)
            for human in s.pedestrians
                euclidean_distance = sqrt((s.cart.x - human.x)^2 + (s.cart.y - human.y)^2)
                if(euclidean_distance < dist_to_closest_human)
                    dist_to_closest_human = euclidean_distance
                end
                if(dist_to_closest_human < d_near_threshold)
                    # println("Too close ", dist_to_closest_human )
                    tbr_a = POMDP_2D_action_type(-10.0,-1.0,nearest_prm_point.closest_prm_vertex_x, nearest_prm_point.closest_prm_vertex_y,
                                                nearest_prm_point.next_prm_vertex_x, nearest_prm_point.next_prm_vertex_y)
                    if(io!=nothing)
                        write_and_print(io,"Action -> " * string(tbr_a))
                    else
                        println("Action -> ", tbr_a)
                    end
                    return tbr_a
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

    #This condition is true only when all the states in the belief are terminal. In that case, just return (0.0,0.0)
    if(first_execution_flag == true)
        #@show(0.0,0.0)
        tbr_a = POMDP_2D_action_type(-10.0,0.0,nearest_prm_point.closest_prm_vertex_x, nearest_prm_point.closest_prm_vertex_y,
                                    nearest_prm_point.next_prm_vertex_x, nearest_prm_point.next_prm_vertex_y)
        if(io!=nothing)
            write_and_print(io,"Action -> " * string(tbr_a) )
        else
            println("Action -> ", tbr_a)
        end
        return tbr_a
    end

    #This means all humans are away and you can accelerate.
    if(speed_change_to_be_returned == 1.0)
        #@show(0.0,speed_change_to_be_returned)
        tbr_a = POMDP_2D_action_type(-10.0,speed_change_to_be_returned,nearest_prm_point.closest_prm_vertex_x, nearest_prm_point.closest_prm_vertex_y,
                                    nearest_prm_point.next_prm_vertex_x, nearest_prm_point.next_prm_vertex_y)
        if(io!=nothing)
            write_and_print(io,"Action -> " * string(tbr_a) )
        else
            println("Action -> ", tbr_a)
        end
        return tbr_a
    end

    #If code has reached this point, then the best action is to maintain your current speed.
    #We have already found the best steering angle to take.
    #@show(best_delta_angle,0.0)
    tbr_a = POMDP_2D_action_type(-10.0,0.0,nearest_prm_point.closest_prm_vertex_x, nearest_prm_point.closest_prm_vertex_y,
                                nearest_prm_point.next_prm_vertex_x, nearest_prm_point.next_prm_vertex_y)
    if(io!=nothing)
        write_and_print(io,"Action -> " * string(tbr_a) )
    else
        println("Action -> ",  tbr_a)
    end
    return tbr_a

end

function get_the_delta_angle_action_from_pomdp_state(m,pomdp_state)
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
end
