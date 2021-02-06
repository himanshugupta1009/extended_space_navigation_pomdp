bad = []


function debug_ub_lb_error_2D_action_space(all_observed_environments,all_generated_beliefs,which_env)
    pomdp_ub_debugging_env = deepcopy(all_observed_environments[which_env])
    current_belief_debugging = all_generated_beliefs[which_env]

    golfcart_pomdp_debug =  POMDP_Planner_2D_action_space(0.99,2.0,-1000.0,1.0,-1000.0,0.0,1.0,100.0,5.0,pomdp_ub_debugging_env)
    discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
    isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
    actions(::POMDP_Planner_2D_action_space) = [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space), 100, reward_to_be_awarded_at_max_depth_in_lower_bound_policy_rollout),
            debug_golfcart_upper_bound_2D_action_space, check_terminal=true),K=100,D=100,T_max=0.5, tree_in_info=true)
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

    golfcart_pomdp_debug =  POMDP_Planner_2D_action_space(0.99,2.0,-1000.0,1.0,-1000.0,1.0,1.0,100.0,5.0,pomdp_ub_debugging_env)
    discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
    isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
    actions(::POMDP_Planner_2D_action_space) = [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space)),
            debug_golfcart_upper_bound_2D_action_space, check_terminal=true),K=100,D=50,T_max=0.5, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_pomdp_debug);

    m_ub_debugging = golfcart_pomdp_debug

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
