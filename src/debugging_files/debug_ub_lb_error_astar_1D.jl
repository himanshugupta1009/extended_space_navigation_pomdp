#bad = []


function debug_ub_lb_error_1D_action_space(which_env)
    pomdp_ub_debugging_env = deepcopy(astar_1D_all_observed_environments[which_env])
    current_belief_debugging = astar_1D_all_generated_beliefs[which_env]

    golfcart_pomdp_debug() = POMDP_Planner_1D_action_space(0.99,1.0,-100.0,1.0,1.0,1000.0,7.0,pomdp_ub_debugging_env,1)
    discount(p::POMDP_Planner_1D_action_space) = p.discount_factor
    isterminal(::POMDP_Planner_1D_action_space, s::POMDP_state_1D_action_space) = is_terminal_state_pomdp_planning(s,terminal_cart_state);
    actions(::POMDP_Planner_1D_action_space) = [-1.0, 0.0, 1.0, -10.0]

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_1D_action_space)),
            debug_golfcart_upper_bound_1D_action_space, check_terminal=true),K=100,D=50,T_max=0.5, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_pomdp_debug());

    m_ub_debugging = golfcart_pomdp_debug()
    b_ub_debugging = POMDP_1D_action_space_state_distribution(m_ub_debugging.world,current_belief_debugging,m.start_path_index)
    a, info = action_info(planner, b_ub_debugging);
    @show(a)
end

function print_belief_states(which_env,bad,loc)

    pomdp_ub_debugging_env = deepcopy(astar_1D_all_observed_environments[which_env])
    current_belief_debugging = astar_1D_all_generated_beliefs[which_env]

    golfcart_pomdp_debug() = POMDP_Planner_1D_action_space(0.9,2.0,-100.0,1.0,1.0,1000.0,7.0,pomdp_ub_debugging_env,1)
    discount(p::POMDP_Planner_1D_action_space) = p.discount_factor
    isterminal(::POMDP_Planner_1D_action_space, s::POMDP_state_1D_action_space) = is_terminal_state_pomdp_planning(s,terminal_cart_state);
    actions(::POMDP_Planner_1D_action_space) = [-1.0, 0.0, 1.0, -10.0]

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_1D_action_space)),
            debug_golfcart_upper_bound_1D_action_space, check_terminal=true),K=100,D=50,T_max=0.5, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_pomdp_debug());

    m_ub_debugging = golfcart_pomdp_debug()

    value_sum = 0.0
    for (s,w) in weighted_particles(bad[loc][3])
        @show(s,w)
        #@show(s.cart)
        if(s.cart.x == -100.0 && s.cart.y == -100.0)
            value_sum += 0.0
        elseif(is_within_range(location(s.cart.x,s.cart.y), s.cart.goal, m_ub_debugging.cart_goal_reached_distance_threshold))
            value_sum += w*m_ub_debugging.goal_reward
            @show("A")
        elseif(debug_is_collision_state_pomdp_planning_1D_action_space(s,m_ub_debugging))
            @show("B")
            value_sum += w*m_ub_debugging.pedestrian_collision_penalty
        else
            @show("C")
            value_sum += w*((discount(m_ub_debugging)^time_to_goal_pomdp_planning_1D_action_space(s,m_ub_debugging))*m_ub_debugging.goal_reward)
            @show(value_sum)
        end
        temp = POMDPs.gen(m_ub_debugging, s, (1.0), MersenneTwister(1234))
        @show(POMDPs.isterminal(m_ub_debugging,s))
        println(temp)
        #search_state = s
    end
    @show(@time calculate_lower_bound_policy_pomdp_planning_1D_action_space(bad[loc][3]))
    @show(value_sum)
end
