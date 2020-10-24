bad = []

function debug_not_correctly_working_error_2D_action_space(which_env)
    pomdp_debugging_env = deepcopy(all_observed_environments[which_env])
    current_belief_debugging = all_generated_beliefs[which_env]

    golfcart_pomdp_debug() =  POMDP_Planner_2D_action_space(0.99,1.0,-100.0,2.0,-100.0,1.0,1.0,100.0,7.0,pomdp_debugging_env,1)
    discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
    isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,terminal_cart_state);
    actions(::POMDP_Planner_2D_action_space) = [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]

    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space)),
            debug_golfcart_upper_bound_2D_action_space, check_terminal=true),K=100,D=50,T_max=0.8, tree_in_info=true)
    planner = POMDPs.solve(solver, golfcart_2D_action_space_pomdp());

    m_ub_debugging = golfcart_pomdp_debug()
    b_ub_debugging = POMDP_2D_action_space_state_distribution(m_ub_debugging.world,current_belief_debugging)
    a, info = action_info(planner, b_ub_debugging);
    @show(a)
end
