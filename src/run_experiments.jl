include("new_main_2d_action_space_pomdp.jl")

rand_rng = MersenneTwister(100)

discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
actions(::POMDP_Planner_2D_action_space) = [(-10.0,-10.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0)]

function run_experiment_for_given_world_and_noise(world, rand_noise_generator, iteration_num)

    #Create POMDP for env_right_now
    env_right_now = deepcopy(world)
    m = POMDP_Planner_2D_action_space(0.99,2.0,-100.0,1.0,-100.0,1.0,1.0,100.0,5.0,env_right_now,1)
    solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(calculate_lower_bound_policy_pomdp_planning_2D_action_space),max_depth=100,
                            final_value=reward_to_be_awarded_at_max_depth_in_lower_bound_policy_rollout),
                            calculate_upper_bound_value_pomdp_planning_2D_action_space, check_terminal=true),K=100,D=100,T_max=0.5, tree_in_info=true)
    planner = POMDPs.solve(solver, m);

    #display_env(golfcart_2D_action_space_pomdp().world)
    just_2D_pomdp_all_gif_environments, just_2D_pomdp_all_observed_environments, just_2D_pomdp_all_generated_beliefs,
            just_2D_pomdp_all_generated_trees, just_2D_pomdp_all_risky_scenarios, just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,
            just_2D_pomdp_time_taken_by_cart, just_2D_pomdp_cart_reached_goal_flag = run_one_simulation(env_right_now, rand_noise_generator, m, planner)

    anim = @animate for i ∈ 1:length(just_2D_pomdp_all_observed_environments)
        display_env(just_2D_pomdp_all_observed_environments[i]);
        savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*string(i)*".png")
    end

    gif_name = "./scenario_1_gifs/just_2D_action_space_pomdp_planner_run_"*string(iteration_num)*"_"*string(just_2D_pomdp_cart_reached_goal_flag)
    gif_name = gif_name*"_"*string(just_2D_pomdp_time_taken_by_cart)*"_"*string(just_2D_pomdp_number_risks)*".gif"
    gif(anim, gif_name, fps = 2)

    return just_2D_pomdp_number_risks,just_2D_pomdp_number_of_sudden_stops,just_2D_pomdp_time_taken_by_cart, just_2D_pomdp_cart_reached_goal_flag
end

for iteration_num in 1:20
    rand_noise_generator_seed = Int(ceil(100*rand(rand_rng)))
    rand_noise_generator = MersenneTwister(rand_noise_generator_seed)
    env = generate_environment_no_obstacle(100, rand_noise_generator)
    run_experiment_for_given_world_and_noise(env, MersenneTwister(7), iteration_num)
end
