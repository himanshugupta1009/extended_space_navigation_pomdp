#=
expt_file_name = "./scenario_2/2D/risky_scenarios/expt_50.jld2";
expt_details_dict = load(expt_file_name);
test_time_step = "34";
b = POMDP_2D_action_space_state_distribution(expt_details_dict["all_observed_environments"]["t="*test_time_step],expt_details_dict["all_generated_beliefs"]["t="*test_time_step]);
copy_of_planner = deepcopy(expt_details_dict["all_planners"]["t="*test_time_step]);
supposed_a, supposed_info = action_info(copy_of_planner, b);
inchrome(D3Tree(supposed_info[:tree]))
supposed_a[1]*180/pi
=#

#=
test_time_step = "43";
b = POMDP_2D_action_space_state_distribution(h["all_observed_environments"]["t="*test_time_step],h["all_generated_beliefs"]["t="*test_time_step]);
copy_of_planner = deepcopy(h["all_planners"]["t="*test_time_step]);
supposed_a, supposed_info = action_info(copy_of_planner, b);
inchrome(D3Tree(supposed_info[:tree]))
supposed_a
inchrome(D3Tree(supposed_info[:tree]))
=#


#=
test_time_step = "57";
b = POMDP_2D_action_space_state_distribution(just_2D_pomdp_all_observed_environments["t="*test_time_step],just_2D_pomdp_all_generated_beliefs["t="*test_time_step]);
copy_of_planner = deepcopy(just_2D_pomdp_all_planners["t="*test_time_step]);
supposed_a, supposed_info = action_info(copy_of_planner, b);
despot_tree = supposed_info[:tree];
curr_scenario_belief = ARDESPOT.get_belief(despot_tree,1,deepcopy(just_2D_pomdp_all_planners["t="*test_time_step]).rs);
ARDESPOT.lbound(copy_of_planner.bounds.lower, copy_of_planner.pomdp, curr_scenario_belief)
ARDESPOT.ubound(copy_of_planner.bounds.upper, copy_of_planner.pomdp, curr_scenario_belief)

supposed_a[1]*180/pi
#

# curr_scenarios = supposed_info[:tree].scenarios[1];
# curr_scenario_belief =
ARDESPOT.lbound(copy_of_planner.bounds.lower, copy_of_planner.pomdp, curr_scenario_belief)
ARDESPOT.ubound(copy_of_planner.bounds.upper, copy_of_planner.pomdp, curr_scenario_belief)


test_time_step = "34";
b = POMDP_2D_action_space_state_distribution(expt_details_dict["all_observed_environments"]["t="*test_time_step],expt_details_dict["all_generated_beliefs"]["t="*test_time_step]);
copy_of_planner = deepcopy(expt_details_dict["all_planners"]["t="*test_time_step]);
root_scenarios = [i=>rand(copy_of_planner.rng, b) for i in 1:copy_of_planner.sol.K];
curr_scenario_belief = ScenarioBelief(root_scenarios, copy_of_planner.rs, 0, b);
L_0, U_0 = bounds(copy_of_planner.bounds, copy_of_planner.pomdp, curr_scenario_belief)
lb_policy = DefaultPolicyLB(FunctionPolicy(b->calculate_lower_bound_policy_pomdp_planning_2D_action_space_debug(golfcart_2D_action_space_pomdp, b,io)),max_depth=100);
lb_policy = DefaultPolicyLB(FunctionPolicy(b->calculate_lower_bound_policy_pomdp_planning_2D_action_space(golfcart_2D_action_space_pomdp, b)),max_depth=100);
io = open("random_file.txt", "w")
ARDESPOT.lbound(lb_policy, copy_of_planner.pomdp, curr_scenario_belief)
close(io)
ARDESPOT.ubound(copy_of_planner.bounds.upper, copy_of_planner.pomdp, curr_scenario_belief)
calculate_lower_bound_policy_pomdp_planning_2D_action_space_debug(copy_of_planner.pomdp, curr_scenario_belief)


gen_action_debugging = POMDP_2D_action_type(0.0, -1.0, false)
gen_action_debugging = POMDP_2D_action_type(0.0, 0.0, false)
curr_pomdp_state = curr_scenario_belief.scenarios[1][2]
POMDPs.gen(golfcart_2D_action_space_pomdp, curr_pomdp_state, gen_action_debugging, MersenneTwister(1234))

new_pomdp_state = POMDP_state_2D_action_space(cart_state(11.254059234304371, 33.770025148236186, 0.9201807793229126, 2.0, 1.0, location(100.0, 75.0)), human_state[human_state(26.540962076986272, 32.66428415194191, 1.0, location(100.0, 100.0), 2.0), human_state(23.34710220661714, 47.10950052256582, 1.0, location(100.0, 100.0), 9.0), human_state(25.81733060754584, 32.05392277150804, 1.0, location(100.0, 100.0), 105.0), human_state(5.80010818880632, 50.37685216243477, 1.0, location(0.0, 100.0), 140.0), human_state(5.853906380814106, 13.429549932455878, 1.0, location(0.0, 0.0), 256.0), human_state(1.0409728300598835, 4.163891320239533, 1.0, location(0.0, 0.0), 281.0)])
new_scenario = [1=>new_pomdp_state]
new_scenario_belief = ScenarioBelief(new_scenario, copy_of_planner.rs, 0, b);
action_according_to_rollout_policy_at_this_belief = calculate_lower_bound_policy_pomdp_planning_2D_action_space_debug(copy_of_planner.pomdp, new_scenario_belief)
println(action_according_to_rollout_policy_at_this_belief)
POMDPs.gen(golfcart_2D_action_space_pomdp, new_pomdp_state, action_according_to_rollout_policy_at_this_belief, MersenneTwister(1234))


x_point =  floor(Int64,new_pomdp_state.cart.x/ 1.0)+1
y_point =  floor(Int64,new_pomdp_state.cart.y/ 1.0)+1
theta_point = clamp(floor(Int64,new_pomdp_state.cart.theta/(pi/18))+1,1,36)
nearest_prm_point = lookup_table[x_point,y_point,theta_point]
=#


#=
anim = @animate for k ∈ keys(just_2D_pomdp_all_gif_environments)
    display_env(just_2D_pomdp_all_gif_environments[k],k);
    #savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*all_gif_environments[i][1]*".png")
end
gif(anim, "just_2D_action_space_pomdp_planner_run.gif", fps = 20)
=#

#=
anim = @animate for k ∈ keys(just_2D_pomdp_all_gif_environments)
    display_env(just_2D_pomdp_all_gif_environments[k],k);
    #savefig("./plots_just_2d_action_space_pomdp_planner/plot_"*all_gif_environments[i][1]*".png")
end
gif(anim, "just_2D_action_space_pomdp_planner_run.gif", fps = 20)
=#

#inchrome(D3Tree(just_2D_pomdp_all_generated_trees[9][:tree]))



#=
test_time_step = "31";
display_env(h["all_observed_environments"]["t="*test_time_step])
h["all_observed_environments"]["t="*test_time_step].cart
=#



#=
test_time_step = "26";
b = POMDP_2D_action_space_state_distribution(just_2D_pomdp_all_observed_environments["t="*test_time_step],just_2D_pomdp_all_generated_beliefs["t="*test_time_step]);
copy_of_planner = deepcopy(planner);
supposed_a, supposed_info = action_info(copy_of_planner, b);
despot_tree = supposed_info[:tree];
curr_scenario_belief = ARDESPOT.get_belief(despot_tree,1,deepcopy(just_2D_pomdp_all_planners["t="*test_time_step]).rs);
ARDESPOT.lbound(copy_of_planner.bounds.lower, copy_of_planner.pomdp, curr_scenario_belief)
ARDESPOT.ubound(copy_of_planner.bounds.upper, copy_of_planner.pomdp, curr_scenario_belief)
=#


#=
test_time_step = "258";
b = POMDP_2D_action_space_state_distribution(h["all_observed_environments"]["t="*test_time_step],h["all_generated_beliefs"]["t="*test_time_step]);
copy_of_planner = deepcopy(planner);
supposed_a, supposed_info = action_info(copy_of_planner, b);
despot_tree = supposed_info[:tree];
curr_scenario_belief = ARDESPOT.get_belief(despot_tree,1,deepcopy(planner.rs));
ARDESPOT.lbound(copy_of_planner.bounds.lower, copy_of_planner.pomdp, curr_scenario_belief)
ARDESPOT.ubound(copy_of_planner.bounds.upper, copy_of_planner.pomdp, curr_scenario_belief)
=#
