#Code to visualize the Hybrid A* path at any moment of the cart's journey

# hybrid_a_star_search(var1.cart.x, var1.cart.y,
#                    var1.cart.theta, var1.cart.goal.x, var1.cart.goal.y, var1, var1_belief);


#debug_num_index = 49
function debugging_path(debug_num_index)

    var1 = all_observed_environments[debug_num_index][1]
    var1_belief = all_observed_environments[debug_num_index][2]

    hybrid_astar_debugging_env = deepcopy(var1)
    humans_to_avoid = get_nearest_six_pedestrians(hybrid_astar_debugging_env,var1_belief)
    hybrid_astar_debugging_env.cart_hybrid_astar_path = @time hybrid_a_star_search(var1.cart.x, var1.cart.y,
                       var1.cart.theta, var1.cart.goal.x, var1.cart.goal.y, var1,humans_to_avoid );
    x_init,y_init,theta_init = hybrid_astar_debugging_env.cart.x,
                            hybrid_astar_debugging_env.cart.y, hybrid_astar_debugging_env.cart.theta
    initial_state = [x_init,y_init,theta_init]
    path_x, path_y, path_theta = [x_init],[y_init],[theta_init]

    for steering_angle in hybrid_astar_debugging_env.cart_hybrid_astar_path
        extra_parameters = [1.0, hybrid_astar_debugging_env.cart.L, steering_angle]
        x,y,theta = get_intermediate_points(initial_state, 1.0, extra_parameters);
        push!(path_x,last(x))
        push!(path_y,last(y))
        push!(path_theta,last(theta))
        initial_state = [last(x),last(y),last(theta)]
    end
    hybrid_astar_debugging_env.cart_hybrid_astar_path = []
    anim = @animate for i ∈ 1:length(path_x)
        hybrid_astar_debugging_env.cart.x = path_x[i]
        hybrid_astar_debugging_env.cart.y = path_y[i]
        display_env(hybrid_astar_debugging_env);
    end
    gif(anim, "hybrid_a_star_path_checking.gif", fps = 10)
    @show(humans_to_avoid)
end
#debugging_path(debug_num_index)

# function something_random()
#
#     dummy_env = experiment_environment(100.0, 100.0, 20,
#     location[location(0.0, 0.0), location(0.0, 100.0), location(100.0, 100.0), location(100.0, 0.0)],
#
#     human_state[human_state(94.63217173850562, 61.88841934339001, 1.0, location(100.0, 100.0), 1.0), human_state(64.04850386605376, 78.75593410266809, 1.0, location(100.0, 100.0), 2.0), human_state(13.548569495513004, 90.76233898033203, 1.0, location(0.0, 100.0), 3.0), human_state(65.01290758670531, 49.66742638448621, 1.0, location(0.0, 100.0), 4.0), human_state(25.41966711479815, 69.91141443554505, 1.0, location(0.0, 100.0), 5.0), human_state(100.0, 100.0, 1.0, location(100.0, 100.0), 6.0), human_state(7.63414045316239, 59.80076688310543, 1.0, location(0.0, 0.0), 7.0), human_state(61.49668649646925, 27.745034730485386, 1.0, location(100.0, 0.0), 8.0), human_state(47.40178948747595, 43.609646328477865, 1.0, location(0.0, 0.0), 9.0), human_state(46.27910986919878, 87.92788985824689, 1.0, location(100.0, 100.0), 10.0), human_state(27.865867173996342, 71.06236870392688, 1.0, location(0.0, 100.0), 11.0), human_state(52.04386487257814, 48.70771968843845, 1.0, location(0.0, 0.0), 12.0), human_state(11.686216933337255, 88.31378306666272, 1.0, location(0.0, 100.0), 13.0), human_state(89.24705068285756, 46.83263948746251, 1.0, location(100.0, 100.0), 14.0), human_state(82.33001286624723, 40.309658148873446, 1.0, location(100.0, 0.0), 15.0), human_state(40.673623577045205, 10.84460644290571, 1.0, location(100.0, 0.0), 16.0), human_state(75.23528979591792, 60.608369709990384, 1.0, location(100.0, 0.0), 17.0), human_state(85.43231967957287, 82.74879962054679, 1.0, location(100.0, 100.0), 18.0), human_state(74.24519071299902, 56.68509347186195, 1.0, location(100.0, 100.0), 19.0), human_state(34.0818131907156, 31.184310473823977, 1.0, location(100.0, 100.0), 20.0),
#     human_state(44.0818131907156, 31.184310473823977, 1.0, location(100.0, 100.0), 21.0)],
#
#     obstacle_location[obstacle_location(50.0, 50.0, 3.0), obstacle_location(33.0, 69.0, 5.0), obstacle_location(73.0, 79.0, 3.0)],
#
#     cart_state(46.58537465073062, 57.21889235742753, 0.4188790204786391, 5.0, 1.0, location(100.0, 50.0)),
#
#     human_state[human_state(64.04850386605376, 78.75593410266809, 1.0, location(100.0, 100.0), 2.0), human_state(65.01290758670531, 49.66742638448621, 1.0, location(0.0, 100.0), 4.0), human_state(25.41966711479815, 69.91141443554505, 1.0, location(0.0, 100.0), 5.0), human_state(47.40178948747595, 43.609646328477865, 1.0, location(0.0, 0.0), 9.0), human_state(27.865867173996342, 71.06236870392688, 1.0, location(0.0, 100.0), 11.0), human_state(52.04386487257814, 48.70771968843845, 1.0, location(0.0, 0.0), 12.0), human_state(75.23528979591792, 60.608369709990384, 1.0, location(100.0, 0.0), 17.0), human_state(74.24519071299902, 56.68509347186195, 1.0, location(100.0, 100.0), 19.0), human_state(34.0818131907156, 31.184310473823977, 1.0, location(100.0, 100.0), 20.0),
#     human_state(44.0818131907156, 31.184310473823977, 1.0, location(100.0, 100.0), 21.0)],
#
#     Float64[])
#
#
#
#
#     dummy_belief = [human_probability_over_goals([0.19869494402858412, 0.21765175520088775, 0.3288955917175596, 0.2547577090529685]),
#      human_probability_over_goals([0.21416987762877016, 0.616420776904841, 0.15337546052465253, 0.016033884941736347]),
#      human_probability_over_goals([1.4247432938124952e-10, 0.9999999933247531, 6.5327725964555004e-9, 1.281326111489158e-17]),
#      human_probability_over_goals([0.9992210662411697, 0.0005304407392283801, 4.633958562230638e-9, 0.0002484883856433518])  ,
#      human_probability_over_goals([1.2530391372467218e-8, 0.999999961338052, 2.6131552067268344e-8, 4.447982278219262e-15])  ,
#      human_probability_over_goals([0.9973100171370589, 0.0016696049610358478, 1.3069737806285022e-8, 0.0010203648321673916]) ,
#      human_probability_over_goals([0.26638849526023195, 0.2092748419127956, 0.2192339014830492, 0.3051027613439232])         ,
#      human_probability_over_goals([0.1664200821927678, 0.2625274219573986, 0.36717497175263325, 0.2038775240972004])         ,
#      human_probability_over_goals([1.4347058114331389e-8, 0.016179992761376192, 0.9697525728941828, 0.014067419997382934])   ,
#      human_probability_over_goals([0.25,0.25,0.25,0.25])
#     ]
#
#
#     golfcart_dummy() = Speed_Planner_POMDP(0.9,4.0,-100.0,1.0,1.0,100.0,7.0,dummy_env,1)
#     discount(p::Speed_Planner_POMDP) = p.discount_factor
#     isterminal(::Speed_Planner_POMDP, s::SP_POMDP_state) = isgoalstate_pomdp_planning(s,terminal_cart_state);
#     actions(::Speed_Planner_POMDP) = [-1.0, 0.0, 1.0]
#
#
#     m = golfcart_dummy()
#     b = @time initialstate_distribution(m,dummy_belief);
#
#
#
#     which_env = 41
#     golfcart_temp() = Speed_Planner_POMDP(0.9,4.0,-100.0,1.0,1.0,100.0,7.0,all_observed_environments[which_env][1],1)
#     discount(p::Speed_Planner_POMDP) = p.discount_factor
#     isterminal(::Speed_Planner_POMDP, s::SP_POMDP_state) = isgoalstate_pomdp_planning(s,terminal_cart_state);
#     actions(::Speed_Planner_POMDP) = [-1.0, 0.0, 1.0]
#     m = golfcart_temp()
#     b = @time initialstate_distribution(m,all_observed_environments[which_env][2]);
#
#
#
#     temp_belief = [
#     human_probability_over_goals([0.25,0.25,0.25,0.25]),
#     human_probability_over_goals([0.25,0.25,0.25,0.25]),
#     human_probability_over_goals([0.25,0.25,0.25,0.25]),
#     human_probability_over_goals([0.25,0.25,0.25,0.25]),
#     human_probability_over_goals([0.25,0.25,0.25,0.25]),
#     human_probability_over_goals([0.25,0.25,0.25,0.25]),
#     human_probability_over_goals([0.25,0.25,0.25,0.25]),
#     human_probability_over_goals([0.25,0.25,0.25,0.25])
#     ]
# end

function lala()
    current_x, current_y, current_theta = 0.0,0.0,0.0
    new_cart_speed = 5.0
    arc_length = new_cart_speed
    time_interval = 1.0/new_cart_speed
    steering_angle = pi/6
    cart_path = Tuple{Float64,Float64,Float64}[]
    push!(cart_path,(Float64(current_x), Float64(current_y), Float64(current_theta)))
    for i in (1:new_cart_speed)
        new_theta = current_theta + (arc_length * tan(steering_angle) * time_interval / env.cart.L)
        new_theta = wrap_between_0_and_2Pi(new_theta)
        new_x = current_x + ((env.cart.L / tan(steering_angle)) * (sin(new_theta) - sin(current_theta)))
        new_y = current_y + ((env.cart.L / tan(steering_angle)) * (cos(current_theta) - cos(new_theta)))
        push!(cart_path,(Float64(new_x), Float64(new_y), Float64(new_theta)))
        current_x, current_y,current_theta = new_x,new_y,new_theta
    end
    return cart_path
end
#lala()

function lala(env,param)
    new_cart_speed = 7.0
    current_x, current_y, current_theta = env.cart.x,env.cart.y,env.cart.theta
    arc_length = new_cart_speed
    steering_angle = pi/9
    cart_path = Tuple{Float64,Float64,Float64}[]
    push!(cart_path,(Float64(current_x), Float64(current_y), Float64(current_theta)))

    if(param == 1)
        time_interval = 1.0/new_cart_speed
        for i in (1:new_cart_speed)
            new_theta = current_theta + (arc_length * tan(steering_angle) * time_interval / env.cart.L)
            new_theta = wrap_between_0_and_2Pi(new_theta)
            new_x = current_x + ((env.cart.L / tan(steering_angle)) * (sin(new_theta) - sin(current_theta)))
            new_y = current_y + ((env.cart.L / tan(steering_angle)) * (cos(current_theta) - cos(new_theta)))
            push!(cart_path,(Float64(new_x), Float64(new_y), Float64(new_theta)))
            current_x, current_y,current_theta = new_x,new_y,new_theta
        end
    else
        time_interval = 1.0
        new_theta = current_theta + (arc_length * tan(steering_angle) * time_interval / env.cart.L)
        new_theta = wrap_between_0_and_2Pi(new_theta)
        new_x = current_x + ((env.cart.L / tan(steering_angle)) * (sin(new_theta) - sin(current_theta)))
        new_y = current_y + ((env.cart.L / tan(steering_angle)) * (cos(current_theta) - cos(new_theta)))
        push!(cart_path,(Float64(new_x), Float64(new_y), Float64(new_theta)))
        current_x, current_y,current_theta = new_x,new_y,new_theta
    end

    hybrid_astar_debugging_env = deepcopy(env)
    anim = @animate for i ∈ 1:length(cart_path)
        hybrid_astar_debugging_env.cart.x = cart_path[i][1]
        hybrid_astar_debugging_env.cart.y = cart_path[i][2]
        display_env(hybrid_astar_debugging_env);
    end
    gif(anim, "debug_lala.gif", fps = 10)

    return cart_path
end
#lala(env,1)
