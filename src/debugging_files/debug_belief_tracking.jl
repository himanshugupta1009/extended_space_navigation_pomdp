user_defined_rng = MersenneTwister(7)
one_time_step = 0.5
env_copy = deepcopy(env)
all_observed_environments = []
lidar_range = 30

function some_dummy_env_and_belief()
    dummy_env = experiment_environment(100.0, 100.0, 20,
    location[location(0.0, 0.0), location(0.0, 100.0), location(100.0, 100.0), location(100.0, 0.0)],

    human_state[human_state(94.63217173850562, 61.88841934339001, 1.0, location(100.0, 100.0), 1.0), human_state(64.04850386605376, 78.75593410266809, 1.0, location(100.0, 100.0), 2.0), human_state(13.548569495513004, 90.76233898033203, 1.0, location(0.0, 100.0), 3.0), human_state(65.01290758670531, 49.66742638448621, 1.0, location(0.0, 100.0), 4.0), human_state(25.41966711479815, 69.91141443554505, 1.0, location(0.0, 100.0), 5.0), human_state(100.0, 100.0, 1.0, location(100.0, 100.0), 6.0), human_state(7.63414045316239, 59.80076688310543, 1.0, location(0.0, 0.0), 7.0), human_state(61.49668649646925, 27.745034730485386, 1.0, location(100.0, 0.0), 8.0), human_state(47.40178948747595, 43.609646328477865, 1.0, location(0.0, 0.0), 9.0), human_state(46.27910986919878, 87.92788985824689, 1.0, location(100.0, 100.0), 10.0), human_state(27.865867173996342, 71.06236870392688, 1.0, location(0.0, 100.0), 11.0), human_state(52.04386487257814, 48.70771968843845, 1.0, location(0.0, 0.0), 12.0), human_state(11.686216933337255, 88.31378306666272, 1.0, location(0.0, 100.0), 13.0), human_state(89.24705068285756, 46.83263948746251, 1.0, location(100.0, 100.0), 14.0), human_state(82.33001286624723, 40.309658148873446, 1.0, location(100.0, 0.0), 15.0), human_state(40.673623577045205, 10.84460644290571, 1.0, location(100.0, 0.0), 16.0), human_state(75.23528979591792, 60.608369709990384, 1.0, location(100.0, 0.0), 17.0), human_state(85.43231967957287, 82.74879962054679, 1.0, location(100.0, 100.0), 18.0), human_state(74.24519071299902, 56.68509347186195, 1.0, location(100.0, 100.0), 19.0), human_state(34.0818131907156, 31.184310473823977, 1.0, location(100.0, 100.0), 20.0),
    human_state(44.0818131907156, 31.184310473823977, 1.0, location(100.0, 100.0), 21.0)],

    obstacle_location[obstacle_location(50.0, 50.0, 3.0), obstacle_location(33.0, 69.0, 5.0), obstacle_location(73.0, 79.0, 3.0)],

    cart_state(46.58537465073062, 57.21889235742753, 0.4188790204786391, 5.0, 1.0, location(100.0, 50.0)),

    human_state[human_state(64.04850386605376, 78.75593410266809, 1.0, location(100.0, 100.0), 2.0), human_state(65.01290758670531, 49.66742638448621, 1.0, location(0.0, 100.0), 4.0), human_state(25.41966711479815, 69.91141443554505, 1.0, location(0.0, 100.0), 5.0), human_state(47.40178948747595, 43.609646328477865, 1.0, location(0.0, 0.0), 9.0), human_state(27.865867173996342, 71.06236870392688, 1.0, location(0.0, 100.0), 11.0), human_state(52.04386487257814, 48.70771968843845, 1.0, location(0.0, 0.0), 12.0), human_state(75.23528979591792, 60.608369709990384, 1.0, location(100.0, 0.0), 17.0), human_state(74.24519071299902, 56.68509347186195, 1.0, location(100.0, 100.0), 19.0), human_state(34.0818131907156, 31.184310473823977, 1.0, location(100.0, 100.0), 20.0),
    human_state(44.0818131907156, 31.184310473823977, 1.0, location(100.0, 100.0), 21.0)],

    Float64[])


    dummy_belief = [human_probability_over_goals([0.19869494402858412, 0.21765175520088775, 0.3288955917175596, 0.2547577090529685]),
     human_probability_over_goals([0.21416987762877016, 0.616420776904841, 0.15337546052465253, 0.016033884941736347]),
     human_probability_over_goals([1.4247432938124952e-10, 0.9999999933247531, 6.5327725964555004e-9, 1.281326111489158e-17]),
     human_probability_over_goals([0.9992210662411697, 0.0005304407392283801, 4.633958562230638e-9, 0.0002484883856433518])  ,
     human_probability_over_goals([1.2530391372467218e-8, 0.999999961338052, 2.6131552067268344e-8, 4.447982278219262e-15])  ,
     human_probability_over_goals([0.9973100171370589, 0.0016696049610358478, 1.3069737806285022e-8, 0.0010203648321673916]) ,
     human_probability_over_goals([0.26638849526023195, 0.2092748419127956, 0.2192339014830492, 0.3051027613439232])         ,
     human_probability_over_goals([0.1664200821927678, 0.2625274219573986, 0.36717497175263325, 0.2038775240972004])         ,
     human_probability_over_goals([1.4347058114331389e-8, 0.016179992761376192, 0.9697525728941828, 0.014067419997382934])   ,
     human_probability_over_goals([0.25,0.25,0.25,0.25])
    ]
end

#Function that updates the belief based on old cart_lidar_data and new cart_lidar_data
function update_belief_from_old_world_and_new_world(current_belief, old_world, new_world)
    updated_belief = update_belief(current_belief, old_world.goals,
        old_world.cart_lidar_data, new_world.cart_lidar_data)
    return updated_belief
end


function function_to_check_if_modified_update_belief_function_is_working(env_copy)
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
    global current_belief_1 = initial_belief
    global current_belief_2 = initial_belief


    num_of_times_loop_to_run = 5

    for i in 1:num_of_times_loop_to_run
        @show(i)
        #Propogate humans for one time step
        moved_human_positions = Array{human_state,1}()
        for human in env_copy.humans
            push!(moved_human_positions,get_new_human_position_actual_environemnt(human,env_copy,one_time_step,MersenneTwister(7)))
        end
        env_copy.humans = moved_human_positions

        #Sense humans near cart after the first time step
        new_lidar_data = Array{human_state,1}()
        for human in env_copy.humans
            if(is_within_range(location(env_copy.cart.x,env_copy.cart.y), location(human.x,human.y), lidar_range))
                push!(new_lidar_data,human)
            end
        end

        global current_belief_1 = update_belief(current_belief_1, env_copy.goals,
                                            env_copy.cart_lidar_data, new_lidar_data)
        env_copy.cart_lidar_data = new_lidar_data
    end



    env_copy = deepcopy(env)
    env_copy.cart_lidar_data = initial_cart_lidar_data
    for i in 1:num_of_times_loop_to_run

        #Propogate humans for one time step
        moved_human_positions = Array{human_state,1}()
        for human in env_copy.humans
            push!(moved_human_positions,get_new_human_position_actual_environemnt(human,env_copy,one_time_step,MersenneTwister(7)))
        end
        env_copy.humans = moved_human_positions

        #Sense humans near cart after the first time step
        new_lidar_data = Array{human_state,1}()
        for human in env_copy.humans
            if(is_within_range(location(env_copy.cart.x,env_copy.cart.y), location(human.x,human.y), lidar_range))
                push!(new_lidar_data,human)
            end
        end
        new_world = deepcopy(env_copy)
        new_world.cart_lidar_data = new_lidar_data

        global current_belief_2 = update_belief_from_old_world_and_new_world(current_belief_2, env_copy, new_world)
        env_copy.cart_lidar_data = new_lidar_data
    end

    return current_belief_1, current_belief_2
end
