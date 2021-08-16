function lala()
    folder_location = "./scenario_3/"
    #Delete output txt files
    #1D planner
    foreach(rm, filter(endswith(".txt"), readdir(folder_location*"1D",join=true)))
    #2D planner
    foreach(rm, filter(endswith(".txt"), readdir(folder_location*"2D",join=true)))

    #Delete jld2 files
    #1D planner
    foreach(rm, filter(endswith(".jld2"), readdir(folder_location*"1D/risky_scenarios",join=true)))
    #2D planner
    foreach(rm, filter(endswith(".jld2"), readdir(folder_location*"2D/risky_scenarios",join=true)))
end

function damn()
    try
        try
            sqrt("ten")
        catch e
            println("You should have entered a numeric value")
            println(e)
        end
        ss
    catch f
        println("2 step verification DONE" , f)
    end
end

function haha(a::Union{Tuple{Float64,Float64}, Tuple{Int,Int,Float64}})::Float64
    typeof_action = typeof(a)
    println(typeof_action)
    if(typeof_action == Tuple{Float64,Float64})
        println("Action is Float")
    elseif(typeof_action == Tuple{Int,Int})
        println("Action is PRM vertex")
    end
    println("hey, your action is " ,a)
    return 5.0
end

tbr_flag = false
if(tbr_flag)
    # solver = DESPOTSolver(epsilon_0=0.0,
    #                       K=500,
    #                       D=50,
    #                       bounds=bounds,
    #                       T_max=Inf,
    #                       max_trials=100,
    #                       rng=MersenneTwister(4),
    #                       # random_source=FastMersenneSource(500, 10)
    #                       random_source=MemorizingSource(500, 90, MersenneTwister(5))
    #                       )
    #
    #
    #                       solver = DESPOTSolver(epsilon_0=0.0,
    #                                             K=100,
    #                                             lambda=0.01,
    #                                             bounds=bounds,
    #                                             max_trials=3,
    #                                             rng=MersenneTwister(4),
    #                                             random_source=MemorizingSource(500, 90, MersenneTwister(5))
    #                                            )
    #
    #
    # rand_noise_generator_seed_for_env = 1100709942
    # rand_noise_generator_seed_for_sim = 2447832946
    # rand_noise_generator_seed_for_prm = 11
    # rand_noise_generator_seed_for_solver = 1216188435
    #
    # loc = 2
    # value_sum = 0.0
    # for (s,w) in weighted_particles(bad[loc][3])
    #     @show(s,w)
    #     #@show(s.cart)
    #     if(s.cart.x == -100.0 && s.cart.y == -100.0)
    #         value_sum += 0.0
    #     elseif(is_within_range(location(s.cart.x,s.cart.y), s.cart.goal, bad[loc][4].cart_goal_reached_distance_threshold))
    #         @show(w*bad[loc][4].goal_reward)
    #         value_sum += w*bad[loc][4].goal_reward
    #         @show("A")
    #     elseif(debug_is_collision_state_pomdp_planning_2D_action_space(s,bad[loc][4]))
    #         @show("B")
    #         @show(w*bad[loc][4].pedestrian_collision_penalty)
    #         value_sum += w*bad[loc][4].pedestrian_collision_penalty
    #     else
    #         @show("C")
    #         @show( w*((discount(bad[loc][4])^time_to_goal_pomdp_planning_2D_action_space(s,bad[loc][4].max_cart_speed))*bad[loc][4].goal_reward) )
    #         value_sum += w*((discount(bad[loc][4])^time_to_goal_pomdp_planning_2D_action_space(s,bad[loc][4].max_cart_speed))*bad[loc][4].goal_reward)
    #         @show(value_sum)
    #     end
    #     temp = POMDPs.gen(bad[loc][4], s, (0.0,1.0), MersenneTwister(1234))
    #     @show(POMDPs.isterminal(bad[loc][4],s))
    #     println(temp)
    #     println()
    #     #search_state = s
    # end
    # @show(value_sum)
    # end
    #
    # st = POMDP_state_2D_action_space(
    # cart_state(4.680500532311589, 30.34981953744709, 1.2598421566986628, 1.0, 1.0, location(100.0, 75.0)),
    # human_state[human_state(5.10174584838373, 33.51861483465669, 1.0, location(100.0, 0.0), 9.0), human_state(5.490315956158676, 38.80774833521536, 1.0, location(100.0, 100.0), 42.0), human_state(5.26381686605569, 37.474119131596765, 1.0, location(100.0, 100.0), 139.0), human_state(9.155031191527854, 28.463563363880436, 1.0, location(100.0, 100.0), 140.0), human_state(0.8362702404760353, 32.61453937856538, 1.0, location(0.0, 0.0), 146.0), human_state(6.4193580609883165, 36.723470541686616, 1.0, location(0.0, 100.0), 241.0)] )
end


function lala()
    s = 2
    for i in 1:10
        s = i+2
    end
    println(s)
end

struct temp_check
    num::Int64
    x::Float64
    y::Float64
end
