include("utils.jl")

#Functions for tracking belief

#This function returns the distance of a human's position to all the possible goal locations
function calculate_human_dist_from_all_goals(human_position,list_of_goals)
    distance_list = Float64[]
    for goal in list_of_goals
        distance = ((human_position.x-goal.x)^2 + (human_position.y-goal.y)^2)^0.5
        push!(distance_list,distance)
    end
    return distance_list
end

function is_human_present_in_the_list(human,old_lidar_data)
    for human_old_lidar_data_index in 1:length(old_lidar_data)
        if(human.id == old_lidar_data[human_old_lidar_data_index].id)
            return human_old_lidar_data_index
        end
    end
    return -1
end

function update_belief(current_belief,all_goals_list, old_cart_lidar_data, new_cart_lidar_data)
    #@show("INSIDE",current_belief, old_cart_lidar_data, new_cart_lidar_data,"*****")
    new_belief = Array{human_probability_over_goals,1}()
    for human_index_in_new_lidar_data in 1:length(new_cart_lidar_data)
        human_index_in_old_lidar_data = is_human_present_in_the_list(new_cart_lidar_data[human_index_in_new_lidar_data],
            old_cart_lidar_data)
        if(human_index_in_old_lidar_data == -1)
            push!(new_belief,human_probability_over_goals([0.25,0.25,0.25,0.25]))
        else
            old_human_dist_from_all_goals_list = calculate_human_dist_from_all_goals(
                old_cart_lidar_data[human_index_in_old_lidar_data],all_goals_list)
            new_human_dist_from_all_goals_list = calculate_human_dist_from_all_goals(
                new_cart_lidar_data[human_index_in_new_lidar_data],all_goals_list)
            human_prob_over_goals_list = old_human_dist_from_all_goals_list .- new_human_dist_from_all_goals_list
            minimum_unnormalized_value = abs(minimum(human_prob_over_goals_list))
            for i in 1:length(human_prob_over_goals_list)
                human_prob_over_goals_list[i] += (minimum_unnormalized_value + 1)
            end
            #human_prob_over_goals_list = broadcast(x-> x+1+abs(minimum(human_prob_over_goals_list)),human_prob_over_goals_list)
            updated_belief_for_current_human = (current_belief[human_index_in_old_lidar_data].distribution).*human_prob_over_goals_list
            updated_belief_for_current_human = updated_belief_for_current_human/sum(updated_belief_for_current_human)
            #@show(updated_belief_for_current_human)
            push!(new_belief,human_probability_over_goals(updated_belief_for_current_human))
        end
    end
    return new_belief
end

# temp_old_lidar_data = [env.humans[1], env.humans[3], env.humans[5]]
# temp_current_belief = [human_probability_over_goals([0.25, 0.25, 0.25, 0.25]),
# human_probability_over_goals([0.25, 0.25, 0.25, 0.25]),
# human_probability_over_goals([0.25, 0.25, 0.25, 0.25])]
# temp_new_lidar_data = [get_new_human_position_actual_environemnt(env.humans[1],env,1),
#     get_new_human_position_actual_environemnt(env.humans[2],env,1),
#     get_new_human_position_actual_environemnt(env.humans[3],env,1),
#     get_new_human_position_actual_environemnt(env.humans[4],env,1),
#     get_new_human_position_actual_environemnt(env.humans[5],env,1)]

# @code_warntype update_belief(temp_current_belief, env.goals,temp_old_lidar_data,temp_new_lidar_data)
#lala = update_belief(temp_current_belief, env.goals,temp_old_lidar_data,temp_new_lidar_data)
#lala = @code_warntype update_belief([], env.goals,[],temp_new_lidar_data)
