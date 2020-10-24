# Base.isequal(a::golfcart_observations, b::golfcart_observations) = isequal(a.observed_human_positions, b.observed_human_positions)
# Base.hash(x::golfcart_observations, h::UInt) = hash(x.observed_human_positions, h)

Base.isequal(a::POMDP_state_2D_action_space, b::POMDP_state_2D_action_space) = isequal( (a.cart.x,a.cart.y,a.cart.theta,a.cart.v), (b.cart.x,b.cart.y,b.cart.theta,b.cart.v) )
Base.hash(a::POMDP_state_2D_action_space, h::UInt) = hash((a.cart.x,a.cart.y,a.cart.theta,a.cart.v), h)

function check_if_all_cart_states_are_same_in_same_scenario_belief(tree)
    for scenario in tree.scenarios
        pomdp_state = scenario[1].second
        for i in 2:length(scenario)
            if( scenario[i].second.cart.x == -100.0 || pomdp_state.cart.x == -100.0)
                continue
            end
            if ( !isequal(pomdp_state, scenario[i].second ) )
                @show("I suck!")
                println(pomdp_state)
                println(scenario[i].second)
            end
        end
    end
end

# scenarios_to_explore = [()]
# scenarios_with_names = [ ( tree.scenarios[1].cart )]
# initial_children_indices = tree.children[1]
#
# for ba_child_index in initial_children_indices
#     tree.ba_children[ba_child_index]
#     scenarios_to_explore

function see_states_visited_in_DESPOT(all_trees, which_index, pomdp)

    tree = all_trees[which_index][:tree]

    scenarios_to_explore = Queue{Any}()
    enqueue!(scenarios_to_explore,1)
    explored_cart_states = Dict()

    while( !isempty(scenarios_to_explore) )

        index = dequeue!(scenarios_to_explore)
        current_scenario = tree.scenarios[index]

        cart_state_in_current_scenario = current_scenario[1].second.cart
        for i in 1:length(current_scenario)
            if(current_scenario[i].second.cart.x == -100.0 && current_scenario[i].second.cart.y == -100.0)
                continue
            else
                cart_state_in_current_scenario = current_scenario[i].second.cart
                break
            end
        end

        if(index == 1)
            explored_cart_states[ current_scenario.second ] =  (nothing, 0 , current_scenario[1].second.cart )
        else
            if( ! (current_scenario[1].second in keys(explored_cart_states) )
                parent_index = tree.parent[index]
                explored_cart_states[ current_scenario.second ] = ( tree.scenarios[parent_index].  )



    for scenario in tree.scenarios
        pomdp_state = scenario[1].second
        if(pomdp_state in keys(explored_cart_states))
            explored_cart_states[pomdp_state] =
