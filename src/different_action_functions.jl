#function get_available_actions(m::POMDP_Planner_2D_action_space,pomdp_state)
function get_available_actions_non_holonomic(m::POMDP_Planner_2D_action_space,b)
    pomdp_state = first(particles(b))
    x_point =  floor(Int64,pomdp_state.cart.x/ 1.0)+1
    y_point =  floor(Int64,pomdp_state.cart.y/ 1.0)+1
    theta_point = clamp(floor(Int64,pomdp_state.cart.theta/(pi/18))+1,1,36)
    # nearest_prm_point -> Format (vertex_num, x_coordinate, y_coordinate, prm_dist_to_goal)
    nearest_prm_point = m.lookup_table[x_point,y_point,theta_point]
    required_orientation = get_heading_angle( nearest_prm_point[2][1], nearest_prm_point[2][2], pomdp_state.cart.x, pomdp_state.cart.y)
    delta_angle = required_orientation - pomdp_state.cart.theta
    abs_delta_angle = abs(delta_angle)
    if(abs_delta_angle<=pi)
        delta_angle = clamp(delta_angle, -pi/4, pi/4)
    else
        if(delta_angle>=0.0)
            delta_angle = clamp(delta_angle-2*pi, -pi/4, pi/4)
        else
            delta_angle = clamp(delta_angle+2*pi, -pi/4, pi/4)
        end
    end
    if(pomdp_state.cart.v == 0.0)
        if(delta_angle==pi/4 || delta_angle==-pi/4)
            return [(-pi/4,1.0),(-pi/6,1.0),(-pi/12,1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,1.0),(pi/4,1.0)]
        else
            return [(delta_angle, 1.0),(-pi/4,1.0),(-pi/6,1.0),(-pi/12,1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,1.0),(pi/4,1.0)]
        end
    # elseif (pomdp_state.cart.v == m.max_cart_speed)
    #     if(delta_angle==pi/4 || delta_angle==-pi/4)
    #         # return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
    #         #Without immediate stop action
    #         return [(-pi/4,-1.0),(-pi/6,-1.0),(-pi/12,-1.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(pi/12,-1.0),(pi/6,-1.0),(pi/4,-1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0)]
    #     else
    #         # return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
    #         #Without immediate stop action
    #         return [(delta_angle, 0.0), (delta_angle, -1.0),(-pi/4,-1.0),(-pi/6,-1.0),(-pi/12,-1.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(pi/12,-1.0),(pi/6,-1.0),(pi/4,-1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0)]
    #         # return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0)]
    #     end
    else
        if(delta_angle==pi/4 || delta_angle==-pi/4)
            # return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0)]
        else
            # return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0)]
        end
    end
end
#@code_warntype get_available_actions(POMDP_state_2D_action_space(env.cart,env.humans))

function get_available_actions_holonomic(m::POMDP_Planner_2D_action_space,b)
    pomdp_state = first(particles(b))
    x_point =  floor(Int64,pomdp_state.cart.x/ 1.0)+1
    y_point =  floor(Int64,pomdp_state.cart.y/ 1.0)+1
    # nearest_prm_point -> Format (vertex_num, x_coordinate, y_coordinate, prm_dist_to_goal)
    nearest_prm_point = m.lookup_table[x_point,y_point]
    required_orientation = get_heading_angle( nearest_prm_point[2], nearest_prm_point[3], pomdp_state.cart.x, pomdp_state.cart.y)
    delta_angle = required_orientation - pomdp_state.cart.theta
    if(pomdp_state.cart.v == 0.0)
        return [(delta_angle, 1.0),(-pi/4,1.0),(-pi/6,1.0),(-pi/12,1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,1.0),(pi/4,1.0)]
    else
        if(delta_angle==pi/4 || delta_angle==-pi/4)
            # return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0)]
        else
            # return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0)]
        end
    end
end

function get_available_actions_with_prm_vertex_non_holonomic(m::POMDP_Planner_2D_action_space,b)::Array{ Union{ Tuple{Float64,Float64}, Tuple{ Tuple{Float64,Float64}, Tuple{Float64,Float64}, Float64 } } ,1}
    pomdp_state = first(particles(b))
    x_point =  floor(Int64,pomdp_state.cart.x/ 1.0)+1
    y_point =  floor(Int64,pomdp_state.cart.y/ 1.0)+1
    theta_point = clamp(floor(Int64,pomdp_state.cart.theta/(pi/18))+1,1,36)
    # nearest_prm_point -> Format (vertex_num, x_coordinate, y_coordinate, prm_dist_to_goal)
    nearest_prm_point = m.lookup_table[x_point,y_point,theta_point]
    a = Array{ Union{ Tuple{Float64,Float64}, Tuple{ Tuple{Float64,Float64}, Tuple{Float64,Float64}, Float64 } } ,1}()
    if(pomdp_state.cart.v == 0.0)
        if(nearest_prm_point[1] == -1)
            a = [(-pi/4,1.0),(-pi/6,1.0),(-pi/12,1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,1.0),(pi/4,1.0)]
        else
            a =  [(nearest_prm_point[2],nearest_prm_point[4],1.0),(-pi/4,1.0),(-pi/6,1.0),(-pi/12,1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,1.0),(pi/4,1.0)]
        end
    # elseif (pomdp_state.cart.v == m.max_cart_speed)
    else
        if(nearest_prm_point[1] == -1)
            # return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            a =  [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0)]
        else
            # return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            a = [(nearest_prm_point[2],nearest_prm_point[4],0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,0.0),(pi/6,0.0),(pi/4,0.0)]
        end
    end
    return a
end
#@code_warntype get_available_actions(POMDP_state_2D_action_space(env.cart,env.humans))


function get_available_custom_actions_non_holonomic(m::POMDP_Planner_2D_action_space,b)
    pomdp_state = first(particles(b))
    x_point =  floor(Int64,pomdp_state.cart.x/ 1.0)+1
    y_point =  floor(Int64,pomdp_state.cart.y/ 1.0)+1
    theta_point = clamp(floor(Int64,pomdp_state.cart.theta/(pi/18))+1,1,36)
    nearest_prm_point = m.lookup_table[x_point,y_point,theta_point]
    # a = Array{ Union{ Tuple{Float64,Float64}, Tuple{ Tuple{Float64,Float64}, Tuple{Float64,Float64}, Float64 } } ,1}()
    if(pomdp_state.cart.v == 0.0)
        if(nearest_prm_point[1] == -1)
            a = [ POMDP_2D_action_type(-pi/4,1.0,(0.0,0.0),(0.0,0.0)) , POMDP_2D_action_type(-pi/6,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(-pi/12,1.0,(0.0,0.0),(0.0,0.0)), POMDP_2D_action_type(0.0,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(0.0,1.0,(0.0,0.0),(0.0,0.0)), POMDP_2D_action_type(pi/12,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(pi/6,1.0,(0.0,0.0),(0.0,0.0)), POMDP_2D_action_type(pi/4,1.0,(0.0,0.0),(0.0,0.0)) ]
        else
            a = [ POMDP_2D_action_type(-pi/4,1.0,(0.0,0.0),(0.0,0.0)) , POMDP_2D_action_type(-pi/6,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(-pi/12,1.0,(0.0,0.0),(0.0,0.0)), POMDP_2D_action_type(0.0,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(0.0,1.0,(0.0,0.0),(0.0,0.0)), POMDP_2D_action_type(pi/12,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(pi/6,1.0,(0.0,0.0),(0.0,0.0)), POMDP_2D_action_type(pi/4,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(-10.0,1.0,nearest_prm_point[2],nearest_prm_point[4]) ]
        end
    # elseif (pomdp_state.cart.v == m.max_cart_speed)
    else
        if(nearest_prm_point[1] == -1)
            # return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            a = [ POMDP_2D_action_type(-pi/4,0.0,(0.0,0.0),(0.0,0.0)) , POMDP_2D_action_type(-pi/6,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(-pi/12,0.0,(0.0,0.0),(0.0,0.0)), POMDP_2D_action_type(0.0,-1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(0.0,0.0,(0.0,0.0),(0.0,0.0)),POMDP_2D_action_type(0.0,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(pi/12,0.0,(0.0,0.0),(0.0,0.0)),POMDP_2D_action_type(pi/6,0.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(pi/4,0.0,(0.0,0.0),(0.0,0.0)) ]
        else
            # return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            a = [ POMDP_2D_action_type(-pi/4,0.0,(0.0,0.0),(0.0,0.0)) , POMDP_2D_action_type(-pi/6,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(-pi/12,0.0,(0.0,0.0),(0.0,0.0)), POMDP_2D_action_type(0.0,-1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(0.0,0.0,(0.0,0.0),(0.0,0.0)),POMDP_2D_action_type(0.0,1.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(pi/12,0.0,(0.0,0.0),(0.0,0.0)),POMDP_2D_action_type(pi/6,0.0,(0.0,0.0),(0.0,0.0)),
                POMDP_2D_action_type(pi/4,0.0,(0.0,0.0),(0.0,0.0)), POMDP_2D_action_type(-10.0,0.0,nearest_prm_point[2],nearest_prm_point[4]) ]
        end
    end
    return a
end

function get_available_custom_actions_non_holonomic(m::POMDP_Planner_2D_action_space,b)
    pomdp_state = first(particles(b))
    x_point =  floor(Int64,pomdp_state.cart.x/ 1.0)+1
    y_point =  floor(Int64,pomdp_state.cart.y/ 1.0)+1
    theta_point = clamp(floor(Int64,pomdp_state.cart.theta/(pi/18))+1,1,36)
    nearest_prm_point = m.lookup_table[x_point,y_point,theta_point]
    # a = Array{ Union{ Tuple{Float64,Float64}, Tuple{ Tuple{Float64,Float64}, Tuple{Float64,Float64}, Float64 } } ,1}()
    if(pomdp_state.cart.v == 0.0)
        if(nearest_prm_point.closest_prm_vertex_num == -1)
            a = [ POMDP_2D_action_type(-pi/4,1.0,0,0.0,0.0,0,0.0,0.0) , POMDP_2D_action_type(-pi/6,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(-pi/12,1.0,0,0.0,0.0,0,0.0,0.0), POMDP_2D_action_type(0.0,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(0.0,1.0,0,0.0,0.0,0,0.0,0.0), POMDP_2D_action_type(pi/12,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(pi/6,1.0,0,0.0,0.0,0,0.0,0.0), POMDP_2D_action_type(pi/4,1.0,0,0.0,0.0,0,0.0,0.0) ]
        else
            a = [ POMDP_2D_action_type(-pi/4,1.0,0,0.0,0.0,0,0.0,0.0) , POMDP_2D_action_type(-pi/6,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(-pi/12,1.0,0,0.0,0.0,0,0.0,0.0), POMDP_2D_action_type(0.0,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(0.0,1.0,0,0.0,0.0,0,0.0,0.0), POMDP_2D_action_type(pi/12,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(pi/6,1.0,0,0.0,0.0,0,0.0,0.0), POMDP_2D_action_type(pi/4,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(-10.0,1.0,nearest_prm_point.closest_prm_vertex_num,nearest_prm_point.closest_prm_vertex_x,
                                            nearest_prm_point.closest_prm_vertex_y,nearest_prm_point.next_prm_vertex_num,
                                            nearest_prm_point.next_prm_vertex_x, nearest_prm_point.next_prm_vertex_y) ]
        end
    # elseif (pomdp_state.cart.v == m.max_cart_speed)
    else
        if(nearest_prm_point.closest_prm_vertex_num == -1)
            # return [(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            a = [ POMDP_2D_action_type(-pi/4,0.0,0,0.0,0.0,0,0.0,0.0) , POMDP_2D_action_type(-pi/6,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(-pi/12,0.0,0,0.0,0.0,0,0.0,0.0), POMDP_2D_action_type(0.0,-1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(0.0,0.0,0,0.0,0.0,0,0.0,0.0),POMDP_2D_action_type(0.0,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(pi/12,0.0,0,0.0,0.0,0,0.0,0.0),POMDP_2D_action_type(pi/6,0.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(pi/4,0.0,0,0.0,0.0,0,0.0,0.0) ]
        else
            # return [(delta_angle, 0.0),(-pi/4,0.0),(-pi/6,0.0),(-pi/12,0.0),(0.0,-1.0),(0.0,0.0),(0.0,1.0),(pi/12,1.0),(pi/6,0.0),(pi/4,0.0),(-10.0,-10.0)]
            #Without immediate stop action
            a = [ POMDP_2D_action_type(-pi/4,0.0,0,0.0,0.0,0,0.0,0.0) , POMDP_2D_action_type(-pi/6,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(-pi/12,0.0,0,0.0,0.0,0,0.0,0.0), POMDP_2D_action_type(0.0,-1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(0.0,0.0,0,0.0,0.0,0,0.0,0.0),POMDP_2D_action_type(0.0,1.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(pi/12,0.0,0,0.0,0.0,0,0.0,0.0),POMDP_2D_action_type(pi/6,0.0,0,0.0,0.0,0,0.0,0.0),
                POMDP_2D_action_type(pi/4,0.0,0,0.0,0.0,0,0.0,0.0), POMDP_2D_action_type(-10.0,0.0,nearest_prm_point.closest_prm_vertex_num,nearest_prm_point.closest_prm_vertex_x,
                                                                                    nearest_prm_point.closest_prm_vertex_y,nearest_prm_point.next_prm_vertex_num,
                                                                                    nearest_prm_point.next_prm_vertex_x, nearest_prm_point.next_prm_vertex_y) ]
        end
    end
    return a
end
