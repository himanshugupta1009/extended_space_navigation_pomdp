#Various different miscellaneous functions that are needed by different components and are common to multiple files

function is_within_range_check_with_points(p1_x,p1_y, p2_x, p2_y, threshold_distance)
    euclidean_distance = ((p1_x - p2_x)^2 + (p1_y - p2_y)^2)^0.5
    if(euclidean_distance<=threshold_distance)
        return true
    else
        return false
    end
end

function is_within_range(location1, location2, threshold_distance)
    euclidean_distance = ((location1.x - location2.x)^2 + (location1.y - location2.y)^2)^0.5
    if(euclidean_distance<=threshold_distance)
        return true
    else
        return false
    end
end
#@code_warntype is_within_range(location(0,0,-1), location(3,4,-1), 1)

function wrap_between_0_and_2Pi(theta)
   return mod(theta,2*pi)
end

function travel!(du,u,p,t)
    x,y,theta = u
    v,L,alpha = p

    du[1] = v*cos(theta)
    du[2] = v*sin(theta)
    du[3] = (v/L)*tan(alpha)
end

function get_intermediate_points(initial_state, time_interval, extra_parameters)
    prob = ODEProblem(travel!,initial_state,time_interval,extra_parameters)
    sol = DifferentialEquations.solve(prob,saveat=0.1)
    x = []
    y = []
    theta = []

    for i in 1:length(sol.u)
        push!(x,sol.u[i][1])
        push!(y,sol.u[i][2])
        push!(theta,wrap_between_0_and_2Pi(sol.u[i][3]))
    end

    return x,y,theta
end

#Given three points, find the center and the radius of the circle they lie on
function find_center_and_radius(x1,y1,x2,y2,x3,y3)
    x12 = x1-x2
    x13 = x1-x3

    y12 = y1-y2
    y13 = y1-y3

    y31 = y3-y1
    y21 = y2-y1

    x31 = x3-x1
    x21 = x2-x1

    sx13 = (x1*x1) - (x3*x3)
    sy13 = (y1*y1) - (y3*y3)
    sx21 = (x2*x2) - (x1*x1)
    sy21 = (y2*y2) - (y1*y1)

    f = ((sx13) * (x12) + (sy13) * (x12) + (sx21) * (x13) + (sy21) * (x13))/(2 * ((y31) * (x12) - (y21) * (x13)))
    g = ((sx13) * (y12)+ (sy13) * (y12)+ (sx21) * (y13)+ (sy21) * (y13))/(2 * ((x31) * (y12) - (x21) * (y13)))
    c = -(x1*x1) - (y1*y1) - (2*g*x1) - (2*f*y1)
    r = sqrt( (g*g) + (f*f) - c );

    return -g, -f, r
end

#Given a circle's center and radius and a line segment, find if they intersect
function find_if_circle_and_line_segment_intersect(cx::Float64,cy::Float64,cr::Float64,
                                    ex::Float64,ey::Float64,lx::Float64,ly::Float64)
    dx = lx-ex
    dy = ly-ey
    fx = ex-cx
    fy = ey-cy

    #Quadratic equation is  t^2 ( d · d ) + 2t ( d · f ) +  (f · f - r^2) = 0
    #(ex,ey) is the starting point of the ray and (lx,ly) is the end point
    #Refer to this link if needed - https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
    #Standard form is a.t^2 + b.t + c = 0

    a = (dx^2 + dy^2)
    b = 2*(dx*fx + dy*fy)
    c = (fx^2 + fy^2) - (cr^2)
    discriminant = (b^2 - 4*a*c)

    if(discriminant<0)
        return false
    elseif (discriminant == 0)
        t = -b/(2*a)
        if(t>=0 && t<=1)
            return true
        end
    else
        discriminant = sqrt(discriminant)
        t = (-b-discriminant)/(2*a)
        if(t>=0 && t<=1)
            return true
        end
        t = (-b+discriminant)/(2*a)
        if(t>=0 && t<=1)
            return true
        end
    end
    return false
end

# Check if line segment joining (x1,y1) to (x2,y2) and line segment joining (x3,y3)) to (x4,y4) intersect or not
function find_if_two_line_segments_intersect(x1::Float64,y1::Float64,x2::Float64,y2::Float64,
                                        x3::Float64,y3::Float64,x4::Float64,y4::Float64)

    #Refer to this link for the logic
    #http://paulbourke.net/geometry/pointlineplane/

    epsilon = 10^-6
    same_denominator = ( (y4-y3)*(x2-x1) ) - ( (x4-x3)*(y2-y1) )
    numerator_ua = ( (x4-x3)*(y1-y3) ) - ( (y4-y3)*(x1-x3) )
    numerator_ub = ( (x2-x1)*(y1-y3) ) - ( (y2-y1)*(x1-x3) )

    if(abs(same_denominator) < epsilon && abs(numerator_ua) < epsilon
                                        && abs(numerator_ub) < epsilon)
         return true;
    elseif (abs(same_denominator) < epsilon)
        return false
    else
        ua = numerator_ua/same_denominator
        ub = numerator_ub/same_denominator
        if(ua>=0.0 && ua<=1.0 && ub>=0.0 && ub<=1.0)
            return true;
        else
            return false;
        end
    end
end

function find_if_two_circles_intersect(c1x::Float64,c1y::Float64,c1r::Float64,c2x::Float64,c2y::Float64,c2r::Float64)
    dist_c1_c2 = (c1x - c2x)^2 + (c1y - c2y)^2
    if(dist_c1_c2 > (c1r + c2r)^2 )
        return false
    else
        return true
    end
end

function get_heading_angle(human_x, human_y, cart_x, cart_y)

    #First Quadrant
    if(human_x >= cart_x && human_y >= cart_y)
        if(human_x == cart_x)
            heading_angle = pi/2.0
        elseif(human_y == cart_y)
            heading_angle = 0.0
        else
            heading_angle = atan((human_y - cart_y) / (human_x - cart_x))
        end
    #Second Quadrant
    elseif(human_x <= cart_x && human_y >= cart_y)
        if(human_x == cart_x)
            heading_angle = pi/2.0
        elseif(human_y == cart_y)
            heading_angle = pi/1.0
        else
            heading_angle = atan((human_y - cart_y) / (human_x - cart_x)) + pi
        end
    #Third Quadrant
    elseif(human_x <= cart_x && human_y <= cart_y)
        if(human_x == cart_x)
            heading_angle = 3*pi/2.0
        elseif(human_y == cart_y)
            heading_angle = pi/1.0
        else
            heading_angle = atan((human_y - cart_y) / (human_x - cart_x)) + pi
        end
    #Fourth Quadrant
    else(human_x >= cart_x && human_y <= cart_y)
        if(human_x == cart_x)
            heading_angle = 3*pi/2.0
        elseif(human_y == cart_y)
            heading_angle = 0.0
        else
            heading_angle = 2.0*pi + atan((human_y - cart_y) / (human_x - cart_x))
        end
    end

    return heading_angle
end

function get_lidar_data(world,lidar_range)
    initial_cart_lidar_data = Array{human_state,1}()
    for human in world.humans
        if(is_within_range(location(world.cart.x,world.cart.y), location(human.x,human.y), lidar_range))
            if(human.x!= human.goal.x || human.y!= human.goal.y)
                push!(initial_cart_lidar_data,human)
            end
        end
    end
    return initial_cart_lidar_data
end

#Function for actually moving human in the environment
function get_new_human_position_actual_environemnt(human, world, time_step, user_defined_rng)

    rand_num = (rand(user_defined_rng) - 0.5)*0.1
    #rand_num = 0.0
    #First Quadrant
    if(human.goal.x >= human.x && human.goal.y >= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y + (human.v)*time_step + rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x + (human.v)*time_step + rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x + ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y + ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    #Second Quadrant
    elseif(human.goal.x <= human.x && human.goal.y >= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y + (human.v)*time_step + rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x - (human.v)*time_step - rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x - ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y - ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    #Third Quadrant
    elseif(human.goal.x <= human.x && human.goal.y <= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y - (human.v)*time_step - rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x - (human.v)*time_step - rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x - ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y - ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    #Fourth Quadrant
    else(human.goal.x >= human.x && human.goal.y <= human.y)
        if(human.goal.x == human.x)
            new_x = human.x
            new_y = human.y - (human.v)*time_step - rand_num
        elseif(human.goal.y == human.y)
            new_x = human.x + (human.v)*time_step + rand_num
            new_y = human.y
        else
            heading_angle = atan((human.goal.y - human.y) / (human.goal.x - human.x))
            new_x = human.x + ((human.v)*time_step + rand_num)*cos(heading_angle)
            new_y = human.y + ((human.v)*time_step + rand_num)*sin(heading_angle)
        end
    end

    new_x = clamp(new_x,0,world.length)
    new_y = clamp(new_y,0,world.breadth)
    #@show(new_x,new_y)
    new_human_state = human_state(new_x, new_y, human.v, human.goal,human.id)
    return new_human_state
end

function move_human_for_one_time_step_in_actual_environment(world,time_step,user_defined_rng)
    moved_human_positions = Array{human_state,1}()
    for human in world.humans
        if(human.x == human.goal.x && human.y==human.goal.y)
            new_human_state = human_state(human.x,human.y,human.v,human.goal,human.id)
        elseif (is_within_range(location(human.x,human.y), human.goal, 1.0))
            new_human_state = human_state(human.goal.x,human.goal.y,human.v,human.goal,human.id)
        else
            new_human_state = get_new_human_position_actual_environemnt(human,world,time_step,user_defined_rng)
        end
        push!(moved_human_positions,new_human_state)
    end
    return moved_human_positions
end

function get_count_number_of_risks(world)
    risks = 0
    if(world.cart.v>=1.0)
        for human in world.complete_cart_lidar_data
            euclidean_distance = sqrt( (human.x - world.cart.x)^2 + (human.y - world.cart.y)^2 )
            if(euclidean_distance<=0.5)
                println( "A risky scenario encountered and the distance is : ", euclidean_distance )
                risks += 1
            end
        end
    else
        return 0;
    end
    return risks;
end

function write_and_print(io::IOStream, string_to_be_written_and_printed::String)
    write(io, string_to_be_written_and_printed * "\n")
    println(string_to_be_written_and_printed)
end

function get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(cart, cart_lidar_data, n, cone_half_angle::Float64=pi/3.0)
    nearest_n_pedestrians = Array{human_state,1}()
    priority_queue_nearest_n_pedestrians = PriorityQueue{human_state,Float64}(Base.Order.Forward)
    for i in 1:length(cart_lidar_data)
        human = cart_lidar_data[i]
        angle_between_cart_and_human = get_heading_angle(human.x, human.y, cart.x, cart.y)
        difference_in_angles = abs(cart.theta - angle_between_cart_and_human)
        if(difference_in_angles <= cone_half_angle)
            euclidean_distance = sqrt( (cart.x - human.x)^2 + (cart.y - human.y)^2 )
            priority_queue_nearest_n_pedestrians[human] = euclidean_distance
        elseif ( (2*pi - difference_in_angles) <= cone_half_angle )
            euclidean_distance = sqrt( (cart.x - human.x)^2 + (cart.y - human.y)^2 )
            priority_queue_nearest_n_pedestrians[human] = euclidean_distance
        else
            continue
        end
    end

    priority_queue_nearest_n_pedestrians_ordered_by_id = PriorityQueue{human_state,Float64}(Base.Order.Forward)
    for i in 1:n
        if(length(priority_queue_nearest_n_pedestrians) != 0)
            pedestrian = dequeue!(priority_queue_nearest_n_pedestrians)
            priority_queue_nearest_n_pedestrians_ordered_by_id[pedestrian] = pedestrian.id
        else
            break
        end
    end

    for pedestrian in priority_queue_nearest_n_pedestrians_ordered_by_id
        push!(nearest_n_pedestrians,dequeue!(priority_queue_nearest_n_pedestrians_ordered_by_id))
    end

    return nearest_n_pedestrians
end

function get_nearest_n_pedestrians_hybrid_astar_search(world,current_belief,n,cone_half_angle::Float64=pi/3.0)
    nearest_n_pedestrians = Array{Tuple{human_state,human_probability_over_goals},1}()
    priority_queue_nearest_n_pedestrians = PriorityQueue{Tuple{human_state,human_probability_over_goals},Float64}(Base.Order.Forward)
    cone_half_angle = pi/3.0
    for i in 1:length(world.cart_lidar_data)
        human = world.cart_lidar_data[i]
        angle_between_cart_and_human = get_heading_angle(human.x, human.y, world.cart.x, world.cart.y)
        difference_in_angles = abs(world.cart.theta - angle_between_cart_and_human)
        if(difference_in_angles <= cone_half_angle)
            euclidean_distance = sqrt( (world.cart.x - human.x)^2 + (world.cart.y - human.y)^2 )
            priority_queue_nearest_n_pedestrians[(human,current_belief[i])] = euclidean_distance
        elseif ( (2*pi - difference_in_angles) <= cone_half_angle )
            euclidean_distance = sqrt( (world.cart.x - human.x)^2 + (world.cart.y - human.y)^2 )
            priority_queue_nearest_n_pedestrians[(human,current_belief[i])] = euclidean_distance
        else
            continue
        end
    end
    for i in 1:n
        if(length(priority_queue_nearest_n_pedestrians) != 0)
            push!(nearest_n_pedestrians,dequeue!(priority_queue_nearest_n_pedestrians))
        else
            break
        end
    end
    return nearest_n_pedestrians
end

# Function that updates the belief based on cart_lidar_data of old world and
# cart_lidar_data of the new world
function update_belief_from_old_world_and_new_world(current_belief, old_world, new_world)
    updated_belief = update_belief(current_belief, old_world.goals,
        old_world.complete_cart_lidar_data, new_world.complete_cart_lidar_data)
    return updated_belief
end

function update_current_belief_by_creating_temp_world(old_world, new_world, old_belief, lidar_range,
                                                        num_humans_to_care_about, cone_half_angle)

    #Create the temp world at t = 0.5 second
    temp_world = deepcopy(old_world)
    for human_index in 1:length(old_world.humans)
        temp_world.humans[human_index].x = 0.5*(old_world.humans[human_index].x + new_world.humans[human_index].x)
        temp_world.humans[human_index].y = 0.5*(old_world.humans[human_index].y + new_world.humans[human_index].y)
    end

    complete_new_lidar_data = get_lidar_data(temp_world,lidar_range)
    new_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(temp_world.cart, complete_new_lidar_data,
                                                                                num_humans_to_care_about, cone_half_angle )
    #Update belief
    temp_world.complete_cart_lidar_data = complete_new_lidar_data
    temp_world.cart_lidar_data = new_lidar_data
    updated_belief = update_belief(old_belief, temp_world.goals,
        old_world.complete_cart_lidar_data, temp_world.complete_cart_lidar_data)

    final_updated_belief = update_belief(updated_belief, new_world.goals,
        temp_world.complete_cart_lidar_data, new_world.complete_cart_lidar_data)

    return final_updated_belief
end

function get_belief_for_selected_humans_from_belief_over_complete_lidar_data(belief_over_complete_lidar_data, complete_lidar_data, shortlisted_lidar_data)
    shortlisted_belief = Array{human_probability_over_goals,1}();
    for human in shortlisted_lidar_data
        for index in 1:length(complete_lidar_data)
            if(human.id == complete_lidar_data[index].id)
                push!(shortlisted_belief, belief_over_complete_lidar_data[index])
                break;
            end
        end
    end
    return shortlisted_belief
end

function respawn_humans_in_environment(world, lidar_range, num_humans_to_care_about_while_pomdp_planning,
                                                        cone_half_angle, current_belief, user_defined_rng_for_env)

    old_world = deepcopy(world)
    num_humans_already_at_goal = 0
    for human in world.humans
        if( (human.x == human.goal.x) && (human.y == human.goal.y) )
            num_humans_already_at_goal += 1
        end
    end

    total_number_of_humans = length(world.humans)
    total_humans_not_at_goal = total_number_of_humans - num_humans_already_at_goal

    if( total_humans_not_at_goal < world.num_humans)
        for i in 1:total_humans_not_at_goal
            rand_num = rand(user_defined_rng_for_env)
            if(rand_num > 0.5)
                x = world.length
                y = floor(100*rand_num)
                while(is_within_range_check(x,y,world.cart.goal.x,world.cart.goal.y,5.0))
                    y = floor(100*rand(user_defined_rng))
                end
                new_human_state = human_state(x,y,1.0,world.goals[Int(ceil(rand(user_defined_rng)*4))], float(length(world.humans) + 1))
            else
                x = 0.0
                y = floor(100*rand_num)
                while(is_within_range_check(x,y,world.cart.goal.x,world.cart.goal.y,5.0))
                    y = floor(100*rand(user_defined_rng))
                end
                new_human_state = human_state(x,y,1.0,world.goals[Int(ceil(rand(user_defined_rng)*4))], float(length(world.humans) + 1))
            end
            push!(world.humans,new_human_state)
        end
        world.cart_lidar_data = get_lidar_data(world,lidar_range)
        world.cart_lidar_data = get_nearest_n_pedestrians_in_cone_pomdp_planning_1D_or_2D_action_space(world.cart,
                                                            world.cart_lidar_data, num_humans_to_care_about_while_pomdp_planning,
                                                            cone_half_angle)

        updated_belief = update_belief_from_old_world_and_new_world(current_belief, old_world, env_right_now)

    else
        updated_belief = current_belief
    end

    return current_belief
end
