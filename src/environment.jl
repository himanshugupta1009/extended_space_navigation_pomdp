using Plots
using Random
using MetaGraphs
using LightGraphs
using Revise

#Global Variables
plot_size = 1000; #number of pixels
cart_size = 1; # radius in meters

#Various different Struct definitions
struct location
    x::Float64
    y::Float64
end

mutable struct human_state
    x::Float64
    y::Float64
    v::Float64
    goal::location
    id::Float64
end

struct obstacle_location
    x::Float64
    y::Float64
    r::Float64 #Radius of the obstacle which is assumed to be a circle
end

mutable struct cart_state
    x::Float64
    y::Float64
    theta::Float64
    v::Float64
    L::Float64
    goal::location
end

struct human_probability_over_goals
    distribution::Array{Float64,1}
end

mutable struct experiment_environment
    length::Float64
    breadth::Float64
    max_num_humans::Float64
    num_humans::Int64
    goals::Array{location,1}
    humans::Array{human_state,1}
    obstacles::Array{obstacle_location,1}
    cart::cart_state
    cart_lidar_data::Array{human_state,1}
    complete_cart_lidar_data::Array{human_state,1}
    cart_hybrid_astar_path::Array{Float64,1}
    cart_start_location::location
end

#Define the Environment
function generate_environment_no_obstacles(number_of_humans, user_defined_rng)

    world_length = 100.0
    world_breadth = 100.0
    g1 = location(0.0,0.0)
    g2 = location(0.0,world_breadth)
    g3 = location(world_length,world_breadth)
    g4 = location(world_length,0.0)
    cart_goal = location(world_length,75.0)
    all_goals_list = [g1,g2,g3,g4]
    all_obstacle_list = obstacle_location[]
    max_num_humans = number_of_humans

    golfcart = cart_state(1.0,25.0,0.0,0.0,1.0,cart_goal)
    initial_cart_lidar_data = Array{human_state,1}()
    initial_complete_cart_lidar_data = Array{human_state,1}()

    human_state_start_list = Array{human_state,1}()
    for i in 1:max_num_humans
        human =  human_state(floor(world_length*rand(user_defined_rng)), floor(world_breadth*rand(user_defined_rng)) , 1.0
                                                , all_goals_list[Int(ceil(rand(user_defined_rng)*4))] , float(i))
        while(is_within_range_check_with_points(human.x,human.y, golfcart.x, golfcart.y, 5.0))
            human =  human_state(floor(world_length*rand(user_defined_rng)), floor(world_breadth*rand(user_defined_rng)) , 1.0
                                                    , all_goals_list[Int(ceil(rand(user_defined_rng)*4))] , float(i))
        end
        push!(human_state_start_list,human)
    end

    world = experiment_environment(world_length,world_breadth,max_num_humans,number_of_humans,
                    all_goals_list,human_state_start_list,all_obstacle_list,golfcart,initial_cart_lidar_data,
                    initial_complete_cart_lidar_data,Float64[],location(golfcart.x, golfcart.y))

    return world
end

function generate_environment_small_circular_obstacles(number_of_humans,user_defined_rng)

    world_length = 100.0
    world_breadth = 100.0
    g1 = location(0.0,0.0)
    g2 = location(0.0,world_breadth)
    g3 = location(world_length,world_breadth)
    g4 = location(world_length,0.0)
    cart_goal = location(world_length,75.0)
    all_goals_list = [g1,g2,g3,g4]

    o1 = obstacle_location(50.0,70.0,5.0)
    o2 = obstacle_location(25.0,70.0,5.0)
    o3 = obstacle_location(50.0,50.0,5.0)
    o4 = obstacle_location(30.0,20.0,5.0)
    o5 = obstacle_location(70.0,20.0,5.0)
    o6 = obstacle_location(80.0,50.0,5.0)
    #o2 = obstacle_location(50.0,50.0,10.0)
    # o3 = obstacle_location(50.0,30.0,15.0)
    #o2 = obstacle_location(33.0,69.0,8.0)
    #o2 = obstacle_location(25.0,50.0,25.0)
    # o3 = obstacle_location(73.0,79.0,3.0)
    # o4 = obstacle_location(65.0,40.0,7.0)
    # all_obstacle_list = [o1,o2,o3,o4]
    all_obstacle_list = [o1,o2,o3,o4,o5,o6]

    golfcart = cart_state(1.0,25.0,0.0,0.0,1.0,cart_goal)
    initial_cart_lidar_data = Array{human_state,1}()
    initial_complete_cart_lidar_data = Array{human_state,1}()

    max_num_humans = number_of_humans
    human_state_start_list = Array{human_state,1}()
    for i in 1:max_num_humans
        human =  human_state(floor(world_length*rand(user_defined_rng)), floor(world_breadth*rand(user_defined_rng)) , 1.0
                                                , all_goals_list[Int(ceil(rand(user_defined_rng)*4))] , float(i))
        while(is_within_range_check_with_points(human.x,human.y, golfcart.x, golfcart.y, 5.0))
            human =  human_state(floor(world_length*rand(user_defined_rng)), floor(world_breadth*rand(user_defined_rng)) , 1.0
                                                    , all_goals_list[Int(ceil(rand(user_defined_rng)*4))] , float(i))
        end
        push!(human_state_start_list,human)
    end

    world = experiment_environment(world_length,world_breadth,max_num_humans,number_of_humans,
                    all_goals_list,human_state_start_list,all_obstacle_list,golfcart,initial_cart_lidar_data,
                    initial_complete_cart_lidar_data,Float64[],location(golfcart.x, golfcart.y))

    return world
end

function generate_environment_large_circular_obstacles(number_of_humans,user_defined_rng)

    world_length = 100.0
    world_breadth = 100.0
    g1 = location(0.0,0.0)
    g2 = location(0.0,world_breadth)
    g3 = location(world_length,world_breadth)
    g4 = location(world_length,0.0)
    cart_goal = location(world_length,75.0)
    all_goals_list = [g1,g2,g3,g4]

    o1 = obstacle_location(50.0,75.0,15.0)
    o2 = obstacle_location(50.0,25.0,15.0)
    o3 = obstacle_location(30.0,50.0,10.0)
    o4 = obstacle_location(70.0,50.0,10.0)
    all_obstacle_list = [o1,o2,o3,o4]

    golfcart = cart_state(1.0,25.0,0.0,0.0,1.0,cart_goal)
    initial_cart_lidar_data = Array{human_state,1}()
    initial_complete_cart_lidar_data = Array{human_state,1}()

    max_num_humans = number_of_humans
    human_state_start_list = Array{human_state,1}()
    for i in 1:max_num_humans
        human =  human_state(floor(world_length*rand(user_defined_rng)), floor(world_breadth*rand(user_defined_rng)) , 1.0
                                                , all_goals_list[Int(ceil(rand(user_defined_rng)*4))] , float(i))
        while(is_within_range_check_with_points(human.x,human.y, golfcart.x, golfcart.y, 5.0))
            human =  human_state(floor(world_length*rand(user_defined_rng)), floor(world_breadth*rand(user_defined_rng)) , 1.0
                                                    , all_goals_list[Int(ceil(rand(user_defined_rng)*4))] , float(i))
        end
        push!(human_state_start_list,human)
    end

    world = experiment_environment(world_length,world_breadth,max_num_humans,number_of_humans,
                    all_goals_list,human_state_start_list,all_obstacle_list,golfcart,initial_cart_lidar_data,
                    initial_complete_cart_lidar_data,Float64[],location(golfcart.x, golfcart.y))

    return world
end

function generate_environment_L_shaped_corridor(number_of_humans,user_defined_rng)

    world_length = 100.0
    world_breadth = 100.0
    g1 = location(0.0,0.0)
    g2 = location(0.0,world_breadth)
    g3 = location(world_length,world_breadth)
    g4 = location(world_length,0.0)
    cart_goal = location(world_length,75.0)
    all_goals_list = [g1,g2,g3,g4]

    o1 = obstacle_location(65.0,35.0,35.0)
    all_obstacle_list = [o1]
    # o1 = obstacle_location(50.0,50.0,20.0)
    # o2 = obstacle_location(50.0,20.0,20.0)
    # o3 = obstacle_location(80.0,50.0,20.0)
    # o4 = obstacle_location(80.0,20.0,20.0)
    # all_obstacle_list = [o1,o2,o3,o4]

    golfcart = cart_state(1.0,25.0,0.0,0.0,1.0,cart_goal)
    initial_cart_lidar_data = Array{human_state,1}()
    initial_complete_cart_lidar_data = Array{human_state,1}()

    max_num_humans = number_of_humans
    human_state_start_list = Array{human_state,1}()
    for i in 1:max_num_humans
        human =  human_state(floor(world_length*rand(user_defined_rng)), floor(world_breadth*rand(user_defined_rng)) , 1.0
                                                , all_goals_list[Int(ceil(rand(user_defined_rng)*4))] , float(i))
        while(is_within_range_check_with_points(human.x,human.y, golfcart.x, golfcart.y, 5.0))
            human =  human_state(floor(world_length*rand(user_defined_rng)), floor(world_breadth*rand(user_defined_rng)) , 1.0
                                                    , all_goals_list[Int(ceil(rand(user_defined_rng)*4))] , float(i))
        end
        push!(human_state_start_list,human)
    end

    world = experiment_environment(world_length,world_breadth,max_num_humans,number_of_humans,
                    all_goals_list,human_state_start_list,all_obstacle_list,golfcart,initial_cart_lidar_data,
                    initial_complete_cart_lidar_data,Float64[],location(golfcart.x, golfcart.y))

    return world
end


#Function to display the environment
function display_env(env::experiment_environment, time_step=nothing, gif_env_num=nothing, graph = nothing)

    #Plot Boundaries
    p = plot([0.0],[0.0],legend=false,grid=false)
    plot!([env.length], [env.breadth],legend=false)

    #Plot Humans in the cart lidar data
    for i in 1: length(env.cart_lidar_data)
        scatter!([env.cart_lidar_data[i].x], [env.cart_lidar_data[i].y],color="green",msize=0.5*plot_size/env.length)
    end

    #Plot humans in complete_cart_lidar_data
    for i in 1: length(env.complete_cart_lidar_data)
        in_lidar_data_flag = false
        for green_human in env.cart_lidar_data
            if(env.complete_cart_lidar_data[i].id == green_human.id)
                in_lidar_data_flag = true
                break
            end
        end
        if(!in_lidar_data_flag)
            scatter!([env.complete_cart_lidar_data[i].x], [env.complete_cart_lidar_data[i].y],color="red",msize=0.5*plot_size/env.length)
        end
    end

    # #Plot Rest of the Humans
    # for i in 1: length(env.humans)
    #     in_lidar_data_flag = false
    #     for green_human in env.cart_lidar_data
    #         if(env.humans[i].id == green_human.id)
    #             in_lidar_data_flag = true
    #             break
    #         end
    #     end
    #     if(!in_lidar_data_flag)
    #         scatter!([env.humans[i].x], [env.humans[i].y],color="red",msize=0.5*plot_size/env.length)
    #     end
    # end

    #Plot Obstacles
    for i in 1: length(env.obstacles)
        scatter!([env.obstacles[i].x], [env.obstacles[i].y],color="black",shape=:circle,msize=plot_size*env.obstacles[i].r/env.length)
    end

    #Plot Golfcart
    scatter!([env.cart.x], [env.cart.y], shape=:circle, color="blue", msize= 0.3*plot_size*cart_size/env.length)
    quiver!([env.cart.x],[env.cart.y],quiver=([cos(env.cart.theta)],[sin(env.cart.theta)]), color="blue")

    #Plot the Hybrid A* path if it exists
    if(length(env.cart_hybrid_astar_path)!=0)
        #Plotting for normal environments that are 1 sec apart
        if(gif_env_num==nothing)
            initial_state = [env.cart.x,env.cart.y,env.cart.theta]
            path_x, path_y = [env.cart.x],[env.cart.y]
            for steering_angle in env.cart_hybrid_astar_path
                extra_parameters = [1.0, env.cart.L, steering_angle]
                x,y,theta = get_intermediate_points(initial_state, 1.0, extra_parameters);
                for pos_x in 2:length(x)
                    push!(path_x,x[pos_x])
                end
                for pos_y in 2:length(y)
                    push!(path_y,y[pos_y])
                end
                initial_state = [last(x),last(y),last(theta)]
            end
            plot!(path_x,path_y,color="black")
        #Plotting for gif environments that are 0.1 sec apart
        else
            current_gif_env_time_index = parse(Int, split(gif_env_num,"_")[2])
            current_time_stamp = 0.1*current_gif_env_time_index
            upper_cap_on_time = ceil(current_time_stamp/(1/env.cart.v))
            time_remaining_for_current_steering_angle = ((1/env.cart.v)*upper_cap_on_time) - current_time_stamp
            if(time_remaining_for_current_steering_angle!=0.0 && env.cart.v!=0.0)
                initial_state = [env.cart.x,env.cart.y,env.cart.theta]
                path_x, path_y = [env.cart.x],[env.cart.y]
                steering_angle = env.cart_hybrid_astar_path[1]
                extra_parameters = [env.cart.v, env.cart.L, steering_angle]
                x,y,theta = get_intermediate_points(initial_state,time_remaining_for_current_steering_angle, extra_parameters);
                for pos_x in 2:length(x)
                    push!(path_x,x[pos_x])
                end
                for pos_y in 2:length(y)
                    push!(path_y,y[pos_y])
                end
                initial_state = [last(path_x),last(path_y),last(theta)]
                start_index = 2
            else
                initial_state = [env.cart.x,env.cart.y,env.cart.theta]
                path_x, path_y = [env.cart.x],[env.cart.y]
                start_index = 1
            end
            for steering_angle in env.cart_hybrid_astar_path[start_index:end]
                extra_parameters = [1.0, env.cart.L, steering_angle]
                x,y,theta = get_intermediate_points(initial_state, 1.0, extra_parameters);
                for pos_x in 2:length(x)
                    push!(path_x,x[pos_x])
                end
                for pos_y in 2:length(y)
                    push!(path_y,y[pos_y])
                end
                initial_state = [last(x),last(y),last(theta)]
            end
            plot!(path_x,path_y,color="black")
        end
    end

    if(graph!=nothing)
        #Plot the PRM vertices
        for i in 1:nv(graph)
            if(i!=3)
                scatter!([get_prop(graph,i,:x)], [get_prop(graph,i,:y)],color="Grey",shape=:circle,msize=0.3*plot_size/env.length)
            end
        end

        #Format of vertex_tuple -> (current_vertex, parent_vertex)
        vertex_tuple = nothing
        if(vertex_tuple != nothing)
            for n in neighbors(graph, vertex_tuple[1])
                if( n!= vertex_tuple[2] && n!=3 )
                    plot!( [get_prop(graph,vertex_tuple[1],:x),get_prop(graph,n,:x)], [get_prop(graph,vertex_tuple[1],:y),
                                                                        get_prop(graph,n,:y)], color="LightGrey")
                end
            end
        else
            #Plot all the PRM edges
            all_edges = collect(edges(graph))
            for edge in all_edges
                if( get_prop(graph,edge.dst,:x) != -100.0 )
                    plot!( [get_prop(graph,edge.src,:x),get_prop(graph,edge.dst,:x) ], [get_prop(graph,edge.src,:y),get_prop(graph,edge.dst,:y)], color="LightGrey")
                end
                # scatter!([get_prop(env.prm,i,:x)], [get_prop(env.prm,i,:y)],color="LightGrey",shape=:circle,msize=0.3*plot_size/env.length)
            end
        end
    end

    annotate!(env.cart_start_location.x, env.cart_start_location.y, text("S", :purple, :right, 20))
    annotate!(env.cart.goal.x, env.cart.goal.y, text("G", :purple, :right, 20))
    if(time_step!=nothing)
        annotate!(env.length/2, env.breadth, text(time_step, :blue, :right, 20))
    end
    plot!(size=(plot_size,plot_size))
    display(p)
end

function display_prm_path_from_given_vertex(env::experiment_environment, prm, vertex_num)

    #Plot Boundaries
    p = plot([0.0],[0.0],legend=false,grid=false)
    plot!([env.length], [env.breadth],legend=false)

    #Plot Humans in the cart lidar data
    for i in 1: length(env.cart_lidar_data)
        scatter!([env.cart_lidar_data[i].x], [env.cart_lidar_data[i].y],color="green",msize=0.5*plot_size/env.length)
    end

    #Plot humans in complete cart_lidar_data
    for i in 1: length(env.complete_cart_lidar_data)
        in_lidar_data_flag = false
        for green_human in env.cart_lidar_data
            if(env.complete_cart_lidar_data[i].id == green_human.id)
                in_lidar_data_flag = true
                break
            end
        end
        if(!in_lidar_data_flag)
            scatter!([env.complete_cart_lidar_data[i].x], [env.complete_cart_lidar_data[i].y],color="red",msize=0.5*plot_size/env.length)
        end
    end

    #Plot Obstacles
    for i in 1: length(env.obstacles)
        scatter!([env.obstacles[i].x], [env.obstacles[i].y],color="black",shape=:circle,msize=plot_size*env.obstacles[i].r/env.length)
    end

    #Plot Golfcart
    scatter!([env.cart.x], [env.cart.y], shape=:circle, color="blue", msize= 0.3*plot_size*cart_size/env.length)
    quiver!([env.cart.x],[env.cart.y],quiver=([cos(env.cart.theta)],[sin(env.cart.theta)]), color="blue")

    prm_path = get_prop(prm,vertex_num,:path_to_goal)
    prm_path_x = []
    prm_path_y = []

    for waypoint in prm_path
        push!(prm_path_x, get_prop(prm,waypoint,:x))
        push!(prm_path_y, get_prop(prm,waypoint,:y))
        scatter!([get_prop(prm,waypoint,:x)], [get_prop(prm,waypoint,:y)], shape=:circle, color="green", msize= 0.3*plot_size*cart_size/env.length)
    end

    plot!(prm_path_x,prm_path_y)

    annotate!(env.cart_start_location.x, env.cart_start_location.y, text("S", :purple, :right, 20))
    annotate!(env.cart.goal.x, env.cart.goal.y, text("G", :purple, :right, 20))
    plot!(size=(plot_size,plot_size))
    display(p)
end

function display_fmm_path_from_given_vertex(env::experiment_environment, given_x_point, given_y_point, gradient_information_matrix, dx=0.1, dy=0.1)

    #Plot Boundaries
    p = plot([0.0],[0.0],legend=false,grid=false)
    plot!([env.length], [env.breadth],legend=false)

    #Plot Humans in the cart lidar data
    for i in 1: length(env.cart_lidar_data)
        scatter!([env.cart_lidar_data[i].x], [env.cart_lidar_data[i].y],color="green",msize=0.5*plot_size/env.length)
    end

    #Plot humans in complete cart_lidar_data
    for i in 1: length(env.complete_cart_lidar_data)
        in_lidar_data_flag = false
        for green_human in env.cart_lidar_data
            if(env.complete_cart_lidar_data[i].id == green_human.id)
                in_lidar_data_flag = true
                break
            end
        end
        if(!in_lidar_data_flag)
            scatter!([env.complete_cart_lidar_data[i].x], [env.complete_cart_lidar_data[i].y],color="red",msize=0.5*plot_size/env.length)
        end
    end

    #Plot Obstacles
    for i in 1: length(env.obstacles)
        scatter!([env.obstacles[i].x], [env.obstacles[i].y],color="black",shape=:circle,msize=plot_size*env.obstacles[i].r/env.length)
    end

    #Plot Golfcart
    scatter!([env.cart.x], [env.cart.y], shape=:circle, color="blue", msize= 0.3*plot_size*cart_size/env.length)
    quiver!([env.cart.x],[env.cart.y],quiver=([cos(env.cart.theta)],[sin(env.cart.theta)]), color="blue")

    #Generate fmm path
    x_points,y_points = @time find_path_from_given_point(given_x_point,given_y_point,dx,dy,env.cart.goal.x,env.cart.goal.y,gradient_information_matrix)
    plot!(x_points,y_points)

    annotate!(env.cart_start_location.x, env.cart_start_location.y, text("S", :purple, :right, 20))
    annotate!(env.cart.goal.x, env.cart.goal.y, text("G", :purple, :right, 20))
    plot!(size=(plot_size,plot_size))
    display(p)
    return x_points,y_points
end
