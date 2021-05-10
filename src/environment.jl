using Plots
using Random

#Global Variables
plot_size = 800; #number of pixels
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
    num_humans::Int64
    goals::Array{location,1}
    humans::Array{human_state,1}
    obstacles::Array{obstacle_location,1}
    cart::cart_state
    cart_lidar_data::Array{human_state,1}
    complete_cart_lidar_data::Array{human_state,1}
    cart_hybrid_astar_path::Array{Float64,1}
end

#Function to plot an ellipse
function ellipse(h,k,a,b)
    θ = LinRange(0,2*pi,500)
    h.+ a*cos.(θ), k.+ b*sin.(θ)
end

#Function to display the environment
function display_env(env::experiment_environment,particles=nothing,weights=nothing)

    #Plot Boundaries
    p = plot([0.0],[0.0],legend=false,grid=false)
    plot!([env.length], [env.breadth],legend=false)

    #Plot Humans in the cart lidar data
    for i in 1: length(env.cart_lidar_data)
        scatter!([env.cart_lidar_data[i].x], [env.cart_lidar_data[i].y],color="green",msize=0.5*plot_size/env.length)
    end

    #Plot Rest of the Humans
    for i in 1: length(env.humans)
        in_lidar_data_flag = false
        for green_human in env.cart_lidar_data
            if(env.humans[i].id == green_human.id)
                in_lidar_data_flag = true
                break
            end
        end
        if(!in_lidar_data_flag)
            scatter!([env.humans[i].x], [env.humans[i].y],color="red",msize=0.5*plot_size/env.length)
        end
    end

    #Plot Obstacles
    for i in 1: length(env.obstacles)
        scatter!([env.obstacles[i].x], [env.obstacles[i].y],color="black",shape=:circle,msize=plot_size*env.obstacles[i].r/env.length)
    end

    #Plot Golfcart
    scatter!([env.cart.x], [env.cart.y], shape=:circle, color="blue", msize= 0.3*plot_size*cart_size/env.length)

    if(length(env.cart_hybrid_astar_path)!=0)
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
            push!(path_x,last(x))
            push!(path_y,last(y))
            initial_state = [last(x),last(y),last(theta)]
        end
        plot!(path_x,path_y,color="black")
    end

    if(particles!=nothing)
        # for particle in particles
        #     scatter!([particle.x], [particle.y], shape=:square, color="LightGrey", msize= 0.6*plot_size*cart_size/env.length)
        # end
        m,v = get_mean_covar_from_particles(particles,weights)
        scatter!([m[1]], [m[2]], shape=:plus, color="black", msize= 0.6*plot_size*cart_size/env.length)
        plot!(ellipse(m[1],m[2],v[1,1],v[2,2]), seriestype=[:shape,],lw=0.5,c=:blue,linecolor=:black,legend=false,fillalpha=0.2,aspect_ratio=1)
        println(v[1,1]," ",v[2,2])
    end
    plot!(size=(plot_size,plot_size))
    display(p)
end

#Define the Environment
function generate_environment_no_obstacles(number_of_humans, user_defined_rng)

    world_length = 50.0
    world_breadth = 50.0
    g1 = location(0.0,0.0)
    g2 = location(0.0,world_breadth)
    g3 = location(world_length,world_breadth)
    g4 = location(world_length,0.0)
    cart_goal = location(world_length,30.0)
    all_goals_list = [g1,g2,g3,g4]
    all_obstacle_list = []
    max_num_humans = number_of_humans

    golfcart = cart_state(1.0,25.0,0.0,0.0,0.5,cart_goal)
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

    world = experiment_environment(world_length,world_breadth,length(human_state_start_list),
                    all_goals_list,human_state_start_list,all_obstacle_list,golfcart,initial_cart_lidar_data,
                    initial_complete_cart_lidar_data,Float64[])

    return world
end

function generate_environment_circular_obstacles(number_of_humans,user_defined_rng)

    world_length = 100.0
    world_breadth = 100.0
    g1 = location(0.0,0.0)
    g2 = location(0.0,world_breadth)
    g3 = location(world_length,world_breadth)
    g4 = location(world_length,0.0)
    cart_goal = location(world_length,70.0)
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

    golfcart = cart_state(1.0,20.0,0.0,0.0,1.0,cart_goal)
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

    world = experiment_environment(world_length,world_breadth,length(human_state_start_list),
                    all_goals_list,human_state_start_list,all_obstacle_list,golfcart,initial_cart_lidar_data,
                    initial_complete_cart_lidar_data,Float64[])

    return world
end
