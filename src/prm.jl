using LightGraphs
using MetaGraphs
using Plots
using Debugger
include("environment.jl")
include("utils.jl")

function get_distance_between_two_prm_vertices(prm, first_vertex_index, second_vertex_index)
    euclidean_distance = (get_prop(prm,first_vertex_index,:x) - get_prop(prm,second_vertex_index,:x))^2
    euclidean_distance += (get_prop(prm,first_vertex_index,:y) - get_prop(prm,second_vertex_index,:y))^2
    return sqrt(euclidean_distance)
end

function add_vertex_to_prm_graph(prm_graph, world)

    prm_vertex_found_flag = false
    obstacle_padding_threshold_distance = 2.0
    min_distance_between_vertices_threshold = 2

    while(prm_vertex_found_flag!=true)
        sampled_x_point = rand()*world.length
        sampled_y_point = rand()*world.breadth
        collision_flag = false

        for obstacle in world.obstacles
            if( is_within_range_check_with_points(sampled_x_point,sampled_y_point,obstacle.x,obstacle.y,
                                                        obstacle.r+obstacle_padding_threshold_distance) )
                collision_flag = true
                break
            end
        end
        if(!collision_flag)
            for i in 1:nv(prm_graph)
                if(i == 3)
                    continue
                else
                    euclidean_distance = (get_prop(prm_graph,i,:x) - sampled_x_point)^2
                    euclidean_distance += (get_prop(prm_graph,i,:y) - sampled_y_point)^2
                    if( sqrt(euclidean_distance) < min_distance_between_vertices_threshold)
                        collision_flag = true
                    end
                end
            end
        end
        if(collision_flag == false)
            add_vertex!(prm_graph)
            set_props!(prm_graph, nv(prm_graph), Dict(:x=>sampled_x_point, :y => sampled_y_point, :dist_to_goal => 0.0))
            prm_vertex_found_flag = true
        end
    end
end

function generate_prm_vertices(max_num_vertices, world)

    prm_graph = MetaGraph()
    set_prop!(prm_graph, :description, "This is the PRM for global planning")
    add_vertex!(prm_graph)
    set_props!(prm_graph, nv(prm_graph), Dict(:x=>world.cart_start_location.x, :y => world.cart_start_location.y, :dist_to_goal => 0.0))
    add_vertex!(prm_graph)
    set_props!(prm_graph, nv(prm_graph), Dict(:x=>world.cart.goal.x, :y => world.cart.goal.y, :dist_to_goal => 0.0))
    num_vertices_so_far = 2

    while(num_vertices_so_far<max_num_vertices)
        #@show("Vertices so far : ", num_vertices_so_far )
        add_vertex_to_prm_graph(prm_graph, world)
        num_vertices_so_far += 1
    end
    return prm_graph
end

function get_max_weight_edge_from_prm(prm)
    max_weight_so_far = -1.0
    for edge in collect(edges(prm))
        if( get_prop(prm,edge,:weight) > max_weight_so_far)
            max_weight_so_far = get_prop(prm,edge,:weight)
        end
    end
    return max_weight_so_far
end

function generate_prm_edges(world, num_nearest_nebhrs)

    dist_dict = Dict{Int, Array{Tuple{Int64,Float64},1}}()

    for i in 1:nv(world.graph)
        dist_array =  Array{Tuple{Int64,Float64},1}()
        for j in 1:nv(world.graph)
            if(i==j)
                push!(dist_array, (j,0.0))
            else
                push!( dist_array, (j,get_distance_between_two_prm_vertices(world.graph,i,j)) )
            end
        end
        dist_dict[i] = sort(dist_array, by = x->x[2])
    end

    for i in 1:nv(world.graph)
        j = 3
        num_out_edges = 0
        while(num_out_edges<num_nearest_nebhrs && j<=nv(world.graph))
            if(has_edge(world.graph,i,dist_dict[i][j][1]))
                num_out_edges +=1
                #Don't do anything
            else
                add_edge_flag = true
                for obstacle in world.obstacles
                    if( find_if_circle_and_line_segment_intersect(obstacle.x,obstacle.y,obstacle.r+2,get_prop(world.graph,i,:x),
                        get_prop(world.graph,i,:y),get_prop(world.graph,dist_dict[i][j][1],:x),get_prop(world.graph,dist_dict[i][j][1],:y)) )
                        add_edge_flag = false
                    end
                end
                if(add_edge_flag)
                    add_edge!(world.graph,i,dist_dict[i][j][1], :weight, dist_dict[i][j][2] )
                    # set_prop!(world.graph, Edge(i, dist_dict[i][j][1]), :weight, dist_dict[i][j][2])
                    num_out_edges +=1
                end
            end
            j +=1
        end
    end
    # gplot(env.prm)
    for i in 1:nv(world.graph)
        dsp = dijkstra_shortest_paths(world.graph, i)
        path_to_goal = Int64[]
        curr_par = 2
        while(curr_par!=0)
            push!(path_to_goal,curr_par)
            curr_par = dsp.parents[curr_par]
        end
        reverse!(path_to_goal)
        set_prop!(world.graph, i, :dist_to_goal, dsp.dists[2])
        set_prop!(world.graph, i, :path_to_goal, path_to_goal)
    end
    return dist_dict
end

#=
env = generate_environment_small_circular_obstacles(300, MersenneTwister(15))
env.graph = generate_prm_vertices(100,env)
d = generate_prm_edges(env, 10)
=#

function generate_prm_points_lookup_table(world)
    discretization_width_in_x = 1.0
    discretization_width_in_y = 1.0
    discretization_width_in_theta = 10.0     #in degrees
    lookup_table = Dict{String,Int}()

    for i in 0:(world.length/discretization_width_in_x)-1
        for j in 0:(world.breadth/discretization_width_in_y)-1
            println("Doing it for this i and j at the moment " * string(i)* ","* string(j))
            for k in 0:(360/discretization_width_in_theta)-1
                dict_key = "xbin_"*string(i*discretization_width_in_x)*"_"*string((i+1)*discretization_width_in_x)*"_ybin_"*
                                                        string(j*discretization_width_in_y)*"_"*string((j+1)*discretization_width_in_y)*
                                                        "_theta_"*string(k)
                x_point = ( (i*discretization_width_in_x) + ((i+1)*discretization_width_in_x) )/ 2
                y_point = ( (j*discretization_width_in_y) + ((j+1)*discretization_width_in_y) )/ 2
                theta_point = k*discretization_width_in_theta

                closest_vertex = get_nearest_prm_point_in_cone(x_point,y_point,theta_point,world.graph)

                # for prm_ver_index in 1:nv(world.graph)
                #     dist_to_curr_vertex = (x_point - get_prop(world.graph,prm_ver_index,:x))^2 + (y_point - get_prop(world.graph,prm_ver_index,:y))^2
                #     dist_from_curr_point_to_goal = dist_to_curr_vertex + get_prop(world.graph,prm_ver_index,:dist_to_goal)
                #     if(dist_from_curr_point_to_goal < closest_dist_so_far )
                #         closest_vertex = prm_ver_index
                #         closest_dist_so_far = dist_from_curr_point_to_goal
                #     end
                # end
                lookup_table[dict_key] = closest_vertex
            end
        end
    end

    return lookup_table
end


function Distances.evaluate(dist::myMet,a::Array{Float64},b::Array{Float64})
           println("HG")
           dist = b[3] + sqrt( (a[1]-b[1])^2 + (a[2]-b[2])^2 )
end


function get_nearest_prm_point_in_cone(x,y,theta,prm,cone_half_angle::Float64=(2*pi/9.0))

    closest_prm_vertex = -1
    closest_dist_so_far = Inf

    for i in 1:nv(prm)
        angle_between_given_point_and_prm_vertex = get_heading_angle(get_prop(prm,i,:x), get_prop(prm,i,:y), x, y)
        difference_in_angles = abs(theta - angle_between_given_point_and_prm_vertex)
        if(difference_in_angles <= cone_half_angle)
            euclidean_distance = (x - get_prop(prm,i,:x))^2 + (y - get_prop(prm,i,:y))^2
            if(euclidean_distance + get_prop(prm,i,:dist_to_goal) < closest_dist_so_far)
                 closest_prm_vertex = i
                 closest_dist_so_far = euclidean_distance + get_prop(prm,i,:dist_to_goal)
            end
        elseif ( (2*pi - difference_in_angles) <= cone_half_angle )
            euclidean_distance = (x - get_prop(prm,i,:x))^2 + (y - get_prop(prm,i,:y))^2
            if(euclidean_distance + get_prop(prm,i,:dist_to_goal) < closest_dist_so_far)
                 closest_prm_vertex = i
                 closest_dist_so_far = euclidean_distance + get_prop(prm,i,:dist_to_goal)
            end
        else
            continue
        end
    end
    return closest_prm_vertex
end
