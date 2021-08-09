using LightGraphs
using MetaGraphs
using Plots
using Debugger
include("environment.jl")
include("utils.jl")

struct lookup_table_struct
    closest_prm_vertex_num::Int64
    closest_prm_vertex_x::Float64
    closest_prm_vertex_y::Float64
    next_prm_vertex_num::Int64
    next_prm_vertex_x::Float64
    next_prm_vertex_y::Float64
end


function get_distance_between_two_prm_vertices(prm, first_vertex_index, second_vertex_index)
    euclidean_distance = (get_prop(prm,first_vertex_index,:x) - get_prop(prm,second_vertex_index,:x))^2
    euclidean_distance += (get_prop(prm,first_vertex_index,:y) - get_prop(prm,second_vertex_index,:y))^2
    return sqrt(euclidean_distance)
end

function check_if_edge_can_exist_between_p1_and_p2(p1_x,p1_y,p2_x,p2_y,world)
    for obstacle in world.obstacles
        if( find_if_circle_and_line_segment_intersect(obstacle.x,obstacle.y,obstacle.r+2,p1_x,p1_y,p2_x,p2_y) )
            return false
        end
    end
    return true
end

function add_vertex_to_prm_graph(prm_graph, rand_rng, world)

    prm_vertex_found_flag = false
    obstacle_padding_threshold_distance = 2.0 + world.cart.L
    min_distance_between_vertices_threshold = 2

    while(prm_vertex_found_flag!=true)
        sampled_x_point = rand(rand_rng)*world.length
        sampled_y_point = rand(rand_rng)*world.breadth
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

function generate_prm_vertices(max_num_vertices, rand_rng, world)

    prm_graph = MetaGraph()
    set_prop!(prm_graph, :description, "This is the PRM for global planning")
    add_vertex!(prm_graph)
    set_props!(prm_graph, nv(prm_graph), Dict(:x=>world.cart_start_location.x, :y => world.cart_start_location.y, :dist_to_goal => 0.0))
    add_vertex!(prm_graph)
    set_props!(prm_graph, nv(prm_graph), Dict(:x=>world.cart.goal.x, :y => world.cart.goal.y, :dist_to_goal => 0.0))
    num_vertices_so_far = 2

    while(num_vertices_so_far<max_num_vertices)
        #@show("Vertices so far : ", num_vertices_so_far )
        add_vertex_to_prm_graph(prm_graph, rand_rng, world)
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

function generate_prm_edges(world, graph, num_nearest_nebhrs)

    dist_dict = Dict{Int, Array{Tuple{Int64,Float64},1}}()

    for i in 1:nv(graph)
        dist_array =  Array{Tuple{Int64,Float64},1}()
        for j in 1:nv(graph)
            if(i==j)
                push!(dist_array, (j,0.0))
            else
                push!( dist_array, (j,get_distance_between_two_prm_vertices(graph,i,j)) )
            end
        end
        dist_dict[i] = sort(dist_array, by = x->x[2])
    end

    for i in 1:nv(graph)
        j = 3
        num_out_edges = 0
        while(num_out_edges<num_nearest_nebhrs && j<=nv(graph))
            if(has_edge(graph,i,dist_dict[i][j][1]))
                num_out_edges +=1
                #Don't do anything
            else
                add_edge_flag = true
                for obstacle in world.obstacles
                    if( find_if_circle_and_line_segment_intersect(obstacle.x,obstacle.y,obstacle.r+2,get_prop(graph,i,:x),
                        get_prop(graph,i,:y),get_prop(graph,dist_dict[i][j][1],:x),get_prop(graph,dist_dict[i][j][1],:y)) )
                        add_edge_flag = false
                    end
                end
                if(add_edge_flag)
                    add_edge!(graph,i,dist_dict[i][j][1], :weight, dist_dict[i][j][2] )
                    # set_prop!(world.graph, Edge(i, dist_dict[i][j][1]), :weight, dist_dict[i][j][2])
                    num_out_edges +=1
                end
            end
            j +=1
        end
    end
    # gplot(env.prm)
    for i in 1:nv(graph)
        dsp = dijkstra_shortest_paths(graph, i)
        path_to_goal = Int64[]
        curr_par = 2
        while(curr_par!=0)
            push!(path_to_goal,curr_par)
            curr_par = dsp.parents[curr_par]
        end
        reverse!(path_to_goal)
        set_prop!(graph, i, :dist_to_goal, dsp.dists[2])
        set_prop!(graph, i, :path_to_goal, path_to_goal)
    end
    return dist_dict
end

function generate_prm_points_coordinates_lookup_table_holonomic(world, graph)
    discretization_width_in_x = 1.0
    discretization_width_in_y = 1.0
    lookup_table = Array{lookup_table_struct,2}(undef,100,100)

    for i in 1:(world.length/discretization_width_in_x)
        for j in 1:(world.breadth/discretization_width_in_y)
            println("Doing it for this i and j at the moment " * string(i)* ","* string(j))
            x_point = ( (i*discretization_width_in_x) + ((i-1)*discretization_width_in_x) )/ 2
            y_point = ( (j*discretization_width_in_y) + ((j-1)*discretization_width_in_y) )/ 2
            closest_prm_vertex = -1
            closest_dist_so_far = Inf
            for prm_ver_index in 1:nv(graph)
                is_traversal_possible = check_if_edge_can_exist_between_p1_and_p2(x_point,y_point,get_prop(graph,prm_ver_index,:x), get_prop(graph,prm_ver_index,:y),world)
                if(is_traversal_possible)
                    dist_to_curr_vertex = (x_point - get_prop(graph,prm_ver_index,:x))^2 + (y_point - get_prop(graph,prm_ver_index,:y))^2
                    dist_from_curr_point_to_goal = dist_to_curr_vertex + get_prop(graph,prm_ver_index,:dist_to_goal)
                    if(dist_from_curr_point_to_goal < closest_dist_so_far )
                        closest_prm_vertex = prm_ver_index
                        closest_dist_so_far = dist_from_curr_point_to_goal
                    end
                end
            end
            if(closest_prm_vertex == -1)
                closest_prm_vertex_x = -1
                closest_prm_vertex_y = -1
                second_prm_vertex = -1
                second_prm_vertex_x = -1
                second_prm_vertex_y = -1
            elseif( length(get_prop(graph,closest_prm_vertex,:path_to_goal)) == 1 )
                closest_prm_vertex_x = get_prop(graph,closest_prm_vertex,:x)
                closest_prm_vertex_y = get_prop(graph,closest_prm_vertex,:y)
                second_prm_vertex = closest_prm_vertex
                second_prm_vertex_x = closest_prm_vertex_x
                second_prm_vertex_y = closest_prm_vertex_y
            else
                closest_prm_vertex_x = get_prop(graph,closest_prm_vertex,:x)
                closest_prm_vertex_y = get_prop(graph,closest_prm_vertex,:y)
                second_prm_vertex = get_prop(graph,closest_prm_vertex,:path_to_goal)[2]
                second_prm_vertex_x = get_prop(graph,second_prm_vertex,:x)
                second_prm_vertex_y = get_prop(graph,second_prm_vertex,:y)
            end

            struct_to_be_written = lookup_table_struct(closest_prm_vertex, closest_prm_vertex_x, closest_prm_vertex_y, second_prm_vertex,
                                                                    second_prm_vertex_x, second_prm_vertex_y)
            lookup_table[Int(i),Int(j)] = struct_to_be_written
        end
    end
    return lookup_table
end

function generate_prm_points_lookup_table_non_holonomic(world, graph)
    discretization_width_in_x = 1.0
    discretization_width_in_y = 1.0
    discretization_width_in_theta = 10     #in degrees
    lookup_table = Array{Tuple{Int64,Float64,Float64,Float64},3}(undef,100,100,36)

    for i in 1:(world.length/discretization_width_in_x)
        for j in 1:(world.breadth/discretization_width_in_y)
            println("Doing it for this i and j at the moment " * string(i)* ","* string(j))
            for k in 1:(360/discretization_width_in_theta)
                # dict_key = "xbin_"*string(i*discretization_width_in_x)*"_"*string((i+1)*discretization_width_in_x)*"_ybin_"*
                #                                         string(j*discretization_width_in_y)*"_"*string((j+1)*discretization_width_in_y)
                dict_key = "xbin_"*string(i*discretization_width_in_x)*"_"*string((i+1)*discretization_width_in_x)*"_ybin_"*
                                                        string(j*discretization_width_in_y)*"_"*string((j+1)*discretization_width_in_y)*
                                                        "_theta_bin_"*string(k*discretization_width_in_theta)*"_"*string((k+1)*discretization_width_in_theta)
                x_point = ( (i*discretization_width_in_x) + ((i-1)*discretization_width_in_x) )/ 2
                y_point = ( (j*discretization_width_in_y) + ((j-1)*discretization_width_in_y) )/ 2
                theta_point = ( (k*discretization_width_in_theta) + ((k-1)*discretization_width_in_theta) )/ 2
                theta_point = theta_point*pi/180  #need to be in radians
                closest_vertex = get_nearest_prm_point_in_cone(x_point,y_point,theta_point,graph)

                # for prm_ver_index in 1:nv(world.graph)
                #     dist_to_curr_vertex = (x_point - get_prop(world.graph,prm_ver_index,:x))^2 + (y_point - get_prop(world.graph,prm_ver_index,:y))^2
                #     dist_from_curr_point_to_goal = dist_to_curr_vertex + get_prop(world.graph,prm_ver_index,:dist_to_goal)
                #     if(dist_from_curr_point_to_goal < closest_dist_so_far )
                #         closest_vertex = prm_ver_index
                #         closest_dist_so_far = dist_from_curr_point_to_goal
                #     end
                # end
                lookup_table[Int(i),Int(j),Int(k)] = Tuple{Int64,Float64,Float64,Float64}((closest_vertex[1],closest_vertex[2],closest_vertex[3],closest_vertex[4]))
            end
        end
    end
    return lookup_table
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
    if(closest_prm_vertex!=-1)
        return (closest_prm_vertex, get_prop(prm,closest_prm_vertex,:x), get_prop(prm,closest_prm_vertex,:y), get_prop(prm,closest_prm_vertex,:dist_to_goal))
    else
        return (closest_prm_vertex, 0.0, 0.0, 0.0)
    end
end

function get_best_possible_prm_vertex_closest_to_given_point_non_holonomic(x,y,theta,prm,world,cone_half_angle::Float64=(2*pi/9.0))

    best_prm_vertex = -1
    closest_dist_so_far = Inf

    for i in 1:nv(prm)
        if(i!=2)
            angle_between_given_point_and_prm_vertex = get_heading_angle(get_prop(prm,i,:x), get_prop(prm,i,:y), x, y)
            difference_in_angles = abs(theta - angle_between_given_point_and_prm_vertex)
            if(difference_in_angles <= cone_half_angle)
                is_traversal_possible = check_if_edge_can_exist_between_p1_and_p2(x,y,get_prop(prm,i,:x), get_prop(prm,i,:y),world)
                if(is_traversal_possible)
                    distance_between_point_and_prm_vertex = sqrt((x - get_prop(prm,i,:x))^2 + (y - get_prop(prm,i,:y))^2)
                    if(distance_between_point_and_prm_vertex + get_prop(prm,i,:dist_to_goal) < closest_dist_so_far)
                        best_prm_vertex = i
                        closest_dist_so_far = distance_between_point_and_prm_vertex + get_prop(prm,i,:dist_to_goal)
                    end
                end
            elseif ( (2*pi - difference_in_angles) <= cone_half_angle )
                is_traversal_possible = check_if_edge_can_exist_between_p1_and_p2(x,y,get_prop(prm,i,:x), get_prop(prm,i,:y),world)
                if(is_traversal_possible)
                    distance_between_point_and_prm_vertex = sqrt((x - get_prop(prm,i,:x))^2 + (y - get_prop(prm,i,:y))^2)
                    if(distance_between_point_and_prm_vertex + get_prop(prm,i,:dist_to_goal) < closest_dist_so_far)
                        best_prm_vertex = i
                        closest_dist_so_far = distance_between_point_and_prm_vertex + get_prop(prm,i,:dist_to_goal)
                    end
                end
            else
                continue
            end
        end
    end

    if(best_prm_vertex!=-1)
        return (best_prm_vertex, get_prop(prm,best_prm_vertex,:x), get_prop(prm,best_prm_vertex,:y), get_prop(prm,best_prm_vertex,:dist_to_goal))
    else
        return (best_prm_vertex, 0.0, 0.0, 0.0)
    end
end

function generate_prm_points_coordinates_lookup_table_non_holonomic(world, graph)
    discretization_width_in_x = 1.0
    discretization_width_in_y = 1.0
    discretization_width_in_theta = 10     #in degrees
    lookup_table = Array{Tuple{ Int64, Tuple{Float64, Float64}, Int64, Tuple{Float64, Float64}, Float64},3}(undef,100,100,36)
    # lookup_table = Array{Tuple{ Int64,Float64,Float64,Float64},3}(undef,100,100,36)

    for i in 1:(world.length/discretization_width_in_x)
        for j in 1:(world.breadth/discretization_width_in_y)
            # println("Doing it for this i and j at the moment " * string(i)* ","* string(j))
            for k in 1:(360/discretization_width_in_theta)
                x_point = ( (i*discretization_width_in_x) + ((i-1)*discretization_width_in_x) )/ 2
                y_point = ( (j*discretization_width_in_y) + ((j-1)*discretization_width_in_y) )/ 2
                theta_point = ( (k*discretization_width_in_theta) + ((k-1)*discretization_width_in_theta) )/ 2
                theta_point = theta_point*pi/180  #need to be in radians
                println("Doing it for this i, j, k at the moment " * string(i)* ","* string(j)* ","* string(k))
                println("x, y, theta are -  " * string(x_point)* ","* string(y_point)* ","* string(theta_point))
                closest_vertex = get_best_possible_prm_vertex_closest_to_given_point_non_holonomic(x_point,y_point,theta_point,graph,world)
                if(closest_vertex[1] == -1)
                    tuple_to_be_written = (-1, (0.0,0.0), -1, (0.0,0.0), 0.0)
                else
                    path_to_goal_from_closest_vertex = get_prop(graph, closest_vertex[1],:path_to_goal)
                    second_vertex_num = get_prop(graph, closest_vertex[1],:path_to_goal)[2]
                    second_vertex_num_x = get_prop(graph, second_vertex_num,:x)
                    second_vertex_num_y = get_prop(graph, second_vertex_num,:y)
                    tuple_to_be_written = (closest_vertex[1], (closest_vertex[2],closest_vertex[3]), second_vertex_num, (second_vertex_num_x,second_vertex_num_y), closest_vertex[4])
                end
                lookup_table[Int(i),Int(j),Int(k)] = tuple_to_be_written
                # lookup_table[Int(i),Int(j),Int(k)] = Tuple{Int64,Float64,Float64,Float64}((closest_vertex[1],closest_vertex[2],closest_vertex[3],closest_vertex[4]))
            end
        end
    end
    return lookup_table
end

#=
env = generate_environment_large_circular_obstacles(300, MersenneTwister(15))
graph = generate_prm_vertices(500,MersenneTwister(15),env)
d = generate_prm_edges(env, graph, 10)
display_env(env)
display_env(env,nothing,nothing,graph)
=#

# function Distances.evaluate(dist::myMet,a::Array{Float64},b::Array{Float64})
#            println("HG")
#            dist = b[3] + sqrt( (a[1]-b[1])^2 + (a[2]-b[2])^2 )
# end


#=
for i in 1:nv(graph)
   display_prm_path_from_given_vertex(env,graph,i)
   savefig("./prm_paths/from_vertex_"*string(i)*".png")
end
=#
