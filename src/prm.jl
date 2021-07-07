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
    max_edge_weight_threshold = 5.0
    for i in 1:nv(world.graph)
        dist_array =  Array{Tuple{Int64,Float64},1}()
        if(i == 3)
            dist_array = repeat([(-10,-10)],nv(world.graph))
        else
            for j in 1:nv(world.graph)
                if(i==j)
                    push!(dist_array, (j,0.0))
                elseif(j==3)
                    push!(dist_array, (j,-10.0))
                else
                    push!( dist_array, (j,get_distance_between_two_prm_vertices(world.graph,i,j)) )
                end
            end
        end
        dist_dict[i] = sort(dist_array, by = x->x[2])
    end
    #return dist_dict
    for i in 1:nv(world.graph)
        if(i!=3)
            add_edge!(world.graph,i,3,:weight, 0.0)
        end
    end
    keep_adding_edge_flag = true
    j=3
    while( keep_adding_edge_flag && j<=nv(world.graph) )
        for i in 1:nv(world.graph)
            if(i!=3)
                num_out_edges_from_i = length( neighbors(world.graph,i) )
                num_out_edges_from_j = length( neighbors(world.graph,dist_dict[i][j][1]) )
                if( (num_out_edges_from_i < num_nearest_nebhrs) && (num_out_edges_from_j < num_nearest_nebhrs) )
                    if( !has_edge(world.graph,i,dist_dict[i][j][1]) )
                        add_edge_flag = true
                        for obstacle in world.obstacles
                            if( find_if_circle_and_line_segment_intersect(obstacle.x,obstacle.y,obstacle.r+2,get_prop(world.graph,i,:x),
                                get_prop(world.graph,i,:y),get_prop(world.graph,dist_dict[i][j][1],:x),get_prop(world.graph,dist_dict[i][j][1],:y)) )
                                add_edge_flag = false
                            end
                        end
                        if(add_edge_flag && dist_dict[i][j][2]<=max_edge_weight_threshold)
                            add_edge!(world.graph,i,dist_dict[i][j][1], :weight, dist_dict[i][j][2] )
                            add_edge!(world.graph,dist_dict[i][j][1],i, :weight, dist_dict[i][j][2] )
                        end
                    end
                end
            end
        end
        j+=1
    end

    for i in 1:nv(world.graph)
        path_to_goal = Int64[]
        if(i!=3)
            dsp = dijkstra_shortest_paths(world.graph, i)
            curr_par = 2
            while(curr_par!=0)
                push!(path_to_goal,curr_par)
                curr_par = dsp.parents[curr_par]
            end
            reverse!(path_to_goal)
            set_prop!(world.graph, i, :dist_to_goal, dsp.dists[2])
            set_prop!(world.graph, i, :path_to_goal, path_to_goal)
        else
            set_prop!(world.graph, i, :dist_to_goal, Inf)
            set_prop!(world.graph, i, :path_to_goal, path_to_goal)
        end
    end
    return dist_dict
end

function old_generate_prm_edges(world, num_nearest_nebhrs)

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
        if(i!=3)
            j = 3
            num_out_edges = length( neighbors(world.graph,i) )
            while(num_out_edges<num_nearest_nebhrs && j<=nv(world.graph))
                if(has_edge(world.graph,i,dist_dict[i][j][1]))
                    # num_out_edges +=1
                    #Don't do anything
                else
                    add_edge_flag = true
                    for obstacle in world.obstacles
                        if( find_if_circle_and_line_segment_intersect(obstacle.x,obstacle.y,obstacle.r+2,get_prop(world.graph,i,:x),
                            get_prop(world.graph,i,:y),get_prop(world.graph,dist_dict[i][j][1],:x),get_prop(world.graph,dist_dict[i][j][1],:y)) )
                            add_edge_flag = false
                        end
                    end
                    if( length(neighbors(world.graph,dist_dict[i][j][1])) >= num_nearest_nebhrs )
                        add_edge_flag = false
                    end
                    if(add_edge_flag)
                        add_edge!(world.graph,i,dist_dict[i][j][1], :weight, dist_dict[i][j][2] )
                        add_edge!(world.graph,dist_dict[i][j][1],i, :weight, dist_dict[i][j][2] )
                        # set_prop!(world.graph, Edge(i, dist_dict[i][j][1]), :weight, dist_dict[i][j][2])
                        num_out_edges +=1
                    end
                end
                j +=1
            end
            add_edge!(world.graph,i,3,:weight, 0.0)
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
