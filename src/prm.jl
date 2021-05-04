using LightGraphs
using MetaGraphs
using Plots

function generate_prm_vertices(max_num_vertices, world)

    prm_graph = MetaGraph()
    set_prop!(prm_graph, :description, "This is the PRM for global planning")
    add_vertex!(prm_graph)
    set_props!(prm_graph, nv(prm_graph), Dict(:x=>world.cart.x, :y => world.cart.y, :dist => 0.0))
    add_vertex!(prm_graph)
    set_props!(prm_graph, nv(prm_graph), Dict(:x=>world.cart.goal.x, :y => world.cart.goal.y, :dist => 0.0))
    num_vertices_so_far = 2

    while(num_vertices_so_far<max_num_vertices)
        #@show("Vertices so far : ", num_vertices_so_far )
        sampled_x_point = rand()*world.length
        sampled_y_point = rand()*world.breadth
        collision_flag = false
        threshold_distance = 2.0
        for obstacle in world.obstacles
            if( is_within_range_check_with_points(sampled_x_point,sampled_y_point,obstacle.x,obstacle.y,obstacle.r+threshold_distance) )
                collision_flag = true
                break
            end
        end
        if(collision_flag == false)
            add_vertex!(prm_graph)
            set_props!(prm_graph, nv(prm_graph), Dict(:x=>sampled_x_point, :y => sampled_y_point, :dist => 0.0))
            num_vertices_so_far += 1
        end
    end
    return prm_graph
end

function get_distance_between_two_prm_vertices(prm, first_vertex_index, second_vertex_index)
    euclidean_distance = (get_prop(prm,first_vertex_index,:x) - get_prop(prm,second_vertex_index,:x))^2
    euclidean_distance += (get_prop(prm,first_vertex_index,:y) - get_prop(prm,second_vertex_index,:y))^2
    return sqrt(euclidean_distance)
end


function generate_prm_edges(world, num_nearest_nebhrs)

    dist_dict = Dict{Int, Array{Tuple{Int64,Float64},1}}()
    for i in 1:nv(world.prm)
        dist_array =  Array{Tuple{Int64,Float64},1}()
        for j in 1:nv(world.prm)
            if(i==j)
                push!(dist_array, (j,0.0))
            else
                push!( dist_array, (j,get_distance_between_two_prm_vertices(world.prm,i,j)) )
            end
        end
        dist_dict[i] = sort(dist_array, by = x->x[2])
    end

    for i in 1:nv(world.prm)
        j = 2
        num_out_edges = 0
        while(num_out_edges<num_nearest_nebhrs)
            if(has_edge(world.prm,i,dist_dict[i][j][1]))
                num_out_edges +=1
            else
                add_edge_flag = true
                for obstacle in world.obstacles
                    if( find_if_circle_and_line_segment_intersect(obstacle.x,obstacle.y,obstacle.r+2,
                                                get_prop(world.prm,i,:x),get_prop(world.prm,i,:y),get_prop(world.prm,j,:x),get_prop(world.prm,j,:y)) )
                        add_edge_flag = false
                    end
                end
                if(add_edge_flag)
                    add_edge!(world.prm,i,dist_dict[i][j][1])
                    num_out_edges +=1
                end
            end
            j +=1
        end
    end
    # gplot(env.prm)
    return dist_dict
end
