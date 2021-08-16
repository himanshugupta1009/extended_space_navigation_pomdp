using FEFMM

struct gradient_info_struct
    grad_x::Float64
    grad_y::Float64
    mod_grad::Float64
    alpha::Float64
end

function solve_eikonal_equation_on_given_map(k, dx, source)
    size_k = size(k)
    bk = ones( size_k[1]+2, size_k[2]+2 )
    bk[2:size_k[1]+1,2:size_k[2]+1] = k

    modified_source = CartesianIndex(source[1]+1,source[2]+1)
    (t, ordering) = fefmm(bk,dx,modified_source)
    for i in 1:size(t)[1]
        for j in 1:size(t)[2]
            if( t[i,j] == Inf )
                t[i,j] = 10^6
            end
        end
    end
    return t
end

function calculate_gradients(padded_time_matrix)

    x_kernel = [ [1, 1, 1] [ 0, 0, 0] [-1, -1 ,-1] ]
    #x_kernel = x_kernel[1:end,end:-1:1]
    y_kernel = [ [ 1, 0, -1] [1 ,0, -1] [1, 0, -1] ]

    size_pm = size(padded_time_matrix)
    grad_info_matrix = Array{gradient_info_struct,2}(undef,size_pm[1]-2,size_pm[2]-2)

    for i in 2:size_pm[1]-1
        for j in 2:size_pm[2]-1
            grad_x = dot(x_kernel, padded_time_matrix[i-1:i+1,j-1:j+1])
            grad_y = dot(y_kernel, padded_time_matrix[i-1:i+1,j-1:j+1])
            mod_grad = sqrt( (grad_x)^2 + (grad_y)^2 )
            if(grad_x>=0.0 && grad_y>=0.0)
                alpha = atan(grad_y/grad_x)
            elseif(grad_x<=0.0 && grad_y>=0.0)
                alpha = atan(grad_y/grad_x) + pi
            elseif(grad_x<=0.0 && grad_y<=0.0)
                alpha = atan(grad_y/grad_x) + pi
            elseif(grad_x>=0.0 && grad_y<=0.0)
                alpha = atan(grad_y/grad_x) + (2*pi)
            end
            grad_info_matrix[i-1,j-1] = gradient_info_struct(grad_x,grad_y,mod_grad,alpha)
        end
    end
    return grad_info_matrix
end

function get_env_matrix_from_t_matrix(fmm_matrix)
    fmm_matrix_size = size(fmm_matrix)
    env_fmm_matrix = typeof(fmm_matrix)(undef,fmm_matrix_size[1],fmm_matrix_size[2])
    for matrix_ver_index in 1:fmm_matrix_size[1]
        env_ver_index = fmm_matrix_size[1] - matrix_ver_index + 1
        for matrix_hor_index in 1:fmm_matrix_size[2]
            env_hor_index = matrix_hor_index
            env_fmm_matrix[env_hor_index, env_ver_index] = fmm_matrix[matrix_ver_index, matrix_hor_index]
        end
    end
    return env_fmm_matrix
end

function find_path_from_given_point(x,y,dx,dy,goal_x,goal_y,grad_info_matrix,debug_flag=false)

    curr_x = x
    curr_y = y
    path_x = Float64[curr_x]
    path_y = Float64[curr_y]
    grad_mat_size = size(grad_info_matrix)
    println("\nCart's position is " ,curr_x," ",curr_y)
    try
        while( !is_within_range_check_with_points(curr_x,curr_y,goal_x,goal_y,1.0) )
            env_x_index = convert(Int,floor(curr_x/dx))
            env_y_index = convert(Int,floor(curr_y/dy))
            mat_ver_index = grad_mat_size[1] - env_y_index
            mat_hor_index = env_x_index + 1
            println("Matrix indices are : ", mat_ver_index, " ", mat_hor_index)
            grad_info = grad_info_matrix[mat_ver_index, mat_hor_index]
            println("Gradient's direction is : ", grad_info.alpha*180/pi)
            println("Gradient's magnitude is : ", grad_info.mod_grad)
            # mag = (grad_info.mod_grad)
            mag = clamp(grad_info.mod_grad,0,0.1)
            #mag = 0.1
            new_x = curr_x + mag*cos(-grad_info.alpha)
            new_y = curr_y + mag*sin(-grad_info.alpha)
            println("\nCart's position is " ,new_x," ",new_y)
            curr_x, curr_y = new_x,new_y
            push!(path_x,curr_x)
            push!(path_y,curr_y)
        end
        return path_x,path_y
    catch e
        println("An error was encountered")
        #println(e)
        return path_x,path_y
    end
end

function display_fmm_old(grad_info_matrix)
    p = plot([0.0],[0.0],legend=false,grid=true)
    #plot!([100], [100],legend=false)
    for matrix_ver_index in 1:10
        environment_ver_index = 10 - matrix_ver_index
       for matrix_hor_index in 1:10
           environment_hor_index = matrix_hor_index - 1
           theta = grad_info_matrix[matrix_ver_index,matrix_hor_index].alpha
           mag = grad_info_matrix[matrix_ver_index,matrix_hor_index].mod_grad
           println("i=",matrix_ver_index ," j=",matrix_hor_index," theta=", theta*180/pi)
           quiver!([environment_hor_index],[environment_ver_index],quiver=([mag*cos(theta)],[mag*sin(theta)]), color="blue")
       end
   end
   display(p)
end

function display_fmm(given_matrix)
    p = plot([0.0],[0.0],legend=false,grid=true)
    size_matrix = size(given_matrix)
    #plot!([100], [100],legend=false)
    for i in 1:size_matrix[1]
       for j in 1:size_matrix[2]
           theta = given_matrix[i,j].alpha
           mag = given_matrix[i,j].mod_grad
           println("i=",i ," j=",j," theta=", theta*180/pi)
           quiver!([i-1],[j-1],quiver=([mag*cos(theta)],[mag*sin(theta)]), color="blue")
       end
   end
   display(p)
end

function find_path_from_given_point_old(x,y,dx,dy,grad_info_matrix)

    curr_x = x
    curr_y = y
    path_x = Float64[curr_x]
    path_y = Float64[curr_y]
    while( !is_within_range_check_with_points(curr_x,curr_y,10,5,1.0) )
        matrix_x_index = convert(Int,floor(curr_x/dx))
        matrix_y_index = convert(Int,floor(curr_y/dy))
        println("Matrix indices are : ", matrix_x_index, " ", matrix_y_index)
        grad_info = grad_info_matrix[matrix_x_index,matrix_y_index]
        mag = grad_info.mod_grad
        #mag = 0.1
        new_x = curr_x + mag*cos(grad_info.alpha)
        new_y = curr_y + mag*sin(grad_info.alpha)
        println(new_x," ",new_y)
        curr_x, curr_y = new_x,new_y
        push!(path_x,curr_x)
        push!(path_y,curr_y)
    end
    return path_x,path_y
end

function check_if_grid_intersects_with_circular_obstacle_using_corner_points(blc_x,blc_y,trc_x,trc_y,obstacle, obstacle_padding)
     nearest_point_x = max(blc_x, min(obstacle.x,trc_x))
     nearest_point_y = max(blc_y, min(obstacle.y,trc_y))

     dist_to_center_of_circle = sqrt( (nearest_point_x-obstacle.x)^2 + (nearest_point_y-obstacle.y)^2 )
     if(dist_to_center_of_circle > obstacle.r + obstacle_padding)
         return false
     else
         return true
     end
end

function check_if_grid_intersects_with_circular_obstacle_using_grid_center(grid_center_x, grid_center_y, grid_width_x, grid_width_y, obstacle,obstacle_padding)
    abs_dist_x = abs(obstacle.x - grid_center_x)
    abs_dist_y = abs(obstacle.y - grid_center_y)

    if(abs_dist_x > (grid_width_x/2)+obstacle.r+obstacle_padding)
        return false
    end
    if(abs_dist_y > (grid_width_y/2)+obstacle.r+obstacle_padding)
        return false
    end

    if(abs_dist_x <= grid_width_x/2)
        return true
    end
    if(abs_dist_y <= grid_width_y/2)
        return true
    end

    corner_dist_sq = (abs_dist_x - (grid_width_x/2))^2 + (abs_dist_y - (grid_width_y/2))^2

    if(corner_dist_sq <= (obstacle_padding+obstacle.r)^2)
        return true
    else
        return false
    end
end

function generate_slowness_map_from_given_environment(world, obstacle_padding=2.0)
    #=
    Note - In the environment, left bottom corner is (0,0) and the right top corner is (100,100).
    In the slowness map, (1,1) is left top corner while (100,100) is the right bottom corner.
    So, the indices of slowness map/matrix can be assumed to be a 90 degress clockwise rotated form of the environment.
    That's why we need to handle these indices very properly.
    =#

    horizontal_discretization_for_slowness_map = 0.1
    vertical_discretization_for_slowness_map = 0.1
    vertical_grid_size = convert(Int,(world.breadth/vertical_discretization_for_slowness_map)) + 1          #number of rows
    horizontal_grid_size = convert(Int,(world.length/horizontal_discretization_for_slowness_map)) + 1       #number of columns

    #Generate a slowness map assuming there are no static obstacles.
    k = ones(vertical_grid_size, horizontal_grid_size)
    slowness_distance_threshold_around_obstacles = 0.0
    #Put static obstacles on the slowness map
    for i in vertical_grid_size:-1:1
        for j in 1:horizontal_grid_size
            # point_corresponding_to_bottom_left_corner_of_the_grid_in_env_x = (j-1)*horizontal_discretization_for_slowness_map
            # point_corresponding_to_bottom_left_corner_of_the_grid_in_env_y = (vertical_grid_size - i)*vertical_discretization_for_slowness_map
            # point_corresponding_to_top_right_corner_of_the_grid_in_env_x = j*horizontal_discretization_for_slowness_map
            # point_corresponding_to_top_right_corner_of_the_grid_in_env_y = (vertical_grid_size - i+1)*vertical_discretization_for_slowness_map
            grid_center_x = horizontal_discretization_for_slowness_map * ( ( (j-1) + j ) / 2 )
            grid_center_y = vertical_discretization_for_slowness_map * (  ( (vertical_grid_size - i) + (vertical_grid_size - i + 1) ) / 2 )
            collision_with_obstacle_flag = false
            slow_speed_flag = false
            for obstacle in world.obstacles
                slow_speed_flag = check_if_grid_intersects_with_circular_obstacle_using_grid_center(grid_center_x, grid_center_y,
                                                    horizontal_discretization_for_slowness_map, vertical_discretization_for_slowness_map,
                                                    obstacle, obstacle_padding+world.cart.L+slowness_distance_threshold_around_obstacles)
                collision_with_obstacle_flag = check_if_grid_intersects_with_circular_obstacle_using_grid_center(grid_center_x, grid_center_y,
                                                    horizontal_discretization_for_slowness_map, vertical_discretization_for_slowness_map,
                                                    obstacle, obstacle_padding+world.cart.L)
                if(slow_speed_flag==true)
                    k[i,j] = 0.5
                end
                if(collision_with_obstacle_flag==true)
                    k[i,j] = Inf
                    break
                end
            end
        end
    end
    return k
end

function convert_given_env_point_to_matrix_indices(x_env,y_env,env_length,env_breadth,dx=0.1,dy=0.1)
    matrix_index_j = convert(Int, floor(x_env/dx))
    num_grids_in_y_direction = convert(Int, env.breadth/dy)
    matrix_index_i = num_grids_in_y_direction - convert(Int, floor(y_env/dy))
    return matrix_index_i,matrix_index_j
end

function convert_given_matrix_index_to_env_point(matrix_index_i,matrix_index_j,env_length,env_breadth,dx=0.1,dy=0.1)
    x_env = matrix_index_j*dx
    num_grids_in_y_direction = convert(Int, env.breadth/dy)
    y_env = (num_grids_in_y_direction - matrix_index_i)*dy
    return x_env,y_env
end

function find_path_hack(x,y,dx,dy,goal_x,goal_y,t)
    curr_x = x
    curr_y = y
    mat_index_x,mat_index_y = convert_given_env_point_to_matrix_indices(curr_x,curr_y,100,100)
    #mat_index_x,mat_index_y = mat_index_x+1,mat_index_y+1
    path_x = Float64[mat_index_x]
    path_y = Float64[mat_index_y]
    # new_t = t[2:end-1,2:end-1]
    # println("\nCart's position is " ,curr_x," ",curr_y)
    try
        while( !is_within_range_check_with_points(curr_x,curr_y,goal_x,goal_y,1.0) )
            println("Time Matrix indices are : ", mat_index_x, " ", mat_index_y)
            # neighbourhood = t[mat_index_x:mat_index_x+2, mat_index_y:mat_index_y+2]
            curr_min = Inf
            min_i = 0
            min_j = 0
            for i in mat_index_x-1:mat_index_x+1
                for j in mat_index_y-1:mat_index_y+1
                    if(t[i,j] < curr_min)
                        curr_min = t[i,j]
                        min_i = i
                        min_j = j
                    end
                end
            end
            # println("Gradient's direction is : ", grad_info.alpha*180/pi)
            # println("Gradient's magnitude is : ", grad_info.mod_grad)
            # mag = grad_info.mod_grad
            #mag = 0.1
            new_x, new_y = convert_given_matrix_index_to_env_point(min_i-1,min_j-1,100,100)
            println("\nCart's position is " ,new_x," ",new_y)
            curr_x, curr_y = new_x,new_y
            # mat_index_x,mat_index_y = convert_given_env_point_to_matrix_indices(curr_x,curr_y,100,100)
            # mat_index_x,mat_index_y = mat_index_x+1,mat_index_y+1
            mat_index_x,mat_index_y = min_i,min_j
            push!(path_x , mat_index_x)
            push!(path_y , mat_index_y)
        end
    catch e
        println("Encountered an error")
        #println(e)
        return path_x,path_y
    end
    return path_x,path_y
end


#=
k = generate_slowness_map_from_given_environment(env,2.0)
dx = [0.1,0.1]
source = CartesianIndex(250,1000)
t = solve_eikonal_equation_on_given_map(k, dx, source)
g = calculate_gradients(t)
display_fmm_path_from_given_vertex(env,31.6,75,g)
=#

#=
p = plot([0.0],[0.0],legend=false,grid=true)
xi = 85
yi = 75
dx=0.1
dy=0.1
theta = e[xi,yi].alpha
mag = e[xi,yi].mod_grad
quiver!([xi*dx],[yi*dy],quiver=([mag*cos(theta)],[mag*sin(theta)]), color="blue")
=#

# k = ones(100,100) #slowness squared
# for i in 50:100
#     for j in 1:50
#         k[i,j] = Inf
#     end
# end
# dx = [1.0, 1.0] # grid spacing
#
# l = rotr90(rotr90(k))
# x0 = CartesianIndex(75,100)
# (t, ordering) = fefmm(l,dx,x0)
#
# x1 = CartesianIndex(100,75)
# (t2, ordering) = fefmm(k,dx,x1)
#
#
# x_kernel = [ [-1, -1 ,-1] [ 0, 0,0] [1, 1,1] ]
# y_kernel =[ [ 1, 0, -1] [1 ,0, -1] [1,0,-1] ]
#
#
# k = ones(1001,1001) #slowness squared
# dx = [0.1, 0.1]
# x0 = CartesianIndex(1001,251)
# (t, ordering) = fefmm(k,dx,x0)
# padded_t = zeros(1003,1003)
# padded_t[2:1002,2:1002] = t
# grad_info_matrix = Array{gradient_info_struct,2}(undef,1001,1001)
# for i in 2:1002
#     for j in 2:1002
#         grad_x = dot(x_kernel, padded_t[i-1:i+1,j-1:j+1])
#         grad_y = dot(y_kernel, padded_t[i-1:i+1,j-1:j+1])
#         mod_grad = sqrt( (grad_x)^2 + (grad_y)^2 )
#         alpha = atan(grad_y/grad_x)
#         grad_info_matrix[i-1,j-1] = gradient_info_struct(grad_x,grad_y,mod_grad,alpha)
#     end
# end
#
#
# k = ones(12,12) #slowness squared
# dx = [1.0, 1.0]
# x0 = CartesianIndex(6,11)
# (t, ordering) = fefmm(k,dx,x0)
# grad_info_matrix = Array{gradient_info_struct,2}(undef,10,10)
# for i in 2:11
#     for j in 2:11
#         grad_x = dot(x_kernel, t[i-1:i+1,j-1:j+1])
#         grad_y = dot(y_kernel, t[i-1:i+1,j-1:j+1])
#         mod_grad = sqrt( (grad_x)^2 + (grad_y)^2 )
#         alpha = wrap_between_0_and_2Pi(atan(grad_y/grad_x))
#         grad_info_matrix[i-1,j-1] = gradient_info_struct(grad_x,grad_y,mod_grad,alpha)
#     end
# end

#=
k = ones(1000,1000)
dx = [0.1,0.1]
source = CartesianIndex(250,1000)
for i in 500:1000
    for j in 500:1000
        k[i,j] = Inf
    end
end
t = solve_eikonal_equation_on_given_map(k, dx, source)
g = calculate_gradients(t)
a,b = find_path_from_given_point(40,30,0.1,0.1,100,75,g)
=#

#=
for i in 1:10
   for j in 1:10
       print(e[i,j].alpha*180/pi, "\t")
   end
   println()
end

des_mat = g;
for i in 1:size(des_mat)[1]
   for j in 1:size(des_mat)[2]
       print(des_mat[i,j].alpha*180/pi, "\t")
   end
   println()
end

for i in 1:10
   for j in 1:10
       print(grad_info_matrix[i,j].alpha*180/pi, "\t")
   end
   println()
end

=#

#=
a,b = find_path_hack(70,30,0.1,0.1,100,75,t[2:end-1, 2:end-1])
x_points = []
y_points = []
for i in 1:length(a)
   x,y = convert_given_matrix_index_to_env_point(a[i],b[i],100,100)
   push!(x_points,x)
   push!(y_points,y)
end
display_fmm_path_from_given_point(env,x_points,y_points,g)
=#
