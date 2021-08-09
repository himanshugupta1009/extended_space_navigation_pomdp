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
    return t
end

function calculate_gradients(padded_time_matrix)

    x_kernel = [ [-1, -1 ,-1] [ 0, 0, 0] [1, 1, 1] ]
    y_kernel = [ [ 1, 0, -1] [1 ,0, -1] [1, 0, -1] ]

    size_pm = size(padded_time_matrix)
    grad_info_matrix = Array{gradient_info_struct,2}(undef,size_pm[1]-2,size_pm[2]-2)

    for i in 2:size_pm[1]-1
        for j in 2:size_pm[2]-1
            grad_x = dot(x_kernel, padded_time_matrix[i-1:i+1,j-1:j+1])
            grad_y = dot(y_kernel, padded_time_matrix[i-1:i+1,j-1:j+1])
            mod_grad = sqrt( (grad_x)^2 + (grad_y)^2 )
            alpha = atan(grad_y/grad_x)
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

function find_path_from_given_point(x,y,dx,dy,goal_x,goal_y,grad_info_matrix)

    curr_x = x
    curr_y = y
    path_x = Float64[curr_x]
    path_y = Float64[curr_y]
    grad_mat_size = size(grad_info_matrix)
    println("\nCart's position is " ,curr_x," ",curr_y)
    while( !is_within_range_check_with_points(curr_x,curr_y,goal_x,goal_y,1.0) )
        env_x_index = convert(Int,floor(curr_x/dx))
        env_y_index = convert(Int,floor(curr_y/dy))
        mat_hor_index = grad_mat_size[1] - env_y_index + 1
        mat_ver_index = env_x_index
        println("Matrix indices are : ", mat_hor_index, " ", mat_ver_index)
        grad_info = grad_info_matrix[mat_hor_index,mat_ver_index]
        println("Gradient's direction is : ", grad_info.alpha*180/pi)
        println("Gradient's magnitude is : ", grad_info.mod_grad)
        mag = grad_info.mod_grad
        #mag = 0.1
        new_x = curr_x + mag*cos(grad_info.alpha)
        new_y = curr_y + mag*sin(grad_info.alpha)
        println("\nCart's position is " ,new_x," ",new_y)
        curr_x, curr_y = new_x,new_y
        push!(path_x,curr_x)
        push!(path_y,curr_y)
    end
    return path_x,path_y
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

function check_if_grid_intersects_with_circular_obstacle(blc_x,blc_y,trc_x,trc_y,obstacle, obstacle_padding)
     nearest_point_x = max(blc_x, min(obstacle.x,trc_x))
     nearest_point_y = max(blc_y, min(obstacle.y,trc_y))

     dist_to_center_of_circle = sqrt( (nearest_point_x-obstacle.x)^2 + (nearest_point_y-obstacle.y)^2 )
     if(dist_to_center_of_circle > obstacle.r + obstacle_padding)
         return false
     else
         return true
     end
end

function lala(grid_center_x, grid_center_y, grid_width_x, grid_width_y, obstacle,obstacle_padding)
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
    vertical_grid_size = (world.breadth/vertical_discretization_for_slowness_map)       #number of rows
    horizontal_grid_size = (world.length/horizontal_discretization_for_slowness_map     #number of columns

    #Generate a slowness map assuming there are no static obstacles.
    k = ones(vertical_grid_size, horizontal_grid_size)

    #Put static obstacles on the slowness map
    for i in vertical_grid_size:-1:1
        for j in 1:horizontal_grid_size
            point_corresponding_to_bottom_left_corner_of_the_grid_in_env_x = (j-1)*horizontal_discretization_for_slowness_map
            point_corresponding_to_bottom_left_corner_of_the_grid_in_env_y = (vertical_grid_size - i)*vertical_discretization_for_slowness_map
            point_corresponding_to_top_right_corner_of_the_grid_in_env_x = j*horizontal_discretization_for_slowness_map
            point_corresponding_to_top_right_corner_of_the_grid_in_env_y = (vertical_grid_size - i+1)*vertical_discretization_for_slowness_map
            collision_with_obstacle_flag = false
            for obstacle in world.obstacles
                collision_with_obstacle_flag = check_if_grid_intersects_with_circular_obstacle(obstacle, obstacle_padding)
                if(collision_with_obstacle_flag==true)
                    k[i,j] = Inf
                    break
                end
            end
            #Check if this cell collides with any static obstacle in the environment
            #If yes, set k[i,j] = Inf , else continue

    for obstacle in world.obstacles




end
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

function
