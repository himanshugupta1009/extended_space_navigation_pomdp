using Random, Distributions

mutable struct cart_state_particle
    x::Float64
    y::Float64
    theta::Float64
end

function find_distance_between_points(x1,y1,x2,y2)
    dist = ((x1 - x2)^2 + (y1 - y2)^2)^0.5
    return dist
end

function get_measurement(world,MN_mixand_weight,MN_mixand_mean,MN_mixand_covariance)
    true_meas = find_distance_between_points(world.cart.x,world.cart.y,world.cart.goal.x,world.cart.goal.y)
    index = rand(SparseCat([1,2,3,4,5],MN_mixand_weight))
    mn_nd = Normal(MN_mixand_mean[index],MN_mixand_covariance[index])
    mn = rand(mn_nd)
    return true_meas+mn
end

function get_initial_particles_and_weights(num_particles,world)
    #num_particles = 100
    particles = cart_state_particle[]
    weights = Float64[]
    for i in 1:num_particles
        x = world.cart.x
        x += rand(Distributions.Uniform(-2,2))
        x = clamp(x,0,world.length)
        y = world.cart.y
        y += rand(Distributions.Uniform(-2,2))
        y = clamp(y,0,world.breadth)
        theta = world.cart.x
        theta += rand(Distributions.Uniform(-pi/36,pi/36))
        theta = wrap_between_0_and_2Pi(theta)
        theta = world.cart.theta
        push!(particles, cart_state_particle(x,y,theta))
        push!(weights,1/num_particles)
    end
    return particles,weights
end

function propogate_individual_particle(current_cart_position, new_cart_velocity, starting_index, world)
    current_x, current_y, current_theta = current_cart_position.x, current_cart_position.y, current_cart_position.theta
    length_hybrid_a_star_path = length(world.cart_hybrid_astar_path)
    cart_path = Tuple{Float64,Float64,Float64}[ (Float64(current_x), Float64(current_y), Float64(current_theta)) ]
    if(new_cart_velocity == 0.0)
        return cart_path
    else
        time_interval = 1.0/new_cart_velocity
        for i in (1:new_cart_velocity)
            #@show(starting_index, length(world.cart_hybrid_astar_path))
            steering_angle = world.cart_hybrid_astar_path[starting_index]
            if(steering_angle == 0.0)
                new_theta = current_theta
                new_x = current_x + new_cart_velocity*cos(current_theta)*time_interval
                new_y = current_y + new_cart_velocity*sin(current_theta)*time_interval
            else
                new_theta = current_theta + (new_cart_velocity * tan(steering_angle) * time_interval / world.cart.L)
                new_theta = wrap_between_0_and_2Pi(new_theta)
                new_x = current_x + ((world.cart.L / tan(steering_angle)) * (sin(new_theta) - sin(current_theta)))
                new_y = current_y + ((world.cart.L / tan(steering_angle)) * (cos(current_theta) - cos(new_theta)))
            end
            push!(cart_path,(Float64(new_x), Float64(new_y), Float64(new_theta)))
            current_x, current_y,current_theta = new_x,new_y,new_theta
            starting_index = starting_index + 1
            if(starting_index>length_hybrid_a_star_path)
                break
            end
        end
    end
    return cart_path
end

function propogate_particles(particles, world, new_cart_velocity,PN_mixand_weight,PN_mixand_mean,PN_mixand_covariance)
    new_particles = Array{cart_state_particle,1}()
    for particle in particles
        particle_path::Vector{Tuple{Float64,Float64,Float64}} = propogate_individual_particle(particle,
                                                                                    new_cart_velocity, 1, world)
        new_particle_position = particle_path[end]
        index = rand(SparseCat([1,2,3,4,5],PN_mixand_weight))
        pn_nd = MvNormal(PN_mixand_mean[index],PN_mixand_covariance[index])
        sampled_pn = rand(pn_nd)
        #new_particle = cart_state_particle(new_particle_position[1],new_particle_position[2],new_particle_position[3])
        new_particle = cart_state_particle(new_particle_position[1]+sampled_pn[1],
                                        new_particle_position[2]+sampled_pn[2],new_particle_position[3])
        push!(new_particles,new_particle)
    end
    return new_particles
end

function get_weights_for_new_particles(new_particles, old_weights, yk_observed, mn_covar, world)
    new_weights = Float64[]
    for index in 1:length(new_particles)
        particle = new_particles[index]
        yk_predicted = find_distance_between_points(particle.x,particle.y,world.cart.goal.x,world.cart.goal.y)
        dist = Normal(yk_predicted, mn_covar)
        new_weight = pdf(dist,yk_observed)*old_weights[index]
        push!(new_weights,new_weight)
    end
    new_weights = new_weights/sum(new_weights)
    return new_weights
end

function calculate_NEES(particle_weights)
    Ns = length(particle_weights)
    m = mean(particle_weights)
    v = var(particle_weights)
    cv = v/(m*m)
    NESS = Ns/(1+cv)
    return NESS
end

function get_mean_covar_from_particles(current_particles,current_weights)
    mean_x,mean_y,mean_theta = 0.0,0.0,0.0
    for i in 1:length(current_particles)
        mean_x += current_weights[i]*current_particles[i].x
        mean_y += current_weights[i]*current_particles[i].y
        mean_theta += current_weights[i]*current_particles[i].theta
    end
    mean_vec = [mean_x,mean_y,mean_theta]
    covar = [0 0 0; 0 0 0; 0 0 0]
    for i in 1:length(current_particles)
        diff_from_mmse = [current_particles[i].x,current_particles[i].y,current_particles[i].theta] - mean_vec
        covar = covar +  current_weights[i]*( diff_from_mmse*transpose(diff_from_mmse))
    end
    return mean_vec,covar
end

function resample_particles(current_particles, current_weights)
    new_particles = cart_state_particle[]
    new_weights = Float64[]
    for i in 1:length(current_particles)
        new_particle = rand(SparseCat(current_particles,current_weights))
        push!(new_particles,new_particle)
        push!(new_weights,1/length(current_particles))
    end
    return new_particles,new_weights
end
