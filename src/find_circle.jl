using Debugger

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

# initial_state = [10.0,10.0,0.0]
# extra_parameters = [5.0, 1.0, pi/12]
# x,y,theta = get_intermediate_points(initial_state, 1.0, extra_parameters);
# cx,cy,cr = find_center_and_radius(x[1],y[1],x[11],y[11],x[6],y[6])

#Given a circle's center and radius and a line segment, find if they intersect
function find_if_circle_and_line_segment_intersect(cx::Float64,cy::Float64,cr::Float64,
                                    ex::Float64,ey::Float64,lx::Float64,ly::Float64)
    dx = lx-ex
    dy = ly-ey
    fx = ex-cx
    fy = ey-cy

    #Quadratic equation is  t^2 ( d · d ) + 2t ( d · f ) +  (f · f - r^2) = 0
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

#Given three colinear points p, q, r, the function checks if point q lies on line segment 'pr'
function onSegment(px::Float64, py::Float64, qx::Float64, qy::Float64, rx::Float64, ry::Float64)
    if (qx <= max(px, rx) && qx >= min(px, rx) && qy <= max(py, ry) && qy >= min(py, ry))
       return true;
   end
    return false;
end

#Given three points p, q, r, the function finds the orientation of the triplet (p,q,r)
#Return 0 if p, q and r are colinear; Return 1 if Clockwise and Return 2 if Counterclockwise
function orientation(px::Float64, py::Float64, qx::Float64, qy::Float64, rx::Float64, ry::Float64)
    val = ( (qy-py)*(rx-qx) ) - ( (qx-px)*(ry-qy) )
    if (val == 0)  #collinear
        return 0;
    elseif val>0
        return 1; #clockwise
    else
        return 2; #counter clockwise
    end
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
