using Plots
function ellipse(h,k,a,b)
    θ = LinRange(0,2*pi,500)
    h.+ a*cos.(θ), k.+ b*sin.(θ)
end

plot(ellipse(0,0,10,5))
