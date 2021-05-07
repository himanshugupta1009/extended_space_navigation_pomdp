using Random, Distributions
using Debugger
using LinearAlgebra

A = [0 1;0 0]
B = [0;1]
C = [1 0]
Tau = transpose([0 1])
W = [1]
V = [1]
delta_t = 0.1
total_num_steps = 150

A_hat = [ A B ; 0 0 0 ]
A_hat_exp = exp(A_hat*delta_t)
F = A_hat_exp[1:2,1:2]
G = A_hat_exp[1:2,3]
H = C

Z1 = [-A Tau*W*transpose(Tau)]
Z2 = [ 0 0; 0 0]
Z2 = [Z2 transpose(A)]
Z = [Z1; Z2]
Z_exp = exp(Z*delta_t)
Q = transpose(Z_exp[3:4,3:4])*Z_exp[1:2,3:4]
R = V/delta_t

mn_nd = Normal(0.0,R[1])
pn_nd = MvNormal([0;0],[1 0; 0 1])

function simulate_truth_data()
    is = [0.0 ;0.0]
    ic = [ 10 0; 0 10]
    im = H*is
    ins = rand(mn_nd)
    im = im .+ ins

    tsd = [is]
    tmd = [im[1]]
    tpnd = [ [0.0;0.0] ]
    tmnd = [ ins ]

    cs = is
    for i in (1:total_num_steps)
        #Control Input
        u = 2*cos(0.75*i*delta_t)
        #Get new state
        ns = F*(cs) + G*u
        chol_decom = cholesky(Q).L
        sampled_pn = rand(pn_nd)
        sampled_pn = chol_decom*sampled_pn
        ns = ns+sampled_pn
        #Get corresponding measurement
        nm = H*ns
        sampled_mn = rand(mn_nd)
        nm = nm .+ sampled_mn
        push!(tsd,ns)
        push!(tmd,nm[1])
        push!(tpnd,sampled_pn)
        push!(tmnd,sampled_mn)
        cs = ns
    end
    return tsd,tmd,tpnd,tmnd
end
