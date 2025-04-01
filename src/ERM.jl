include("util.jl")
#include("variational.jl")
#ENV["JULIA_PROJECT"] = "/home/wenlab-user/wangzezhen/FISHRNN_TEST"
#ENV["JULIA_PROJECT"] = ""

pyplot()
N = 2^10#number of neurons
ξ = 10^18
d = 2 #dimensionality
ρ = 1024#1024
L = (N/ρ)^(1/d)
δ = 0.1
ϵ = 0.03125
μ = 0.5

different_neural_variability = true
if different_neural_variability
    σ² = vec(rand(LogNormal(0,0.5),N,1))
    σ² = σ²/mean(σ²)
else
    σ² = ones(N,1)
end

#σ² = σ²/mean(σ²) #normalize so that σ̄² = 1
σ = vec(broadcast(√,σ²))
Δ = diagm(σ)

p = ERMParameter(;N = N, L = L, ρ = ρ, n = d, ϵ = ϵ, μ = μ, ξ = ξ, β = 0, σ̄² = mean(σ²), σ̄⁴ = mean(σ².^2))

#=
N number of neurons
L box size
ρ density
n dimensionality
ϵ parameter in distance function f
μ parameter in distance function f
ξ parameter in distance function f
β = 1/ξ parameter in distance function f
σ̄² mean variance of neural activity
σ̄⁴ mean 4th moment of neural activity
#parameters in the ERM model
=#

points = rand(Uniform(-L/2,L/2),N,d) #points are uniformly distributed in a region 
C = reconstruct_covariance(points, p, subsample = false)
Ĉ = Δ*C*Δ #covariance matrix
Ĉ = (Ĉ+Ĉ)/2

#subsampling(C, "C:\\Users\\wzz\\Documents\\GitHubold\\RNN\\figures")

λ_sim, p_sim, λ_theory, p_theory = eigendensity(Ĉ, p, correction = false, λ_min = 0.1)
plot(λ_sim[1:end], p_sim[1:end], label = "simulation")
plot!(λ_theory, p_theory, label = "theory_no_correction")


plot!(xlabel = L"\lambda", ylabel = "pdf", xaxis = :log, yaxis = :log)
plot!(title=L"\rho = %$(round(p.ρ,digits=3)), \mu = %$(p.μ), \epsilon = %$(round(p.ϵ,digits=3)), n =  %$(p.n), \beta = %$(round(p.β,digits=3)), E(\sigma^4) = %$(round(p.σ̄⁴,digits=2))")


#λ_sim, p_sim, λ_theory_correction, p_theory_correction = eigendensity(Ĉ, p, correction = true, λ_min = 0.5)
#plot!(λ_theory_correction, p_theory_correction, label = "theory_2rd_correction")

λ_sim, p_sim, λ_theory_GV, p_theory_GV = eigendensity(Ĉ, p; length = 20, use_variation_model = true, λ_min = 1, λ_max = 20)
index, dlnpdlnρ, λ_index = collapse_index_theory(p, λ_max = 5)
println(index)
plot!(λ_theory_GV[1:end], p_theory_GV[1:end], label = "variation, N = $N")



