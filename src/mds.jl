include("util.jl")

if !@isdefined ifish
    ifish = 210106
end

if !@isdefined n
    n = 2
end

if !@isdefined root
    root = "ERM-scale/results/mds"
end



if !isdir(joinpath(root))
    mkdir(joinpath(root)) 
end

if !isdir(joinpath(root,"$ifish/$n"))
    if !isdir(joinpath(root,"$ifish"))
        mkdir(joinpath(root,"$ifish"))
    end
    mkdir(joinpath(root,"$ifish/$n"))
    mkdir(joinpath(root,"$ifish/$n/h"))
    mkdir(joinpath(root,"$ifish/$n/cov"))
    mkdir(joinpath(root,"$ifish/$n/corr"))
    mkdir(joinpath(root,"$ifish/$n/cov/data"))
    mkdir(joinpath(root,"$ifish/$n/corr/data"))
end


include("../load_data/load_fish_data2.jl")
#subsampling(Corr, "/home/wenlab-user/wangzezhen/FISHRNN_TEST/results")
fit = Distributions.fit

#p = ERMParameter(;N = 1024, L = 10, ρ = 1024/10^2, n = 2, ϵ = 0.03125, μ = 0.5, ξ = 10, σ̄² = 1.134, σ̄⁴ = 1.653)

d = n
ρ, μ = Parameter_estimation(Corr, n = n)
N = size(Corr)[1]
L = (N/ρ)^(1/n)
p = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = 0.03125, μ = μ, ξ = 10^18, β = 0, σ̄² = 1, σ̄⁴ = 1)

ϵ = 0.03125
println("(ϵ/L)^d = ",(ϵ/L)^d," d=",d)

C = Corr#[id,id]
#C = max.(C,10^-9)
C = abs.(C)
D = copy(C)


function corr(D,p)  
    @unpack μ, ϵ, β = p 
    C_orr = copy(D)
    C_orr = ϵ^μ*(D.^2 .+ϵ.^2).^(-μ/2)
    return C_orr
end


function find_D(C,p)
    @unpack μ, ϵ, β, L = p 
    β2 = 1/L
    L = L
    fc(D,C) = ϵ^μ*(D).^(-μ).*exp.(-(D.- ϵ)*β) - C
    f(d) = ϵ^μ*(d).^(-μ).*exp.(-(d.- ϵ)*β)
    N = size(C,1)
    fcd = ZeroProblem(fc,ϵ)
    #a = solve(fcd,c)
    #d = find_zero(fc,ϵ)
    [D[i,j] = solve(fcd,C[i,j]) for i = 1:N for j = 1:N]
    D[D.>L] .= -log.(C[D.>L]/f(L))./β2 .+L
    return D
end


function find_D(C,p)
    @unpack μ, ϵ, β, L = p 
    L = L
    D .= ϵ*sqrt.(abs.(C.^(-2/μ) .- 1))
    D[D.>L] .= L*log.((D[D.>L] ./L)) .+L
    return D
end







D = find_D(C, p)
D = D - Diagonal(D)
W = ones(N,N)
W[Corr .< 0] .= 0.01
X, stress = mdscale(D,n = n, criterion = "sammon")
matwrite(joinpath(root,"$ifish/$n/X.mat"),Dict("X"=>X))

println(stress)
if n == 1.0
    D2 = pairwise(Euclidean(), X')
else
    D2 = pairwise(Euclidean(), X', dims=1)
end
C2 = corr(D2,p) 

λ_sim, p_sim = eigendensity(C2, correction = false, λ_min = 0.5)
λ_id = findall(λ_sim.> 0.1 )
plot(λ_sim[λ_id], p_sim[λ_id], label = "$ifish _refactoring")
plot!(xlabel = L"\lambda", ylabel = "pdf", xaxis = :log, yaxis = :log)

λ_sim, p_sim = eigendensity(Corr, correction = false, λ_min = 0.5)
plot!(λ_sim[1:end], p_sim[1:end], label = "$ifish")
plot!(title=L"L = %$(round(p.L,digits=3)), \mu = %$(round(p.μ ,digits=3)), \epsilon = %$(round(p.ϵ,digits=3)), n =  %$(p.n), \beta = %$(round(p.β,digits=3)), E(\sigma^4) = %$(round(p.σ̄⁴,digits=2))")


savefig(joinpath(root,"$ifish/$n/corr/lambda_density.png"))
savefig(joinpath(root,"$ifish/$n/corr/lambda_density.svg"))

C3 = std_c * C2 * std_c

#subsampling(C2, joinpath(root,"$ifish/$n/corr"))
#subsampling(C3, joinpath(root,"$ifish/$n/cov"))
#subsampling(Cᵣ, joinpath(root,"$ifish/$n/cov/data"))
#subsampling(Corr, joinpath(root,"$ifish/$n/corr/data"))

R = hclust(exp.(-100*Corr),linkage = :average)
R = hclust(D,linkage = :average)
heatmap(Corr[R.order, R.order],theme = :dark, clim = (0,1))
savefig(joinpath(root,"$ifish/$n/corr/corr.png"))

heatmap(C2[R.order, R.order],theme = :dark, clim = (0,1))
savefig(joinpath(root,"$ifish/$n/corr/corr_refactoring.png"))

scatter(X[1,R.order],X[2,R.order], marker_z = 1:size(X,2), markersize = 2,  color = :jet)
savefig(joinpath(root,"$ifish/$n/corr/point_cloud.png"))

marginalkde(X[1,R.order],X[2,R.order])
marginalkde(X[1,:],X[2,:])
savefig(joinpath(root,"$ifish/$n/corr/marginalkde.png"))

heatmap(r[R.order,:],clim = (0,1))
savefig(joinpath(root,"$ifish/$n/corr/neuron_activate.png"))

