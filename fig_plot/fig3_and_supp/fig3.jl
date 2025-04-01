using Pkg
# Switch to the desired Julia environment
Pkg.activate("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/")


include("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/src/util.jl") # ZZ's code
include("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/src/util2.jl")
include("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/src/subsampling_functions.jl") 

fig_dir = joinpath(FIGURE_ROOT, "fig3_and_supp")
mkpath(fig_dir)

N = 2^10 #number of neurons
d = 2 #dimensionality
L = 10 #(N/ρ)^(1/d) #box size
ξ = 10^18
δ = 0.1
ϵ = 0.03125
μ = 0.5
β = 0
ρ = N/L^d
p = ERMParameter(;N = N, L = L, ρ = ρ, n = d, ϵ = ϵ, μ = μ, ξ = ξ, β = β, σ̄² = mean(σ²), σ̄⁴ = mean(σ².^2))
points = rand(Uniform(-L/2,L/2),N,d) #points are uniformly distributed in a region
D = pairwise(Euclidean(), points, dims=1) 
C = reconstruct_covariance(points, p, subsample = false)


c_max = 0.7
using Plots
R = hclust(D,linkage = :average)
cbar = :thermal
theme(:default)
colorbar_ticks = range(0, stop=c_max, length=2)  # Adjust the number of ticks as needed
p1 = heatmap(C[R.order, R.order],
    clim = (0,c_max),
    ratio=:equal,
    framestyle = :none,
    color=cgrad(cbar),
    colorbar = true,
    size = (320, 300),
    dpi = 500,
    legend = :none,
    colorbar_ticks = nothing,  # Use the custom ticks for the colorbar
    )
# Customize plot attributes
plot!(p1,
# ytickfont = font(20),   # Customize y-axis tick font size
)
savefig(p1, joinpath(fig_dir, "fig3B_ERM_corr.png"))


import PyPlot as plt
uppercase_letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
fig = plt.figure(figsize=(6.85,4),facecolor="w",frameon=true, dpi=300)
plt.rc("font",size=8,family="Arial")
# plt.rc("font",size=6,family="Arial")
plt.rc("font", serif="Helvetica")
plt.rc("pdf", fonttype=42)
yratio = 6.85/4 

n_sim = 100

ipanel = 2

#################################
#----ERM rank plot and pdf-----##
#################################
x0 = .53 # x position 
y00 = [.6, .05] # y position
xs = 0.25 # length of x axis
ys = xs * yratio # length of y axis
ipanel+=1
ax1 = fig.add_axes([x0, y00[1], xs,ys])

ipanel+=1
ax2 = fig.add_axes([x0, y00[2], xs,ys])



color_map = plt.get_cmap("inferno")

subsampling_multi_sim_subplot(ax1,ax2,p,n_sim)
ax1.text(-0.2, 1.05, uppercase_letters[ipanel-1],transform=ax1.transAxes,size=14,weight="bold")
ax1.legend(borderpad=0.5,handlelength=0.8,frameon=false,fontsize=8,loc="upper right")

ax2.text(-0.2, 1.05, uppercase_letters[ipanel],transform=ax2.transAxes,size=14,weight="bold")
ax2.legend(borderpad=0.5,handlelength=0.8,frameon=false,fontsize=8,loc="lower left")

fig.savefig(joinpath(fig_dir, "fig3CD.pdf"), bbox_inches = "tight")

