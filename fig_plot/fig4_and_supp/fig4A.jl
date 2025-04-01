"""
load CI file and plot ERM rank plot for power law f(x)
The CI file is located at figure_script/fig4_and_supp/CI_ERM.txt


Note: parameters in the ERM model

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

"""


using Pkg
# Switch to the desired Julia environment
Pkg.activate("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/")


include("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/src/util.jl") # ZZ's code
include("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/src/util2.jl")

fig4_dir = joinpath(FIGURE_ROOT, "fig4_and_supp")
mkpath(fig4_dir)
import PyPlot as plt

uppercase_letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
fig = plt.figure(figsize=(6.85,5),facecolor="w",frameon=true, dpi=300)
# plt.rc("font",size=6,family="sans-serif")
plt.rc("font",size=6,family="Arial")
# plt.rc("font", serif="Helvetica")
plt.rc("pdf", fonttype=42)
yratio = 6.85/5


########################################################
##---------ERM rank plot for power law f(x)--------##
########################################################

ipanel = 0

f_r = f_r_t_pdf
N = 2^11 #number of neurons
# d = 2 #dimensionality
ρ = 256
ξ = 10^18
δ = 0.1
ϵ = 0.03125
# μ = 0.5
β = 0

different_neural_variability = false
if different_neural_variability
    σ² = vec(rand(LogNormal(0,0.5),N,1))
else
    σ² = ones(N,1)
end

#σ² = σ²/mean(σ²) #normalize so that σ̄² = 1
σ = vec(broadcast(√,σ²))
Δ = diagm(σ)

x00 = [.08, .3, .52, .74]*0.77 .+.01 # x position 
y00 = [.7, .4, .1]*0.77 .+ 0.1 # y position
xs = 0.15 # length of x axis
ys = xs * yratio # length of y axis

# if not CI file, calculate CI and save it
if !isfile(joinpath(fig4_dir, "fig4A_CI_ERM.txt"))
    index_all = zeros(3,3)
else
    index_all = open(joinpath(fig4_dir, "fig4A_CI_ERM.txt")) do file
        readlines(file)
    end
    # transform to array
    index_all = [parse(Float64, index_all[i]) for i = 1:length(index_all)]
    index_all = reshape(index_all, 3, 3) # assgin along columns
    index_all = transpose(index_all) # transpose to assign along rows
end

μ_list = [0.5, 0.9, 1.3]
for d = [1,2,3]
    for μ_id = [1,2,3]
        
        μ = μ_list[μ_id]
        L = (N/ρ)^(1/d) #box size

        p = ERMParameter(;N = N, L = L, ρ = ρ, n = d, ϵ = ϵ, μ = μ, ξ = ξ, β = β, σ̄² = mean(σ²), σ̄⁴ = mean(σ².^2))
        points = rand(Uniform(-L/2,L/2),N,d) #points are uniformly distributed in a region 
        C = reconstruct_covariance(points, p, f_r; subsample = false)


        ipanel+=1
        ax = fig.add_axes([x00[μ_id], y00[d], xs,ys])
        subsampling_rank_subplot(C,ax)

        if ipanel in [2, 3, 5, 6]
            # clear x and y ticks
            ax.set_xticks([]); ax.set_yticks([])
        elseif ipanel in [1, 4]
            ax.set_xticks([])
        elseif ipanel in [8, 9]
            ax.set_yticks([])
        end
        # if ipanel == 9
        #     ax.legend(bbox_to_anchor=(1.0, 0.1), loc="lower right", bbox_transform=fig.transFigure)
        # end
        
        # if no CI file, calculate CI and save it
        if index_all[d,μ_id] == 0.0
            index,cutoff1,cutoff2 = collapse_index_rank(C;isData=false)
            index_all[d,μ_id] = copy(index)
        else
            index = index_all[d,μ_id]
        end        
        ax.text(0.05,0.05, "CI = "*string(round(index,digits=3)),transform=ax.transAxes,size=8)
        # clear x any label
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_ylim([1e-1, 1e+2])

    end
end
fig.text(0.28, 0.1, "rank/N",size=10)
fig.text(0.01, 0.42, "eigenvalue λ",size=10, rotation=90)
# fig.text(0.01, 0.86, uppercase_letters[1],size=12,weight="bold")

fig.text(0.59, 0.45, L"Decreasing $\it{d}$",size=10, rotation=270)
ax = fig.add_axes([0.585, 0.5, 0.1,0.1])
ax.annotate("", xy=(0, 2.8), xycoords="axes fraction", xytext=(0, -2), 
            arrowprops=Dict("arrowstyle"=>"-|>,head_length=1,head_width=0.5"))
ax.axis("off")

fig.text(0.26, 0.89, L"Increasing $\it{μ}$",size=10)
ax = fig.add_axes([0.24, 0.88, 0.1,0.1])
ax.annotate("", xy=(2.5,0), xycoords="axes fraction", xytext=(-1,0), 
            arrowprops=Dict("arrowstyle"=>"-|>,head_length=1,head_width=0.5"))
ax.axis("off")

fig.savefig(joinpath(fig4_dir, "fig4A.png"),bbox_inches="tight",dpi=500)

plt.close(fig)

##
# save CI file to txt, each element separated by comma
if !isfile(joinpath(fig4_dir, "fig4A_CI_ERM.txt"))
    open(joinpath(fig4_dir, "fig4A_CI_ERM.txt"), "w") do file
        for i = 1:3
            for j = 1:3
                println(file, index_all[i,j])
            end
        end
    end
end

