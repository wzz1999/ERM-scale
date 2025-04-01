using Pkg
# Switch to the desired Julia environment
Pkg.activate("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/")



include("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/src/util.jl") # ZZ's code
include("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/src/util2.jl")
include("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/src/PR_ratio.jl")
import .new_load_data as nld
import .new_analyses as na

data_type = "light_field"
data_dict = nld.get_data_dict(data_type)


fig_dir = joinpath(FIGURE_ROOT, "fig5_and_supp")
mkpath(fig_dir)

# Define result root directory to avoid repetition
const RESULT_ROOT = joinpath(nld.PR_ROOT, data_type)

nData = length(data_dict)

function subsampling_PR_3_types_publish(C,ax1, y_axis, x_axis=nothing; clus_type="rand")
    # plot the correlation matrix dimensionality
    N = size(C,1)
    if clus_type == "rand"
        neural_set = random_clusters(N)
    elseif clus_type == "anatomical" 
        neural_set = anatomical_clusters(N, y_axis)
    elseif clus_type == "RG_v2"
        neural_set = RG_clusters_v2(C)
    end

    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))
    PR_ratio = zeros(4)
    errorbar = zeros(4)
    K = zeros(4)
    i = 0
    for n in range(iterations, iterations-3, step=-1)
        i +=1
        # PR_ratio_all: PR_ratio for each cluster in one iteration
        # PR_ratio: the mean of PR_ratio_all
        PR_ratio, errorbar, K, PR_ratio_all = pca_cluster2_PR(neural_set, C, Int(2^n))
        x = [K./N for _ in 1:length(PR_ratio_all)]
        if isnan(errorbar)
            errorbar = 0
        end
        if clus_type == "rand"

            # ax1.errorbar(K./N.-0.02, PR_ratio, yerr=errorbar, fmt="o", color=colorList[i,:],ms=4.5, label=clus_type)
            ax1.scatter(K./N.-0.02, PR_ratio, marker="o", color=colorList[i,:],s=18, label=clus_type, facecolor="none")

        elseif clus_type == "anatomical" || clus_type == "ROI_CCA"

            # ax1.errorbar(K./N, PR_ratio, yerr=errorbar, fmt="s", color=colorList[i,:], ms=4.5, label=clus_type)
            ax1.scatter(K./N, PR_ratio, marker="s", color=colorList[i,:],s=18, label=clus_type, facecolor="none")

            ax1.scatter(x, PR_ratio_all, color=colorList[i, :], marker="o",s=1.5)
        # RG or functional 
        elseif clus_type == "RG" || clus_type == "RG_v2" || clus_type == "functional" || clus_type == "X_CCA" 
            ax1.scatter(K./N.+0.02, PR_ratio, color=colorList[i, :], marker="v",s=18, facecolor="none")
            # ax1.errorbar(K./N.+0.02, PR_ratio, yerr=errorbar, fmt="v", color=colorList[i,:], ms=4.5, label=clus_type)


        end
    end
    ax1.set_xlabel("subsampled fraction", fontsize=8)
    ax1.set_ylabel("dimensionality", fontsize=8)
    ax1.set_xlim(0.08,0.55)
    return ax1
end




import PyPlot as plt
uppercase_letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
fig = plt.figure(figsize=(6.85,5),facecolor="w",frameon=true, dpi=300)
# plt.rc("font",size=6,family="sans-serif")
plt.rc("font",size=8,family="Arial")
# plt.rc("font", serif="Helvetica")
plt.rc("pdf", fonttype=42)
yratio = 6.85/5



x00 = [.1, .42, .74] #.+.01 # x position 
y00 = [.5, .1] # y position
xs = 0.2 # length of x axis
ys = xs * yratio # length of y axis

ipanel = 0

#################
# dim across 6 fish
#################

function plot_dim_3type_across_fish(ax, data_dict, subject_i, ipanel; RCCA=0)
    data_type = "light_field"
    isubject = data_dict[subject_i]

    C_ls = nld.load_correlation_matrices(isubject, data_type)

    C = C_ls[1]

    y_axis = 1:size(C,1) # already sorted along AP-axis

    subsampling_PR_3_types_publish(C, ax, y_axis; clus_type="rand")
    subsampling_PR_3_types_publish(C, ax, y_axis; clus_type="anatomical")
    subsampling_PR_3_types_publish(C, ax, y_axis; clus_type="RG_v2")

    N = size(C, 1)
    iterations = Int(floor(log2(N)))
    cluster_set = random_clusters(N)
    d = 2
    μ = Parameter_estimation(C, n = d)[2]

    # Calculate the number of iterations based on your range
    K_pred_ls = collect(64:8:1024)
    K_pred_count = length(K_pred_ls)
    D_rand = zeros(K_pred_count)
    D_RG = zeros(K_pred_count)
    D_ana = zeros(K_pred_count)

    # Now you can use the index within the loop directly
    for (i, K_pred) in enumerate(K_pred_ls)
        K = N
        #D_rand[i], D_RG[i], D_ana[i] = theory_PR(cluster_set, C, K, K_pred, μ, d; iClus=1, RCCA = RCCA)
        D_rand[i], D_RG[i], D_ana[i] = theory_PR(cluster_set, C, K, K_pred; iClus=1, RCCA = RCCA)
    end

    ax.plot(K_pred_ls./N, D_rand, label="rand_theory", color="black", linestyle="--")
    ax.plot(K_pred_ls./N, D_RG, label="RG_theory", color="black", linestyle="-")
    ax.plot(K_pred_ls./N, D_ana, label="anatomical_theory", color="black", linestyle=":")


    ax.set_ylabel("dimensionality")
    ax.set_xticks([0.1, 0.3, 0.5])
    ax.text(-0.1, 1.05, uppercase_letters[ipanel],transform=ax.transAxes,size=14,weight="bold")
    return ax
end

# load RCCA_knee.mat
RCCA_knee = matread(joinpath(fig_dir, "RCCA_knee.mat"))

for subject_i = 1:3
    ipanel += 1
    ax = fig.add_axes([x00[ipanel], y00[1], xs,ys*1.1])
    key = "ifish"*data_dict[subject_i]
    plot_dim_3type_across_fish(ax, data_dict, subject_i, ipanel; RCCA = RCCA_knee[key])
end

for subject_i = 4:6
    ipanel += 1
    global ax = fig.add_axes([x00[ipanel-3], y00[2], xs,ys*1.1])
    key = "ifish"*data_dict[subject_i]
    plot_dim_3type_across_fish(ax, data_dict, subject_i, ipanel; RCCA = RCCA_knee[key])
end

handle1 = ax.scatter(-1, 0, marker="o", color="blue",s=10,facecolor="none")
handle2 = ax.scatter(-1, 0, marker="s", color="blue",s=10,facecolor="none")
handle3 = ax.scatter(-1, 0, marker="v", color="blue",s=10,facecolor="none")
handle4, = ax.plot(-1, 0, color="black", linestyle="--", label="rand_theory")
handle5, = ax.plot(-1, 0, color="black", linestyle="-", label="RG_theory")
handle6, = ax.plot(-1, 0, color="black", linestyle=":", label="anatomical_theory")

ax.legend([handle1, handle2, handle3, handle4, handle5, handle6], 
    ["RSap", "ASap", "FSap", "RSap theory", "uniform FSap theory", "ASap theory"], 
    loc="lower right",
    fontsize=6,
    handletextpad=0.5, borderpad=0.5, labelspacing=0.5,bbox_to_anchor=(1.8, 0),
)

plt.savefig(joinpath(fig_dir, "suppfig_dim_3type_across_fish_v2.pdf"), 
    bbox_inches="tight", dpi=300)
plt.close(fig)


# subject_i = 4
# isubject = data_dict[subject_i]
# data_type = "light_field"
# C_ls = nld.load_correlation_matrices(isubject, data_type)

# C = C_ls[1]
# cluster_set = RG_clusters_v2(C);
# X = na.matlab_mds(C)

# fig, ax = plt.subplots(1,1)
# # ax.scatter(X[:,1], X[:,2], s=1)
# ax.scatter(X[cluster_set[10][1],1], X[cluster_set[10][1],2], s=1,label="RG clus 1")
# ax.scatter(X[cluster_set[10][2],1], X[cluster_set[10][2],2], s=1, label="RG clus 2")
# ax.legend()
# plt.savefig(joinpath(fig_dir, "suppfig_dim_3type_fish4_mds.pdf"), 
#     bbox_inches="tight", dpi=300)


