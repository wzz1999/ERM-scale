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


for subject_i = 1:nData
    if subject_i == 1
        isubject = data_dict[subject_i]
        global combined_CI_all = na.combine_CI(RESULT_ROOT, isubject)
    else
        isubject = data_dict[subject_i]
        combined_CI = na.combine_CI(RESULT_ROOT, isubject)
        combined_CI_all = vcat(combined_CI_all, combined_CI)
    end
end


import PyPlot as plt
uppercase_letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
fig = plt.figure(figsize=(6.85,5),facecolor="w",frameon=true, dpi=300)
# plt.rc("font",size=6,family="sans-serif")
plt.rc("font",size=8,family="Arial")
# plt.rc("font", serif="Helvetica")
plt.rc("pdf", fonttype=42)
yratio = 6.85/5



x00 = [.08, .3, .52, .74] #.+.01 # x position 
y00 = [.7, .5, .1] # y position
xs = 0.2 # length of x axis
ys = xs * yratio # length of y axis



#################
# fish 6 dim
#################
ipanel = 5
ax = fig.add_axes([x00[1], y00[2], xs,ys*1.1])

data_type = "light_field"
subject_i = 6
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

# Now you can use the index within the loop directly
for (i, K_pred) in enumerate(K_pred_ls)
    K = N
    D_rand[i], D_RG[i] = theory_PR(cluster_set, C, K, K_pred, μ, d; iClus=1)
end

ax.plot(K_pred_ls./N, D_rand, label="rand_theory", color="black", linestyle="--")
ax.plot(K_pred_ls./N, D_RG, label="RG_theory", color="black", linestyle="-")

handle1 = ax.scatter(-1, 0, marker="o", color="blue",s=10,facecolor="none")
handle2 = ax.scatter(-1, 0, marker="s", color="blue",s=10,facecolor="none")
handle3 = ax.scatter(-1, 0, marker="v", color="blue",s=10,facecolor="none")
handle4, = ax.plot(-1, 0, color="black", linestyle="--", label="rand_theory")
handle5, = ax.plot(-1, 0, color="black", linestyle="-", label="RG_theory")

ax.legend([handle1, handle2, handle3, handle4, handle5], 
    ["RSap", "ASap", "FSap", "RSap theory", "uniform FSap theory"], 
    loc="lower right",
    fontsize=6,
    handletextpad=0.5, borderpad=0.5, labelspacing=0.5,bbox_to_anchor=(1.8, 0),
    )
ax.set_ylabel("dimensionality")
ax.set_xticks([0.1, 0.3, 0.5])
ax.text(-0.1, 1.05, uppercase_letters[ipanel],transform=ax.transAxes,size=14,weight="bold")


#################
# light field CI
#################
ipanel = 6
ax = fig.add_axes([x00[1], y00[3], xs,ys])
for i = 1:size(combined_CI_all, 1)  # size(combined_CI_all, 1) gives the number of rows
    ax.plot(1:size(combined_CI_all, 2), combined_CI_all[i, :], "-o", label = "fish " * string(i), ms=3)  # Assuming you want x-values from 1 to number of columns
end
ax.set_xticks([1, 2, 3], ["RSap", "ASap", "FSap"])
ax.text(-0.1, 1.05, uppercase_letters[ipanel],transform=ax.transAxes,size=14,weight="bold")
ax.legend(loc="lower right", fontsize=6, ncol=1, columnspacing=0.5, 
    handletextpad=0.5, borderpad=0.5, labelspacing=0.5,bbox_to_anchor=(1.4, 0))
ax.set_ylabel("CI")
ax.set_ylim([0.05, 0.25])
# plt.savefig(joinpath("C:/Fish-Brain-Behavior-Analysis/Fish-Brain-Behavior-Analysis/figure/version_24Mar/fig5", "light_field_dim_and_CI.pdf"))
plt.savefig(joinpath(fig_dir, "light_field_dim_and_CI.pdf"))
plt.close()

using HypothesisTests

# Initialize an array to hold the p-values
p_values = zeros(3)


function one_side_paired_t_test(x, y)
    # Perform paired t-test on the column
    # Subtract the first observation from the rest to get the paired differences
    diffs = x .- y
    test = OneSampleTTest(diffs)
    # Check the sign of the test statistic and divide the two-sided p-value by 2 for one-sided test
    p_value = (test.t > 0) ? (pvalue(test) / 2) : 1 - (pvalue(test) / 2)
    println(p_value)

    return p_value
end

p_values[1] = one_side_paired_t_test(combined_CI_all[:,1], combined_CI_all[:,2])

p_values[2] = one_side_paired_t_test(combined_CI_all[:,1], combined_CI_all[:,3])

p_values[3] = one_side_paired_t_test(combined_CI_all[:,2], combined_CI_all[:,3])


# save p_values
open(joinpath(fig_dir, "light_field_dim_and_CI_p_values.txt"), "w") do io
    println(io, "RSap vs ASap p_values: ", p_values[1])
    println(io, "RSap vs FSap p_values: ", p_values[2])
    println(io, "ASap vs FSap p_values: ", p_values[3])
end

# keep 3 digits
p_values = round.(p_values, digits=4)
