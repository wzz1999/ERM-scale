
using Pkg
# Switch to the desired Julia environment
#Pkg.activate("C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/")


include("../../src/util.jl") # ZZ's code
include("../../src/util2.jl")

FIGURE_ROOT =  "/home/wenlab-user/wangzezhen/ERM-scale/figures"

fig4_dir = joinpath(FIGURE_ROOT, "fig4_and_supp")
mkpath(fig4_dir)
using Colors

function CI_data_plot(ax)
    
    dataroot = "/home/wenlab-user/wangzezhen/ERM-scale/results/CI_rank/normalization4"
    ci_cor201106 = matread(joinpath(dataroot, "cor_CI_normlization.mat"))["ci_cor201106"]
    ci_cor201116 = matread(joinpath(dataroot, "cor_CI_normlization.mat"))["ci_cor201116"]
    ci_cor201117 = matread(joinpath(dataroot, "cor_CI_normlization.mat"))["ci_cor201117"]
    ci_cor201125 = matread(joinpath(dataroot, "cor_CI_normlization.mat"))["ci_cor201125"]
    ci_cor201126 = matread(joinpath(dataroot, "cor_CI_normlization.mat"))["ci_cor201126"]
    ci_cor210106 = matread(joinpath(dataroot, "cor_CI_normlization.mat"))["ci_cor210106"]
    ci_cor_light_sheet_fish11 = matread(joinpath(dataroot, "cor_light_sheet_ci_mean.mat"))["ci_cor_light_sheet_fish11"]
    ci_cor_Waksman = matread(joinpath(dataroot, "cor_mouse_ci_mean.mat"))["ci_cor_Waksman"]
    ci_cor_spont_M160907_MP028_2016_09_26 = matread(joinpath(dataroot, "cor_mouse_ci_mean.mat"))["ci_cor_spont_M160907_MP028_2016_09_26"]

    ci_cov201106 = matread(joinpath(dataroot, "cov_CI_normlization.mat"))["ci_cov201106"]
    ci_cov201116 = matread(joinpath(dataroot, "cov_CI_normlization.mat"))["ci_cov201116"]
    ci_cov201117 = matread(joinpath(dataroot, "cov_CI_normlization.mat"))["ci_cov201117"]
    ci_cov201125 = matread(joinpath(dataroot, "cov_CI_normlization.mat"))["ci_cov201125"]
    ci_cov201126 = matread(joinpath(dataroot, "cov_CI_normlization.mat"))["ci_cov201126"]
    ci_cov210106 = matread(joinpath(dataroot, "cov_CI_normlization.mat"))["ci_cov210106"]
    ci_cov_light_sheet_fish11 = matread(joinpath(dataroot, "cov_light_sheet_ci_mean.mat"))["ci_cov_light_sheet_fish11"]
    ci_cov_Waksman = matread(joinpath(dataroot, "cov_mouse_ci_mean.mat"))["ci_cov_Waksman"]
    ci_cov_spont_M160907_MP028_2016_09_26 = matread(joinpath(dataroot, "cov_mouse_ci_mean.mat"))["ci_cov_spont_M160907_MP028_2016_09_26"]

    index_num_cov_all = matread(joinpath(dataroot, "../num_var_shuffle_rank_5.mat"))["index_num_cov_all"];

    
    sizes = 30

    c = palette(:turbo, 10) 
    # Convert to an array of RGB values
    rgb_values = [Float64[Colors.red(color), Colors.green(color), Colors.blue(color)] for color in c]

    # Alternatively, create a matrix if you need this format
    c = hcat(rgb_values...)'

    c1 = copy(c)

    ax.scatter(1, ci_cor201106,sizes, color=c[1,:], marker="s")
    ax.scatter(2, ci_cor201116, sizes, color=c[2,:], marker="s")
    ax.scatter(3, ci_cor201117, sizes, color=c[3,:], marker="s")
    ax.scatter(4, ci_cor201125, sizes, color=c[4,:], marker="s")
    ax.scatter(5, ci_cor201126, sizes, color=c[5,:], marker="s")
    ax.scatter(6, ci_cor210106, sizes, color=c[6,:], marker="s")
    ax.scatter(7, ci_cor_light_sheet_fish11, sizes, color=c[7,:], marker="s")
    ax.scatter(8, ci_cor_Waksman, sizes, color=c[8,:], marker="s")
    ax.scatter(9, ci_cor_spont_M160907_MP028_2016_09_26, sizes, color=c[9,:], marker="s")
    ax.scatter(10, index_num_cov_all[1], sizes, color=c[10,:], marker="s")


    ax.scatter(1, ci_cov201106, sizes, color=c1[1,:], marker="s", linewidth=1, facecolor="none")
    ax.scatter(2, ci_cov201116, sizes, color=c1[2,:], marker="s", linewidth=1, facecolor="none")
    ax.scatter(3, ci_cov201117, sizes, color=c1[3,:], marker="s", linewidth=1, facecolor="none")
    ax.scatter(4, ci_cov201125, sizes, color=c1[4,:], marker="s", linewidth=1, facecolor="none")
    ax.scatter(5, ci_cov201126, sizes, color=c1[5,:], marker="s", linewidth=1, facecolor="none")
    ax.scatter(6, ci_cov210106, sizes, color=c1[6,:], marker="s", linewidth=1, facecolor="none")
    ax.scatter(7, ci_cov_light_sheet_fish11, sizes, color=c1[7,:], marker="s", linewidth=1, facecolor="none")
    ax.scatter(8, ci_cov_Waksman, sizes, color=c1[8,:], marker="s", linewidth=1, facecolor="none")
    ax.scatter(9, ci_cov_spont_M160907_MP028_2016_09_26, sizes, color=c1[9,:], marker="s", linewidth=1, facecolor="none")
    ax.scatter(10, index_num_cov_all[end], sizes, color=c1[10,:], marker="s", linewidth=1, facecolor="none")


    color_l = [0.5, 0.5, 0.5]
    ax.plot([1,1],[ci_cor201106,ci_cov201106], color=color_l)
    ax.plot([2,2],[ci_cor201116,ci_cov201116], color=color_l)
    ax.plot([3,3],[ci_cor201117,ci_cov201117], color=color_l)
    ax.plot([4,4],[ci_cor201125,ci_cov201125], color=color_l)
    ax.plot([5,5],[ci_cor201126,ci_cov201126], color=color_l)
    ax.plot([6,6],[ci_cor210106,ci_cov210106], color=color_l)
    ax.plot([7,7],[ci_cor_light_sheet_fish11,ci_cov_light_sheet_fish11], color=color_l)
    ax.plot([8,8],[ci_cor_Waksman,ci_cov_Waksman], color=color_l)
    ax.plot([9,9],[ci_cor_spont_M160907_MP028_2016_09_26,ci_cov_spont_M160907_MP028_2016_09_26], color=color_l)
    ax.plot([10,10],[index_num_cov_all[1],index_num_cov_all[end]], color=color_l)

    ax.set_xlabel("dataset")
    ax.set_ylabel("collapse index (CI)")


    # Create a modified version of the tick labels
    labels_modif = ["f1", "f2", "f3", "f4", "f5", "f6", "fl", "mn", "mp", "ERM"]

    ax.set_xticks(1:length(labels_modif),labels_modif,fontsize=6)

    ax.set_ylim([0, 0.2])
    
    return ax
end

########################################################################################
## fig4BC


isubject = "201106"
data_dir = "/home/data2/wangzezhen/fishdata" #"C:/Fish-Brain-Behavior-Analysis/Fish-Brain-Behavior-Analysis/results/"
FR = matread(joinpath(data_dir, isubject, "spike_OASIS.mat"))["sMatrix_total"]
Judge2 = matread(joinpath(data_dir, isubject, "Judge2.mat"))["Judge2"]
y_conv_range = matread(joinpath(data_dir, isubject, "y_conv_range.mat"))["y_conv_range"]
FR = FR[vec(Judge2),1:7200]

nC,nT = size(FR)

FR_raw, Cov_M = data_preprocessing_and_subsample(FR;flag_rand_seed=true);
Corr_M = cor(convert(Matrix, FR_raw'));

uppercase_letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
fig = plt.figure(figsize=(6.85,5),facecolor="w",frameon=true, dpi=300)
plt.rc("pdf", fonttype=42)
plt.rc("font",size=8,family="Arial")
yratio = 6.85/5

x0 = .08 # x position 
y00 = [.7, .4, .1]*1.05 # y position
xs = 0.18 # length of x axis
ys = xs * yratio # length of y axis


ipanel = 1 # start from B



###############################
##----corr vs cov spectrum---------##
###############################

ipanel+=1

ax = fig.add_axes([x0, y00[1], xs,ys])
subsampling_rank_subplot(Corr_M,ax)
ax.set_ylim([1e-1, 1e+2+10])
ax.legend(borderpad=0.5,handlelength=0.6,frameon=false,fontsize=6,loc="upper right")
ax.text(-0.2, 1.1, uppercase_letters[ipanel],transform=ax.transAxes,size=12,weight="bold")
ax.text(0.05,0.05, "Corr", transform=ax.transAxes,fontsize=8)
ax.set_xticks([])
ax.set_xlabel("")

ax = fig.add_axes([x0, y00[2]+0.06, xs,ys])
subsampling_rank_subplot(Cov_M,ax)
ax.set_ylim([1e-1, 1e+2+10])
# ax.legend(borderpad=0.5,handlelength=0.6,frameon=false,fontsize=6.5,loc="upper right")
ax.text(0.05,0.05, "Cov", transform=ax.transAxes,fontsize=8)

#####################
# CI_data_plot #
#####################
ipanel+=1
ax = fig.add_axes([x0-0.07, y00[3]+0.1, xs*1.4,ys*0.7])
CI_data_plot(ax)
ax.text(-0.2, 1.1, uppercase_letters[ipanel],transform=ax.transAxes,size=12,weight="bold")

fig.savefig(joinpath(fig4_dir, "fig4BC.pdf"), bbox_inches="tight")
plt.close(fig)


c = [
        0.1900 0.0718 0.2322;
        0.2769 0.4658 0.9370;
        0.1080 0.8127 0.8363;
        0.3857 0.9896 0.4202;
        0.8207 0.9143 0.2063;
        0.9967 0.6082 0.1778;
        0.8568 0.2250 0.0276;
        0.4796 0.0158 0.0106
    ]