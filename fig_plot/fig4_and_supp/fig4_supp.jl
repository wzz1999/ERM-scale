function CI_theory_plot(ax)

    resultroot2 = "C:\\Fish-Brain-Behavior-Analysis\\Fish-Brain-Behavior-Analysis\\results\\CI_rank"

    mu = 5

    @unpack var_all, index_num_cov_all = matread(joinpath(resultroot2,"num_var_shuffle_rank_$mu.mat"))
    var_all_1 = var_all; index_num_cov_all_1 = index_num_cov_all
    rankx = 1:11
    var_all2 = reshape(var_all, :, 11)
    index_num_cov_all2 = reshape(index_num_cov_all, :, 11)
    #p3 = StatsPlots.errorline(var_all2[1,rankx], index_num_cov_all2[:,rankx], errorstyle=:ribbon,label = "simulation")
    error_bar1 = std(index_num_cov_all2[:,rankx],dims =1)'
    mean_index_num_cov = mean(index_num_cov_all2[:,rankx],dims =1)'

    ax.plot(var_all2[1,rankx], mean_index_num_cov, color="#33B8FF", label="ERM simulation")
    ax.fill_between(var_all2[1,rankx], vec(mean_index_num_cov.-error_bar1), vec(mean_index_num_cov.+error_bar1), alpha=0.2, color="#33B8FF")
    ax.set_xlabel(L"E(\sigma^4)")
    ax.set_ylabel("collapse index (CI)")
    ax.set_ylim(0,0.15)
    ax.set_xlim(0.995,1.25)

    idd = 1:11
    @unpack var_all, index_theory_cov_all = matread(joinpath(resultroot3,"theory_var_fixc_0$mu.mat"))
    ax.plot(var_all[idd], index_theory_cov_all[idd], label="variational method", color="red",lw=0.8)
    ax.legend(borderpad=0.5,handlelength=0.6,frameon=false,fontsize=7)

    return ax
end