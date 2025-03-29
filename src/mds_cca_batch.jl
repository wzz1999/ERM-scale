include("util.jl")

root = "ERM-scale/results/mds_CCA"

color0 = :RdBu
#ifish_all = [201106, 201116, 201117, 210106, 210430]
ifish_all = [210106]


for ifish0 in ifish_all
    global ifish = Int(ifish0)
    for n0 = 2:3
        global n = n0
        include("mds.jl")
        println("$ifish")
        ROI_MDS_CCA = fit(CCA,ROIxyz',X)
        correlations_CCA = correlations(ROI_MDS_CCA)
        ROI_pjec = xprojection(ROI_MDS_CCA)
        X_pjec = yprojection(ROI_MDS_CCA)
        ROI_CCA = ROIxyz*ROI_pjec
        X_CCA = X'*X_pjec
        scatter(ROIxyz[:,2],ROIxyz[:,1], marker_z = ordinalrank(X_CCA[:,1]), markersize = 3,  color = color0, markerstrokecolor = :white, label = "$ifish")
        savefig(joinpath(root,"$ifish/$n/fish_space.png"))
        savefig(joinpath(root,"$ifish/$n/fish_space.svg"))
        matwrite(joinpath(root,"$ifish/$n/X_CCA.mat"),
            Dict("X_CCA" => X_CCA, "X" =>X,"C2" =>C2,"C3" =>C3,"D" =>D , "ROIxyz" => ROIxyz, "correlations_CCA" => correlations_CCA,
            "X_pjec" => X_pjec, "ROI_pjec" => ROI_pjec, "X_rank" =>ordinalrank(X_CCA[:,1])))
    end
    if ifish == 201117
        corr_dim, corr_shuffle = CCA_corr_dim(Corr; n_all = 2:10, ifish = ifish, figroot = joinpath(root,"$ifish"))
    else
        corr_dim, corr_shuffle = CCA_corr_dim(Corr; n_all = 1:10, ifish = ifish, figroot = joinpath(root,"$ifish"))
    end
    matwrite(joinpath(root,"$ifish/corr_dim.mat"), Dict("corr_dim" => corr_dim, "corr_shuffle" => corr_shuffle))
end

