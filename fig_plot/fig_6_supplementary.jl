include("../src/util.jl")
include("../src/util_weihao.jl")
include("kernel_regression_function.jl")

pyplot()
using PyPlot
plt.rcParams["figure.dpi"] = 300
plot = Plots.plot
plt.rc("font",family="Arial")
plt.rc("pdf", fonttype=42)

Plots.default(titlefont = ("Arial"), legendfont = ("Arial"))

dpi = 300
bsonroot = "D:/Fish-Brain-Behavior-Analysis/results/mds_CCA/visualization"
matroot = "C:/Users/wzz/Desktop/fig_server/mds_CCA"
resultroot = "D:/Fish-Brain-Behavior-Analysis/figure/fig6"
load_data_root = "../src/load_fish_data2.jl"

ifish_all = [201106,201116,201117,201125,201126]
#ifish_all = [201116,210106,201106]
n = 2
cbar = :thermal

N_color = 1000
cc = colormap("RdBu", N_color)
cc2 = copy(cc)

function f_color(x; N_color = N_color, a = 3) 
    x0 = x/(N_color+1) *2 -1
    y = (tanh(a*x0) + 1)/2
    y = Int(round(y*N_color))
    y = min(y,N_color)
    y = max(y,1)
    return y
end

for i = 1:N_color
    cc[i] = cc2[f_color(i)]
end

l = @layout [
    [a b c _] 
    [e f g _]  
]

title_G = Dict(); title_H = Dict()
title_G[201106] = "A"; title_G[201116] = "B"; title_G[201117] ="C"; title_G[210106] ="D";title_G[201125] = "D";title_G[201126] = "E"
title_H[201106] = "F"; title_H[201116] = "G"; title_H[201117] ="H"; title_H[210106] ="H";title_H[201125] = "I";title_H[201126] = "J"
title_A = title_G; title_B = title_H; title_E = title_G
title_C = Dict(); title_D = Dict()
title_C[201106] = "K"; title_C[201116] = "L"; title_C[201117] ="M"; title_C[210106] ="N"; title_C[201125] ="N"; title_C[201126] ="O"
title_D[201106] = "P"; title_D[201116] = "Q"; title_D[201117] ="R"; title_D[210106] ="S"; title_D[201125] ="S"; title_D[210126] ="T"
title_F = title_C
title_B2 = title_C
title_C = title_A
title_D = title_B

function plot_fig_A(ifish0)
    global ifish = ifish0    
    p1 = plot(title = title_A[ifish],grid = :none,xlabel = "rank/N", ylabel = "eigenvalue "* L"\lambda", dpi =dpi)
    return p1
end

function plot_fig_B(ifish0)
    global ifish = ifish0    
    p1 = plot(title = title_B[ifish],grid = :none,xlabel = "rank/N", ylabel = "eigenvalue "* L"\lambda", dpi =dpi)
    return p1
end

function plot_fig_B2(ifish0)
    global ifish = ifish0    
    p1 = plot(title = title_B2[ifish],grid = :none,xlabel = "rank/N", ylabel = "eigenvalue "* L"\lambda", dpi =dpi)
    return p1
end

function plot_fig_C(ifish0; load_data_root = "../src/load_fish_data2.jl", cbar = :thermal, use_c = "cov")
    global ifish = ifish0
    include(load_data_root)
    #@unpack X = matread(joinpath(matroot,"$ifish","$n","X.mat"))
    @unpack X, C2, D, X_CCA, ROI_pjec, X_pjec = matread(joinpath(matroot,"$ifish","$n","X_CCA.mat"))
    R = hclust(D,linkage = :average)
    X = X["parent"]' 
    if use_c == "cov"
        C = Cᵣ
    else
        C = Corr
    end
    ip_file = string(bsonroot, "/", "permutation_index_", "all", "_", "$ifish", "_", "2021-07-02", ".bson")
    if isfile(ip_file)
        BSON.@load ip_file ip
    else
        R = hclust(D,linkage = :average)
        ip = R.order
    end
    #BSON.@load string(bsonroot, "/", "permutation_index_", "all", "_", "$ifish", "_", "2021-07-02", ".bson") ip
    #p3 = heatmap(C[R.order, R.order],theme = :dark, clim = (0,0.7),ratio=:equal,title = title_C[ifish],framestyle = :none,legend = :none, color=cgrad(cbar), dpi =dpi)
    p3 = heatmap(C[ip, ip],theme = :dark, clim = (0,0.5),ratio=:equal,title = title_C[ifish],framestyle = :none,legend = :none, color=cgrad(cbar), dpi =dpi)

    return p3
end

function plot_fig_D(ifish0; load_data_root = "../src/load_fish_data2.jl", cbar = :thermal, use_c = "cov")
    global ifish = ifish0
    include(load_data_root)
    #@unpack X = matread(joinpath(matroot,"$ifish","$n","X.mat"))
    @unpack X, C2, D, X_CCA, ROI_pjec, X_pjec = matread(joinpath(matroot,"$ifish","$n","X_CCA.mat"))
    R = hclust(D,linkage = :average)
    X = X["parent"]' 
    if use_c == "cov"
        C = corr2cov(C2, Cᵣ)
    else
        C = C2
    end
    #BSON.@load string(bsonroot, "/", "permutation_index_", "all", "_", "$ifish", "_", "2021-07-02", ".bson") ip
    ip_file = string(bsonroot, "/", "permutation_index_", "all", "_", "$ifish", "_", "2021-07-02", ".bson")
    if isfile(ip_file)
        BSON.@load ip_file ip
    else
        R = hclust(D,linkage = :average)
        ip = R.order
    end
    #p4 = heatmap(C[R.order, R.order],theme = :dark, clim = (0,0.7),ratio=:equal,title = title_D[ifish],framestyle = :none,legend = :none, color=cgrad(cbar), dpi =dpi)
    p4 = heatmap(C[ip, ip],theme = :dark, clim = (0,0.5),ratio=:equal,title = title_D[ifish],framestyle = :none,legend = :none, color=cgrad(cbar), dpi =dpi)
    return p4
end

function plot_fig_E(ifish0; load_data_root = "../src/load_fish_data2.jl", isloglog = true)
    global ifish = ifish0
    include(load_data_root)
    @unpack X, C2, D, X_CCA, ROI_pjec, X_pjec = matread(joinpath(matroot,"$ifish","$n","X_CCA.mat"))
    X = X["parent"]'
    D2 = pairwise(Euclidean(), X', dims=1)
    d = n
    ρ, μ = Parameter_estimation(Corr, n = n)
    N = size(Corr)[1]
    L = (N/ρ)^(1/n)
    p = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = 0.03125, μ = μ, ξ = 10^18, β = 0, σ̄² = 1, σ̄⁴ = 1)
    K = length(D2[0.999 .>Corr.>0])
    id = findall(D2[0.999 .>Corr.>0].<0.2)
    id1 = sample(findall(0.01 .<D2[0.999 .>Corr.>0].<0.1),2500,replace = true)
    id2 = sample(findall(0.1 .<D2[0.999 .>Corr.>0].<1),3333,replace = true)
    id3 = sample(findall(1 .<D2[0.999 .>Corr.>0].<10),3333,replace = true)
    if L>10
        id4 = sample(findall(10 .<D2[0.999 .>Corr.>0].<100),min(2500,length(10 .<D2[0.999 .>Corr.>0].<100)),replace = true)
        id = vcat(id1,id2,id3,id4)
    else
        id = vcat(id1,id2,id3)
    end
    global x_train = D2[0.999 .>Corr.>0][id]#-5:0.5:5
    global x_test = 0.1:0.01:round(1.5*L)#-7:0.1:7
    if n==2
        global x_test = 0.1:0.01:min(100,round(1.5*L))
    end
    if ifish==201106||ifish==201126
        #global x_test = 0.01:0.01:min(10,round(2*L))
        global x_test = 0.1:0.01:min(10,round(2*L))
    end
    global y_train = Corr[0.999 .>Corr.>0][id]#f_truth.(x_train) + noise
    global y_test = corr(x_test,p) #f_truth.(x_test)
    lambda=1e-3; kernel= GaussianKernel()
    y_pred = kernel_ridge_regression(kernel, x_train, y_train, x_test, lambda)
    y_error = error_bar(x_test,x_train,y_train)
    #Plots.scatter(x_train, y_train; label=nothing,xaxis = :log, yaxis = :log)
    if  isloglog == true
        p5 = plot(x_test, y_test; label="f(x)", title=title_H[ifish],grid = :none,xaxis = :log, yaxis = :log, legend=:bottomleft, ylim = (0.01,1), dpi =dpi)#, legendfontsize=6
    else
        p5 = plot(x_test, y_test; label="Model f(x)", title=title_E[ifish],grid = :none, dpi =dpi)#, legendfontsize=6
    end
    p5 = plot!(x_test, y_pred;ribbon = y_error, label="Data f(x)",xlabel = "distance", ylabel = "correlation")
    return p5
end
#plot_fig_E(210106, isloglog = false)

function plot_fig_F(ifish0; load_data_root = "../src/load_fish_data2.jl")
    global ifish = ifish0
    include(load_data_root)
    #@unpack X = matread(joinpath(matroot,"$ifish","$n","X.mat"))
    @unpack X, C2, D, X_CCA, ROI_pjec, X_pjec = matread(joinpath(matroot,"$ifish","$n","X_CCA.mat"))
    @unpack corr_dim, corr_shuffle= matread(joinpath(matroot,"$ifish","corr_dim.mat"))
    X = X["parent"]' 
    dim = 1:10
    dim2 = 1:10
    if ifish == 201117
        dim = 1:9
        dim2 = 2:10
    end
    p6 = plot(dim2,corr_dim[1,dim],title = title_F[ifish],ylim = (0,1),grid = :none,label = "data", xlim = (0.9,dim[end]), dpi =dpi)#scatter(D2[0.999 .>Corr.>0], Corr[0.999 .>Corr.>0], xaxis = :log, yaxis = :log, markersize = 2)
    p6 = Plots.scatter!(dim2,corr_dim[1,dim],label = :none, color = :blue,xlabel = "dimension "*L"d", ylabel = "canonical correlation",markersize = 2)
    p6 = plot!(dim2,mean(corr_shuffle,dims =2)[dim],yerr = std(corr_shuffle,dims =2)[dim],markerstrokecolor=:auto,label = "shuffle")
    return p6
end


X01_all = Dict(); k01_all = Dict(); a1_loc_all = Dict()#; title_G = Dict()
X01_all[201116] = [-18,28]; X01_all[201117] = [-50,155]; X01_all[210106] = [-8,20]; X01_all[201106] = [-1,1]; X01_all[201125] = [0,60]; X01_all[201126] = [-10,10]
k01_all[201116] = 250; k01_all[201117] =5000; k01_all[210106] =70; k01_all[201106] =2; k01_all[201125] =1000; k01_all[201126] =105
a1_loc_all[201116] = [-20,40]; a1_loc_all[201117] = [-100,180]; a1_loc_all[210106] = [-12,15]; a1_loc_all[201106] = [-1,2.5]; a1_loc_all[201125] = [-60,60]; a1_loc_all[201126] = [-8,20]
#title_G[201116] = "A"; title_G[201117] = "B"; title_G[210106] ="C"; title_G[201106] ="D"
function plot_fig_G(ifish0; load_data_root = "../src/load_fish_data2.jl")
    global ifish = ifish0
    include(load_data_root)
    #@unpack X = matread(joinpath(matroot,"$ifish","$n","X.mat"))
    @unpack X, C2, D, X_CCA, ROI_pjec, X_pjec = matread(joinpath(matroot,"$ifish","$n","X_CCA.mat"))
    X = X["parent"]' 
    if ifish0 == 201117
        ROI_pjec = -ROI_pjec; X_pjec = -X_pjec 
    end
    p7 = Plots.scatter(X[1,:],X[2,:], markersize = 2,size=(400,300),grid = :none, label = :none, title = title_G[ifish], ratio=:equal, dpi =dpi,
    marker_z= (ROIxyz*ROI_pjec)[:,1],markercolor = :PRGn, markerstrokecolor = :white, markerstrokewidth = 0)
    X01 = X01_all[ifish]
    k01 = k01_all[ifish]
    a1_loc = a1_loc_all[ifish]
    #p7 = plot!([X01[1],X01[1]+k01*X_pjec[1,1]],[X01[2],X01[2]+k01*X_pjec[2,1]],arrow=true,color=:black,linewidth=2,label="",legend=:topleft)
    p7 = arrow_color(X01, k01, X_pjec,color_map = palette(cc))
    #p7 = annotate!(a1_loc[1],a1_loc[2], L"\vec{a}_1")
    return p7
end


X02_all = Dict(); k02_all = Dict()#; title_H = Dict()
X02_all[201116] = [130,250]; X02_all[201117] = [95,200]; X02_all[210106] = [60,203]; X02_all[201106] = [100,260]; X02_all[201125] = [60,203]; X02_all[201126] = [60,203]
k02_all[201116] = 2500; k02_all[201117] =5000; k02_all[210106] =10000; k02_all[201106] =5000; k02_all[201125] =5000; k02_all[201126] =5000
#title_H[201116] = "E"; title_H[201117] = "F"; title_H[210106] ="G"; title_H[201106] ="H"
function plot_fig_H(ifish0; load_data_root = "../src/load_fish_data2.jl")
    global ifish = ifish0
    include(load_data_root)
    #@unpack X = matread(joinpath(matroot,"$ifish","$n","X.mat"))
    @unpack X, C2, D, X_CCA, ROI_pjec, X_pjec = matread(joinpath(matroot,"$ifish","$n","X_CCA.mat"))
    if (ifish == 201117) | (ifish == 201126)
        ROI_pjec = -ROI_pjec; X_pjec = -X_pjec; X_CCA = -X_CCA
    end
    X = X["parent"]'
    if (ifish == 201116) | (ifish == 201117)
        global ROIxyz = ROIxyz[:,[2,1,3]]
        ROI_pjec = ROI_pjec[[2,1,3],:]
    end
    p8 = Plots.scatter(ROIxyz[:,1],ROIxyz[:,2], marker_z = (X_CCA[:,1]), markersize = 2, size = (400,300), dpi =dpi, ratio=:equal,
        markercolor = palette(cc), markerstrokecolor = :white, label = "",clim = (-2+median(X_CCA[:,1]),2+median(X_CCA[:,1])),title = title_H[ifish],
        grid = :none, xlims = (50,400),ylims = (25,275), markerstrokewidth = 0)
    X02 = X02_all[ifish]
    k02 = k02_all[ifish]
    #p8 = plot!([X02[1],X02[1]+k02*ROI_pjec[1,1]],[X02[2],X02[2]+k02*ROI_pjec[2,1]],arrow=true,color=:black,linewidth=2,label="",legend=:topleft)
    p8 = arrow_color(X02, k02, ROI_pjec,color_map = palette(palette_colormap(:PRGn)))
    p8 = plot_scale_fig6(dpi=dpi,fsize = 12,yscale = 260)
    #p8 = annotate!(100,230, L"\vec{b}_1")
    return p8
end
#plot_fig_H(201117)
#=
uppercase_letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
title_all = Dict()
j = 0
for i in ifish_all
    j = j+1
    title_all[i] = j
end
=#
show_fig = "e"

if show_fig == "abb"
    pabcd = plot((plot_fig_A(i) for i in ifish_all)..., (plot_fig_B(i) for i in ifish_all)..., (plot_fig_B2(i) for i in ifish_all)..., 
        layout= (3,length(ifish_all)),size = (1125,650), titleloc = :left, left_margin = 10Plots.mm)

    #pabcd = heatmap!((0:0.01:1).*ones(101,1), legend=:none, xticks=:none, yticks=(1:100:101, string.(0:0.7:0.7)), color=cgrad(cbar),
        #inset = (9, bbox(-0.20, 0.05, 0.10, 2.1, :left)),subplot = 17)

    fig = pabcd
    Plots.prepare_output(fig)
    for i = 1:length(ifish_all)
        global ifish = ifish_all[i]
        include(load_data_root)
        @unpack X, C2, D, X_CCA, ROI_pjec, X_pjec = matread(joinpath(matroot,"$ifish","$n","X_CCA.mat"))
        ###################################################################################################################
        ϵ = 0.03125
        d = n
        ρ, μ = Parameter_estimation(Corr, n = n)
        N = size(Corr)[1]
        L = (N/ρ)^(1/n)
        p = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = 0.03125, μ = μ, ξ = 10^18, β = 0, σ̄² = 1, σ̄⁴ = 1)
        points = rand(Uniform(-L/2,L/2),N,d)
        #Correrm = reconstruct_covariance(points, p, subsample = false)
        #C_erm = corr2cov(Correrm, Cᵣ)
        LogNormal_param = fit_mle(LogNormal, diag(Cᵣ)) 
        σ² = vec(rand(LogNormal_param,N))
        σ = vec(broadcast(√,σ²))
        Δ = diagm(σ)
        Correrm = reconstruct_covariance(points, p, subsample = false)
        #C_erm = corr2cov(Correrm, Cᵣ)
        C_erm = Δ*Correrm*Δ
        ###################################################################################################################

        C3 = corr2cov(C2, Cᵣ)
        id = sample(1:N,1024,replace=false)
        global pyfig = subsample_fig6(Cᵣ[id,id], fig; subfig = i)
        global pyfig = subsample_fig6(C3[id,id], pyfig; subfig = i+length(ifish_all), is_pyfig = true)
        global pyfig = subsample_fig6(C_erm[id,id], pyfig; subfig = i+length(ifish_all)+length(ifish_all), is_pyfig = true)
    end

    pyfig.savefig(joinpath(resultroot,"fig6_sup_abb.png"), dpi = dpi)
    pyfig.savefig(joinpath(resultroot,"fig6_sup_abcd.pdf"), dpi = dpi)
    pyfig.savefig(joinpath(resultroot,"fig6_sup_abb.jpeg"), dpi = dpi)
end

if show_fig == "df"
    pabcd = plot((plot_fig_C(i) for i in ifish_all)..., (plot_fig_D(i) for i in ifish_all)..., 
        layout= (2,length(ifish_all)),size = (1200,400), titleloc = :left, left_margin = 40Plots.mm)

    pabcd = heatmap!((0:0.01:1).*ones(101,1), legend=:none, xticks=:none, yticks=(1:100:101, string.(0:0.5:0.5)), color=cgrad(cbar),
        inset = (1, bbox(-0.15, 0.00, 0.10, 2.05, :left)),subplot = 11)

    fig = pabcd
    Plots.prepare_output(fig)
    pyfig = fig.o

    pyfig.savefig(joinpath(resultroot,"fig6_sup_df.png"), dpi = dpi)
    #pyfig.savefig(joinpath(resultroot,"fig6_sup_df.pdf"), dpi = dpi)
    pyfig.savefig(joinpath(resultroot,"fig6_sup_df.jpeg"), dpi = dpi)
end

if show_fig == "abcd"
    pabcd = plot((plot_fig_A(i) for i in ifish_all)..., (plot_fig_B(i) for i in ifish_all)..., (plot_fig_C(i) for i in ifish_all)..., (plot_fig_D(i) for i in ifish_all)..., 
        layout= (4,length(ifish_all)),size = (900,800), titleloc = :left, left_margin = 10Plots.mm)

    pabcd = heatmap!((0:0.01:1).*ones(101,1), legend=:none, xticks=:none, yticks=(1:100:101, string.(0:0.7:0.7)), color=cgrad(cbar),
        inset = (9, bbox(-0.20, 0.05, 0.10, 2.1, :left)),subplot = 17)

    fig = pabcd
    Plots.prepare_output(fig)
    for i = 1:length(ifish_all)
        global ifish = ifish_all[i]
        include(load_data_root)
        @unpack X, C2, D, X_CCA, ROI_pjec, X_pjec = matread(joinpath(matroot,"$ifish","$n","X_CCA.mat"))
        C3 = corr2cov(C2, Cᵣ)
        id = sample(1:N,1024,replace=false)
        global pyfig = subsample_fig6(Cᵣ[id,id], fig; subfig = i)
        global pyfig = subsample_fig6(C3[id,id], pyfig; subfig = i+length(ifish_all), is_pyfig = true)
    end

    pyfig.savefig(joinpath(resultroot,"fig6_sup_abcd.png"), dpi = dpi)
    #pyfig.savefig(joinpath(resultroot,"fig6_sup_abcd.pdf"), dpi = dpi)
    pyfig.savefig(joinpath(resultroot,"fig6_sup_abcd.jpeg"), dpi = dpi)
end
#pc = plot((plot_fig_C(i) for i in ifish_all)..., layout= (1,length(ifish_all)),size = (1200,600))
#pd = plot((plot_fig_D(i) for i in ifish_all)..., layout= (1,length(ifish_all)),size = (1200,600))

#=
pcd = plot((plot_fig_C(i) for i in ifish_all)...,(plot_fig_D(i) for i in ifish_all)..., layout= (2,length(ifish_all)),size = (1200,600))
pcd = heatmap!((0:0.01:1).*ones(101,1), legend=:none, xticks=:none, yticks=(1:100:101, string.(0:0.7:0.7)), color=cgrad(cbar),
     inset = (1, bbox(-0.20, 0, 0.10, 0.3, :left)),subplot = 5)
=#
#pcd.savefig(joinpath(resultroot,"fig6_sup_cd.png")
#pcd.savefig(joinpath(resultroot,"fig6_sup_cd.pdf")
#show_fig = "ef"

if show_fig == "ef"
    #pe = plot((plot_fig_E(i) for i in ifish_all)..., layout= (1,length(ifish_all)),size = (1200,300))
    #pf = plot((plot_fig_F(i) for i in ifish_all)..., layout= (1,length(ifish_all)),size = (1200,300))
    pef = plot((plot_fig_E(i,isloglog = false) for i in ifish_all)..., (plot_fig_E(i) for i in ifish_all)..., (plot_fig_F(i) for i in ifish_all)..., 
        layout= (3,length(ifish_all)),size = (1500,900), titleloc = :left)
    Plots.savefig(joinpath(resultroot,"fig6_sup_ef.png"))
    #Plots.savefig(joinpath(resultroot,"fig6_sup_ef.pdf"))
    fig = pef
    Plots.prepare_output(fig)
    pyfig = fig.o
    pyfig.savefig(joinpath(resultroot,"fig6_sup_ef.pdf"), dpi=dpi)
    #Plots.savefig(joinpath(resultroot,"fig6_sup_ef.jpg"))
end

if show_fig == "e"
    #pe = plot((plot_fig_E(i) for i in ifish_all)..., layout= (1,length(ifish_all)),size = (1200,300))
    #pf = plot((plot_fig_F(i) for i in ifish_all)..., layout= (1,length(ifish_all)),size = (1200,300))
    pef = plot((plot_fig_E(i,isloglog = false) for i in ifish_all)..., (plot_fig_E(i) for i in ifish_all)..., 
        layout= (2,length(ifish_all)),size = (1500,600), titleloc = :left)
    Plots.savefig(joinpath(resultroot,"fig6_sup_e.png"))
    #Plots.savefig(joinpath(resultroot,"fig6_sup_ef.pdf"))
    fig = pef
    Plots.prepare_output(fig)
    pyfig = fig.o
    pyfig.savefig(joinpath(resultroot,"fig6_sup_e.pdf"), dpi=dpi)
    #Plots.savefig(joinpath(resultroot,"fig6_sup_ef.jpg"))
end

if show_fig == "f"
    ifish_all = [201106,201116,201117,201125,201126,210106]
    #pe = plot((plot_fig_E(i) for i in ifish_all)..., layout= (1,length(ifish_all)),size = (1200,300))
    #pf = plot((plot_fig_F(i) for i in ifish_all)..., layout= (1,length(ifish_all)),size = (1200,300))
    pef = plot((plot_fig_F(i) for i in ifish_all)..., 
        layout= (2,3),size = (900,600), titleloc = :left)
    Plots.savefig(joinpath(resultroot,"fig6_sup_f.png"))
    #Plots.savefig(joinpath(resultroot,"fig6_sup_ef.pdf"))
    fig = pef
    Plots.prepare_output(fig)
    pyfig = fig.o
    pyfig.savefig(joinpath(resultroot,"fig6_sup_f.pdf"), dpi=dpi)
    #Plots.savefig(joinpath(resultroot,"fig6_sup_ef.jpg"))
end

#show_fig = "gh"

if show_fig == "gh"
    pg = plot((plot_fig_G(i) for i in ifish_all)..., layout= (length(ifish_all),1),size = (500,2250), titleloc = :left)
    ph = plot((plot_fig_H(i) for i in ifish_all)..., layout= (length(ifish_all),1),size = (700,2250), titleloc = :left)
    #pgh = plot((plot_fig_G(i) for i in ifish_all)...,(plot_fig_H(i) for i in ifish_all)..., layout= (length(ifish_all),2), size = (600,2250), titleloc = :left)
    #pgh = plot(([plot_fig_G(i), plot_fig_H(i)] for i in ifish_all)..., layout= (length(ifish_all),2), size = (600,2250), titleloc = :left)
    lgh = @layout [a{0.43w} b{0.55w}]
    pgh = plot(pg,ph,layout = lgh,size = (1300,2250))
    Plots.savefig(joinpath(resultroot,"fig6_sup_gh.png"))
    Plots.savefig(joinpath(resultroot,"fig6_sup_gh.pdf"))
    #fig = pgh
    #Plots.prepare_output(fig)
    #pyfig = fig.o
    #pyfig.savefig(joinpath(resultroot,"fig6_sup_gh.jpeg"), dpi=dpi)
    #pyfig.savefig(joinpath(resultroot,"fig6_sup_gh2.pdf"), dpi=dpi)
end
