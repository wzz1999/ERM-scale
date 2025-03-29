dpi = 300

include("../src/util.jl")
include("../src/subsampling_functions.jl")


pyplot()
matroot = "ERM-scale/results/mds_CCA"
resultroot = "ERM-scale/figures"
bsonroot = "ERM-scale/results/mds_CCA/visualization"
ifish = Int(210106)
include("../load_data/load_fish_data2.jl")
n = 2


@unpack X, C2, D, X_CCA, ROI_pjec, X_pjec = matread(joinpath(matroot,"$ifish","$n","X_CCA.mat"))
@unpack corr_dim, corr_shuffle= matread(joinpath(matroot,"$ifish","corr_dim.mat"))
C3 = corr2cov(C2, Cᵣ)
X = X["parent"]'
D2 = pairwise(Euclidean(), X', dims=1)


include("kernel_regression.jl")
using PyPlot
plt.rcParams["figure.dpi"] = dpi
plot = Plots.plot
ip_file = string(bsonroot, "/", "permutation_index_", "all", "_", "$ifish", "_", "2021-07-02", ".bson")
if isfile(ip_file)
    BSON.@load ip_file ip
else
    R = hclust(D,linkage = :average)
    ip = R.order
end


###################################################################################################################
n =2
ϵ = 0.03125
d = n
ρ, μ = Parameter_estimation(Corr, n = n)
N = size(Corr)[1]
L = (N/ρ)^(1/n)
p = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = 0.03125, μ = μ, ξ = 10^18, β = 0, σ̄² = 1, σ̄⁴ = 1)
points = rand(Uniform(-L/2,L/2),N,d)
LogNormal_param = fit_mle(LogNormal, diag(Cᵣ)) 
σ² = vec(rand(LogNormal_param,N))
σ = vec(broadcast(√,σ²))
Δ = diagm(σ)
Correrm = reconstruct_covariance(points, p, subsample = false)
C_erm = Δ*Correrm*Δ

###################################################################################################################

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

R = hclust(D,linkage = :average)


l = @layout [
    [a{0.5h} b{0.5h}; c{0.5h} d{0.5h}] [h{0.5h} ; e{0.5h} f{0.5h} ] g{1h} 
]


l123456 = @layout [
    [a b] 
        [c d]  
            [e f]
]
λ1 = eigvals(Corr)
λ1 = sort(λ1,rev = true)
λ2 = eigvals(C2)
λ2 = sort(λ2,rev = true)
top_range = 1:N


##############
p1 = plot(title = "A",grid = :none,xlabel = "rank/N", ylabel = "eigenvalue "* L"\lambda", dpi = dpi)
p1 = annotate!(10^-1.5,10^2.1, Plots.text("Data",10))


p2 = plot(title = "B",grid = :none,xlabel = "rank/N", ylabel = "eigenvalue "* L"\lambda", dpi = dpi)
 
p2 = annotate!(10^-1.5,10^2.35, Plots.text("Fitted ERM",10))
p2 = annotate!(10^-1.5,10^2.1, Plots.text("(with MDS coordinates)",7))

p6 = plot(title = "C",grid = :none,xlabel = "rank/N", ylabel = "eigenvalue "* L"\lambda", dpi = dpi)
p6 = annotate!(10^-1.5,10^2.1, Plots.text("Fitted ERM",10))

cbar = :thermal

c_max =0.5
p3 = heatmap(Cᵣ[ip, ip],theme = :dark, clim = (0,c_max),ratio=:equal,title = "D",framestyle = :none,legend = :none, color=cgrad(cbar), dpi = dpi)
p4 = heatmap(C3[ip, ip],theme = :dark, clim = (0,c_max),ratio=:equal,title = "E",framestyle = :none,legend = :none, color=cgrad(cbar), dpi = dpi)




p5 = kernelized_fit_and_plot(GaussianKernel())
use_dim = 5

p123456 = plot(p1,p3,p2,p4,p6,p5, layout = l123456, size=(400,600))

p123456 = heatmap!((0:0.01:1).*ones(101,1), legend=:none, xticks=:none, yticks=(1:100:101, string.(0:c_max:c_max)), color=cgrad(cbar),
    inset = (2, bbox(-0.20, 0.02, 0.10, 0.94, :left)),subplot = 7)
p123456 = heatmap!((0:0.01:1).*ones(101,1), legend=:none, xticks=:none, yticks=(1:100:101, string.(0:c_max:c_max)), color=cgrad(cbar),
    inset = (4, bbox(-0.20, 0.02, 0.10, 0.94, :left)),subplot = 8)

    p7 = Plots.scatter(X[1,:],X[2,:], markersize = 2,grid = :none, label = :none, title = "G", ratio=:equal,#size=(400,300),
    marker_z= (ROIxyz*ROI_pjec)[:,1],markercolor = :PRGn, markerstrokecolor = :white, markerstrokewidth = 0, dpi = dpi)

X01 = [-8,20]
k01 = 70

p7 = arrow_color(X01, k01, X_pjec,color_map = palette(cc))
p7 = annotate!(-12,15, L"\vec{a}_1")

p8 = Plots.scatter(ROIxyz[:,1],ROIxyz[:,2], marker_z = (X_CCA[:,1]), markersize = 1.5, dpi = dpi,ylims = (0,300), ratio=:equal,  
    markercolor = palette(cc), markerstrokecolor = :white, label = "",clim = (-2+median(X_CCA[:,1]),2+median(X_CCA[:,1])),title = "H",grid = :none, markerstrokewidth = 0)
p8 = annotate!(170,65-9, Plots.text("Optic tectum",10))
p8 = annotate!(100,100, Plots.text("Forebrain",10))
p8 = annotate!(370-5,110-20, Plots.text("Hindbrain",10))
p8 = plot_scale_fig6(dpi=dpi)
X02 = [60,203]
k02 = 10000

p8 = arrow_color(X02, k02, ROI_pjec,color_map = palette(palette_colormap(:PRGn)))
p8 = annotate!(100,230+10, L"\vec{b}_1")
l12345678 = @layout [a{0.55w} [b{0.93w} 
                                c{0.93w}] ]
p12345678 = plot(p123456,p7, p8,layout = l12345678, size=(800,600), titleloc = :left, left_margin = 10Plots.mm)


id = sample(1:N,1024,replace=false)
fig = p12345678
Plots.prepare_output(fig)

fts = 5.5
pyfig = subsample_fig6(Cᵣ[id,id], fig; subfig = 1, fontsize =fts )
pyfig = subsample_fig6(C3[id,id], pyfig; subfig = 3, is_pyfig = true, fontsize =fts )
pyfig = subsample_fig6(C_erm[id,id], pyfig; subfig = 5, is_pyfig = true, fontsize =fts )

pyfig.savefig(joinpath(resultroot,"fig6.png"), dpi =dpi)
#pyfig.savefig(joinpath(resultroot,"fig6.jpeg"), dpi =dpi)
