using DataFrames
import GLM as glm
import PyPlot as plt

const ERM_ROOT = "/home/wenlab-user/wangzezhen/ERM-scale"
const FIGURE_ROOT = "/home/wenlab-user/wangzezhen/ERM-scale/fig_plot"

#################################
##-------different f(x)--------##
#################################

function f_r_power_law(r, p::ERMParameter)
    @unpack μ, ϵ = p
    y = ϵ^μ.*(r).^(-μ)
    return y
end

function f_r_power_law_tent(r, p::ERMParameter;flag_tangent=true) # distance function
    # y = pdf.(TD,r)
    @unpack μ, ϵ, β = p

    c = (1+μ)^(1/μ)
    ϵ2 = c * ϵ
    k = (c^(-μ)-1)/(ϵ2)
    if ~flag_tangent
        k = k / 2
        f_singularity_zero_x = x -> k*x+1-ϵ^μ/x^μ
        fx = ZeroProblem(f_singularity_zero_x,ϵ/2)
        ϵ2 = solve(fx)
        
    end

    if length(r) > 1
        y = copy(r)   
        y[r .<= ϵ2] .= k.* r[r .<= ϵ2] .+ 1

        y[r .> ϵ2] .= ϵ^μ*(r[r .> ϵ2]).^(-μ).*exp.(-(r[r .> ϵ2] .- ϵ2)*β)
    else
        if r <= ϵ2
            y = k * r + 1
            # k = (ϵ^μ*ϵ2.^(-μ)-1)/(ϵ2)^2
            # y = k * (r^2) + 1 
        else
            y = ϵ^μ*(r)^(-μ)*exp(-(r - ϵ2)*β)
        end
    end
    return y
end

function f_r_power_law_parabola(r, p::ERMParameter) # distance function
    # y = pdf.(TD,r)
    @unpack μ, ϵ, β = p

    f_singularity_zero_x = x -> 2*x*(ϵ^μ/x^μ - 1)/x^2 - ϵ^μ*(-μ)*x^(-μ-1)
    fx = ZeroProblem(f_singularity_zero_x,ϵ)
    ϵ2 = solve(fx)       
    k = (ϵ^μ*ϵ2.^(-μ)-1)/(ϵ2)^2
    # c = (1+μ)^(1/μ)
    # ϵ2 = c * ϵ
    # k = (ϵ^μ/ϵ2^μ - 1)/ϵ2^2
    if length(r) > 1
        y = copy(r)   
        y[r .<= ϵ2] .= k.* (r[r .<= ϵ2]).^2 .+ 1

        y[r .> ϵ2] .= ϵ^μ*(r[r .> ϵ2]).^(-μ).*exp.(-(r[r .> ϵ2] .- ϵ2)*β)
    else
        if r <= ϵ2
            y = k * (r^2) + 1 
        else
            y = ϵ^μ*(r)^(-μ)*exp(-(r - ϵ2)*β)
        end
    end
    return y
end



function f_r_power_law_flat(r, p::ERMParameter) # distance function
    @unpack μ, ϵ, β = p

    if length(r) > 1
        y = copy(r)   
  
        y[r .<= ϵ] .= 1

        y[r .> ϵ] .= ϵ^μ*(r[r .> ϵ]).^(-μ).*exp.(-(r[r .> ϵ] .- ϵ)*β)
    else
        if r <= ϵ
            y = 1
        else
            y = ϵ^μ*(r)^(-μ)*exp(-(r - ϵ)*β)
        end
    end
    return y
end

function f_r_t_pdf(r, p::ERMParameter)
    @unpack μ, ϵ, β = p
    c = (1/ϵ^μ)^(-2/μ)
    y = ϵ^μ.*(c .+ r.^2).^(-μ/2)
    return y
end

function f_r_exp(r, p::ERMParameter)
    @unpack β = p
    y = exp.(-(r)*β)
    return y
end

function f_r_exp_eta(r, p::ERMParameter)
    @unpack μ, ϵ, β = p
    y = exp.(-(r.^μ))
    return y
end

function f_r_gaussian(r, p::ERMParameter)
    # y = pdf.(Normal(),r)
    y = exp.(-0.5.*r.^2)
end



#################################
##---- data preprocessing -----##
#################################
function normalize_activity(a)
    N = size(a,1)
    epsilon = 1e-4
    a[a.<epsilon] .= 0
    for i = 1:N
        IDX = a[i,:] .> 0
        mean_activity = mean(a[i,IDX]);
        if mean_activity > 0
            a[i,:] = a[i,:]./mean_activity
        end
    end
    return a
end


function covariance_preprocessing(C::Matrix)
    s = diag(C)
    s = mean(s)
    C = C ./ s
    return C
end

function data_preprocessing_and_subsample(FR::Matrix;K = 1024, flag_rand_seed=false, flag_cov = true, flag_same_time_frame=false)

    tf = vec(sum(FR, dims=2) .> 0)
    FR = convert(Array{Float64}, FR[tf, :])
    FR = normalize_activity(FR);

    # K = 1024
    if flag_rand_seed
        Random.seed!(1)
    end
    i_rand = randperm(size(FR, 1))[1:K]
    FR_sub = FR[i_rand,:]
    if flag_same_time_frame
        FR_sub = FR_sub[:, 1:7200]
    end
    if flag_cov
        C = cov(convert(Matrix, FR_sub'))
    else
        C = cor(convert(Matrix, FR_sub'))
    end    

    s = diag(C)
    s = mean(s)
    C = C ./ s

    return FR_sub, C
end

function data_preprocessing(FR::Matrix)

    tf = vec(sum(FR, dims=2) .> 0)
    if sum(tf) < length(tf)
        @warn "There are neurons with zero activity!"
    end
    FR = convert(Array{Float64}, FR[tf, :])
    FR = normalize_activity(FR);


    FR_sub = FR
    C = cov(convert(Matrix, FR_sub'))
    s = diag(C)
    s = mean(s)
    C = C ./ s

    return FR_sub, C
end


#################################
##-------subsampling-----------##
#################################

function subsampling(C)
    N = size(C,1)
    neural_set = random_clusters(N)
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))

    fig1, ax1 = plt.subplots()
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []

    fig2, ax2 = plt.subplots()
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel("eigenvalue " * L"\lambda", fontsize=8)
    ax2.set_ylabel("probability density" * L"p(\lambda)",fontsize=8)
    i = 0
    for n in range(iterations, 7, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=1,marker="o",color=colorList[i,:],
        ms=2,label=@sprintf("N=%d",2^n))   
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3)

        id, λ = dropzeros_and_less(eigenvalues)
        N = size(C,1)
        length = 50

        h_λ = fit(Histogram, λ, LogRange(minimum(λ),maximum(λ),length))
        h_λ = normalize(h_λ, mode=:pdf)

        λᵣ, p_sim = dropzero(h_λ)
        # p_sim = p_sim.*size(id,1)/2^n
        ax2.plot(λᵣ, p_sim, linewidth=1,color=colorList[i,:],label=@sprintf("N=%d",2^n))
        
        append!(eigenspectrums,eigenvalues[4:end])
        append!(nranks, k[4:end])

    end
    ax1.set_ylim([1e-2, 1e+3])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N",fontsize=8)
    ax1.set_ylabel("eigenvalue" * L"(\lambda)",fontsize=8)
    ax1.legend(frameon=false)

    ax2.legend(frameon=false)

    println("The data covariance matrix eigenspectrum:")
    IDX = (nranks .< 0.1) .& (nranks .> 4/N)
    slope, ranks, eig_fit = powerlaw_fit(nranks, eigenspectrums, IDX)
    push!(leg2, "data fit")
    ax1.text(0.01, 50, ["slope(data):", string(slope)], fontsize=12)
end
    
function subsampling_subplot(C,ax1,ax2)
    N = size(C,1)
    neural_set = random_clusters(N)
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []

    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel("eigenvalue " * L"\lambda",fontsize=8)
    ax2.set_ylabel("probability density " * L"p(\lambda)",fontsize=8)
    i = 0
    for n in range(iterations, 7, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.5,marker="o",color=colorList[i,:],
        ms=0.6,label=@sprintf("N=%d",2^n))   
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color=colorList[i,:])

        id, λ = dropzeros_and_less(eigenvalues)
        N = size(C,1)
        length = 50

        h_λ = fit(Histogram, λ, LogRange(minimum(λ),maximum(λ),length))
        h_λ = normalize(h_λ, mode=:pdf)

        λᵣ, p_sim = dropzero(h_λ)
        # p_sim = p_sim.*size(id,1)/2^n
        ax2.plot(λᵣ, p_sim, linewidth=1,color=colorList[i,:],label=@sprintf("N=%d",2^n))
        
        append!(eigenspectrums,eigenvalues[4:end])
        append!(nranks, k[4:end])

    end
    ax1.set_ylim([1e-1, 1e+2])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N",fontsize=8)
    ax1.set_ylabel("eigenvalue " * L"\lambda",fontsize=8)
    ax1.legend(borderpad=0.5,handlelength=0.6,frameon=false)

    ax2.legend(borderpad=0.5,handlelength=0.3,frameon=false)

    println("The data covariance matrix eigenspectrum:")
    IDX = (nranks .< 0.1) .& (nranks .> 4/N)
    slope, ranks, eig_fit = powerlaw_fit(nranks, eigenspectrums, IDX)
    push!(leg2, "data fit")
    # ax1.text(0.01, 50, ["slope(data):", string(slope)], fontsize=12)
    return ax1, ax2
end


function subsampling_fitting(C)
    N = size(C,1)
    neural_set = random_clusters(N)
    iterations = floor(log2(N))

    eigenspectrums = []
    nranks = []

    i = 0
    for n in range(iterations, 7, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]        
        append!(eigenspectrums,eigenvalues[4:end])
        append!(nranks, k[4:end])
    end

    println("The data covariance matrix eigenspectrum:")
    IDX = (nranks .< 0.1) .& (nranks .> 4/N)
    slope, r_sq, _, _, _ = powerlaw_fit(nranks, eigenspectrums, IDX)
    return slope, r_sq
#     IDX = (nranks .< 0.1) .& (nranks .> 4/N)
#     slope, xmin = plfit(nranks, eigenspectrums, IDX)
    
#     return slope, xmin

end

function plfit(x, y, range)
    y = y[range]
    y = map(float, y)
    mat"""
        matlabroot = getenv('MATLAB_PROJECT');
        addpath([matlabroot, '/code/RG/functions']);
        [$alpha,$xmin] = plfit2($y,'finite',1);
    """
    return alpha, xmin
end
    

function subsampling_rank_subplot(C,ax1)
    N = size(C,1)
    neural_set = random_clusters(N)
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []
    i = 0
    for n in range(iterations, iterations-3, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.5,marker="o",color=colorList[i,:],
        ms=0.3,label=@sprintf("N=%d",2^n))   
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color=colorList[i,:])
    end
    ax1.set_ylim([1e-2, 1e+3])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N", fontsize=8)
    ax1.set_ylabel("eigenvalue " * L"\lambda", fontsize=8)
    return ax1
end

# the function to make animation of the figure
function subsampling_rank_subplot2(C,ax1)
    N = size(C,1)
    neural_set = random_clusters(N)
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []
    i = 0
    for n in range(iterations, iterations-3, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.5,marker="o",color=colorList[i,:],
        ms=0.3,label=@sprintf("N=%d",2^n))   
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color=colorList[i,:])
    end
    ax1.set_ylim([1e-2, 1e+3])
    # c_diag = sort(diag(C), rev=true)
    # ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N", fontsize=8)
    ax1.set_ylabel("eigenvalue " * L"\lambda", fontsize=8)
    return ax1
end

function subsampling_rank_subplot_all_neuron(C,ax1)
    N = size(C,1)
    neural_set = random_clusters(N)
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []
    i = 0
    for n in range(1, 4)
        i+=1
        eigenvalues, errorbars, k, K = pca_cluster3(neural_set, C, n)
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.2, marker="o",color=colorList[i,:],
        ms=0.2,label=@sprintf("N=%d",K))   
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color=colorList[i,:])
    end
    ax1.set_ylim([1e-2, 1e+3])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N", fontsize=8)
    ax1.set_ylabel("eigenvalue " * L"\lambda", fontsize=8)
    return ax1
end

function subsampling_comparison_rank_subplot(C, C2, C3, n, ax1)
    N = size(C,1)
    neural_set = random_clusters(N)
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    # colorList = cmap(range(0, 1, length=4))
    # colorList = ["red","blue","green"]
    colorList = ["#FF5733","#178BF1","#37C618"]


    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []
    i = 0
    # for n in range(iterations, 7, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.5,marker="o",color=colorList[1],
            ms=0.6,label="C_exp")   
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        # ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color=colorList[i,:])
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C2, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.5,marker="o",color=colorList[2],
        ms=0.6,label="C2")   
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        # ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color=colorList[i,:])
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C3, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.5,marker="o",color=colorList[3],
        ms=0.6,label="C3")   
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        # ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color=colorList[i,:])
    # end
    ax1.set_ylim([1e-2, 1e+3])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker=".",label="diagonal_exp",color=colorList[1],alpha=0.3,s=2)
    c_diag = sort(diag(C2), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal_2",color=colorList[2],alpha=0.3,s=2)
    c_diag = sort(diag(C3), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="+",label="diagonal_3",color=colorList[3],alpha=0.3,s=2)
    # ax1.set_xlabel("rank/N", fontsize=8)
    # ax1.set_ylabel("eigenvalue " * L"\lambda", fontsize=8)
    return ax1
end


function random_clusters(N)
    iterations = Int(floor(log2(N)))
    cluster_set = Vector{Any}(undef, iterations+1)
    L = N
    cluster_set[1] = Vector{Any}(undef, L)
    y = randperm(N)
    for k = 1:L
        cluster_set[1][k] = y[k]
    end
    for n = 2:iterations+1
        L = Int(floor(L/2))
        cluster_set[n] = Vector{Any}(undef, L)
        for k = 1:L
            cluster_set[n][k] = [cluster_set[n-1][2*k-1]; cluster_set[n-1][2*k]]
        end
    end
    return cluster_set
end
    

function pca_cluster2(cluster_set, C_original, K)
    n = Int(log2(K) + 1)
    N = length(cluster_set[n])
    eigenspectrum = zeros(K,N)
    for i in 1:N
        IDX = cluster_set[n][i]
        c = C_original[IDX,:]
        c = c[:,IDX]
        LAMBDA = sort(eigvals(Hermitian(c)), rev=true)
        LAMBDA = LAMBDA[LAMBDA .> 0]
        ranks = length(LAMBDA)
        eigenspectrum[1:ranks,i] = LAMBDA
    end
    eigen = mean(eigenspectrum, dims=2)
    errorbars = 2 * std(eigenspectrum, dims=2) / sqrt(N) # 2*SEM is a history issue of the code. It has been fixed when plotting (by divide 2 again before plotting)
    k = (1:K) / K
    return eigen, errorbars, k
end


function pca_cluster3(cluster_set, C_original, n)
    # !!! only for the case when that it is random subsampling !!!
    # attention: this function is for all neurons when the number of neuron is not the power of 2
    # it can also be used when number of neurons is the power of 2 (only for the case that it is random subsampling)
    # the order of cluster_set is reversed compared to pca_cluster2
    K = length(cluster_set[n])
    
    cluster_set_n = hcat(cluster_set[n]...)'
    Ncol = size(cluster_set_n,2)
    eigenspectrum = zeros(K,Ncol)
    
    for i in 1:Ncol
        IDX = cluster_set_n[:,i]
        c = C_original[IDX,:]
        c = c[:,IDX]
        LAMBDA = sort(eigvals(Hermitian(c)), rev=true)
        LAMBDA = LAMBDA[LAMBDA .> 0]
        ranks = length(LAMBDA)
        eigenspectrum[1:ranks,i] = LAMBDA
    end
    eigen = mean(eigenspectrum, dims=2)
    errorbars = 2 * std(eigenspectrum, dims=2) / sqrt(Ncol)
    k = (1:K) / K
    return eigen, errorbars, k, K
end

# function powerlaw_fit(x, y, range)
#     if size(x, 1) == 1
#         x = transpose(x)
#     end
#     if size(y, 1) == 1
#         y = transpose(y)
#     end
#     p = Polynomials.polyfitA(log.(x[range]), log.(y[range]), 1)
#     println("slope = $(p[1])")
#     slope = p[1]
#     x_fit = logspace(log10(minimum(x)), log10(maximum(x)), 10)
#     y_fit = exp(p[2]) .* x_fit .^ p[1]
#     return slope, x_fit, y_fit, p
# end

function powerlaw_fit(x, y, range)
    if size(x, 1) == 1
        x = transpose(x)
    end
    if size(y, 1) == 1
        y = transpose(y)
    end
    p, r_sq = linearFit(log.(x[range]), log.(y[range]))
    x_fit = logspace(log10(minimum(x)), log10(maximum(x)), 10)
    y_fit = exp(p[1]) .* x_fit .^ p[2] # p[1]: y-intercept, p[2]: slope
    slope = p[2]
    return slope, r_sq, x_fit, y_fit, p
end

function linearFit(x,y)
    df = DataFrame(X=x,Y=y)
    fm = glm.@formula(Y ~ X)
    linearRegressor = glm.lm(fm, df)
    r_sq = glm.r2(linearRegressor)
    coeffs = glm.coef(linearRegressor)
    return coeffs, r_sq
end




function logspace(start, stop, n)
    return exp.(range(start, stop, length=n))
end

function mean_sem_nonzero(mat::Matrix;dims=1)
    # mean across multiple simulations: non-zero values are included
    nrow,ncol = size(mat)
    
    if dims == 1 # the mean of rows
        m = zeros(ncol)
        sem = zeros(ncol)
        for i = 1:ncol
            tf = mat[:,i] .!= 0 
            m[i] = mean(mat[tf,i])
            sem[i] = std(mat[tf,i])/sqrt(length(mat[tf,i]))
        end
    elseif dims == 2 # the mean of columns
        m = zeros(nrow)
        sem = zeros(nrow)
        for i = 1:nrow
            tf = mat[i,:] .!= 0 
            m[i] = mean(mat[i,tf])
            sem[i] = std(mat[i,tf])/sqrt(length(mat[i,tf]))
        end
    end 
    return m,sem
end

using LinearAlgebra, Random, Statistics
function collapse_index_rank(C;isData=true)
    

    # Input:
    # Covariance C,
    # large eig cut-off percentile p0
    # sampling ratio s = Ns/N
    # N = 1000
    # ggg = 0.5
    # C = I(N) - ggg*randn(N,N)/sqrt(N)
    # C = inv(C)
    # C = C * C'
    # C *= 1-ggg^2
    s = 0.5
    p0 = 0.01

    N,_ = size(C)
    Ns = round(Int, N*s)
    lb_N = -sort(-eigvals(Hermitian(C)))
    if isData
        ntrial = 2000
    else
        ntrial = 20
    end
    lb_Ns = zeros(Ns,ntrial)
    for i in 1:ntrial
        i_s = randperm(N)[1:Ns]
        Cs = copy(C[:,i_s])
        Cs = Cs[i_s,:]
        lb_Ns[:,i] = -sort(-eigvals(Hermitian(Cs)))
    end
    lb_Ns = vec(mean(lb_Ns,dims=2))

    r0 = round(Int, Ns*p0-0.5)
    r1 = maximum(findall(lb_Ns .> 1))
    f_Ns = log.(lb_Ns[r0+1:r1])
    f_N = zeros(length(f_Ns))
    for k in 1:length(f_N)
        i = r0 + k
        j = (i)/Ns*N
        if isinteger(j)
            j = convert(Int,j)
            f_N[k] = log.(lb_N[j])
        else
            j0 = convert(Int,floor(j))
            # f_N[k] = (j0+1-j)*lb_N[j0] + (j-j0)*lb_N[j0+1]

            x = log(j/N)
            x0 = log(j0/N)
            x1 = log((j0+1)/N)
            y0 = log(lb_N[j0])
            y1 = log(lb_N[j0+1])
            f_N[k] = (y0*(x1-x) + y1*(x-x0)) / (x1-x0)
        end
    end
    S = 0
    for k in 1:length(f_N)-1
        i = r0 + k
        a = (i)/Ns
        b = (i+1)/Ns
        y1 = abs(f_N[k] - f_Ns[k])
        y2 = abs(f_N[k+1] - f_Ns[k+1])
        S += log(b/a)*(y1+y2)/2
    end
    CI = S/log(N/Ns)/log((r1)/(r0+1))
    return CI, (r0+1)/Ns, r1/Ns
end


# deprecated code
# function parse_required_arg(argname)
#     #=
#     Search the global `ARGS` array for an argument that matches `argname` and return its associated value. 
#     This function expects arguments to be in the form of `--argname value`. It will return the `value` associated 
#     with the given `argname`. If the `argname` is not found, the function will throw an error, indicating that 
#     the argument is required.

#     # Arguments
#     - `argname::String`: The name of the command-line argument to find.

#     # Returns
#     - `String`: The value associated with the `argname` in the command-line arguments.

#     # Example
#     ```julia
#     # If you run your script as follows:
#     # julia script.jl --data_type light_field

#     # You can use the function in the script to get 'image':
#     data_type = parse_required_arg("--data_type")
#     println("The specified data type is: $(data_type)")
#     ```
#     =#

#     found_arg = nothing
#     for i in 1:2:length(ARGS)-1
#         if ARGS[i] == argname
#             found_arg = ARGS[i+1]
#             break
#         end
#     end
#     if isnothing(found_arg)
#         error("Argument $(argname) is required")
#     end
#     return found_arg
# end


