##
# just a backup. Don't need to use this functions
##

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
    ax2.set_xlabel("eigenvalue" * L"(\lambda)",fontsize=8)
    ax2.set_ylabel(L"p(\lambda)",fontsize=8)
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
    ax1.legend()

    ax2.legend()

    println("The data covariance matrix eigenspectrum:")
    IDX = (nranks .< 0.1) .& (nranks .> 4/N)
    exponent, ranks, eig_fit = powerlaw_fit(nranks, eigenspectrums, IDX)
    push!(leg2, "data fit")
    ax1.text(0.01, 50, ["exponent(data):", string(exponent)], fontsize=12)
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
    ax2.set_xlabel("eigenvalue" * L"(\lambda)",fontsize=8)
    ax2.set_ylabel(L"p(\lambda)",fontsize=8)
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
        p_sim = p_sim.*size(id,1)/2^n
        ax2.plot(λᵣ, p_sim, linewidth=1,color=colorList[i,:],label=@sprintf("N=%d",2^n))
        
        append!(eigenspectrums,eigenvalues[4:end])
        append!(nranks, k[4:end])

    end
    ax1.set_ylim([1e-2, 1e+3])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N",fontsize=8)
    ax1.set_ylabel("eigenvalue" * L"(\lambda)",fontsize=8)
    ax1.legend()

    ax2.legend()

    println("The data covariance matrix eigenspectrum:")
    IDX = (nranks .< 0.1) .& (nranks .> 4/N)
    exponent, ranks, eig_fit = powerlaw_fit(nranks, eigenspectrums, IDX)
    push!(leg2, "data fit")
    # ax1.text(0.01, 50, ["exponent(data):", string(exponent)], fontsize=12)
    return ax1, ax2
end

function random_clusters(N)
    iterations = Int(log2(N))
    cluster_set = Vector{Any}(undef, iterations+1)
    L = N
    cluster_set[1] = Vector{Any}(undef, L)
    y = randperm(N)
    for k = 1:L
        cluster_set[1][k] = y[k]
    end
    for n = 2:iterations+1
        L = Int(L/2)
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
    errorbars = 2 * std(eigenspectrum, dims=2) / sqrt(N)
    k = (1:K) / K
    return eigen, errorbars, k
end

function pca_cluster2_multi_sim(cluster_set, C_original, K)
    # multi simulations
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
    # eigen = mean(eigenspectrum, dims=2)
    # errorbars = 2 * std(eigenspectrum, dims=2) / sqrt(N)
    k = (1:K) / K
    return eigenspectrum, k
end

function powerlaw_fit(x, y, range)
    if size(x, 1) == 1
        x = transpose(x)
    end
    if size(y, 1) == 1
        y = transpose(y)
    end
    p = Polynomials.polyfitA(log.(x[range]), log.(y[range]), 1)
    println("exponent = $(p[1])")
    exponent = p[1]
    x_fit = logspace(log10(minimum(x)), log10(maximum(x)), 10)
    y_fit = exp(p[2]) .* x_fit .^ p[1]
    return exponent, x_fit, y_fit, p
end

function logspace(start, stop, n)
    return exp.(range(start, stop, length=n))
end

function mean_sem_nonzero(mat::Matrix;dims=1)
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

function subsampling_multi_sim_subplot(ax1,ax2,p,n_sim)
    N = p.N
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
    # n_sim = 10
    for n in range(iterations, 7, step=-1)
        i+=1
        points = rand(Uniform(-p.L/2,p.L/2),p.N,p.n) #points are uniformly distributed in a region 
        C = reconstruct_covariance(points, p, subsample = false)
        eigenspectrum, k = pca_cluster2_multi_sim(neural_set, C, Int(2^n))
        eigenspectrum_all = copy(eigenspectrum)
        Threads.@threads for ii = 2:n_sim
            points = rand(Uniform(-p.L/2,p.L/2),p.N,p.n) #points are uniformly distributed in a region 
            C = reconstruct_covariance(points, p, subsample = false)
            eigenspectrum, k = pca_cluster2_multi_sim(neural_set, C, Int(2^n))
            eigenspectrum_all = hcat(eigenspectrum_all,eigenspectrum)
        end
        eigenvalues = mean(eigenspectrum_all, dims=2)
        errorbars = std(eigenspectrum_all, dims=2) ./ sqrt(size(eigenspectrum_all,2))

        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.6,marker="o",color=colorList[i,:],
        ms=1,label=@sprintf("N=%d",2^n))   
        λ_err_l = vec(eigenvalues .- errorbars) # lower
        λ_err_u = vec(eigenvalues .+ errorbars) # upper
        print(size(λ_err_l))
        print(n)
        ax1.fill_between(k[1:end],λ_err_l[1:end],λ_err_u[1:end],alpha=0.3,color=colorList[i,:])
        
        λ_min = minimum(eigenspectrum[:])
        λ_max = maximum(eigenspectrum[:])
        
        id, λ = dropzeros_and_less(eigenspectrum_all[:,1])
        N = size(C,1)
        length = 200
        h_λ = fit(Histogram, λ, LogRange(λ_min,λ_max,length))
        # edges, p_sim = normalize(h_λ, mode=:pdf)
        h_λ = normalize(h_λ, mode=:pdf)
        @unpack edges, weights = h_λ       
        binsize = edges[1][2] - edges[1][1]
        x = collect(edges[1][2:end].-binsize/2)
        p_sim_all = vec(weights.*size(id,1)/2^n)
        
        for ii = 2:size(eigenspectrum_all,2)
            id, λ = dropzeros_and_less(eigenspectrum_all[:,ii])
            N = size(C,1)
            length = 200
            h_λ = fit(Histogram, λ, LogRange((λ_min),maximum(λ_max),length))
            h_λ = normalize(h_λ, mode=:pdf)
            @unpack edges, weights = h_λ   
            binsize = edges[1][2] - edges[1][1]
            x = collect(edges[1][2:end].-binsize/2)
            p_sim = vec(weights.*size(id,1)/2^n)
            p_sim_all = hcat(p_sim_all,p_sim)
        end
        
        p_sim,errorbars = mean_sem_nonzero(p_sim_all, dims=2)
        # errorbars = std(p_sim_all, dims=2) ./ sqrt(size(p_sim_all,2))
        p_err_l = p_sim .- errorbars
        p_err_u = p_sim .+ errorbars
        
        ax2.plot(x, p_sim, linewidth=1,color=colorList[i,:],label=@sprintf("N=%d",2^n))        
        ax2.fill_between(x,p_err_l[1:end],p_err_u[1:end],alpha=0.3,color=colorList[i,:])

        


    end
    
    points = rand(Uniform(-p.L/2,p.L/2),p.N,p.n) #points are uniformly distributed in a region 
    C = reconstruct_covariance(points, p, subsample = false)

    ax1.set_ylim([1e-2, 1e+3])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N",fontsize=8)
    ax1.set_ylabel("eigenvalue" * L"(\lambda)",fontsize=8)
    ax1.legend()

    ax2.legend()

    println("The data covariance matrix eigenspectrum:")
    # IDX = (nranks .< 0.1) .& (nranks .> 4/N)
    # exponent, ranks, eig_fit = powerlaw_fit(nranks, eigenspectrums, IDX)
    # push!(leg2, "data fit")
    # ax1.text(0.01, 50, ["exponent(data):", string(exponent)], fontsize=12)
    return ax1, ax2
end

###########
# reconstruct covariance, option for corr or cov
###########
function reconstruct_covariance2(p::ERMParameter;different_neural_variability=false)
    if different_neural_variability
        σ² = vec(rand(LogNormal(0,0.5),N,1))
    else
        σ² = ones(N,1)
    end

    σ² = σ²/mean(σ²) #normalize so that σ̄² = 1
    σ = vec(broadcast(√,σ²))
    Δ = diagm(σ)
    points = rand(Uniform(-p.L/2,p.L/2),p.N,p.n) #points are uniformly distributed in a region 
    C = reconstruct_covariance(points, p, subsample = false)
    Ĉ = Δ*C*Δ;
    return Ĉ
end


function multi_sim_subplot(ax,p,n_sim;different_neural_variability=false)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("eigenvalue " *L"\lambda",fontsize=8)
    ax.set_ylabel("probability density " * L"p(\lambda)",fontsize=8)
    # n_sim = 10
    neural_set = random_clusters(N)
    points = rand(Uniform(-p.L/2,p.L/2),p.N,p.n) #points are uniformly distributed in a region 
    C = reconstruct_covariance2(p,different_neural_variability= different_neural_variability)
    eigenspectrum, k = pca_cluster2_multi_sim(neural_set, C, p.N)
    eigenspectrum_all = copy(eigenspectrum)
    Threads.@threads for ii = 2:n_sim
        points = rand(Uniform(-p.L/2,p.L/2),p.N,p.n) #points are uniformly distributed in a region 
        C = reconstruct_covariance2(p,different_neural_variability= different_neural_variability)
        eigenspectrum, k = pca_cluster2_multi_sim(neural_set, C, p.N)
        eigenspectrum_all = hcat(eigenspectrum_all,eigenspectrum)
    end
    eigenvalues = mean(eigenspectrum_all, dims=2)
    errorbars = std(eigenspectrum_all, dims=2) ./ sqrt(size(eigenspectrum_all,2))

    λ_min = minimum(eigenspectrum[:])
    λ_max = maximum(eigenspectrum[:])

    id, λ = dropzeros_and_less(eigenspectrum_all[:,1])
    length = 200
    h_λ = fit(Histogram, λ, LogRange(λ_min,λ_max,length))
    # edges, p_sim = normalize(h_λ, mode=:pdf)
    h_λ = normalize(h_λ, mode=:pdf)
    @unpack edges, weights = h_λ       
    binsize = edges[1][2] - edges[1][1]
    x = collect(edges[1][2:end].-binsize/2)
    p_sim_all = vec(weights.*size(id,1)/p.N)

    for ii = 2:size(eigenspectrum_all,2)
        id, λ = dropzeros_and_less(eigenspectrum_all[:,ii])
        length = 200
        h_λ = fit(Histogram, λ, LogRange((λ_min),maximum(λ_max),length))
        h_λ = normalize(h_λ, mode=:pdf)
        @unpack edges, weights = h_λ   
        binsize = edges[1][2] - edges[1][1]
        x = collect(edges[1][2:end].-binsize/2)
        p_sim = vec(weights.*size(id,1)/p.N)
        p_sim_all = hcat(p_sim_all,p_sim)
    end

    p_sim,errorbars = mean_sem_nonzero(p_sim_all, dims=2)
    # errorbars = std(p_sim_all, dims=2) ./ sqrt(size(p_sim_all,2))
    p_err_l = p_sim .- errorbars
    p_err_u = p_sim .+ errorbars

    ax.plot(x, p_sim, linewidth=1,label="ERM simulation")        
    ax.fill_between(x,p_err_l[1:end],p_err_u[1:end],alpha=0.3)


    return ax
end

###########################
## rank plot subsampling ##
###########################
function multi_sim_rank_subplot(ax,p;n_sim,different_neural_variability)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("eigenvalue " * L"\lambda",fontsize=8)
    ax.set_ylabel("probability density " * L"p(\lambda)",fontsize=8)
    # n_sim = 10
    neural_set = random_clusters(p.N)
    C = reconstruct_covariance2(p, different_neural_variability=different_neural_variability)
    eigenspectrum, k = pca_cluster2_multi_sim(neural_set, C, p.N)
    eigenspectrum_all = copy(eigenspectrum)
    Threads.@threads for ii = 2:n_sim
        C = reconstruct_covariance2(p, different_neural_variability=different_neural_variability)
        eigenspectrum, k = pca_cluster2_multi_sim(neural_set, C, p.N)
        eigenspectrum_all = hcat(eigenspectrum_all,eigenspectrum)
    end
    eigenspectrum_all = eigenspectrum_all ./ p.ρ # normalized eigenvalues
    eigenvalues = mean(eigenspectrum_all, dims=2)
    errorbars = std(eigenspectrum_all, dims=2) ./ sqrt(size(eigenspectrum_all,2))

    λ_err_l = eigenvalues .- errorbars
    λ_err_u = eigenvalues .+ errorbars
    n_plot = Int(round(p.N/10))
    x = (1:n_plot)./p.N
    if different_neural_variability == false
        ax.plot(x, eigenvalues[1:n_plot],"-^",ms=1,markerfacecolor="none",linewidth=0.5,label="sim: corr")  
    else
        ax.plot(x, eigenvalues[1:n_plot],"-v",ms=1,linewidth=0.5,label="sim: cov",color="red",alpha=0.5)  
    end
    ax.fill_between(x,λ_err_l[1:n_plot],λ_err_u[1:n_plot],alpha=0.3)


    return ax,n_plot
end

function rank_match_subplot(ax,p,n_sim)
    _,_ = multi_sim_rank_subplot(ax,p,n_sim=n_sim,different_neural_variability=false)
    _,nplot = multi_sim_rank_subplot(ax,p,n_sim=n_sim,different_neural_variability=true)

    N = p.N
    x_p = [1:1:N;]
    x_p = x_p./N
    lb_th_ls = zeros(nplot)
    lb_num_ls = zeros(nplot)
    for i = 1:nplot
        lb_th_ls[i] = f̂₀_th(i, p; flag_k = false)
        lb_num_ls[i] = f̂₀_num(i, p; flag_k = false)
    end

    p_ls = x_p[1:nplot]
    ax.loglog(p_ls,lb_th_ls, "o-",markerfacecolor="none",lw=0.5,ms=1.5,label="high-density theory",color="cyan")
    # ax.loglog(p_ls,sort(lb_num_ls,rev=true), "+-",markerfacecolor="none",label="num")
    # ax.plot(p_ls,lb_num_ls, ".-",lw=0.5,ms=0.5,label="numerical",color="gray")
    ax.set_xlabel("Rank/N",fontsize=8)
    # ax.xlabel("Rank/N (top 10% eigenvalues)",fontsize=S)
    ax.set_ylabel(L"\lambda/\rho",fontsize=8)
    ax.legend()
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
    for n in range(iterations, 7, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], "-o", linewidth=0.1,
            color = colorList[i,:], ms=0.5,label=@sprintf("N=%d",2^n))   
        # ax1.scatter(k[1:end], eigenvalues[1:end], marker=".",markerfacecolor=colorList[i,:],
        # ms=0.5,fillstyle="full")
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.2,color=colorList[i,:])
    end
    ax1.set_ylim([1e-1, 1e+2])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.2,s=2)
    ax1.set_xlabel("rank/N", fontsize=8)
    ax1.set_ylabel("eigenvalue " * L"\lambda", fontsize=8)
    return ax1
end


function RG_step(C; neural_set = reshape(1:size(C,1),:,1))
    Cr = cov2cor(C)
    #L = size(C,1)
    N,S = size(neural_set)
    C_new = zeros(Int(N/2),Int(N/2))
    neural_set_new = zeros(Int(N/2),Int(2*S))
    Ru,subs = linearize_triu(Cr)
    #R = sort(Ru,rev=true)
    IDX = sortperm(Ru,rev=true)
    L = size(IDX,1)
    temp = Int.(zeros(N,1));
    k = 1
    for h=1:L  
        i = Int.(subs[IDX[h],1])    
        j = Int.(subs[IDX[h],2])      
        n_i=neural_set[Int.(subs[IDX[h],1]),:]
        n_j=neural_set[Int.(subs[IDX[h],2]),:]
        if !(i in temp)&&!(j in temp)
            temp[2*k-1:2*k] =[i;j]
            #neural_set = push_cluster_index(neural_set,2^n,k,i,j)
            neural_set_new[k,:] = hcat(n_i,n_j)
            k=k+1
        end
        if k==N/2+1
            break
        end
    end
    for i = 1:Int(N/2)
        for j = 1:Int(N/2)
            C_new[i,j] = sum(Cr[temp[2*i-1:2*i],temp[2*j-1:2*j]])
        end
    end
    return C_new, Int.(neural_set_new)
end

function RG_clusters(C)
    N = size(C,1)
    neural_set = reshape(1:N,:,1)
    iterations = Int(log2(N))
    cluster_set = Vector{Any}(undef, iterations+1)
    L = N
    cluster_set[1] = Vector{Any}(undef, L)
    y = randperm(N)
    for k = 1:L
        cluster_set[1][k] = y[k]
    end
    for n = 2:iterations+1
        L = Int(L/2)
        cluster_set[n] = Vector{Any}(undef, L)
        C, neural_set = RG_step(C, neural_set = neural_set)
        for k = 1:L
            cluster_set[n][k] = neural_set[k,:]
        end
    end
    return cluster_set
end

function ERM_RG_rank_subplot(C,ax1)
    N = size(C,1)
    neural_set = RG_clusters(C)
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []
    i = 0
    for n in range(iterations, 7, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], "-o", linewidth=0.1,
            color = colorList[i,:], ms=0.5,label=@sprintf("N=%d",2^n))   
        # ax1.scatter(k[1:end], eigenvalues[1:end], marker=".",markerfacecolor=colorList[i,:],
        # ms=0.5,fillstyle="full")
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.2,color=colorList[i,:])
    end
    ax1.set_ylim([1e-1, 1e+2])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.2,s=2)
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
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.5,marker="o",color=colorList[i,:],
        ms=0.3,label=@sprintf("N=%d",K))   
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

function anatomical_clusters(N, y_axis)
    # this function is only for N which is power to 2
    y_sorted_indices = sortperm(y_axis)
    iterations = Int(floor(log2(N)))
    cluster_set = Vector{Any}(undef, iterations+1)
    L = N
    cluster_set[1] = Vector{Any}(undef, L)
    y = y_sorted_indices
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

function functional_RG_rank_subplot(C,ax1,y_axis)
    N = size(C,1)
    neural_set = anatomical_clusters(N, y_axis)
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []
    i = 0
    for n in range(iterations, 7, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], "-o", linewidth=0.1,
            color = colorList[i,:], ms=0.5,label=@sprintf("N=%d",2^n))   
        # ax1.scatter(k[1:end], eigenvalues[1:end], marker=".",markerfacecolor=colorList[i,:],
        # ms=0.5,fillstyle="full")
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.2,color=colorList[i,:])
    end
    ax1.set_ylim([1e-1, 1e+2])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.2,s=2)
    ax1.set_xlabel("rank/N", fontsize=8)
    ax1.set_ylabel("eigenvalue " * L"\lambda", fontsize=8)
    return ax1
end