using KernelDensity
import MultivariateStats as MVS
using Statistics
using PyCall
import PyPlot as plt



function subsampling_PR(C,ax1, y_axis, x_axis=nothing; flag_face_color=true, clus_type="rand", flag_normalized=false)
    # plot the covariance matrix dimensionality
    # raw dim without normalized by N
    N = size(C,1)
    if clus_type == "rand"
        neural_set = random_clusters(N)
    elseif clus_type == "anatomical"
        neural_set = anatomical_clusters(N, y_axis, x_axis)
    elseif clus_type == "RG"
        neural_set = RG_clusters(C)
    elseif clus_type == "RG_v2"
        neural_set = RG_clusters_v2(C)
    end
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))
    # PR_ratio = zeros(4)
    # errorbar = zeros(4)
    # K = zeros(4)
    i = 0
    for n in range(iterations, iterations-3, step=-1)
        i +=1
        PR_ratio, errorbar, K, PR_ratio_all = pca_cluster2_PR(neural_set, C, Int(2^n); flag_normalized=flag_normalized)
        x = [K./N for _ in 1:length(PR_ratio_all)]
        if flag_face_color == true
            ax1.errorbar(K./N, PR_ratio, yerr=errorbar, fmt="o", color=colorList[i,:], label="N=$K")
            ax1.scatter(x, PR_ratio_all, color=colorList[i, :], marker="o",s=1.5)
        else
            # ax1.errorbar(K./N.+0.02, PR_ratio, yerr=errorbar, fmt="o", mfc="none", color=colorList[i,:], label="N=$K")
            # ax1.scatter(x.+0.02, PR_ratio_all, color=colorList[i, :], marker="o",s=2, facecolor="none")
            ax1.errorbar(K./N, PR_ratio, yerr=errorbar, fmt="o", mfc="none", color=colorList[i,:], label="N=$K")
            # ax1.scatter(x, PR_ratio_all, color=colorList[i, :], marker="o",s=1.5, facecolor="none")

        end
    end
    ax1.set_xlabel("subsampled fraction", fontsize=8)
    ax1.set_ylabel("dimensionality", fontsize=8)
    return ax1
end





function subsampling_PR2(C,ax1, y_axis; flag_face_color=true, clus_type="rand")
    # plot the covariance matrix dimensionality
    # normalized dim
    N = size(C,1)
    if clus_type == "rand"
        neural_set = random_clusters(N)
    elseif clus_type == "anatomical"
        neural_set = anatomical_clusters(N, y_axis)
    elseif clus_type == "RG"
        neural_set = RG_clusters(C)
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
        PR_ratio, errorbar, K, PR_ratio_all = pca_cluster2_PR(neural_set, C, Int(2^n);flag_normalized=true)
        x = [K./N for _ in 1:length(PR_ratio_all)]
        if flag_face_color == true
            ax1.errorbar(K./N.-0.02, PR_ratio, yerr=errorbar/2, fmt="o", color=colorList[i,:], label="N=$K")
            ax1.scatter(x.-0.02, PR_ratio_all, color=colorList[i, :], marker="o",s=2)
        else
            ax1.errorbar(K./N.+0.02, PR_ratio, yerr=errorbar/2, fmt="o", mfc="none", color=colorList[i,:], label="N=$K")
            ax1.scatter(x.+0.02, PR_ratio_all, color=colorList[i, :], marker="o",s=2, facecolor="none")

        end
    end
    ax1.set_xlabel("subsampled fraction", fontsize=8)
    ax1.set_ylabel("normalized dimensionality", fontsize=8)
    return ax1
end

# the relationship between pca_cluster2_PR and pca_cluster3_PR is that 
# pca_cluster2_PR uses one row as a cluster
# pca_cluster3_PR uses one column as a cluster

function pca_cluster2_PR(cluster_set, C_original, K; flag_normalized=false)
    n = Int(log2(K) + 1)
    Nrow = length(cluster_set[n])
    println("cluster number: "*string(Nrow))
    PR_ratio = zeros(Nrow)

    for i in 1:Nrow
        IDX = cluster_set[n][i]
        c = C_original[IDX,:]
        c = c[:,IDX]
        LAMBDA = sort(eigvals(Hermitian(c)), rev=true)
        LAMBDA = LAMBDA[LAMBDA .> 0]
        # p1 = percentile(data, 90) 
        # LAMBDA = LAMBDA[LAMBDA .> p1]
        # print(length(LAMBDA))
        # print("\n")
        if flag_normalized
            PR_ratio[i] = (sum(LAMBDA))^2/sum(LAMBDA.^2)/length(IDX)
        else
            PR_ratio[i] = (sum(LAMBDA))^2/sum(LAMBDA.^2)
        end
    end
    errorbar = std(PR_ratio) / sqrt(Nrow)
    PR_ratioMean = mean(PR_ratio)
    return PR_ratioMean, errorbar, K, PR_ratio
end

function pca_cluster2_PR2(cluster_set, C_original, K)
    # normalized dim
    n = Int(log2(K) + 1)
    Nrow = length(cluster_set[n])
    PR_ratio = zeros(Nrow)

    for i in 1:Nrow
        IDX = cluster_set[n][i]
        c = C_original[IDX,:]
        c = c[:,IDX]
        LAMBDA = sort(eigvals(Hermitian(c)), rev=true)
        LAMBDA = LAMBDA[LAMBDA .> 0]
        p1 = percentile(data, 90) 
        LAMBDA = LAMBDA[LAMBDA .< p1]
        print(length(LAMBDA))
        print("\n")
        PR_ratio[i] = (sum(LAMBDA))^2/sum(LAMBDA.^2)/length(LAMBDA)
    end
    errorbar = std(PR_ratio) / sqrt(Nrow)
    PR_ratioMean = mean(PR_ratio)
    return PR_ratioMean, errorbar, K, PR_ratio
end

function pca_cluster3_PR(cluster_set, C_original, n)
    # calculate covariance matrix dimensionality
    # attention: this function is for all neurons when the number of neuron is not the power of 2
    # it can also be used when number of neurons is the power of 2
    # the order of cluster_set is reversed compared to pca_cluster2
    K = length(cluster_set[n])
    
    cluster_set_n = hcat(cluster_set[n]...)'
    Ncol = size(cluster_set_n,2)
    PR_ratio = zeros(Ncol)
    for i in 1:Ncol
        IDX = cluster_set_n[:,i]
        c = C_original[IDX,:]
        c = c[:,IDX]
        LAMBDA = sort(eigvals(Hermitian(c)), rev=true)
        LAMBDA = LAMBDA[LAMBDA .> 0]
        PR_ratio[i] = (sum(LAMBDA))^2/sum(LAMBDA.^2)/K
    end
    errorbar = std(PR_ratio) / sqrt(Ncol)
    PR_ratio = mean(PR_ratio)
    return PR_ratio, errorbar, K
end

function anatomical_clusters(N, y_axis, x_axis= nothing)
    # this function is only for N which is power to 2
    iterations = Int(floor(log2(N))) # if N=1024, iterations = 10
    cluster_set = Vector{Any}(undef, iterations+1)

    if isnothing(x_axis)
        y_sorted_indices = sortperm(y_axis)
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

    else # if x_axis exists, then we need to divide neurons according to both x_axis and y_axis

        cluster_set = Vector{Any}(undef, iterations+1)
        cluster_set[iterations+1] = Vector{Any}(undef, 1)
        cluster_set[iterations+1][1] = sortperm(y_axis)

        for n = iterations:-1:1
    
            # if n is even number
            if n % 2 == 0
                one_axis = copy(y_axis)
            else
                one_axis = copy(x_axis)
            end          
        
            nClus = 2^(iterations-n+1) # the number of clusters for each iteration
            L = Int(floor(N/2^(iterations-n+1))) # the length of cluster, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1
        
            cluster_set[n] = Vector{Any}(undef, nClus)
            ii = 0
            for i = 1:2:nClus
                ii += 1
                relative_idx = sortperm(one_axis[cluster_set[n+1][ii]]) # cluster_set[n+1][ii] is the raw index of neurons in the larger cluster 
                cluster_set[n][i] = cluster_set[n+1][ii][relative_idx][1:L] # divide the larger cluster into two smaller clusters
            end
            ii = 0
            for i = 2:2:nClus
                ii += 1
                relative_idx = sortperm(one_axis[cluster_set[n+1][ii]]) 
                cluster_set[n][i] = cluster_set[n+1][ii][relative_idx][L+1:end]
            end
        
        end


    end
    return cluster_set
end

# function RG_clusters(C)
#     N = size(C,1)
#     neural_set = reshape(1:N,:,1)
#     iterations = Int(log2(N))
#     cluster_set = Vector{Any}(undef, iterations+1)
#     L = N
#     cluster_set[1] = Vector{Any}(undef, L)
#     y = randperm(N)
#     for k = 1:L # we have L clusters for the first iteration
#         cluster_set[1][k] = y[k]
#     end
#     for n = 2:iterations+1
#         L = Int(L/2) # we have L/2 clusters for the nth iteration
#         cluster_set[n] = Vector{Any}(undef, L)
#         C, neural_set = RG_step(C, neural_set = neural_set)
#         for k = 1:L
#             cluster_set[n][k] = neural_set[k,:]
#         end
#     end
#     return cluster_set
# end

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

# function RG_step(C; neural_set = reshape(1:size(C,1),:,1))
#     Cr = cov2cor(C)
#     #L = size(C,1)
#     N,S = size(neural_set)
#     C_new = zeros(Int(N/2),Int(N/2))
#     neural_set_new = zeros(Int(N/2),Int(2*S))
#     Ru,subs = linearize_triu(Cr)
#     #R = sort(Ru,rev=true)
#     IDX = sortperm(Ru,rev=true)
#     L = size(IDX,1)
#     temp = Int.(zeros(N,1));
#     k = 1
#     for h=1:L  
#         i = Int.(subs[IDX[h],1])    
#         j = Int.(subs[IDX[h],2])      
#         n_i=neural_set[Int.(subs[IDX[h],1]),:]
#         n_j=neural_set[Int.(subs[IDX[h],2]),:]
#         if !(i in temp)&&!(j in temp)
#             temp[2*k-1:2*k] =[i;j]
#             #neural_set = push_cluster_index(neural_set,2^n,k,i,j)
#             neural_set_new[k,:] = hcat(n_i,n_j)
#             k=k+1
#         end
#         if k==N/2+1
#             break
#         end
#     end
#     for i = 1:Int(N/2)
#         for j = 1:Int(N/2)
#             C_new[i,j] = sum(Cr[temp[2*i-1:2*i],temp[2*j-1:2*j]])
#         end
#     end
#     return C_new, Int.(neural_set_new)
# end

function RG_step(C; neural_set = reshape(1:size(C,1),:,1))
    Cr = cov2cor(C)
    #L = size(C,1)
    N,S = size(neural_set)
    C_new = zeros(Int(N/2),Int(N/2))
    neural_set_new = zeros(Int(N/2),Int(2*S))
    Ru,subs = linearize_triu(C)
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

function RG_clusters_v2(C)
    # use the most correlated cluster
    # the first version: cluster_set[n][1] as the most correlated cluster
    # the second version: the cluster with max(mean of c) as the most correlated cluster 
    N = size(C,1)

    iterations = floor(log2(N))

    cluster_set = RG_clusters(C);
    cluster_set_new = copy(cluster_set)

    # modify RG to RG_one
    for (idx, n) in enumerate(range(iterations-3, iterations, step=1))
        n = Int(n)
        mean_c_ls = zeros(size(cluster_set[n],1))
        for i in 1:size(cluster_set[n],1)
            c = C[cluster_set[n][i],cluster_set[n][i]];
            c_ls,_ = linearize_triu(c)
            mean_c_ls[i] = mean(c_ls) 
        end
        max_idx = argmax(mean_c_ls)
        println("The index of cluster before spliting: "*string(max_idx))
        println(mean_c_ls)
        for i in 1:size(cluster_set[n],1)
            cluster_set_new[n][i] = cluster_set_new[n][max_idx]
        end

    end

    return cluster_set_new

end

function most_correlated_pairs(C)
    N = size(C,1)
    cluster_set = RG_clusters(C)
    iterations = floor(log2(N))

    # example: 
    # cl[2][1:2] is a 2-element Vector{Any}:
    # [551, 220]
    # [171, 222]
    # vcat(cl[2][1:2]...) is a 4-element Vector{Int64}:
    # 551
    # 220
    # 171
    # 222
    # the most correlated pairs of neurons as one cluster, 
    # for example K = 128, 
    # then the first iteration one neuron as one cluster, totally 128 clusters,
    # the second iteration 64 clusters, the third iteration 32 clusters, etc.
    # here we take the top 32 clusters from 64 clusters as the most correlated pairs of neurons
    # that is why K/4 is used here
    IDX_sub = vcat(cluster_set[2][1:Int(N/4)]...) 
    n_IDX_sub = length(IDX_sub)

    for (idx, n) in enumerate(range(iterations, iterations-3, step=-1))
        n = Int(n)
        stop = n_IDX_sub/(2^(idx-1))
        stop = Int(stop)
        for i in 1:size(cluster_set[n],1)
            cluster_set[n][i] = IDX_sub[1:stop]
        end
    end

    return cluster_set
end

function functional_clusters(X)
    N = size(X,1)
    iterations = Int(log2(N))
    cluster_set = Vector{Any}(undef, iterations+1)
    cluster_set[iterations+1] = Vector{Any}(undef, 1)
    cluster_set[iterations+1][1] = collect(1:N)

    for (idx, n) in enumerate(iterations:-1:6) # don't need to go to the last iteration
        nClus = 2^(idx) # 2^1 or 2^2 or 2^3 or 2^4 or 2^5
        cluster_set[n] = Vector{Any}(undef, nClus)
        for (iClus_wh, iClus) in enumerate(1:2:nClus) # 1 or 1,3 or 1,3,5 or 1,3,5,7 or 1,3,5,7,9
            println("The index of cluster before spliting: "*string(iClus_wh))
            println("The index of cluster after spliting: "*string(iClus))
            neurid = cluster_set[n+1][iClus_wh]
            println("The number of neurons in the cluster before spliting: "*string(length(neurid)))
            println("\n")
            # the first cluster
            id_sub = functional_steps(X[neurid, :])
            cluster_set[n][iClus] = neurid[id_sub]
            # the second cluster
            cluster_set[n][iClus+1] = setdiff(neurid, cluster_set[n][iClus])
        end

    end

    return cluster_set

end

function functional_steps(X_mds)
    """
    functional_steps(X_mds), a function called by functional_clusters(X_mds)

    This function performs functional clustering on a 2D scatter plot of data points.

    1. Find the highest density point.
    2. Divide the neurons into two equal-sized groups using a straight line passing through 
    the highest density point.
    3. Compute the variance of neuron coordinates in MDS space for each group.
    4. Find the division that has the lowest difference between the two variances.

    # Arguments
    - `X_mds`: A matrix representing the 2D scatter plot, 
    where each row represents a data point and 
    the first column represents the x-coordinate and the second column represents the y-coordinate.

    # Returns
    - `neuron_ids`: An array of indices representing the selected neurons.
    """

    N = size(X_mds, 1)
    data_x = X_mds[:, 1]
    data_y = X_mds[:, 2]
    neuron_ids = collect(1:length(data_x))



    # Find the highest density point
    kde_result = kde((data_x, data_y))
    # Corrected to properly identify the index in a 2D array
    max_index = argmax(kde_result.density)
    highest_density_point = [kde_result.x[max_index[1]], kde_result.y[max_index[2]]]

    # plt.figure()
    # # First subplot for the clustered data
    # plt.scatter(highest_density_point[1], highest_density_point[2], c="red", label="highest density", alpha=1)
    # plt.scatter(data_x, data_y, c="gray", label="All Data", alpha=0.2)
    # plt.savefig("functional_test.png")
    
    N = length(data_x) # The number of data points
    
    # Initialize a list to store the possible divisions that meet the criteria
    division_ls = []
    
    # Iterate over all the other points to define division lines
    for idx in 1:N
        # The current point to create the division line with the highest density point
        current_point = [data_x[idx], data_y[idx]]
        
        # Skip if it is the highest density point itself
        if current_point == highest_density_point
            continue
        end
        
        # Calculate the direction vector from the highest density point to the current point
        direction_vector = current_point - highest_density_point
        
        # Use the direction vector to divide the dataset
        # Positive if on one side of the line, and negative if on the other
        division = [(data_x[i] - highest_density_point[1]) * direction_vector[2] - 
                    (data_y[i] - highest_density_point[2]) * direction_vector[1] < 0 for i in 1:N]
        # Assign the current_point to one group and the highest_density_point to the other
        division[idx] = !division[idx]
        # Calculate sizes of the divided groups, correct for equal
        if sum(division) == Int(N/2)            
            push!(division_ls, division)
        end
    end
    
    # Now you have a list of valid divisions in `division_ls`
    # You can further analyze these divisions as per your requirements

    # Compute the variance of neuron coordinates in MDS space for each group
    diff_ls = []
    for division in division_ls
        group1_indices = findall(x -> x, division)
        group2_indices = findall(x -> !x, division)
        group1_variance = var(X_mds[group1_indices, 1]) + var(X_mds[group1_indices, 2])
        group2_variance = var(X_mds[group2_indices, 1]) + var(X_mds[group2_indices, 2])
        push!(diff_ls, abs(group1_variance - group2_variance))
    end

    # Find the division that has the lowest difference between the two variances
    min_difference_idx = argmin(diff_ls)
    optimal_division = division_ls[min_difference_idx]

    return neuron_ids[optimal_division]
end

# old version of functional_steps: should be removed someday
# function functional_sampling(data_x, data_y)

#     """
#     functional_sampling(data_x, data_y), a function called by functional_clusters(X_mds)

#     Perform kernel density estimation on the given dataset and filter points within a certain radius (median).

#     # Arguments
#     - `data_x`: Array{Float64} - The x-coordinates of the dataset.
#     - `data_y`: Array{Float64} - The y-coordinates of the dataset.

#     # Returns
#     - `data_x_within_radius`: Array{Float64} - The x-coordinates of the points within the radius.
#     - `data_y_within_radius`: Array{Float64} - The y-coordinates of the points within the radius.
#     - `points_within_radius_filter`: BitArray{1} - A boolean array indicating which points are within the radius.
#     """

#     # Perform kernel density estimation on the current dataset
#     kde_result = kde((data_x, data_y))

#     # Find the highest density
#     max_density = maximum(kde_result.density)
#     max_index = argmax(kde_result.density)
#     highest_density_point = (kde_result.x[max_index[1]], kde_result.y[max_index[2]])

#     # Calculate distances to the highest density point
#     distances = [sqrt((x - highest_density_point[1])^2 + (y - highest_density_point[2])^2) for (x, y) in zip(data_x, data_y)]

#     # Sort distances and find the median, which is our radius containing half of the points
#     sorted_distances = sort(distances)
#     median_index = ceil(Int, length(sorted_distances) / 2)
#     radius = sorted_distances[median_index]

#     # Filter points within this radius
#     points_within_radius_filter = distances .<= radius
#     data_x_within_radius = data_x[points_within_radius_filter]
#     data_y_within_radius = data_y[points_within_radius_filter]

#     # Print out the result for this iteration (or plot them as needed)
#     println("Highest density point: ", highest_density_point)
#     println("Density value: ", max_density)
#     println("Points within radius: ", length(data_x_within_radius))
#     println("–––––––––––––––––––––––")

#     # To avoid running the next iteration if we have too few points to proceed
#     if length(data_x) <= 1
#         println("Insufficient data points to proceed to the next iteration.")
#     end

#     return data_x_within_radius, data_y_within_radius, points_within_radius_filter
# end
    

using Random

function functional_sampling_v2(X_mds, sample_fraction)
    """
    As constrol? forget what this function was doing ...
    This function performs functional subsampling on a 2D scatter plot of data points by selecting a random neuron 
    and then selecting a given percentage of total neurons within a radius for each iteration.

    # Arguments
    - `X_mds`: A matrix representing the 2D scatter plot, where each row represents a data point.
    - `sample_fraction`: The fraction of neurons to be included at each iteration. Should be a value between 0 and 100.
    - `Ncenter`: Number of neurons to be selected as centers. Default value is 10.

    # Returns
    - `closest_indices`: An array of indices representing the selected neurons.

    """
    N = size(X_mds, 1)
    neuron_ids = collect(1:N)  # An array containing all neuron IDs
    index = sample(neuron_ids, 1, replace = false)
    index_point = X_mds[index, :]
    data_x = X_mds[:,1]
    data_y = X_mds[:,2]
  
    # Calculate distances to the highest density point
    distances = [sqrt((x - index_point[1])^2 + (y - index_point[2])^2) for (x, y) in zip(data_x, data_y)]

    # Sort distances and find the fraction_index, which is our radius containing half of the points
    sorted_distances = sort(distances)
    fraction_index = ceil(Int, N*sample_fraction)
    radius = sorted_distances[fraction_index]

    # Filter points within this radius
    points_within_radius_filter = distances .<= radius
    closest_indices = neuron_ids[points_within_radius_filter]

    return closest_indices
end


function kmeans_c_clusters(X; flag_pca=true)
    # use kmeans_constrained to split data into two clusters with equal size
    N = size(X,1)
    iterations = Int(log2(N))
    cluster_set = Vector{Any}(undef, iterations+1)
    cluster_set[iterations+1] = Vector{Any}(undef, 1)
    cluster_set[iterations+1][1] = collect(1:N)

    for (idx, n) in enumerate(iterations:-1:6) # don't need to go to the last iteration
        nClus = 2^(idx) # 2^1 or 2^2 or 2^3 or 2^4 or 2^5
        cluster_set[n] = Vector{Any}(undef, nClus)
        for (iClus_wh, iClus) in enumerate(1:2:nClus) # 1 or 1,3 or 1,3,5 or 1,3,5,7 or 1,3,5,7,9
            println("The index of cluster before spliting: "*string(iClus_wh))
            println("The index of cluster after spliting: "*string(iClus))
            neurid = cluster_set[n+1][iClus_wh]
            println("The number of neurons in the cluster before spliting: "*string(length(neurid)))
            println("\n")
            # the first cluster
            id_sub = kmeans_c_steps(X[neurid, :]; flag_pca)
            cluster_set[n][iClus] = neurid[id_sub]
            # the second cluster
            cluster_set[n][iClus+1] = setdiff(neurid, cluster_set[n][iClus])
        end

    end

    return cluster_set
end

function kmeans_c_steps(X; flag_pca=true)
    function performPCA(X::Matrix, explained_variance::Float64)
        # Standardize the data
        # time points are Features to be Standardized
        X_centered = X .- mean(X, dims=1)
        std_X = std(X, dims=1)
        X_std = X_centered ./ (std_X .+ (std_X .== 0))
    
        # Fit the PCA model
        pca_model = MVS.fit(MVS.PCA, X_std')
    
        # Find the number of components needed to explain the specified variance
        cumulative_var = cumsum(MVS.principalvars(pca_model)) / sum(MVS.principalvars(pca_model))
        n_components = findfirst(cumulative_var .>= explained_variance)
    
        if isnothing(n_components)
            error("Explained variance too high, cannot find a component count that satisfies it.")
        end
    
        # Transform the dataset using the appropriate number of components
        pca_transformed = MVS.transform(pca_model, X_std')
    
        # Select the desired number of components
        Y = pca_transformed[1:n_components, :]
    
        return Y', n_components
    end
        
    function perform_clustering(data)
        clus_size = Int(size(data, 1)/ 2)
        # Importing numpy to handle the conversion to numpy arrays
        np = PyCall.pyimport("numpy")
    
        # Convert Julia array to NumPy array for use with Python package
        data_np = np.array(data)
    
        # Import the k_means_constrained package
        k_means_constrained = PyCall.pyimport("k_means_constrained")
    
        # Create an instance of KMeansConstrained, specifying 2 clusters with equal size
        kmeans = k_means_constrained.KMeansConstrained(n_clusters=2, size_min=clus_size, size_max=clus_size,random_state=0)
    
        # Fit the model to the data
        kmeans.fit(data_np)
    
        # Retrieve the labels and centroids
        labels = PyCall.copy(kmeans.labels_)
        centroids = PyCall.copy(kmeans.cluster_centers_)
    
        return labels, centroids
    end
    if flag_pca
        X_pca, _ = performPCA(X, 0.9)
        labels, _ = perform_clustering(X_pca)
    else
        labels, _ = perform_clustering(X)
    end
    return findall(labels .== 0)
end

function clusters_large_k(C, ROIxyz; clus_type="rand")
    # extend the dim to 0.75. For example , the whole set of neuron is 1024, now we can have a colum of 768
    N = size(C,1)
    iterations = Int(floor(log2(N)))
    cluster_net_new = Vector{Any}(undef, iterations+2)
    if clus_type == "rand"
        
        cluster_net = random_clusters(N)
        addition_set1 = reduce(vcat, cluster_net[iterations-1][1:3])
        addition_set2 = reduce(vcat, cluster_net[iterations-1][2:4])

        cluster_net_new[1:iterations] = cluster_net[1:iterations]
        cluster_net_new[iterations+2] = cluster_net[iterations+1]
        
        cluster_net_new[iterations+1] = Vector{Any}(undef, 2)

        cluster_net_new[iterations+1][1] = addition_set1
        cluster_net_new[iterations+1][2] = addition_set2

    elseif clus_type == "anatomical"

        x_axis = ROIxyz[:,1]
        y_axis = ROIxyz[:,2]
        cluster_net = anatomical_clusters(N, y_axis, x_axis)
    
        # get the vector contains 3 clusters as a fraction of 0.75 as another iteration
        addition_set1 = reduce(vcat, cluster_net[iterations-1][1:3])
        addition_set2 = reduce(vcat, cluster_net[iterations-1][2:4])

        cluster_net_new[1:iterations] = cluster_net[1:iterations]
        cluster_net_new[iterations+2] = cluster_net[iterations+1]
        
        cluster_net_new[iterations+1] = Vector{Any}(undef, 2)

        cluster_net_new[iterations+1][1] = addition_set1
        cluster_net_new[iterations+1][2] = addition_set2

    elseif clus_type == "RG_v2"
        cluster_net = RG_clusters_v2(C)
        cluster_net_original = RG_clusters(C)
        addition_set1 = reduce(vcat, cluster_net_original[iterations-1][1:3])

        cluster_net_new[1:iterations] = cluster_net[1:iterations]
        cluster_net_new[iterations+2] = cluster_net[iterations+1]
        
        cluster_net_new[iterations+1] = Vector{Any}(undef, 1)

        cluster_net_new[iterations+1][1] = addition_set1

    end

    return cluster_net_new

end


function subsampling_PR_3_types_large_k(ax1, C, ROIxyz; clus_type="rand")
    # plot the covariance matrix dimensionality for extended fraction (0.75)
    N = size(C,1)

    neural_set = clusters_large_k(C, ROIxyz; clus_type)


    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=5))

    i = 0
    for n in range(iterations, iterations-4, step=-1)
        i +=1
        # PR_ratio_all: PR_ratio for each cluster in one iteration
        # PR_ratio: the mean of PR_ratio_all
        # be careful about n+1 here!! it is for large k
        PR_ratio, errorbar, K, PR_ratio_all = pca_cluster2_PR(neural_set, C, Int(2^(n+1)))
        # x = [0.125, 0.25, 0.5, 0.75, 1]
        x = [1.0, 0.75, 0.5, 0.25, 0.125]
        x2 = x[i]
        x2 = [x2 for i in 1:length(PR_ratio_all)]
        if isnan(errorbar)
            errorbar = 0
        end
        if clus_type == "rand"
            if !isnan(errorbar)
                ax1.errorbar(x[i] - 0.02, PR_ratio, yerr=errorbar, fmt="o", color=colorList[i,:],ms=6, label=clus_type)
            end
            ax1.scatter(x2.-0.02, PR_ratio_all, color=colorList[i, :], marker="o",s=2, facecolor="none")
        elseif clus_type == "anatomical" || clus_type == "ROI_CCA"
            if !isnan(errorbar)
                ax1.errorbar(x[i], PR_ratio, yerr=errorbar, fmt="s", color=colorList[i,:], ms=6, label=clus_type)
            end
            ax1.scatter(x2, PR_ratio_all, color=colorList[i, :], marker="o",s=2, facecolor="none")
        # RG or functional 
        elseif clus_type == "RG" || clus_type == "RG_v2" || clus_type == "functional" || clus_type == "X_CCA" 
            if !isnan(errorbar)
                ax1.errorbar(x[i].+0.02, PR_ratio, yerr=errorbar, fmt="v", color=colorList[i,:], ms=6, label=clus_type)
            end
            ax1.scatter(x2.+0.02, PR_ratio_all, color=colorList[i, :], marker="o", s=2, facecolor="none")

        end
    end
    ax1.set_xlabel("subsampled fraction", fontsize=8)
    ax1.set_ylabel("dimensionality", fontsize=8)
    # tick_positions = [, 2, 3, 4]
    # tick_labels = ["N/8", "N/4", "N/2", "N/N"]
    # plt.xticks(tick_positions, tick_labels)
    ax1.set_xlim(0,1.1)
    return ax1
end


function two_eigen_rank_plot(C,ax1,y_axis,K_i)
    N = size(C,1)
    rand_set = random_clusters(N)
    ana_set = anatomical_clusters(N, y_axis)
    RG_set = RG_clusters(C)
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []
    i = 0
    c_ls = ["red","blue","green"]
    for n in range(iterations-K_i, iterations-K_i, step=-1)
        i+=1
        for j in [1,3]
            if j == 1
                eigenvalues, errorbars, k = pca_cluster2(rand_set, C, Int(2^n))
            elseif j == 2
                eigenvalues, errorbars, k = pca_cluster2(ana_set, C, Int(2^n))
            elseif j == 3
                eigenvalues, errorbars, k = pca_cluster2(RG_set, C, Int(2^n))
            end
            I = findall(eigenvalues .> 0)
            eigenvalues = eigenvalues[I]
            k = k[I]
            errorbars = errorbars[I]  
            λ_err_l = eigenvalues - errorbars./2
            λ_err_u = eigenvalues + errorbars./2
            ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.5,marker="o",color=c_ls[j],
                ms=0.3,label=@sprintf("N=%d",2^n)) 
            ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color=c_ls[j])
        end
    end
    ax1.set_ylim([1e-2, 1e+3])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N", fontsize=8)
    ax1.set_ylabel("eigenvalue " * L"\lambda", fontsize=8)
    return ax1
end


function three_eigen_rank_plot(C,ax1,rand_set, ana_set, func_set, K_i)
    N = size(C,1)
    iterations = floor(log2(N))
    # cmap = plt.get_cmap("winter")
    # colorList = cmap(range(0, 1, length=4))

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    # leg2 = []
    # eigenspectrums = []
    # nranks = []
    i = 0
    c_ls = ["red","blue","green"]
    for n in range(iterations-K_i, iterations-K_i, step=-1)
        i+=1
        for j in [1,2,3]
            if j == 1
                eigenvalues, errorbars, k = pca_cluster2(rand_set, C, Int(2^n))
            elseif j == 2
                eigenvalues, errorbars, k = pca_cluster2(ana_set, C, Int(2^n))
            elseif j == 3
                eigenvalues, errorbars, k = pca_cluster2(func_set, C, Int(2^n))
            end
            I = findall(eigenvalues .> 0)
            eigenvalues = eigenvalues[I]
            k = k[I]
            errorbars = errorbars[I]  
            λ_err_l = eigenvalues - errorbars./2
            λ_err_u = eigenvalues + errorbars./2
            ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.5,marker="o",color=c_ls[j],
                ms=0.3,label=@sprintf("N=%d",2^n)) 
            ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color=c_ls[j])
        end
    end
    ax1.set_ylim([1e-2, 1e+3])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N", fontsize=8)
    ax1.set_ylabel("eigenvalue " * L"\lambda", fontsize=8)
    return ax1
end

function diff_sampling_rank_subplot(C,ax1,y_axis, x_axis=nothing; clus_type ="RG" )
    N = size(C,1)
    
    if clus_type == "rand"
        neural_set = random_clusters(N)
    elseif clus_type == "anatomical"
        if isnothing(x_axis)
            neural_set = anatomical_clusters(N, y_axis)
        else
            neural_set = anatomical_clusters(N, y_axis, x_axis)
        end
    elseif clus_type == "RG"
        neural_set = RG_clusters(C)

    elseif clus_type == "RG_v2"
        neural_set = RG_clusters_v2(C)

    elseif clus_type == "functional"
        if isnothing(x_axis)
            println("Please provide mds coordinates for functional subsampling")
        else
            X_mds = hcat(x_axis, y_axis)
            neural_set = functional_clusters(X_mds)
        end    
    end
    
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

###################################################

###################################################

function collapse_index_rank2(C, y_axis, x_axis=nothing;clus_type)
    """
    collapse_index_rank2(C, y_axis; clus_type)

    Collapse index also for RG and anatomical/regional subsampling. 
    The function collapse_index_rank is only for random subsampling.

    # Arguments
    - `C`: Matrix representing the covariance or correlation.
    - `y_axis`: Anatomical Axis.
    - `clus_type`: Type of clustering to be performed.

    # Returns
    - `CI`: Collapse index.
    - `(r0+1)/Ns`: Lower bound ratio.
    - `r1/Ns`: Upper bound ratio.
    """ 
  
    s = 0.5 # subsample percentage
    
    
    N,_ = size(C)

    if clus_type == "rand"
        neural_set = random_clusters(N)
    elseif clus_type == "anatomical"
        if isnothing(x_axis)
            neural_set = anatomical_clusters(N, y_axis)
        else
            neural_set = anatomical_clusters(N, y_axis, x_axis)
        end
    elseif clus_type == "RG"
        neural_set = RG_clusters(C)
    elseif clus_type == "functional"
        if isnothing(x_axis)
            println("Please provide mds coordinates for functional subsampling")
        else
            X_mds = hcat(x_axis, y_axis)
            neural_set = functional_clusters(X_mds)
        end
    end
         
    iterations = Int(log2(N)); # e.g., if N is 1024 totally 10 iterations and size(neural_set) is 11
    # iterations = iterations - 2
    Ns = round(Int, N*s)
    
    
    lb_N = -sort(-eigvals(Hermitian(C)))
    N_sets = size(neural_set[iterations],1)
    if clus_type == "rand"
        ntrial = 2000
        lb_Ns = zeros(Ns,ntrial)
        for i in 1:ntrial
            i_s = randperm(N)[1:Ns]
            Cs = copy(C[:,i_s])
            Cs = Cs[i_s,:]
            lb_Ns[:,i] = -sort(-eigvals(Hermitian(Cs)))
        end
    else
        lb_Ns = zeros(Ns,N_sets)
        for i in 1:N_sets
            i_s = neural_set[iterations][i] 
            Cs = copy(C[:,i_s])
            Cs = Cs[i_s,:]
            lb_Ns[:,i] = -sort(-eigvals(Hermitian(Cs)))
        end
    end
    lb_Ns = vec(mean(lb_Ns,dims=2))

    # r0 = round(Int, Ns*p0-0/.5) 
    # r0 = 5

    
    if N >= 1024 # take out the first 1% of eigenvalues
        p0 = 0.01 
        r0 = round(Int, Ns*p0-0/.5) 
    else # take out the first 5 eigenvalues
        r0 = 5
    end
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


function subsampling_PR_3_types(C,ax1, y_axis, x_axis=nothing; clus_type="rand")
    # plot the covariance matrix dimensionality
    N = size(C,1)
    if clus_type == "rand"
        neural_set = random_clusters(N)
    elseif clus_type == "anatomical" || clus_type == "X_CCA" || clus_type == "ROI_CCA"
        if isnothing(x_axis)
            neural_set = anatomical_clusters(N, y_axis)
        else
            neural_set = anatomical_clusters(N, y_axis, x_axis)
        end
    elseif clus_type == "RG"
        neural_set = RG_clusters(C)
    elseif clus_type == "RG_v2"
        neural_set = RG_clusters_v2(C)
    elseif clus_type == "functional"
        if isnothing(x_axis)
            println("Please provide mds coordinates for functional subsampling")
        else
            X_mds = hcat(x_axis, y_axis)
            neural_set = functional_clusters(X_mds)
        end    
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
            if !isnan(errorbar)
                ax1.errorbar(K./N.-0.02, PR_ratio, yerr=errorbar, fmt="o", color=colorList[i,:],ms=6, label=clus_type)
            end
            ax1.scatter(x.-0.02, PR_ratio_all, color=colorList[i, :], marker="o",s=2, facecolor="none")
        elseif clus_type == "anatomical" || clus_type == "ROI_CCA"
            if !isnan(errorbar)
                ax1.errorbar(K./N, PR_ratio, yerr=errorbar, fmt="s", color=colorList[i,:], ms=6, label=clus_type)
            end
            ax1.scatter(x, PR_ratio_all, color=colorList[i, :], marker="o",s=2, facecolor="none")
        # RG or functional 
        elseif clus_type == "RG" || clus_type == "RG_v2" || clus_type == "functional" || clus_type == "X_CCA" 
            if !isnan(errorbar)
                ax1.errorbar(K./N.+0.02, PR_ratio, yerr=errorbar, fmt="v", color=colorList[i,:], ms=6, label=clus_type)
            end
            ax1.scatter(x.+0.02, PR_ratio_all, color=colorList[i, :], marker="o", s=2, facecolor="none")

        end
    end
    ax1.set_xlabel("subsampled fraction", fontsize=8)
    ax1.set_ylabel("dimensionality", fontsize=8)
    ax1.set_xlim(0,1.1)
    # tick_positions = [, 2, 3, 4]
    # tick_labels = ["N/8", "N/4", "N/2", "N/N"]
    # plt.xticks(tick_positions, tick_labels)
    return ax1
end

function subsampling_one_iteration_PR_3_types(C, ax1, y_axis, x_axis=nothing; clus_type="rand", iteration=1, x=1)
    # plot the covariance matrix dimensionality for one iteration
    # input x: the x position of the data point. e.g., x=mouse1, x=mouse2, x=mouse3
    N = size(C,1)
    if clus_type == "rand"
        neural_set = random_clusters(N)
    elseif clus_type == "anatomical"
        if isnothing(x_axis)
            neural_set = anatomical_clusters(N, y_axis)
        else
            neural_set = anatomical_clusters(N, y_axis, x_axis)
        end
    elseif clus_type == "RG"
        neural_set = RG_clusters(C)
    end
    iterations = floor(log2(N))
    colorList = plt.get_cmap("winter")(range(0, 1, length=4))
    PR_ratio = zeros(4)
    errorbar = zeros(4)
    K = zeros(4)

    # the first iteration: 1024 neurons (n = 10), the second iteration: 512 neurons (n=9), etc.
    n = iterations - iteration + 1

    PR_ratio, errorbar, K, PR_ratio_all = pca_cluster2_PR(neural_set, C, Int(2^n))
    print(errorbar)
    if isnan(errorbar)
        errorbar = 0
    end
    x_ = [x for _ in 1:length(PR_ratio_all)]
    if clus_type == "rand"
        ax1.errorbar(x .- 0.2, PR_ratio, yerr=errorbar, fmt="o", color=colorList[iteration, :], ms=6)
        ax1.scatter(x_ .- 0.2, PR_ratio_all, color=colorList[iteration, :], marker="o", s=2, facecolor="none")
    elseif clus_type == "anatomical"
        ax1.errorbar(x, PR_ratio, yerr=errorbar, fmt="s", color=colorList[iteration, :], ms=6)
        ax1.scatter(x_, PR_ratio_all, color=colorList[iteration, :], marker="o", s=2, facecolor="none")
    elseif clus_type == "RG"
        ax1.errorbar(x .+ 0.2, PR_ratio, yerr=errorbar, fmt="v", color=colorList[iteration, :], ms=6)
        ax1.scatter(x_ .+ 0.2, PR_ratio_all, color=colorList[iteration, :], marker="o", s=2, facecolor="none")
    end

    return ax1
end

# function subsampling_PR_3_types_theory(C,ax1, y_axis, x_axis=nothing; clus_type="rand")
#     # plot the covariance matrix dimensionality
#     N = size(C,1)

#     iterations = floor(log2(N))
#     cmap = plt.get_cmap("winter")
#     colorList = cmap(range(0, 1, length=4))
#     PR_ratio = zeros(4)
#     errorbar = zeros(4)
#     K = zeros(4)
#     i = 0
#     for n in range(iterations, iterations-3, step=-1)
#         i +=1
#         # PR_ratio_all: PR_ratio for each cluster in one iteration
#         # PR_ratio: the mean of PR_ratio_all
#         PR_ratio, errorbar, K, PR_ratio_all = pca_cluster2_PR(neural_set, C, Int(2^n))
#         d = 2
#         μ = Parameter_estimation(C, n = d)[2]
        
#         x = [K./N for _ in 1:length(PR_ratio_all)]
#         if isnan(errorbar)
#             errorbar = 0
#         end
#         if clus_type == "rand"
#             if !isnan(errorbar)
#                 ax1.errorbar(K./N.-0.02, PR_ratio, yerr=errorbar, fmt="o", color=colorList[i,:],ms=6, label=clus_type)
#             end
#             ax1.scatter(x.-0.02, PR_ratio_all, color=colorList[i, :], marker="o",s=2, facecolor="none")
#         elseif clus_type == "anatomical" || clus_type == "ROI_CCA"
#             if !isnan(errorbar)
#                 ax1.errorbar(K./N, PR_ratio, yerr=errorbar, fmt="s", color=colorList[i,:], ms=6, label=clus_type)
#             end
#             ax1.scatter(x, PR_ratio_all, color=colorList[i, :], marker="o",s=2, facecolor="none")
#         # RG or functional 
#         elseif clus_type == "RG" || clus_type == "RG_v2" || clus_type == "functional" || clus_type == "X_CCA" 
#             if !isnan(errorbar)
#                 ax1.errorbar(K./N.+0.02, PR_ratio, yerr=errorbar, fmt="v", color=colorList[i,:], ms=6, label=clus_type)
#             end
#             ax1.scatter(x.+0.02, PR_ratio_all, color=colorList[i, :], marker="o", s=2, facecolor="none")

#         end
#     end
#     ax1.set_xlabel("subsampled fraction", fontsize=8)
#     ax1.set_ylabel("dimensionality", fontsize=8)
#     ax1.set_xlim(0,1.1)
#     # tick_positions = [, 2, 3, 4]
#     # tick_labels = ["N/8", "N/4", "N/2", "N/N"]
#     # plt.xticks(tick_positions, tick_labels)
#     return ax1
# end


###################################################
###################################################
# interval prediction related functions
###################################################
###################################################
function theory_PR(cluster_set, C_original, K, K_pred, μ=nothing, d=nothing; iClus=1, RCCA = 0)

    # smaller anatomical cluster data to calculate E(σ^2)^2, E(σ^4) and E(c_ij^2)
    # if K > K_pred, then it is using whole brain data to calculate E(σ^2)^2, E(σ^4) and E(c_ij^2)
    # K_pred can be any integer

    # ERM simulation or data
    if isnothing(μ) || isnothing(d)
        flag_data = true
    else
        flag_data = false
    end

    n = Int(log2(K) + 1)
    Nrow = length(cluster_set[n])
    #println("cluster number: "*string(Nrow))
    PR_pred_random = zeros(Nrow)
    PR_pred_RG = zeros(Nrow)
    PR_pred_ana = zeros(Nrow)
    CI_corr = zeros(Nrow)


    for i in 1:Nrow
        #println("i: "*string(i))
        IDX = cluster_set[n][i]
        c = C_original[IDX,:]
        c = c[:,IDX]

        if flag_data
            d = 2 # change d would not modify the result of μ/d
            ρ, μ = Parameter_estimation(c, n = d)
            println("ρ: "*string(ρ)*", μ: "*string(μ))
        end

        # one anatomical cluster has one CI
        # CI_corr[i] = collapse_index_rank2(c, [];clus_type="rand")[1]
        # E(c_ij^2)
        c_ls,_ = linearize_triu(c)

        a = mean(diag(c))^2 * K_pred # raw dim prediction
        k_pred_ana = sqrt.(1 .- RCCA^2 .+ (K_pred/K).^2*RCCA^2)
        b = mean(diag(c).^(2)) + (K_pred - 1) * mean(c_ls.^2)
        b2 = mean(diag(c).^(2)) + (K_pred - 1) * mean(c_ls.^2) * (K_pred/K)^(-2*μ/d)
        b3 = mean(diag(c).^(2)) + (K_pred - 1) * mean(c_ls.^2) * (k_pred_ana)^(-2*μ/d)
        # one anatomical cluster has one prediction
        PR_pred_random[i] = a/b
        PR_pred_RG[i] = a/b2
        PR_pred_ana[i] = a/b3

    end

    # PR_pred = mean(PR_pred)
    # CI_corr = mean(CI_corr)

    return PR_pred_random[iClus], PR_pred_RG[iClus], PR_pred_ana[iClus]#, CI_corr
end


function theory_PR_v2(cluster_set, C_original, K, K_pred, μ=nothing, d=nothing; iClus=1, flag_compute_CI=true)

    # smaller anatomical cluster data to calculate E(σ^2)^2, E(σ^4) and E(c_ij^2)
    # plugin N to do prediction: N_pred to predict PR

    # ERM simulation or data
    if isnothing(μ) || isnothing(d)
        flag_data = true
    else
        flag_data = false
    end

    n = Int(log2(K) + 1)
    Nrow = length(cluster_set[n])
    println("cluster number: "*string(Nrow))
    PR_pred_random = zeros(Nrow)
    PR_pred_RG = zeros(Nrow)
    CI_corr = zeros(Nrow)

    n_pred = Int(log2(K_pred) + 1)
    IDX_pred = cluster_set[n_pred][1]
    N_pred = length(IDX_pred)

    # check N_pred is equal to K_pred
    if N_pred != K_pred
        @warn "N_pred is not equal to K_pred"
    end

    for i in 1:Nrow
        println("i: "*string(i))
        IDX = cluster_set[n][i]
        c = C_original[IDX,:]
        c = c[:,IDX]

        if flag_data


            # cluster_set_temp = RG_clusters_v2(c)
            # IDX_sub = cluster_set_temp[end-1][1] 

            # not using RG_clusters_v2
            cluster_set_temp = RG_clusters(c)

            # example: 
            # cl[2][1:2] is a 2-element Vector{Any}:
            # [551, 220]
            # [171, 222]
            # vcat(cl[2][1:2]...) is a 4-element Vector{Int64}:
            # 551
            # 220
            # 171
            # 222
            # the most correlated pairs of neurons as one cluster, 
            # for example K = 128, 
            # then the first iteration one neuron one cluster totally 128 clusters,
            # the second iteration 64 clusters, the third iteration 32 clusters, etc.
            # here we take the top 32 clusters from 64 clusters as the most correlated pairs of neurons
            # that is why K/4 is used here
            IDX_sub = vcat(cluster_set_temp[2][1:Int(K/4)]...) 

            c_sub = c[IDX_sub,:]
            c_sub = c_sub[:,IDX_sub]
            c_sub_ls,_ = linearize_triu(c_sub)

            c_ls,_ = linearize_triu(c)
            E_k = mean(c_sub_ls.^2)
            E = mean(c_ls.^2)
            θ = log(E_k/E)/log(1/2)
        else 
            θ = -2*μ/d

        end
        println("θ: "*string(θ))


        # one anatomical cluster has one CI
        if flag_compute_CI
            CI_corr[i] = collapse_index_rank2(c, [];clus_type="rand")[1]
        else
            CI_corr[i] = NaN
        end
        # E(c_ij^2)
        c_ls,_ = linearize_triu(c)

        a = mean(diag(c))^2

        b = mean(diag(c).^(2)) + (N_pred - 1) * mean(c_ls.^2)
        # b2 = b*(K_pred/K)^(-2*μ/d)
        b2 = mean(diag(c).^(2)) + (N_pred - 1) * mean(c_ls.^2) * (K_pred/K)^(θ)
        # one anatomical cluster has one prediction
        PR_pred_random[i] = a/b
        PR_pred_RG[i] = a/b2

    end

    # PR_pred = mean(PR_pred)
    # CI_corr = mean(CI_corr)

    return PR_pred_random[iClus], PR_pred_RG[iClus], CI_corr[iClus]
end

########################################################################################

function clusters_expand(N, y_axis, x_axis=nothing; start_K=128)
    # cluster expand: a population of neurons which contains the points (neurons) that you started with.

    # This function expands clusters of neurons to the next fraction by checking if 
    # the current cluster is a subset of any cluster in the next fraction.
    # It makes sure that the clusters are a subset of the next fraction.

    # structure example: 
    # cluster_set[1][1] = [1,2,3,4,5]
    # cluster_set[2][1] = [1,2,3,4,5,6,7,8,9,10]
    # cluster_set[1][2] = [6,7,8,9,10]
    # cluster_set[2][2] = [1,2,3,4,5,6,7,8,9,10]


    function check_subset(a, b)
        # Check if cluster a is a subset of cluster b
        is_in_b = [element in b for element in a]
        return sum(is_in_b) == length(a)
    end

    cluster_set = anatomical_clusters(N, y_axis, x_axis)
    cluster_set_new = copy(cluster_set)
    iterations = Int(floor(log2(N))) # 1024 neurons, 10 iterations

    n = Int(floor(log2(start_K))) + 1
    nClus = length(cluster_set[n])

    for iFraction = 1:(iterations - n + 1)

        cluster_set_new[n+iFraction] = Vector{Any}(undef, nClus)

        for i in 1:nClus
            a = cluster_set[n][i]
            # expand to the next fraction
            for ii in 1:length(cluster_set[n+iFraction])
                b = cluster_set[n+iFraction][ii]
                if check_subset(a, b)
                    cluster_set_new[n+iFraction][i] = b
                    break
                end
            end
        end
    end

    return cluster_set_new

end


########################################################################################
module new_load_data

using MAT
using Parameters
import PyPlot as plt
include("../src/util.jl") # ZZ's code
include("../src/util2.jl")
const PR_ROOT = ERM_ROOT * "/results/PR/Corr_mat"

stringer_neuropixels_dict = Dict(
    1 => "Krebs",
    2 => "Waksman",
    3 => "Robbins",
)

function load_stringer_neuropixels_data(isubject, seed_i=1, C_type = "cov", flag_rand_wb = true)
    data_dir = ERM_ROOT * "/results/firing_rate/"
    @unpack FR, Wh = matread(joinpath(data_dir, isubject * "_10Hz_FR.mat"));
    FR = FR[:,1:7200]

    tf = vec(sum(FR, dims=2) .> 7200*0.05)
    if sum(tf) < length(tf)
        @warn "There are neurons with zero activity. Already removed."
    end
    FR = convert(Array{Float64}, FR[tf, :])
    Wh = convert(Array{Float64}, Wh[tf, :])


    K = 1024
    if flag_rand_wb # randomly sample K neurons from the whole brain
        Random.seed!(seed_i)
        i_select = sample(1:size(FR, 1), K, replace = false)
    else
        # select whole brain neurons based on the height of the neurons
        i_select = sortperm(Wh[:], rev=true)
        i_select = i_select[1:K]

    end
    # already sorted by the height of the neurons
    FR = FR[i_select, :]
    Wh = Wh[i_select, :]
    # center_Coord = center_Coord[i_select,:]
    FR = normalize_activity(FR);
    if C_type == "cov"
        FR, C_data = data_preprocessing(FR);
    elseif C_type == "corr"
        C_data = cor(convert(Matrix, FR'));
    end


    return FR, C_data, Wh
end

function load_neuropixels_C(isubject)
    C_data = load_stringer_neuropixels_data(isubject, 2, "corr", false)[2]
    return C_data
end

fish_dict = Dict(
    1=>"201106",
    2=>"201116",
    3=>"201117",
    4=>"201125",
    5=>"201126",
    6=>"210106",
    # 7 =>"210430",
)

function load_FR_raw(ifish)
    fishdata_dir = "/home/data2/wangzezhen/fishdata/"
    FR = matread(joinpath(fishdata_dir, ifish, "spike_OASIS.mat"))["sMatrix_total"]
    Judge = vec(matread(joinpath(fishdata_dir, ifish, "Judge.mat"))["Judge"])
    if eltype(Judge)==Bool
        Judge = vec(Judge)
    else
        Judge = vec(Judge.==1)
    end
    if sum(Judge) != size(FR,1)
        println("Warning: check the variable Judge!")
    end
    Judge2 = vec(matread(joinpath(fishdata_dir, ifish, "Judge2.mat"))["Judge2"])
    FR = FR[Judge2,:]

    if ifish == "201125"
        nC = size(FR,1)
        tf = fill(true, Int(nC))
        # tf[[76,855]] .= false
        tf[[76]] .= false
        FR = FR[tf,:]
    end
    return FR
end

function load_XYZ_raw(ifish)
    n = 2
    matroot = ERM_ROOT*"/results/mds_CCA"
    # ROI_pjec: the vector plotted in anatomical space
    # X_pjec: the vector plotted in functional space
    @unpack X, ROI_pjec, X_pjec, ROIxyz, correlations_CCA = matread(joinpath(matroot,"$ifish","$n","X_CCA.mat"))

    if ifish == "201116" || ifish == "201117"
        ROIxyz[:,[1,2]] = ROIxyz[:,[2,1]]
    end

    return ROIxyz

end

function load_fish_data(ifish, seed_i=1, C_type = "cov", flag_rand_wb = true)

    FR = load_FR_raw(ifish)
    FR = FR[:,1:7200]

    ROIxyz = load_XYZ_raw(ifish)

    # check the raw size of ROIxyz and FR
    if size(ROIxyz,1) != size(FR,1)
        println("Warning: check the variable ROIxyz!")
    end
    println("Raw ROIxyz size: "*string(size(ROIxyz)))

    K = 1024
    if flag_rand_wb # randomly sample K neurons from the whole brain
        Random.seed!(seed_i)
        i_select = sample(1:size(FR, 1), K, replace = false)
    else
        # select whole brain neurons based on the xyz axis
        # Default use seed_i = 2 which is anterior-posterior axis
        if seed_i <=3
            i_select = sortperm(ROIxyz[:,seed_i])
            i_select = i_select[1:K]
        else
            i_select = sortperm(ROIxyz[:,seed_i-3], rev=true)
            i_select = i_select[1:K]
        end
    end
    FR = FR[i_select, :]
    # center_Coord = center_Coord[i_select,:]
    FR = normalize_activity(FR);
    if C_type == "cov"
        FR, C_data = data_preprocessing(FR);
    elseif C_type == "corr"
        C_data = cor(convert(Matrix, FR'));
    end

    # CCA axis
    ROI_CCA = ROIxyz[i_select,:]*ROI_pjec
    X_CCA = X["parent"][i_select,:]*X_pjec

    ROIxyz = ROIxyz[i_select,:]

    return FR, C_data, ROI_CCA, X_CCA, X["parent"][i_select,:], ROIxyz, correlations_CCA
end

load_fish_XYZ(ifish, seed_i=2, C_type = "cov", flag_rand_wb=false) = load_fish_data(ifish, seed_i, C_type,flag_rand_wb)[6]

load_fish_C(ifish, seed_i=2, C_type = "cov", flag_rand_wb=false) = load_fish_data(ifish, seed_i, C_type, flag_rand_wb)[2]

load_fish_FR(ifish, seed_i=2, C_type = "cov", flag_rand_wb=false) = load_fish_data(ifish, seed_i, C_type, flag_rand_wb)[1]


function load_fish_C2(ifish, N_end=1024)
    # select neurons along the anterior-posterior axis starting from the median point
    # select 1:7200 time frames
    # filtering neurons with smaller activity
    # compute correlation matrix

    FR_raw = load_FR_raw(ifish)
    ROIxyz_raw = load_XYZ_raw(ifish)


    # anterior--posterior axis
    axis_for_sort = copy(ROIxyz_raw[:,1])
    i_sort = sortperm(axis_for_sort, rev=false)

    # filtering neurons with smaller activity
    tf = vec(sum(FR_raw[:,1:7200] .> 0, dims=2) .> 7200*0.01)
    if sum(tf) < length(tf)
        println("Removed small activity neurons.")
    end

    FR = convert(Array{Float64}, FR_raw[tf, 1:7200])
    i_sort = i_sort[tf]

    nC = size(FR,1)

    # Median index
    # select 1200 neurons around the median point, leave some room to remove no response neurons
    N_select = 1024
    median_index = div(nC+1, 2)
    # Calculate start and end indices for the range taking care not to go out of bounds
    start_index = max(1, median_index - Int(N_select/2))
    end_index = min(median_index + Int(N_select/2), length(axis_for_sort))
    # Select the range
    i_sort = i_sort[start_index:end_index-1]

    # check length of i_sort
    if length(i_sort) != N_select
        println("Warning: check the variable i_sort!")
    end

    # here i_sort are index in raw data
    # because i_sort is already filtered by tf, we can order neuron along AP axis by FR_raw[i_sort, 1:7200] without tf
    FR = convert(Array{Float64}, FR_raw[i_sort, 1:7200])
    # check length of FR and i_sort
    if size(FR,1) != length(i_sort)
        println("Warning: check the variable i_sort!")
    end
    
    FR = normalize_activity(FR);
    C = cor(convert(Matrix, FR'));


    plt.figure()
    plt.scatter(ROIxyz_raw[:,1], ROIxyz_raw[:,2], s=3, color="k", label="all neurons")
    plt.scatter(ROIxyz_raw[i_sort,1], ROIxyz_raw[i_sort,2], s=1, color="r", label="selected neurons")
    plt.title("check whole set neuron anatomical distribution")
    plt.savefig(ERM_ROOT * "/results/PR/Corr_mat/light_field/"*ifish*"_ROI_xyz.png")
    plt.close()

    return C # correlation matrix neuron order is already sorted along anatomical axis
end

light_sheet_dict = Dict(
    1=>"8",
    2=>"9",
    3=>"11",
)

function load_light_sheet_C(ifish)
    # preprocessing: code\data_analysis_Weihao\ERM_paper\preprocessing\light_sheet_fish_preprocessing_v2.m
    # load the light sheet data
    data_dir = "C:/Fish-Brain-Behavior-Analysis/Fish-Brain-Behavior-Analysis/results/firing_rate/light_sheet_fish/"
    data_dir = data_dir * "AP_axis"

    @unpack C_ls, XYZ_ls = matread(joinpath(data_dir, "light_sheet_fish$(ifish)_C_XYZ.mat"));
    return C_ls, XYZ_ls

end

twoP_dict = Dict(
    1=>"spont_M150824_MP019_2016-04-05",
    2=>"spont_M160907_MP028_2016-09-26",
    3=>"spont_M160825_MP027_2016-12-12",
    4=>"spont_M161025_MP030_2016-11-20",
    5=>"spont_M161025_MP030_2017-06-16",
    6=>"spont_M161025_MP030_2017-06-23",
    7=>"spont_M170714_MP032_2017-08-04",
    8=>"spont_M170717_MP033_2017-08-18",
    9=>"spont_M170717_MP034_2017-08-25"
    )

function load_2p_C(isubject)
    data_dir = "D:/OneDrive - HKUST Connect/data/stringer_et_al/stringer_visual_two_photon"
    FR = matread(joinpath(data_dir, isubject * ".mat"))["Fsp"]
    XYZ = matread(joinpath(data_dir, isubject * ".mat"))["med"]
    
    # Pre-select a window of 7200 time points as in the previous example
    FR = convert(Array{Float64}, FR[:, 1:7200])

    # Filter neurons by activity as before
    tf = vec(sum(FR .> 0, dims=2) .> 7200 * 0.05)
    FR = FR[tf, :]
    XYZ = XYZ[tf, :]

    # Sort neurons along the indicated axis
    axis_for_sort = copy(XYZ[:, 1])
    i_sort = sortperm(axis_for_sort, rev=false)

    C_ls = []
    XYZ_ls = []
    N_select = 1024
    N_all_neuron = length(axis_for_sort)
    # Here we start at 1 and go until we have less than N_select neurons left
    for (i, N_idx) in enumerate(1:N_select:N_all_neuron-(N_select-1))
        selected_indices = i_sort[N_idx:N_idx+N_select-1]

        FR_selected = FR[selected_indices, :]
        XYZ_selected = XYZ[selected_indices, :]

        # Assume normalize_activity is a function that normalizes the activity matrix
        FR_normalized = normalize_activity(FR_selected)

        # Compute the correlation matrix for the selected neurons
        C = cor(FR_normalized')

        push!(C_ls, C)
        push!(XYZ_ls, XYZ_selected)

        fig_path = joinpath("../results/PR/Corr_mat/2p", "$(isubject)_$(i)_ROI_xyz.png")
        # Check if the figure file already exists
        if !isfile(fig_path)
            plt.figure()
            plt.scatter(XYZ[:, 1], XYZ[:, 2], s=3, color="k", label="all neurons")
            plt.scatter(XYZ_selected[:, 1], XYZ_selected[:, 2], s=1, color="r", label="selected neurons")
            plt.title("Neuron anatomical distribution after selection")
            plt.legend()
          
            # Save the figure to disk only if it does not exist
            plt.savefig(fig_path)
            plt.close()
        end
    end

    # Return the list of correlation matrices and XYZ coordinates for each chunk
    return C_ls, XYZ_ls
end

function get_data_dict(data_type)
    return Dict(
        "neuropixels" => stringer_neuropixels_dict,
        "light_field" => fish_dict,
        "light_sheet" => light_sheet_dict,
        "2p" => twoP_dict, # stringer 2p
        # "ERM" => nld.ERM_dict  # Assuming ERM_dict is uncommented in the actual code
    )[data_type]
end

# Function to load correlation matrices based on data type
function load_correlation_matrices(isubject, data_type)
    if data_type == "light_field"
        return [load_fish_C2(isubject, 1024)]
    elseif data_type == "light_sheet"
        return load_light_sheet_C(isubject)[1]
    elseif data_type == "neuropixels"
        return [load_neuropixels_C(isubject)]
    elseif data_type == "2p"
        return load_2p_C(isubject)[1]
    end
end

end

########################################################################################
module new_analyses

using MAT
using Parameters
include("../src/util.jl") # ZZ's code
include("../src/util2.jl")


"""
    compute_D(Corr, n=nothing)

Compute the distance matrix D based on the correlation matrix Corr.

# Arguments
- `Corr`: The correlation matrix.
- `n`: The number of dimensions. Default is `nothing`.

# Returns
The matrix D.

"""

function compute_D(Corr, n=nothing)

    function find_D(C,p)
        @unpack μ, ϵ, β, L = p 
        L = L
        D .= ϵ*sqrt.(abs.(C.^(-2/μ) .- 1))
        D[D.>L] .= L*log.((D[D.>L] ./L)) .+L
        return D
    end

    if isnothing(n)
        n = 2
    end

    d = n
    ρ, μ = Parameter_estimation(Corr, n = n)
    N = size(Corr)[1]
    L = (N/ρ)^(1/n)
    ϵ = 0.03125
    p = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = ϵ, μ = μ, ξ = 10^18, β = 0, σ̄² = 1, σ̄⁴ = 1)

    
    println("(ϵ/L)^d = ",(ϵ/L)^d," d=",d)

    C = Corr
    C = max.(C,10^-2)
    D = copy(C)

    D = find_D(C, p)
    D = D - Diagonal(D)
    W = ones(N,N)
    W[Corr .< 0] .= 0.01

    return D
end

function matlab_mds(C,d=nothing)
    code_path = "C:/Users/Public/code/Fish-Brain-Behavior-Analysis/code/data_analysis_Weihao/ERM_paper/"
    D = compute_D(C,d)

    # if you can run matlab inside Julia, you can use the following code to compute the mds
    # X, stress = mdscale(D; n = 2, criterion="sammon")

    # if you can't run matlab inside Julia, save D as intermidiate file and run the following code in matlab
    matwrite(code_path * "src/D.mat", Dict("D"=>D))

    # bash command to run matlab
    run(`matlab -nodisplay -nosplash -nodesktop -r "run('$(code_path * "src/mds_step2.m")');exit;"`)

    # load the result from matlab
    while !isfile(code_path * "src/X_mds.mat")
        sleep(0.5)  # Wait for 1 second before checking again
    end
    X = matread(code_path * "src/X_mds.mat")["X_mds"]
 
    # Replace forward slashes with backslashes
    code_path = replace(code_path, "/" => "\\")

    # Now `code_path` has backslashes
    println(code_path)
    # Delete the intermediate files in Win10
    run(`cmd /c del $(code_path * "src\\D.mat")`)
    run(`cmd /c del $(code_path * "src\\X_mds.mat")`)

    return X, D

end

using MAT
# Pkg.add("Glob")
using Glob


function combine_CI(RESULT_ROOT, isubject)
    # each subject has multiple correlation matries
    # Find all the matching files for each CI type
    CI_rand_files = glob("CI_rand_$(isubject)*.mat", RESULT_ROOT)
    CI_anatomical_files = glob("CI_anatomical_$(isubject)*.mat", RESULT_ROOT)
    CI_RG_files = glob("CI_RG_$(isubject)*.mat", RESULT_ROOT)

    # Initialize arrays to store combined data
    combined_rand = []
    combined_anatomical = []
    combined_RG = []

    # Check if there are any files to process
    if isempty(CI_rand_files) || isempty(CI_anatomical_files) || isempty(CI_RG_files)
        error("No files were found for one or more CI types")
    end

    # Read and combine data from each file type
    for rand_file in CI_rand_files
        data = matread(rand_file)["CI_whole_set"]
        push!(combined_rand, data)
    end

    for anatomical_file in CI_anatomical_files
        data = matread(anatomical_file)["CI_whole_set"]
        push!(combined_anatomical, data)
    end

    for RG_file in CI_RG_files
        data = matread(RG_file)["CI_whole_set"]
        push!(combined_RG, data)
    end

    # Combine into a single matrix, with each column representing a CI type
    combined_CI = hcat(vcat(combined_rand...), vcat(combined_anatomical...), vcat(combined_RG...))

    return combined_CI
end

end