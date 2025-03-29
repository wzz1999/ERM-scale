using MAT, MATLAB
using Parameters, Plots, LinearAlgebra, BSON, StatsBase
using MultivariateStats, Statistics
using NPZ
using Dates

include("../src/util.jl")

if ~ @isdefined ifish
    ifish = 210106
end

sim = false
do_normlization = true
if !@isdefined do_normlization
    do_normlization = false
end
do_zscore = false
condition = "all"
if !@isdefined ifish
    ifish = 201116
end

if !@isdefined date
    date = today()
end

######## load spike data ################
#dataroot = get(ENV, "FISHDATA", "D:/Fish-Brain-Behavior-Analysis/results")
if isdir("/home/data2/wangzezhen/fishdata")
    dataroot = "/home/data2/wangzezhen/fishdata"
elseif isdir("D:/Fish-Brain-Behavior-Analysis/results")
    dataroot = "D:/Fish-Brain-Behavior-Analysis/results"
end

if isdir("/home/data2/wenquan/fishdata")
    dataroot2 = "/home/data2/wenquan/fishdata"
elseif isdir("D:/Fish-Brain-Behavior-Analysis/results")
    dataroot2 = "D:/Fish-Brain-Behavior-Analysis/results"
end

spikedata = joinpath(dataroot, "$ifish", "spike_OASIS.mat")
@unpack sMatrix_total = matread(spikedata)

####### load corrected ROI   ###############
judgedata = joinpath(dataroot,"$ifish","Judge.mat")

if isfile(judgedata)
    @unpack Judge = matread(judgedata)
    if eltype(Judge)==Bool
        judge = findall(vec(Judge))
    else
        judge = findall(!iszero, vec(Judge))
    end
    
    if length(Judge) == size(sMatrix_total,1)
        r = convert(Array{Float32},sMatrix_total[judge,:])
    else
        r = convert(Array{Float32},sMatrix_total)
    end

end

judgedata2 = joinpath(dataroot,"$ifish","Judge2.mat")

if isfile(judgedata2)
    @unpack Judge2 = matread(judgedata2)
    if eltype(Judge2)==Bool
        judge2 = findall(vec(Judge2))
    else
        judge2 = findall(!iszero, vec(Judge2))
    end

    if length(Judge2) == size(r,1)
        r = r[judge2,:]
    end

end

#firingrate = r[:,6001:9000]
firingrate = r

(N,T) = size(firingrate)

#=
if N>1024
    id = sample(1:N,1024,replace = false)
    if T > 6000
        firingrate = firingrate[id, 1:6000]
    else
        firingrate = firingrate[id, :]
    end
else
    if T > 6000
        firingrate = firingrate[:, 1:6000]
    end
end
=#

if do_normlization
    A = normalization(firingrate, ϵ=1e-4)
    A = A .- mean(A,dims=2)
elseif do_zscore
    μ = mean(firingrate, dims=2)
    μ = repeat(μ, 1, T)
    σ = std(firingrate, dims=2)
    σ = repeat(σ, 1, T)
    A = zscore(firingrate,μ,σ)
else
    A = firingrate .- mean(firingrate, dims=2)
end

dims = []
a = C_projection(A, dims=dims)
if !isempty(dims)
    condition = condition*"_exclude $dims dims"
end

resultdata = joinpath(dataroot, "$ifish", "Result.mat")
resultdata2 = joinpath(dataroot, "$ifish", "Results.mat")
resultdata3 = joinpath(dataroot2, "$ifish", "Result.mat")
resultdata4 = joinpath(dataroot2, "$ifish", "Results.mat")
center_Coord_data = joinpath(dataroot, "$ifish", "center_Coord.mat")
if isfile(center_Coord_data)
    @unpack center_Coord = matread(center_Coord_data)
    ROIxyz = center_Coord[judge,:] 
elseif isfile(resultdata) && ifish != 210702
    @unpack Result = matread(resultdata)
    @unpack center_Coord = Result[1]
    ROIxyz = center_Coord[judge,:] 
elseif isfile(resultdata2) && ifish != 210702
    @unpack Result = matread(resultdata2)
    @unpack center_Coord = Result[1]
    ROIxyz = center_Coord[judge,:] 
elseif isfile(resultdata3) && ifish != 210702
    @unpack Result = matread(resultdata3)
    @unpack center_Coord = Result[1]
    ROIxyz = center_Coord[judge,:] 
elseif isfile(resultdata4) && ifish != 210702
    @unpack Result = matread(resultdata4)
    @unpack center_Coord = Result[1]
    ROIxyz = center_Coord[judge,:] 
end

if @isdefined judge2
    if @isdefined ROIxyz
        if size(ROIxyz,1)==length(Judge2)
            ROIxyz=ROIxyz[judge2,:];
        end
    end
end 

#ROIxyz = ROIxyz[id,:]

query_regions_data = joinpath(dataroot, "$ifish", "query_regions_in.mat")
if isfile(query_regions_data)
    @unpack coordinates, pixelnumber, areaName = matread(query_regions_data)
end

#=
region2standardArea_data = joinpath(dataroot, "$ifish", "region2standardArea.mat")
if isfile(sequencedata)
    @unpack sequences = matread(region2standardArea_data)
    sequences = convert(Array{Int},sequences)
end
=#


#project out the neural activity in dims dimensions

ROIxyz0 = ROIxyz

(N,T)=size(a)
Cᵣ = a*a'/T 
β = N/tr(Cᵣ)
Cᵣ = Cᵣ*β
#Cₛ = construct_Toeplitz(Cᵣ)
C = copy(Cᵣ)
C[Cᵣ.>1] .= 1
id_un0 = findall(diag(Cᵣ) .!=0) 
Cᵣ =  Cᵣ[id_un0,id_un0]
ROIxyz = ROIxyz[id_un0,:]

std_c = sqrt(Diagonal(Cᵣ))
Corr = std_c\Cᵣ/std_c
Corr = (Corr + Corr')/2
