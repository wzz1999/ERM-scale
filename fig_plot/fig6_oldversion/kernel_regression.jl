
if !@isdefined dpi
    dpi =300
end

ϵ = 1
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

#id3 = sample(findall(0.001 .<D2[0.999 .>Corr.>0].<0.01),3333,replace = true)
id = vcat(id1,id2,id3)
#id = vcat(id1,id2)
x_train = D2[0.999 .>Corr.>0][id] #-5:0.5:5
x_test = 0.01:0.01:round(1.5*L)#-7:0.1:7
if n==2
    x_test = 0.1:0.01:min(10,round(1.5*L))./ϵ
end

y_train = Corr[0.999 .>Corr.>0][id]#f_truth.(x_train) + noise
y_test = corr(x_test,p) #f_truth.(x_test)

#plot(x_test, y_test; label=raw"$f(x)$")
#Plots.scatter!(x_train, y_train,xaxis = :log, yaxis = :log)

function error_bar(X_test,X,Y)
    n = length(X_test)
    X_error = zeros(n)
    for i = 1:n
        id = findall(X_test[i]*0.9 .<X .<X_test[i]*1.1)
        X_error[i] = std(Y[id])
    end
    return X_error
end

function kernel_regression(k, X, y, Xstar)
    kstar = kernelmatrix(k, Xstar, X)
    return kstar * y ./ sum(kstar, dims =2)
end;

function kernel_ridge_regression(k, X, y, Xstar, lambda)
    K = kernelmatrix(k, X)
    kstar = kernelmatrix(k, Xstar, X)
    return kstar * ((K + lambda * I) \ y)
end;

function kernelized_fit_and_plot(kernel, lambda=1e-3)
    y_pred = kernel_ridge_regression(kernel, x_train, y_train, x_test, lambda)
    #y_pred = kernel_regression(kernel, x_train, y_train, x_test)
    title = "F"
    y_error = error_bar(x_test,x_train,y_train)
    #Plots.scatter(x_train, y_train; label=nothing,xaxis = :log, yaxis = :log)
    plot(x_test, y_test; label="Model "*L"f(\vec{x})",grid = :none, dpi = dpi)#,xaxis = :log, yaxis = :log)
    return plot!(x_test, y_pred;ribbon = y_error, label="Data "*L"f(\vec{x})", title=title,xlabel = "distance", ylabel = "correlation")
end

function kernelized_fit_and_plot_loglog(kernel, lambda=1e-3)
    y_pred = kernel_ridge_regression(kernel, x_train, y_train, x_test, lambda)
    title = "F"
    y_error = error_bar(x_test,x_train,y_train)
    #Plots.scatter(x_train, y_train; label=nothing,xaxis = :log, yaxis = :log)
    plot(x_test/ϵ, y_test; label="f(x)",grid = :none,xaxis = :log, yaxis = :log, legend=:bottomleft, legendfontsize=6, dpi = dpi)
    return plot!(x_test/ϵ, y_pred;ribbon = y_error, label="", title=title,xlabel = "distance", ylabel = "correlation")
end

function kernelized_fit_and_subplot(kernel, lambda=1e-3)
    y_pred = kernel_ridge_regression(kernel, x_train, y_train, x_test, lambda)
    title = "F"
    y_error = error_bar(x_test,x_train,y_train)
    #Plots.scatter(x_train, y_train; label=nothing,xaxis = :log, yaxis = :log)
    return plot!(x_test, y_test; label=nothing,grid = :none,inset = (5, bbox(0, 0.5, 0.5, 0.5, :left)),subplot = 8)#,xaxis = :log, yaxis = :log)
    #return plot!(x_test, y_pred;ribbon = y_error, label=nothing,xlabel = nothing, ylabel = nothing,inset = (5, bbox(0, 0.5, 0.5, 0.5, :left)),subplot = 8)
end

#plot((kernelized_fit_and_plot(PolynomialKernel(; degree=degree, c=1)) for degree in 1:4)...)

#kernelized_fit_and_plot(GaussianKernel())
#kernelized_fit_and_plot(SqExponentialKernel())


#x_error = error_bar(x_test,x_train,y_train)
#=
function linear_regression(X, y, Xstar)
    weights = (X' * X) \ (X' * y)
    return Xstar * weights
end;

y_pred = linear_regression(x_train, y_train, x_test)
Plots.scatter(x_train, y_train; label="observations",xaxis = :log, yaxis = :log)
plot!(x_test, y_pred; label="linear fit")

function featurize_poly(x; degree=1)
    return repeat(x, 1, degree + 1) .^ (0:degree)'
end

function featurized_fit_and_plot(degree)
    X = featurize_poly(x_train; degree=degree)
    Xstar = featurize_poly(x_test; degree=degree)
    y_pred = linear_regression(X, y_train, Xstar)
    Plots.scatter(x_train, y_train; legend=false, title="fit of order $degree")
    return plot!(x_test, y_pred)
end

plot((featurized_fit_and_plot(degree) for degree in 1:4)...)

function ridge_regression(X, y, Xstar, lambda)
    weights = (X' * X + lambda * I) \ (X' * y)
    return Xstar * weights
end

function regularized_fit_and_plot(degree, lambda)
    X = featurize_poly(x_train; degree=degree)
    Xstar = featurize_poly(x_test; degree=degree)
    y_pred = ridge_regression(X, y_train, Xstar, lambda)
    Plots.scatter(x_train, y_train; legend=false, title="\$\\lambda=$lambda\$",xaxis = :log, yaxis = :log)
    return plot!(x_test, y_pred)
end

plot((regularized_fit_and_plot(20, lambda) for lambda in (1e-3, 1e-2, 1e-1, 1))...)
=#

#plot(1:3, 1:3,ribbon = ([1 1 1],[2 2 2]))

function plot_fig_E(ifish0; load_data_root = "../src/load_fish_data2.jl")
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
    id = vcat(id1,id2,id3)
    global x_train = D2[0.999 .>Corr.>0][id]#-5:0.5:5
    global x_test = 0.1:0.01:round(1.5*L)#-7:0.1:7
    if n==2
        global x_test = 0.1:0.01:min(10,round(1.5*L))
    end
    if ifish==201106
        global x_test = 0.01:0.01:min(10,round(2*L))
    end
    global y_train = Corr[0.999 .>Corr.>0][id]#f_truth.(x_train) + noise
    global y_test = corr(x_test,p) #f_truth.(x_test)
    lambda=1e-3; kernel= GaussianKernel()
    y_pred = kernel_ridge_regression(kernel, x_train, y_train, x_test, lambda)
    title = "E"
    y_error = error_bar(x_test,x_train,y_train)
    #Plots.scatter(x_train, y_train; label=nothing,xaxis = :log, yaxis = :log)
    p5 = plot(x_test, y_test; label="f(x) of model",grid = :none,xaxis = :log, yaxis = :log, legend=:bottomleft, legendfontsize=6, ylim = (0.01,1))
    p5 = plot!(x_test, y_pred;ribbon = y_error, label=raw"f(x) of data", title=title,xlabel = "distance", ylabel = "correlation")
    return p5
end


#plot_fig_E(201106)