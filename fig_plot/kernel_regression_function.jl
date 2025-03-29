function error_bar(X_test,X,Y)
    n = length(X_test)
    X_error = zeros(n)
    for i = 1:n
        id = findall(X_test[i]*0.9 .<X .<X_test[i]*1.1)
        X_error[i] = std(Y[id])
    end
    return X_error
end

function kernel_ridge_regression(k, X, y, Xstar, lambda)
    K = kernelmatrix(k, X)
    kstar = kernelmatrix(k, Xstar, X)
    return kstar * ((K + lambda * I) \ y)
end;

function kernelized_fit_and_plot(kernel, lambda=1e-3)
    y_pred = kernel_ridge_regression(kernel, x_train, y_train, x_test, lambda)
    title = "E"
    y_error = error_bar(x_test,x_train,y_train)
    #Plots.scatter(x_train, y_train; label=nothing,xaxis = :log, yaxis = :log)
    plot(x_test, y_test; label=raw"$f(x)$",grid = :none, dpi = dpi)#,xaxis = :log, yaxis = :log)
    return plot!(x_test, y_pred;ribbon = y_error, label=nothing, title=title,xlabel = "distance", ylabel = "correlation")
end

function kernelized_fit_and_plot_loglog(kernel, lambda=1e-3)
    y_pred = kernel_ridge_regression(kernel, x_train, y_train, x_test, lambda)
    title = "E"
    y_error = error_bar(x_test,x_train,y_train)
    #Plots.scatter(x_train, y_train; label=nothing,xaxis = :log, yaxis = :log)
    plot(x_test, y_test; label="f(x) of model",grid = :none,xaxis = :log, yaxis = :log, legend=:bottomleft, legendfontsize=6, dpi = dpi)
    return plot!(x_test, y_pred;ribbon = y_error, label=raw"f(x) of data", title=title,xlabel = "distance", ylabel = "correlation")
end

function kernelized_fit_and_subplot(kernel, lambda=1e-3)
    y_pred = kernel_ridge_regression(kernel, x_train, y_train, x_test, lambda)
    title = "E"
    y_error = error_bar(x_test,x_train,y_train)
    #Plots.scatter(x_train, y_train; label=nothing,xaxis = :log, yaxis = :log)
    return plot!(x_test, y_test; label=nothing,grid = :none,inset = (5, bbox(0, 0.5, 0.5, 0.5, :left)),subplot = 8)#,xaxis = :log, yaxis = :log)
    #return plot!(x_test, y_pred;ribbon = y_error, label=nothing,xlabel = nothing, ylabel = nothing,inset = (5, bbox(0, 0.5, 0.5, 0.5, :left)),subplot = 8)
end