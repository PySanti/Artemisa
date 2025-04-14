
knn_param_grid = {  
    'n_neighbors': [3, 5, 7, 9, 11, 15],  
    'weights': ['uniform', 'distance'],  
    'metric': ['euclidean', 'manhattan', 'chebyshev']  
}

param_grid_lr = {  
    'fit_intercept': [True, False],  
    'copy_X': [True, False]  ,
}

param_grid_lasso = {  
    'alpha': [0.001, 0.0001, 0.00001, 0.000001, 0.01],  
    'fit_intercept': [True, False],  
    'tol' : [1e-7],
}  

param_grid_ridge = {  
    'alpha': [0.001, 0.0001, 0.00001, 0.000001, 0.01],  
    'solver': ['svd', 'cholesky', 'lsqr'],  
    'fit_intercept': [True, False],  
    'tol' : [1e-7],
}  


