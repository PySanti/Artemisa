from sklearn.model_selection import GridSearchCV

def find_best_hp(estimator, param_grid, df_train, target):
    core_count = 5
    return GridSearchCV(
        estimator=estimator, 
        param_grid=param_grid,
        verbose=10,
        cv=4,
        n_jobs=core_count,
        scoring="r2",
    ).fit(df_train.drop(target, axis=1), df_train[target]).best_params_


