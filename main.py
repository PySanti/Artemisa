import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from utils.find_best_hp import find_best_hp
from sklearn.linear_model import LinearRegression
from utils.constants import *
from utils.basic_preprocess import basic_preprocess
from utils.model_performance import model_performance

target = "price"

[df_train, df_test] = basic_preprocess(pd.read_csv("./data/data.csv"), target)


knn_best_hp = find_best_hp(KNeighborsRegressor(), knn_param_grid, df_train, target)
ridge_best_hp = find_best_hp(Ridge(), param_grid_ridge, df_train, target)
lasso_best_hp = find_best_hp(Lasso(), param_grid_lasso, df_train, target)
rl_best_hp = find_best_hp(LinearRegression(), param_grid_lr, df_train, target)

model_performance(KNeighborsRegressor, knn_best_hp, df_train, df_test, target, "knn")
model_performance(Ridge, ridge_best_hp, df_train, df_test, target, "Ridge")
model_performance(Lasso, lasso_best_hp, df_train, df_test, target, "Lasso")
model_performance(LinearRegression, rl_best_hp, df_train, df_test, target, "Regresion lineal")
