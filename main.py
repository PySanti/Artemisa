import pandas as pd
from utils.basic_preprocess import basic_preprocess

[df_train, df_test] = basic_preprocess(pd.read_csv("./data/data.csv"), "price")


