import numpy as np
import pandas as pd
from mf2 import MF2

# Đọc dữ liệu test
r_cols = ['user_id', 'item_id', 'rating', 'unix_timestamp']
ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')
rate_test = ratings_test.values
rate_test[:, :2] -= 1

# Load mô hình
rs_loaded = MF2.load("model/mf_model")
rmse = rs_loaded.evaluate_RMSE(rate_test)
print("RMSE (loaded model): ", rmse)