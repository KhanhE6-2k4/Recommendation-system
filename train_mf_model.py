import numpy as np
import os
import pandas as pd
from mf import MF
from mf2 import MF2
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
# r_cols = ['user_id', 'item_id', 'rating', 'unix_timestamp']
# ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
# rate_train = ratings_base.values
# rate_train[:, :2] -= 1

# # Khởi tạo và train model
# rs = MF2(rate_train, K=100, lam=0.01, print_every=20, learning_rate=2, max_iter=500)
# rs.fit()

# # Tạo thư mục lưu model
# os.makedirs("model", exist_ok=True)

# # Lưu mô hình
# rs.save("model/mf_model")
# print("Mô hình đã được huấn luyện!")


# 1. Đọc và chuẩn bị dữ liệu
def load_data():
    r_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    
    # Đọc tập train và test
    ratings_train = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')
    
    # Chuyển sang mảng numpy và chuyển index về 0-based
    train_data = ratings_train.values
    test_data = ratings_test.values
    train_data[:, :2] -= 1
    test_data[:, :2] -= 1
    
    # Chia train thành train/validation
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    return train_data, val_data, test_data

# 2. Huấn luyện và đánh giá mô hình
def train_and_evaluate():
    # Load dữ liệu
    train_data, val_data, test_data = load_data()
    
    # Khởi tạo model với tham số hợp lý hơn
    model = MF2(
        train_data, 
        K=10,                  # Giảm K so với 100 để tránh overfitting 30
        lam=0.1,                # 0.01
        learning_rate=0.75,     # Giảm learning rate từ 2 xuống 0.05
        max_iter=100,            # 100
        print_every=10,
        user_based=False,
        early_stopping=True,   # Thêm early stopping
        patience=5
    )
    
    # Huấn luyện với validation set
    model.fit(train_data, val_data)
    
    # Đánh giá trên tập test
    test_rmse = model.evaluate_RMSE(test_data)
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Lưu model
    os.makedirs("model", exist_ok=True)
    model.save("model/mf_model")
    print("Mô hình đã được huấn luyện và lưu!")

if __name__ == "__main__":
    train_and_evaluate()