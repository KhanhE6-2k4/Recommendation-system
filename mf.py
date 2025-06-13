import numpy as np
import json
import pandas as pd
import os

class MF:
    def __init__(self, Y_data, K, lam=0.1, Xinit=None, Winit=None, 
                 learning_rate=0.5, max_iter=1000, print_every=100, user_based=1):
        self.Y_raw_data = Y_data              # Ma tran rating goc
        self.K = K                            # So chieu an(latent factors) - do phuc tap bieu dien nguoi dung / san pham
        self.lam = lam                        # he so regularization de tranh overfitting
        self.learning_rate = learning_rate    # toc do hoc
        self.max_iter = max_iter              # So vong lap
        self.print_every = print_every        # In ra log moi n vong lap
        self.user_based = user_based          # Chuan hoa theo user/item, 1 -> user, 0 -> item
        self.n_users = int(np.max(Y_data[:, 0])) + 1  # So luong user = gia tri lon nhat tren cot 0 cua ma tran rating + 1 (do user_id bat dau tu 0) 
        self.n_items = int(np.max(Y_data[:, 1])) + 1  # So luong iten = gia tri lon nhat tren cot 1 cua ma tran rating + 1 (do item_id bat dau tu 0)

        self.X = np.random.randn(self.n_items, K) if Xinit is None else Xinit   # Ma tran dac trung cua cac item vs KT (so luong item * latent factors)
		                                                                        # Moi hang cua X bieu dien 1 item trong khong gian an K chieu
																				# Neu Xinit khong duoc cung cap, khoi tao ngau nhien bang np.random,.rand() (co phan phoi chuan ~ N(0, 1))
																				
        self.W = np.random.randn(K, self.n_users) if Winit is None else Winit   # Ma tran dac trung cua cac user vs KT (latent factors * so luong item)

        self.n_ratings = Y_data.shape[0]        # So luong danh gia (ratings) hien co = so donng cua ma tran rating 
        self.Y_data_n = self.Y_raw_data.copy()  # Tao ban sao de thuc hien chuan hoa va tien xu ly (Chuyen doi chi so cua user-id hoac item-id sang STT khong am) 
		
		# Cong thuc du doan rating cua user u cho item i:
		#  r_ui = X[i] . W[:, u] -> So thuc bieu dien rating du doan
		# voi X[i] la vector dac trung cua item i, W[:, u] la vector dac trung cua user u

    # Ham chuẩn hóa
    def normalize_Y(self):
        user_col = 0 if self.user_based else 1
        item_col = 1 - user_col
        n_objects = self.n_users if self.user_based else self.n_items  # user-based -> n_objects = n_users, else n_objects = n_items

        users = self.Y_raw_data[:, user_col]   # vector-id user (hoac item) cho tung dong rating
        self.mu = np.zeros((n_objects,))       # trung binh rating cua moi user hay item (vector do dai n_objects voi cac phan tu toan 0). VD: n = 3 -> [0, 0, 0]
        for n in range(n_objects):
            ids = np.where(users == n)[0].astype(np.int32)  # Tim toan bo chi so dong trong ma tran rating goc Y_data ma user (hoặc item) có id  = n
            item_ids = self.Y_data_n[ids, item_col]         # lay ra danh sach cac item-id tuong ung voi cac dong danh gia ma user n da danh gia
            ratings = self.Y_data_n[ids, 2]                 # lay ra mang ratings cua user n
            m = np.mean(ratings)                            # Lay ra trung binh rating cua user n
            self.mu[n] = 0 if np.isnan(m) else m            # np.mean([]) -> nan, khi ratings la mang rong (co nghia la user n chua danh gia cai gi) thi tb rating của user n = 0 
            self.Y_data_n[ids, 2] = ratings - self.mu[n]    # Chuẩn hóa: thay giá trị rating mới = rating cũ - (tb rating)

    # Hàm mất mát
    def loss(self):
        L = 0 
        for i in range(self.Y_data_n.shape[0]):   # self.Y_data_n.shape[0] là số hàng của ma trận rating 
            n, m, rate = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i, 1]), self.Y_data_n[i, 2] # lấy các giá trị user_id, item_id, rating ứng với hàng i
            L += 0.5 * (rate - self.X[m, :].dot(self.W[:, n])) ** 2 # L += 1/2 * (rmn - xm un) ^ 2

        L /= self.n_ratings  # L = 1/2n * (rmn - xm un) ^ 2 - lý do sử dụng 1/2 là để thuận tiện cho việc lấy đao hàm
        L += 0.5 * self.lam * (np.linalg.norm(self.X, 'fro') ** 2 + np.linalg.norm(self.W, 'fro') ** 2) # L = Mean Square Error + Regularization
        return L 

    def get_items_rated_by_user(self, user_id):
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0] # vector chỉ số những hàng ứng với user với user_id truyền vảo
        item_ids = self.Y_data_n[ids, 1].astype(np.int32) # vector items_id với điều kiện item đượt rate bởi user với user_id truyền vào
        ratings = self.Y_data_n[ids, 2]                   # vector ratings với điều kiện rating đươc thực hiện bởi user với user_id truyền vào 
        return item_ids, ratings                          # Trả về cả hai

    def get_users_who_rate_item(self, item_id):           # tương tự hàm trên nhưng đầu vào là item_id
        ids = np.where(self.Y_data_n[:, 1] == item_id)[0] 
        user_ids = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return user_ids, ratings

    def updateX(self): # Cố định W tối ưu X
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m) 
            Wm = self.W[:, user_ids]
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T) / self.n_ratings + self.lam * self.X[m, :]
            self.X[m, :] -= self.learning_rate * grad_xm.reshape((self.K,))

    def updateW(self): # Cố định X tối ưu W
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n])) / self.n_ratings + self.lam * self.W[:, n]
            self.W[:, n] -= self.learning_rate * grad_wn.reshape((self.K,))

# đến đây rồi
    def fit(self):  # hàm huấn luyện mô hình Matrix Factorization sử dụng kỹ thuật Alternating Least Squares
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                print(f"iter = {it + 1}, loss = {self.loss()}, RMSE train = {rmse_train}")
    
    # Dùng để dự đoán giá trị rating cho 1 cặp user - item riêng lẻ -> sử dụng trong lúc đánh giá
    def pred(self, u, i):
        u, i = int(u), int(i)
        bias = self.mu[u] if self.user_based else self.mu[i]
        pred = self.X[i, :].dot(self.W[:, u]) + bias
        return min(5, max(1, pred))
    
    # Dùng để gợi ý danh sách sản phẩm cho người dùng (recommendation list)
    def pred_for_user(self, user_id):
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_data_n[ids, 1].tolist()
        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]
        return [(i, y_pred[i]) for i in range(self.n_items) if i not in items_rated_by_u]
    
    # Tinh toán RMSE (Root mean square error - Độ lêch bình phương trung bình gốc)
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2]) ** 2
        return np.sqrt(SE / n_tests)

    def save(self, path_prefix):
        np.savez_compressed(f"{path_prefix}_factors.npz", X=self.X, W=self.W, mu=self.mu)
        meta = {"K": self.K, "lam": self.lam, "n_users": self.n_users, "n_items": self.n_items, "user_based": self.user_based}
        with open(f"{path_prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path_prefix):
        with open(f"{path_prefix}_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        dummy_data = np.zeros((1, 3))
        model = cls(dummy_data, K=meta["K"], lam=meta["lam"])
        model.n_users = meta["n_users"]
        model.n_items = meta["n_items"]
        model.user_based = meta["user_based"]
        data = np.load(f"{path_prefix}_factors.npz")
        model.X = data["X"]
        model.W = data["W"]
        model.mu = data["mu"]
        return model
    
# r_cols = ['user_id', 'item_id', 'rating', 'unix_timestamp']
# ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
# ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

# rate_train = ratings_base.values
# rate_test = ratings_test.values

# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1

# rs = MF(rate_train, K = 100, lam = .01, print_every = 20, learning_rate = 2, max_iter = 200)

# # Huan luyen mo hinh
# rs.fit()

# # Đảm bảo thư mục 'model' tồn tại
# os.makedirs("model", exist_ok=True)

# # luu lai mo hinh sau khi huan luyen
# rs.save('model/mf_model') 
# print('Huan luyen hoan thanh!\n')
