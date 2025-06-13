import numpy as np
import json
from tqdm import tqdm  # For progress bar
from scipy.sparse import csr_matrix

class MF2:
    def __init__(self, Y_data, K, lam=0.1, Xinit=None, Winit=None, 
                 learning_rate=0.5, max_iter=1000, print_every=100, 
                 user_based=True, early_stopping=True, patience=5):
        # Validate input
        # assert Y_data.shape[1] == 3, "Y_data must have 3 columns: user_id, item_id, rating"
        # assert np.all(Y_data[:, 2] >= 1) and np.all(Y_data[:, 2] <= 5), "Ratings must be in [1,5]"
        
        self.Y_raw_data = Y_data
        self.K = K
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.user_based = bool(user_based)
        self.early_stopping = early_stopping
        self.patience = patience
        
        # Convert to 0-based index if needed
        if np.min(Y_data[:, :2]) > 0:
            Y_data[:, :2] -= 1
            
        self.n_users = int(np.max(Y_data[:, 0])) + 1
        self.n_items = int(np.max(Y_data[:, 1])) + 1
        
        # Initialize factors with Xavier initialization
        if Xinit is None:
            self.X = np.random.normal(0, 1/np.sqrt(K), (self.n_items, K))
        else:
            self.X = Xinit
            
        if Winit is None:
            self.W = np.random.normal(0, 1/np.sqrt(K), (K, self.n_users))
        else:
            self.W = Winit

            
        # Create sparse matrices for faster lookup
        self._create_sparse_matrices()
        
    def _create_sparse_matrices(self):
        """Create CSR matrices for fast row/column access"""
        users, items, ratings = self.Y_raw_data[:, 0], self.Y_raw_data[:, 1], self.Y_raw_data[:, 2]
        self.user_item_matrix = csr_matrix((ratings, (users, items)), 
                                           shape=(self.n_users, self.n_items))
        self.item_user_matrix = self.user_item_matrix.T.tocsr()
        
    # def normalize_Y(self):
    #     """Normalize using global mean if user/item has no ratings"""
    #     if self.user_based:
    #         user_means = np.array(self.user_item_matrix.mean(axis=1)).flatten()
    #         global_mean = np.nanmean(user_means)
    #         self.mu = np.nan_to_num(user_means, nan=global_mean)
    #     else:
    #         item_means = np.array(self.item_user_matrix.mean(axis=1)).flatten()
    #         global_mean = np.nanmean(item_means)
    #         self.mu = np.nan_to_num(item_means, nan=global_mean)
            
    #     # Create normalized copy
    #     self.Y_data_n = self.Y_raw_data.copy()
    #     norm_col = 0 if self.user_based else 1
    #     for idx in range(len(self.mu)):
    #         mask = self.Y_data_n[:, norm_col] == idx
    #         self.Y_data_n[mask, 2] -= self.mu[idx]

    # def normalize_Y(self):
    #     user_col = 0 if self.user_based else 1
    #     item_col = 1 - user_col
    #     n_objects = self.n_users if self.user_based else self.n_items

    #     self.mu = np.zeros((n_objects,))
    #     self.Y_data_n = self.Y_raw_data.copy().astype(np.float64)

    #     # Lấy danh sách ID (user hoặc item) cho toàn bộ dòng
    #     ids = self.Y_data_n[:, user_col].astype(np.int32)

    #     # Tạo tổng rating và số lượng rating cho mỗi user/item
    #     rating_sums = np.bincount(ids, weights=self.Y_data_n[:, 2])
    #     rating_counts = np.bincount(ids)

    #     # Tránh chia cho 0
    #     nonzero_mask = rating_counts != 0
    #     self.mu[nonzero_mask] = rating_sums[nonzero_mask] / rating_counts[nonzero_mask]

    #     # Chuẩn hóa ratings
    #     self.Y_data_n[:, 2] -= self.mu[ids]

    def normalize_Y(self):
        user_col = 0 if self.user_based else 1
        item_col = 1 - user_col
        n_objects = self.n_users if self.user_based else self.n_items

        self.mu = np.zeros((n_objects,))
        self.Y_data_n = self.Y_raw_data.copy().astype(np.float64)

        # Lấy danh sách ID (user hoặc item)
        ids = self.Y_data_n[:, user_col].astype(np.int32)

        # Tạo tổng rating và số lượng rating cho mỗi user/item
        rating_sums = np.bincount(ids, weights=self.Y_data_n[:, 2])
        rating_counts = np.bincount(ids)

        # Tìm ra những ID hợp lệ (có rating)
        valid_ids = np.arange(len(rating_counts))[rating_counts != 0]

        # Chỉ cập nhật self.mu ở các ID hợp lệ
        self.mu[valid_ids] = rating_sums[valid_ids] / rating_counts[valid_ids]

        # Chuẩn hóa ratings
        self.Y_data_n[:, 2] -= self.mu[ids]
    
    def loss(self):
        """Vectorized loss calculation"""
        users = self.Y_data_n[:, 0].astype(int)
        items = self.Y_data_n[:, 1].astype(int)
        ratings = self.Y_data_n[:, 2]
        
        predictions = np.sum(self.X[items] * self.W[:, users].T, axis=1)
        mse = np.mean((ratings - predictions)**2)
        reg = 0.5 * self.lam * (np.sum(self.X**2) + np.sum(self.W**2))
        return mse + reg
    
    def _get_rated_indices(self, entity_id, by_user=True):
        """Get rated items/users using sparse matrix"""
        if by_user:
            return self.user_item_matrix[entity_id].indices
        return self.item_user_matrix[entity_id].indices
    
    def updateX(self):
        """Vectorized update for X"""
        for m in range(self.n_items):
            user_ids = self._get_rated_indices(m, by_user=False)
            if len(user_ids) == 0:
                continue
                
            W_m = self.W[:, user_ids]
            ratings_m = self.Y_data_n[self.Y_data_n[:, 1] == m, 2]
            error = ratings_m - self.X[m].dot(W_m)
            
            grad = -error.dot(W_m.T)/len(user_ids) + self.lam*self.X[m]
            self.X[m] -= self.learning_rate * grad
    
    def updateW(self):
        """Vectorized update for W"""
        for n in range(self.n_users):
            item_ids = self._get_rated_indices(n, by_user=True)
            if len(item_ids) == 0:
                continue
                
            X_n = self.X[item_ids]
            ratings_n = self.Y_data_n[self.Y_data_n[:, 0] == n, 2]
            error = ratings_n - X_n.dot(self.W[:, n])
            
            grad = -X_n.T.dot(error)/len(item_ids) + self.lam*self.W[:, n]
            self.W[:, n] -= self.learning_rate * grad
    
    def fit(self, train_data, val_data = None):
        """Training with early stopping"""
        self.normalize_Y()
        # best_loss = float('inf')
        best_val_rmse = float('inf')
        no_improvement = 0
        
        for it in tqdm(range(self.max_iter), desc="Training"):
            self.updateX()
            self.updateW()
            
            # current_loss = self.loss()
            
            # if (it + 1) % self.print_every == 0:
            #     print(f"Iter {it+1}: Loss = {current_loss:.4f}")
                
            # if self.early_stopping:
            #     if current_loss < best_loss:
            #         best_loss = current_loss
            #         no_improvement = 0
            #     else:
            #         no_improvement += 1
                    
            #     if no_improvement >= self.patience:
            #         print(f"Early stopping at iter {it+1}")
            #         break
            if val_data is not None:
                val_rmse = self.evaluate_RMSE(val_data) # Tính rmse trên validation data

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement >= self.patience:
                    break # Dừng nếu validation RMSE không cải thiện
    
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
    
    def add_new_user(self, new_user_ratings): # Hàm này chỉ dùng với item-based (để tối ưn hơn)
        """
        Thêm người dùng mới vào mô hình

        Args: 
            new_user_ratings: List hoặc ndarray (n, 2) gồm [item_id, rating],
                              item_id là index trong range(0, n_items)
        """
        new_user_id = self.n_users
        
        # Tạo dữ liệu mới: [user_id, item_id, rating]
        new_data = np.array([[new_user_id, item, rating] for item, rating in new_user_ratings])


        # Tính trung bình rating của từng item (mu[i] đã có sẵn)
         # Có thể chấp nhận nếu số lượng đánh giá cũ của item i lớn, vì thêm một đánh giá mới không ảnh hưởng đáng kể tới TB đánh giá với item i
        item_ids = new_data[:, 1].astype(int)
        normalized_ratings = new_data[:, 2] - self.mu[item_ids]

        # Lưu lại chuẩn hóa
        new_data[:, 2] = normalized_ratings

         # Gộp vào tập dữ liệu
        new_data = np.hstack([new_data, np.zeros((new_data.shape[0], 1))])
        self.Y_raw_data = np.vstack((self.Y_raw_data, new_data))

        self.n_users += 1
        self.mu = np.append(self.mu, 0.0)  # chỉ để giữ đồng bộ shape, không dùng mu[u]

        # Thêm W mới
        new_W = np.random.normal(0, 1/np.sqrt(self.K), (self.K, 1))
        self.W = np.hstack((self.W, new_W))

        # Cập nhật sparse matrix
        self._create_sparse_matrices()

        self.normalize_Y()

        # Huấn luyện W mới trong vài vòng
        for i in range(50):
            self.updateW()
            self.updateX()

    def save(self, path_prefix):
        np.savez_compressed(f"{path_prefix}_factors.npz", X=self.X, W=self.W, mu=self.mu, Y_raw=self.Y_raw_data)
        meta = {
            "K": self.K, "lam": self.lam, "n_users": self.n_users, "n_items": self.n_items,
            "user_based": self.user_based
        }
        with open(f"{path_prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path_prefix):
        with open(f"{path_prefix}_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        data = np.load(f"{path_prefix}_factors.npz")
        
        Y_raw = data["Y_raw"]
        model = cls(Y_raw, K=meta["K"], lam=meta["lam"], user_based=meta["user_based"])
        model.n_users = meta["n_users"]
        model.n_items = meta["n_items"]
        model.X = data["X"]
        model.W = data["W"]
        model.mu = data["mu"]

        # Tạo lại dữ liệu chuẩn hóa
        model.normalize_Y()
        return model