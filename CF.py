import panda as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# Input data
    # user_id   item_id   rating

# Memory Based method
class CF(object):
    """
    Collaborative filter using lazy Knn
    Assume input schema is 
    user_id, item_id, rating
    """
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF  # user - user collaborative filtering
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0 ,2]]
        self.k = k        # number of neighbor points
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since the id starts from 0 
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
    
    def normalize_Y(self):
        first = self.Y_data[:, 0] # first col of the Y_data
        sec = self.Y_data[:, 1]   # second col of the Y_data
        if self.uuCF:
            n_first, n_sec = self.n_users , self.n_items
        else:
            n_sec, n_first = self.n_users, self.n_items
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((n_first, ))     # Create a numpy array of length n_first filled with zeros
        for n in range(n_first):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(first == n)[0].astype(np.int32)
            # indices of all ratings asscociated with user n
            item_ids = self.Y_data[ids, 1]
            # and the corresponding ratings
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings) # average rating 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize 
            self.Ybar_data[ids, 2] = ratings - self.mu[n]
            ################################################
            # form the rating matrix as a sparse matrix. Sparsity is important 
            # for both memory and computing efficiency. For example, if #user = 1M, 
            # #item = 100k, then shape of the rating matrix would be (100k, 1M), 
            # you may not have enough memory to store this. Then, instead, we store 
            # nonzeros only, and, of course, their locations.
            self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2], (self.Ybar_data)))


