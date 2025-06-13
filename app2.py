import streamlit as st
import pandas as pd
import numpy as np
from mf2 import MF2
import random

r_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDB_url'] + [f'genre_{i}' for i in range(19)]

movie_info = pd.read_csv('ml-100k/u.item', sep='|', names=r_cols, encoding='latin-1') # item_info la 1 df (dataframe)
movie_title = dict(zip(movie_info['movie_id'], movie_info['title']))
title_to_id = {title : id for id, title in movie_title.items()}


# Đọc dữ liệu test
r_cols = ['user_id', 'item_id', 'rating', 'unix_timestamp']
ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')
rate_test = ratings_test.values
rate_test[:, :2] -= 1

st.title("🎬 Demo hệ gợi ý tổng hợp")
st.subheader("Chọn ít nhất 5 phim yêu thích để nhận gợi ý!")

movie_choices = movie_info['title'].tolist()
selected_movies = st.multiselect("Chọn phim yêu thích: ", movie_choices)

if st.button("🎁 Gợi ý cho tôi!"):
    if len(selected_movies) < 5:
        st.warning("⚠️ Bạn cần chọn ít nhất 5 bộ phim.")
    else:
        # Mapping title -> movie_id và gán rating = 5.0 cho mỗi bộ phim yêu thích
        selected_ids = [title_to_id[title] for title in selected_movies]
        # selected_ids = random.sample(range(1000), 200)
        # new_user_ratings = [[movie_id, 5.0] for movie_id in selected_ids]

        # Mỗi phim yêu thích sẽ có rating ngẫu nhiên từ 4.0 đến 5.0 thay vì luôn là 5.0 -> giúp dữ liệu mô phỏng chân thực hơn.
        new_user_ratings = [[movie_id, round(random.uniform(4.0, 5.0), 1)] for movie_id in selected_ids]

        # Load model và dự đoán
        mf_model = MF2.load("model/mf_model")
        mf_model.add_new_user(new_user_ratings)

        print(f"{mf_model.evaluate_RMSE(rate_test)}")


        predictions = mf_model.pred_for_user(mf_model.n_users - 1)
        top_k_recs = sorted(predictions, key = lambda x: x[1], reverse=True)[:10]

        st.markdown("## 📢 Gợi ý theo Matrix Factorization:")
        for movie_id, rating in top_k_recs:
            if movie_id not in selected_ids:
                rating = min(5, max(1, round(rating, 1)))
                st.write(f"🎬 {movie_title[movie_id]} — ⭐ {rating:.1f}")
