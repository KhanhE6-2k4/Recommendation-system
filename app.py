# app.py
import streamlit as st
import pandas as pd
import numpy as np
from mf2 import MF2
# from recommenders import user_user, item_item, content_based, matrix_factorization
# from utils import load_data

# Đọc dữ liệu từ file u.item, ánh xạ item_id -> item.title
# Chữ f ở đầu chuỗi biến chuỗi đó thành một f-string (formatted string). Nó cho phép bạn nhúng biến trực tiếp vào chuỗi bằng cú pháp {}
r_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDB_url'] + [f'genre_{i}' for i in range(19)]

movie_info = pd.read_csv('ml-100k/u.item', sep='|', names=r_cols, encoding='latin-1') # item_info la 1 df (dataframe)
movie_title = dict(zip(movie_info['movie_id'], movie_info['title']))


st.title("🎬 Demo hệ gợi ý tổng hợp")

# Load dữ liệu
# ratings, movies = load_data()

st.sidebar.header("📌 Tùy chọn")
# method = st.sidebar.radio("Chọn phương pháp gợi ý:", 
#             ["User-User", "Item-Item", "Content-Based", "Matrix Factorization"])

# Load model tương ứng vào
mf_model = MF2.load("model/mf_model")

# user_ids = sorted(np.unique(mf_model.Y_raw_data[:, 0]))
# Dùng dòng trên khi id ko tăng dần
# Do user_id trong file ub.base là tăng dần nên và liên tục nên ta ko cần xử lý phức tạp với data
# Chú ý là user_id trong file ub.base bắt đầu từ 1 còn ta cài đặt thuật toán thì sử dụng chỉ số từ 0 do ma trận trong python idx bắt đầu từ 0
user_ids = [x for x in range(1, mf_model.n_users)] # danh sách id theo file ub.base
user_id = st.sidebar.selectbox("Chọn User:", user_ids)

top_k = st.sidebar.slider("Số lượng gợi ý", 5, 20, 10)

# Nút gợi ý
if st.button("🎁 Gợi ý cho tôi!"):
    # if method == "User-User":
    #     recs = user_user.recommend(user_id, ratings, movies, top_k)
    # elif method == "Item-Item":
    #     recs = item_item.recommend(user_id, ratings, movies, top_k)
    # elif method == "Content-Based":
    #     recs = content_based.recommend(user_id, ratings, movies, top_k)
    # elif method == "Matrix Factorization":
    #     mf_model = matrix_factorization.load_model("mf_model/mf")  # path đến mô hình đã lưu
    #     recs = matrix_factorization.recommend(user_id, mf_model, movies, top_k)

    # Lấy dự đoán cho tất cả item mà user chưa đánh giá
    predictions = mf_model.pred_for_user(user_id - 1)  # Trừ user_id đi 1 do ta chuyển từ id trong file ub.base thành id sử dụng trong model
    # Sắp xếp theo predicted rating giảm dần là lấy ra top_k item đầu tiên
    top_k_recs = sorted(predictions, key = lambda x: x[1], reverse=True)[:top_k]

    # Hiển thi danh sách gợi ý
    st.subheader("📢 Danh sách gợi ý:")
    for movie_id, rating in top_k_recs:
        st.write(f"🎬 {movie_title[movie_id]} — rating: ⭐ {rating:.2f}")

# !!!
# Khi chạy từ Run/Debug trong VSCode hoặc dùng nút ▶ Run, thì Python được chạy qua môi trường wrapper
# Streamlit phát hiện điều đó và cảnh báo rằng một số chức năng hiếm (như hot-reload hoặc log) có thể không hoạt động tối ưu.

# Chạy bằng streamlit run app.py trong vsc terminal
