#Deploytmen
import streamlit as st
import pandas as pd
import joblib
import numpy as np 
import threading
import time
import os
import joblib

model = joblib.load('model.pkl')

def run_streamlit():
    os.system("streamlit run app.py --server.port 8501")


# Load model dan dataset gabungan
# Pastikan file 'model.pkl' dan 'film.csv' berada di direktori yang sama dengan app.py
try:
    model = joblib.load("model.pkl") # Model GaussianNB yang dilatih untuk rekomendasi film
    dataset = pd.read_csv("film.csv") # Dataset yang telah diproses (termasuk 'rating_class')
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Pastikan 'model.pkl' dan 'film.csv' ada.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model atau dataset: {e}")
    st.stop()

# Bagian relevan di app.py yang menyiapkan fitur prediksi
# ...
pred_data_df = dataset[["userId", "movieId", "year", "genre"]]
# ...
predicted_proba = model.predict_proba(pred_data_df)
# ...X = df[['userId', 'movieId', 'year', 'genre']]

st.title("🎬 Sistem Rekomendasi Film")

# Input user ID dan jumlah rekomendasi
user_input = st.text_input("Masukkan User ID:", value="1")
n_recs = st.slider("Jumlah rekomendasi film:", min_value=1, max_value=10, value=5)

if user_input: # Memastikan input tidak kosong
    if not user_input.isdigit():
        st.warning("User ID harus berupa angka.")
    else:
        user_id = int(user_input)

        # Cek apakah user_id ada di dataset
        if user_id not in dataset["userId"].unique():
            st.error(f"User ID {user_id} tidak ditemukan dalam dataset.")
        else:
            st.success(f"User ID valid: {user_id}")

            # Data film yang sudah ditonton user
            user_watched_movie_ids = dataset[dataset["userId"] == user_id]["movieId"].tolist()

            # Ambil semua film unik yang belum ditonton user
            # Pastikan kolom yang diambil sesuai dengan yang digunakan saat training dan prediksi
            all_movies_for_prediction = dataset[["movieId", "title", "genre", "year"]].drop_duplicates(subset=["movieId"])
            unseen_movies_df = all_movies_for_prediction[~all_movies_for_prediction["movieId"].isin(user_watched_movie_ids)].copy()

            if unseen_movies_df.empty:
                st.info(f"User ID {user_id} telah menonton semua film yang ada atau tidak ada film baru untuk direkomendasikan.")
            else:
                # Tambahkan kolom userId untuk prediksi
                unseen_movies_df["userId"] = user_id

                # Susun urutan kolom sesuai model yang dilatih
                # Urutan saat training: ['userId', 'movieId', 'year', 'genre']
                pred_data_df = unseen_movies_df[["userId", "movieId", "year", "genre"]]

                try:
                    # Prediksi probabilitas untuk setiap kelas rating
                    # Berdasarkan notebook Anda, LabelEncoder akan mengkodekan kelas rating sbb:
                    # 'High' -> 0
                    # 'Low'  -> 1
                    # 'Medium' -> 2
                    # Kita ingin probabilitas kelas 'High' (indeks 0)
                    predicted_proba = model.predict_proba(pred_data_df)
                    unseen_movies_df["proba_high_rating"] = predicted_proba[:, 0] # Probabilitas untuk kelas 'High'

                    # Ambil top-N berdasarkan probabilitas prediksi rating "High"
                    top_n_df = unseen_movies_df.sort_values("proba_high_rating", ascending=False).head(n_recs)

                    st.subheader(f"🎯 Rekomendasi Teratas untuk User ID {user_id}:")
                    # Tampilkan kolom yang relevan
                    st.table(top_n_df[["title", "genre", "year", "proba_high_rating"]])

                except ValueError as ve:
                    st.error(f"ValueError saat prediksi: {ve}")
                    st.error("Ini mungkin terjadi jika fitur untuk prediksi tidak cocok dengan saat training. Pastikan urutan dan nama kolom benar.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
else:
    st.info("Masukkan User ID untuk memulai.")

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

time.sleep(5)

public_url = ngrok.connect(addr=8501)
print(f"Streamlit app is live at: {public_url}")