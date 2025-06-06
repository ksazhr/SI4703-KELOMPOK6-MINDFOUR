import pandas as pd
import streamlit as st
import pandas as pd
from pyngrok import ngrok
import threading
import time
import os
import joblib

# Load model dan dataset
try:
    model = joblib.load("model.pkl")
    dataset = pd.read_csv("film.csv")
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e}")
    st.stop()
except Exception as e:
    st.error(f"Kesalahan saat memuat file: {e}")
    st.stop()

st.title("🎬 Sistem Rekomendasi Film Berdasarkan Preferensi")

# Mapping genre ID ke nama
genre_mapping = {
    1: 'Adventure', 2: 'Comedy', 3: 'Action', 4: 'Drama', 5: 'Crime', 6: 'Children',
    7: 'Mystery', 8: 'Animation', 9: 'Documentary', 10: 'Thriller', 11: 'Horror',
    12: 'Fantasy', 13: 'Western', 14: 'Film-Noir', 15: 'Romance', 16: 'Sci-Fi',
    17: 'Musical', 18: 'War', 19: '(no genres listed)'
}
reverse_genre_mapping = {v: k for k, v in genre_mapping.items()}

# Sidebar: Input preferensi pengguna
st.sidebar.header("Preferensi Anda")

# Rentang tahun
min_year = 1900
max_year = int(dataset["year"].max())
year_range = st.sidebar.slider("Rentang Tahun Film:", min_value=min_year, max_value=max_year, value=(min_year, max_year))

# Genre
genre_names = ["Tidak Ada Preferensi"] + list(genre_mapping.values())
genre_input_name = st.sidebar.selectbox("Genre:", options=genre_names)

# Rating minimum
min_rating_input = st.sidebar.slider("Minimum Rating", 0.0, 1.0, 0.5, 0.05)

# Jumlah rekomendasi
n_recs = st.slider("Jumlah Rekomendasi:", 1, 10, 5)

# Persiapan data film
film_df = dataset[["movieId", "title", "genre", "year"]].drop_duplicates(subset="movieId").copy()

# Filter tahun
film_df = film_df[(film_df["year"] >= year_range[0]) & (film_df["year"] <= year_range[1])]
st.info(f"🎞️ Filter tahun: {year_range[0]} - {year_range[1]}")

# Filter genre
if genre_input_name != "Tidak Ada Preferensi":
    genre_input = reverse_genre_mapping.get(genre_input_name)
    film_df = film_df[film_df["genre"] == genre_input]
    st.info(f"🎭 Filter genre: {genre_input_name}")

if film_df.empty:
    st.warning("Tidak ada film yang sesuai filter.")
else:
    # Tambahkan userId dummy karena model membutuhkan
    film_df["userId"] = 1
    pred_data = film_df[["userId", "movieId", "year", "genre"]]

    try:
        predicted_proba = model.predict_proba(pred_data)
        film_df["Higher_Rating"] = predicted_proba[:, 0]

        # Filter berdasarkan probabilitas
        filtered = film_df[film_df["Higher_Rating"] >= min_rating_input]

        if filtered.empty:
            st.warning("Tidak ada film memenuhi kriteria probabilitas tinggi.")
        else:
            top_n = filtered.sort_values("Higher_Rating", ascending=False).head(n_recs)
            top_n["genre"] = top_n["genre"].map(genre_mapping)  # Ubah ID genre ke nama
            st.subheader("🎯 Rekomendasi Film:")
            st.table(top_n[["title", "genre", "year", "Higher_Rating"]])

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")



# Jalankan Streamlit secara paralel
def run_streamlit():
    os.system("streamlit run stream-rekomendasi-film.py --server.port 8501")

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

# Tunggu Streamlit siap
time.sleep(5)

# Buat tunnel ngrok
public_url = ngrok.connect(addr=8501)
print(f"✅ Streamlit app is live at: {public_url}")