import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Rekomendasi Film", layout="wide")

@st.cache_data
def load_data():
    """Memuat semua DataFrame yang sudah disimpan dan melakukan pre-processing."""
    movies_df = pd.read_csv('streamlit_movies_df.csv')
    ratings_df = pd.read_csv('streamlit_ratings_df.csv')
    user_latent_features_df = pd.read_csv('streamlit_user_latent_features_df.csv', index_col='userId')
    item_latent_features_df = pd.read_csv('streamlit_item_latent_features_df.csv', index_col='movieId')
    movie_tag_features_df = pd.read_csv('streamlit_movie_tag_features_df.csv')
    all_movie_features_df = pd.read_csv('streamlit_all_movie_features_df.csv')
    feature_columns_order = joblib.load('feature_columns_order.pkl')

    # --- Pre-processing for Genres ---
    # Identify genre columns (assuming they are binary 0/1 and named as genres)
    genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    # Create a 'genres_str' column
    def get_genres_str(row):
        genres = [genre for genre in genre_columns if genre in row and row[genre] == 1]
        return ", ".join(genres) if genres else "N/A"
    
    movies_df['genres_str'] = movies_df.apply(get_genres_str, axis=1)

    # --- Pre-processing for Tags ---
    # Identify tag columns from movie_tag_features_df (excluding 'movieId')
    tag_columns_from_df = [col for col in movie_tag_features_df.columns if col != 'movieId']

    # Create a 'tags_str' column in movie_tag_features_df
    def get_tags_str(row):
        tags = [tag for tag in tag_columns_from_df if tag in row and row[tag] == 1]
        return ", ".join(tags) if tags else "N/A"
    
    movie_tag_features_df['tags_str'] = movie_tag_features_df.apply(get_tags_str, axis=1)

    # Merge tags_str into movies_df
    movies_df = movies_df.merge(
        movie_tag_features_df[['movieId', 'tags_str']],
        on='movieId',
        how='left'
    )
    # Fill NaN tags_str with 'N/A' for movies that might not have tags
    movies_df['tags_str'] = movies_df['tags_str'].fillna('N/A')

    return movies_df, ratings_df, user_latent_features_df, item_latent_features_df, movie_tag_features_df, all_movie_features_df, feature_columns_order

@st.cache_resource
def load_model():
    """Memuat model classifier yang sudah dilatih."""
    best_classifier = joblib.load('best_classifier_model.pkl')
    return best_classifier

# Muat semua data dan model
movies_df, ratings_df, user_latent_features_df, item_latent_features_df, movie_tag_features_df, all_movie_features_df, feature_columns_order = load_data()
best_classifier = load_model()

# --- Fungsi Rekomendasi (Diperbarui untuk Cold Start dan Sorting) ---
def get_movie_recommendations_for_user(user_id, classifier_model, n=10, base_movies_df_filtered=None, sort_order='relevance'):
    """
    Menghasilkan rekomendasi film untuk pengguna tertentu, termasuk penanganan cold start dan sorting.
    Parameter base_movies_df_filtered: DataFrame film yang sudah difilter oleh UI.
    Parameter sort_order: 'relevance', 'newest', 'oldest'.
    """
    st.write(f"Mencari rekomendasi...")

    # Define the desired output columns for display
    display_cols = ['title', 'year', 'genres_str', 'tags_str']

    # --- Jalur untuk pengguna umum/cold start (prioritas: popularitas/rata-rata rating) ---
    if user_id is None or user_id not in user_latent_features_df.index:
        st.warning(f"Merekomendasikan film populer dari hasil filter.")
        
        if base_movies_df_filtered is None or base_movies_df_filtered.empty:
            st.info("Tidak ada film yang memenuhi filter.")
            return pd.DataFrame(columns=display_cols)

        movies_with_ratings = base_movies_df_filtered.merge(ratings_df, on='movieId', how='inner')
        
        if movies_with_ratings.empty:
            st.info("Tidak ada film yang memenuhi filter yang dipilih.")
            return pd.DataFrame(columns=display_cols)

        popular_filtered_movies = movies_with_ratings.groupby('movieId')['rating'].mean().reset_index()
        
        # Merge back with the full movie info to get title, year, genres_str, tags_str
        popular_movies_info = popular_filtered_movies.merge(
            movies_df[['movieId', 'title', 'year', 'genres_str', 'tags_str']],
            on='movieId',
            how='inner'
        )

        # Apply sorting for cold start
        if sort_order == 'relevance':
            # For cold start, 'relevance' means highest average rating
            recommendations = popular_movies_info.sort_values(by='rating', ascending=False).head(n)
        elif sort_order == 'newest':
            recommendations = popular_movies_info.sort_values(by='year', ascending=False).head(n)
        elif sort_order == 'oldest':
            recommendations = popular_movies_info.sort_values(by='year', ascending=True).head(n)
        else: # Default to relevance if unknown sort_order
            recommendations = popular_movies_info.sort_values(by='rating', ascending=False).head(n)
        
        return recommendations[display_cols].copy()

    # --- Jalur untuk pengguna spesifik (bukan cold start) ---
    watched_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    
    # Ensure source_movies_for_prediction includes the genres_str and tags_str columns
    source_movies_for_prediction = base_movies_df_filtered if base_movies_df_filtered is not None else movies_df # Use movies_df which already has 'genres_str' and 'tags_str'
    
    unwatched_movies_df = source_movies_for_prediction[~source_movies_for_prediction['movieId'].isin(watched_movie_ids)].copy()

    if unwatched_movies_df.empty:
        st.info(f"User ID {user_id} telah menonton semua film yang memenuhi filter atau tidak ada film yang tersisa untuk direkomendasikan.")
        return pd.DataFrame(columns=display_cols)

    user_lfs = user_latent_features_df.loc[user_id]
    X_predict = unwatched_movies_df[feature_columns_order].copy()

    for col_name, value in user_lfs.items():
        X_predict[col_name] = value
    X_predict = X_predict[feature_columns_order]

    predicted_proba = classifier_model.predict_proba(X_predict)[:, 1]
    unwatched_movies_df['predicted_proba'] = predicted_proba # Keep for sorting

    # Apply sorting for specific user recommendations
    if sort_order == 'relevance':
        top_n_recommendations = unwatched_movies_df.sort_values(by='predicted_proba', ascending=False).head(n)
    elif sort_order == 'newest':
        top_n_recommendations = unwatched_movies_df.sort_values(by='year', ascending=False).head(n)
    elif sort_order == 'oldest':
        top_n_recommendations = unwatched_movies_df.sort_values(by='year', ascending=True).head(n)
    else: # Default to relevance if unknown sort_order
        top_n_recommendations = unwatched_movies_df.sort_values(by='predicted_proba', ascending=False).head(n)

    # Select and return the desired columns
    return top_n_recommendations[display_cols].copy()

# --- Bagian UI Streamlit dengan Filter Baru ---
st.title("🎬 Sistem Rekomendasi Film")
st.markdown("---")

# Tab untuk Rekomendasi dan Performa Model
tab1, tab2 = st.tabs(["Rekomendasi Film", "Performa Model"])

with tab1:
    st.header("Temukan Film Favorit Anda!")

    # Input Filter
    with st.form("filter_form"):
        st.subheader("Filter Preferensi")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            genre_input = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            selected_genre = st.multiselect(
                "Pilih genre film yang Anda suka:",
                options=genre_input
            )
            
            if 'movieId' in movie_tag_features_df.columns:
                tag_input_display_options = [col for col in movie_tag_features_df.columns if col != 'movieId']
                if not tag_input_display_options:
                    st.warning("Tidak ada kolom tag yang ditemukan.")
            else:
                st.warning("Kolom 'movieId' tidak ditemukan.")
                tag_input_display_options = [] 

            tag_input = st.multiselect(
                "Pilih Tags:",
                options=tag_input_display_options,
                default=[]
            )
        with col2:
            if 'year' in movies_df.columns and pd.api.types.is_numeric_dtype(movies_df['year']):
                min_year = int(movies_df['year'].min())
                max_year = int(movies_df['year'].max())
            else:
                st.warning("Kolom 'year' tidak ditemukan. Menggunakan tahun default.")
                min_year = 1900
                max_year = 2023 

            release_year_input = st.slider(
                "Tahun Rilis:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year) 
            )
        with col3:
            num_recommendations = st.slider(
                "Jumlah Rekomendasi:",
                min_value=1,
                max_value=20,
                value=1, 
                step=1
            )
            
            # --- Tambah Radio Button untuk Sorting ---
            sort_by_option = st.radio(
                "Urutkan hasil berdasarkan:",
                ('Relevansi (Default)', 'Tahun Rilis (Terbaru)', 'Tahun Rilis (Terlama)'),
                index=0 # Default to 'Relevansi'
            )
            
            # Map radio button selection to internal sort_order string
            if sort_by_option == 'Relevansi (Default)':
                selected_sort_order = 'relevance'
            elif sort_by_option == 'Tahun Rilis (Terbaru)':
                selected_sort_order = 'newest'
            else: # 'Tahun Rilis (Terlama)'
                selected_sort_order = 'oldest'

        submitted = st.form_submit_button("Submit Pilihan")

    # Proses Rekomendasi
    if submitted:
        st.write(f"Genre yang dipilih: {selected_genre}") 
        st.write(f"Tags yang dipilih: {tag_input}") 
        st.write(f"Rentang tahun: {release_year_input}")
        st.write(f"Diurutkan berdasarkan: {sort_by_option}")
        st.success("Form berhasil disubmit!")
        with st.spinner("Mencari rekomendasi..."):
            current_filtered_df = movies_df.copy() # Start with movies_df containing genres_str and tags_str
            
            # Filter berdasarkan genre yang dipilih (logika OR)
            if selected_genre:
                genre_filter_condition = pd.Series([False] * len(current_filtered_df), index=current_filtered_df.index)
                for genre in selected_genre:
                    if genre in current_filtered_df.columns:
                        genre_filter_condition = genre_filter_condition | (current_filtered_df[genre] == 1)
                    else:
                        st.warning(f"Genre '{genre}' tidak ditemukan sebagai kolom di data film. Filter ini akan diabaikan.")
                current_filtered_df = current_filtered_df[genre_filter_condition]

            # Filter berdasarkan tags (logika OR)
            if tag_input:
                # Merge movie_tag_features_df temporarily for filtering.
                # Drop tags_str from movie_tag_features_df before merging to avoid duplicate columns if present.
                temp_tag_df = movie_tag_features_df.drop('tags_str', axis=1, errors='ignore')
                temp_df_for_tag_filter = current_filtered_df.merge(temp_tag_df, on='movieId', how='inner', suffixes=('', '_tag_oh'))
                
                tag_filter_condition = pd.Series([False] * len(temp_df_for_tag_filter), index=temp_df_for_tag_filter.index)
                for tag in tag_input:
                    if tag in temp_df_for_tag_filter.columns:
                        tag_filter_condition = tag_filter_condition | (temp_df_for_tag_filter[tag] == 1)
                    else:
                        st.warning(f"Tag '{tag}' tidak ditemukan sebagai kolom di data tag one-hot. Filter ini akan diabaikan.")
                
                # Apply tag filter and select original columns plus the new string columns (genres_str, tags_str)
                # Ensure we retain the 'genres_str' and 'tags_str' columns from current_filtered_df
                cols_to_keep = [col for col in movies_df.columns] # All original columns including movieId, title, year, genres_str, tags_str, and one-hot genres
                current_filtered_df = temp_df_for_tag_filter[tag_filter_condition][cols_to_keep].drop_duplicates(subset=['movieId']).copy()

            # Filter berdasarkan tahun rilis
            current_filtered_df = current_filtered_df[
                (current_filtered_df['year'] >= release_year_input[0]) &
                (current_filtered_df['year'] <= release_year_input[1])
            ]
            
            # Ensure only unique movieIds are passed for recommendation after filtering
            filtered_movies_for_recommendation = current_filtered_df.drop_duplicates(subset=['movieId']).copy()


            if filtered_movies_for_recommendation.empty:
                st.warning("Tidak ada film yang memenuhi filter yang dipilih.")
            else:
                st.subheader("Hasil Rekomendasi Film:")
                recommendations = get_movie_recommendations_for_user(
                    user_id=None,  # Tetap None untuk mode rekomendasi umum
                    classifier_model=best_classifier,
                    n=num_recommendations,
                    base_movies_df_filtered=filtered_movies_for_recommendation, # Meneruskan DataFrame yang sudah difilter
                    sort_order=selected_sort_order # Meneruskan opsi sorting
                )
                
                if recommendations.empty:
                    st.info("Tidak ada rekomendasi yang ditemukan setelah menerapkan semua filter dan kriteria sorting.")
                else:
                    st.dataframe(recommendations, use_container_width=True)

with tab2:
    st.header("Metrik Performa Model")
    st.write("Visualisasi ini menunjukkan seberapa baik model klasifikasi memprediksi apakah pengguna akan menyukai film.")
    st.info(
    "Aplikasi ini merekomendasikan film berdasarkan preferensi pengguna menggunakan model Hybrid (Collaborative Filtering "
    "dengan TruncatedSVD untuk fitur laten, dan Content-Based melalui fitur genre & tag) yang dilatih dengan RandomForestClassifier."
    "Model ini memprediksi kemungkinan pengguna menyukai suatu film."
    )

    metrics = {
        "Accuracy": 0.7555,
        "Precision": 0.7565,
        "Recall": 0.8854,
        "F1-Score": 0.8159,
        "ROC AUC Score": 0.8211
    }
    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Score"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Metric", y="Score", data=metrics_df, palette="viridis", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Performa Model Klasifikasi Rekomendasi")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metrik")
    for index, row in metrics_df.iterrows():
        ax.text(row.name, row.Score + 0.02, round(row.Score, 4), color='black', ha="center")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)