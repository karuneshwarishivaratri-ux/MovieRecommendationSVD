
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

st.set_page_config(
    page_title="CineMatrix | SVD Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

DATA_DIR = Path(__file__).parent / "ml-100k"

@st.cache_data
def load_data():
    ratings = pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    
    genre_names = [
        "unknown", "Action", "Adventure", "Animation",
        "Children", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]

    movie_data = pd.read_csv(
        DATA_DIR / "u.item",
        sep="|",
        encoding="latin-1",
        header=None
    )

    movies = movie_data[[0, 1]].copy()
    movies.columns = ["movie_id", "title"]

    for index, genre in enumerate(genre_names, start=5):
        movies[genre] = movie_data[index].values

    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["movie_id"] = ratings["movie_id"].astype(str)
    movies["movie_id"] = movies["movie_id"].astype(str)
    return ratings, movies

@st.cache_resource
def train_model():
    ratings, _ = load_data()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings[["user_id", "movie_id", "rating"]], reader
    )
    trainset, testset = train_test_split(
        data, test_size=0.20, random_state=42
    )
    model = SVD(n_factors=50, n_epochs=20, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    mean_rating = np.mean(
        [rating for _, _, rating in trainset.all_ratings()]
    )
    errors = np.array([true - mean_rating for _, _, true in testset])
    baseline_rmse = float(np.sqrt(np.mean(errors ** 2)))
    baseline_mae = float(np.mean(np.abs(errors)))

    return model, trainset.n_ratings, len(testset), rmse, mae, baseline_rmse, baseline_mae

try:
    ratings, movies = load_data()
    model, train_count, test_count, rmse, mae, base_rmse, base_mae = train_model()
except Exception as error:
    st.error(f"Could not load data or train model: {error}")
    st.stop()

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #151b30);
    color: #f5f7ff;
}
h1, h2, h3 { color: #f5f7ff; }
[data-testid="stMetric"] {
    background: #202942;
    border: 1px solid #394765;
    padding: 18px;
    border-radius: 14px;
}
.movie-card {
    background: #202942;
    border: 1px solid #394765;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 CineMatrix")
st.subheader("Personalized Movie Recommendations using SVD")
st.caption("Computational Mathematics Laboratory · Collaborative Filtering")

st.latex(r"\hat r_{ui} = \mu + b_u + b_i + q_i^T p_u")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Ratings", f"{len(ratings):,}")
c2.metric("Users", f"{ratings.user_id.nunique():,}")
c3.metric("Movies", f"{movies.movie_id.nunique():,}")
c4.metric("SVD RMSE", f"{rmse:.4f}")

st.divider()

left, right = st.columns([1, 2])

with left:
    st.subheader("Find Recommendations")
    user_ids = sorted(ratings.user_id.unique(), key=int)
    selected_user = st.selectbox(
        "Select MovieLens User ID", user_ids, index=user_ids.index("196")
    )
    top_n = st.radio("Number of recommendations", [5, 10], horizontal=True)
    
    genre_options = [
        "All Genres", "Action", "Adventure", "Animation",
        "Children", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]

    selected_genre = st.selectbox(
        "Filter by Genre",
        genre_options
    )

    st.info("Recommendations exclude movies already rated by this user.")

    user_rating_count = int(
        (ratings.user_id == selected_user).sum()
    )
    st.metric("Ratings submitted by user", user_rating_count)

rated_movies = set(
    ratings.loc[ratings.user_id == selected_user, "movie_id"]
)

candidates = movies[~movies.movie_id.isin(rated_movies)].copy()

if selected_genre != "All Genres":
    candidates = candidates[candidates[selected_genre] == 1]

candidates["Predicted Rating"] = candidates.movie_id.apply(
    lambda movie_id: model.predict(selected_user, movie_id).est
)
recommendations = candidates.sort_values(
    "Predicted Rating", ascending=False
).head(top_n)
st.subheader("Top-N Movie Recommendations")

if recommendations.empty:
    st.warning("No recommendations found. Try selecting All Genres.")
else:
    for _, row in recommendations.iterrows():
        st.markdown(
            f"### 🎬 {row['title']}"
        )
        st.write(
            f"Predicted Rating: ⭐ {row['Predicted Rating']:.2f} / 5"
        )
        st.divider()

with right:
    st.subheader(f"Top {top_n} picks for User {selected_user}")

    for rank, (_, row) in enumerate(recommendations.iterrows(), 1):
        movie_genres = [
            genre for genre in [
                "unknown", "Action", "Adventure", "Animation",
                "Children", "Comedy", "Crime", "Documentary",
                "Drama", "Fantasy", "Film-Noir", "Horror",
                "Musical", "Mystery", "Romance", "Sci-Fi",
                "Thriller", "War", "Western"
            ]
            if row.get(genre, 0) == 1
        ]

        genre_text = ", ".join(movie_genres) or "Unknown"

        st.markdown(
            f"""
            <div class="movie-card">
                <b>{rank}. {row['title']}</b><br>
                Genre: {genre_text}<br>
                Predicted rating: ⭐ {row['Predicted Rating']:.2f} / 5
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------- MOVIE RATING PREDICTOR ----------
st.divider()
st.header("🎯 Predict a Movie Rating")
st.write("Estimate how a selected user might rate a movie.")

movie_titles = sorted(movies["title"].dropna().unique().tolist())

selected_movie_title = st.selectbox(
    "Choose a movie",
    movie_titles,
    key="rating_predictor_movie"
)

if st.button("Predict My Rating"):
    movie_row = movies[movies["title"] == selected_movie_title].iloc[0]

    predicted_rating = model.predict(
        selected_user,
        movie_row["movie_id"]
    ).est

    predicted_rating = max(1.0, min(5.0, predicted_rating))

    st.metric(
        "Predicted Rating",
        f"{predicted_rating:.2f} / 5 ⭐"
    )

    st.caption(
        "This is an estimated rating, not a guaranteed user rating."
    )

st.divider()
st.header("Model Evaluation")

m1, m2 = st.columns(2)
m1.metric("SVD MAE", f"{mae:.4f}")
m2.metric("Baseline MAE", f"{base_mae:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
labels = ["Mean Baseline", "SVD"]
x = np.arange(2)
width = 0.35

ax.bar(x - width / 2, [base_rmse, rmse], width, label="RMSE")
ax.bar(x + width / 2, [base_mae, mae], width, label="MAE")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Prediction Error")
ax.set_title("Prediction Error: Lower is Better")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("Dataset and Mathematical Method"):
    st.write(f"Training ratings: {train_count:,}")
    st.write(f"Testing ratings: {test_count:,}")
    st.write("SVD learns latent user and movie factors from observed ratings.")
    st.write("Unknown ratings are predicted, then candidate movies are ranked.")
    st.write("Lower RMSE and MAE indicate better prediction accuracy.")
