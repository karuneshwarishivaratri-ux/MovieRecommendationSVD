
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# 1. Load MovieLens ratings
ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Convert IDs to strings for Surprise
ratings["user_id"] = ratings["user_id"].astype(str)
ratings["movie_id"] = ratings["movie_id"].astype(str)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    ratings[["user_id", "movie_id", "rating"]],
    reader
)

# 2. Split into training and testing data
trainset, testset = train_test_split(
    data, test_size=0.20, random_state=42
)

# 3. Train the SVD model
model = SVD(
    n_factors=50,
    n_epochs=20,
    random_state=42
)
model.fit(trainset)

# 4. Evaluate SVD
predictions = model.test(testset)
svd_rmse = accuracy.rmse(predictions, verbose=False)
svd_mae = accuracy.mae(predictions, verbose=False)

# 5. Mean-rating baseline
global_mean = np.mean([rating for _, _, rating in trainset.all_ratings()])

baseline_errors = [
    true_rating - global_mean
    for _, _, true_rating in testset
]

baseline_rmse = np.sqrt(np.mean(np.square(baseline_errors)))
baseline_mae = np.mean(np.abs(baseline_errors))

print("\n===== MODEL EVALUATION =====")
print(f"Training ratings: {trainset.n_ratings}")
print(f"Testing ratings: {len(testset)}")
print(f"Mean baseline RMSE: {baseline_rmse:.4f}")
print(f"Mean baseline MAE : {baseline_mae:.4f}")
print(f"SVD RMSE          : {svd_rmse:.4f}")
print(f"SVD MAE           : {svd_mae:.4f}")

# 6. Load movie titles
movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)
movies["movie_id"] = movies["movie_id"].astype(str)

# 7. Recommend movies for a selected user
def recommend_movies(user_id, top_n=10):
    user_id = str(user_id)

    if user_id not in ratings["user_id"].values:
        print("User ID not found.")
        return

    rated_movies = set(
        ratings.loc[ratings["user_id"] == user_id, "movie_id"]
    )

    candidates = movies[~movies["movie_id"].isin(rated_movies)].copy()

    candidates["predicted_rating"] = candidates["movie_id"].apply(
        lambda movie_id: model.predict(user_id, movie_id).est
    )

    recommendations = candidates.sort_values(
        "predicted_rating", ascending=False
    ).head(top_n)

    print(f"\n===== TOP {top_n} MOVIES FOR USER {user_id} =====")
    print(
        recommendations[["title", "predicted_rating"]]
        .to_string(
            index=False,
            formatters={"predicted_rating": "{:.2f}".format}
        )
    )

if __name__ == "__main__":
    recommend_movies(user_id=196, top_n=10)
