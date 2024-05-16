import numpy as np
import pandas as pd

# Sample movie data
movies_data = {
    'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
    'genres': ['Drama', 'Crime, Drama', 'Action, Crime, Drama', 'Crime, Drama', 'Drama, Romance']
}

# Sample ratings data
ratings_data = {
    'userId': [1, 1, 2, 2, 3],
    'title': ['The Shawshank Redemption', 'The Godfather', 'The Shawshank Redemption', 'Pulp Fiction', 'Forrest Gump'],
    'rating': [5, 4, 4, 5, 4]
}

# Create DataFrame from sample data
movies = pd.DataFrame(movies_data)
ratings = pd.DataFrame(ratings_data)

# Merge movies and ratings data
movie_ratings = pd.merge(movies, ratings)

# Create a user-item matrix
user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# Fill missing values with 0
user_movie_matrix = user_movie_matrix.fillna(0)

# Calculate similarity between users (cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity

# Define a function to recommend movies to a user
def recommend_movies(user_id, n=5):
    # Exclude user's ratings when calculating similarity
    user_similarity = cosine_similarity(user_movie_matrix[user_movie_matrix.index != user_id])
    
    # Get similarity scores for the given user
    user_similarities = user_similarity[user_id-1]
    
    # Get indices of top similar users
    similar_users_indices = np.argsort(user_similarities)[::-1]  # No need to exclude the user itself
    
    # Get movies the user has not seen yet
    user_unseen_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] == 0]
    
    # Calculate scores for unseen movies
    movie_scores = user_unseen_movies.dot(user_similarity) / np.sum(user_similarity)
    
    # Convert movie_scores to pandas Series for sorting
    movie_scores_series = pd.Series(movie_scores, index=user_unseen_movies.index)
    
    # Get top recommended movies
    top_movies_indices = movie_scores_series.sort_values(ascending=False).index[:n]
    top_movies = movies[movies['title'].isin(top_movies_indices)]
    
    return top_movies

# Test the recommendation system for a user
user_id = 2
recommended_movies = recommend_movies(user_id)
print("Recommended movies for user", user_id)
print(recommended_movies[['title', 'genres']])
