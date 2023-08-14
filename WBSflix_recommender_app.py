import pandas as pd
import matplotlib as plt
# import seaborn as sns

from surprise import Reader, Dataset, KNNBasic, SVDpp, BaselineOnly, accuracy
from surprise.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

#### getting and setting datasets

path_ratings = 'Jupyter_notebooks/8_Recommender/ratings.csv'
ratings_df = pd.read_csv(path_ratings)

path_links = 'Jupyter_notebooks/8_Recommender/links.csv'
links_df = pd.read_csv(path_links)

path_movies = 'Jupyter_notebooks/8_Recommender/movies.csv'
movies_df = pd.read_csv(path_movies)

path_tags = 'Jupyter_notebooks/8_Recommender/tags.csv'
tags_df = pd.read_csv(path_tags)

movie_genres = ["No",
                "Action",
                "Adventure",
                "Animation",
                "Children's",
                "Comedy",
                "Crime",
                "Documentary",
                "Drama",
                "Fantasy",
                "Film-Noir",
                "Horror",
                "Musical",
                "Mystery",
                "Romance",
                "Sci-Fi",
                "Thriller",
                "War",
                "Western"]


#### popular movies

rating_count_df = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
rating_count_df.sort_values(by=['count', 'mean'], ascending=False)
rating_count_df['popularity_rating'] = rating_count_df['mean'] + rating_count_df['count'] * 0.01
popularity_df_formula = rating_count_df.merge(movies_df[['movieId', 'title']], on='movieId', how='left')
popular_movies = popularity_df_formula.sort_values('popularity_rating', ascending=False)


#### item-based

def top_n_item_base(given_movie, n_recom):
    thrshld_both_movies = 5
    
    # given_movie_mask = movies_df["title"].str.contains(given_movie, case=False)
    
    given_movie_id = movies_df.loc[movies_df["title"] == given_movie, "movieId"].values[0]
        
    user_movie_matrix = pd.pivot_table(data=ratings_df,
                                       values='rating',
                                       index='userId',
                                       columns='movieId',
                                       fill_value=0)
    
    movies_cosines_matrix = pd.DataFrame(cosine_similarity(user_movie_matrix.T),
                                         columns=user_movie_matrix.columns,
                                         index=user_movie_matrix.columns)
    
    # Create a DataFrame using the values from 'movies_cosines_matrix' for the 'given_movie_id' movie
    given_movie_cosines_df = pd.DataFrame(movies_cosines_matrix[given_movie_id])
    
    # Rename the column to '{given_movie}_cosine'
    given_movie_cosines_df = given_movie_cosines_df.rename(columns={given_movie_id: f'{given_movie}_cosine'})
    
    # Remove the row with the index 'given_movie_id'
    given_movie_cosines_df = given_movie_cosines_df[given_movie_cosines_df.index != given_movie_id]
    
    # Sort the 'given_movie_cosines_df' by the column '{given_movie}_cosine' column in descending order
    given_movie_cosines_df = given_movie_cosines_df.sort_values(by=f'{given_movie}_cosine', ascending=False)
    
    # Find out the number of users rated both given movie and the other movie
    num_of_users_rated_both_movies = [sum((user_movie_matrix[given_movie_id] > 0) & (user_movie_matrix[movie_id] > 0)) for movie_id in given_movie_cosines_df.index]
    
    # Create a column for the number of users who rated given_movie and the other movie
    given_movie_cosines_df['users_who_rated_both_movies'] = num_of_users_rated_both_movies
    
    # Remove recommendations that have less than 'thrshld_both_movies' users who rated both books
    given_movie_cosines_df = given_movie_cosines_df[given_movie_cosines_df["users_who_rated_both_movies"] > thrshld_both_movies]
    
    given_movie_top_n_cosine = (given_movie_cosines_df
                                 .head(n_recom)
                                 .merge(movies_df,
                                        on='movieId',
                                        how='left')
                                 [['movieId', 
                                   'title', 
                                   f'{given_movie}_cosine', 
                                   'users_who_rated_both_movies']])
    
    return given_movie_top_n_cosine


#### user-based

def get_top_n(user_id, n):
    data = ratings_df[['userId', 'movieId', 'rating']]
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data, reader)

    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()
    
    bsl_options = {'method': 'sgd', 'reg': 0.02, 'learning_rate': 0.01, 'n_epochs': 20}
    algo_bsl = BaselineOnly(bsl_options=bsl_options)
    algo_bsl.fit(trainset)
    predictions_bsl = algo_bsl.test(testset)  

    user_recommendations = []

    # Iterate through each prediction tuple
    for uid, iid, true_r, est, _ in predictions_bsl:
        # Check if the user ID matches the target user
        if user_id == uid:
            # Append item_id and estimated_rating to the user_recommendations list
            user_recommendations.append((iid, est))
        else:
            # Skip to the next prediction if user ID doesn't match
            continue

    # Sort the user_recommendations list based on estimated_rating in descending order
    ordered_recommendations = sorted(user_recommendations, key=lambda x: x[1], reverse=True)

    # Get the top n predictions from the ordered_recommendations
    ordered_recommendations_top_n = ordered_recommendations[:n]
    ordered_recommendations_top_n_df = pd.DataFrame(ordered_recommendations_top_n, 
                                                  columns=["movieId", "estimated_rating"])
    ordered_recommendations_top_n_full_df = ordered_recommendations_top_n_df.merge(movies_df, 
                                                                                 on='movieId', 
                                                                                 how='left')  
    return ordered_recommendations_top_n_full_df


#### UI
st.title('Welcome to WBSFlix Recommender')

st.sidebar.header("User Inputs")

selected_movie_title = st.sidebar.selectbox("Select a movie:", movies_df['title'].unique())
selected_genre = st.sidebar.selectbox("Select a genre: [under construction]", movie_genres)
selected_user_id = st.sidebar.selectbox("Select a user:", ratings_df['userId'].unique())
n = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=30, value=5)

if st.sidebar.button("Get Recommendations"):
    recommendations_item = top_n_item_base(selected_movie_title, n)
    if selected_genre != 'No':
        recommendations_popular = popular_movies.loc[movies_df['genres'].str.contains(selected_genre, case=False)].head(n)
    else:
        recommendations_popular = popular_movies.head(n)
    # recommendations_user = get_top_n(selected_user_id, n)
    st.write("Popular movies")
    st.dataframe(recommendations_popular, height=300)
    # st.dataframe(recommendations_popular[['title']], height=300)
    st.write(f'Movies similar to {selected_movie_title}')
    st.dataframe(recommendations_item[['title']], height=300)
    st.write(f'Movies for user with ID {selected_user_id}')
    # st.dataframe(recommendations_user, height=300)
    
    
