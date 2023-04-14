import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)

ratings_df = pd.read_csv('data/ml-25m/ratings.csv')
movies_df = pd.read_csv('data/ml-25m/movies.csv')

# consider only the last 24 * 2 months
num_round = 24
num_day_per_round = 60
second_per_round = (60 * 60 * 24 * num_day_per_round)
diff = (ratings_df['timestamp'].max() - ratings_df['timestamp'].min()) % second_per_round
ratings_df['round'] = (ratings_df['timestamp'] - ratings_df['timestamp'].min() - diff - 1) / second_per_round

ratings_df['roundId'] = np.floor(ratings_df['round']).astype(int)
max_round = ratings_df['roundId'].max()
new_ratings_df = ratings_df[ratings_df['round'] >= max_round - num_round]

# consider users that gives at least one rating every 30 days
user_counts_df = new_ratings_df.groupby(['userId', 'roundId']).size().reset_index(name='counts')
user_counts_counts_df = user_counts_df.groupby(['userId']).size().reset_index(name='counts')
max_count = user_counts_counts_df['counts'].max()
active_userId_list = list(user_counts_counts_df[user_counts_counts_df['counts'] >= max_count // 2]['userId'])  # max_count // 2
num_user = len(active_userId_list)
print(num_user)

final_ratings_df = new_ratings_df[new_ratings_df['userId'].isin(active_userId_list)]

# build reward matrix
genre_dict = {
    'Action': 0,
    'Adventure': 1,
    'Animation': 2,
    'Children': 3,
    'Comedy': 4,
    'Crime': 5,
    'Documentary': 6,
    'Drama': 7,
    'Fantasy': 8,
    'Film-Noir': 9,
    'Horror': 10,
    'Musical': 11,
    'Mystery': 12,
    'Romance': 13,
    'Sci-Fi': 14,
    'Thriller': 15,
    'War': 16,
    'Western': 17,
    'IMAX': 18,
    '(no genres listed)': 19
}
genre_index_map = lambda genre: genre_dict[genre]

all_Ys = []
round_id = 0
min_round_id = final_ratings_df['roundId'].min()
max_round_id = final_ratings_df['roundId'].max()
num_round = max_round_id - min_round_id + 1
for round_id in range(num_round):
    current_round_id = min_round_id + round_id
    print(round_id, current_round_id)
    this_round_df = final_ratings_df[final_ratings_df['roundId'] == current_round_id]

    rating_mat = np.zeros([num_user, len(genre_dict)])
    user_count = 0
    for user_id in active_userId_list:
        merge_df = pd.merge(this_round_df[this_round_df['userId'] == user_id][['userId', 'movieId', 'rating']], movies_df)
        movie_count = np.zeros([len(genre_dict)])
        for index, row in merge_df.iterrows():
            genre_index = list(map(genre_index_map, row['genres'].split('|')))
            rating_mat[user_count, genre_index] += row['rating']
            movie_count[genre_index] += 1.0
        rating_mat[user_count] = rating_mat[user_count] / movie_count.clip(min=1.0)
        user_count += 1
    rating_mat = rating_mat[:, :19]  # remove no genres
    all_Ys.append(rating_mat)

np.save('data/ml-25m/ns_data_Y_{}_{}'.format(num_round, num_day_per_round), all_Ys)
