import pandas as pd
from surprise import dump
import json
from collections import defaultdict

##Preload this
model = dump.load('./SVD_V1')[1]
movie_list = set(json.loads(open('../Data/movielist.json', 'r').read()))
top_20_list = json.loads(open('../Data/top20_movie.json', 'r').read())
user_hist = defaultdict(list, json.loads(open('../Data/userhist.json', 'r').read()))
##
##When you implement this in code, call the function with:
#get_recommend_movie(model, movie_list, top_20_list, user_hist[str(user_id)], user_id)
def get_recommend_movie(model, movie_list, top_20_list, user_hist, user_id, n = 20):
    if user_hist == []:
        return top_20_list
    else:
        user_hist = set(user_hist)
        new_movies = movie_list-user_hist
        result = []
        for movie in new_movies:
            prediction = model.predict(user_id, movie)
            movie = prediction.iid
            rating = prediction.est
            result.append((movie,rating))
        result_df = pd.DataFrame(result, columns=['movie_id', 'predicted_rating']).sort_values(by='predicted_rating', ascending=False)[:n]
        return result_df.movie_id.values.tolist()
