import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import re
movies = pd.read_csv('/Users/varunsachan/Desktop/Projects/basic movie recommender/tmdb_5000_movies.csv')
credits = pd.read_csv('/Users/varunsachan/Desktop/Projects/basic movie recommender/tmdb_5000_credits.csv')
movies = movies.merge(credits,on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+= 1
        else:
            break
    return L
def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] =='Director':
            L.append(i['name'])
            break
    return L
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))
movies['cast'] = movies['cast'].apply(lambda x: ' '.join(x))
movies['crew'] = movies['crew'].apply(lambda x: ' '.join(x))


movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])


movies['tags'] = movies['overview'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['genres'] + ' ' + \
                 movies['cast'] + ' ' + \
                 movies['crew'] + ' ' + \
                 movies['keywords'].apply(lambda x: ' '.join(x))


new_df = movies[['movie_id', 'title', 'tags']]

new_df['tags'].apply(lambda x:"".join(x))
new_df['tags'].apply(lambda x:x.lower())
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i)) 
    return " ".join(y)

new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)

similarity = cosine_similarity(vectors)



def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def recommend(movie=None, genre=None):
    titles_processed = new_df['title'].apply(preprocess_text)  
    
    if movie and genre:
        movie_processed = preprocess_text(movie)  
        genre_processed = preprocess_text(genre)  
        
        filtered_movies = new_df[(titles_processed == movie_processed) & (new_df['tags'].str.contains(genre_processed))]
    elif movie:
        movie_processed = preprocess_text(movie) 
        
        # Finding similar movies based on cosine similarity
        movie_index = new_df[new_df['title'].apply(preprocess_text) == movie_processed].index
        if len(movie_index) == 0:
            print("Movie not found.")
            return
        movie_index = movie_index[0]
        similar_movies = list(enumerate(similarity[movie_index]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        similar_movies = similar_movies[1:11]  # Exclude the movie itself and take top 10 similar movies
        
        for i in range(len(similar_movies)):
            idx = similar_movies[i][0]
            print(new_df.iloc[idx]['title'])
            
        return
    elif genre:
        genre_processed = preprocess_text(genre)  
        
        filtered_movies = new_df[new_df['tags'].str.contains(genre_processed)]
    else:
        print('Please provide either a movie title or a genre.')
        return
    
    if not filtered_movies.empty:
        for index, row in filtered_movies.iterrows():
            print(row['title'])
    else:
        print('No movies found matching the provided criteria.')

recommend(input("Tell The Name Of The Movie "), input("Tell the Genre "))
