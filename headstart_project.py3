# coding: utf-8
# Your code here!

import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sys import exit


# loading the data from the csv file to apandas dataframe
movies_data = pd.read_csv('/home/aditya/Documents/books/headstart/movies.csv')

# selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']


# replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')


# combining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

# converting the text data to feature vectors

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)


# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)

end = 1

while(end != 0):
  # getting the movie name from the user
  movie_name = input(' Enter your favourite movie name : ')
  if movie_name == 0:
  	end = 0
  	sys.exit()
  # creating a list with all the movie names given in the dataset
  list_of_all_titles = movies_data['title'].tolist()
  # finding the close match for the movie name given by the user
  find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
  if len(find_close_match)== 0:
    if end == 0:
      print('have a good day')
      exit()
    else:
      print('given movie does not exits is the current database')
      continue 
  
  close_match = find_close_match[0]
  # finding the index of the movie with title
  index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
  # getting a list of similar movies
  similarity_score = list(enumerate(similarity[index_of_the_movie]))
  # print the name of similar movies based on the index
  sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
  print('Movies suggested for you : \n')
  i = 1
  for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if (i<30):
      print(i, '.',title_from_index)
      i+=1
