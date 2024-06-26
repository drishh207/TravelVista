from flask import Flask, request, jsonify
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from concurrent.futures import ProcessPoolExecutor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Helper function to read a list from a pickle file
def read_list_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        my_list = pickle.load(file)
    return my_list

# Function to preprocess reviews by removing special characters and converting to lowercase
def remove_special_characters(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation and symbols
    return text

good_words_counter = read_list_from_pickle("good_words.pkl")
good_words = list(good_words_counter.keys())
bad_words_counter = read_list_from_pickle("bad_words.pkl")
bad_words = list(bad_words_counter.keys())

# Function to rank places
def rank_places(input_data, good_words=good_words, bad_words=bad_words):
    input_city = input_data['city']

    df = pd.read_csv('places_final_preprocess.csv')
    result = df[df['City'] == input_city]

    if not result.empty:
        filename = result.iloc[0]['places_final']
        with open(filename, 'r') as file:
            json_data = json.load(file)
    else:
        print("No entry found")
        json_data = input_data

    places_with_reviews = json_data.get('places_with_reviews', [])

    all_words = list(set(good_words + bad_words))
    max_length = max(len(good_words), len(bad_words))
    padding_word = "padding_word"

    if len(good_words) < max_length:
        good_words += [padding_word] * (max_length - len(good_words))

    if len(bad_words) < max_length:
        bad_words += [padding_word] * (max_length - len(bad_words))

    vectorizer = CountVectorizer(vocabulary=all_words, tokenizer=lambda text: text.split())

    ranked_places_with_original = []

    # Lists to store overall scores and place names
    overall_scores = []
    place_names = []
    # List to store ranked places with original JSON objects
    ranked_places_with_original = []
    # Loop through each place with reviews
    for place in places_with_reviews:
        place_name = place['name']
        #print("Place:", place_name)

        # Extract reviews and preprocess them
        reviews = place['preprocessed_reviews']
        preprocessed_reviews = []
        problematic_reviews = []  # List to store problematic reviews
        for review in reviews:
            try:
                preprocessed_review = remove_special_characters(review.lower())  # Fix here
                preprocessed_review = " ".join(preprocessed_review.split())
                preprocessed_reviews.append(preprocessed_review)
            except Exception as e:
                problematic_reviews.append(review)
                print("Error processing review:", review)
                print("Error message:", str(e))

        # If there are no reviews, assign a rating of 2.5
        if not preprocessed_reviews:
            avg_rating = 2.5
        else:
            # Transform reviews into vectors
            review_vectors = vectorizer.fit_transform(preprocessed_reviews)

            # Calculate cosine similarity between reviews and good/bad words
            good_similarity = cosine_similarity(review_vectors, vectorizer.transform(good_words).toarray())
            bad_similarity = cosine_similarity(review_vectors, vectorizer.transform(bad_words).toarray())

            # Calculate average rating
            avg_rating = np.mean(place['rating'])

        # Calculate overall score
        overall_score = np.mean(good_similarity - bad_similarity) + avg_rating

        place_copy = place.copy()  # Make a copy to avoid modifying the original object
        place_copy['overall_score'] = overall_score
        ranked_places_with_original.append(place_copy)


    ranked_places_with_original = sorted(ranked_places_with_original, key=lambda x: x['overall_score'], reverse=True)

    ranked_json_object = {"places_with_reviews": ranked_places_with_original}

    return ranked_json_object

@app.route('/place_rank', methods=['POST'])
def rank_places_route():
    input_data = request.json
    with ProcessPoolExecutor() as executor:
        result = executor.submit(rank_places, input_data).result()
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(port=9003, host='0.0.0.0', debug=True)
