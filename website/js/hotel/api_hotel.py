from flask import Flask, request, jsonify
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

@app.route('/hotel_rank', methods=['POST', 'OPTIONS'])
def rank_places():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'success'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    input_city = request.json.get('city')
    df = pd.read_csv('hotel_final_preprocess.csv')
    result = df[df['City'] == input_city]

    if not result.empty:
        filename = result.iloc[0]['hotel_final']
        with open(filename, 'r') as file:
            json_data = json.load(file)
    else:
        json_data = request.get_json()

    places_with_reviews = json_data.get('places_with_reviews', [])

    good_words_counter = read_list_from_pickle("good_words.pkl")
    good_words = list(good_words_counter.keys())
    bad_words_counter = read_list_from_pickle("bad_words.pkl")
    bad_words = list(bad_words_counter.keys())

    max_length = max(len(good_words), len(bad_words))
    padding_word = "padding_word"

    if len(good_words) < max_length:
        good_words += [padding_word] * (max_length - len(good_words))

    if len(bad_words) < max_length:
        bad_words += [padding_word] * (max_length - len(bad_words))

    vectorizer = CountVectorizer()

    ranked_places_with_original = []
    for place in places_with_reviews:
        preprocessed_reviews = []
        for review_data in place['reviews']:
            review_text = review_data.get('text', '')
            try:
                preprocessed_review = remove_special_characters(review_text.lower())
                preprocessed_review = " ".join(preprocessed_review.split())
                preprocessed_reviews.append(preprocessed_review)
            except Exception as e:
                print("Error processing review:", review_text)
                print("Error message:", str(e))

        review_vectors = vectorizer.fit_transform(preprocessed_reviews)
        good_similarity = cosine_similarity(review_vectors, vectorizer.transform(good_words))
        bad_similarity = cosine_similarity(review_vectors, vectorizer.transform(bad_words))

        avg_rating = np.mean([rating_data for review_data in place['reviews'] for rating_data in review_data.get('rating', [])])
        overall_score = np.mean(good_similarity - bad_similarity) + avg_rating

        place_copy = place.copy()
        place_copy['overall_score'] = overall_score
        ranked_places_with_original.append(place_copy)

    ranked_places_with_original = sorted(ranked_places_with_original, key=lambda x: x['overall_score'], reverse=True)
    ranked_json_object = {"places_with_reviews": ranked_places_with_original}

    response = jsonify(ranked_json_object)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(port=9002, host='0.0.0.0', debug=True)

