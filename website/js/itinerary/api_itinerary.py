from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/get_itinerary', methods=['POST'])
def get_itinerary():
    try:
        data = request.json
        city = data.get('city')
        hotel_name = data.get('hotel')
        places_to_visit = data.get('places')

        # Extract only the names from the places_to_visit list
        if isinstance(places_to_visit, list):
            place_names = [place.get('name', place) if isinstance(place, dict) else place for place in places_to_visit]
        else:
            place_names = []

        # Join the place names into a comma-separated string
        places_to_visit_str = ', '.join(place_names)

        # Construct JSON payload for the API
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"""Create two separate itineraries, with 2 days each, for {city} and the hotel or accommodation selected is {hotel_name}. The places to visit in {city} are {places_to_visit_str}. Provide local transportation options and total budget for both itineraries. Give the result as two JSON objects, one for each itinerary. Give both the itineraries in the below format:
          	Day 1: Give attractions to visit, along with some info about the attractions of what can be done, with transportation required to reach them, along with some famous food locations
          	Day 2: Give attractions to visit, along with some info about the attractions of what can be done, with transportation required to reach them, along with some famous food locations
          	
          	Total Budget:
          	Hotel:
          	Food:
          	Local Transportation:
          	Estimated Budget: """
                }]
            }]
        }

        # Print the payload for debugging
        print("Payload being sent:", payload)

        # Make a POST request to the API endpoint
        response = requests.post(
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyDLRQP-7VqPcia7GDRoJsi_Pa5X7mIQf_M',
            json=payload
        )

        # Check if the request was successful
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({"error": "Failed to get itinerary"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=9004, host='0.0.0.0', debug=True)

