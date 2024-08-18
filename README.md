
# TravelVista: Your Personalized Travel Planner for Literary Escapades

**TravelVista** is a novel application designed to simplify the travel planning process and provide users with tailored recommendations for an optimized travel experience.

## Overview

Travel planning is an intricate process that often involves extensive research, decision-making, and organization. With the growing availability of travel information online, users are inundated with options, making it challenging to streamline their preferences into a coherent itinerary. TravelVista addresses this issue by offering personalized recommendations and a hassle-free way to discover, plan, and organize trips.

## Key Features

- **Popular Travel Destinations**: Based on the user’s query, the system provides the best recommendations for travel destinations.
  
- **User Engagement Tracking**: The system tracks user engagement with recommendations to gain insights into user preferences, enhancing future recommendations using Collaborative Filtering.
  
- **Personalized Itinerary**: Tailored itineraries based on the user’s chosen destination, dates of visit, and budget, including:
  - **Hotels/Resorts**
  - **Flight/Train/Cab/Bus** options from home to destination
  - **Tourist Attractions**
  - **Cuisine Options**
  - **Transportation** within the city
  
- **Feedback System**: The system manually asks for user feedback on recommendations, updating preferences for future reference.

## Implementation Details

1. **City Recommendation**:
   We have manually curated the dataset for city recommendation, using authentic sources like Google Real time Map API and chatgpt, supervised manually by the team. The dataset consists of Destination/City Name, Region, State, User Reviews and a general description (content) of the place.
   For all the reviews, content are concatenated, we are using *FAISS Index* to store the embeddings of each entry and then using *BERT based Langchain Model* for mapping the user query to each embedding. We are using BERT and Langchain since we have to find the best matching places by understanding the users' intent and sentiment and then ranking as well *(Matching + Ranking)*. Therefore, Langchain helps to provide this facility more accurately than just using LLMs like BERT or DistilBERT.

2. **Hotel and Places to Visit Ranking**:
  For ranking the hotels and places to visit within the selected city, we have used a custom ranking algorithm. For hotels and places having rating >= 3.7 are considered as good otherwise bad. Good and Bad words are taken out from the reviews accordingly. Each review is matched with the Good and bad word list and a cosine similarity score is calculated and the results are ranked accordingly.

3. **Itinerary Generation**:
   Using Map API and LLMs like Gemini, we have carefully curated two suitable itineraries for the user.

4. **Engagement Score Calculator**:
   We employ a mechanism where the system gets insights from the user engagements on the webiste, for eg, time he/she spends on a particular destination - taken from cursor hovering time, viewing more images of 1 category, or patterns of dissatisfaction as a feedback for the system. This helps us to understand the user requirements better. 

**Implementation Diagram**
![TravelVista Diagram.png](https://github.com/drishh207/TravelVista/blob/main/TravelVista%20Diagram.png)

As we advance in the website, we will apply *Collaborative Filtering* to improve the personalization and provide better recommendations to first time users. 

## Usage Details
1. Clone the repository.
2. In the webiste/js folder, in each of the folders - city, hotel, itinerary, places, run the .py files (api files for each of the pages)
3. Open index.html and you can use the website.
4. Demo of the website: https://github.com/drishh207/TravelVista/blob/main/TravelVista%20Presentation.mp4

P.S. We are still working on the Front End of the website. Right now, It is a very simple HTML Page. 

## Summary

TravelVista revolutionizes the travel planning experience by offering a streamlined, personalized approach to discovering and organizing trips. By leveraging advanced technology and user-centric design, TravelVista aims to make travel planning more efficient, enjoyable, and tailored to individual needs.
