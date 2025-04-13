# predict_music.py

import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('../model/music_recommender_model.joblib')
gender_encoder = joblib.load('../model/gender_encoder.joblib')
mood_encoder = joblib.load('../model/mood_encoder.joblib')
genre_encoder = joblib.load('../model/genre_encoder.joblib')

# User input
age = int(input("Enter your age: "))
gender = input("Enter your gender (e.g., Male/Female): ").capitalize()
mood = input("Enter your mood (e.g., Happy/Sad/Energetic/Relaxed): ").capitalize()

# Encode inputs
gender_encoded = gender_encoder.transform([gender])[0]
mood_encoded = mood_encoder.transform([mood])[0]

# Predict
input_df = pd.DataFrame([[age, gender_encoded, mood_encoded]], columns=['Age', 'Gender', 'Mood'])
predicted_genre_encoded = model.predict(input_df)[0]
predicted_genre = genre_encoder.inverse_transform([predicted_genre_encoded])[0]

print(f"Recommended Genre: {predicted_genre}")
