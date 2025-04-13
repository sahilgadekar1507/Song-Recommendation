# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Load dataset
data = pd.read_csv('../data/music_data.csv')  # Path to your new CSV

# Encode Gender and Mood
gender_encoder = LabelEncoder()
mood_encoder = LabelEncoder()
genre_encoder = LabelEncoder()  # Genre instead of Recommended_Song

data['Gender'] = gender_encoder.fit_transform(data['Gender'])
data['Mood'] = mood_encoder.fit_transform(data['Mood'])
data['Genre'] = genre_encoder.fit_transform(data['Genre'])

# Features and Target
X = data[['Age', 'Gender', 'Mood']]
y = data['Genre']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Create model directory if not exist
os.makedirs('../model', exist_ok=True)

# Save model and encoders
joblib.dump(model, '../model/music_recommender_model.joblib')
joblib.dump(gender_encoder, '../model/gender_encoder.joblib')
joblib.dump(mood_encoder, '../model/mood_encoder.joblib')
joblib.dump(genre_encoder, '../model/genre_encoder.joblib')

print("Training completed and model saved.")
