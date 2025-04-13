from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load models and encoders
model = joblib.load('../model/music_recommender_model.joblib')
gender_encoder = joblib.load('../model/gender_encoder.joblib')
mood_encoder = joblib.load('../model/mood_encoder.joblib')
genre_encoder = joblib.load('../model/genre_encoder.joblib')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender'].capitalize()  # Capitalize to match training
        mood = request.form['mood'].capitalize()

        # Encode inputs
        gender_encoded = gender_encoder.transform([gender])[0]
        mood_encoded = mood_encoder.transform([mood])[0]

        # Prepare input
        input_df = pd.DataFrame([[age, gender_encoded, mood_encoded]], columns=['Age', 'Gender', 'Mood'])

        # Predict
        predicted_genre_encoded = model.predict(input_df)[0]
        predicted_genre = genre_encoder.inverse_transform([predicted_genre_encoded])[0]

        prediction = predicted_genre

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
