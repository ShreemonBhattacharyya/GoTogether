from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- 1. DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('traffic.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, start_point TEXT, end_point TEXT, 
                  prediction_time TEXT, congestion_level TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

# --- 2. SIMPLE AI MODEL TRAINING ---
# We'll create a small synthetic dataset to "train" the model on startup
def train_model():
    # Features: Hour (0-23), DayOfWeek (0-6), DistanceFactor (1-3)
    # Target: Congestion Percentage (0-100)
    data = []
    for _ in range(1000):
        hour = np.random.randint(0, 24)
        day = np.random.randint(0, 7)
        dist = np.random.uniform(1, 3)
        # Simulate rush hours: 8-10 AM and 5-8 PM
        base = 20
        if 8 <= hour <= 10 or 17 <= hour <= 20:
            base = 70
        congestion = base + (dist * 5) + np.random.normal(0, 5)
        data.append([hour, day, dist, min(100, max(0, congestion))])
    
    df = pd.DataFrame(data, columns=['hour', 'day', 'dist', 'congestion'])
    model = RandomForestRegressor(n_estimators=10)
    model.fit(df[['hour', 'day', 'dist']], df['congestion'])
    return model

traffic_model = train_model()

# --- 3. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    start = data.get('start')
    destination = data.get('destination')
    time_str = data.get('time') # Format HH:MM
    
    # Process inputs for AI
    hour = int(time_str.split(':')[0])
    day_of_week = datetime.now().weekday()
    
    # Distance heuristic based on your "allPlaces" logic
    dist_factor = 1.5
    if "esplanade" in start or "esplanade" in destination: dist_factor = 2.5
    
    # AI Prediction
    prediction = traffic_model.predict([[hour, day_of_week, dist_factor]])[0]
    
    # Determine Level
    level = "High" if prediction > 65 else "Medium" if prediction > 35 else "Low"
    
    # Save to Database
    conn = sqlite3.connect('traffic.db')
    c = conn.cursor()
    c.execute("INSERT INTO history (start_point, end_point, prediction_time, congestion_level, timestamp) VALUES (?, ?, ?, ?, ?)",
              (start, destination, time_str, level, datetime.now()))
    conn.commit()
    conn.close()

    return jsonify({
        "congestion_percent": round(prediction, 2),
        "level": level,
        "travel_time_min": round(15 * dist_factor * (prediction/50 + 0.5)),
        "signals": int(5 + (prediction/10)),
        "vehicles": int(20 + prediction)
    })

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)