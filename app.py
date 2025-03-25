from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Flask app setup
app = Flask(__name__)

# Serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")  # Make sure index.html is inside the "templates" folder

# API Endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    distance = np.array([[data["distance"]]])
    fare = model.predict(distance)[0]
    return jsonify({"fare": float(fare)})

if __name__ == "__main__":
    app.run(debug=True)
