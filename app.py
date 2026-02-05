from flask import Flask, request, render_template, send_from_directory
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Load trained model
model = joblib.load("salary_model.pkl")

# Load dataset (for scatter plot)
df = pd.read_csv("Salary_data.csv")
df = df[df["YearsExperience"] >= 1]

# Initialize Flask app
app = Flask(__name__)

# Ensure folder for saving plots
if not os.path.exists("static"):
    os.makedirs("static")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input
        if request.method == "POST":
                years = float(request.form["experience"])
                prediction = model.predict(np.array([[years]]))[0]
        # --- Generate plot ---
        plt.figure(figsize=(6,4))
        # Scatter: actual data
        plt.scatter(df["YearsExperience"], df["Salary"], color='purple', label="Actual Data")  
        # Regression line
        X_sorted = np.sort(df["YearsExperience"].values).reshape(-1,1)
        y_sorted = model.predict(X_sorted)
        plt.plot(X_sorted, y_sorted, color='red', label="Regression Line")
        # Highlight predicted point
        plt.scatter(years, prediction, color='green', s=100, marker="X", label="Your Prediction")
        plt.title("Salary Prediction")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.legend()
        # Save image to static folder
        plot_path = "static/plot.png"
        plt.savefig(plot_path)
        plt.close()

        return render_template("index.html",
                               prediction_text=f"Predicted Salary: ₹{prediction:,.2f}",
                               plot_url=plot_path)
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=False)
