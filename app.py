from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("pcos_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    bmi = float(request.form['bmi'])
    irregularity = int(request.form['irregularity'])
    testosterone = float(request.form['testosterone'])
    follicle = int(request.form['follicle'])
    age=int(request.form['age'])

    # Arrange into numpy array for prediction
    data = np.array([[bmi, irregularity, testosterone, follicle, age]])

    # Make prediction
    prediction = model.predict(data)[0]

    result = "PCOS Detected ✅" if prediction == 1 else "No PCOS ❌"
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
