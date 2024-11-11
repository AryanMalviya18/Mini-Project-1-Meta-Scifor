from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model from the file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Create a simple index.html for the form

# Define the route to handle the form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    features = [float(x) for x in request.form.values()]  # Convert inputs to float
    features_array = np.array([features])  # Create an array for prediction
    
    # Make prediction
    prediction = model.predict(features_array)
    
    # Return the result
    return f'Predicted Happiness Score: {prediction[0]:.2f}'

if __name__ == '__main__':
    app.run(debug=True)