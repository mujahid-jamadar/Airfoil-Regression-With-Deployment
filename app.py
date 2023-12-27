from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model from the pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_values = [
        float(request.form['frequency']),
        float(request.form['angle_of_attack']),
        float(request.form['chord_length']),
        float(request.form['free_stream_velocity']),
        float(request.form['suction_side'])
    ]

    # Make a prediction using the loaded model
    prediction = model.predict(np.array(input_values).reshape(1, -1))

    # Render the result page with the prediction
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
