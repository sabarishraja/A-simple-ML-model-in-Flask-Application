from flask import Flask, render_template, request
import joblib 
import numpy as np


#Initializing Flask app
#This creates an instance of the Flask app
app = Flask(__name__)

#Load the model
model = joblib.load('iris_model.pkl')
scaler = joblib.load('scaler.pkl')

#Defining routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    #Get the user input
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    #Scale the input
    input_data = np.array([sepal_length, sepal_width, petal_length, petal_width])
    scaled_input = scaler.transform(input_data)

    #Make prediction
    prediction = model.predict(scaled_input)[0]
    from website import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler
model = joblib.load('iris_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

#The following route handles POST requests which are sent when user submits a form
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Scale the input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    species = ['Setosa', 'Versicolor', 'Virginica'][prediction]

    return render_template('index.html', prediction=f"The predicted species is: {species}")

if __name__ == '__main__':
    app.run(debug=True)


