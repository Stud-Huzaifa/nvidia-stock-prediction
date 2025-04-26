import pickle
import numpy as np
from flask import Flask, request, render_template

# Load the trained model
model = pickle.load(open('F:/Data Science/Machine Learning/Nvidia_Stock_Prediction_Project/model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from form
    Open = float(request.form['Open'])
    High = float(request.form['High'])
    Low = float(request.form['Low'])
    Volume = float(request.form['Volume'])
    Year = int(request.form['Year'])
    Month = int(request.form['Month'])
    Day = int(request.form['Day'])
    Adj_Close = float(request.form['Adj_Close'])
    
    # Prepare features
    features = np.array([[Open, High, Low, Volume, Year, Month, Day, Adj_Close]])
    
    # Predict
    prediction = model.predict(features)

    return render_template('index.html', prediction_text='The predicted stock price is ${:.2f}'.format(prediction[0]))

# Main program
if __name__ == '__main__':
    app.run(debug=True)
