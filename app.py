from flask import Flask, request, jsonify, render_template

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

# Load the dataset and model
data = pd.read_csv('Dist Data 3.1 - Sheet1.csv')

# Data preprocessing
data.ffill(inplace=True)  # Use ffill() instead of fillna(method='ffill')

# Encode categorical variables
le_location = LabelEncoder()
le_crop = LabelEncoder()

data['Location'] = le_location.fit_transform(data['Location'])
data['cropname'] = le_crop.fit_transform(data['cropname'])

# Features and target variable
X = data[['Location', 'cropname']]
y = data['Yield']

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model and encoders
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(le_location, 'le_location.pkl')
joblib.dump(le_crop, 'le_crop.pkl')

# Load the model and encoders
model = joblib.load('random_forest_model.pkl')
le_location = joblib.load('le_location.pkl')
le_crop = joblib.load('le_crop.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/updates')
def updates():
    return render_template('updates.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    cropname = request.form.get('cropname')

    try:
        # Validate and encode location and cropname
        if location not in le_location.classes_:
            raise ValueError(f"Invalid location: {location}")
        if cropname not in le_crop.classes_:
            raise ValueError(f"Invalid cropname: {cropname}")

        location_encoded = le_location.transform([location])[0]
        cropname_encoded = le_crop.transform([cropname])[0]

        input_data = pd.DataFrame([[location_encoded, cropname_encoded]], columns=['Location', 'cropname'])
        predicted_yield = model.predict(input_data)[0]

        result = {'predicted_yield': predicted_yield}
    except Exception as e:
        result = {'error': str(e)}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
