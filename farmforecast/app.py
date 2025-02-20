from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the CSV as a DataFrame
file_path = r"C:\Users\ADMIN\OneDrive\Desktop\predicted_crop_prices_evaluation.csv"  # Update this path if needed
df = pd.read_csv(file_path)

# Endpoint to get unique dropdown options for React frontend
@app.route('/options', methods=['GET'])
def get_options():
    # Get unique values for each dropdown
    places = df['AmcName'].dropna().unique().tolist()
    yards = df['YardName'].dropna().unique().tolist()
    crops = df['CommName'].dropna().unique().tolist()
    varities = df['VarityName'].dropna().unique().tolist()
    
    return jsonify({
        'places': places,
        'yards': yards,
        'crops': crops,
        'varities': varities
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    data = request.get_json()
    state = data.get('State', None)
    yard = data['YardName']
    market = data['AmcName']
    crop = data['CommName']
    varity = data['VarityName']
    year = data['Year']
    month = data['Month']
    day = data['Day']
    
    # Filter the DataFrame to find the matching row
    filtered_df = df[
        (df['AmcName'] == market) &
        (df['YardName'] == yard) &
        (df['CommName'] == crop) &
        (df['VarityName'] == varity) &
        (df['Year'] == year) &
        (df['Month'] == month) &
        (df['Day'] == day)
    ]

    # Check if a matching row is found
    if not filtered_df.empty:
        result = filtered_df.iloc[0]  # Get the first matching row
        return jsonify({
            'MinPrice': float(result['MinPrice']),
            'MaxPrice': float(result['MaxPrice']),
            'AvgPrice': float(result['AvgPrice'])
        })
    else:
        return jsonify({'error': 'No data found for the selected inputs'}), 404

if __name__ == '__main__':
    app.run(debug=True)
