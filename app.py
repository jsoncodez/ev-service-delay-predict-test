from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Load the trained models
model_classification = joblib.load('./models/rf_model_classification_delay_risk.pkl')  # delay prediction
model_regression = joblib.load('./models/rf_model_regression_wait_time.pkl')  # wait time prediction

app = Flask(__name__)

# Explicit CORS config — covers all routes and methods
CORS(app)

# Route to predict service delay and wait time
@app.route('/predict', methods=['POST'])
def predict():
    # debug message
    print("Request Headers:", request.headers)  # Print headers
    print("Request Body:", request.data)  # Print raw body

    # Get data
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    # Extract individual features from the incoming request
    service_type = data['service_type']
    date = data['date']
    issue_type = data['issue_type']
    avg_service_time = data['avg_service_time']
    appointments_per_day = data['appointments_per_day']
    service_slots = data['service_slots']
    num_technicians = data['num_technicians']
    backlog_size = data['backlog_size']
    demand_capacity_ratio = data['demand_capacity_ratio']

    # Extract date-related features
    date = pd.to_datetime(date)
    month = date.month
    day_of_week = date.dayofweek  # Monday=0, Sunday=6
    is_weekend = 1 if day_of_week >= 5 else 0

    # Preprocess data to vector
    features = preprocess_input(
        month, day_of_week, is_weekend, service_type, avg_service_time,
        appointments_per_day, service_slots, num_technicians, backlog_size, demand_capacity_ratio
    )

    # predict delay_risk
    delay_risk = model_classification.predict([features])

    # predict wait time estimator
    wait_time = model_regression.predict([features])

    # return jsonify
    return jsonify({
        'delay_risk': int(delay_risk[0]),
        'wait_time': float(wait_time[0])
    })


def preprocess_input(month, day_of_week, is_weekend, service_type, avg_service_time,
                     appointments_per_day, service_slots, num_technicians, backlog_size, demand_capacity_ratio):

    # Map service_type to numerical value
    service_type_mapping = {'repair': 0, 'maintenance': 1, 'inspection': 2}
    service_type = service_type_mapping.get(service_type, -1)

    # Combining all features into a single list
    features = [
        month,  # Month of the service request
        day_of_week,  # Day of the week (0=Monday, 6=Sunday)
        is_weekend,  # Whether it's a weekend (1=weekend, 0=weekday)
        service_type,  # Service type encoded as an integer
        avg_service_time,  # Average time required for the service
        appointments_per_day,  # Number of appointments per day
        service_slots,  # Number of available service slots
        num_technicians,  # Number of technicians available
        backlog_size,  # Number of requests in the service backlog
        demand_capacity_ratio  # Demand-to-capacity ratio
    ]

    return features


if __name__ == '__main__':
    app.run(debug=True)
