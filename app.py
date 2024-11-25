from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import os
from joblib import load

# Load the model
model_path = "churn_prediction_model.joblib"
model = load(model_path)
print("Model loaded successfully!")


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('form.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    filename = file.filename

    # Load the uploaded file
    if filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        data = pd.read_excel(file)
    else:
        return "Invalid file format. Only CSV and Excel files are supported.", 400

    required_columns = [
        'customer_id', 'late_payments_last_year', 'missed_payments_last_year', 'plan_tenure',
        'num_employees', 'avg_monthly_contribution', 'annual_revenue',
        'support_calls_last_year', 'support_engagement_per_year',
        'major_issue_Technical Issue'
    ]

    if not all(col in data.columns for col in required_columns):
        return f"Missing required columns. Ensure your file contains: {', '.join(required_columns)}", 400

    # Handle one-hot encoding for 'major_issue'
    if 'major_issue' in data.columns:
        data = pd.get_dummies(data, columns=['major_issue'], drop_first=True)

    # Ensure all required columns are present
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0  

    # Extract features for prediction
    data_features = data[required_columns]

    # Make predictions
    data['predicted_churn'] = model.predict(data_features)

    # Select the desired output columns
    output_columns = [
        'customer_id', 'late_payments_last_year', 'missed_payments_last_year',
        'plan_tenure', 'num_employees', 'avg_monthly_contribution',
        'annual_revenue', 'support_calls_last_year', 'support_engagement_per_year',
        'major_issue_Technical Issue', 'predicted_churn'
    ]

    output_data = data[output_columns]

    # Save the output to a CSV file
    output_file = os.path.join(os.getcwd(), 'test_data_with_predictions_and_ids.csv')
    output_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Provide a download link
    return send_file(output_file, as_attachment=True, download_name="predictions.csv")


if __name__ == "__main__":
    app.run(debug=True)
