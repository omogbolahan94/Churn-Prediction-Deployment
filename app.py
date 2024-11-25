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
        test_data = pd.read_csv(file)
    elif filename.endswith('.xlsx'): 
        test_data = pd.read_excel(file, engine='openpyxl')
    elif filename.endswith('.xls'):
        test_data = pd.read_csv(file, sep='\t') #, engine='xlrd')
    else:
        return "Invalid file format. Only CSV and Excel files are supported.", 400

    # Ensure all training-time columns are present
    required_columns = [
        'late_payments_last_year', 'missed_payments_last_year', 'plan_tenure',
        'num_employees', 'avg_monthly_contribution', 'annual_revenue',
        'support_calls_last_year', 'support_engagement_per_year',
        'major_issue_Technical Issue'
    ]

    # Ensure 'major_issue' is one-hot encoded if needed
    if 'major_issue_Technical Issue' not in test_data.columns:
        test_data = pd.get_dummies(test_data, columns=['major_issue_Technical Issue'], drop_first=True)

    
    for col in required_columns:
        if col not in test_data.columns:
            test_data[col] = 0  # Add missing columns with default value 0

    customer_ids = test_data['customer_id']  
    # # Extract features and customer IDs
    test_data_features = test_data[required_columns]

    # # Make predictions
    test_data['predicted_churn'] = model.predict(test_data_features)

    # # Select the desired output columns (excluding churn_probability)
    output_columns = [
        'customer_id', 'late_payments_last_year', 'missed_payments_last_year',
        'plan_tenure', 'num_employees', 'avg_monthly_contribution',
        'annual_revenue', 'support_calls_last_year', 'support_engagement_per_year',
        'major_issue_Technical Issue', 'predicted_churn'
    ]
    output_data = test_data[output_columns]

    # # Specify the file name
    output_file = 'test_data_with_predictions_and_ids.csv'

    # # Select the desired output columns from the DataFrame
    output_data = test_data[output_columns]

    # # Save the selected output to a CSV file
    output_file_path = os.path.join(os.getcwd(), output_file)
    output_data.to_csv(output_file_path, index=False)

    # # Provide a download link for the saved CSV file
    return send_file(output_file_path, as_attachment=True, download_name="predictions.csv")


if __name__ == "__main__":
    app.run(debug=True)
