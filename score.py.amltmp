import json
import os
import pandas as pd
import joblib

# Define the expected columns based on training data
expected_columns = [
    'age', 'balance', 'day', 'duration', 'campaign', 'previous',
    'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
    'job_retired', 'job_self-employed', 'job_services', 'job_student',
    'job_technician', 'job_unemployed', 'job_unknown', 'marital_married',
    'marital_single', 'education_secondary', 'education_tertiary',
    'education_unknown', 'contact_telephone', 'contact_unknown',
    'month_aug', 'month_dec', 'month_feb', 'month_jan', 'month_jul',
    'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct',
    'month_sep', 'poutcome_other', 'poutcome_success', 'poutcome_unknown',
    'housing_new', 'loan_new'
]

# Initialize the model
def init():
    global model
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], 'random_forest_model.pkl')
    model = joblib.load(model_path)

# Run function for inference
def run(raw_data):
    try:
        # Parse input JSON
        input_data = json.loads(raw_data)

        # Validate input format
        if "data" not in input_data:
            return {"error": "Invalid input format. 'data' key is missing."}

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data["data"])

        # Add missing columns with default values
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default to 0 for missing numeric fields

        # Reorder columns to match training data
        input_df = input_df[expected_columns]

        # Ensure all columns are numeric
        input_df = input_df.astype(float, errors='ignore')

        # Make predictions
        predictions = model.predict(input_df)

        # Return predictions as JSON
        return {"predictions": predictions.tolist()}

    except Exception as e:
        # Log and return the error for debugging
        return {"error": str(e)}