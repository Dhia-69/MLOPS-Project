import pandas as pd
import pickle
import numpy as np

"""
This function serves to clean the incoming new data in production when it is JSON format.
"""

# Load pre-saved encoders and dummy column structure
with open("label_encoders.pkl", "rb") as le_file:
    label_encoders = pickle.load(le_file)

with open("dummy_columns.pkl", "rb") as dc_file:
    dummy_columns = pickle.load(dc_file)

# Define the training columns to ensure consistent data alignment
training_cols = [
    "Age", "Number of Children", "Smoking Status", "Physical Activity Level", 
    "Employment Status", "Income", "Alcohol Consumption", "Dietary Habits", 
    "Sleep Patterns", "History of Mental Illness", "History of Substance Abuse", 
    "Family History of Depression", "Chronic Medical Conditions", 
    "Marital Status_Divorced", "Marital Status_Married", 
    "Marital Status_Single", "Marital Status_Widowed", 
    "Education Level_Associate Degree", "Education Level_Bachelor's Degree", 
    "Education Level_High School", "Education Level_Master's Degree", 
    "Education Level_PhD"
]

def fix_missing_cols(training_cols, new_data):
    """
    Ensures the new data aligns with training columns by adding missing columns
    and ensuring the correct order.

    Args:
        training_cols (list): List of columns from the training data.
        new_data (pd.DataFrame): New data to be aligned.

    Returns:
        pd.DataFrame: Aligned data.
    """
    missing_cols = set(training_cols) - set(new_data.columns)
    for c in missing_cols:
        new_data[c] = 0  # Add missing columns with default value 0
    new_data = new_data[training_cols]  # Ensure column order matches training
    return new_data

def clean_data_json(df):
    """
    Cleans the incoming JSON data for prediction.

    Args:
        df (pd.DataFrame): Raw data as a Pandas DataFrame.

    Returns:
        pd.DataFrame: Cleaned and processed data ready for prediction.
    """
    # Cube root transformation for Income
    df["Income"] = np.cbrt(df["Income"])

    # Handle categorical features
    for col in ["Marital Status", "Education Level"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Ensure dummy columns align with training columns
    for col in dummy_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing dummy columns
    df = df[dummy_columns]  # Align dummy columns order

    # Use LabelEncoder for other categorical features
    for col in ["Smoking Status", "Physical Activity Level", "Employment Status", 
                "Alcohol Consumption", "Dietary Habits", "Sleep Patterns", 
                "History of Substance Abuse", "Family History of Depression", 
                "Chronic Medical Conditions"]:
        if col in df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = le.transform(df[col])
            else:
                raise ValueError(f"No LabelEncoder found for column: {col}")

    # Align with training columns
    df = fix_missing_cols(training_cols, df)
    return df
