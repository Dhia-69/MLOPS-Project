import pandas as pd
import pickle
import numpy as np


"""
This function serves to clean the incoming new data in production when it is JSON format.
"""

# Load pre-saved encoders and dummy column structure
with open("./src/label_encoders.pkl", "rb") as le_file:
    label_encoders = pickle.load(le_file)

with open("./src/dummy_columns.pkl", "rb") as dc_file:
    dummy_columns = pickle.load(dc_file)


def clean_data_json(df):
    # Print the DataFrame as it is received
    print("Received DataFrame (original):", flush=True)
    print(df, flush=True)

    # Rename columns if needed
    df = df.rename(columns={
        "MaritalStatus": "Marital Status",
        "EducationLevel": "Education Level",
        "NumberOfChildren": "Number of Children",
        "SmokingStatus": "Smoking Status",
        "PhysicalActivityLevel": "Physical Activity Level",
        "EmploymentStatus": "Employment Status",
        "AlcoholConsumption": "Alcohol Consumption",
        "DietaryHabits": "Dietary Habits",
        "SleepPatterns": "Sleep Patterns",
        "HistoryOfSubstanceAbuse": "History of Substance Abuse",
        "FamilyHistoryOfDepression": "Family History of Depression",
        "ChronicMedicalConditions": "Chronic Medical Conditions"
    })

    print("After renaming columns:", flush=True)
    print(df, flush=True)

    # Cube root transformation for Income
    df["Income"] = np.cbrt(df["Income"])
    print("After cube root transformation of Income:", flush=True)
    print(df, flush=True)

    # Handle categorical features
    for col in ["Marital Status", "Education Level"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    print("After one-hot encoding Marital Status and Education Level:", flush=True)
    print(df, flush=True)

    # Ensure dummy columns align with training columns
    for col in dummy_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[dummy_columns]

    print("After aligning dummy columns:", flush=True)
    print(df, flush=True)

    # Use LabelEncoder for other categorical features
    for col in [
        "Smoking Status", "Physical Activity Level", "Employment Status",
        "Alcohol Consumption", "Dietary Habits", "Sleep Patterns",
        "History of Substance Abuse", "Family History of Depression",
        "Chronic Medical Conditions"
    ]:
        if col in df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = le.transform(df[col])
            else:
                raise ValueError(f"No LabelEncoder found for column: {col}")

    print("After label encoding remaining categorical features:", flush=True)
    print(df, flush=True)

    return df

