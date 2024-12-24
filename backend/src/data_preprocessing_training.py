import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Load pre-saved encoders and dummy column structure
with open("./label_encoders.pkl", "rb") as le_file:
    label_encoders = pickle.load(le_file)

with open("./dummy_columns.pkl", "rb") as dc_file:
    dummy_columns = pickle.load(dc_file)

def transform_data(df):
    y = df['History of Mental Illness']
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
            df[col] = 0
    df = df[dummy_columns]
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
    X = df
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)  # Split data 80/20
    #scaler = MinMaxScaler()  # Normalize train & test features
    #X_train[num_features] = scaler.fit_transform(X_train[num_features])
    #X_test[num_features] = scaler.transform(X_test[num_features])
    return X_train,X_test,y_train,y_test
    