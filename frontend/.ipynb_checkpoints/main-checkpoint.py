import requests
import streamlit as st
import pandas as pd
import json
import os
from streamlit_option_menu import option_menu  # Importing the option menu library

# Get the backend URL from the environment variable
backend_url = os.getenv("BACKEND_URL", "http://localhost:8080")

# Create a horizontal navigation bar
selected_page = option_menu(
    menu_title=None,  # No title for the menu
    options=["Home", "Predict with Form", "Predict with CSV", "Power BI Dashboard"],  # Menu options
    icons=["house", "clipboard-data", "file-earmark-arrow-up", "graph-up"],  # Icons from Bootstrap
    menu_icon="cast",  # Icon for the menu button (if collapsible)
    default_index=0,  # Default active menu
    orientation="horizontal",  # Orientation: horizontal or vertical
    styles={
        "container": {"padding": "0!important", "background-color": "#f9f9f9"},
        "icon": {"color": "blue", "font-size": "20px"}, 
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "color": "#2c3e50",
        },
        "nav-link-selected": {"background-color": "#4b72d9", "color": "white"},
    },
)

# Apply global styling
st.markdown(
    """
    <style>
    body { font-family: 'Arial', sans-serif; }
    .main { border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    h1, h2, h3 { font-family: 'Helvetica', sans-serif; color: #2c3e50; }
    h1 { border-bottom: 2px solid #4b72d9; padding-bottom: 10px; }
    .stButton>button { background-color: #4b72d9; color: white; border-radius: 5px; padding: 10px 20px; }
    .stButton>button:hover { background-color: #3a5bb1; }
    .stFileUploader { padding: 10px; border: 1px dashed #4b72d9; border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Home Page
if selected_page == "Home":
    st.title("🧠 Mental Health Prediction App")
    st.image(
        "sddefault.jpg", 
        caption="Empowering mental health awareness and prediction"
    )
    st.markdown(
        """
        ## Welcome!
        This application leverages advanced machine learning to assess mental health risks based on user inputs.

        ### Features:
        - **Interactive Form**: Provide personal details to get an instant prediction.
        - **CSV Upload**: Analyze multiple records simultaneously.
        - **Power BI Dashboard**: Visualize mental health trends and data insights.
        
        Use the navigation bar to explore the app.
        """
    )

# Predict with Form Page
elif selected_page == "Predict with Form":
    st.title("📋 Predict Mental Health - Form")
    st.subheader("Fill in the details to receive a mental health risk prediction.")

    with st.form("form1", clear_on_submit=False):
        # Input fields
        col1, col2 = st.columns(2)
        with col1:
            Age = st.number_input("Enter Age", min_value=0, max_value=120, value=35)
            Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            Education_Level = st.selectbox(
                "Education Level", 
                ["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"]
            )
            Physical_Activity_Level = st.selectbox(
                "Physical Activity Level", 
                ["Sedentary", "Moderate", "Active"]
            )
            Employment_Status = st.selectbox("Employment Status", ["Employed", "Unemployed"])

        with col2:
            Number_of_Children = st.number_input("Number of Children", min_value=0, max_value=20, value=2)
            Smoking_Status = st.selectbox("Smoking Status", ["Non-smoker", "Former", "Current"])
            Alcohol_Consumption = st.selectbox("Alcohol Consumption", ["Low", "Moderate", "High"])
            Sleep_Patterns = st.selectbox("Sleep Patterns", ["Poor", "Fair", "Good"])
            Income = st.number_input("Income (USD)", min_value=0.0, value=60000.0, step=1000.0)

        Dietary_Habits = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"])
        History_of_Substance_Abuse = st.selectbox("History of Substance Abuse", ["Yes", "No"])
        Family_History_of_Depression = st.selectbox("Family History of Depression", ["Yes", "No"])
        Chronic_Medical_Conditions = st.selectbox("Chronic Medical Conditions", ["Yes", "No"])

        # Collect data in a dictionary
        input_data = {
            "Age": Age,
            "Marital Status": Marital_Status,
            "Education Level": Education_Level,
            "Number of Children": Number_of_Children,
            "Smoking Status": Smoking_Status,
            "Physical Activity Level": Physical_Activity_Level,
            "Employment Status": Employment_Status,
            "Income": Income,
            "Alcohol Consumption": Alcohol_Consumption,
            "Dietary Habits": Dietary_Habits,
            "Sleep Patterns": Sleep_Patterns,
            "History of Substance Abuse": History_of_Substance_Abuse,
            "Family History of Depression": Family_History_of_Depression,
            "Chronic Medical Conditions": Chronic_Medical_Conditions,
        }

        # Submit button
        submit = st.form_submit_button("💡 Get Prediction")

        if submit:
            try:
                response = requests.post(f"{backend_url}/predict", data=json.dumps(input_data))
                predictions = response.json().get("predictions")

                # Display the prediction result
                if predictions == [0]:
                    st.success("✅ The individual is not at risk of mental health issues.")
                elif predictions == [1]:
                    st.warning("⚠️ The individual may be at risk of mental health issues.")
                else:
                    st.error(predictions)
                    st.error("❌ An error occurred while processing the prediction.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Predict with CSV Page
elif selected_page == "Predict with CSV":
    st.title("📁 Predict Mental Health - CSV")
    st.subheader("Upload a CSV file to analyze multiple records at once.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            file = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{backend_url}/predict/csv", files=file)
            predictions = response.json().get("predictions")

            # Display predictions for each record
            st.subheader("Predictions for uploaded data")
            for idx, prediction in enumerate(predictions, start=1):
                st.text(f"Record {idx}: {'⚠️ At risk' if prediction == 1 else '✅ Not at risk'}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# Power BI Dashboard Page
elif selected_page == "Power BI Dashboard":
    st.title("📊 Mental Health Dashboard")
    st.markdown(
        """
        This dashboard provides an overview of trends and insights into mental health data.
        """
    )

    st.markdown(
        f'<iframe title="Depression" width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiYmM3MDA5NTUtMjg5Mi00ZTM2LWIxODktNzRlMDJiYjU2NzVlIiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9&pageName=337f1f81144651557047" frameborder="0" allowFullScreen="true"></iframe>',
        unsafe_allow_html=True,
    )
