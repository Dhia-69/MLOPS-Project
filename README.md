# MLOps Project: Mental Illness Detection

This repository contains the complete implementation of an MLOps pipeline for a **IA models** capable of providing detailed mental health analysis, lifestyle-based insights, and symptom-based detection. The project follows industry-standard MLOps practices to ensure efficient development, deployment, and monitoring.

---

## Features

1. **Data Workflow:**
   - **Data:** CSV datasets containing lifestyle inputs such as sleep patterns, activity levels.
   - **EDA & Preprocessing:** Performed exploratory data analysis (EDA) and applied robust preprocessing techniques to clean and standardize the data.
   - **Modeling:** Built and trained machine learning models using **MLflow** for experiment tracking.

2. **Version Control:**
   - Integrated **DVC (Data Version Control)** for efficient management of data, pipelines, and models.
   - Managed versioning of datasets and trained models for reproducibility and traceability.

3. **Backend and Frontend Integration:**
   - Developed a **backend** to expose the mental health analysis functionality as an API.
   - Built a user-friendly **frontend** to interact with the chatbot.

4. **Testing:**
   - Used **Deepchecks** to validate the quality of data and the performance of the trained models.

5. **Deployment:**
   - Deployed the application on **Render**, ensuring scalability and reliability.

6. **CI/CD Automation:**
   - Set up **Jenkins** for continuous integration (CI), continuous training (CT), and continuous deployment (CD).

7. **Monitoring:**
   - Integrated **Arize** for end-to-end monitoring of the model's performance and data drift.

---

## Architecture

The MLOps pipeline includes the following components:

1. **Data Management**
   - Raw data ingestion and preprocessing pipelines.
   - Versioning with **DVC**.

2. **Modeling**
   - Model training and hyperparameter tuning using **MLflow**.
   - Model versioning and registry.

3. **Application Development**
   - FastAPI backend for API development.
   - Streamlit frontend for user interaction.

4. **Deployment**
   - Render platform for hosting the application.
   - Dockerized services for easy deployment.

5. **Testing & Validation**
   - Automated testing pipelines with **Deepchecks**.

6. **CI/CD Pipeline**
   - Jenkins pipeline for automated build, test, and deploy processes.

7. **Monitoring**
   - **Arize** platform for tracking model performance and identifying anomalies in predictions.

---

## Technologies Used

- **Data Management:** DVC
- **Model Tracking & Experimentation:** MLflow
- **Testing:** Deepchecks
- **Deployment:** Render, Docker
- **CI/CD:** Jenkins
- **Monitoring:** Arize
- **Frontend:** Streamlit
- **Backend:** FastAPI

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KoubaaMahdi/MLops.git
   cd mental-illness-detection-mlops
   ```

2. Set up DVC:
   ```bash
   dvc init
   ```

3. Train the model and track experiments with MLflow:
   ```bash
   mlflow run .
   ```

4. Deploy the application on Render:
   - Configure `render.yaml` for deployment.
   - Push the repository to trigger deployment.

5. Set up Jenkins for CI/CD:
   - Use the provided `Jenkinsfile`.
   - Configure the pipeline to automate build, test, and deploy steps.

6. Monitor the model:
   - Integrate Arize for real-time monitoring.

---

## Contributions

This project was collaboratively developed by **Mahdi Koubaa**, **Mohamed Taha Mourali**, and **Dhia Elhak Toukebri**.

Feel free to fork the repository and open pull requests to enhance the pipeline or add new features.

---


Thank you for exploring the MLOps pipeline for the Mental Illness Detection system! 🚀
