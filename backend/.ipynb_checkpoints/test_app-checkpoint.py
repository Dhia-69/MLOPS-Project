from pathlib import Path
import pickle
import pandas as pd
import sys
from src.clean_data_json import clean_data_json
import mlflow
#from dotenv import load_dotenv
import os

from dotenv import load_dotenv

load_dotenv("../backend/src/.env")

DagsHub_username = os.getenv("DagsHub_username")
DagsHub_token=os.getenv("DagsHub_token")

os.environ['MLFLOW_TRACKING_USERNAME']= DagsHub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = DagsHub_token



#setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/KoubaaMahdi/MLOps_project.mlflow') #your mlfow tracking uri


#tests if the model works as expected

def test_model_use():

    #let's call the model from the model registry ( in production stage)

    all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
    df_mlflow = mlflow.search_runs(experiment_ids=all_experiments,filter_string="metrics.F1_score_test <1")
    run_id = df_mlflow.loc[df_mlflow['metrics.F1_score_test'].idxmax()]['run_id']


    logged_model = f'runs:/{run_id}/ML_models'

    # Load model as a PyFuncModel.
    model = mlflow.pyfunc.load_model(logged_model)

    d = {'Age': 23,"Marital Status":"Single" ,
                "Education Level": "Master's Degree",
                "Number of Children": 0,
                "Smoking Status": "Current",
                "Physical Activity Level": 'Moderate',
                "Employment Status": "Unemployed",
                "Income": 3600 ,
                "Alcohol Consumption": "Low",
                "Dietary Habits": "Healthy",
                "Sleep Patterns": "Fair",
                "History of Substance Abuse": "No",
                "Family History of Depression": "No",
                "Chronic Medical Conditions": "No"}

    df = pd.DataFrame(data=d,index=[0])
    dd = clean_data_json(df)
    predict_result = model.predict(dd)
    print(predict_result[0])
    assert predict_result[0] == 1
test_model_use()