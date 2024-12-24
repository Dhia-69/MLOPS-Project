#from dotenv import load_dotenv
import os
from data_preprocessing_training import transform_data
import pandas as pd
from imblearn.over_sampling import SMOTE
import mlflow
from treatement import naive_bayes_model
from treatement import knn_model

#Load environement variable (Dagshub credentials)
# from dotenv import load_dotenv
# import os
# load_dotenv("../backend/src/.env")

# DagsHub_username = os.getenv("DagsHub_username")
# DagsHub_token=os.getenv("DagsHub_token")

#Get Dagshub credentials
# os.environ['MLFLOW_TRACKING_USERNAME']= DagsHub_username
# os.environ["MLFLOW_TRACKING_PASSWORD"] = DagsHub_token

#Affect Daghsub credentials 

os.environ['MLFLOW_TRACKING_USERNAME']= "KoubaaMahdi"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "666511bfd2b2180d037f16fac402803ea2a8aed1"

#setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/KoubaaMahdi/MLOps_project.mlflow') #your mlfow tracking uri

#setup mlflow
mlflow.set_experiment("depression-detection-experiment")

#Data Url and version
version = "v2.0"
data_url = "../../data/depression_data.csv"

#read the data
df = pd.read_csv(data_url)

#cleaning and preprocessing
X_train,X_test,y_train,y_test = transform_data(df)

#Execute the models with new version of data:
naive_bayes_model(data_url,version,df,X_train,y_train,X_test,y_test)
knn_model(data_url,version,df,X_train,y_train,X_test,y_test)
