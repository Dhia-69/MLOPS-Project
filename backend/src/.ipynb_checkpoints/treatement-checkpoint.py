import pandas as pd
from sklearn.metrics      import confusion_matrix, classification_report
from sklearn.metrics      import precision_recall_fscore_support as score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import mlflow
import warnings

def naive_bayes_model(data_url, version, df, x_train, y_train, x_test, y_test):
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='Naive Bayes'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])
        nb = GaussianNB()
        params = nb.get_params()
        mlflow.set_tag(key= "model", value="NaiveBayes")
        mlflow.log_params(params)
        nb.fit(x_train,y_train)
        train_features_name = f'{x_train=}'.split('=')[0]
        train_label_name = f'{y_train=}'.split('=')[0]
        mlflow.set_tag(key="train_features_name",value= train_features_name)
        mlflow.set_tag(key= "train_label_name",value=train_label_name)
        predicted=nb.predict(x_test)
        precision,recall,fscore,support=score(y_test,predicted,average='macro')
        mlflow.log_metric("Precision_test",precision)
        mlflow.log_metric("Recall_test",recall)
        mlflow.log_metric("F1_score_test",fscore)
        mlflow.sklearn.log_model(nb,artifact_path="ML_models")

def knn_model(data_url, version, df, x_train, y_train, x_test, y_test):
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='K-Nearest Neighbors'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])
        knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan')
        params = knn.get_params()
        mlflow.set_tag(key="model", value="K-Nearest Neighbors")
        mlflow.log_params(params)
        knn.fit(x_train,y_train)
        train_features_name = f'{x_train=}'.split('=')[0]
        train_label_name = f'{y_train=}'.split('=')[0]
        mlflow.set_tag(key="train_features_name",value= train_features_name)
        mlflow.set_tag(key= "train_label_name",value=train_label_name)
        predicted=knn.predict(x_test)
        precision,recall,fscore,support=score(y_test,predicted,average='macro')
        mlflow.log_metric("Precision_test",precision)
        mlflow.log_metric("Recall_test",recall)
        mlflow.log_metric("F1_score_test",fscore)
        mlflow.sklearn.log_model(knn,artifact_path="ML_models")
    