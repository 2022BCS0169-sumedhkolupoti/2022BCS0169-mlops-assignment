import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import json
import os
import argparse

# Mandatory Identification
STUDENT_NAME = "Sumedh Kolupoti"
ROLL_NO = "2022BCS0169"

def train(run_name, data_version, model_type='logistic', C=1.0, feature_selection=False):
    # Set experiment name as per requirement
    mlflow.set_experiment(f"{ROLL_NO}_experiment")
    
    with mlflow.start_run(run_name=run_name):
        # Load tracked data
        df = pd.read_csv('data/breast_cancer.csv')
        X = df.drop('target', axis=1)
        y = df['target']
        
        selected_features = list(X.columns)
        if feature_selection:
            # Mandatory feature selection requirement
            selector = SelectKBest(f_classif, k=10)
            X_new = selector.fit_transform(X, y)
            selected_features = [X.columns[i] for i in selector.get_support(indices=True)]
            X = pd.DataFrame(X_new, columns=selected_features)
            mlflow.log_param("feature_selection", "SelectKBest(k=10)")
            mlflow.log_dict({"selected_features": selected_features}, "features.json")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model selection and Hyperparameter tuning
        if model_type == 'logistic':
            model = LogisticRegression(C=C, max_iter=10000)
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("C", C)
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics Calculation
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Mandatory Identification logging
        mlflow.log_param("student_name", STUDENT_NAME)
        mlflow.log_param("roll_no", ROLL_NO)
        mlflow.log_param("data_version", data_version)
        
        # Log multiple metrics (Mandatory > 2)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log model artifact
        mlflow.sklearn.log_model(model, "model")
        
        # Mandatory Metrics JSON including Identification
        metrics_data = {
            "name": STUDENT_NAME,
            "roll_no": ROLL_NO,
            "run_name": run_name,
            "data_version": data_version,
            "accuracy": acc,
            "f1_score": f1,
            "model_type": model_type,
            "selected_features": selected_features if feature_selection else "all"
        }
        
        metrics_path = f"metrics_{ROLL_NO}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)
        
        mlflow.log_artifact(metrics_path)
        
        print(f"[{run_name}] Acc: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--model", type=str, default='logistic')
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--fs", action="store_true")
    args = parser.parse_args()
    
    train(args.run, args.version, args.model, args.C, args.fs)
