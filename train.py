import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix


def set_seed(seed: int):
    """Garantiza la reproducibilidad fijando las semillas."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_and_preprocess_data(csv_path: str):
    """Carga los datos y aplica el truncado de outliers (Winsorizing)."""
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['Diabetes_binary'])
    y = df['Diabetes_binary'].astype(int)

    num_cols = X.select_dtypes(include=[np.number]).columns

    X_truncated = X.copy()
    for col in num_cols:
        Q1 = X_truncated[col].quantile(0.25)
        Q3 = X_truncated[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        X_truncated[col] = X_truncated[col].clip(lower=limite_inferior, upper=limite_superior)

    return X_truncated, y


def plot_and_save_confusion_matrix(y_true, y_pred, output_path="confusion_matrix.png"):
    """Genera y guarda la matriz de confusión como imagen."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def run_experiment(args):
    """Flujo principal de entrenamiento y registro en MLflow."""
    set_seed(args.seed)

    mlflow.set_experiment(args.experiment_name)

    X, y = load_and_preprocess_data(args.data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    params = {
        "C": args.C,
        "max_iter": args.max_iter,
        "solver": args.solver,
        "class_weight": args.class_weight,
        "random_state": args.seed
    }

    run_name = f"LogReg_C_{args.C}_solver_{args.solver}_cw_{args.class_weight}"
    with mlflow.start_run(run_name=run_name):
        # A) Registrar Parámetros y Tags
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.log_params(params)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("test_size", args.test_size)

        # B) Entrenar Modelo
        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)

        # C) Predicciones y Evaluación
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        recall_train = recall_score(y_train, y_train_pred)
        recall_test = recall_score(y_test, y_test_pred)
        precision_test = precision_score(y_test, y_test_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        # D) Registrar Métricas
        mlflow.log_metric("train_recall", recall_train)
        mlflow.log_metric("test_recall", recall_test)
        mlflow.log_metric("test_precision", precision_test)
        mlflow.log_metric("test_accuracy", accuracy_test)

        # E) Registrar Artefacto (Imagen)
        cm_path = plot_and_save_confusion_matrix(y_test, y_test_pred)
        mlflow.log_artifact(cm_path)
        if os.path.exists(cm_path):
            os.remove(cm_path)

        # F) Registrar el Modelo
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"=== Corrida Finalizada: {run_name} ===")
        print(f"Recall Test: {recall_test:.4f} | Precision Test: {precision_test:.4f} | Accuracy Test: {accuracy_test:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de entrenamiento parametrizado para MLflow")

    parser.add_argument("--data_path", type=str, default="diabetes_binary_health_indicators_BRFSS2015.csv", help="Ruta al CSV")
    parser.add_argument("--experiment_name", type=str, default="Clasificacion_Diabetes_MLOps", help="Nombre del experimento MLflow")
    
    # Hiperparámetros
    parser.add_argument("--C", type=float, default=1.0, help="Fuerza de regularización")
    parser.add_argument("--max_iter", type=int, default=1000, help="Iteraciones máximas")
    parser.add_argument("--solver", type=str, default="lbfgs", choices=["lbfgs", "liblinear", "saga"])
    parser.add_argument("--class_weight", type=str, default="balanced", choices=["balanced", "None"])

    # Semilla
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--test_size", type=float, default=0.20, help="Proporción de prueba")

    args = parser.parse_args()
    
    if args.class_weight == "None":
        args.class_weight = None

    run_experiment(args)