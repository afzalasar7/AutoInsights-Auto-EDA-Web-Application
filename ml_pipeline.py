import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle


def validate_dataset(df: pd.DataFrame) -> dict:
    """Return basic validation stats for a dataframe."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def detect_problem_type(series: pd.Series) -> str:
    """Infer problem type based on the target series."""
    if series.dtype == object or series.nunique() <= 20:
        return "classification"
    return "regression"


def split_dataset(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """Split dataframe into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    stratify = y if detect_problem_type(y) == "classification" else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def train_baseline_model(X_train, y_train, problem_type: str):
    """Train a simple baseline model."""
    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, problem_type: str) -> dict:
    """Evaluate the model and return metrics."""
    preds = model.predict(X_test)
    if problem_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_score": float(f1_score(y_test, preds, average="weighted")),
        }
    return {
        "r2_score": float(r2_score(y_test, preds)),
        "mse": float(mean_squared_error(y_test, preds)),
    }


def export_model(model) -> bytes:
    """Serialize a trained model as bytes."""
    return pickle.dumps(model)
