# src/model_utils.py
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


NUMERIC_FEATURES: List[str] = [
    "age",
    "dependents",
    "monthly_income",
    "employment_months",
    "requested_amount",
    "loan_term_months",
    "interest_rate",
    "installment",
    "debt_to_income",
    "num_open_loans",
    "num_credit_cards",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "marital_status",
    "employment_type",
    "has_mortgage",
    "channel",
    "region",
]


def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[target_column].values
    return X, y


def build_model_pipeline(
    random_state: int = 42,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    class_weight: str = None,
    l1_ratio: float = 0.5,
) -> Pipeline:
    """
    Build a model pipeline with configurable hyperparameters.
    
    Args:
        random_state: Random state for reproducibility
        C: Inverse of regularization strength (smaller = stronger regularization)
        penalty: Regularization penalty ('l1', 'l2', 'elasticnet', None)
        solver: Algorithm to use ('lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga')
        max_iter: Maximum number of iterations
        class_weight: Class weight strategy ('balanced', None, or dict)
        
    Returns:
        Scikit-learn Pipeline
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Handle solver compatibility with penalty
    if penalty == "l1":
        if solver not in ["liblinear", "saga"]:
            solver = "liblinear"
    elif penalty == "elasticnet":
        if solver != "saga":
            solver = "saga"
    elif penalty == "l2" or penalty is None:
        if solver not in ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]:
            solver = "lbfgs"

    # Build LogisticRegression with appropriate parameters
    lr_params = {
        "C": C,
        "penalty": penalty,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": random_state,
        "class_weight": class_weight,
    }
    
    # Add l1_ratio only for elasticnet penalty
    if penalty == "elasticnet":
        lr_params["l1_ratio"] = l1_ratio
    
    clf = LogisticRegression(**lr_params)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )
    return model
