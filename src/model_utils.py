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


def build_model_pipeline(random_state: int = 42) -> Pipeline:
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

    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )
    return model
