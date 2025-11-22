# src/data_prep.py
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RAW_DATA_PATH, PROCESSED_DATA_DIR, TARGET_COLUMN, RANDOM_STATE


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["application_date"])

    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    train_path = PROCESSED_DATA_DIR / "train.csv"
    valid_path = PROCESSED_DATA_DIR / "valid.csv"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    print(f"Train guardado en: {train_path} ({len(train_df)} filas)")
    print(f"Valid guardado en: {valid_path} ({len(valid_df)} filas)")


if __name__ == "__main__":
    main()
