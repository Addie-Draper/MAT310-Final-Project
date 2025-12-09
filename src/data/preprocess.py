import os
import pandas as pd
import numpy as np
from .load_data import load_dataset


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by replacing null values in the feature columns
    with the rounded column mean."""

    """Means were calculated previously and are hardcoded here for simplicity"""
    
    df = df.copy()

    # Replace string "none" with actual NaN
    
    df = df.replace("none", np.nan)

    # Handle Support Calls: fill NaN with 2, then cast to int safely
    if "Support Calls" in df.columns:
        df["Support Calls"] = (
            df["Support Calls"]
            .fillna(2)
            .astype(int)
        )

    # Handle Payment Delay: fill NaN with 10.0
    if "Payment Delay" in df.columns:
        df["Payment Delay"] = df["Payment Delay"].fillna(10.0)

    # Handle Last Interaction: fill NaN with 14.0
    if "Last Interaction" in df.columns:
        df["Last Interaction"] = df["Last Interaction"].fillna(14.0)

        

    return df



if __name__ == "__main__":
    # Load the raw dataset
    raw = load_dataset("data/raw/train(1).csv")
    # Clean the dataset
    cleaned = clean_dataset(raw)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path = "data/processed/train(1)_cleaned.csv"
    cleaned.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")
