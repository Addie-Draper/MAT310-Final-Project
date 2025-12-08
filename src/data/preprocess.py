import os
import pandas as pd
import numpy as np
from .load_data import load_dataset


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by replacing null values with the rounded column mean."""
    ##return df.apply(lambda col: col.fillna(round(col.mean())) if col.dtype.kind in 'biufc' else col )
    """
    df_filled = df.copy()
    for col in df_filled.select_dtypes(include="number"):
        df_filled[col] = df_filled[col].fillna(round(df_filled[col].mean()))
    return df_filled """

    """
    df = df.replace("none", np.nan)

    numeric_cols = df.select_dtypes(include="number").columns
    
    for col in numeric_cols:
        mean_val = df[col].mean()
        if pd.notna(mean_val):  # only if mean is valid
            df[col] = df[col].fillna(round(mean_val))
    
    for col in df.columns:
        if col == "Support Calls":
            df[col] = df[col].astype(int)
    """
    """
    Clean the training dataset by replacing 'none' with NaN and filling
    specific columns with default values.
    
    Parameters:
        df (pd.DataFrame): Input training dataset
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
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

    print(df.head())
        

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
