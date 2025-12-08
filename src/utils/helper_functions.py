"""Utility helpers for the project."""
import pandas as pd


def main() -> None:
    pass

def oneHotEncode(df, columns):
    """One-hot encode specified columns in the DataFrame."""
    return pd.get_dummies(df, columns=columns, drop_first=True)

def bestModel(auc_scores, models, labels):
    """Select the best model based on AUC scores."""
    best_index = auc_scores.index(max(auc_scores))
    return models[best_index], labels[best_index]

if __name__ == "__main__":
    main()
