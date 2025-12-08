import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train and return a random forest classifier."""
    model = RandomForestClassifier(random_state=123)
    model.fit(X_train, y_train)
    return model