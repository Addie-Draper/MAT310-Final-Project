import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Train and return a decision tree classifier."""
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    model.fit(X_train, y_train)
    return model