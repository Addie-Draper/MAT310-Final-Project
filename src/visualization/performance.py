import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def plot_confusion_matrices(y_test, y_pred_dt, y_pred_rf) -> None:
    """Plot confusion matrices for both models."""
    conf_dt = confusion_matrix(y_test, y_pred_dt)
    conf_rf = confusion_matrix(y_test, y_pred_rf)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(conf_dt, annot=True, fmt='d', cmap='Reds', ax=axes[0])
    axes[0].set_title('Decision Tree')
    sns.heatmap(conf_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Random Forest')
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(y_test, y_pred_dt, y_pred_rf) -> None:
    """Create a bar chart comparing model metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    dt_scores = [
        accuracy_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_dt)
    ]
    rf_scores = [
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf)
    ]
    df = pd.DataFrame({'Metric': metrics, 'Decision Tree': dt_scores, 'Random Forest': rf_scores})
    df.plot(x='Metric', kind='bar', figsize=(8, 5))
    plt.ylim(0, 1)
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset
    from src.models.train_model import train_models
    print("I'm the problem")

    raw = load_dataset("data/raw/train(1).csv")
    clean = clean_dataset(raw)
    y_test, baseline, knn = train_models(clean)
    plot_confusion_matrices(y_test, baseline, knn)
    plot_performance_comparison(y_test, baseline, knn)
