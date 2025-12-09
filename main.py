from src.models.decision_tree_model import train_decision_tree_model
from src.models.random_forest_model import train_random_forest_model
from src.data.load_data import load_dataset
from src.data.preprocess import clean_dataset
from src.visualization.eda import plot_eda
from src.models.train_model import split_data, plot_roc_curve
from src.models.knn_model import train_knn_model
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.helper_functions import bestModel


def main() -> None:
    print("---Loading data...")
    raw_df = load_dataset("data/raw/train(1).csv")
    
    # Print shape of the raw dataset
    print(f"Raw dataset shape: {raw_df.shape}")

    
    print("---Cleaning data...")
    clean_df = clean_dataset(raw_df)
    
    print(f"Cleaned dataset shape: {clean_df.shape}")
    

    print("---Creating EDA visuals...")
    plot_eda(clean_df)

    print("---Selecting features...")
    from src.features.build_features import select_features
    clean_df = select_features(clean_df)

    print("---One-hot encoding categorical variables...")
    from src.utils.helper_functions import oneHotEncode
    oneHot_df = oneHotEncode(clean_df, columns=["Contract Length"])

    print("---Splitting data...")
    oneHot_df = oneHot_df.copy()  # Assuming oneHot_df is already preprocessed
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(oneHot_df)
    

    print("---Training models...")
    knn_model = train_knn_model(X_train, y_train)
    decision_tree_model = train_decision_tree_model(X_train, y_train)
    random_forest_model = train_random_forest_model(X_train, y_train)


    print("---Evaluating on validation set...")
    y_val_pred_knn = knn_model.predict(X_val)
    y_val_pred_dtree = decision_tree_model.predict(X_val)
    y_val_pred_rf = random_forest_model.predict(X_val)

    val_prob_knn = knn_model.predict_proba(X_val)[:, 1]
    val_prob_dtree = decision_tree_model.predict_proba(X_val)[:, 1]
    val_prob_rf = random_forest_model.predict_proba(X_val)[:, 1]

    #plot_confusion_matrices(y_val, y_val_pred_knn, y_val_pred_dtree)
    plot_confusion_matrices(y_val, y_val_pred_dtree, y_val_pred_rf)
    plot_performance_comparison(y_val, y_val_pred_dtree, y_val_pred_rf)

    auc_knn = plot_roc_curve(y_val, val_prob_knn, "3-NN")
    auc_dtree = plot_roc_curve(y_val, val_prob_dtree, "Decision Tree")
    auc_rf = plot_roc_curve(y_val, val_prob_rf, "Random Forest")
    
    models = [knn_model, decision_tree_model, random_forest_model]
    auc_scores = [auc_knn, auc_dtree, auc_rf]
    labels = ["3-NN", "Decision Tree", "Random Forest"]

    best_model, best_label = bestModel(auc_scores, models, labels)

    print(f"---Testing best model ({best_label})...")
    y_test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, test_prob, f"Test {best_label}")

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Best Model Confusion Matrix: "+best_label)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
