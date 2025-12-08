from matplotlib import ticker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_eda(df: pd.DataFrame) -> None:
    """Create exploratory plots including bivariate views colored by churn."""
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Class Distribution')
    plt.show()

    sns.countplot(x='Support Calls', data=df, hue="Churn")
    plt.title('Support Calls Distribution vs Churn')
    plt.show()
    
    """
    sns.countplot(x='Total Spend', data=df, hue="Churn")
    plt.title('Total Spend Distribution vs Churn')
    plt.show()
    """

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Last Interaction', data=df, hue="Churn")
    plt.title('Last Interaction Distribution vs Churn')
    plt.tight_layout()
    plt.show()

    sns.histplot(df['Total Spend'], bins=30)
    plt.title('Total Spend Distribution')
    plt.show()

    # Bivariate visualizations colored by churn
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Age', hue='Churn', palette=['green', 'red'])
    plt.title('Age Distribution vs Churn')
    plt.xlabel('Age')
    plt.ylabel('Count')
    #plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title='Churn', labels=['Non-Churn', 'Churn'])
    plt.show()

    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='distance_from_home',
        y='ratio_to_median_purchase_price',
        hue='fraud',
        palette={0: 'green', 1: 'red'},
    )
    plt.title(
        'Churn vs Non-Churn: Distance from Home vs Ratio to Median Purchase Price'
    )
    plt.xlabel('Distance from Home')
    plt.ylabel('Ratio to Median Purchase Price')
    plt.show()
    """


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset

    raw = load_dataset("data/raw/train(1).csv")
    clean = clean_dataset(raw)
    plot_eda(clean)
