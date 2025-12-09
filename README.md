# Final Project -- Addie Draper

This repository is my final project for MAT311, about modelling and machine learning, in the context of a company and customers churning.

## Purpose

The purpose of this project was to take a dataset about customers in a company and the data to predict whether or not that customer is going to churn. Using features like the total amount of money they spent on the product, the number of support calls they had, and others, I was able to create numerous models, including a knn model, a decision tree model, and a random forest model. 

## Project layout

```
.
├── main.py                 # Entry point that runs the entire pipeline
├── requirements.txt        # Python dependencies
├── data/
│   ├── processed/          # Created after running the pipeline
│   └── raw/
│       └── train(1).csv
│       └── test.csv
├── notebooks/
│   └── 3nn.ipynb
│   └── model5.ipynb
│   └── simple_classifer.ipynb
└── src/
    ├── data/
    │   ├── load_data.py
    │   ├── preprocess.py
    │   └── split_data.py
    ├── features/
    │   └── build_features.py
    ├── models/
    │   ├── train_model.py
    │   └── knn_model.py
    │   └── decision_tree_model.py
    │   └── random_forest_model.py
    ├── utils/
    │   └── helper_functions.py
    └── visualization/
        ├── eda.py
        └── performance.py
```

`main.py` imports the modules inside `src/` and executes them to reproduce the analysis and results. Jupyter notebooks were used for data exploration and model creation before converting the code into python scripts. 

## How to run:

Install the dependencies and run the pipeline. You should use the versions of the dependencies as specified by the requirements file:

```bash
conda create -n final_proj --file requirements.txt
conda activate final_proj
python main.py
```

This will load the dataset, perform basic feature engineering, train a simple model and produce visualizations similar to those in the notebooks.
The cleaned data will be written to `data/processed/` and all plots will be displayed interactively.
