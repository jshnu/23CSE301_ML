import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


DATASET = "ASVspoof2017_V2_CQCC_Lab7_500.csv"


def load_dataset():

    data = pd.read_csv(DATASET)

    feature_columns = [
        column
        for column in data.columns
        if column.startswith("CQCC_")
    ]

    X = data[feature_columns]

    y = data["Label"]

    return X, y


def split_dataset(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def perform_grid_search(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=42
    )

    parameter_grid = {

        "criterion": [
            "gini",
            "entropy"
        ],

        "max_depth": [
            2,
            3,
            4,
            5,
            7
        ],

        "min_samples_split": [
            2,
            5,
            10,
            20
        ],

        "min_samples_leaf": [
            1,
            2,
            5
        ]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    return grid_search


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions
    )

    return accuracy, report


X, y = load_dataset()

X_train, X_test, y_train, y_test = split_dataset(
    X,
    y
)

grid_search = perform_grid_search(
    X_train,
    y_train
)

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation accuracy:")
print(grid_search.best_score_)

best_model = grid_search.best_estimator_

test_accuracy, report = evaluate_model(
    best_model,
    X_test,
    y_test
)

print("\nTest accuracy:")
print(test_accuracy)

print("\nClassification report:")
print(report)
