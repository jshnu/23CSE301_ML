
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

from asvspoof_utils import (
    DEFAULT_DATASET,
    classification_metrics,
    get_positive_class_scores,
    load_asvspoof_data,
    make_output_dir,
    stratified_split,
)


def build_search_configurations(random_state: int = 42):
    """Return the two preprocessing pipelines and their search spaces."""
    perceptron_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                Perceptron(
                    max_iter=3000,
                    tol=1e-4,
                    random_state=random_state,
                ),
            ),
        ]
    )
    perceptron_parameters = {
        "classifier__penalty": [None, "l1", "l2", "elasticnet"],
        "classifier__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "classifier__eta0": [0.01, 0.05, 0.1, 0.5, 1.0],
        "classifier__fit_intercept": [True, False],
        "classifier__early_stopping": [False, True],
    }

    mlp_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    solver="adam",
                    max_iter=700,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=random_state,
                ),
            ),
        ]
    )
    mlp_parameters = {
        "classifier__hidden_layer_sizes": [(50,), (100,), (100, 50), (150,)],
        "classifier__activation": ["relu", "tanh"],
        "classifier__alpha": [1e-5, 1e-4, 1e-3],
        "classifier__learning_rate_init": [5e-4, 1e-3, 5e-3],
        "classifier__batch_size": ["auto", 64],
    }
    return {
        "Perceptron": (perceptron_pipeline, perceptron_parameters),
        "MLPClassifier": (mlp_pipeline, mlp_parameters),
    }


def tune_one_model(
    model_name,
    estimator,
    parameter_space,
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir,
    n_iter=10,
    cv_folds=5,
    n_jobs=1,
    random_state=42,
):
    """Run randomized CV, evaluate the refitted best estimator, and save outputs."""
    cross_validator = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_space,
        n_iter=n_iter,
        scoring="f1",
        cv=cross_validator,
        refit=True,
        n_jobs=n_jobs,
        random_state=random_state,
        return_train_score=True,
        verbose=1,
        error_score="raise",
    )

    start_time = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(X_train, y_train)
    search_seconds = time.perf_counter() - start_time

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    scores = get_positive_class_scores(best_model, X_test)
    metrics = classification_metrics(y_test, predictions, scores)
    metrics.update(
        {
            "Model": model_name,
            "Best_CV_F1": float(search.best_score_),
            "Search_Time_Seconds": float(search_seconds),
            "Candidates_Tested": int(len(search.cv_results_["params"])),
            "Convergence_Warnings": int(len(caught_warnings)),
            "Best_Parameters": str(search.best_params_),
        }
    )

    safe_name = model_name.lower()
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        output_dir / f"A2_{safe_name}_cv_results.csv", index=False
    )
    joblib.dump(best_model, output_dir / f"A2_best_{safe_name}.joblib")
    (output_dir / f"A2_{safe_name}_classification_report.txt").write_text(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["genuine", "spoof"],
            zero_division=0,
        ),
        encoding="utf-8",
    )
    return metrics


def run_experiment(
    data_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
    n_iter: int,
    cv_folds: int,
    n_jobs: int,
):
    """Tune both classifiers and write a consolidated test metric table."""
    X, y, _ = load_asvspoof_data(data_path)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=test_size, random_state=random_state
    )

    result_rows = []
    configurations = build_search_configurations(random_state)
    for model_name, (estimator, parameter_space) in configurations.items():
        result_rows.append(
            tune_one_model(
                model_name,
                estimator,
                parameter_space,
                X_train,
                y_train,
                X_test,
                y_test,
                output_dir,
                n_iter=n_iter,
                cv_folds=cv_folds,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )

    results = pd.DataFrame(result_rows).sort_values("F1", ascending=False)
    results.to_csv(output_dir / "A2_tuned_model_metrics.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="A2: Tune Perceptron and MLP using RandomizedSearchCV"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=10, help="Randomized candidates per model.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel CV jobs; use -1 for all cores.")
    args = parser.parse_args()

    if args.n_iter < 1 or args.cv_folds < 2:
        parser.error("--n-iter must be >= 1 and --cv-folds must be >= 2.")
    output_dir = make_output_dir(args.output_dir, "A2")
    results = run_experiment(
        args.data,
        output_dir,
        args.test_size,
        args.random_state,
        args.n_iter,
        args.cv_folds,
        args.n_jobs,
    )
    print("A2 completed: RandomizedSearchCV hyperparameter tuning")
    print(results.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
