from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from asvspoof_utils import (
    DEFAULT_DATASET,
    classification_metrics,
    get_positive_class_scores,
    load_asvspoof_data,
    make_output_dir,
    stratified_split,
)


def build_baseline_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Create baseline Perceptron and MLP models with numeric preprocessing."""
    return {
        "Perceptron": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    Perceptron(
                        max_iter=3000,
                        tol=1e-4,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "MLPClassifier": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(100,),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=800,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def run_experiments(data_path: Path, output_dir: Path, test_size: float, random_state: int):
    """Train and evaluate baseline models; save metrics, reports and plots."""
    X, y, source_frame = load_asvspoof_data(data_path)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=test_size, random_state=random_state
    )
    results = []
    reports = []
    fitted_models = {}

    for model_name, model in build_baseline_models(random_state).items():
        start_time = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start_time

        predictions = model.predict(X_test)
        scores = get_positive_class_scores(model, X_test)
        row = {"Model": model_name, **classification_metrics(y_test, predictions, scores)}
        row["Fit_Time_Seconds"] = fit_seconds
        classifier = model.named_steps["classifier"]
        row["Training_Iterations"] = int(classifier.n_iter_)
        row["Convergence_Warnings"] = len(caught_warnings)
        results.append(row)
        fitted_models[model_name] = model
        reports.append(
            f"===== {model_name} =====\n"
            + classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["genuine", "spoof"],
                zero_division=0,
            )
        )

        if model_name == "MLPClassifier":
            figure, axis = plt.subplots(figsize=(5.5, 4.5))
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                predictions,
                display_labels=["genuine", "spoof"],
                cmap="Blues",
                values_format="d",
                ax=axis,
            )
            axis.set_title("A1 - MLPClassifier Confusion Matrix")
            figure.tight_layout()
            figure.savefig(output_dir / "A1_MLP_confusion_matrix.png", dpi=160)
            plt.close(figure)

    metrics_frame = pd.DataFrame(results).sort_values("F1", ascending=False)
    metrics_frame.to_csv(output_dir / "A1_baseline_metrics.csv", index=False)
    (output_dir / "A1_classification_reports.txt").write_text(
        "\n\n".join(reports), encoding="utf-8"
    )
    return metrics_frame, source_frame.shape, len(X_train), len(X_test)


def main():
    parser = argparse.ArgumentParser(description="A1: Baseline ASVspoof project-data experiment")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = make_output_dir(args.output_dir, "A1")
    metrics, dataset_shape, train_count, test_count = run_experiments(
        args.data, output_dir, args.test_size, args.random_state
    )
    print("A1 completed: Lab 08 project-dataset baseline experiment")
    print(f"Dataset rows and columns: {dataset_shape}")
    print(f"Training rows: {train_count} | Test rows: {test_count}")
    print(metrics.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
