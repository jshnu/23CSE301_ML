
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from asvspoof_utils import (
    DEFAULT_DATASET,
    build_lime_pipeline,
    classification_metrics,
    get_positive_class_scores,
    load_asvspoof_data,
    make_output_dir,
)


def split_with_indices(X, y, test_size=0.20, random_state=42):
    """Split row indices so the source FileName can be attached to predictions."""
    all_indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return train_indices, test_indices


def run_pipeline_experiment(data_path: Path, output_dir: Path, test_size: float, random_state: int):
    """Fit the pipeline and export its model, selected features and test results."""
    X, y, source_frame = load_asvspoof_data(data_path)
    train_indices, test_indices = split_with_indices(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    pipeline = build_lime_pipeline(n_features=X.shape[1], random_state=random_state)
    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start_time

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, predictions, probabilities)
    metrics["Fit_Time_Seconds"] = fit_seconds
    metrics["Selected_Features"] = int(
        pipeline.named_steps["feature_selection"].get_support().sum()
    )

    joblib.dump(pipeline, output_dir / "A4_pipeline.joblib")
    pd.DataFrame([metrics]).to_csv(output_dir / "A4_pipeline_metrics.csv", index=False)

    selector = pipeline.named_steps["feature_selection"]
    selected_names = X.columns[selector.get_support()]
    selected_scores = selector.scores_[selector.get_support()]
    pd.DataFrame(
        {"Feature": selected_names, "ANOVA_F_Score": selected_scores}
    ).sort_values("ANOVA_F_Score", ascending=False).to_csv(
        output_dir / "A4_selected_features.csv", index=False
    )

    file_names = (
        source_frame.iloc[test_indices]["FileName"].astype(str).to_numpy()
        if "FileName" in source_frame.columns
        else test_indices.astype(str)
    )
    prediction_frame = pd.DataFrame(
        {
            "FileName": file_names,
            "Actual_Label": y_test.map({0: "genuine", 1: "spoof"}).to_numpy(),
            "Predicted_Label": pd.Series(predictions, index=y_test.index)
            .map({0: "genuine", 1: "spoof"})
            .to_numpy(),
            "Spoof_Probability": probabilities,
        }
    )
    prediction_frame.to_csv(output_dir / "A4_test_predictions.csv", index=False)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["genuine", "spoof"],
        zero_division=0,
    )
    (output_dir / "A4_classification_report.txt").write_text(report, encoding="utf-8")

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["genuine", "spoof"],
        cmap="Blues",
        values_format="d",
        ax=axis,
    )
    axis.set_title("A4 - Pipeline Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "A4_pipeline_confusion_matrix.png", dpi=160)
    plt.close(figure)

    return metrics, report


def main():
    parser = argparse.ArgumentParser(description="A4: Build and run an ML pipeline")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = make_output_dir(args.output_dir, "A4")
    metrics, report = run_pipeline_experiment(
        args.data, output_dir, args.test_size, args.random_state
    )
    print("A4 completed: preprocessing and classification pipeline")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"\n{report}")
    print(f"Pipeline saved to: {output_dir / 'A4_pipeline.joblib'}")


if __name__ == "__main__":
    main()
