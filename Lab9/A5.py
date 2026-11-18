
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from asvspoof_utils import (
    DEFAULT_DATASET,
    build_lime_pipeline,
    load_asvspoof_data,
    make_output_dir,
)


def split_with_indices(X, y, test_size=0.20, random_state=42):
    """Recreate the stratified split used by A4."""
    all_indices = np.arange(len(X))
    return train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def load_or_train_pipeline(model_path, X_train, y_train, n_features, random_state, fallback_path):
    """Load A4's fitted pipeline, or train an identical pipeline when unavailable."""
    if model_path is not None and Path(model_path).is_file():
        return joblib.load(model_path), "loaded"

    pipeline = build_lime_pipeline(n_features=n_features, random_state=random_state)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, fallback_path)
    return pipeline, "trained"


def create_lime_explanation(
    pipeline,
    X_train,
    X_test,
    y_test,
    feature_names,
    output_dir,
    test_index=None,
    num_samples=5000,
):
    """Explain one test prediction and save its LIME HTML, plot and feature weights."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as error:
        raise ModuleNotFoundError(
            "The 'lime' package is required for A5. Install it with:\n"
            "    python -m pip install lime"
        ) from error

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    if test_index is None:
        # Prefer a correctly classified spoof trial; otherwise explain the first test row.
        suitable_rows = np.flatnonzero((y_test.to_numpy() == 1) & (predictions == 1))
        selected_position = int(suitable_rows[0]) if len(suitable_rows) else 0
    else:
        selected_position = int(test_index)
        if selected_position < 0 or selected_position >= len(X_test):
            raise IndexError(f"--test-index must be from 0 to {len(X_test) - 1}.")

    explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(dtype=float),
        feature_names=list(feature_names),
        class_names=["genuine", "spoof"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    row = X_test.iloc[selected_position].to_numpy(dtype=float)
    explanation = explainer.explain_instance(
        data_row=row,
        predict_fn=pipeline.predict_proba,
        labels=(0, 1),
        num_features=min(20, len(feature_names)),
        num_samples=num_samples,
    )

    explanation.save_to_file(str(output_dir / "A5_LIME_explanation.html"))
    figure = explanation.as_pyplot_figure(label=1)
    figure.suptitle("A5 - LIME explanation for spoof-class probability", y=1.02)
    figure.tight_layout()
    figure.savefig(output_dir / "A5_LIME_explanation.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    weights = explanation.as_list(label=1)
    pd.DataFrame(weights, columns=["Feature_Condition", "LIME_Weight"]).to_csv(
        output_dir / "A5_LIME_feature_weights.csv", index=False
    )
    sample_details = {
        "Test_Set_Position": selected_position,
        "Actual_Label": "spoof" if int(y_test.iloc[selected_position]) == 1 else "genuine",
        "Predicted_Label": "spoof" if int(predictions[selected_position]) == 1 else "genuine",
        "Genuine_Probability": float(probabilities[selected_position, 0]),
        "Spoof_Probability": float(probabilities[selected_position, 1]),
    }
    pd.DataFrame([sample_details]).to_csv(output_dir / "A5_explained_sample.csv", index=False)
    return sample_details, weights


def run_experiment(
    data_path,
    output_dir,
    model_path,
    test_size,
    random_state,
    test_index,
    num_samples,
):
    """Load the data and pipeline, then generate a local LIME explanation."""
    X, y, _ = load_asvspoof_data(data_path)
    train_indices, test_indices = split_with_indices(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    pipeline, model_source = load_or_train_pipeline(
        model_path,
        X_train,
        y_train,
        X.shape[1],
        random_state,
        output_dir / "A5_pipeline_copy.joblib",
    )
    sample_details, weights = create_lime_explanation(
        pipeline,
        X_train,
        X_test,
        y_test,
        X.columns,
        output_dir,
        test_index=test_index,
        num_samples=num_samples,
    )
    return model_source, sample_details, weights


def main():
    default_model_path = Path(__file__).resolve().parent / "outputs" / "A4" / "A4_pipeline.joblib"
    parser = argparse.ArgumentParser(description="A5: Explain an ASVspoof pipeline prediction using LIME")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--model-path", type=Path, default=default_model_path)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-index", type=int, default=None, help="Position within the test split; default selects a correctly predicted spoof if possible.")
    parser.add_argument("--num-samples", type=int, default=5000)
    args = parser.parse_args()

    output_dir = make_output_dir(args.output_dir, "A5")
    model_source, sample_details, weights = run_experiment(
        args.data,
        output_dir,
        args.model_path,
        args.test_size,
        args.random_state,
        args.test_index,
        args.num_samples,
    )
    print("A5 completed: LIME local explanation")
    print(f"Pipeline source: {model_source}")
    print(pd.DataFrame([sample_details]).to_string(index=False))
    print("\nLargest displayed local contributions:")
    print(pd.DataFrame(weights, columns=["Feature_Condition", "LIME_Weight"]).to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
