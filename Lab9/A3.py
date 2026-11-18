
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from asvspoof_utils import (
    DEFAULT_DATASET,
    classification_metrics,
    get_positive_class_scores,
    load_asvspoof_data,
    make_output_dir,
    stratified_split,
)


def build_classifiers(random_state: int = 42):
    """Build the seven classifiers named in the Lab 09 specification."""
    classifiers = {
        "Support Vector Machine": SVC(
            C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=None, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=random_state
        ),
        "Naive Bayes": GaussianNB(),
    }

    unavailable_models = []
    try:
        from catboost import CatBoostClassifier

        classifiers["CatBoost"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            random_seed=random_state,
            thread_count=-1,
        )
    except ImportError:
        unavailable_models.append("CatBoost (install with: pip install catboost)")

    try:
        from xgboost import XGBClassifier

        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        unavailable_models.append("XGBoost (install with: pip install xgboost)")

    return classifiers, unavailable_models


def evaluate_classifier(model_name, classifier, X_train, y_train, X_test, y_test):
    """Fit one classifier and return its metrics, report and fitted pipeline."""
    start_time = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        if model_name == "CatBoost":
            # CatBoost 1.2.x can lack the scikit-learn estimator tag API expected
            # by newer scikit-learn Pipeline implementations. Fit it directly.
            imputer = SimpleImputer(strategy="median")
            X_train_processed = imputer.fit_transform(X_train)
            X_test_processed = imputer.transform(X_test)
            classifier.fit(X_train_processed, y_train)
            model = classifier
            predictions = classifier.predict(X_test_processed).ravel()
            probabilities = classifier.predict_proba(X_test_processed)
            scores = probabilities[:, 1]
        else:
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", classifier),
                ]
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            scores = get_positive_class_scores(model, X_test)
    fit_seconds = time.perf_counter() - start_time
    metrics = {
        "Classifier": model_name,
        **classification_metrics(y_test, predictions, scores),
        "Fit_Time_Seconds": float(fit_seconds),
        "Warnings_During_Fit": int(len(caught_warnings)),
    }
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["genuine", "spoof"],
        zero_division=0,
    )
    return metrics, report, model


def run_comparison(data_path: Path, output_dir: Path, test_size: float, random_state: int):
    """Train all requested classifiers and save the comparison table and plot."""
    X, y, _ = load_asvspoof_data(data_path)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=test_size, random_state=random_state
    )
    classifiers, unavailable_models = build_classifiers(random_state)
    result_rows = []
    report_sections = []

    for model_name, classifier in classifiers.items():
        metrics, report, _ = evaluate_classifier(
            model_name, classifier, X_train, y_train, X_test, y_test
        )
        result_rows.append(metrics)
        report_sections.append(f"===== {model_name} =====\n{report}")

    results = pd.DataFrame(result_rows).sort_values(
        ["F1", "ROC_AUC"], ascending=False
    )
    results.to_csv(output_dir / "A3_classifier_comparison.csv", index=False)
    (output_dir / "A3_classification_reports.txt").write_text(
        "\n\n".join(report_sections), encoding="utf-8"
    )

    plot_data = results.sort_values("F1", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(plot_data["Classifier"], plot_data["F1"])
    axis.set_xlabel("F1-score (spoof class)")
    axis.set_ylabel("Classifier")
    axis.set_title("A3 - Classifier Performance Comparison")
    axis.set_xlim(0, 1)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "A3_classifier_f1_comparison.png", dpi=160)
    plt.close(figure)
    return results, unavailable_models


def main():
    parser = argparse.ArgumentParser(description="A3: Compare seven classification algorithms")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = make_output_dir(args.output_dir, "A3")
    results, unavailable = run_comparison(
        args.data, output_dir, args.test_size, args.random_state
    )
    print("A3 completed: classifier comparison")
    print(results.to_string(index=False))
    if unavailable:
        print("\nClassifiers skipped because packages are unavailable:")
        for item in unavailable:
            print(f"- {item}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
