"""Participant-grouped comparison of spectral and multi-domain EEG features."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SPECTRAL_POSITIONS = [5, 6, 7, 8, 9]
CHANNELS = ["F3", "F4", "Fz", "P3", "P4", "Pz", "O1", "O2"]


def spectral_indices() -> list[int]:
    return [ch * 24 + pos for ch in range(len(CHANNELS)) for pos in SPECTRAL_POSITIONS]


def make_model(name: str, n_features: int):
    k = min(25, n_features)
    if name == "Dummy":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DummyClassifier(strategy="prior")),
        ])
    common = [("imputer", SimpleImputer(strategy="median")),
              ("selector", SelectKBest(score_func=f_classif, k=k))]
    if name == "Logistic Regression":
        return Pipeline(common + [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=3000,
                                               class_weight="balanced", random_state=42)),
        ])
    if name == "Random Forest":
        return Pipeline(common + [
            ("classifier", RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                  class_weight="balanced_subsample",
                                                  random_state=42, n_jobs=-1)),
        ])
    if name == "SVM":
        return Pipeline(common + [
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=1.0, kernel="rbf", gamma="scale",
                               probability=False, class_weight="balanced")),
        ])
    raise ValueError(f"Unknown model: {name}")


def fit_model(name: str, X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray):
    base = make_model(name, X_train.shape[1])
    if name != "SVM":
        base.fit(X_train, y_train)
        return base
    inner = GroupKFold(n_splits=3)
    splits = list(inner.split(X_train, y_train, groups_train))
    for inner_train, inner_valid in splits:
        if np.unique(y_train[inner_train]).size != 2:
            raise ValueError("An inner SVM calibration training fold lacks one class.")
        if np.unique(y_train[inner_valid]).size != 2:
            raise ValueError("An inner SVM calibration validation fold lacks one class.")
    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid",
                                        cv=splits, ensemble=False)
    calibrated.fit(X_train, y_train)
    return calibrated


def metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "Sensitivity": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "Specificity": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Macro F1": f1_score(y_true, pred, average="macro", zero_division=0),
        "ROC AUC": roc_auc_score(y_true, prob),
    }


def subject_aggregate(groups: np.ndarray, y: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"Subject": groups, "True": y, "Probability": prob})
    label_counts = df.groupby("Subject")["True"].nunique()
    if (label_counts > 1).any():
        raise ValueError("At least one participant has inconsistent labels.")
    out = (df.groupby("Subject", as_index=False)
             .agg({"True": "first", "Probability": "mean"}))
    out["Predicted"] = (out["Probability"] >= 0.5).astype(int)
    return out


def run(features_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(features_csv)
    required = {"ID", "Class"}
    if not required.issubset(df.columns):
        raise ValueError("Feature CSV must contain ID and Class columns.")
    feature_cols = [c for c in df.columns if c not in required]
    if len(feature_cols) != 199:
        raise ValueError(f"Expected 199 multi-domain features, found {len(feature_cols)}")

    X_full = df[feature_cols].to_numpy(dtype=float)
    X_full[~np.isfinite(X_full)] = np.nan
    y = df["Class"].to_numpy(dtype=int)
    groups = df["ID"].astype(str).to_numpy()
    X_spec = X_full[:, spectral_indices()]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    feature_sets = {"Spectral": X_spec, "Multi-domain": X_full}
    model_names = ["Dummy", "Logistic Regression", "Random Forest", "SVM"]
    cv = GroupKFold(n_splits=5)
    rows, all_predictions = [], {}

    for feature_name, X in feature_sets.items():
        for model_name in model_names:
            oof = np.full(len(y), np.nan)
            fold_scores = []
            for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
                gtr, gte = groups[tr], groups[te]
                if not set(gtr).isdisjoint(set(gte)):
                    raise RuntimeError("Participant leakage detected in outer fold.")
                if np.unique(y[tr]).size != 2 or np.unique(y[te]).size != 2:
                    raise ValueError("A GroupKFold split contains only one class; inspect group labels/folds.")
                model = fit_model(model_name, X[tr], y[tr], gtr)
                class_index = list(model.classes_).index(1)
                oof[te] = model.predict_proba(X[te])[:, class_index]
                test_subjects = subject_aggregate(gte, y[te], oof[te])
                fold_scores.append(metrics(test_subjects["True"].to_numpy(),
                                           test_subjects["Probability"].to_numpy()))

            if not np.isfinite(oof).all():
                raise RuntimeError("Some epochs did not receive out-of-fold predictions.")
            subject_pred = subject_aggregate(groups, y, oof)
            res = {"Feature Set": feature_name, "Model": model_name,
                   **metrics(subject_pred["True"].to_numpy(), subject_pred["Probability"].to_numpy()),
                   "Fold Balanced Accuracy Mean": np.mean([x["Balanced Accuracy"] for x in fold_scores]),
                   "Fold Balanced Accuracy SD": np.std([x["Balanced Accuracy"] for x in fold_scores], ddof=1)}
            rows.append(res)
            all_predictions[(feature_name, model_name)] = subject_pred
            subject_pred.to_csv(output_dir / "predictions" /
                                f"{feature_name}_{model_name.replace(' ', '_')}.csv", index=False)
            print(feature_name, model_name, res)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "metrics.csv", index=False)
    results.to_markdown(output_dir / "metrics.md", index=False, floatfmt=".4f")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, feature_name in zip(axes, feature_sets):
        for model_name in model_names:
            p = all_predictions[(feature_name, model_name)]
            fpr, tpr, _ = roc_curve(p["True"], p["Probability"])
            auc = roc_auc_score(p["True"], p["Probability"])
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", linewidth=0.8)
        ax.set(title=f"{feature_name} features", xlabel="False positive rate",
               ylabel="True positive rate")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "roc_comparison.png", dpi=200)
    plt.close(fig)

    # Confusion matrix for the best cross-validated configuration (descriptive only).
    best_idx = results["Balanced Accuracy"].idxmax()
    best = results.loc[best_idx]
    best_pred = all_predictions[(best["Feature Set"], best["Model"])]
    cm = confusion_matrix(best_pred["True"], best_pred["Predicted"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Control", "ADHD"],
           yticklabels=["Control", "ADHD"], xlabel="Predicted", ylabel="Actual",
           title=f"{best['Model']} / {best['Feature Set']}")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "best_confusion_matrix.png", dpi=200)
    plt.close(fig)
    print("\nSaved results to", output_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/features_199.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.features, args.output)


if __name__ == "__main__":
    main()
