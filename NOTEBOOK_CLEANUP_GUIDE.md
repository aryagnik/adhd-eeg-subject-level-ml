# Cleaning the Colab notebook for the final submission

The final notebook should be a readable record of the experiment reported in the paper. Keep the code that generates the reported features, models, participant-level predictions, metrics, and figures. Remove old experiments that are not part of the final paper.

## Keep

- Dataset loading and data validation (without exposed credentials).
- The final epoching and feature extraction that produced the reported results: 2-second windows (256 samples) with a 3-second step (384 samples).
- Both feature representations: 40 spectral and 199 multi-domain features.
- The active model-building function and Scikit-learn `Pipeline` objects.
- `SimpleImputer`, `SelectKBest`, `StandardScaler` where used, Logistic Regression, Random Forest, DummyClassifier, and the SVM with group-aware probability calibration.
- Five-fold participant-level `GroupKFold`, participant-ID checks, epoch-to-subject probability aggregation, metric computation, ROC plots, and confusion matrices.
- A final cell that saves `metrics.csv`, `metrics.json`, figures, and participant-level predictions.

## Remove from the final experiment path (or move to a clearly labelled exploratory appendix)

- Old `XGBClassifier` experiments and `StackingClassifier` experiments: these are not among the models reported in the paper.
- The stacking model's ordinary `cv=5` setting: it does not group epochs by participant and should not be used for subject-independent evaluation.
- Global `SelectKBest.fit_transform(X, y)` performed before cross-validation: this leaks label information from held-out participants into feature selection.
- PCA fitted on the full dataset if its components are used in model evaluation. A PCA scatter plot may remain only as clearly labelled exploratory visualization, not as part of the reported classifier pipeline.
- Duplicate definitions of `wavelet_entropy`, `add_connectivity`, repeated imports, repeated `pip install` cells, and unused functions/imports such as unused `nolds` calculations.
- Redundant dataset-download routes. Keep one documented access route.
- Cells that upload, display, or copy `kaggle.json`. Never save credentials in the notebook or repository.

## Do not remove the Scikit-learn Pipeline

The `Pipeline` used by the final models is important. It ensures imputation, feature selection, and scaling are fitted within each training fold. It is not the obsolete stacking pipeline. Keep the pipelines used by the four final model configurations.

## Final notebook checks

1. Restart the runtime and run all cells from top to bottom.
2. Check that all participant IDs and labels remain aligned with feature rows.
3. Confirm the feature shapes and that each participant appears in exactly one outer test fold.
4. Confirm that all eight feature-set/model combinations complete and reproduce the committed metrics (within numeric tolerance).
5. Clear any output that reveals API keys, personal paths, or credentials.
6. Save as `notebooks/ADHD_EEG_Final.ipynb`.
