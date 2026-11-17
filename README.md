# ES 678 Course Project: Subject-Level ADHD Classification from EEG

This repository contains code and result files for an individual ES 678 course project at IIT Gandhinagar. The study compares spectral-only EEG features with a multi-domain representation for participant-level ADHD/control classification.

## Research question

Does a multi-domain EEG feature representation improve subject-level ADHD classification compared with spectral features alone, and which classical machine-learning model performs most consistently on previously unseen participants?

## Dataset and access

- Original dataset: *EEG data for ADHD/Control Children*, IEEE DataPort, DOI: [10.21227/rzfh-zn36](https://doi.org/10.21227/rzfh-zn36).
- The analysis notebook accessed a CSV mirror: [Kaggle dataset `danizo/eeg-dataset-for-adhd`](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd).
- The dataset contains 121 children (61 ADHD and 60 controls), 19 EEG channels, sampled at 128 Hz.
- Eight channels were used: F3, F4, Fz, P3, P4, Pz, O1, and O2.
- The submitted notebook version formed 5,621 epochs using 2-second windows (256 samples) and a 3-second step (384 samples), leaving a 1-second gap.
- The feature matrix contains 199 multi-domain features per epoch. The spectral subset contains 40 features: theta, alpha, and beta power, theta/beta ratio, and theta/alpha ratio across eight channels.

Do not redistribute the raw EEG dataset unless its terms explicitly allow redistribution. Obtain it directly from the original source or the Kaggle mirror. Store `adhdata.csv` locally at `data/adhdata.csv`; raw data must remain untracked by Git.

### Kaggle access

You may download the CSV through the Kaggle website or configure the Kaggle CLI locally. Never commit `kaggle.json`, API tokens, passwords, or credential-bearing notebook outputs. For Colab, use a private secret or a secure runtime upload.

## Methods

The comparison uses a prior-probability Dummy Classifier baseline and three candidate models:

- Logistic Regression
- Random Forest
- RBF-kernel Support Vector Machine (SVM)

Each is assessed with spectral and multi-domain features. Five-fold `GroupKFold` keeps all epochs from a participant in one outer fold. Epoch probabilities are averaged to produce one participant-level prediction. Imputation, feature selection, and scaling are fitted inside the training pipelines; SVM probability calibration uses three participant-grouped inner folds. The model settings are fixed for the reported experiment; no global hyperparameter search is claimed.

## Repository structure

```text
.
├── README.md
├── AI_USE_DISCLOSURE.md
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── ADHD_EEG_Final.ipynb     # Contains extra code for Pipelining and Stacking.
    └── ADHD_EEG_Cleaned.ipynb   # Removed excess code from the colab file.
├── src/
│   ├── extract_features.py
│   └── evaluate.py
├── results/
│   ├── metrics.json
│   ├── metrics.csv
│   ├── metrics.md
│   └── figures/
│       └── roc_curves.png
        └── feature_model_comparison.pdf
└── paper/
    ├── es678_project_template.tex
    ├── es678_project_template.pdf
    ├── neurips_2026.sty
    ├── references.bib
    └── figures/
        ├── feature_model_comparison.pdf
        └── roc_curves.png
```

## Setup

Python 3.10 or later is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / WSL / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Results

Participant-level pooled out-of-fold metrics are stored in `results/metrics.csv` and `results/metrics.json`.

- Spectral features: best candidate balanced accuracy is 64.3% (Random Forest).
- Multi-domain features: all three learned models obtain 77.7% accuracy and approximately 77.6-77.7% balanced accuracy.
- Multi-domain Random Forest has the highest ROC-AUC (86.3%) and specificity (75.0%).
- Multi-domain SVM has the highest ADHD sensitivity (88.5%).

The ROC plots are included in the paper. They show higher ROC-AUC for the multi-domain representation for all three learned models: Logistic Regression 0.644 to 0.800, Random Forest 0.695 to 0.863, and SVM 0.622 to 0.837. The Dummy classifier's pooled ROC-AUC is not treated as a meaningful ranking measure because its fold-specific prior scores are not on a common learned ranking scale; the 50.0% balanced accuracy is the more useful baseline summary.

## Reproducibility

The primary executable implementation of this project is provided in notebooks/ADHD_EEG_Final.ipynb. The notebook contains the final feature extraction, model training, participant-level evaluation, and result visualization pipeline. It also includes clearly labelled exploratory analysis.

## Dataset Access

The dataset, EEG data for ADHD/Control Children, is publicly available through IEEE DataPort. The notebook uses a CSV mirror available through Kaggle. The dataset must be downloaded separately; raw EEG data and Kaggle API credentials are not included in this repository.

## Experimental Configuration

The experiment uses eight EEG channels (F3, F4, Fz, P3, P4, Pz, O1, O2), with 2-second windows (256 samples) and a 3-second step (384 samples). It compares 40 spectral features with 199 multi-domain features using a Dummy Classifier, Logistic Regression, Random Forest, and RBF-kernel SVM.

Evaluation uses five-fold GroupKFold cross-validation, with participant IDs used as grouping variables. Epoch-level prediction probabilities are averaged to obtain participant-level predictions, using a threshold of 0.5.

## Running the Experiment

1. Clone the repository
2. Install the dependencies listed in requirements.txt.
3. Download the dataset using the provided access instructions.
4. Configure the dataset path in the notebook.
5. Run all notebook cells sequentially.

The notebook generates model-comparison metrics, participant-level predictions, ROC curves, and confusion matrices under the results/ directory.

## Reproducibility Notes

The committed metrics are the reference results for the reported experiment. A clean execution of the notebook should be used to verify these results. Minor numerical differences may occur across software environments.

The Python files in src/ are companion implementations and should not be assumed to reproduce the notebook exactly unless independently verified.

The reported experiment uses 2-second windows and a 3-second step. Any change to epoching, preprocessing, feature extraction, model configuration, or evaluation must be followed by a complete rerun of the experiments and an update to the reported results and paper.

## AI-use disclosure

See [`AI_USE_DISCLOSURE.md`](AI_USE_DISCLOSURE.md).

## Limitations

The independent sample comprises 121 participants, not 5,621 independent cases. Participants contribute unequal numbers of epochs; GroupKFold prevents participant overlap but does not guarantee class-stratified folds. There is no external test cohort. The analysis does not include a separate band-pass filtering or ICA artifact-removal stage. The findings are dataset-specific and do not establish clinical diagnostic validity.
