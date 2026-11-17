# Updating the existing GitHub repository

You do not need to create another repository or upload the whole project manually. Copy/replace the changed files inside your existing local repository, then commit and push the changes.

From PowerShell, open the existing repo folder and check:

```powershell
git status
```

After copying the updated `paper` folder and README/guide files over the corresponding local files:

```powershell
git status
git diff -- README.md paper/es678_project_template.tex
```

Review the changes. Then stage only the relevant files:

```powershell
git add README.md GIT_UPDATE_GUIDE.md NOTEBOOK_CLEANUP_GUIDE.md `
  paper/es678_project_template.tex paper/neurips_2026.sty paper/references.bib `
  paper/es678_project_template.pdf paper/figures/feature_model_comparison.pdf `
  paper/figures/roc_curves.png results/figures/roc_curves.png
```

If you also edited the notebook, stage it separately after verifying it contains no secrets:

```powershell
git add notebooks/ADHD_EEG_Final.ipynb
```

Then commit and push:

```powershell
git commit -m "Add ROC analysis and update ES 678 paper"
git push
```

Git sends the new commit and changed file objects; you do not re-upload the entire repository. Never stage raw EEG data, `kaggle.json`, API tokens, `.venv`, or local environment files.
