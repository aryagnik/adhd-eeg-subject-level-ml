# GitHub setup

## 1. Prepare this folder

1. Download and extract the repository ZIP.
2. Open a terminal in the extracted `ADHD_EEG_ES678_GitHub_Repo` folder.
3. Export your actual executed Colab notebook as `notebooks/ADHD_EEG_Final.ipynb`.
4. Review the notebook and remove credential-upload cells/outputs. Do not add raw EEG data or `kaggle.json`.
5. Replace the roll-number and GitHub URL placeholders in `paper/es678_project_template.tex` before final paper compilation.

## 2. Create an empty GitHub repository

On GitHub, choose **New repository** and name it `adhd-eeg-subject-level-ml` (or another clear name). Choose Public only if your course requires a public repository and the contents are safe to share. Do not select “Add a README”, `.gitignore`, or license because these files already exist locally.

Copy the HTTPS repository URL, for example:

`https://github.com/YOUR-USERNAME/adhd-eeg-subject-level-ml.git`

## 3. Initialize and push (PowerShell, Git Bash, WSL, or macOS terminal)

Run these commands from the extracted project folder:

```bash
git init
git branch -M main
git add README.md GIT_SETUP.md AI_USE_DISCLOSURE.md .gitignore requirements.txt notebooks src results paper
git status
```

**Stop and inspect `git status` before committing.** Make sure it does not list `kaggle.json`, `adhdata.csv`, raw EEG files, tokens, passwords, or other private data. If any appear, do not commit; remove them from the staging area and fix `.gitignore`.

Then commit and push:

```bash
git commit -m "Add ES 678 ADHD EEG classification project"
git remote add origin https://github.com/YOUR-USERNAME/adhd-eeg-subject-level-ml.git
git push -u origin main
```

Replace the URL with the actual URL copied from your GitHub repository. GitHub may prompt you to authenticate in the browser or through Git Credential Manager; do not put a password or token directly into the commands.

## 4. Check the published repository

Open the repository page and verify that:

- README displays correctly;
- `src/`, `results/`, and `paper/` are present;
- your actual notebook appears under `notebooks/`;
- no credentials or raw restricted dataset were uploaded;
- the metrics in the repository match the paper.

Copy the repository URL into the course portal and the paper's reproducibility section.

## 5. Push later changes

After updating files:

```bash
git status
git add <specific-files-you-changed>
git commit -m "Update final notebook and documentation"
git push
```

Use `git add .` only after checking that no private data or credentials are in the project folder.
