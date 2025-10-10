# hdb-resale
Exploration and modelling of HDB resale price.

# Primary Questions
1. Transaction-level valuation: Given a flat’s attributes and location today, what is its expected resale price (or $/sqm)?

2. Market-level forecasting (time series): How will the monthly HDB resale price index / median $/sqm evolve over the next 3–12 months given macro + policy?

3. Policy impact: How did specific cooling measures affect prices/volume across flat types/locations?

# Setup
## Python & Dependencies
- Use `Python 3.13` (see `.python-version`).

### Using uv (recommended)
- Install all project dependencies into a managed environment and run commands without manual activation:
  - `uv sync`  
    - resolves from `pyproject.toml` / `uv.lock` and prepares `.venv`
  - `uv run jupyter lab`  
    - example: run Jupyter using the project environment
  - `uv run python -c "import pandas; print(pandas.__version__)"`
    - if it outputs `2.3.3`, then the required packages are installed correctly
- Note: `uv` locates/creates the project environment automatically; you do not need to `source` a venv when using `uv run`.

### Using pip
- If you prefer `pip`:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`

## Data (DVC + Google Drive)
- We use DVC with a Google Drive remote to store datasets.
- Only pickle files and large CSVs are stored via DVC. Jupyter notebooks are not stored in DVC.
- Accessing the remote:
  - Obtain the `config.local` file (shared privately via Telegram or other means).
  - Save it as `.dvc/config.local`.
  - This file contains private keys/credentials for the Google Cloud project; do not share it and do not commit it.

## DVC Setup (Local)
- Assumes you have cloned the repo and have `dvc` installed.
- Steps:
  - Place `.dvc/config.local` as described above.
  - Verify the remote: `dvc remote list`.
  - Pull data: `dvc pull` (downloads tracked files like `data/data.csv`, `data/cleaned_data.pkl`).
  - Optional: check state: `dvc status`.

## DVC Basics
- `dvc pull` — download the latest tracked data from the remote.
- `dvc push` — upload your tracked data changes to the remote (only for allowed file types here: pickles, large CSVs).
- `dvc status` — see what’s changed compared to the cache/remote.
- `dvc add <path>` — start tracking a new large file. Then:
  - `git add <path>.dvc .gitignore && git commit -m "Track data with DVC"`
  - `dvc push` to upload the data contents.
- Do not track notebooks with DVC.
- Learn more: DVC docs at https://dvc.org/doc

## Notebooks & Jupytext
- We version control only Jupytext files: `*.ju.py`.
- Jupyter notebooks (`*.ipynb`) are not version controlled and are not stored via DVC due to file size and limited value in versioning.
- Convert with `jupytext`:
  - `jupytext --to py notebook.ipynb`  # convert a notebook to a `.py` file
  - `jupytext --to notebook notebook.py`  # convert a `.py` file to `.ipynb`
- Consult the Jupytext documentation for additional conversion options.
