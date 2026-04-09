## Student Score predictor end-to-end ML project (Streamlit UI)

This project trains a regression model to predict **student math score** and serves predictions via a **Streamlit** app.

### What gets created

- **Training artifacts**: `artifacts/model.pkl`, `artifacts/preprocessor.pkl`, plus `artifacts/train.csv`, `artifacts/test.csv`
- **App**: `app.py` (Streamlit UI)

### Run locally (Windows / PowerShell)

Create a venv (recommended) and install deps:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train the model (creates `artifacts/`):

```bash
python -m src.pipeline.train_pipeline
```

Run the Streamlit UI:

```bash
streamlit run app.py
```

### Deploy

#### Option A: Streamlit Community Cloud (easiest)

- Push this repo to GitHub.
- Go to Streamlit Community Cloud and create a new app.
- **App file**: `app.py`
- **Python deps**: detected from `requirements.txt`
- After deploy, open the app URL.

Notes:
- Your app needs the trained artifacts. In Streamlit Cloud you can either:
  - **Train locally and commit the `artifacts/` folder**, or
  - Add a small “train” step in the app/startup (not recommended for heavy training), or
  - Use GitHub Actions to train and upload artifacts (more advanced).
- **Scikit-learn version must match training**: `preprocessor.pkl` and `model.pkl` are pickled with the sklearn version from `requirements.txt`. If you see errors like `SimpleImputer` / `_fill_dtype` on Streamlit, pin `scikit-learn` (already pinned), push, **redeploy**, and if needed **re-run training** with that pinned version and commit fresh `artifacts/`.

#### Option B: Docker

Create a `Dockerfile` (example) and deploy anywhere Docker runs.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

Build & run:

```bash
docker build -t mlproject .
docker run -p 8501:8501 mlproject
```
