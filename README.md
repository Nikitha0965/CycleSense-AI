# AI-powered Menstrual Health App (MHA)

An open-source project to help users **track periods**, **predict cycles (including irregular ones)**, discover **personalized home remedies**, and get **risk flags** for potential health issues for timely doctor consultation.

##  Features
- **Cycle tracking**: log periods, symptoms, pain levels, mood, and notes.
- **AI predictions**: baseline statistical + ML (upgradeable to deep learning) for next period/ovulation.
- **Personalized remedies**: rule-based recommendations by age, body type, and symptoms (extensible).
- **Insights dashboard**: trends (cycle length, pain intensity, flow), adherence, and anomaly alerts.
- **Privacy-first**: local-only mode or API mode; no third-party analytics by default.
- **Deployable**: run locally via Python, Docker, or deploy to cloud.
- **MVP frontend**: Streamlit app; swap-in React later.

##  Architecture
```
frontend-streamlit/  -> MVP UI (Streamlit)
backend/             -> FastAPI REST API (predict, logs, remedies)
ml/                  -> training code + model registry
data/                -> sample dataset (synthetic)
models/              -> saved models (optional; API falls back to baseline)
tests/               -> pytest backend tests
.github/workflows/   -> CI (lint + tests)
docker-compose.yml   -> one command to run full stack
```

##  Quickstart (Local)
### 1) Backend (FastAPI)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2) Frontend (Streamlit MVP)
Open a new terminal:
```bash
cd frontend-streamlit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Visit: http://localhost:8501  (Frontend)  
Backend docs: http://localhost:8000/docs

### 3) Docker (optional)
```bash
docker compose up --build
```

##  Run Tests
```bash
pip install -r backend/requirements-dev.txt
pytest -q
```

## 🔧 Train a Baseline Model
```bash
cd ml
python train_baseline.py  # saves models/baseline_model.pkl
```

##  Tech Stack
- **Backend**: FastAPI, Pydantic, Uvicorn
- **ML**: scikit-learn, pandas, numpy, joblib
- **Frontend**: Streamlit (MVP)
- **CI**: GitHub Actions (pytest)
- **Packaging**: Docker & docker-compose



## 📄 License
MIT — see [LICENSE](LICENSE).
