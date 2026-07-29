# Cementing Quality Prediction

ML system for cementing quality prediction

## Architecture

Synthetic Data Generator -> Scikit-Learn Training -> Flask API -> Streamlit Dashboard

## Quick Start

```bash
pip install -r requirements.txt
python train.py
python app.py
streamlit run streamlit_app.py
```

## API

All endpoints require `X-API-Key` header.
