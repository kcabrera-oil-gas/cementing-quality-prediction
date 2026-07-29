# Models

## Training

1. `data_generator.py` generates synthetic data
2. `train.py` trains Random Forest with scaling
3. Model saved to `outputs/models/model.pkl`

## Model Format

```python
{
    "model": RandomForestRegressor,
    "scaler": StandardScaler,
    "feature_names": [...],
    "target_name": "target",
    "metrics": {"r2": 0.81}
}
```
