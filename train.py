import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, classification_report
import os, json

DATA_FILE = "generated_data.csv"
MODEL_DIR = "outputs/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    df = pd.read_csv(DATA_FILE)
    X = df["density", "yield_stress", "plastic_viscosity", "fluid_loss", "thickening_time", "temp_bottom", "annular_vel", "pipe_centralization"]
    y = df["bond_quality"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    le = None
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_encoded)
    preds = model.predict(X_test_scaled)
    score = accuracy_score(y_encoded_test, preds)
    print(f"Accuracy: {score:.4f}")
    print(classification_report(y_encoded_test, preds))

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": ['density', 'yield_stress', 'plastic_viscosity', 'fluid_loss', 'thickening_time', 'temp_bottom', 'annular_vel', 'pipe_centralization'],
        "target_name": "bond_quality",
        "metrics": {"r2" if le is None else "accuracy": float(score)}
    }, os.path.join(MODEL_DIR, "model.pkl"))
    print(f"Model saved to {MODEL_DIR}/model.pkl")

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(DATA_FILE, index=False)
    train()
