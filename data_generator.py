import numpy as np
import pandas as pd

def generate_dataset(n_samples=1000, seed=42):
    np.random.seed(seed)
    n = n_samples
    density = np.random.uniform(0, 100, n)
    yield_stress = np.random.uniform(0, 100, n)
    plastic_viscosity = np.random.uniform(0, 100, n)
    fluid_loss = np.random.uniform(0, 100, n)
    thickening_time = np.random.uniform(0, 100, n)
    temp_bottom = np.random.uniform(0, 100, n)
    annular_vel = np.random.uniform(0, 100, n)
    pipe_centralization = np.random.uniform(0, 100, n)
    score = (0.5*density + 0.5*yield_stress + 0.5*plastic_viscosity + 0.5*fluid_loss + 0.5*thickening_time + 0.5*temp_bottom + 0.5*annular_vel + 0.5*pipe_centralization) / 8
    bond_quality = np.where(score > score.mean(), "good", "poor")

    df = pd.DataFrame({
        "density": density,
        "yield_stress": yield_stress,
        "plastic_viscosity": plastic_viscosity,
        "fluid_loss": fluid_loss,
        "thickening_time": thickening_time,
        "temp_bottom": temp_bottom,
        "annular_vel": annular_vel,
        "pipe_centralization": pipe_centralization,
        "bond_quality": bond_quality,
    })
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print(f"Generated {len(df)} samples")
