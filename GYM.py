import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

MUSCLE_PARAMS = {
    "Biceps":     {"Gmax": 6.0,  "k": 0.045, "mrv_time": 70,  "recovery_penalty": 0.020},
    "Triceps":    {"Gmax": 6.5,  "k": 0.040, "mrv_time": 75,  "recovery_penalty": 0.018},
    "Shoulders":  {"Gmax": 7.0,  "k": 0.030, "mrv_time": 90,  "recovery_penalty": 0.015},
    "Chest":      {"Gmax": 8.0,  "k": 0.026, "mrv_time": 100, "recovery_penalty": 0.014},
    "Back":       {"Gmax": 9.0,  "k": 0.020, "mrv_time": 130, "recovery_penalty": 0.010},
    "Quads":      {"Gmax": 10.0, "k": 0.017, "mrv_time": 150, "recovery_penalty": 0.009},
    "Hamstrings": {"Gmax": 8.5,  "k": 0.022, "mrv_time": 110, "recovery_penalty": 0.012},
    "Calves":     {"Gmax": 5.0,  "k": 0.035, "mrv_time": 80,  "recovery_penalty": 0.022},
}
 
MUSCLES = list(MUSCLE_PARAMS.keys())

def true_growth_function(muscle, weekly_time, sessions_per_week, intensity_pct, training_age):
    
    p = MUSCLE_PARAMS[muscle]
 
    base_growth = p["Gmax"] * (1 - np.exp(-p["k"] * weekly_time))
 
    excess = np.maximum(0, weekly_time - p["mrv_time"])
    overtraining_penalty = p["recovery_penalty"] * excess
 
    freq_factor = 1 + 0.08 * np.minimum(sessions_per_week, 4) - 0.02 * np.maximum(0, sessions_per_week - 4)
    intensity_factor = 1 - 0.0009 * (intensity_pct - 75) ** 2 / 10
 
    age_factor = 1.35 / (1 + 0.15 * training_age)
 
    growth = (base_growth - overtraining_penalty) * freq_factor * intensity_factor * age_factor
    return np.maximum(growth, 0)

def simulate_dataset(n_samples_per_muscle=300):
    rows = []
    for muscle in MUSCLES:
        weekly_time = rng.uniform(10, 200, n_samples_per_muscle)
        sessions = rng.integers(1, 6, n_samples_per_muscle)      
        intensity = rng.uniform(55, 95, n_samples_per_muscle)     
        training_age = rng.uniform(0, 15, n_samples_per_muscle)        
 
        expected_growth = true_growth_function(muscle, weekly_time, sessions, intensity, training_age)
 
        noise = rng.normal(0, 0.15 * expected_growth + 0.15, n_samples_per_muscle)
        observed_growth = np.maximum(expected_growth + noise, 0)
 
        for i in range(n_samples_per_muscle):
            rows.append({
                "muscle_group": muscle,
                "weekly_training_time_min": round(weekly_time[i], 1),
                "sessions_per_week": int(sessions[i]),
                "avg_intensity_pct_1rm": round(intensity[i], 1),
                "training_age_years": round(training_age[i], 1),
                "muscle_growth_pct_8wk": round(observed_growth[i], 3),
            })
 
    return pd.DataFrame(rows)

 
print("Simulating training dataset...")
df = simulate_dataset(n_samples_per_muscle=300)
print(f"Dataset shape: {df.shape}")
print()

feature_cols = ["muscle_group", "weekly_training_time_min", "sessions_per_week",
                 "avg_intensity_pct_1rm", "training_age_years"]
target_col = "muscle_growth_pct_8wk"
 
X = df[feature_cols]
y = df[target_col]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
 
preprocessor = ColumnTransformer(
    transformers=[("muscle_ohe", OneHotEncoder(handle_unknown="ignore"), ["muscle_group"])],
    remainder="passthrough",
)
 
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_SEED)),
])
 
print("Training RandomForestRegressor...")
model.fit(X_train, y_train)

r2 = r2_score(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))


st.title("Muscle Growth Prediction App")

my_input = pd.DataFrame({
    "muscle_group": [st.selectbox("Select muscle group:", ["Chest", "Back", "Legs", "Shoulders"])],
    "weekly_training_time_min": [st.number_input("Weekly training time (minutes):", min_value=10, max_value=200, value=60)],
    "sessions_per_week": [st.number_input("Sessions per week:", min_value=1, max_value=10, value=3)],
    "avg_intensity_pct_1rm": [st.slider("Average intensity (% 1RM):", min_value=50, max_value=90, value=75)],
    "training_age_years": [st.slider("Training age (years):", min_value=0, max_value=20, value=2)],
})

predicted_growth = model.predict(my_input)
st.write(f"Predicted growth: {predicted_growth[0]:.2f}%")
print(f"\nModel performance on held-out test set:")
print(f"  R^2  = {r2:.3f}")
print(f"  MAE  = {mae:.3f} percentage points of growth")
 
ohe_feature_names = model.named_steps["preprocess"].named_transformers_["muscle_ohe"].get_feature_names_out(["muscle_group"])
all_feature_names = list(ohe_feature_names) + [c for c in feature_cols if c != "muscle_group"]
importances = model.named_steps["regressor"].feature_importances_
importance_df = pd.DataFrame({"feature": all_feature_names, "importance": importances}) \
                   .sort_values("importance", ascending=False)
print("\nFeature importances:")
print(importance_df.to_string(index=False))
 
plt.figure(figsize=(10, 6.5))
time_range = np.linspace(10, 200, 100)
 
for muscle in MUSCLES:
    query_df = pd.DataFrame({
        "muscle_group": muscle,
        "weekly_training_time_min": time_range,
        "sessions_per_week": 3,             
        "avg_intensity_pct_1rm": 75,          
        "training_age_years": 2,              
    })
    predicted_growth = model.predict(query_df)
    plt.plot(time_range, predicted_growth, label=muscle, linewidth=2)
 
plt.xlabel("Weekly Training Time for Muscle Group (minutes of working sets)")
plt.ylabel("Predicted Muscle Growth over 8 Weeks (%)")
plt.title("Predicted Muscle Growth vs. Weekly Training Time\n(at 3 sessions/week, 75% 1RM, 2 yrs training age)")
plt.legend(title="Muscle Group", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("muscle_growth_vs_time.png", dpi=150)
print("\nSaved plot to muscle_growth_vs_time.png")


st.title("Muscle Growth Prediction App")

my_input = pd.DataFrame({
    "muscle_group": [st.selectbox("Select muscle group:", ["Chest", "Back", "Legs", "Shoulders"])],
    "weekly_training_time_min": [st.number_input("Weekly training time (minutes):", min_value=10, max_value=200, value=60)],
    "sessions_per_week": [st.number_input("Sessions per week:", min_value=1, max_value=10, value=3)],
    "avg_intensity_pct_1rm": [st.slider("Average intensity (% 1RM):", min_value=50, max_value=90, value=75)],
    "training_age_years": [st.slider("Training age (years):", min_value=0, max_value=20, value=2)],
})

predicted_growth = model.predict(my_input)
st.write(f"Predicted growth: {predicted_growth[0]:.2f}%")