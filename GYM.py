import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="AI Muscle Growth Predictor",
    page_icon="💪",
    layout="wide"
)

RANDOM_SEED = 42

MUSCLES = [
    "Biceps",
    "Triceps",
    "Shoulders",
    "Chest",
    "Back",
    "Quads",
    "Hamstrings",
    "Calves"
]


MUSCLE_PARAMS = {
    "Biceps": {
        "Gmax": 6.0,
        "k": 0.045,
        "mrv_time": 70,
        "recovery_penalty": 0.020
    },

    "Triceps": {
        "Gmax": 6.5,
        "k": 0.040,
        "mrv_time": 75,
        "recovery_penalty": 0.018
    },

    "Shoulders": {
        "Gmax": 7.0,
        "k": 0.030,
        "mrv_time": 90,
        "recovery_penalty": 0.015
    },

    "Chest": {
        "Gmax": 8.0,
        "k": 0.026,
        "mrv_time": 100,
        "recovery_penalty": 0.014
    },

    "Back": {
        "Gmax": 9.0,
        "k": 0.020,
        "mrv_time": 130,
        "recovery_penalty": 0.010
    },

    "Quads": {
        "Gmax": 10.0,
        "k": 0.017,
        "mrv_time": 150,
        "recovery_penalty": 0.009
    },

    "Hamstrings": {
        "Gmax": 8.5,
        "k": 0.022,
        "mrv_time": 110,
        "recovery_penalty": 0.012
    },

    "Calves": {
        "Gmax": 5.0,
        "k": 0.035,
        "mrv_time": 80,
        "recovery_penalty": 0.022
    }
}

def calculate_growth(
    muscle,
    weekly_time,
    sessions,
    intensity,
    training_age,
    calories,
    protein,
    body_weight,
    sleep,
    recovery
):

    p = MUSCLE_PARAMS[muscle]

    base_growth = p["Gmax"] * (
        1 - np.exp(-p["k"] * weekly_time)
    )

    excess_volume = np.maximum(
        0,
        weekly_time - p["mrv_time"]
    )

    volume_penalty = (
        p["recovery_penalty"] * excess_volume
    )

    frequency_factor = (
        1
        + 0.08 * np.minimum(sessions, 4)
        - 0.02 * np.maximum(0, sessions - 4)
    )

    intensity_factor = (
        1
        - 0.0009 * (intensity - 75) ** 2 / 10
    )

    experience_factor = (
        1.35 / (1 + 0.15 * training_age)
    )

    protein_per_kg = protein / max(body_weight, 1)

    protein_factor = np.clip(
        0.75 + 0.25 * (protein_per_kg / 1.6),
        0.70,
        1.05
    )

    calorie_factor = np.clip(
        0.85 + 0.15 * ((calories - 1800) / 1000),
        0.80,
        1.05
    )

    sleep_factor = np.clip(
        0.70 + 0.05 * sleep,
        0.70,
        1.05
    )

    recovery_factor = np.clip(
        recovery / 10,
        0.50,
        1.00
    )

    growth = (
        (base_growth - volume_penalty)
        * frequency_factor
        * intensity_factor
        * experience_factor
        * protein_factor
        * calorie_factor
        * sleep_factor
        * recovery_factor
    )

    return np.maximum(growth, 0)


# ============================================================
# DATASET GENERATOR
# ============================================================

@st.cache_data
def generate_dataset(samples_per_muscle=1000):

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    for muscle in MUSCLES:

        for _ in range(samples_per_muscle):

            weekly_time = rng.uniform(10, 220)

            sessions = rng.integers(1, 7)

            intensity = rng.uniform(55, 95)

            training_age = rng.uniform(0, 15)

            body_weight = rng.uniform(45, 120)

            protein_per_kg = rng.uniform(0.7, 2.5)

            protein = body_weight * protein_per_kg

            calories = rng.uniform(1600, 3800)

            sleep = rng.uniform(4.5, 10)

            recovery = rng.uniform(4, 10)

            expected_growth = calculate_growth(
                muscle=muscle,
                weekly_time=weekly_time,
                sessions=sessions,
                intensity=intensity,
                training_age=training_age,
                calories=calories,
                protein=protein,
                body_weight=body_weight,
                sleep=sleep,
                recovery=recovery
            )

            # Realistic random variation
            noise = rng.normal(
                0,
                0.12 * expected_growth + 0.12
            )

            observed_growth = max(
                expected_growth + noise,
                0
            )

            rows.append({
                "muscle_group": muscle,
                "weekly_training_time_min": round(
                    weekly_time, 1
                ),
                "sessions_per_week": int(sessions),
                "avg_intensity_pct_1rm": round(
                    intensity, 1
                ),
                "training_age_years": round(
                    training_age, 1
                ),
                "body_weight_kg": round(
                    body_weight, 1
                ),
                "daily_calories": round(
                    calories
                ),
                "daily_protein_g": round(
                    protein, 1
                ),
                "sleep_hours": round(
                    sleep, 1
                ),
                "recovery_score": round(
                    recovery, 1
                ),
                "muscle_growth_pct_8wk": round(
                    observed_growth, 3
                )
            })

    return pd.DataFrame(rows)

df = generate_dataset()

feature_cols = [
    "muscle_group",
    "weekly_training_time_min",
    "sessions_per_week",
    "avg_intensity_pct_1rm",
    "training_age_years",
    "body_weight_kg",
    "daily_calories",
    "daily_protein_g",
    "sleep_hours",
    "recovery_score"
]

target_col = "muscle_growth_pct_8wk"


X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "muscle",
            OneHotEncoder(handle_unknown="ignore"),
            ["muscle_group"]
        )
    ],
    remainder="passthrough"
)

@st.cache_resource
def train_model(X_train, y_train):

    model = Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor
            ),

            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=RANDOM_SEED,
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    return model


model = train_model(
    X_train,
    y_train
)
y_pred = model.predict(X_test)

r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)


st.title("💪 AI Muscle Growth Predictor")

st.markdown(
    """
### Personalized training & recovery analysis

Adjust your training, nutrition and recovery variables
to generate an **educational 8-week muscle-growth simulation**.
"""
)

st.sidebar.header("🏋️ Training")

muscle = st.sidebar.selectbox(
    "Muscle Group",
    MUSCLES,
    key="muscle_selector"
)

weekly_time = st.sidebar.slider(
    "Weekly Training Time (min)",
    10,
    220,
    60,
    5,
    key="weekly_training_time"
)

sessions = st.sidebar.slider(
    "Sessions Per Week",
    1,
    6,
    3,
    key="sessions_per_week"
)

intensity = st.sidebar.slider(
    "Average Intensity (% 1RM)",
    55,
    95,
    75,
    key="intensity"
)

training_age = st.sidebar.slider(
    "Training Age (years)",
    0.0,
    15.0,
    2.0,
    0.5,
    key="training_age"
)

st.sidebar.header("🍗 Nutrition")

body_weight = st.sidebar.number_input(
    "Body Weight (kg)",
    min_value=35.0,
    max_value=180.0,
    value=62.0,
    step=0.5,
    key="body_weight"
)

calories = st.sidebar.number_input(
    "Daily Calories",
    min_value=1200,
    max_value=6000,
    value=2500,
    step=50,
    key="daily_calories"
)

protein = st.sidebar.number_input(
    "Daily Protein (g)",
    min_value=30.0,
    max_value=350.0,
    value=120.0,
    step=5.0,
    key="daily_protein"
)

st.sidebar.header("😴 Recovery")

sleep = st.sidebar.slider(
    "Average Sleep (hours)",
    4.0,
    10.0,
    7.0,
    0.5,
    key="sleep"
)

recovery = st.sidebar.slider(
    "Recovery Score",
    1.0,
    10.0,
    8.0,
    0.5,
    key="recovery"
)

user_input = pd.DataFrame({

    "muscle_group": [muscle],

    "weekly_training_time_min": [
        weekly_time
    ],

    "sessions_per_week": [
        sessions
    ],

    "avg_intensity_pct_1rm": [
        intensity
    ],

    "training_age_years": [
        training_age
    ],

    "body_weight_kg": [
        body_weight
    ],

    "daily_calories": [
        calories
    ],

    "daily_protein_g": [
        protein
    ],

    "sleep_hours": [
        sleep
    ],

    "recovery_score": [
        recovery
    ]
})

prediction = model.predict(
    user_input
)[0]


protein_per_kg = protein / body_weight


st.subheader("🎯 Your Prediction")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Predicted 8-Week Growth",
        f"{prediction:.2f}%"
    )

with col2:

    st.metric(
        "Protein / kg",
        f"{protein_per_kg:.2f} g"
    )

with col3:

    st.metric(
        "Training Time",
        f"{weekly_time} min"
    )

with col4:

    st.metric(
        "Sleep",
        f"{sleep:.1f} h"
    )


st.subheader("🧠 AI Analysis")

recommendations = []


if protein_per_kg < 1.2:

    recommendations.append(
        "🍗 Your protein intake is relatively low for a muscle-building simulation."
    )

elif protein_per_kg >= 1.6:

    recommendations.append(
        "🍗 Your protein intake is within a high-protein range."
    )

else:

    recommendations.append(
        "🍗 Your protein intake is moderate."
    )


if sleep < 7:

    recommendations.append(
        "😴 Sleep is below 7 hours. Recovery may be an important limitation."
    )

elif sleep >= 8:

    recommendations.append(
        "😴 Your sleep input indicates strong recovery conditions."
    )

else:

    recommendations.append(
        "😴 Your sleep input is around the commonly recommended range."
    )


if weekly_time > MUSCLE_PARAMS[muscle]["mrv_time"]:

    recommendations.append(
        "⚠️ Your simulated weekly training time is above this muscle's "
        "configured recovery threshold."
    )

else:

    recommendations.append(
        "💪 Your training time is below the simulated recovery threshold."
    )

if sessions > 5:

    recommendations.append(
        "🔄 High training frequency may increase recovery demands."
    )


if recovery < 6:

    recommendations.append(
        "🛌 Your recovery score is low. Consider reducing training stress "
        "if fatigue is accumulating."
    )


for recommendation in recommendations:

    st.info(recommendation)

st.subheader("📈 Growth vs Weekly Training Time")

time_range = np.linspace(
    10,
    220,
    100
)

curve_df = pd.DataFrame({

    "muscle_group": [muscle] * 100,

    "weekly_training_time_min":
        time_range,

    "sessions_per_week":
        [sessions] * 100,

    "avg_intensity_pct_1rm":
        [intensity] * 100,

    "training_age_years":
        [training_age] * 100,

    "body_weight_kg":
        [body_weight] * 100,

    "daily_calories":
        [calories] * 100,

    "daily_protein_g":
        [protein] * 100,

    "sleep_hours":
        [sleep] * 100,

    "recovery_score":
        [recovery] * 100
})


curve_prediction = model.predict(
    curve_df
)


fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    time_range,
    curve_prediction,
    linewidth=2
)

ax.axvline(
    MUSCLE_PARAMS[muscle]["mrv_time"],
    linestyle="--",
    label="Simulated Recovery Threshold"
)

ax.set_xlabel(
    "Weekly Training Time (minutes)"
)

ax.set_ylabel(
    "Predicted 8-Week Growth (%)"
)

ax.set_title(
    f"{muscle}: Training Time vs Simulated Growth"
)

ax.grid(
    alpha=0.3
)

ax.legend()

st.pyplot(fig)

st.subheader("🔍 What Influenced the Model?")

ohe = model.named_steps[
    "preprocess"
].named_transformers_["muscle"]


ohe_names = list(
    ohe.get_feature_names_out(
        ["muscle_group"]
    )
)


numeric_features = [
    c for c in feature_cols
    if c != "muscle_group"
]


all_features = (
    ohe_names +
    numeric_features
)


importances = (
    model.named_steps[
        "regressor"
    ].feature_importances_
)


importance_df = pd.DataFrame({

    "Feature": all_features,

    "Importance": importances

}).sort_values(
    "Importance",
    ascending=False
)


