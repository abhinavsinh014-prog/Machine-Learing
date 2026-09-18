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


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Muscle Growth Predictor",
    page_icon="💪",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

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


# ============================================================
# SYNTHETIC GROWTH FUNCTION
# ============================================================

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

    # Training stimulus
    base_growth = p["Gmax"] * (
        1 - np.exp(-p["k"] * weekly_time)
    )

    # Excess training volume
    excess_volume = np.maximum(
        0,
        weekly_time - p["mrv_time"]
    )

    volume_penalty = (
        p["recovery_penalty"] * excess_volume
    )

    # Frequency
    frequency_factor = (
        1
        + 0.08 * np.minimum(sessions, 4)
        - 0.02 * np.maximum(0, sessions - 4)
    )

    # Intensity
    intensity_factor = (
        1 -
        0.0009 * (intensity - 75) ** 2 / 10
    )

    # Training experience
    experience_factor = (
        1.35 /
        (1 + 0.15 * training_age)
    )

    # Protein
    protein_per_kg = protein / max(body_weight, 1)

    protein_factor = np.clip(
        0.75 + 0.25 * (protein_per_kg / 1.6),
        0.70,
        1.05
    )

    # Calories
    calorie_factor = np.clip(
        0.85 + 0.15 * ((calories - 1800) / 1000),
        0.80,
        1.05
    )

    # Sleep
    sleep_factor = np.clip(
        0.70 + 0.05 * sleep,
        0.70,
        1.05
    )

    # Recovery
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
# GENERATE TRAINING DATA
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

            growth = calculate_growth(
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
            )

            noise = rng.normal(
                0,
                0.12 * growth + 0.12
            )

            observed_growth = max(
                growth + noise,
                0
            )

            rows.append({
                "muscle_group": muscle,
                "weekly_training_time_min": weekly_time,
                "sessions_per_week": sessions,
                "avg_intensity_pct_1rm": intensity,
                "training_age_years": training_age,
                "body_weight_kg": body_weight,
                "daily_calories": calories,
                "daily_protein_g": protein,
                "sleep_hours": sleep,
                "recovery_score": recovery,
                "muscle_growth_pct_8wk": observed_growth
            })

    return pd.DataFrame(rows)


# ============================================================
# TRAIN MODEL
# ============================================================

df = generate_dataset()

features = [
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

target = "muscle_growth_pct_8wk"

X = df[features]
y = df[target]

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

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=14,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        )
    ])

    model.fit(X_train, y_train)

    return model


model = train_model(
    X_train,
    y_train
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

predictions = model.predict(X_test)

r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)


# ============================================================
# HEADER
# ============================================================

st.title("💪 AI Muscle Growth Predictor")

st.write(
    "Enter your personal training, nutrition and recovery "
    "information to generate an educational 8-week prediction."
)

st.divider()


# ============================================================
# USER INFORMATION
# ============================================================

st.header("👤 Your Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=80,
        value=20,
        key="user_age"
    )

with col2:

    body_weight = st.number_input(
        "Body Weight (kg)",
        min_value=35.0,
        max_value=200.0,
        value=65.0,
        step=0.5,
        key="user_weight"
    )

with col3:

    height = st.number_input(
        "Height (cm)",
        min_value=130.0,
        max_value=220.0,
        value=170.0,
        step=0.5,
        key="user_height"
    )


# ============================================================
# TRAINING INFORMATION
# ============================================================

st.header("🏋️ Training Information")

col1, col2 = st.columns(2)

with col1:

    muscle = st.selectbox(
        "Which muscle do you want to analyze?",
        MUSCLES,
        key="user_muscle"
    )

    weekly_time = st.number_input(
        "Weekly training time for this muscle (minutes)",
        min_value=10,
        max_value=300,
        value=60,
        step=5,
        key="user_training_time"
    )

    sessions = st.number_input(
        "How many times do you train this muscle per week?",
        min_value=1,
        max_value=7,
        value=3,
        step=1,
        key="user_sessions"
    )


with col2:

    intensity = st.slider(
        "Average training intensity (% 1RM)",
        min_value=50,
        max_value=100,
        value=75,
        key="user_intensity"
    )

    training_age = st.number_input(
        "How many years have you been training?",
        min_value=0.0,
        max_value=30.0,
        value=1.0,
        step=0.5,
        key="user_training_age"
    )


# ============================================================
# NUTRITION
# ============================================================

st.header("🍗 Nutrition")

col1, col2 = st.columns(2)

with col1:

    calories = st.number_input(
        "Average daily calories",
        min_value=1000,
        max_value=6000,
        value=2500,
        step=50,
        key="user_calories"
    )


with col2:

    protein = st.number_input(
        "Average daily protein (grams)",
        min_value=30.0,
        max_value=400.0,
        value=120.0,
        step=5.0,
        key="user_protein"
    )


# ============================================================
# RECOVERY
# ============================================================

st.header("😴 Recovery")

col1, col2 = st.columns(2)

with col1:

    sleep = st.slider(
        "Average sleep per night (hours)",
        min_value=3.0,
        max_value=12.0,
        value=7.0,
        step=0.5,
        key="user_sleep"
    )


with col2:

    recovery = st.slider(
        "How would you rate your recovery?",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        key="user_recovery"
    )

    st.caption(
        "1 = very poor recovery | 10 = excellent recovery"
    )


# ============================================================
# CALCULATIONS
# ============================================================

protein_per_kg = protein / body_weight

bmi = body_weight / (
    (height / 100) ** 2
)

st.divider()

predict_button = st.button(
    "🚀 Predict My Muscle Growth",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    user_data = pd.DataFrame({

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


    predicted_growth = model.predict(
        user_data
    )[0]


    # ========================================================
    # RESULT
    # ========================================================

    st.success(
        "Prediction generated successfully!"
    )

    st.header("🎯 Your Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Predicted Growth",
            f"{predicted_growth:.2f}%"
        )

    with col2:

        st.metric(
            "Protein / kg",
            f"{protein_per_kg:.2f} g"
        )

    with col3:

        st.metric(
            "BMI",
            f"{bmi:.1f}"
        )

    with col4:

        st.metric(
            "Training Frequency",
            f"{sessions}× / week"
        )

