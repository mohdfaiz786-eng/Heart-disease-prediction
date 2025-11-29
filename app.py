# app.py
# ---------------------------------------------
# Heart Disease Prediction using Streamlit
# with separate CSV dataset (heart_disease.csv)
# ---------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease.csv")
    return df

try:
    df = load_data()
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ Dataset not found! Please place 'heart_disease.csv' in the same folder as this app.")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --- Features and target ---
X = df.drop(columns=['target'])
y = df['target']

# --- Model selection ---
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if st.button("🚀 Train Model"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    st.success(f"Model trained successfully! ✅ Accuracy: {acc:.2f}")
    st.write("Confusion Matrix:")
    st.write(cm)
    joblib.dump(model, "heart_model.joblib")
    st.info("💾 Model saved as 'heart_model.joblib'")

# --- Prediction Section ---
st.header("🔍 Predict for a Single Patient")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120mg/dl", [0, 1])
with col2:
    restecg = st.number_input("Resting ECG (0-2)", 0, 2, 1)
    thalach = st.number_input("Max Heart Rate", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.number_input("Slope (0-2)", 0, 2, 1)
    ca = st.number_input("CA (0-3)", 0, 3, 0)
    thal = st.number_input("Thal (0-3)", 0, 3, 1)

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

if st.button("🧠 Predict Heart Disease"):
    try:
        model = joblib.load("heart_model.joblib")
        pred = model.predict(input_data)[0]
        result = "🚨 Likely Heart Disease" if pred == 1 else "💚 No Heart Disease Detected"
        st.success(f"Prediction: {result}")
    except:
        st.warning("Please train the model first!")

st.markdown("---")
st.caption("Demo dataset inspired by UCI Heart Disease dataset. For educational use only.")
