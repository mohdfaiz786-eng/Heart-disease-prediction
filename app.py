# app_modern.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import io
import plotly.express as px

st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='text-align:center; color:red;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Interactive ML-based prediction using Streamlit</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for dataset and model options
st.sidebar.header("1️⃣ Dataset & Model Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
model_choice = st.sidebar.selectbox("Choose model", ["Logistic Regression", "Random Forest"])
test_size = st.sidebar.slider("Test size (%)", 10, 50, 25)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

EXPECTED_COLS = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

# --- DATA LOADING ---
def make_demo_df(n=200, random_state=42):
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame({
        'age': rng.randint(29, 77, size=n),
        'sex': rng.randint(0,2,size=n),
        'cp': rng.randint(0,4,size=n),
        'trestbps': rng.randint(94,200,size=n),
        'chol': rng.randint(126,564,size=n),
        'fbs': rng.randint(0,2,size=n),
        'restecg': rng.randint(0,3,size=n),
        'thalach': rng.randint(70,202,size=n),
        'exang': rng.randint(0,2,size=n),
        'oldpeak': np.round(rng.uniform(0,6.2,size=n),1),
        'slope': rng.randint(0,3,size=n),
        'ca': rng.randint(0,4,size=n),
        'thal': rng.randint(0,4,size=n),
    })
    logits = 0.03*(df.age-50) + 0.8*df.exang + 0.02*(df.chol-200)/10 - 0.02*(df.thalach-150)/5 + 0.4*df.cp
    probs = 1/(1+np.exp(-logits))
    df['target'] = (probs > 0.5).astype(int)
    return df

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV loaded successfully!")
    except:
        st.sidebar.error("Failed to read CSV. Using demo dataset.")
        df = make_demo_df()
else:
    st.sidebar.info("No file uploaded — using demo dataset.")
    df = make_demo_df()

# Preview
with st.expander("Preview Dataset"):
    st.dataframe(df.head(10))

# Check columns
missing = [c for c in EXPECTED_COLS if c not in df.columns]
if missing:
    st.warning(f"Missing columns: {missing}. Using demo dataset or 'target' must exist.")
    if 'target' not in df.columns:
        st.error("'target' missing. Cannot train.")
        st.stop()

X = df.drop(columns=['target'])
y = df['target']

# --- TRAIN MODEL ---
if st.button("Train Model"):
    st.info("Training model... ⏳")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state, stratify=y if len(np.unique(y))>1 else None
    )

    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000, class_weight='balanced') if model_choice=="Logistic Regression" else RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced')
    pipeline = Pipeline([("scaler", scaler), ("clf", clf)])
    pipeline.fit(X_train, y_train)
    
    # Metrics
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline, "predict_proba") else np.zeros_like(y_pred)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_proba)
    }
    col1, col2, col3, col4, col5 = st.columns(5)
    for c, (name, val) in zip([col1, col2, col3, col4, col5], metrics.items()):
        c.metric(label=name, value=f"{val:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="reds")
    st.plotly_chart(fig_cm)

    # Feature importance
    if model_choice=="Random Forest":
        try:
            importances = pipeline.named_steps['clf'].feature_importances_
            df_imp = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(by="importance", ascending=False)
            st.subheader("Top 10 Feature Importances")
            fig_imp = px.bar(df_imp.head(10), x="feature", y="importance", color="importance", color_continuous_scale="blues")
            st.plotly_chart(fig_imp)
        except:
            st.write("Cannot show feature importance.")

    # Save model
    joblib.dump(pipeline, "heart_model.joblib")
    st.success("Model trained and saved as `heart_model.joblib`.")
    st.download_button("Download Model", data=open("heart_model.joblib","rb").read(), file_name="heart_model.joblib")

# --- SINGLE PREDICTION ---
st.header("2️⃣ Predict Single Case")
with st.form("predict_form"):
    cols = st.columns(2)
    input_data = {}
    for i, col in enumerate(X.columns):
        with cols[i%2]:
            input_data[col] = st.number_input(col, value=float(df[col].median()))
    submit = st.form_submit_button("Predict")
    if submit:
        input_df = pd.DataFrame([input_data])
        try:
            model = joblib.load("heart_model.joblib")
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            st.success(f"Prediction: {'❤️ Disease' if pred==1 else '💚 No Disease'}")
            if proba is not None:
                st.info(f"Probability of disease: {proba:.2f}")
        except:
            st.error("No trained model found. Train a model first.")

# --- DATA SUMMARY ---
with st.expander("Data Summary & Target Distribution"):
    st.write(df.describe())
    st.write("Target class distribution:")
    fig_dist = px.histogram(df, x="target", color="target", text_auto=True)
    st.plotly_chart(fig_dist)

st.markdown("---")
st.caption("⚠️ This is a demo. Not for clinical use.")
