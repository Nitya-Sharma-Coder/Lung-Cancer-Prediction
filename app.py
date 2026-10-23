import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🫁", layout="wide")

MODEL_PATH = Path(__file__).parent / "best_lung_cancer_model.pkl"

st.title("🫁 Lung Cancer Prediction")
st.caption("Educational machine-learning project using survey-based data")
st.warning(
    "This tool is for educational demonstration only. It is not a medical diagnosis, "
    "does not estimate your actual cancer risk, and must not replace a qualified clinician."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Copy best_lung_cancer_model.pkl into this folder.")
    st.info("Download the model from Google Colab after running the model-saving cells.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Enter survey details")
st.write("Choose the values below, then click **Run prediction**.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["F", "M"])
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        yellow_fingers = st.selectbox("Yellow fingers", ["No", "Yes"])
        anxiety = st.selectbox("Anxiety", ["No", "Yes"])
        peer_pressure = st.selectbox("Peer pressure", ["No", "Yes"])
        chronic_disease = st.selectbox("Chronic disease", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])

    with col2:
        allergy = st.selectbox("Allergy", ["No", "Yes"])
        wheezing = st.selectbox("Wheezing", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol consuming", ["No", "Yes"])
        coughing = st.selectbox("Coughing", ["No", "Yes"])
        shortness_breath = st.selectbox("Shortness of breath", ["No", "Yes"])
        swallowing_difficulty = st.selectbox("Swallowing difficulty", ["No", "Yes"])
        chest_pain = st.selectbox("Chest pain", ["No", "Yes"])

    submitted = st.form_submit_button("Run prediction", type="primary")

if submitted:
    # The original dataset encodes binary survey responses as 1/2.
    # Mapping is consistent with the common Kaggle dataset: 1 = No, 2 = Yes.
    no_yes = {"No": 1, "Yes": 2}
    input_data = pd.DataFrame([{
        "GENDER": gender,
        "AGE": int(age),
        "SMOKING": no_yes[smoking],
        "YELLOW_FINGERS": no_yes[yellow_fingers],
        "ANXIETY": no_yes[anxiety],
        "PEER_PRESSURE": no_yes[peer_pressure],
        "CHRONIC DISEASE": no_yes[chronic_disease],
        "FATIGUE": no_yes[fatigue],
        "ALLERGY": no_yes[allergy],
        "WHEEZING": no_yes[wheezing],
        "ALCOHOL CONSUMING": no_yes[alcohol],
        "COUGHING": no_yes[coughing],
        "SHORTNESS OF BREATH": no_yes[shortness_breath],
        "SWALLOWING DIFFICULTY": no_yes[swallowing_difficulty],
        "CHEST PAIN": no_yes[chest_pain],
    }])

    try:
        prediction = int(model.predict(input_data)[0])
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_data)[0][1])
        else:
            probability = None

        st.subheader("Model output")
        if prediction == 1:
            st.error("Survey pattern classified as: YES")
        else:
            st.success("Survey pattern classified as: NO")

        if probability is not None:
            st.metric("Model-estimated YES score", f"{probability * 100:.1f}%")
            st.caption(
                "This is the model's output score for the YES class, not a medically "
                "validated probability or an individual's cancer risk."
            )

        st.subheader("Input summary")
        st.dataframe(input_data, use_container_width=True)

        st.subheader("Important next step")
        st.write(
            "A survey-based prediction cannot confirm or rule out lung cancer. "
            "For symptoms or health concerns, consult a qualified healthcare professional."
        )
    except Exception as exc:
        st.error("Prediction could not be completed. Check that the saved model matches the dataset columns.")
        st.code(str(exc))
