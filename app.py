import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        # body { background-color: #0f0f0f; }
        # .main { background-color: #0e1117; }

        .disease-card {
            background: linear-gradient(145deg, #1f1f2e, #2a2a3e);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
            margin-bottom: 20px;
        }
        .result-box {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: 700;
            margin-top: 20px;
        }
        .positive {
            background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
            border: 2px solid #ff4b4b;
            color: #ff4b4b;
        }
        .negative {
            background: linear-gradient(135deg, #00c85322, #00c85344);
            border: 2px solid #00c853;
            color: #00c853;
        }
        .stTextInput > div > div > input {
            # background-color: #1f1f2e;
            border: 1px solid #3a3a5c;
            border-radius: 8px;
            color: white;
            padding: 10px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #4e9af1;
            box-shadow: 0 0 0 2px rgba(78,154,241,0.2);
        }
        .stButton > button {
            background: linear-gradient(135deg, #4e9af1, #2563eb);
            color: white;
            font-weight: 700;
            font-size: 16px;
            padding: 12px 40px;
            border: none;
            border-radius: 10px;
            width: 100%;
            transition: all 0.3s ease;
            margin-top: 10px;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #60a8ff, #3b72f5);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(78,154,241,0.4);
        }
        .metric-label {
            color: #888;
            font-size: 12px;
            margin-bottom: 4px;
        }
        h1, h2, h3 {
            color: white !important;
        }
        .info-box {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-left: 4px solid #4e9af1;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
            color: #aaaacc;
            font-size: 14px;
        }
        section[data-testid="stSidebar"] {
            background-color: #0d0d1a;
        }
    </style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
    heart_scaler = pickle.load(open('heart_scaler.sav', 'rb'))
    parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
    parkinsons_scaler = pickle.load(open('parkinsons_scaler.sav', 'rb'))
    return diabetes_model, scaler, heart_disease_model, heart_scaler, parkinsons_model, parkinsons_scaler

diabetes_model, scaler, heart_disease_model, heart_scaler, parkinsons_model, parkinsons_scaler = load_models()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧑‍⚕️ Health Assistant")
    st.markdown("---")
    selected = option_menu(
        'Disease Prediction',
        ['Diabetes', 'Heart Disease', "Parkinson's"],
        icons=['activity', 'heart-pulse', 'person-lines-fill'],
        default_index=0,
        styles={
            "container": {"background-color": "#0d0d1a"},
            "icon": {"color": "#4e9af1", "font-size": "18px"},
            "nav-link": {"color": "#aaaacc", "font-size": "15px"},
            "nav-link-selected": {"background-color": "#1a1a3e", "color": "white"},
        }
    )
    st.markdown("---")
    st.markdown("""
        <div style='color: #666; font-size: 12px; padding: 8px;'>
        ⚠️ This app is for educational purposes only. 
        Always consult a qualified medical professional 
        for actual diagnosis.
        </div>
    """, unsafe_allow_html=True)

# ── Helper function ────────────────────────────────────────────────────────────
def show_result(diagnosis, positive_msg, negative_msg):
    if diagnosis == 1:
        st.markdown(f"""
            <div class="result-box positive">
                ⚠️ {positive_msg}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box negative">
                ✅ {negative_msg}
            </div>
        """, unsafe_allow_html=True)

# ── Diabetes Page ──────────────────────────────────────────────────────────────
if selected == 'Diabetes':
    st.title('🩸 Diabetes Prediction')
    st.markdown("""
        <div class="info-box">
        Enter the patient's medical details below. All fields are required. 
        This model is trained on the Pima Indians Diabetes Dataset.
        </div>
    """, unsafe_allow_html=True)

    with st.form("diabetes_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.text_input('🤰 Pregnancies', placeholder='e.g. 2')
            SkinThickness = st.text_input('🩹 Skin Thickness (mm)', placeholder='e.g. 20')
            DiabetesPedigreeFunction = st.text_input('🧬 Diabetes Pedigree Function', placeholder='e.g. 0.5')
        with col2:
            Glucose = st.text_input('🍬 Glucose Level', placeholder='e.g. 120')
            Insulin = st.text_input('💉 Insulin Level', placeholder='e.g. 80')
            Age = st.text_input('🎂 Age', placeholder='e.g. 30')
        with col3:
            BloodPressure = st.text_input('💓 Blood Pressure (mm Hg)', placeholder='e.g. 70')
            BMI = st.text_input('⚖️ BMI', placeholder='e.g. 25.0')

        submitted = st.form_submit_button("🔍 Run Diabetes Test")

    if submitted:
        try:
            input_data = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            input_array = np.asarray(input_data).reshape(1, -1)
            std_data = scaler.transform(input_array)
            prediction = diabetes_model.predict(std_data)[0]
            show_result(prediction,
                        "The person is likely Diabetic. Please consult a doctor.",
                        "The person is likely not Diabetic. Keep up the healthy lifestyle!")
        except ValueError:
            st.error("⚠️ Please fill in all fields with valid numeric values.")

# ── Heart Disease Page ─────────────────────────────────────────────────────────
if selected == 'Heart Disease':
    st.title('❤️ Heart Disease Prediction')
    st.markdown("""
        <div class="info-box">
        Enter the patient's cardiac metrics below. 
        This model is trained on the Cleveland Heart Disease Dataset.
        </div>
    """, unsafe_allow_html=True)

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.text_input('🎂 Age', placeholder='e.g. 45')
            trestbps = st.text_input('💓 Resting Blood Pressure', placeholder='e.g. 120')
            restecg = st.text_input('📋 Resting ECG Results (0/1/2)', placeholder='e.g. 0')
            oldpeak = st.text_input('📉 ST Depression (Oldpeak)', placeholder='e.g. 1.0')
            thal = st.text_input('🧪 Thal (0/1/2)', placeholder='e.g. 2')
        with col2:
            sex = st.text_input('⚧ Sex (1=Male, 0=Female)', placeholder='e.g. 1')
            chol = st.text_input('🩸 Cholesterol (mg/dl)', placeholder='e.g. 200')
            thalach = st.text_input('💗 Max Heart Rate Achieved', placeholder='e.g. 150')
            slope = st.text_input('📈 Slope of Peak ST (0/1/2)', placeholder='e.g. 1')
        with col3:
            cp = st.text_input('😣 Chest Pain Type (0-3)', placeholder='e.g. 1')
            fbs = st.text_input('🍬 Fasting Blood Sugar > 120? (1=Yes, 0=No)', placeholder='e.g. 0')
            exang = st.text_input('🏃 Exercise Induced Angina (1=Yes, 0=No)', placeholder='e.g. 0')
            ca = st.text_input('🫀 Major Vessels (0-3)', placeholder='e.g. 0')

        submitted = st.form_submit_button("🔍 Run Heart Disease Test")

    if submitted:
        try:
            input_data = [
                float(age), float(sex), float(cp),
                float(trestbps), float(chol), float(fbs),
                float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca),
                float(thal)
            ]
            input_array = np.asarray(input_data).reshape(1, -1)
            std_data = heart_scaler.transform(input_array)
            prediction = heart_disease_model.predict(std_data)[0]
            show_result(prediction,
                        "The person likely has Heart Disease. Please consult a cardiologist.",
                        "The person likely does not have Heart Disease. Stay heart-healthy!")
        except ValueError:
            st.error("⚠️ Please fill in all fields with valid numeric values.")

# ── Parkinsons Page ────────────────────────────────────────────────────────────
if selected == "Parkinson's":
    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("""
        <div class="info-box">
        Enter the patient's vocal biomarker measurements below. 
        This model is trained on the UCI Parkinson's Dataset.
        </div>
    """, unsafe_allow_html=True)

    with st.form("parkinsons_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            fo = st.text_input('MDVP:Fo (Hz)', placeholder='e.g. 119.99')
            Jitter_percent = st.text_input('MDVP:Jitter (%)', placeholder='e.g. 0.003')
            RAP = st.text_input('MDVP:RAP', placeholder='e.g. 0.001')
            Shimmer = st.text_input('MDVP:Shimmer', placeholder='e.g. 0.02')
            APQ3 = st.text_input('Shimmer:APQ3', placeholder='e.g. 0.01')
            NHR = st.text_input('NHR', placeholder='e.g. 0.01')
            RPDE = st.text_input('RPDE', placeholder='e.g. 0.4')
            spread1 = st.text_input('Spread1', placeholder='e.g. -5.6')
        with col2:
            fhi = st.text_input('MDVP:Fhi (Hz)', placeholder='e.g. 157.30')
            Jitter_Abs = st.text_input('MDVP:Jitter (Abs)', placeholder='e.g. 0.00002')
            PPQ = st.text_input('MDVP:PPQ', placeholder='e.g. 0.001')
            Shimmer_dB = st.text_input('MDVP:Shimmer (dB)', placeholder='e.g. 0.2')
            APQ5 = st.text_input('Shimmer:APQ5', placeholder='e.g. 0.01')
            HNR = st.text_input('HNR', placeholder='e.g. 21.0')
            DFA = st.text_input('DFA', placeholder='e.g. 0.7')
            spread2 = st.text_input('Spread2', placeholder='e.g. 0.2')
        with col3:
            flo = st.text_input('MDVP:Flo (Hz)', placeholder='e.g. 74.99')
            DDP = st.text_input('Jitter:DDP', placeholder='e.g. 0.003')
            APQ = st.text_input('MDVP:APQ', placeholder='e.g. 0.01')
            DDA = st.text_input('Shimmer:DDA', placeholder='e.g. 0.03')
            D2 = st.text_input('D2', placeholder='e.g. 2.3')
            PPE = st.text_input('PPE', placeholder='e.g. 0.2')

        submitted = st.form_submit_button("🔍 Run Parkinson's Test")

    if submitted:
        try:
            input_data = [
                float(fo), float(fhi), float(flo),
                float(Jitter_percent), float(Jitter_Abs),
                float(RAP), float(PPQ), float(DDP),
                float(Shimmer), float(Shimmer_dB),
                float(APQ3), float(APQ5), float(APQ),
                float(DDA), float(NHR), float(HNR),
                float(RPDE), float(DFA),
                float(spread1), float(spread2),
                float(D2), float(PPE)
            ]
            input_array = np.asarray(input_data).reshape(1, -1)
            std_data = parkinsons_scaler.transform(input_array)
            prediction = parkinsons_model.predict(std_data)[0]
            show_result(prediction,
                        "The person likely has Parkinson's Disease. Please consult a neurologist.",
                        "The person likely does not have Parkinson's Disease.")
        except ValueError:
            st.error("⚠️ Please fill in all fields with valid numeric values.")