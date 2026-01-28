import streamlit as st
from streamlit_shap import st_shap
import shap
import tensorflow as tf
import joblib
import numpy as np
import os

# ---------------------------------------------------------
# 1. PAGE SETUP & STYLING
# ---------------------------------------------------------
st.set_page_config(page_title="Sentinel-AI Dashboard", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. ASSET LOADING (Absolute Paths for Hugging Face)
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    
    # Load your exported files
    model = tf.keras.models.load_model(os.path.join(base_path, 'sentinel_ai_model.keras'))
    le = joblib.load(os.path.join(base_path, 'label_encoder.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    
    # Create a small dummy background for SHAP (43 features)
    # This is used as a reference to explain why a prediction is high or low
    background = np.zeros((20, 43)) 
    return model, le, scaler, background

model, le, scaler, background = load_assets()

# ---------------------------------------------------------
# 3. UI LAYOUT & INPUTS
# ---------------------------------------------------------
st.title("🛡️ Sentinel-AI: Intrusion Detection System")
st.write("Real-time network traffic analysis using Deep Learning.")
st.markdown("---")

# Feature names (standard 43 features for UNSW-NB15)
feature_names = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
    'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 
    'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'res_bdy_len', 
    'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
    'ct_dst_src_ltm', 'is_sm_ips_ports', 'ct_ftp_cmd_2', 'ct_flw_http_mthd_2', 
    'ct_src_ltm_2', 'ct_srv_dst_2'
]

# Sidebar for metadata
st.sidebar.header("System Status")
st.sidebar.success("✅ Neural Network Loaded")
st.sidebar.success("✅ Scaler Active")

# Main Input Grid
col1, col2, col3 = st.columns(3)

with col1:
    dur = st.number_input("Duration", value=0.12)
    sbytes = st.number_input("Source Bytes", value=250)
    dbytes = st.number_input("Dest Bytes", value=150)
    rate = st.number_input("Packet Rate", value=75.0)

with col2:
    sttl = st.slider("Source TTL", 0, 255, 31)
    dttl = st.slider("Dest TTL", 0, 255, 29)
    sload = st.number_input("Source Load", value=15000.0)
    dload = st.number_input("Dest Load", value=8000.0)

with col3:
    spkts = st.number_input("Source Packets", value=6)
    dpkts = st.number_input("Dest Packets", value=4)
    swin = st.number_input("Source Window", value=255)
    dwin = st.number_input("Dest Window", value=255)

# ---------------------------------------------------------
# 4. PREDICTION & EXPLAINABILITY
# ---------------------------------------------------------
if st.button("🚀 ANALYZE NETWORK TRAFFIC"):
    # 4.1 Construct the 43-feature input array
    input_data = np.zeros((1, 43))
    # Map the UI inputs to correct feature positions
    input_data[0, 0] = dur       # dur
    input_data[0, 1] = spkts     # spkts
    input_data[0, 2] = dpkts     # dpkts
    input_data[0, 3] = sbytes    # sbytes
    input_data[0, 4] = dbytes    # dbytes
    input_data[0, 5] = rate      # rate
    input_data[0, 6] = sttl      # sttl
    input_data[0, 7] = dttl      # dttl
    input_data[0, 8] = sload     # sload
    input_data[0, 9] = dload     # dload
    input_data[0, 16] = swin     # swin (position 16)
    input_data[0, 19] = dwin     # dwin (position 19)
    
    # 4.2 Scale and Predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input, verbose=0)
    class_idx = np.argmax(prediction)
    threat_type = le.inverse_transform([class_idx])[0]
    confidence = np.max(prediction) * 100

    # 4.3 Display Result
    st.markdown("---")
    if threat_type == 'Normal':
        st.success(f"### 🟢 STATUS: SECURE")
        st.write(f"The traffic pattern appears safe. Confidence: **{confidence:.2f}%**")
    else:
        st.error(f"### 🔴 ALERT: {threat_type.upper()} DETECTED!")
        st.write(f"The AI has flagged this connection as a threat. Confidence: **{confidence:.2f}%**")

    # 4.4 SHAP Explanation (The "Explainable AI" section)
    st.subheader("🔍 Decision Logic (SHAP Force Plot)")
    
    try:
        with st.spinner("Calculating feature contributions..."):
            # KernelExplainer is used for Neural Networks
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(scaled_input)
            
            # --- FIXED TYPE HANDLING ---
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class output: list of arrays
                sv = shap_values[class_idx][0]  # Get values for predicted class
                ev = explainer.expected_value[class_idx]
            else:
                # Single array output
                if len(shap_values.shape) == 3:
                    # Shape: (samples, features, classes)
                    sv = shap_values[0, :, class_idx]
                    ev = explainer.expected_value[class_idx]
                else:
                    # Shape: (samples, features)
                    sv = shap_values[0]
                    if isinstance(explainer.expected_value, np.ndarray):
                        ev = explainer.expected_value[0]
                    else:
                        ev = explainer.expected_value
            
            # Ensure everything is the right type
            sv = np.array(sv).flatten()
            ev = float(ev)
            features_row = input_data[0].flatten()
            # ---------------------------------------

            # Render the interactive plot
            st_shap(shap.force_plot(
                ev, 
                sv, 
                features_row, 
                feature_names=feature_names
            ), height=200, width=1000)

        st.info("💡 **How to read this:** Red features push the AI toward detecting an attack, while blue features push it toward 'Normal'.")
        
    except Exception as e:
        st.warning(f"⚠️ SHAP visualization unavailable: {str(e)}")
        st.info("Prediction completed successfully, but the explainability plot encountered an issue.")