import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# --- Configuration ---
# Set the title and a favicon for the browser tab
st.set_page_config(page_title="Engine RUL Predictor", page_icon="✈️", layout="wide")

# --- Load Model and Scaler ---
# Use st.cache_resource to load the model and scaler only once, which speeds up the app
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and scaler from disk."""
    try:
        model = load_model('rul_prediction_model.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.error("Please ensure 'rul_prediction_model.h5' and 'scaler.pkl' are in the same folder as app.py.")
        return None, None

model, scaler = load_assets()

# --- Main App Interface ---
st.title("✈️ Predictive Maintenance: Aircraft Engine RUL Predictor")
st.write(
    "This application uses a trained LSTM deep learning model to predict the Remaining Useful Life (RUL) "
    "of an aircraft turbofan engine. Enter the engine's current operational settings and sensor readings "
    "in the sidebar to get a prediction."
)

st.markdown("---")

# --- Sidebar for User Input ---
st.sidebar.header("Engine Input Parameters")
st.sidebar.write("Please provide the most recent sensor readings.")

# Define the user-friendly names and the corresponding internal names from our training
# This mapping makes the app usable for a non-technical user
feature_mapping = {
    'Altitude Setting': 'setting_1',
    'Speed (Mach) Setting': 'setting_2',
    'Throttle Setting': 'setting_3',
    'Total temperature at fan inlet (T2)': 's_1',
    'Total temperature at LPC outlet (T24)': 's_2',
    'Total temperature at HPC outlet (T30)': 's_3',
    'Total temperature at LPT outlet (T50)': 's_4',
    'Pressure at fan inlet (P2)': 's_5',
    'Total pressure in bypass-duct (P15)': 's_6',
    'Total pressure at HPC outlet (P30)': 's_7',
    'Physical fan speed (Nf)': 's_8',
    'Physical core speed (Nc)': 's_9',
    'Engine pressure ratio (epr)': 's_10',
    'Static pressure at HPC outlet (Ps30)': 's_11',
    'Ratio of fuel flow to Ps30 (phi)': 's_12',
    'Corrected fan speed': 's_13',
    'Corrected core speed': 's_14',
    'Bypass Ratio': 's_15',
    'Burner fuel-air ratio': 's_16',
    'Bleed Enthalpy': 's_17',
    'Demanded fan speed': 's_18',
    'Demanded corrected fan speed': 's_19',
    'HPT coolant bleed': 's_20',
    'LPT coolant bleed': 's_21'
}

# Create a dictionary to hold user inputs
user_inputs = {}

# Create number input fields for each feature in the sidebar
for display_name, internal_name in feature_mapping.items():
    user_inputs[internal_name] = st.sidebar.number_input(display_name, value=0.0, format="%.4f")

# --- Prediction Logic ---
if st.sidebar.button("Predict RUL", type="primary"):
    if model is not None and scaler is not None:
        # 1. Convert user inputs into a DataFrame
        input_df = pd.DataFrame([user_inputs])

        # 2. Scale the user inputs using the loaded scaler
        scaled_input = scaler.transform(input_df)

        # 3. Create the sequence for the LSTM model
        # The model expects a history of 40 cycles. For a single prediction,
        # we create a "dummy" history by repeating the current snapshot 40 times.
        sequence_length = 40
        input_sequence = np.array([scaled_input[0]] * sequence_length)
        input_sequence = input_sequence.reshape(1, sequence_length, len(feature_mapping))

        # 4. Make a prediction
        predicted_rul = model.predict(input_sequence)[0][0]

        # --- Display the Result ---
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Predicted Remaining Useful Life",
                value=f"{predicted_rul:.0f} cycles",
                help="This is the estimated number of flights remaining before maintenance is required."
            )
        
        with col2:
            # Provide a simple health status based on the prediction
            health_status = "Good"
            if predicted_rul < 100:
                health_status = "Warning"
            if predicted_rul < 50:
                health_status = "Critical"
            st.metric(label="Engine Health Status", value=health_status)

        st.success("Prediction successful! See the estimated RUL and health status above.")
        
        # Display the user's input for confirmation
        with st.expander("Show User Input"):
            st.write("The prediction was based on the following input values:")
            st.dataframe(input_df)
    else:
        st.error("Model is not loaded. Cannot make a prediction.")
else:
    st.info("Enter the engine parameters in the sidebar and click 'Predict RUL' to get a result.")
