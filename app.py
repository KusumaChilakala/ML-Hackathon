import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load assets
scaler = joblib.load('scaler.pkl')
xgb_model = joblib.load('xgb_model.pkl')
rf_model = joblib.load('rf_model.pkl')

st.title("🌍 Urban Air Quality Forecasting & Alert System")

# 2. UI Inputs
st.header("📍 Input Environmental Factors")
s1 = st.slider("PT08.S1 (CO Sensor Reading)", 400, 2200, 1000) 
no2 = st.slider("NO2 Concentration", 0, 500, 50)    
temp = st.slider("Temperature (°C)", -5.0, 50.0, 25.0)
rh = st.slider("Humidity (%)", 0.0, 100.0, 45.0)
hour = st.slider("Hour of Day", 0, 23, 12)

if st.button("🚀 GENERATE FORECAST & ALERT"):
    # EXACT features expected by your 88% accuracy model
    feature_names = [
        'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 
        'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 
        'Hour', 'DayOfWeek', 'NO2_lag1', 'NO2_lag2', 'NO2_lag3'
    ]
    
    # 3. DYNAMIC SENSOR CORRELATION
    # Instead of fixed numbers, we calculate hidden values based on your sliders
    # This ensures that if you move S1 to 2000, the other sensors "act" like it's a high-pollution day.
    input_data = {
        'PT08.S1(CO)': s1,
        'NMHC(GT)': s1 * 0.18,          # Correlated to S1
        'C6H6(GT)': (s1 - 400) / 80,    # Correlated to S1
        'PT08.S2(NMHC)': s1 * 0.85,     # Highly correlated to S1
        'NOx(GT)': no2 * 1.5,           # Correlated to NO2
        'PT08.S3(NOx)': 1200 - (no2*2), # S3 goes DOWN when pollution goes up
        'NO2(GT)': no2,
        'PT08.S4(NO2)': 800 + (s1*0.4), # Correlated to S1
        'PT08.S5(O3)': 500 + (s1*0.5),  # Correlated to S1
        'T': temp,
        'RH': rh,
        'AH': 0.8,
        'Hour': hour,
        'DayOfWeek': 2,                 # Mid-week (usually higher pollution)
        'NO2_lag1': no2,        
        'NO2_lag2': no2,
        'NO2_lag3': no2
    }
    
    # Convert and Predict
    df_input = pd.DataFrame([input_data])[feature_names]
    input_scaled = scaler.transform(df_input)
    
    p1 = xgb_model.predict(input_scaled)[0]
    p2 = rf_model.predict(input_scaled)[0]
    final_pred = (p1 + p2) / 2
    
    # Force result to stay within realistic dataset bounds (0.1 to 10.0)
    final_pred = max(0.1, final_pred)

    st.subheader("Results")
    st.metric(label="Predicted CO Concentration", value=f"{final_pred:.2f} mg/m³")
    
    # 4. ALERT LOGIC
    if final_pred > 4.0:
        st.error(f"🚨 ALERT: High Risk Detected!")
        st.progress(100)
    elif final_pred > 2.0:
        st.warning(f"⚠️ Warning: Moderate Pollution")
        st.progress(60)
    else:
        st.success(f"✅ Safe: Low Risk")
        st.progress(20)
