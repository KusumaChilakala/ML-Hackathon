# 🌍 Urban Air Quality Forecasting & Alert System

An end-to-end Machine Learning solution designed to predict Carbon Monoxide (CO) concentrations and provide real-time health alerts based on environmental sensor data.

---

## 🚀 Live Demonstration
**Current Status:** Live via Ngrok (active during demonstration periods).  
**Link:** [View Live Dashboard](https://daisy-unmoldered-mirta.ngrok-free.dev/)

---

## 📊 Project Highlights
* **Model Accuracy ($R^2$ Score):** 88%
* **Algorithm:** Ensemble Learning (XGBoost + Random Forest Regressor)
* **Dataset:** UCI Machine Learning Repository - Air Quality Data
* **Key Techniques:** Time-series Lag Features, Linear Interpolation, and Cyclic Time Encoding

---

## 🛠️ Development Stages

### 1. Data Preprocessing
* Cleaned sensor errors (replaced -200 with NaN).
* Used **Linear Interpolation** to maintain time-series continuity.
* Normalized data using `StandardScaler` to ensure feature parity.



### 2. Feature Engineering
* Created **Lag Features** (Previous 1-3 hours of sensor data) to capture temporal trends.
* Engineered **Cyclic Time Features** (Sine/Cosine transformations) for hours and days.
* Linked sensor correlations (CO, $NO_2$, and Benzene) for realistic dashboard simulation.

### 3. Machine Learning Pipeline
* Trained and evaluated multiple models: **Linear Regression, SVR, Random Forest, and XGBoost**.
* Developed an **Ensemble Model** by averaging the top two performing regressors (RF + XGB) to achieve the final 88% accuracy.



### 4. Deployment
* Built an interactive **Streamlit** UI allowing users to manipulate sensor readings.
* Integrated an **Alert System** that triggers "High Risk" warnings when CO levels exceed **4.5 $mg/m^3$**.
* Deployed via **Ngrok** to expose the local Streamlit server to a public URL.

---

## 📂 File Structure
* `project2.ipynb`: Full research, data visualization, and training code.
* `app.py`: Streamlit application script for the user interface.
* `scaler.pkl`: Saved StandardScaler object.
* `xgb_model.pkl` & `rf_model.pkl`: Pre-trained models for the ensemble engine.
* `requirements.txt`: List of dependencies for reproducibility.

---

## 🔧 Installation & Usage
To run this project locally:

1.  **Clone the repository:** `git clone https://github.com/KusumaChilakala/ML-Hackathon.git`
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Launch the dashboard:** `streamlit run app.py`
