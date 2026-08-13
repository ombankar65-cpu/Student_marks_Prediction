import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load the trained Naive Bayes model
@st.cache_resource
def load_model():
    with open('naive_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file `naive_model.pkl`: {e}")
    st.stop()

# Title and Description
st.title("🎓 Student Academic Performance Predictor")
st.markdown("""
Predict whether a student is likely to **Pass** or **Fail** based on demographic data, 
attendance, study habits, and exam scores using the trained Naive Bayes Classifier.
---
""")

# Input Form
with st.form("prediction_form"):
    st.subheader("📋 Enter Student Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Demographics")
        age = st.number_input("Age", min_value=15, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", options=[0, 1], help="0: Female, 1: Male (or encoded index)")
        department = st.selectbox("Department", options=[0, 1, 2, 3], help="Categorical index for department")

    with col2:
        st.markdown("### 📚 Habit Metrics")
        study_hours = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
        attendance = st.slider("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        assignments = st.number_input("Assignments Completed", min_value=0, max_value=50, value=10, step=1)

    with col3:
        st.markdown("### 📝 Test Scores")
        midterm_score = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        final_score = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=78.0, step=1.0)

    st.markdown("---")
    submit_button = st.form_submit_button(label="🚀 Predict Result", use_container_width=True)

# Prediction Logic
if submit_button:
    # Construct input dataframe matching exact feature names expected by sklearn
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Department': department,
        'Study_Hours_Per_Day': study_hours,
        'Attendance_Percentage': attendance,
        'Assignments_Completed': assignments,
        'Midterm_Score': midterm_score,
        'Final_Score': final_score
    }])

    try:
        # Perform prediction and get probability scores
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Get class labels from the model
        classes = list(model.classes_)
        pass_idx = classes.index("Pass") if "Pass" in classes else 1
        fail_idx = classes.index("Fail") if "Fail" in classes else 0

        st.subheader("📊 Prediction Results")
        
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == "Pass":
                st.success(f"### Result: **PASS** 🎉")
            else:
                st.error(f"### Result: **FAIL** ⚠️")

        with res_col2:
            st.metric(label="Pass Probability", value=f"{probabilities[pass_idx] * 100:.2f}%")
            st.metric(label="Fail Probability", value=f"{probabilities[fail_idx] * 100:.2f}%")

        # Visual progress bar
        st.progress(float(probabilities[pass_idx]))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
