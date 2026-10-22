from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Model, Scaler aur Feature names load karna
with open('breast cancer model.pkl', 'rb') as f:
    saved_data = pickle.load(f)

model = saved_data['model']
scaler = saved_data['scaler']
feature_names = saved_data['features']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form se direct numerical/binary inputs lena
        age = float(request.form['Age'])
        gender = int(request.form['Gender'])
        bmi = float(request.form['BMI'])
        family_history = int(request.form['Family_History'])
        smoking = int(request.form['Smoking'])
        alcohol = int(request.form['Alcohol_Consumption'])
        hormone = int(request.form['Hormone_Therapy'])
        menopause = int(request.form['Menopause_Status'])
        genetic = int(request.form['Genetic_Mutation'])
        tumor_size = float(request.form['Tumor_Size_cm'])
        blood_pressure = float(request.form['Blood_Pressure'])
        cholesterol = float(request.form['Cholesterol'])
        diabetes = int(request.form['Diabetes'])
        exercise = float(request.form['Exercise_Days_Per_Week'])
        breastfeeding = float(request.form['Breastfeeding_History'])
        annual_income = float(request.form['Annual_Income_USD'])

        # Categorical dropdowns jinka one-hot encoding tha
        physical_activity = request.form['Physical_Activity']  # 'High', 'Low', 'Moderate'
        mammogram = request.form['Mammogram_Result']            # 'Benign', 'Normal', 'Suspicious'

        # One-hot encoding logic (matching notebook get_dummies drop_first=True)
        physical_activity_low = 1 if physical_activity == 'Low' else 0
        physical_activity_moderate = 1 if physical_activity == 'Moderate' else 0

        mammogram_result_normal = 1 if mammogram == 'Normal' else 0
        mammogram_result_suspicious = 1 if mammogram == 'Suspicious' else 0

        # DataFrame create karna exact usi columns sequence me jisme train hua tha
        input_dict = {
            'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'Family_History': family_history,
            'Smoking': smoking,
            'Alcohol_Consumption': alcohol,
            'Hormone_Therapy': hormone,
            'Menopause_Status': menopause,
            'Genetic_Mutation': genetic,
            'Tumor_Size_cm': tumor_size,
            'Blood_Pressure': blood_pressure,
            'Cholesterol': cholesterol,
            'Diabetes': diabetes,
            'Exercise_Days_Per_Week': exercise,
            'Breastfeeding_History': breastfeeding,
            'Annual_Income_USD': annual_income,
            'Physical_Activity_Low': physical_activity_low,
            'Physical_Activity_Moderate': physical_activity_moderate,
            'Mammogram_Result_Normal': mammogram_result_normal,
            'Mammogram_Result_Suspicious': mammogram_result_suspicious
        }

        input_df = pd.DataFrame([input_dict])[feature_names]

        # Features ko scale karna
        input_scaled = scaler.transform(input_df)

        # Prediction aur Probability
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] * 100

        result_text = "Cancer Positive" if prediction == 1 else "Cancer Negative (No Cancer)"
        risk_text = f"Risk Probability: {prob:.2f}%"

        return render_template('index.html', 
                               prediction_text=result_text, 
                               probability_text=risk_text,
                               is_cancer=int(prediction))

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)