# predict.py

import joblib
import numpy as np
from xgboost import XGBClassifier
import mysql.connector

# Load model and encoders
model = joblib.load("models/disease_predictor_model.joblib")
le_gender = joblib.load("models/le_gender.joblib")
le_treatment = joblib.load("models/le_treatment.joblib")
le_disease = joblib.load("models/le_disease.joblib")

def log_prediction_to_db(age, gender, treatment, predicted_disease):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="your_mysql_password",  # ⬅️ Update this
            database="health_db"             # ⬅️ Update this
        )
        cursor = conn.cursor()
        query = """
            INSERT INTO predictions_log (age, gender, treatment, predicted_disease)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (age, gender, treatment, predicted_disease))
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Prediction logged in database.")
    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")

def predict_disease(age, gender, treatment):
    try:
        # Validate treatment
        if treatment not in le_treatment.classes_:
            print(f"\n❌ Invalid treatment: '{treatment}'")
            print("✅ Available treatment values:", list(le_treatment.classes_))
            return

        # Validate gender
        if gender not in le_gender.classes_:
            print(f"\n❌ Invalid gender: '{gender}'")
            print("✅ Available gender values:", list(le_gender.classes_))
            return

        # Encode input
        gender_encoded = le_gender.transform([gender])[0]
        treatment_encoded = le_treatment.transform([treatment])[0]

        # Make prediction
        input_features = np.array([[age, gender_encoded, treatment_encoded]])
        prediction = model.predict(input_features)[0]

        predicted_disease = le_disease.inverse_transform([prediction])[0]
        print(f"\n✅ Predicted Disease: {predicted_disease}")

        # Log to database
        log_prediction_to_db(age, gender, treatment, predicted_disease)

    except Exception as e:
        print(f"\n❌ Error: {e}")

# Sample prediction (change these values to test)
predict_disease(age=58, gender="Female", treatment="Radiation")
predict_disease(age=40, gender="Male", treatment="Surgery")
