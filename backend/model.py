# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def train_and_save_model(data_path, model_dir='models', model_filename='disease_predictor_model.joblib'):
    try:
        # Step 1: Load dataset
        print("Loading dataset...")
        df = pd.read_csv(data_path)

        # Step 2: Preprocess data
        print("Preprocessing data...")
        required_cols = ['Age', 'Gender', 'Treatment', 'Disease']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset must include columns: {required_cols}")

        df = df.drop(columns=['Patient ID', 'Admission Date', 'Discharge Date'], errors='ignore')

        le_gender = LabelEncoder()
        le_treatment = LabelEncoder()
        le_disease = LabelEncoder()

        df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
        df['Treatment_encoded'] = le_treatment.fit_transform(df['Treatment'])
        df['Disease_encoded'] = le_disease.fit_transform(df['Disease'])

        features = ['Age', 'Gender_encoded', 'Treatment_encoded']
        target = 'Disease_encoded'

        X = df[features]
        y = df[target]

        # Step 3: Train/test split
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Train XGBoost model
        print("Training model...")
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        model.fit(X_train, y_train)

        # Step 5: Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Model Accuracy: {accuracy:.4f}")
        print("\n🧾 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le_disease.classes_))

        # Step 6: Save model and encoders
        print("Saving model and encoders...")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, model_filename))
        joblib.dump(le_gender, os.path.join(model_dir, 'le_gender.joblib'))
        joblib.dump(le_treatment, os.path.join(model_dir, 'le_treatment.joblib'))
        joblib.dump(le_disease, os.path.join(model_dir, 'le_disease.joblib'))

        print(f"\n✅ All files saved in '{model_dir}/' folder.")

    except FileNotFoundError:
        print(f"❌ Error: The file '{data_path}' was not found.")
    except ValueError as ve:
        print(f"❌ Value Error: {ve}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == '__main__':
    # ✅ This is the correct relative path assuming your CSV is inside: backend/data/
    csv_file_path = 'data/healthcare_patient_records_large.csv'
    train_and_save_model(csv_file_path)
