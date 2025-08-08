import mysql.connector
from mysql.connector import Error

# Change these based on your MySQL setup
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'azhar',
    'database': 'health_platform'
}

def create_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"Error: {e}")
        return None

# Insert new patient record
def insert_patient(name, age, weight, bp, sugar, symptoms):
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        query = """
            INSERT INTO patients (name, age, weight, blood_pressure, blood_sugar, symptoms)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (name, age, weight, bp, sugar, symptoms)
        cursor.execute(query, values)
        conn.commit()
        patient_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return patient_id
    return None

# Fetch patient by ID
def get_patient(patient_id):
    conn = create_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM patients WHERE id = %s", (patient_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    return None


# Test insert
pid = insert_patient("Alice", 35, 62.5, "120/80", 92.3, "Fatigue, thirst")
print(f"Inserted patient ID: {pid}")

# Test fetch
patient = get_patient(pid)
print("Patient Data:", patient)
