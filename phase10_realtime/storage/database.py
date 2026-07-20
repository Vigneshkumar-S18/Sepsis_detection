# Real-Time SQLite Database Manager
import sqlite3
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from phase10_realtime.config import DB_PATH


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create Patients Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        name TEXT,
        age REAL,
        gender INTEGER
    )
    """)

    # 2. Create Measurements Table
    # Contains all 40 PhysioNet clinical variables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        timestamp TEXT,
        HR REAL, O2Sat REAL, Temp REAL, SBP REAL, MAP REAL, DBP REAL, Resp REAL, EtCO2 REAL,
        BaseExcess REAL, HCO3 REAL, FiO2 REAL, pH REAL, PaCO2 REAL, SaO2 REAL, AST REAL, BUN REAL,
        Alkalinephos REAL, Calcium REAL, Chloride REAL, Creatinine REAL, Bilirubin_direct REAL,
        Glucose REAL, Lactate REAL, Magnesium REAL, Phosphate REAL, Potassium REAL,
        Bilirubin_total REAL, TroponinI REAL, Hct REAL, Hgb REAL, PTT REAL, WBC REAL,
        Fibrinogen REAL, Platelets REAL, Age REAL, Gender INTEGER, Unit1 REAL, Unit2 REAL,
        HospAdmTime REAL, ICULOS REAL,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    """)

    # 3. Create Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        timestamp TEXT,
        risk REAL,
        status TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    """)

    # 4. Create Alerts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        risk REAL,
        priority TEXT,
        acknowledged INTEGER DEFAULT 0,
        acknowledged_at TEXT,
        created_at TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    """)
    
    conn.commit()
    conn.close()


if __name__ == "__main__":
    initialize_database()
    print("Database schema initialized successfully.")
