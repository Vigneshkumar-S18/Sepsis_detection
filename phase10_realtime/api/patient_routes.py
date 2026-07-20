# FastAPI Routing Endpoints
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import sqlite3
import time
from datetime import datetime
import numpy as np

from phase10_realtime.config import DB_PATH
from phase10_realtime.storage.database import get_db_connection
from phase10_realtime.sequence.rolling_window import RollingWindowBuffer
from phase10_realtime.inference.predictor import RealTimePredictor
from phase10_realtime.explainability.online_integrated_gradients import OnlineIGExplainer
from phase10_realtime.alerts.alert_engine import AlertEngine

router = APIRouter()

# Instantiate singletons at API initialization
window_manager = RollingWindowBuffer()
predictor = RealTimePredictor()
explainer = OnlineIGExplainer(predictor.model)
alert_engine = AlertEngine()


class PatientUpdateModel(BaseModel):
    patient_id: str
    timestamp: str
    HR: float
    MAP: float
    Temp: float
    SpO2: float
    Resp: float
    SBP: float = 120.0
    DBP: float = 80.0
    Age: float = 65.0
    Gender: int = 1
    ICULOS: float = 1.0


@router.post("/patient/update")
async def update_patient_data(data: PatientUpdateModel):
    # 1. Clinical validation: reject impossible measurements
    if data.HR <= 0 or data.HR > 300:
        raise HTTPException(status_code=400, detail=f"Invalid Heart Rate value: {data.HR}")
    if data.Temp < 25.0 or data.Temp > 45.0:
        raise HTTPException(status_code=400, detail=f"Invalid Temperature value: {data.Temp}")
    if data.SpO2 <= 0 or data.SpO2 > 100:
        raise HTTPException(status_code=400, detail=f"Invalid SpO2 value: {data.SpO2}")
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Ensure patient exists in database
    cursor.execute(
        "INSERT OR IGNORE INTO patients (patient_id, name, age, gender) VALUES (?, ?, ?, ?)",
        (data.patient_id, f"Patient {data.patient_id}", data.Age, data.Gender)
    )
    
    # Store measurement
    cursor.execute("""
    INSERT INTO measurements (
        patient_id, timestamp, HR, MAP, Temp, O2Sat, Resp, SBP, DBP, Age, Gender, ICULOS
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.patient_id, data.timestamp, data.HR, data.MAP, data.Temp, data.SpO2,
        data.Resp, data.SBP, data.DBP, data.Age, data.Gender, data.ICULOS
    ))
    
    conn.commit()
    conn.close()
    
    # Return success
    return {"status": "success", "message": "Telemetry record registered successfully."}


@router.get("/prediction/{patient_id}")
async def get_patient_prediction(patient_id: str):
    t0 = time.time()
    try:
        sequence = window_manager.get_patient_sequence(patient_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
        
    # Inference
    risk = predictor.predict_risk(sequence)
    status = "High Risk" if risk >= 0.50 else "Low Risk"
    
    # Save prediction history
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO predictions (patient_id, timestamp, risk, status) VALUES (?, ?, ?, ?)",
        (patient_id, timestamp, risk, status)
    )
    conn.commit()
    conn.close()
    
    # Evaluate alerts
    alert = alert_engine.evaluate_patient_risk(patient_id, risk)
    
    latency_ms = (time.time() - t0) * 1000.0
    
    return {
        "patient_id": patient_id,
        "risk": risk,
        "status": status,
        "alert_triggered": alert is not None,
        "alert": alert,
        "latency_ms": latency_ms
    }


@router.get("/explanation/{patient_id}")
async def get_patient_explanation(patient_id: str):
    try:
        sequence = window_manager.get_patient_sequence(patient_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
        
    explanation = explainer.attribute_sequence(sequence)
    return {
        "patient_id": patient_id,
        "top_features": explanation["top_features"],
        "attribution_scores": explanation["attribution_scores"]
    }


@router.get("/alerts")
async def get_active_alerts():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    alerts = [dict(row) for row in rows]
    return {"active_alerts": alerts, "count": len(alerts)}


@router.post("/alerts/acknowledge/{patient_id}")
async def acknowledge_patient_alerts(patient_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    ack_time = datetime.now().isoformat()
    cursor.execute(
        "UPDATE alerts SET acknowledged = 1, acknowledged_at = ? WHERE patient_id = ? AND acknowledged = 0",
        (ack_time, patient_id)
    )
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"Alerts for patient {patient_id} acknowledged."}


@router.get("/dashboard")
async def get_dashboard_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total patients count
    cursor.execute("SELECT COUNT(*) as cnt FROM patients")
    patients_count = cursor.fetchone()["cnt"]
    
    # High risk patients count (risk >= 0.50 in last predictions)
    cursor.execute("""
    SELECT COUNT(DISTINCT patient_id) as cnt FROM predictions 
    WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY patient_id) AND risk >= 0.50
    """)
    high_risk_count = cursor.fetchone()["cnt"]
    
    # Active alerts count
    cursor.execute("SELECT COUNT(*) as cnt FROM alerts WHERE acknowledged = 0")
    active_alerts = cursor.fetchone()["cnt"]
    
    # Average risk score
    cursor.execute("""
    SELECT AVG(risk) as avg_risk FROM predictions 
    WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY patient_id)
    """)
    avg_risk = cursor.fetchone()["avg_risk"] or 0.0
    
    conn.close()
    
    return {
        "occupancy": patients_count,
        "high_risk_count": high_risk_count,
        "active_alerts": active_alerts,
        "average_risk": float(avg_risk)
    }
