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
        
    # Get previous prediction to calculate delta
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT risk FROM predictions WHERE patient_id = ? ORDER BY id DESC LIMIT 1",
        (patient_id,)
    )
    prev_row = cursor.fetchone()
    prev_risk = prev_row["risk"] if prev_row else None
    
    # Inference
    risk = predictor.predict_risk(sequence)
    
    # Classifications
    status = "High Risk" if risk >= 0.50 else "Low Risk"
    risk_level = "High" if risk >= 0.70 else ("Medium" if risk >= 0.50 else "Low")
    
    delta_risk = risk - prev_risk if prev_risk is not None else 0.0
    trend = "Increasing" if delta_risk > 0.05 else ("Decreasing" if delta_risk < -0.05 else "Stable")
    
    # Save prediction history
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO predictions (patient_id, timestamp, risk, status) VALUES (?, ?, ?, ?)",
        (patient_id, timestamp, risk, status)
    )
    conn.commit()
    conn.close()
    
    # Evaluate alerts
    alert = alert_engine.evaluate_patient_risk(patient_id, risk)
    
    # Local explainability (top features attribution) in real-time
    explanation = explainer.attribute_sequence(sequence)
    top_features = explanation["top_features"]
    
    # Clinical recommendation logic matching decision support guidelines
    if risk >= 0.70:
        recommendation = "Immediate physician review. Check vital stability and lactate clearance."
    elif risk >= 0.50:
        recommendation = "Escalate monitoring frequency. Consider checking WBC count and repeating vitals."
    else:
        recommendation = "Routine clinical monitoring. Patient is physiologically stable."

    certainty = 0.5 + abs(risk - 0.5)
    latency_ms = (time.time() - t0) * 1000.0

    return {
        "patient_id": patient_id,
        "timestamp": timestamp,
        "risk": risk,
        "risk_level": risk_level,
        "status": status,
        "trend": trend,
        "delta_risk": delta_risk,
        "top_features": top_features,
        "certainty": certainty,
        "alert_triggered": alert is not None,
        "alert": alert,
        "recommendation": recommendation,
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
        "attribution_scores": explanation["attribution_scores"],
        "feature_names": explainer.feature_names,
        "temporal_attributions": explanation["temporal_attributions"]
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


@router.get("/patients")
async def get_patients_list():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Fetch all patients
    cursor.execute("SELECT * FROM patients")
    patients_rows = cursor.fetchall()
    
    patients_list = []
    for p in patients_rows:
        p_id = p["patient_id"]
        
        # Get latest measurements
        cursor.execute(
            "SELECT * FROM measurements WHERE patient_id = ? ORDER BY id DESC LIMIT 1",
            (p_id,)
        )
        meas_row = cursor.fetchone()
        
        # Get latest prediction
        cursor.execute(
            "SELECT * FROM predictions WHERE patient_id = ? ORDER BY id DESC LIMIT 1",
            (p_id,)
        )
        pred_row = cursor.fetchone()
        
        # Check active alerts
        cursor.execute(
            "SELECT * FROM alerts WHERE patient_id = ? AND acknowledged = 0 ORDER BY id DESC LIMIT 1",
            (p_id,)
        )
        alert_row = cursor.fetchone()
        
        latest_meas = dict(meas_row) if meas_row else {}
        latest_pred = dict(pred_row) if pred_row else {}
        
        patients_list.append({
            "patient_id": p_id,
            "name": p["name"],
            "age": p["age"],
            "gender": "Male" if p["gender"] == 1 else "Female",
            "latest_vitals": {
                "HR": latest_meas.get("HR", 0.0),
                "MAP": latest_meas.get("MAP", 0.0),
                "Temp": latest_meas.get("Temp", 0.0),
                "Resp": latest_meas.get("Resp", 0.0),
                "SpO2": latest_meas.get("O2Sat", 0.0),
                "ICULOS": latest_meas.get("ICULOS", 0.0)
            },
            "latest_prediction": {
                "risk": latest_pred.get("risk", 0.0),
                "status": latest_pred.get("status", "Low Risk")
            },
            "alert_triggered": alert_row is not None,
            "alert": dict(alert_row) if alert_row else None
        })
        
    conn.close()
    return patients_list


@router.get("/prediction/{patient_id}/history")
async def get_patient_prediction_history(patient_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, risk FROM predictions WHERE patient_id = ? ORDER BY id ASC",
        (patient_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
