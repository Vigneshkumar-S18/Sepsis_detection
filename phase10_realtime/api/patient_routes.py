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
from phase10_realtime.preprocessing.online_preprocessing import OnlinePreprocessor
from phase10_realtime.inference.predictor import RealTimePredictor
from phase10_realtime.explainability.online_integrated_gradients import OnlineIGExplainer
from phase10_realtime.alerts.alert_engine import AlertEngine
import pandas as pd
import io

router = APIRouter()

# Instantiate singletons at API initialization
window_manager = RollingWindowBuffer()
online_preprocessor = OnlinePreprocessor()
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
    top_features = [
        {"feature": str(name), "attribution": float(score)}
        for name, score in zip(explanation["top_features"], explanation["attribution_scores"])
    ]
    
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


class DatasetPayloadModel(BaseModel):
    file_content: str = ""
    patient_id: str = "Uploaded_Patient"


@router.post("/evaluate_dataset")
async def evaluate_dataset_input(payload: DatasetPayloadModel):
    t0 = time.time()
    content = str(payload.file_content).strip()
    
    if not content:
        raise HTTPException(status_code=400, detail="Empty dataset content provided.")
        
    df = None
    parse_errors = []
    
    # 1. Try plain text CSV/PSV/TSV reading
    try:
        first_line = content.split('\n')[0] if '\n' in content else content
        if '\t' in first_line:
            delimiter = '\t'
        elif '|' in first_line:
            delimiter = '|'
        elif ';' in first_line:
            delimiter = ';'
        else:
            delimiter = ','
        df = pd.read_csv(io.StringIO(content), sep=delimiter)
    except Exception as e:
        parse_errors.append(f"CSV read: {e}")
        
    # 2. If CSV failed, try Base64 decoded Excel (.xlsx/.xls)
    if df is None or df.empty or len(df.columns) < 2:
        try:
            import base64
            decoded_bytes = base64.b64decode(content)
            df = pd.read_excel(io.BytesIO(decoded_bytes), engine='openpyxl')
        except Exception as e:
            parse_errors.append(f"Base64 Excel read: {e}")

    # 3. If Base64 failed, try raw latin1 encoded Excel (.xlsx/.xls)
    if df is None or df.empty or len(df.columns) < 2:
        try:
            df = pd.read_excel(io.BytesIO(content.encode('latin1')), engine='openpyxl')
        except Exception as e:
            parse_errors.append(f"Raw Excel read: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded dataset/Excel content. Errors: {parse_errors}")
        
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded dataset contains 0 rows.")
        
    # Cap trajectory length to last 24 hours if very long
    if len(df) > 24:
        df = df.iloc[-24:].copy()
        
    # Run online preprocessing pipeline with schema validation
    try:
        seq = online_preprocessor.preprocess_sequence(df)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(err)}")
    
    # Run step-by-step risk trajectory calculation
    risk_timeline = []
    k_len = seq.shape[0]
    for step in range(1, k_len + 1):
        step_seq = seq[:step, :]
        p_risk = predictor.predict_risk(step_seq)
        risk_timeline.append(float(p_risk))
        
    final_risk = risk_timeline[-1] if risk_timeline else 0.0
    
    # Calculate Captum Integrated Gradients attributions on the sequence
    explanation = explainer.attribute_sequence(seq)
    top_features = [
        {"feature": str(name), "attribution": float(score)}
        for name, score in zip(explanation["top_features"], explanation["attribution_scores"])
    ]
    
    # Extract vitals timelines for charts with robust case-insensitive column matching
    vitals_timeline = {}
    col_map = {str(c).strip().upper(): c for c in df.columns}
    
    mapping_rules = [
        ("HR", ["HR", "HEARTRATE", "HEART_RATE"]),
        ("MAP", ["MAP", "MEANARTERIALPRESSURE", "MEAN_MAP"]),
        ("Resp", ["RESP", "RESPRATE", "RESPIRATORY_RATE", "RR"]),
        ("Temp", ["TEMP", "TEMPERATURE"]),
        ("O2Sat", ["O2SAT", "SPO2", "SATURATION"]),
        ("SpO2", ["SPO2", "O2SAT", "SATURATION"]),
        ("WBC", ["WBC", "WHITE_BLOOD_CELLS"]),
        ("Lactate", ["LACTATE"]),
        ("Creatinine", ["CREATININE"]),
        ("SBP", ["SBP"]),
        ("DBP", ["DBP"])
    ]
    
    for key, aliases in mapping_rules:
        found = False
        for alias in aliases:
            if alias in col_map:
                vitals_timeline[key] = [float(x) for x in df[col_map[alias]].fillna(0).tolist()]
                found = True
                break
        if not found:
            vitals_timeline[key] = []

            
    # Risk Level & Status
    if final_risk >= 0.70:
        risk_level = "Critical"
        status = "HIGH SEPSIS RISK"
        recommendations = [
            "Order STAT Blood Cultures (x2 sets) and Serum Lactate clearance.",
            "Initiate Broad-Spectrum IV Antibiotics within 1 hour.",
            "Administer 30 mL/kg IV Crystalloids for fluid resuscitation.",
            "Notify ICU Attending Physician and alert Rapid Response Team."
        ]
    elif final_risk >= 0.40:
        risk_level = "Warning"
        status = "MODERATE / RISING RISK"
        recommendations = [
            "Re-check vital signs telemetry every 15–30 minutes.",
            "Review recent CBC, WBC count, and inflammatory markers.",
            "Assess patient for potential infectious etiology."
        ]
    else:
        risk_level = "Low"
        status = "STABLE CONTROL"
        recommendations = [
            "Maintain continuous vital telemetry monitoring.",
            "Re-evaluate laboratory measurement panels at next scheduled draw."
        ]
        
    latency_ms = (time.time() - t0) * 1000.0
    
    return {
        "status": "success",
        "model_name": "BiLSTM Champion Engine (391 Features)",
        "patient_id": payload.patient_id,
        "observations_count": k_len,
        "final_risk": final_risk,
        "risk_level": risk_level,
        "alert_status": status,
        "risk_timeline": risk_timeline,
        "vitals_timeline": vitals_timeline,
        "top_features": top_features,
        "recommendations": recommendations,
        "latency_ms": latency_ms
    }
