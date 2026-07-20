# Real-Time Clinical Deterioration Alert Engine
import os
import sys
import sqlite3
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from phase10_realtime.config import (
    CRITICAL_RISK_THRESHOLD, INCREASE_RISK_THRESHOLD,
    CONSECUTIVE_RISK_THRESHOLD, ALERT_SUPPRESSION_MINUTES
)
from phase10_realtime.storage.database import get_db_connection


class AlertEngine:
    """
    Evaluates risk trajectories and triggers clinical escalation and alerts.
    """
    def __init__(self):
        pass

    def evaluate_patient_risk(self, patient_id: str, current_risk: float) -> dict:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Fetch previous prediction
        cursor.execute(
            "SELECT risk FROM predictions WHERE patient_id = ? ORDER BY id DESC LIMIT 2",
            (patient_id,)
        )
        prev_preds = cursor.fetchall()
        
        prev_risk = prev_preds[0]["risk"] if len(prev_preds) > 0 else None
        older_risk = prev_preds[1]["risk"] if len(prev_preds) > 1 else None

        # 2. Check if there is an active acknowledged alert to respect suppression rule
        cursor.execute("""
        SELECT acknowledged, acknowledged_at, risk FROM alerts 
        WHERE patient_id = ? ORDER BY id DESC LIMIT 1
        """, (patient_id,))
        last_alert = cursor.fetchone()
        
        is_suppressed = False
        if last_alert and last_alert["acknowledged"] == 1:
            # Check elapsed time
            ack_time = last_alert["acknowledged_at"]
            # Acknowledge time is ISO string
            try:
                from datetime import datetime
                elapsed = (datetime.now() - datetime.fromisoformat(ack_time)).total_seconds() / 60.0
                if elapsed < ALERT_SUPPRESSION_MINUTES:
                    # Suppress alert UNLESS the risk increased by > 10% since acknowledgement
                    if current_risk <= last_alert["risk"] + 0.10:
                        is_suppressed = True
            except Exception:
                pass

        # 3. Evaluate Deterioration Rules
        trigger_alert = False
        priority = "Low"
        reasons = []

        # Rule 1: Risk > 70% -> Critical
        if current_risk >= CRITICAL_RISK_THRESHOLD:
            trigger_alert = True
            priority = "Critical"
            reasons.append(f"Sepsis risk exceeds critical threshold ({current_risk:.1%})")
            
        # Rule 2: Risk increased by > 20% compared to previous prediction
        elif prev_risk is not None and (current_risk - prev_risk) >= INCREASE_RISK_THRESHOLD:
            trigger_alert = True
            priority = "High"
            reasons.append(f"Risk increased significantly by {(current_risk - prev_risk):.1%}")

        # Rule 3: Risk remains > 80% for two consecutive predictions -> Escalate
        if prev_risk is not None and current_risk >= CONSECUTIVE_RISK_THRESHOLD and prev_risk >= CONSECUTIVE_RISK_THRESHOLD:
            trigger_alert = True
            priority = "Critical"
            reasons.append("Risk remains critically high (>80%) for consecutive periods")

        alert_data = None
        if trigger_alert and not is_suppressed:
            # Save alert to database
            created_at = time.strftime('%Y-%m-%dT%H:%M:%S')
            cursor.execute("""
            INSERT INTO alerts (patient_id, risk, priority, acknowledged, created_at)
            VALUES (?, ?, ?, 0, ?)
            """, (patient_id, current_risk, priority, created_at))
            conn.commit()
            
            alert_data = {
                "patient_id": patient_id,
                "risk": current_risk,
                "priority": priority,
                "reasons": reasons,
                "created_at": created_at
            }
            
        conn.close()
        return alert_data
