# Rolling Window Sequence Buffer Manager
import os
import sys
import numpy as np
import pandas as pd
import sqlite3

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from phase10_realtime.config import DB_PATH
from phase10_realtime.storage.database import get_db_connection
from phase10_realtime.preprocessing.online_preprocessing import OnlinePreprocessor


class RollingWindowBuffer:
    """
    Retrieves the last 12 hours of patient telemetry from the database
    and formats it into sequence tensors of shape (12, 95).
    """
    def __init__(self):
        self.preprocessor = OnlinePreprocessor()

    def get_patient_sequence(self, patient_id: str) -> np.ndarray:
        conn = get_db_connection()
        
        # Query measurements for this patient sorted by time/stay duration
        query = "SELECT * FROM measurements WHERE patient_id = ? ORDER BY ICULOS ASC"
        df = pd.read_sql_query(query, conn, params=[patient_id])
        conn.close()
        
        if len(df) == 0:
            raise ValueError(f"No telemetry records found for patient: {patient_id}")
            
        # Limit to the most recent 12 hours of observations
        df_window = df.tail(12)
        
        # Preprocess the windowed DataFrame (returns scaled array)
        processed_arr = self.preprocessor.preprocess_sequence(df_window)  # shape: (k, 95)
        
        # Apply padding if stay length is under 12 hours
        seq_len = processed_arr.shape[0]
        if seq_len < 12:
            padded_arr = np.zeros((12, 95), dtype=np.float32)
            padded_arr[12-seq_len:] = processed_arr
            return padded_arr
        else:
            return processed_arr
