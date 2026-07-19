import os
import sys
import pandas as pd
import numpy as np
from feature_engineering.config import ROLLING_VITALS, ROLLING_WINDOW_SIZE, SPARSITY_LABS, LAG_VITALS, LAG_INTERVAL

def compute_cardiovascular_features(df_unscaled, df_out):
    """
    Computes Shock Index, Pulse Pressure, and MAP deviation features.
    """
    # Shock Index = HR / SBP
    df_out['Shock_Index'] = df_unscaled['HR'] / df_unscaled['SBP'].clip(lower=40)
    
    # Pulse Pressure = SBP - DBP
    df_out['Pulse_Pressure'] = df_unscaled['SBP'] - df_unscaled['DBP']
    
    # Estimated MAP = (DBP * 2 + SBP) / 3
    est_map = (df_unscaled['DBP'] * 2 + df_unscaled['SBP']) / 3
    df_out['MAP_deviation'] = df_unscaled['MAP'] - est_map
    
    return df_out

def compute_clinical_ratios(df_unscaled, df_out):
    """
    Computes clinical ratios representing physiological distress.
    """
    df_out['HR_MAP_Ratio'] = df_unscaled['HR'] / df_unscaled['MAP'].clip(lower=20)
    df_out['Resp_O2Sat_Ratio'] = df_unscaled['Resp'] / df_unscaled['O2Sat'].clip(lower=10)
    df_out['HR_Temp_Ratio'] = df_unscaled['HR'] / df_unscaled['Temp'].clip(lower=25)
    return df_out

def compute_rolling_statistics(df_unscaled, df_out):
    """
    Computes rolling mean and standard deviation over 6 hours grouped by PatientID.
    """
    # Grouped rolling operations in Pandas can be slow if not written carefully.
    # Using groupby + rolling on the columns at once is highly optimized.
    grouped = df_unscaled.groupby('PatientID')[ROLLING_VITALS]
    
    # 6h Rolling Mean
    roll_mean = grouped.rolling(ROLLING_WINDOW_SIZE, min_periods=1).mean().reset_index(level=0, drop=True)
    for col in ROLLING_VITALS:
        df_out[f"{col}_roll_mean_6h"] = roll_mean[col]
        
    # 6h Rolling Std (Variability)
    roll_std = grouped.rolling(ROLLING_WINDOW_SIZE, min_periods=1).std().reset_index(level=0, drop=True)
    for col in ROLLING_VITALS:
        # First hour std is NaN, fill with 0.0
        df_out[f"{col}_roll_std_6h"] = roll_std[col].fillna(0.0)
        
    return df_out

def compute_trends_and_slopes(df_unscaled, df_out):
    """
    Computes 1-hour lag difference and slope features.
    """
    # Group by patient and shift by 1 hour
    lagged = df_unscaled.groupby('PatientID')[LAG_VITALS].shift(LAG_INTERVAL)
    
    for col in LAG_VITALS:
        # Slope is change per hour (since interval is 1 hour, diff equals slope)
        df_out[f"{col}_diff_1h"] = (df_unscaled[col] - lagged[col]).fillna(0.0)
        
    return df_out

def compute_hours_since_measured(df_unscaled, df_out):
    """
    Tracks elapsed hours since sparse laboratory tests were last measured.
    """
    # For each sparse lab, calculate the hours since the flag col_measured was 1.
    for col in SPARSITY_LABS:
        measured_col = f"{col}_measured"
        if measured_col not in df_unscaled.columns:
            continue
            
        # Re-index index where measured == 1, forward-fill, and subtract from current ICULOS
        temp_col = f"temp_{col}_last"
        df_unscaled[temp_col] = df_unscaled['ICULOS'].where(df_unscaled[measured_col] == 1)
        last_measured_ffilled = df_unscaled.groupby('PatientID')[temp_col].ffill()
        df_unscaled.drop(columns=[temp_col], inplace=True)
            
        hours_since = df_unscaled['ICULOS'] - last_measured_ffilled
        # Impute periods before the first measurement with 999.0
        df_out[f"hours_since_last_{col}"] = hours_since.fillna(999.0)
        
    return df_out

def compute_icu_time_features(df_unscaled, df_out):
    """
    Extracts ICU timeline flags.
    """
    df_out['First_24h_Flag'] = (df_unscaled['ICULOS'] <= 24).astype(int)
    # Proxy diurnal shift hour (hours of stay modulo 24)
    df_out['Diurnal_Proxy_Hour'] = (df_unscaled['ICULOS'] % 24).astype(int)
    return df_out

def generate_engineered_features(df_unscaled, df_scaled_base):
    """
    Main orchestrator that computes all clinical, rolling, trend, and sparsity features,
    returning a DataFrame combining original scaled columns and new features.
    """
    # Output df starts with a copy of the scaled base df
    df_out = df_scaled_base.copy()
    
    df_out = compute_cardiovascular_features(df_unscaled, df_out)
    df_out = compute_clinical_ratios(df_unscaled, df_out)
    df_out = compute_rolling_statistics(df_unscaled, df_out)
    df_out = compute_trends_and_slopes(df_unscaled, df_out)
    df_out = compute_hours_since_measured(df_unscaled, df_out)
    df_out = compute_icu_time_features(df_unscaled, df_out)
    
    return df_out
