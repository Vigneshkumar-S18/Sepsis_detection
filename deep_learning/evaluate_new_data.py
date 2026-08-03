import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger
from deep_learning.models.bilstm import BiLSTMClassifier
from deep_learning.models.lstm import LSTMClassifier
from deep_learning.models.gru import GRUClassifier

def get_labels_and_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    
    return acc, prec, rec, f1, auroc, auprc

def generate_sequences(df, feature_cols, window_size=12, horizon=0):
    features = df[feature_cols].values.astype(np.float32)
    labels = df['SepsisLabel'].values
    
    X_list = []
    y_list = []
    n_rows = len(df)
    
    for start in range(n_rows - window_size + 1):
        end = start + window_size - 1
        
        target_idx = end + horizon
        if target_idx >= n_rows:
            continue
            
        label = int(labels[target_idx])
        X_window = features[start : start + window_size]
        
        X_list.append(X_window)
        y_list.append(label)
        
    return X_list, y_list

def main():
    logger.info("Evaluating models on new preprocessed data...")
    
    data_dir = os.path.join(project_root, "newpreprocessedtrainingdata", "completed_preprocessed_complete")
    psv_files = glob.glob(os.path.join(data_dir, "*.psv"))
    if not psv_files:
        logger.error(f"No .psv files found in {data_dir}")
        return
        
    logger.info(f"Found {len(psv_files)} files. Reading feature columns...")
    
    test_feat_path = os.path.join(project_root, "datasets", "processed", "test_features.parquet")
    test_df = pd.read_parquet(test_feat_path)
    feature_cols = [col for col in test_df.columns if col not in ["PatientID", "SepsisLabel"]]
    
    all_X = []
    all_y = []
    
    logger.info("Extracting sequences from PSV files...")
    # Read a sample of files to ensure it finishes quickly enough but gives a representative score
    for f in tqdm(psv_files[:1000]):
        df = pd.read_csv(f, sep='|')
        
        if "Patient_ID" in df.columns:
            df.rename(columns={"Patient_ID": "PatientID"}, inplace=True)
            
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            for m in missing:
                df[m] = 0.0
                
        df.fillna(0.0, inplace=True)
                
        X_list, y_list = generate_sequences(df, feature_cols, window_size=12, horizon=0)
        all_X.extend(X_list)
        all_y.extend(y_list)
        
    if not all_X:
        logger.error("No valid sequences generated.")
        return
        
    X_tensor = torch.tensor(np.array(all_X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(all_y), dtype=torch.float32)
    
    logger.info(f"Generated {len(all_X)} sequences of shape {all_X[0].shape}.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = X_tensor.to(device)
    
    input_dim = len(feature_cols)
    hidden_dim = 64
    
    models = {
        "BiLSTM": BiLSTMClassifier(input_dim, hidden_dim, num_layers=2, dropout=0.2),
        "LSTM": LSTMClassifier(input_dim, hidden_dim, num_layers=2, dropout=0.2),
        "GRU": GRUClassifier(input_dim, hidden_dim, num_layers=2, dropout=0.2)
    }
    
    checkpoints_dir = os.path.join(project_root, "experiments", "checkpoints")
    results = []
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    
    for model_name, model in models.items():
        ckpt_path = os.path.join(checkpoints_dir, f"{model_name.lower()}_w12_h0_final_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(checkpoints_dir, f"{model_name.lower()}_w12_h0_best.pt")
            
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint for {model_name} not found. Skipping.")
            continue
            
        logger.info(f"Evaluating {model_name}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        all_probs = []
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                
        acc, prec, rec, f1, auroc, auprc = get_labels_and_predictions(all_y, all_probs)
        results.append({
            "Model": model_name,
            "AUROC": auroc,
            "AUPRC": auprc,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
    # Use the last time step of the sequence for baseline ML models
    X_flat = X_tensor[:, -1, :].cpu().numpy()
    baselines = ["random_forest", "lightgbm", "xgboost"]
    
    for bl in baselines:
        ckpt_path = os.path.join(checkpoints_dir, f"{bl}_dataset_a.pkl")
        if os.path.exists(ckpt_path):
            logger.info(f"Evaluating Baseline {bl}...")
            with open(ckpt_path, 'rb') as f:
                clf = pickle.load(f)
            
            try:
                probs = clf.predict_proba(X_flat)[:, 1]
                acc, prec, rec, f1, auroc, auprc = get_labels_and_predictions(all_y, probs)
                results.append({
                    "Model": bl.replace("_", " ").title(),
                    "AUROC": auroc,
                    "AUPRC": auprc,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-Score": f1
                })
            except Exception as e:
                logger.error(f"Failed evaluating {bl}: {e}")

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by="AUROC", ascending=False)
        
        artifact_path = os.path.join("C:\\Users\\VIGNESH KUMAR\\.gemini\\antigravity-ide", "brain", "18c1c177-350e-4fc7-944c-18afae575c3c", "model_accuracy_report.md")
        
        md_content = "# Model Accuracy Report on New Preprocessed Dataset\n\n"
        md_content += "This report evaluates the accuracy of the trained Sepsis models against the newly provided dataset (`newpreprocessedtrainingdata`).\n\n"
        md_content += "### Evaluation Results\n\n"
        md_content += df_res.to_markdown(index=False, floatfmt=".4f")
        
        with open(artifact_path, "w", encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"Report successfully saved to {artifact_path}")
    else:
        logger.warning("No results to compile.")

if __name__ == "__main__":
    main()
