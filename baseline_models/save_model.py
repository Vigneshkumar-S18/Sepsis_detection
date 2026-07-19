# Model Checkpoint Serializer
import os
import pickle


def save_checkpoint(model, model_name, dataset_name, output_dir):
    """
    Serializes a trained baseline classical ML model checkpoint.

    Parameters
    ----------
    model : object
        A fitted classifier model.
    model_name : str
        Algorithm name (e.g. 'random_forest').
    dataset_name : str
        Dataset identifier (e.g. 'dataset_a').
    output_dir : str
        Directory to save checkpoints.

    Returns
    -------
    str
        Path to the saved checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_{dataset_name}.pkl"
    file_path = os.path.join(output_dir, filename)

    # Use pickle for standard scikit-learn, XGBoost, and LightGBM structures
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

    return file_path
