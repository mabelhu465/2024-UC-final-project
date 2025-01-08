import os
import json
import numpy as np
import tensorflow as tf
from attention import Attention
from tensorflow.keras.losses import MeanSquaredError
from file_loader import file_loader
import properscoring as ps

def load_latest_model(model_folder):
    """Loads the latest model from the specified folder."""
    saved_models = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.keras')]
    if not saved_models:
        raise FileNotFoundError("No saved models found in the specified folder.")
    latest_model_path = max(saved_models, key=os.path.getctime)
    print(f"Loaded model from: {latest_model_path}")
    model = tf.keras.models.load_model(latest_model_path, custom_objects={'Attention': Attention, 'mse': MeanSquaredError()})
    return model

#def save_predictions_first_sequence(predictions, true_labels, timeslots, regions, output_file="predictions.json"):
#    """Saves only the first-sequence predictions along with metadata to a JSON file."""
#    results = []
#    for idx, region in enumerate(regions):
#        # Only take the first prediction for each region
#        results.append({
#            "region": region,
#            "timeslot": timeslots[0],  # First sequence prediction corresponds to the first timeslot
#            "true_label": true_labels[0, idx].tolist(),  # True label for the first sequence
#            "prediction": predictions[0, idx].tolist()   # Prediction for the first sequence
#        })
#    with open(output_file, 'w') as f:
#        json.dump(results, f, indent=4)
#    print(f"First-sequence predictions saved to {output_file}")

def save_predictions_all(predictions, true_labels, timeslots, regions, output_file_prefix="predictions_all",threshold=0.1):
    """
    Saves predictions and true labels for each volume type separately.
    """
    volume_types = true_labels.shape[-1]  # Number of volume types (e.g., 2 for inflow and outflow)
    num_regions = len(regions)
    results = {f"volume_type_{v}": [] for v in range(volume_types)}
    metrics = {f"volume_type_{v}": {"RMSE": None, "MAPE": None, "MAE": None, "CRPS": None} for v in range(volume_types)}

    for t_idx, timeslot in enumerate(timeslots[:477]):  # take all predictions, the sequecens number is 960-483=477
        for r_idx, region in enumerate(regions):
            label_idx = t_idx * num_regions + r_idx  # Flattened index in predictions and true_labels
            if label_idx >= len(true_labels):  # Ensure index is within bounds
                print(f"Skipping region {region} at timeslot {timeslot}: Index out of bounds.")
                continue
            for v in range(volume_types):
                true_val = float(true_labels[label_idx, v])
                pred_val = float(predictions[label_idx, v])
                results[f"volume_type_{v}"].append({
                    "region": region,
                    "timeslot": timeslot,
                    "true_label": float(true_labels[label_idx, v]),  # Specific volume type
                    "prediction": float(predictions[label_idx, v])  # Specific volume type
                })
    
    
     # Evaluate metrics for each volume type with filtering
    for v in range(volume_types):
        true_vals = np.array([item["true_label"] for item in results[f"volume_type_{v}"]])
        pred_vals = np.array([item["prediction"] for item in results[f"volume_type_{v}"]])

        # Apply filtering based on the threshold
        mask = true_vals > threshold
        filtered_true_vals = true_vals[mask]
        filtered_pred_vals = pred_vals[mask]
        
        if len(filtered_true_vals) > 0:
            metrics[f"volume_type_{v}"]["RMSE"] = np.sqrt(np.mean((filtered_true_vals - filtered_pred_vals) ** 2))
            metrics[f"volume_type_{v}"]["MAPE"] = np.mean(np.abs(filtered_true_vals - filtered_pred_vals) / (filtered_true_vals + 1e-8))  # Avoid division by zero
            metrics[f"volume_type_{v}"]["MAE"] = np.mean(np.abs(filtered_true_vals - filtered_pred_vals))
            metrics[f"volume_type_{v}"]["CRPS"] = np.mean(ps.crps_gaussian(filtered_true_vals, mu=filtered_pred_vals, sig=1e-6))
        else:
            metrics[f"volume_type_{v}"]["RMSE"] = None
            metrics[f"volume_type_{v}"]["MAPE"] = None
            metrics[f"volume_type_{v}"]["MAE"] = None
            metrics[f"volume_type_{v}"]["CRPS"] = None

    # Save results and metrics
    for v in range(volume_types):
        output_file = os.path.join("./taxi_result/", f"{output_file_prefix}_volume_type_{v}.json")
        with open(output_file, 'w') as f:
            json.dump(results[f"volume_type_{v}"], f, indent=4)
        print(f"Volume type {v} predictions saved to {output_file}")

    metrics_file = os.path.join("./taxi_result/", f"{output_file_prefix}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_file}")




def save_predictions_first(predictions, true_labels, timeslots, regions, output_file_prefix="predictions_first",threshold=0.1):
    """
    Saves predictions and true labels for each volume type separately.
    """
    volume_types = true_labels.shape[-1]  # Number of volume types (e.g., 2 for inflow and outflow)
    num_regions = len(regions)
    results = {f"volume_type_{v}": [] for v in range(volume_types)}
    metrics = {f"volume_type_{v}": {"RMSE": None, "MAPE": None, "MAE": None, "CRPS": None} for v in range(volume_types)}

    for t_idx, timeslot in enumerate(timeslots[:1]):  # Only take the first time step
        for r_idx, region in enumerate(regions):
            label_idx = t_idx * num_regions + r_idx  # Flattened index in predictions and true_labels
            if label_idx >= len(true_labels):  # Ensure index is within bounds
                print(f"Skipping region {region} at timeslot {timeslot}: Index out of bounds.")
                continue
            for v in range(volume_types):
                true_val = float(true_labels[label_idx, v])
                pred_val = float(predictions[label_idx, v])
                results[f"volume_type_{v}"].append({
                    "region": region,
                    "timeslot": timeslot,
                    "true_label": float(true_labels[label_idx, v]),  # Specific volume type
                    "prediction": float(predictions[label_idx, v])  # Specific volume type
                })
    print(f"Sample prediction: {predictions[0, 0]}")
    print(f"Sample true label: {true_labels[0, 0]}")
    
     # Evaluate metrics for each volume type with filtering
    for v in range(volume_types):
        true_vals = np.array([item["true_label"] for item in results[f"volume_type_{v}"]])
        pred_vals = np.array([item["prediction"] for item in results[f"volume_type_{v}"]])

        # Apply filtering based on the threshold
        mask = true_vals > threshold
        filtered_true_vals = true_vals[mask]
        filtered_pred_vals = pred_vals[mask]
        
        if len(filtered_true_vals) > 0:
            metrics[f"volume_type_{v}"]["RMSE"] = np.sqrt(np.mean((filtered_true_vals - filtered_pred_vals) ** 2))
            metrics[f"volume_type_{v}"]["MAPE"] = np.mean(np.abs(filtered_true_vals - filtered_pred_vals) / (filtered_true_vals + 1e-8))  # Avoid division by zero
            metrics[f"volume_type_{v}"]["MAE"] = np.mean(np.abs(filtered_true_vals - filtered_pred_vals))
            metrics[f"volume_type_{v}"]["CRPS"] = np.mean(ps.crps_gaussian(filtered_true_vals, mu=filtered_pred_vals, sig=1e-6))
        else:
            metrics[f"volume_type_{v}"]["RMSE"] = None
            metrics[f"volume_type_{v}"]["MAPE"] = None
            metrics[f"volume_type_{v}"]["MAE"] = None
            metrics[f"volume_type_{v}"]["CRPS"] = None


    # Save results and metrics
    for v in range(volume_types):
        output_file = os.path.join("./taxi_result/", f"{output_file_prefix}_volume_type_{v}.json")
        with open(output_file, 'w') as f:
            json.dump(results[f"volume_type_{v}"], f, indent=4)
        print(f"Volume type {v} predictions saved to {output_file}")

    metrics_file = os.path.join("./taxi_result/", f"{output_file_prefix}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_file}")
    
def save_predictions_first_20(predictions, true_labels, timeslots, regions, output_file_prefix="predictions_first_20", threshold=0.1):
    """
    Saves only the first 20 predictions for each region along with metadata to a JSON file for each volume type.
    Computes metrics (RMSE, MAPE, MAE, CRPS) with filtering based on the threshold.
    """
    
    volume_types = true_labels.shape[-1]  # Number of volume types (e.g., 2 for inflow and outflow)
    num_regions = len(regions)
    results = {f"volume_type_{v}": [] for v in range(volume_types)}
    metrics = {f"volume_type_{v}": {"RMSE": None, "MAPE": None, "MAE": None, "CRPS": None} for v in range(volume_types)}

    for t_idx, timeslot in enumerate(timeslots[:20]):  # Take only the first 20 time steps
        for r_idx, region in enumerate(regions):
            label_idx = t_idx * num_regions + r_idx  # Flattened index in predictions and true_labels
            if label_idx >= len(true_labels):  # Ensure index is within bounds
                print(f"Skipping region {region} at timeslot {timeslot}: Index out of bounds.")
                continue
            for v in range(volume_types):
                true_val = float(true_labels[label_idx, v])
                pred_val = float(predictions[label_idx, v])
                results[f"volume_type_{v}"].append({
                    "region": region,
                    "timeslot": timeslot,
                    "true_label": true_val,
                    "prediction": pred_val
                })

    # Compute metrics for each volume type
    for v in range(volume_types):
        true_vals = np.array([item["true_label"] for item in results[f"volume_type_{v}"]])
        pred_vals = np.array([item["prediction"] for item in results[f"volume_type_{v}"]])

        # Apply filtering based on the threshold
        mask = true_vals > threshold
        filtered_true_vals = true_vals[mask]
        filtered_pred_vals = pred_vals[mask]

        if len(filtered_true_vals) > 0:
            metrics[f"volume_type_{v}"]["RMSE"] = np.sqrt(np.mean((filtered_true_vals - filtered_pred_vals) ** 2))
            metrics[f"volume_type_{v}"]["MAPE"] = np.mean(np.abs(filtered_true_vals - filtered_pred_vals) / (filtered_true_vals + 1e-8))  # Avoid division by zero
            metrics[f"volume_type_{v}"]["MAE"] = np.mean(np.abs(filtered_true_vals - filtered_pred_vals))
            metrics[f"volume_type_{v}"]["CRPS"] = np.mean(ps.crps_gaussian(filtered_true_vals, mu=filtered_pred_vals, sig=1e-6))
        else:
            metrics[f"volume_type_{v}"]["RMSE"] = None
            metrics[f"volume_type_{v}"]["MAPE"] = None
            metrics[f"volume_type_{v}"]["MAE"] = None
            metrics[f"volume_type_{v}"]["CRPS"] = None

    # Save results and metrics
    for v in range(volume_types):
        output_file = os.path.join("./taxi_result/", f"{output_file_prefix}_volume_type_{v}.json")
        with open(output_file, 'w') as f:
            json.dump(results[f"volume_type_{v}"], f, indent=4)
        print(f"Volume type {v} predictions saved to {output_file}")

    metrics_file = os.path.join("./taxi_result/", f"{output_file_prefix}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_file}")

    




def main():
    model_folder = "./hdf5s_taxi/"
    #output_file = "predictions_first_sequence.json"
    
    # Load the latest model
    model = load_latest_model(model_folder)

    # Load test data
    sampler = file_loader()
    att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(
        datatype="test", 
        att_lstm_num=3, 
        long_term_lstm_seq_len=3,
        short_term_lstm_seq_len=7,
        nbhd_size=2,
        cnn_nbhd_size=3
    )
    print("Test data loaded.")

    # Perform predictions
    print("Starting predictions...")
    y_pred = model.predict(x=att_cnnx + att_flow + att_x + cnnx + flow + [x, ])
    # Debug shapes
    print("Predictions shape:", y_pred.shape)
    print("True labels shape:", y.shape)
    
    
    # Prepare metadata
    regions = [(i, j) for i in range(10) for j in range(20)]
    print("Number of regions:", len(regions))
    timeslots = list(range(y.shape[0]))
    
    # Save sequence predictions
    #save_predictions_by_volume_type(y_pred, y, timeslots, regions, output_file="predictions_first_sequence.json")
    #save_predictions_first_20(y_pred, y, timeslots, regions,output_file= "predictions_first_20_sequence.json")
    threshold = float(sampler.threshold) / sampler.config["volume_train_max"]
    print("Evaluating threshold: {0}.".format(threshold))
    save_predictions_all(y_pred, y, timeslots, regions, output_file_prefix="predictions_all", threshold=threshold)
    save_predictions_first(y_pred, y, timeslots, regions, output_file_prefix="predictions_first", threshold=threshold)
    save_predictions_first_20(y_pred, y, timeslots, regions, output_file_prefix="predictions_20", threshold=threshold)

    

if __name__ == "__main__":
    main()
