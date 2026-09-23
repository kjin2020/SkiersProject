# Test file for MLP and LSTM
import sys
import os
# Ensure project root is in sys.path
# Need to go two paths up here
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it
    

import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

from utils.preprocess_signals import pad_sequences, normalize_full_signal, normalize_per_timestamp, replace_nan_with_first_value, gaussian_filter1d
from utils.nets import LSTMNet, SimpleMLP, load_classifier_checkpoint
from utils.CustomDataset import CustomDataset
from utils.training_utils import validation, initialize_loss


MODEL_PATH = "./pretrained_models/trained_model_2025_04_06_07_48_lr0.0001_lstm.pth"
TEST_DATA_PATH = "./data/split_data/test_full.json"
TEST_OUTPUT = "./classification/training/runs/test"

def main():
    start_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    # check and select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device = {device}")
    # Load the checkpoint to the model as we need it for certain data
    print("Loading Model...")
    checkpoint = load_classifier_checkpoint(MODEL_PATH, map_location=device)
    
    # Extract the state_dict and custom parameters
    state_dict = checkpoint["state_dict"]
    custom_params = checkpoint["custom_params"]

    # Load in the test data
    print("Loading test data...")
    test_data = []
    labels = []
    skiers_ids = []
    with open(TEST_DATA_PATH, 'r') as f:
        data_json = json.load(f)
        
        
    for cycle in data_json.values():
        # Extract joint data as (num_joints, time_steps)
        cycle_data = [np.array(cycle[joint], dtype=np.float32) for joint in custom_params["choosen_joints"]]

        # Stack into a (num_joints, time_steps) tensor
        cycle_tensor = np.stack(cycle_data)  # Shape: (num_joints, time_steps)

        test_data.append(cycle_tensor)
        labels.append(cycle["Label"])
        skiers_ids.append(cycle["Skier_id"])

    # Preprocess data based on train data
    print("Preprocessing data...")
    # padding
    test_data = pad_sequences(test_data, max_length=custom_params["train_max_length"], pad_value=float('nan'))
    
    # normalization
    if custom_params["normalization"]:
        print("Normalizing the data...")
        if custom_params["norm_type"] == "full_signal":
            test_data = normalize_full_signal(test_data, custom_params["train_mean"], custom_params["train_std"])
        else:
            test_data = normalize_per_timestamp(test_data, custom_params["train_mean"], custom_params["train_std"])
    
    # adjust padding    
    test_data = replace_nan_with_first_value(test_data)
    
    # smoothing
    if custom_params["smoothing"] > 0:
        test_data = gaussian_filter1d(test_data, sigma=custom_params["smoothing"], axis=2)
        
    # convert data to a tensor
    test_data = torch.tensor(test_data, dtype=torch.float32)

    test_dataset = CustomDataset(test_data, labels, custom_params["labels"], skier_id=skiers_ids)

    test_loader = DataLoader(test_dataset, batch_size=custom_params["batch_size"], shuffle=False)
    
    # Extract input and output sizes
    input_channels = custom_params["input_channels"]
    output_channels = custom_params["output_channels"]

    # initialize the model
    if custom_params["network_type"] == "lstm":
        print("Initializing LSTM...")
        net = LSTMNet(input_channels, 
                    custom_params["hidden_size"], 
                    output_channels, 
                    custom_params["num_layers"], 
                    custom_params["dropout"])

    elif custom_params["network_type"] == "mlp":  # MLP
        print("Initializing MLP...")
        net = SimpleMLP(input_channels, 
                        custom_params["hidden1"], 
                        custom_params["hidden2"], 
                        output_channels)
    else:
        print("Invalid network type, not lstm or mlp")

    # Load weights into the model
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    loss_func = initialize_loss(custom_params, custom_params, output_channels)

    avg_test_loss, epoch_accuracy, precision, recall, f1, (conf_matrix, conf_matrix_norm), skiers_acc = validation(test_loader, net, loss_func, device, custom_params["network_type"])

    test_results_text = (
        f"Test Loss: {avg_test_loss:.4f}\n"
        f"Test Accuracy: {epoch_accuracy:.2f}%\n"
        f"Test Precision: {precision:.4f}\n"
        f"Test Recall: {recall:.4f}\n"
        f"Test F1 Score: {f1:.4f}\n"
    )

    for skier_id, acc in skiers_acc.items():
        test_results_text += f"Skier {skier_id} Accuracy: {acc:.2f}%\n"

    # Print results
    print(test_results_text)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Annotate with both values: count and percentage
    annot = np.empty_like(conf_matrix).astype(str)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            count = conf_matrix[i, j]
            perc = conf_matrix_norm[i, j] * 100
            annot[i, j] = f"{count}\n{perc:.1f}%"

    # Plot using seaborn heatmap
    sns.heatmap(conf_matrix_norm, annot=annot, fmt='', cmap="Blues", xticklabels=[*custom_params["labels"]], yticklabels=[*custom_params["labels"]], cbar=True, ax=ax)

    # Labeling
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix: Count and %")

    plt.tight_layout()

    # Save cm and output
    cm_out_path = os.path.join(TEST_OUTPUT, f"cm_{start_time}.png")
    os.makedirs(TEST_OUTPUT, exist_ok=True)
    plt.savefig(cm_out_path)

    for key, item in custom_params.items():
        test_results_text += str(key) + " " + str(item) + "\n"

    test_results_file = os.path.join(TEST_OUTPUT, f"test_results_{start_time}.txt")
    with open(test_results_file, "w") as file:
        file.write(test_results_text)
    

if __name__ == '__main__':
    main()
