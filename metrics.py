import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from model import TransformerModel
from loaders.loader import CombinedDataset
from loaders.youtube_loader import FORMAT_RANGES, FORMAT_SIZE
from inference import generate, load_model
from collections import defaultdict

# Hyperparameters (make sure these match your training settings)
TOKEN_DIM = FORMAT_SIZE
MODEL_PARAMS = (TOKEN_DIM, 256, 16, 4, 1024)  # d_input, d_model, nhead, num_layers, dim_feedforward
CONTEXT_WINDOW = 16
device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

def basic_loss_metrics(obj_name, loss_name, obj, loss_fn, num_samples, dataset_name, model_name, loss_unit="meters", ymax=2.0):    
    dataset = loader.lab_test_dataset
    mask = loader.lab_mask
    context_size = -15
    
    plt.figure(figsize=(10, 6))  # Increase figure size for better readability
    
    losses_dict = defaultdict(list)
    for i in range(num_samples):
        predicted_sequences, fps = generate(
            model, 
            dataset, 
            mask, 
            context_size, 
            device,
            1, 
            use_mask_on_generation=True
        )
        predicted_sequences = predicted_sequences[:, context_size:]
        
        predicted_sequences = predicted_sequences * (dataset.std + 1e-8) + dataset.mean
        ground_truth = predicted_sequences[0][:, FORMAT_RANGES[obj][0]:FORMAT_RANGES[obj][1]]
        pred = predicted_sequences[1][:, FORMAT_RANGES[obj][0]:FORMAT_RANGES[obj][1]]
        ground_truth = ground_truth.numpy()
        pred = pred.numpy()
        
        loss = loss_fn(ground_truth, pred)
        timesteps = np.arange(len(ground_truth)) / fps
        for i in range(len(ground_truth)):
            losses_dict[timesteps[i]].append(loss[i])
        plt.plot(timesteps, loss, color="grey", alpha=0.1)
    
    timesteps = sorted(list(losses_dict.keys()))
    losses = []
    for t in timesteps:
        losses.append(np.nanmedian(losses_dict[t]))
    
    plt.plot(timesteps, losses, color="black", label="Median loss")
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel(f"Loss ({loss_unit})", fontsize=12)
    plt.xlim(xmin=0, xmax=0.4)
    plt.ylim(ymin=0, ymax=ymax)
    plt.title(f"{loss_name} loss of {model_name} on {obj_name}", fontsize=14)
    
    # Add vertical bar at the point where the opponent hits the ball
    opponent_hit_time = abs(context_size) / 100
    plt.axvline(x=opponent_hit_time, color='blue', linestyle='--', label='Opponent hits ball')
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    if not os.path.exists(f"plots/{model_name.replace(' ', '')}/"):
        os.makedirs(f"plots/{model_name.replace(' ', '')}/")
    plt.savefig(f"plots/{model_name.replace(' ', '')}/metric_{model_name}_{obj_name}_{dataset_name} Data_{time.time()}.png", dpi=300)
    plt.clf()
        
def dist_loss(a, b):
    return np.sqrt(np.sum((a - b)**2, axis=-1))

def many_dist_loss(a, b):
    a = a.reshape(a.shape[0], -1, 3)
    b = b.reshape(b.shape[0], -1, 3)
    return np.mean(np.sqrt(np.sum((a - b)**2, axis=-1)), axis=-1)

def pad_pos_loss(a, b):
    a = a[:, :3]
    b = b[:, :3]
    return dist_loss(a, b)
        
def depth_loss(a, b):
    a = a[:, 0:1]
    b = b[:, 0:1]
    return dist_loss(a, b)

def speed_loss(a, b):
    a_speeds = (a[2:]-a[:-2])/2
    b_speeds = (b[2:]-b[:-2])/2
    a_speeds = np.linalg.norm(a_speeds, axis=-1)
    b_speeds = np.linalg.norm(b_speeds, axis=-1)
    a_speeds = np.concatenate(([np.linalg.norm(a[1] - a[0])], a_speeds, [np.linalg.norm(a[-1] - a[-2])]), 0)
    b_speeds = np.concatenate(([np.linalg.norm(b[1] - b[0])], b_speeds, [np.linalg.norm(b[-1] - b[-2])]), 0)
    return abs(a_speeds - b_speeds)

def pad_orientation_loss(q1, q2):
    q1 = q1[:, 3:]
    q2 = q2[:, 3:]
    dot_products = np.einsum('ij,ij->i', q1, q2)
    thetas = np.arccos(2 * dot_products**2 - 1)
    return thetas

# Load data
loader = CombinedDataset("recons", "recons_lab", "recons_test", "recons_lab_test", CONTEXT_WINDOW, val_split=0.05, constant_fps=True)

# MODEL_NAME = "Masked Ball Model"
# DATASET_NAME = "Lab"
# model = load_model('models/masked_ball_model.pth', device)

MODEL_NAME = "Full Model"
DATASET_NAME = "Lab"
model = load_model('models/full_model.pth', device)

# MODEL_NAME = "Lab Model"
# DATASET_NAME = "Lab"
# model = load_model('models/lab_model.pth', device)

# MODEL_NAME = "Finetuned Model"
# DATASET_NAME = "Lab"
# model = load_model('models/finetuned_model.pth', device)

basic_loss_metrics("Ball Speed", "Absolute", "b", speed_loss, 100, DATASET_NAME, MODEL_NAME, ymax=0.25, loss_unit="meters/sec")
basic_loss_metrics("Ball X Position", "Absolute", "b", depth_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.0)
basic_loss_metrics("Ball Position", "Absolute", "b", dist_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.5)
basic_loss_metrics("Rel. Player Joints", "Absolute", "p1", many_dist_loss, 100, DATASET_NAME, MODEL_NAME, ymax=0.2)
basic_loss_metrics("Player Root", "Absolute", "p1_root", dist_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.0)
basic_loss_metrics("Paddle Position", "Absolute", "p1_pad", pad_pos_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.0)
basic_loss_metrics("Paddle Orientation", "Absolute", "p1_pad", pad_orientation_loss, 100, DATASET_NAME, MODEL_NAME, ymax=6)
