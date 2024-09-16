import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import time

from model import TransformerModel
from loaders.loader import CombinedDataset
from loaders.youtube_loader import FORMAT_RANGES, FORMAT_SIZE
from utils.render import MultiSampleVideoRenderer

# Hyperparameters (make sure these match your training settings)
ANTICIPATORY_WINDOW = 8
NUM_SAMPLES = 5
CONTEXT_WINDOW = 16
TOKEN_DIM = FORMAT_SIZE
MODEL_PARAMS = (TOKEN_DIM, 256, 16, 4, 1024)  # d_input, d_model, nhead, num_layers, dim_feedforward
BATCH_SIZE = 1        
    
device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = TransformerModel(*MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_inference(model, sample, length, token_mask, fps, use_mask_on_generation, num_generations=1):
    t = time.time()
    with torch.no_grad():
        generated = model.generate(sample.repeat(num_generations, 1, 1), length, token_mask.repeat(num_generations, 1, 1), np.array([fps]*num_generations), use_mask_on_generation=use_mask_on_generation).detach().cpu()
    print(generated.shape)
    print(f"Generation Time: {time.time() - t}")
    return generated

def main():
    # Set random seed for reproducibility
    random_seed = 8
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load the datasets
    loader = CombinedDataset("recons", "recons_lab", "recons_test", CONTEXT_WINDOW, val_split=0.05)
    
    # Choose a dataset to use for inference
    dataset = loader.test_dataset
    idx = torch.randint(0, len(dataset), (1,)).item() 

    # Load the trained model
    model = load_model('best_model.pth')

    # Get a random sample from the validation set
    sample = dataset[idx]
    sample = sample.unsqueeze(0)
    sample = sample.to(device)
    batch, token_mask = sample[:, :, :TOKEN_DIM], sample[:, :, TOKEN_DIM:]
    
    token_mask *= loader.youtube_mask.to(device)
    # token_mask[:, ANTICIPATORY_WINDOW+4:, FORMAT_RANGES["p1_pad"][0]:FORMAT_RANGES["p1_pad"][1]] = 1.0
    use_mask_on_generation = False
    
    fps = round(batch[0, 0, -1].item() * 120.0)
    
    # Run inference
    generation_length = batch.shape[1]+1
    predicted_sequences = run_inference(model, batch[:, :ANTICIPATORY_WINDOW], generation_length, token_mask, fps, use_mask_on_generation, num_generations=NUM_SAMPLES)
    predicted_pad1, predicted_pad2 = [], []
    
    if use_mask_on_generation:
        batch = batch * token_mask
    predicted_sequences = torch.concat((batch, predicted_sequences), dim=0)  # ground truth
    
    predicted_sequences = [ seq for seq in predicted_sequences ]
    
    # Convert tensors to appropriate shape for rendering
    input_sequence = batch.squeeze(0) # Remove batch dimension
    for i in range(len(predicted_sequences)):
        pred_scene, pred_pad1, pred_pad2 = dataset.return_to_raw(predicted_sequences[i], fps=fps) 
        predicted_sequences[i] = pred_scene
        predicted_pad1.append(pred_pad1)
        predicted_pad2.append(pred_pad2)
        
    # Renderings
    print(f"FPS: {fps}")
    MultiSampleVideoRenderer(predicted_sequences, predicted_pad1, predicted_pad2).render()

if __name__ == "__main__":
    main()