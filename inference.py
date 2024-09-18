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
TOKEN_DIM = FORMAT_SIZE
MODEL_PARAMS = (TOKEN_DIM, 256, 16, 4, 1024)  # d_input, d_model, nhead, num_layers, dim_feedforward
CONTEXT_WINDOW = 16
    
def load_model(model_path, device):
    model = TransformerModel(*MODEL_PARAMS).to(device)
    try: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def run_inference(model, context, generation_length, token_mask, fps, use_mask_on_generation=True, num_gens_per_sample=1):
    t = time.time()
    with torch.no_grad():
        generated = model.generate(context.repeat(num_gens_per_sample, 1, 1), generation_length, token_mask.repeat(num_gens_per_sample, 1, 1), np.array([fps]*num_gens_per_sample), use_mask_on_generation=use_mask_on_generation).detach().cpu()
    print(f"Generation Time: {time.time() - t}")
    return generated

def generate(model, dataset, mask, context_size, device, num_gens_per_sample, use_mask_on_generation=True):    
    # Get a random sample from the dataset
    idx = torch.randint(0, len(dataset), (1,)).item() 
    sample = dataset[idx]
    sample = sample.unsqueeze(0)
    sample = sample.to(device)
    batch, token_mask = sample[:, :, :TOKEN_DIM], sample[:, :, TOKEN_DIM:]
    fps = round(batch[0, 0, -1].item() * 100.0)
        
    # Apply the mask
    token_mask *= mask.to(device)
    
    if context_size < 0:
        hit_time = max(dataset.hit_times[idx] + context_size, 5)
        hit_time = round(hit_time / (100 / fps))
        context_size = hit_time
    
    # Run inference
    generation_length = batch.shape[1]+1
    predicted_sequences = run_inference(model, batch[:, :context_size], generation_length, token_mask, fps, use_mask_on_generation=use_mask_on_generation, num_gens_per_sample=num_gens_per_sample)
    if use_mask_on_generation:
        batch = batch * token_mask
    predicted_sequences = torch.concat((batch, predicted_sequences), dim=0)  # ground truth
    
    return predicted_sequences, fps

def main():
    # Set random seed for reproducibility
    random_seed = 8
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context_size, num_gens_per_sample = -15, 1
    loader = CombinedDataset("recons", "recons_lab", "recons_test", "recons_lab_test", CONTEXT_WINDOW, val_split=0.05)
    model = load_model('models/lab_model.pth', device)
    
    dataset = loader.lab_test_dataset
    mask = loader.lab_mask
    
    predicted_sequences, fps = generate(
        model, 
        dataset, 
        mask, 
        context_size, 
        device,
        num_gens_per_sample, 
        use_mask_on_generation=False
    )
    
    # Convert tensors to appropriate shape for rendering
    predicted_sequences, predicted_pad1, predicted_pad2 = [seq for seq in predicted_sequences], [], []
    for i in range(len(predicted_sequences)):
        pred_scene, pred_pad1, pred_pad2 = dataset.return_to_raw(predicted_sequences[i], fps=fps) 
        predicted_sequences[i] = pred_scene
        predicted_pad1.append(pred_pad1)
        predicted_pad2.append(pred_pad2)
    
    # Renderings
    MultiSampleVideoRenderer(predicted_sequences, predicted_pad1, predicted_pad2).render()

if __name__ == "__main__":
    main()