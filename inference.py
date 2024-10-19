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

def load_model(model_path, device):
    model = TransformerModel(*MODEL_PARAMS).to(device)
    try: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def run_inference(model, context, generation_length, token_mask, fps, use_mask_on_generation=True, num_gens_per_sample=1, verbose=True):
    t = time.time()
    with torch.no_grad():
        generated = model.generate(context.repeat(num_gens_per_sample, 1, 1), generation_length, token_mask.repeat(num_gens_per_sample, 1, 1), np.array([fps]*num_gens_per_sample), use_mask_on_generation=use_mask_on_generation).detach()
    if verbose: print(f"Generation Time: {time.time() - t}")
    return generated

def generate(model, dataset, idx, mask, context_size, device, num_gens_per_sample, use_mask_on_generation=True, verbose=True, hit_time_idx=None):    
    # Get a random sample from the dataset
    sample = dataset[idx]
    sample = sample.unsqueeze(0)
    sample = sample.to(device)
    batch, token_mask = sample[:, :, :TOKEN_DIM], sample[:, :, TOKEN_DIM:]
    fps = round(batch[0, 0, -1].item() * 100.0)
    generation_length = batch.shape[1]
        
    # Apply the mask
    token_mask *= mask.to(device)
    
    if context_size < 0:
        hit_times = dataset.hit_times[idx]
        if hit_time_idx is not None:
            ht = hit_times[2*hit_time_idx]
            generation_length = round(hit_times[2*hit_time_idx+1] / (100 / fps)) + 1
        else:
            ht = hit_times
        f = 100/fps
        start_time = max(ht + context_size * f, 4)
        start_idx  = round(start_time / f)
        
    # Run inference
    predicted_sequences = run_inference(model, batch[:, :start_idx], generation_length+1, token_mask, fps, use_mask_on_generation=use_mask_on_generation, num_gens_per_sample=num_gens_per_sample, verbose=verbose)
    if use_mask_on_generation:
        batch = batch * token_mask
    predicted_sequences = torch.concat((batch[:, :generation_length], predicted_sequences), dim=0)  # ground truth
    predicted_sequences = predicted_sequences.cpu()
    
    return predicted_sequences, fps, start_idx

def main():
    # Set random seed for reproducibility
    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt_size, num_gens_per_sample = -5, 1
    loader = CombinedDataset("data/recons", "data/recons_lab", "data/recons_extreme", "data/recons_lab_test", val_split=0.05, constant_fps=True)
    
    models = []
    ens_number, num_models = 2, 5
    for i in range(num_models):
        model = load_model(f'models/ensemble{ens_number}/best_model_{i}.pth', device)
        models.append(model)
    
    dataset = loader.youtube_test_dataset
    mask = loader.youtube_mask
    idx = 4 # torch.randint(0, len(dataset), (1,)).item() 
    print((len(dataset.hit_times[idx]) // 2) - 1)
    hit_time = 1 # (len(dataset.hit_times[idx]) // 2) - 1  # torch.randint(0, len(dataset.hit_times[idx]) // 2, (1,)).item()
    
    preds = []
    for model in models:
        predicted_sequences, fps, preds_start_index = generate(
            model, 
            dataset, 
            idx,
            mask, 
            prompt_size, 
            device,
            num_gens_per_sample, 
            use_mask_on_generation=False,
            hit_time_idx=hit_time
        )
        if len(preds) == 0:
            preds.append(predicted_sequences)
        else:
            preds.append(predicted_sequences[1:])
    predicted_sequences = torch.concat(preds, dim=0)
    
    # Convert tensors to appropriate shape for rendering
    predicted_sequences, predicted_pad1, predicted_pad2 = [seq for seq in predicted_sequences], [], []
    for i in range(len(predicted_sequences)):
        pred_scene, pred_pad1, pred_pad2 = dataset.return_to_raw(predicted_sequences[i], fps=fps) 
        predicted_sequences[i] = pred_scene
        predicted_pad1.append(pred_pad1)
        predicted_pad2.append(pred_pad2)
    
    # Renderings
    renderer = MultiSampleVideoRenderer(predicted_sequences, predicted_pad1, predicted_pad2, preds_start_index)
    renderer.save()
    renderer.render()

if __name__ == "__main__":
    main()
    
    
    
