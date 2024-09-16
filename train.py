import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from model import TransformerModel, generate_square_subsequent_mask
from loaders.loader import CombinedDataset, collate_fn
from loaders.youtube_loader import FORMAT_RANGES, FORMAT_SIZE

# Hyperparameters
CONTEXT_WINDOW = 32
TOKEN_DIM = FORMAT_SIZE
MODEL_PARAMS = (TOKEN_DIM, 512, 16, 4, 1024)  # d_input, d_model, nhead, num_layers, dim_feedforward
INIT_LR = 5e-4
FINAL_LR = 5e-6
NUM_EPOCHS = 200
BATCH_SIZE = 512 
RANDOM_MASK_PROB = 0.5

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def loss_fn(output, target, weight=5):   
    output[:, FORMAT_RANGES["p1_pad"][0]:FORMAT_RANGES["p1_pad"][1]] = weight * output[:, FORMAT_RANGES["p1_pad"][0]:FORMAT_RANGES["p1_pad"][1]]
    output[:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]           = weight * output[:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]
    target[:, FORMAT_RANGES["p1_pad"][0]:FORMAT_RANGES["p1_pad"][1]] = weight * target[:, FORMAT_RANGES["p1_pad"][0]:FORMAT_RANGES["p1_pad"][1]]
    target[:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]           = weight * target[:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]
    return F.mse_loss(output, target)

def train(rank, world_size):
    print(f"Running on rank {rank}.")
    setup(rank, world_size)

    # Load data
    loader = CombinedDataset("recons", "recons_lab", "recons_test", CONTEXT_WINDOW, val_split=0.05)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        loader.train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        loader.train_dataset, 
        batch_size=BATCH_SIZE//world_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=collate_fn
    ) 
    val_loader = torch.utils.data.DataLoader(
        loader.val_dataset, 
        batch_size=BATCH_SIZE//world_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Load model
    model = TransformerModel(*MODEL_PARAMS).to(rank)
    model = DDP(model, device_ids=[rank])

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

    # Learning rate scheduler for linear decay
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=FINAL_LR/INIT_LR, total_iters=NUM_EPOCHS)

    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_samples = 0
        train_loader.sampler.set_epoch(epoch)

        for batch, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch", disable=rank!=0):
            batch, token_mask = batch[:, :, :TOKEN_DIM], batch[:, :, TOKEN_DIM:]
            if np.random.rand() < RANDOM_MASK_PROB:
                x = batch * loader.random_mask
            else:
                x = batch
        
            batch = batch.to(rank)
            x = x.to(rank)
            
            fps = x[:, 0, -1] * 120.0
            token_mask = token_mask.to(rank)
            future_mask = generate_square_subsequent_mask(x.shape[1]).to(rank)
            
            optimizer.zero_grad()
            y = model(x, mask=future_mask, token_mask=token_mask, fps=fps)
            
            lengths_mask = torch.arange(y.size(1)).unsqueeze(1) < lengths.unsqueeze(0)
            y, target, token_mask, lengths_mask = y[:, :-1], batch[:, 1:], token_mask[:, 1:], lengths_mask[1:].permute(1, 0)
            y = y*token_mask
            target = target*token_mask
            y = y[lengths_mask]
            target = target[lengths_mask]
            
            loss = loss_fn(y, target)
            loss.backward()
            
            optimizer.step()        
            
            train_loss += loss.item()
            train_samples += 1
        
        train_loss /= train_samples
        train_losses.append(train_loss)
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for batch, lengths in val_loader:
                batch, token_mask = batch[:, :, :TOKEN_DIM], batch[:, :, TOKEN_DIM:]
                if np.random.rand() < RANDOM_MASK_PROB:
                    x = batch * loader.random_mask
                else:
                    x = batch
                
                batch = batch.to(rank)
                x = x.to(rank)
                
                fps = x[:, 0, -1] * 120.0
                token_mask = token_mask.to(rank)
                future_mask = generate_square_subsequent_mask(x.shape[1]).to(rank)
                y = model(x, mask=future_mask, token_mask=token_mask, fps=fps)
                
                lengths_mask = torch.arange(y.size(1)).unsqueeze(1) < lengths.unsqueeze(0)
                y, target, token_mask, lengths_mask = y[:, :-1], batch[:, 1:], token_mask[:, 1:], lengths_mask[1:].permute(1, 0)
                y = y*token_mask
                target = target*token_mask
                y = y[lengths_mask]
                target = target[lengths_mask]
                
                loss = loss_fn(y, target)
                
                val_loss += loss.item()
                val_samples += 1
        
        val_loss /= val_samples
        val_losses.append(val_loss)
        
        if rank == 0:
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), 'best_model.pth')
                print(f"New best model saved with validation loss: {best_val_loss:.6f}")
        
            # Save the model
            torch.save(model.module.state_dict(), 'curr_model.pth')
        
            # Update learning rate
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
        
            plt.clf()
            plt.plot(train_losses, label="train error")
            plt.plot(val_losses, label="validation error")
            plt.legend()
            plt.savefig("track.png")
        
    if rank == 0:
        print("Training completed.")
    
    cleanup()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)