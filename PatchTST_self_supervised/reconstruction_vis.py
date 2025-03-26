import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse

from src.models.patchTST import PatchTST
from src.callback.patch_mask import create_patch, random_masking
from datautils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type=str, default='tuh_2000', help='dataset name')
    parser.add_argument('--context_points', type=int, default=25000, help='sequence length')
    parser.add_argument('--patch_len', type=int, default=2500, help='patch length')
    parser.add_argument('--stride', type=int, default=25000, help='stride between patch')
    parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--pretrained_model', type=str, default='', help='path to pretrained model')
    
    args = parser.parse_args()
    return args

def visualize_reconstruction(model, data, patch_len, stride, mask_ratio):
    """
    Visualize original signal, masked patches and reconstruction
    """
    # Create a copy of the data for visualization
    x_orig = data.clone()
    
    # Apply patching and masking using functions from patch_mask.py
    # First create patches
    xb_patch, num_patch = create_patch(x_orig, patch_len, stride)  # xb_patch: [bs x num_patch x n_vars x patch_len]
    
    # Then apply random masking
    x_masked, _, mask, _ = random_masking(xb_patch, mask_ratio)  # x_masked: [bs x num_patch x n_vars x patch_len]
    mask = mask.bool()  # Convert to boolean mask: [bs x num_patch x n_vars]
    
    # Get reconstruction from model
    with torch.no_grad():
        reconstruction = model(x_masked)
    
    # Rest of your visualization code stays the same...
    # Convert tensors to numpy for plotting
    x_orig_np = xb_patch[0].detach().cpu().numpy()  # First sample in batch (original patches)
    x_masked_np = x_masked[0].detach().cpu().numpy()
    reconstruction_np = reconstruction[0].detach().cpu().numpy()
    mask_np = mask[0].detach().cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    
    # Get dimensions
    num_patches = x_orig_np.shape[0]
    n_vars = x_orig_np.shape[1]
    
    # Choose a variable to visualize if multivariate
    var_idx = 0  # Visualize first variable
    
    # Reconstruct the full sequence from patches
    seq_len = num_patches * patch_len
    orig_seq = np.zeros(seq_len)
    masked_seq = np.zeros(seq_len)
    recon_seq = np.zeros(seq_len)
    
    # Fill in the sequences
    for i in range(num_patches):
        start_idx = i * patch_len
        end_idx = start_idx + patch_len
        orig_seq[start_idx:end_idx] = x_orig_np[i, var_idx]
        
        if not mask_np[i, var_idx]:  # Unmasked patch (mask=0 means keep)
            masked_seq[start_idx:end_idx] = x_masked_np[i, var_idx]
        
        # Fill in reconstructed sequence
        recon_seq[start_idx:end_idx] = reconstruction_np[i, var_idx]
    
    # Plot original
    axes[0].plot(orig_seq, 'b-', label='Original Signal')
    axes[0].set_title('Original Signal')
    axes[0].legend()
    
    # Plot masked
    axes[1].plot(orig_seq, 'b-', alpha=0.3, label='Original Signal')
    
    # Highlight masked regions
    for i in range(num_patches):
        if mask_np[i, var_idx]:  # Masked patch (mask=1 means remove)
            start_idx = i * patch_len
            end_idx = start_idx + patch_len
            axes[1].axvspan(start_idx, end_idx, color='r', alpha=0.3)
    axes[1].plot(masked_seq, 'g-', label='Visible Signal')
    axes[1].set_title('Masked Signal (Red regions are masked)')
    axes[1].legend()
    
    # Plot reconstruction compared to original
    axes[2].plot(orig_seq, 'b-', label='Original Signal')
    
    # For each patch that was masked, plot the reconstruction
    for i in range(num_patches):
        if mask_np[i, var_idx]:  # Masked patch (was reconstructed)
            start_idx = i * patch_len
            end_idx = start_idx + patch_len
            axes[2].plot(range(start_idx, end_idx), 
                        recon_seq[start_idx:end_idx], 
                        'r-', linewidth=2)
    
    axes[2].set_title('Original vs Reconstructed (Red shows reconstruction of masked patches)')
    axes[2].legend(['Original', 'Reconstruction'])
    
    plt.tight_layout()
    plt.savefig('patchtst_reconstruction.png')
    plt.show()
    
    print(f"Visualization saved to 'patchtst_reconstruction.png'")

def main():
    args = get_args()
    
    # Load sample data
    print("Loading data...")
    args.dset = args.dset
    args.target_points = 0  # Not used in reconstruction
    args.features = 'M'
    args.num_workers = 1
    dls = get_dls(args)
    
    # Get a batch of data
    batch = next(iter(dls.valid))
    x_batch = batch[0]  # Get input tensor
    
    # Calculate num_patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1
    
    # Create model with same config as pretrained
    print("Loading model...")
    model = PatchTST(
        c_in=dls.vars,
        target_dim=96,  # Not used for pretrain anyway
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=3,
        d_model=128,
        n_heads=16,
        d_ff=512,
        head_dropout=0.2,
        dropout=0.2,
        head_type='pretrain'
    )
    
    # Load pretrained weights if provided
    if args.pretrained_model:
        model_path = args.pretrained_model
    else:
        # Use default path pattern from training script
        model_path = f'saved_models/{args.dset}/masked_patchtst/based_model/patchtst_pretrained_cw{args.context_points}_patch{args.patch_len}_stride{args.stride}_epochs-pretrain10_mask{args.mask_ratio}_model1.pth'
    
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Warning: Pretrained model not found at {model_path}. Using random weights.")
    
    # Visualize
    print("Generating visualization...")
    visualize_reconstruction(model, x_batch, args.patch_len, args.stride, args.mask_ratio)

if __name__ == "__main__":
    main()
