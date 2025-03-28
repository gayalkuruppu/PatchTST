import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from safetensors.torch import load_file
from transformers import AutoConfig
import json
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    set_seed,
    Trainer,
    TrainingArguments,
    PatchTSTForPretraining
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='path to HuggingFace checkpoint directory')
    parser.add_argument('--data_path', type=str, default='/mnt/ssd_4tb_0/data/tuh_preprocessed_npy/aaaaaqrm_s001_t000_preprocessed.npy',
                        help='path to input data file (.pt or .npy)')
    parser.add_argument('--patch_len', type=int, default=2500, 
                        help='patch length')
    parser.add_argument('--stride', type=int, default=2500, 
                        help='stride between patches')
    parser.add_argument('--mask_ratio', type=float, default=0.4, 
                        help='masking ratio')
    parser.add_argument('--output_file', type=str, default='hf_reconstruction.png',
                        help='output file for visualization')
    parser.add_argument('--channel_idx', type=int, default=0,
                        help='channel index to visualize (for multivariate data)')
    
    args = parser.parse_args()
    return args

def create_patch_manual(x, patch_len, stride):
    """Manually create patches from a sequence"""
    # x: [batch_size, seq_len, n_vars]
    batch_size, seq_len, n_vars = x.shape
    
    # Calculate number of patches
    num_patch = (seq_len - patch_len) // stride + 1
    
    # Initialize output
    patches = []
    
    # Create patches
    for i in range(num_patch):
        start_idx = i * stride
        end_idx = start_idx + patch_len
        patch = x[:, start_idx:end_idx, :]  # [batch_size, patch_len, n_vars]
        patch = patch.permute(0, 2, 1)  # [batch_size, n_vars, patch_len]
        patches.append(patch)
    
    # Stack patches
    patches = torch.stack(patches, dim=1)  # [batch_size, num_patch, n_vars, patch_len]
    
    return patches, num_patch

def random_masking_manual(x, mask_ratio):
    """Apply random masking to patches"""
    # x: [batch_size, num_patch, n_vars, patch_len]
    batch_size, num_patch, n_vars, patch_len = x.shape
    
    # Create mask
    mask = torch.rand(batch_size, num_patch, n_vars) < mask_ratio  # True = mask
    
    # Create masked input
    x_masked = x.clone()
    
    # Apply masking
    for b in range(batch_size):
        for p in range(num_patch):
            for v in range(n_vars):
                if mask[b, p, v]:
                    # Replace with zeros (or another masking strategy)
                    x_masked[b, p, v] = torch.zeros_like(x_masked[b, p, v])
    
    return x_masked, mask

def visualize_reconstruction(original_data, reconstructed_data, mask, patch_len, output_file, channel_idx=0):
    """Visualize original signal, masked regions, and reconstruction"""
    # original_data: [batch_size, num_patch, n_vars, patch_len]
    # reconstructed_data: [batch_size, num_patch, n_vars, patch_len]
    # mask: [batch_size, num_patch, n_vars] (True = masked)
    
    # Convert tensors to numpy for plotting
    x_orig_np = original_data[0].detach().cpu().numpy()  # First sample in batch
    recon_np = reconstructed_data[0].detach().cpu().numpy()
    mask_np = mask[0].detach().cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    
    # Get dimensions
    num_patches = x_orig_np.shape[0]
    
    # Channel/variable to visualize
    var_idx = channel_idx
    
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
        
        if not mask_np[i, var_idx]:  # Unmasked patch
            masked_seq[start_idx:end_idx] = x_orig_np[i, var_idx]
        
        # Fill in reconstructed sequence
        recon_seq[start_idx:end_idx] = recon_np[i, var_idx]
    
    # Plot original
    axes[0].plot(orig_seq, 'b-', label='Original Signal')
    axes[0].set_title('Original Signal')
    axes[0].legend()
    
    # Plot masked
    axes[1].plot(orig_seq, 'b-', alpha=0.3, label='Original Signal')
    
    # Highlight masked regions
    for i in range(num_patches):
        if mask_np[i, var_idx]:  # Masked patch
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
    plt.savefig(output_file)
    print(f"Visualization saved to '{output_file}'")
    plt.show()

def load_huggingface_model(checkpoint_path):
    """Load a model from a HuggingFace checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Load model config
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config: {config.keys()}")
    else:
        print("Config file not found!")
        return None
    
    # Load model weights
    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
        print(f"Loaded weights with {len(state_dict)} parameters")
        print(f"Sample keys: {list(state_dict.keys())[:5]}")
    else:
        print("Model weights file not found!")
        return None
    
    # For now, return the state_dict to manually initialize the model
    # In a real implementation, you would create the actual model here
    return state_dict, config

def load_data(data_path):
    """Load input data from file"""
    if data_path.endswith('.pt'):
        data = torch.load(data_path)
    elif data_path.endswith('.npy'):
        data = torch.from_numpy(np.load(data_path))
    else:
        raise ValueError(f"Unsupported data file format: {data_path}")
    
    # Ensure proper shape: [batch_size, seq_len, n_vars]
    if len(data.shape) == 2:
        # Add batch dimension if missing
        data = data.unsqueeze(0)
    
    if len(data.shape) == 3 and data.shape[1] > data.shape[2]:
        # Assume [batch_size, seq_len, n_vars]
        pass
    elif len(data.shape) == 3:
        # Assume [batch_size, n_vars, seq_len], transpose to [batch_size, seq_len, n_vars]
        data = data.transpose(1, 2)
    
    return data

def main():
    # Parse arguments
    args = get_args()
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_data(args.data_path)
    print(f"Data shape: {data.shape}")
    
    # # Create patches
    patches, num_patch = create_patch_manual(data, args.patch_len, args.stride)
    print(f"Created {num_patch} patches with shape {patches.shape}")
    first_seq = data[:, 2500:5000, 0]

    # plot the first sequence
    plt.plot(first_seq[0])
    plt.savefig("first_seq.png")
    plt.close()
    
    # Apply masking
    masked_patches, mask = random_masking_manual(patches, args.mask_ratio)
    print(f"Applied masking with ratio {args.mask_ratio}")
    
    # Load HuggingFace model
    state_dict, config = load_huggingface_model(args.checkpoint_path)
    
    # get reconstructions
    # Initialize model with same config as pretrained
    context_length = 500
    patch_length = 25
    patch_stride = 25
    print("Loading model...")
    config = PatchTSTConfig(
        num_input_channels=19, #6,
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        mask_type='random',
        random_mask_ratio=0.4,
        use_cls_token=True,
        # prediction_length=forecast_horizon,
    )
    model = PatchTSTForPretraining(config)

    # Load pretrained weights if provided
    if state_dict:
        print(f"Loading pretrained model from HuggingFace checkpoint")
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Pretrained model not found. Using random weights.")

    # Get reconstruction
    with torch.no_grad():
        reconstruction = model(masked_patches)
    
    # Visualize
    visualize_reconstruction(
        patches, reconstruction, mask, 
        args.patch_len, args.output_file, args.channel_idx
    )

if __name__ == "__main__":
    main()
