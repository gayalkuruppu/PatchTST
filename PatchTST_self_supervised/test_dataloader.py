import time
import torch
import sys
from datautils import get_dls
from tqdm import tqdm

# Define basic parameters for testing
class Params:
    dset = 'tuh_test'
    context_points = 25000 #384
    target_points = 0 #96
    batch_size = 64
    num_workers = 8
    features = 'M'

def test_dataloader_speed(batch_size=64, num_workers=8, max_batches=20):
    """Test the speed of the TUH dataloader with configurable parameters"""
    # Create parameters with specific batch size and workers
    params = Params()
    params.batch_size = batch_size
    params.num_workers = num_workers
    
    # Time dataloader creation
    print(f"Creating dataloader with batch_size={batch_size}, num_workers={num_workers}")
    start_time = time.time()
    dls = get_dls(params)
    creation_time = time.time() - start_time
    print(f"Dataloader creation time: {creation_time:.2f} seconds")
    
    # Print dataset information
    print(f"Dataset dimensions: {dls.vars} variables, {dls.len} context points")
    print(f"Number of batches in train: {len(dls.train)}")
    if dls.valid:
        print(f"Number of batches in validation: {len(dls.valid)}")
    if dls.test:
        print(f"Number of batches in test: {len(dls.test)}")
    
    # Time batch loading for training set
    print("\nTesting training dataloader speed:")
    test_batch_loading(dls.train, max_batches=max_batches)
    
    # Time batch loading for validation set if available
    if dls.valid:
        print("\nTesting validation dataloader speed:")
        test_batch_loading(dls.valid, max_batches=max_batches)
    
    return dls

def test_batch_loading(dataloader, max_batches=20):
    """Test the speed of loading batches from a dataloader"""
    batch_times = []
    memory_usage = []
    
    print(f"Testing {min(max_batches, len(dataloader))} batches")
    
    # Use a subset of batches for quick testing
    for i, batch in enumerate(tqdm(dataloader)):
        if i >= max_batches:
            break
            
        # Time the batch loading
        start_time = time.time()
        x, y = batch
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        # Get memory usage of tensors
        batch_memory = (x.element_size() * x.nelement() + 
                        y.element_size() * y.nelement()) / (1024 * 1024)  # MB
        memory_usage.append(batch_memory)
        
        # Print information about the first batch
        if i == 0:
            print(f"First batch shapes - X: {x.shape}, Y: {y.shape}")
            print(f"Data types - X: {x.dtype}, Y: {y.dtype}")
            print(f"Memory usage: {batch_memory:.2f} MB")
    
    # Calculate statistics 
    if batch_times:
        avg_time = sum(batch_times) / len(batch_times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        total_samples = min(max_batches, len(dataloader)) * dataloader.batch_size
        samples_per_sec = total_samples / sum(batch_times)
        
        print(f"Average batch loading time: {avg_time:.4f} seconds")
        print(f"Average batch memory usage: {avg_memory:.2f} MB")
        print(f"Samples per second: {samples_per_sec:.2f}")

def profile_different_configurations():
    """Test different dataloader configurations to find optimal settings"""
    configs = [
        {"batch_size": 32, "num_workers": 4},
        {"batch_size": 64, "num_workers": 8},
        {"batch_size": 128, "num_workers": 16},
    ]
    
    results = []
    for config in configs:
        print(f"\n\n===== Testing with batch_size={config['batch_size']}, num_workers={config['num_workers']} =====")
        dls = test_dataloader_speed(**config, max_batches=10)
        # You could collect and return performance metrics here

if __name__ == "__main__":
    print("Testing TUH dataloader performance")
    
    # Basic test with default settings
    dls = test_dataloader_speed()
    
    # Uncomment to test multiple configurations
    # profile_different_configurations()