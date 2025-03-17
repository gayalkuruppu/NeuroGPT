import os
import numpy as np
import torch
from pathlib import Path

def create_random_eeg_data(output_dir="./synthetic_eeg_data", num_samples=50, sample_length=5000, num_channels=22):
    """
    Creates synthetic EEG data for testing NeuroGPT model.
    
    Args:
        output_dir: Directory to save the generated data
        num_samples: Number of synthetic EEG recordings to create
        sample_length: Length of each recording in time points (must be >= 1000)
        num_channels: Number of EEG channels (default: 22)
    """
    # Ensure we're generating data that meets the minimum length threshold
    if sample_length < 1000:
        print("Warning: Setting sample_length to 1000 to meet training script requirements")
        sample_length = 1000
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Generate random EEG-like data using PyTorch directly
        eeg_data = torch.randn(num_channels, sample_length) * 0.5
        
        # Add some low-frequency oscillations to make it more EEG-like
        t = torch.arange(sample_length).float()
        for ch in range(num_channels):
            # Add alpha-like oscillations (8-12 Hz)
            freq = torch.tensor(np.random.uniform(8, 12)).float()
            amplitude = torch.tensor(np.random.uniform(0.5, 1.5)).float()
            eeg_data[ch] += amplitude * torch.sin(2 * np.pi * freq * t / 250)
            
            # Add some random drift
            # Use cumulative sum to create drift effect
            drift = torch.cumsum(torch.randn(sample_length) * 0.01, dim=0)
            eeg_data[ch] += drift
        
        # IMPORTANT CHANGE: Save only the tensor data directly, not wrapped in a dictionary
        # This matches what load_tensor() expects
        filename = f"synthetic_eeg_sample_{i:03d}.pt"
        torch.save(eeg_data, output_path / filename)
        
    print(f"Created {num_samples} synthetic EEG samples in {output_dir}")
    return output_path

# Update create_synthetic_data_csv.py as well
def create_csv_for_synthetic_data(data_dir="./synthetic_eeg_data", output_csv="./inputs/sub_list2.csv"):
    """
    Creates a CSV file with the correct time_len for the synthetic data
    """
    import pandas as pd
    import glob
    
    # Get all PT files in the directory
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    
    # Create relative paths from the current directory
    file_list = []
    time_lens = []
    
    # Generate the CSV data
    for pt_file in pt_files:
        # Get basename of file
        filename = os.path.basename(pt_file)
        
        # Get actual time length from the tensor file
        try:
            # IMPORTANT CHANGE: Load directly as tensor, not dictionary
            data = torch.load(pt_file)
            time_len = data.shape[1]  # second dimension is time
        except Exception as e:
            print(f"Could not load {pt_file}: {str(e)}")
            # Use the default value from generation
            time_len = 5000
            
        file_list.append(filename)
        time_lens.append(time_len)
    
    # Create dataframe and save to CSV
    df = pd.DataFrame({
        'filename': file_list,
        'time_len': time_lens
    })
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save the CSV
    df.to_csv(output_csv, index=False)
    print(f"Created CSV at {output_csv} with {len(file_list)} files")

if __name__ == "__main__":
    # Create synthetic data
    data_dir = create_random_eeg_data(
        output_dir="./synthetic_eeg_data", 
        num_samples=2000,  # 2000 samples to ensure enough data
        sample_length=5000,  # 5000 is well above the 1000 threshold
        num_channels=22
    )
    
    # Create the CSV with correct time_len values
    create_csv_for_synthetic_data(str(data_dir))
    
    print("\nSynthetic data and CSV created successfully. Run the training script with:")
