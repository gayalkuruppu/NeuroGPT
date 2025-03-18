#!/usr/bin/env python3
# filepath: /home/gayal/ssl-analyses-repos/NeuroGPT/preprocess_tuh.py
import os
import glob
import numpy as np
import pandas as pd
import torch
import mne
from mne.channels import make_standard_montage
from scipy import signal
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
import logging
import time
from datetime import datetime

from src.eeg_utils import map_tuh_to_standard_channels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('eeg_preprocessing')

warnings.filterwarnings("ignore", category=RuntimeWarning)

def preprocess_eeg(file_path, output_dir, channels_to_use=None):
    """
    Preprocess a single EEG file and save as PT file.
    
    Args:
        file_path: Path to the EDF file
        output_dir: Directory to save the preprocessed file
        channels_to_use: List of channels to keep (if None, will use default 22 channels)
    """
    logger.debug(f"Starting preprocessing of: {file_path}")
    
    # Define standard channel names we want to use (10-20 system)
    if channels_to_use is None:
        # channels_to_use = [
        #     'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
        #     'T1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T2',   #'T1', 'T2'
        #     'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2' # 'OZ'
        # ]
        # channels_to_use = [
        #     'EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
        #     'EEG T1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG T2-REF',
        #     'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG OZ-REF', 'EEG O2-REF'
        # ]
        channels_to_use = [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
            'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF',
            'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG O2-REF'
        ]
    
    # Read EDF file
    try:
        logger.debug(f"Reading EDF file: {file_path}")
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        logger.debug(f"Successfully read EDF file: {file_path}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None
    
    # Get all available channel names and standardize them
    available_channels = [ch.upper() for ch in raw.ch_names]
    logger.debug(f"Available channels: {available_channels}")
    
    # Select only channels that we want to use and are available
    channels_present = [ch for ch in channels_to_use if ch in available_channels]
    logger.debug(f"Selected channels: {channels_present} ({len(channels_present)}/{len(channels_to_use)})")
    
    # If we don't have enough channels, skip this file
    if len(channels_present) < len(channels_to_use):  # All the channels should be present
        logger.warning(f"Not enough channels in {file_path}, skipping... (Found {len(channels_present)}/{len(channels_to_use)})")
        logger.debug(f"Missing channels: {set(channels_to_use) - set(channels_present)}")
        return None
    
    # Pick only the channels we want to use
    try:
        logger.debug(f"Picking channels: {[ch.upper() for ch in channels_present]}")
        raw.pick([ch.upper() for ch in channels_present])
        logger.debug("Successfully picked channels")
    except Exception as e:
        logger.error(f"Error picking channels in {file_path}: {e}")
        return None
    
    # Create a standard 10-20 montage
    logger.debug("Setting up 10-20 montage")
    montage = make_standard_montage('standard_1020')

    # Map TUH channel names to standard 10-20 names
    logger.debug(f"Original channels: {raw.ch_names}")
    raw = map_tuh_to_standard_channels(raw)
    logger.debug(f"Renamed channels: {raw.ch_names}")
        
    # Now apply the montage
    raw.set_montage(montage, match_case=False)
    
    # Detect bad channels (completely flat or missing signal)
    logger.debug("Detecting bad channels")
    flat_threshold = 1e-7
    bad_channels = []
    data = raw.get_data()
    
    for i, ch_name in enumerate(raw.ch_names):
        if np.std(data[i]) < flat_threshold or np.all(data[i] == 0):
            bad_channels.append(ch_name)
    
    if bad_channels:
        logger.info(f"Found {len(bad_channels)} bad channels in {file_path}: {bad_channels}")
        raw.info['bads'] = bad_channels
        
        # Interpolate bad channels
        try:
            logger.debug(f"Interpolating bad channels: {bad_channels}")
            raw.interpolate_bads(reset_bads=True)
            logger.debug("Successfully interpolated bad channels")
        except Exception as e:
            logger.warning(f"Error interpolating bad channels in {file_path}: {e}")
            logger.debug("Setting bad channel data to zeros instead")
            # If interpolation fails, we'll just set the bad channels to zeros
            for ch in bad_channels:
                idx = raw.ch_names.index(ch)
                data[idx] = np.zeros_like(data[idx])
            raw._data = data
    
    # Re-reference to average
    logger.debug("Re-referencing to average")
    raw.set_eeg_reference('average', projection=False)
    
    # Apply notch filter to remove power line noise (60 Hz)
    logger.debug("Applying 60Hz notch filter")
    raw.notch_filter(freqs=[60], picks='eeg')
    
    # Apply bandpass filter (0.5 - 100 Hz)
    logger.debug("Applying bandpass filter (0.5-100 Hz)")
    raw.filter(l_freq=0.5, h_freq=100, picks='eeg')
    
    # Resample to 250 Hz
    logger.debug(f"Resampling to 250 Hz (from {raw.info['sfreq']} Hz)")
    raw.resample(250)
    
    # Get the data
    data = raw.get_data()
    logger.debug(f"Data shape after preprocessing: {data.shape}")
    
    # DC offset correction (remove mean from each channel)
    logger.debug("Applying DC offset correction")
    data = data - np.mean(data, axis=1, keepdims=True)
    
    # Remove linear trend from **each channel**
    logger.debug("Removing linear trends")
    for i in range(data.shape[0]):
        data[i] = signal.detrend(data[i])
    
    # Z-transform along time dimension
    logger.debug("Applying Z-transform")
    for i in range(data.shape[0]):
        std = np.std(data[i])
        if std > 0:  # Avoid division by zero
            data[i] = (data[i] - np.mean(data[i])) / std
    
    # Make sure we have exactly 22 channels
    original_channel_count = data.shape[0]
    if data.shape[0] < 22:
        # If we have fewer channels, we'll pad with zeros
        logger.debug(f"Padding channels: {data.shape[0]} -> 22")
        padded_data = np.zeros((22, data.shape[1]))
        padded_data[:data.shape[0]] = data
        data = padded_data
    elif data.shape[0] > 22:
        # If we have more channels, we'll take only the first 22
        logger.debug(f"Truncating channels: {data.shape[0]} -> 22")
        data = data[:22]
    
    # Create output filename and path
    base_name = os.path.basename(file_path)
    output_name = base_name.replace('.edf', '_preprocessed.pt')
    output_path = os.path.join(output_dir, output_name)
    
    # Save as PyTorch tensor
    logger.debug(f"Saving preprocessed data to {output_path}")
    torch.save(torch.FloatTensor(data), output_path)
    
    logger.info(f"Successfully preprocessed {file_path} -> {output_path} (Original channels: {original_channel_count}, Time points: {data.shape[1]})")
    
    # Return output path and time length
    return output_path, data.shape[1]

def find_edf_files(data_dir):
    """Find all EDF files in the dataset."""
    logger.info(f"Searching for EDF files in {data_dir}...")
    edf_files = glob.glob(os.path.join(data_dir, '**', '*.edf'), recursive=True)
    logger.info(f"Found {len(edf_files)} EDF files")
    return edf_files

def create_csv_file(processed_files, output_csv):
    """Create a CSV file with the filenames and time lengths."""
    logger.info(f"Creating CSV file at {output_csv}")
    df = pd.DataFrame(processed_files, columns=['filepath', 'time_len'])
    df['filename'] = df['filepath'].apply(os.path.basename)
    df = df[['filename', 'time_len']]
    
    # Log some statistics
    logger.info(f"Time length statistics:")
    logger.info(f"- Min: {df['time_len'].min()}")
    logger.info(f"- Max: {df['time_len'].max()}")
    logger.info(f"- Mean: {df['time_len'].mean():.2f}")
    logger.info(f"- Median: {df['time_len'].median()}")
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Created CSV file with {len(df)} files at {output_csv}")
    return df

def main():
    parser = argparse.ArgumentParser(description='Preprocess TUH EEG dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with the TUH EEG dataset')
    parser.add_argument('--output-dir', type=str, default='./preprocessed_eeg_data',
                        help='Directory to save preprocessed data')
    parser.add_argument('--csv-path', type=str, default='./inputs/sub_list2.csv',
                        help='Path to save the CSV file')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (default: all)')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log_path = f"./logs/preprocess_{timestamp}.log"
    parser.add_argument('--log-file', type=str, default=default_log_path,
                        help='Path to save log file (default: logs/preprocess_timestamp.log)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose (debug) logging')
    
    args = parser.parse_args()
    
    # Always ensure logs directory exists
    logs_dir = "./logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Configure file logging if specified
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {args.log_file}")
    
    # Set log level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Log start time and arguments
    start_time = time.time()
    logger.info(f"Starting preprocessing with args: {vars(args)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    
    # Find all EDF files
    edf_files = find_edf_files(args.data_dir)
    
    if args.max_files:
        logger.info(f"Limiting processing to {args.max_files} files")
        edf_files = edf_files[:args.max_files]
    
    logger.info(f"Starting preprocessing of {len(edf_files)} files...")
    
    # Process each file
    processed_files = []
    error_count = 0
    
    for file_path in tqdm(edf_files, desc="Preprocessing files"):
        try:
            result = preprocess_eeg(file_path, args.output_dir)
            if result is not None:
                output_path, time_len = result
                processed_files.append([output_path, time_len])
            else:
                error_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {str(e)}")
            error_count += 1
    
    # Create CSV file
    if processed_files:
        df = create_csv_file(processed_files, args.csv_path)
    else:
        logger.error("No files were successfully processed!")
    
    # Log summary
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Preprocessing complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"- Total files found: {len(edf_files)}")
    logger.info(f"- Successfully processed: {len(processed_files)}")
    logger.info(f"- Failed: {error_count}")
    if processed_files:
        logger.info(f"- Success rate: {len(processed_files)/len(edf_files)*100:.2f}%")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"CSV file saved to: {args.csv_path}")

if __name__ == "__main__":
    main()
