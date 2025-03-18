def get_standard_channel_lists():
    """Return standard EEG channel lists for different naming conventions."""
    
    # Standard 10-20 channel names (e.g., used by MNE montage)
    standard_1020 = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
        'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2'
    ]
    
    # TUH EEG channel names with REF
    tuh_ref_channels = [
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
        'EEG T1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG T2-REF',
        'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG OZ-REF', 'EEG O2-REF'
    ]
    
    # TUH reduced channel set (19 channels)
    tuh_reduced_channels = [
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F7-REF', 'EEG F3-REF', 'EEG FZ-REF', 'EEG F4-REF', 'EEG F8-REF',
        'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF',
        'EEG T5-REF', 'EEG P3-REF', 'EEG PZ-REF', 'EEG P4-REF', 'EEG T6-REF', 'EEG O1-REF', 'EEG O2-REF'
    ]
    
    return {
        'standard_1020': standard_1020,
        'tuh_ref': tuh_ref_channels,
        'tuh_reduced': tuh_reduced_channels
    }

def map_tuh_to_standard_channels(raw):
    """
    Map TUH EEG channel names to standard 10-20 system names.
    
    Args:
        raw: MNE Raw object with TUH channel names
        
    Returns:
        raw: MNE Raw object with renamed channels
    """
    # Create mapping from TUH to standard format
    mapping = {}
    for ch in raw.ch_names:
        if 'EEG ' in ch and '-REF' in ch:
            # Extract the electrode name (e.g., FP1 from EEG FP1-REF)
            electrode = ch.replace('EEG ', '').replace('-REF', '')
            
            # Handle specific case differences
            if electrode.upper() == 'FP1':
                mapping[ch] = 'Fp1'
            elif electrode.upper() == 'FP2':
                mapping[ch] = 'Fp2'
            elif electrode.upper() == 'FZ':
                mapping[ch] = 'Fz'
            elif electrode.upper() == 'CZ':
                mapping[ch] = 'Cz'
            elif electrode.upper() == 'PZ':
                mapping[ch] = 'Pz'
            elif electrode.upper() == 'OZ':
                mapping[ch] = 'Oz'
            else:
                # For other channels, just use the electrode part
                mapping[ch] = electrode
    
    # Rename the channels
    raw.rename_channels(mapping)
    
    return raw
