import librosa
import numpy as np
import os
import tensorflow as tf

# --- DSP GOLDEN CONSTANTS (MUST BE PORTED TO C++) ---
SAMPLE_RATE = 16000 # Fixed audio rate
CLIP_LENGTH_SAMPLES = SAMPLE_RATE # 1 second clip
N_FFT = 512      # FFT window size (determines frequency resolution)
HOP_LENGTH = 160 # Hop size (10ms step at 16kHz) - determines time resolution
N_MELS = 40      # Number of Mel frequency bands (the final feature depth)
FMAX = 8000      # Maximum frequency to consider (usually half the sample rate)

def compute_log_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """
    Computes the Log Mel-Spectrogram features from a raw audio waveform.
    This function defines the exact mathematical steps (the Gold Standard).
    """
    # 1. Input Validation / Pad / Trim
    # Ensure the waveform is exactly 1 second (16000 samples)
    if len(waveform) > CLIP_LENGTH_SAMPLES:
        waveform = waveform[:CLIP_LENGTH_SAMPLES]
    elif len(waveform) < CLIP_LENGTH_SAMPLES:
        # Pad with zeros if necessary
        padding = CLIP_LENGTH_SAMPLES - len(waveform)
        waveform = np.pad(waveform, (0, padding), 'constant')

    # 2. Compute the Mel Spectrogram
    # This single line chains Framing, Windowing (Hann), FFT, and Mel-Filterbank.
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX,
        power=2.0  # Use power spec (amplitude squared)
    )

    # 3. Convert to Log Scale (Decibels)
    # This compresses the dynamic range of the features.
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 4. Transpose (Feature Map Shape)
    # Librosa outputs (n_mels, n_frames). CNNs usually expect (n_frames, n_mels).
    features = log_mel_spectrogram.T
    
    return features

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Running DSP Golden Reference Test ---")
    
    # Generate dummy data directly (No file I/O needed here anymore)
    dummy_waveform = np.random.randn(CLIP_LENGTH_SAMPLES).astype(np.float32)
    
    # Run the function
    features = compute_log_mel_spectrogram(dummy_waveform)
    
    print(f"Input Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Frame Size (N_FFT): {N_FFT} samples")
    print(f"Hop Length: {HOP_LENGTH} samples")
    print(f"Actual Output Shape: {features.shape}")
    print(f"First 5 feature values:\n{features[0, :5]}")