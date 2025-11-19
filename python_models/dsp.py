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

def compute_log_mel_spectrogram(audio_path: str) -> np.ndarray:
    """
    Loads an audio file and computes the Log Mel-Spectrogram features.
    This function defines the exact mathematical steps (the Gold Standard).
    """
    # 1. Load Audio (Handles decoding, resampling, and normalization)
    # Since we are focusing on DSP, librosa handles the I/O for simplicity.
    try:
        # Load audio, forcing the target sample rate
        waveform, sr = librosa.load(
            audio_path, 
            sr=SAMPLE_RATE, 
            mono=True, 
            duration=1.0 # Ensure we only take 1 second
        )
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return np.zeros((98, N_MELS), dtype=np.float32) # Return dummy features on failure

    # Ensure the waveform is exactly 1 second (16000 samples)
    if len(waveform) > CLIP_LENGTH_SAMPLES:
        waveform = waveform[:CLIP_LENGTH_SAMPLES]
    elif len(waveform) < CLIP_LENGTH_SAMPLES:
        # Pad with zeros if necessary (embedded devices must have fixed input size)
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
    # This compresses the dynamic range of the features, making them easier for the CNN.
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 4. Transpose (Feature Map Shape)
    # Librosa outputs (n_mels, n_frames). CNNs usually expect (n_frames, n_mels).
    features = log_mel_spectrogram.T
    
    return features

# --- Example Usage ---
if __name__ == '__main__':
    # You can adapt this to load a real WAV file from your dataset folder later
    # For now, let's generate 1 second of random noise to test the shape
    print("--- Running DSP Golden Reference Test ---")
    
    # Create dummy data (must be float32, which is standard for ML models)
    dummy_waveform = np.random.randn(CLIP_LENGTH_SAMPLES).astype(np.float32)
    
    # Save a temporary WAV file to simulate the I/O path (FFMPEG will replace this later)
    temp_wav_path = "temp_test_audio.wav"
    librosa.output.write_wav(temp_wav_path, dummy_waveform, sr=SAMPLE_RATE)

    # Run the function
    features = compute_log_mel_spectrogram(temp_wav_path)
    
    # Calculate the expected number of frames: (16000 - 512) / 160 + 1 = 98.675 -> 99 frames
    n_frames = features.shape[0] if features.ndim == 2 else 0

    print(f"Input Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Frame Size (N_FFT): {N_FFT} samples")
    print(f"Hop Length: {HOP_LENGTH} samples")
    print(f"Expected Time Steps (Frames): Approx. 99 (10ms steps)")
    print(f"Actual Output Shape: {features.shape}")
    print(f"First 5 feature values (for later C++ comparison):\n{features[0, :5]}")
    
    # Clean up dummy file
    os.remove(temp_wav_path)