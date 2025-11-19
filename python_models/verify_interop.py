import numpy as np
import sys
import os

# Add the build directory to the Python path so we can import the C++ module
# (Adjust 'build' if your build folder is named differently)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

try:
    import audioguard_core
    print("✅ Successfully imported C++ module 'audioguard_core'")
except ImportError as e:
    print("❌ Could not import 'audioguard_core'. Did you compile the C++ project?")
    print(f"Error: {e}")
    sys.exit(1)

import gold_standard_dsp

def compare_outputs():
    print("\n--- 🧪 AudioGuard Interop Verification 🧪 ---\n")

    # 1. Generate Dummy Input (1 Second of Audio)
    # Must be float32 because our C++ vector expects floats!
    sample_rate = 16000
    # Create a simple sine wave so we can inspect values easily if needed
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    raw_audio = 0.5 * np.sin(2 * np.pi * 440 * t) # 440Hz tone
    raw_audio = raw_audio.astype(np.float32)

    # 2. Run Python Gold Standard
    print("1️⃣  Running Python Reference (Librosa)...")
    py_features = gold_standard_dsp.compute_log_mel_spectrogram(raw_audio)
    print(f"    Python Output Shape: {py_features.shape}")

    # 3. Run C++ Implementation
    print("2️⃣  Running C++ Implementation (AudioGuard)...")
    # Initialize the C++ class
    preprocessor = audioguard_core.Preprocessor()
    
    # Pass the numpy array directly (PyBind11 handles the conversion)
    cpp_features_list = preprocessor.compute_log_mel_spectrogram(raw_audio)
    
    # Convert the result back to a numpy array for comparison
    cpp_features = np.array(cpp_features_list, dtype=np.float32)
    print(f"    C++ Output Shape:    {cpp_features.shape}")

    # 4. Verification Logic
    print("\n3️⃣  Comparing Results...")
    
    # A. Shape Check
    if py_features.shape != cpp_features.shape:
        print(f"❌ SHAPE MISMATCH! Py: {py_features.shape} vs C++: {cpp_features.shape}")
        return

    # B. Value Check
    # We allow a small tolerance (atol) because C++ sin/cos/log might strictly differ
    # slightly from Python's numpy implementation due to floating point precision.
    tolerance = 1e-4 
    is_close = np.allclose(py_features, cpp_features, atol=tolerance)

    if is_close:
        print(f"✅ SUCCESS! C++ and Python outputs match within {tolerance} tolerance.")
        
        # Calculate Mean Squared Error (MSE) just to be fancy
        mse = np.mean((py_features - cpp_features) ** 2)
        print(f"    Mean Squared Error: {mse:.8f}")
    else:
        print("❌ VALUE MISMATCH!")
        diff = np.abs(py_features - cpp_features)
        print(f"    Max Difference: {np.max(diff)}")
        print(f"    Mean Difference: {np.mean(diff)}")
        
        print("\n    First 5 Python values:\n", py_features[0][:5])
        print("    First 5 C++ values:   \n", cpp_features[0][:5])

if __name__ == "__main__":
    compare_outputs()