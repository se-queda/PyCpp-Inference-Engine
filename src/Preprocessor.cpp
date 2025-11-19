#include "audioguard/Preprocessor.h"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace AudioGuard
{

    // =============================================================
    // 1. CONSTRUCTOR: The "One-Time Setup"
    // This runs ONLY when the app starts. It builds the Lookup Tables.
    // =============================================================
    Preprocessor::Preprocessor()
    {
        // Calculate the heavy math once and store it in RAM
        init_hann_window();
        init_mel_filterbank();

        std::cout << "[Preprocessor] Initialized. "
                  << "Mel Banks: " << N_MELS
                  << ", FFT Size: " << N_FFT << std::endl;
    }

    Preprocessor::~Preprocessor()
    {
        // Vectors clean themselves up
    }

    // =============================================================
    // 2. THE "REAL-TIME" LOOP (The Dynamic Part)
    // This uses the pre-calculated tables to process live audio.
    // =============================================================
    std::vector<vector> Preprocessor::compute_log_mel_spectrogram(const vector &input_wav)
    {
        std::vector<vector> log_mel_spec;

        // We slide our window across the audio (160 sample hop)
        size_t num_samples = input_wav.size();

        for (size_t i = 0; i + N_FFT <= num_samples; i += HOP_LENGTH)
        {
            // A. Extract Frame (Copy audio chunk)
            vector frame(input_wav.begin() + i, input_wav.begin() + i + N_FFT);

            // B. Apply Window (Fast multiply using the 'hann_window' LUT)
            vector windowed = apply_window(frame);

            // C. FFT & Power Spectrum
            vector power_spec = compute_power_spectrum(windowed);

            // D. Apply Mel Filterbank (Matrix Multiply using 'mel_filterbank' LUT)
            vector mel_spec = apply_mel_filter(power_spec);

            // E. Log Scaling
            vector log_mel = apply_log_scale(mel_spec);

            log_mel_spec.push_back(log_mel);
        }

        return log_mel_spec;
    }

    // =============================================================
    // 3. STATIC SETUP FUNCTIONS (The Math)
    // =============================================================

    void Preprocessor::init_hann_window()
    {
        hann_window.resize(N_FFT);
        for (int i = 0; i < N_FFT; ++i)
        {
            // Standard Hann Formula: 0.5 * (1 - cos(2*pi*n / (N-1)))
            hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (N_FFT - 1)));
        }
    }

    // Helpers for Hz <-> Mel conversion
    float hz_to_mel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
    float mel_to_hz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

    void Preprocessor::init_mel_filterbank()
    {
        // We are building a matrix of size [40 x 257]
        int num_fft_bins = N_FFT / 2 + 1;
        mel_filterbank.assign(N_MELS, vector(num_fft_bins, 0.0f));

        float mel_min = hz_to_mel(F_MIN);
        float mel_max = hz_to_mel(F_MAX);

        // Create 42 points to make 40 triangles
        vector mel_points(N_MELS + 2);
        for (int i = 0; i < N_MELS + 2; ++i)
        {
            mel_points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (N_MELS + 1));
        }

        // Build the triangles
        for (int m = 0; m < N_MELS; ++m)
        {
            float f_left = mel_points[m];
            float f_center = mel_points[m + 1];
            float f_right = mel_points[m + 2];

            for (int k = 0; k < num_fft_bins; ++k)
            {
                float freq = (float)k * SAMPLE_RATE / N_FFT;

                if (freq > f_left && freq < f_center)
                {
                    mel_filterbank[m][k] = (freq - f_left) / (f_center - f_left);
                }
                else if (freq >= f_center && freq < f_right)
                {
                    mel_filterbank[m][k] = (f_right - freq) / (f_right - f_center);
                }
            }
        }
    }

    // =============================================================
    // 4. RUNTIME HELPERS
    // =============================================================

    vector Preprocessor::apply_window(const vector &frame)
    {
        vector result(N_FFT);
        for (int i = 0; i < N_FFT; ++i)
        {
            result[i] = frame[i] * hann_window[i]; // Look up the value!
        }
        return result;
    }

    vector Preprocessor::compute_power_spectrum(const vector &frame)
    {
        // Simple DFT (O(N^2)) - We use this to avoid external dependencies for now.
        // In Phase 4/Optimization, we will swap this for KissFFT.
        int num_bins = N_FFT / 2 + 1;
        vector power_spec(num_bins);

        for (int k = 0; k < num_bins; ++k)
        {
            float real = 0.0f;
            float imag = 0.0f;

            for (int n = 0; n < N_FFT; ++n)
            {
                float angle = 2.0f * M_PI * k * n / N_FFT;
                real += frame[n] * std::cos(angle);
                imag -= frame[n] * std::sin(angle);
            }
            power_spec[k] = (real * real + imag * imag) / N_FFT;
        }
        return power_spec;
    }

    vector Preprocessor::apply_mel_filter(const vector &power_spec)
    {
        // Matrix Multiplication: [40 x 257] * [257] -> [40]
        vector mel_energies(N_MELS, 0.0f);
        int num_bins = power_spec.size();

        for (int m = 0; m < N_MELS; ++m)
        {
            for (int k = 0; k < num_bins; ++k)
            {
                mel_energies[m] += mel_filterbank[m][k] * power_spec[k];
            }
        }
        return mel_energies;
    }

    vector Preprocessor::apply_log_scale(const vector &mel_spec)
    {
        vector log_mels(N_MELS);
        float epsilon = 1e-10f; // Prevent log(0)

        for (int i = 0; i < N_MELS; ++i)
        {
            log_mels[i] = 10.0f * std::log10(mel_spec[i] + epsilon);
        }
        return log_mels;
    }
}