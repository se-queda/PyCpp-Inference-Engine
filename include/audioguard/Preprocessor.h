#pragma once

#include <vector>
#include <complex>
#include <cmath>

namespace AudioGuard
{
    using vector = std::vector<float>;
    using complex_vec = std::vector<std::complex<float>>;

    class Preprocessor
    {
    public:
        Preprocessor();
        ~Preprocessor();

        std::vector<vector> compute_log_mel_spectrogram(const vector &input_wav);

    private:
        const int SAMPLE_RATE = 16000;
        const int N_FFT = 512;
        const int HOP_LENGTH = 160;
        const int N_MELS = 40;
        const float F_MIN = 0.0f;
        const float F_MAX = 8000.0f;

        vector hann_window;
        std::vector<vector> mel_filterbank;

        void init_hann_window();
        void init_mel_filterbank();

        vector apply_window(const vector &frame);
        vector compute_power_spectrum(const vector &windowed_frame);
        vector apply_mel_filter(const vector &power_spec);
        vector apply_log_scale(const vector &mel_spec);
    };

}