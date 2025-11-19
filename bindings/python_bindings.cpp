#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "audioguard/Preprocessor.h"

namespace py = pybind11;

PYBIND11_MODULE(audioguard_core, m)
{

    m.doc() = "AudioGuard C++ Core Plugin";
    py::class_<AudioGuard::Preprocessor>(m, "Preprocessor")

        .def(py::init<>()) // constructor

        .def("compute_log_mel_spectrogram",
             &AudioGuard::Preprocessor::compute_log_mel_spectrogram, // callable function
             py::call_guard<py::gil_scoped_release>(),               // GIL release
             "Computes Log Mel-Spectrogram from raw audio input.");
}