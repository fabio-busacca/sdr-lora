/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_encoder(py::module& m);
void bind_decoder(py::module& m);
PYBIND11_MODULE(sdrlora_python, m)
{
    py::module::import("gnuradio.gr");
    m.doc() = "SDR-LoRa GNU Radio blocks";
    bind_encoder(m);
    bind_decoder(m);
}
