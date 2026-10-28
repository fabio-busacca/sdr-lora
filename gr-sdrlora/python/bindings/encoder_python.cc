/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sdrlora/encoder.h>
#include "sdrlora/lora_phy.h"

namespace py = pybind11;

void bind_encoder(py::module& m)
{
    using encoder = gr::sdrlora::encoder;
    py::class_<encoder, gr::block, gr::basic_block, std::shared_ptr<encoder> >(
        m, "encoder", "LoRa encoder: PDUs in, baseband IQ out")
        .def(py::init(&encoder::make),
             py::arg("sf") = 7, py::arg("cr") = 1, py::arg("bw") = 125e3,
             py::arg("samp_rate") = 1e6, py::arg("f0") = 868e6,
             py::arg("has_crc") = true, py::arg("implicit_header") = false,
             py::arg("preamble_symbols") = 8, py::arg("src") = 0,
             py::arg("dst") = 0, py::arg("pseudo_header") = true, py::arg("seqn") = 0,
             py::arg("cfo_ppm") = 0.0, py::arg("sync_word") = std::vector<int>{ 0x12 },
             py::arg("ldro") = 2, py::arg("lead_in") = -1,   /* auto; matches
                                                    encoder.h = -1 */
             py::arg("invert_iq") = false,
             py::arg("trise") = gr::sdrlora::TRISE,
             py::arg("t0_frac") = 0.0, py::arg("phi0") = 0.0,
             py::arg("amplitude") = 1.0,
             py::arg("hop_period") = 0,
             py::arg("hop_freqs") = std::vector<double>{},
             py::arg("burst_tail") = -1)    /* auto; matches encoder.h */
        .def("set_seqn", &encoder::set_seqn)
        .def("seqn", &encoder::seqn);

    /* The PHY core, exposed directly. Handy for generating and inspecting
     * waveforms from a Python session without a flowgraph. */
    using tx_params = gr::sdrlora::tx_params;
    py::class_<tx_params>(m, "tx_params")
        .def(py::init<>())
        .def_readwrite("SF", &tx_params::SF)
        .def_readwrite("cr", &tx_params::cr)
        .def_readwrite("has_crc", &tx_params::has_crc)
        .def_readwrite("implicit_header", &tx_params::implicit_header)
        .def_readwrite("preamble_symbols", &tx_params::preamble_symbols)
        .def_readwrite("src", &tx_params::src)
        .def_readwrite("dst", &tx_params::dst)
        .def_readwrite("seqn", &tx_params::seqn)
        .def_readwrite("f0", &tx_params::f0)
        .def_readwrite("bw", &tx_params::bw)
        .def_readwrite("fs", &tx_params::fs)
        /* Every tx_params field is bound. pseudo_header in particular is
         * what lets phy_encode() express a standard LoRa frame:
         * pseudo_header=false is what a third-party receiver expects and
         * what interoperating with gr-lora_sdr requires. A field added to
         * tx_params needs a binding here too. */
        .def_readwrite("pseudo_header", &tx_params::pseudo_header)
        .def_readwrite("cfo_ppm", &tx_params::cfo_ppm)
        /* Off-grid stimulus: a fractional sampling instant and an initial
         * phase. Both default to 0, so the default waveform is
         * unchanged. */
        .def_readwrite("t0_frac", &tx_params::t0_frac)
        .def_readwrite("phi0", &tx_params::phi0)
        /* The shaping and receive-tuning parameters. Exposed on the PHY
           core so they can be set and read back numerically rather than
           by reading the header. */
        .def_readwrite("trise", &tx_params::trise)
        /* receive: the carrier for sample-clock tracking, 0 off */
        .def_readwrite("center_freq", &tx_params::center_freq)
        .def_readwrite("amplitude", &tx_params::amplitude)
        .def_readwrite("cutoff_ratio", &tx_params::cutoff_ratio)
        .def_readwrite("numtaps", &tx_params::numtaps)
        .def_readwrite("sync_threshold", &tx_params::sync_threshold)
        .def_readwrite("search_tail_symbols", &tx_params::search_tail_symbols)
        .def_readwrite("min_osf", &tx_params::min_osf)
        .def_readwrite("interp_target_osf", &tx_params::interp_target_osf)
        .def_readwrite("hop_period", &tx_params::hop_period)
        .def_readwrite("hop_freqs", &tx_params::hop_freqs)
        .def_readwrite("sync_word", &tx_params::sync_word)
        .def_readwrite("ldro", &tx_params::ldro)
        .def_readwrite("lead_in", &tx_params::lead_in)
        .def_readwrite("invert_iq", &tx_params::invert_iq);

    /* Bound so the widening formula can be checked NUMERICALLY rather than
       by reading the source. */
    m.def("hop_cutoff_ratio", &gr::sdrlora::hop_cutoff_ratio,
          py::arg("hop_freqs"), py::arg("bw"),
          "Smallest receive-filter cutoff, in multiples of bw, that "
          "passes every channel in a hop plan.");
    /* And the de-hop itself, so its block edges -- which drift with the
       sample clock -- can be checked sample for sample. That drift barely
       affects delivery, so an error would not show at the decode
       level. */
    m.def("apply_hops",
          [](std::vector<std::complex<float> > s, long frame_ofs,
             int hop_period, const std::vector<double>& hop_freqs,
             long n_sym, double fs, bool invert, double sfo, double ref) {
              gr::sdrlora::apply_hops(s, frame_ofs, hop_period, hop_freqs,
                                      n_sym, fs, invert, sfo, ref);
              return s;
          },
          py::arg("samples"), py::arg("frame_ofs"), py::arg("hop_period"),
          py::arg("hop_freqs"), py::arg("n_sym"), py::arg("fs"),
          py::arg("invert") = false, py::arg("sfo") = 0.0,
          py::arg("ref") = 0.0,
          "apply_hops() on a copy of the samples, returned.");
    /* Bound so the auto-lead-in rule can be checked NUMERICALLY: reading
     * the source text cannot tell whether the body actually computes the
     * rule. */
    m.def("default_lead_in", &gr::sdrlora::default_lead_in,
          py::arg("SF"), py::arg("bw"), py::arg("fs"),
          py::arg("preamble_symbols") = 8,
          "Zero samples ahead of the preamble for a negative lead_in.");
    m.def("resolved_lead_in", &gr::sdrlora::resolved_lead_in,
          py::arg("p"),
          "What a tx_params' lead_in means; negative resolves to auto.");

    m.def("phy_encode",
          [](const tx_params& p, const std::vector<uint8_t>& msg) {
              std::vector<std::complex<float> > out;
              gr::sdrlora::encode(p, msg, out);
              return out;
          },
          py::arg("params"), py::arg("payload"),
          "Payload bytes -> baseband IQ.");

    m.def("phy_encode_symbols",
          [](const tx_params& p, const std::vector<uint8_t>& msg) {
              std::vector<uint32_t> kh, kp;
              gr::sdrlora::encode_symbols(p, msg, kh, kp);
              return std::make_pair(kh, kp);
          },
          py::arg("params"), py::arg("payload"),
          "Payload bytes -> (k_hdr, k_payload) symbol indices");

    /* The public modulate(), so what it refuses on its own can be
     * exercised directly: it carries the exact carrier-phase check that
     * encode()'s estimate stands in front of. */
    m.def("phy_modulate",
          [](const tx_params& p, const std::vector<uint32_t>& k_hdr,
             const std::vector<uint32_t>& k_payload) {
              std::vector<std::complex<float> > out;
              gr::sdrlora::modulate(p, k_hdr, k_payload, out);
              return out;
          },
          py::arg("params"), py::arg("k_hdr"), py::arg("k_payload"),
          "(k_hdr, k_payload) symbol indices -> baseband IQ");
}
