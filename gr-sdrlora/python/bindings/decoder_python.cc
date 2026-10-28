/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sdrlora/decoder.h>
#include "sdrlora/lora_rx.h"

namespace py = pybind11;

void bind_decoder(py::module& m)
{
    using decoder = gr::sdrlora::decoder;
    py::class_<decoder, gr::sync_block, gr::basic_block, std::shared_ptr<decoder> >(
        m, "decoder", "LoRa decoder: baseband IQ in, PDUs out")
        .def(py::init(&decoder::make),
             py::arg("sf") = 7, py::arg("bw") = 125e3,
             py::arg("samp_rate") = 1e6, py::arg("window") = 0,
             py::arg("channel_filter_rx") = true,
             py::arg("soft_decision") = true,
             py::arg("robust_sync") = true,
             py::arg("sync_phases") = 0, py::arg("pseudo_header") = true,
             py::arg("implicit_header") = false, py::arg("payload_len") = 0,
             py::arg("cr") = 1, py::arg("has_crc") = true,
             py::arg("sync_word") = std::vector<int>{ 0x12 },
             py::arg("ldro") = 2,
             py::arg("fec_correct") = true, py::arg("interpolate") = true,
             py::arg("report_crc_errors") = false,
             py::arg("max_payload_len") = 251,
             py::arg("invert_iq") = false,
             py::arg("trise") = gr::sdrlora::TRISE,
             py::arg("cutoff_ratio") = 1.15,
             py::arg("numtaps") = 255,
             py::arg("sync_threshold") = 5.5,
             py::arg("search_tail_symbols") =
                 gr::sdrlora::SEARCH_TAIL_SYMBOLS_DEFAULT,
             py::arg("min_osf") = gr::sdrlora::MIN_OSF_DEFAULT,
             py::arg("interp_target_osf") =
                 gr::sdrlora::INTERP_TARGET_OSF_DEFAULT,
             py::arg("hop_period") = 0,
             py::arg("hop_freqs") = std::vector<double>{},
             py::arg("low_latency") = false,
             py::arg("center_freq") = 0.0)
        .def("packets", &decoder::packets)
        .def("crc_errors", &decoder::crc_errors)
        .def("overruns", &decoder::overruns)
        .def("scans", &decoder::scans)
        .def("set_idle_flush", &decoder::set_idle_flush,
             py::call_guard<py::gil_scoped_release>())
        .def("idle_flush", &decoder::idle_flush,
             py::call_guard<py::gil_scoped_release>())
        .def("retention_samples", &decoder::retention_samples);

    /* The receive core, exposed directly. Lets you decode a captured buffer
     * from a Python session without a flowgraph. */
    py::class_<gr::sdrlora::rx_result>(m, "rx_result")
        .def_readonly("hdr_ok", &gr::sdrlora::rx_result::hdr_ok)
        .def_readonly("crc_ok", &gr::sdrlora::rx_result::crc_ok)
        /* has_crc and truncated let a caller see whether the frame carried
         * a payload CRC and tell a clipped frame from a failed one. */
        .def_readonly("has_crc", &gr::sdrlora::rx_result::has_crc)
        .def_readonly("truncated", &gr::sdrlora::rx_result::truncated)
        /* consumed is samples used past p_ofs. The streaming contract is
         * built on it, so a Python caller can read it off each packet. */
        .def_readonly("consumed", &gr::sdrlora::rx_result::consumed)
        .def_readonly("cr", &gr::sdrlora::rx_result::cr)
        .def_readonly("length", &gr::sdrlora::rx_result::length)
        .def_readonly("src", &gr::sdrlora::rx_result::src)
        .def_readonly("dst", &gr::sdrlora::rx_result::dst)
        .def_readonly("seqn", &gr::sdrlora::rx_result::seqn)
        .def_readonly("payload", &gr::sdrlora::rx_result::payload)
        .def_readonly("k_hdr", &gr::sdrlora::rx_result::k_hdr)
        .def_readonly("k_payload", &gr::sdrlora::rx_result::k_payload)
        .def_readonly("gap_score", &gr::sdrlora::rx_result::gap_score);

    /* The receive filter, bound so its KERNEL (its impulse response) can be
       inspected numerically: a Blackman-windowed sinc, 255 taps by default
       -- lora_sync.cc's window matches np.blackman. */
    m.def("channel_filter",
          [](const std::vector<std::complex<float> >& in, double bw,
             double fs, double cutoff_ratio, int numtaps) {
              std::vector<std::complex<float> > out;
              gr::sdrlora::channel_filter(in, bw, fs, out,
                                          cutoff_ratio, numtaps);
              return out;
          },
          py::arg("in"), py::arg("bw"), py::arg("fs"),
          py::arg("cutoff_ratio") = 1.15, py::arg("numtaps") = 255,
          "Band-limit to the LoRa channel.");
    /* Streaming geometry. Pure functions, so they can be checked directly
     * without a waveform. */
    m.def("max_packet_symbols", &gr::sdrlora::max_packet_symbols,
          py::arg("SF"), py::arg("de") = -1, py::arg("max_payload") = 251,
          "Longest legal packet in symbols");
    m.def("live_overlap_samples", &gr::sdrlora::live_overlap_samples,
          py::arg("SF"), py::arg("osf"), py::arg("de") = -1,
          py::arg("max_payload") = 251,
          "Samples consecutive windows must share");

    /* Same as phy_decode, but also returns how many leading samples the
     * caller is finished with -- decode_buffer's consumed_out. Exposed
     * because this is the streaming contract, and an error here silently
     * eats packets rather than failing loudly. */
    m.def("phy_decode_consumed",
          /* BY VALUE, not by reference: the GIL is released below, and a
           * reference here points into the live Python object's storage --
           * every field is def_readwrite, so another thread assigning
           * params.sync_word would free the vector mid-read. Copying costs
           * one small struct and happens while the GIL is still held. */
          [](gr::sdrlora::tx_params p,
             const std::vector<std::complex<float> >& s,
             bool channel_filter_rx, bool robust_sync, int sync_phases,
             bool soft_decision, bool implicit_header, int payload_len,
             bool fec_correct, bool interpolate, bool report_crc_errors,
             long search_from, long walk_end, bool final_walk) {
              long consumed = 0;
              std::vector<gr::sdrlora::rx_result> out;
              {
                  py::gil_scoped_release unlock;   /* see phy_decode */
                  out =
                  gr::sdrlora::decode_buffer(s, p, channel_filter_rx,
                                             robust_sync, sync_phases,
                                             soft_decision, implicit_header,
                                             payload_len, fec_correct,
                                             interpolate, report_crc_errors,
                                             &consumed, search_from,
                                             walk_end, 0, final_walk);
              }
              return std::make_pair(out, consumed);
          },
          py::arg("params"), py::arg("samples"),
          py::arg("channel_filter_rx") = true, py::arg("robust_sync") = true,
          py::arg("sync_phases") = 0, py::arg("soft_decision") = true,
          py::arg("implicit_header") = false, py::arg("payload_len") = 0,
          py::arg("fec_correct") = true, py::arg("interpolate") = true,
          py::arg("report_crc_errors") = false,
          /* Where the walk starts -- what the decoder block passes after a
           * scan whose buffer it rewound for the filter. Exposed so the
           * resume can be exercised on captured samples. */
          py::arg("search_from") = 0,
          /* And where it stops, with the whole buffer filtered -- the
           * block's hold-back. -1 walks everything. */
          py::arg("walk_end") = -1,
          /* nothing more is coming. A caller asking for `consumed` -- a
           * streaming scan -- leaves it off, hence the default */
          py::arg("final") = false,
          "decode_buffer(), returning (packets, consumed_samples)");

    m.def("phy_decode",
          /* BY VALUE, not by reference: the GIL is released below, and a
           * reference here points into the live Python object's storage --
           * every field is def_readwrite, so another thread assigning
           * params.sync_word would free the vector mid-read. Copying costs
           * one small struct and happens while the GIL is still held. */
          [](gr::sdrlora::tx_params p,
             const std::vector<std::complex<float> >& s,
             bool channel_filter_rx, bool robust_sync, int sync_phases,
             bool soft_decision, bool implicit_header, int payload_len,
             bool fec_correct, bool interpolate, bool report_crc_errors) {
              /* Release the GIL for the duration. decode_buffer touches no
               * Python object -- it works on the std::vector pybind already
               * converted -- and at SF12 it runs for seconds, during which
               * holding the GIL would block every other thread in the
               * process. */
              py::gil_scoped_release unlock;
              return gr::sdrlora::decode_buffer(s, p, channel_filter_rx,
                                                robust_sync, sync_phases,
                                                soft_decision, implicit_header,
                                                payload_len, fec_correct,
                                                interpolate,
                                                report_crc_errors);
          },
          py::arg("params"), py::arg("samples"),
          py::arg("channel_filter_rx") = true, py::arg("robust_sync") = true,
          py::arg("sync_phases") = 0, py::arg("soft_decision") = true,
          /* implicit_header and payload_len are what decode an
           * implicit-header frame, and so SF5 and SF6 at all. */
          py::arg("implicit_header") = false, py::arg("payload_len") = 0,
          py::arg("fec_correct") = true, py::arg("interpolate") = true,
          py::arg("report_crc_errors") = false,
          "Scan a buffer and decode every packet.");

    m.def("phy_packet_rx",
          /* BY VALUE, not by reference: the GIL is released below, and a
           * reference here points into the live Python object's storage --
           * every field is def_readwrite, so another thread assigning
           * params.sync_word would free the vector mid-read. Copying costs
           * one small struct and happens while the GIL is still held. */
          [](gr::sdrlora::tx_params p,
             const std::vector<std::complex<float> >& s, long p_ofs,
             double cfo_hz, bool soft_decision, bool implicit_header,
             int payload_len, int cr, bool has_crc, bool fec_correct) {
              /* Release the GIL, as phy_decode and phy_decode_consumed do;
               * the by-value copy above is what makes it safe. Without the
               * release, packet_rx would block every other Python thread
               * for its whole duration, which at SF12 is the entire
               * decode. */
              py::gil_scoped_release unlock;
              return gr::sdrlora::packet_rx(s, p, p_ofs, cfo_hz,
                                            implicit_header, payload_len, cr,
                                            has_crc, soft_decision,
                                            fec_correct);
          },
          py::arg("params"), py::arg("samples"), py::arg("p_ofs"),
          py::arg("cfo_hz"), py::arg("soft_decision") = true,
          /* The implicit-header settings, so this can demodulate an
           * implicit-header frame as well as an explicit one. */
          py::arg("implicit_header") = false, py::arg("payload_len") = 0,
          py::arg("cr") = 1, py::arg("has_crc") = true,
          py::arg("fec_correct") = true,
          "Demodulate one packet at a known offset and CFO, skipping the sync "
          "search.");
}
