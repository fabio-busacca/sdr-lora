/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
#ifndef INCLUDED_SDRLORA_ENCODER_IMPL_H
#define INCLUDED_SDRLORA_ENCODER_IMPL_H

#include <sdrlora/encoder.h>
#include "sdrlora/lora_phy.h"

#include <deque>
#include <mutex>

namespace gr {
namespace sdrlora {

class encoder_impl : public encoder
{
private:
    tx_params d_p;
    int d_seqn;

    /* Encoded samples waiting to be handed to the scheduler. A whole packet is
     * produced at once when a PDU arrives, then drained; this keeps the
     * modulator out of general_work() and avoids tagged-stream length-tag
     * bookkeeping entirely. */
    std::deque<gr_complex> d_queue;
    /* Each queued frame's length, and how much of the one being emitted is
     * left, so its first and last samples carry tx_sob / tx_eob. */
    std::deque<size_t> d_frames;
    size_t d_frame_left = 0;
    /* Zero samples closing each burst, before tx_eob; the constructor says
     * why and how many. */
    size_t d_tail = 0;
    /* mutable: seqn() is const and takes this lock, because d_seqn is
       written by the message-handler thread. */
    mutable std::mutex d_mutex;

    void handle_pdu(pmt::pmt_t pdu);

public:
    encoder_impl(int sf, int cr, double bw, double samp_rate, double f0,
                 bool has_crc, bool implicit_header, int preamble_symbols,
                 int src, int dst, bool pseudo_header, int seqn,
                 double cfo_ppm, std::vector<int> sync_word, int ldro,
                 int lead_in,
                 bool invert_iq,
                 double trise, double t0_frac, double phi0,
                 double amplitude, int hop_period,
                 std::vector<double> hop_freqs, int burst_tail);
    ~encoder_impl() override;

    void set_seqn(int seqn) override;
    int seqn() const override;

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override;
};

} // namespace sdrlora
} // namespace gr

#endif
