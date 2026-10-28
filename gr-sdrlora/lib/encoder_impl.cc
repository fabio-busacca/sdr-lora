/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
#include "encoder_impl.h"

#include <gnuradio/io_signature.h>

/* <cmath> explicitly for std::floor and std::abs(double) at the rate check
 * below: relying on <complex>'s transitive include risks std::abs resolving
 * to the INTEGER overload, which would make the integer-ratio check stop
 * rejecting fractional rates. */
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gr {
namespace sdrlora {

encoder::sptr encoder::make(int sf, int cr, double bw, double samp_rate, double f0,
                            bool has_crc, bool implicit_header,
                            int preamble_symbols, int src, int dst,
                            bool pseudo_header, int seqn, double cfo_ppm,
                            std::vector<int> sync_word, int ldro,
                            int lead_in,
                            bool invert_iq,
                            double trise, double t0_frac, double phi0,
                            double amplitude,
                            int hop_period,
                            std::vector<double> hop_freqs,
                            int burst_tail)
{
    return gnuradio::make_block_sptr<encoder_impl>(
        sf, cr, bw, samp_rate, f0, has_crc, implicit_header, preamble_symbols,
        src, dst, pseudo_header, seqn, cfo_ppm, sync_word, ldro, lead_in,
        invert_iq, trise, t0_frac, phi0, amplitude, hop_period,
        hop_freqs, burst_tail);
}

encoder_impl::encoder_impl(int sf, int cr, double bw, double samp_rate, double f0,
                           bool has_crc, bool implicit_header,
                           int preamble_symbols, int src, int dst,
                           bool pseudo_header, int seqn, double cfo_ppm,
                           std::vector<int> sync_word, int ldro,
                           int lead_in,
                           bool invert_iq,
                           double trise, double t0_frac, double phi0,
                           double amplitude,
                           int hop_period,
                           std::vector<double> hop_freqs,
                           int burst_tail)
    : gr::block("sdrlora_encoder",
                gr::io_signature::make(0, 0, 0),
                gr::io_signature::make(1, 1, sizeof(gr_complex))),
      d_seqn(seqn & 0xff)
{
    if (sf < 5 || sf > 12)
        throw std::invalid_argument("sdrlora encoder: sf must be 5..12");
    if (sf < 7 && !implicit_header)
        throw std::invalid_argument(
            "sdrlora encoder: SF5/SF6 require implicit_header (an explicit "
            "header needs 20 bits and the header block holds 4*(SF-2))");
    if (cr < 1 || cr > 4)
        throw std::invalid_argument("sdrlora encoder: cr must be 1..4");
    /* A frame with no preamble is not a LoRa frame: no receiver can lock
     * onto it. Three up-chirps is the fewest this receiver decodes on a
     * clean SF7 frame; the standard uses eight. */
    if (preamble_symbols > gr::sdrlora::PREAMBLE_SYMBOLS_MAX)
        throw std::invalid_argument(
            "sdrlora encoder: preamble_symbols must be <= 65535 -- the width "
            "of the SX1276's preamble-length register, and 8192 times the "
            "standard 8");
    if (preamble_symbols < 1)
        throw std::invalid_argument(
            "sdrlora encoder: preamble_symbols must be at least 1 (8 is "
            "standard; below 3 this receiver cannot lock even on a clean "
            "signal)");
    /* src, dst and seqn go on air as single bytes. Out-of-range values are
     * refused rather than truncated by the cast (dst=256 would go out as
     * 0). WHATEVER THE FRAMING, here and in the GRC asserts, although
     * check_tx_params skips them without the pseudo-header: there they are
     * a re-sent packet's absent fields, here a value typed into a palette,
     * where 300 is a typo. */
    if (src < 0 || src > 255 || dst < 0 || dst > 255)
        throw std::invalid_argument(
            "sdrlora encoder: src and dst are single bytes, 0..255");
    /* The counter itself is meant to wrap -- it is a byte on air and
     * handle_pdu() increments it mod 256. An out-of-range STARTING value is a
     * different thing: masking 300 to 44 silently is a typo accepted, not a
     * counter wrapping. */
    if (seqn < 0 || seqn > 255)
        throw std::invalid_argument(
            "sdrlora encoder: seqn is a single byte, 0..255 (it wraps as it "
            "increments; this is the starting value)");
    /* THE RATE FIRST, finite, positive and bounded, before anything
     * converts it. default_lead_in below does, and it runs before
     * check_tx_params: every comparison with NaN is false, so a NaN rate
     * would otherwise pass the ratio test below and reach
     * default_lead_in's (long)(fs/bw + 0.5). */
    gr::sdrlora::check_rate_bounds(bw, samp_rate);
    const double osf = samp_rate / bw;
    /* RELATIVE tolerance, like the other copies of this rule --
     * check_rx_params, decode_buffer and decoder_impl -- so that every
     * entry point accepts the same pairs (e.g. bw = 125000,
     * samp_rate = 16000000.01). lora_phy.cc's note
     * on the transmit/receive nrise spellings names the relative rule as
     * the contract that makes them agree.
     *
     * AND `< 1` ON THE ROUNDED RATIO, as the other copies test it, so a
     * ratio that is 1 to within the tolerance (samp_rate 124999.99999 at
     * bw 125000) is accepted here as it is elsewhere. */
    const double osf_r = std::floor(osf + 0.5);
    if (osf_r < 1.0 || std::abs(osf - osf_r) > 1e-9 * (osf > 1.0 ? osf : 1.0))
        throw std::invalid_argument(
            "sdrlora encoder: samp_rate must be an integer multiple of bw");

    d_p.SF = sf;
    d_p.cr = cr;
    d_p.bw = bw;
    d_p.fs = samp_rate;
    d_p.f0 = f0;
    d_p.has_crc = has_crc;
    d_p.implicit_header = implicit_header;
    d_p.preamble_symbols = preamble_symbols;
    d_p.src = src;
    d_p.dst = dst;
    d_p.pseudo_header = pseudo_header;
    /* cfo_ppm is a simulated offset; 0, the default here and in encode(),
     * is a clean carrier, as a transmitter should send. */
    d_p.cfo_ppm = cfo_ppm;
    d_p.sync_word = sync_word;
    /* Zero samples before each preamble. NEGATIVE MEANS AUTO, which is the
     * block's default. A flat 10000 is not enough warm-up for robust_sync
     * at a high SF with a short preamble, and the frame then does not
     * decode -- see default_lead_in(). An explicit non-negative value is
     * honoured exactly. */
    if (lead_in < 0)
        lead_in = gr::sdrlora::default_lead_in(sf, bw, samp_rate,
                                               preamble_symbols);
    if ((long)lead_in > gr::sdrlora::LEAD_IN_MAX)
        throw std::invalid_argument(
            "sdrlora encoder: lead_in must be <= 67108864 samples -- pure "
            "padding ahead of the preamble, bounded so an absurd value is an "
            "error naming it rather than a bad_alloc");
    d_p.lead_in = lead_in;
    d_p.invert_iq = invert_iq;
    d_p.ldro = ldro;
    /* Set BEFORE the check_tx_params call at the end of this constructor,
       whose bandwidth ceiling is 2^SF / trise -- so a lengthened taper is
       held to the tighter limit rather than the constant's, and a bad
       trise, t0_frac, amplitude or hop plan throws at construction instead
       of dropping every PDU at runtime. */
    d_p.trise = trise;
    d_p.t0_frac = t0_frac;
    d_p.phi0 = phi0;
    d_p.amplitude = amplitude;
    d_p.hop_period = hop_period;
    d_p.hop_freqs = hop_freqs;
    {
        int s1 = 0, s2 = 0;
        if (!sync_word_symbols(sync_word, s1, s2))
            throw std::invalid_argument(
                "sdrlora encoder: sync_word must be one byte with two "
                "non-zero nibbles (0x12 is the private/default value) or two "
                "raw symbols in 1..4095, e.g. [24, 32]");
        if (!sync_symbols_separable(s1, s2, sf))
            throw std::invalid_argument(
                "sdrlora encoder: those sync symbols are within one bin of the "
                "preamble at this SF -- bins are cyclic mod 2^SF and the sync "
                "check has a one-bin tolerance, so they cannot be told from "
                "more preamble. Use symbols 2..2^SF-2.");
    }
    if (ldro < 0 || ldro > 2)
        throw std::invalid_argument(
            "sdrlora encoder: ldro must be 0 (off), 1 (on) or 2 (auto)");
    /* The whole rule, on the assembled struct. The per-field checks above
       give better-worded errors for the fields they cover; this catches
       everything they do not. */
    gr::sdrlora::check_tx_params(d_p);
    /* THE PAYLOAD-INDEPENDENT FLOOR OF check_frame_size.
     *
     * check_frame_size lives in encode_symbols, because the full estimate
     * needs the payload symbol count, so check_tx_params never sees it.
     * handle_pdu catches the throw and logs, so without this a block whose
     * preamble alone puts the frame past MAX_FRAME_SAMPLES (e.g. a long
     * preamble at SF12) would construct cleanly and then drop every PDU.
     *
     * n_sym_payload = 0 is the floor of the very same formula, so this
     * cannot refuse anything encode() accepts: every real frame's estimate
     * is at least this one. Same reasoning as decoder_impl's constructor
     * lifting decode_buffer's rules. */
    gr::sdrlora::check_frame_size(d_p, 0);

    /* SILENCE CLOSING EACH BURST, inside it (tx_eob goes on its last
     * sample). This block is driven by messages, so after a frame its output
     * simply stops. The decoder's tx_eob flush and idle flush deliver a
     * burst on a direct connection, but a filter or resampler in between
     * (a decimating FIR, a Channel Model) keeps the frame's last samples in
     * its history until more input arrives, and no decoder-side flush can
     * reach samples still inside an upstream block.
     *
     * The auto length (negative burst_tail) is 512 samples AT fs/bw 1 --
     * 512 * osf here -- so it scales with the rate and still covers the
     * decoder's default hold-back of 127 samples, with room for the group
     * delay of blocks in between, behind a decimate-by-D resampler. That is
     * 4 ms at 125 kHz, short enough not to key a USRP sink's transmitter for
     * long; 0 sends none. Bounded like lead_in, so an absurd value is an
     * error naming it rather than a bad_alloc per PDU. */
    if ((long)burst_tail > gr::sdrlora::LEAD_IN_MAX)
        throw std::invalid_argument(
            "sdrlora encoder: burst_tail must be <= 67108864 samples -- "
            "zero padding closing each burst (negative is auto, 0 is none)");
    d_tail = (burst_tail >= 0)
                 ? (size_t)burst_tail
                 : (size_t)512 * (size_t)osf_r;

    message_port_register_in(pmt::mp("pdus"));
    set_msg_handler(pmt::mp("pdus"),
                    [this](pmt::pmt_t msg) { this->handle_pdu(msg); });
}

encoder_impl::~encoder_impl() {}

/* Both under d_mutex. d_seqn is incremented in the message-handler thread
   (handle_pdu) while these two are called from whatever thread owns the
   flowgraph; without the lock a race yields a duplicated or skipped
   sequence number. */
void encoder_impl::set_seqn(int seqn)
{
    /* REFUSED, as the constructor refuses it, rather than masked: 300
     * becoming 44 is a typo accepted, not a counter wrapping. Checked
     * before the lock: nothing is shared yet. */
    if (seqn < 0 || seqn > 255)
        throw std::invalid_argument(
            "sdrlora encoder: set_seqn: seqn is a single byte, 0..255 (it "
            "wraps as it increments; this is the next value)");
    std::lock_guard<std::mutex> lock(d_mutex);
    d_seqn = seqn;
}

int encoder_impl::seqn() const
{
    std::lock_guard<std::mutex> lock(d_mutex);
    return d_seqn;
}

void encoder_impl::handle_pdu(pmt::pmt_t pdu)
{
    pmt::pmt_t payload;
    if (pmt::is_pair(pdu))
        payload = pmt::cdr(pdu);      /* standard PDU: (metadata . u8vector) */
    else
        payload = pdu;

    if (!pmt::is_u8vector(payload)) {
        GR_LOG_WARN(d_logger, "dropping PDU: payload is not a u8vector");
        return;
    }

    size_t len = 0;
    const uint8_t* data = pmt::u8vector_elements(payload, len);
    /* The PHY header's LENGTH field is 8 bits, so 255 bytes go on air. The
     * pseudo-header spends 4 of them on DST/SRC/SEQNO/LENGTH, leaving 251 for
     * the message -- but only when it is enabled. With pseudo_header off
     * (standard LoRa framing, and what a third-party receiver expects) the
     * limit is 255. */
    const size_t max_len = d_p.pseudo_header ? 251 : 255;
    /* The LOWER bound is pseudo-header-dependent too. With the prefix on, a
     * 0-byte MESSAGE is a legal 4-byte-on-air frame, which the encoder
     * builds and the decoder reads back. */
    const size_t min_len = d_p.pseudo_header ? 0 : 1;
    if (len < min_len || len > max_len) {
        GR_LOG_WARN(d_logger,
                    "dropping PDU: payload length " + std::to_string(len) +
                        " outside " + std::to_string(min_len) + ".." +
                        std::to_string(max_len));
        return;
    }

    tx_params p = d_p;
    {
        /* Take and advance under the same lock set_seqn()/seqn() use. */
        std::lock_guard<std::mutex> lock(d_mutex);
        p.seqn = d_seqn;
        d_seqn = (d_seqn + 1) & 0xff;
    }

    std::vector<uint8_t> msg(data, data + len);
    std::vector<std::complex<float> > iq;
    try {
        encode(p, msg, iq);
    } catch (const std::exception& e) {
        GR_LOG_ERROR(d_logger, std::string("encode failed: ") + e.what());
        return;
    }
    iq.resize(iq.size() + d_tail, gr_complex(0.0f, 0.0f));

    std::lock_guard<std::mutex> lock(d_mutex);
    d_queue.insert(d_queue.end(), iq.begin(), iq.end());
    d_frames.push_back(iq.size());
}

int encoder_impl::general_work(int noutput_items,
                               gr_vector_int& ninput_items,
                               gr_vector_const_void_star& input_items,
                               gr_vector_void_star& output_items)
{
    (void)ninput_items;
    (void)input_items;
    gr_complex* out = (gr_complex*)output_items[0];

    std::lock_guard<std::mutex> lock(d_mutex);
    const int n = std::min<int>(noutput_items, (int)d_queue.size());
    /* BURST TAGS on each burst's first and last sample -- the last one is
     * the end of the closing silence (see the constructor): tx_sob / tx_eob
     * are what a USRP sink uses to transmit in bursts, and what the decoder
     * block flushes on, so a loopback flowgraph delivers each packet when
     * its frame ends rather than when the next one is sent. */
    const uint64_t base = nitems_written(0);
    for (int i = 0; i < n; i++) {
        if (d_frame_left == 0 && !d_frames.empty()) {
            d_frame_left = d_frames.front();
            d_frames.pop_front();
            add_item_tag(0, base + (uint64_t)i, pmt::mp("tx_sob"),
                         pmt::PMT_T);
        }
        out[i] = d_queue.front();
        d_queue.pop_front();
        if (d_frame_left > 0 && --d_frame_left == 0)
            add_item_tag(0, base + (uint64_t)i, pmt::mp("tx_eob"),
                         pmt::PMT_T);
    }
    /* Returning 0 with nothing queued tells the scheduler we are simply idle;
     * this block is driven by messages, not by an input stream. */
    return n;
}

} // namespace sdrlora
} // namespace gr
