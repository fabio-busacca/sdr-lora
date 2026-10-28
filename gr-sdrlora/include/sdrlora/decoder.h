/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
/*
 * LoRa decoder block: baseband IQ in on a stream, PDUs out on a message port.
 *
 * A "tx_eob" stream tag (the encoder block puts one on each burst's last
 * sample, as a USRP sink expects) flushes the buffer at that sample, as
 * stop() does, so a tagged burst is decoded as soon as it ends. A live
 * receive stream carries no such tag.
 *
 * The DSP is in lora_rx.{h,cc} and lora_sync.cc, neither of which depends on
 * GNU Radio. This block is only the streaming wrapper.
 */
#ifndef INCLUDED_SDRLORA_DECODER_H
#define INCLUDED_SDRLORA_DECODER_H

#include <cstddef>

#include <gnuradio/sync_block.h>
#include <sdrlora/api.h>
/* For TRISE and the SEARCH_TAIL_SYMBOLS / MIN_OSF /
   INTERP_TARGET_OSF defaults used in make() below. */
#include <sdrlora/lora_phy.h>

namespace gr {
namespace sdrlora {

class SDRLORA_API decoder : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<decoder> sptr;

    /*!
     * \param sf                spreading factor, 5..12. Below 7 an explicit
     *                          header does not fit, so implicit_header is
     *                          required there.
     * \param bw                bandwidth in Hz
     * \param samp_rate         sample rate; must be a POWER-OF-TWO multiple of
     *                          bw, because the sync FFTs are length N and 4N.
     *                          1x is allowed and interpolated up internally,
     *                          but it is lossy: a LoRa chirp is not strictly
     *                          bandlimited, so at critical sampling its skirts
     *                          alias in-band at the ADC and no receiver
     *                          recovers that. Prefer 2x or more.
     * \param window            NEW samples accumulated before scanning again.
     *                          This is the only latency the receiver adds on
     *                          top of a packet's own duration, and it trades
     *                          against CPU because every scan re-examines the
     *                          retained tail. The interval, in samples
     *                          after the frame's last symbol arrives, is
     *                          [numtaps/2, window + numtaps/2) with the
     *                          channel filter on and [0, window) with it
     *                          off: the filter holds its half-width back so
     *                          the next scan can filter it with real context,
     *                          so with the filter on the minimum is NOT zero
     *                          -- except at fs = bw without interpolation,
     *                          where the filter is the identity and nothing
     *                          is held back: [0, window).
     *                          Below min_osf, where the block interpolates,
     *                          the hold-back is at least 63 samples either
     *                          way -- half the interpolator's INTERP_TAPS.
     *                          It holds for any call sizes the scheduler
     *                          picks: work() consumes only up to the next
     *                          window boundary. Where in the interval a
     *                          frame lands is decided by its length, so no
     *                          single latency figure describes an SF.
     *
     *                          0 selects the DEFAULT: 32 symbols, and never
     *                          under 4096 chips -- 128 symbols at SF5, 64 at
     *                          SF6 -- clamped to the buffer ceiling. The CPU
     *                          cost goes as (tail + window)/window over the
     *                          24-symbol retained tail, so a short window is
     *                          expensive: 7.0 for a four-symbol window
     *                          against 1.75 for the default, and a decoder
     *                          that falls behind real time loses frames
     *                          silently at the radio. A bursty source does not wait for a
     *                          window: see set_idle_flush().
     * \param center_freq       the RF carrier in Hz the source is tuned
     *                          to; 0, the default, is off. Given, the
     *                          receiver reads the sample-clock error off
     *                          the carrier offset it measures -- one
     *                          crystal drives a radio's clock and its LO --
     *                          and moves each symbol window with it, as
     *                          gr-lora_sdr's center_freq does.
     *                          It assumes the offset is crystal error; a
     *                          carrier shifted any other way would be read
     *                          as drift. Either way a frame that locates and
     *                          fails its CRC is demodulated again with the
     *                          drift read from the signal, kept only if the
     *                          CRC then passes (lora_rx.h, DRIFT_*).
     * \param low_latency       halve the DEFAULT window: 16 symbols, never
     *                          under 2048 chips. Half the delivery latency
     *                          -- about 0.5 s at SF12 / 125 kHz instead of
     *                          1.05 s -- for about 1.45x the default's CPU,
     *                          close to the 1.43x (tail + window)/window
     *                          predicts. Ignored
     *                          when `window` is set.
     * \param channel_filter_rx band-limit to the channel first (~6 dB)
     * \param soft_decision     LLR + ML payload decoding
     * \param robust_sync       accumulate the preamble to detect it
     * \param sync_phases       scan granularity; 0 picks a per-SF default
     *                          (4 below SF7, where the symbol is too short for
     *                          2 to find the sync word reliably, 2 above). At
     *                          SF7 and up, 4 is more sensitive and about twice
     *                          the cost, which exceeds a live budget.
     * \param cr                coding rate 4/(4+cr), IMPLICIT MODE ONLY --
     *                          an explicit header carries it on air. Ignored
     *                          when implicit_header is false.
     * \param has_crc           whether the frame carries a payload CRC,
     *                          IMPLICIT MODE ONLY, same reasoning. Ignored
     *                          when implicit_header is false.
     * \param sync_word        LoRa sync word: one element is a byte split
     *                          into nibbles, two are the raw symbols (the
     *                          form gr-lora_sdr takes). Carried as the two
     *                          preamble symbols after the up-chirps (high
     *                          nibble * 8, then low nibble * 8). 0x12 is the
     *                          private/default value. In the byte form
     *                          both nibbles must be non-zero; the pair form
     *                          has no nibbles, so [24, 32] and [3, 125] are
     *                          legal.
     * \param ldro              low-data-rate optimisation: 0 off, 1 on,
     *                          2 auto. Auto enables it once a symbol lasts
     *                          longer than 16 ms, which depends on BANDWIDTH
     *                          as well as SF -- at 250 kHz that is SF12 only,
     *                          not SF11.
     * \param pseudo_header    keep the 4-byte DST/SRC/SEQNO/LENGTH prefix
     *                          inside the payload. Not part of LoRa: pass
     *                          false for a standard frame, which is required
     *                          to interoperate with other implementations.
     */
    /*! `invert_iq` swaps I and Q -- the SX127x InvertIQ setting, and the
     *  polarity a LoRaWAN DOWNLINK uses. An inverted frame does not decode as
     *  a normal one and vice versa; that separation is the point of it. See
     *  tx_params::invert_iq. */
    /* trise and the six that follow it: front end and search tuning.
       cutoff_ratio and numtaps shape the channel filter, sync_threshold
       sets the detection threshold, and trise, search_tail_symbols,
       min_osf and interp_target_osf default to the PHY constants in
       lora_phy.h. Documented here rather than inline, to keep the
       argument list free of prose. */
    static sptr make(int sf = 7,
                     double bw = 125e3,
                     double samp_rate = 1e6,
                     int window = 0,
                     bool channel_filter_rx = true,
                     bool soft_decision = true,
                     bool robust_sync = true,
                     int sync_phases = 0,
                     bool pseudo_header = true,
                     bool implicit_header = false,
                     int payload_len = 0,
                     int cr = 1,
                     bool has_crc = true,
                     std::vector<int> sync_word = { 0x12 },
                     int ldro = 2,
                     bool fec_correct = true,
                     bool interpolate = true,
                     bool report_crc_errors = false,
                     int max_payload_len = 251,
                     bool invert_iq = false,
                     double trise = TRISE,
                     double cutoff_ratio = 1.15,
                     int numtaps = 255,
                     double sync_threshold = 5.5,
                     int search_tail_symbols = SEARCH_TAIL_SYMBOLS_DEFAULT,
                     int min_osf = MIN_OSF_DEFAULT,
                     int interp_target_osf = INTERP_TARGET_OSF_DEFAULT,
                     int hop_period = 0,
                     std::vector<double> hop_freqs = {},
                     bool low_latency = false,
                     double center_freq = 0.0);

    /*! Packets emitted since start.
     *
     *  There is no drop count for a slow host: a GNU Radio sink that falls
     *  behind meets backpressure, not loss. What the block does shed -- the
     *  oldest samples, to stay under its buffer ceiling -- is counted by
     *  overruns(). */
    virtual uint64_t packets() const = 0;
    /* Packets that demodulated with a valid header and failed their payload
     * CRC. Always 0 unless report_crc_errors was set. packets() and this
     * together are a packet error rate. */
    virtual uint64_t crc_errors() const = 0;
    /*! Times process() shed the oldest samples to stay under the buffer
     *  ceiling.
     *
     *  Zero in normal operation. The ceiling holds one maximum-length frame,
     *  the retained tail and the front end's hold-back and left context --
     *  the channel filter's and, below min_osf, the interpolator's -- so
     *  only a lock that never completes reaches it: a false
     *  lock, or a frame longer than max_payload_len. */
    virtual uint64_t overruns() const = 0;
    /*! Scans run -- decode_buffer() calls -- since construction.
     *
     *  While a located frame is still arriving, process() skips the scans
     *  that could only find it truncated again. That changes CPU and
     *  nothing else, and this counter makes it observable: a long frame in
     *  flight costs a handful of scans rather than one per window. */
    virtual uint64_t scans() const = 0;
    /*! THE IDLE FLUSH: when no samples have arrived for this many seconds
     *  of wall time, the block scans what it holds as stop() does,
     *  hold-back and all. A live radio never goes quiet, so there it never
     *  fires; a bursty source -- a LoRa Encoder, through a Channel Model,
     *  a filter, a resampler, an Add block -- has each packet delivered
     *  about this long after its frame ends, whatever the window and
     *  whether or not its tx_eob tag survives the way there.
     *  Default 0.25; 0 disables it. A host stalled this long mid-stream
     *  costs one early scan, whose last numtaps/2 samples are filtered
     *  against zero padding instead of the samples still to come, and
     *  after which the next scan starts without left context; a frame
     *  caught by such a stall is still delivered exactly once. NOT A GRC
     *  PARAMETER: from a flowgraph, call set_idle_flush() from a Python
     *  Snippet block. */
    virtual void set_idle_flush(double seconds) = 0;
    virtual double idle_flush() const = 0;
    /*! Samples the buffer's CEILING reserves for the scan tail, so a
     *  preamble straddling a work() boundary is still found.
     *
     *  max(search_tail_symbols, SEARCH_TAIL_SYMBOLS) * 2^SF * osf.
     *
     *  THE FLOOR IS ON THIS TERM ONLY. decode_buffer()'s walk stops
     *  short of the end by the RAW knob (lora_sync.cc, `search_tail =
     *  p.search_tail_symbols * N`), so below the constant the buffer
     *  keeps MORE than the walk looks at -- e.g. 24 symbols kept against
     *  13 scanned at search_tail_symbols = 13. That direction is safe:
     *  nothing is discarded before it has been scanned. Above the
     *  constant the two coincide, which is what matters: the buffer must
     *  never keep less than the walk leaves unscanned, or everything in
     *  between would be dropped.
     *
     *  So this is what the RETENTION CEILING is, not what the walk
     *  retains, and not what the block asks the allocator for: the
     *  reserve hint is derived from the DEFAULT retention and
     *  deliberately does not follow the knob (see decoder_impl's
     *  constructor).
     *
     *  Exposed for observation, like packets() and crc_errors(): a
     *  mismatch between retention and walk shows only under sustained
     *  undecodable input, and this value makes it visible. */
    virtual std::size_t retention_samples() const = 0;
};

} // namespace sdrlora
} // namespace gr

#endif
