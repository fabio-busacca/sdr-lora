/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
#ifndef INCLUDED_SDRLORA_DECODER_IMPL_H
#define INCLUDED_SDRLORA_DECODER_IMPL_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <sdrlora/decoder.h>
#include "sdrlora/lora_rx.h"

namespace gr {
namespace sdrlora {

class decoder_impl : public decoder
{
private:
    tx_params d_p;
    size_t d_chunk;            /* new samples accumulated before a scan */
    size_t d_cap;              /* hard ceiling on the live buffer */
    size_t d_tail;             /* kept so a preamble across a call is found */
    size_t d_since;            /* samples since the last scan */
    size_t d_sym = 1;          /* one symbol, in samples at the input rate */
    size_t d_overlap;          /* one maximum-length frame: a term of the
                                * ceiling d_cap, not a carried overlap */
    bool d_filter_runs;        /* the channel filter does something: on,
                                * and fs > bw after any interpolation */
    size_t d_hold;             /* the front end's right-hand context, held
                                * back from every walk: numtaps/2 with the
                                * filter on, at least INTERP_TAPS/2 below
                                * min_osf */
    size_t d_foot;             /* the hold-back plus the left context kept
                                * before the consume point, 2 * d_hold:
                                * live samples d_cap must hold */
    bool d_filter;
    bool d_soft;
    bool d_robust;
    int d_phases;
    bool d_fec;         /* correct Hamming parity: the header, and the
                         * payload syndrome too (see lora_rx.cc) */
    bool d_interp;      /* interpolate when fs/bw is below MIN_OSF */
    bool d_report_bad;  /* emit packets whose payload CRC failed */
    /* ATOMIC. Written in process(), which runs in the GNU Radio scheduler
     * thread, and read by packets()/crc_errors(), which Python calls from
     * whatever thread owns the flowgraph. Atomics rather than a mutex
     * because process() is the hot path and these are plain counters. */
    std::atomic<uint64_t> d_packets;
    std::atomic<uint64_t> d_crc_errors;
    std::atomic<uint64_t> d_overruns;   /* ceiling trims; see overruns() */
    std::atomic<uint64_t> d_scans;      /* decode_buffer calls; scans() */
    bool d_ih;          /* implicit header: no length on air */
    int d_ih_len;       /* message length the caller promises */

    std::vector<gr_complex> d_buf;
    long d_consumed;           /* what the last scan finished with */
    long d_complete_at = -1;   /* where a located, still-arriving frame's
                                * last symbol ends, as an index into d_buf;
                                * -1 when none is pending */
    long d_search_from;        /* where the next scan's walk starts */

    /* final_walk: nothing more is coming -- the end of a burst or a
     * stream, or stop(); not the idle flush, after which more may come. */
    void process(bool flush = false, bool final_walk = false);
    bool scan_would_finish_nothing() const;

public:
    decoder_impl(int sf, double bw, double samp_rate, int window,
                 bool channel_filter_rx, bool soft_decision, bool robust_sync,
                 int sync_phases, bool pseudo_header,
                 bool implicit_header, int payload_len, int cr,
                 bool has_crc, std::vector<int> sync_word, int ldro,
                 bool fec_correct,
                 bool interpolate, bool report_crc_errors,
                 int max_payload_len,
                 bool invert_iq,
                 double trise, double cutoff_ratio, int numtaps,
                 double sync_threshold, int search_tail_symbols,
                 int min_osf, int interp_target_osf,
                 int hop_period, std::vector<double> hop_freqs,
                 bool low_latency, double center_freq);
    ~decoder_impl() override;

    uint64_t packets() const override;
    uint64_t crc_errors() const override;
    uint64_t overruns() const override;
    uint64_t scans() const override;
    std::size_t retention_samples() const override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

    /* Decode whatever is left when the stream ends. work() only calls
     * process() once a full window has accumulated, so this is what decodes
     * the last partial window of a file source or a burst transmitter. */
    bool stop() override;
    /* Starts the idle-flush watchdog; see watch(). */
    bool start() override;

    void set_idle_flush(double seconds) override;
    double idle_flush() const override;

private:
    /* THE IDLE FLUSH. d_mx serialises work(), stop() and the
     * watchdog thread over every member process() touches. */
    void watch();
    mutable std::mutex d_mx;
    std::condition_variable d_cv;
    std::thread d_watch;
    bool d_watch_stop = false;
    bool d_dirty = false;      /* samples in since the last flush */
    std::chrono::steady_clock::time_point d_last_work;
    std::atomic<double> d_idle{ 0.25 };   /* seconds; 0 disables */
};

} // namespace sdrlora
} // namespace gr

#endif
