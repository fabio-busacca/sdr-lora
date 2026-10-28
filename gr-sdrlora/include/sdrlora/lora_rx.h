/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
/*
 * LoRa PHY, receive chain. Pure C++, no GNU Radio dependency, so like the
 * transmit side it can be compiled and used on its own.
 *
 * Split into two halves on purpose:
 *
 *   packet_rx()  demodulation and decode GIVEN a known offset and CFO. Fully
 *                deterministic. This is also the hot path -- it is where the
 *                decoder spends its time.
 *
 *   find_preamble() and decode_buffer()  the sync search and the buffer
 *                walk, below. Threshold-dependent, so their behaviour is
 *                characterised statistically -- detection performance on
 *                captures -- rather than byte-wise.
 *
 * The FFT is over the DECIMATED symbol, whose length is K = 2^SF, always a
 * power of two. A self-contained radix-2 transform is therefore exact for every
 * SF and every oversampling factor, and the module keeps its no-dependency
 * property.
 */
#ifndef INCLUDED_SDRLORA_LORA_RX_H
#define INCLUDED_SDRLORA_LORA_RX_H

#include "sdrlora/lora_phy.h"

#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

namespace gr {
namespace sdrlora {

/* In-place radix-2 DFT. sign = -1 for the forward transform, +1 for the
 * inverse (unscaled). n must be a power of two. The DEMODULATOR's
 * transform: the deterministic half of the receiver keeps it. */
void fft_radix2(std::vector<std::complex<double> >& a, int sign);

/* The same contract, faster: a Stockham autosort FFT, radix-4 stages and
 * one radix-2 stage when log2(n) is odd, with its twiddles computed
 * directly and cached per thread. The SYNC SEARCH's transform -- the
 * channel filter, the preamble search and fine sync, which are judged
 * statistically rather than byte for byte. Several times faster
 * than fft_radix2 from 256 to 131072 points, where the transform dominates
 * an idle scan, and more accurate: directly computed twiddles do not
 * accumulate the rounding error of fft_radix2's recurrence-built ones. */
void fft_fast(std::vector<std::complex<double> >& a, int sign);

struct rx_result {
    bool truncated = false;
    bool hdr_ok = false;
    bool crc_ok = false;
    bool has_crc = false;
    int cr = -1;          /* read from the header */
    int length = -1;      /* from the in-payload pseudo-header */
    int src = -1, dst = -1, seqn = -1;
    std::vector<uint8_t> payload;
    /* Symbol estimates, SIGNED: stored as argmax(...) - 1, so -1 occurs
     * whenever the peak lands in bin 0. Normalising it to K-1 is numerically
     * equivalent downstream but changes the reported indices, so don't. */
    std::vector<int32_t> k_hdr;
    std::vector<int32_t> k_payload;
    /* Samples consumed from p_ofs. The buffer walk uses it to step past this
     * packet. */
    long consumed = 0;
    /* When `truncated` with the frame's length already known -- the header
     * decoded, or an implicit length -- where its last symbol ends, in
     * samples from p_ofs; -1 otherwise. decode_buffer reports it so a
     * streaming caller need not re-scan the frame until it can complete. */
    long complete_at = -1;
    /* A frame WITHOUT a CRC: the mean peak gap over its symbols after the
     * sync word, each capped at NOCRC_GAP_CAP (peak_gap), which the no-CRC
     * check compares with nocrc_min_gap(SF). NaN for a frame with a CRC, or
     * one that did not decode that far. */
    double gap_score = std::numeric_limits<double>::quiet_NaN();
};

/* Result of the preamble search. */
struct sync_result {
    bool found = false;
    bool truncated = false;
    long p_ofs = 0;       /* packet start, relative to the buffer handed in */
    double cfo_hz = 0.0;
};

/* Band-limit received samples to the LoRa channel. Worth ~6 dB against noise
 * that extends beyond the channel, because the symbol demodulator's decimation
 * folds every Nyquist zone onto the signal. */
void channel_filter(const std::vector<std::complex<float> >& in, double bw,
                    double fs, std::vector<std::complex<float> >& out,
                    double cutoff_ratio = 1.15, int numtaps = 255);

/* Lowest fs/bw the demodulator works at. Below it decode_buffer interpolates
 * rather than refusing -- see upsample(). */
static const int MIN_OSF = 2;

/* What decode_buffer interpolates UP TO when fs/bw is below MIN_OSF --
 * deliberately a different number. Clearing the minimum is what makes the
 * timing estimator work at all; 4 is what makes it work well, and it costs
 * nothing anywhere else because fs/bw = 1 is the only ratio below the minimum
 * in the first place.
 *
 * Against 2x, interpolating to 4x decodes clearly more frames at SF8..SF9
 * and is indistinguishable elsewhere; it is never worse. The gain sits in
 * that band because below it the preamble search dominates and above it the
 * processing gain carries the frame; SF8..SF9 is where the timing estimator
 * is what binds.
 *
 * Not "more is better" either: 8x decodes fewer frames than 4x at SF8..SF9,
 * because interpolating reshuffles which aliased artefacts the sync path sees
 * rather than removing them. Hence a specific number. */
static const int INTERP_TARGET_OSF = 4;

/* The interpolation filter's length: upsample()'s default, and what
 * decoder_impl holds back below MIN_OSF (INTERP_TAPS / 2 caller samples)
 * so no walked sample is interpolated against zero padding. */
static const int INTERP_TAPS = 127;

/* Band-limited interpolation by an integer factor: zero-stuff, then lowpass at
 * the ORIGINAL Nyquist. What a polyphase interpolator computes, without the
 * phase factoring.
 *
 * This creates no information. fs = bw is critically sampled for complex IQ,
 * not undersampled, so the samples already determine the waveform; what fails
 * there is the timing estimator, whose parabolic peak interpolation reads low
 * when the dechirped preamble peak is distorted. Handing the same estimator
 * the same samples on a finer grid fixes it; how much finer is
 * INTERP_TARGET_OSF above.
 *
 * Against FFT zero-padding -- the ideal band-limited interpolator -- any
 * windowed-sinc from 63 taps up decodes the same frames, so the taps stay
 * modest and the cutoff sits exactly at fs/2. A cutoff above that is worse:
 * the zero-stuffing image lands on the chirp's own sweep extremes. */
void upsample(const std::vector<std::complex<float> >& in, double fs,
              int factor, std::vector<std::complex<float> >& out,
              int numtaps = INTERP_TAPS);

/* Scan granularity to use when the caller does not choose one (sync_phases
 * <= 0). The preamble search advances by N/sync_phases samples, so what
 * matters is the step in chips: 2^SF / sync_phases. Two phases is half a
 * symbol at every SF, but at SF5 that is 16 chips of 32 and the fine-sync
 * estimate has a quarter of the preamble SF7 has, so whether the sync word is
 * found depends on where the frame falls in the scan grid. Hence 4 below
 * SF7, 2 from SF7 up, which leaves the real-time cost from SF7 up unchanged
 * and costs little where it applies -- an SF5 packet is a sixteenth of an
 * SF7 one. */
int default_sync_phases(int SF);

/*! The oversampling find_preamble will actually run at, or 0 when no
 *  interpolation happens.
 *
 *  decode_buffer interpolates only when `osf < min_osf && interpolate`, and
 *  then by ceil(interp_target_osf / osf) -- so the ratio the sync FFTs are
 *  built from is osf * that factor, not osf and not interp_target_osf.
 */
long effective_osf(int osf, int min_osf, int interp_target_osf,
                   bool interpolate);

/*! Refuse an interpolated oversampling the power-of-two sync FFTs cannot use.
 *
 *  FACTORED so decoder_impl's constructor and decode_buffer share one copy.
 *  Refusing inside decode_buffer alone is fine for a one-shot decode and
 *  not for a GNU Radio block: a throw from work() leaves through the
 *  scheduler's thread body, which does not catch std::exception, so the
 *  whole flowgraph would die mid-stream. The block therefore refuses at
 *  construction. At osf 1 with the default min_osf, most interp_target_osf
 *  values the palette permits give a ratio that is not a power of two, e.g.
 *  sf 7 / bw 125000 / samp_rate 125000 / interp_target_osf 3.
 *
 *  Same root cause as the power-of-two rate rule: the sync FFTs need
 *  power-of-two lengths.
 */
void check_effective_osf(int osf, int min_osf, int interp_target_osf,
                         bool interpolate);


/* Geometry of the live decoder block's buffering, declared here rather than
 * inside decoder_impl.cc so that everything that needs it shares one copy
 * that cannot drift.
 *
 * max_packet_symbols: the longest legal packet, in symbols. 251 payload bytes
 * plus the 4-byte pseudo-header plus the 2-byte CRC, at the weakest coding
 * rate, carried 4*PPM bits per (4+cr)-symbol block with cr = 4 over
 * PPM = SF - 2*DE bits per symbol, plus the 8-symbol header and the
 * 12.25-symbol preamble.
 *
 * live_overlap_samples: what consecutive windows must share, or a frame on a
 * boundary is truncated in the window that starts it and has no preamble in
 * the window that ends it -- lost by both. One maximum-length packet.
 *
 * live_window_samples: how far an OVERLAP-WINDOW receiver advances between
 * decode attempts; each sample is then scanned (window + overlap) / window
 * times, so the window sets the steady-state cost as well as the latency.
 * The decoder block does NOT use this scheme -- it advances by its window,
 * max(32 * 2^SF, 4096) * osf by default, and retires on SEARCH_TAIL_SYMBOLS
 * -- so this function describes the cost model of an overlap-window
 * receiver, as a reference for real-time budgeting. */
/* `de` is the low-data-rate flag. It MATTERS here: LDRO carries 2 fewer bits
 * per symbol, so the longest legal packet grows by about 40% at SF7. Passing
 * -1 applies the SF>10 rule, which is what the auto rule gives at
 * bw = 125 kHz; the decoder block passes its own resolved flag, so forcing
 * LDRO on at a low spreading factor does not undersize the overlap and drop
 * frames that straddle a window boundary. */
long max_packet_symbols(int SF, int de = -1, int max_payload = 251);
size_t live_overlap_samples(int SF, int osf, int de = -1,
                            int max_payload = 251);
size_t live_window_samples(int SF, int osf);

/* Scan for a preamble. Detection uses an accumulation of the 8 identical
 * preamble up-chirps when robust_sync is set; the sync-word check is strict
 * either way. sync_phases sets the scan granularity (window advance N/phases);
 * <= 0 means default_sync_phases(p.SF).
 *
 * Requires OSF = fs/bw to be a power of two, because the sync FFTs are length
 * N and 4N rather than the decimated K. Reports found=false otherwise. */
sync_result find_preamble(const std::vector<std::complex<float> >& s,
                          const tx_params& p, long start = 0,
                          bool robust_sync = true, int sync_phases = 0);

/* Scan a buffer and decode every packet in it.
 *
 * search_from and walk_end are for a STREAMING caller, and both count the
 * caller's samples. The walk starts at search_from -- where the previous
 * scan stopped -- and stops at walk_end (-1: the whole buffer), while the
 * front end filters ALL of `in`: samples past walk_end are right-hand
 * context for the ones before it and are walked by a later scan, once
 * their own context has arrived. */
std::vector<rx_result> decode_buffer(const std::vector<std::complex<float> >& in,
                                     const tx_params& p,
                                     bool channel_filter_rx = true,
                                     bool robust_sync = true,
                                     int sync_phases = 0,
                                     bool soft_decision = true,
                                     bool implicit_header = false,
                                     int ih_payload_len = 0,
                                     bool fec_correct = true,
                                     bool interpolate = true,
                                     bool report_crc_errors = false,
                                     long* consumed_out = 0,
                                     long search_from = 0,
                                     long walk_end = -1,
                                     long* complete_at_out = 0,
                                     /* nothing more is coming: a frame cut
                                      * off by the end is a failed lock, not
                                      * the end of the walk. 1 yes, 0 no;
                                      * -1, the default, means
                                      * final unless the caller asked for
                                      * consumed_out, which only a streaming
                                      * caller does -- a default of "final"
                                      * would make every such caller trim
                                      * frames still arriving. The block
                                      * passes it explicitly. */
                                     int final_walk = -1);

/* Upper bound on sync_phases. It bounds two things: the preamble walk
 * advances by N / sync_phases, integer division, so a value above N would
 * make the step 0 and the search loop would never terminate; and
 * robust_sync's accumulator ring is sync_phases * (ACC_SYMBOLS + ACC_LAG) *
 * N * 8 bytes, which grows without limit with it. Four times the largest
 * default (default_sync_phases returns 2 or 4), which caps the ring at 42 MB
 * at SF12 / osf 8. */
static const int SYNC_PHASES_MAX = 16;

/* Samples a streaming caller must keep after decode_buffer() returns, so that
 * a preamble straddling the end of one call is still found in the next.
 * find_preamble() needs the 12.25-symbol preamble/sync/down-chirp pattern plus
 * the accumulation history behind it (ACC_LAG + ACC_SYMBOLS); 24 symbols is
 * that with room to spare. A retained tail of this length, rather than a
 * whole maximum-length frame of overlap, is what keeps the receiver's
 * delivery latency low. */
static const int SEARCH_TAIL_SYMBOLS = 24;

/* Half the channel filter's length. It is applied `same`-mode, so the first
 * and last FILTER_HALF samples of any buffer are filtered against zero padding
 * rather than signal. A streaming caller holds back that much at the end and
 * backs its consume point off by that much at the start, so every sample is
 * scanned at least once with real context on both sides. */
/* The DEFAULT half-width, i.e. FILTER_TAPS/2. Callers that hold samples
 * back must use their own numtaps/2 -- decoder_impl does -- because
 * numtaps is a parameter over 3..8191 and this constant only describes
 * the default: as the hold-back at numtaps = 8191 it would leave 3968
 * samples per scan unfiltered. */
static const size_t FILTER_HALF = 127;

/* Hamming syndrome decoding.
 * CR 4/5 and 4/6 have d_min = 2 and cannot locate an error, so they detect
 * only; 4/7 and 4/8 correct a single bit. Returns (corrected, uncorrectable).
 * C is [rows][4+cr], data bits first, and is modified in place. */
void hamming_correct(std::vector<std::vector<uint8_t> >& C, int cr,
                     int& corrected, int& uncorrectable);

/* Per-bit LLRs from dechirped FFT magnitudes, max-log approximation.
 * `mag` is n_sym rows of K magnitudes; returns n_sym rows of PPM LLRs, with a
 * POSITIVE value meaning the bit is more likely 0. */
void symbol_llr(const std::vector<std::vector<double> >& mag, int SF, int de,
                std::vector<std::vector<double> >& llr);

/* Maximum-likelihood decode of soft codewords: exhaustive over the 16
 * candidates, which is optimal and cheap. Unlike syndrome decoding this helps
 * at every coding rate, including the d_min = 2 rates where a hard decoder
 * cannot locate an error at all. Returns 4 data bits per row. */
void hamming_soft_decode(const std::vector<std::vector<double> >& L, int cr,
                         std::vector<std::vector<uint8_t> >& data);

/* Demodulate and decode one packet whose start and CFO are already known.
 *
 * s        baseband samples
 * p_ofs    index of the packet's rising edge within s
 * cfo_hz   carrier frequency offset to remove
 * ih       implicit header: pass length/cr/has_crc when true
 * soft_decision
 *          decode the payload from the FFT magnitudes via LLRs and ML rather
 *          than committing to a hard argmax first. Supersedes the hard
 *          syndrome correction for the payload when enabled.
 * sfo      sample-clock offset as a fraction (cfo_hz / carrier), 0 for none;
 *          see sym_start in lora_rx.cc.
 */
rx_result packet_rx(const std::vector<std::complex<float> >& s,
                    const tx_params& p, long p_ofs, double cfo_hz,
                    bool implicit_header = false, int ih_length = 0,
                    int ih_cr = 1, bool ih_has_crc = true,
                    bool soft_decision = true, bool fec_correct = true,
                    double sfo = 0.0,
                    std::vector<double>* fracs_out = 0);

/* Where, in symbols past the frame's leading edge, the fine-sync timing
 * estimate is exact: the mean centre of its four windows. */
constexpr double SFO_REF_SYMBOLS = 8.5;

/* THE DRIFT READ FROM THE SIGNAL, DRIFT_* and drift_from_fracs(): when a
 * frame locates and fails its CRC, the slope of
 * its payload symbols' fractional peak offsets is the sample-clock drift the
 * windows did not follow, and decode_buffer demodulates again with it,
 * keeping the result only if the CRC passes. */
constexpr int DRIFT_RETRIES = 3;
constexpr int DRIFT_MIN_SYMBOLS = 8;
constexpr double DRIFT_MIN_BINS = 0.1;
constexpr int DRIFT_LAGS[3] = { 1, 4, 16 };   /* coarse to fine */
/* Below fs/BW 4 the fractional offset misreads a small residual, so after
 * the estimate these offsets around the carrier's are tried, CRC-gated. */
constexpr int DRIFT_MIN_OSF = 4;
constexpr double DRIFT_GRID_PPM[6] = { 2.0, -2.0, 4.0, -4.0, 6.0, -6.0 };
/* The LDRO flip is tried only from SF11, where LDRO mismatches between
 * devices occur in practice; below it the flip only costs time. */
constexpr int LDRO_RETRY_MIN_SF = 11;

/* The residual sample-clock offset, as a fraction, from the fractional
 * peak offsets packet_rx collected (fracs_out). */
double drift_from_fracs(const std::vector<double>& fracs, int SF);

/* THE NO-CRC CHECK, peak_gap() and nocrc_min_gap(): a frame without a CRC
 * is reported crc_ok only if the mean,
 * over its symbols after the sync word, of how far each dechirped peak
 * stands above the largest bin outside its main lobe -- in units of those
 * bins' mean power, capped at NOCRC_GAP_CAP -- reaches nocrc_min_gap(SF).
 * In noise each gap is Exp(1), mean 1, at every SF. */
constexpr double NOCRC_GAP_CAP = 8.0;
constexpr double NOCRC_MIN_GAP_DEFAULT = 2.5;
double peak_gap(const std::vector<double>& power, size_t pos);
double nocrc_min_gap(int SF);

} /* namespace sdrlora */
} /* namespace gr */

#endif /* INCLUDED_SDRLORA_LORA_RX_H */
