/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
/*
 * LoRa PHY, transmit chain. Pure C++ with no GNU Radio dependency, so it can
 * be compiled and used on its own.
 */
#ifndef INCLUDED_SDRLORA_LORA_PHY_H
#define INCLUDED_SDRLORA_LORA_PHY_H

/* <algorithm> for std::min below; included directly rather than relying on
   a transitive include from <vector>, which not every standard library
   provides. */
#include <algorithm>
#include <cmath>        /* std::isfinite, std::floor: check_rate_bounds */
#include <complex>
#include <cstdint>
#include <cstdio>       /* std::snprintf: check_rate_bounds' messages */
#include <stdexcept>    /* std::invalid_argument: check_rate_bounds */
#include <vector>

namespace gr {
namespace sdrlora {

/* Rising/falling edge duration. */
static const double TRISE = 50e-6;

/* Smallest hop dwell, in symbols. A DELIBERATE FLOOR, not a limit of the
 * receiver: the schedule is anchored at the header, so nothing before it is
 * rotated at any hop period. 13 is the length of the sync pattern -- 8
 * preamble symbols, 2 sync words, 2.25 down-chirps -- and a hop shorter than
 * that is outside the range this implementation is characterised for. Not a
 * practical limit either: hopping exists for dwell-time rules, and 400 ms at
 * SF7/125 kHz is ~390 symbols. */
static const int HOP_PERIOD_MIN = 13;
static const int HOP_PERIOD_MAX = 65535;
static const int HOP_FREQS_MAX = 256;

/* Largest frame, in samples. The per-parameter bounds do not compose:
 * every field of a SF12 / fs-over-bw-8 / 65535-preamble frame is individually
 * legal and the waveform is 2.15e9 samples, 17.2 GB of complex64. */
static const long MAX_FRAME_SAMPLES = 1L << 28;

/* THE SAMPLE-RATE BOUNDS, shared by every entry point that converts the rate
 * to an integer.
 *
 * Finiteness alone does not make the conversions defined: (int)(ratio + 0.5)
 * overflows from 2^31 up, and (long)ceil(trise*fs) for bw = fs = 1e19.
 * Bounding both fs and the quotient fs/bw keeps every conversion in range.
 *
 * OSF_MAX: 500 MS/s over the narrowest LoRa channel, 7.8125 kHz, is 64000,
 * so this admits every real front end, and 2^SF * OSF_MAX = 2^28 keeps the
 * symbol length well inside a long. FS_MAX: three orders of magnitude above
 * any ADC; it bounds trise*fs (trise <= 1 s) and the hop anchor, the other
 * sample-index conversions of the rate. */
static const long OSF_MAX = 65536;
static const double FS_MAX = 1e12;

inline void check_rate_bounds(double bw, double fs)
{
    if (!std::isfinite(bw) || !std::isfinite(fs))
        throw std::invalid_argument("sdrlora: bw and fs must be finite");
    if (bw <= 0.0 || fs <= 0.0)
        throw std::invalid_argument("sdrlora: bw and fs must be positive");
    /* The limits are FORMATTED IN from the constants, not spelled in the
     * text, so a message can never disagree with the bound it reports. */
    char msg[256];
    if (fs > FS_MAX) {
        std::snprintf(msg, sizeof msg,
                      "sdrlora: fs must be <= %g Hz -- three orders of "
                      "magnitude above any ADC; the bound keeps trise*fs and "
                      "the hop anchor, which are converted to sample indices, "
                      "inside a long", FS_MAX);
        throw std::invalid_argument(msg);
    }
    /* `!(x <= bound)`, so an overflowed quotient (1e12 / 1e-300 = inf) is
     * refused here too rather than reaching a conversion. */
    if (!(fs / bw <= (double)OSF_MAX + 0.5)) {
        std::snprintf(msg, sizeof msg,
                      "sdrlora: fs/bw must be <= %ld -- 500 MS/s over the "
                      "narrowest LoRa channel (7.8125 kHz) is 64000; the bound "
                      "keeps the oversampling factor and the 2^SF * fs/bw "
                      "symbol inside the integer types they are converted to",
                      OSF_MAX);
        throw std::invalid_argument(msg);
    }
}

/* Copies of lora_rx.h's SEARCH_TAIL_SYMBOLS, MIN_OSF and INTERP_TARGET_OSF.
 * They are duplicated because this header cannot include lora_rx.h without a
 * cycle. Both copies must agree; the lora_rx.h copy is the one
 * decoder_impl.cc sizes the block's retained tail from. */
static const int SEARCH_TAIL_SYMBOLS_DEFAULT = 24;
static const int MIN_OSF_DEFAULT = 2;
static const int INTERP_TARGET_OSF_DEFAULT = 4;
/* A typical crystal error, in ppm, for simulating an impaired link:
 * set tx_params::cfo_ppm = CFO_PPM. Not the default. */
static const double CFO_PPM = 17.0;

/* Symbol duration above which the low-data-rate option applies, in ms. Same
 * constant gr-lora_sdr uses (LDRO_MAX_DURATION_MS). */
static const double LDRO_MAX_DURATION_MS = 16.0;

/* The DE flag: 1 when the low-data-rate option is in force. ldro is 0 (off),
 * 1 (on) or 2 (auto, by symbol duration). */
inline int ldro_enabled(int SF, double bw, int ldro)
{
    if (ldro == 0) return 0;
    if (ldro == 1) return 1;
    return (((double)(1u << SF) * 1e3 / bw) > LDRO_MAX_DURATION_MS) ? 1 : 0;
}

/* The two preamble symbols a sync word is carried as: nibbles scaled by 8.
 * Returns false for a zero nibble, whose symbol would coincide with the
 * preamble up-chirps and be undetectable. */
inline bool sync_word_symbols(int sync_word, int& s1, int& s2)
{
    if (sync_word < 0 || sync_word > 0xFF) return false;
    const int hi = (sync_word >> 4) & 0xF, lo = sync_word & 0xF;
    if (hi == 0 || lo == 0) return false;
    s1 = hi * 8;
    s2 = lo * 8;
    return true;
}

/* The list form. One element is a byte, two are the raw symbols. Matches
 * gr-lora_sdr: [0x34] and [24, 32] are the same waveform there and here. */
inline bool sync_word_symbols(const std::vector<int>& sync_word, int& s1,
                              int& s2)
{
    if (sync_word.size() == 1) return sync_word_symbols(sync_word[0], s1, s2);
    if (sync_word.size() != 2) return false;
    s1 = sync_word[0];
    s2 = sync_word[1];
    /* Symbol 0 is refused here; symbol 1 is refused too, by
     * sync_symbols_separable() below, which takes out every symbol within a
     * bin of the preamble -- 1 and 2^SF - 1 -- once SF is known. Symbol 0
     * coincides with the preamble up-chirps outright; symbol 1 sits ONE bin
     * from them and the sync check runs with a tolerance of one bin, so it is
     * not reliably separable from more preamble. Refusing it costs nothing
     * real: a byte sync word always yields multiples of 8.
     *
     * Upper bound is the widest symbol alphabet (SF12); encode and the sync
     * check reduce mod 2^SF. */
    return s1 > 0 && s2 > 0 && s1 < 4096 && s2 < 4096;
}

/* Can the sync symbols be told apart from the preamble at this SF?
 *
 * A sync symbol is detected as a bin offset from the preamble's own bin with a
 * tolerance of ONE bin, and bins are cyclic mod 2^SF -- so symbol 1 and symbol
 * 2^SF - 1 are both one bin away. 127 is adjacent to the preamble at SF7 and
 * 127 bins from it at SF12, which is why this needs SF and why
 * sync_word_symbols() cannot decide it.
 *
 * Pairs within a bin of the preamble are marginal: whether one is found
 * depends on implementation details, so receivers disagree on them, and
 * gr-lora_sdr does not reliably receive them from itself. Refusing the band
 * outright makes the behaviour deterministic, and it costs nothing real: a
 * byte sync word gives multiples of 8, never within a bin of the preamble at
 * any SF. */
inline bool sync_symbols_separable(int s1, int s2, int SF)
{
    const int K = 1 << SF;
    const int a = ((s1 % K) + K) % K, b = ((s2 % K) + K) % K;
    return std::min(a, K - a) >= 2 && std::min(b, K - b) >= 2;
}


struct tx_params {
    int SF = 7;          /* 5..12; below 7 implicit_header is mandatory      */
    int cr = 1;          /* 1..4, i.e. coding rate 4/(4+cr)                   */
    bool has_crc = true;
    bool implicit_header = false;
    int preamble_symbols = 8;
    int src = 0;
    int dst = 0;
    int seqn = 0;
    double f0 = 868e6;   /* carrier, only used for the CFO term               */

    /* Carrier offset baked into the waveform, in ppm of f0, for simulating
     * crystal error. Applied twice (as Cfo_TX and again as Cfo, one per end
     * of the link), so the offset on the waveform is 2 * cfo_ppm * 1e-6 *
     * f0. Default 0, a clean carrier: a real transmitter
     * brings its own offset. CFO_PPM (17) is a typical crystal, 29.5 kHz at
     * 868 MHz.
     *
     * gr-lora_sdr's capture range is about a quarter of the bandwidth, and
     * 29.5 kHz is 0.24 of 125 kHz: a simulated 17 ppm is just inside its
     * limit from SF6 up and just outside it at SF5, where its capture range
     * is slightly narrower. A margin, not a property of SF5. */
    double cfo_ppm = 0.0;

    /* Fractional sampling instant, in samples, and the initial phase: the
     * chirp's time axis is (t0_frac + n) * Ts, n = 0..N-1, starting at phase
     * phi0.
     *
     * encode() leaves these at 0, so they do not change the default waveform
     * -- they exist to construct an off-grid stimulus, e.g. a signal sampled
     * half a sample late, which is the hard case for the timing estimator at
     * fs/bw = 1. */
    double t0_frac = 0.0;
    double phi0 = 0.0;

    /* Raised-edge duration in seconds, and the output level. The defaults
     * are TRISE and unit magnitude. */
    double trise = TRISE;
    double amplitude = 1.0;

    /* ---- receive side ----
     * decode_buffer takes this same struct (p.ldro and p.sync_word are
     * already read on both the transmit and the receive side), so the
     * receive knobs live here rather than in a second struct.
     *
     * cutoff_ratio and numtaps are passed through to channel_filter.
     * sync_threshold is find_preamble's detection threshold; 5.5 is a
     * working value rather than a characterised optimum, which is why it is
     * a parameter. */
    /* The RF carrier in Hz the samples were tuned to, for sample-clock
     * tracking; 0 is off and the default. NOT f0, which is the transmit
     * side's and defaults to 868 MHz: reading it here would turn tracking on
     * for every caller that leaves f0 alone. check_rx_params validates
     * it. */
    double center_freq = 0.0;
    /* Frequency hopping: hop_period symbols on each channel, with the
     * list supplied by the host rather than carried on the air. 0 is off,
     * matching RegHopPeriod.
     *
     * ANCHORED AT THE HEADER, not at the start of the packet, which is
     * where the SX127x anchors it. Everything before the header goes out
     * unrotated, so the preamble does NOT hop: modulate() keys the schedule
     * to p_ofs + nrise + (preamble_symbols + 4.25)*N. The frame start is a
     * point the receiver cannot locate, because its timing estimate is
     * computed as if the preamble were 8 symbols long; the header start is
     * exact at any preamble length. Hopped and unhopped frames are therefore
     * byte-identical over the whole pre-header region.
     *
     * hop_freqs[0] must be 0: the list is relative to the channel the
     * receiver STARTS on, and it acquires the preamble at baseband. The
     * phase restarts at each hop, because a real synthesiser retunes and
     * comes back with an arbitrary phase -- which also makes the transform
     * exactly invertible, so neither end needs an absolute time origin. */
    int hop_period = 0;
    std::vector<double> hop_freqs;

    double cutoff_ratio = 1.15;
    int numtaps = 255;
    double sync_threshold = 5.5;
    int search_tail_symbols = SEARCH_TAIL_SYMBOLS_DEFAULT;
    int min_osf = MIN_OSF_DEFAULT;
    int interp_target_osf = INTERP_TARGET_OSF_DEFAULT;

    /* LoRa sync word: one byte, carried as the two preamble symbols that
     * follow the up-chirps -- high nibble * 8, then low nibble * 8. 0x12 is
     * the private/default value and gives 8 and 16. Configurable, as in
     * gr-lora_sdr, so frames with any other sync word can be produced and
     * received. */
    /* The same two forms gr-lora_sdr's `sync_word` list takes: one element
     * is a byte split into nibbles, two are the raw symbol values. The raw
     * form reaches what the byte form cannot -- (18, 52) is a sync word they
     * transmit and receive, and nibbles*8 are always multiples of 8. */
    std::vector<int> sync_word{ 0x12 };

    /* Low-data-rate optimisation: 0 off, 1 on, 2 auto. The real LoRa rule is
     * about symbol DURATION, not spreading factor -- enable it once 2^SF / bw
     * exceeds LDRO_MAX_DURATION_MS. "SF > 10" is the same thing only at
     * bw = 125 kHz: at 250 kHz the standard rule leaves SF11 off, and a
     * receiver that disagrees on the flag decodes nothing from the other. */
    int ldro = 2;

    /* Zero samples emitted before the preamble. The receiver here decodes
     * with none at all, so this is pure transmitter-on dead air -- 10 ms per
     * frame at fs = 1 MHz. gr-lora_sdr's equivalent is `frame_zero_padd`. */
    int lead_in = -1;          /* negative = auto, see resolved_lead_in */

    double bw = 125e3;
    double fs = 1e6;     /* must be an integer multiple of bw                 */

    /* This implementation prefixes every payload with its own 4-byte
     * DST/SRC/SEQNO/LENGTH block. That is NOT part of LoRa. Set false for a
     * standard frame -- message at payload byte 0, the PHY header's LENGTH is
     * the message length, src/dst/seqn unused -- which is both 4 bytes
     * cheaper on air and what a third-party transmitter sends. Required to
     * receive from gr-lora_sdr: with the prefix assumed, its first four
     * payload bytes are eaten as DST/SRC/SEQNO/LENGTH and nothing decodes. */
    bool pseudo_header = true;

    /* Transmit (and receive) with I and Q swapped. For a complex baseband
     * signal that is exactly the conjugate, which is what the SX127x InvertIQ
     * bits do: every chirp in the frame reverses direction, so the preamble
     * sweeps DOWN and the SFD up.
     *
     * This is the polarity a LoRaWAN DOWNLINK uses. Gateways transmit
     * inverted and listen normal, end devices the reverse, so neither hears
     * its own kind -- an inverted frame does not decode as a normal one and
     * vice versa, and that separation is the point of the feature rather than
     * a side effect. Without it this implementation can speak only the uplink
     * half of a LoRaWAN link.
     *
     * Applied to the MODULATION only, before the carrier offset, because a
     * transmitter's oscillator error comes from its crystal and does not
     * invert with the modulator. With cfo_ppm = 0 an inverted frame is
     * bit-identical to conj() of the normal one; with the offset on, the
     * offset's sign is the only difference. */
    bool invert_iq = false;
};

/* ---------------------------------------------------------------- helpers */

/* MSB-first bits of `value`, `nbits` wide. */
void num2binary(uint32_t value, int nbits, std::vector<uint8_t>& out);

/* Gray tables for a PPM-bit alphabet.
 *   gray[v]   -- index whose Gray code is v   (the transmitter's mapping)
 *   degray[i] -- Gray code of index i         (the receiver's mapping)
 */
void gray_lut(int ppm, std::vector<uint32_t>& gray, std::vector<uint32_t>& degray);

/* LoRa's non-standard CRC-16. `bits` is a multiple of 8, LSB-first per byte
 * as laid out by payload_init(). Returns 16 bits. */
void crc16(const std::vector<uint8_t>& bits, std::vector<uint8_t>& out);

/* ------------------------------------------------------------ tx chain */

/* Header symbol count and the payload bits the header carries. */
void header_init(int SF, bool implicit_header, int& n_sym_hdr, int& n_bits_hdr);

/* Byte packing, CRC and whitening. Produces the whitened payload bit array
 * and the number of payload symbols it will occupy. */
void payload_init(const tx_params& p, const std::vector<uint8_t>& msg,
                  std::vector<uint8_t>& payload_bits, int& n_sym_payload);

/* Header symbols. Advances `payload_ofs` past the payload bits the header
 * absorbed (non-zero from SF8 up). */
void header_symbols(const tx_params& p, int length,
                    const std::vector<uint8_t>& payload_bits,
                    std::vector<uint32_t>& k_hdr, int& payload_ofs);

/* Payload symbols: Hamming, diagonal interleave, Gray. */
void payload_symbols(const tx_params& p, int n_sym_payload,
                     const std::vector<uint8_t>& payload_bits, int payload_ofs,
                     std::vector<uint32_t>& k_payload);

/* Payload bytes -> symbol indices. This is the whole integer half of the
 * transmit chain. */
void encode_symbols(const tx_params& p, const std::vector<uint8_t>& msg,
                    std::vector<uint32_t>& k_hdr,
                    std::vector<uint32_t>& k_payload);

/* Symbol indices -> baseband IQ, including preamble, edges and the CFO
 * term. */
void modulate(const tx_params& p, const std::vector<uint32_t>& k_hdr,
              const std::vector<uint32_t>& k_payload,
              std::vector<std::complex<float> >& out);

/* Reject a tx_params the PHY cannot honour -- SF out of range, SF5/SF6 with an
 * explicit header, cr outside 1..4, no preamble, a lead-in past LEAD_IN_MAX (a
 * negative one means auto), a non-positive rate, an ldro outside 0..2. The
 * GNU Radio blocks check these in their constructors; this is the same rule
 * for callers of the core API. Called by encode(). */
void check_tx_params(const tx_params& p);

/* The receive-side subset: everything above except the transmit-only fields
 * (preamble_symbols, lead_in, amplitude, t0_frac, src, dst, seqn), and the
 * seven receive knobs. Called by decode_buffer() and by decoder_impl's
 * constructor, so that e.g. numtaps = 1 (an all-NaN channel filter) or
 * interp_target_osf = 1e6 (a terabyte allocation request) is refused up
 * front. check_tx_params() calls this first, so the two agree by
 * construction rather than by a duplicated list. */
void check_rx_params(const tx_params& p);

/* Samples the frame will occupy, and the refusals built on it: past
 * MAX_FRAME_SAMPLES, and a carrier phase 2*pi*cfo*t that overflows over the
 * modulated part. Separate from check_tx_params because the
 * payload symbol count is only known once payload_init has run. */
double frame_samples_estimate(const tx_params& p, long n_sym_payload);
void check_frame_size(const tx_params& p, long n_sym_payload);

/* Largest bandwidth whose raised edge still fits inside one symbol.
 *
 * The edge is TRISE seconds long, so nrise = ceil(TRISE*bw*osf) samples, and a
 * symbol is N = 2^SF*osf. nrise > N therefore reduces to bw > 2^SF / TRISE,
 * independent of the sample rate -- 640 kHz at SF5, 1.28 MHz at SF6. Past that
 * the taper would index before the start of the preamble, so such a
 * combination is refused. Not hypothetical at low SF: the SX1280 uses 812 kHz
 * and 1625 kHz. */
inline double max_bandwidth_for_sf(int SF, double trise = TRISE)
{
    /* Against the CALLER's trise, not the constant: a longer edge means a
     * tighter bandwidth limit, and checking the constant would let a
     * lengthened taper index before the start of the preamble -- the
     * fault this bound exists to prevent. */
    return (double)(1L << SF) / trise;
}

/* Apply (or, with invert, undo) a hop schedule to `s`, anchored at
 * `frame_ofs`. Handles a NEGATIVE anchor -- the fine timing estimate can
 * land before the buffer, and an unguarded negative index would wrap to the
 * end of the array. */
/* sfo, ref: the sample-clock offset and the absolute sample where the
 * timing estimate is exact -- the block edges drift as the symbol windows
 * do. sfo == 0 places the edges with no drift correction. */
void apply_hops(std::vector<std::complex<float> >& s,
                            long frame_ofs, int hop_period,
                            const std::vector<double>& hop_freqs,
                            long n_sym, double fs, bool invert,
                            double sfo = 0.0, double ref = 0.0);

/* Smallest receive-filter cutoff, in multiples of bw, that passes every
 * channel in the plan. The filter runs BEFORE the de-hop, so a channel
 * outside the passband is gone before anything can bring it back. */
double hop_cutoff_ratio(const std::vector<double>& hop_freqs,
                                    double bw);

/*! The hop-plan rules: period range, a non-empty list of at most 256
 *  channels starting at 0, every channel finite and inside the sampled
 *  band. Called by check_rx_params and by packet_rx -- which does not call
 *  check_rx_params, so it validates the plan itself -- in both AFTER the
 *  rate, because the band is (fs - bw)/2 and negative when fs < bw. */
void check_hop_plan(const tx_params& p);

/*! Refuse a hop plan whose WIDENED receive cutoff the filter cannot take.
 *
 *  decode_buffer raises the channel-filter cutoff to
 *  hop_cutoff_ratio(hop_freqs, bw) = 1 + 2*max|f|/bw whenever a plan is in
 *  force -- the filter runs before the de-hop, so a channel outside the
 *  passband is destroyed before anything can recover it -- and
 *  channel_filter refuses a ratio above 64.
 *
 *  Validating the KNOB (<= 64) and the hop band (|f| <= (fs-bw)/2)
 *  separately is not enough: the real constraint is their combination,
 *  max|f| <= 31.5*bw. At bw 125 kHz / fs 16 MHz, max|f| = 4 MHz is inside
 *  the hop band but needs a ratio of 65; refusing it here, at
 *  construction, names the hop plan instead of letting the first
 *  process() fail on a cutoff_ratio the caller never set.
 */
void check_effective_cutoff(double cutoff_ratio, int hop_period,
                            const std::vector<double>& hop_freqs, double bw);

/* Upper bound on preamble_symbols. The SX1276's preamble-length register is
 * 16 bits wide, so this is the longest preamble any of the silicon this talks
 * to can be asked for -- 8192 times the standard 8. Bounded so that e.g.
 * preamble_symbols = 1 << 30 is refused by name instead of reaching the
 * modulator as an allocation request, where std::vector would throw
 * length_error or bad_alloc naming neither the parameter nor a limit. */
static const int PREAMBLE_SYMBOLS_MAX = 65535;

/* Upper bound on lead_in, in SAMPLES. Pure transmitter padding with no
 * hardware analogue, so the bound is chosen rather than cited: 2^26 is 67.1
 * million samples, 67 s of dead air at 1 Msps and 537 MB of complex float --
 * 6700 times the 10000-sample default. */
static const long LEAD_IN_MAX = 1L << 26;

/* The floor the auto value never goes below: the default lead-in. */
static const int LEAD_IN_FLOOR = 10000;

/* Symbols robust_sync accumulates before its detection statistic exists.
 * Shared because the TRANSMITTER has to know it too: see default_lead_in.
 * lora_sync.cc's ACC_SYMBOLS is this value. */
static const int SYNC_ACC_SYMBOLS = 6;

/* Zero samples to put ahead of the preamble when the caller asks for the
 * default (a negative lead_in on the encoder block).
 *
 * 10000 samples, topped up so that robust_sync can lock onto the result. The
 * accumulator needs SYNC_ACC_SYMBOLS symbols of history and it can come from
 * the lead-in or from the preamble -- but the lead-in is counted in SAMPLES,
 * so what it contributes shrinks as the symbol grows: 9.8 symbols at SF7 with
 * fs/bw = 8, and 0.31 at SF12. With a fixed lead-in, the shortest preamble
 * the default receiver could lock onto would rise with SF, and below that
 * line the encoder would emit a legal frame the decoder does not read.
 *
 * At the standard preamble_symbols = 8 the shortfall is zero, so this returns
 * exactly 10000 at every SF and rate. */
inline int default_lead_in(int SF, double bw, double fs,
                           int preamble_symbols = 8)
{
    /* VALIDATED, not clamped to a floor: the (long)(fs/bw + 0.5) conversion
     * below is undefined for NaN, +-inf and ratios past 9.2e18, and this can
     * be called before check_tx_params. */
    if (SF < 5 || SF > 12)
        throw std::invalid_argument("sdrlora: default_lead_in: sf must be 5..12");
    check_rate_bounds(bw, fs);
    /* osf is an INTEGER oversampling factor, rounded half up as
     * floor(ratio + 0.5), and defined because the ratio is bounded. */
    long osf = (long)std::floor(fs / bw + 0.5);
    if (osf < 1)
        osf = 1;
    const double n_sym = (double)(1L << SF) * (double)osf;
    /* In double, so an extreme preamble_symbols cannot overflow the
     * subtraction; SATURATED below, because the result is returned as int:
     * e.g. default_lead_in(12, 1.0, 1e6, 1) would otherwise wrap negative
     * and slip past a `> LEAD_IN_MAX` guard. */
    const double shortfall = (double)SYNC_ACC_SYMBOLS - (double)preamble_symbols;
    const double want = (shortfall > 0.0 ? shortfall * n_sym : 0.0);
    if (want >= (double)LEAD_IN_MAX)
        return (int)LEAD_IN_MAX;
    const long w = (long)want;
    return (int)(w > LEAD_IN_FLOOR ? w : LEAD_IN_FLOOR);
}

/* The lead-in a tx_params actually means. NEGATIVE MEANS AUTO -- one rule for
 * every entry point: the encoder block, the GRC yaml, check_tx_params and
 * phy_encode() all treat a negative value the same way. */
inline int resolved_lead_in(const tx_params& p)
{
    return p.lead_in < 0
               ? default_lead_in(p.SF, p.bw, p.fs, p.preamble_symbols)
               : p.lead_in;
}

/* Payload bytes -> IQ. Validates first. */
void encode(const tx_params& p, const std::vector<uint8_t>& msg,
            std::vector<std::complex<float> >& out);

} /* namespace sdrlora */
} /* namespace gr */

#endif /* INCLUDED_SDRLORA_LORA_PHY_H */
