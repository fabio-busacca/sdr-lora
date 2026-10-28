/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
#include "decoder_impl.h"

#include <gnuradio/block_detail.h>     /* the input's done flag, work() */
#include <gnuradio/buffer.h>
#include <gnuradio/buffer_reader.h>
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <string>

#include <cmath>
/* <sstream> for the window-versus-cap message below. */
#include <sstream>
#include <stdexcept>

namespace gr {
namespace sdrlora {

decoder::sptr decoder::make(int sf, double bw, double samp_rate, int window,
                            bool channel_filter_rx, bool soft_decision,
                            bool robust_sync, int sync_phases,
                            bool pseudo_header, bool implicit_header,
                            int payload_len, int cr, bool has_crc,
                            std::vector<int> sync_word, int ldro,
                            bool fec_correct,
                            bool interpolate, bool report_crc_errors,
                            int max_payload_len,
                            bool invert_iq,
                            double trise, double cutoff_ratio,
                            int numtaps, double sync_threshold,
                            int search_tail_symbols, int min_osf,
                            int interp_target_osf,
                            int hop_period,
                            std::vector<double> hop_freqs,
                            bool low_latency, double center_freq)
{
    return gnuradio::make_block_sptr<decoder_impl>(
        sf, bw, samp_rate, window, channel_filter_rx, soft_decision,
        robust_sync, sync_phases, pseudo_header, implicit_header,
        payload_len, cr, has_crc, sync_word, ldro, fec_correct, interpolate,
        report_crc_errors, max_payload_len, invert_iq,
        trise, cutoff_ratio, numtaps, sync_threshold,
        search_tail_symbols, min_osf, interp_target_osf, hop_period,
        hop_freqs, low_latency, center_freq);
}

decoder_impl::decoder_impl(int sf, double bw, double samp_rate, int window,
                           bool channel_filter_rx, bool soft_decision,
                           bool robust_sync, int sync_phases,
                           bool pseudo_header, bool implicit_header,
                           int payload_len, int cr, bool has_crc,
                           std::vector<int> sync_word, int ldro,
                           bool fec_correct,
                           bool interpolate, bool report_crc_errors,
                           int max_payload_len,
                           bool invert_iq,
                           double trise, double cutoff_ratio,
                           int numtaps, double sync_threshold,
                           int search_tail_symbols, int min_osf,
                           int interp_target_osf,
                           int hop_period,
                           std::vector<double> hop_freqs,
                           bool low_latency, double center_freq)
    : gr::sync_block("sdrlora_decoder",
                     gr::io_signature::make(1, 1, sizeof(gr_complex)),
                     gr::io_signature::make(0, 0, 0)),
      d_filter(channel_filter_rx),
      d_soft(soft_decision),
      d_robust(robust_sync),
      d_phases(sync_phases),
      d_fec(fec_correct),
      d_interp(interpolate),
      d_report_bad(report_crc_errors),
      d_packets(0),
      d_crc_errors(0),
      d_overruns(0),
      d_scans(0),
      d_consumed(0),
      d_search_from(0)
{
    if (sf < 5 || sf > 12)
        throw std::invalid_argument("sdrlora decoder: sf must be 5..12");
    /* Checked at construction like every other parameter here: left to
     * find_preamble it would throw out of work() on the first full window,
     * and an exception out of work() kills the flowgraph. */
    if (sync_phases > gr::sdrlora::SYNC_PHASES_MAX)
        throw std::invalid_argument(
            "sdrlora decoder: sync_phases must be <= 16 (0 selects the "
            "per-SF default)");
    if (sf < 7 && !implicit_header)
        throw std::invalid_argument(
            "sdrlora decoder: SF5/SF6 require implicit_header (an explicit "
            "header needs 20 bits and the header block holds 4*(SF-2))");

    /* FINITE AND POSITIVE BEFORE THE CONVERSION. check_rx_params checks
     * these too, but it does not run until the end of this constructor,
     * and converting a non-finite ratio to an integer below is undefined
     * behaviour (bw = 0 gives inf, a NaN operand gives NaN). Checking here
     * also gives the right message rather than "samp_rate must be an
     * integer multiple of bw". encoder_impl validates the rate in the same
     * order. */
    if (!std::isfinite(bw) || !std::isfinite(samp_rate))
        throw std::invalid_argument(
            "sdrlora decoder: bw and samp_rate must be finite");
    if (bw <= 0.0 || samp_rate <= 0.0)
        throw std::invalid_argument(
            "sdrlora decoder: bw and samp_rate must be positive");
    /* AND BOUNDED -- see check_rate_bounds in lora_phy.h. Finite operands
     * with a finite quotient can still overflow later integer conversions
     * (e.g. bw = 1, samp_rate = 2**31). */
    gr::sdrlora::check_rate_bounds(bw, samp_rate);
    const double osf_d = samp_rate / bw;
    /* Defined: the bound holds osf_d <= OSF_MAX + 0.5, which also refuses
     * an overflowed quotient (1e300 / 1e-10). */
    const long osf = (long)std::floor(osf_d + 0.5);
    /* RELATIVE tolerance, the same rule as check_rx_params and
     * decode_buffer. */
    if (osf < 1 || std::fabs(osf_d - (double)osf)
                       > 1e-9 * (osf_d > 1.0 ? osf_d : 1.0))
        throw std::invalid_argument(
            "sdrlora decoder: samp_rate must be an integer multiple of bw");
    /* The sync FFTs are length N and 4N with N = 2^sf * osf, and the transform
     * takes power-of-two lengths, so osf has to be a power of two. 1 is allowed:
     * decode_buffer interpolates it up to INTERP_TARGET_OSF internally, which
     * is where it works best rather than merely at all. It stays a lossy rate
     * -- the chirp's skirts alias in-band at the ADC and no receiver recovers
     * that -- so prefer 2 or more, and band-limit ahead of the ADC if 1 is
     * forced on you. */
    if ((osf & (osf - 1)) != 0 || osf < 1)
        throw std::invalid_argument(
            "sdrlora decoder: samp_rate/bw must be a power of two (the sync "
            "FFTs need power-of-two lengths); 8 is the usual choice");

    /* THE OTHER HALF OF check_rx_params' sync_phases rule: the phase count
     * must divide the symbol length (lora_sync.cc says why). Checked here
     * so that a non-dividing value is refused at construction rather than
     * thrown from inside work(), which leaves through GNU Radio's thread
     * body -- it does not catch std::exception -- and kills the flowgraph.
     * The GRC asserts cover the palette, but sdrlora.decoder(...) from
     * Python and hand-written flowgraphs reach this directly. */
    {
        const long _Nsym = (long)(1L << sf) * osf;
        const int _mph = (sync_phases < 1)
                             ? gr::sdrlora::default_sync_phases(sf)
                             : sync_phases;
        if (_mph > 0 && (_Nsym % (long)_mph) != 0)
            throw std::invalid_argument(
                "sdrlora decoder: sync_phases must divide the symbol length "
                "2**sf * samp_rate/bw -- a grid that does not divide it slips "
                "every cycle and never aligns to a symbol boundary. 0 selects "
                "the per-SF default");
    }

    d_p.SF = sf;
    d_p.bw = bw;
    d_p.fs = samp_rate;
    d_p.f0 = 868e6;    /* only the CFO term uses it, and the RX estimates that */
    d_p.pseudo_header = pseudo_header;
    /* Nothing on the receive path reads this -- implicit mode is carried
     * separately in d_ih -- but it is kept in step with the argument so the
     * struct is a faithful copy of the parameter set. */
    d_p.implicit_header = implicit_header;
    /* Implicit-header frames carry no header, so the coding rate and the CRC
     * flag have to come from here. In explicit mode the header supplies both
     * and these are unused. */
    d_p.cr = cr;
    d_p.has_crc = has_crc;
    d_p.sync_word = sync_word;
    d_p.ldro = ldro;
    d_p.invert_iq = invert_iq;
    d_p.center_freq = center_freq;   /* checked by check_rx_params below */
    /* Set before the check_rx_params call at the end of this constructor,
       which refuses values such as numtaps = 1 (an all-NaN channel
       filter) at construction. */
    d_p.trise = trise;
    d_p.cutoff_ratio = cutoff_ratio;
    d_p.numtaps = numtaps;
    d_p.sync_threshold = sync_threshold;
    d_p.search_tail_symbols = search_tail_symbols;
    d_p.min_osf = min_osf;
    d_p.interp_target_osf = interp_target_osf;
    d_p.hop_period = hop_period;
    d_p.hop_freqs = hop_freqs;
    {
        int s1 = 0, s2 = 0;
        if (!sync_word_symbols(sync_word, s1, s2))
            throw std::invalid_argument(
                "sdrlora decoder: sync_word must be one byte with two "
                "non-zero nibbles (0x12 is the private/default value) or two "
                "raw symbols in 1..4095, e.g. [24, 32]");
        if (!sync_symbols_separable(s1, s2, sf))
            throw std::invalid_argument(
                "sdrlora decoder: those sync symbols are within one bin of the "
                "preamble at this SF -- bins are cyclic mod 2^SF and the sync "
                "check has a one-bin tolerance, so they cannot be told from "
                "more preamble. Use symbols 2..2^SF-2.");
    }
    if (ldro < 0 || ldro > 2)
        throw std::invalid_argument(
            "sdrlora decoder: ldro must be 0 (off), 1 (on) or 2 (auto)");
    d_ih = implicit_header;
    d_ih_len = payload_len;
    /* pseudo_header ? 0 : 1 -- the same rule as decode_buffer. payload_len
     * is the MESSAGE length and the 4-byte prefix is added to it downstream,
     * so 0 is legitimate when that prefix supplies the bytes. */
    if (implicit_header && payload_len < (pseudo_header ? 0 : 1))
        throw std::invalid_argument(
            "sdrlora decoder: implicit_header needs payload_len -- with no "
            "header on air the frame length cannot be discovered");
    /* AND THE CEILING. decode_buffer refuses
     * payload_len + (pseudo_header ? 4 : 0) > 255 because the on-air length
     * field is one byte, but only on the first decoded window, inside the
     * scheduler; refuse it here at construction. */
    /* COMPARED WITHOUT ADDING: payload_len + 4 would overflow an int for
     * payload_len >= 2147483644. */
    if (implicit_header && payload_len > 255 - (pseudo_header ? 4 : 0))
        throw std::invalid_argument(
            "sdrlora decoder: payload_len plus the pseudo-header prefix must "
            "be <= 255 -- the on-air length field is one byte");
    /* decode_buffer enforces these too, but it does so when the first window
     * is decoded -- inside the scheduler, where a GRC flowgraph has already
     * started and the exception surfaces as a runtime crash. Refuse while the
     * block is being built instead, like every other constraint here. */
    if (report_crc_errors && implicit_header)
        throw std::invalid_argument(
            "sdrlora decoder: report_crc_errors needs an explicit header -- "
            "the header FCS is what separates a corrupted packet from a false "
            "lock, and implicit mode has no header to check");
    if (report_crc_errors && pseudo_header)
        throw std::invalid_argument(
            "sdrlora decoder: report_crc_errors needs pseudo_header=false -- "
            "with the 4-byte prefix the frame length is read from inside the "
            "damaged payload, so corrupted packets would be under-counted");
    /* Only in implicit mode: with an explicit header cr is a "don't
       care". */
    if (implicit_header && (cr < 1 || cr > 4))
        throw std::invalid_argument(
            "sdrlora decoder: implicit_header needs cr in 1..4 -- with no "
            "header on air the coding rate cannot be discovered");
    /* The whole receive-side rule, on the assembled struct, at construction
       rather than in the scheduler. On a copy carrying the block's own
       implicit_header, which is a separate constructor argument from
       d_p.implicit_header -- the same copy decode_buffer makes. */
    {
        gr::sdrlora::tx_params q = d_p;
        q.implicit_header = implicit_header;
        gr::sdrlora::check_rx_params(q);
    }

    /* d_chunk: how many NEW samples accumulate before the next scan. This
     * is the only latency the receiver adds on top of a packet's own
     * duration, and it trades directly against CPU, because every scan
     * re-examines the retained tail: the cost goes roughly as
     * (tail + window) / window, so a 4-symbol window costs several times
     * what a 64-symbol one does. An idle channel is the worst case -- the
     * search never locks and so scans everything; while a frame is in
     * flight process() skips the scans that cannot finish it (see
     * scan_would_finish_nothing).
     *
     * THE LATENCY BOUND. decode_buffer() itself needs zero trailing
     * samples. What this block adds is the scan cadence and the front
     * end's context, d_hold, which process() keeps back: numtaps/2 (not
     * FILTER_HALF, the default half-width -- numtaps runs to 8191) with
     * the channel filter running and, below min_osf, at least
     * INTERP_TAPS/2 = 63 for the interpolator. The end of a frame's last
     * symbol must therefore sit at least d_hold inside the buffer, and the
     * packet is reported at the first window boundary at or after
     * end + d_hold: the latency is in [d_hold, window + d_hold). That is
     * [numtaps/2, window + numtaps/2) with the filter on, [0, window) with
     * it off -- or on but inert, at fs = bw without interpolation -- and
     * [63, window + 63) with it off below min_osf. Where in that interval
     * a frame lands is set by its length. The bound holds for any call
     * size because work() is capped with set_max_noutput_items(d_chunk)
     * and consumes only up to the next boundary (see work()).
     *
     * gr-lora_sdr reports with 0 trailing samples, so a small window is
     * the way to its latency, at the CPU cost above. */
    /* NEGATIVE IS NOT "AUTO". Only 0 selects the default; a negative value
     * is refused by name, like every other out-of-range knob here, rather
     * than silently replaced by the default. */
    if (window < 0)
        throw std::invalid_argument(
            "sdrlora decoder: window must be >= 0 (0 selects the per-SF "
            "default of max(32 * 2^sf, 4096) * samp_rate/bw); a negative "
            "value is not "
            "'auto' and would have been silently replaced by that default");
    /* THE DEFAULT RUNS IN REAL TIME: 32 symbols, and never under 4096
     * chips -- 128 symbols at SF5, 64 at SF6, 32 from SF7, so about 33 ms
     * at 125 kHz up to SF7, 131 ms at SF9 and 1.05 s at SF12, roughly one
     * frame's airtime there. A window of a few symbols costs a multiple of
     * real time on an idle channel, because every scan re-examines the
     * 24-symbol retained tail, and a decoder behind real time loses frames
     * silently at the radio, while latency is bounded and visible. A
     * bursty source does not wait for a window -- see set_idle_flush().
     * low_latency halves it -- 16 symbols, never under 2048 chips -- for
     * half the latency at about 1.45x the CPU. */
    d_chunk = (window > 0)
                  ? (size_t)window
                  : low_latency
                        ? (size_t)std::max(16L * (1L << sf), 2048L) * (size_t)osf
                        : (size_t)std::max(32L * (1L << sf), 4096L) * (size_t)osf;
    /* Resolved LDRO flag, not the SF>10 default: with LDRO forced on at a
     * low SF the longest packet is ~40% longer and a flat rule would undersize
     * the tail carried between windows. */
    if (max_payload_len < 1 || max_payload_len > 251)
        throw std::invalid_argument(
            "sdrlora decoder: max_payload_len must be 1..251 -- it bounds the "
            "longest frame the receiver holds, and so the delivery latency "
            "and the buffer size");
    /* AN IMPLICIT FRAME WITHIN WHAT max_payload_len DECLARES. In implicit
     * mode the frame length is known here, and max_payload_len is the
     * longest frame the block is declared for: d_cap is sized for
     * max_payload_len + 4 + 2 bytes on air at CR 4/8 with the CRC
     * (max_packet_symbols) plus the retained tail. A DECLARED ENVELOPE, not
     * the exact edge where frames drop -- how far past it a frame still
     * fits depends on CR, CRC and that tail. The GRC palette refuses by the
     * same rule.
     *
     * payload_len is <= 255 here, refused above, which keeps the addition
     * below from overflowing an int. */
    if (implicit_header &&
        payload_len + (pseudo_header ? 4 : 0) > max_payload_len + 4)
        throw std::invalid_argument(
            "sdrlora decoder: implicit payload_len plus the pseudo-header "
            "prefix is longer than max_payload_len + 4, the longest frame "
            "max_payload_len declares and sizes the receive buffer by; past "
            "it a frame fits only as far as the buffer's slack allows. Raise "
            "max_payload_len");
    /* Samples kept between scans so a preamble straddling the boundary is
     * still found. About 23 symbols is all find_preamble looks at;
     * SEARCH_TAIL_SYMBOLS (24) is that with room.
     *
     * THE ARGUMENT, not the constant -- and never below it.
     * d_p.search_tail_symbols (set with the rest of d_p, above) is what
     * decode_buffer's walk stops short by, and this is what the block
     * RETAINS between scans; sizing the retention from the same knob keeps
     * a raised value from leaving samples between the walk's stop and the
     * buffer's start that are discarded unsearched. max(), because
     * lowering the knob must not shrink the retention below what
     * find_preamble needs. */
    {
        const long tail_sym = (long)(d_p.search_tail_symbols > SEARCH_TAIL_SYMBOLS
                                     ? d_p.search_tail_symbols
                                     : SEARCH_TAIL_SYMBOLS);
        d_tail = (size_t)(tail_sym * (1L << sf) * (long)osf);
    }
    /* A ceiling, not a working size. The buffer holds a packet only while one
     * is in flight; if a false lock is never resolved -- the located packet
     * never completes because there was no packet -- nothing would retire it
     * and the buffer would grow without bound. One maximum-length frame plus
     * the tail is the most that can ever be live. */
    d_overlap = live_overlap_samples(sf, (int)osf,
                                     ldro_enabled(sf, bw, ldro),
                                     max_payload_len);
    /* AND THE CHANNEL FILTER'S FOOTPRINT. process() holds back the last
     * d_hold samples of every scan and keeps d_hold before the consume
     * point as the filter's left context; both are live, so the ceiling
     * carries them as d_foot.
     *
     * THE FRONT END'S CONTEXT, d_hold: numtaps/2 with the channel filter
     * running -- AND, below min_osf, the interpolation filter's.
     * decode_buffer interpolates with an INTERP_TAPS-long same-mode FIR
     * first, so without that hold the last samples of every scan would be
     * interpolated against zero padding. Counted as INTERP_TAPS/2 CALLER
     * samples, conservative by the interpolation factor exactly as the
     * channel filter's term is, and enough for the two filters in
     * cascade: that needs ceil((INTERP_TAPS/2 + numtaps/2) / factor)
     * caller samples, and the factor is at least 2 whenever decode_buffer
     * interpolates. */
    /* The channel filter's term only where the filter RUNS: at fs = bw
     * without interpolation channel_filter is the identity (fs <= bw), and
     * holding numtaps/2 back there -- or refusing a numtaps past the tail,
     * below -- would be stricter than decode_buffer. */
    d_filter_runs = channel_filter_rx &&
                    (osf > 1 || (interpolate && osf < (long)min_osf));
    d_hold = d_filter_runs ? (size_t)(d_p.numtaps / 2) : 0;
    if (interpolate && osf < (long)min_osf &&
        (size_t)(INTERP_TAPS / 2) > d_hold)
        d_hold = (size_t)(INTERP_TAPS / 2);
    d_foot = 2 * d_hold;
    /* AND THE WORKING SET HAS TO BE WITHIN REACH OF A HOST. At legal but
     * extreme rates it is not: SF12 at bw 7812.5 Hz and samp_rate 512 MS/s
     * (fs/bw 65536) would retain a tail of 6.44e9 samples (51.5 GB). Such
     * configurations are refused by name here, as check_frame_size and
     * LEAD_IN_MAX do on the transmit side, rather than dying out of memory
     * at the first scan. The bound is twice MAX_FRAME_SAMPLES: room for any
     * frame the encoder will build plus a retained tail no longer than it.
     * It bounds the buffer, not the resident set -- the decode's working
     * arrays come on top, ~55 bytes per window sample by the note on the
     * window ceiling below -- so it removes what no host holds, not every
     * configuration a given host cannot. */
    const double ws_limit = 2.0 * (double)MAX_FRAME_SAMPLES;
    if ((double)d_overlap + (double)d_tail > ws_limit) {
        std::ostringstream os;
        os << "sdrlora decoder: one maximum-length frame and the retained "
              "tail come to " << (d_overlap + d_tail) << " samples, past "
              "the working-set limit of " << (long)ws_limit
           << " (twice MAX_FRAME_SAMPLES). Lower max_payload_len, "
              "search_tail_symbols or the sample rate";
        throw std::invalid_argument(os.str());
    }
    d_cap = d_overlap + d_tail + d_foot;
    /* AND THE WINDOW HAS TO FIT IN IT. `window` is not in tx_params, so
     * check_rx_params never sees it, and the yaml declares it as free-text
     * int. The peak resident set is about 55 bytes per window sample,
     * because the decode copies d_buf and fir_same allocates two
     * complex<double> arrays over the copy; window = 2^31-1 would be
     * ~120 GB, the kind of OOM-kill that check_frame_size and LEAD_IN_MAX
     * refuse by name.
     *
     * d_cap is the right ceiling on TWO counts, not just memory: d_buf is
     * trimmed to d_cap inside process(), so a window larger than the cap
     * accumulates samples that are shed before they are ever scanned.
     * Practical windows are orders of magnitude under it -- even 64
     * symbols at SF12 / fs-bw 8 is 2097152 samples, against a cap in the
     * tens of millions. */
    /* THE DEFAULT FITS, a choice does not: at SF5 the 128-symbol default
     * outgrows the ceiling of a decoder declared for small frames
     * (max_payload_len near 1), and refusing a setting the user never
     * made would be a defect of its own. So the default is clamped to the
     * ceiling, and an explicit window past it is still refused below. */
    if (window == 0 && d_chunk > d_cap) d_chunk = d_cap;
    if (d_chunk > d_cap) {
        std::ostringstream os;
        os << "sdrlora decoder: window " << d_chunk
           << " exceeds the buffer ceiling of " << d_cap
           << " samples (one maximum-length frame, the retained tail and "
              "the front end's hold-back and left context). "
              "process() trims to that ceiling, so a larger window "
              "accumulates samples it then sheds before scanning them -- and "
              "the peak resident set is about 55 bytes per window sample. "
              "Lower window, or raise max_payload_len";
        throw std::invalid_argument(os.str());
    }
    /* THE FILTER'S HALF-WIDTH MUST FIT IN THE FLOOR TAIL. A CONSERVATIVE
     * ENVELOPE, and the message says so.
     *
     * Delivery with a wide filter is exactly-once because of three
     * mechanisms:
     *
     *   - the walk resumes where the last one stopped (d_search_from;
     *     decode_buffer's search_from) rather than re-searching the
     *     samples the rewind keeps for the filter's left context;
     *   - the ceiling has room for the filter's footprint -- d_foot;
     *   - fir_same returns exact zeros in digital silence, so FFT roundoff
     *     there cannot pass the detector's relative energy gate.
     *
     * The bound stays because the hold-back is delivery LATENCY, and past
     * this point it exceeds the 24-symbol tail the block retains. Relaxing
     * it is a decision, not a repair.
     *
     * ANCHORED TO THE FLOOR TAIL, SEARCH_TAIL_SYMBOLS, not to d_tail, which
     * grows with the knob. The knob moves nothing here.
     *
     * ONLY WHEN THE FILTER RUNS. Otherwise process() holds back at most
     * the interpolator's context, which does not depend on numtaps, so
     * numtaps is inert, and refusing then would be stricter than
     * decode_buffer.
     *
     * COUNTED IN THE CALLER'S SAMPLES, so conservative below min_osf: the
     * filter then runs at the interpolated rate (4x at the defaults), where
     * numtaps/2 is a quarter as many caller samples as the hold-back, d_foot
     * and this bound count. At SF7/osf1 it refuses numtaps 6147..8191,
     * whose real footprint fits the tail. Left conservative. */
    const size_t hold_limit = (size_t)SEARCH_TAIL_SYMBOLS
                              * (size_t)(1L << sf) * (size_t)osf;
    if (d_filter_runs && (size_t)(d_p.numtaps / 2) > hold_limit)
        throw std::invalid_argument(
            "sdrlora decoder: numtaps/2 exceeds SEARCH_TAIL_SYMBOLS * 2^SF * "
            "osf, the floor search tail. The channel filter's half-width is "
            "held back from every scan -- it is delivery latency -- and must "
            "fit inside the tail the streaming receivers retain. Use fewer "
            "taps or a higher oversampling ratio; search_tail_symbols does "
            "not move this bound");
    /* AND THE INTERPOLATED OSF, which decode_buffer also checks. Same
     * shape as the sync_phases and payload_len rules above: checked at
     * construction because a throw out of work() leaves through GNU
     * Radio's thread body, which does not catch std::exception. */
    gr::sdrlora::check_effective_osf((int)osf, min_osf, interp_target_osf,
                                     interpolate);
    /* AND THE WIDENED HOP CUTOFF. With a hop plan in force and the channel
     * filter on, decode_buffer raises the cutoff to 1 + 2*max|f|/bw, and
     * channel_filter refuses anything above 64, so the real limit is
     * max|f| <= 31.5*bw. At high rates that is tighter than the hop-band
     * rule's (samp_rate-bw)/2, so it is checked here as well rather than
     * thrown from the first process(). */
    if (channel_filter_rx)
        gr::sdrlora::check_effective_cutoff(d_p.cutoff_ratio, d_p.hop_period,
                                            d_p.hop_freqs, bw);
    d_since = 0;
    d_sym = ((size_t)1 << d_p.SF) * (size_t)(d_p.fs / d_p.bw + 0.5);

    /* No whole-frame overlap is buffered ahead of a scan. find_preamble()
     * looks at about 23 symbols to recognise a preamble, and a located
     * packet's own length says where it ends, so the only live state is the
     * last SEARCH_TAIL_SYMBOLS or, while a packet is arriving, that packet.
     * decode_buffer() reports how much is finished with and process()
     * retires exactly that. d_overlap is only the ceiling a false lock can
     * never exceed. */

    /* THE CADENCE IS THE BLOCK'S, not the scheduler's. This block is a
     * sink, so noutput_items is whatever is available -- up to
     * max_noutput_items, 16384 by default -- and the gate in work() is
     * `d_since >= d_chunk`. work() aligns its consumption to the window
     * boundaries, which is what holds the latency interval for any call
     * size; this cap is a second guard, keeping every call to at most one
     * window. */
    set_max_noutput_items((int)d_chunk);

    message_port_register_out(pmt::mp("pdus"));
    /* A HINT, NOT THE CEILING -- and one that does not follow the knob.
     *
     * d_cap is one maximum-length frame plus the RETENTION, and the
     * retention is search_tail_symbols * 2^SF * osf, so at SF12, osf 8 and
     * the knob at its validated maximum of 4096 reserving d_cap would ask
     * the allocator for 148570112 gr_complex -- 1.19 GB -- before a sample
     * has arrived. The buffer only reaches d_cap under a false lock that
     * never resolves, which is the case d_cap exists to bound.
     *
     * The DEFAULT retention rather than a fixed byte figure, because the
     * working set is genuinely rate-dependent: one maximum-length SF12
     * frame plus 24 symbols is 15138816 samples at osf 8 and 30277632 at
     * osf 16, so any absolute figure would under-reserve exactly the
     * configurations that need the room. At any knob value of 24 or below
     * this is d_cap unchanged.
     *
     * The vector still grows past this hint on demand. */
    {
        const size_t hint = d_overlap + (size_t)SEARCH_TAIL_SYMBOLS
                            * (size_t)(1L << sf) * (size_t)osf + d_foot;
        /* The clamp cannot bind today -- d_cap uses
         * max(knob, SEARCH_TAIL_SYMBOLS) and this uses
         * SEARCH_TAIL_SYMBOLS, so hint <= d_cap for every knob value --
         * and it is kept so that a future change to either term cannot
         * make the hint exceed the ceiling it is meant to sit under. */
        /* AND A FIXED CEILING ON THE HINT. At legal high rates the default
         * working set is itself enormous: SF12 at fs/bw 256 gives a
         * 484442366-sample hint (d_foot included), 3.9 GB, and reserving
         * that can raise an unnamed std::bad_alloc out of this
         * constructor. 2^24 samples (128 MB) covers the SF12 / fs-bw 8
         * working set of 15138816 and caps everything larger; d_buf grows
         * past the hint on demand. A false lock that never resolves can
         * still grow the buffer to d_cap at such rates. */
        const size_t kReserveMax = (size_t)1 << 24;
        size_t r = hint < d_cap ? hint : d_cap;
        if (r > kReserveMax) r = kReserveMax;
        d_buf.reserve(r);
    }
}

decoder_impl::~decoder_impl()
{
    {
        std::lock_guard<std::mutex> lk(d_mx);
        d_watch_stop = true;
    }
    d_cv.notify_all();
    if (d_watch.joinable()) d_watch.join();
}

/* ATOMIC, NOT UNDER d_mx: the mutex is held for a whole scan, up to
 * 0.7 s at SF12 / fs-bw 8, and these are called from Python -- a GUI
 * thread polling idle_flush() would block for that long, holding the GIL
 * while it waited. The binding releases the GIL as well. */
void decoder_impl::set_idle_flush(double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0 || seconds > 3600.0)
        throw std::invalid_argument(
            "sdrlora decoder: set_idle_flush takes 0 (off) to 3600 seconds");
    d_idle = seconds;
}

double decoder_impl::idle_flush() const { return d_idle; }

/* THE IDLE FLUSH. A sync block runs only when samples arrive, and a
 * bursty source -- a LoRa Encoder in a loopback -- sends none after a
 * frame: the decoder would then hold the frame's last d_hold samples back
 * and wait for a window boundary the next PDU would bring. The tx_eob
 * flush covers a direct connection only; a Channel Model or an Add block
 * drops the tag, a filter or a resampler delays the frame past it, and a
 * raised window outruns any closing silence the encoder adds. This
 * watchdog flushes what the block holds once its input has been quiet for
 * d_idle seconds of wall time, measured from the END of the last work()
 * call -- a long scan inside work() is not quiet time -- and once per
 * quiet spell (d_dirty). A live radio never goes quiet, so there it never
 * fires. */
void decoder_impl::watch()
{
    std::unique_lock<std::mutex> lk(d_mx);
    while (!d_watch_stop) {
        d_cv.wait_for(lk, std::chrono::milliseconds(50));
        if (d_watch_stop) break;
        if (!d_dirty || d_idle <= 0.0) continue;
        const std::chrono::duration<double> quiet =
            std::chrono::steady_clock::now() - d_last_work;
        if (quiet.count() < d_idle) continue;
        process(true);          /* flush: nothing is arriving */
        d_dirty = false;
        d_since = 0;            /* the cadence restarts with the next */
    }
}

bool decoder_impl::start()
{
    {
        std::lock_guard<std::mutex> lk(d_mx);
        d_watch_stop = false;
        d_dirty = false;
        d_last_work = std::chrono::steady_clock::now();
    }
    if (!d_watch.joinable())
        d_watch = std::thread(&decoder_impl::watch, this);
    return true;
}

uint64_t decoder_impl::packets() const { return d_packets; }
uint64_t decoder_impl::crc_errors() const { return d_crc_errors; }
uint64_t decoder_impl::overruns() const { return d_overruns; }
uint64_t decoder_impl::scans() const { return d_scans; }
std::size_t decoder_impl::retention_samples() const { return d_tail; }

void decoder_impl::process(bool flush, bool final_walk)
{
    /* The channel filter is numtaps long applied `same`-mode, so the last
     * half-width of any buffer is filtered against zero padding. Hold those
     * back from the WALK -- they are still filtered, as the right-hand
     * context of the samples before them -- and walk them on a later scan,
     * once their own context has arrived. */
    /* `flush` says no more samples are coming -- end of a file source or a
     * finite capture, not a live radio. The last half-filter-width is held
     * back normally because those samples are filtered against zero padding
     * and will be filtered properly once more arrive; at the end of a stream
     * nothing more arrives, so a packet ending inside that stretch would just
     * be lost. */
    /* THE CONFIGURED numtaps, not the default constant. FILTER_HALF is
     * 127 = 255/2, and numtaps is a parameter validated to 3..8191, so the
     * filter's half-width runs to 4095; holding back less would retire
     * samples filtered against zero padding, which lora_rx.h's invariant
     * rules out. Identical at the default, where numtaps/2 IS FILTER_HALF.
     * Below min_osf d_hold covers the interpolator too -- see the
     * constructor. */
    const size_t hold = flush ? 0 : d_hold;
    if (d_buf.size() <= hold) return;
    const size_t usable = d_buf.size() - hold;
    /* THE WHOLE BUFFER, walked to `usable`: the held-back samples are the
     * right-hand context of the ones before them, so they are filtered
     * with the rest rather than cut off against zero padding -- see
     * walk_end in decode_buffer. */
    d_consumed = 0;
    d_complete_at = -1;
    d_scans++;
    const std::vector<rx_result> got =
        decode_buffer(d_buf, d_p, d_filter, d_robust, d_phases, d_soft,
                      d_ih, d_ih_len, d_fec, d_interp, d_report_bad,
                      &d_consumed,
                      d_search_from < (long)usable ? d_search_from
                                                   : (long)usable,
                      (long)usable, &d_complete_at, final_walk);

    for (size_t i = 0; i < got.size(); i++) {
        const rx_result& R = got[i];
        /* An empty payload is a real frame when the pseudo-header supplies
         * the on-air bytes -- see the implicit-length rule in the
         * constructor -- and is counted. Only a failed header or CRC
         * suppresses a result. */
        if (!R.hdr_ok) continue;
        /* decode_buffer only returns a failed-CRC packet when it was asked
         * to, and only one per corrupted frame -- see the walk there. */
        if (!R.crc_ok && !d_report_bad) continue;

        pmt::pmt_t meta = pmt::make_dict();
        meta = pmt::dict_add(meta, pmt::mp("src"), pmt::from_long(R.src));
        meta = pmt::dict_add(meta, pmt::mp("dst"), pmt::from_long(R.dst));
        meta = pmt::dict_add(meta, pmt::mp("seqn"), pmt::from_long(R.seqn));
        meta = pmt::dict_add(meta, pmt::mp("cr"), pmt::from_long(R.cr));
        meta = pmt::dict_add(meta, pmt::mp("sf"), pmt::from_long(d_p.SF));
        meta = pmt::dict_add(meta, pmt::mp("crc_ok"),
                             pmt::from_bool(R.crc_ok));

        pmt::pmt_t vec = pmt::init_u8vector(R.payload.size(), R.payload.data());
        message_port_pub(pmt::mp("pdus"), pmt::cons(meta, vec));
        /* packets() counts only what passed; crc_errors() is the other
         * half of a packet error rate. */
        if (R.crc_ok) d_packets++;
        else d_crc_errors++;
    }

    /* Retire what decode_buffer says is finished with, backing off by half
     * the channel filter so the samples kept are re-filtered with context on
     * both sides next time rather than against zero padding. */
    /* REWIND FOR THE FILTER, NOT FOR THE WALK. Keeping from consumed - hold
     * gives the next scan's filter `hold` samples of left context; the next
     * walk starts at the old `consumed`, because the rewound samples were
     * already searched and re-searching them would deliver a frame inside
     * the rewind again. */
    long keep_from = d_consumed - (long)hold;
    if (keep_from > 0) {
        if (keep_from > (long)d_buf.size()) keep_from = (long)d_buf.size();
        d_buf.erase(d_buf.begin(), d_buf.begin() + keep_from);
    } else {
        keep_from = 0;
    }
    d_search_from = d_consumed - keep_from;
    if (d_search_from < 0) d_search_from = 0;
    if (d_complete_at >= 0) d_complete_at -= keep_from;

    /* The ceiling. Reached only when a located packet never completes, which
     * means it was a false lock: drop the oldest samples so the search moves
     * on rather than holding a buffer that will never resolve. */
    if (d_buf.size() > d_cap) {
        const long trim = (long)(d_buf.size() - d_cap);
        d_buf.erase(d_buf.begin(), d_buf.end() - (long)d_cap);
        d_overruns++;               /* reported by overruns() */
        d_search_from = (d_search_from > trim) ? d_search_from - trim : 0;
        d_complete_at = -1;         /* the frame it named was shed */
    }
}

/* A SCAN THAT CANNOT FINISH ANYTHING, skipped. While a located frame whose
 * length is known is still arriving, a scan would re-filter the whole
 * growing buffer, re-run the preamble search and re-demodulate every symbol
 * received so far, only to find the frame truncated again -- so the cost
 * of a frame would grow with the square of its length. Skipped until the
 * buffer holds the frame's end, one symbol early, plus the hold-back -- a
 * sooner scan only finds it truncated, so the delivery latency is
 * unchanged -- or until the buffer has passed its ceiling, so a false lock
 * still sheds. */
bool decoder_impl::scan_would_finish_nothing() const
{
    if (d_complete_at < 0) return false;
    if (d_buf.size() > d_cap) return false;
    const long need = d_complete_at - (long)d_sym + (long)d_hold;
    return (long)d_buf.size() < need;
}

/* End of stream. A live SDR never stops producing samples, so the partial
 * window left in d_buf would eventually be flushed by more arriving data --
 * but a file source, a finite capture, or a bursty transmitter does stop, and
 * then the tail is simply lost. Decode it here instead.
 *
 * Guarded on having at least one preamble plus header worth of samples, so
 * stopping an idle flowgraph does not run a full sync search over a handful
 * of samples for nothing. */
bool decoder_impl::stop()
{
    {
        std::lock_guard<std::mutex> lk(d_mx);
        d_watch_stop = true;
    }
    d_cv.notify_all();
    if (d_watch.joinable()) d_watch.join();
    std::lock_guard<std::mutex> lk(d_mx);
    d_dirty = false;
    const size_t sym = ((size_t)1 << d_p.SF) * (size_t)(d_p.fs / d_p.bw + 0.5);
    /* A live SDR never stops, but a file source or a bursty transmitter does,
     * and then whatever is still buffered would be lost. Scan it. */
    if (d_buf.size() >= 13 * sym) {
        /* The last resort, and a lossy one: GNU Radio calls stop() after
         * telling the message subscribers this block is done, so a packet
         * found HERE is counted and reaches no sink. work() flushes at the
         * end of the stream to keep this empty; say so when it is not. */
        const uint64_t before = d_packets + d_crc_errors;
        process(true, true);    /* flush: nothing follows */
        if (d_packets + d_crc_errors > before)
            GR_LOG_WARN(d_logger,
                        "decoder: " + std::to_string(d_packets + d_crc_errors
                                                     - before)
                            + " packet(s) decoded in stop(), after the "
                              "flowgraph's message sinks were told this "
                              "block was done -- counted in packets(), but "
                              "not delivered");
    }
    d_buf.clear();
    d_search_from = 0;          /* an index into what was just cleared */
    /* AND SO IS d_complete_at: the flush above sets it when a frame is in
     * flight, and a restarted decoder (tb.start() again, or lock()/unlock())
     * would compare its new buffer against that stale index and skip every
     * scan until the buffer grew past it. d_since restarts the cadence with
     * the buffer. */
    d_complete_at = -1;
    d_since = 0;
    return true;
}

int decoder_impl::work(int noutput_items,
                       gr_vector_const_void_star& input_items,
                       gr_vector_void_star& output_items)
{
    (void)output_items;
    const gr_complex* in = (const gr_complex*)input_items[0];
    std::lock_guard<std::mutex> lk(d_mx);
    /* the watchdog measures quiet from the END of this call */
    struct stamp {
        std::chrono::steady_clock::time_point& t;
        ~stamp() { t = std::chrono::steady_clock::now(); }
    } stamp_end{ d_last_work };

    /* ALIGNED TO THE WINDOW, not merely capped by it. set_max_noutput_items
     * (d_chunk) bounds each call at one window, but a bound does not align:
     * with calls of 1000 samples against a 1024-sample window the scan gate
     * would fire on almost every call and then, after a scan that landed on
     * a boundary, not for 2000 samples, nearly doubling the latency
     * ceiling. Consuming only up to the next boundary puts every scan ON
     * one, whatever the scheduler offers; the items not consumed are
     * offered again on the next call, which is the scheduler's contract
     * with a sink. This is what holds the interval for any call size; the
     * cap is a second guard. */
    const size_t room = d_chunk - d_since;        /* >= 1: d_since < d_chunk */
    int take = ((size_t)noutput_items < room) ? noutput_items : (int)room;
    /* THE END OF A BURST. An encoder block marks each frame's last sample
     * with "tx_eob" (as a USRP sink expects), and after it nothing follows
     * until the next burst: in a loopback flowgraph the decoder would wait
     * for samples that only the NEXT PDU brings, so every packet would come
     * out one PDU late and the last one never. Consume up to the tag and
     * flush, as stop() does. A live receive stream carries no such tag, so
     * on a radio nothing changes. */
    bool eob = false;
    /* Only inside a flowgraph: work() called directly on a bare block has
     * no buffer reader for the tag lookup to read. */
    if (detail()) {
        const uint64_t n0 = nitems_read(0);
        std::vector<gr::tag_t> tags;
        get_tags_in_range(tags, 0, n0, n0 + (uint64_t)take,
                          pmt::mp("tx_eob"));
        if (!tags.empty()) {
            uint64_t first = tags[0].offset;
            for (size_t i = 1; i < tags.size(); i++)
                if (tags[i].offset < first) first = tags[i].offset;
            take = (int)(first - n0) + 1;
            eob = true;
        }
    }
    /* THE END OF THE STREAM, seen while the flowgraph still runs. A finite
     * source -- a file, a vector -- ends, and GNU Radio then calls stop()
     * only after the executor has told the message subscribers this block
     * is done, so a packet flushed there reaches no sink. So the call that
     * takes the input's last sample once it reports done flushes HERE, and
     * its PDUs reach a sink that is still running; stop() keeps a flush as
     * the last resort, and warns when it finds anything.
     *
     * AND THE CALL THAT DRAINS THE INPUT BEFORE THE FLAG waits for it.
     * Upstream sets done on its NEXT iteration after its last output, so
     * with the source fanned out -- to a GUI or file sink, or a second
     * decoder for another SF -- this block can take the last chunk first;
     * the executor then sees done with nothing left, never calls work()
     * again, and the tail goes to stop(), whose PDU reaches no sink. So a
     * call that leaves the input empty polls, sleeping, for up to 50 ms:
     * more data ends the wait, done ends it with the flush here, and a
     * full buffer that this block's own hold keeps full ends it at once
     * (see below). NOT by holding a sample back until done: a sink that
     * returns 0 with input left is called again at once, so the block
     * would spin on an idle input, and the busy calls would keep the idle
     * flush from ever firing. */
    /* NOT WHILE THIS BLOCK IS WHAT HOLDS THE WRITER. Until work() returns,
     * the `take` items stay unconsumed: if the buffer is full and this
     * reader is the furthest behind, upstream cannot write -- nor reach
     * the iteration that sets done -- so the wait could only time out, and
     * at fs/bw 8 a scan can outlast the 8192-item buffer. And on a CLOCK,
     * not a count: sleep granularity makes a count of short sleeps last
     * far longer than its nominal total. */
    bool eos = false;
    if (detail()) {
        const gr::buffer_reader_sptr rd = detail()->input(0);
        eos = rd->done() && take >= rd->items_available();
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
        while (!eos && !eob && take >= rd->items_available()
               && std::chrono::steady_clock::now() < deadline) {
            const gr::buffer_sptr b = rd->buffer();
            int need = 1;
            if (gr::block_sptr wb = b->link())
                need = std::max(std::max(1, wb->output_multiple()),
                                wb->min_noutput_items());
            /* UNDER THE BUFFER'S MUTEX: in GR 3.10 space_available()
             * prunes the buffer's tag map, which the mutex guards -- the
             * executor takes it around the same call. Unlocked, the
             * writer's add_item_tag and the other readers' tag lookups
             * race this erase on a tagged stream. Same order as
             * get_tags_in_range above: d_mx, then the buffer's. */
            bool held = false;
            {
                gr::thread::scoped_lock guard(*b->mutex());
                if (b->space_available() < need) {
                    int others = 0;
                    for (size_t i = 0; i < b->nreaders(); i++)
                        if (b->reader(i) != rd.get())
                            others = std::max(
                                others, b->reader(i)->items_available());
                    held = rd->items_available() >= others;
                }
            }
            if (held)
                break;              /* this block's hold blocks the writer */
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            eos = rd->done() && take >= rd->items_available();
        }
    }
    d_buf.insert(d_buf.end(), in, in + take);
    d_since += (size_t)take;
    if (take > 0) d_dirty = true;
    if (eob || eos) {
        process(true, true);        /* flush: the burst or stream is over */
        d_since = 0;                /* the cadence restarts with the next */
        d_dirty = false;
        return take;
    }
    if (d_since >= d_chunk) {
        /* SUBTRACT, do not zero. Zeroing would discard whatever the call
         * overshot by, so a run of work() calls at 0.6*d_chunk would scan
         * every 1.2 windows instead of every one. With the alignment above
         * d_since reaches exactly d_chunk and this is 0 either way;
         * subtracting stays right if that ever changes. */
        d_since -= d_chunk;
        if (!scan_would_finish_nothing())
            process();
    }

    return take;
}

} // namespace sdrlora
} // namespace gr
