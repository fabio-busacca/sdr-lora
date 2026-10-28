/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
/*
 * Preamble detection, fine sync, CFO estimation, the receive channel filter,
 * and the multi-packet buffer walk.
 *
 * This is the half of the receiver that CANNOT be held to byte identity. It
 * performs a search: scanning offsets, thresholding an accumulated statistic,
 * interpolating a peak. Two computations can agree to fifteen digits and
 * still land on opposite sides of one threshold, and then diverge completely --
 * one finds the packet, the other does not. So its correctness here is
 * statistical (packet delivery against SNR), not byte-level.
 *
 * FFT sizes here are N and 4N, unlike the symbol demodulator which works on the
 * decimated length K. N = 2^SF * OSF, so the power-of-two transforms here
 * (fft_fast, radix-4/2) require OSF to be a power of two; 8 is the usual
 * choice. Anything else needs a mixed-radix FFT; find_preamble() reports
 * failure rather than silently producing nonsense.
 */
#include "sdrlora/lora_rx.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <cstring>
#include <stdexcept>

namespace gr {
namespace sdrlora {

static const double kPi = 3.14159265358979311600;

/* ------------------------------------------------------- channel filter */
/* Windowed-sinc lowpass, unit DC gain, linear phase. */
static void lowpass_taps(double cutoff_hz, double fs, int numtaps,
                         std::vector<double>& h)
{
    if (numtaps % 2 == 0) numtaps++;
    h.assign((size_t)numtaps, 0.0);
    const double fc = cutoff_hz / fs;
    double sum = 0.0;
    for (int i = 0; i < numtaps; i++) {
        const double n = (double)i - (double)(numtaps - 1) / 2.0;
        const double x = 2.0 * fc * n;
        const double sinc = (std::fabs(x) < 1e-12) ? 1.0 : std::sin(kPi * x) / (kPi * x);
        /* Blackman window, matching np.blackman */
        const double w = 0.42 - 0.5 * std::cos(2.0 * kPi * (double)i / (double)(numtaps - 1))
                              + 0.08 * std::cos(4.0 * kPi * (double)i / (double)(numtaps - 1));
        h[(size_t)i] = 2.0 * fc * sinc * w;
        sum += h[(size_t)i];
    }
    for (int i = 0; i < numtaps; i++) h[(size_t)i] /= sum;
}

static size_t next_pow2(size_t v)
{
    size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

/* Convolve with a symmetric FIR, trimmed like numpy's mode='same'. Shared by
 * channel_filter() and upsample(). */
static void fir_same(const std::vector<std::complex<float> >& in,
                     const std::vector<double>& h,
                     std::vector<std::complex<float> >& out)
{
    const size_t M = h.size();
    const size_t n = in.size();
    if (n == 0) { out.clear(); return; }
    const size_t lead = (M - 1) / 2;          /* symmetric h: centre it */
    const size_t n_causal = n + lead;
    const size_t nfft = next_pow2(std::max<size_t>(1u << 14, 4 * M));
    const size_t step = nfft - M + 1;

    std::vector<std::complex<double> > H(nfft, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < M; i++) H[i] = std::complex<double>(h[i], 0.0);
    fft_fast(H, -1);

    /* Overlap-save, so a megasample buffer costs O(n log n) with bounded memory. */
    std::vector<std::complex<double> > xp(M - 1 + n + nfft,
                                          std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < n; i++) xp[M - 1 + i] = std::complex<double>(in[i]);

    std::vector<std::complex<double> > causal(n_causal, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double> > seg(nfft);
    size_t pos = 0;
    while (pos < n_causal) {
        for (size_t i = 0; i < nfft; i++)
            seg[i] = (pos + i < xp.size()) ? xp[pos + i] : std::complex<double>(0.0, 0.0);
        fft_fast(seg, -1);
        for (size_t i = 0; i < nfft; i++) seg[i] *= H[i];
        fft_fast(seg, +1);
        const double inv = 1.0 / (double)nfft;
        const size_t take = std::min(step, n_causal - pos);
        for (size_t i = 0; i < take; i++)
            causal[pos + i] = seg[M - 1 + i] * inv;
        pos += step;
    }

    out.assign(n, std::complex<float>(0.0f, 0.0f));
    for (size_t i = 0; i < n; i++)
        out[i] = std::complex<float>((float)causal[lead + i].real(),
                                     (float)causal[lead + i].imag());

    /* EXACT ZEROS WHERE THE ARITHMETIC SAYS ZERO. out[j] reads
     * in[j-left .. j+right], and the buffer is zero-extended at both ends
     * like numpy's 'same', so where every one of those inputs is zero the
     * true output is 0 -- and overlap-save leaves FFT roundoff there
     * instead (~1e-16 of the filtered peak). find_preamble's energy gate is
     * relative to the slice's own mean power, so once a frame has been
     * consumed that roundoff could pass it, and the plain detector has no
     * gate at all: in implicit mode with standard framing and CRC off,
     * where any lock is a packet, a garbage frame would come out after a
     * real one. An all-zero window fails the gate and the sync-word test
     * outright. Not a threshold: a sample with any non-zero input in its
     * support is untouched. */
    {
        const long nn = (long)n;
        const long right = (long)lead;
        const long left = (long)M - 1 - right;
        long a = -1;                    /* start of the current zero run */
        for (long i = 0; i <= nn; i++) {
            const bool zero = i < nn && in[(size_t)i].real() == 0.0f &&
                              in[(size_t)i].imag() == 0.0f;
            if (zero) {
                if (a < 0) a = i;
                continue;
            }
            if (a >= 0) {
                /* a run touching an end continues into the zero padding */
                const long lo = (a == 0) ? 0 : a + left;
                const long hi = (i == nn) ? nn : i - right;
                for (long j = lo; j < hi; j++)
                    out[(size_t)j] = std::complex<float>(0.0f, 0.0f);
                a = -1;
            }
        }
    }
}

void channel_filter(const std::vector<std::complex<float> >& in, double bw,
                    double fs, std::vector<std::complex<float> >& out,
                    double cutoff_ratio, int numtaps)
{
    /* VALIDATED, because this is a public entry point (phy_channel_filter)
     * and lowpass_taps has no defence of its own: it divides by
     * numtaps - 1, so numtaps = 1 or 0 would return a full-length buffer of
     * NaNs, SILENTLY. Bounds and wording follow check_rx_params. */
    /* THE RATE BOUNDS, not only finite and positive: bw = 1e-320 is
     * both, and would give a buffer of NaNs without a word. */
    check_rate_bounds(bw, fs);
    if (numtaps < 3 || numtaps > 8191 || (numtaps % 2) == 0)
        throw std::invalid_argument(
            "sdrlora: numtaps must be an ODD number in 3..8191 -- the "
            "windowed sinc divides by numtaps - 1, so 1 and 0 produce a "
            "filter of NaNs, and an even length has no whole-sample group "
            "delay");
    if (!std::isfinite(cutoff_ratio) || cutoff_ratio <= 0.0
            || cutoff_ratio > 64.0)
        throw std::invalid_argument(
            "sdrlora: cutoff_ratio must be finite, > 0 and <= 64");
    if (fs <= bw) { out = in; return; }
    std::vector<double> h;
    lowpass_taps(bw / 2.0 * cutoff_ratio, fs, numtaps, h);
    fir_same(in, h, out);
}

void upsample(const std::vector<std::complex<float> >& in, double fs,
              int factor, std::vector<std::complex<float> >& out, int numtaps)
{
    if (factor <= 1 || in.empty()) { out = in; return; }
    std::vector<std::complex<float> > up(in.size() * (size_t)factor,
                                         std::complex<float>(0.0f, 0.0f));
    for (size_t i = 0; i < in.size(); i++) up[i * (size_t)factor] = in[i];
    std::vector<double> h;
    lowpass_taps(fs / 2.0, fs * (double)factor, numtaps, h);
    /* Zero-stuffing divides the passband gain by `factor`; put it back. */
    for (size_t i = 0; i < h.size(); i++) h[i] *= (double)factor;
    fir_same(up, h, out);
}

/* ------------------------------------------------------- preamble search */
static bool is_pow2(long v) { return v > 0 && (v & (v - 1)) == 0; }

int default_sync_phases(int SF) { return SF < 7 ? 4 : 2; }

long effective_osf(int osf, int min_osf, int interp_target_osf,
                   bool interpolate)
{
    if (osf < 1 || !interpolate || osf >= min_osf) return 0;
    /* CEIL, matching decode_buffer below -- truncation could land the
     * receiver under its own min_osf. */
    const int factor = (interp_target_osf + osf - 1) / osf;
    return (long)osf * (long)factor;
}

void check_effective_osf(int osf, int min_osf, int interp_target_osf,
                         bool interpolate)
{
    const long eff = effective_osf(osf, min_osf, interp_target_osf,
                                   interpolate);
    if (eff == 0) return;            /* no interpolation: nothing to check */
    if (eff < 1 || (eff & (eff - 1)) != 0) {
        const int factor = (interp_target_osf + osf - 1) / osf;
        std::ostringstream os;
        os << "sdrlora: interpolating fs/bw " << osf << " up by "
           << factor << " gives " << eff
           << ", which is not a power of two -- the sync FFTs are "
              "power-of-two transforms and the receiver would report an empty band. "
              "Choose an interp_target_osf that is a power-of-two "
              "multiple of fs/bw.";
        throw std::invalid_argument(os.str());
    }
}

/* Sync-word offsets are cyclic mod K, the SYMBOL alphabet -- not mod N, the
 * transform length.
 *
 * A dechirped LoRa symbol is two tone segments, because the chirp wraps at the
 * symbol value: one beats to k*BW/K and the other to k*BW/K - BW. In a
 * K-length (decimated) spectrum the two fold into the same bin. In the
 * N = K*osf transform used here they do not -- they land at k and k-K, 2^SF
 * bins apart -- and the argmax takes whichever segment is stronger. So a
 * measured offset of k and one of k-K are the same sync symbol, and only a
 * fold mod K treats them that way.
 *
 * Folding mod N instead would let the default 0x12 through (symbols 8 and
 * 16, far from the wrap) while rejecting sync words with a symbol near K --
 * at SF7, [3, 125] and [100, 7], which are legal sync words. */
static double bin_off(double measured, double reference, double want, long K)
{
    const double d = measured - reference - want;
    const double k = (double)K;
    return std::fmod(std::fmod(d + k / 2.0, k) + k, k) - k / 2.0;
}


long max_packet_symbols(int SF, int de, int max_payload)
{
    if (SF < 5 || SF > 12)
        throw std::invalid_argument("sdrlora: sf must be 5..12");
    if (de < 0) de = (SF > 10) ? 1 : 0;
    /* de is the low-data-rate flag: 0 or 1, nothing else. Taken as given,
     * de = 3 would make ppm = SF - 6 = 0 at SF6 and the block-count division
     * below divide by zero (SIGFPE); de = 4 would make it negative and the
     * sample count cast to about 1.8e19, a number a caller would size a
     * buffer from. */
    if (de > 1)
        throw std::invalid_argument(
            "sdrlora: de is the low-data-rate flag and must be 0, 1, or -1 "
            "for the automatic per-SF rule");
    /* Refused, not clamped, as decoder_impl refuses it. A caller asking to
     * reserve for 1000 bytes has misunderstood something, and getting the
     * 251-byte answer back would hide it. */
    if (max_payload < 1 || max_payload > 251)
        throw std::invalid_argument(
            "sdrlora: max_payload must be 1..251 (the on-air length field is "
            "one byte, and the 4-byte pseudo-header prefix takes it to 255)");
    const int ppm = SF - 2 * de;
    /* The longest frame the receiver must be able to hold. It sets the overlap
     * carried between windows, and so the delivery latency and the buffer
     * size: at SF11 a 251-byte bound is half a second and megabytes before
     * anything is reported, while a deployment that only ever sends 32-byte
     * frames needs about a fifth of that. Bounding it is the only lever a
     * buffer-scanning receiver has -- gr-lora_sdr's frame_sync is a streaming
     * state machine and reports a frame the moment it ends. */
    const int worst_bits = 8 * (max_payload + 4 + 2);
    const int blocks = (worst_bits + (4 * ppm) - 1) / (4 * ppm);
    return (long)blocks * 8L + 9L + 13L;   /* + header + preamble */
}

size_t live_overlap_samples(int SF, int osf, int de, int max_payload)
{
    /* osf TOO. At osf 0 the reservation would be ZERO, and a caller
     * sizing a ring from it would lose every frame with only a counter
     * moving; at osf -1 it would wrap to about 1.8e19, and bad_alloc. */
    if (osf < 1)
        throw std::invalid_argument(
            "sdrlora: osf (fs/bw) must be >= 1 -- it scales the frame "
            "length, so a zero or negative ratio returns a reservation "
            "that is silently too small");
    /* max_packet_symbols FIRST: it validates SF, and as one expression the
     * shift could be evaluated before it, and `1L << -1` is undefined. */
    const long n_sym = (long)max_packet_symbols(SF, de, max_payload);
    return (size_t)(n_sym * (1L << SF) * (long)osf);
}

size_t live_window_samples(int SF, int osf)
{
    /* SF7..SF12 come from an empirical per-SF window table. That table is
     * about one maximum-length packet per SF: the RAW entries agree with
     * live_overlap_samples() at osf = 8, UNDIVIDED, to within about 8% at
     * every one (SF7: 650000 against 628736). The value RETURNED carries the
     * 1.1 headroom factor below on top of that, so it sits roughly 10% to
     * 19% above one packet. The table is kept verbatim rather than derived,
     * because the agreement is an observation, not a definition. The
     * (void)osf below makes the same point: the returned count is a SAMPLE
     * count that only describes osf = 8, so "one packet per window" holds at
     * that rate and nowhere else.
     *
     * NOTE ON SCOPE: the GNU Radio decoder block does not use this.
     * decoder_impl computes its own chunk -- max(32 * 2^SF, 4096) * osf by
     * default -- and retires on SEARCH_TAIL_SYMBOLS, so the
     * (window + overlap)/window cost model below describes the
     * OVERLAP-WINDOW scheme, for callers that scan that way.
     *
     * SF5 and SF6 are not in it; the table is SF7 and up. Deriving them
     * the same way would give ~220k and ~370k samples, and that is the wrong
     * choice: the steady-state cost is (window + overlap)/window, so a window
     * of one packet means every sample is scanned about twice. At SF5 that
     * lands near 100% of real time, because SF5 also needs four scan phases
     * (see default_sync_phases). Holding the window at the SF7 value instead
     * amortises the SF5 overlap -- three times smaller than SF7's -- over the
     * same number of samples, which brings the factor to ~1.3, in line with
     * every other SF. The price is latency: 715k samples is 715 ms at
     * fs = 1 MHz, against 238 ms for the derived window. */
    static const double win[6] = {650e3, 1.2e6, 2e6, 3.6e6, 8e6, 14.5e6};
    const int i = (SF < 7) ? 0 : ((SF > 12) ? 5 : SF - 7);
    (void)osf;
    return (size_t)(1.1 * win[i]);
}

sync_result find_preamble(const std::vector<std::complex<float> >& s,
                          const tx_params& p, long start, bool robust_sync,
                          int sync_phases)
{
    sync_result R;
    const int SF = p.SF;
    const int osf = (int)(p.fs / p.bw + 0.5);
    const long K = 1L << SF;
    const long N = K * osf;
    const double ts = 1.0 / p.fs;
    const long nrise = (long)std::ceil(p.trise * p.fs);
    const int OSF_FINE = 4;

    if (!is_pow2(N)) return R;   /* power-of-two lengths only; see the file header */

    /* Base up-chirp and its conjugate. */
    std::vector<std::complex<double> > u0((size_t)N), d0((size_t)N);
    {
        const double f_ini = -p.bw / 2.0;
        const double df = p.bw / ((double)N * ts);
        for (long i = 0; i < N; i++) {
            const double t = (double)i * ts;
            const double ph = 2.0 * kPi * f_ini * t + kPi * df * (t * t);
            u0[(size_t)i] = std::complex<double>(std::cos(ph), std::sin(ph));
            d0[(size_t)i] = std::conj(u0[(size_t)i]);
        }
    }

    /* < 1 means "auto". An upper bound is needed for two separate faults: the
     * walk advances by N / m_phases, which is integer division, so
     * m_phases > N would make the step 0 and the loop never terminate; and
     * the robust ring below is m_phases * 10 * N * 8 bytes, 15 GiB at
     * m_phases = 200000. Sixteen is four times the largest value anything
     * documents or uses (default_sync_phases returns 2 or 4), and caps the
     * ring at 42 MB in the worst configuration. */
    if (sync_phases > SYNC_PHASES_MAX)
        throw std::invalid_argument(
            "sdrlora: sync_phases must be <= 16 (0 or less selects the "
            "per-SF default); finer than that buys nothing, the walk step "
            "N/sync_phases truncates to zero, and the robust-sync ring grows "
            "without bound");
    const int m_phases = sync_phases < 1 ? default_sync_phases(SF)
                                        : sync_phases;
    /* m_vec[phase][0] = up-chirp bin history, [1] = down-chirp, 6 deep. */
    std::vector<std::vector<std::vector<double> > > m_vec(
        (size_t)m_phases,
        std::vector<std::vector<double> >(2, std::vector<double>(6, -1.0)));

    /* SYNC_ACC_SYMBOLS, not a second copy of 6: the transmitter reads
     * the same constant through default_lead_in(). */
    const int ACC_SYMBOLS = SYNC_ACC_SYMBOLS, ACC_LAG = 4;
    const int acc_depth = ACC_SYMBOLS + ACC_LAG;
    std::vector<std::vector<std::vector<double> > > pu_ring;
    std::vector<long> pu_count;
    if (robust_sync) {
        pu_ring.assign((size_t)m_phases,
                       std::vector<std::vector<double> >((size_t)acc_depth,
                                                         std::vector<double>((size_t)N, 0.0)));
        pu_count.assign((size_t)m_phases, 0);
    }

    const long ns = (long)s.size();
    long s_ofs = start;
    int m_phase = 0;
    double stat_ref = 0.0, m_ref = 0.0;
    /* Sync-word symbol offsets, in bins above the preamble reference. */
    int sw1 = 8, sw2 = 16;
    if (!sync_word_symbols(p.sync_word, sw1, sw2)) return R;
    /* Not separable from the preamble at this SF -- see the helper. */
    if (!sync_symbols_separable(sw1, sw2, p.SF)) return R;

    /* Energy floor for the detector.
     *
     * mx/mean is scale-invariant, which is exactly what makes it a good
     * detection statistic -- and meaningless when the window holds no signal.
     * Over a run of zeros both are ~0 and the ratio is floating-point noise,
     * which sails past the threshold; the sync-word checks that follow are
     * then comparing argmaxes of denormals and pass by chance often enough.
     *
     * The case that matters: the encoder emits a 10000-sample zero lead-in,
     * which is 78 scan positions of silence at SF5 against 19 at SF7.
     * Without the floor a false lock appears inside the lead-in at low
     * spreading factors, and the buffer walk then advances past the real
     * packet.
     *
     * The floor is relative to the buffer's own mean power, so a uniformly
     * quiet capture still detects normally -- it only rejects windows that
     * are essentially empty compared with the rest of the buffer. */
    /* A negative start has to be refused HERE, before the buf_pow loop below
     * dereferences s[start + i]: the scan guard is floored at `start`, so it
     * does not catch it. decode_buffer never passes a negative cumulative,
     * but this is public API with a defaulted argument. */
    if (start < 0) return R;

    /* From `start`, not from 0. decode_buffer passes the whole buffer with
     * start = cumulative rather than a slice, so every read here is floored
     * at start. This one sets win_pow_floor below, the energy gate that
     * stops robust_sync locking onto silence -- averaging a different span
     * would move the false-lock threshold silently. */
    double buf_pow = 0.0;
    const long pow_n = ns - start;
    for (long i = start; i < ns; i++) {
        const double re = (double)s[(size_t)i].real();
        const double im = (double)s[(size_t)i].imag();
        buf_pow += re * re + im * im;
    }
    buf_pow = (pow_n > 0) ? buf_pow / (double)pow_n : 0.0;
    const double win_pow_floor = 1e-6 * buf_pow;

    std::vector<std::complex<double> > win((size_t)N), tmpf((size_t)N);

    while (true) {
        /* start is the access floor as well as the scan origin -- see the
         * note on buf_pow above. */
        if (s_ofs < start || s_ofs + N > ns) return R;

        double win_pow = 0.0;
        for (long i = 0; i < N; i++) {
            win[(size_t)i] = std::complex<double>(s[(size_t)(s_ofs + i)]);
            win_pow += std::norm(win[(size_t)i]);
        }
        win_pow /= (double)N;
        const bool has_signal = (win_pow > win_pow_floor);

        /* Su: dechirp with the down-chirp; Sd: with the up-chirp. */
        for (long i = 0; i < N; i++) tmpf[(size_t)i] = win[(size_t)i] * d0[(size_t)i];
        fft_fast(tmpf, -1);
        std::vector<double> Su((size_t)N);
        for (long i = 0; i < N; i++) Su[(size_t)i] = std::abs(tmpf[(size_t)i]);

        for (long i = 0; i < N; i++) tmpf[(size_t)i] = win[(size_t)i] * u0[(size_t)i];
        fft_fast(tmpf, -1);
        long m_d_i = 0;
        double bv = -1.0;
        for (long i = 0; i < N; i++) {
            const double m = std::abs(tmpf[(size_t)i]);
            if (m > bv) { bv = m; m_d_i = i; }
        }

        long m_u_i = 0;
        bv = -1.0;
        for (long i = 0; i < N; i++)
            if (Su[(size_t)i] > bv) { bv = Su[(size_t)i]; m_u_i = i; }

        if (robust_sync) {
            /* Su is already computed, so the accumulation costs no extra FFT. */
            const int idx = (int)(pu_count[(size_t)m_phase] % acc_depth);
            for (long i = 0; i < N; i++)
                pu_ring[(size_t)m_phase][(size_t)idx][(size_t)i] =
                    Su[(size_t)i] * Su[(size_t)i];
            pu_count[(size_t)m_phase]++;
            stat_ref = 0.0;
            if (pu_count[(size_t)m_phase] >= acc_depth) {
                std::vector<double> acc((size_t)N, 0.0);
                for (int j = 0; j < ACC_SYMBOLS; j++) {
                    const int r = ((idx - ACC_LAG - j) % acc_depth + acc_depth) % acc_depth;
                    const std::vector<double>& row = pu_ring[(size_t)m_phase][(size_t)r];
                    for (long i = 0; i < N; i++) acc[(size_t)i] += row[(size_t)i];
                }
                double mean = 0.0, mx = -1.0;
                long am = 0;
                for (long i = 0; i < N; i++) {
                    mean += acc[(size_t)i];
                    if (acc[(size_t)i] > mx) { mx = acc[(size_t)i]; am = i; }
                }
                mean /= (double)N;
                if (mean > 0.0) {
                    stat_ref = mx / mean;
                    m_ref = (double)(((am - 1 + N / 2) % N + N) % N) - (double)N / 2.0;
                }
            }
        }

        const double m_u = (double)(((m_u_i - 1 + N / 2) % N + N) % N) - (double)N / 2.0;
        const double m_d = (double)(((m_d_i - 1 + N / 2) % N + N) % N) - (double)N / 2.0;

        std::vector<double>& hu = m_vec[(size_t)m_phase][0];
        std::vector<double>& hd = m_vec[(size_t)m_phase][1];
        for (int i = 5; i > 0; i--) { hu[(size_t)i] = hu[(size_t)i - 1];
                                      hd[(size_t)i] = hd[(size_t)i - 1]; }
        hu[0] = m_u;
        hd[0] = m_d;

        bool detected;
        /* has_signal gates the ROBUST detector only: the floor sits with
         * the accumulator, against a false lock built from the zero
         * lead-in. Applying it to the plain detector too would change its
         * decisions on real samples -- a strong co-channel frame followed
         * by a weak one at another SF is enough -- and `consumed`, the
         * streaming contract, would change with them. */
        if (robust_sync && !has_signal) {
            detected = false;          /* nothing here to detect */
        } else if (robust_sync) {
            /* Detection from the accumulated preamble, then the SAME strict
             * sync-word requirement measured against that reliable reference. */
            detected = (stat_ref > p.sync_threshold &&
                        std::fabs(hd[0] - hd[1]) <= 1.0 &&
                        std::fabs(bin_off(hu[3], m_ref, sw1, K)) <= 1.0 &&
                        std::fabs(bin_off(hu[2], m_ref, sw2, K)) <= 1.0);
        } else {
            detected = (std::fabs(hd[0] - hd[1]) <= 1.0 &&
                        std::fabs(bin_off(hu[2], hu[3], sw2 - sw1, K)) <= 1.0 &&
                        std::fabs(bin_off(hu[3], hu[4], sw1, K)) <= 1.0 &&
                        std::fabs(hu[4] - hu[5]) <= 1.0);
        }

        /* A DETECTION IS ONLY USABLE IF ITS HISTORY IS IN THE BUFFER.
         * The fine-sync estimator below looks back (4 + 2) = 6 symbols
         * from s_ofs. Reporting a look-back off the front as truncated
         * would tell decode_buffer "a packet that has not finished
         * arriving" and stop the whole buffer walk -- but this is the
         * opposite case: history that is already gone, which no amount of
         * further data can supply, so the walk would discard every
         * COMPLETE packet behind the partial one.
         *
         * Refusing the candidate lets the scan run on through the rest
         * of the buffer, which is the only useful response: advancing
         * the window start instead would REDUCE the history available
         * to the same detection. */
        if (detected && s_ofs - 6 * N < start)
            detected = false;

        if (detected) {
            /* ---- fine sync: zero-padded FFT + parabolic peak interpolation */
            const long NF = N * OSF_FINE;
            std::vector<std::complex<double> > pad((size_t)NF);

            /* The fine estimate as a function of the ANCHOR a: s_ofs, and
             * +-N/2 when the estimates there are ambiguous (below).
             * `main` marks s_ofs itself: a look-back off the
             * window there is the unreachable truncation below, and its
             * down windows always fit; a shifted anchor whose windows leave
             * the buffer is simply not a candidate. */
            struct fine_t {
                bool ok = false;
                double m_u0 = 0.0, m_d0 = 0.0, peak = 0.0;
                double u[2] = {0.0, 0.0}, d[2] = {0.0, 0.0};
            };
            auto fine_at = [&](long a, bool main) -> fine_t {
                fine_t F;
                for (int i = 1; i <= 2; i++) {
                    const long base = a - (4 + i) * N;
                    /* `< start`, not `< 0`: the look-back is clipped at
                     * `start`, the access floor (see the note on buf_pow),
                     * so the estimate depends only on the samples the scan
                     * was handed from there on. */
                    if (base < start || base + N > ns) return F;
                    for (long j = 0; j < NF; j++)
                        pad[(size_t)j] = (j < N) ? std::complex<double>(s[(size_t)(base + j)]) * d0[(size_t)j]
                                                 : std::complex<double>(0.0, 0.0);
                    fft_fast(pad, -1);
                    long mi = 0;
                    double mv = -1.0;
                    for (long j = 0; j < NF; j++) {
                        const double m = std::abs(pad[(size_t)j]);
                        if (m > mv) { mv = m; mi = j; }
                    }
                    F.peak += mv;
                    double mf = (double)mi;
                    if (mi > 0 && mi < NF - 1) {
                        const double a1 = std::abs(pad[(size_t)(mi - 1)]);
                        const double b1 = std::abs(pad[(size_t)mi]);
                        const double c1 = std::abs(pad[(size_t)(mi + 1)]);
                        const double den = a1 - 2.0 * b1 + c1;
                        if (den != 0.0) mf = (double)mi + 0.5 * (a1 - c1) / den;
                    }
                    /* Zero-based bins: see the note above R.cfo_hz. */
                    F.u[i - 1] = std::fmod(std::fmod(mf + (double)NF / 2.0, (double)NF)
                                           + (double)NF, (double)NF) - (double)NF / 2.0;
                    F.m_u0 += F.u[i - 1];
                }
                F.m_u0 /= 2.0;
                for (int i = 1; i <= 2; i++) {
                    const long base = a - (i - 1) * N;
                    if (base < start || base + N > ns) {
                        F.ok = false;
                        F.peak = main ? -2.0 : -1.0;   /* main: truncated */
                        return F;
                    }
                    for (long j = 0; j < NF; j++)
                        pad[(size_t)j] = (j < N) ? std::complex<double>(s[(size_t)(base + j)]) * u0[(size_t)j]
                                                 : std::complex<double>(0.0, 0.0);
                    fft_fast(pad, -1);
                    long mi = 0;
                    double mv = -1.0;
                    for (long j = 0; j < NF; j++) {
                        const double m = std::abs(pad[(size_t)j]);
                        if (m > mv) { mv = m; mi = j; }
                    }
                    F.peak += mv;
                    double mf = (double)mi;
                    if (mi > 1 && mi < NF - 1) {
                        const double a1 = std::abs(pad[(size_t)(mi - 1)]);
                        const double b1 = std::abs(pad[(size_t)mi]);
                        const double c1 = std::abs(pad[(size_t)(mi + 1)]);
                        const double den = a1 - 2.0 * b1 + c1;
                        if (den != 0.0) mf = (double)mi + 0.5 * (a1 - c1) / den;
                    }
                    F.d[i - 1] = std::fmod(std::fmod(mf + (double)NF / 2.0, (double)NF)
                                           + (double)NF, (double)NF) - (double)NF / 2.0;
                    F.m_d0 += F.d[i - 1];
                }
                F.m_d0 /= 2.0;
                F.ok = true;
                return F;
            };

            fine_t F = fine_at(s_ofs, true);
            /* Unreachable while a candidate without 6 symbols of history
             * is refused above; kept because the failure mode if it became
             * reachable is a silent estimate formed from the wrong
             * samples. */
            if (!F.ok) { R.truncated = true; return R; }
            long anchor = s_ofs;
            /* ONE CHIRP SEGMENT, NOT TWO. A window straddling a symbol
             * boundary holds two
             * tones one BW apart, K * OSF_FINE fine bins, and averaging
             * estimates from different segments puts cfo_hz off by BW/4 to
             * BW/2, enough to lose frames at low fs/bw. When the two windows
             * of a pair sit ONE BW apart -- the two-segment signature, within
             * an eighth of a BW -- the estimate is formed half a symbol
             * either side too and the strongest-peaked anchor kept; a wider
             * trigger would fire under co-channel LoRa and cut its delivery.
             * The order, and `>` for the comparison, are deliberate: a tie
             * keeps s_ofs, then the earlier candidate. */
            const double B = (double)K * (double)OSF_FINE;
            if (std::fabs(std::fabs(F.u[0] - F.u[1]) - B) < B / 8.0
                || std::fabs(std::fabs(F.d[0] - F.d[1]) - B) < B / 8.0) {
                const long cand[2] = { s_ofs - N / 2, s_ofs + N / 2 };
                for (int c = 0; c < 2; c++) {
                    const fine_t G = fine_at(cand[c], false);
                    if (G.ok && G.peak > F.peak) { F = G; anchor = cand[c]; }
                }
            }
            const double m_u0 = F.m_u0;
            const double m_d0 = F.m_d0;

            /* ZERO-BASED BINS. These indices need no 1-based argmax
             * conversion (the MATLAB original's `mf - 1.0`): it would
             * cancel in t_est, a difference, but bias cfo_hz, a mean, by
             * -1 fine bin, -bw/(4 * 2^SF) -- -247 Hz at SF7. */
            R.cfo_hz = (m_u0 + m_d0) / 2.0 * p.bw / (double)K / (double)OSF_FINE;
            const double t_est = (m_d0 - m_u0) * (double)osf / 2.0 / (double)OSF_FINE
                                 + (double)anchor - 11.0 * (double)N - (double)nrise;
            /* ROUND, not ceil. A timing error of n samples
             * moves the dechirped peak by n/osf bins, so the offsets that
             * still decode span about +-osf/2 samples. ceil() would bias the
             * estimate up by anything in [0, 1): negligible at osf = 8, but
             * at osf = 3 and 4 it lands outside that window about half the
             * time.
             *
             * nearbyint, NOT floor(t_est + 0.5): the estimate is rounded
             * half to EVEN, and floor(x + 0.5) is round-half-UP; they
             * disagree on every exact .5 -- 2.5 gives 3 one way and 2 the
             * other. With the default FE_TONEAREST rounding mode nearbyint
             * IS round-half-to-even. An exact .5 needs t_est to land on a
             * half sample, which is rare. */
            R.p_ofs = (long)std::nearbyint(t_est);
            R.found = true;
            return R;
        }

        m_phase = (m_phase + 1) % m_phases;
        s_ofs += N / m_phases;
    }
}

/* --------------------------------------------------- full buffer decode */
std::vector<rx_result> decode_buffer(const std::vector<std::complex<float> >& in,
                                     const tx_params& p, bool channel_filter_rx,
                                     bool robust_sync, int sync_phases,
                                     bool soft_decision, bool implicit_header,
                                     int ih_payload_len, bool fec_correct,
                                     bool interpolate, bool report_crc_errors,
                                     long* consumed_out, long search_from,
                                     long walk_end, long* complete_at_out,
                                     int final_walk_in)
{
    const bool final_walk =
        final_walk_in < 0 ? (consumed_out == 0) : (final_walk_in != 0);
    std::vector<rx_result> out;
    if (consumed_out) *consumed_out = 0;
    if (complete_at_out) *complete_at_out = -1;
    long complete_at = -1;
    /* the open reported failure: its index in `out`, where its frame ends,
     * and whether the walk ended on a frame still arriving */
    long fail_idx = -1, fail_end = -1;
    bool exit_truncated = false;

    /* The sample-rate contract, enforced rather than warned about because
     * there is no warnings channel here: fs must be an integer
     * multiple of bw, that multiple must be a power of two (the sync FFTs are
     * length N and 4N, power-of-two transforms), and it must be at least 1.
     * Any power of two from 1 up is accepted: 1 is interpolated to
     * INTERP_TARGET_OSF first, and 2 works at every SF (see trap 3 in
     * lora_rx.cc). osf = 5..7 is refused, because those are not powers of
     * two. Returning empty on a rate that
     * never works would be indistinguishable from an empty band.
     *
     * Checked before osf is computed: p.bw = 0 would otherwise divide by zero
     * and cast an infinity to int. */
    if (p.SF < 5 || p.SF > 12)
        throw std::invalid_argument("sdrlora: sf must be 5..12");
    /* A packet error rate needs to tell a corrupted packet from a false
     * lock, and only the header's own FCS does that. Implicit mode has no
     * header -- hdr_ok is set unconditionally -- so every false lock would be
     * counted as a corrupted packet. */
    if (report_crc_errors && implicit_header)
        throw std::invalid_argument(
            "decode_buffer: report_crc_errors needs an explicit header -- the "
            "header FCS is what separates a corrupted packet from a false "
            "lock, and implicit mode has no header to check");
    /* The 4-byte pseudo-header lives INSIDE the CRC-protected payload and the
     * interleaver spreads damage across the whole block, so its LENGTH byte is
     * corrupted along with everything else and a badly hit packet cannot be
     * sized: a frame with a few payload symbols destroyed is no longer
     * counted, where standard framing counts it at every level of damage. A
     * silent under-count is worse than a refusal. */
    if (report_crc_errors && p.pseudo_header)
        throw std::invalid_argument(
            "decode_buffer: report_crc_errors needs pseudo_header=false -- "
            "with the 4-byte prefix the frame length is read from inside the "
            "damaged payload and heavily corrupted packets would be "
            "under-counted");
    if (p.SF < 7 && !implicit_header)
        throw std::invalid_argument(
            "sdrlora: SF5/SF6 have no room for an explicit header (it needs "
            "20 bits and the header block holds 4*(SF-2)); use implicit "
            "header mode and supply the payload length");
    /* The seven receive knobs -- trise, cutoff_ratio, numtaps,
     * sync_threshold, search_tail_symbols, min_osf, interp_target_osf --
     * are bounded on this path by check_rx_params: unchecked, numtaps = 1
     * would divide by numtaps-1 in
     * lowpass_taps and make every output sample NaN, and
     * interp_target_osf = 1e6 would be a 1.6 TB allocation request. Placed
     * after the checks above so their more specific messages still win.
     *
     * On a COPY with implicit_header taken from the argument: decode_buffer
     * carries it separately from p.implicit_header, and the receiver's
     * argument is the one that decides how this buffer is read. */
    {
        tx_params q = p;
        q.implicit_header = implicit_header;
        check_rx_params(q);
    }
    if (!(p.bw > 0.0) || !(p.fs > 0.0))
        throw std::invalid_argument("sdrlora: bw and fs must be positive");
    const double ratio = p.fs / p.bw;
    const int osf = (int)(ratio + 0.5);
    if (std::fabs(ratio - (double)osf) > 1e-9 * std::max(1.0, ratio))
        throw std::invalid_argument(
            "sdrlora: fs must be an integer multiple of bw");
    /* 2 is supported at every SF: with the payload decimation phase at 0
     * the two dechirped segments stay in phase at this rate (see trap 3 in
     * lora_rx.cc).
     *
     * 1 is accepted and interpolated up to INTERP_TARGET_OSF below. It is not
     * undersampled -- for complex IQ the Nyquist rate is fs >= bw -- but at
     * critical sampling the timing estimator degrades, so the samples are
     * re-sampled onto a finer grid before anything else runs. See upsample().
     * It stays lossy for a reason no receiver can fix: a LoRa chirp is not
     * strictly bandlimited, so its skirts alias in-band at the ADC unless
     * the signal is band-limited first, as an analogue anti-alias filter
     * does. */
    if (osf < 1 || !is_pow2(osf))
        throw std::invalid_argument(
            "sdrlora: fs/bw must be a power of two and at least 1 "
            "(the sync FFTs need power-of-two lengths)");

    /* Implicit-header framing carries no length on air, so the caller has to
     * supply one -- with neither, there is nothing to decode and the walk
     * would run over a zero-length payload. Refused, saying why. */
    /* ih_payload_len is the MESSAGE length; packet_rx() adds the 4-byte
     * pseudo-header prefix to it. So the quantity that must be at least one
     * is the ON-AIR length: 0 is legitimate when the prefix supplies those
     * four bytes, and it round-trips at every SF. */
    if (implicit_header) {
        const int min_ih = p.pseudo_header ? 0 : 1;
        if (ih_payload_len < min_ih)
            throw std::invalid_argument(
                "sdrlora: implicit_header needs a message length >= 0 with "
                "the pseudo-header prefix, >= 1 without it; with no header "
                "on air the receiver cannot discover the frame length");
        /* Without adding: ih_payload_len + 4 would wrap negative near
         * INT_MAX and pass, leaving the refusal to depend on whether the
         * buffer held a frame for packet_rx to refuse it again. */
        if (ih_payload_len > 255 - (p.pseudo_header ? 4 : 0))
            throw std::invalid_argument(
                "sdrlora: ih_payload_len plus the pseudo-header prefix must "
                "be <= 255 (the on-air length field is one byte)");
    }
    /* In implicit mode p.cr is passed through to the Hamming tables, where an
     * out-of-range value would be a heap-buffer-overflow WRITE in codebook().
     * In explicit mode the header carries the real rate and p.cr is unused,
     * so it is only checked where it is read. */
    if (implicit_header && (p.cr < 1 || p.cr > 4))
        throw std::invalid_argument(
            "sdrlora: cr must be 1..4 in implicit-header mode (it is read "
            "from the header otherwise)");
    /* ldro_enabled() treats anything outside 0..2 as auto, so an out-of-range
     * value would decode normally here while the GNU Radio blocks reject
     * it. */
    if (p.ldro < 0 || p.ldro > 2)
        throw std::invalid_argument(
            "sdrlora: ldro must be 0 (off), 1 (on) or 2 (auto)");
    /* THE WIDENED CUTOFF, eagerly and only when the filter runs. The
     * widening itself is further down, inside `if (channel_filter_rx)`, and
     * channel_filter refuses the result above 64 -- which would otherwise
     * surface as "cutoff_ratio must be finite, > 0 and <= 64", naming a
     * parameter the caller never set, from the middle of the decode. */
    if (channel_filter_rx)
        check_effective_cutoff(p.cutoff_ratio, p.hop_period, p.hop_freqs,
                               p.bw);
    /* Checked HERE and not only inside find_preamble, which runs only once
     * the buffer holds a symbol -- otherwise the same sync_phases would be
     * accepted on a short buffer and rejected on a long one. Validation must
     * not depend on the data. */
    if (sync_phases > SYNC_PHASES_MAX)
        throw std::invalid_argument(
            "sdrlora: sync_phases must be <= 16 (0 or less selects the "
            "per-SF default)");
    /* The scan grid must DIVIDE the symbol, or the window slips
     * (N % m_phases) samples every cycle and never returns to a symbol
     * boundary; integer division alone would quietly scan a different grid.
     * Checked here for the same reason as the bound above:
     * data-independent, so a short buffer and a long one agree. */
    {
        const long _N = (long)(1L << p.SF)
                      * (long)((int)(p.fs / p.bw + 0.5));
        const int _m = sync_phases < 1 ? default_sync_phases(p.SF)
                                       : sync_phases;
        if (_m > 0 && (_N % (long)_m) != 0)
            throw std::invalid_argument(
                "sdrlora: sync_phases must divide the symbol length "
                "2**SF * fs/bw -- a grid that does not divide it slips every "
                "cycle and never aligns to a symbol boundary");
    }
    /* find_preamble returns found=false for a sync word it cannot use, so an
     * invalid one would come back as an empty packet list -- indistinguishable
     * from an empty band, which is the failure mode the osf check above
     * refuses to have. encode() and the decoder block both throw for the
     * same value. */
    {
        int s1 = 0, s2 = 0;
        if (!sync_word_symbols(p.sync_word, s1, s2))
            throw std::invalid_argument(
                "sdrlora: sync_word must be one byte with two non-zero "
                "nibbles, or two raw symbols in 1..4095");
        if (!sync_symbols_separable(s1, s2, p.SF))
            throw std::invalid_argument(
                "sdrlora: sync symbols are within one bin of the preamble at "
                "this SF and cannot be told from more preamble");
    }

    tx_params pe = p;
    int eff_osf = osf;

    /* Undo the transmitter's I/Q inversion before anything else looks at the
     * samples: interpolation, the channel filter, the preamble search and the
     * demodulator then all run unchanged on a normal frame. See
     * tx_params::invert_iq. */
    /* NON-FINITE SAMPLES BECOME ZERO HERE, before anything transforms
     * them. Without it one NaN anywhere would make find_preamble's buf_pow
     * NaN, `win_pow > win_pow_floor` false at every scan position, and the
     * whole buffer silent, even frames demodulated long before the NaN. A
     * corrupt capture should cost the frames it touches, not the ones
     * either side of it.
     *
     * The copy is made only when there is something to clean, so the
     * ordinary path still costs one scan and no allocation. */
    std::vector<std::complex<float> > finite;
    const std::vector<std::complex<float> >* clean = &in;
    {
        bool bad = false;
        for (size_t i = 0; i < in.size() && !bad; i++)
            bad = !(std::isfinite(in[i].real())
                    && std::isfinite(in[i].imag()));
        if (bad) {
            finite = in;
            for (size_t i = 0; i < finite.size(); i++)
                if (!(std::isfinite(finite[i].real())
                      && std::isfinite(finite[i].imag())))
                    finite[i] = std::complex<float>(0.0f, 0.0f);
            clean = &finite;
        }
    }

    std::vector<std::complex<float> > deinv;
    const std::vector<std::complex<float> >* src = clean;
    if (p.invert_iq) {
        deinv.resize(clean->size());
        for (size_t i = 0; i < clean->size(); i++)
            deinv[i] = std::conj((*clean)[i]);
        src = &deinv;
        /* EXCEPT the hop plan. conj() flips the sign of the transmitter's hop
         * rotation along with everything else, so packet_rx's de-hop -- which
         * applies exp(-j2*pi*f*t) unconditionally -- would DOUBLE the offset
         * rather than remove it. Negating the channel list is the same
         * correction expressed where the reason for it is visible, and it
         * leaves apply_hops with one sign convention. */
        for (size_t i = 0; i < pe.hop_freqs.size(); i++)
            pe.hop_freqs[i] = -pe.hop_freqs[i];
    }

    std::vector<std::complex<float> > interp;
    const std::vector<std::complex<float> >* raw = src;
    if (osf < p.min_osf && interpolate) {
        /* Up to INTERP_TARGET_OSF, not MIN_OSF: clearing the minimum is
         * necessary, not sufficient. See lora_rx.h.
         *
         * CEIL, not truncation. Truncating division could land the
         * receiver BELOW its own min_osf -- neither validator requires the
         * ratio to be integral, only that
         * interp_target_osf >= min_osf: at osf 2 / min_osf 3 / target 3,
         * ceil reaches an effective osf of 4 and truncation 2. The defaults
         * (min_osf 2, target 4) give 4 either way. */
        const int factor = (p.interp_target_osf + osf - 1) / osf;
        /* The RESULT has to be a power of two, not just the input.
         * find_preamble's FFTs are length N and 4N with N = 2^SF * osf,
         * and it returns empty on a non-power-of-two N -- indistinguishable
         * from an empty band, which is the outcome this file refuses
         * everywhere else. The input osf is checked above; the
         * interpolated one is checked here. */
        /* ONE COPY of the rule, shared with decoder_impl's constructor,
         * which applies it at construction so that the first process()
         * cannot throw out through GNU Radio's thread body. */
        check_effective_osf(osf, p.min_osf, p.interp_target_osf, interpolate);
        upsample(*src, p.fs, factor, interp);
        raw = &interp;
        pe.fs = p.fs * (double)factor;
        eff_osf = osf * factor;
    }
    const long N = (1L << p.SF) * eff_osf;

    std::vector<std::complex<float> > filtered;
    const std::vector<std::complex<float> >* s = raw;
    if (channel_filter_rx) {
        /* Widened for the hop plan when one is in force: the filter runs
           BEFORE the de-hop -- it must, since the de-hop needs a frame
           offset the preamble search has not produced yet -- so a channel
           outside the passband is destroyed before anything can bring it
           back. Never narrows, so a caller's own value still wins whenever
           it is already wide enough. */
        double _cut = p.cutoff_ratio;
        if (p.hop_period > 0 && !p.hop_freqs.empty()) {
            const double _need = hop_cutoff_ratio(p.hop_freqs, pe.bw);
            if (_need > _cut) _cut = _need;
        }
        channel_filter(*raw, pe.bw, pe.fs, filtered, _cut, p.numtaps);
        s = &filtered;
    }

    /* FILTERED WHOLE, WALKED TO walk_end. A streaming caller holds back the
     * last numtaps/2 samples of its buffer so that no sample it walks is
     * filtered against zero padding. Handing only the rest in would move
     * the zero padding rather than remove it: the samples just before the
     * hold-back would be filtered with no right-hand context. So the whole
     * buffer goes through the front end and the walk stops at walk_end,
     * and the held-back samples are the context the hold-back exists to
     * provide. */
    const long _factor = (long)(eff_osf / osf);
    const long _in_end = (walk_end >= 0 && walk_end < (long)in.size())
                         ? walk_end : (long)in.size();
    std::vector<std::complex<float> > walked;
    if (_in_end < (long)in.size()) {
        const size_t _end = (size_t)(_in_end * _factor);
        if (s == &filtered) {
            filtered.resize(_end);
        } else {
            walked.assign(s->begin(), s->begin() + (long)_end);
            s = &walked;
        }
    }

    /* How many leading samples are finished with, for a streaming caller.
     * Everything before this has been scanned with full context and holds no
     * packet that has not already been reported. Expressed in the samples the
     * CALLER handed in, which is not the same domain when osf < MIN_OSF and
     * the input was interpolated up. */
    const long search_tail = (long)p.search_tail_symbols * N;
    /* THE REAL LENGTH, before the pad below: `done` never passes it. */
    const long ns_eff = (long)s->size();
    /* NOTHING MORE IS COMING, so what follows the buffer is silence -- one
     * symbol of it is appended. The timing
     * estimate can land a sample late, and a frame whose last modulated
     * sample is the buffer's last, or one or two short of it, would ask
     * for samples past the end and be lost. Only a buffer that can hold a
     * preamble and sync word (12 symbols) can hold a frame's end: below
     * that the pad finds nothing, and at fs/BW 65536 one symbol is 8 M
     * samples for a 1024-sample buffer. */
    if (final_walk && ns_eff > 12 * N) {
        if (s == &filtered) {
            filtered.resize(filtered.size() + (size_t)N);
        } else if (s == &walked) {
            walked.resize(walked.size() + (size_t)N);
        } else {
            walked.assign(s->begin(), s->end());
            walked.resize(walked.size() + (size_t)N);
            s = &walked;
        }
    }
    long done = 0;                    /* in the effective (post-interp) domain */
    bool done_set = false;

    /* RESUMES at search_from, scaled into the interpolated domain the way
     * *consumed_out is scaled back out of it. A streaming caller rewinds its
     * buffer so the channel filter has left context; the samples before
     * search_from are that context and were searched by the previous scan.
     * Re-searching them would deliver a frame that lay inside the rewind a
     * second time. */
    /* CLAMPED TO THE CALLER'S BUFFER FIRST, then scaled. Scaled first, a
     * huge search_from (2**62 at fs/bw 1, factor 4) would overflow and the
     * walk restart from the wrapped value. */
    /* And to the WALKED region, which is what search_from indexes into. */
    long _from = (search_from > 0) ? search_from : 0;
    if (_from > _in_end) _from = _in_end;
    long cumulative = _from * (_factor > 1 ? _factor : 1);
    if (cumulative > (long)s->size()) cumulative = (long)s->size();
    while ((long)s->size() - cumulative >= N) {
        /* No slice. find_preamble's `start` parameter is both the scan
         * origin and the access floor, so no copy of s[cumulative:] is made
         * per iteration -- that would be O(n^2) in copies across a window
         * with many locks, and a transient allocation the size of the
         * remaining buffer each time (~127 MB per symbol of advance at
         * SF12/osf 8).
         *
         * Consequence for everything below: sy.p_ofs is ABSOLUTE in the
         * *s domain, because t_est is formed as `... + s_ofs`. */
        const sync_result sy = find_preamble(*s, pe, cumulative, robust_sync,
                                             sync_phases);
        if (sy.truncated) {
            /* Ran out of samples mid-search: keep from where it was looking. */
            done = cumulative;
            done_set = true;
            break;
        }
        if (!sy.found) {
            /* Nothing here. Only a preamble straddling the end could still be
             * live, so keep the last search_tail samples and no more. */
            done = std::max(cumulative, ns_eff - search_tail);
            done_set = true;
            break;
        }

        /* Implicit header carries no length on air, so the caller supplies
         * it. Required for SF5 and SF6, which cannot use an explicit header
         * at all: the header block is the first SF-2 codewords at CR 4/8,
         * i.e. 4*(SF-2) bits, and the header needs 20. SF7 gives exactly 20,
         * SF6 gives 16, SF5 gives 12. */
        /* packet_rx takes the MESSAGE length and adds the pseudo-header
         * itself, so it is not pre-added here -- counting the prefix twice
         * would make every implicit decode fail. */
        /* The sample-clock offset the carrier offset implies: the carrier
         * NEGATED under invert_iq, whose conj() flips cfo_hz but not the
         * clock. */
        const double sfo_f0 = p.invert_iq ? -p.center_freq : p.center_freq;
        const double sfo = sfo_f0 != 0.0 ? sy.cfo_hz / sfo_f0 : 0.0;
        /* One attempt: packet_rx, then THE DRIFT READ FROM THE SIGNAL when
         * the frame located and failed its CRC -- kept only if the CRC then
         * passes, so a retry never loses a frame the first pass decoded. */
        auto attempt = [&](const tx_params& pp) -> rx_result {
            std::vector<double> fr;
            rx_result A = packet_rx(*s, pp, sy.p_ofs, sy.cfo_hz,
                                    implicit_header, ih_payload_len, pp.cr,
                                    pp.has_crc, soft_decision, fec_correct,
                                    sfo, &fr);
            const bool retry = !A.truncated && A.hdr_ok && !A.crc_ok &&
                               A.has_crc &&
                               (long)fr.size() >= DRIFT_MIN_SYMBOLS;
            double sfo_try = sfo;
            /* the effective osf, after any interpolation */
            if (retry && eff_osf >= DRIFT_MIN_OSF) {
                for (int t = 0; t < DRIFT_RETRIES; t++) {
                    const double d = drift_from_fracs(fr, p.SF);
                    if (std::fabs(d) * (double)fr.size() *
                            (double)(1L << p.SF) < DRIFT_MIN_BINS)
                        break;
                    sfo_try = sfo_try + d;
                    fr.clear();
                    rx_result R2 = packet_rx(*s, pp, sy.p_ofs, sy.cfo_hz,
                                             implicit_header, ih_payload_len,
                                             pp.cr, pp.has_crc, soft_decision,
                                             fec_correct, sfo_try, &fr);
                    if (!R2.truncated && R2.hdr_ok && R2.crc_ok) return R2;
                    if (R2.truncated || !R2.hdr_ok ||
                        (long)fr.size() < DRIFT_MIN_SYMBOLS)
                        break;
                }
            } else if (retry) {
                /* Below fs/BW 4 only the first estimate is trusted, and the
                 * search runs around it and around the carrier's drift. */
                const double d = drift_from_fracs(fr, p.SF);
                double centres[2] = { sfo, sfo + d };
                bool try_centre[2] = { false, true };
                const int n_centres =
                    std::fabs(d) * (double)fr.size() *
                            (double)(1L << p.SF) >= DRIFT_MIN_BINS ? 2 : 1;
                for (int ci = 0; ci < n_centres; ci++) {
                    for (int gi = try_centre[ci] ? -1 : 0; gi < 6; gi++) {
                        const double g = gi < 0 ? 0.0 : DRIFT_GRID_PPM[gi];
                        rx_result R2 = packet_rx(*s, pp, sy.p_ofs, sy.cfo_hz,
                                                 implicit_header,
                                                 ih_payload_len, pp.cr,
                                                 pp.has_crc, soft_decision,
                                                 fec_correct,
                                                 centres[ci] + g * 1e-6);
                        if (!R2.truncated && R2.hdr_ok && R2.crc_ok)
                            return R2;
                    }
                }
            }
            return A;
        };
        rx_result R = attempt(pe);
        /* LDRO THE OTHER WAY, when it is automatic and the frame still
         * located and failed its CRC: Sandeep Mistry's Arduino LoRa 0.8.0
         * transmits SF11 / 125 kHz without the LDRO the rule asks for. Kept
         * only if the CRC then passes. LDRO on costs symbols, so such a
         * frame read with it can run past the end: on the last feed the
         * shorter reading is tried too -- while streaming the frame
         * completes and fails its CRC instead. Only a
         * frame that carries a CRC: without one every reading "passes", and
         * a real LDRO frame cut off at the end would come back as wrong
         * bytes with crc_ok set. */
        const bool cut = final_walk && R.truncated && R.hdr_ok &&
                         R.has_crc && ldro_enabled(pe.SF, pe.bw, 2);
        if (pe.ldro == 2 && R.hdr_ok && p.SF >= LDRO_RETRY_MIN_SF &&
            (cut || (!R.truncated && !R.crc_ok && R.has_crc))) {
            tx_params pf = pe;
            pf.ldro = ldro_enabled(pe.SF, pe.bw, 2) ? 0 : 1;
            rx_result R2 = attempt(pf);
            if (!R2.truncated && R2.hdr_ok && R2.crc_ok) R = R2;
        }
        if (R.truncated && final_walk) {
            /* NOTHING MORE IS COMING (final_walk): a
             * frame cut off by the end -- or a corrupted header claiming a
             * length past it -- will never complete, and ending the walk on
             * it would lose every complete frame behind it. A failed lock:
             * one symbol on. */
            const long nx = std::max(sy.p_ofs, cumulative) + N;
            if (nx <= cumulative) {
                done = cumulative;
                done_set = true;
                break;
            }
            cumulative = nx;
            continue;
        }
        if (R.truncated) {
            exit_truncated = true;
            /* A located packet that has not finished arriving. Keep it from
             * its own start -- that is the whole buffer a streaming receiver
             * ever needs to hold, and only while a packet is in flight. */
            /* p_ofs is absolute, so this is max(p_ofs, cumulative): a fine
             * estimate landing before this window must not retire samples
             * ahead of it. */
            done = std::max(sy.p_ofs, cumulative);
            done_set = true;
            /* and where it will be complete, when its length is known */
            if (R.complete_at >= 0) complete_at = sy.p_ofs + R.complete_at;
            break;
        }

        /* The walk below advances one symbol on anything that did not fully
         * decode, because a lock that failed may not be where a packet starts.
         * That lumps together two things the receiver can tell apart:
         *
         *   a FALSE LOCK -- find_preamble fired on noise or a neighbour. The
         *   PHY header carries its own FCS, so this fails it. Nothing is known
         *   about where anything ends: advance one symbol.
         *
         *   a LOCATED PACKET -- the header FCS passed, so a real LoRa header
         *   is here and its LENGTH and CR say exactly where the packet ends.
         *   A failed payload CRC there is a corrupted packet. Report it once
         *   and step over it.
         *
         * Splitting those is what makes a packet error rate countable: without
         * it the same corrupted frame is re-found at every later symbol offset
         * and each re-find looks like another failure. On noise a lock that
         * also passes the header FCS is rare, which is what makes the count
         * trustworthy.
         *
         * decode_buffer rejects the two configurations where the discriminator
         * does not exist, so this does not have to re-check them. */
        const bool decoded = R.hdr_ok && R.crc_ok;
        const bool located = report_crc_errors && !decoded && R.hdr_ok &&
                             !R.payload.empty() && R.consumed > 0;
        /* A REPORTED FAILURE DOES NOT SKIP ITS FRAME: it steps one symbol
         * like any failed lock and
         * remembers the frame's span; a later lock inside it that decodes
         * REPLACES the failure, one that fails again is not counted twice.
         * Skipping the frame would cost good frames, because a re-lock one
         * symbol on often decodes the same frame cleanly. */
        const bool same = fail_idx >= 0 && sy.p_ofs < fail_end;
        if (decoded) {
            if (same) {
                out[(size_t)fail_idx] = R;
                fail_idx = -1;
                fail_end = -1;
            } else {
                out.push_back(R);
            }
        } else if (located && !same) {
            out.push_back(R);
            fail_idx = (long)out.size() - 1;
            fail_end = sy.p_ofs + R.consumed;
        }

        /* Advance past a decoded packet, and by one symbol on a FAILED
         * decode: skipping a packet's worth from a lock that did
         * not decode is wrong, because the reason it did not decode may be
         * that sy.p_ofs is not where a packet starts -- and then the skip
         * lands inside, or past, a real one. A false lock a few thousand
         * samples ahead of a packet (the detection window's tail reaching the
         * preamble across the lead-in) would make the walk step over the
         * packet entirely; any false lock, from noise or an adjacent channel,
         * would cost every later packet in the window.
         *
         * So on failure advance one symbol: it rules out only what was ruled
         * out, and still guarantees progress. */
        /* p_ofs is clamped at 0 because the fine timing estimate can land
         * before the window it was measured in and come back negative, and
         * then an advance of p_ofs + N moves by less than a symbol or not at
         * all -- the walk stalls on the same window forever. */
        /* Absolute, so this is the NEXT cumulative rather than a delta.
         * R.consumed is relative to p_ofs ("samples used past p_ofs"), so
         * p_ofs + consumed is the packet's end. The clamp floor is
         * `cumulative` instead of 0 for the same reason as above. */
        const long next = (decoded && R.consumed > 0)
                              ? (sy.p_ofs + R.consumed)
                              : (std::max(sy.p_ofs, cumulative) + N);
        /* The stall guard: a walk that does not move forward would
         * re-examine the same window forever. */
        if (next <= cumulative) {
            done = cumulative;
            done_set = true;
            break;
        }
        cumulative = next;
    }

    if (consumed_out) {
        if (!done_set) done = cumulative;        /* fell out of the while */
        /* A reported failure's frame stays retired: a streaming scan ending
         * inside its span would
         * re-walk it next time and report it again. Not past a frame still
         * arriving. */
        if (fail_end > done && !exit_truncated) done = fail_end;
        if (done < 0) done = 0;
        if (done > ns_eff) done = ns_eff;
        /* Back to the caller's sample domain. */
        *consumed_out = (_factor > 1) ? (done / _factor) : done;
    }
    /* Rounded DOWN into the caller's domain: an early estimate only costs a
     * scan that finds the frame truncated again. */
    if (complete_at_out && complete_at >= 0)
        *complete_at_out = (_factor > 1) ? (complete_at / _factor)
                                         : complete_at;
    return out;
}

} /* namespace sdrlora */
} /* namespace gr */
