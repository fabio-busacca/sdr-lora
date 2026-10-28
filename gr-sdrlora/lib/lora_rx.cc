/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
/*
 * LoRa receive chain.
 *
 * Numerical traps, all of them load-bearing:
 *
 *  1. The demodulator uses the INVERSE FFT, not the forward one. That is a
 *     positive exponent, so the bin ordering is mirrored relative to a
 *     forward transform. Getting it wrong yields plausible-looking but wrong
 *     symbols.
 *
 *  2. The header maps (K - 1 - k) / 4 while the payload maps
 *     (K - 2 - k) / 2^(2*DE). Different constant, and the header always
 *     divides by 4 because DE_hdr is 1 regardless of SF.
 *
 *  3. The decimation phase must be 0 for BOTH header and payload. It is
 *     tempting to argue that a one-sample shift only applies a constant
 *     phase to the dechirped tone and so cannot move an argmax. That is
 *     wrong, and at fs/bw = 2 it breaks reception below SF11.
 *
 *     A dechirped symbol is TWO tone segments, not one: the received chirp
 *     wraps at the symbol value, so the part before the wrap beats down to
 *     k*bw/K and the part after to k*bw/K - bw. Both fold into the same bin
 *     -- that is what makes the symbol value cyclic -- and they sum there
 *     COHERENTLY. Decimating from index p delays the block by p samples,
 *     which gives the two segments a RELATIVE phase of 2*pi*p/osf, since
 *     they differ in frequency by exactly bw. At p/osf = 1/2 they arrive 180
 *     degrees apart and cancel -- partially, since complete cancellation
 *     needs a symbol value that splits the block equally. Averaged over all K
 *     values at SF7, the correct bin keeps 1.00, 0.95, 0.81, 0.63, 0.50 of
 *     its amplitude as p/osf goes 0 to 1/2, and its argmax is right 100%,
 *     100%, 100%, 65%, 52% of the time. Half the symbols wrong is far past
 *     what the FEC carries.
 *
 *     So p = 1 is 1/8 of the way to that notch at osf = 8 and nearly
 *     invisible, and sits exactly in it at osf = 2. The notch tracks osf/2
 *     at every rate, exactly as the phase argument requires.
 *
 *  4. The symbol mapping rounds half to EVEN. It only matters at SF11/SF12
 *     where DE = 1 makes the division by 4 land on .5, but it does matter.
 */
#include "sdrlora/lora_rx.h"
#include "sdrlora/lora_tables.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

namespace gr {
namespace sdrlora {

static const double kPi = 3.14159265358979311600;

void fft_radix2(std::vector<std::complex<double> >& a, int sign)
{
    const size_t n = a.size();
    if (n < 2) return;

    /* bit-reversal permutation */
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = (double)sign * 2.0 * kPi / (double)len;
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; k++) {
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

/* W_n^k = exp(sign * 2 pi i k / n), k = 0..n-1, each from cos/sin
 * directly rather than by recurrence, cached per thread -- two decoder
 * blocks never share a table. */
static const std::vector<std::complex<double> >& fft_table(size_t n,
                                                           int sign)
{
    thread_local std::map<std::pair<size_t, int>,
                          std::vector<std::complex<double> > > cache;
    std::vector<std::complex<double> >& t = cache[std::make_pair(n, sign)];
    if (t.empty()) {
        t.resize(n);
        const double step = (double)sign * 2.0 * kPi / (double)n;
        for (size_t k = 0; k < n; k++)
            t[k] = std::complex<double>(std::cos(step * (double)k),
                                        std::sin(step * (double)k));
    }
    return t;
}

/* Real arithmetic, so the inner loop carries no NaN-recovery branch;
 * identical to std::complex's product for finite operands. */
static inline std::complex<double> cmul(const std::complex<double>& a,
                                        const std::complex<double>& b)
{
    return std::complex<double>(a.real() * b.real() - a.imag() * b.imag(),
                                a.real() * b.imag() + a.imag() * b.real());
}

void fft_fast(std::vector<std::complex<double> >& a, int sign)
{
    typedef std::complex<double> cd;
    const size_t N = a.size();
    if (N < 2) return;
    const std::vector<cd>& W = fft_table(N, sign);
    thread_local std::vector<cd> work;
    if (work.size() < N) work.resize(N);
    cd* x = a.data();
    cd* y = work.data();
    size_t n = N, s = 1;
    /* Stockham: stage n reads x at stride s and writes y in place order,
     * so no bit reversal; W_n^p is W_N^(p * N/n). */
    while (n >= 4) {
        const size_t n0 = n / 4;
        const size_t stride = N / n;
        for (size_t p = 0; p < n0; p++) {
            const cd w1 = W[p * stride];
            const cd w2 = W[2 * p * stride];
            const cd w3 = W[3 * p * stride];
            const cd* xa = x + s * p;
            const cd* xb = x + s * (p + n0);
            const cd* xc = x + s * (p + 2 * n0);
            const cd* xd = x + s * (p + 3 * n0);
            cd* y0 = y + s * (4 * p);
            cd* y1 = y0 + s;
            cd* y2 = y1 + s;
            cd* y3 = y2 + s;
            for (size_t q = 0; q < s; q++) {
                const cd A = xa[q], B = xb[q], C = xc[q], D = xd[q];
                const cd apc = A + C, amc = A - C, bpd = B + D, bmd = B - D;
                /* -sign * i * (B - D) */
                const cd jbmd = (sign < 0) ? cd(-bmd.imag(), bmd.real())
                                           : cd(bmd.imag(), -bmd.real());
                y0[q] = apc + bpd;
                y1[q] = cmul(w1, amc - jbmd);
                y2[q] = cmul(w2, apc - bpd);
                y3[q] = cmul(w3, amc + jbmd);
            }
        }
        n = n0;
        s *= 4;
        std::swap(x, y);
    }
    if (n == 2) {
        for (size_t q = 0; q < s; q++) {
            const cd A = x[q], B = x[q + s];
            y[q] = A + B;
            y[q + s] = A - B;
        }
        std::swap(x, y);
    }
    if (x != a.data())
        for (size_t i = 0; i < N; i++) a[i] = x[i];
}

/* Round half to even. */
static long round_half_even(double x)
{
    const double f = std::floor(x);
    const double d = x - f;
    long r;
    if (d > 0.5) r = (long)f + 1;
    else if (d < 0.5) r = (long)f;
    else r = ((long)f % 2 == 0) ? (long)f : (long)f + 1;
    return r;
}

static long pymod(long a, long m) { const long r = a % m; return r < 0 ? r + m : r; }

/* MSB-first bits of value, nbits wide. */
static void bits_msb_first(uint32_t value, int nbits, uint8_t* out)
{
    for (int i = 0; i < nbits; i++)
        out[i] = (uint8_t)((value >> (nbits - 1 - i)) & 1u);
}

/* packbits(..., bitorder='little'): bits[0] is the LSB. */
static uint32_t bits_to_uint_lsb_first(const uint8_t* b, int n)
{
    uint32_t v = 0;
    for (int i = 0; i < n; i++)
        v |= (uint32_t)(b[i] & 1) << i;
    return v;
}

/* ------------------------------------------------------- Hamming decoding */
static const uint8_t* hamming_matrix(int cr)
{
    switch (cr) {
    case 1: return &HAMMING_P1[0][0];
    case 2: return &HAMMING_P2[0][0];
    case 3: return &HAMMING_P3[0][0];
    case 4: return &HAMMING_P4[0][0];
    default:
        /* Refused rather than mapped to the CR-4 matrix, which would make
         * every out-of-range coding rate look valid: codebook() sizes each
         * word 4 + cr and writes four bytes into it, so cr = -3 would be a
         * heap overflow WRITE and cr = 10 a global overread. */
        throw std::invalid_argument(
            "sdrlora: cr must be 1..4 (coding rate 4/5..4/8)");
    }
}

/* syndrome -> the single position whose flip explains it. Empty for CR 4/5 and
 * 4/6, whose d_min is 2: every single-bit syndrome there is ambiguous. */
static void syndrome_lut(int cr, std::vector<int>& lut)
{
    const int n = 4 + cr;
    const uint8_t* P = hamming_matrix(cr);
    std::vector<int> count(1 << cr, 0), pos(1 << cr, -1);
    for (int p = 0; p < n; p++) {
        int syn = 0;
        for (int c = 0; c < cr; c++) {
            int bit;
            if (p < 4) bit = P[p * cr + c];
            else bit = (c == p - 4) ? 1 : 0;
            syn |= (bit & 1) << c;
        }
        if (syn) { count[syn]++; pos[syn] = p; }
    }
    lut.assign(1 << cr, -1);
    for (int s = 1; s < (1 << cr); s++)
        if (count[s] == 1) lut[s] = pos[s];
}

void hamming_correct(std::vector<std::vector<uint8_t> >& C, int cr,
                     int& corrected, int& uncorrectable)
{
    corrected = 0;
    uncorrectable = 0;
    if (cr < 1 || cr > 4) return;
    const uint8_t* P = hamming_matrix(cr);
    std::vector<int> lut;
    syndrome_lut(cr, lut);

    for (size_t r = 0; r < C.size(); r++) {
        int syn = 0;
        for (int c = 0; c < cr; c++) {
            int acc = 0;
            for (int i = 0; i < 4; i++)
                acc ^= (C[r][i] & P[i * cr + c]);
            syn |= ((acc ^ C[r][4 + c]) & 1) << c;
        }
        if (!syn) continue;
        const int p = lut[syn];
        if (p < 0) uncorrectable++;
        else { C[r][p] ^= 1; corrected++; }
    }
}

/* ---------------------------------------------- de-interleave + de-gray */
/* Inverse of the transmit mapping: symbol estimates -> the block's data bits.
 * `bits_blk` is laid out symbol-major, PPM bits per symbol. */
/* The de-interleaver is a pure permutation, so LLRs ride the identical index
 * arithmetic as bits do -- hence the template. */
template <typename T>
static void deinterleave_block_t(const std::vector<T>& blk, int ppm,
                                 int n_sym_blk, std::vector<std::vector<T> >& C)
{
    C.assign(ppm, std::vector<T>(n_sym_blk, T()));
    std::vector<std::vector<T> > F(ppm, std::vector<T>(n_sym_blk, T()));
    for (int ii = 0; ii < ppm; ii++)
        for (int jj = 0; jj < n_sym_blk; jj++)
            F[ii][jj] = blk[(size_t)((ii + jj) % ppm) + (size_t)jj * ppm];
    for (int i = 0; i < ppm; i++)
        C[i] = F[ppm - 1 - i];      /* row flip */
}

static void deinterleave_block(const std::vector<uint8_t>& bits_blk, int ppm,
                               int n_sym_blk,
                               std::vector<std::vector<uint8_t> >& C)
{
    /* reshape(bits_blk, (ppm, n_sym_blk), order='F').transpose() gives
     * S[c][r] = bits_blk[r + c*ppm], i.e. bit r of symbol c. */
    deinterleave_block_t(bits_blk, ppm, n_sym_blk, C);
}

/* ------------------------------------------------------- soft decision */
/* (PPM, K) mask: does FFT bin `pos` carry a 1 at bit `b`?
 * Built by running the HARD decoder's own bin -> bits rule over every bin, so
 * the soft path cannot drift from the hard path's symbol mapping. */
static void bit_label_mask(int SF, int de, std::vector<std::vector<uint8_t> >& mask)
{
    const int ppm = SF - 2 * de;
    const long K = 1L << SF;
    const long mod = 1L << ppm;
    const long step = 1L << (2 * de);
    std::vector<uint32_t> gray, degray;
    gray_lut(ppm, gray, degray);

    mask.assign((size_t)ppm, std::vector<uint8_t>((size_t)K, 0));
    std::vector<uint8_t> bits((size_t)ppm, 0);
    for (long pos = 0; pos < K; pos++) {
        const long k = pos - 1;                       /* the stored pos-1 */
        const long b = round_half_even((double)(K - 2 - k) / (double)step);
        bits_msb_first(degray[pymod(b, mod)], ppm, &bits[0]);
        for (int i = 0; i < ppm; i++)
            mask[(size_t)i][(size_t)pos] = bits[(size_t)i];
    }
}

void symbol_llr(const std::vector<std::vector<double> >& mag, int SF, int de,
                std::vector<std::vector<double> >& llr)
{
    /* Public entry point, so all three arguments are checked. SF outside
     * 5..12 makes 1L << SF nonsense and ppm = SF - 2*de can go negative,
     * which mask.assign() would turn into a huge size_t; and the inner loop
     * below runs to mag[r].size() while indexing mask[b], which is 2^SF
     * wide, so a row of the wrong width would READ PAST THE MASK. */
    if (SF < 5 || SF > 12)
        throw std::invalid_argument("sdrlora: symbol_llr: SF must be 5..12");
    if (de < 0 || de > 1)
        throw std::invalid_argument(
            "sdrlora: symbol_llr: de must be 0 or 1 -- it is the "
            "low-data-rate flag and the label mask is ppm = SF - 2*de bits "
            "wide, which stops meaning anything outside that range");
    const size_t K_ = (size_t)1 << SF;
    for (size_t r = 0; r < mag.size(); r++)
        if (mag[r].size() != K_)
            throw std::invalid_argument(
                "sdrlora: symbol_llr: every row of mag must hold 2^SF "
                "magnitudes, one per FFT bin of a dechirped symbol");

    std::vector<std::vector<uint8_t> > mask;
    bit_label_mask(SF, de, mask);
    const size_t ppm = mask.size();
    llr.assign(mag.size(), std::vector<double>(ppm, 0.0));

    for (size_t r = 0; r < mag.size(); r++) {
        const std::vector<double>& m = mag[r];
        for (size_t b = 0; b < ppm; b++) {
            /* max-log: strongest bin that would carry a 0, against the
             * strongest that would carry a 1. Squared magnitudes, because
             * under noncoherent detection that is the natural metric. */
            double best0 = -1.0, best1 = -1.0;
            for (size_t i = 0; i < m.size(); i++) {
                const double p2 = m[i] * m[i];
                if (mask[b][i]) { if (p2 > best1) best1 = p2; }
                else            { if (p2 > best0) best0 = p2; }
            }
            llr[r][b] = best0 - best1;   /* positive => bit 0 */
        }
    }
}

/* The 16 codewords and their +/-1 signs, for ML decoding. */
static void codebook(int cr, std::vector<std::vector<uint8_t> >& words,
                     std::vector<std::vector<double> >& signs)
{
    const int n = 4 + cr;
    const uint8_t* P = hamming_matrix(cr);
    words.assign(16, std::vector<uint8_t>((size_t)n, 0));
    signs.assign(16, std::vector<double>((size_t)n, 0.0));
    for (int v = 0; v < 16; v++) {
        for (int i = 0; i < 4; i++)
            words[(size_t)v][(size_t)i] = (uint8_t)((v >> (3 - i)) & 1);
        for (int c = 0; c < cr; c++) {
            int acc = 0;
            for (int i = 0; i < 4; i++)
                acc ^= (words[(size_t)v][(size_t)i] & P[i * cr + c]);
            words[(size_t)v][(size_t)(4 + c)] = (uint8_t)(acc & 1);
        }
        for (int i = 0; i < n; i++)
            signs[(size_t)v][(size_t)i] = 1.0 - 2.0 * (double)words[(size_t)v][(size_t)i];
    }
}

void hamming_soft_decode(const std::vector<std::vector<double> >& L, int cr,
                         std::vector<std::vector<uint8_t> >& data)
{
    std::vector<std::vector<uint8_t> > words;
    std::vector<std::vector<double> > signs;
    codebook(cr, words, signs);
    const int n = 4 + cr;

    data.assign(L.size(), std::vector<uint8_t>(4, 0));
    for (size_t r = 0; r < L.size(); r++) {
        int best = 0;
        double bestm = -1e300;
        for (int v = 0; v < 16; v++) {
            double acc = 0.0;
            for (int i = 0; i < n; i++)
                acc += L[r][(size_t)i] * signs[(size_t)v][(size_t)i];
            if (acc > bestm) { bestm = acc; best = v; }
        }
        for (int i = 0; i < 4; i++)
            data[r][(size_t)i] = words[(size_t)best][(size_t)i];
    }
}

/* ------------------------------------------------------------ header decode */
struct header_fields {
    bool fcs_ok = false;
    int length = -1;
    bool has_crc = false;
    int cr = -1;
    std::vector<uint8_t> payload_bits; /* bits the header carried */
};

/* hdr_rows: 5 for an explicit header, 0 for implicit, where the same block
 * exists but carries only payload and has no FCS to check. */
static header_fields header_decode(int SF, const std::vector<int32_t>& k_hdr,
                                   bool fec_correct, int hdr_rows = 5)
{
    header_fields h;
    const int de_hdr = 1, cr_hdr = 4;
    const int ppm = SF - 2 * de_hdr;
    const int n_sym_hdr = 4 + cr_hdr;
    const long K = 1L << SF;
    const long mod = 1L << ppm;

    std::vector<uint32_t> gray, degray;
    gray_lut(ppm, gray, degray);

    std::vector<uint8_t> bits((size_t)ppm * n_sym_hdr, 0);
    for (int sym = 0; sym < n_sym_hdr; sym++) {
        /* header: (K - 1 - k) / 4, always /4 because DE_hdr is 1 */
        const long b = round_half_even((double)(K - 1 - (long)k_hdr[sym]) / 4.0);
        bits_msb_first(degray[pymod(b, mod)], ppm, &bits[(size_t)sym * ppm]);
    }

    std::vector<std::vector<uint8_t> > C;
    deinterleave_block(bits, ppm, n_sym_hdr, C);
    if (fec_correct) {
        int a, b2;
        hamming_correct(C, cr_hdr, a, b2);
    }

    if (hdr_rows == 0) {
        /* Implicit header: no header bits, so no FCS and no fields -- every
         * codeword is payload and the caller already knows length/cr/crc.
         *
         * This must return BEFORE the FCS below, which indexes C[0..4]. At
         * SF5 ppm is only 3, so reaching it segfaults. */
        h.fcs_ok = true;
        h.length = -1;
        h.cr = -1;
        h.has_crc = false;
        h.payload_bits.assign((size_t)(ppm * 4), 0);
        for (int i = 0; i < ppm; i++)
            for (int j = 0; j < 4; j++)
                h.payload_bits[(size_t)i * 4 + j] = C[i][j];
        return h;
    }

    /* FCS over the header bits in the order below, with the identity block
     * appended. */
    uint8_t v[17];
    int n = 0;
    for (int i = 3; i >= 0; i--) v[n++] = C[0][i];
    for (int i = 3; i >= 0; i--) v[n++] = C[1][i];
    for (int i = 3; i >= 0; i--) v[n++] = C[2][i];
    v[n++] = C[3][0];
    for (int i = 3; i >= 0; i--) v[n++] = C[4][i];
    bool any = false;
    for (int c = 0; c < 5; c++) {
        int acc = 0;
        for (int r = 0; r < 12; r++)
            acc ^= (v[r] & HEADER_FCS[r][c]);
        for (int r = 12; r < 17; r++)
            acc ^= (v[r] & ((r - 12 == c) ? 1 : 0));
        if (acc & 1) any = true;
    }
    h.fcs_ok = !any;
    if (!h.fcs_ok) return h;

    uint8_t lb[8];
    for (int i = 0; i < 4; i++) lb[i] = C[1][i];
    for (int i = 0; i < 4; i++) lb[4 + i] = C[0][i];
    h.length = (int)bits_to_uint_lsb_first(lb, 8);
    h.has_crc = C[2][0] != 0;
    h.cr = (int)bits_to_uint_lsb_first(&C[2][1], 3);
    /* 3 bits on air but only 1..4 are legal, so out of range means the
     * header is corrupt whatever its FCS says. Rejecting it here keeps an
     * invalid CR out of the FEC decoders and strengthens the FCS. */
    if (h.cr < 1 || h.cr > 4) { h.fcs_ok = false; return h; }

    const int n_bits_hdr = ppm * 4 - 4 * hdr_rows;
    h.payload_bits.assign((size_t)(n_bits_hdr > 0 ? n_bits_hdr : 0), 0);
    for (int i = hdr_rows; i < ppm; i++)
        for (int j = 0; j < 4; j++)
            h.payload_bits[(size_t)(i - hdr_rows) * 4 + j] = C[i][j];
    return h;
}

/* ----------------------------------------------------------- payload decode */
static int payload_n_sym(int SF, int length, bool has_crc, int cr,
                         int n_bits_hdr, int de)
{
    const int ppm = SF - 2 * de;
    const int n_bits_blk = ppm * 4;
    const int n_bits_tot = 8 * length + 16 * (has_crc ? 1 : 0);
    /* Clamped, and computed in a signed type: a LENGTH read off a corrupt
     * header can make this difference negative, and zero blocks is the
     * legitimate answer when the payload fits the header block's spare
     * bits. */
    int n_blk = (int)std::ceil((double)(n_bits_tot - n_bits_hdr) /
                               (double)n_bits_blk);
    if (n_blk < 0) n_blk = 0;
    return n_blk * (4 + cr);
}

/* ------------------------------------------------------------- packet_rx */
/* Bins per symbol the peaks move, from offsets that wrap at +-0.5. Not
 * unwrapped -- a noisy step
 * past half a bin would slip a whole bin and drag a fitted line toward
 * zero. Each offset is a phase, and the rate is the angle
 * of sum z[i+L] * conj(z[i]) over 2*pi*L; lag 1 is unambiguous, 4 and 16
 * refine it, each placed by the one before. */
static double drift_rate(const std::vector<double>& fracs)
{
    const long n = (long)fracs.size();
    std::vector<double> zr((size_t)n), zi((size_t)n);
    for (long i = 0; i < n; i++) {
        zr[(size_t)i] = std::cos(2.0 * kPi * fracs[(size_t)i]);
        zi[(size_t)i] = std::sin(2.0 * kPi * fracs[(size_t)i]);
    }
    double rate = 0.0;
    for (size_t li = 0; li < sizeof(DRIFT_LAGS) / sizeof(DRIFT_LAGS[0]);
         li++) {
        const long lag = DRIFT_LAGS[li];
        if (lag >= n) break;
        double cr = 0.0, ci = 0.0;
        for (long i = 0; i < n - lag; i++) {
            const size_t j = (size_t)(i + lag), k = (size_t)i;
            cr += zr[j] * zr[k] + zi[j] * zi[k];
            ci += zi[j] * zr[k] - zr[j] * zi[k];
        }
        const double raw = std::atan2(ci, cr) / (2.0 * kPi * (double)lag);
        if (li == 0) {
            rate = raw;
        } else {
            const double k = std::floor((rate - raw) * (double)lag + 0.5);
            rate = raw + k / (double)lag;
        }
    }
    return rate;
}

double drift_from_fracs(const std::vector<double>& fracs, int SF)
{
    return -drift_rate(fracs) / (double)(1L << SF);
}

/* The peak's power less the largest outside its main lobe
 * (the peak and its two neighbours), over the mean power outside it. */
double peak_gap(const std::vector<double>& power, size_t pos)
{
    const size_t K = power.size();
    const size_t lo = (pos + K - 1) % K, hi = (pos + 1) % K;
    double sum = 0.0, mx = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < K; i++) {
        if (i == pos || i == lo || i == hi) continue;
        sum += power[i];
        if (power[i] > mx) mx = power[i];
        n++;
    }
    const double m = sum / (double)n;
    if (m > 0.0) return (power[pos] - mx) / m;
    return power[pos] > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
}

/* The no-CRC threshold: where a frame of 8 noise symbols passes 1 time in
 * 1000, per SF; flat from SF8. */
double nocrc_min_gap(int SF)
{
    switch (SF) {
    case 5: return 2.9;
    case 6: return 2.65;
    case 7: return 2.55;
    default: return NOCRC_MIN_GAP_DEFAULT;
    }
}

rx_result packet_rx(const std::vector<std::complex<float> >& s_in,
                    const tx_params& p, long p_ofs, double cfo_hz,
                    bool implicit_header, int ih_length, int ih_cr,
                    bool ih_has_crc, bool soft_decision, bool fec_correct,
                    double sfo, std::vector<double>* fracs_out)
{
    /* Everything below is reachable from phy_packet_rx(), whose
     * implicit_header argument DEFAULTS to false -- so these are not
     * theoretical. Validated here rather than only in decode_buffer()
     * because this is a public entry point in its own right. */
    if (p.SF < 5 || p.SF > 12)
        throw std::invalid_argument("sdrlora: sf must be 5..12");
    /* And ldro, for the same reason. ldro_enabled() returns the AUTO rule
       for EVERYTHING that is not 0 or 1 -- including -1 and 3 -- so an
       unchecked ldro=3 would demodulate with whatever AUTO decided and
       return a packet. check_rx_params() carries this bound, but
       packet_rx() is a public entry point that does not call it, and so
       validates inline.
       NOT the whole of check_rx_params(): that tests p.implicit_header,
       while this function takes implicit_header as its own argument, so
       the SF5/SF6 rule there would be applied to the wrong flag. */
    if (p.ldro < 0 || p.ldro > 2)
        throw std::invalid_argument(
            "sdrlora: ldro must be 0 (off), 1 (on) or 2 (auto)");
    /* AND THE RATE, for the same reason: finite, positive and BOUNDED,
     * through the same check_rate_bounds as check_rx_params. Unchecked, a
     * non-finite, zero, negative or huge bw/fs would reach d0((size_t)N)
     * with N = 2^SF * garbage through undefined int conversions. */
    check_rate_bounds(p.bw, p.fs);
    /* AND trise, which this function READS AND CONVERTS TO INT:
     * `nrise = (long)ceil(p.trise * p.fs)` below is undefined for a
     * non-finite product, and the result would be a silently truncated
     * packet rather than an error. Bound and wording follow
     * check_rx_params, which this entry point deliberately does not call
     * (it tests p.implicit_header, and this function takes
     * implicit_header as its own argument). */
    if (!std::isfinite(p.trise))
        throw std::invalid_argument("sdrlora: trise must be finite");
    if (!(p.trise > 0.0) || p.trise > 1.0)
        throw std::invalid_argument(
            "sdrlora: trise must be > 0 and <= 1 s");
    {
        const double _ratio = p.fs / p.bw;
        /* Defined: check_rate_bounds holds the ratio to OSF_MAX. */
        const int _osf = (int)(_ratio + 0.5);
        const double _tol = 1e-9 * (_ratio > 1.0 ? _ratio : 1.0);
        if (_osf < 1 || _ratio - (double)_osf > _tol
                || _ratio - (double)_osf < -_tol)
            throw std::invalid_argument(
                "sdrlora: fs must be an integer multiple of bw");
    }
    /* And the hop plan, which this function reads and APPLIES below --
     * AFTER the rate, as in check_rx_params: its band is (fs - bw)/2,
     * negative when fs < bw, so checking it first would blame the plan for
     * a bad rate. */
    check_hop_plan(p);
    /* De-hopped HERE, where the frame offset is already known and before a
       single symbol is demodulated, so everything below sees an ordinary
       single-channel frame. A copy is needed because the input is const;
       the copy is made only when a schedule is actually in force. `s`
       below is an alias, so the body is unchanged whichever way this
       goes. */
    /* A FRAME START FURTHER FROM THE BUFFER THAN ITS LENGTH locates nothing
     * in it, and every offset below is formed as p_ofs + something, which
     * would overflow for an extreme p_ofs. Returned as truncated here,
     * before any sum is formed. */
    if (p_ofs > (long)s_in.size() || p_ofs < -(long)s_in.size()) {
        rx_result T;
        T.truncated = true;
        return T;
    }
    std::vector<std::complex<float> > _dehopped;
    const bool _hopping = (p.hop_period > 0 && !p.hop_freqs.empty());
    if (_hopping) {
        _dehopped = s_in;
        /* Anchored at the HEADER, not at p_ofs. p_ofs is a frame start
           computed as if the preamble were 8 symbols long, so it is off the
           true frame start by (preamble_symbols - 8)*N -- and a schedule
           anchored there would be de-hopped against the wrong phase at
           every other preamble length. p_ofs + nrise + 12.25*N is the header
           start EXACTLY, for any preamble length, by the same cancellation
           that makes demodulation work. */
        const long _N = (long)(1L << p.SF)
                      * (long)((int)(p.fs / p.bw + 0.5));
        const long _anchor = p_ofs + (long)std::ceil(p.trise * p.fs)
                           + (long)(12.25 * (double)_N);
        /* the block edges drift with the symbol windows: sym_start's
         * rule, with its reference made absolute */
        apply_hops(_dehopped, _anchor, p.hop_period, p.hop_freqs,
                   _N, p.fs, true, sfo,
                   (double)p_ofs + std::ceil(p.trise * p.fs)
                       + SFO_REF_SYMBOLS * (double)_N);
    }
    const std::vector<std::complex<float> >& s = _hopping ? _dehopped : s_in;
    if (p.SF < 7 && !implicit_header)
        throw std::invalid_argument(
            "sdrlora: SF5/SF6 have no room for an explicit header (it needs "
            "20 bits and the header block holds 4*(SF-2)); use implicit "
            "header mode and supply the payload length. Reaching the FCS at "
            "SF5 reads past the end of the codeword array");
    if (implicit_header) {
        if (ih_cr < 1 || ih_cr > 4)
            throw std::invalid_argument(
                "sdrlora: ih_cr must be 1..4; no Hamming generator exists "
                "outside that range");
        /* 8 * length would overflow int for a large enough value, and the
         * payload arrays are sized from it, so it is bounded first. The
         * on-air LENGTH field is one byte, and ih_length is the MESSAGE
         * length -- the pseudo-header prefix is added to it below, so 0 is
         * legitimate when that prefix supplies the bytes. */
        /* Bounded BEFORE anything adds the prefix: ih_length + 4 would wrap
         * negative near INT_MAX. The sum is formed only once ih_length is
         * known to be small. */
        const int prefix = p.pseudo_header ? 4 : 0;
        if (ih_length < 0 || ih_length > 255 - prefix
                || ih_length + prefix < 1)
            throw std::invalid_argument(
                "sdrlora: ih_length plus the pseudo-header prefix must be "
                "1..255 (the on-air length field is one byte)");
    }
    rx_result R;
    const int SF = p.SF;
    const int osf = (int)(p.fs / p.bw + 0.5);
    const long K = 1L << SF;
    const long N = K * osf;
    const double ts = 1.0 / p.fs;
    /* p.trise, NOT the TRISE constant. find_preamble() measures p_ofs with
     * p.trise (lora_sync.cc); this skips nrise + 12.25*N from it to reach the
     * header. Mixing the two would put the header (TRISE - p.trise)*fs
     * samples off: at trise = 200 us and fs = 1 Msps that is 150 samples,
     * 18.75 FFT bins at osf 8, so frames sent at a non-default trise would
     * not decode. */
    /* long, MATCHING lora_sync.cc's identical expression, so the two
     * spellings of one quantity agree. The conversion is defined because
     * check_rate_bounds holds fs <= FS_MAX and trise is bounded to (0, 1],
     * which keeps the product within 1e12; an int would not hold it. */
    const long nrise = (long)std::ceil(p.trise * p.fs);
    /* bw is known here, so the duration-based LDRO rule is resolved once and
     * used for both the payload sizing and the de-interleave below. */
    const int de = ldro_enabled(SF, p.bw, p.ldro);

    /* Reference down-chirp: chirp(BW/2, N, Ts, -BW/(N*Ts)). */
    std::vector<std::complex<double> > d0((size_t)N);
    {
        const double f_ini = p.bw / 2.0;
        const double df = -p.bw / ((double)N * ts);
        for (long i = 0; i < N; i++) {
            const double t = (double)i * ts;
            const double ph = 2.0 * kPi * f_ini * t + kPi * df * (t * t);
            d0[(size_t)i] = std::complex<double>(std::cos(ph), std::sin(ph));
        }
    }

    /* SAMPLE-CLOCK TRACKING, sym_start's rule: the first sample of
     * the symbol nominally `ofs` into the frame. One crystal drives a
     * radio's clock and its LO, so the carrier offset fine sync measured
     * is the clock error times the carrier, and the same error stretches
     * every symbol; a symbol sits sfo*ofs samples early, counted from where
     * the timing estimate was made. sfo == 0 is the nominal placement
     * exactly. nearbyint is round-half-even. */
    const double sfo_ref = (double)nrise + SFO_REF_SYMBOLS * (double)N;
    auto sym_start = [&](long ofs) -> long {
        if (sfo == 0.0) return p_ofs + ofs;
        return p_ofs + ofs -
               (long)std::nearbyint(sfo * ((double)ofs - sfo_ref));
    };

    /* Demodulate one symbol at sample offset `ofs`, decimating from `start`. */
    std::vector<std::complex<double> > temp((size_t)N);
    std::vector<std::complex<double> > dec;
    std::vector<double>* spec_out = 0;   /* when set, receives |ifft| per bin */
    std::vector<double> pw;              /* |ifft|^2 of the latest symbol */
    std::vector<double> gaps;            /* peak_gap of every symbol */
    auto demod = [&](long ofs, int start, bool& ok) -> long {
        ok = false;
        const long st = sym_start(ofs);
        if (st < 0 || st + N > (long)s.size()) return 0;
        for (long i = 0; i < N; i++) {
            const double ph = -2.0 * kPi * cfo_hz * ts * (double)(ofs + i);
            const std::complex<double> rot(std::cos(ph), std::sin(ph));
            temp[(size_t)i] =
                rot * std::complex<double>(s[(size_t)(st + i)]) * d0[(size_t)i];
        }
        /* Exactly K bins. `start` is 0 for both header and payload and the
         * argument is kept only so the choice is visible at the call sites;
         * any other value costs coherent gain and osf/2 destroys the symbol
         * outright (trap 3). Taking osf*K samples rather than a
         * `[start:-1:OSF]` slice also keeps the length a power of two at
         * osf = 2, which fft_radix2 requires. */
        dec.clear();
        for (long i = start; i < start + osf * K; i += osf)
            dec.push_back(temp[(size_t)i]);
        /* Inverse DFT -> positive exponent, scaled by 1/K. The scale is
         * irrelevant to argmax but is applied anyway so the LLRs below are
         * computed on the normalised magnitudes. */
        fft_radix2(dec, +1);
        const double inv = 1.0 / (double)dec.size();
        size_t best = 0;
        double bv = -1.0;
        if (spec_out) spec_out->assign(dec.size(), 0.0);
        pw.resize(dec.size());
        for (size_t i = 0; i < dec.size(); i++) {
            const double m = std::abs(dec[i]) * inv;
            if (spec_out) (*spec_out)[i] = m;
            const double p2 = m * m;
            pw[i] = p2;
            if (p2 > bv) { bv = p2; best = i; }
        }
        /* every symbol's, for the no-CRC check below; O(K) beside the FFT */
        gaps.push_back(peak_gap(pw, best));
        ok = true;
        return (long)best - 1; /* stored as pos - 1 */
    };

    long ofs = (long)(nrise + 12.25 * (double)N);

    int length = 0, cr = p.cr;
    bool has_crc = p.has_crc;
    std::vector<uint8_t> hdr_payload_bits;

    /* The first block is 8 symbols at CR 4/8 in BOTH modes; with an implicit
     * header it simply carries no header bits. Skipping it for implicit
     * frames would make them 3 symbols shorter than everyone else's. */
    {
        R.k_hdr.resize(8);
        for (int i = 0; i < 8; i++) {
            bool ok = false;
            const long k = demod(ofs, 0, ok);  /* guarded at entry, above */
            if (!ok) { R.truncated = true; return R; }
            R.k_hdr[(size_t)i] = (int32_t)k;   /* signed: pos - 1, see rx_result */
            ofs += N;
        }
        const header_fields h =
            header_decode(SF, R.k_hdr, fec_correct,
                          implicit_header ? 0 : 5);
        if (implicit_header) {
            R.hdr_ok = true;                   /* no FCS exists to check */
            length = ih_length + (p.pseudo_header ? 4 : 0);
            cr = ih_cr;
            has_crc = ih_has_crc;
        } else {
            R.hdr_ok = h.fcs_ok;
            if (!h.fcs_ok) return R;
            length = h.length;
            cr = h.cr;
            has_crc = h.has_crc;
        }
        hdr_payload_bits = h.payload_bits;
        R.cr = cr;
        R.has_crc = has_crc;
    }

    const int n_bits_hdr = (int)hdr_payload_bits.size();
    const int n_sym_payload =
        payload_n_sym(SF, length, has_crc, cr, n_bits_hdr, de);
    /* ZERO is legal and must decode: the whole payload fitted in the bits
     * the first interleaver block already carries, which happens for short
     * frames (any message up to 3 bytes at SF9 without the CRC, and 0-1 bytes
     * at SF12 with the pseudo-header). The transmitter emits such frames. */
    if (n_sym_payload < 0) return R;

    /* Where this frame's last symbol ends: known from here on, so a
     * truncation below can say how long the frame still needs. */
    const long frame_end = ofs + (long)n_sym_payload * N;
    R.k_payload.resize((size_t)n_sym_payload);
    std::vector<std::vector<double> > mag_payload;
    if (soft_decision) mag_payload.resize((size_t)n_sym_payload);
    for (int i = 0; i < n_sym_payload; i++) {
        bool ok = false;
        spec_out = soft_decision ? &mag_payload[(size_t)i] : 0;
        /* Phase 0, same as the header loop above -- see trap 3. */
        const long k = demod(ofs, 0, ok);
        spec_out = 0;
        if (!ok) { R.truncated = true; R.complete_at = frame_end; return R; }
        R.k_payload[(size_t)i] = (int32_t)k;   /* signed: pos - 1, see rx_result */
        if (fracs_out) {
            /* The peak's offset from its bin: |X| at the peak and its larger
             * neighbour, scaled 1/K as the inverse DFT above is. `dec` still
             * holds this symbol's spectrum. */
            const long K = (long)dec.size();
            const long pos = k + 1;
            const double inv = 1.0 / (double)K;
            const double a = std::abs(dec[(size_t)((pos - 1 + K) % K)]) * inv;
            const double b = std::abs(dec[(size_t)pos]) * inv;
            const double c = std::abs(dec[(size_t)((pos + 1) % K)]) * inv;
            double f;
            if (c >= a)
                f = (b + c) > 0.0 ? c / (b + c) : 0.0;
            else
                f = (b + a) > 0.0 ? -a / (b + a) : 0.0;
            fracs_out->push_back(f);
        }
        ofs += N;
    }

    /* THE NO-CRC CHECK: a frame without a CRC is crc_ok only if the mean of
     * its symbols' peak gaps, each capped at NOCRC_GAP_CAP, reaches
     * nocrc_min_gap(SF). Summed in index order; a nan propagates and
     * passes. */
    auto nocrc_check = [&]() -> bool {
        double acc = 0.0;
        for (size_t i = 0; i < gaps.size(); i++)
            acc += std::min(gaps[i], NOCRC_GAP_CAP);
        R.gap_score = gaps.empty() ? std::numeric_limits<double>::infinity()
                                   : acc / (double)gaps.size();
        return !(R.gap_score < nocrc_min_gap(SF));
    };

    /* ---- payload: de-gray, de-interleave, FEC, de-whiten, parse ---- */
    const int ppm = SF - 2 * de;
    const int n_sym_blk = 4 + cr;
    const int n_bits_blk = ppm * 4;
    const int n_blk = n_sym_payload / n_sym_blk;
    const long mod = 1L << ppm;
    const long step = 1L << (2 * de);

    std::vector<uint32_t> gray, degray;
    gray_lut(ppm, gray, degray);

    std::vector<uint8_t> PAYLOAD;
    PAYLOAD.reserve((size_t)(n_bits_hdr + n_blk * n_bits_blk));
    PAYLOAD = hdr_payload_bits;
    PAYLOAD.resize((size_t)(n_bits_hdr + n_blk * n_bits_blk), 0);
    size_t ofs_bits = (size_t)n_bits_hdr;

    /* Soft path: LLRs for every payload symbol, computed once. */
    std::vector<std::vector<double> > llr_all;
    if (soft_decision)
        symbol_llr(mag_payload, SF, de, llr_all);

    for (int blk = 0; blk < n_blk; blk++) {
        if (soft_decision) {
            /* LLRs go through the identical permutation as the bits would. */
            std::vector<double> lblk((size_t)ppm * n_sym_blk, 0.0);
            for (int sym = 0; sym < n_sym_blk; sym++)
                for (int b = 0; b < ppm; b++)
                    lblk[(size_t)sym * ppm + (size_t)b] =
                        llr_all[(size_t)(blk * n_sym_blk + sym)][(size_t)b];
            std::vector<std::vector<double> > Lc;
            deinterleave_block_t(lblk, ppm, n_sym_blk, Lc);
            std::vector<std::vector<uint8_t> > data;
            hamming_soft_decode(Lc, cr, data);
            for (int k = 0; k < ppm; k++) {
                for (int j = 0; j < 4; j++)
                    PAYLOAD[ofs_bits + (size_t)j] = data[(size_t)k][(size_t)j];
                ofs_bits += 4;
            }
            continue;
        }

        std::vector<uint8_t> bits((size_t)ppm * n_sym_blk, 0);
        for (int sym = 0; sym < n_sym_blk; sym++) {
            const long k = (long)R.k_payload[(size_t)(blk * n_sym_blk + sym)];
            /* payload: (K - 2 - k) / 2^(2*DE)  -- note K-2, not K-1 */
            const long b = round_half_even((double)(K - 2 - k) / (double)step);
            bits_msb_first(degray[pymod(b, mod)], ppm, &bits[(size_t)sym * ppm]);
        }
        std::vector<std::vector<uint8_t> > C;
        deinterleave_block(bits, ppm, n_sym_blk, C);
        /* GUARDED BY fec_correct, as the HEADER path above is:
         * fec_correct=false asks for the original, parity-ignoring
         * receiver, on the payload as on the header. */
        if (fec_correct) {
            int a, b2;
            hamming_correct(C, cr, a, b2);
        }
        for (int k = 0; k < ppm; k++) {
            for (int j = 0; j < 4; j++)
                PAYLOAD[ofs_bits + (size_t)j] = C[k][j];
            ofs_bits += 4;
        }
    }

    /* de-whitening: the same LFSR as the transmitter */
    {
        uint8_t W[8];
        std::memset(W, 1, sizeof(W));
        const size_t nbytes = PAYLOAD.size() / 8;
        for (size_t k = 0; k < nbytes; k++) {
            for (int i = 0; i < 8; i++)
                PAYLOAD[k * 8 + (size_t)i] ^= W[i];
            int fb = 0;
            for (int i = 0; i < 8; i++)
                fb ^= (W[i] & WHITEN_FB[i]);
            for (int i = 7; i > 0; i--) W[i] = W[i - 1];
            W[0] = (uint8_t)(fb & 1);
        }
    }

    if (!p.pseudo_header) {
        /* Standard framing: the PHY header's LENGTH is the message length and
         * every payload byte is message. The CRC is unchanged -- it already
         * covers the whole payload, which is the standard definition. */
        const int len2 = length;
        const size_t need = has_crc ? (size_t)(len2 + 2) * 8 : (size_t)len2 * 8;
        /* has_crc with len2 == 0 reaches crc16(), which throws for a
           zero-length input -- and this throw leaves work() through GNU
           Radio's thread body, which does not catch std::exception, so the
           whole flowgraph dies on a header that passed its 5-bit FCS by
           chance. A false lock must look like a failed lock. */
        if (len2 < 0 || (has_crc && len2 < 1) || need > PAYLOAD.size()) {
            R.consumed = ofs; return R;
        }
        R.length = len2;
        R.dst = R.src = R.seqn = -1;   /* not present on air in this mode */
        R.payload.assign((size_t)len2, 0);
        for (int i = 0; i < len2; i++)
            R.payload[(size_t)i] =
                (uint8_t)bits_to_uint_lsb_first(&PAYLOAD[(size_t)(8 * i)], 8);
        if (!has_crc) {
            R.crc_ok = nocrc_check();
        } else {
            std::vector<uint8_t> pre(PAYLOAD.begin(),
                                     PAYLOAD.begin() + (size_t)len2 * 8);
            std::vector<uint8_t> want;
            crc16(pre, want);
            bool same = true;
            for (int i = 0; i < 16; i++)
                if (PAYLOAD[(size_t)(8 * len2 + i)] != want[(size_t)i])
                    same = false;
            R.crc_ok = same;
        }
        R.consumed = ofs;
        return R;
    }

    if (PAYLOAD.size() < 32) { R.consumed = ofs; return R; }  /* pseudo-header does not fit */

    R.dst = (int)bits_to_uint_lsb_first(&PAYLOAD[0], 8);
    R.src = (int)bits_to_uint_lsb_first(&PAYLOAD[8], 8);
    R.seqn = (int)bits_to_uint_lsb_first(&PAYLOAD[16], 8);
    int len2 = (int)bits_to_uint_lsb_first(&PAYLOAD[24], 8);
    if (len2 == 0) len2 = length;
    R.length = len2;

    const int msg_len = len2 - 4;
    const size_t need = has_crc ? (size_t)(len2 + 2) * 8 : (size_t)len2 * 8;
    if (need > PAYLOAD.size() || len2 < 4) { R.consumed = ofs; return R; }

    R.payload.assign((size_t)msg_len, 0);
    for (int i = 0; i < msg_len; i++)
        R.payload[(size_t)i] =
            (uint8_t)bits_to_uint_lsb_first(&PAYLOAD[(size_t)(8 * (4 + i))], 8);

    if (!has_crc) {
        R.crc_ok = nocrc_check();
    } else {
        std::vector<uint8_t> pre(PAYLOAD.begin(),
                                 PAYLOAD.begin() + (size_t)len2 * 8);
        std::vector<uint8_t> want;
        crc16(pre, want);
        bool same = true;
        for (int i = 0; i < 16; i++)
            if (PAYLOAD[(size_t)(8 * len2 + i)] != want[(size_t)i]) same = false;
        R.crc_ok = same;
    }
    R.consumed = ofs;   /* samples used past p_ofs, for the buffer walk */
    return R;
}

} /* namespace sdrlora */
} /* namespace gr */
