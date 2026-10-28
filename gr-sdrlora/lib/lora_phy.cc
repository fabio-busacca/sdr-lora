/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
/*
 * LoRa PHY transmit chain.
 *
 * Two things are deliberate and worth knowing before editing:
 *
 *  1. The bit arrays are LSB-first *within each byte*:
 *     PAYLOAD[base + i] = bit i of the value. Getting this backwards is the
 *     single most likely mistake.
 *
 *  2. The chirp phase is accumulated symbol by symbol in a fixed order and
 *     with fixed operations, so the output is reproducible to the bit. In
 *     particular phi_fin is ((phi + a) + b), not phi + (a + b) --
 *     pre-summing the two terms shifts the result by an ULP and changes the
 *     emitted float32.
 */
#include "sdrlora/lora_phy.h"
#include "sdrlora/lora_tables.h"

#include <cmath>
#include <cstring>
#include <ios>
#include <sstream>
#include <stdexcept>

namespace gr {
namespace sdrlora {

static const double kPi = 3.14159265358979311600;

void num2binary(uint32_t value, int nbits, std::vector<uint8_t>& out)
{
    out.assign(nbits, 0);
    for (int i = 0; i < nbits; i++)
        out[i] = (uint8_t)((value >> (nbits - 1 - i)) & 1u); /* MSB first */
}

void gray_lut(int ppm, std::vector<uint32_t>& gray, std::vector<uint32_t>& degray)
{
    const uint32_t n = 1u << ppm;
    gray.assign(n, 0);
    degray.assign(n, 0);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t g = i ^ (i >> 1); /* standard binary -> Gray */
        degray[i] = g;
        gray[g] = i;                     /* inverse permutation */
    }
}

void crc16(const std::vector<uint8_t>& bits, std::vector<uint8_t>& out)
{
    const int length = (int)bits.size() / 8;
    if (length < 1 || length > 255)
        throw std::runtime_error("crc16: length outside 1..255");

    std::vector<uint8_t> state;
    num2binary(length < 5 ? CRC16_STATE_SHORT[length - 1]
                          : CRC16_STATE[length - 5], 16, state);

    int pos = 0, pos4 = 4, pos11 = 11;
    for (int j = 0; j < length; j++) {
        const int base = j * 8 + 7;
        for (int k = 0; k < 8; k++) {
            const uint8_t add = state[pos];
            state[pos] = bits[base - k];
            if (add) {
                state[pos4] ^= 1;
                state[pos11] ^= 1;
                state[pos] ^= 1;
            }
            pos = (pos + 1) % 16;
            pos4 = (pos4 + 1) % 16;
            pos11 = (pos11 + 1) % 16;
        }
    }
    out.assign(16, 0);
    for (int i = 0; i < 16; i++)
        out[i] = state[(pos + 15 - i) % 16];
}

void header_init(int SF, bool implicit_header, int& n_sym_hdr, int& n_bits_hdr)
{
    const int cr_hdr0 = 4, de_hdr0 = 1;
    if (implicit_header) {
        /* The first interleaver block is 8 symbols at CR 4/8 whether or not a
         * header occupies it. With an implicit header all 4*(SF-2) of its
         * data bits are payload. Returning (0, 0) here -- no first block at
         * all -- would be self-consistent but matches nothing else on air:
         * implicit frames would be 3 symbols shorter than gr-lora_sdr's and
         * the silicon's at every SF. */
        n_sym_hdr = 4 + cr_hdr0;
        n_bits_hdr = 4 * (SF - 2 * de_hdr0);
    } else {
        const int cr_hdr = 4, de_hdr = 1;
        n_sym_hdr = 4 + cr_hdr;
        const int intlv = (SF - 2 * de_hdr) * (4 + cr_hdr);
        n_bits_hdr = intlv * 4 / (4 + cr_hdr) - 20;
    }
}

/* PAYLOAD[base + i] = bit i of value (LSB first within the byte). */
static void put_byte_lsb_first(std::vector<uint8_t>& bits, int base, uint32_t value)
{
    for (int i = 0; i < 8; i++)
        bits[base + i] = (uint8_t)((value >> i) & 1u);
}

void payload_init(const tx_params& p, const std::vector<uint8_t>& msg,
                  std::vector<uint8_t>& payload_bits, int& n_sym_payload)
{
    /* on-air LENGTH counts the pseudo-header when there is one */
    const int prefix = p.pseudo_header ? 4 : 0;
    const int length = prefix + (int)msg.size();
    /* A LoRa CRC needs at least two payload bytes to exist. The checksum is a
     * 16-bit register clocked over the payload, so the last TWO bytes are
     * simply shifted in and appear in the result as a plain XOR: at
     * length 2 the CRC is exactly those two bytes XORed with a constant,
     * and at length 3 the last two still only XOR in while byte 0 goes through
     * the register. With one byte half the register is still the initial
     * state, and nothing defines what that should be.
     *
     * gr-lora_sdr says the same thing ("CRC not supported for payload smaller
     * than 2 bytes") and then sets the header's CRC flag anyway and appends no
     * checksum, which its own receiver rejects. Refuse instead of emitting a
     * frame whose checksum means nothing. Note this is the ON-AIR length: with
     * the pseudo-header a one-byte message is five bytes and well defined. */
    if (p.has_crc && length < CRC_MIN_PAYLOAD)
        throw std::runtime_error(
            "sdrlora: a LoRa CRC is not defined for a payload shorter than "
            "two bytes -- the checksum is a 16-bit register clocked over the "
            "payload and needs two bytes to fill. Send it with has_crc=false, "
            "or with pseudo_header=true. gr-lora_sdr refuses the same case.");
    const int de = ldro_enabled(p.SF, p.bw, p.ldro);
    const int ppm = p.SF - 2 * de;
    const int n_bits_blk = ppm * 4;
    /* payload_init is public and reachable without going through
     * check_rx_params, exactly as payload_symbols is -- and payload_symbols,
     * below, carries the same guard. SF 0, or SF 2 with LDRO, gives
     * ppm = 0, and the ceil() division below would then be inf, which the
     * cast to int makes undefined behaviour before a vector is sized off
     * it. */
    if (n_bits_blk <= 0)
        throw std::invalid_argument(
            "sdrlora: SF - 2*ldro must be positive -- there are no bits in "
            "an interleaver block at this spreading factor");
    const int n_bits_tot = 8 * length + 16 * (p.has_crc ? 1 : 0);

    int n_sym_hdr, n_bits_hdr;
    header_init(p.SF, p.implicit_header, n_sym_hdr, n_bits_hdr);

    /* Clamped. With an empty message, no CRC and no pseudo-header,
       n_bits_tot is 0 and equals n_bits_hdr for a DE=1 block, so this ceil
       is exactly -1 -- reachable at the DEFAULT ldro=2 at SF11 and SF12.
       Zero is the legitimate answer -- the payload fits inside the header
       block -- and negative has no meaning. The receiver applies the same
       clamp (lora_rx.cc, payload_n_sym). An if rather than std::max:
       <algorithm> is not included here. */
    int n_blk_tot =
        (int)std::ceil((double)(n_bits_tot - n_bits_hdr) / (double)n_bits_blk);
    if (n_blk_tot < 0) n_blk_tot = 0;
    n_sym_payload = n_blk_tot * (4 + p.cr);

    payload_bits.assign((size_t)(n_bits_hdr + n_blk_tot * n_bits_blk), 0);

    if (p.pseudo_header) {
        put_byte_lsb_first(payload_bits, 0, (uint32_t)p.dst);
        put_byte_lsb_first(payload_bits, 8, (uint32_t)p.src);
        put_byte_lsb_first(payload_bits, 16, (uint32_t)p.seqn);
        put_byte_lsb_first(payload_bits, 24, (uint32_t)length);
    }
    for (size_t k = 0; k < msg.size(); k++)
        put_byte_lsb_first(payload_bits, 8 * (prefix + (int)k), msg[k]);

    if (p.has_crc) {
        std::vector<uint8_t> pre(payload_bits.begin(),
                                 payload_bits.begin() + 8 * length);
        std::vector<uint8_t> crc;
        crc16(pre, crc);
        for (int i = 0; i < 16; i++)
            payload_bits[8 * length + i] = crc[i];
    }

    /* Whitening: XOR each byte with the LFSR state, then step the LFSR. */
    uint8_t W[8];
    std::memset(W, 1, sizeof(W));
    const int nbytes = (int)(payload_bits.size() / 8);
    for (int k = 0; k < nbytes; k++) {
        for (int i = 0; i < 8; i++)
            payload_bits[k * 8 + i] ^= W[i];
        int fb = 0;
        for (int i = 0; i < 8; i++)
            fb ^= (W[i] & WHITEN_FB[i]);
        for (int i = 7; i > 0; i--)
            W[i] = W[i - 1];
        W[0] = (uint8_t)(fb & 1);
    }
}

/* Diagonal interleave + Gray map for one block of (4+cr) symbols.
 * C is [ppm][4+cr] with the data bits first: a row flip, then the
 * S[jj][(ii+jj) % ppm] = C[ii][jj] assignment. */
static void interleave_and_map(const std::vector<std::vector<uint8_t> >& C,
                               int ppm, int n_sym_blk, int SF, int de,
                               const std::vector<uint32_t>& gray,
                               std::vector<uint32_t>& k_out)
{
    /* row flip */
    std::vector<std::vector<uint8_t> > F(ppm, std::vector<uint8_t>(n_sym_blk, 0));
    for (int i = 0; i < ppm; i++)
        F[i] = C[ppm - 1 - i];

    /* S[jj][(ii+jj) % ppm] = F[ii][jj] */
    std::vector<std::vector<uint8_t> > S(n_sym_blk, std::vector<uint8_t>(ppm, 0));
    for (int ii = 0; ii < ppm; ii++)
        for (int jj = 0; jj < n_sym_blk; jj++)
            S[jj][(ii + jj) % ppm] = F[ii][jj];

    const uint32_t K = 1u << SF;
    const uint32_t step = 1u << (2 * de);
    for (int sym = 0; sym < n_sym_blk; sym++) {
        uint32_t v = 0;
        for (int b = 0; b < ppm; b++)      /* MSB-first within the symbol */
            v = (v << 1) | S[sym][b];
        k_out.push_back(K - 1u - step * gray[v]);
    }
}

void header_symbols(const tx_params& p, int length,
                    const std::vector<uint8_t>& payload_bits,
                    std::vector<uint32_t>& k_hdr, int& payload_ofs)
{
    k_hdr.clear();
    /* hdr_rows: how many of the ppm codewords carry header bits. 5 for an
     * explicit header (20 bits = five 4-bit codewords), 0 for implicit, where
     * the block still exists and is entirely payload. */
    const int hdr_rows = p.implicit_header ? 0 : 5;
    const int cr_hdr = 4, de_hdr = 1;
    const int ppm = p.SF - 2 * de_hdr;
    const int n_sym_hdr = 4 + cr_hdr;

    std::vector<uint32_t> gray, degray;
    gray_lut(ppm, gray, degray);

    /* 20-bit header: length nibbles swapped, CRC flag, CR, then 8 FCS bits. */
    std::vector<uint8_t> len_bits, cr_bits;
    num2binary((uint32_t)length, 8, len_bits);
    num2binary((uint32_t)p.cr, 3, cr_bits);

    std::vector<uint8_t> hdr(20, 0);
    for (int i = 0; i < 4; i++) hdr[i] = len_bits[3 - i];
    for (int i = 0; i < 4; i++) hdr[4 + i] = len_bits[7 - i];
    hdr[8] = p.has_crc ? 1 : 0;
    for (int i = 0; i < 3; i++) hdr[9 + i] = cr_bits[2 - i];

    /* FCS over the first 12 header bits, taken in the order below. */
    int idx[12];
    for (int i = 0; i < 4; i++) idx[i] = 3 - i;
    for (int i = 0; i < 4; i++) idx[4 + i] = 7 - i;
    for (int i = 0; i < 4; i++) idx[8 + i] = 11 - i;

    uint8_t chk[5] = {0, 0, 0, 0, 0};
    for (int c = 0; c < 5; c++) {
        int acc = 0;
        for (int r = 0; r < 12; r++)
            acc ^= (hdr[idx[r]] & HEADER_FCS[r][c]);
        chk[c] = (uint8_t)(acc & 1);
    }
    hdr[12] = chk[0];
    for (int i = 0; i < 4; i++) hdr[16 + i] = chk[4 - i];

    /* Hamming(8,4) over 5 header rows, then payload bits for the remainder. */
    std::vector<std::vector<uint8_t> > C(ppm, std::vector<uint8_t>(4 + cr_hdr, 0));
    payload_ofs = 0;
    for (int k = 0; k < ppm; k++) {
        for (int i = 0; i < 4; i++)
            C[k][i] = (k < hdr_rows) ? hdr[k * 4 + i]
                                     : payload_bits[payload_ofs + i];
        if (k >= hdr_rows) payload_ofs += 4;
        for (int c = 0; c < cr_hdr; c++) {
            int acc = 0;
            for (int r = 0; r < 4; r++)
                acc ^= (C[k][r] & HAMMING_P4[r][c]);
            C[k][4 + c] = (uint8_t)(acc & 1);
        }
    }
    interleave_and_map(C, ppm, n_sym_hdr, p.SF, de_hdr, gray, k_hdr);
}

void payload_symbols(const tx_params& p, int n_sym_payload,
                     const std::vector<uint8_t>& payload_bits, int payload_ofs,
                     std::vector<uint32_t>& k_payload)
{
    k_payload.clear();
    const int de = ldro_enabled(p.SF, p.bw, p.ldro);
    const int ppm = p.SF - 2 * de;
    const int n_sym_blk = 4 + p.cr;
    const int n_blk_tot = n_sym_payload / n_sym_blk;

    std::vector<uint32_t> gray, degray;
    gray_lut(ppm, gray, degray);

    const uint8_t* P[4] = {&HAMMING_P1[0][0], &HAMMING_P2[0][0],
                           &HAMMING_P3[0][0], &HAMMING_P4[0][0]};
    /* Checked, because payload_symbols() is public and reachable without
       going through check_tx_params(): cr = 0 would read BEFORE a
       four-element table of pointers and cr = 5 past the end of it. */
    if (p.cr < 1 || p.cr > 4)
        throw std::invalid_argument(
            "sdrlora: cr must be 1..4 (coding rate 4/(4+cr))");
    const uint8_t* Pm = P[p.cr - 1];

    int ofs = payload_ofs;
    for (int blk = 0; blk < n_blk_tot; blk++) {
        std::vector<std::vector<uint8_t> > C(ppm, std::vector<uint8_t>(n_sym_blk, 0));
        for (int k = 0; k < ppm; k++) {
            for (int i = 0; i < 4; i++)
                C[k][i] = payload_bits[ofs + i];
            ofs += 4;
            for (int c = 0; c < p.cr; c++) {
                int acc = 0;
                for (int r = 0; r < 4; r++)
                    acc ^= (C[k][r] & Pm[r * p.cr + c]);
                C[k][4 + c] = (uint8_t)(acc & 1);
            }
        }
        interleave_and_map(C, ppm, n_sym_blk, p.SF, de, gray, k_payload);
    }
}

void encode_symbols(const tx_params& p, const std::vector<uint8_t>& msg,
                    std::vector<uint32_t>& k_hdr, std::vector<uint32_t>& k_payload)
{
    /* Public, and bound as phy_encode_symbols, so it validates its own
     * input: an unchecked SF of 31 would reach gray_lut(29) and ask the
     * allocator for 4 GiB -- the OOM check_tx_params() exists to prevent.
     * encode() validates too; the cost is a few microseconds. */
    check_tx_params(p);
    /* The on-air LENGTH field is one byte. Above 255 it would wrap, and a
     * 300-byte payload would come back as a 44-byte packet with a passing
     * header FCS -- silently, whether or not the CRC is on. */
    {
        const int on_air = (p.pseudo_header ? 4 : 0) + (int)msg.size();
        if (on_air > 255)
            throw std::invalid_argument(
                "sdrlora: the on-air payload length is one byte, so "
                "pseudo_header prefix + payload must be <= 255 bytes");
    }

    std::vector<uint8_t> payload_bits;
    int n_sym_payload = 0;
    payload_init(p, msg, payload_bits, n_sym_payload);
    /* Here, not in modulate(): before any of the per-symbol work, and on
     * the one path every encoder takes. */
    check_frame_size(p, (long)n_sym_payload);

    int payload_ofs = 0;
    header_symbols(p, (p.pseudo_header ? 4 : 0) + (int)msg.size(),
                   payload_bits, k_hdr, payload_ofs);
    payload_symbols(p, n_sym_payload, payload_bits, payload_ofs, k_payload);
}

/* ------------------------------------------------------------- modulation */

/* One linear chirp segment. The returned phase is ((phi_ini + a) + b), and
 * the two terms are kept separate on purpose. */
static double chirp_seg(double f_ini, int n, double ts, double df,
                        double phi_ini, std::vector<std::complex<double> >& out,
                        double t0_frac = 0.0)
{
    const double two_pi = 2.0 * kPi;
    for (int i = 0; i < n; i++) {
        /* (t0_frac + i) * ts. The END phase below deliberately uses n*ts
         * and not (t0_frac + n)*ts, because the phase advances by one whole
         * segment regardless of where within a sample the grid sits. */
        const double t = (t0_frac + (double)i) * ts;
        const double ph = phi_ini + two_pi * f_ini * t + kPi * df * (t * t);
        out.push_back(std::complex<double>(std::cos(ph), std::sin(ph)));
    }
    const double T = (double)n * ts;
    const double a = two_pi * f_ini * T;
    const double b = kPi * df * (T * T);
    return (phi_ini + a) + b;
}

/* One LoRa symbol: two chirp segments, k*OSF then (K-k)*OSF samples. */
static double lora_symbol(uint32_t k, double bw, int K, int osf, double phi,
                          std::vector<std::complex<double> >& out,
                          double t0_frac = 0.0)
{
    const double fs = bw * (double)osf;
    const double ts = 1.0 / fs;
    const int N = K * osf;
    const double T = (double)N * ts;
    const double df = bw / T;

    const int n1 = (int)k * osf;
    const double f1 = bw * (0.5 - (double)k / (double)K);
    const double f2 = -bw / 2.0;

    phi = chirp_seg(f1, n1, ts, df, phi, out, t0_frac);
    phi = chirp_seg(f2, N - n1, ts, df, phi, out, t0_frac);
    return phi;
}

void modulate(const tx_params& p, const std::vector<uint32_t>& k_hdr,
              const std::vector<uint32_t>& k_payload,
              std::vector<std::complex<float> >& out)
{
    check_tx_params(p);                 /* public entry point, see above */
    /* AND THE FRAME-SIZE FLOOR, payload-independent, as encoder_impl's
     * constructor applies it: modulate() is public (phy_modulate), and SF12
     * with a 65535-symbol preamble would otherwise go straight to a 17 GB
     * allocation -- std::bad_alloc, or the OOM killer -- where encode()
     * names the size. encode()'s own estimate counts the payload too, so it
     * is never below this and this refuses nothing encode() accepts. */
    if (frame_samples_estimate(p, 0) > (double)MAX_FRAME_SAMPLES)
        throw std::invalid_argument(
            "sdrlora: the requested frame is past the MAX_FRAME_SAMPLES "
            "limit before any payload -- the preamble and lead-in alone");
    const int K = 1 << p.SF;
    /* Symbols are FFT bin indices, so they have to be inside the alphabet.
     * lora_symbol() makes the first chirp segment k*osf samples and the
     * second N - k*osf; past K the second length is negative, chirp_seg's
     * loop simply does not run, and the result is a SHORT WAVEFORM with a
     * wrong end phase -- no error, no crash, just a frame nothing can decode.
     * k is unsigned, so a value above INT_MAX also converts to a negative
     * int and makes the FIRST segment the short one. */
    for (int pass = 0; pass < 2; pass++) {
        const std::vector<uint32_t>& ks = pass ? k_payload : k_hdr;
        for (size_t i = 0; i < ks.size(); i++)
            if (ks[i] >= (uint32_t)K)
                throw std::invalid_argument(
                    "sdrlora: modulate: every symbol must be < 2^SF -- a "
                    "symbol is an FFT bin index, and outside the alphabet "
                    "the two chirp segments cannot both have a length");
    }
    const int osf = (int)(p.fs / p.bw + 0.5);
    const int N = K * osf;
    const double ts = 1.0 / p.fs;
    const double bw = p.bw;
    /* bw*osf, not p.fs -- and they are the same number, which is worth
     * saying because the receiver writes it the other way round:
     * lora_rx.cc and lora_sync.cc both use ceil(p.trise * p.fs). The two
     * EXPRESSIONS agree only because
     * check_tx_params rejects a non-integer fs/bw to within 1e-9 relative,
     * so osf == fs/bw exactly for anything that gets this far. Without
     * that check they would diverge -- BW 41.7 kHz at fs 500 kHz gives
     * osf 12, a transmit taper of 26 samples and a receive offset computed
     * from 25 -- so this relies on the check. */
    const int nrise = (int)std::ceil(p.trise * bw * (double)osf);

    /* Preamble: n up-chirps at k=0, the sync word's two symbols, two full
     * down-chirps and a quarter down-chirp. */
    std::vector<std::complex<double> > pre;
    double phi = p.phi0;
    for (int i = 0; i < p.preamble_symbols; i++)
        (void)lora_symbol(0, bw, K, osf, p.phi0, pre, p.t0_frac);
    /* The preamble tiles one up-chirp, so every preamble up-chirp is the
     * same block -- built at phi0 and concatenated, which is why the starting
     * phase is p.phi0 and not the running one. Re-derive the phase held after
     * the tile. */
    {
        std::vector<std::complex<double> > tmp;
        phi = lora_symbol(0, bw, K, osf, p.phi0, tmp, p.t0_frac);
    }
    {
        /* Placed below K because the modulator's mapping is mirrored; K-8 and
         * K-16 are what the default 0x12 produces. */
        int s1 = 8, s2 = 16;
        if (!sync_word_symbols(p.sync_word, s1, s2))
            throw std::invalid_argument(
                "sdrlora: sync_word must be one byte with two non-zero "
                "nibbles, or two raw symbols in 1..4095");
        if (!sync_symbols_separable(s1, s2, p.SF))
            throw std::runtime_error(
                "sdrlora: sync symbols are within one bin of the preamble at "
                "this SF (bins are cyclic mod 2^SF and the sync check has a "
                "one-bin tolerance), so they cannot be told from more "
                "preamble");
        /* Fold to a real bin first. sync_symbols_separable() compares mod K
         * and guarantees 2 <= f <= K-2, so K - f is a valid symbol index;
         * without the fold a raw sync symbol above K would go negative. */
        const int f1 = ((s1 % K) + K) % K;
        const int f2 = ((s2 % K) + K) % K;
        phi = lora_symbol((uint32_t)(K - f1), bw, K, osf, phi, pre,
                          p.t0_frac);
        phi = lora_symbol((uint32_t)(K - f2), bw, K, osf, phi, pre,
                          p.t0_frac);
    }

    const double T = (double)N * ts;
    const double df_dn = -bw / T;
    /* The down-chirp d0 is built once and concatenated TWICE, so the second
     * down-chirp is the same block with the same starting phase -- it does not
     * continue the chain. d0_4 then resumes from the phase after a single d0. */
    std::vector<std::complex<double> > d0blk;
    const double phi_after_d0 = chirp_seg(bw / 2.0, N, ts, df_dn, phi, d0blk,
                                         p.t0_frac);
    pre.insert(pre.end(), d0blk.begin(), d0blk.end());
    pre.insert(pre.end(), d0blk.begin(), d0blk.end());
    phi = chirp_seg(bw / 2.0, N / 4, ts, -(bw * bw) / (double)K, phi_after_d0,
                    pre, p.t0_frac);

    const size_t n_sym = k_hdr.size() + k_payload.size();
    std::vector<std::complex<double> > body;
    body.reserve(n_sym * (size_t)N);
    for (size_t i = 0; i < k_hdr.size(); i++)
        phi = lora_symbol(k_hdr[i], bw, K, osf, phi, body, p.t0_frac);
    for (size_t i = 0; i < k_payload.size(); i++)
        phi = lora_symbol(k_payload[i], bw, K, osf, phi, body, p.t0_frac);

    /* Assemble: rising edge, preamble, symbols, falling edge. */
    std::vector<std::complex<double> > s;
    s.resize((size_t)nrise + pre.size() + body.size() + (size_t)nrise,
             std::complex<double>(0.0, 0.0));
    for (int i = 0; i < nrise; i++) {
        const double w = std::sin(kPi / 2.0 * (double)(i + 1) / (double)nrise);
        s[i] = pre[(size_t)(N - 1 - nrise + 1 + i)] * (w * w);
    }
    /* .data() and a guard, not &v[0]: a zero-symbol payload is a legal,
       block-accepted frame (pseudo_header=false, has_crc=false, 0 bytes at
       SF7), and the preamble gets the same guard. &v[0] on an empty vector
       is out of range before memcpy ever sees the count -- benign in a
       plain Release build, an abort under _GLIBCXX_ASSERTIONS or UBSan. */
    if (!pre.empty())
        std::memcpy(&s[nrise], pre.data(),
                    pre.size() * sizeof(std::complex<double>));
    if (!body.empty())
        std::memcpy(&s[(size_t)nrise + pre.size()], body.data(),
                    body.size() * sizeof(std::complex<double>));
    /* A NO-OP: `tail` is already past the last symbol, so this multiplies
       the trailing zero padding by cos^2 and the frame ends at full
       amplitude with a hard step to zero. Only the rising edge above is a
       real taper. Kept because it is the published waveform; its
       out-of-band cost against a true taper is a fraction of a dB. */
    const size_t tail = (size_t)nrise + pre.size() + body.size();
    for (int i = 0; i < nrise; i++) {
        const double w = std::cos(kPi / 2.0 * (double)i / (double)nrise);
        s[tail + i] = s[tail + i] * (w * w);
    }

    /* CFO term and the zero lead-in. A negative
     * lead_in means AUTO and is resolved here rather than refused -- see
     * resolved_lead_in(). Resolving it in the core API, not only in the GNU
     * Radio block, is what lets phy_encode() emit frames phy_decode() can
     * read at SF10-SF12 with a short preamble -- the case default_lead_in()
     * exists for. */
    const int p_ofs = resolved_lead_in(p);
    const double cfo = 2.0 * p.cfo_ppm * 1e-6 * p.f0;
    /* EXACT, over this frame, for a caller of the public modulate() --
     * encode() refuses on check_frame_size's estimate first. Without it a
     * finite but huge cfo_ppm (1.7e304 at SF7 / 1 MS/s) would come back as
     * a frame of NaNs with no error. */
    if (!std::isfinite(2.0 * kPi * cfo * ((double)s.size() * ts)))
        throw std::invalid_argument(
            "sdrlora: the carrier phase overflows: 2*pi*cfo*t over this "
            "frame is past the largest double, so every modulated sample "
            "would be NaN");
    out.assign((size_t)p_ofs + s.size(), std::complex<float>(0.0f, 0.0f));
    for (size_t i = 0; i < s.size(); i++) {
        const double t = (double)i * ts;
        const double ph = 2.0 * kPi * cfo * t;
        /* Inverting I/Q is conjugation, and it applies to the modulation
         * only -- the CFO below is the transmitter's crystal and does not
         * invert with it. See tx_params::invert_iq. */
        const std::complex<double> m = p.invert_iq ? std::conj(s[i]) : s[i];
        const std::complex<double> v =
            m * std::complex<double>(std::cos(ph), std::sin(ph));
        /* Accumulate onto zero, which normalises -0.0. */
        out[(size_t)p_ofs + i] =
            std::complex<float>((float)(0.0 + v.real()), (float)(0.0 + v.imag()));
    }
    /* Scaled last, and skipped entirely at 1.0, so the default path is
     * bit-identical to the unscaled one. A COMPLEX multiply by
     * (amplitude, 0) rather than a scalar one, because the sign of a zero
     * component follows from it. */
    /* Hops BEFORE the level -- both are multiplicative, so this only fixes
     * the rounding.
     *
     * ANCHORED AT THE HEADER, not at p_ofs. The receiver's frame-start
     * estimate is computed as if the preamble were 8 symbols long, so it is
     * off the true frame start by (preamble_symbols - 8)*N and a
     * frame-anchored schedule would be de-hopped against the wrong phase at
     * every other preamble length. p_ofs + nrise + (preamble_symbols + 4.25)*N is
     * the header start, and the receiver reaches the identical sample as
     * p_ofs_est + nrise + 12.25*N whatever the preamble was. Everything
     * before it -- lead-in, preamble, sync words, down-chirps -- is left
     * untouched by apply_hops and goes out on hop_freqs[0], which is
     * required to be 0 and is the only channel a receiver could acquire on. */
    {
        const long _N = (long)(1L << p.SF) * (long)osf;
        const long _anchor = (long)p_ofs + (long)nrise
            + (long)(((double)p.preamble_symbols + 4.25) * (double)_N);
        apply_hops(out, _anchor, p.hop_period, p.hop_freqs, _N, p.fs, false);
    }
    if (p.amplitude != 1.0) {
        const std::complex<float> g((float)p.amplitude, 0.0f);
        for (size_t i = 0; i < out.size(); i++)
            out[i] = out[i] * g;
    }
}

void apply_hops(std::vector<std::complex<float> >& s, long frame_ofs,
                int hop_period, const std::vector<double>& hop_freqs,
                long n_sym, double fs, bool invert, double sfo,
                double ref)
{
    if (hop_period <= 0 || hop_freqs.empty() || n_sym <= 0) return;
    const long step = (long)hop_period * n_sym;
    const double sign = invert ? -1.0 : 1.0;
    const long n = (long)s.size();
    long pos = frame_ofs;
    long k = 0;
    /* Whole blocks entirely before the buffer are skipped, keeping the
       channel index in step -- the partial block that straddles 0 still
       gets the phase its own start implies.

       A loop-trip optimisation, NOT a bounds guard: `lo` below clamps the
       slice whether or not this runs. */
    if (pos < 0) {
        const long skip = (-pos) / step;
        k += skip;
        pos += skip * step;
    }
    /* where nominal sample q of the schedule really falls: the rule of
     * lora_rx.cc's sym_start (nearbyint is round-half-even) */
    auto at = [&](long q) -> long {
        if (sfo == 0.0) return q;
        return q - (long)std::nearbyint(sfo * ((double)q - ref));
    };
    while (at(pos) < n) {
        const long start = at(pos);
        const long e1 = at(pos + step);
        const long end = (e1 < n) ? e1 : n;
        const long lo = (start > 0) ? start : 0;
        if (lo < end) {
            const double f = hop_freqs[(size_t)(k % (long)hop_freqs.size())];
            if (f != 0.0) {
                for (long i = lo; i < end; i++) {
                    /* t counts from the BLOCK start, not the slice start.
                       Computed in double and multiplied in double: a
                       complex128 rotation, with the product cast back to
                       complex64. */
                    const double t = (double)(i - start) / fs;
                    const double ph = sign * 2.0 * kPi * f * t;
                    const std::complex<double> v =
                        std::complex<double>((double)s[(size_t)i].real(),
                                             (double)s[(size_t)i].imag())
                        * std::complex<double>(std::cos(ph), std::sin(ph));
                    s[(size_t)i] = std::complex<float>((float)v.real(),
                                                       (float)v.imag());
                }
            }
        }
        pos += step;
        k++;
    }
}

double hop_cutoff_ratio(const std::vector<double>& hop_freqs, double bw)
{
    if (hop_freqs.empty()) return 1.15;
    /* BY NAME: bw = 0 would otherwise give
     * inf here (nan with an all-zero plan). Every caller in this library
     * has validated bw already. */
    if (!std::isfinite(bw) || !(bw > 0.0))
        throw std::invalid_argument(
            "sdrlora: hop_cutoff_ratio: bw must be finite and positive");
    double far = 0.0;
    for (size_t i = 0; i < hop_freqs.size(); i++) {
        /* AND FINITE CHANNELS: a NaN fails `a > far` and would be skipped
         * silently. Refused by name. */
        if (!std::isfinite(hop_freqs[i]))
            throw std::invalid_argument(
                "sdrlora: hop_cutoff_ratio: every hop frequency must be "
                "finite");
        const double a = hop_freqs[i] < 0.0 ? -hop_freqs[i] : hop_freqs[i];
        if (a > far) far = a;
    }
    return 1.0 + 2.0 * far / bw;
}

void check_effective_cutoff(double cutoff_ratio, int hop_period,
                            const std::vector<double>& hop_freqs, double bw)
{
    if (hop_period <= 0 || hop_freqs.empty() || !(bw > 0.0)) return;
    const double need = hop_cutoff_ratio(hop_freqs, bw);
    const double use = (need > cutoff_ratio) ? need : cutoff_ratio;
    if (!(use > 0.0) || use > 64.0) {
        double far = 0.0;
        for (size_t i = 0; i < hop_freqs.size(); i++) {
            const double a = std::fabs(hop_freqs[i]);
            if (a > far) far = a;
        }
        std::ostringstream os;
        os << "sdrlora: this hop plan needs a receive cutoff of " << use
           << " * bw/2 and the channel filter accepts at most 64 -- the "
              "filter runs BEFORE the de-hop, so a channel outside its "
              "passband is destroyed before the de-hop can bring it back. "
              "The reachable limit is max|hop_freqs| <= 31.5 * bw, which is "
           << (31.5 * bw) << " Hz here; this plan reaches " << far
           << " Hz. Narrow the plan, raise bw, or set channel_filter_rx "
              "false (which skips the widening entirely)";
        throw std::invalid_argument(os.str());
    }
}


/* check_rx_params (below): everything a RECEIVER needs to be true, which
 * is everything except the transmit-only fields (preamble_symbols,
 * lead_in, amplitude, t0_frac, src, dst, seqn). decode_buffer() calls it,
 * so the seven receive knobs -- trise, cutoff_ratio, numtaps,
 * sync_threshold, search_tail_symbols, min_osf, interp_target_osf -- are
 * bounded on the decode path too; numtaps = 1,
 * for one, would make lowpass_taps evaluate cos(2*pi*i/0) = NaN and the
 * whole channel filter output NaN. The shared checks run in the same order
 * from either entry point, so a parameter set with one bad field is
 * refused with the same message. */
void check_hop_plan(const tx_params& p)
{
    /* FACTORED out of check_rx_params so packet_rx can apply it too. That
     * entry point validates SF, ldro, bw/fs and trise inline because it is
     * public and does not call check_rx_params; an unchecked plan there --
     * a NaN channel, or hop_freqs[0] != 0 -- would blank or mis-rotate the
     * frame silently instead of being refused by name. */
    if (p.hop_period == 0) return;
    if (p.hop_period < HOP_PERIOD_MIN || p.hop_period > HOP_PERIOD_MAX)
        throw std::invalid_argument(
            "sdrlora: hop_period must be 0 (off) or 13..65535 symbols -- "
            "a deliberate floor: the schedule is anchored at the header, "
            "so nothing before it is rotated at any hop period, but a hop "
            "must not be shorter than the 12.25-symbol sync pattern");
    if (p.hop_freqs.empty())
        throw std::invalid_argument(
            "sdrlora: hop_period needs hop_freqs -- the channel list is "
            "not carried on the air, so both ends need the same one");
    if ((int)p.hop_freqs.size() > HOP_FREQS_MAX)
        throw std::invalid_argument(
            "sdrlora: hop_freqs holds at most 256 channels");
    if (p.hop_freqs[0] != 0.0)
        throw std::invalid_argument(
            "sdrlora: hop_freqs[0] must be 0 -- the list is relative to "
            "the channel the receiver starts on, and it acquires the "
            "preamble at baseband");
    const double edge = (p.fs - p.bw) / 2.0;
    for (size_t i = 0; i < p.hop_freqs.size(); i++) {
        const double v = p.hop_freqs[i];
        /* std::isfinite, not a magnitude test: 1e301 is finite, and is
         * left to the band check below. */
        if (!std::isfinite(v))
            throw std::invalid_argument(
                "sdrlora: hop_freqs must be finite");
        if (v > edge || v < -edge)
            throw std::invalid_argument(
                "sdrlora: a hop channel is outside the sampled band -- "
                "the receiver holds every channel at once and needs "
                "bw/2 either side of the outermost one");
    }
}

void check_rx_params(const tx_params& p)
{
    if (p.SF < 5 || p.SF > 12)
        throw std::invalid_argument("sdrlora: sf must be 5..12");
    if (p.SF < 7 && !p.implicit_header)
        throw std::invalid_argument(
            "sdrlora: SF5/SF6 require implicit_header (an explicit "
            "header needs 20 bits and the header block holds 4*(SF-2))");
    /* cr is NOT checked here. In explicit-header mode the header carries
     * the real coding rate and this field is ignored, so refusing an
     * out-of-range value would reject a configuration that decodes
     * correctly -- callers legitimately pass cr = 0 as "don't care".
     *
     * It is checked where it is READ: check_tx_params below for the
     * transmitter, payload_symbols for the codebook, and decoder_impl
     * for implicit mode, where there is no header to carry it. */
    /* FINITE FIRST, then positive, then BOUNDED: see check_rate_bounds in
     * lora_phy.h. `<= 0.0` passes inf and nan, and a non-finite or huge
     * ratio would reach `(int)(ratio + 0.5)` below -- an undefined
     * conversion -- and be refused, if at all, with the wrong message. */
    check_rate_bounds(p.bw, p.fs);
    if (p.ldro < 0 || p.ldro > 2)
        throw std::invalid_argument(
            "sdrlora: ldro must be 0 (off), 1 (on) or 2 (auto)");
    /* The SEVEN receive knobs, non-finite values included. (Amplitude is a
       transmit knob, validated in check_tx_params below.) */
    if (!(p.trise > 0.0) || p.trise > 1.0)
        throw std::invalid_argument(
            "sdrlora: trise must be > 0 and <= 1 s");
    /* center_freq: 0 is off; anything else is a carrier in Hz
     * of at least 1 MHz -- below it reads as MHz, a million times off. */
    if (!std::isfinite(p.center_freq) || p.center_freq < 0.0 ||
        (p.center_freq > 0.0 && p.center_freq < 1e6))
        throw std::invalid_argument(
            "sdrlora: center_freq must be 0 (off) or a carrier in Hz of at "
            "least 1e6");
    if (!(p.cutoff_ratio > 0.0) || p.cutoff_ratio > 64.0)
        throw std::invalid_argument(
            "sdrlora: cutoff_ratio must be > 0 and <= 64");
    /* NOT the widened cutoff. That rule applies only when the receive
     * channel filter runs, and this function knows nothing about
     * channel_filter_rx -- while check_tx_params calls it first, so
     * checking here would refuse a wide hop plan on the ENCODER for a
     * receive-filter reason the encoder never applies. It lives in
     * decode_buffer's eager block and decoder_impl's constructor, the two
     * places that know the flag. See check_effective_cutoff. */
    if (p.numtaps < 3 || p.numtaps > 8191)
        throw std::invalid_argument("sdrlora: numtaps must be 3..8191");
    if (p.numtaps % 2 == 0)
        throw std::invalid_argument(
            "sdrlora: numtaps must be ODD -- an even-length windowed sinc "
            "has a half-sample group delay, which shifts the whole frame off "
            "the sample grid for nothing");
    /* FINITE as well as > 1.0. `!(x > 1.0)` catches NaN and -inf, but
       +inf passes it -- and this is the only float knob here without an
       upper bound, so it is the only one where that matters (trise and
       cutoff_ratio reject inf through `> 1.0` and `> 64.0`). An infinite
       threshold declares no lock ever and therefore reports nothing at
       all, a silent failure. */
    if (!(p.sync_threshold > 1.0) || !std::isfinite(p.sync_threshold))
        throw std::invalid_argument(
            "sdrlora: sync_threshold must be finite and > 1.0 -- it is a "
            "peak-to-MEAN ratio, and a spectrum whose peak does not exceed "
            "its own mean is not a detection");
    if (p.search_tail_symbols < 13 || p.search_tail_symbols > 4096)
        throw std::invalid_argument(
            "sdrlora: search_tail_symbols must be 13..4096");
    if (p.min_osf < 1 || p.min_osf > 64)
        throw std::invalid_argument("sdrlora: min_osf must be 1..64");
    if (p.interp_target_osf < 1 || p.interp_target_osf > 64)
        throw std::invalid_argument(
            "sdrlora: interp_target_osf must be 1..64");
    if (p.interp_target_osf < p.min_osf)
        throw std::invalid_argument(
            "sdrlora: interp_target_osf must be >= min_osf -- interpolating "
            "up to a ratio still below the minimum clears nothing");

    /* A non-integer fs/bw would reach modulate(), which rounds osf while
     * keeping ts = 1/fs, so the chirp rate and the symbol length would
     * disagree and the result would not be a LoRa frame at any rate. */
    {
        const double ratio = p.fs / p.bw;
        /* DEFINED: check_rate_bounds above holds ratio <= OSF_MAX + 0.5,
         * which also refuses an overflowed quotient (1e300 / 1e-10). */
        const int osf = (int)(ratio + 0.5);
        const double err = ratio - (double)osf;
        const double tol = 1e-9 * (ratio > 1.0 ? ratio : 1.0);
        if (osf < 1 || err > tol || err < -tol)
            throw std::invalid_argument(
                "sdrlora: fs must be an integer multiple of bw");
    }
    /* The hop plan AFTER THE RATE: check_hop_plan computes its band as
     * (fs - bw)/2, which is negative when fs < bw, so checking it first
     * would blame the plan ("a hop channel is outside the sampled band")
     * for a bad rate. Factored into
     * check_hop_plan so packet_rx, which does not call this function,
     * applies the same rules in the same order. */
    check_hop_plan(p);
    /* Checked here as well as in modulate(), so that a caller validating a
     * parameter set up front gets the same answer the frame builder will. */
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
}

/* Reject a tx_params the PHY cannot honour.
 *
 * The GNU Radio blocks check all of this in their constructors; the core
 * API that phy_encode() exposes needs the same checks. Some of the cases
 * are worse than a wrong answer: SF = -1 and SF = 13 make 2^SF nonsense,
 * and the modulator would then try to allocate it, so the process would be
 * OOM-killed rather than told what is wrong. An exception is recoverable;
 * a SIGKILL takes the caller's flowgraph with it.
 *
 * The wording mirrors encoder_impl.cc deliberately -- two copies of a rule
 * that disagree about what is legal are worse than one. */
void check_tx_params(const tx_params& p)
{
    check_rx_params(p);
    /* HERE, not in check_rx_params: the transmitter READS cr to pick a
     * codebook, while an explicit-header receiver reads it off the air
     * and ignores this field. */
    /* TRANSMIT ONLY. The raised edge must fit inside a symbol -- above
       this the taper would index before the start of the preamble. That
       is a hazard of BUILDING the taper: nothing on the receive path
       builds one, so the receiver accepts SX1280 rates (SF5 at 812 kHz,
       SF6 at 1.625 MHz) that the encoder cannot produce. Frames at those
       rates come from real
       silicon, which is exactly the case a receiver exists for. */
    if (p.bw > max_bandwidth_for_sf(p.SF, p.trise))
        throw std::invalid_argument(
            "sdrlora: bw is too wide for this SF -- the raised edge would be "
            "longer than one symbol. The limit is 2^SF / TRISE (640 kHz at "
            "SF5, 1.28 MHz at SF6, 2.56 MHz at SF7)");
    if (p.cr < 1 || p.cr > 4)
        throw std::invalid_argument("sdrlora: cr must be 1..4");
    if (p.preamble_symbols < 1)
        throw std::invalid_argument(
            "sdrlora: preamble_symbols must be >= 1; a frame with no "
            "preamble is not a LoRa frame and no receiver can lock onto it");
    /* And from above. See PREAMBLE_SYMBOLS_MAX: unbounded, a huge value
     * would be an allocation request rather than an error. */
    if (p.preamble_symbols > PREAMBLE_SYMBOLS_MAX)
        throw std::invalid_argument(
            "sdrlora: preamble_symbols must be <= 65535 -- the width of the "
            "SX1276's preamble-length register, and 8192 times the standard "
            "8");
    /* Negative is AUTO, not an error; the bound applies to what it
     * resolves to. */
    if ((long)resolved_lead_in(p) > LEAD_IN_MAX)
        throw std::invalid_argument(
            "sdrlora: lead_in must be <= 67108864 samples -- it is pure "
            "padding ahead of the preamble, and the bound is what turns an "
            "absurd value into an error naming it rather than a bad_alloc");
    if (!(p.amplitude > 0.0) || p.amplitude > 1e6)
        throw std::invalid_argument(
            "sdrlora: amplitude must be > 0 and <= 1e6");
    if (!(p.t0_frac > -1.0 && p.t0_frac < 1.0))
        throw std::invalid_argument(
            "sdrlora: t0_frac must be in (-1, 1) -- it is a fraction of one "
            "SAMPLE, and NEGATIVE means sampled early. A whole sample either "
            "way is a lead_in change, not a sub-sample offset");
    /* The header carries these as single bytes; out of range they would be
     * truncated silently -- src = 256 would produce a frame byte-identical
     * to src = 0 -- and encoder_impl.cc rejects the same values.
     *
     * ONLY WITH THE PSEUDO-HEADER, which is the only place they go: with
     * standard framing encode_symbols() never reads them -- a packet
     * received with standard framing carries -1 in all three, and
     * re-sending it must work. The encoder
     * BLOCK still refuses them whatever the framing: they are typed into
     * its palette, where 300 is a typo rather than a packet being
     * forwarded. */
    if (p.pseudo_header &&
        (p.src < 0 || p.src > 255 || p.dst < 0 || p.dst > 255 ||
         p.seqn < 0 || p.seqn > 255))
        throw std::invalid_argument(
            "sdrlora: src, dst and seqn are one byte each and must be "
            "0..255; out of range they would be truncated into a frame that "
            "says something different from what was asked for");
    /* FINITE. cfo_ppm multiplies into the phase of every modulated sample
     * (`cfo = 2*cfo_ppm*1e-6*f0`, then cos/sin of it), so a NaN or an
     * infinity does not fail on its own -- it would come back as a buffer
     * of NaNs, finite only in the zero lead-in the modulation never
     * touches, with no error at all. Every floating-point knob on this
     * struct is checked finite.
     *
     * The bound here is finiteness: a large ppm is a real request -- tens
     * of ppm is ordinary for a cheap crystal -- and what an extreme value
     * costs is a frame outside the receiver's capture range, which is a
     * decision for the caller and not an error. Up to the point where the
     * carrier PHASE overflows, which check_frame_size refuses once the
     * frame length is known. */
    /* isfinite alone: it is false for NaN, so an `x == x` test beside it
     * would add nothing. */
    if (!std::isfinite(p.cfo_ppm))
        throw std::invalid_argument(
            "sdrlora: cfo_ppm must be finite -- it scales the carrier and a "
            "non-finite value emits a waveform of NaNs rather than failing");
    /* AND f0, for the same reason and by the same arithmetic. The carrier
     * term is `cfo = 2*cfo_ppm*1e-6*f0`, so the two are interchangeable
     * factors of one product: guarding one and leaving the other open
     * would leave the identical failure reachable one field over. Zeroing
     * the ppm does not rescue a non-finite f0 -- 0*inf is nan.
     *
     * f0 is a free-text `real` on the GRC encoder block, so a flowgraph
     * reaches it. The block yaml asserts it.
     *
     * Zero and negative stay legal -- f0 = 0 simply means no carrier offset.
     * Only nan and +-inf are refused here, matching cfo_ppm; a carrier whose
     * PHASE overflows -- cfo_ppm = f0 = 1e300 -- is check_frame_size's. */
    if (!std::isfinite(p.f0))
        throw std::invalid_argument(
            "sdrlora: f0 must be finite -- it multiplies cfo_ppm into the "
            "phase of every modulated sample, and a non-finite value emits a "
            "waveform of NaNs rather than failing");
    /* phi0 is the initial carrier phase and is handed straight to cos/sin
     * for every preamble sample, so a non-finite value is the same fault
     * again. */
    if (!std::isfinite(p.phi0))
        throw std::invalid_argument(
            "sdrlora: phi0 must be finite -- it is the initial carrier phase "
            "and goes into cos/sin of every modulated sample, so a non-finite "
            "value emits a waveform of NaNs rather than failing");
}

double frame_samples_estimate(const tx_params& p, long n_sym_payload)
{
    /* The frame-size estimate (the check against MAX_FRAME_SAMPLES) --
     * preamble + 32 + payload symbols, times
     * N, plus the lead-in. The 32 covers the header's first interleaver
     * block, the two sync symbols and the 2.25 down-chirps, with margin.
     * Returned as a double: the product overflows a long before it reaches
     * anything worth reporting. */
    /* The INTEGER ratio, as modulate() itself uses: the unrounded fs/bw
     * could refuse, by a fraction of a sample, a frame at a near-integer
     * rate (fs = 1e6 * (1 + 5e-10)) that the modulator makes exactly 2**28
     * samples long. */
    const double osf = (double)(long)(p.fs / p.bw + 0.5);
    const double N = (double)(1L << p.SF) * osf;
    const double n_sym = (double)p.preamble_symbols + 32.0
                       + (double)n_sym_payload;
    return (double)resolved_lead_in(p) + n_sym * N;
}

void check_frame_size(const tx_params& p, long n_sym_payload)
{
    /* The per-parameter bounds in check_tx_params do not COMPOSE: SF12 at
     * fs/bw = 8 with the legal preamble_symbols = 65535 is 2.15e9 samples,
     * 17.2 GB of complex64. This names that as an error rather than
     * reaching std::bad_alloc -- or, on a Linux host
     * with overcommit, the OOM killer, which is worse than a wrong answer. */
    const double n = frame_samples_estimate(p, n_sym_payload);
    if (!(n >= 0.0) || n > (double)MAX_FRAME_SAMPLES) {
        std::ostringstream os;
        os.setf(std::ios::fixed);
        os.precision(0);
        os << "sdrlora: the requested frame is about " << n
           << " samples (";
        os.precision(1);
        os << n * 8.0 / 1e9 << " GB of complex64), past the "
           << MAX_FRAME_SAMPLES << "-sample limit -- a "
           << p.preamble_symbols << "-symbol preamble and " << n_sym_payload
           << " payload symbols at SF" << p.SF << ", fs/BW "
           << (long)(p.fs / p.bw + 0.5) << ", plus " << resolved_lead_in(p)
           << " samples of lead-in";
        throw std::invalid_argument(os.str());
    }
    /* THE CARRIER PHASE over the same frame. cfo_ppm and f0 are each
     * checked finite, but their product is not bounded by that: the
     * modulator computes cfo = 2*cfo_ppm*1e-6*f0 and then 2*pi*cfo*t, and
     * past about 1.65e304 ppm at 868 MHz (or cfo_ppm = f0 = 1e300) that
     * overflows, so a FINITE setting would emit every modulated sample as
     * NaN with no error. Over the estimate above WITHOUT the lead-in, which
     * is zeros the CFO term never touches, so a long lead-in ahead of a
     * frame whose modulated part stays finite is not refused; computed in
     * the order the GRC assert uses, so the two agree at the boundary.
     * modulate() checks the exact frame. */
    const double cfo = 2.0 * p.cfo_ppm * 1e-6 * p.f0;
    const double n_mod = n - (double)resolved_lead_in(p);
    if (!std::isfinite(2.0 * kPi * cfo * (n_mod / p.fs))) {
        std::ostringstream os;
        os << "sdrlora: the carrier phase overflows: cfo_ppm " << p.cfo_ppm
           << " at f0 " << p.f0 << " is a " << cfo << " Hz carrier offset, "
              "and 2*pi*cfo*t over this frame is past the largest double, "
              "so every modulated sample would be NaN";
        throw std::invalid_argument(os.str());
    }
}

void encode(const tx_params& p, const std::vector<uint8_t>& msg,
            std::vector<std::complex<float> >& out)
{
    check_tx_params(p);
    std::vector<uint32_t> k_hdr, k_payload;
    encode_symbols(p, msg, k_hdr, k_payload);
    modulate(p, k_hdr, k_payload, out);
}

} /* namespace sdrlora */
} /* namespace gr */
