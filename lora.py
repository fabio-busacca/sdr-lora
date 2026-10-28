# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2024 SDR-LoRa contributors.
#
# Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
# See LICENSE for the full text.

import warnings

import math
import numpy as np

# Single source of truth for the package version.
__version__ = "2.0.0"


#############CONSTANTS#####################
# rising and falling edges duration
Trise = 50e-6
# Codewords of the first interleaver block that carry header bits. Five is the
# standard (the explicit header is 20 bits at CR 4/8); see _check_hdr_rows.
HDR_ROWS_DEFAULT = 5
# A typical crystal error, in ppm, for simulating an impaired link:
# encode(..., cfo_ppm=Cfo_PPM) bakes it in at both ends, 29.5 kHz at
# 868 MHz. encode() defaults to 0, a clean carrier. At BW = 125 kHz a
# clean frame survives a baked-in offset of at least 0.23 of BW below
# fs/BW = 2 and 0.48 at or above it; at narrower bandwidths the same ppm
# is a larger fraction of BW and the frame may not decode.
Cfo_PPM = 17
# Receiver Noise Figure
NF_dB = 6
SYNC_WORD_DEFAULT = 0x12

# Low-data-rate optimisation. LoRa drops 2 bits per symbol once a symbol lasts
# long enough that crystal drift moves it across a bin during the symbol, and
# the threshold is a DURATION, not a spreading factor: 16 ms. "SF > 10" is the
# same thing only at BW = 125 kHz; at 250 kHz the standard rule leaves SF11
# off. gr-lora_sdr uses the same constant (LDRO_MAX_DURATION_MS) and the same
# three modes.
LDRO_MAX_DURATION_MS = 16.0
LDRO_OFF, LDRO_ON, LDRO_AUTO = 0, 1, 2


def ldro_enabled(SF, BW, ldro=LDRO_AUTO):
    """The DE flag: 1 when the low-data-rate option is in force.

    ldro is LDRO_OFF, LDRO_ON, or LDRO_AUTO (the default), which enables it
    when the symbol duration 2**SF / BW exceeds LDRO_MAX_DURATION_MS.
    """
    # int(2.5) is 2, which is LDRO_AUTO, so non-integral values are rejected
    # rather than truncated; numpy integer scalars and 2.0 still work. _real
    # gives None, nan, inf and strings a named error.
    _fl = _real(ldro, "ldro")
    if not np.isfinite(_fl) or int(_fl) != _fl:
        raise ValueError("ldro must be a whole number (0 off, 1 on, 2 auto), "
                         "got %r" % (ldro,))
    mode = int(_fl)
    if mode == LDRO_OFF:
        return 0
    if mode == LDRO_ON:
        return 1
    if mode != LDRO_AUTO:
        raise ValueError("ldro must be 0 (off), 1 (on) or 2 (auto), got %r"
                         % (ldro,))
    # SF must be whole and BW real and positive, validated by name where the
    # auto rule reads them.
    _sf = _whole(SF, "SF")
    if not 5 <= _sf <= 12:
        raise ValueError("SF must be 5..12, got %r" % (SF,))
    _bw = _real(BW, "BW")
    if not (np.isfinite(_bw) and _bw > 0):
        raise ValueError("BW must be finite and positive, got %r" % (BW,))
    return 1 if (2.0 ** _sf) * 1e3 / _bw > LDRO_MAX_DURATION_MS else 0


def sync_word_symbols(sync_word):
    """The two preamble symbols a LoRa sync word is carried as.

    Two forms, the same two gr-lora_sdr accepts in its `sync_word` list:

    * **a byte**, split into nibbles and each scaled by 8: 0x12 -> (8, 16),
      the private/default value (0x34 is the public LoRaWAN one);
      0x34 -> (24, 32). This is what a LoRa radio's register holds, and it can
      only ever produce multiples of 8.
    * **a pair of raw symbol values**, `(s1, s2)`, which reaches the ones the
      byte form cannot -- gr-lora_sdr transmits and receives (18, 52), for
      instance.

    `[0x34]` and `[24, 32]` are the same waveform in gr-lora_sdr's encoder.

    A symbol of 0 is rejected in either form: it would coincide with the
    preamble up-chirps and be indistinguishable from the preamble it follows.
    """
    if isinstance(sync_word, np.ndarray) and sync_word.ndim == 0:
        # A 0-d array is ONE value, the byte form, not a sequence.
        sync_word = sync_word[()]
    if isinstance(sync_word, (list, tuple, np.ndarray)):
        # _whole, not int(): int() would truncate [8.9, 16.9] to [8, 16] and
        # accept strings inconsistently.
        vals = [_whole(v, "sync_word[%d]" % i)
                for i, v in enumerate(sync_word)]
        if len(vals) == 1:
            return sync_word_symbols(vals[0])
        if len(vals) != 2:
            raise ValueError(
                "sync_word sequence must be 1 value (a byte) or 2 (the raw "
                "symbols), got %d" % len(vals))
        s1, s2 = vals
        # Symbol 0 is refused here; symbol 1 is refused too, by
        # check_sync_symbols(), which takes out every symbol within a bin of
        # the preamble -- 1 and 2**SF - 1 -- once SF is known. Symbol 0
        # coincides with the preamble up-chirps outright; symbol 1 sits ONE
        # bin from them, and the sync check runs with a tolerance of one bin,
        # so it is not reliably separable from more preamble (gr-lora_sdr's
        # own loopback fails such pairs too). Refusing it keeps this receiver
        # and the C++ one behaving identically, and costs nothing real: a byte
        # sync word always yields multiples of 8.
        #
        # Upper bound is the widest symbol alphabet (SF12); encode() and the
        # sync check reduce mod 2**SF, matching gr-lora_sdr.
        if not (0 < s1 < 4096 and 0 < s2 < 4096):
            raise ValueError(
                "sync symbols must be in 1..4095, got (%d, %d); 0 lands on "
                "the preamble up-chirps" % (s1, s2))
        # How far they must sit from the preamble bin depends on SF, so that
        # check is check_sync_symbols(), called from encode() and decode().
        return s1, s2

    sw = _whole(sync_word, "sync_word")
    if not 0 <= sw <= 0xFF:
        raise ValueError("sync_word must be a byte or a pair of raw symbols, "
                         "got %r" % (sync_word,))
    hi, lo = (sw >> 4) & 0xF, sw & 0xF
    if hi == 0 or lo == 0:
        raise ValueError(
            "sync_word 0x%02X has a zero nibble; that sync symbol would sit "
            "on the preamble up-chirps and be undetectable" % sw)
    return hi * 8, lo * 8


def check_sync_symbols(s1, s2, SF):
    """Reject sync symbols that cannot be separated from the preamble.

    The sync word is detected as a bin offset from the preamble's own bin with
    a tolerance of one bin, and bins are cyclic mod 2**SF -- so symbol 1 and
    symbol 2**SF - 1 are both ONE bin away. 127 is adjacent to the preamble at
    SF7 and 127 bins from it at SF12, which is why this takes SF and
    sync_word_symbols() cannot do it.

    Symbols in that band are not reliably detected by any receiver
    (gr-lora_sdr's own loopback fails them too), and refusing the band
    outright keeps the Python and C++ receivers identical. It costs nothing
    real: a byte sync word gives multiples of 8, which are never within a bin
    of the preamble at any SF.
    """
    K = 1 << int(SF)
    for s in (s1, s2):
        d = int(s) % K
        if min(d, K - d) < 2:
            raise ValueError(
                "sync symbol %d is %d bin(s) from the preamble at SF%d "
                "(bins are cyclic mod %d, and the sync check has a tolerance "
                "of one bin, so it cannot be told from more preamble). Use a "
                "symbol between 2 and %d."
                % (int(s), min(d, K - d), int(SF), K, K - 2))


# Samples a streaming caller must keep after samples_decoding() returns, so a
# preamble straddling the end of one call is still found in the next.
# rf_decode() needs the 12.25-symbol preamble/sync/down-chirp pattern plus the
# accumulation history behind it (_SYNC_ACC_LAG + _SYNC_ACC_SYMBOLS); 24
# symbols is that with room to spare. Mirrors SEARCH_TAIL_SYMBOLS in
# gr-sdrlora/include/sdrlora/lora_rx.h.
SEARCH_TAIL_SYMBOLS = 24


# Zero samples the transmitter emits before each preamble. See
# complex_lora_packet(); gr-lora_sdr's equivalent is `frame_zero_padd`.
LEAD_IN_DEFAULT = 10000


def _resolve_lead_in(lead_in, SF, BW, fs, preamble_bits):
    """lead_in in samples: None, or a negative whole number, is the
    default.

    One rule for encode() and complex_lora_packet(). A non-whole value
    (100.7, nan, inf) raises ValueError rather than being truncated. Mirrors
    gr::sdrlora::resolved_lead_in, whose lead_in is an int.
    """
    if lead_in is not None:
        v = _whole(lead_in, "lead_in")
        if v >= 0:
            return v
    return default_lead_in(SF, BW, fs, preamble_bits)


def default_lead_in(SF, BW, fs, preamble_bits=8):
    """Zero samples to put ahead of the preamble when `lead_in` is None.

    10000 samples, topped up so that robust_sync can actually lock onto the
    result. That detector accumulates _SYNC_ACC_SYMBOLS symbols before its
    statistic exists, and the history can come from the lead-in or from the
    preamble -- but the lead-in is counted in SAMPLES, so what it contributes
    shrinks as the symbol grows: 9.8 symbols at SF7 with fs/BW = 8, and 0.31
    at SF12. Without the top-up the shortest preamble the default receiver
    could lock onto would rise with SF (3 symbols at SF7, 6 at SF12), and
    encode() would emit legal frames that decode() cannot read.

    Topping up the shortfall in whole symbols removes that dependence, for
    any preamble of 3 symbols or more. One and two symbols remain
    undetectable for both detectors and at every SF -- that is the detection
    floor _check_preamble's docstring describes, not this warm-up.

    At the standard preamble_bits=8 the shortfall is zero, so this returns
    exactly 10000 at every SF and rate. Mirrors gr::sdrlora::default_lead_in().
    """
    # Validated by the same rules as gr::sdrlora::default_lead_in: a bad rate
    # raises ValueError (std::invalid_argument in C++).
    SF = _whole(SF, "SF")
    if SF < 5 or SF > 12:
        raise ValueError("default_lead_in: SF must be 5..12, got %r" % (SF,))
    _check_rate_bounds(BW, fs)
    # Round half UP, as the C++ mirror does; int(round(...)) would round half
    # to even and differ at half-integer ratios.
    ratio = float(fs) / float(BW)
    osf = max(1, int(np.floor(ratio + 0.5)))
    n_sym = (1 << int(SF)) * osf
    # Whole, by name: int() would truncate 3.9 to 3.
    shortfall = max(0, _SYNC_ACC_SYMBOLS - _whole(preamble_bits,
                                                  "preamble_bits"))
    # Saturated at LEAD_IN_MAX, like gr::sdrlora::default_lead_in, so a huge
    # SF x osf cannot overflow the C++ int return.
    return int(min(LEAD_IN_MAX, max(LEAD_IN_DEFAULT, shortfall * n_sym)))


#Minimum duration for a LoRa packet
min_time_lora_packet = 20e-3 #20 milliseconds


##########################################################





def lora_packet(BW, OSF, SF, k1, k2, n_pr, IH, CR, MAC_CRC, SRC, DST, SEQNO, MESSAGE, Trise, t0_frac, phi0, pseudo_header=True, ldro=LDRO_AUTO, hdr_rows=None):
    # in this implementation, the payload includes a 4 byte pseudo-header: (SRC, DST, SEQNO, LENGTH)
    # The on-air LENGTH counts everything after the header, so it includes the
    # 4-byte pseudo-header when that is present.
    LENGTH = np.uint16((4 if pseudo_header else 0) + (MESSAGE.size))

    # n_sym_hdr: number of chirps encoding the header (0 if IH: True)
    # n_bits_hdr: payload bits carried in the first (header) interleaver block
    (n_sym_hdr, n_bits_hdr) = lora_header_init(SF, IH, hdr_rows)
    # BW is known here, so this is where the duration-based LDRO rule can be
    # applied; everything below takes the resolved flag.
    _de = ldro_enabled(SF, BW, ldro)
    [PAYLOAD, n_sym_payload] = lora_payload_init(SF, LENGTH, MAC_CRC, CR, n_bits_hdr, DST, SRC, SEQNO, MESSAGE, pseudo_header, _de)

    # -------------------------------------------------BIT TO SYMBOL MAPPING
    payload_ofs = 0
    # The first block exists either way; with an implicit header none of its
    # codewords carry header bits, so all of them carry payload and hdr_rows
    # is inert.
    [k_hdr, payload_ofs] = lora_header(
        SF, LENGTH, CR, MAC_CRC, PAYLOAD, payload_ofs,
        0 if IH else (HDR_ROWS_DEFAULT if hdr_rows is None else int(hdr_rows)))

    k_payload = lora_payload(SF, CR, n_sym_payload, PAYLOAD, payload_ofs, _de)

    # --------------------------------- CSS MODULATION

    # number of samples per chirp
    K = np.power(2, SF)
    N = int(K * OSF)
    # number of samples in the rising/falling edge
    Nrise = int(np.ceil(Trise * BW * OSF))

    # preamble modulation
    (p, phi) = lora_preamble(n_pr, k1, k2, BW, K, OSF, t0_frac, phi0)

    # samples initialization
    s = np.concatenate((np.zeros(Nrise), p, np.zeros(N * (n_sym_hdr + n_sym_payload) + Nrise)))

    # rising edge samples
    s[0:Nrise] = p[N - 1 + np.arange(-Nrise + 1, 1)] * np.power(
        np.sin(np.pi / 2 * np.arange(1, Nrise + 1) / Nrise), 2)
    s_ofs = p.size + Nrise

    # header + payload modulation. Header and payload symbols are modulated
    # identically (up-chirps, same phase chain), so they go out in one pass.
    sym_k = np.concatenate((np.asarray(k_hdr, dtype=np.float64).ravel(),
                            np.asarray(k_payload, dtype=np.float64).ravel()))
    if sym_k.size:
        n_out = sym_k.size * N
        (s[s_ofs:s_ofs + n_out], phi) = lora_symbols(sym_k, BW, K, OSF, t0_frac, phi)
        s_ofs = s_ofs + n_out

    # falling edge samples
    #
    # This is a no-op, deliberately. s_ofs has already advanced past the
    # last symbol, so s[s_ofs : s_ofs+Nrise] is the trailing zero padding
    # and multiplying zeros by cos^2 leaves zeros: the frame ends at full
    # amplitude and steps to zero. Only the RISING edge is a real taper.
    #
    # It is kept because this is the published waveform, unchanged from the
    # paper's implementation, and a true taper is worth very little (about
    # 0.2 dB of out-of-band power at SF7/OSF 8, because the chirp's own
    # skirts dominate). The symmetric construction would be
    # s[s_ofs+i] = s[s_ofs-N+i] * cos^2(...), the cyclic suffix of the last
    # symbol, mirroring the cyclic prefix the rising edge uses.
    # gr-sdrlora/lib/lora_phy.cc carries the identical no-op.
    s[s_ofs:s_ofs + Nrise] = s[s_ofs:s_ofs + Nrise] * np.power(
        np.cos(np.pi / 2 * np.arange(0, Nrise) / Nrise), 2)

    return s, k_hdr, k_payload



def lora_header_init(SF, IH, hdr_rows=None):
    """Size of the first interleaver block, which always exists.

    LoRa's interleaver codes the first SF-2 codewords at CR 4/8 whether or not
    a header occupies them -- 8 symbols carrying 4*(SF-2) data bits. With an
    explicit header, 20 of those bits are the header and the rest are payload
    (`n_bits_hdr`). With an implicit header there is no header, so all
    4*(SF-2) bits are payload, but the block is still 8 symbols at CR 4/8,
    as other LoRa implementations (gr-lora_sdr included) transmit it.
    """
    if (IH):
        # Plain ints, not np.uint16: n_sym_hdr feeds N * (n_sym_hdr +
        # n_sym_payload), and a uint16 there wraps once the sample count passes
        # 65535 -- which is every SF >= 9.
        CR_hdr = 4
        DE_hdr = 1
        n_sym_hdr = 4 + CR_hdr
        n_bits_hdr = int(4 * (SF - 2 * DE_hdr))
    else:
        # interleaving block size, respectively for header & payload
        CR_hdr = 4
        DE_hdr = 1
        n_sym_hdr = 4 + CR_hdr
        intlv_hdr_size = (SF - 2 * DE_hdr) * (4 + CR_hdr)
        # The header is hdr_rows codewords of 4 bits and the rest of the
        # block is payload; hdr_rows = 5 (20 bits) is the standard.
        _rows = HDR_ROWS_DEFAULT if hdr_rows is None else int(hdr_rows)
        n_bits_hdr = int(intlv_hdr_size * 4 / (4 + CR_hdr) - 4 * _rows)

    return n_sym_hdr, n_bits_hdr


# function [k_hdr,payload_ofs] =
def lora_header(SF, LENGTH, CR, MAC_CRC, PAYLOAD, payload_ofs,
                hdr_rows=HDR_ROWS_DEFAULT):
    # HDR_ROWS_DEFAULT, not the literal 5, so encode and its decoding
    # mirror lora_header_decode share one constant.
    """Build the first interleaver block: 8 symbols at CR 4/8.

    hdr_rows: how many of the PPM codewords carry header bits. 5 for an
        explicit header (the 20-bit header is five 4-bit codewords); 0 for an
        implicit header, where the whole block is payload.
    """
    # header parity check matrix (reverse engineered through brute force search)

    header_FCS = np.array(([1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 1, 0, 0],
                           [0, 1, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1],
                           [0, 0, 1, 1, 1],
                           [0, 1, 0, 1, 1]))

    CR_hdr = 4
    Hamming_hdr = np.array(([1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]), dtype=np.uint8)
    n_sym_hdr = 4 + CR_hdr
    DE_hdr = 1
    PPM = SF - 2 * DE_hdr
    gray_hdr = gray_lut(PPM)[0]
    intlv_hdr_size = PPM * n_sym_hdr

    # header (20 bit) -- only when the block actually carries one
    LENGTH_bits = num2binary(LENGTH, 8)
    CR_bits = num2binary(CR, 3)
    hdr = np.concatenate((LENGTH_bits[np.arange(3, -1, -1, dtype=np.uint8)],
                          LENGTH_bits[np.arange(7, 3, -1, dtype=np.uint8)], np.array([MAC_CRC], dtype=np.uint8),
                          CR_bits[np.arange(2, -1, -1, dtype=np.uint8)], np.zeros(8, dtype=np.uint8)))
    hdr_chk_indexes = np.concatenate((np.arange(3, -1, -1, dtype=np.uint8), np.arange(7, 3, -1, dtype=np.uint8),
                                      np.arange(11, 7, -1, dtype=np.uint8)))

    hdr_chk = np.mod(hdr[hdr_chk_indexes] @ header_FCS, 2)
    hdr[12] = hdr_chk[0]
    hdr[16:20] = hdr_chk[4:-0:-1]

    # parity bit calculation
    C = np.zeros((PPM, 4 + CR_hdr))
    for k in range(0, hdr_rows):
        C[k, 0:4] = hdr[k * 4 + np.arange(0, 4)]
        C[k, 3 + np.arange(1, CR_hdr + 1)] = np.mod(C[k, 0:4] @ Hamming_hdr, 2)

    for k in range(hdr_rows, PPM):
        C[k, 0:4] = PAYLOAD[payload_ofs + np.arange(0, 4)]
        payload_ofs = payload_ofs + 4
        C[k, 3 + np.arange(1, CR_hdr + 1)] = np.mod(C[k, 0:4] @ Hamming_hdr, 2)

    # rows flip
    C = np.flip(C, 0)

    S = np.zeros((4 + CR_hdr, PPM), dtype=np.uint8)
    for ii in range(0, PPM):
        for jj in range(0, 4 + CR_hdr):
            S[jj, np.mod(ii + jj, PPM)] = C[ii, jj]

    bits_hdr = np.reshape(S.transpose(), intlv_hdr_size, order='F')

    # bit to symbol mapping
    k_hdr = np.zeros(n_sym_hdr)
    K = np.power(2, SF)
    for sym in range(0, n_sym_hdr):
        k_hdr[sym] = K - 1 - np.power(2, (2 * DE_hdr)) * gray_hdr[
            bits_hdr[sym * PPM + np.arange(0, PPM, dtype=np.uint16)] @ np.power(2, np.arange(PPM - 1, -1, -1))]

    return k_hdr, payload_ofs



def lora_payload_init(SF, LENGTH, MAC_CRC, CR, n_bits_hdr, DST, SRC, SEQNO,
                      MESSAGE, pseudo_header=True, DE=None):
    """Build the whitened payload bit vector.

    pseudo_header: keep this implementation's own 4-byte DST/SRC/SEQNO/LENGTH
        prefix ahead of the message. It is NOT part of LoRa -- a standard
        transmitter puts the message at byte 0 and the PHY header's LENGTH is
        the message length. Pass False for a frame any LoRa receiver can read
        without stripping four bytes (and 4 bytes less airtime).
    """
    # bigger spreading factors (11 and 12) use 2 less bits per symbol
    if DE is None:
        # No caller-supplied flag: fall back to the spreading-factor rule,
        # which is correct at BW = 125 kHz; lora_packet() and
        # lora_packet_rx() pass the duration-based value instead.
        DE = 1 if SF > 10 else 0
    PPM = SF - 2 * DE
    n_bits_blk = PPM * 4
    n_bits_tot = 8 * LENGTH + 16 * MAC_CRC
    # int() before subtracting: LENGTH is np.uint16, so the difference would
    # wrap when the payload is shorter than the header block. Zero is the
    # legitimate answer there (the payload fits the header block).
    n_blk_tot = max(0, int(np.ceil(
        (int(n_bits_tot) - int(n_bits_hdr)) / float(n_bits_blk))))
    n_sym_blk = 4 + CR
    n_sym_payload = n_blk_tot * n_sym_blk

    byte_range = np.arange(7, -1, -1)
    PAYLOAD = np.zeros(int(n_bits_hdr + n_blk_tot * n_bits_blk), dtype=np.uint8)
    msg_ofs = 4 if pseudo_header else 0
    if pseudo_header:
        PAYLOAD[byte_range] = num2binary(DST, 8)
        PAYLOAD[8 + byte_range] = num2binary(SRC, 8)
        PAYLOAD[8 * 2 + byte_range] = num2binary(SEQNO, 8)
        PAYLOAD[8 * 3 + byte_range] = num2binary(LENGTH, 8)
    # PAYLOAD[base+i] holds bit i of each byte, which is exactly what
    # unpackbits(bitorder='little') produces -- one call for the whole message
    # instead of a num2binary() call per byte.
    if MESSAGE.size:
        PAYLOAD[8 * msg_ofs: 8 * (msg_ofs + MESSAGE.size)] = np.unpackbits(
            np.ascontiguousarray(MESSAGE, dtype=np.uint8), bitorder='little')

    if MAC_CRC:
        PAYLOAD[8 * LENGTH + np.arange(0, 16)] = CRC16(PAYLOAD[0:8 * LENGTH])

    # ----------------------------------------------------------- WHITENING
    W = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    W_fb = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
    for k in range(1, int(np.floor(len(PAYLOAD) / 8) + 1)):
        PAYLOAD[(k - 1) * 8 + np.arange(0, 8, dtype=np.int32)] = np.mod(
            PAYLOAD[(k - 1) * 8 + np.arange(0, 8, dtype=np.int32)] + W, 2)
        W1 = np.array([np.mod(np.sum(W * W_fb), 2)])
        W = np.concatenate((W1, W[0:-1]))

    return PAYLOAD, n_sym_payload


# CRC-16 initial states for on-air lengths 1..4. The entries for 2, 3 and 4
# are solved against gr-lora_sdr frames.
#
# The length-1 entry is not defined by anything: the checksum is a 16-bit
# register clocked over the payload, so it needs two bytes to fill, and with
# one byte half of it is still whatever the initial state was. encode()
# refuses the case -- see _check_crc_length(). Asked for one, gr-lora_sdr sets
# the header's CRC flag but appends no checksum. The value is kept only so the
# RECEIVE path stays total: such a frame gets a computed value that will not
# match and is reported as crc_ok = 0 rather than raising inside the decoder.
_CRC16_SHORT_STATE_VEC = np.array([34666, 55825, 56317, 11633], dtype=np.int64)
_CRC16_STATE_VEC = np.array([46885, 27367, 35014, 54790, 18706, 15954, \
                              9784, 59350, 12042, 22321, 46211, 20984, 56450, 7998, 62433, 35799, \
                              2946, 47628, 30930, 52144, 59061, 10600, 56648, 10316, 34962, 55618, \
                              57666, 2088, 61160, 25930, 63354, 24012, 29658, 17909, 41022, 17072, \
                              42448, 5722, 10472, 56651, 40183, 19835, 21851, 13020, 35306, 42553, \
                              12394, 57960, 8434, 25101, 63814, 29049, 27264, 213, 13764, 11996, \
                              46026, 6259, 8758, 22513, 43163, 38423, 62727, 60460, 29548, 18211, \
                              6559, 61900, 55362, 46606, 19928, 6028, 35232, 29422, 28379, 55218, \
                              38956, 12132, 49339, 47243, 39300, 53336, 29575, 53957, 5941, 63650, \
                              9502, 28329, 44510, 28068, 19538, 19577, 36943, 59968, 41464, 33923, \
                              54504, 49962, 64357, 12382, 44678, 11234, 58436, 47434, 63636, 51152, \
                              29296, 61176, 33231, 32706, 27862, 11005, 41129, 38527, 32824, 20579, \
                              37742, 22493, 37464, 56698, 29428, 27269, 7035, 27911, 55897, 50485, \
                              10543, 38817, 54183, 52989, 24549, 33562, 8963, 38328, 13330, 24139, \
                              5996, 8270, 49703, 60444, 8277, 43598, 1693, 60789, 32523, 36522, \
                              17339, 33912, 23978, 55777, 34725, 2990, 13722, 60616, 61229, 19060, \
                              58889, 43920, 9043, 10131, 26896, 8918, 64347, 42307, 42863, 7853, \
                              4844, 60762, 21736, 62423, 53096, 19242, 55756, 26615, 53246, 11257, \
                              2844, 47011, 10022, 13541, 18296, 44005, 23544, 18733, 23770, 33147, \
                              5237, 45754, 4432, 22560, 40752, 50620, 32260, 2407, 26470, 2423, \
                              33831, 34260, 1057, 552, 56487, 62909, 4753, 7924, 40021, 7849, \
                              4895, 10401, 32039, 40207, 63952, 10156, 53647, 51938, 16861, 46769, \
                              7703, 9288, 33345, 16184, 56808, 30265, 10696, 4218, 7708, 32139, \
                              34174, 32428, 20665, 3869, 43003, 6609, 60431, 22531, 11704, 63584, \
                              13620, 14292, 37000, 8503, 38414, 38738, 10517, 48783, 30506, 63444, \
                              50520, 34666, 341, 34793, 26234], dtype=np.uint16)
# The last entry is the state for on-air LENGTH 255, the longest frame. A
# round trip through this library cannot validate this table, because
# encode() and decode() share it (as does gr-sdrlora); its entries agree with
# an independent receiver, gr-lora_sdr's crc_verif.

#CRC-16 calculation for LoRa (reverse engineered from a Libelium board)
def CRC16(bits, init=None):
    length = int(len(bits) / 8)
    # Initial states for the CRC16: _CRC16_STATE_VEC covers lengths 5..255
    # and _CRC16_SHORT_STATE_VEC lengths 1..4. Indexing the long table below
    # 5 would wrap a negative index to its far end. `init` lets a caller
    # supply the state explicitly.
    if init is None:
        if length < 1:
            # Zero bytes has no state; a negative index would wrap silently.
            # Reachable from lora_payload_decode when a chance-passing header
            # gives LENGTH = 0.
            raise ValueError(
                "a LoRa CRC is not defined over zero bytes; no initial state "
                "exists for it and the result would be meaningless")
        if length < 5:
            init = int(_CRC16_SHORT_STATE_VEC[length - 1])
        else:
            init = int(_CRC16_STATE_VEC[length - 5])
    crc_tmp = num2binary(int(init), 16)
    # crc_poly = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1]
    # 	for j = 1:numel(bits)/8
    # 		for k = 1:8
    # 			add = crc_tmp(1)
    # 			crc_tmp = [crc_tmp(2: ),bits((j-1)*8+9-k)]
    # 			if add
    # 				crc_tmp = mod(crc_tmp+crc_poly(2: ),2)
    #
    #
    #
    # 	CRC=crc_tmp(16:-1:1)
    # Plain Python ints and lists here: the loop runs 8*length times, and
    # numpy scalar indexing and np.mod() would dominate it.
    state = [int(b) for b in crc_tmp]
    in_bits = [int(b) for b in bits]
    pos = 0
    pos4 = 4
    pos11 = 11
    for j in range(0, length):
        base = j * 8 + 7
        for k in range(0, 8):
            add = state[pos]
            state[pos] = in_bits[base - k]
            if add:
                state[pos4] = 1 - state[pos4]
                state[pos11] = 1 - state[pos11]
                state[pos] = 1 - state[pos]

            pos = (pos + 1) % 16
            pos4 = (pos4 + 1) % 16
            pos11 = (pos11 + 1) % 16

    CRC = np.array([state[(pos + i) % 16] for i in range(15, -1, -1)], dtype=np.uint8)
    return CRC


# gray and reversed-gray mappings

def gray_lut(n):
    n = int(n)
    pow_n = 2 ** n
    g = np.zeros(pow_n, dtype=np.uint16)

    vec = np.atleast_2d(np.arange(0, pow_n, dtype=np.uint16)).transpose()
    vec = np.flip((vec.view(np.uint8)), 1)
    vec = np.unpackbits(vec).reshape(pow_n, 16)
    vec = vec[:, -n:]
    vec = vec.transpose()
    support_x = np.zeros((n, pow_n))
    support_x[1:, :] = vec[:-1, :]

    ig = np.matmul(np.power(2, np.arange(n - 1, -1, -1)), np.mod(vec + support_x, 2))
    ig = ig.astype(int)
    g[ig] = np.arange(0, pow_n, dtype=np.uint16)
    return g, ig


# function k_payload = \
def lora_payload(SF, CR, n_sym_payload, PAYLOAD, payload_ofs, DE=None):
    # varargout = {DST, SRC, SEQNO, MESSAGE}

    # hamming parity check matrices 
    Hamming_P1 = np.array(([1], [1], [1], [1]), dtype=np.uint8)
    Hamming_P2 = np.array(([1, 0], [1, 1], [1, 1], [0, 1]), dtype=np.uint8)
    Hamming_P3 = np.array(([1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]), dtype=np.uint8)
    Hamming_P4 = np.array(([1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]), dtype=np.uint8)

    if CR == 1:
        Hamming = Hamming_P1
    elif CR == 2:
        Hamming = Hamming_P2
    elif CR == 3:
        Hamming = Hamming_P3
    elif CR == 4:
        Hamming = Hamming_P4
    else:
        # A named error for a CR outside 1..4, which a direct caller of this
        # public function can pass; the C++ throws for the same input.
        raise ValueError(
            "CR must be 1, 2, 3 or 4 (coding rates 4/5 to 4/8), got %r -- "
            "there is no Hamming parity matrix for it" % (CR,))

    if DE is None:
        # No caller-supplied flag: fall back to the spreading-factor rule,
        # which is correct at BW = 125 kHz; lora_packet() and
        # lora_packet_rx() pass the duration-based value instead.
        DE = 1 if SF > 10 else 0

    PPM = SF - 2 * DE
    n_sym_blk = (4 + CR)
    intlv_blk_size = PPM * n_sym_blk
    gray = gray_lut(PPM)[0]
    K = np.power(2, SF)
    n_blk_tot = int(n_sym_payload / n_sym_blk)
    C = np.zeros((PPM, 4 + CR))
    S = np.zeros((4 + CR, PPM))
    k_payload = np.zeros(n_sym_payload)
    for blk in range(0, n_blk_tot):
        for k in range(0, PPM):
            C[k, 0:4] = PAYLOAD[payload_ofs + np.arange(0, 4)]
            payload_ofs = payload_ofs + 4
            C[k, 4: 4 + CR] = np.mod(C[k, 0:4] @ Hamming, 2)

        # row flip
        C = np.flip(C, 0)

        # interleaving
        for ii in range(0, PPM):
            for jj in range(0, 4 + CR):
                S[jj, np.mod(ii + jj, PPM)] = C[ii, jj]

        bits_blk = np.reshape(S.transpose(), intlv_blk_size, order='F')

        # bit to symbol mapping
        for sym in range(0, n_sym_blk):
            k_payload[(blk) * n_sym_blk + sym] = K - 1 - np.power(2, (2 * DE)) * gray[int(
                bits_blk[(sym * PPM + np.arange(0, PPM, dtype=np.uint16))] @ np.power(2, np.arange(PPM - 1, -1, -1)))]

    return k_payload





def chirp(f_ini,N,Ts,Df,t0_frac = 0,phi_ini = 0):
    t = (t0_frac+np.arange(0,N, dtype = np.int32))*Ts
    T = N*Ts
    # t*t is bit-identical to np.power(t,2) and avoids the generic pow path.
    two_pi = 2*np.pi
    s = _cis(phi_ini + two_pi*f_ini*t + np.pi*Df*(t*t))
    phi_fin = phi_ini + two_pi*f_ini*T + np.pi*Df*(T*T)
    return s, phi_fin


# Where, in symbols past the frame's leading edge, the fine-sync timing
# estimate is exact: the mean centre of its four windows, two preamble
# up-chirps at 5..7 and the two down-chirps at 10..12 (rf_decode's _fine).
SFO_REF_SYMBOLS = 8.5


def _sym_start(p_ofs_est, ofs, sfo, Nrise, N):
    """The first sample of the symbol `ofs` samples into the frame.

    SAMPLE-CLOCK TRACKING. An SDR derives its sample clock and its local
    oscillator from one crystal, and so does a LoRa node, so the carrier
    offset fine sync measures, Cfo_est, is the crystals' joint error times
    the carrier -- and the same error stretches or shrinks every symbol.
    sfo = Cfo_est / carrier is that error as a fraction; a symbol nominally
    `ofs` in then sits sfo*ofs samples early, counted from where the timing
    estimate was made (SFO_REF_SYMBOLS), not from the frame start. Without
    it the windows walk off long frames (a few ppm is enough over a
    251-byte frame from SF9 up); gr-lora_sdr's center_freq argument does
    the same thing. sfo = 0 is the untracked placement.
    """
    if not sfo:
        return p_ofs_est + ofs
    ref = Nrise + SFO_REF_SYMBOLS * N
    return int(p_ofs_est + ofs - np.round(sfo * (ofs - ref)))


# THE DRIFT READ FROM THE SIGNAL. _sym_start follows the clock offset the
# carrier offset implies, which assumes the carrier offset IS crystal error.
# On some SDRs the carrier and the sample clock differ by a few ppm, and over
# a 251-byte SF12 frame (9 s) 3 ppm is ~3.7 chips -- past the +-2 the
# low-data-rate symbols absorb. A drift the windows do not follow moves each
# dechirped peak by a growing fraction of a bin (a timing error of d chips
# is a d-bin frequency shift), so the slope of the payload symbols'
# fractional peak offsets IS the residual drift. When a frame locates and
# fails its CRC, rf_decode reads it and demodulates again, keeping the
# result only if the CRC then passes. Without center_freq the same reads
# the whole drift.
_DRIFT_RETRY = True        # private switch: isolates the carrier tracking
_LDRO_RETRY = True         # private switch: likewise for the LDRO flip
DRIFT_RETRIES = 3          # further demodulations at most; each estimate
                           # recovers ~80% of what remains, so 3 leave <1%
DRIFT_MIN_SYMBOLS = 8      # fewer payload symbols say nothing about a slope
DRIFT_MIN_BINS = 0.1       # a correction below this over the frame is noise
DRIFT_LAGS = (1, 4, 16)    # coarse to fine; see _drift_rate
# BELOW fs/BW 4 THE FRACTIONAL OFFSET IS NOT A CLOCK. A window moves in
# half-chip steps at fs/BW 2 and the decimation phase reshapes the peak
# (the two-segment interference, see the payload loop), so a small residual
# of a few ppm can read with the wrong sign. A LARGE drift still reads right
# there, so below fs/BW 4 one estimate is taken and these offsets are tried
# around it and around the carrier's drift, CRC-gated.
DRIFT_MIN_OSF = 4
DRIFT_GRID_PPM = (2.0, -2.0, 4.0, -4.0, 6.0, -6.0)
# The LDRO flip only from SF11, where the known LDRO mismatches between
# stacks lie: the Arduino LoRa library 0.8.0 leaves LDRO off at SF11 /
# 125 kHz, older releases of it never set LDRO, and some stacks set it by
# spreading factor at any bandwidth. Below SF11 the flip would only cost
# decode time.
LDRO_RETRY_MIN_SF = 11
# THE NO-CRC CHECK. A frame without a CRC has nothing to check, and a header
# that passes its 5-bit FCS by chance on noise and claims no CRC would come
# out as a frame of noise. Such a frame is crc_ok only if its symbols after
# the sync word look like signal: the mean of their peak gaps (_peak_gap),
# each capped at NOCRC_GAP_CAP, must reach nocrc_min_gap(SF).
#
# The gap is the peak's power less the largest outside its main lobe, over
# the mean power there -- in noise the top spacing of K exponentials, which
# is Exp(1) whatever K, so its floor barely moves with SF (a second-over-
# first peak ratio would). The cap keeps a few strong symbols -- a frame
# overlapping part of a junk frame's span -- from carrying the rest. Each
# per-SF threshold is where a frame of 8 noise symbols, the fewest any frame
# has, passes about 1 time in 1000, at fs/BW 2, 4 and 8 alike. SF5 and SF6
# sit higher: with 32 and 64 bins the mean outside the lobe is itself noisy.
#
# Rejecting a junk lock also stops the walk from skipping the frames behind
# it, so the check costs essentially nothing in delivered frames.
_NOCRC_CHECK = True        # private switch: turns the check off
NOCRC_GAP_CAP = 8.0
_NOCRC_MIN_GAP = {5: 2.9, 6: 2.65, 7: 2.55}
NOCRC_MIN_GAP_DEFAULT = 2.5    # SF8 to SF12, within their sampling spread


def nocrc_min_gap(SF):
    """The capped-mean peak gap a frame without a CRC needs to be crc_ok."""
    return _NOCRC_MIN_GAP.get(int(SF), NOCRC_MIN_GAP_DEFAULT)


def _nocrc_score(gaps):
    """The mean of the gaps, each capped at NOCRC_GAP_CAP. Summed in index
    order, as gr-sdrlora's mirror sums them."""
    acc = 0.0
    for g in gaps:
        acc += min(float(g), NOCRC_GAP_CAP)
    return acc / len(gaps) if gaps else math.inf


def _peak_gap(spec, pos):
    """How far one dechirped symbol's peak stands above the largest bin
    outside its main lobe (the peak and its two neighbours), in units of
    the mean power of the bins outside it."""
    p = np.asarray(spec, dtype=np.float64) ** 2
    K = p.size
    keep = np.ones(K, dtype=bool)
    keep[[(pos - 1) % K, pos, (pos + 1) % K]] = False
    rest = p[keep]
    m = float(np.mean(rest))
    if m > 0.0:
        return (float(p[pos]) - float(np.max(rest))) / m
    return math.inf if p[pos] > 0.0 else 0.0


def _frac_offset(spec, pos):
    """The peak's offset from bin `pos`, in bins, from |X| at the peak and
    its larger neighbour: exact for a rectangular-windowed tone, which a
    dechirped symbol is. (A parabola through the three is not: it
    compresses small offsets and exaggerates ones near half a bin.)"""
    K = spec.size
    a = float(spec[(pos - 1) % K])
    b = float(spec[pos])
    c = float(spec[(pos + 1) % K])
    if c >= a:
        return c / (b + c) if (b + c) > 0.0 else 0.0
    return -a / (b + a) if (b + a) > 0.0 else 0.0


def _drift_rate(fracs):
    """Bins per symbol the peaks move, from offsets that wrap at +-0.5.

    NOT UNWRAPPED. At 20 ppm on SF12 the peak moves 0.08 bin a symbol, and
    per-symbol noise pushes an occasional step past half a bin: unwrapping
    then slips a whole bin and drags a fitted line toward zero (-8 ppm
    read for -20). Each offset is a phase instead, z = exp(j*2*pi*f), and
    the rate is the angle of sum z[i+L] * conj(z[i]) over 2*pi*L: a symbol
    demodulated wrong adds zero-mean noise, not a slip. Lag 1 is
    unambiguous for any rate under half a bin a symbol; lags 4 and 16
    refine it, each placed by the one before. Summed in index order with
    math's cos/sin, so gr-sdrlora's mirror can match it.
    """
    n = len(fracs)
    zr = [math.cos(2.0 * math.pi * float(f)) for f in fracs]
    zi = [math.sin(2.0 * math.pi * float(f)) for f in fracs]
    rate = 0.0
    for lag in DRIFT_LAGS:
        if lag >= n:
            break
        cr = 0.0
        ci = 0.0
        for i in range(n - lag):
            # z[i+lag] * conj(z[i])
            cr += zr[i + lag] * zr[i] + zi[i + lag] * zi[i]
            ci += zi[i + lag] * zr[i] - zr[i + lag] * zi[i]
        raw = math.atan2(ci, cr) / (2.0 * math.pi * lag)
        if lag == DRIFT_LAGS[0]:
            rate = raw
        else:
            # the alias of raw (period 1/lag) nearest the coarser estimate
            k = math.floor((rate - raw) * lag + 0.5)
            rate = raw + k / float(lag)
    return rate


def _drift_from_fracs(fracs, SF):
    """The residual sample-clock offset, as a fraction, from the payload
    symbols' fractional peak offsets (_drift_rate): a clock fast by e moves
    the peaks by -e * 2^SF bins per symbol."""
    return -_drift_rate(fracs) / float(1 << int(SF))


def _demod_with_retries(s,SF,BW,OSF,Trise,p_ofs_est,Cfo_est,ih,fec_correct,soft_decision,pseudo_header,ldro,hdr_rows,hop_period,hop_freqs,_sfo):
    """lora_packet_rx, then the drift retries when the frame located and
    failed its CRC. Returns lora_packet_rx's tuple: the first
    attempt's, or the first retry's whose CRC passed."""
    _fr = []
    _res = lora_packet_rx(s,SF,BW,OSF,Trise,p_ofs_est,Cfo_est,ih,fec_correct,soft_decision,pseudo_header,ldro,hdr_rows,hop_period,hop_freqs,_sfo,_fr)
    # THE DRIFT READ FROM THE SIGNAL, when the frame located and failed
    # its CRC; see _drift_from_fracs. Kept only if the CRC then passes, so
    # no frame the first attempt decoded is lost.
    _retry = _DRIFT_RETRY and (not _res[10]) and _res[1] \
        and (not _res[3]) and _res[9] and len(_fr) >= DRIFT_MIN_SYMBOLS
    if _retry and int(OSF) >= DRIFT_MIN_OSF:
        for _ in range(DRIFT_RETRIES):
            _d = _drift_from_fracs(_fr, SF)
            if abs(_d) * len(_fr) * (1 << int(SF)) < DRIFT_MIN_BINS:
                break
            _sfo = _sfo + _d
            _fr = []
            _r2 = lora_packet_rx(s,SF,BW,OSF,Trise,p_ofs_est,Cfo_est,ih,fec_correct,soft_decision,pseudo_header,ldro,hdr_rows,hop_period,hop_freqs,_sfo,_fr)
            if (not _r2[10]) and _r2[1] and _r2[3]:
                _res = _r2
                break
            if _r2[10] or not _r2[1] or len(_fr) < DRIFT_MIN_SYMBOLS:
                break
    elif _retry:
        # Below fs/BW 4 only the FIRST estimate is trusted -- a large
        # drift reads right there, a small residual does not -- and the
        # search runs around it and
        # around the carrier's drift, the attempt already made first.
        _d = _drift_from_fracs(_fr, SF)
        _centres = [(_sfo, False)]
        if abs(_d) * len(_fr) * (1 << int(SF)) >= DRIFT_MIN_BINS:
            _centres.append((_sfo + _d, True))
        _found = False
        for _c, _try_centre in _centres:
            for _g in ((0.0,) if _try_centre else ()) + DRIFT_GRID_PPM:
                _r2 = lora_packet_rx(s,SF,BW,OSF,Trise,p_ofs_est,Cfo_est,ih,fec_correct,soft_decision,pseudo_header,ldro,hdr_rows,hop_period,hop_freqs,_c + _g * 1e-6)
                if (not _r2[10]) and _r2[1] and _r2[3]:
                    _res = _r2
                    _found = True
                    break
            if _found:
                break
    return _res


def lora_packet_rx(s,SF,BW,OSF,Trise,p_ofs_est,Cfo_est,ih=None,fec_correct=True,soft_decision=True,pseudo_header=True,ldro=LDRO_AUTO,hdr_rows=HDR_ROWS_DEFAULT,hop_period=0,hop_freqs=None,sfo=0.0,_fracs=None,_gaps=None):
    """Demodulate one packet at a known offset.

    ih: None for explicit header (read LENGTH/CR/HAS_CRC off the air), or the
        tuple (LENGTH, CR, HAS_CRC) for implicit-header mode, where there is no
        header to read and the parameters must be supplied by the caller.
    sfo: the sample-clock offset as a fraction (Cfo_est / carrier), 0 for
        none. See _sym_start.
    _fracs: a list to receive each payload symbol's fractional peak offset
        (_frac_offset), for rf_decode's drift retry. Private.
    """
    truncated = False
    fs = BW*OSF
    Ts = 1/fs
    N = np.power(2,SF)*OSF
    Nrise = np.ceil(Trise*fs)
    # BW is known here, so the duration-based LDRO rule is applied once and
    # the resolved flag handed to the payload sizing and decode below.
    _de = ldro_enabled(SF, BW, ldro)
    # De-hopped HERE, where the frame offset is already known and before any
    # symbol is demodulated -- so every stage below sees an ordinary
    # single-channel frame and needs no change. The preamble search that
    # produced p_ofs_est ran on the hopped buffer, but nothing before the
    # header is rotated at any hop period, so it saw an unhopped sync
    # pattern: HOP_PERIOD_MIN is a deliberate floor, not a requirement of
    # this stage. See HOP_PERIOD_MIN.
    # The plan is validated before it is applied, as gr-sdrlora's packet_rx
    # does (a nan channel would blank every sample past the first hop), and
    # whenever one is given: hop_period with no channel list is an error,
    # not "no hopping". _check_hops returns (0, ()) for hop_period 0, so an
    # unhopped call costs one comparison.
    hop_period, hop_freqs = _check_hops(hop_period, hop_freqs, fs, BW)
    if hop_period:
        # The HEADER start, which is p_ofs_est + Nrise + 12.25*N whatever the
        # transmitter's preamble length was -- see complex_lora_packet for
        # why the frame start itself is not usable as an anchor here.
        s = _apply_hops(s, int(p_ofs_est + Nrise + 12.25 * N), hop_period,
                        hop_freqs, int(N), fs, invert=True,
                        # the block edges drift with the windows
                        sfo=sfo,
                        ref=p_ofs_est + Nrise + SFO_REF_SYMBOLS * N)
    d0 = chirp(BW/2,N,Ts,-BW/(N*Ts))[0]
    DST = -1
    SRC = -1
    SEQNO = -1
    CR = -1
    HAS_CRC = -1
    # demodulate the header symbols; ofs starts past the preamble
    ofs = int(Nrise+12.25*N)

    # The first interleaver block is 8 symbols at CR 4/8 in BOTH modes; with an
    # implicit header it simply carries no header bits.
    if True:
        CR_hdr = 4
        n_sym_hdr = 4+CR_hdr
        k_hdr_est = np.zeros((n_sym_hdr))
        _hdr_specs = []        # for the no-CRC check, once HAS_CRC is known
        for i in range(0,n_sym_hdr):

            try:
                # A NEGATIVE start index is refused, not indexed: numpy fancy
                # indexing wraps a negative index to the end of the buffer
                # instead of raising, which would read samples from the far
                # end of the capture and report truncated=False.
                #
                # p_ofs_est + ofs only reaches 0 from below at fs/BW >= 16:
                # it equals (m_d0-m_u0)*OSF/8 + s_ofs + 1.25*N, s_ofs is at
                # least 6*N (the history guard in rf_decode) and the folded
                # residual difference is bounded by 4*N. The same guard sits
                # at the same place in gr-sdrlora/lib/lora_rx.cc.
                _st = _sym_start(p_ofs_est, ofs, sfo, Nrise, N)
                if _st < 0:
                    raise IndexError("negative packet offset")
                temp = np.exp(-1j * 2 * np.pi * Cfo_est * Ts * (ofs+np.arange(0,N))) * s[_st+np.arange(0,N, dtype = np.int32)] *d0
            except IndexError:
                k_hdr_est = None
                MAC_CRC_OK = False
                HDR_FCS_OK = False
                k_payload_est = None
                MSG = None
                truncated = True
                return k_hdr_est,HDR_FCS_OK,k_payload_est,MAC_CRC_OK,MSG,DST,SRC,SEQNO,CR,HAS_CRC,truncated,ofs


            ofs = ofs+N
            # Phase 0, K bins exactly, spelled the same way as the payload
            # loop below; see the decimation comment there.
            spec = np.abs(np.fft.ifft(temp[0:OSF * (2 ** int(SF)):OSF]))
            pos = np.argmax(spec)
            k_hdr_est[i]=pos-1
            _hdr_specs.append((spec, int(pos)))


        #in case of header checksum failure, the message is taken as lost/corrupted
        (HDR_FCS_OK,LENGTH,HAS_CRC,CR,PAYLOAD_bits_hdr) = lora_header_decode(
            SF, k_hdr_est, fec_correct,
            0 if ih is not None else int(hdr_rows))
        if ih is not None:
            # Implicit header: the parameters come from the caller, not the
            # air. PAYLOAD_bits_hdr above is the whole first block's payload.
            (LENGTH, CR, HAS_CRC) = ih
            HDR_FCS_OK = True                  # no FCS exists to check
        if not HDR_FCS_OK:
            k_hdr_est = None
            MAC_CRC_OK = False
            k_payload_est = None
            MSG = None

            return k_hdr_est,HDR_FCS_OK,k_payload_est,MAC_CRC_OK,MSG,DST,SRC,SEQNO,CR,HAS_CRC,truncated,ofs
    if False:
        pass

    # ---------------- payload demodulation (both header modes)
    n_bits_hdr = (PAYLOAD_bits_hdr.size)
    n_sym_payload = lora_payload_n_sym(SF,LENGTH,HAS_CRC,CR,n_bits_hdr,_de)
    # ZERO is legal and must decode: it means the whole payload fitted in the
    # bits the first interleaver block already carries, which happens for
    # short implicit frames (any payload up to 3 bytes at SF9, and 0-1 bytes
    # at SF12 with the default pseudo-header).
    if n_sym_payload < 0:
        MAC_CRC_OK = False
        return k_hdr_est,HDR_FCS_OK,None,MAC_CRC_OK,None,DST,SRC,SEQNO,CR,HAS_CRC,truncated,ofs

    k_payload_est = np.zeros((n_sym_payload))
    # K bins per dechirped symbol; the decimated spectrum has exactly this length
    mag_payload = np.zeros((n_sym_payload, 2 ** int(SF))) if soft_decision else None
    # The peak gaps (_peak_gap) of every symbol after the sync word, which
    # the no-CRC check below reads; only gathered when it can apply.
    _want_gaps = (_gaps is not None) or (_NOCRC_CHECK and not HAS_CRC)
    if _gaps is None:
        _gaps = []
    if _want_gaps:
        _gaps.extend(_peak_gap(sp, ps) for sp, ps in _hdr_specs)
    for i in range(0, n_sym_payload):
        try:
            # Negative start refused rather than wrapped; see the header loop
            # above and gr-sdrlora/lib/lora_rx.cc's demod() guard.
            _st = _sym_start(p_ofs_est, ofs, sfo, Nrise, N)
            if _st < 0:
                raise IndexError("negative packet offset")
            temp = np.exp(-1j*2*np.pi*Cfo_est*Ts*(ofs+np.arange(0,N, dtype = np.int32))) * s[_st+np.arange(0,N, dtype = np.int32)] * d0
        except IndexError:
            # IndexError only, not a bare except: samples_decoding reads
            # `truncated` as "a packet that has not finished arriving" and
            # stops the buffer walk, so only the slice running off the end
            # of the buffer may reach it; anything else (KeyboardInterrupt,
            # a genuine bug) must propagate.
            k_hdr_est = None
            MAC_CRC_OK = False
            k_payload_est = None
            MSG = None

            truncated = True
            return k_hdr_est,HDR_FCS_OK,k_payload_est,MAC_CRC_OK,MSG,DST,SRC,SEQNO,CR,HAS_CRC,truncated,ofs

        ofs = ofs+N
        # Decimate the dechirped block to K bins, starting at sample 0.
        #
        # The phase is not a free choice. A dechirped LoRa symbol is TWO
        # tone segments, not one: the received
        # chirp wraps at the symbol value, so the part before the wrap beats
        # down to k*BW/K and the part after to k*BW/K - BW. Both land in the
        # same FFT bin -- that fold is what makes the symbol value cyclic --
        # and they are summed COHERENTLY there.
        #
        # Decimating from index p delays the block by p samples, which gives
        # those two segments a RELATIVE phase of 2*pi*p/OSF, because they
        # differ in frequency by exactly BW and p samples of BW is p/OSF of a
        # cycle. At p/OSF = 1/2 they arrive 180 degrees apart and cancel.
        #
        # Not completely -- that needs a symbol value that splits the block
        # equally. Averaged over all K values at SF7, the correct bin keeps
        # 1.00, 0.95, 0.81, 0.63, 0.50 of its amplitude as p/OSF goes 0 to
        # 1/2, and the argmax is right 100%, 100%, 100%, 65%, 52% of the time.
        # Half the symbols wrong is well past what the FEC can carry, and the
        # loss depends only on p/OSF -- which is why one sample of phase is
        # invisible at OSF 8 and fatal at OSF 2.
        #
        # p = 1 is therefore 1/8 of the way to that notch at fs/BW = 8 and
        # harmless, but sits exactly IN it at fs/BW = 2; the notch tracks
        # p = OSF/2 at every rate. (With LDRO, SF11/SF12 would hide a
        # one-bin error, because symbol values are quantised to every
        # fourth bin.)
        #
        # Slicing to OSF*K rather than -1 makes the length exactly K at every
        # OSF, as the soft-decision store below requires.
        spec = np.abs(np.fft.ifft(temp[0:OSF * (2 ** int(SF)):OSF]))
        pos = np.argmax(spec)
        k_payload_est[i] = pos-1
        if _fracs is not None:
            _fracs.append(_frac_offset(spec, int(pos)))
        if _want_gaps:
            _gaps.append(_peak_gap(spec, int(pos)))
        if mag_payload is not None:
            mag_payload[i] = spec


    (MAC_CRC_OK,DST,SRC,SEQNO,MSG,HAS_CRC) = lora_payload_decode(SF,k_payload_est,PAYLOAD_bits_hdr,HAS_CRC,CR,LENGTH,fec_correct,mag_payload,pseudo_header,_de)
    # THE NO-CRC CHECK: see _NOCRC_CHECK. MSG still comes out,
    # as for a failed CRC, so report_crc_errors counts it.
    if (_NOCRC_CHECK and MAC_CRC_OK and not HAS_CRC
            and _nocrc_score(_gaps) < nocrc_min_gap(SF)):
        MAC_CRC_OK = False
    # if MAC_CRC_OK:
    #
    # 	#fprintf('DST: #d SRC: #d SEQ: #d LEN: #d DATA: "',...
    # 		#DST,SRC,SEQNO,LENGTH)
    # 	for i in range(1,len(MSG)+1):
    # 		if MSG[i-1]>=32 and MSG[i-1]<=127:
    # 			#fprintf('#c',MSG(i))
    # 			pass
    # 		else:
    # 			pass
    # 			#fprintf('\\#d',MSG(i))
    return k_hdr_est, HDR_FCS_OK, k_payload_est, MAC_CRC_OK, MSG, DST,SRC,SEQNO, CR,HAS_CRC, truncated,ofs






def lora_header_decode(SF,k_hdr,fec_correct=True,hdr_rows=HDR_ROWS_DEFAULT):
    """Decode the first interleaver block (8 symbols, CR 4/8).

    hdr_rows: how many codewords carry header bits. 5 for an explicit header;
        0 for an implicit one, where there is no header and no FCS to check
        and every codeword is payload.
    """

    if SF < 7 and hdr_rows > 0:
        # SF5 and SF6 cannot carry an explicit header at all: the header block
        # is the first SF-2 codewords at CR 4/8, i.e. 4*(SF-2) data bits, and
        # the LoRa header needs 20. SF7 gives exactly 20; SF6 gives 16 and SF5
        # gives 12. This is why the SX1276 datasheet requires implicit header
        # mode at SF6, and why gr-lora_sdr cannot use an explicit header at
        # SF5/SF6 either. The full five-value tuple is returned so the caller
        # can unpack it.
        #
        # Only when hdr_rows > 0: with an implicit header the same 8-symbol
        # block exists and carries payload, which SF5/SF6 handle fine.
        return False, -1, False, -1, -1






    DE_hdr = 1
    PPM = SF-2*DE_hdr
    (_,degray) = gray_lut(PPM)

    #np.fft(downsample(s_rx.*chirp)) demodulates the signal
    CR_hdr = 4
    n_sym_hdr = 4+CR_hdr
    intlv_hdr_size = PPM*n_sym_hdr
    bits_est = np.zeros((intlv_hdr_size), dtype = np.uint8)
    K = np.power(2,SF)
    for sym in range(0,n_sym_hdr):
        # gray decode
        # K-1 HERE, K-2 IN THE PAYLOAD LOOP, and the difference is
        # deliberate. Both loops store pos-1 from the same demod, so at zero
        # timing error this argument evaluates to g + 0.25 while the
        # payload's evaluates to exactly g. Rounding absorbs the quarter-bin
        # bias, so both round-trip correctly; what differs is the tolerance
        # window. The header survives bin errors in (-1, +3) -- asymmetric --
        # and the payload |e| < step/2, symmetric.
        #
        # Do NOT 'harmonise' the two constants. Making this K-2 for symmetry
        # shifts the header's timing-tolerance window by a bin and changes
        # the decoded bits. lora_rx.cc carries the same pair
        # for the same reason: header_decode() maps (K - 1 - k) / 4 and
        # packet_rx()'s payload loop (K - 2 - k) / 2^(2*DE), as does
        # bit_label_mask() for the soft path.
        bin = int(np.round((K-1-k_hdr[sym])/4))
        bits_est[(sym)*PPM+np.arange(0,PPM, dtype = np.int32)] = num2binary(degray[np.mod(bin,np.power(2,PPM))],PPM)


    # interleaver del brevetto
    S = np.reshape(bits_est,(PPM,4+CR_hdr),order = 'F').transpose()
    C = np.zeros((PPM,4+CR_hdr),dtype = np.uint8)
    for ii in range(0,PPM):
        for jj in range (0,4+CR_hdr):
            C[ii,jj] = S[jj,np.mod(ii+jj,PPM)]



    # row flip
    C = np.flip(C,0)

    # The header is protected by the (8,4) code as well as its own 5-bit FCS.
    # Correcting first means a single-bit hit does not cost the whole packet.
    if fec_correct:
        C = np.ascontiguousarray(C)
        hamming_correct(C, CR_hdr)

    # header parity check matrix (reverse engineered through brute force search)
    header_FCS = np.array(([1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,0], \
        [0,1,0,1,0],[0,1,0,0,1],[0,0,1,1,0],[0,0,1,0,1],[0,0,0,1,1],[0,0,1,1,1], \
        [0,1,0,1,1]))

    if hdr_rows < 0 or hdr_rows > HDR_ROWS_DEFAULT:
        # BOTH ENDS. Same 20-bit ceiling as _check_hdr_rows, repeated
        # because this builder is public and reachable without going
        # through it. The FCS below is read from codewords 3 and 4 and
        # the fields from 0 to 2, so there is nothing for a sixth header
        # codeword to hold. A negative value would size the output larger
        # than the block it is filled from and index C from a negative row,
        # which numpy wraps instead of raising.
        raise ValueError(
            "hdr_rows must be 0 (implicit) or 1..%d: the explicit header "
            "is 20 bits, which is %d codewords of 4 at CR 4/8, got %r"
            % (HDR_ROWS_DEFAULT, HDR_ROWS_DEFAULT, hdr_rows))
    if hdr_rows == 0:
        # Implicit header: no header bits, so no FCS exists and every codeword
        # is payload. The caller already knows LENGTH, CR and HAS_CRC.
        n_bits_hdr = PPM * 4
        PAYLOAD_bits = np.zeros((1, n_bits_hdr), dtype=np.uint8)
        for i in range(0, PPM):
            PAYLOAD_bits[0, i * 4 + np.arange(0, 4, dtype=np.int32)] = C[i, 0:4]
        return True, -1, False, -1, PAYLOAD_bits

    FCS_chk = np.mod(np.concatenate((C[0,np.arange(3,-1,-1, dtype = np.int32)],C[1,np.arange(3,-1,-1)],C[2,np.arange(3,-1,-1, dtype = np.int32)],np.array([C[3,0]],dtype=np.uint8),C[4,np.arange(3,-1,-1, dtype = np.int32)])) @ [np.concatenate((header_FCS,np.eye(5)),0)],2)
    FCS_OK = not np.any(FCS_chk)
    if not FCS_OK:
        LENGTH = -1
        HAS_CRC = False
        CR = -1
        PAYLOAD_bits = -1
      

    else:
        # header decoding
        # Header=char(['len:',C(1,4:-1:1)+'0',C(2,4:-1:1)+'0',...
        # 	' CR:',C(3,4:-1:2)+'0',...
        # 	' MAC-CRC:',C(3,1)+'0',...
        # 	' HDR-FCS:',C(4,4:-1:1)+'0',C(5,4:-1:1)+'0'])
        #fprintf('Header Decode: #s [OK]\n',Header)


        LENGTH = bit2uint8(np.concatenate((C[1,0:4],C[0,0:4])))
        HAS_CRC = int(C[2,0]) # HAS_CRC
        CR = bit2uint8(C[2,1:4])
        # The CR field is 3 bits but only 1..4 are legal, so an out-of-range
        # value means the header is corrupt whatever its FCS says. Rejecting it
        # here both strengthens the FCS and stops CR=0 (or 5..7) reaching the
        # FEC decoders, which have no parity matrix for it.
        if CR < 1 or CR > 4:
            return False, -1, False, -1, -1
        # FCS_chk above decides FCS_OK; the five received header FCS bits
        # are C[4, 0:4] and C[3, 0].

        # 4 * hdr_rows in both the size and the loop: the header occupies the
        # first hdr_rows codewords and the payload starts after them, as in
        # gr-sdrlora/lib/lora_rx.cc's `for (i = hdr_rows; ...)`.
        _rows = int(hdr_rows)
        n_bits_hdr = PPM*4 - 4*_rows
        PAYLOAD_bits=np.zeros((1,n_bits_hdr),dtype=np.uint8)
        for i in range(_rows,PPM):
            #C(i,4+(1:CR_hdr)) = mod(C(i,1:4)*Hamming_hdr,2)
            PAYLOAD_bits[0,(i-_rows)*4+np.arange(0,4, dtype = np.int32)]=C[i,0:4]


    return FCS_OK,LENGTH,HAS_CRC,CR,PAYLOAD_bits


def num2binary(num,length = 0):

    num = np.array([num],dtype=np.uint16)
    num = np.flip(num.view(np.uint8))
    num = np.unpackbits(num)
    return num[-length:]



def lora_payload_n_sym(SF,LENGTH,MAC_CRC,CR,n_bits_hdr,DE=None):

    # bigger spreading factors (11 and 12) use 2 less bits per symbol
    if DE is None:
        # No caller-supplied flag: fall back to the spreading-factor rule,
        # which is correct at BW = 125 kHz; lora_packet() and
        # lora_packet_rx() pass the duration-based value instead.
        DE = 1 if SF > 10 else 0
    PPM = SF-2*DE
    n_bits_blk = PPM*4
    n_bits_tot = 8*LENGTH+16*MAC_CRC
    # Signed, and clamped: same unsigned-wrap hazard as lora_payload_init.
    # A LENGTH read from a corrupt header can be anything, so the receiver
    # reaches this with differences the encoder never would.
    n_blk_tot = max(0, int(np.ceil(
        (int(n_bits_tot) - int(n_bits_hdr)) / float(n_bits_blk))))
    n_sym_blk = 4+CR
    n_sym_payload = int(n_blk_tot*n_sym_blk)
    return n_sym_payload




# Parity matrices, indexed by CR. Identical to the ones the encoder applies in
# lora_payload(); the header always uses the CR=4 matrix.
_HAMMING_P = {
    1: np.array(([1], [1], [1], [1]), dtype=np.int64),
    2: np.array(([1, 0], [1, 1], [1, 1], [0, 1]), dtype=np.int64),
    3: np.array(([1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]), dtype=np.int64),
    4: np.array(([1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]), dtype=np.int64),
}

_SYNDROME_LUT = {}


def _syndrome_lut(CR):
    """syndrome -> the codeword position whose flip explains it.

    Only unambiguous syndromes are listed. At CR 4/5 and 4/6 the code has
    d_min = 2, every single-bit syndrome collides, and the table is empty --
    those rates can detect an error but genuinely cannot locate it.
    """
    if CR in _SYNDROME_LUT:
        return _SYNDROME_LUT[CR]
    P = _HAMMING_P[CR]
    n = 4 + CR
    weights = 1 << np.arange(CR)
    seen = {}
    for pos in range(n):
        e = np.zeros(n, dtype=np.int64)
        e[pos] = 1
        syn = (e[4:] ^ ((e[0:4] @ P) % 2)) % 2
        key = int(syn @ weights)
        seen.setdefault(key, []).append(pos)
    lut = {k: v[0] for k, v in seen.items() if len(v) == 1 and k != 0}
    _SYNDROME_LUT[CR] = lut
    return lut


def hamming_correct(C, CR):
    """Correct single-bit errors in the de-interleaved codewords `C`, in place.

    C is (n_codewords, 4 + CR) with the four data bits first. Returns
    (n_corrected, n_uncorrectable): rows whose syndrome pinned a single bit, and
    rows that failed the parity check but could not be located.
    """
    CR = int(CR)
    if CR not in _HAMMING_P or C.size == 0:
        return 0, 0
    P = _HAMMING_P[CR]
    data = C[:, 0:4].astype(np.int64)
    par = C[:, 4:4 + CR].astype(np.int64)
    syn = (par ^ ((data @ P) % 2)) % 2
    bad = np.flatnonzero(syn.any(axis=1))
    if bad.size == 0:
        return 0, 0

    lut = _syndrome_lut(CR)
    weights = 1 << np.arange(CR)
    corrected = uncorrectable = 0
    for r in bad:
        pos = lut.get(int(syn[r] @ weights))
        if pos is None:
            uncorrectable += 1          # detected but not locatable
        else:
            C[r, pos] ^= 1
            corrected += 1
    return corrected, uncorrectable


_BIT_LABEL_CACHE = {}
_CODEBOOK_CACHE = {}


def _bit_label_mask(SF, DE):
    """(PPM, K) boolean mask: does FFT bin `pos` carry a 1 at bit `b`?

    Built by running the hard decoder's own bin -> bits formula over every bin,
    so the soft path cannot drift from the hard path's symbol mapping.
    """
    key = (int(SF), int(DE))
    if key in _BIT_LABEL_CACHE:
        return _BIT_LABEL_CACHE[key]
    SF = int(SF)
    DE = int(DE)
    K = 2 ** SF
    PPM = SF - 2 * DE
    degray = gray_lut(PPM)[1]
    mask = np.zeros((PPM, K), dtype=bool)
    for pos in range(K):
        k = pos - 1
        b = round((K - 2 - k) / 2 ** (2 * DE))
        label = degray[int(np.mod(b, 2 ** PPM))]
        mask[:, pos] = num2binary(label, PPM).astype(bool)
    _BIT_LABEL_CACHE[key] = mask
    return mask


def symbol_llr(mag, SF, DE):
    """Per-bit LLRs from dechirped FFT magnitudes.

    mag: (n_sym, K) magnitudes straight out of the demodulator.
    Returns (n_sym, PPM) LLRs, positive meaning the bit is more likely 0.

    Max-log approximation: for each bit, compare the strongest bin that would
    carry a 0 against the strongest that would carry a 1. Under noncoherent
    detection the winning bin is Rice-distributed and the rest Rayleigh, so the
    difference of squared magnitudes is the natural metric.
    """
    SF = int(SF)
    DE = int(DE)
    if SF < 5 or SF > 12:
        raise ValueError("symbol_llr: SF must be 5..12, got %d" % SF)
    if DE < 0 or DE > 1:
        raise ValueError(
            "symbol_llr: DE must be 0 or 1, got %d -- it is the low-data-rate "
            "flag and the label mask is PPM = SF - 2*DE bits wide, which "
            "stops meaning anything outside that range" % DE)
    mask = _bit_label_mask(SF, DE)              # (PPM, K)
    m = np.asarray(mag, dtype=np.float64) ** 2  # (n_sym, K)
    # The mask is 2^SF wide and is indexed by the magnitude columns, so the
    # width is checked up front, even for zero rows, with a named error
    # instead of numpy's boolean-index IndexError. Mirrors the check in
    # gr::sdrlora::symbol_llr.
    if m.ndim != 2 or m.shape[1] != mask.shape[1]:
        raise ValueError(
            "symbol_llr: mag must be (n_sym, %d) for SF%d, got %r -- one "
            "magnitude per FFT bin of a dechirped symbol"
            % (mask.shape[1], SF, (m.shape,)))
    PPM = mask.shape[0]
    llr = np.empty((m.shape[0], PPM), dtype=np.float64)
    # Loop the PPM bits (7..10 of them) rather than broadcasting to
    # (n_sym, PPM, K): that temporary is ~166 MB at SF12 with a full payload.
    for b in range(PPM):
        ones = mask[b]
        llr[:, b] = m[:, ~ones].max(axis=1) - m[:, ones].max(axis=1)
    return llr


def _codebook(CR):
    """(16, 4+CR) codewords and their +/-1 signs, for ML decoding."""
    CR = int(CR)
    if CR in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[CR]
    if CR not in _HAMMING_P:
        # Defensive: hamming_correct() already guards this, and the header
        # decoder rejects an out-of-range CR, but the soft path must not be
        # the one place a corrupt header can raise.
        raise ValueError("coding rate %r is outside 1..4" % (CR,))
    P = _HAMMING_P[CR]
    words = np.zeros((16, 4 + CR), dtype=np.uint8)
    for v in range(16):
        d = np.array([(v >> (3 - i)) & 1 for i in range(4)], dtype=np.uint8)
        words[v, 0:4] = d
        words[v, 4:4 + CR] = (d.astype(np.int64) @ P) % 2
    signs = 1.0 - 2.0 * words.astype(np.float64)     # bit 0 -> +1
    _CODEBOOK_CACHE[CR] = (words, signs)
    return words, signs


def hamming_soft_decode(L, CR):
    """Maximum-likelihood decode of soft codewords.

    L: (n_codewords, 4+CR) LLRs, positive meaning bit 0.
    Returns (n_codewords, 4) hard data bits.

    Exhaustive over the 16 codewords, which is both optimal and cheap. Works at
    every coding rate, including the d_min = 2 rates where hard-decision
    syndrome decoding cannot locate an error.
    """
    words, signs = _codebook(CR)
    metric = np.asarray(L, dtype=np.float64) @ signs.T     # (rows, 16)
    return words[np.argmax(metric, axis=1), 0:4]


def lora_payload_decode(SF,k_payload,PAYLOAD_hdr_bits,HAS_CRC,CR,LENGTH_FROM_HDR,fec_correct=True,mag_payload=None,pseudo_header=True,DE=None):


    # hamming parity check matrices 
    Hamming_P1 = np.array(([1],[1],[1],[1]), dtype=np.uint8)
    Hamming_P2 = np.array(([1,0], [1,1], [1,1], [0,1]), dtype=np.uint8)
    Hamming_P3 = np.array(([1,0,1], [1,1,1], [1,1,0], [0,1,1]), dtype=np.uint8)
    Hamming_P4 = np.array(([1,0,1,1], [1,1,1,0], [1,1,0,1], [0,1,1,1]),dtype = np.uint8)

    if CR == 1:
        Hamming = Hamming_P1
    elif CR == 2:
        Hamming = Hamming_P2
    elif CR == 3:
        Hamming = Hamming_P3
    elif CR == 4:
        Hamming = Hamming_P4
    else:
        # A named error for a CR outside 1..4. lora_packet_rx and
        # lora_payload_decode are public, and decode() validates ahead of
        # them but a direct caller does not. The C++ throws for the same
        # input.
        raise ValueError(
            "CR must be 1, 2, 3 or 4 (coding rates 4/5 to 4/8), got %r -- "
            "there is no Hamming parity matrix for it" % (CR,))

    if DE is None:
        # No caller-supplied flag: fall back to the spreading-factor rule,
        # which is correct at BW = 125 kHz; lora_packet() and
        # lora_packet_rx() pass the duration-based value instead.
        DE = 1 if SF > 10 else 0
    PPM = SF-2*DE
    n_sym_blk = (4+CR)
    n_bits_blk = PPM*4
    intlv_blk_size = PPM*n_sym_blk
    [_,degray] = gray_lut(PPM)
    K = np.power(2,SF)
    n_sym_payload = len(k_payload)
    n_blk_tot = int(n_sym_payload/n_sym_blk)
    try:
        PAYLOAD = np.concatenate((np.squeeze(PAYLOAD_hdr_bits),np.zeros((int(n_bits_blk*n_blk_tot)), dtype = np.uint8)))
    except ValueError:
        PAYLOAD = np.zeros((int(n_bits_blk*n_blk_tot)), dtype = np.uint8)
    payload_ofs = (PAYLOAD_hdr_bits.size)
    # Soft path: LLRs for every payload symbol, computed once.
    llr_all = None
    if mag_payload is not None:
        llr_all = symbol_llr(mag_payload, SF, DE)      # (n_sym_payload, PPM)

    for blk in range (0,n_blk_tot):

        if llr_all is not None:
            # The de-interleaver is a permutation, so LLRs go through exactly
            # the same reshape / index / flip as the bits would.
            llr_blk = llr_all[blk*n_sym_blk:(blk+1)*n_sym_blk, :].reshape(-1)
            Sl = np.reshape(llr_blk,(PPM, (4+CR)),order = 'F').transpose()
            Lc = np.zeros((PPM,4+CR), dtype=np.float64)
            for ii in range(0,PPM):
                for jj in range(0,4+CR):
                    Lc[ii,jj] = Sl[jj,np.mod(ii+jj,PPM)]
            Lc = np.flip(Lc,0)
            data = hamming_soft_decode(Lc, CR)
            for k in range(0,PPM):
                PAYLOAD[payload_ofs+np.arange(0,4, dtype = np.int32)] = data[k]
                payload_ofs = payload_ofs+4
            continue

        bits_blk = np.zeros((intlv_blk_size))
        for sym in range(0,n_sym_blk):
            bin = round((K-2-k_payload[(blk)*n_sym_blk+sym])/np.power(2,(2*DE)))
            # gray decode
            bits_blk[(sym)*PPM+np.arange(0,PPM, dtype = np.int32)] = num2binary(degray[int(np.mod(bin,np.power(2,PPM)))],PPM)


        # interleaving



        S = np.reshape(bits_blk,(PPM, (4+CR)),order = 'F').transpose()
        C = np.zeros((PPM,4+CR),dtype=np.uint8)
        for ii in range(0,PPM):
            for jj in range(0,4+CR):
                C[ii,jj]= S[jj,np.mod(ii+jj,PPM)]


        # row flip
        C = np.flip(C,0)

        # Use the parity bits before discarding them. Without this the FEC is
        # transmitted and thrown away, and the coding rate buys nothing.
        if fec_correct:
            C = np.ascontiguousarray(C)
            hamming_correct(C, CR)

        for k in range(0,PPM):
            PAYLOAD[payload_ofs+np.arange(0,4, dtype = np.int32)] = C[k,0:4]
            payload_ofs = payload_ofs+4



    # ----------------------------------------------------------- WHITENING
    W = np.array([1,1,1,1,1,1,1,1], dtype=np.uint8)
    W_fb = np.array([0,0,0,1,1,1,0,1], dtype = np.uint8)
    for k in range (1,int(np.floor(len(PAYLOAD)/8) + 1)):
        PAYLOAD[(k-1)*8+np.arange(0,8, dtype = np.int32)] = np.mod(PAYLOAD[(k-1)*8+np.arange(0,8, dtype = np.int32)]+W,2)
        W1 = np.array([np.mod(np.sum(W*W_fb),2)])
        W = np.concatenate((W1,W[0:-1]))




    if not pseudo_header:
        # ------------------------------------------------ standard framing
        # No 4-byte prefix: the PHY header's LENGTH is the message length and
        # every payload byte belongs to the message. This is what any other
        # LoRa implementation transmits, and what is needed to receive from
        # one: with the pseudo-header, a standard frame's first four payload
        # bytes would be consumed as DST/SRC/SEQNO/LENGTH. The CRC itself is
        # the same: it covers the whole payload, which is the standard
        # definition.
        LENGTH = int(LENGTH_FROM_HDR)
        # LENGTH 0 with the CRC flag set is refused rather than decoded: a
        # LoRa CRC is not defined over zero bytes, so CRC16 raises, and this
        # header is one that passed its 5-bit FCS by chance on noise. In the
        # C++ port the same raise would leave work() through GNU Radio's
        # thread body and stop the flowgraph. A false lock must look like a
        # failed lock, not like an exception. LENGTH 0 WITHOUT the flag
        # stays legal: that is a real zero-byte frame. 1 byte with a CRC also stays legal -- gr-lora_sdr
        # sets the flag and appends no checksum below 2 bytes, and CRC16's
        # short-state table is correct from 1 byte up.
        if (LENGTH < 0 or (HAS_CRC and LENGTH < 1)
                or (LENGTH + 2 * bool(HAS_CRC)) * 8 > len(PAYLOAD)):
            return False, -1, -1, -1, None, HAS_CRC
        MSG = np.zeros(LENGTH, dtype=np.uint8)
        for i in range(LENGTH):
            MSG[i] = bit2uint8(PAYLOAD[8 * i + np.arange(0, 8, dtype=np.int32)])
        if not HAS_CRC:
            return True, -1, -1, -1, MSG, HAS_CRC
        ok = not np.any(PAYLOAD[8 * LENGTH + np.arange(0, 16, dtype=np.int32)]
                        != CRC16(PAYLOAD[0:8 * LENGTH]))
        # DST/SRC/SEQNO do not exist on air in this mode.
        # MSG comes out whether or not the CRC checked out -- `ok` is what says
        # which, and a caller counting corrupted packets needs the bytes. The
        # pseudo-header branch below returns them the same way.
        return ok, -1, -1, -1, MSG, HAS_CRC

    #NOTE HOW THE TOTAL LENGTH IS 4 BYTES + THE PAYLOAD LENGTH
    #INDEED, THE FIRST 4 BYTES ENCODE DST, SRC, SEQNO AND LENGTH INFOS
    # A header that passed its FCS by chance -- which happens on noise -- can
    # give a LENGTH that sizes PAYLOAD below this 32-bit pseudo-header. The
    # sanity check further down runs too late to protect these four reads.
    if len(PAYLOAD) < 32:
        return False, -1, -1, -1, None, HAS_CRC

    DST = bit2uint8(PAYLOAD[0:8])
    SRC = bit2uint8(PAYLOAD[8+np.arange(0,8, dtype = np.int32)])
    SEQNO = bit2uint8(PAYLOAD[8*2+np.arange(0,8, dtype = np.int32)])
    LENGTH = bit2uint8(PAYLOAD[8*3+np.arange(0,8, dtype = np.int32)])
    if (LENGTH == 0):
    	LENGTH = LENGTH_FROM_HDR
    	
    
    MSG_LENGTH = LENGTH-4
    if (((LENGTH+2)*8 > len(PAYLOAD) and HAS_CRC) or (LENGTH*8 > len(PAYLOAD) and not (HAS_CRC)) or LENGTH<4):
        MAC_CRC_OK = False

        return MAC_CRC_OK, DST, SRC, SEQNO, None, HAS_CRC

    MSG=np.zeros((int(MSG_LENGTH)), dtype = np.uint8)
    for i in range (0,int(MSG_LENGTH)):
        MSG[i]=bit2uint8(PAYLOAD[8*(4+i)+np.arange(0,8, dtype = np.int32)])

    if not HAS_CRC:
        MAC_CRC_OK = True
    else:
        #fprintf('CRC-16: 0x#02X#02X ',...
            #PAYLOAD(8*LENGTH+(1:8))*2.^(0:7)',...
            #PAYLOAD(8*LENGTH+8+(1:8))*2.^(0:7)')
        # Computed once: CRC16 is a pure-Python loop over 8*LENGTH bits.
        _crc = CRC16(PAYLOAD[0:8*LENGTH])
        if np.any(PAYLOAD[8*LENGTH+np.arange(0,16, dtype = np.int32)] != _crc):
            #fprintf(' [CRC FAIL]\n')
            MAC_CRC_OK = False
        else:
            #fprintf(' [CRC OK]\n')
            MAC_CRC_OK = True



    return MAC_CRC_OK,DST,SRC,SEQNO,MSG,HAS_CRC




def bit2uint8(bits):
    # int(), not uint8: LENGTH feeds bit-offset arithmetic that overflows 8 bits
    return int(np.packbits(bits, bitorder='little')[0])




# LoRa preamble (reverse engineered from a real signal, is actually different from the one described in the patent)
def lora_preamble(n_preamble, k1, k2, BW, K, OSF, t0_frac=0, phi0=0):
    (u0, phi) = lora_chirp(+1, 0, BW, K, OSF, t0_frac, phi0)
    (u1, phi) = lora_chirp(+1, k1, BW, K, OSF, t0_frac, phi)
    (u2, phi) = lora_chirp(+1, k2, BW, K, OSF, t0_frac, phi)
    (d0, phi) = lora_chirp(-1, 0, BW, K, OSF, t0_frac, phi)
    (d0_4, phi) = chirp(BW / 2, K * OSF / 4, 1 / (BW * OSF), -np.power(BW, 2) / K, t0_frac, phi)
    s = np.concatenate((np.tile(u0, (n_preamble)), u1, u2, d0, d0, d0_4))
    return s, phi



# Number of (symbol x sample) elements lora_symbols() works on at a time.
# This is purely a cache-tuning knob -- the output is identical for any value.
# Keeping the intermediates inside L2 matters far more than making the batches
# big: 8192 runs up to 1.67x faster than 1<<20 on SF7-SF12, because the larger
# batch spills to memory.
_SYMBOL_CHUNK_ELEMS = 1 << 13


def _cis(phase, out=None):
    """exp(1j*phase) for real `phase`.

    Bitwise identical to np.exp(1j*phase) but ~1.5x faster: the complex
    exponential evaluates exp(0) for the real part and allocates a temporary,
    whereas this writes cos/sin directly into the result's real and imag views.
    """
    if out is None:
        out = np.empty(np.shape(phase), dtype=np.complex128)
    np.cos(phase, out=out.real)
    np.sin(phase, out=out.imag)
    return out


def lora_symbols(ks, BW, K, OSF, t0_frac=0, phi0=0):
    """Modulate a sequence of up-chirp symbols in one vectorised pass.

    Equivalent to calling lora_chirp(+1, k, BW, K, OSF, t0_frac, phi) for every
    k in `ks`, feeding each returned phase into the next call, and concatenating
    the results -- and bit-identical to doing so.

    Returns (samples, phi_end).
    """
    K = int(K)
    OSF = int(OSF)
    # Every symbol is a bin index, so it has to be inside the alphabet: the
    # first chirp segment is k*OSF samples and the second K-k, and past K the
    # second length goes negative. Mirrors the check in gr::sdrlora::modulate.
    # Integral first: int(-0.5) is 0, so a fractional index would slip past
    # the `min() < 0` test and make the first chirp segment negative in
    # length. The code below relies on ks being exactly integral.
    _ks = np.asarray(ks)
    if _ks.size and not np.all(np.equal(np.mod(_ks, 1), 0)):
        raise ValueError(
            "lora_symbols: every symbol must be a whole number; got a "
            "fractional one. A symbol is an FFT bin index.")
    if _ks.size and (int(_ks.min()) < 0 or int(_ks.max()) >= K):
        raise ValueError(
            "lora_symbols: every symbol must be 0..%d for K=%d, got "
            "min %d max %d -- a symbol is an FFT bin index and outside the "
            "alphabet the two chirp segments cannot both have a length"
            % (K - 1, K, int(_ks.min()), int(_ks.max())))
    N = K * OSF
    Ts = 1.0 / (BW * OSF)
    T = N * Ts
    Df = BW / T                       # mu = +1 throughout the LoRa uplink
    two_pi = 2 * np.pi

    ks = np.asarray(ks, dtype=np.float64).ravel()
    n_sym = ks.size
    if n_sym == 0:
        return np.empty(0, dtype=np.complex128), phi0

    # Segment geometry. N1 + N2 == N for every symbol.
    n1 = ks * OSF                     # exactly integral: ks and OSF are integral
    t1 = n1 * Ts
    t2 = (N - n1) * Ts
    f1 = BW * (0.5 - ks / K)
    f2 = -BW / 2.0

    # The two phase-advance terms of each segment, kept SEPARATE. chirp()
    # computes phi_fin as ((phi_ini + a) + b), so pre-summing a + b and adding
    # once re-associates the addition and shifts the result by an ULP -- which
    # then tips the float32 rounding of a handful of output samples.
    a1 = two_pi * f1 * t1
    b1 = np.pi * Df * (t1 * t1)
    a2 = two_pi * f2 * t2
    b2 = np.pi * Df * (t2 * t2)

    # Accumulate the phase chain sequentially, in chirp()'s exact add order.
    # This loop is O(n_sym) scalar arithmetic; the expensive O(n_sym * N)
    # per-sample work below is what gets vectorised.
    phi_start = np.empty(n_sym, dtype=np.float64)
    phi_mid = np.empty(n_sym, dtype=np.float64)
    phi = phi0
    for i in range(n_sym):
        phi_start[i] = phi
        phi = (phi + a1[i]) + b1[i]
        phi_mid[i] = phi
        phi = (phi + a2[i]) + b2[i]
    phi_end = phi

    n1 = n1.astype(np.int64)
    j = np.arange(N, dtype=np.int64)
    out = np.empty(n_sym * N, dtype=np.complex128)
    step = max(1, _SYMBOL_CHUNK_ELEMS // N)

    for lo in range(0, n_sym, step):
        hi = min(lo + step, n_sym)
        edge = n1[lo:hi, None]
        in_seg1 = j[None, :] < edge
        # Each segment restarts its own time base at t0_frac.
        local = np.where(in_seg1, j[None, :], j[None, :] - edge)
        t = (t0_frac + local) * Ts
        f = np.where(in_seg1, f1[lo:hi, None], f2)
        base = np.where(in_seg1, phi_start[lo:hi, None], phi_mid[lo:hi, None])
        phase = base + two_pi * f * t + np.pi * Df * (t * t)
        _cis(phase, out=out[lo * N:hi * N].reshape(hi - lo, N))

    return out, phi_end


#generate a lora chirp
def lora_chirp(mu, k, BW, K, OSF, t0_frac=0, phi0=0):
    fs = BW * OSF
    Ts = 1 / fs
    # number of samples in one period T
    N = K * OSF
    T = N * Ts
    # derivative of the instant frequency
    Df = mu * BW / T
    if k > 0:
        (s1, phi) = chirp(mu * BW * (1 / 2 - k / K), k * OSF, Ts, Df, t0_frac, phi0)
        (s2, phi) = chirp(-mu * BW / 2, (K - k) * OSF, Ts, Df, t0_frac, phi)
        s = np.concatenate((s1, s2))
    else:
        (s, phi) = chirp(-mu * BW / 2, K * OSF, Ts, Df, t0_frac, phi0)
    return s, phi

# Default receive channel filter. A windowed sinc's transition band is roughly
# 5.5*fs/numtaps, so a short filter droops INSIDE the channel and attenuates
# the chirp's own sweep extremes. The tap count and the corner are SEPARABLE
# choices and both matter. Response of _lowpass_taps at 0.9*BW/2, for
# fs = 1 MHz and BW = 125 kHz:
#
#     numtaps    corner 1.15*BW/2    corner 1.00*BW/2
#        63          -1.41 dB            -3.63 dB
#       127          -0.11 dB            -1.95 dB
#       255          +0.00 dB            -0.35 dB
#
# At the default corner the tap count alone is worth 1.41 dB. 255 taps at
# 1.15*BW/2 give -0.02 dB at BW/2 and -122.1 dB at exactly BW, the first
# folded zone's edge; the worst sidelobe beyond BW is -103.8 dB.
_CHANNEL_FILTER_TAPS = 255
# Interpolation kernel for fs/BW below _MIN_OSF. 63 taps already match the
# ideal interpolator; 127 adds margin, and it only ever runs at fs/BW = 1.
# See upsample().
_INTERP_TAPS = 127
_CHANNEL_FILTER_CUTOFF_RATIO = 1.15


def _lowpass_taps(cutoff_hz, fs, numtaps):
    """Odd-length windowed-sinc lowpass, unit DC gain, linear phase."""
    if numtaps % 2 == 0:
        numtaps += 1
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    fc = cutoff_hz / fs
    h = 2 * fc * np.sinc(2 * fc * n) * np.blackman(numtaps)
    return h / h.sum()


def _fir_same(x, h, block=1 << 14):
    """Convolve `x` with FIR `h`, trimmed like np.convolve(mode='same').

    Overlap-save via FFT, so a multi-megasample receive buffer costs O(n log n)
    with bounded memory instead of the O(n * numtaps) of a direct convolution.
    """
    M = h.size
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.complex128)
    lead = (M - 1) // 2                     # symmetric h: centre the response
    n_causal = n + lead                     # causal samples needed for 'same'
    nfft = 1 << int(np.ceil(np.log2(max(block, 4 * M))))
    step = nfft - M + 1
    H = np.fft.fft(h, nfft)
    # Trailing pad only has to complete the final block, not mirror the input;
    # padding by n_causal would double the peak allocation on megasample
    # buffers.
    xp = np.concatenate((np.zeros(M - 1, dtype=np.complex128),
                         np.asarray(x, dtype=np.complex128),
                         np.zeros(nfft, dtype=np.complex128)))
    causal = np.empty(n_causal, dtype=np.complex128)
    pos = 0
    while pos < n_causal:
        seg = xp[pos:pos + nfft]
        if seg.size < nfft:                 # final block: zero-pad to length
            seg = np.concatenate((seg, np.zeros(nfft - seg.size, dtype=np.complex128)))
        y = np.fft.ifft(np.fft.fft(seg) * H)
        take = min(step, n_causal - pos)
        causal[pos:pos + take] = y[M - 1:M - 1 + take]
        pos += step
    out = causal[lead:]
    _zero_unsupported(out, np.asarray(x), M - 1 - lead, lead)
    return out


def _zero_unsupported(y, x, left, right):
    """Set y[j] = 0 wherever x[j-left .. j+right] is all zero, in place.

    EXACT ZEROS WHERE THE ARITHMETIC SAYS ZERO. _fir_same's y[j] reads
    exactly those inputs, and 'same' mode zero-extends the buffer at both
    ends, so there the true output is 0 -- but overlap-save returns FFT
    roundoff instead, of order 1e-16 of the signal.

    That roundoff would be DETECTED. rf_decode's energy gate compares each
    window with 1e-6 of the slice's own mean power, and once a frame has
    been consumed the slice is nothing but roundoff, so the gate would pass
    it; the plain detector has no gate at all. In implicit mode with
    standard framing and CRC off -- the one mode in which any lock becomes
    a packet -- that would deliver garbage frames beside a real one. An
    all-zero window fails the gate (0 is never above a floor) and the
    sync-word test (every bin is equal, so the argmaxes cannot sit 8 and 16
    bins apart).

    Not a threshold: a sample with any non-zero input in its support is
    untouched, so no signal, however weak, can be affected.
    gr-sdrlora/lib/lora_sync.cc's fir_same applies the same rule.
    """
    nz = x != 0
    n = nz.size
    if n == 0 or nz.all():
        return                          # a real capture: nothing to do
    if not nz.any():
        y[:] = 0
        return
    # A RUNNING COUNT of non-zero inputs, padded so that y[j]'s support
    # x[j-left .. j+right] -- zero-extended past both ends, as 'same' mode
    # is -- holds cz[j + left + right + 1] - cz[j] of them. Both terms are
    # views of one array, so this costs the count and a mask, and memory
    # does not grow with the number of zero runs -- one per input sample on
    # the zero-stuffed input upsample() hands in.
    w =left + right + 1
    cz = np.zeros(n + w, dtype=np.int32 if n + w < 2 ** 31 else np.int64)
    np.cumsum(nz, dtype=cz.dtype, out=cz[left + 1:left + 1 + n])
    cz[left + 1 + n:] = cz[left + n]
    y[cz[w:w + n] == cz[:n]] = 0


def upsample(s, fs, factor, numtaps=None):
    """Band-limited interpolation by an integer factor.

    Zero-stuff and lowpass at the ORIGINAL Nyquist -- what a polyphase
    interpolator computes, without the phase factoring. This creates no
    information: for complex IQ the Nyquist rate is fs >= BW, so a chirp
    spanning BW is already fully determined at fs = BW. It only re-samples it
    on a finer grid.

    It is needed because the receiver does not work at fs/BW = 1, and the
    reason is the timing estimator rather than the data. At critical sampling
    the dechirped preamble peak is distorted, so the parabolic interpolation
    in rf_decode reads low: at SF7, a sampling instant moving across a whole
    sample gives t_est of 0.02, 0.06, 0.10, 0.14 where the truth is 0.00,
    0.125, 0.25, 0.375 -- a gain of about 0.3 instead of 1, then a jump. It is
    not a resolution limit; raising OSF_fine_sync from 4 to 64 moves those
    numbers by less than 0.01. Hand the same samples to the same estimator on
    a finer grid and it tracks correctly. How much finer is set by
    _check_sample_rate; decode() uses _INTERP_TARGET_OSF.

    The information is all in the samples: a buffer the receiver fails on
    decodes once resampled to a different phase, which is only possible if
    the samples determine the waveform.

    The awkward part is that a critically sampled signal fills its whole band,
    so after zero-stuffing the image begins exactly where the signal ends and
    there is no transition band to put a filter in. It matters less than it
    sounds: against FFT zero-padding, the ideal band-limited interpolator,
    every windowed-sinc from 63 taps up decodes identically. Only a cutoff
    ABOVE the original Nyquist does worse, because the image lands on the
    chirp's own sweep extremes. So the cutoff sits exactly at fs/2 and the
    taps stay modest.
    """
    # WHOLE AND AT LEAST 1, by name: int(2.5) would interpolate by 2, and a
    # factor of 0 or -1 would return the input unchanged.
    factor = _whole(factor, "factor")
    # and BOUNDED: decode() interpolates to at most OSF_BOUND_MAX, and the
    # output is factor times the input's length
    if not 1 <= factor <= OSF_BOUND_MAX:
        raise ValueError("upsample: factor must be 1..%d, got %r"
                         % (OSF_BOUND_MAX, factor))
    # fs and numtaps too: fs = nan would make every sample non-finite, and
    # fs = 0 would divide by zero.
    _f = _real(fs, "fs")
    if not (np.isfinite(_f) and _f > 0):
        raise ValueError("upsample: fs must be finite and positive, got %r"
                         % (fs,))
    if numtaps is not None:
        numtaps = _check_numtaps(numtaps)
    s = np.asarray(s)
    if factor == 1 or s.size == 0:
        return s
    if numtaps is None:
        numtaps = _INTERP_TAPS
    up = np.zeros(s.size * factor, dtype=np.complex128)
    up[::factor] = s
    # Zero-stuffing divides the passband gain by `factor`; put it back.
    h = _lowpass_taps(fs / 2.0, fs * factor, numtaps) * factor
    return _fir_same(up, h).astype(np.complex64)


def channel_filter(s, BW, fs, cutoff_ratio=None, numtaps=None):
    """Band-limit received samples to the LoRa channel before demodulation.

    Worth about 6 dB of SENSITIVITY against wideband noise at the usual
    fs/BW = 8 -- the SNR shift at equal delivery.

    That is not the same as the filter's power ratio: at fs/BW = 8 the
    filter takes 8.57 dB off white noise and 0.86 dB off the chirp, so
    +7.7 dB of in-band SNR, the same at SF7, SF9 and SF12. The gap between
    6 and 7.7 dB is the resolution of the delivery comparison, not a loss;
    "about 6 dB" is a lower bound on the shift. The ideal brickwall figure,
    10*log10(fs/BW) = 9.03 dB here, is 1.3 dB above what this filter
    achieves.

    THE 0.86 dB IS CLIPPING, not the filter's cost on a centred chirp. It is
    the figure for a 17 ppm (Cfo_PPM) carrier offset, 29.5 kHz at
    868 MHz: the corner sits at cutoff_ratio * BW/2 about 0 Hz, so the
    offset chirp's upper sweep runs into the transition band. At zero
    offset the filter takes 0.00-0.03 dB off the chirp; at 5 ppm 0.07, at
    10 ppm 0.35. At SF7, fs/BW 8 and 17 ppm that costs about 1.3 to 1.5 dB
    of sensitivity against the same filter centred on the carrier.
    Widening the cutoff does not help: the noise it admits costs more.
    Filtering around the lock's own CFO estimate would, but the filter runs
    before the preamble search that produces that estimate, so it would
    need a second filtering pass per lock; it is not done.

    THE FRONT END DOES NOT GIVE THAT BACK. The decimation by OSF FOLDS
    every Nyquist zone onto the signal, which is the reason this function
    exists (see the module note below). Through the demodulator's own front
    end (dechirp, decimate [0:N:OSF], K-point FFT) the noise reduction seen
    at the FFT matches the one at the input to within about 0.07 dB at SF7
    and 0.03 dB at SF9, with no resolvable SF dependence.

    See the module note on why the *decimation* itself must stay unfiltered.

    cutoff_ratio scales the BW/2 corner. The default sits slightly above the
    channel edge so the filter's transition band falls outside it -- a corner
    exactly at BW/2 attenuates the chirp's own sweep extremes. A no-op when
    fs <= BW, where there is no oversampling and so nothing to fold.
    """
    s = np.asarray(s)
    # VALIDATED, because this is a public entry point and the kernel has
    # no defence of its own: _lowpass_taps divides by numtaps - 1, so
    # numtaps of 0 or 1 would return a buffer of NaNs. gr-sdrlora's
    # phy_channel_filter binding reaches the same code. These are the
    # checkers decode() uses.
    numtaps = _check_numtaps(numtaps)
    cutoff_ratio = _check_cutoff_ratio(cutoff_ratio)
    # THE RATE BOUNDS, not only finite and positive: BW = 1e-320 is both,
    # and would put the corner at a subnormal frequency and return NaNs.
    # Past fs/BW = OSF_MAX nothing here is meaningful.
    _check_rate_bounds(BW, fs)
    if fs <= BW:
        return s
    if numtaps is None:
        numtaps = _CHANNEL_FILTER_TAPS
    if cutoff_ratio is None:
        cutoff_ratio = _CHANNEL_FILTER_CUTOFF_RATIO
    h = _lowpass_taps(BW / 2.0 * cutoff_ratio, fs, numtaps)
    return _fir_same(s, h).astype(np.complex64)


def _mean_power(x):
    """Mean |x|**2, accumulated in float64 and blind to non-finite samples.

    The detection gate at the bottom of the scan loop compares a window's
    power against a floor derived from the whole buffer, and EVERY
    degenerate comparison -- `inf > inf`, `0 > 0`, `x > nan` -- evaluates
    False, which the gate reads as "reject". A degenerate power would make
    the buffer look silent rather than loud, so a plain
    `float(np.mean(np.abs(x) ** 2))` is not enough:

      * One non-finite sample anywhere would make the floor nan, and the
        gate would reject every scan position in the buffer.
      * |x| above ~1.3e17 in complex64 overflows a float32 accumulator: each
        square fits comfortably, the sum of thousands does not. Accumulating
        in float64 keeps decode scale-free -- receiver gain does not change
        the bits.

    Masking rather than raising keeps the receive path total, which is what
    StreamDecoder needs: one corrupt sample in a live stream must not take
    the decoder down with it.
    """
    if x.size == 0:
        return 0.0
    a = np.abs(x)
    good = np.isfinite(a)
    if not good.all():
        a = a[good]
        if a.size == 0:
            return 0.0
    return float(np.mean(a.astype(np.float64) ** 2))


def samples_decoding(s,BW,N,Ts,K,OSF,Nrise,SF,Trise,ih=None,fec_correct=True,soft_decision=True,robust_sync=True,sync_phases=None,pseudo_header=True,sync_word=SYNC_WORD_DEFAULT,ldro=LDRO_AUTO,report_crc_errors=False,with_consumed=False,sync_threshold=None,search_tail_symbols=None,hdr_rows=HDR_ROWS_DEFAULT,hop_period=0,hop_freqs=None,search_from=0,final=False,center_freq=0.0):

    # Collected as a list, not pre-sized from a minimum packet duration: an
    # SF5 frame is 5.2 ms and an SF6 one 10.4 ms, shorter than
    # min_time_lora_packet, so a buffer can hold more frames than such a
    # bound allows.
    pack_list = []



    OSF = int(OSF)
    N = int(N)
    K = int(K)
    SF = int(SF)
    # THE WALK RESUMES AT search_from, and `consumed` never reports less.
    # A streaming caller rewinds its buffer so that the channel filter has
    # left context, and the samples before search_from are that context:
    # the previous scan already walked them. Walking them again would
    # deliver a second time any frame that lies inside the rewind. 0, the
    # default, is every one-shot caller.
    search_from = max(0, min(int(search_from), int(s.size)))
    # THE REAL LENGTH, before the pad below: `consumed` never passes it.
    _n_real = int(s.size)
    if final:
        # NOTHING MORE IS COMING, so what follows the buffer is silence --
        # one symbol of it is appended. The timing estimate can land a
        # sample late, and a frame whose last modulated sample is the
        # buffer's last, or one or two short of it, would then ask for
        # samples past the end and be lost. Demodulated against zeros, the
        # last symbol loses a sample or two of N; the CRC still decides.
        # gr-sdrlora/lib/lora_sync.cc pads its final walk the same way.
        # Only a buffer that can hold a preamble and sync word (12 symbols)
        # can hold a frame's end: below that the pad finds nothing, and at
        # fs/BW 65536 one symbol is 8 M samples for a 1024-sample buffer.
        if _n_real > 12 * N:
            s = np.concatenate([s, np.zeros(N, dtype=s.dtype)])
    cumulative_index = search_from
    last_index = 0
    received = 0
    # How many leading samples are finished with: scanned with full context and
    # holding no packet that has not already been reported. A streaming caller
    # keeps the rest and nothing else. Set at every exit below.
    consumed = search_from
    _search_tail = (SEARCH_TAIL_SYMBOLS if search_tail_symbols is None
                    else int(search_tail_symbols)) * N
    # The open REPORTED FAILURE: its index in pack_list and where
    # its frame ends. See `located` below.
    _fail_idx, _fail_end = -1, -1
    _exit_truncated = False




    while True:
        if s.size - cumulative_index < N:
            # Less than one symbol left, counted from the walk position.
            break

        (success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset) = rf_decode(s[cumulative_index:] ,BW,N,Ts,K,OSF,Nrise,SF,Trise,ih,fec_correct,soft_decision,robust_sync,sync_phases,pseudo_header,sync_word,ldro,sync_threshold,hdr_rows,hop_period,hop_freqs,center_freq,final)

        if truncated and final:
            # NOTHING MORE IS COMING: a one-shot decode, or the last feed of
            # a stream. A frame cut off by the end of the buffer -- or a
            # corrupted header claiming a length that runs past it -- will
            # never complete, and ending the walk on it would lose every
            # complete frame behind it. A failed lock, then: one symbol on.
            cumulative_index = cumulative_index + max(int(last_index), 0) + N
            consumed = cumulative_index
            continue
        if truncated:
            # A packet that has not finished arriving. Keep it from its own
            # start -- that is the whole buffer a streaming caller ever needs
            # to hold, and only while a packet is in flight.
            consumed = cumulative_index + max(int(last_index), 0)
            _exit_truncated = True
            break

        # A header whose FCS passed locates a real packet: its LENGTH and CR
        # say where it ends, so a failed payload CRC there is a corrupted
        # packet rather than a false lock. Only meaningful with a header on
        # air -- implicit mode sets HDR_FCS_OK unconditionally, and decode()
        # refuses the combination.
        located = (report_crc_errors and (not success) and bool(HDR_FCS_OK)
                   and payload is not None and len(payload) > 0)

        # A REPORTED FAILURE DOES NOT SKIP ITS FRAME. With the flag off the
        # walk steps one symbol and often re-locks the SAME frame cleanly;
        # skipping the whole frame would make asking for the failures cost
        # good frames and bias packet error rates high. So it steps one
        # symbol like any failed lock, and remembers the frame's span: a
        # later lock inside it that decodes REPLACES the failure, and one
        # that fails again is the same frame, not counted twice. A lock starting at or after the span is a new frame.
        _same = (_fail_idx >= 0
                 and cumulative_index + int(last_index) < _fail_end)
        if success:
            _pkt = LoRaPacket(payload,SRC,DST,SEQNO,HDR_FCS_OK,HAS_CRC,MAC_CRC_OK,CR,int(ih is not None),SF,BW)
            if _same:
                pack_list[_fail_idx] = _pkt
                _fail_idx, _fail_end = -1, -1
            else:
                pack_list.append(_pkt)
                received = received + 1
        elif located and not _same:
            pack_list.append(LoRaPacket(payload,SRC,DST,SEQNO,HDR_FCS_OK,HAS_CRC,MAC_CRC_OK,CR,int(ih is not None),SF,BW))
            received = received + 1
            _fail_idx = len(pack_list) - 1
            _fail_end = cumulative_index + int(last_index) + int(offset)



        # FOUND NOTHING is HDR_FCS_OK still None -- only lora_packet_rx sets
        # it, and rf_decode calls that only after a detection -- and NOT
        # last_index == -1. last_index is the detected frame's start estimate
        # relative to this slice, and -1 is a legitimate value of it: a frame
        # starting one sample before the walk's start, which is common when
        # a streaming walk resumes exactly at the previous estimate.
        # gr-sdrlora's walk tests sy.found, the same distinction.
        if HDR_FCS_OK is None:
            # No preamble in what is left. Only one straddling the end could
            # still be live, so keep the last _search_tail samples.
            consumed = max(cumulative_index, _n_real - _search_tail)
            break

        else:
            #cumulative_index = cumulative_index + last_index + 10*N
            #cumulative_index = cumulative_index + last_index + 28 * N
            # Only a decode skips its frame. A located failure steps one
            # symbol like any failed lock, and its span (_fail_end) does the
            # counting-once.
            if success and offset != -1:
                # Decoded: skip the whole packet. last_index is where it
                # started in this window, offset how long it ran.
                _next = cumulative_index + last_index + offset
                # STALL GUARD, matching lora_sync.cc's `if (next <=
                # cumulative) break;`. The failure branch below clamps
                # last_index at 0 and always adds N, so it cannot stall; this
                # branch adds last_index + offset with no floor, and
                # last_index is a fine timing estimate that can come back
                # negative. If it does not move the walk forward, the SAME
                # window would be re-decoded and the SAME packet appended on
                # every pass -- an unbounded loop, not a lost packet.
                #
                # Reachable only at fs/BW >= 32: last_index + offset is
                # (m_d0-m_u0)*OSF/8 + s_ofs + 9.25*N or more, s_ofs >= 6*N,
                # and |m_d0-m_u0| < 4*N, so the sum stays positive below
                # OSF 32. 32 and 64 are accepted by _check_sample_rate and,
                # being powers of two, by the C++ decoder as well.
                if _next <= cumulative_index:
                    consumed = cumulative_index
                    break
                cumulative_index = _next
            else:
                # Detected a preamble but could not decode it. Skipping a
                # packet's worth from here would be wrong, because the reason
                # the decode failed may be that last_index is not where a
                # packet starts -- and then the skip lands inside, or past, a
                # real one. Any false lock -- from noise, from an adjacent
                # channel, or the detection window's tail reaching a preamble
                # across the lead-in -- would then cost every later packet in
                # the window.
                #
                # Advance one symbol instead. That only rules out what was
                # actually ruled out, and still guarantees progress, so the
                # walk terminates. It costs repeated attempts while it steps
                # clear of a bad lock, but each one stops at the first
                # detection and so is cheap.
                #
                # last_index is clamped at 0 because the fine timing estimate
                # can land BEFORE the window it was measured in and come back
                # negative. -1 is the "nothing found" sentinel and broke out
                # above, but -1024 is a real estimate, and both last_index +
                # offset and a bare last_index + N would then move by less
                # than a symbol or not at all.
                cumulative_index = cumulative_index + max(int(last_index), 0) + N
        consumed = cumulative_index
    # A REPORTED FAILURE'S FRAME stays retired: a streaming scan that ended
    # inside its span would otherwise re-walk it next time and report it
    # again. Not past a frame still arriving, which is the other way out.
    if _fail_end > consumed and not _exit_truncated:
        consumed = min(_fail_end, _n_real)
    if with_consumed:
        return (np.array(pack_list, dtype=LoRaPacket),
                int(max(search_from, min(consumed, _n_real))))
    return np.array(pack_list, dtype=LoRaPacket)





# Preamble detection. The 8 preamble up-chirps are identical, so accumulating
# their dechirped power spectra before locating the peak is far more reliable
# than any single symbol. _SYNC_ACC_SYMBOLS spectra are summed, ending
# _SYNC_ACC_LAG scan steps back -- by the time the sync-word pattern is
# visible in the history the newest windows are down-chirps, so the
# accumulation has to look back to when the window was still on the preamble.
_SYNC_ACC_SYMBOLS = 6
_SYNC_ACC_LAG = 4
# peak/mean of the accumulation. The noise-only p99.9 is 4.1 (SF7) to 4.6
# (SF12); the preamble sits at 18 (SF7, -22 dB) to 4400 (SF12, -15 dB).
_SYNC_STAT_THRESHOLD = 5.5


def rf_decode(s,BW,N,Ts,K,OSF,Nrise,SF,Trise,ih=None,fec_correct=True,soft_decision=True,robust_sync=True,sync_phases=None,pseudo_header=True,sync_word=SYNC_WORD_DEFAULT,ldro=LDRO_AUTO,sync_threshold=None,hdr_rows=HDR_ROWS_DEFAULT,hop_period=0,hop_freqs=None,center_freq=0.0,_final=False):
    truncated = False
    payload = None
    last_index = -1
    success = False
    HDR_FCS_OK = None
    MAC_CRC_OK = None
    DST = -1
    SRC = -1
    SEQNO = -1
    CR = -1
    HAS_CRC = -1
    ns = len(s)
    # base upchirp & downchirp
    u0 = chirp(-BW/2,N,Ts,BW/(N*Ts))[0]

    d0 = np.conj(u0)

    # parameters and state variables in the synch block
    # Scan granularity: the window advances by N/m_phases, so this sets how
    # closely a scan window can align to a symbol boundary. The sync-word
    # argmaxes are single-symbol and degrade with misalignment, which makes
    # this the binding constraint on sensitivity: at SF7, 4 phases deliver
    # several times as many frames as 2 near the sensitivity limit. They
    # also cost ~1.7-2x decode time, which takes a live receiver past its
    # real-time budget, so the default stays at 2 there.
    # Below SF7 the symbol is short enough that 2 phases miss the sync word
    # outright, and short enough that 4 are cheap -- see default_sync_phases.
    # Detection energy floor: see the gate below.
    # Resolved once, outside the scan loop.
    _sync_thr = (_SYNC_STAT_THRESHOLD if sync_threshold is None
                 else float(sync_threshold))
    _buf_pow = _mean_power(s)
    _win_pow_floor = 1e-6 * _buf_pow

    m_phases = int(default_sync_phases(SF) if sync_phases is None
                   else sync_phases)
    # Sync-word symbol offsets, in bins above the preamble reference.
    _sw1, _sw2 = sync_word_symbols(sync_word)
    check_sync_symbols(_sw1, _sw2, SF)

    def _bin_off(measured, reference, want):
        # Bin indices are cyclic mod K, so the offset has to be taken mod K
        # and folded into [-K/2, K/2). Without that, a sync word whose nibbles
        # are large enough to wrap would fail its own receiver: 0xA5 puts a
        # symbol 80 bins up at SF7, and 80 - 128 = -48 reads as a 48-bin
        # error.
        return np.mod(measured - reference - want + K / 2.0, K) - K / 2.0
    m_phase = 0
    m_vec = -1*np.ones((2*m_phases,6)) # peak values positioning in the DFT
    # oversampling factor for fine-grained esteem
    OSF_fine_sync = 4

    missed_sync = True
    # KNOWN LIMIT: a NEGATIVE carrier offset can make this scan lock exactly
    # half a symbol early.
    #
    # The two-chirp fine estimator folds its residual into +-N/2 and so
    # cannot tell +1858 samples from -190: at SF9, fs/BW 8, sync word 0x34
    # and cfo_ppm -14 (0.19 BW of offset) the true alignment is
    # s_ofs = 55106, the scan grid offers 53248 and 55296, both detect, and
    # both report a residual of -190. The earlier one is visited first and
    # is wrong by half a symbol.
    #
    # `sync_metric` below would break that tie but is inert: it starts at
    # inf, so the first candidate always wins, and the loop breaks
    # immediately afterwards. Continuing the scan and taking the best
    # candidate is not done: under interference the first detection is
    # often the right one, and sync_metric is not a good enough
    # discriminator to overrule it.
    #
    # The fold is JOINT: a lock that is N/2 early always reads a Cfo_est
    # that is BW/2 high, so the two hypotheses are (t, c) and
    # (t - N/2, c + BW/2). Keeping whichever has the smaller |Cfo_est| does
    # not resolve it: that rule holds only while |Cfo_est| is under BW/4 and
    # inverts above it, so it breaks correct locks at |ppm| >= 18. At the
    # offsets that matter the two hypotheses are equally plausible in
    # frequency, and the fold has to be resolved from the DATA --
    # dechirping the first payload symbol at both hypotheses, say.
    #
    # WHAT IT COSTS: with any error detection on -- an explicit header's
    # FCS, or a CRC -- nothing; SF9 fs/BW 8 explicit+CRC delivers the
    # correct payload at every cfo_ppm in +-25. The exposure is implicit
    # header WITH CRC OFF, the one LoRa mode carrying no integrity check at
    # all, where a mis-lock hands back wrong bytes with nothing able to flag
    # them. SF5 is the worst case because a given offset in Hz is four times
    # as many bins as at SF7.
    #
    # AND AT ZERO OFFSET, at a few grid phases -- a second mechanism. With
    # cfo_ppm 0, implicit header and CRC off, a clean frame decodes to wrong
    # bytes at 1 of the 8 lead-in phases of the scan grid at SF5/osf1, 1 of
    # 16 at SF5/osf2 and 1 of 32 at SF6/osf2 (every phase decodes at
    # SF6/osf1, and at SF7 at osf 1, 2 and 8). There the two candidate
    # alignments TIE and floating-point roundoff settles it, so the outcome
    # can differ between NumPy versions and the C++ receiver; on a real
    # signal each of those phases is a coin flip. With a CRC every phase
    # delivers, because a wrong lock fails it and the walk moves on.
    #
    # gr-sdrlora/lib/lora_sync.cc has the identical behaviour, byte for byte,
    # except where the choice is such a tie and roundoff decides it.
    sync_metric = np.inf


    # Ring of recent dechirped power spectra, one ring per scan phase.
    _acc_depth = _SYNC_ACC_SYMBOLS + _SYNC_ACC_LAG
    if robust_sync:
        pu_ring = np.zeros((m_phases, _acc_depth, int(N)))
        pu_count = np.zeros(m_phases, dtype=np.int64)
    stat_ref = 0.0
    m_ref = 0.0

    offset = -1
    main_loop = True
    s_ofs = 0

    #         s = s(620000:1000000)
    #         ns = numel(s)
    while main_loop:
      
        # sample window for a full chirp
        try:

            s_win = s[np.arange(s_ofs,s_ofs+N, dtype = np.int32)]
            _win_pow = _mean_power(s_win)
        except IndexError:
            success = False
            payload = None

            return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset

        # multiplication of the signal with a downchirp and a upchirp, respectively.





        Su = np.abs(np.fft.fft(s_win * d0))
        Sd = np.abs(np.fft.fft(s_win * u0))

        if robust_sync:
            # Su is already computed, so the accumulation costs no extra FFT.
            idx = int(pu_count[m_phase] % _acc_depth)
            pu_ring[m_phase, idx] = Su * Su
            pu_count[m_phase] += 1
            stat_ref = 0.0
            if pu_count[m_phase] >= _acc_depth:
                # sum the _SYNC_ACC_SYMBOLS spectra ending _SYNC_ACC_LAG back
                rows = [(idx - _SYNC_ACC_LAG - j) % _acc_depth
                        for j in range(_SYNC_ACC_SYMBOLS)]
                acc = pu_ring[m_phase, rows].sum(axis=0)
                acc_mean = acc.mean()
                # An energy gate as well as acc_mean > 0. The ratio
                # max/mean is scale-invariant, which is what makes it a good
                # detector -- and meaningless over a run of zeros, where both
                # are ~0 and the ratio is floating-point noise that sails past
                # the threshold -- a lock inside the encoder's zero lead-in,
                # which spans the most scan positions at low spreading
                # factors. The floor is relative to the buffer's own mean
                # power, so a uniformly quiet capture still detects normally.
                if acc_mean > 0 and _win_pow > _win_pow_floor:
                    stat_ref = float(acc.max() / acc_mean)
                    r = int(np.argmax(acc))
                    m_ref = np.mod(r - 1 + N / 2, N) - N / 2
        m_u=np.argmax(Su)
        m_d=np.argmax(Sd)

        # convert the positioning in values in the range [-N/2,N/2-1] 
        m_u=np.mod(m_u-1+N/2,N)-N/2
        m_d=np.mod(m_d-1+N/2,N)-N/2


        m_vec[np.arange(m_phase * 2, m_phase * 2 + 2), 1:] = m_vec[np.arange(m_phase * 2, m_phase * 2 + 2), :-1]
        m_vec[np.arange(m_phase * 2, m_phase * 2 + 2), 0] = np.array([m_u, m_d]) #Numpy automatically converts the row into a column





        # three upchirps followed by the two sync-word symbols, further
        # followed by two downchirps, correspond to a preamble. The sync
        # symbols sit at sync_word's nibbles times 8 -- 8 and 16 for the
        # default 0x12.
        # The four conditions chaining ADJACENT symbols. Costs nothing but
        # comparisons on values already computed, so it is always evaluated:
        # robust_sync needs it as a fallback, see below.
        _plain = (
            np.abs(m_vec[m_phase*2+1,0]-m_vec[m_phase*2+1,1])<=1 and
            np.abs(_bin_off(m_vec[m_phase*2,2], m_vec[m_phase*2,3],
                            _sw2-_sw1))<=1 and
            np.abs(_bin_off(m_vec[m_phase*2,3], m_vec[m_phase*2,4],
                            _sw1))<=1 and
            np.abs(m_vec[m_phase*2,4]-m_vec[m_phase*2,5]) <=1)

        if robust_sync:
            # Detection from the accumulated preamble, then the SAME strict
            # sync-word requirement (+8 and +16 bins, tolerance 1) measured
            # against the reliable reference rather than a noisy neighbour.
            #
            # stat_ref is 0 until the ring holds _acc_depth spectra, which
            # is about 6 symbols of buffer ahead of the sync word -- lead-in
            # and preamble both count towards it. That warm-up is deliberate
            # and must not be short-circuited by falling back to _plain: a
            # decoy preamble inside the warm-up window would be locked by
            # the plain chain and hide the frame behind it. The warm-up
            # alone would make the minimum preamble length depend on SF;
            # default_lead_in() covers the shortfall, leaving a uniform
            # three-symbol detection floor that both detectors share.
            _detected = (
                stat_ref > _sync_thr and
                np.abs(m_vec[m_phase*2+1,0]-m_vec[m_phase*2+1,1]) <= 1 and
                np.abs(_bin_off(m_vec[m_phase*2,3], m_ref, _sw1)) <= 1 and
                np.abs(_bin_off(m_vec[m_phase*2,2], m_ref, _sw2)) <= 1)
        else:
            _detected = _plain

        # A DETECTION IS ONLY USABLE IF ITS HISTORY IS IN THE BUFFER. The
        # fine-sync estimator below looks back (4 + Nu) = 6 symbols from
        # s_ofs. History that has run off the front is already gone, and no
        # amount of further data can supply it, so this is NOT the
        # `truncated` case, which samples_decoding reads as "a packet that
        # has not finished arriving" and which stops the whole buffer walk.
        #
        # Refusing the candidate lets the scan run on through the rest of
        # the buffer, so complete packets behind the partial one are still
        # found. Advancing the slice start instead would REDUCE the history
        # available to the same detection, not increase it.
        # gr-sdrlora/lib/lora_sync.cc does the same.
        if _detected and (s_ofs - 6 * N) < 0:
            _detected = False

        if _detected:

            missed_sync = False
            #keyboard
            tmp = np.sum(np.abs(m_vec[m_phase*2+1,1:2])+np.abs(m_vec[m_phase*2,5:6]))
            if tmp < sync_metric:
                sync_metric = tmp

                #fprintf('phase: #g\n',m_phase)
                #display(m_vec(m_phase*2+(1:2),:))
                Nu = 2
                Nd = 2
                # fine-grained estimation of the positions of the maximums in the DFT,
                # as a function of the ANCHOR a: s_ofs, and +-N/2 when the
                # estimates there are ambiguous (see below). Returns None
                # when an anchor's windows leave the buffer; `main` marks
                # s_ofs itself, whose down-chirp windows always fit.
                def _fine(a, main):
                    ups, downs, peak = [], [], 0.0
                    for i in range (1,Nu+1):
                        # The look-back can only run off the FRONT of the buffer
                        # (the loop guard bounds the far end), and numpy fancy
                        # indexing WRAPS negative indices instead of raising,
                        # so it is checked explicitly.
                        _lo = a - (4 + i) * N
                        if _lo < 0:
                            # Unreachable at s_ofs while a candidate without 6
                            # symbols of history is refused above; kept so the
                            # failure mode can never be an estimate formed
                            # from the wrong end of the capture.
                            return None
                        Su = np.abs(np.fft.fft(((s[np.arange(_lo, _lo + N, dtype = np.int32)]) * d0),N*OSF_fine_sync))

                        m_u = np.argmax(Su)
                        peak += Su[m_u]
                        # NO ZERO-DENOMINATOR GUARD, and none is needed, though
                        # gr-sdrlora/lib/lora_sync.cc carries a harmless
                        # `if (den != 0.0)` at the same two places.
                        #
                        # np.argmax returns the FIRST index of the maximum, so
                        # Su[m_u-1] < Su[m_u] STRICTLY (a tie below would have
                        # been chosen instead). Su[m_u+1] <= Su[m_u]. So
                        # den = (Su[m_u-1]-Su[m_u]) + (Su[m_u+1]-Su[m_u]) is
                        # strictly negative in exact arithmetic, and the first
                        # term alone is at least one ulp.
                        #
                        # Rounding can only collapse it to exactly 0.0 if
                        # Su[m_u+1] == Su[m_u] bit for bit AND Su[m_u]-Su[m_u-1]
                        # is at most half an ulp of Su[m_u] -- which additionally
                        # needs Su[m_u] to sit on a binade boundary, because
                        # `a - 2*b + c` is evaluated as fl(fl(a-2b)+c) and the
                        # second add is exact by Sterbenz. The triple
                        # (0.9999999999999999, 1.0, 1.0) does it; nothing an FFT
                        # of these windows produces does. If it ever did, numpy
                        # would emit a RuntimeWarning rather than fail silently.
                        if (m_u > 0 and m_u < N*OSF_fine_sync-1):
                            m_u = m_u + 0.5*(Su[m_u-1]-Su[m_u+1]) / (Su[m_u-1]-2*Su[m_u]+Su[m_u+1])

                        ups.append(np.mod(m_u+N*OSF_fine_sync/2,N*OSF_fine_sync)-N*OSF_fine_sync/2)

                    for i in range(1,Nd+1):
                        _b = a - (i - 1) * N
                        # the s_ofs windows always fit (the loop guard
                        # keeps s_ofs + N <= ns); a shifted anchor may not
                        if not main and _b + N > ns:
                            return None
                        Sd = np.abs(np.fft.fft(((s[np.arange(_b, _b + N, dtype = np.int32)]) * u0), int(N * OSF_fine_sync)))
                        m_d = np.argmax(Sd)
                        peak += Sd[m_d]
                        # `> 1` is CONSERVATIVE. The parabola below reads
                        # Sd[m_d-1] and Sd[m_d+1], so bin 0 would fold Sd[-1]
                        # round to the far end of the spectrum and fit through
                        # a point that is not adjacent to the peak. Where the
                        # bound skips the refinement it would not move the
                        # estimate far enough to change a decode.
                        # lora_sync.cc's find_preamble carries the same bound.
                        if m_d > 1 and m_d < N*OSF_fine_sync:
                            try:
                                m_d = m_d + 0.5*(Sd[m_d-1]-Sd[m_d+1]) / (Sd[m_d-1]-2*Sd[m_d]+Sd[m_d+1])
                            except IndexError:
                                pass

                        downs.append(np.mod(m_d+N*OSF_fine_sync/2,N*OSF_fine_sync)-N*OSF_fine_sync/2)

                    m_u0 = 0
                    for v in ups:
                        m_u0 = m_u0 + v
                    m_u0 = m_u0/Nu
                    m_d0 = 0
                    for v in downs:
                        m_d0 = m_d0 + v
                    m_d0 = m_d0/Nd
                    return m_u0, m_d0, ups, downs, peak

                _r = _fine(s_ofs, True)
                if _r is None:
                    truncated = True
                    return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset
                _anchor = s_ofs
                # ONE CHIRP SEGMENT, NOT TWO. A dechirped window that straddles
                # a symbol boundary holds two tones exactly one BW apart --
                # K*OSF_fine_sync fine bins -- one per segment, and the argmax
                # takes whichever segment is longer. The detector fires up to
                # about half a symbol off alignment, where the two are nearly
                # equal and noise picks one per window; averaging estimates
                # from DIFFERENT segments puts Cfo_est off by BW/4 to BW/2 and
                # t_est off by a quarter to half a symbol. It matters at SF5
                # and SF6 at low fs/BW; from SF7 up the scan grid sits closer.
                # So when the two windows of a pair sit ONE BW apart -- the
                # signature of two segments, K*OSF_fine_sync fine bins within
                # an eighth of a BW -- the same estimate is formed half a
                # symbol either side and the anchor with the strongest peaks
                # kept: there the frame is within a quarter symbol of
                # alignment, one segment dominates every window, and the
                # branch is consistent. Anything else is used unchanged. The
                # trigger is the signature, not any disagreement: a wider one
                # (pairs more than half a BW apart, or a timing more than a
                # quarter symbol off) fires under co-channel LoRa, whose
                # windows scatter, and costs delivery there.
                # gr-sdrlora/lib/lora_sync.cc carries the identical rule.
                _B = K*OSF_fine_sync
                if (abs(abs(_r[2][0] - _r[2][1]) - _B) < _B/8
                        or abs(abs(_r[3][0] - _r[3][1]) - _B) < _B/8):
                    for _a in (s_ofs - int(N)//2, s_ofs + int(N)//2):
                        _c = _fine(_a, False)
                        if _c is not None and _c[4] > _r[4]:
                            _r, _anchor = _c, _a
                m_u0, m_d0 = _r[0], _r[1]

                # Cfo_est: frequency error
                #t_est: timing error
                #
                # ZERO-BASED BINS: np.argmax is 0-based, so the sums above
                # fold no `m - 1` (the MATLAB original's 1-based conversion).
                # An extra -1 would cancel in t_est, a DIFFERENCE of the two,
                # but bias Cfo_est, their MEAN, by -1 fine bin,
                # -BW/(4 * 2^SF). lora_sync.cc does the same.
                Cfo_est = (m_u0+m_d0)/2*BW/K/OSF_fine_sync
                t_est = (m_d0-m_u0)*OSF/2/OSF_fine_sync + _anchor-11*N-Nrise # n_pr = 8 + 2 syncword + 1 downchirp
                break





        m_phase = np.mod(m_phase+1,m_phases)
        # INTEGER division, matching the C++ (`s_ofs += N / m_phases`,
        # lora_sync.cc). True division would make s_ofs a drifting float, and
        # np.arange(s_ofs, s_ofs+N, dtype=np.int32) would then return N+1
        # elements once the fraction crossed an integer, for every sync_phases
        # value that is not a power of two.
        s_ofs = s_ofs + int(N)//int(m_phases)
        if s_ofs+N > ns:
            main_loop = False
            success = False
            return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset



    if not missed_sync:

        missed_sync = True

       
       #symbol-level receiver
        # ROUND, not ceil. A timing error of n samples moves the dechirped
        # peak by exactly n/OSF bins, so the offsets that still decode span
        # about +-OSF/2 samples (+-8 at fs/BW = 16, +-4 at 8, +-2 at 4, +-1
        # at 2). ceil() would bias the estimate up by anything in [0, 1),
        # which at fs/BW = 8 is at most an eighth of that budget -- but at 3
        # and 4 it lands outside the window about half the time. Nearest
        # rounding halves the worst-case residual and costs nothing.
        #
        # Folding the leftover fraction into Cfo_est as an equivalent
        # frequency offset (on a chirp the two are interchangeable) makes no
        # measurable difference with nearest rounding, so it is not worth
        # the coupling.
        p_ofs_est = int(np.round(t_est))

        last_index = p_ofs_est

        # The fractional timing residual is NOT applied:
        # lora_packet_rx takes the integer p_ofs_est and does its
        # own sub-sample work.
        #keyboard
        # the sample-clock offset the carrier offset implies; see
        # _sym_start. center_freq is signed: decode() negates it under
        # invert_iq, whose conj() flips Cfo_est but not the clock.
        _sfo = (Cfo_est / center_freq) if center_freq else 0.0
        _res = _demod_with_retries(s,SF,BW,OSF,Trise,p_ofs_est,Cfo_est,ih,fec_correct,soft_decision,pseudo_header,ldro,hdr_rows,hop_period,hop_freqs,_sfo)
        # LDRO THE OTHER WAY, when it is automatic and the frame still
        # located and failed its CRC. The header is read the same either
        # way; the payload is not. Sandeep Mistry's Arduino LoRa library
        # (0.8.0) rounds SF11 / 125 kHz's 16.38 ms symbol down to 16 ms and
        # transmits WITHOUT LDRO where the rule says with, so its frames
        # decode only with LDRO flipped. Kept only if the CRC then passes.
        # LDRO on costs symbols, so read with it a frame sent without it
        # can run past the buffer's end: "truncated", not a CRC failure. On
        # the last feed nothing more is coming, and the shorter reading is
        # tried too; while streaming the frame completes and fails its CRC
        # instead -- trying it on every feed until then would demodulate
        # each real LDRO frame again per block.
        # Only a frame that CARRIES a CRC: without one every reading
        # "passes", so a real LDRO frame cut off at the end would come back
        # as wrong bytes with crc_ok set.
        _cut = (_final and _res[10] and _res[1] and _res[9]
                and ldro_enabled(SF, BW, LDRO_AUTO))
        if (_LDRO_RETRY and int(ldro) == LDRO_AUTO and _res[1]
                and SF >= LDRO_RETRY_MIN_SF
                and (_cut or ((not _res[10]) and (not _res[3]) and _res[9]))):
            _flip = 0 if ldro_enabled(SF, BW, LDRO_AUTO) else 1
            _r2 = _demod_with_retries(s,SF,BW,OSF,Trise,p_ofs_est,Cfo_est,ih,fec_correct,soft_decision,pseudo_header,_flip,hdr_rows,hop_period,hop_freqs,_sfo)
            if (not _r2[10]) and _r2[1] and _r2[3]:
                _res = _r2
        (k_hdr_est,HDR_FCS_OK,k_payload_est,MAC_CRC_OK,MSG,DST,SRC,SEQNO,CR,HAS_CRC,truncated,offset) = _res


        if (not truncated) and (HDR_FCS_OK and MAC_CRC_OK):
            success = True
            payload = MSG
        elif (not truncated) and HDR_FCS_OK:
            # Header FCS passed, payload CRC did not: a located packet that
            # arrived corrupted. `success` stays False -- every existing caller
            # and the walk depend on that -- but the bytes come out so
            # samples_decoding() can count it.
            payload = MSG

        return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset


#SUPPORT FUNCTION TO ENCODE A LORA PACKET
def complex_lora_packet(K, n_pr, IH, CR, MAC_CRC, SRC, DST, SEQNO, BW, OSF, SF, Trise, N, Ts, fs, Cfo_PPM, f0,  MESSAGE, t0_frac, phi0, pseudo_header=True, sync_word=SYNC_WORD_DEFAULT, ldro=LDRO_AUTO, lead_in=None, invert_iq=False, hdr_rows=None, amplitude=1.0,
                        hop_period=0, hop_freqs=None):
    # THE SAME RULES encode() APPLIES, because the C++ mirror applies them
    # here too: gr::sdrlora::modulate() and encode_symbols() both open with
    # check_tx_params(p). This is a public entry point, and without them an
    # out-of-range address would wrap into a frame that decodes with
    # crc_ok=1, and a non-finite amplitude, phase, carrier offset or f0
    # would produce non-finite samples.
    #
    # The same helpers as encode(), so the two entry points cannot drift.
    # t0_frac is deliberately included: it permits negatives; it rejects
    # non-finite values and any outside (-1, 1), a whole sample being a
    # lead_in change.
    # The switches, as encode() checks them: a direct call would otherwise
    # take pseudo_header "False" as a request for the prefix.
    pseudo_header = _check_flag(pseudo_header, "pseudo_header")
    invert_iq = _check_flag(invert_iq, "invert_iq")
    IH = int(_check_flag(IH, "implicit_header"))
    SRC, DST, SEQNO = _check_addresses(SRC, DST, SEQNO, pseudo_header)
    Cfo_PPM = _check_cfo_ppm(Cfo_PPM)
    f0 = _check_f0(f0)
    amplitude = _check_amplitude(amplitude)
    phi0 = _check_phi0(phi0)
    t0_frac = _check_t0_frac(t0_frac)
    # _whole BEFORE the range check, and rebound: _check_preamble
    # validates and returns None, so the rebound value is what reaches the
    # modulator. np.tile needs an int, and 8.5 must not be truncated.
    n_pr = _whole(n_pr, "preamble_bits")
    _check_preamble(n_pr)
    # MAC_CRC IS A HEADER BIT. lora_payload_init sizes the buffer
    # 8*LENGTH + 16*MAC_CRC while lora_header takes the LOW BIT for the
    # flag, so MAC_CRC=2 would build a frame that reserves four CRC bytes,
    # writes two, and declares it has no CRC.
    MAC_CRC = _check_enable_crc(MAC_CRC)
    CR = _whole(CR, "cr")
    _check_coding_rate(CR)
    # _whole first: _check_spreading_factor does not test wholeness, so
    # SF=7.9 would build an SF7 waveform.
    SF = _whole(SF, "SF")
    # IH FIRST, as encode() does it. _check_spreading_factor takes the
    # header flag, and bool(0.5) is True, so a fractional flag must be
    # rejected before it can wave the SF5/SF6 rule through.
    IH = _whole(IH, "implicit_header")
    _check_spreading_factor(SF, bool(IH))
    Trise = _check_trise(Trise)
    # WITH the resolved Trise, and HERE rather than only in encode(): this
    # function builds the taper, and a bandwidth whose raised edge is
    # longer than a symbol would index the leading taper off the front of
    # the preamble (numpy wraps, the C++ mirror reads off the heap).
    _check_edge_bandwidth(BW, SF, Trise)
    # AND the rate. A non-integral fs/BW such as 8.5 makes lora_packet's
    # int(K*OSF) and lora_symbols' int(OSF)*K disagree, and the modulator
    # fails with an error naming neither parameter. The return value is
    # used for the same reason encode() uses it: _check_sample_rate
    # tolerates a ratio one ULP below an integer, and OSF is by definition
    # fs/BW, so the validated integer is what the two int() calls must both
    # see.
    # warn_below_min=False -- the minimum-ratio warning is about the
    # DEMODULATOR and has nothing to say to a transmitter.
    OSF = _check_sample_rate(BW, fs, warn_below_min=False)
    # REBOUND, not just inspected: _payload_bytes accepts bytes, bytearray,
    # memoryview, str and a list of ints, and lora_packet needs the uint8
    # array it returns.
    MESSAGE = _payload_bytes(MESSAGE)
    _on_air = (4 if pseudo_header else 0) + _payload_len(MESSAGE)
    _check_payload_length(_on_air)
    _check_crc_length(_on_air, MAC_CRC)
    # The hop schedule, rebound: _check_hops RESOLVES it (None -> (),
    # a period to a whole number of symbols) as well as validating it, and
    # refuses non-finite channels, a period with no channel list, a
    # channel past the front-end edge, a period below the 13-symbol floor
    # and an acquisition channel off zero, where no receiver can find it.
    hop_period, hop_freqs = _check_hops(hop_period, hop_freqs, fs, BW)
    # The sync symbols are placed below K because the modulator's mapping is
    # mirrored; K-8 and K-16 are what 0x12 produces here.
    hdr_rows = _check_hdr_rows(hdr_rows, SF, IH)
    s1, s2 = sync_word_symbols(sync_word)
    check_sync_symbols(s1, s2, SF)
    # Fold to a real bin FIRST. sync_word_symbols accepts raw symbols up to
    # 4095 and check_sync_symbols compares mod 2^SF, so the symbol is
    # reduced mod K here too: at SF5 (K = 32) a sync word of (34, 16)
    # would otherwise give k1 = -2 and a corrupt frame. The separability
    # check guarantees 2 <= s mod K <= K-2, so K - s is a valid symbol index.
    # gr-sdrlora/lib/lora_phy.cc modulate() carries the identical fold.
    s1 = int(s1) % int(K)
    s2 = int(s2) % int(K)
    k1 = K - s1
    k2 = K - s2
    SF = int(SF)
    # THE FRAME-SIZE FLOOR before anything is built, payload-independent,
    # as gr-sdrlora's modulate() and the encoder block apply it: otherwise
    # a direct call with SF12 and a 65535-symbol preamble would build
    # 2.1e9 samples of chirp before any check ran. encode()'s estimate
    # counts the payload as well, so it is never below this.
    _floor = (_resolve_lead_in(lead_in, SF, BW, fs, n_pr)
              + (int(n_pr) + 32) * int(K) * int(OSF))
    if _floor > MAX_FRAME_SAMPLES:
        raise ValueError(
            "the requested frame is about %d samples before any payload, "
            "past the %d-sample limit -- the preamble and lead-in alone"
            % (_floor, MAX_FRAME_SAMPLES))
    p = lora_packet(BW,OSF,SF,k1,k2,n_pr,IH,CR,MAC_CRC,SRC,DST,SEQNO,MESSAGE,Trise,t0_frac,phi0,pseudo_header,ldro,hdr_rows)[0]
    if invert_iq:
        # Inverting I and Q is, for a complex baseband signal, exactly the
        # conjugate -- which is what the SX127x InvertIQ bits do. Every chirp
        # in the frame reverses direction: the preamble sweeps down, the SFD
        # up. It is applied to the MODULATION only, before the carrier offset
        # below, because a transmitter's oscillator error is a property of its
        # crystal and does not invert with the modulator.
        #
        # Equivalent to rebuilding every symbol with mu = -1 -- lora_chirp
        # negates both the sweep rate and the starting offset with mu, so its
        # phase is negated and conj() gives the same samples -- but it costs
        # one pass over the buffer instead of a second modulator.
        p = np.conj(p)
    size_p = p.size
     
     #p_ofs=ceil((ns-np-ntail)*rand)
    # Zero samples before the preamble. Both receivers here decode with NO
    # lead-in at all GIVEN A FULL PREAMBLE, so it buys nothing on reception
    # there and costs 10 ms of transmitter-on dead air per frame at
    # fs = 1 MHz. LEAD_IN_DEFAULT stays the floor so existing waveforms are
    # reproduced exactly, and is topped up for a SHORT preamble, the one
    # case where the lead-in is load-bearing. See default_lead_in.
    # NEGATIVE MEANS AUTO, the same rule as gr::sdrlora::resolved_lead_in,
    # so the GNU Radio block and phy_encode() agree; None is the Pythonic
    # spelling of it.
    p_ofs = _resolve_lead_in(lead_in, SF, BW, fs, n_pr)
    if p_ofs > LEAD_IN_MAX:
        raise ValueError(
            "lead_in must be <= %d samples, got %r -- it is pure padding "
            "ahead of the preamble, and the bound is what turns an absurd "
            "value into an error naming it rather than a MemoryError from "
            "inside np.zeros" % (LEAD_IN_MAX, lead_in))
    # Only p_ofs..p_ofs+size_p is ever populated, so allocate exactly that.
    s = np.zeros(p_ofs + size_p, dtype=np.complex64)

    t = np.arange(0,size_p)*Ts
    # Exact, over this frame, for a caller who reaches this directly;
    # encode() checks the same rule on its frame-size estimate first.
    _check_carrier_phase(Cfo_PPM, f0, size_p * Ts)
    # Cfo_TX = Cfo_PPM*1e-6*f0*(2*rand-1)
    # Cfo = Cfo_TX+Cfo_PPM*1e-6*f0*(2*rand-1)
    Cfo_TX = Cfo_PPM * 1e-6 * f0 * 1
    Cfo = Cfo_TX + Cfo_PPM * 1e-6 * f0 * 1
    # Slice rather than fancy-index, but KEEP the `+ s[...]`: s holds +0.0 here,
    # and 0.0 + (-0.0) normalises the sign bit. Dropping it preserves -0.0
    # instead, which is numerically identical but changes the emitted bytes.
    # No phi0 here: lora_packet() above already starts every chirp at
    # phi0, and adding it to this term as well would apply it twice.
    # The CFO's own doubling (Cfo_TX + Cfo, see above) is deliberate.
    s[p_ofs:p_ofs+size_p] = s[p_ofs:p_ofs+size_p] + p * _cis(2*np.pi*Cfo*t)
    # Scaled at the very END, after the -0.0 normalisation above and after the
    # CFO term, and skipped entirely at 1.0 -- so the default path is
    # bit-identical to the unscaled one.
    # Hops BEFORE the level, so amplitude scales the hopped frame rather
    # than the other way round. Both are multiplicative, so this only fixes
    # the rounding order -- and the C++ mirrors it.
    # ANCHORED AT THE HEADER, not at the frame start. The receiver's
    # p_ofs_est is a frame start computed as if the preamble were 8 symbols
    # long (see t_est's `- 11*N`, and lora_packet_rx's matching `12.25*N`:
    # the two cancel, which is why demodulation works at any preamble
    # length). It is therefore off the TRUE frame start by exactly
    # (n_pr - 8)*N, and a schedule anchored there would be de-hopped against
    # the wrong phase for every preamble length except 8.
    #
    # p_ofs_est + Nrise + 12.25*N is the header start, EXACTLY, for any
    # preamble length -- the same cancellation that makes demodulation work.
    # So the schedule is keyed to that instead, and the preamble, sync words
    # and down-chirps -- everything before the anchor, which _apply_hops
    # leaves untouched -- go out unrotated on hop_freqs[0]. That is also the
    # only channel a receiver could acquire on, and hop_freqs[0] == 0 is
    # already required for exactly that reason.
    _hop_anchor = int(p_ofs + np.ceil(Trise * fs) + (n_pr + 4.25) * N)
    s = _apply_hops(s, _hop_anchor, hop_period, hop_freqs, int(N), fs)
    if float(amplitude) != 1.0:
        s = (s * np.complex64(amplitude)).astype(np.complex64)
    return s



#ENCODER FUNCTION. GIVEN A PAYLOAD (IN BYTES), AND THE INTENDED TRANSMISSION PARAMETERS, GENERATES THE SAMPLES FOR THE CORRESPONDING LORA PACKET
def encode(f0, SF, BW, payload, fs, src, dst, seqn, cr=1, enable_crc=1, implicit_header=0, preamble_bits=8, pseudo_header=True, cfo_ppm=0.0, sync_word=SYNC_WORD_DEFAULT, ldro=LDRO_AUTO, lead_in=None, invert_iq=False, trise=None, t0_frac=0.0, phi0=0.0, amplitude=1.0,
           hop_period=0, hop_freqs=None):
    """Payload bytes -> baseband IQ.

    pseudo_header: prefix the payload with this implementation's own 4-byte
    DST/SRC/SEQNO/LENGTH block (the default). Pass False for a standard LoRa
    frame: the message starts at payload byte 0, the PHY header's LENGTH is
    the message length, and `src`/`dst`/`seqn` are ignored -- not validated,
    and None is accepted, which is what decode() returns for them in that
    mode, so a received packet can be sent on again. That saves 4 bytes of airtime
    and is what a third-party receiver expects -- though gr-lora_sdr decodes
    the default frames too, prefix and all, since to LoRa those are just
    payload bytes.

    invert_iq: transmit with I and Q swapped, which for complex baseband is
    the conjugate of the frame. This is the SX127x InvertIQ setting and the
    polarity a LoRaWAN DOWNLINK uses: gateways transmit inverted and listen
    normal, end devices the reverse, so neither hears its own kind. An
    inverted frame does not decode as a normal one and vice versa -- that
    separation is the entire point of the feature.

    cfo_ppm: carrier frequency offset to bake into the waveform, in ppm of
    `f0`, for simulating crystal error. The packet builder applies it twice
    -- as Cfo_TX and again as Cfo, one per end of the link -- so the offset
    on the waveform is 2 * cfo_ppm * 1e-6 * f0. Default 0, a clean carrier:
    a real transmitter brings its own offset, so leave it at 0 for anything
    sent over the air. lora.Cfo_PPM (17) is a typical crystal, 29.5 kHz at
    868 MHz.

    gr-lora_sdr's capture range is about a quarter of the bandwidth, and
    29.5 kHz is 0.24 of 125 kHz -- right on the line. Its receiver loses SF5
    between 24.3 and 29.5 kHz of offset while SF6 and SF7 hold to between
    29.5 and 41.7 kHz, so a simulated cfo_ppm=17 lands just inside for SF6
    up and just outside at SF5. It is a margin, not a property of SF5.
    """
    # Coerced to plain ints, not merely checked. LoRaPacket fields are
    # np.uint8, and under NumPy 2 (NEP 50) 2**uint8 wraps to 0 for SF>=8
    # and arithmetic on cr=pkt.cr overflows uint8 inside the modulator, so
    # a received packet's fields can be passed straight back in.
    # VALIDATED BEFORE COERCED where the coercion would hide the problem:
    # int(0.5) is 0, so coercing enable_crc first would make a fractional
    # value silently mean "no CRC" instead of an error naming it.
    enable_crc = _check_enable_crc(enable_crc)
    # THE SWITCHES, by name, as decode() checks them: a truthy string such
    # as "False" or "no" must not silently select the 4-byte prefix or an
    # inverted frame.
    pseudo_header = _check_flag(pseudo_header, "pseudo_header")
    invert_iq = _check_flag(invert_iq, "invert_iq")
    # _whole, not int(), by the same rule.
    SF = _whole(SF, "SF")
    cr = _whole(cr, "cr")
    implicit_header = int(_check_flag(implicit_header, "implicit_header"))
    preamble_bits = _whole(preamble_bits, "preamble_bits")
    _check_spreading_factor(SF, bool(implicit_header))
    _check_coding_rate(cr)
    _check_preamble(preamble_bits)
    src, dst, seqn = _check_addresses(src, dst, seqn, pseudo_header)
    # Coerced HERE, so every path below sees a uint8 array whatever came in:
    # bytes, a list of ints, a bytearray, a memoryview or a str.
    payload = _payload_bytes(payload)
    _on_air = (4 if pseudo_header else 0) + _payload_len(payload)
    _check_payload_length(_on_air)
    _check_crc_length(_on_air, enable_crc)
    # KNOWN ASYMMETRY, deliberately left. An empty payload with
    # implicit_header=1 and pseudo_header=False encodes to a full frame
    # that decode() then refuses ("payload_len must be 1..255 in
    # implicit-header mode"), as both C++ receivers do. It stays encodable:
    # the frame is well-formed, and it is the input that exercises
    # lora_payload_init's block-count clamp, the `max(0, ...)`, identically
    # on both implementations.
    # The rate is validated by name, and the RETURN VALUE used, not just
    # the side effect. _check_sample_rate tolerates
    # |ratio - round(ratio)| <= 1e-9*ratio, so a ratio one ULP below an
    # integer passes -- and `OSF = fs / BW` would hand that float on to
    # lora_symbols, which truncates it with int(OSF) while lora_packet
    # uses int(K*OSF), and the two would disagree by one sample.
    # Reachable with STANDARD LoRa bandwidths: 125000/12 Hz at 7x and 14x
    # oversampling, and likewise 125000/6 and 125000/3. decode() uses the
    # return value too.
    #
    # warn_below_min=False: the minimum-ratio warning it can emit is
    # about the DEMODULATOR, and has nothing to say to a transmitter.
    _osf_checked = _check_sample_rate(BW, fs, warn_below_min=False)
    # Waveform knobs; each defaults to the module's standard value.
    trise = _check_trise(trise)
    t0_frac = _check_t0_frac(t0_frac)
    phi0 = _check_phi0(phi0)
    amplitude = _check_amplitude(amplitude)
    cfo_ppm = _check_cfo_ppm(cfo_ppm)
    # And the OTHER factor of the same product: cfo_ppm and f0 multiply
    # together into the carrier term, so both are checked. See _check_f0.
    f0 = _check_f0(f0)
    hop_period, hop_freqs = _check_hops(hop_period, hop_freqs, fs, BW)
    # The raised edge is trise seconds, so Nrise = ceil(trise*fs) samples
    # against a symbol of N = 2**SF * OSF. Nrise > N reduces to
    # BW > 2**SF / Trise regardless of the rate, and past that the edge slice
    # p[N - 1 + np.arange(-Nrise + 1, 1)] runs off the FRONT of the preamble.
    # numpy wraps a negative index instead of raising, so the frame would be
    # tapered with samples from the wrong end, and the C++ port would read
    # off the heap. 640 kHz at SF5, 1.28 MHz at SF6;
    # the SX1280's 812 kHz and 1625 kHz reach it at low SF.
    # Against the RESOLVED trise, not the module constant: a caller who
    # shortens the edge is entitled to the wider bandwidth that allows, and
    # one who lengthens it must be held to the tighter limit.
    _check_edge_bandwidth(BW, SF, trise)
    ldro_enabled(SF, BW, ldro)        # validates ldro eagerly; result unused

    OSF = _osf_checked
    Ts = 1 / fs
    # No Nrise here: complex_lora_packet derives the taper length itself.
    K = np.power(2, SF)
    N = K * OSF

    # THE TOTAL, before anything is allocated. The per-parameter bounds name
    # the offending knob, which is what they are for, but they do not bound
    # MEMORY: preamble_bits = 65535 is the maximum ALLOWED value and still
    # asks for a 64 GiB preamble at SF12 with fs/BW = 16. Checked HERE,
    # before complex_lora_packet builds anything, so an oversized request
    # is an error naming it rather than a MemoryError.
    _lead = _resolve_lead_in(lead_in, SF, BW, fs, preamble_bits)
    # The SPECIFIC bound first, so a lead_in cause gets a lead_in message
    # rather than the total's. complex_lora_packet checks the same constant
    # for callers who reach it directly.
    if _lead > LEAD_IN_MAX:
        raise ValueError(
            "lead_in must be <= %d samples, got %r -- it is pure padding "
            "ahead of the preamble, and the bound is what turns an absurd "
            "value into an error naming it rather than a MemoryError from "
            "inside np.zeros" % (LEAD_IN_MAX, lead_in))
    # The REAL payload length, not a constant. A flat allowance for
    # everything that is not the preamble would be ~57x high for a short
    # payload, and multiplied by N = 2**SF * fs/BW it would refuse small
    # frames: SF12 at BW 7812.5 Hz and fs 1 MHz -- the narrowest LoRa
    # bandwidth at the commonest SDR rate -- with a 2-byte payload is
    # 127 MB. An error message naming a size the frame does not have is
    # worse than no guard at all.
    #
    # 32 symbols covers the rest: 8 for the first interleaver block, 2 sync
    # words, 2.25 down-chirps, and margin. The preamble is still the only
    # unbounded contributor, which is what this guard is for.
    _de_est = ldro_enabled(SF, BW, ldro)
    # [1] only: the symbol count is not used here, complex_lora_packet
    # derives the geometry itself.
    _n_bits_hdr = lora_header_init(SF, bool(implicit_header))[1]
    _n_sym = (int(preamble_bits) + 32
              + int(lora_payload_n_sym(SF, _on_air, int(enable_crc), cr,
                                       _n_bits_hdr, _de_est)))
    if _lead + _n_sym * N > MAX_FRAME_SAMPLES:
        raise ValueError(
            "the requested frame is about %d samples (%.1f GB of complex64), "
            "past the %d-sample limit -- a %d-symbol preamble and %d symbols "
            "of frame at SF%d, fs/BW %d, plus %d samples of lead-in"
            % (_lead + _n_sym * N, (_lead + _n_sym * N) * 8 / 1e9,
               MAX_FRAME_SAMPLES, preamble_bits, _n_sym, SF, int(OSF), _lead))
    # The carrier phase over the same estimate, as check_frame_size does in
    # lora_phy.cc -- the MODULATED part only: the lead-in is zeros the CFO
    # term never touches, so a long lead-in must not count against it.
    _check_carrier_phase(cfo_ppm, f0, (_n_sym * N) / fs)
    complex_samples = complex_lora_packet(K, preamble_bits, implicit_header, cr, enable_crc, src, dst, seqn, BW, OSF, SF, trise, N,
                            # cfo_ppm, plainly: _check_cfo_ppm resolved it,
                            # default included, at the top of encode().
                            Ts, fs, cfo_ppm,
                            f0, payload, t0_frac, phi0, pseudo_header,
                            sync_word, ldro, lead_in, invert_iq,
                            None, amplitude, hop_period, hop_freqs)
    return complex_samples

SF_RANGE = (5, 12)
SF_MIN_EXPLICIT_HEADER = 7

# Shortest on-air payload a LoRa CRC is defined for.
CRC_MIN_PAYLOAD = 2


def _whole(value, name):
    """int(value), or a ValueError naming `name` if it is not whole.

    A blanket int() would truncate in silence: SF=7.9 would build an SF7
    waveform, cr=1.9 a cr=1 one, preamble_bits=8.9 an 8-symbol preamble,
    and implicit_header=0.5 an EXPLICIT-header frame that bool() reads as
    implicit.
    """
    try:
        iv = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError TOO: int(inf) raises it where int(nan) raises
        # ValueError, and both get the same message naming the parameter.
        raise ValueError("%s must be a whole number, got %r" % (name, value))
    if iv != value:
        raise ValueError(
            "%s must be a whole number, got %r -- truncating it would build a "
            "frame for a different setting than the one asked for"
            % (name, value))
    return iv


def _real(value, name):
    """float(value), or a ValueError naming `name` if it is not a real number.

    The float-valued twin of _whole. A bare float() would leak CPython's
    own message for anything it cannot convert, naming neither the
    parameter nor the rule. A string is refused even when it parses: "1e6"
    is text, not a sample rate. Finiteness and range stay with each caller,
    whose rules differ.
    """
    if isinstance(value, (str, bytes)):
        raise ValueError("%s must be a real number, got %r" % (name, value))
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("%s must be a real number, got %r" % (name, value))


def _check_enable_crc(enable_crc):
    """enable_crc is a single header BIT, so it is 0 or 1.

    Any other value would be read two ways: _check_crc_length treats it as
    TRUTHY, while the header writer takes the low BIT, so enable_crc=2 would
    both demand a CRC and not carry one.
    """
    # CAUGHT, so None, nan, inf and strings get the message below rather
    # than int()'s own.
    try:
        iv = int(enable_crc)
    except (TypeError, ValueError, OverflowError):
        iv = None
    if iv is None or iv != enable_crc or iv not in (0, 1):
        raise ValueError(
            "enable_crc must be 0 or 1, got %r -- it is a single bit in the "
            "header, and any other value is read as the bit by the header "
            "writer and as a request for a CRC by the length check"
            % (enable_crc,))
    return iv


def _check_crc_length(on_air_len, enable_crc):
    """A LoRa CRC needs at least two payload bytes to exist.

    The checksum is a 16-bit shift register clocked over the payload, so the
    LAST TWO bytes are simply shifted in and appear in the result as a pure
    XOR: at length 2 the CRC is exactly those two bytes
    XORed with a constant, and at length 3 the last two still only XOR in
    while byte 0 goes through the register. With one byte only half the
    register is filled and the other half is still the initial state, so
    there is nothing the standard defines the result to be.

    gr-lora_sdr says the same thing its own way -- "CRC not supported for
    payload smaller than 2 bytes" -- and then sets the header's CRC flag
    anyway and appends no checksum, which its own receiver rejects. This
    refuses instead of emitting a frame whose checksum means nothing and
    which no other implementation can validate.

    Note this is the ON-AIR length: with the 4-byte pseudo-header a one-byte
    message is a five-byte payload and perfectly well defined.
    """
    if int(enable_crc) and int(on_air_len) < CRC_MIN_PAYLOAD:
        raise ValueError(
            "a LoRa CRC is not defined for a %d-byte payload: the checksum is "
            "a 16-bit register clocked over the payload, so it needs two bytes "
            "to fill and the last two appear in the result as a plain XOR. "
            "Send it with enable_crc=0, or with pseudo_header=True (which "
            "makes the on-air payload 4 bytes longer). gr-lora_sdr refuses the "
            "same case." % int(on_air_len))


def _check_addresses(src, dst, seqn, pseudo_header=True):
    """Reject an address the header cannot carry.

    src, dst and seqn are one byte each in the pseudo-header. Out of range
    they would be truncated -- src=300 would produce a frame that decodes as
    src=44 with a passing CRC -- or, for a negative value, fail inside
    num2binary without naming the field. check_tx_params in
    gr-sdrlora/lib/lora_phy.cc refuses the same values.

    WITHOUT THE PSEUDO-HEADER there is nowhere for them to go, and a packet
    received with standard framing carries None in all three, so that it can
    be re-sent through encode_packet(). With pseudo_header False they are
    ignored: None is accepted and every value is returned as 0, which the
    modulator never reads. check_tx_params does the same.
    """
    if not pseudo_header:
        return 0, 0, 0
    out = []
    for name, v in (("src", src), ("dst", dst), ("seqn", seqn)):
        # COERCED AND RETURNED, not merely checked, so src=200.9 is refused
        # rather than transmitted as 200; C++ refuses a float outright too.
        # CAUGHT, so None, nan, inf and strings get the message below
        # rather than int()'s own.
        try:
            iv = int(v)
        except (TypeError, ValueError, OverflowError):
            iv = None
        if iv is None or iv != v or iv < 0 or iv > 255:
            raise ValueError(
                "%s must be a whole number 0..255, got %r -- it is a single "
                "byte in the pseudo-header, and out of range it would be "
                "truncated into a frame that says something different from "
                "what was asked for" % (name, v))
        out.append(iv)
    return tuple(out)


def _check_preamble(preamble_bits):
    """Reject a frame with no preamble.

    np.tile(u0, 0) is an empty array, so preamble_bits = 0 would emit a
    frame whose preamble is simply absent -- nothing to lock onto.

    With this receiver, three up-chirps is the fewest that
    still decodes a clean SF7 frame; the standard uses eight and
    gr-lora_sdr sends eight. One is permitted because a caller
    deliberately probing the limit should be able to, but zero cannot
    be a LoRa frame at all. Mirrors the check in encoder_impl.cc.
    """
    if int(preamble_bits) < 1:
        raise ValueError(
            "preamble_bits must be >= 1, got %r -- a frame with no "
            "preamble is not a LoRa frame and no receiver can lock "
            "onto it" % (preamble_bits,))
    if int(preamble_bits) > PREAMBLE_SYMBOLS_MAX:
        raise ValueError(
            "preamble_bits must be <= %d, got %r -- that is the width of the "
            "SX1276's preamble-length register, and 8192 times the standard "
            "8"
            % (PREAMBLE_SYMBOLS_MAX, preamble_bits))


# The on-air LENGTH field is one byte, so this is the longest frame that can
# describe itself. With the 4-byte pseudo-header prefix the payload bound is
# 251; without it, 255.
MAX_ON_AIR_LENGTH = 255

# Upper bound on preamble_symbols. The SX1276's preamble-length register is
# 16 bits wide, so this is the longest preamble any of the silicon this talks
# to can be asked for -- 8192 times the standard 8. Bounded so an absurd count
# is an error naming the parameter rather than a MemoryError from inside
# np.tile. Mirrors PREAMBLE_SYMBOLS_MAX in
# gr-sdrlora/include/sdrlora/lora_phy.h.
PREAMBLE_SYMBOLS_MAX = 65535

# Upper bound on lead_in, in SAMPLES. Pure transmitter padding with no
# hardware analogue, so the bound is chosen rather than cited: 2^26 is 67.1
# million samples, 67 s of dead air at 1 Msps and 537 MB of complex64 -- 6700
# times the 10000-sample default, and far enough above any real use that
# exceeding it is a mistake rather than a choice. Mirrors LEAD_IN_MAX.
LEAD_IN_MAX = 1 << 26

# Upper bound on the whole transmitted waveform, in SAMPLES.
#
# The per-parameter bounds name the parameter, which is what they are for, but
# they do not bound MEMORY: preamble_bits = 65535 is the maximum ALLOWED value
# and still asks for a 64 GiB array at SF12 with fs/BW = 16. This bound
# catches any combination rather than each knob separately.
# 2^28 samples is 2.1 GB of complex64 -- about 18 times the largest frame a
# legal configuration can produce (SF12, 255 bytes, cr 4/8, fs/BW = 8 is
# ~12 M samples), so nothing reachable by a sane caller comes near it.
MAX_FRAME_SAMPLES = 1 << 28

# Upper bound on sync_phases. The scan advances by N/sync_phases, which is
# integer division, so a value above N gives a step of zero and the search
# cannot terminate; and robust_sync's accumulator ring is
# sync_phases * _acc_depth * N floats. Four times the largest value anything
# documents -- default_sync_phases returns 2 or 4. Mirrors
# gr::sdrlora::SYNC_PHASES_MAX.
_SYNC_PHASES_MAX = 16


def _payload_bytes(payload):
    """Coerce any supported payload container to a uint8 array.

    bytes, bytearray, memoryview, a list of ints and a str are all
    accepted, and come out as the array lora_packet consumes (it reads
    MESSAGE.size).

    Values are range-checked here rather than left to numpy's wrap-around:
    payload=[300, 7] would round-trip as [44, 7] with a PASSING CRC, the
    same silent-truncation shape _check_addresses refuses for src.
    """
    if isinstance(payload, str):
        payload = payload.encode()
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(payload), dtype=np.uint8)
    a = np.asarray(payload)
    if a.ndim == 0:
        a = a.reshape(1)
    # NUMERIC, by name: None would become a one-element object array and
    # fail inside the whole-number test below.
    if a.size and a.dtype.kind not in "biuf":
        raise ValueError(
            "payload must be bytes, a str or a sequence of whole numbers "
            "0..255, got %.60r" % (payload,))
    if a.size and a.dtype != np.uint8:
        if not np.all(np.equal(np.mod(a, 1), 0)):
            raise ValueError(
                "payload must hold whole numbers; got a non-integer value. "
                "A LoRa payload is a sequence of bytes.")
        lo, hi = int(np.min(a)), int(np.max(a))
        if lo < 0 or hi > 255:
            raise ValueError(
                "payload bytes must be 0..255, got min %d max %d -- out of "
                "range they are truncated into a frame that carries "
                "something different from what was asked for, with a passing "
                "CRC" % (lo, hi))
    return np.ascontiguousarray(a, dtype=np.uint8)


def _payload_len(payload):
    """Length of a payload in BYTES, whatever container it arrived in.

    np.asarray(b"hello world").size is 1, not 11 -- a bytes object becomes a
    0-d array of one string -- so bytes-like containers are measured with
    len().
    """
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(bytes(payload))
    a = np.asarray(payload)
    return int(a.size)


def _check_payload_length(on_air_len):
    """Reject a frame whose length the header cannot express.

    LENGTH is written with num2binary(v, 8), so above 255 it would wrap: a
    300-byte payload would produce a frame announcing 44 bytes, which
    decodes as a 44-byte packet with a passing header FCS and crc_ok = 1.
    With the CRC enabled the wrap would instead index past the checksum
    initial-state table. gr-sdrlora/lib/lora_phy.cc encode_symbols() carries
    the same rule.
    """
    if int(on_air_len) > MAX_ON_AIR_LENGTH:
        raise ValueError(
            "the on-air payload length is a single byte, so pseudo-header "
            "prefix + payload must be at most %d bytes, got %d -- above that "
            "the length field wraps and the frame describes itself wrongly"
            % (MAX_ON_AIR_LENGTH, int(on_air_len)))


def _check_implicit_payload_len(payload_len, pseudo_header):
    """The implicit-mode frame length, both ends.

    FACTORED so lora_transceiver.StreamDecoder can apply it at
    construction, the same moment gr-sdrlora's decoder_impl refuses a bad
    value (while the block is built), rather than on the first feed().
    """
    # A missing or too-small length is an error, as in the C++ (whose
    # default 0 throws), not an empty result indistinguishable from an
    # empty band.
    if payload_len is None:
        raise ValueError(
            "implicit_header needs payload_len: with no header on air "
            "the receiver cannot discover the frame length")
    # payload_len is the MESSAGE length; the 4-byte pseudo-header prefix
    # is added to it downstream. What has to be at least one byte is the
    # resulting ON-AIR length, so payload_len = 0 is legitimate when the
    # prefix supplies those four bytes -- and it round-trips at every SF.
    _min_ih = 0 if pseudo_header else 1
    _on_air = int(payload_len) + (4 if pseudo_header else 0)
    if int(payload_len) < _min_ih or _on_air > MAX_ON_AIR_LENGTH:
        raise ValueError(
            "payload_len must be %d..%d in implicit-header mode (it is "
            "the message length; the %d-byte prefix is added to it), "
            "got %r"
            % (_min_ih, MAX_ON_AIR_LENGTH - (4 if pseudo_header else 0),
               4 if pseudo_header else 0, payload_len))


# The decode options that are switches. check_decode_opts refuses anything
# else in them by name.
_DECODE_FLAGS = ("channel_filter_rx", "soft_decision", "robust_sync",
                 "fec_correct", "interpolate", "report_crc_errors",
                 "pseudo_header", "invert_iq")


def _check_flag(value, name):
    """A switch: bool, numpy bool, or the integers 0 and 1."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError("%s is a switch (True/False), got %.60r" % (name, value))


def check_decode_opts(SF, BW, fs, opts):
    """Every rule decode() applies eagerly, against a resolved option dict.

    ONE HOME FOR THE SET, so decode() and a long-lived receiver --
    lora_transceiver's resolve_decode_opts in the parent process and
    StreamDecoder.__init__ in the decoder child -- apply the same rules,
    up front. A value refused only on the first feed() would be refused
    inside decoder_process, which catches, prints and keeps feeding, and
    the heartbeat would read `packets 0 crc_err 0 overruns 0`,
    indistinguishable from a quiet channel.

    `opts` is a fully resolved dict (DECODE_DEFAULTS merged with the
    caller's), so a None means "the module default" exactly as it does
    in decode()'s signature. Rates are optional because four rules need
    them -- the integer-multiple rate itself, the hop band, the widened
    cutoff and the sync_phases divisibility -- and a caller may be
    validating options without a radio.
    """
    # WHOLE NUMBERS FIRST, and the header shape: SF5/SF6 with an explicit
    # header, report_crc_errors under the default pseudo_header, and a
    # fractional payload_len or implicit_header are all refused here.
    SF = _whole(SF, "SF")
    # THE SWITCHES FIRST, by name, before anything reads one as a truth
    # value: an array would otherwise raise numpy's "ambiguous" error.
    # implicit_header is one of them; bool(_whole()) would take 2 and -1
    # as True.
    for _k in _DECODE_FLAGS:
        if _k in opts:
            _check_flag(opts[_k], _k)
    implicit_header = _check_flag(opts.get("implicit_header", False),
                                  "implicit_header")
    payload_len = opts.get("payload_len")
    if payload_len is not None:
        payload_len = _whole(payload_len, "payload_len")
    _check_spreading_factor(SF, implicit_header)
    if implicit_header:
        # WHOLE, like SF and payload_len above: cr=2.5 must not decode as
        # CR 2, and encode(cr=2.5) refuses the same value.
        _check_coding_rate(_whole(opts.get("cr"), "cr"))
        _check_enable_crc(opts.get("enable_crc"))
        _check_implicit_payload_len(payload_len,
                                    opts.get("pseudo_header", True))
    # report_crc_errors needs an explicit header (the header FCS is what
    # tells a corrupted packet from a false lock) and standard framing
    # (with the 4-byte prefix the length is read from inside the damaged
    # payload, which under-counts). decode() refuses both.
    if opts.get("report_crc_errors") and implicit_header:
        raise ValueError(
            "report_crc_errors needs an explicit header: the header FCS is "
            "what separates a corrupted packet from a false lock, and "
            "implicit mode has no header to check")
    if opts.get("report_crc_errors") and opts.get("pseudo_header", True):
        raise ValueError(
            "report_crc_errors needs pseudo_header=False: with the 4-byte "
            "prefix the frame length is read from inside the damaged "
            "payload, so heavily corrupted packets cannot be sized and "
            "would be under-counted. Standard framing reads it from the "
            "FCS-protected PHY header")
    _sp = _check_sync_phases(opts.get("sync_phases"))
    _s1, _s2 = sync_word_symbols(opts.get("sync_word", SYNC_WORD_DEFAULT))
    check_sync_symbols(_s1, _s2, SF)
    _check_trise(opts.get("trise"))
    _check_center_freq(opts.get("center_freq"))
    _check_cutoff_ratio(opts.get("cutoff_ratio"))
    _check_numtaps(opts.get("numtaps"))
    _check_sync_threshold(opts.get("sync_threshold"))
    _check_search_tail(opts.get("search_tail_symbols"))
    _lo, _hi = _check_osf_bounds(opts.get("min_osf"),
                                 opts.get("interp_target_osf"))
    # THE RATE BEFORE ANYTHING THAT USES IT. _check_hops computes
    # nyq = (fs - BW)/2 and the LDRO rule and the hop widening both divide
    # by BW, so fs < BW is refused by name here rather than reported as a
    # hop channel outside a negative bound. decode() calls this function
    # first for the same reason.
    if BW is not None and fs is not None:
        # By keyword: positionally, interp_target_osf would bind to
        # `will_interpolate`.
        # warn_below_min=False: the sub-minimum-rate advisory belongs to
        # the decode that actually runs. What is checked here is the rule.
        _osf = _check_sample_rate(BW, fs, warn_below_min=False, min_osf=_lo,
                                  interp_target_osf=_hi)
        # THE NORMALISED PLAN from here on, not the raw option. _check_hops
        # takes any sequence and returns a tuple; the raw value -- a numpy
        # array, say, which encode() accepts -- cannot be read as a truth
        # value by the rule below.
        _hp, _hf = _check_hops(opts.get("hop_period", 0),
                               opts.get("hop_freqs"), fs, BW)
        # AND THE WIDENED CUTOFF, after the hop plan it is computed from.
        # decode() raises the cutoff to 1 + 2*max|f|/BW when a plan is in
        # force and the channel filter runs, and refuses the result above
        # 64, so the product is checked here, not only the knob: the real
        # constraint is max|f| <= 31.5*BW.
        _check_effective_cutoff(opts.get("cutoff_ratio"), _hp, _hf, BW,
                                opts.get("channel_filter_rx", True))
        # AT THE RATE THE SCAN RUNS AT. Below min_osf decode() interpolates
        # by ceil(interp_target_osf / osf) before it searches, so the symbol
        # the grid has to divide is the interpolated one: SF7 at fs/BW 1
        # with interp_target_osf=3 scans a 384-sample symbol, not a
        # 128-sample one. gr-sdrlora needs no
        # change: its effective rate is a power of two, and so is every
        # sync_phases that divides one.
        _eff = _osf
        if _osf < _lo and opts.get("interpolate", True):
            _eff = _osf * int(np.ceil(float(_hi) / _osf))
        _check_sync_phases_divides(SF, _eff, _sp)
    # ldro LAST, and with a rate already known to be usable. With no rate
    # supplied, 1.0 stands in: the result is discarded either way and only
    # the 0..2 range is being checked, which ldro_enabled does before it
    # divides.
    ldro_enabled(SF, 1.0 if BW is None else BW, opts.get("ldro", LDRO_AUTO))


def check_encode_opts(opts, fs=None):
    """The option-level rules encode() applies, against a resolved dict.

    The transmit mirror of check_decode_opts. Without it a bad value would
    pass tx_start and tx_send_burst and be refused by encode() on the first
    packet, inside the daemon worker thread: the worker would die, no
    samples would reach the radio, and every later put() would vanish with
    a traceback on the console as the only sign.

    PER-PACKET RULES ARE NOT HERE, and that is deliberate. SF, coding
    rate, CRC and payload length belong to the packet; sync-word
    separability depends on SF; and the hop BAND depends on each packet's
    own bandwidth, because encode_packet passes pkt.BW rather than the
    radio's. Checking those here against the radio's bandwidth would
    refuse a plan that a narrower-band packet legitimately carries --
    stricter than the code. What is checked is everything that does not
    depend on the packet: the hop plan's shape through _check_hops'
    bw=None mode (which bounds against fs/2, never tighter than a
    packet's real check), and every scalar knob.

    `fs` is optional: without it the hop plan's shape is still checked
    except for the band.
    """
    _check_trise(opts.get("trise"))
    _check_t0_frac(opts.get("t0_frac"))
    _check_phi0(opts.get("phi0"))
    _check_amplitude(opts.get("amplitude"))
    _check_cfo_ppm(opts.get("cfo_ppm"))
    _check_preamble(_whole(opts.get("preamble_bits", 8), "preamble_bits"))
    # The SHAPE of the sync word; its separability depends on SF and is
    # the packet's.
    sync_word_symbols(opts.get("sync_word", SYNC_WORD_DEFAULT))
    _li = opts.get("lead_in")
    if _li is not None:
        # _whole, as in encode(): int() would truncate 100.7 to 100 and
        # -0.5 to 0 -- a zero lead-in, not the default.
        try:
            _li_v = _whole(_li, "lead_in")
        except ValueError:
            raise ValueError(
                "lead_in must be a whole sample count (None or a "
                "negative value selects the default), got %r" % (_li,))
        if _li_v > LEAD_IN_MAX:
            raise ValueError(
                "lead_in must be <= %d samples, got %r -- it is pure padding "
                "ahead of the preamble" % (LEAD_IN_MAX, _li))
    # The ldro RANGE; the AUTO decision needs SF and BW and is the
    # packet's. SF7 and 1.0 stand in, and the result is discarded.
    ldro_enabled(7, 1.0, opts.get("ldro", LDRO_AUTO))
    # And the per-packet overrides, when the caller supplied them.
    if opts.get("cr") is not None:
        _check_coding_rate(_whole(opts["cr"], "cr"))
    if opts.get("enable_crc") is not None:
        _check_enable_crc(opts["enable_crc"])
    if opts.get("implicit_header") is not None:
        _check_flag(opts["implicit_header"], "implicit_header")
    # the transmit switches, as encode() checks them
    for _k in ("pseudo_header", "invert_iq"):
        if opts.get(_k) is not None:
            _check_flag(opts[_k], _k)
    _hp = opts.get("hop_period", 0)
    if fs is not None:
        _check_hops(_hp, opts.get("hop_freqs"), fs, None)
    elif _hp:
        # without a rate, everything but the band: pass an fs that cannot
        # bind, since the band is the only rule that reads it
        _check_hops(_hp, opts.get("hop_freqs"), float("inf"), None)


def _check_sync_phases_divides(SF, OSF, sync_phases):
    """The scan grid must divide the symbol.

    A grid that does not divide it slips N mod sync_phases samples every
    cycle and never returns to a symbol boundary. Whether a slipping
    grid still lands near the sync word is luck: the values that decode
    form a different erratic set at every configuration.

    Not part of _check_sync_phases because the rule needs N, and N needs
    SF and the rate. FACTORED out of decode() for the same reason as
    _check_implicit_payload_len: StreamDecoder applies it at construction
    to match decoder_impl, and one copy of the rule keeps the two in step.

    None means auto, which resolves to default_sync_phases(SF) -- 2 or 4,
    and N is 2^SF * OSF, so the default always divides. gr-sdrlora
    resolves the default before checking, for the same reason.

    Refused BY NAME rather than returning nothing: with the scan advance
    integral, a non-divisor would otherwise give a silent empty result.
    """
    if sync_phases is None:
        return
    N = (1 << int(SF)) * int(OSF)
    if int(N) % int(sync_phases):
        raise ValueError(
            "sync_phases must divide the symbol length: SF%d at fs/BW %d "
            "gives a %d-sample symbol, and %d does not divide it -- the scan "
            "window would slip %d samples every cycle and never align to a "
            "symbol boundary. Usable values here: %s"
            % (SF, int(OSF), int(N), int(sync_phases),
               int(N) % int(sync_phases),
               ", ".join(str(m) for m in range(1, _SYNC_PHASES_MAX + 1)
                         if int(N) % m == 0)))


def _check_sync_phases(sync_phases):
    """Validate the scan granularity. None, or any value below 1, means auto.

    Accepting 0 and negatives as "auto" is deliberate parity: that is what the
    C++ decoder and the GRC block mean by 0, whose dropdown labels it
    "auto (per-SF)", so a flowgraph configuration transcribed into
    lora.decode() works unchanged.
    """
    if sync_phases is None:
        return None
    # FINITE, not whole: the truncation below is deliberate and
    # documented, so this cannot route through _whole. What it must not
    # do is leak int()'s own OverflowError/ValueError for inf and nan,
    # or np.isfinite's TypeError for a string or a list, none of which
    # name the parameter.
    if not np.isfinite(_real(sync_phases, "sync_phases")):
        raise ValueError(
            "sync_phases must be finite, got %r (0 or less selects the "
            "per-SF default)" % (sync_phases,))
    v = int(sync_phases)
    if v < 1:
        return None                      # auto, as in C++ and GRC
    if v > _SYNC_PHASES_MAX:
        raise ValueError(
            "sync_phases must be at most %d (None, or any value below 1, "
            "selects the per-SF default) -- finer than that buys nothing, the "
            "scan step N/sync_phases truncates to zero, and the robust-sync "
            "accumulator grows without bound" % _SYNC_PHASES_MAX)
    return v


# Bounds for the shaping and geometry knobs.
#
# Each knob reproduces the module's standard behaviour at its default, so a
# caller who does not mention it gets byte-identical output. The GNU Radio
# blocks expose everything encode() and decode() take, these included.
#
# TRISE_MAX is generous rather than physical: the real constraint is
# BW <= 2**SF / trise, which encode() checks against the actual SF and BW.
TRISE_MAX = 1.0
AMPLITUDE_MAX = 1e6
# The filter cannot pass more than the sample rate, and fs/BW is bounded by
# the rate check, so 64 is past anything reachable.
CUTOFF_RATIO_MAX = 64.0
NUMTAPS_MAX = 8191
# 13 symbols is the 12.25-symbol pattern rounded up; see _check_search_tail.
SEARCH_TAIL_MIN = 13
SEARCH_TAIL_MAX = 4096
OSF_BOUND_MAX = 64


def _check_edge_bandwidth(BW, SF, trise):
    """Refuse a bandwidth whose raised edge is longer than one symbol.

    The edge is trise seconds, so Nrise = ceil(trise*fs) samples against
    a symbol of N = 2**SF * OSF; Nrise > N reduces to BW > 2**SF / trise
    regardless of the rate. Past that the edge slice
    p[N - 1 + np.arange(-Nrise + 1, 1)] runs off the FRONT of the
    preamble -- numpy wraps a negative index instead of raising, so the
    frame would go out tapered with samples from the wrong end, and the
    C++ port would read off the heap. 640 kHz at SF5, 1.28 MHz at SF6; the
    SX1280's 812 kHz and 1625 kHz reach it at low SF.

    A HELPER rather than two inline copies, because both encode() and
    complex_lora_packet, which is where the taper is built, apply it.
    gr-sdrlora/include/sdrlora/lora_phy.h names the same rule
    max_bandwidth_for_sf().

    Against the RESOLVED trise, not the module constant: a caller who
    shortens the edge is entitled to the wider bandwidth that allows,
    and one who lengthens it must be held to the tighter limit.
    """
    _max_bw = float(np.power(2, SF)) / float(trise)
    if float(BW) > _max_bw:
        raise ValueError(
            "BW = %g Hz is too wide for SF%d: the raised edge (trise = %g s) "
            "would be longer than one symbol, and the taper would be built "
            "from samples before the start of the preamble. The limit is "
            "2**SF / trise = %g Hz" % (BW, SF, trise, _max_bw))
    return _max_bw


def _check_trise(trise):
    """Raised-edge duration in SECONDS. None keeps the module default.

    The taper is what keeps the transmitted spectrum from splattering: the
    first and last Nrise = ceil(trise*fs) samples of the frame are shaped by
    sin^2 / cos^2. The module-level Trise is the default.
    """
    if trise is None:
        return float(Trise)
    v = _real(trise, "trise")
    if not np.isfinite(v):
        raise ValueError("trise must be finite, got %r" % (trise,))
    if v <= 0 or v > TRISE_MAX:
        raise ValueError(
            "trise must be > 0 and <= %g s, got %r -- it is the raised-edge "
            "duration in seconds, and the frame allows one edge at each "
            "end (only the leading one is a real taper; see lora_packet)"
            % (TRISE_MAX, trise))
    return v


# The lowest carrier center_freq accepts. LoRa's lowest band is 137 MHz; this
# is far below it and exists to catch a value in MHz, which would make the
# drift correction 10^6 times too large.
CENTER_FREQ_MIN = 1e6


def _check_center_freq(center_freq):
    """The RF carrier in Hz the capture was tuned to, for sample-clock
    tracking. None or 0 turns tracking off and is the default.

    Returns a float, 0.0 when off.
    """
    if center_freq is None:
        return 0.0
    v = _real(center_freq, "center_freq")
    if not np.isfinite(v) or v < 0:
        raise ValueError("center_freq must be a finite carrier frequency in "
                         "Hz, or 0/None for off, got %r" % (center_freq,))
    if 0 < v < CENTER_FREQ_MIN:
        raise ValueError(
            "center_freq is the carrier in Hz (e.g. 868e6), got %r -- below "
            "%g Hz, which reads as a value in MHz and would make the clock "
            "correction a million times too large" % (center_freq,
                                                       CENTER_FREQ_MIN))
    return v


def _check_t0_frac(t0_frac):
    """Sub-sample timing offset, in fractions of a SAMPLE.

    It is what a receiver's timing recovery has to absorb, so encode()
    exposes it for generating off-grid frames.
    """
    v = _real(0.0 if t0_frac is None else t0_frac, "t0_frac")
    if not np.isfinite(v):
        raise ValueError("t0_frac must be finite, got %r" % (t0_frac,))
    if not (-1.0 < v < 1.0):
        raise ValueError(
            "t0_frac must be in (-1, 1), got %r -- it is a fraction of one "
            "SAMPLE, and a whole sample either way is a lead_in change "
            "rather than a sub-sample offset" % (t0_frac,))
    return v


def _check_phi0(phi0):
    """Initial carrier phase, in RADIANS.

    Applied by lora_packet only (see complex_lora_packet): adding it to the
    CFO term as well would apply it twice.
    """
    v = _real(0.0 if phi0 is None else phi0, "phi0")
    if not np.isfinite(v):
        raise ValueError("phi0 must be finite, got %r" % (phi0,))
    return v


def _check_carrier_phase(cfo_ppm, f0, seconds):
    """The carrier offset's phase, 2*pi*cfo*t, must stay a finite double.

    cfo_ppm and f0 are each checked finite, but their PRODUCT has to be as
    well: the modulator computes cfo = 2*cfo_ppm*1e-6*f0 and then
    2*pi*cfo*t, and past about 1.65e304 ppm at 868 MHz (or cfo_ppm = f0 =
    1e300) that overflows, so a FINITE setting would emit every modulated
    sample as NaN with no error. Computed in the
    order the C++ check_frame_size and the GRC assert use, so the three
    agree at the boundary. `seconds` is the modulated part's duration, or
    encode()'s estimate of it -- the lead-in is zeros and has no phase.
    """
    cfo = 2.0 * float(cfo_ppm) * 1e-6 * float(f0)
    if not np.isfinite(2.0 * np.pi * cfo * float(seconds)):
        raise ValueError(
            "the carrier phase overflows: cfo_ppm %r at f0 %r is a %g Hz "
            "carrier offset, and 2*pi*cfo*t over this %.3g s frame is past "
            "the largest double, so every modulated sample would be NaN"
            % (cfo_ppm, f0, cfo, seconds))


def _check_cfo_ppm(cfo_ppm):
    """Carrier frequency offset to bake in, in ppm of f0. None = 0.

    FINITE, and nothing tighter. It multiplies into the phase of every
    modulated sample, so nan and +-inf do not fail -- they come back as a
    waveform of NaNs (only the zero lead-in finite), which decode() scrubs
    to zero and reports no packets: a transmitter producing silence and
    saying nothing.

    Reachable from the GNU Radio encoder block, which exposes cfo_ppm as a
    free-text float parameter. A large but finite ppm stays legal: what an
    extreme value costs is a frame outside the receiver's capture range,
    which is the caller's decision rather than an error -- up to the point
    where the carrier PHASE overflows, which
    _check_carrier_phase refuses once the frame is known.
    gr::sdrlora::check_tx_params carries the identical rule.
    """
    v = _real(0.0 if cfo_ppm is None else cfo_ppm, "cfo_ppm")
    if not np.isfinite(v):
        raise ValueError(
            "cfo_ppm must be finite, got %r -- it scales the carrier and a "
            "non-finite value emits a waveform of NaNs rather than failing"
            % (cfo_ppm,))
    return v


def _check_f0(f0):
    """Carrier frequency in Hz. FINITE, and nothing tighter.

    The only use the modulator makes of f0 is the carrier-offset term
    `Cfo = 2 * cfo_ppm * 1e-6 * f0`, so f0 and cfo_ppm are interchangeable
    factors of one product, and both are guarded. A non-finite f0 gives a
    waveform of NaNs, and cfo_ppm = 0 does NOT rescue it because 0 * inf is
    nan.

    It is the FIRST positional argument of encode(), and a free-text `real`
    on the GNU Radio encoder block, so both surfaces reach it.
    gr::sdrlora::check_tx_params carries the identical rule and
    sdrlora_encoder.block.yml the matching assert.

    Zero and negative stay legal: f0 = 0 means no carrier offset at all, and
    the sign of a frequency is the caller's business. Only nan and +-inf are
    refused here, exactly as for cfo_ppm -- and, once the frame is known, a
    carrier whose phase overflows: f0 and cfo_ppm are two factors of one
    product, and _check_carrier_phase refuses cfo_ppm = f0 = 1e300.
    """
    v = _real(f0, "f0")
    if not np.isfinite(v):
        raise ValueError(
            "f0 must be finite, got %r -- it multiplies cfo_ppm into the "
            "phase of every modulated sample, and a non-finite value emits "
            "a waveform of NaNs rather than failing" % (f0,))
    return v


def _check_amplitude(amplitude):
    """Output level, linear. 1.0 is the unit-magnitude chirp.

    A parameter rather than left to the caller, so the GNU Radio encoder
    block can carry it too.
    """
    v = _real(1.0 if amplitude is None else amplitude, "amplitude")
    if not np.isfinite(v):
        raise ValueError("amplitude must be finite, got %r" % (amplitude,))
    if v <= 0 or v > AMPLITUDE_MAX:
        raise ValueError(
            "amplitude must be > 0 and <= %g, got %r -- it scales the whole "
            "waveform, and 0 would emit a frame of silence that looks like a "
            "transmitter fault rather than a configuration one"
            % (AMPLITUDE_MAX, amplitude))
    return v


def _check_hdr_rows(hdr_rows, SF, implicit_header):
    """How many of the first block's codewords carry header bits.

    NOT a parameter of encode() or decode(), deliberately. It is DERIVED from
    the explicit header's own field layout: LENGTH (8 bits), the CRC-present
    flag (1), CR (3), 8 reserved, and a 5-bit FCS written to hdr[12] and
    hdr[16:20] -- 20 bits, which is five 4-bit codewords at CR 4/8. Since
    lora_header() places only hdr[0 : 4*hdr_rows], any smaller value DISCARDS
    FCS bits, and the header check can then never pass: hdr_rows of 1
    through 4 produce a frame that not even a receiver told the same value
    can decode. So it is exposed on this low-level builder, where an
    experimenter changing the header layout would also be changing
    lora_header(), and nowhere else.

    FIVE is the standard: the explicit header is 20 bits at CR 4/8, which is
    5 codewords of 4. Anything else is a frame no other LoRa device can
    receive, so this is tunable for experiment and not for interop -- which
    is why the default stays 5 and why the bound is the block's own capacity
    rather than anything a specification says.

    Ignored in implicit-header mode, where no codeword carries header bits;
    lora_packet() passes 0 there regardless.
    """
    if hdr_rows is None:
        return HDR_ROWS_DEFAULT
    v = int(hdr_rows)
    if v != hdr_rows:
        raise ValueError(
            "hdr_rows must be a whole number, got %r" % (hdr_rows,))
    if int(implicit_header):
        # Accept anything in range so a caller can flip implicit_header
        # without also having to clear hdr_rows, but it changes nothing.
        return v if v >= 0 else HDR_ROWS_DEFAULT
    # 1..4 is ACCEPTED AND WARNED. The explicit header is a fixed 20-bit
    # layout whose 5-bit FCS lives at hdr[12] and hdr[16:20], and
    # lora_header() places only hdr[0 : 4*hdr_rows] -- so anything below 5
    # discards FCS bits and the header check can never pass.
    #
    # Refusing them outright would close the door this argument exists to
    # open: it is on the low-level builder so that someone changing the
    # header layout can change lora_header() to match. But a validator
    # whose accepted range is wider than its WORKING range should say so.
    if 1 <= v < HDR_ROWS_DEFAULT:
        warnings.warn(
            "hdr_rows=%d discards FCS bits: the explicit header is 20 bits "
            "and lora_header() places only the first %d of them, so the "
            "header check can never pass and no receiver decodes the "
            "frame -- not even one told the same hdr_rows. Only %d "
            "round-trips with the stock layout; change lora_header() too "
            "if you are experimenting with a different one."
            % (v, 4 * v, HDR_ROWS_DEFAULT),
            RuntimeWarning, stacklevel=3)
    ppm = int(SF) - 2
    # The ceiling is FIVE, not the block's capacity. lora_header() places
    # hdr[0 : 4*hdr_rows] from a hdr array that is 20 bits long, so 6 would
    # read index 20 of a size-20 array. A larger first
    # block does not make a larger header -- the layout is fixed at 20
    # bits -- so 5 is the real limit at every SF.
    if v < 1 or v > HDR_ROWS_DEFAULT:
        raise ValueError(
            "hdr_rows must be 1..%d with an explicit header (the header is "
            "20 bits, which is %d codewords of 4 at CR 4/8; the first "
            "interleaver block holds %d at SF%d but the extra ones carry "
            "payload), got %r -- 5 is the standard and the only "
            "interoperable value"
            % (HDR_ROWS_DEFAULT, HDR_ROWS_DEFAULT, ppm, int(SF), hdr_rows))
    return v


def _check_cutoff_ratio(cutoff_ratio):
    """Receive filter cutoff, as a MULTIPLE of the bandwidth.

    The corner is cutoff_ratio * BW/2. 1.15 is the default because a LoRa
    chirp is not strictly bandlimited -- see channel_filter.
    """
    if cutoff_ratio is None:
        return None
    v = _real(cutoff_ratio, "cutoff_ratio")
    if not np.isfinite(v):
        raise ValueError(
            "cutoff_ratio must be finite, got %r" % (cutoff_ratio,))
    if v <= 0 or v > CUTOFF_RATIO_MAX:
        raise ValueError(
            "cutoff_ratio must be > 0 and <= %g, got %r -- it multiplies BW, "
            "so 1.0 passes exactly the channel and %g passes everything the "
            "sample rate can represent" % (CUTOFF_RATIO_MAX, cutoff_ratio,
                                           CUTOFF_RATIO_MAX))
    return v


def _check_numtaps(numtaps):
    """Receive filter length, in taps. Odd, so the delay is a whole sample."""
    if numtaps is None:
        return None
    # THROUGH _whole, which names the parameter for inf and nan alike.
    v = _whole(numtaps, "numtaps")
    if v < 3 or v > NUMTAPS_MAX:
        raise ValueError(
            "numtaps must be 3..%d, got %r" % (NUMTAPS_MAX, numtaps))
    if v % 2 == 0:
        raise ValueError(
            "numtaps must be ODD, got %r -- an even-length windowed sinc has "
            "a half-sample group delay, which shifts the whole frame off the "
            "sample grid and costs timing accuracy for nothing" % (numtaps,))
    return v


def _check_sync_threshold(sync_threshold):
    """Preamble detection threshold, as a peak-to-mean RATIO.

    The accumulated preamble spectrum's max over its mean must clear this for
    a lock. 5.5 is the default: noise alone stays below about 4.6, and
    ordinary frames sit 3.3 to 800 times above 5.5, so a wide range of
    values works. It is a parameter because the right value depends on how
    much false-lock risk a deployment can carry.
    """
    if sync_threshold is None:
        return None
    v = _real(sync_threshold, "sync_threshold")
    if not np.isfinite(v):
        raise ValueError(
            "sync_threshold must be finite, got %r" % (sync_threshold,))
    if v <= 1.0:
        raise ValueError(
            "sync_threshold must be > 1.0, got %r -- it is a peak-to-MEAN "
            "ratio, and a spectrum whose peak does not exceed its own mean "
            "is not a detection" % (sync_threshold,))
    return v


def _check_search_tail(search_tail_symbols):
    """How many symbols of run-out samples_decoding keeps unconsumed.

    A frame needs the 12.25-symbol preamble/sync/down-chirp pattern plus room
    for the header before rf_decode can judge it, so the walk stops this far
    from the end of the buffer and waits for more. Mirrors
    SEARCH_TAIL_SYMBOLS in gr-sdrlora/include/sdrlora/lora_rx.h.
    """
    if search_tail_symbols is None:
        return None
    # THROUGH _whole, which names this parameter for inf and nan.
    v = _whole(search_tail_symbols, "search_tail_symbols")
    if v < SEARCH_TAIL_MIN or v > SEARCH_TAIL_MAX:
        raise ValueError(
            "search_tail_symbols must be %d..%d, got %r -- below %d it is "
            "shorter than the preamble/sync/down-chirp pattern a frame needs "
            "before it can be judged, and every streaming caller would lose "
            "frames at buffer boundaries"
            % (SEARCH_TAIL_MIN, SEARCH_TAIL_MAX, search_tail_symbols,
               SEARCH_TAIL_MIN))
    return v


def _check_osf_bounds(min_osf, interp_target_osf):
    """The interpolation policy for rates below the demodulator's minimum.

    Returns (min_osf, interp_target_osf). The target must be at least the
    minimum -- interpolating up to something still below it clears nothing --
    and that relation is why they are validated together rather than apart.
    """
    # THROUGH _whole for the same reason as numtaps and
    # search_tail_symbols; it also enforces the wholeness these two need.
    lo = _MIN_OSF if min_osf is None else _whole(min_osf, "min_osf")
    hi = (_INTERP_TARGET_OSF if interp_target_osf is None
          else _whole(interp_target_osf, "interp_target_osf"))
    if lo < 1 or lo > OSF_BOUND_MAX:
        raise ValueError("min_osf must be 1..%d, got %r"
                         % (OSF_BOUND_MAX, min_osf))
    if hi < 1 or hi > OSF_BOUND_MAX:
        raise ValueError("interp_target_osf must be 1..%d, got %r"
                         % (OSF_BOUND_MAX, interp_target_osf))
    if hi < lo:
        raise ValueError(
            "interp_target_osf (%d) must be >= min_osf (%d) -- interpolating "
            "up to a ratio still below the minimum clears nothing, and the "
            "frame would be resampled for no benefit" % (hi, lo))
    return lo, hi


# ---------------------------------------------------------------- FHSS -----
#
# Frequency hopping as the SX127x does it: RegHopPeriod symbols per channel,
# with the frequency list supplied by the host rather than carried on the
# air. RegHopPeriod = 0 means off, which is why hop_period = 0 is the
# default and not None.
#
# The schedule is anchored at the HEADER, so the preamble, the sync words
# and the down-chirps all go out on hop_freqs[0] and hopping begins with
# the first header symbol. The SX127x anchors at the frame start, but that
# is a point the receiver cannot locate: its frame-start estimate assumes
# an 8-symbol preamble, so a frame-anchored schedule would be de-hopped
# against the wrong phase at every other preamble length. Anchoring where
# both ends agree costs nothing: hop_freqs[0] is required to be 0, so the
# acquisition channel is the only one a receiver could lock onto anyway.
#
# Because nothing before the header is rotated, at any hop period, the
# preamble search -- which runs on the buffer as received -- always sees
# an unhopped sync pattern. HOP_PERIOD_MIN = 13 is therefore a deliberate
# floor rather than a correctness limit. Below the
# 8-symbol header a frame spends more time retuning than transmitting, the
# front end still has to hold every channel at once, and no dwell-time rule
# asks for it: 400 ms at SF7 with BW 125 kHz is about 390 symbols.
HOP_PERIOD_MIN = 13
HOP_PERIOD_MAX = 65535
# The SX1276 cycles a host-managed list; 256 is past any regulatory plan.
HOP_FREQS_MAX = 256


def _check_hops(hop_period, hop_freqs, fs, bw=None):
    """Validate a hop schedule. Returns (period_in_symbols, tuple_of_offsets).

    hop_freqs are offsets from the carrier in Hz, cycled. They are bounded by
    the sample rate because a hop the receiver cannot represent is not a hop
    it can follow -- aliased back in band it looks like a different channel.
    """
    raw = 0 if hop_period is None else hop_period
    # BY NAME for every bad value: int() of nan raises a bare ValueError and
    # of inf an OverflowError, neither naming hop_period.
    try:
        period = int(raw)
    except (TypeError, ValueError, OverflowError):
        period = None
    if period is None or period != raw:
        raise ValueError(
            "hop_period must be a whole number of symbols, got %r" % (raw,))
    if period == 0:
        return 0, ()
    if period < HOP_PERIOD_MIN or period > HOP_PERIOD_MAX:
        raise ValueError(
            "hop_period must be 0 (off) or %d..%d symbols, got %r -- a dwell "
            "shorter than the 8-symbol header spends more of the frame "
            "retuning than transmitting, and the front end has to hold "
            "every channel at once either way"
            % (HOP_PERIOD_MIN, HOP_PERIOD_MAX, raw))
    # A FLAT SEQUENCE OF NUMBERS, said so: a string, a 2-D array or a None
    # element would otherwise fail inside float() without naming hop_freqs.
    if hop_freqs is not None:
        try:
            _flat = (not isinstance(hop_freqs, (str, bytes))
                     and hasattr(hop_freqs, "__len__")
                     and np.ndim(hop_freqs) == 1)
        except (TypeError, ValueError):      # ragged: numpy refuses it
            _flat = False
        if not _flat:
            raise ValueError(
                "hop_freqs must be a flat sequence of offsets in Hz, got %r"
                % (hop_freqs,))
    if hop_freqs is None or len(hop_freqs) == 0:
        raise ValueError(
            "hop_period=%d needs hop_freqs: the channel list is not carried "
            "on the air, so both ends must be given the same one" % period)
    if len(hop_freqs) > HOP_FREQS_MAX:
        raise ValueError(
            "hop_freqs holds at most %d channels, got %d"
            % (HOP_FREQS_MAX, len(hop_freqs)))
    # Bounded by what the FRONT END can hold, not just by Nyquist. A
    # hopping receiver sees every channel at once -- it does not retune with
    # the transmitter -- so the whole plan has to fit inside the sample rate
    # WITH a bandwidth's worth of chirp around the outermost channel.
    for i, f in enumerate(hop_freqs):
        # TEXT refused too, as every scalar knob refuses it: float("1e5")
        # parses, but a string is not a frequency.
        try:
            _real(f, "hop_freqs[%d]" % i)
        except ValueError:
            raise ValueError(
                "hop_freqs[%d] must be a number (Hz from the carrier), got %r"
                % (i, f))
    if float(hop_freqs[0]) != 0.0:
        raise ValueError(
            "hop_freqs[0] must be 0: the list is relative to the channel the "
            "receiver STARTS on, and it acquires the preamble at baseband. "
            "A first channel %g Hz away is outside the carrier-offset "
            "capture range (about BW/4), so the frame is never found and "
            "there is nothing to de-hop. Rotate the plan so it begins at 0."
            % float(hop_freqs[0]))
    nyq = float(fs) / 2.0
    if bw is not None:
        nyq = (float(fs) - float(bw)) / 2.0
    out = []
    for i, f in enumerate(hop_freqs):
        v = float(f)
        if not np.isfinite(v):
            raise ValueError(
                "hop_freqs[%d] must be finite, got %r" % (i, f))
        if abs(v) > nyq:
            raise ValueError(
                "hop_freqs[%d] = %g Hz is outside +-%g Hz, so a chirp "
                "centred there does not fit in the sampled band: the "
                "receiver holds every channel at once, and it needs "
                "BW/2 either side of the outermost one" % (i, v, nyq))
        out.append(v)
    return period, tuple(out)


def _check_effective_cutoff(cutoff_ratio, hop_period, hop_freqs, BW,
                            channel_filter_rx=True):
    """Refuse a hop plan whose widened receive cutoff the filter cannot take.

    The channel filter runs BEFORE the de-hop, so decode() widens its
    cutoff to cover the whole schedule and _check_cutoff_ratio caps that
    at 64. The knob and the hop band are each validated on their own; this
    checks their PRODUCT, which sets the reachable limit,
    max|f| <= 31.5*BW.

    Mirrors gr::sdrlora::check_effective_cutoff. Silent when no plan is
    in force, and silent with the channel filter OFF: decode() widens
    only under `if hop_period and hop_freqs and channel_filter_rx`, so
    with the filter off nothing is widened and refusing would be
    stricter than the code -- channel_filter_rx=False is the documented
    way to run a plan wider than 31.5*BW.
    """
    if not channel_filter_rx:
        return
    # BY LENGTH, not truthiness: a numpy array has no truth value.
    if not hop_period or hop_period <= 0 or hop_freqs is None \
            or len(hop_freqs) == 0:
        return
    if BW is None or not np.isfinite(BW) or BW <= 0:
        return          # the rate check reports that, and it runs first
    need = _hop_cutoff_ratio(hop_freqs, BW)
    have = (_CHANNEL_FILTER_CUTOFF_RATIO if cutoff_ratio is None
            else float(cutoff_ratio))
    use = max(have, need)
    if use > CUTOFF_RATIO_MAX:
        far = max(abs(float(f)) for f in hop_freqs)
        raise ValueError(
            "this hop plan needs a receive cutoff of %g * BW/2 and the "
            "channel filter accepts at most %g -- the filter runs BEFORE "
            "the de-hop, so a channel outside its passband is destroyed "
            "before the de-hop can bring it back. The reachable limit is "
            "max|hop_freqs| <= %g * BW, which is %g Hz here; this plan "
            "reaches %g Hz. Narrow the plan, raise BW, or pass "
            "channel_filter_rx=False, which skips the widening entirely."
            % (use, CUTOFF_RATIO_MAX, (CUTOFF_RATIO_MAX - 1.0) / 2.0,
               (CUTOFF_RATIO_MAX - 1.0) / 2.0 * BW, far))


def _hop_cutoff_ratio(hop_freqs, BW):
    """Smallest receive-filter cutoff, in multiples of BW, that passes them all.

    The channel filter runs BEFORE the de-hop -- it has to, because the
    de-hop needs a frame offset the preamble search has not produced yet --
    so a channel outside the passband is destroyed before anything can bring
    it back: at BW 125 kHz the default 1.15 would lose a schedule reaching
    +-150 kHz entirely, and widening to 4.0 passes it.
    """
    if hop_freqs is None or len(hop_freqs) == 0:
        return _CHANNEL_FILTER_CUTOFF_RATIO
    # BY NAME, on both sides, so this and gr::sdrlora::hop_cutoff_ratio
    # agree on the one input neither can use.
    if not (np.isfinite(BW) and BW > 0):
        raise ValueError("_hop_cutoff_ratio: BW must be finite and positive, "
                         "got %r" % (BW,))
    # AND FINITE CHANNELS, on both sides: max() would propagate a NaN by
    # position. decode() refuses such a plan before it gets here; this is
    # for the bound helper.
    _f = [float(f) for f in hop_freqs]
    if not all(np.isfinite(_f)):
        raise ValueError("_hop_cutoff_ratio: every hop frequency must be "
                         "finite, got %r" % (list(hop_freqs),))
    far = max(abs(f) for f in _f)
    return 1.0 + 2.0 * far / float(BW)


def _apply_hops(s, frame_ofs, hop_period, hop_freqs, N, fs, invert=False,
                sfo=0.0, ref=0.0):
    """Apply (or undo) a hop schedule to `s`, anchored at `frame_ofs`.

    sfo, ref: the receiver's sample-clock offset and the absolute sample at
    which its timing estimate is exact. The hop boundaries then drift
    exactly as _sym_start drifts the symbol windows, so a window never
    takes a few samples of the neighbouring channel's rotation at a block
    edge. sfo == 0 is the undrifted placement.

    The phase RESTARTS at each hop rather than running on absolute time. A
    real radio retunes its synthesiser at every hop and comes back with an
    arbitrary phase, so a continuous-phase model would be the less faithful
    of the two -- and restarting makes the transform exactly invertible, so
    the receiver does not have to agree with the transmitter about an
    absolute time origin it has no way to observe.

    Anchored at the frame, not the buffer: lead_in is padding, and a schedule
    keyed to it would move with a parameter that carries no information.
    """
    if not hop_period or hop_freqs is None or len(hop_freqs) == 0:
        return s
    step = int(hop_period) * int(N)
    sign = -1.0 if invert else 1.0
    out = np.array(s, dtype=np.complex64, copy=True)
    pos, k, n = int(frame_ofs), 0, out.size
    # A LOOP-TRIP OPTIMISATION, not a correctness guard: `lo = max(start,
    # 0)` below is what clamps the slice. Skipping whole blocks that lie
    # entirely before the buffer is cheaper than iterating them, and it
    # keeps the channel index in step while doing so.
    # gr-sdrlora/lib/lora_phy.cc's apply_hops() carries the identical block.
    if pos < 0:
        skip = (-pos) // step
        k += skip
        pos += skip * step

    def _at(p):
        # where nominal sample p of the schedule really falls
        if not sfo:
            return p
        return int(p - np.round(sfo * (p - ref)))

    while _at(pos) < n:
        start = _at(pos)
        end = min(_at(pos + step), n)
        lo = max(start, 0)
        if lo < end:
            f = hop_freqs[k % len(hop_freqs)]
            if f != 0.0:
                # t counts from the BLOCK start, not the slice start, so a
                # partially-visible first block keeps the phase the
                # transmitter used for it.
                t = (np.arange(lo - start, end - start, dtype=np.float64)
                     / float(fs))
                out[lo:end] = (out[lo:end]
                               * _cis(sign * 2.0 * np.pi * f * t)).astype(
                                   np.complex64)
        pos, k = pos + step, k + 1
    return out


# Coding rates LoRa defines: 4/5, 4/6, 4/7, 4/8, indexed 1..4.
CR_RANGE = (1, 4)


def _check_coding_rate(cr):
    """Reject a coding rate the payload builder cannot encode.

    lora_payload() picks the Hamming generator by branching on CR, so a value
    outside 1..4 would match no branch and fail deep inside the builder. The
    GNU Radio blocks say "cr must be 1..4"; this says it at the same place.
    """
    lo, hi = CR_RANGE
    if int(cr) < lo or int(cr) > hi:
        raise ValueError(
            "cr must be %d..%d (coding rate 4/%d..4/%d), got %r -- the "
            "payload is carried in (4+cr)-symbol Hamming blocks and no "
            "generator exists outside that range"
            % (lo, hi, 4 + lo, 4 + hi, cr))


def _check_spreading_factor(SF, implicit_header):
    """Validate SF, and reject SF5/SF6 with an explicit header.

    The LoRa header lives in the first SF-2 codewords at CR 4/8, which carry
    4*(SF-2) data bits. The header needs 20: SF7 gives exactly 20, SF6 gives
    16 and SF5 gives 12, so at SF5 and SF6 there is nowhere to put it. That is
    why the SX1276 datasheet mandates implicit header mode at SF6, and
    gr-lora_sdr likewise cannot decode SF5/SF6 with an explicit header at any
    sample rate, LDRO setting or coding rate.

    SF5 and SF6 therefore work here in implicit-header mode only, where the
    caller supplies the length, coding rate and CRC flag instead.
    """
    lo, hi = SF_RANGE
    if SF < lo or SF > hi:
        raise ValueError("SF must be %d..%d, got %r" % (lo, hi, SF))
    if SF < SF_MIN_EXPLICIT_HEADER and not implicit_header:
        raise ValueError(
            "SF%d has no room for an explicit header: it would need 20 bits "
            "and the header block holds only %d at this spreading factor. "
            "Use implicit_header=True and supply payload_len, cr and "
            "enable_crc." % (SF, 4 * (SF - 2)))


def default_sync_phases(SF):
    """Scan granularity to use when the caller does not choose one.

    The preamble search advances by N/sync_phases samples. What the sync-word
    check needs is a window aligned to within roughly a chip, so the step that
    matters is in chips: 2**SF / sync_phases. Two phases is half a symbol at
    every SF -- 64 chips of 128 at SF7, 16 of 32 at SF5 -- but the fine-sync
    estimate has four times less preamble to work from at SF5, and the check
    starts failing.

    At SF5 with 2 phases, whether a frame locks depends on how many zero
    samples precede it -- scan-phase dependent, the signature of too coarse a
    grid. With 4 it locks at every lead-in, including none at all. SF6 and
    SF7 lock either way.

    Hence 4 below SF7 and 2 from SF7 up. That leaves real-time performance
    at SF7 and above untouched and costs little where it does apply: an SF5
    packet is a sixteenth of an SF7 one, so four phases there are cheaper in
    absolute terms than two phases at SF7.
    """
    # _whole: int(7.5) would give SF7's answer, and None a bare TypeError
    _sf = _whole(SF, "SF")
    if not 5 <= _sf <= 12:
        raise ValueError("SF must be 5..12, got %r" % (SF,))
    return 4 if _sf < 7 else 2


_MIN_OSF = 2

# What decode() interpolates UP TO when fs/BW is below _MIN_OSF -- deliberately
# a different number. Clearing the minimum is what makes the timing estimator
# work at all; 4 is what makes it work well, and it costs nothing anywhere
# else because fs/BW = 1 is the only ratio below the minimum in the first
# place. It is not "more is better" either -- 8 is WORSE than 4 at SF8 and
# SF9, because interpolating reshuffles which aliased artefacts the sync path
# sees rather than removing them. See _check_sample_rate.
_INTERP_TARGET_OSF = 4


# THE SAMPLE-RATE BOUNDS. Mirrors gr::sdrlora::OSF_MAX, FS_MAX and
# check_rate_bounds() in gr-sdrlora/include/sdrlora/lora_phy.h, which exist
# because the C++ converts the rate to integers -- (int)(fs/bw + 0.5) is
# undefined from 2^31 up, and (long)ceil(trise*fs) at bw = fs = 1e19, both
# reachable with every finiteness check passing. Python has no such
# conversion to protect, and carries the bounds anyway so that the two
# implementations refuse the same rates.
#
# OSF_MAX: 500 MS/s over the narrowest LoRa channel, 7.8125 kHz, is 64000,
# so it admits every real front end. FS_MAX: three orders of magnitude above
# any ADC.
OSF_MAX = 65536
FS_MAX = 1e12


def _check_rate_bounds(BW, fs):
    """Finite, positive, fs <= FS_MAX and fs/BW <= OSF_MAX -- or ValueError.

    The rules every rate conversion needs, in one place, shared by
    _check_sample_rate and default_lead_in. Mirrors
    gr::sdrlora::check_rate_bounds.
    """
    # FINITE, then positive, so every bad rate gets a ValueError naming the
    # parameter rather than an OverflowError from inside the modulator.
    # REAL NUMBERS FIRST, through _real, and checked on what _real
    # RETURNED: a Python int of 2**64 passes float() but not np.isfinite.
    _bw, _fs = _real(BW, "BW"), _real(fs, "fs")
    if not (np.isfinite(_bw) and np.isfinite(_fs)):
        raise ValueError(
            "BW and fs must be finite, got BW=%r fs=%r" % (BW, fs))
    if _bw <= 0 or _fs <= 0:
        raise ValueError("BW and fs must be positive, got BW=%r fs=%r" % (BW, fs))
    if _fs > FS_MAX:
        raise ValueError(
            "fs must be <= %g Hz, got %r -- three orders of magnitude above "
            "any ADC; the bound keeps trise*fs and the hop anchor, which "
            "gr-sdrlora converts to sample indices, inside a C long"
            % (FS_MAX, fs))
    # `not (x <= bound)`, so an overflowed quotient (1e12 / 1e-300 = inf) is
    # refused here too, with the error class the C++ uses
    # (invalid_argument) rather than an OverflowError from int(round(inf)).
    if not (_fs / _bw <= OSF_MAX + 0.5):
        raise ValueError(
            "fs/BW must be <= %d, got %g -- 500 MS/s over the narrowest LoRa "
            "channel (7.8125 kHz) is 64000; the bound keeps the "
            "oversampling factor and the 2^SF * fs/BW symbol inside the "
            "integer types gr-sdrlora converts them to"
            % (OSF_MAX, _fs / _bw))


def _check_sample_rate(BW, fs, warn_below_min=True, min_osf=None,
                       will_interpolate=True,
                       interp_target_osf=None):
    """Validate fs/BW and return it as an int.

    `warn_below_min` exists because the two callers want different
    halves of this function. The integer-ratio rule applies to both:
    a transmitter at fs/BW = 7.5 produces a waveform whose chirp rate
    and symbol length disagree, which is not a LoRa frame at any rate.
    The minimum-ratio WARNING is a receiver statement, though -- it
    advises interpolating or resampling, and it describes a demodulator
    limit. Generating at fs = BW is valid, if critically sampled, so
    encode() passes False and takes the validation without the advice.

    * A non-integer ratio is rejected. The demodulator decimates the dechirped
      block by exactly OSF and indexes the preamble by whole symbols of N =
      K*OSF samples; neither has a meaning at fs/BW = 7.5. Nothing ever
      decodes, so this is an error rather than a warning -- and not an empty
      result, which would be indistinguishable from "no packet was
      transmitted".

    * A ratio below _MIN_OSF (2) is warned about. 2 itself is fully
      supported: the payload decimation starts at index 0, and index 1 would
      put the two segments of a dechirped symbol 180 degrees apart at exactly
      this ratio (see the payload loop in lora_packet_rx). SF5 is the weakest
      SF at every ratio: it has 8*32 chips of preamble to integrate against
      SF7's 8*128, so its sync estimate is the noisiest thing in the
      receiver.

      1 is usable but lossy, and warned about. It is NOT undersampled: for
      complex IQ the Nyquist rate is fs >= BW, so a chirp spanning BW is
      exactly critically sampled here. Two separate things go wrong.

      The first is fixable. The timing estimator degrades at critical
      sampling -- rf_decode's parabolic peak interpolation reads low, a gain
      of about 0.3 instead of 1 over the first half sample -- so decode()
      interpolates up to _INTERP_TARGET_OSF first (see upsample()). Without
      that, delivery depends on the sub-sample sampling instant; with it,
      every instant decodes. Shifting a frame by whole samples alone does
      not show the effect.

      Interpolating is a net win but not a pure one. At a sampling instant
      that lands exactly on a sample the timing is already right, and
      interpolating only changes which aliased artefacts the OSF-2 sync path
      sees -- at SF8 and SF9 that can lose a clean packet the
      un-interpolated path decodes. An ideal FFT interpolator does the same,
      so it is the rate rather than the kernel. Over sampling instants
      uniform on [0, 1), which is what a radio actually gives, interpolation
      delivers clearly more frames at both high and low SNR.

      The second problem is not fixable in the receiver at all. A LoRa chirp
      is not strictly bandlimited: 1.8% of a SF7 symbol's energy lies beyond
      +-BW/2, 0.34% at SF12. At fs = BW that folds in-band at the ADC, before
      the receiver exists. Band-limiting before sampling -- what an analogue
      anti-alias filter ahead of the ADC gives -- recovers the rate almost
      exactly, which identifies the loss as aliasing rather than anything
      the receiver does.

      HOW FAR to interpolate is a separate question. With the ADC modelled
      without an anti-alias filter and the same noise and sampling phase in
      both arms, 4x beats 2x significantly at SF8 and SF9 and is
      indistinguishable elsewhere: below that band the packet is short
      enough that the preamble search dominates whatever the grid is, and
      above it the processing gain carries the frame regardless. More is not
      monotonically better: 8x is worse than 4x at SF8 and SF9, because
      interpolating reshuffles which aliased artefacts the sync path sees
      rather than removing them. Hence a specific target,
      _INTERP_TARGET_OSF = 4, and not "as high as you can afford". It doubles
      the decode work at fs/BW = 1 and changes nothing at any other rate,
      since no other rate is below the minimum.
    """
    # Finite, positive and bounded -- see _check_rate_bounds.
    _check_rate_bounds(BW, fs)
    ratio = fs / BW
    osf = int(round(ratio))
    if osf < 1 or abs(ratio - osf) > 1e-9 * max(1.0, ratio):
        raise ValueError(
            "fs must be an integer multiple of BW; fs/BW = %g. The demodulator "
            "decimates by exactly this factor and steps the preamble search in "
            "whole symbols, so a fractional ratio never decodes. Resample to "
            "%g Hz (%dx) or %g Hz (%dx)."
            # TWO DISTINCT RATES: `int(ratio) + 1` alone would be 1
            # whenever ratio < 1, suggesting the same rate twice.
            % (ratio, BW * max(1, int(ratio)), max(1, int(ratio)),
               BW * (max(1, int(ratio)) + 1), max(1, int(ratio)) + 1))
    _lo = _MIN_OSF if min_osf is None else int(min_osf)
    _hi = _INTERP_TARGET_OSF if interp_target_osf is None \
        else int(interp_target_osf)
    if osf < _lo and warn_below_min:
        # WHAT THE CALLER WILL ACTUALLY GET: with interpolate=False nothing
        # is interpolated, so the message must not promise a recovery.
        _recovery = ("decode() interpolates to %dx first, so packets do "
                     "decode, but a" % _hi) if will_interpolate else (
            "interpolate=False, so the samples are demodulated at this "
            "ratio as they are and most frames will be lost. A")
        warnings.warn(
            "fs/BW = %d is below the minimum this demodulator works at (%d). "
            "%s "
            "LoRa chirp is not strictly bandlimited and at this ratio its "
            "skirts alias in-band before the receiver sees them. That part is "
            "not recoverable in software: the remedy is an analogue anti-alias "
            "filter ahead of the ADC, or sampling at %g Hz (2x BW) or %g Hz "
            "(8x BW)."
            % (osf, _lo, _recovery, BW * 2, BW * 8),
            RuntimeWarning, stacklevel=3)
    return osf


#DECODER FUNCTION. LOOKS FOR LORA PACKETS IN THE INPUT COMPLEX SAMPLES. RETURNS ALL THE PACKETS FOUND IN THE SAMPLES.

def decode(complex_samples, SF, BW, fs, channel_filter_rx=True,
           implicit_header=False, payload_len=None, cr=1, enable_crc=1,
           fec_correct=True, soft_decision=True, robust_sync=True,
           sync_phases=None, pseudo_header=True, interpolate=True,
           sync_word=SYNC_WORD_DEFAULT, ldro=LDRO_AUTO,
           report_crc_errors=False, with_consumed=False, invert_iq=False,
           trise=None, cutoff_ratio=None, numtaps=None, sync_threshold=None,
           search_tail_symbols=None, min_osf=None, interp_target_osf=None,
           hop_period=0, hop_freqs=None, center_freq=None, _search_from=0,
           _walk_end=None, _final=None):
    """Find and decode every LoRa packet in `complex_samples`.

    channel_filter_rx: band-limit the input to the LoRa channel first. Costs one
    FFT pass over the buffer and is worth about 6 dB against wideband noise at the
    usual fs/BW = 8. Pass False for the unfiltered receiver.

    implicit_header: decode frames sent with encode(..., implicit_header=1).
    There is no header on air, so `payload_len` (the message length in bytes
    handed to encode), `cr` and `enable_crc` must match the transmitter --
    the receiver needs the length before it knows how many symbols to read.

    fec_correct: use the Hamming parity bits to correct single-bit errors
    instead of discarding them. Only CR 4/7 and 4/8 can locate an error; 4/5
    and 4/6 have d_min = 2 and can only detect one. Pass False to ignore the
    parity at every rate.

    soft_decision: keep the dechirped FFT magnitudes, turn them into per-bit
    LLRs, and decode each Hamming codeword by maximum likelihood instead of
    committing to a hard argmax first. Optimal, and unlike fec_correct it helps
    at every coding rate. Supersedes fec_correct for the payload when enabled.

    robust_sync: detect the preamble from an accumulation of its identical
    up-chirps rather than from single-symbol argmax agreement. The sync-word
    requirement is unchanged and still strict, and the detection metric
    improves -- but it needs WARM-UP.

    stat_ref is zero until the ring holds _SYNC_ACC_SYMBOLS + _SYNC_ACC_LAG =
    10 spectra, which works out at about 6 SYMBOLS of buffer ahead of the sync
    word; lead-in and preamble both count towards it. It is not counted in
    scan steps (sync_phases 2, 4 and 8 behave identically), and a frame with
    lead_in=0 and a full preamble decodes at every SF and rate.

    encode()'s lead_in is counted in SAMPLES, so what it contributes shrinks
    as the symbol grows -- 9.8 symbols at SF7, 2.4 at SF9, 0.31 at SF12 -- and
    with a fixed lead-in the shortest preamble this receiver could lock onto
    would rise with SF, from 3 symbols at SF7 to 6 at SF12. default_lead_in()
    prevents that: when `lead_in` is None it tops the padding up to cover the
    shortfall in whole symbols, which leaves the standard 8-symbol preamble at
    exactly 10000 samples and makes three symbols work at every SF and every
    rate. What is left is a detection floor of three symbols that BOTH
    detectors share, which is the limit _check_preamble's docstring
    describes.

    The warm-up is kept because it is what buys the false-lock immunity:
    short-circuited to the plain chain, a decoy preamble inside the warm-up
    window would steal the lock and lose the packet behind it. Pass False for
    the plain four-condition pattern match, which is the right choice for
    truncated preambles and the wrong one near co-channel LoRa.

    What that is worth is narrower than it sounds. Against additive noise,
    an impulsive burst over a preamble symbol or a narrowband tone it buys
    nothing: those impairments degrade every symbol alike, so a chain of
    adjacent-symbol comparisons and an 8-symbol accumulation fail together.
    Where it does help is CO-CHANNEL LORA, the one interferer that can satisfy
    the plain pattern match spuriously -- a second frame is the same
    modulation, so its adjacent symbols hold plausible relationships to each
    other. The benefit is confined to roughly equal power: below it both
    paths find the preamble, above it neither does.

    sync_phases: scan granularity, window advance is N/sync_phases. This is
    what actually limits sensitivity, because the sync-word argmaxes are
    single-symbol and degrade when the window is misaligned. None (the
    default) calls default_sync_phases(SF): 2 from SF7 up, 4 below, where the
    symbol is too short for 2 to find the sync word reliably. At SF7 and above
    4 roughly triples delivered packets at -17/-18 dB but costs up to about
    1.8x the decode time, which leaves a live receiver less margin -- so there
    it is opt-in. Use it for offline analysis of recorded captures, and live
    where the host has the headroom.

    The cost depends strongly on whether a buffer CONTAINS a packet, since the
    scan stops as soon as it locks. In the live receiver's steady state --
    StreamDecoder at rx_example.py's 3,000,000-sample window, the larger of an
    idle channel and back-to-back maximum-length frames -- the default runs at
    roughly 19-43% of real time across SF7-12 and sync_phases=4 at 27-68%, SF7
    the highest, depending on the host.

    pseudo_header: this implementation prefixes every payload with its own
    4-byte DST/SRC/SEQNO/LENGTH block. That is NOT part of LoRa. Leave it True
    to interoperate with encode()'s default; pass False to receive frames from
    a standard transmitter, where the PHY header's LENGTH is the message
    length and every payload byte is message. DST/SRC/SEQNO come back as None
    in that mode because they do not exist on air. gr-lora_sdr (tapparelj)
    frames decode byte-exactly with pseudo_header=False, and not at all with
    it True -- their first four payload bytes get eaten as the prefix.

    interpolate: when `fs/BW` is below the minimum the demodulator works at
    (_MIN_OSF = 2), band-limit-interpolate the input up to _INTERP_TARGET_OSF
    instead of decoding at a rate that does not work. Only fs/BW = 1 is
    affected. Pass False to decode exactly what was handed in, and get the
    warning instead.

    center_freq: the RF carrier in Hz the capture was tuned to. Given, the
    receiver reads the sample-clock error off the carrier offset it measures
    (one crystal drives both, at the transmitter and at an SDR) and moves
    each symbol window with it; long frames at high SF need it. None or 0
    leaves it off, the default, because a capture's carrier is not in the
    samples, and because it assumes the offset is crystal error -- Doppler,
    or encode()'s cfo_ppm, which shifts the carrier only, would be read as
    clock drift. The live receiver passes its own rx_freq. See _sym_start.

    Either way, a frame that locates and fails its CRC is demodulated again
    with the drift read from the signal itself (DRIFT_RETRIES, _drift_rate),
    and kept only if the CRC then passes. That covers a carrier offset that
    is not all crystal error -- some SDR synthesizers add a few ppm the clock
    does not have -- and, with center_freq off, the whole drift. The
    carrier stays the better first guess: it costs nothing, and a frame
    without a CRC cannot be retried.

    ldro: LDRO_AUTO (the default) applies the 16 ms rule and, from SF11
    (LDRO_RETRY_MIN_SF), demodulates a frame that locates and fails its CRC
    again with LDRO the other way, kept only if the CRC then passes: Sandeep
    Mistry's Arduino LoRa 0.8.0 leaves it off at SF11 / 125 kHz. LDRO_ON or
    LDRO_OFF is taken as given.

    A frame WITHOUT a CRC (has_crc 0) is crc_ok only if its symbols look
    like signal (see _NOCRC_CHECK), so a header that passes its 5-bit
    checksum by chance on noise and claims no CRC does not come out as a
    frame of noise with crc_ok set. One that fails is treated as a failed
    CRC -- dropped, or reported with report_crc_errors -- and the walk
    steps on from it rather than skipping the span its header claimed.
    Near sensitivity it also fails some no-CRC frames that decoded wrong
    and, rarely, one that decoded right; on balance it delivers more right
    frames, since the walk does not skip what a junk lock claimed.

    `fs` must be an integer multiple of `BW` -- see _check_sample_rate.
    """
    # THROUGH THE ONE RULE SET FIRST. check_decode_opts is what the
    # long-lived receivers -- resolve_decode_opts in the parent,
    # StreamDecoder in the decoder child -- apply before any sample
    # arrives, so decode() refuses exactly what they refuse, and the rate
    # is checked ahead of the hop plan. As in encode(), a fractional
    # enable_crc or implicit_header is refused rather than coerced, so one
    # value cannot configure the two halves of the library oppositely. The
    # inline calls below remain because they also RESOLVE the values used
    # further down; they cannot refuse anything the rule set accepts.
    check_decode_opts(SF, BW, fs, {
        "implicit_header": implicit_header, "payload_len": payload_len,
        "cr": cr, "enable_crc": enable_crc, "pseudo_header": pseudo_header,
        "report_crc_errors": report_crc_errors, "sync_phases": sync_phases,
        "sync_word": sync_word, "trise": trise, "cutoff_ratio": cutoff_ratio,
        "numtaps": numtaps, "sync_threshold": sync_threshold,
        "search_tail_symbols": search_tail_symbols, "min_osf": min_osf,
        "interp_target_osf": interp_target_osf, "hop_period": hop_period,
        "hop_freqs": hop_freqs, "channel_filter_rx": channel_filter_rx,
        "ldro": ldro, "interpolate": interpolate,
        # EVERY SWITCH, so invert_iq="no" or soft_decision=2 is refused
        # by name here as it is in StreamDecoder.
        "robust_sync": robust_sync, "soft_decision": soft_decision,
        "fec_correct": fec_correct, "invert_iq": invert_iq,
        "center_freq": center_freq})
    # with_consumed is the one switch that is not a decode OPTION -- it
    # picks the return shape -- so the rule set above never sees it, and it
    # is checked by name here.
    with_consumed = _check_flag(with_consumed, "with_consumed")
    SF = _whole(SF, "SF")
    implicit_header = bool(_whole(implicit_header, "implicit_header"))
    if payload_len is not None:
        payload_len = _whole(payload_len, "payload_len")
    _check_spreading_factor(SF, implicit_header)
    # cr and enable_crc are checked only where they are READ. In
    # explicit-header mode the header carries the real coding rate and CRC
    # flag and these arguments are ignored -- callers legitimately pass
    # cr=0 as "don't care", and the C++ decoder accepts the same. In
    # implicit mode they reach the Hamming tables and `ih`.
    if implicit_header:
        cr = _whole(cr, "cr")
        _check_coding_rate(cr)
        enable_crc = _check_enable_crc(enable_crc)
        _check_implicit_payload_len(payload_len, pseudo_header)
    # Validated EAGERLY, so the same call is valid or invalid whatever the
    # buffer holds, not only once a preamble is found.
    sync_phases = _check_sync_phases(sync_phases)
    # sync_word EAGERLY too, separability as well as format:
    # sync_word_symbols checks the SHAPE of the argument; check_sync_symbols
    # rejects a sync word too close to the preamble to be told apart from
    # it. Checked only inside the search, a bad value would return [] on a
    # short buffer (indistinguishable from "no packet was transmitted") and
    # raise on a long one.
    _sw_s1, _sw_s2 = sync_word_symbols(sync_word)
    check_sync_symbols(_sw_s1, _sw_s2, _whole(SF, "SF"))
    # The receive-side knobs; every default resolves to the module's
    # standard value.
    trise = _check_trise(trise)
    center_freq = _check_center_freq(center_freq)
    cutoff_ratio = _check_cutoff_ratio(cutoff_ratio)
    numtaps = _check_numtaps(numtaps)
    sync_threshold = _check_sync_threshold(sync_threshold)
    search_tail_symbols = _check_search_tail(search_tail_symbols)
    _min_osf, _interp_target = _check_osf_bounds(min_osf, interp_target_osf)
    # Against the CALLER's fs, before any interpolation below raises it: a
    # channel list is a property of the radio, not of how this function
    # chooses to resample.
    hop_period, hop_freqs = _check_hops(hop_period, hop_freqs, fs, BW)
    if hop_period and hop_freqs and channel_filter_rx:
        # A hopping receiver does not retune with the transmitter: it
        # holds the whole plan at once, so its front end has to be wide
        # enough for the outermost channel. The filter runs BEFORE the
        # de-hop -- it must, since the de-hop needs a frame offset the
        # preamble search has not produced yet -- so a channel outside the
        # passband is destroyed before anything can bring it back.
        #
        # WIDENED, never narrowed, so a caller's own value still wins
        # wherever it is already wide enough. An explicit value that is too
        # narrow is widened rather than refused: the C++ has no way to tell
        # "unset" from "1.15", and the two implementations must validate
        # alike.
        _need = _hop_cutoff_ratio(hop_freqs, BW)
        _have = (_CHANNEL_FILTER_CUTOFF_RATIO if cutoff_ratio is None
                 else float(cutoff_ratio))
        cutoff_ratio = max(_have, _need)
    # ONE-DIMENSIONAL, checked by name: a (2, 50000) capture -- two
    # channels, the obvious mistake -- would otherwise fail inside
    # _fir_same, naming neither the argument nor decode().
    _nd = np.ndim(complex_samples)
    if _nd != 1:
        raise ValueError(
            "complex_samples must be a one-dimensional array of samples, "
            "got %d dimension(s) with shape %r. A capture with one row "
            "per channel needs the channel selected first, e.g. "
            "complex_samples[0]." % (_nd, np.shape(complex_samples)))
    # Non-finite samples become zero HERE, before anything transforms them.
    # Neither reason is cosmetic:
    #   * np.fft.fft on an inf emits "invalid value encountered in fft", so
    #     under -W error::RuntimeWarning a corrupt capture would fail with a
    #     numpy warning that names nothing, instead of decoding the frames
    #     either side of the corruption.
    #   * the detection gate is a power RATIO, and one nan would poison the
    #     buffer-wide floor every window is compared against. See
    #     _mean_power.
    # Zero is what a receiver does with a dropout: a non-finite sample carries
    # no information either way, and the frames around it still decode. Copied
    # rather than modified in place -- the caller's array is not ours.
    complex_samples = np.asarray(complex_samples)
    if (complex_samples.size
            and np.issubdtype(complex_samples.dtype, np.inexact)
            and not np.all(np.isfinite(complex_samples))):
        complex_samples = complex_samples.copy()
        complex_samples[~np.isfinite(complex_samples)] = 0
    if invert_iq:
        # Undo the transmitter's I/Q inversion before anything else looks at
        # the samples. Everything downstream -- interpolation, the channel
        # filter, the preamble search, the demodulator -- then runs unchanged
        # on a normal frame. See encode(invert_iq=True).
        complex_samples = np.conj(np.asarray(complex_samples))
        # EXCEPT the hop plan. conj() flips the sign of the transmitter's hop
        # rotation along with everything else, so the de-hop inside
        # lora_packet_rx -- which applies exp(-j2*pi*f*t) unconditionally --
        # would DOUBLE the offset rather than remove it. Negating the
        # channel list here is the correction, expressed where the reason
        # for it is visible, and it leaves _apply_hops with one sign
        # convention.
        if hop_period and hop_freqs:
            hop_freqs = tuple(-f for f in hop_freqs)
    OSF = _check_sample_rate(BW, fs, True, _min_osf,
                             bool(interpolate), _interp_target)
    # AFTER the rate check: the LDRO rule divides by BW, so a zero bandwidth
    # must be reported by the rate check first.
    ldro_enabled(SF, BW, ldro)        # validates ldro eagerly; result unused
    interp_factor = 1
    if OSF < _min_osf and interpolate:
        # Below _MIN_OSF the demodulator is fine but the timing estimator is
        # not, so interpolate up to a rate it can work at rather than refuse.
        # See upsample() for why, and why this adds nothing but sample rate.
        # The target is _INTERP_TARGET_OSF and not _MIN_OSF: clearing the
        # minimum is necessary, not sufficient.
        interp_factor = int(np.ceil(float(_interp_target) / OSF))
        complex_samples = upsample(complex_samples, fs, interp_factor)
        fs = fs * interp_factor
        OSF = OSF * interp_factor
    if channel_filter_rx:
        # cutoff_ratio and numtaps, None meaning the filter's defaults.
        complex_samples = channel_filter(complex_samples, BW, fs,
                                         cutoff_ratio, numtaps)
    Ts = 1 / fs
    Nrise = np.ceil(trise * fs)

    K = np.power(2, SF)
    N = K * OSF
    # The scan grid must DIVIDE the symbol. sync_phases is "how many evenly
    # spaced start offsets inside one symbol the search tries", and the loop
    # advances by N//sync_phases -- so unless that divides N exactly, the
    # window never returns to a symbol boundary: it slips N mod sync_phases
    # samples every cycle. Checked here, at the rate the scan runs at, and
    # refused by name; see _check_sync_phases_divides.
    _check_sync_phases_divides(SF, OSF, sync_phases)
    # A packet error rate needs to tell a corrupted packet from a false lock,
    # and the only thing that does that is the header's own FCS. Implicit mode
    # has no header -- HDR_FCS_OK is set true unconditionally -- so every
    # false lock would be counted as a corrupted packet. Refuse rather than
    # return a number made of noise.
    if report_crc_errors and implicit_header:
        raise ValueError(
            "report_crc_errors needs an explicit header: the header FCS is "
            "what separates a corrupted packet from a false lock, and "
            "implicit mode has no header to check")
    # The 4-byte pseudo-header lives INSIDE the CRC-protected payload, and the
    # interleaver spreads damage across the whole block, so its LENGTH byte is
    # corrupted along with everything else. Once the damage reaches it the
    # packet can no longer be sized and is dropped, while standard framing
    # reads the length from the FCS-protected PHY header and counts it at
    # any level of damage. That is a silent UNDER-count, which for an error
    # rate is worse than refusing.
    if report_crc_errors and pseudo_header:
        raise ValueError(
            "report_crc_errors needs pseudo_header=False: with the 4-byte "
            "prefix the frame length is read from inside the damaged payload, "
            "so heavily corrupted packets cannot be sized and would be "
            "under-counted. Standard framing reads it from the FCS-protected "
            "PHY header")
    ih = None
    if implicit_header:
        if payload_len is None:
            raise ValueError(
                "implicit_header=True needs payload_len: with no header on air "
                "the receiver cannot discover the frame length")
        # The on-air LENGTH field counts the 4-byte pseudo-header when there is
        # one, matching what lora_packet() puts in for the explicit case. With
        # standard framing there is no prefix, so the length is the message
        # length.
        ih = (int(payload_len) + (4 if pseudo_header else 0),
              int(cr), int(enable_crc))
    # _walk_end is PRIVATE too, and is the other end of the same streaming
    # contract: the walk stops there while the front end above filtered the
    # WHOLE buffer, so the samples past it -- StreamDecoder's hold-back --
    # are right-hand context for the ones before it. Handing in only the
    # samples before the hold-back would filter a frame ending just before
    # it with no right-hand context, as if there were no hold-back at all.
    # gr-sdrlora's decode_buffer takes the same walk_end, in the same
    # caller domain.
    # NEGATIVE WALKS EVERYTHING, as decode_buffer's -1 does. StreamDecoder
    # never passes one; the two mirrors still have to mean the same thing.
    if _walk_end is not None and int(_walk_end) >= 0:
        complex_samples = complex_samples[:int(_walk_end) * interp_factor]
    # _search_from is PRIVATE, for StreamDecoder: where the previous scan's
    # walk stopped, in the caller's sample domain, so it is scaled by the
    # interpolation factor exactly as `consumed` is scaled back below.
    out = samples_decoding(complex_samples, BW, N, Ts, K, OSF, Nrise, SF, trise, ih, fec_correct, soft_decision, robust_sync, sync_phases, pseudo_header, sync_word, ldro, report_crc_errors, with_consumed, sync_threshold, search_tail_symbols,
                           HDR_ROWS_DEFAULT, hop_period, hop_freqs,
                           max(0, int(_search_from)) * interp_factor,
                           # _final is PRIVATE, for StreamDecoder: a one-shot
                           # call has the whole capture, a feed that is not
                           # the last does not. Unset, a caller
                           # asking for `consumed` is scanning a stream and
                           # one that is not has the whole capture.
                           (not with_consumed) if _final is None
                           else bool(_final),
                           # NEGATED under invert_iq: conj() above flips the
                           # sign of the carrier offset the clock offset is
                           # read from, and not the clock itself.
                           -center_freq if invert_iq else center_freq)
    if with_consumed and interp_factor > 1:
        # samples_decoding counts in the INTERPOLATED domain. The caller only
        # ever saw the samples it handed in, and a streaming caller trims its
        # buffer by this number -- the interpolated count would make it
        # discard interp_factor times as much as is finished with, taking
        # live packets with it. The C++ decode_buffer makes the same
        # conversion.
        packets, consumed = out
        return packets, int(consumed // interp_factor)
    return out


def _packet_byte(value, name):
    """A LoRaPacket field stored as one byte: whole, and 0..255, by name.

    np.uint8() alone would truncate and wrap: SF=7.9 would go out at SF7,
    cr=2.5 as 4/6, src=3.5 as 3, and src=256 would raise numpy's unnamed
    "out of bounds for uint8" or, under NumPy 1.x, wrap to 0.
    pack_lora_data -> encode_packet is the path that transmits them.
    """
    v = _whole(value, "LoRaPacket %s" % name)
    if not 0 <= v <= 255:
        raise ValueError("LoRaPacket %s is one byte, 0..255, got %r"
                         % (name, value))
    return np.uint8(v)


#CLASS TO CONVENIENTLY ENCAPSULATE LORA PACKETS
class LoRaPacket:
    def __init__(self,payload,src,dst,seqn,hdr_ok,has_crc,crc_ok,cr,ih,SF,BW):
        self.payload = payload
        # src/dst/seqn live in this implementation's 4-byte pseudo-header, which
        # is not part of LoRa. In standard framing they do not exist on air, and
        # the decoder signals that with a negative value. Any negative is
        # mapped to None, "absent", BEFORE storage, and a real address goes
        # through _packet_byte, so nothing reaches np.uint8 that it would
        # wrap -- under NumPy 1.x, which does not raise OverflowError there,
        # as well as 2.
        def _addr(v, name):
            if v is None:
                return None
            v = _whole(v, "LoRaPacket %s" % name)
            return None if v < 0 else _packet_byte(v, name)
        self.src = _addr(src, "src")
        self.dst = _addr(dst, "dst")
        self.seqn = _addr(seqn, "seqn")
        # the flags are switches; the rest whole bytes
        self.hdr_ok = np.uint8(_check_flag(hdr_ok, "LoRaPacket hdr_ok"))
        self.has_crc = np.uint8(_check_flag(has_crc, "LoRaPacket has_crc"))
        self.crc_ok = np.uint8(_check_flag(crc_ok, "LoRaPacket crc_ok"))
        self.cr = _packet_byte(cr, "cr")
        self.ih = np.uint8(_check_flag(ih, "LoRaPacket ih"))
        self.SF = _packet_byte(SF, "SF")
        self.BW = BW

    def __eq__(self, other):
        # NotImplemented, not an AttributeError, so `pkt == None`,
        # `pkt != None`, `None in packets` and list.index() against a mixed
        # list all work. Returning NotImplemented lets Python fall back to
        # identity, which answers False for a foreign type and True for the
        # same object.
        if not isinstance(other, LoRaPacket):
            return NotImplemented
        # array_equal, not np.all(a == b): two payloads of different lengths
        # do not broadcast, and `pkt in received`, list.index() and
        # list.remove() over a batch of received packets must not raise.
        payload_eq = np.array_equal(np.asarray(self.payload),
                                    np.asarray(other.payload))
        src_eq = (self.src == other.src)
        dst_eq = (self.dst == other.dst)
        seqn_eq = (self.seqn == other.seqn)
        hdr_ok_eq  = (self.hdr_ok == other.hdr_ok)
        has_crc_eq = (self.has_crc == other.has_crc)
        crc_ok_eq = (self.crc_ok == other.crc_ok)
        cr_eq = (self.cr == other.cr)
        ih_eq = (self.ih == other.ih)
        SF_eq = (self.SF == other.SF)
        BW_eq = (self.BW == other.BW)

        return payload_eq and src_eq and dst_eq and seqn_eq and hdr_ok_eq and has_crc_eq and crc_ok_eq and cr_eq and ih_eq and SF_eq and BW_eq

    def __repr__(self):
        desc = "LoRa Packet Info:\n"
        sf_desc = "Spreading Factor: " + str(self.SF) + "\n"
        bw_desc = "Bandwidth: " + str(self.BW) + "\n"
        if not self.hdr_ok:
            hdr_chk = "Header Integrity Check Failed" + "\n"
            return desc + sf_desc + bw_desc + hdr_chk

        else:
            if self.ih:
                ih_desc = "Implicit Header ON" + "\n"
                if self.has_crc:
                    if self.crc_ok:
                        crc_check = "Payload Integrity Check OK" + "\n"
                        pl_str = "Payload: " + str(self.payload) + "\n"
                        pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                        return desc + sf_desc + bw_desc + ih_desc + crc_check + pl_len + pl_str
                    else:
                        crc_check = "Payload Integrity Check Failed" + "\n"
                        return desc + sf_desc + bw_desc + ih_desc + crc_check
                else:
                    crc_check = "CRC Disabled for this packet. Payload may be corrupted" + "\n"
                    pl_str = "Payload: " + str(self.payload) + "\n"
                    pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                    return desc + sf_desc + bw_desc + ih_desc +  pl_len + pl_str
            else:
                ih_desc = "Explicit Header ON" + "\n"
                hdr_chk = "Header Integrity Check OK" + "\n"
                src_desc = "Source: " + str(self.src) + "\n"
                dest_desc = "Destination: " + str(self.dst) + "\n"
                seq_desc = "Sequence number: " + str(self.seqn) + "\n"
                cr_desc = "Coding Rate: " + str(self.cr) + "\n"
                if self.has_crc:
                    if self.crc_ok:
                        crc_check = "Payload Integrity Check OK" + "\n"
                        pl_str = "Payload: " + str(self.payload) + "\n"
                        pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                        return desc + sf_desc + bw_desc + hdr_chk + ih_desc + src_desc + dest_desc + seq_desc + cr_desc + crc_check + pl_len + pl_str
                    else:
                        crc_check = "Payload Integrity Check Failed" + "\n"
                        return desc + sf_desc + bw_desc + hdr_chk + ih_desc + src_desc + dest_desc + seq_desc + cr_desc + crc_check
                else:
                    crc_check = "CRC Check Disabled for this packet. Payload may be corrupted." + "\n"
                    pl_str = "Payload: " + str(self.payload) + "\n"
                    pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                    return desc + sf_desc + bw_desc + hdr_chk + ih_desc + src_desc + dest_desc + seq_desc + cr_desc + crc_check + pl_len + pl_str


if __name__ == "__main__":
    pass


