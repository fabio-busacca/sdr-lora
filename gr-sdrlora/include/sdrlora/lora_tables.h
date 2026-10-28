/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
/* Lookup tables for the LoRa PHY. They are transcribed by hand; there is no
 * generator script. */
#ifndef INCLUDED_SDRLORA_TABLES_H
#define INCLUDED_SDRLORA_TABLES_H
#include <cstdint>
namespace gr { namespace sdrlora {

/* CRC-16 initial state for payload lengths 1..4, indexed by (length - 1).
 * CRC16_STATE below only covers 5..255, so shorter frames need entries of
 * their own rather than a negative subscript into that table.
 *
 * The entries for lengths 2, 3 and 4 are the initial states, out of all
 * 65536, that reproduce gr-lora_sdr's on-air CRC at those lengths, and they
 * hold across different payloads of each length. The same search at length 5
 * recovers CRC16_STATE[0] = 46885, which confirms the method.
 *
 * The length-1 entry has no such reference: no implementation produces a
 * one-byte CRC, because the checksum is a 16-bit register clocked over the
 * payload and one byte only half fills it. See payload_init(), which refuses
 * to transmit that case. The value is kept so the RECEIVE path stays total: a
 * frame whose header claims a CRC over one byte -- gr-lora_sdr, asked for
 * one, sets the flag and appends no checksum -- gets a computed value that
 * will not match, and is reported as crc_ok = false rather than crashing
 * the decoder. */
static const uint16_t CRC16_STATE_SHORT[4] = { 34666, 55825, 56317, 11633 };

/* Shortest on-air payload a LoRa CRC is defined for. */
static const int CRC_MIN_PAYLOAD = 2;

/* CRC-16 initial state, indexed by (payload length - 5), lengths 5..255. */
static const uint16_t CRC16_STATE[251] = {
    46885, 27367, 35014, 54790, 18706, 15954, 9784, 59350, 12042, 22321, 46211, 20984,
    56450, 7998, 62433, 35799, 2946, 47628, 30930, 52144, 59061, 10600, 56648, 10316,
    34962, 55618, 57666, 2088, 61160, 25930, 63354, 24012, 29658, 17909, 41022, 17072,
    42448, 5722, 10472, 56651, 40183, 19835, 21851, 13020, 35306, 42553, 12394, 57960,
    8434, 25101, 63814, 29049, 27264, 213, 13764, 11996, 46026, 6259, 8758, 22513,
    43163, 38423, 62727, 60460, 29548, 18211, 6559, 61900, 55362, 46606, 19928, 6028,
    35232, 29422, 28379, 55218, 38956, 12132, 49339, 47243, 39300, 53336, 29575, 53957,
    5941, 63650, 9502, 28329, 44510, 28068, 19538, 19577, 36943, 59968, 41464, 33923,
    54504, 49962, 64357, 12382, 44678, 11234, 58436, 47434, 63636, 51152, 29296, 61176,
    33231, 32706, 27862, 11005, 41129, 38527, 32824, 20579, 37742, 22493, 37464, 56698,
    29428, 27269, 7035, 27911, 55897, 50485, 10543, 38817, 54183, 52989, 24549, 33562,
    8963, 38328, 13330, 24139, 5996, 8270, 49703, 60444, 8277, 43598, 1693, 60789,
    32523, 36522, 17339, 33912, 23978, 55777, 34725, 2990, 13722, 60616, 61229, 19060,
    58889, 43920, 9043, 10131, 26896, 8918, 64347, 42307, 42863, 7853, 4844, 60762,
    21736, 62423, 53096, 19242, 55756, 26615, 53246, 11257, 2844, 47011, 10022, 13541,
    18296, 44005, 23544, 18733, 23770, 33147, 5237, 45754, 4432, 22560, 40752, 50620,
    32260, 2407, 26470, 2423, 33831, 34260, 1057, 552, 56487, 62909, 4753, 7924,
    40021, 7849, 4895, 10401, 32039, 40207, 63952, 10156, 53647, 51938, 16861, 46769,
    7703, 9288, 33345, 16184, 56808, 30265, 10696, 4218, 7708, 32139, 34174, 32428,
    20665, 3869, 43003, 6609, 60431, 22531, 11704, 63584, 13620, 14292, 37000, 8503,
    38414, 38738, 10517, 48783, 30506, 63444, 50520, 34666, 341, 34793, 26234
    /* ^ 26234 is the state for on-air LENGTH 255, the maximum.
       gr-lora_sdr's crc_verif is an independent reference, and it accepts
       maximum-length frames built with this value. */
};

/* Header FCS parity matrix, 12x5, reverse engineered. */
static const uint8_t HEADER_FCS[12][5] = {
    {1,1,0,0,0},
    {1,0,1,0,0},
    {1,0,0,1,0},
    {1,0,0,0,1},
    {0,1,1,0,0},
    {0,1,0,1,0},
    {0,1,0,0,1},
    {0,0,1,1,0},
    {0,0,1,0,1},
    {0,0,0,1,1},
    {0,0,1,1,1},
    {0,1,0,1,1},
};

/* Hamming parity matrices, 4 x CR, for CR = 1..4. */
static const uint8_t HAMMING_P1[4][1] = {
    {1},
    {1},
    {1},
    {1},
};
static const uint8_t HAMMING_P2[4][2] = {
    {1,0},
    {1,1},
    {1,1},
    {0,1},
};
static const uint8_t HAMMING_P3[4][3] = {
    {1,0,1},
    {1,1,1},
    {1,1,0},
    {0,1,1},
};
static const uint8_t HAMMING_P4[4][4] = {
    {1,0,1,1},
    {1,1,1,0},
    {1,1,0,1},
    {0,1,1,1},
};

/* Whitening LFSR feedback taps. */
static const uint8_t WHITEN_FB[8] = {0,0,0,1,1,1,0,1};

} } // namespace
#endif
