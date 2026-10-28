/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
/*
 * LoRa encoder block: PDUs in on a message port, baseband IQ out on a stream.
 *
 * The DSP lives in lora_phy.{h,cc}, which has no GNU Radio dependency. This
 * block is only plumbing around it.
 */
#ifndef INCLUDED_SDRLORA_ENCODER_H
#define INCLUDED_SDRLORA_ENCODER_H

#include <gnuradio/block.h>
#include <sdrlora/api.h>
/* for the PHY constants the parameter list below documents */
#include <sdrlora/lora_phy.h>

namespace gr {
namespace sdrlora {

class SDRLORA_API encoder : virtual public gr::block
{
public:
    typedef std::shared_ptr<encoder> sptr;

    /*!
     * \param sf                spreading factor, 5..12 (5 and 6 are
     *                          implicit-header only)
     * \param cr                1..4, coding rate 4/(4+cr)
     * \param bw                bandwidth in Hz
     * \param samp_rate         sample rate, must be an integer multiple of bw
     * \param f0                carrier, used only for the datasheet CFO term
     * \param has_crc           append the MAC CRC-16
     * \param implicit_header   send no header -- length, rate and CRC flag
     *                          are agreed out of band. The 8 symbols of the
     *                          first interleaver block are STILL SENT: all
     *                          4*(SF-2) of their data bits become payload
     *                          instead. Dropping the block outright would be
     *                          self-consistent but match nothing on air:
     *                          such frames are 3 symbols short at every SF
     *                          and do not interoperate with other
     *                          implementations. See
     *                          lora_phy.cc's header_init(). Required at SF5
     *                          and SF6, where 4*(SF-2) is under the 20 bits
     *                          a header needs.
     * \param preamble_symbols  number of preamble up-chirps
     * \param src \param dst    addresses for the in-payload pseudo-header
     * \param pseudo_header    keep the 4-byte DST/SRC/SEQNO/LENGTH prefix
     *                          inside the payload. Not part of LoRa: pass
     *                          false for a standard frame, which is required
     *                          to interoperate with other implementations.
     */
    /*! `invert_iq` swaps I and Q -- the SX127x InvertIQ setting, and the
     *  polarity a LoRaWAN DOWNLINK uses. An inverted frame does not decode as
     *  a normal one and vice versa; that separation is the point of it. See
     *  tx_params::invert_iq. */
    /* trise, t0_frac, phi0 and amplitude: shaping and output level. trise
       is the ramp taper (default TRISE) and amplitude scales the output.
       Documented here rather than inline, to keep the argument list free
       of prose. */
    /* burst_tail: zero samples closing each burst, inside it (tx_eob is on
       the last). Negative, the default, means auto: 512 * samp_rate/bw,
       4 ms at 125 kHz -- enough to push the frame's last samples out of
       the history of a filter or resampler between this block and a LoRa
       Decoder, whose idle flush then delivers it. 0 sends none,
       for a radio with nothing in between; a positive value is taken as
       given -- the decoder's window + 512 * samp_rate/bw for one with its
       idle flush turned off. At most 67108864. */
    static sptr make(int sf = 7,
                     int cr = 1,
                     double bw = 125e3,
                     double samp_rate = 1e6,
                     double f0 = 868e6,
                     bool has_crc = true,
                     bool implicit_header = false,
                     int preamble_symbols = 8,
                     int src = 0,
                     int dst = 0,
                     bool pseudo_header = true,
                     int seqn = 0,
                     double cfo_ppm = 0.0,
                     std::vector<int> sync_word = { 0x12 },
                     int ldro = 2,
                     /* negative means auto; see default_lead_in */
                     int lead_in = -1,
                     bool invert_iq = false,
                     double trise = TRISE,
                     double t0_frac = 0.0,
                     double phi0 = 0.0,
                     double amplitude = 1.0,
                     int hop_period = 0,
                     std::vector<double> hop_freqs = {},
                     /* negative means auto */
                     int burst_tail = -1);

    /*! Sequence number of the next packet; increments per PDU, mod 256.
     *  0..255, as at construction: anything else throws
     *  std::invalid_argument rather than being masked. */
    virtual void set_seqn(int seqn) = 0;
    virtual int seqn() const = 0;
};

} // namespace sdrlora
} // namespace gr

#endif
