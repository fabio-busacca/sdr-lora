# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2024 SDR-LoRa contributors.
#
# Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
# See LICENSE for the full text.


import functools
import threading
import numpy as np
#import queue as Queue
import socket
import time
import lora

# UHD's Python bindings ship with the driver, not from PyPI, so they are absent
# on any machine that just wants the PHY. Import failure must not take the
# module with it: everything here except the four USRP entry points works on
# arrays and files. See _require_uhd().
try:
    import uhd
except (ImportError, OSError):
    # ImportError: no driver on this host. OSError/FileNotFoundError: the
    # bindings are installed but their native library will not load (a
    # broken install rather than an absent one). Either way the module loads.
    uhd = None
import inspect
from collections.abc import Mapping
import multiprocessing as mp
import queue as _queue
import multiprocessing.queues as mp_queues
import sys
import traceback


def _require_uhd(what):
    """Fail with the fix rather than an AttributeError deep in a stream call."""
    if uhd is None:
        raise RuntimeError(
            "%s needs the UHD Python bindings, which are not installed. They "
            "ship with the UHD driver itself and are not on PyPI -- see "
            "https://files.ettus.com/manual/page_python.html. Everything in "
            "this module that works on files or arrays (StreamDecoder, "
            "encode_packet, the decode/encode option helpers) runs without "
            "them." % what)
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import threading
import os





class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'







np.set_printoptions(threshold=sys.maxsize)

# The fs/BW ratio this receiver is tuned for. rx() warns below it; the
# warning states the cost of running lower.
_RX_TUNED_OSF = 8


# Seed for MAX_DECODER_BACKLOG: one less than the number of CPUs, at least 1.
MAX_INFLIGHT_DECODERS = max(1, (os.cpu_count() or 2) - 1)

# Windows allowed to queue up for one spreading factor before the oldest are
# dropped. This is the overload valve: decoding happens in line, and a host
# that cannot keep up would otherwise fall further and further behind the
# ring buffer until it was reading samples the receiver had already
# overwritten.
MAX_DECODER_BACKLOG = MAX_INFLIGHT_DECODERS

# Ceiling on the ring, in windows. rx_start raises block_size to the
# decoder's worst-case lag, which is 2*(MAX_DECODER_BACKLOG+1) and so
# scales with cpu_count, and the ring costs
# complex_data_number * block_size * 8 bytes of shared memory. Past this
# the host cannot keep up with the stream, and a larger buffer would not
# change that.
#
# GB in the warnings below means 10^9 bytes (`_bytes / 1e9`).
RING_WINDOWS_MAX = 32

# Put on a packet queue when that spreading factor will never report
# again -- its decoder died, or the trigger did -- so a reader blocked in
# get() learns of it instead of hanging. decoder_process already puts at
# least one message per window (the int(0) sentinel); this covers the
# permanent case.
#
# A str rather than None, so it survives the queue unambiguously and
# reads clearly in a traceback. Readers that do not know it discard it
# like any other non-packet; readers that do can stop, as the examples do.
DECODER_STOPPED = "sdrlora:decoder-stopped"

# How long rx() waits for a decoder child to come up, and how many times
# it will restart one that keeps dying. Both waits are bounded, because
# is_alive() is False both before start() takes effect and after the
# child has exited, and an unbounded wait would wedge rx() inside its own
# setup, before anything could stop it.
CHILD_START_TIMEOUT = 30.0
CHILD_RESTART_MAX = 3
# How often the receive loop checks that the decoder children are still
# alive. Without it a decoder that dies takes its spreading factor off the
# air silently while the receiver goes on filling the ring. On a timer
# rather than per iteration, because recv() returns every few milliseconds
# and is_alive() is a kernel handle poll. Reported, not restarted: a
# decoder respawned mid-stream would come up against a ring whose contents
# it has no claim on.
CHILD_LIVENESS_PERIOD = 2.0

# Print a full RXMetadata repr for every recv(). Diagnostic only, and off:
# at a few milliseconds per call this is hundreds of lines a second on the
# receive path, and it hides the two lines that matter there -- the
# non-contiguous-stream error and the decoder heartbeat.
RX_DEBUG_METADATA = False

# How long rx_start() waits for the receiver to say its setup is complete
# before returning.
#
# Derived from the two timeouts setup can spend (the respawn sleeps and
# CHILD_START_TIMEOUT), so rx_start() can report either of the
# RuntimeErrors rx() raises by name. The bound costs a healthy start
# nothing, because rx() sets the event as soon as setup completes, and a
# failing start nothing either, because rx() records and signals before it
# raises. Only a setup that has genuinely hung waits the full time.
RX_START_GRACE = CHILD_START_TIMEOUT + CHILD_RESTART_MAX + 5.0
# How long _release() waits for each child to die, so one child that
# ignores terminate() cannot hold up shm.unlink().
CHILD_JOIN_TIMEOUT = 5.0

# Half the channel filter, which is applied `same`-mode: the first and last
# numtaps//2 samples of any buffer are filtered against zero padding rather
# than signal (this is the default numtaps' half, _CHANNEL_FILTER_TAPS//2).
# A streaming decoder holds that much back from the WALK at the end -- still
# filtering it, as the right-hand context of the samples before it -- and
# backs its consume point off by that much at the start, so every sample is
# walked with real context on both sides. Mirrors FILTER_HALF in
# gr-sdrlora/include/sdrlora/lora_rx.h.
_FILTER_HALF = lora._CHANNEL_FILTER_TAPS // 2


class StreamDecoder(object):
    """A rolling buffer that keeps only what is still live.

    Feed it samples as they arrive; it returns the packets found and retains
    the minimum needed to find the next one -- about 24 symbols on an idle
    channel, or a packet while one is arriving. The GNU Radio decoder block
    does the same with decode_buffer()'s consumed_out; this is that
    contract on the Python side.
    """

    def __init__(self, sf, BW, fs, decode_opts=None, max_payload_len=251):
        # Whole, by name, before anything sizes a buffer from it.
        max_payload_len = lora._whole(max_payload_len, "max_payload_len")
        # Range-checked under this name; _packet_symbols below does not
        # check it.
        if not 1 <= max_payload_len <= 251:
            raise ValueError(
                "max_payload_len must be 1..251, got %r -- the on-air "
                "LENGTH field is one byte and the pseudo-header spends "
                "4 of them" % (max_payload_len,))
        # Checked by name, as lora.decode does. BW and fs are required here:
        # check_decode_opts reads BW=None as "no radio" and skips the rate
        # rule, but a StreamDecoder always has a radio.
        self.sf = lora._whole(sf, "SF")
        lora._real(BW, "BW")
        lora._real(fs, "fs")
        self.BW = BW
        self.fs = fs
        # Merged onto DECODE_DEFAULTS and validated through _merge, so a
        # mistyped key is refused by name and a partial dict keeps the
        # other defaults. with_consumed is not the caller's to set.
        self.opts = _merge(DECODE_DEFAULTS, (), decode_opts, "decode")
        # Every rule, from the same list resolve_decode_opts uses, and
        # before the tail guard below: a value out of range at every
        # geometry (numtaps=99999, say) is reported as out of range rather
        # than as too large for the tail, matching the C++ block, where
        # check_rx_params runs before the tail guard.
        #
        # That includes the rate rule (fs an integer multiple of BW), so a
        # bad rate fails at construction instead of in every feed(). The
        # power-of-two rule is deliberately not applied: only the C++ sync
        # FFTs need power-of-two lengths. sync_phases divisibility is
        # checked at the effective (interpolated) rate, as decode() runs.
        lora.check_decode_opts(self.sf, BW, fs, self.opts)
        # An implicit frame must fit what max_payload_len declares. In
        # implicit mode the frame length is known here, and the buffer is
        # sized for max_payload_len + 4 + 2 bytes on air at CR 4/8 with the
        # CRC (_packet_symbols), plus the retained tail. The bound is a
        # declared envelope, not the exact edge where frames drop: how far
        # past it a frame still fits depends on CR, CRC and that tail. The
        # C++ block and the GRC palette refuse by the same rule, so the two
        # receivers agree.
        if self.opts["implicit_header"]:
            _onair = int(self.opts["payload_len"]) + (
                4 if self.opts["pseudo_header"] else 0)
            if _onair > max_payload_len + 4:
                raise ValueError(
                    "implicit payload_len %d%s is %d bytes on air, past the "
                    "%d that max_payload_len=%d declares (max_payload_len + "
                    "4) and sizes the receive buffer by; past it a frame "
                    "fits only as far as the buffer's slack allows, and is "
                    "dropped beyond. Raise max_payload_len to at least %d."
                    % (int(self.opts["payload_len"]),
                       " plus the 4-byte prefix" if self.opts["pseudo_header"]
                       else "", _onair, max_payload_len + 4, max_payload_len,
                       _onair - 4))

        # A ceiling, not a working size. A lock that never resolves -- a
        # false lock on noise that looks like a packet start -- is never
        # retired, and without a ceiling the buffer would grow without bound.
        # The overlap depends on the LDRO flag actually in force, not on the
        # auto rule, so a forced flag cannot undersize the buffer.
        _de = 1 if lora.ldro_enabled(
            self.sf, BW, self.opts.get("ldro", lora.LDRO_AUTO)) else 0
        # fs/BW too: the frame length scales with the sample rate and the
        # reservation has to scale with it.
        # The search tail in force, not the module constant: the larger of
        # the search_tail_symbols option and SEARCH_TAIL_SYMBOLS. The walk
        # in samples_decoding stops that far from the end, and the buffer
        # must keep at least that much, or a raised search_tail_symbols
        # would discard samples before the walk reached them.
        # gr-sdrlora/lib/decoder_impl.cc sizes d_tail by the same rule.
        _tail_sym = self.opts.get("search_tail_symbols")
        _tail_sym = (lora.SEARCH_TAIL_SYMBOLS if _tail_sym is None
                     else max(int(_tail_sym), lora.SEARCH_TAIL_SYMBOLS))
        # The channel filter's footprint is part of the ceiling. feed()
        # holds back the last numtaps/2 samples of every scan so the next
        # one filters them with context on both sides, and keeps numtaps/2
        # samples before the consume point as the filter's left context.
        # Both are live samples; without room for them a large numtaps
        # sheds samples on an idle channel and can shed a frame in flight.
        # gr-sdrlora's d_cap carries the same term.
        #
        # Exactly-once delivery does not depend on the ceiling: feed()
        # resumes the walk where the last one stopped, so the samples the
        # rewind brings back are not searched again.
        #
        # Below min_osf the interpolator's context is held back the same
        # way, so the footprint is 2 * _hold(), which counts both.
        #
        # The footprint is added where `cap` is read (the property below),
        # from the numtaps and channel_filter_rx in force at that moment:
        # feed() reads both per call, so a ceiling frozen at construction
        # would disagree with the hold-back as soon as either option moved.
        _taps0 = self.opts.get("numtaps")
        _hold0 = (_FILTER_HALF if _taps0 is None else int(_taps0) // 2)
        _osf_i = max(1, int(round(fs / BW)))
        # The caller's fs/BW, which decides whether decode() interpolates
        # -- see _hold().
        self._osf = _osf_i
        # Exactly the C++ d_cap: one maximum-length frame (_packet_symbols,
        # as live_overlap_samples) plus the tail, with the footprint added
        # where `cap` is read. Equal ceilings make the two receivers shed
        # the same frames past max_payload_len. window_overlap_samples, with
        # its 1.1x table floor, is not used on the receive path.
        self._cap_base = ((_packet_symbols(self.sf, max_payload_len, _de)
                           + _tail_sym) * (1 << int(self.sf)) * _osf_i)
        # A working set within reach of a host: see _check_working_set.
        _check_working_set(self.sf, _osf_i, max_payload_len, _de, _tail_sym)
        # The filter's half-width must fit in the floor tail. This is a
        # conservative envelope, and the message says so: the hold-back is
        # delivery latency, and past this point it exceeds the 24-symbol
        # tail the receiver retains. Exactly-once delivery rests on three
        # mechanisms, not on this bound:
        #
        #   * feed() resumes the walk where it stopped, so the samples the
        #     rewind keeps for the filter's left context are not re-searched;
        #   * the ceiling has room for the filter's footprint (above);
        #   * lora._zero_unsupported keeps FFT roundoff in digital silence
        #     from passing the detector's relative energy gate.
        #
        # Relaxing the bound is a latency decision, not a repair.
        #
        # Anchored to the floor tail, not the knob: search_tail_symbols
        # does not move it.
        #
        # Only where the filter runs (_filter_runs: on, and above BW after
        # any interpolation -- at fs = BW without interpolation it is the
        # identity). Otherwise feed() holds back at most the interpolator's
        # context (see _hold), which does not depend on numtaps, so numtaps
        # is inert, and refusing then would be stricter than the decoder
        # this wraps.
        #
        # Counted in the caller's samples, so conservative below min_osf.
        # There decode() interpolates first (4x at the defaults) and the
        # filter runs at that rate, so its half-width is numtaps/2
        # interpolated samples -- a quarter of what the hold-back, the
        # ceiling's footprint and this bound count. At SF7/osf1 this
        # refuses numtaps 6147..8191, whose real footprint fits the tail.
        # Counting it exactly would shorten the hold-back, and with it the
        # delivery latency, at those rates; both implementations stay
        # conservative and refuse the same values.
        _sym0 = (1 << int(self.sf)) * _osf_i
        _guard_samples = lora.SEARCH_TAIL_SYMBOLS * _sym0
        if self._filter_runs() and _hold0 > _guard_samples:
            raise ValueError(
                "numtaps/2 = %d exceeds %d samples (%d symbols of %d), the "
                "floor search tail. The channel filter's half-width is held "
                "back from every scan -- it is delivery latency -- and must "
                "fit inside the tail the streaming receivers retain. Use "
                "fewer taps or a higher oversampling ratio; "
                "search_tail_symbols does not move this bound."
                % (_hold0, _guard_samples, lora.SEARCH_TAIL_SYMBOLS, _sym0))
        # Named the same on both sides: decoder::retention_samples() in
        # gr-sdrlora/include/sdrlora/decoder.h returns d_tail, the same
        # quantity.
        #
        # The ceiling's tail term, not what the walk keeps. samples_decoding
        # stops short by the raw knob (lora.py's `_search_tail`), so below
        # SEARCH_TAIL_SYMBOLS this reserves more than the walk looks at --
        # 24 symbols against 13 at search_tail_symbols=13. Safe in that
        # direction, and the same asymmetry as the C++.
        self.retention_samples = (_tail_sym * (1 << int(self.sf))
                                  * max(1, int(round(fs / BW))))
        # The eager checks run above the tail guard. They also live in
        # resolve_decode_opts, which the live receiver goes through: this
        # constructor runs in the decoder child, so it is the second line
        # of defence. A direct StreamDecoder user gets them here.
        self._buf = np.zeros(0, dtype=np.complex64)
        # Where the next scan's WALK starts, as an index into self._buf.
        # See feed(): the buffer is rewound for the filter's sake, and the
        # walk must not re-search what the rewind brought back.
        self._search_from = 0
        self.overruns = 0
        # Mirror decoder::packets() and decoder::crc_errors() in
        # gr-sdrlora/include/sdrlora/decoder.h.
        #
        # Disjoint, as on the C++ side: `packets` counts frames delivered with
        # a passing CRC, `crc_errors` counts frames whose header FCS passed
        # and whose payload CRC did not. The second is always 0 unless
        # report_crc_errors is set, because without it such a frame is never
        # returned. packets + crc_errors is the number of frames located.
        self.packets = 0
        self.crc_errors = 0

    def feed(self, samples, final=False):
        """Append samples, decode what is complete, keep what is live.

        `final` says no more samples are coming -- the end of a capture or a
        file, rather than a live radio. The last half-filter-width is held
        back from the walk normally, because those samples are filtered
        against zero padding and will be filtered properly once more arrive
        -- they are filtered now too, as the right-hand context of the
        samples before them; at the end of a stream nothing more arrives, so
        a packet ending inside that last stretch would simply be lost.
        Flushing costs a slightly degraded filter tail on samples that have
        no successor anyway.
        """
        if samples is not None and len(samples):
            self._buf = np.concatenate(
                (self._buf, np.asarray(samples, dtype=np.complex64)))

        # No scan cadence here, unlike the C++ block. decoder_impl
        # accumulates `window` samples before it scans -- and caps
        # max_noutput_items so that window is its own rather than the
        # scheduler's -- while this decodes on every feed(). Its cadence
        # is the caller's: the live receiver's decoder_process feeds one
        # whole window per call (complex_data_number samples, 3,000,000
        # in rx_example.py, 3 s at 1 MS/s), so there delivery latency is
        # bounded by that window, and a caller feeding smaller chunks gets
        # proportionally less. Both sides apply the hold-back below.
        # The hold-back: the front end's right-hand context, counted in
        # _hold().
        hold = 0 if final else self._hold()
        usable = self._buf.size - hold
        if usable <= 0:
            return []

        # try/except/raise, because the trim and the cap are below this
        # call and an exception would skip both. decoder_process catches,
        # prints and keeps feeding, so without the handler a persistent
        # error would grow the buffer by a window every time. The handler
        # applies the cap, nothing more: the buffer is kept whole below
        # the ceiling and trimmed to it above.
        # The whole buffer, walked to `usable`: the held-back samples are
        # the right-hand context of the ones before them, and slicing at
        # `usable` would filter those against zero padding instead -- see
        # _walk_end in lora.decode.
        try:
            pkts, consumed = lora.decode(self._buf, self.sf, self.BW,
                                         self.fs, with_consumed=True,
                                         _search_from=min(self._search_from,
                                                          usable),
                                         _walk_end=usable, _final=bool(final),
                                         **self.opts)
        except Exception:
            # Apply the ceiling, so a persistent error cannot turn into
            # unbounded memory growth, and move the resume point with the
            # trim. The exception still propagates.
            #
            # Counted in self.overruns, so the shedding shows up in the
            # decoder heartbeat like any other.
            if self._buf.size > self.cap:
                _trim = self._buf.size - self.cap
                self._buf = self._buf[-self.cap:]
                self._search_from = max(0, self._search_from - _trim)
                self.overruns += 1
            raise

        for _p in pkts:
            if _p.crc_ok:
                self.packets += 1
            else:
                self.crc_errors += 1

        # Rewind for the filter, not for the walk. Keeping from
        # consumed - hold gives the next scan's channel filter `hold`
        # samples of left context for everything from `consumed` on. The
        # next walk starts at `consumed` itself, because the rewound
        # samples were already searched: re-searching them would deliver
        # a frame lying inside the rewind more than once.
        keep_from = consumed - hold
        if keep_from > 0:
            keep_from = min(keep_from, self._buf.size)
            self._buf = self._buf[keep_from:]
        else:
            keep_from = 0
        self._search_from = max(0, consumed - keep_from)

        if self._buf.size > self.cap:
            # Only reachable when a located packet never completes, which means
            # it was never a packet. Drop the oldest so the search moves on
            # rather than holding a buffer that will never resolve.
            _trim = self._buf.size - self.cap
            self._buf = self._buf[-self.cap:]
            self._search_from = max(0, self._search_from - _trim)
            self.overruns += 1
        return list(pkts)

    def _hold(self):
        """The front end's right-hand context, in the caller's samples.

        What feed() holds back from every walk and keeps before the
        consume point, for the options in force now; the ceiling carries
        it twice (_footprint). gr-sdrlora's decoder_impl computes the same
        number as d_hold.

        numtaps // 2 with the channel filter on -- the configured numtaps,
        not the module default (numtaps is a decode option over 3..8191).
        None means the default, which is what _FILTER_HALF describes.

        And, below min_osf, the interpolator's. decode() first interpolates
        with a lora._INTERP_TAPS-long same-mode FIR, whose last outputs are
        computed against zero padding, so its context is held back even
        with the channel filter off. Counted as _INTERP_TAPS // 2 caller
        samples, which is conservative by the interpolation factor exactly
        as the channel filter's term is there, and enough for the two
        filters in cascade: that needs
        ceil((_INTERP_TAPS // 2 + numtaps // 2) / factor) caller samples,
        and the factor is at least 2 whenever decode() interpolates.
        """
        half = 0
        if self._filter_runs():
            _taps = self.opts.get("numtaps")
            half = _FILTER_HALF if _taps is None else int(_taps) // 2
        if self._interpolates():
            half = max(half, lora._INTERP_TAPS // 2)
        return half

    def _interpolates(self):
        """decode() interpolates first: fs/BW below min_osf, interpolate on."""
        _lo = self.opts.get("min_osf")
        _lo = lora._MIN_OSF if _lo is None else int(_lo)
        return bool(self.opts.get("interpolate", True)) and self._osf < _lo

    def _filter_runs(self):
        """The channel filter does something: it is on, and the rate it runs
        at is above BW. At fs = BW without interpolation lora.channel_filter
        returns its input, so neither its half-width is held back nor
        numtaps checked against the tail; the C++ block applies the same
        rule."""
        return (bool(self.opts.get("channel_filter_rx", True))
                and (self._osf > 1 or self._interpolates()))

    def _footprint(self):
        """2 * _hold(): the hold-back and the left context kept before the
        consume point -- for the options in force now, which is what
        feed() holds back."""
        return 2 * self._hold()

    @property
    def cap(self):
        """The buffer ceiling: one maximum-length frame, the retained tail
        and the front end's footprint (hold-back plus left context: the
        channel filter's and, below min_osf, the interpolator's).

        Equal to decoder_impl's d_cap -- see the notes in __init__.
        """
        return self._cap_base + self._footprint()

    @cap.setter
    def cap(self, value):
        # Sets the total ceiling; the footprint term keeps following the
        # options from there.
        self._cap_base = int(value) - self._footprint()

    @property
    def buffered(self):
        return int(self._buf.size)


def _check_working_set(sf, osf, max_payload_len, de, tail_symbols):
    """Refuse, by name, a receiver whose buffer is beyond any host.

    A bound on the buffer, not a promise about memory: it counts buffer
    samples, and decode's working arrays come on top (feed() peaks at
    about 41.5 bytes per buffer sample), so an accepted configuration can
    still need far more than a given host has (at the limit, ~22 GB with
    a maximum-length frame in flight). What it removes is the
    configuration no host holds at all -- SF12, BW 7812.5 Hz and fs
    512 MS/s, say, whose tail alone is 6.44e9 samples.

    One maximum-length frame plus the retained tail, counted exactly as
    gr-sdrlora's decoder_impl counts d_overlap + d_tail, so the two
    refuse the same configurations. The bound is twice
    lora.MAX_FRAME_SAMPLES: any frame the encoder will build, plus a tail
    no longer than it.
    """
    sym = (1 << int(sf)) * int(osf)
    need = (_packet_symbols(sf, max_payload_len, de) + int(tail_symbols)) * sym
    limit = 2 * lora.MAX_FRAME_SAMPLES
    if need > limit:
        raise ValueError(
            "one maximum-length frame and the retained tail come to %d "
            "samples at SF%d and fs/BW %d, past the working-set limit of %d "
            "(twice lora.MAX_FRAME_SAMPLES). Lower max_payload_len, "
            "search_tail_symbols or the sample rate"
            % (need, int(sf), int(osf), limit))


def SEARCH_TAIL_SYMBOLS_SAMPLES(sf, BW, fs):
    """The search tail in samples at this rate -- see lora.SEARCH_TAIL_SYMBOLS."""
    osf = max(1, int(round(float(fs) / float(BW))))
    return lora.SEARCH_TAIL_SYMBOLS * (1 << int(sf)) * osf


# ---------------------------------------------------------------- PHY options
#
# Everything lora.decode() takes beyond the three this receiver already knows
# (SF, BW, fs): sync word, implicit-header frame shape, LDRO, the
# sensitivity/CPU trade-offs and the front-end tuning. The GNU Radio decoder
# block takes the same set; this is that surface for the path that needs no
# GNU Radio.
DECODE_DEFAULTS = {
    "channel_filter_rx": True,
    "fec_correct": True,
    "soft_decision": True,
    "robust_sync": True,
    "sync_phases": None,
    "implicit_header": False,
    "payload_len": None,
    "cr": 1,
    "enable_crc": 1,
    "pseudo_header": True,
    "interpolate": True,
    "sync_word": lora.SYNC_WORD_DEFAULT,
    "ldro": lora.LDRO_AUTO,
    "report_crc_errors": False,
    # LoRaWAN downlink polarity. Both ends of a link must agree: an
    # inverted frame does not decode as a normal one.
    "invert_iq": False,
    # Front-end and search tuning. sync_threshold in particular is worth
    # having here: a live receiver carries different false-lock risk from
    # an offline analysis of a capture.
    "trise": None,
    "cutoff_ratio": None,
    "numtaps": None,
    "sync_threshold": None,
    "search_tail_symbols": None,
    "min_osf": None,
    "interp_target_osf": None,
    "hop_period": 0,
    "hop_freqs": None,
    # Sample-clock tracking: the carrier, in Hz. None here means
    # "the radio's own rx_freq" to rx_start(); 0 turns it off. lora.decode
    # itself treats None as off, since a capture does not carry its carrier.
    "center_freq": None,
}

# Everything lora.encode() takes that a LoRaPacket does not already carry.
# cr / enable_crc / implicit_header are absent on purpose: the packet has them
# (pkt.cr, pkt.has_crc, pkt.ih) and the transmitter honours those. They are
# still accepted here as a blanket override for all packets in a burst.
ENCODE_DEFAULTS = {
    "preamble_bits": 8,
    "pseudo_header": True,
    # A clean carrier on the live path, as lora.encode and the GRC encoder
    # default to. A simulated offset such as lora.Cfo_PPM (17, applied
    # twice: 33.7 kHz at 990 MHz, 0.27 BW at 125 kHz) would add to the
    # radio's own crystal error and cost sensitivity; pass cfo_ppm
    # explicitly to inject one on purpose.
    "cfo_ppm": 0.0,
    "sync_word": lora.SYNC_WORD_DEFAULT,
    "ldro": lora.LDRO_AUTO,
    "lead_in": None,
    "invert_iq": False,
    # Shaping and level.
    "trise": None,
    "t0_frac": 0.0,
    "phi0": 0.0,
    "amplitude": 1.0,
    # Frequency hopping. 0 is off, matching RegHopPeriod; the channel
    # list is not carried on the air, so both ends need the same one.
    "hop_period": 0,
    "hop_freqs": None,
}

_ENCODE_FROM_PACKET = ("cr", "enable_crc", "implicit_header")


def _merge(defaults, extra, opts, what):
    """Validate option names before they disappear into **kwargs.

    A typo in a dict key is silent -- lora.decode() would simply use its own
    default and the radio would look like it was configured when it was not.
    """
    out = dict(defaults)
    if opts is not None:
        # A mapping of names, refused by name otherwise: a non-string key
        # would fail obscurely in the join below, and a falsy non-dict
        # ([] or 0) must not pass as "no options".
        if not isinstance(opts, Mapping):
            raise ValueError("%s options must be a dict of option names, "
                             "got %s" % (what, type(opts).__name__))
        named = [k for k in opts if not isinstance(k, str)]
        if named:
            raise ValueError("%s option names must be strings, got %r"
                             % (what, named[0]))
        unknown = sorted(set(opts) - set(defaults) - set(extra))
        if unknown:
            raise ValueError(
                "unknown %s option(s): %s; known: %s"
                % (what, ", ".join(unknown),
                   ", ".join(sorted(set(defaults) | set(extra)))))
        out.update(opts)
    return out


def _sf_key(k):
    """A decode_opts_per_sf key as an int SF.

    Whole, by name: a 7.5 key must not configure SF7 in silence. A string
    of ASCII digits stays a key -- "7" is what a JSON or YAML config
    produces, and is accepted on purpose. ASCII, because str.isdigit()
    also takes "\u00b2", which int() does not.
    """
    if isinstance(k, str) and k.strip().isascii() and k.strip().isdigit():
        return int(k)
    return lora._whole(k, "decode_opts_per_sf key")


def _per_sf_entries(per_sf):
    """decode_opts_per_sf, keyed by int SF, each value a dict.

    Keys are normalised to int before the lookup, so {"7": {...}} -- which
    is what every JSON or YAML config file produces -- configures SF7
    rather than passing validation and then doing nothing: the
    silent-typo failure _merge exists to prevent, one level up.

    Two keys naming one SF ({7: A, "7": B}) are refused rather than one
    being kept in silence. A value that is not a mapping is refused by
    name; None is an empty entry, as opts=None is.
    """
    if not isinstance(per_sf, Mapping):
        raise ValueError("decode_opts_per_sf must map an SF to a dict of "
                         "decode options, got %s" % type(per_sf).__name__)
    by_sf, src = {}, {}
    for k, v in per_sf.items():
        n = _sf_key(k)
        if n not in range(5, 13):
            raise ValueError("decode_opts_per_sf: SF%d out of range 5..12"
                             % n)
        if n in by_sf:
            raise ValueError(
                "decode_opts_per_sf: keys %r and %r both name SF%d, and one "
                "of the two entries would be dropped in silence"
                % (src[n], k, n))
        if v is None:
            v = {}
        elif not isinstance(v, Mapping):
            raise ValueError(
                "decode_opts_per_sf[%r] must be a dict of decode options, "
                "got %s" % (k, type(v).__name__))
        by_sf[n], src[n] = dict(v), k
    return by_sf


def with_rx_carrier(decode_opts, rx_freq):
    """decode_opts with center_freq set to the radio's rx_freq, if unset.

    Sample-clock tracking on for the live receiver. A USRP
    derives its LO and its ADC clock from one reference, as a LoRa node
    does, so the carrier offset the receiver measures is the clock error
    too -- see lora._sym_start. A caller's own value, or 0 for off, wins.
    A decode_opts that is not a mapping is returned as it is, for
    resolve_decode_opts to refuse by name.
    """
    if not rx_freq:
        return decode_opts
    if decode_opts is None or (isinstance(decode_opts, Mapping)
                               and decode_opts.get("center_freq") is None):
        return dict(decode_opts or {}, center_freq=float(rx_freq))
    return decode_opts


def resolve_decode_opts(sf, opts=None, per_sf=None, BW=None, fs=None):
    """Decode options for one spreading factor.

    `opts` applies to every SF; `per_sf` is keyed by SF and wins where it
    overlaps, because a multi-SF listen usually needs different settings per SF
    -- most obviously the implicit-header frame shape, which SF5 and SF6
    require and SF7+ do not.

    BW and fs are optional because four of the rules need a rate: the
    integer-multiple check, the hop band, the widened hop cutoff and the
    sync_phases divisibility (see lora.check_decode_opts, which applies
    them all). Both live callers have them, so both pass them; a caller
    resolving options without a radio still gets every rate-independent
    rule.

    Why here and not only in StreamDecoder: on the live path
    StreamDecoder is constructed inside decoder_process -- the child --
    whose exceptions go nowhere visible, so a rejected option would look
    like a receiver that simply never reports a packet. Every rule is
    applied here, in the parent, before anything starts.
    """
    # SF whole, by name: int(sf) below would take 7.5 as SF7.
    sf = lora._whole(sf, "SF")
    out = _merge(DECODE_DEFAULTS, (), opts, "decode")
    # `is not None`, not truthiness: [] and 0 must not pass as "no
    # entries", and a numpy array must reach the mapping check. An empty
    # dict is still no entries.
    if per_sf is not None:
        by_sf = _per_sf_entries(per_sf)
        # (Every other entry is validated at the end, below.)
        # update(), not dict(out, **entry): a non-string key in the entry
        # must reach _merge, which names it.
        merged = dict(out)
        merged.update(by_sf.get(sf, {}))
        out = _merge(DECODE_DEFAULTS, (), merged, "decode")

    # The switches first, by name, before anything below reads one as a
    # truth value, as lora.check_decode_opts does: report_crc_errors "no"
    # or 2 must be refused as a bad flag, not as "needs
    # pseudo_header=False", and an array must not reach a truth test.
    for _k in lora._DECODE_FLAGS + ("implicit_header",):
        out[_k] = lora._check_flag(out[_k], _k)

    if int(sf) < 7 and not out["implicit_header"]:
        # Same arithmetic the GNU Radio blocks enforce: the first interleaver
        # block carries 4*(SF-2) bits -- 12 at SF5, 16 at SF6 -- and a PHY
        # header needs 20. gr-lora_sdr cannot put an explicit header below SF7
        # either; it just emits nothing instead of saying so.
        raise ValueError(
            "SF%d needs implicit_header=True (an explicit header needs 20 "
            "bits and the header block holds 4*(SF-2) = %d). Pass it in "
            "decode_opts_per_sf={%d: {'implicit_header': True, "
            "'payload_len': N, 'cr': C, 'enable_crc': B}}."
            % (int(sf), 4 * (int(sf) - 2), int(sf)))

    # Counting corrupted packets needs the header FCS to tell one from a
    # false lock, and needs the frame length to come from that header rather
    # than from inside the damaged payload. lora.decode() enforces both; doing
    # it here as well means a misconfigured receive fails in rx_start()
    # instead of inside a decoder process where nothing is watching.
    if out["report_crc_errors"]:
        if out["implicit_header"]:
            raise ValueError(
                "SF%d: report_crc_errors needs an explicit header -- the "
                "header FCS is what separates a corrupted packet from a false "
                "lock, and implicit mode has no header" % int(sf))
        if out["pseudo_header"]:
            raise ValueError(
                "SF%d: report_crc_errors needs pseudo_header=False -- with "
                "the 4-byte prefix the frame length is read from inside the "
                "damaged payload, so corrupted packets would be under-counted"
                % int(sf))

    if out["implicit_header"] and out["payload_len"] is None:
        # There is no header on air to read the length from, and a wrong guess
        # decodes nothing at all rather than something short -- the de-whitening
        # and CRC both run over the assumed length.
        raise ValueError(
            "SF%d: implicit_header=True needs payload_len (no header on air "
            "carries it)" % int(sf))

    # The rest of the rules, in the parent, so that none of them first
    # fires in the decoder child or on the first window: a receiver that
    # starts and never reports. Called rather than restated:
    # lora.check_decode_opts is the same block lora.decode runs, so each
    # rule has one copy and there is one place to add one.
    lora.check_decode_opts(int(sf), BW, fs, out)

    # Every entry is validated, not only this SF's, so a typo such as
    # {8: {"sync_wrod": 1}} is refused even by rx_start([7, 9]) -- the
    # silent-typo shape _merge exists to prevent. Another SF's entry does
    # not apply here; it is resolved as its own SF would resolve it, and
    # the result dropped.
    #
    # With decode_opts applied, because that is what the entry would be
    # used with, and with the rates, so the rate rules -- sync_phases
    # divisibility, the hop band -- reach it too. A refusal names the
    # entry: decode_opts={"sync_word": 0x34} is refused at SF5, whose
    # 32 bins put the second symbol on the preamble, and the message says
    # which SF.
    #
    # The other SFs' entries last, once this SF has resolved: a bad value
    # in decode_opts itself is then refused at this SF in its own words,
    # not as an error in an entry for an SF the operator may not receive.
    if per_sf is not None:
        for k in sorted(by_sf):
            if k != sf:
                try:
                    resolve_decode_opts(k, opts, {k: by_sf[k]}, BW, fs)
                except ValueError as e:
                    raise ValueError(
                        "decode_opts_per_sf[%d], with decode_opts applied: %s"
                        % (k, e)) from None
    return out


def resolve_encode_opts(opts=None, fs=None, f0=None):
    """Transmit options, validated by name and by value.

    A bad value -- an amplitude of -5, a hop plan whose first channel is
    not 0, a sync word of 0x00 -- is refused here, at the entry point,
    rather than by lora.encode on the first packet inside the daemon
    worker thread, where the only sign would be a traceback on the
    console and a queue that stops draining. lora.check_encode_opts is the
    option-level half of encode()'s rules; the per-packet half (SF,
    sync-word separability, the hop band against the packet's own BW) is
    checked when each packet is encoded, because it depends on the packet.

    `fs` is the radio's sample rate, for the hop plan; every live caller
    has it.
    """
    out = _merge(ENCODE_DEFAULTS, _ENCODE_FROM_PACKET, opts, "encode")
    lora.check_encode_opts(out, fs)
    # The carrier phase over the shortest frame any packet could make, when
    # the carrier is known: (preamble + 32) symbols of SF5 at fs/BW 1, the
    # shortest symbol there is. Past it every packet's phase overflows, so
    # the setting is refused here, as the C++ encoder refuses it at
    # construction, rather than packet by packet in the worker thread. Per
    # packet the exact rule still runs in encode().
    if f0 is not None and fs is not None:
        lora._check_carrier_phase(
            lora._check_cfo_ppm(out["cfo_ppm"]), lora._check_f0(f0),
            (int(out["preamble_bits"]) + 32) * 32 / float(fs))
    return out


def encode_packet(center_freq, sf, pkt, sample_rate, opts=None):
    """One LoRaPacket -> IQ, honouring the packet's own cr / ih / has_crc.

    A packet built by lora_utils.pack_lora_data(..., CR=4) goes out at
    CR 4/8, and ih=1 goes out without a header. The cr, enable_crc and
    implicit_header options, when set, override the packet's fields.
    """
    # On top of the defaults, not instead of them: a partial dict that has
    # not been through resolve_encode_opts must still get this layer's
    # defaults for the options it does not name.
    o = dict(ENCODE_DEFAULTS)
    o.update(opts or {})
    cr = o.pop("cr", None)
    crc = o.pop("enable_crc", None)
    ih = o.pop("implicit_header", None)
    return lora.encode(
        center_freq, sf, pkt.BW, pkt.payload, sample_rate,
        pkt.src, pkt.dst, pkt.seqn,
        cr=int(pkt.cr) if cr is None else int(cr),
        enable_crc=int(pkt.has_crc) if crc is None else int(crc),
        implicit_header=int(pkt.ih) if ih is None else int(ih),
        **o)


# Worst-case packet -- 255 bytes at CR 4/8, fs/BW = 8 -- in samples, indexed
# from SF5: one such frame is what a packet straddling a window boundary needs
# carried between windows. Each entry is the length of the frame lora.encode()
# builds, rounded up with a margin of up to 7.5% (SF5: 226228 -> 240e3, SF6:
# 368628 -> 390e3).
SF_MIN_WINDOW_SAMPLES = np.array(
    (240e3, 390e3, 650e3, 1.2e6, 2e6, 3.6e6, 8e6, 14.5e6), dtype=np.uint32)

# The rate the table above is counted at. It is a count of samples, so it
# describes one fs/BW only; window_overlap_samples() rescales it, because the
# frame it has to cover scales with the rate.
_TABLE_OSF = 8


def _packet_symbols(sf, max_payload, de=None):
    """Symbols in the longest frame of `max_payload` bytes. Mirrors
    max_packet_symbols() in gr-sdrlora/lib/lora_sync.cc.

    `de` is the low-data-rate flag and it matters: LDRO carries two fewer bits
    per symbol, so the longest legal frame grows by about 38% at SF7. None
    resolves it from the auto rule at 125 kHz. Callers pass the flag in
    force, so a forced LDRO does not undersize the overlap; the C++ side
    takes the same parameter for the same reason.
    """
    if de is None:
        de = 1 if lora.ldro_enabled(int(sf), 125e3) else 0
    ppm = int(sf) - 2 * int(de)
    blocks = -(-8 * (int(max_payload) + 4 + 2) // (4 * ppm))
    return blocks * 8 + 9 + 13


def window_overlap_samples(sf, max_payload=251, de=None, osf=_TABLE_OSF):
    """Samples of the previous window a decoder must re-examine.

    This is one maximum-length frame -- anything less and a packet straddling
    a window boundary is lost by construction. It is the reference figure,
    the reference transceiver's table rate-scaled. The receive path does not
    use it: StreamDecoder's ceiling is the C++ d_cap (_packet_symbols plus
    the tail), and the delivery latency is [hold, window + hold) whatever
    max_payload_len is. It stays public as the reference against which the
    derived frame length can be compared.

    `max_payload` bounds it; a deployment that only ever sends 32-byte
    frames does not need to reserve for 251. The default reproduces the
    reference table. The block's parameter is called max_payload_len. 251
    also covers a 255-byte standard-framing payload: the frame is sized for
    max_payload + 4 + 2 bytes whether or not the 4-byte pseudo-header is on,
    so 252..255 are refused although standard framing can carry that many,
    and 251 is the value for them.

    `de` is the low-data-rate flag. The table holds whatever the auto rule
    gives at 125 kHz -- LDRO off below SF11, on at SF11 and SF12. Forcing it
    on below SF11 makes the longest frame about 38% longer, so pass the
    resolved flag and the reservation scales with it. None uses the auto
    rule.

    `osf` is fs/BW, and it matters for the same reason `de` does. The table
    is a count of samples at fs/BW = 8, so on its own it describes one
    sample rate: at fs/BW = 16 the frame is twice as long. The value is
    derived, as live_overlap_samples() in gr-sdrlora/lib/lora_sync.cc
    derives it, with the table kept as a rate-scaled floor so the default
    is bit-for-bit what the reference transceiver runs.

    An out-of-range SF would index the wrong end of the table rather than
    fail; rx() rejects those before any decoder process starts, and this
    asserts that it did.
    """
    if int(sf) < 5 or int(sf) > 12:
        raise ValueError("SF%d has no window size" % int(sf))
    # Whole, by name: int() would make 8.5 an 8-byte reservation in
    # silence.
    max_payload = lora._whole(max_payload, "max_payload")
    if not 1 <= max_payload <= 251:
        raise ValueError("max_payload must be 1..251, got %r" % (max_payload,))
    osf = int(osf)
    if osf < 1:
        raise ValueError("osf (fs/BW) must be >= 1, got %r" % (osf,))
    # Before _packet_symbols, which a bad `de` would drive to a zero or
    # negative bits-per-symbol: a ZeroDivisionError, or a reservation
    # collapsed through max(1, ...) to one sample.
    if de is not None and (int(de) != de or int(de) not in (0, 1)):
        # int(de) != de first: the membership test alone accepts 0.5 and
        # the string "1", because int(0.5) is 0 and int("1") is 1.
        raise ValueError(
            "window_overlap_samples: de must be 0, 1 or None, got %r -- it is "
            "the low-data-rate-optimise flag, not a symbol count" % (de,))
    # What the table already accounts for.
    table_de = 1 if lora.ldro_enabled(int(sf), 125e3) else 0
    want = _packet_symbols(sf, max_payload, de)
    have = _packet_symbols(sf, 251, table_de)
    # Arithmetic: symbols * samples-per-symbol, identical to the C++
    # live_overlap_samples(). It is a floor in the mathematical sense -- one
    # maximum-length frame exactly, with no margin -- and over the valid
    # inputs it is never the larger of the two terms: its ratio to the
    # scaled table is a per-SF constant of 0.82-0.91, independent of osf
    # and max_payload. It documents the lower bound and lets a reader
    # check the table against arithmetic.
    derived = want * (1 << int(sf)) * osf
    # The reference reservation, rate-scaled from fs/BW = 8 and
    # payload-scaled so bounding max_payload still shortens it. At the
    # default this is 1.1 * the table entry.
    floor = 1.1 * float(SF_MIN_WINDOW_SAMPLES[int(sf) - 5]) \
        * (osf / float(_TABLE_OSF)) * (want / float(have))
    return max(1, derived, int(floor))


def threshold_trigger_process(index_queue,SF_sample_th, queue_arr_tuple):
    queue_arr = np.array(queue_arr_tuple, dtype= mp_queues.Queue)
    samples_counter = np.zeros((SF_sample_th.size))

    while True:

        item = index_queue.get()
        samples_counter = samples_counter + 1
        for index, th in enumerate(SF_sample_th):
            if samples_counter[index] >= th:
                queue_arr[index].put(1)
                samples_counter[index] = 0


def decoder_process(shm_name,complex_data_number, buf_size,sf_index_queue,pkt_queue, sf, BW, fs, SF_sample_th, decode_opts=None, max_payload_len=251):
    # temp_samples = np.zeros(complex_data_number,dtype=np.complex64)



    print("Decoder on SF", str(sf), ", PID:", str(os.getpid()))

    buf_length = complex_data_number * buf_size
    # One rolling buffer, decoded in line. StreamDecoder asks
    # lora.decode(..., with_consumed=True) what it has finished with, so the
    # carry between windows is exact: about 24 symbols on an idle channel,
    # or a packet while one is arriving. The GNU Radio block does the same,
    # which keeps the two the same receiver.
    dec = StreamDecoder(sf, BW, fs, decode_opts, max_payload_len)
    # Known limit, on POSIX only: this handle is never close()d.
    #
    # Attaching to an existing segment registers it with multiprocessing's
    # resource_tracker by default on every CPython (3.13 adds a
    # `track=False` opt-out, not available at this package's 3.9 floor),
    # so at shutdown the tracker may report leaked shared_memory objects
    # after the parent has already unlink()ed the one real segment. Noise
    # rather than a leak: the parent owns the segment and _release()
    # unlinks it. Windows does not use the tracker.
    #
    # A try/finally around the loop below would not help: _release()
    # stops these children with terminate(), which is SIGTERM, and
    # CPython's default SIGTERM disposition ends the process without
    # unwinding. Running it on terminate() would need a SIGTERM handler in
    # the child, which would change how a live receiver shuts down.
    existing_shm = mp.shared_memory.SharedMemory(name=shm_name)
    rec_buffer = np.ndarray((buf_length,), dtype=np.complex64, buffer=existing_shm.buf)
    sf_windows_len = complex_data_number * SF_sample_th
    old_rec = 0
    rep_count = 0

    win_start_index = 0

    count_debug = 0
    dropped = 0

    while True:

        count_debug = count_debug + 1
        if(count_debug == 5):
            # Heartbeat, with StreamDecoder's counters, so a long capture
            # reports both how many frames it delivered and whether the
            # buffer overflowed.
            #
            # This is also where a false lock becomes visible live: an
            # unbounded search that never resolves shows as `overruns`
            # climbing while `packets` stays flat, which is otherwise
            # indistinguishable from a quiet channel.
            print(bcolors.OKGREEN
                  + "process on SF %d is alive!!! packets %d crc_err %d "
                    "overruns %d dropped %d buffered %d"
                    % (sf, dec.packets, dec.crc_errors, dec.overruns,
                       dropped, dec.buffered)
                  + bcolors.ENDC)
            count_debug = 0
        #sys.stdout.flush()
        # start_t = time.time()
        #print("Decoder")
        sf_index_queue.get()          # block until a window is ready

        # Then take whatever else has piled up behind it. Decoding is in
        # line, so a host that cannot keep up shows as a queue that grows --
        # and once it grows past the ring buffer the decoder is reading
        # samples the receiver has already overwritten. Shedding the oldest
        # is the graceful loss.
        waiting = 0
        while True:
            try:
                sf_index_queue.get_nowait()
                waiting = waiting + 1
            except _queue.Empty:
                break
        # The threshold is the CPU-count valve, not bounded by the ring
        # size: on a small ring that would shed windows a healthy decoder
        # keeps up with. The ring/backlog relation is real -- the drain
        # below takes 1 + waiting windows without re-checking, so the lag
        # reaches ~2*(MAX_DECODER_BACKLOG+1) -- but it is handled where the
        # ring is sized, in rx_start, not in what the valve discards.
        if waiting > MAX_DECODER_BACKLOG:
            skip = waiting - MAX_DECODER_BACKLOG
            win_start_index = (win_start_index
                               + skip * sf_windows_len) % buf_length
            dropped = dropped + skip
            print(bcolors.WARNING + "[SF" + str(sf) + "] decoder behind ("
                  + str(waiting) + " windows queued), dropped " + str(skip)
                  + " (total " + str(dropped) + ")" + bcolors.ENDC)
            # At least one message per window, including the shed ones.
            # A window that decodes N packets emits N messages, and the
            # int(0) sentinel only when N is zero -- so the invariant is
            # "never nothing", not a one-to-one pairing with windows.
            #
            # The reason is liveness, not accounting: a reader blocked in
            # get() has no other way to learn the window came and went,
            # so without this it would stall under precisely the load
            # that makes shedding happen.
            for _ in range(skip):
                pkt_queue.put(int(0))
            waiting = MAX_DECODER_BACKLOG

        for _ in range(1 + waiting):
            c = rec_buffer[win_start_index:win_start_index + sf_windows_len]
            win_start_index = (win_start_index + sf_windows_len) % buf_length

            # The consumer reads at least one message per window and
            # tolerates the int(0) sentinel (see rx_example.py), so a window
            # that decodes
            # nothing still has to produce one or a blocking reader stalls.
            # decode() raises on a sample rate it cannot use -- a
            # configuration mistake, not a transient -- and any other
            # exception is a bug worth seeing rather than a silent stall.
            delivered = 0
            try:
                found = dec.feed(c)
                if found:
                    print("# DECODED PACKETS", len(found))
                for pkt in found:
                    pkt_queue.put(pkt)
                    delivered = delivered + 1
            except Exception as exc:
                traceback.print_exc()
                print(bcolors.FAIL + "[SF" + str(sf) + "] decode failed: "
                      + str(exc) + bcolors.ENDC)
            if delivered == 0:
                pkt_queue.put(int(0))

    return
    #print("esco")

def thread_decode(samples, sf, BW, fs, pkt_queue, decode_opts=None):
    print(bcolors.OKBLUE + "[DECODER PROCESS] Decoding samples from sf", sf, bcolors.ENDC)

    # The parent reads exactly one message per spawned decoder, so anything
    # that escapes here leaves it blocked on a queue that will never fill.
    # decode() raises on a sample rate it cannot use (fs not an integer
    # multiple of BW) -- a configuration mistake, not a transient -- and any
    # other exception is a bug worth seeing rather than a silent stall.
    delivered = 0
    try:
        ans = lora.decode(samples, sf, BW, fs,
                          **(decode_opts if decode_opts is not None
                             else DECODE_DEFAULTS))
        if ans.size > 0:
            print("# DECODED PACKETS", ans.size)
            for pkt in ans:
                pkt_queue.put(pkt)
                delivered += 1
    except Exception as exc:
        traceback.print_exc()
        print(bcolors.FAIL + "[DECODER PROCESS] sf " + str(sf)
              + " failed: " + str(exc) + bcolors.ENDC)
    finally:
        if delivered == 0:
            pkt_queue.put(int(0))

        # for pkt in ans:
        #     print(pkt)



def _releases_the_transmitter(fn):
    """Clear `sending` however the worker ends, including on an exception.

    Each TX worker holds sending.value = True while it runs. Without this,
    one bad packet -- a 300-byte payload, or an SF outside 5..12 in a
    multi-SF list -- would kill the worker with the flag still set, and from
    then on every tx_send_burst would return None with "The radio is already
    transmitting!" and tx_start would hand back a dead queue: the
    transmitter wedged for the life of the object.

    A decorator rather than a try/finally around each body, so the three
    long worker functions need no restructuring.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Bound against the signature, not read out of kwargs, so a
        # positional call releases the transmitter too.
        try:
            bound = inspect.signature(fn).bind(*args, **kwargs)
            sending = bound.arguments.get("sending")
        except TypeError:
            sending = kwargs.get("sending")
        try:
            return fn(*args, **kwargs)
        except Exception:
            # Named, so the failure does not vanish into a dead thread.
            print("Transmitter: worker failed, releasing the transmitter")
            raise
        finally:
            if sending is not None:
                sending.value = False
    return wrapper


def _tx_sleep(seconds, sending):
    """Wait between packets, but notice a stop.

    A flat time.sleep(sleep_time) would keep a worker alive and holding
    the transmitter for the whole gap after tx_stop() had cleared its run
    flag -- and duty-cycle-legal gaps are long: EU868 at SF12 wants
    roughly 100x airtime between frames. The claim refuses while a
    worker is alive, so a sleeping worker would also block every restart
    for that long. Polling at 50 ms costs nothing next to a packet.
    """
    # Monotonic: a wall-clock step (WSL2 jumps a few seconds now and then,
    # and an NTP step can go backwards) would stretch or cut the gap
    # between packets.
    end = time.monotonic() + float(seconds)
    while time.monotonic() < end:
        if not sending.value:
            return
        # max(0.0, ...) is load-bearing. `sending.value` is a Synchronized
        # read that takes a lock, so the clock moves between the loop guard
        # above and the remainder below, and on the last poll of a wait that
        # remainder can already be negative. Without the clamp time.sleep()
        # raises ValueError, @_releases_the_transmitter re-raises it, and the
        # worker dies mid-burst with the rest of the burst silently unsent.
        time.sleep(max(0.0, min(0.05, end - time.monotonic())))


@_releases_the_transmitter
def tx_burst(sample_rate, center_freq, pkt_list, sleep_time, sending, streamer, amplitude, verbose, encode_opts=None):
    _require_uhd("tx_burst()")

    #usrp = uhd.usrp.MultiUSRP("address=" + address)

    #print("TX Acquiring")
    # end_of_burst is reset per packet, below. The metadata object is built
    # once outside the loop and `end_of_burst = True` is set after each
    # packet; left latched, every chunk of the next packet would carry it,
    # which on hardware terminates the burst at every get_max_num_samps()
    # boundary and underflows between them. All three transmit paths reset
    # it.
    metadata = uhd.types.TXMetadata()
    for index,pkt in enumerate(pkt_list):
        # Stoppable. sending.value is the claim and the run flag, so the
        # loop reads it rather than setting it, and tx_stop() can end a
        # burst. It is already True here: _tx_claim() set it before the
        # thread was created.
        if not sending.value:
            print("Transmitter: stopped, %d packet(s) not sent"
                  % (len(pkt_list) - index))
            break
        print("Transmitter: Processing Packet #", index)
        metadata.end_of_burst = False
        buffer_samps = streamer.get_max_num_samps()

        samples = amplitude * encode_packet(center_freq, pkt.SF, pkt,
                                            sample_rate, encode_opts)
        print("Encoded Packet #", index)
        proto_len = samples.shape[-1]
        send_samps = 0
        samples = samples.reshape(1, samples.size)
        while send_samps < proto_len:
            # `samples` is (1, N): slice the sample axis, not axis 0 (size 1).
            chunk = samples[:, send_samps:send_samps + buffer_samps]
            n_samples = streamer.send(chunk, metadata)
            if n_samples <= 0:
                # The break leaves the loop, and the lines below then set
                # end_of_burst, send the end-of-burst sample and go on to
                # the next packet. The burst is ended cleanly around a
                # packet that is missing its tail, so a receiver sees a
                # truncated frame rather than nothing -- the case an
                # operator most needs told apart from a clean send.
                print(bcolors.WARNING
                      + "TX: streamer accepted 0 samples with %d of %d "
                        "still to send. This packet goes out TRUNCATED; "
                        "the burst is ended normally and the transmitter "
                        "continues with the next packet."
                        % (proto_len - send_samps, proto_len)
                      + bcolors.ENDC)
                break
            send_samps += n_samples
            if (verbose):
                print("Sent samples", n_samples)
        metadata.end_of_burst = True
        print("Ending Burst for Packet #", index)

        streamer.send(np.zeros((1, 1), dtype=np.complex64), metadata)

        print("Sent Packet #", index)

        # Send EOB to terminate Tx

        #print("Sent packet with seq number", pkt.seqn)
        _tx_sleep(sleep_time, sending)

    sending.value = False


@_releases_the_transmitter
def tx_burst_multi_sf(sample_rate, center_freq, pkt_list, sleep_time, sending, streamer, sf_list, amplitude, encode_opts=None):
    _require_uhd("tx_burst_multi_sf()")

    #usrp = uhd.usrp.MultiUSRP("address=" + address)


    n_pack = len(pkt_list)
    print("N_pack:", n_pack)
    #remaining_pkt = np.array([len(pkt_list)] * len(sf_list))
    pkt_count = np.zeros((len(sf_list)-1), dtype=np.uint32)
    # Sorted high SF first, so the first packet is the longest and the others
    # fit inside its airtime. sort() in place would also reorder the caller's
    # own list as a side effect of transmitting.
    sf_list = sorted(sf_list, reverse=True)
    # Within UHD's fc32 range, at one gain for the whole run. N superimposed
    # SFs at the default amplitude 1 peak at N -- 2.0 for SF8+SF7, a third
    # of the samples past +-1 -- which UHD clips, or wraps where its
    # converter casts without clamping. Each SF's envelope is at most
    # `amplitude`, so a burst peaks at most at amplitude * N: one factor
    # bringing that to max(1, amplitude) -- what a single SF at this
    # amplitude sends, as tx() and tx_burst() do -- keeps every burst in
    # range and the SFs' relative power, and the level constant across the
    # run. Per-burst 1/peak scaling would send the bursts the secondaries
    # have run out of 6 dB hotter than the collided ones, the reference a
    # collision run compares against.
    bound = float(amplitude) * len(sf_list)
    gain = min(1.0, max(1.0, float(amplitude)) / bound) if bound > 0 else 1.0
    if gain < 1.0:
        print("Transmitter: %d superimposed SFs at amplitude %g would peak "
              "at %g, past UHD's +-1.0 range; every burst is scaled by %.3f "
              "(pass an amplitude of at most 1/%d to avoid it)"
              % (len(sf_list), amplitude, bound, gain, len(sf_list)))
    #print("TX Acquiring")
    metadata = uhd.types.TXMetadata()
    for index,pkt in enumerate(pkt_list):
        if not sending.value:                  # see tx_burst
            print("Transmitter: stopped, %d packet(s) not sent"
                  % (len(pkt_list) - index))
            break
        metadata.end_of_burst = False
        #print("TX Acquired")

        buffer_samps = streamer.get_max_num_samps()
        #print("LOL")



        samples = amplitude * encode_packet(center_freq, sf_list[0], pkt,
                                            sample_rate, encode_opts)
        max_len = samples.size

        for i,sf in enumerate(sf_list[1:]):
            add_samples = np.zeros((samples.size), dtype=np.complex64)
            length = 0
            incr = 0
            coded_pkt = 0
            while (pkt_count[i] < n_pack):
                current_pkt = pkt_list[pkt_count[i]]
                # Scaled like the first SF's, so the relative power between
                # the superimposed signals -- the whole point of a multi-SF
                # burst -- holds at any amplitude.
                current_samples = amplitude * encode_packet(
                    center_freq, sf, current_pkt, sample_rate, encode_opts)


                if length + current_samples.size > add_samples.size:
                    # The break below predicts the next packet's size from the
                    # average so far, which is only exact when every packet is
                    # the same length. A burst of mixed sizes can overshoot,
                    # and the slice assignment would raise ValueError from
                    # inside a transmit thread. Stop cleanly instead.
                    break
                add_samples[length:length + current_samples.size] = current_samples
                length = length + current_samples.size
                coded_pkt = coded_pkt + 1
                pkt_count[i] = pkt_count[i] + 1
                if (length * ((coded_pkt + 1) / coded_pkt)) > max_len:
                    print("Pkt Count for SF", sf,":", pkt_count[i])
                    break




            samples = samples + add_samples

        # Within UHD's fc32 range: see `gain` above the loop.
        if gain != 1.0:
            samples = (samples * gain).astype(np.complex64)

        proto_len = samples.shape[-1]
        send_samps = 0
        samples = samples.reshape(1, samples.size)
        while send_samps < proto_len:
            # `samples` is (1, N): slice the sample axis, not axis 0 (size 1).
            chunk = samples[:, send_samps:send_samps + buffer_samps]
            n_samples = streamer.send(chunk, metadata)
            if n_samples <= 0:
                # The break leaves the loop, and the lines below then set
                # end_of_burst, send the end-of-burst sample and go on to
                # the next packet. The burst is ended cleanly around a
                # packet that is missing its tail, so a receiver sees a
                # truncated frame rather than nothing -- the case an
                # operator most needs told apart from a clean send.
                print(bcolors.WARNING
                      + "TX: streamer accepted 0 samples with %d of %d "
                        "still to send. This packet goes out TRUNCATED; "
                        "the burst is ended normally and the transmitter "
                        "continues with the next packet."
                        % (proto_len - send_samps, proto_len)
                      + bcolors.ENDC)
                break
            send_samps += n_samples

        # End-of-burst belongs after the whole packet is on the wire; sent
        # inside the loop it would truncate the burst after the first chunk.
        metadata.end_of_burst = True
        streamer.send(np.zeros((1, 1), dtype=np.complex64), metadata)


        # Send EOB to terminate Tx

        #print("Sent packet with seq number", pkt.seqn)
        _tx_sleep(sleep_time, sending)

    sending.value = False






@_releases_the_transmitter
def tx(sample_rate, center_freq, pkt_queue, sleep_time, sending, streamer, amplitude, verbose = False, encode_opts=None):
    _require_uhd("tx()")

    #usrp = uhd.usrp.MultiUSRP("address=" + address)

    # sending.value is not re-asserted here. _tx_claim() set it before this
    # thread was created, and setting it again would undo a tx_stop() that
    # lands during startup, putting packets on the air that tx_stop() had
    # dropped. rx() follows the same rule for `receiving`.
    metadata = uhd.types.TXMetadata()
    while sending.value:
        metadata.end_of_burst = False

        pkt = pkt_queue.get()
        # The stop sentinel: tx_stop() puts None to wake a worker parked in
        # get(), and the worker leaves cleanly rather than failing on it.
        if pkt is None:
            break
        if(verbose):
            print("Sending pkt", pkt)

        buffer_samps = streamer.get_max_num_samps()

        samples = amplitude * encode_packet(center_freq, pkt.SF, pkt,
                                            sample_rate, encode_opts)
        proto_len = samples.shape[-1]
        send_samps = 0
        samples = samples.reshape(1, samples.size)
        while send_samps < proto_len:
            # `samples` is (1, N): slice the sample axis, not axis 0 (size 1).
            chunk = samples[:, send_samps:send_samps + buffer_samps]
            n_samples = streamer.send(chunk, metadata)
            if n_samples <= 0:
                # The break leaves the loop, and the lines below then set
                # end_of_burst, send the end-of-burst sample and go on to
                # the next packet. The burst is ended cleanly around a
                # packet that is missing its tail, so a receiver sees a
                # truncated frame rather than nothing -- the case an
                # operator most needs told apart from a clean send.
                print(bcolors.WARNING
                      + "TX: streamer accepted 0 samples with %d of %d "
                        "still to send. This packet goes out TRUNCATED; "
                        "the burst is ended normally and the transmitter "
                        "continues with the next packet."
                        % (proto_len - send_samps, proto_len)
                      + bcolors.ENDC)
                break
            send_samps += n_samples

            if(verbose):
                print("Sent samples", n_samples)
                print("Total Sent Samples", send_samps)

        metadata.end_of_burst = True
        streamer.send(np.zeros((1, 1), dtype=np.complex64), metadata)


        # Send EOB to terminate Tx

        #print("Sent packet with seq number", pkt.seqn)
        _tx_sleep(sleep_time, sending)

    sending.value = False













def rx(sample_rate, sf_list, bandwidth, receiving, packet_queue, complex_data_number, streamer, pause_rec,buf_size = 120, decode_opts=None, decode_opts_per_sf=None, max_payload_len=251, rx_started=None, rx_failure=None):
    _require_uhd("rx()")






    #LORA RECEIVER MAIN THREAD#


    #THE RECEIVER SCRIPT IS STRUCTURED AS A TWO-PROCESS PROGRAM: ONE PROCESS (THE MAIN THREAD OR RECEIVER THREAD) IS RESPONSIBLE FOR READING AND
    #BUFFERING OF RF DATA FROM THE USRP RADIO; THE OTHER PROCESS (THE DECODER THREAD) READS AND PROCESSES DATA FROM THE BUFFER
    #THE PROGRAM RESORTS TO A CIRCULAR BUFFER, AND MAKES USE OF A QUEUE TO EXCHANGDE DATA BETWEEN THE PROCESSES
    #MORE IN DETAIL, THE BUFFER IS LOCATED IN A SHARED MEMORY AREA. EACH TIME A NEW CHUNK OF DATA IS RECEIVED, THE RECEIVER PUTS THE DATA START INDEX IN THE QUEUE.
    #THE PROCESSING THREAD CAN ACCORDINGLY READ AND, POSSIBLY, DECODE THE LORA DATA IN THE CHUNK.


    #MAXIMUM NUMBER OF DATA CHUNKS IN THE SHARED MEMORY BUFFER
    BUF_SIZE = int(buf_size)
    # Any buf_size works. SF_sample_th is [1]*8 below, so sf_windows_len
    # equals complex_data_number and decoder_process's ring slice is one
    # chunk wide -- it cannot straddle the buffer end. A per-SF window
    # count of more than one chunk would need buf_size to be a multiple of
    # it, and a check: the slice would silently shorten rather than wrap.

    #BYTES PER COMPLEX SAMPLES
    data_size = 8  # bytes

    #NUMBER OF COMPLEX SAMPLES IN A DATA CHUNK
    #complex_data_number = 500000



    #SLIDING WINDOW CURSOR FOR THE RECEIVER BUFFER
    cursor = 0




    #SIZE, IN BYTES, OF A DATA CHUNK
    block_size = complex_data_number * data_size
    #samples_buffer = np.zeros(, dtype=np.complex64)

    #PROCESS UTILITIES
    #complex_data_number
    #SF_sample_th = np.array([3e6,6e6,12e6,18e6,42e6,72e6])


    #NUMBER OF RECEIVING WINDOWS PER SF. 
    SF_sample_th = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    sf_arr = np.array([5, 6, 7, 8, 9, 10, 11, 12])

    # np.isin below would drop an out-of-range SF without a word -- rx_start()
    # would come up with no decoder process for it and report nothing at all.
    # Say so instead.
    unsupported = sorted(set(int(sf) for sf in sf_list) - set(int(x) for x in sf_arr))
    if unsupported:
        raise ValueError(
            "this receiver handles SF%d..SF%d; %s not supported."
            % (int(sf_arr[0]), int(sf_arr[-1]),
               ", ".join("SF%d" % sf for sf in unsupported)))

    # SF5/SF6 decode correctly but not in real time here, because at 256
    # samples per symbol the interpreter overhead per scan step dominates.
    # That is a throughput limit, not a correctness one -- windows the decoder
    # cannot keep up with are dropped with a warning by MAX_DECODER_BACKLOG
    # rather than queued -- so say it plainly and let the operator decide. The
    # GNU Radio block in gr-sdrlora/ keeps up there at its default window
    # (128 symbols at SF5, about 30-45% of real time).
    #
    # The figure is quoted at the granularity the live receiver uses:
    # DECODE_DEFAULTS["sync_phases"] is None, so a live decoder calls
    # lora.default_sync_phases(), which is four phases at SF5 -- the default
    # because two are not reliable below SF7. At four phases, through
    # StreamDecoder at rx_example.py's 3,000,000-sample window on an idle
    # channel and an idle machine, SF5 runs at about 150% of real time and
    # SF6 at 80-100%, and more under load, which is why the message gives a
    # range.
    slow = sorted(sf for sf in set(int(x) for x in sf_list) if sf < 7)
    if slow:
        print(bcolors.WARNING
              + "[RX] %s decode correctly but run OVER the real-time budget "
                "in Python at the %s-phase scan granularity they use by "
                "default: about 150%% of real time at SF5 and 80-100%% at "
                "SF6 in steady state, more under load. Expect dropped "
                "windows. The GNU Radio block in gr-sdrlora/ keeps up there "
                "at its default window."
                % (", ".join("SF%d" % sf for sf in slow),
                   "/".join(str(lora.default_sync_phases(sf))
                            for sf in slow))
              + bcolors.ENDC)

    indexes = np.isin(sf_arr, sf_list)

    SF_sample_th = SF_sample_th[indexes]
    sf_arr = sf_arr[indexes]

    # Resolve once, per SF, before anything is spawned: a bad option name or a
    # missing implicit-header length must not surface inside a child process
    # where the only symptom is a decoder that never reports.
    opts_arr = [resolve_decode_opts(int(sf), decode_opts, decode_opts_per_sf,
                                    BW=bandwidth, fs=sample_rate)
                for sf in sf_arr]


    queue_arr = np.empty(shape=(sf_arr.size,), dtype=mp_queues.Queue)
    processes_arr = np.empty(shape=(sf_arr.size,), dtype=mp.Process)





    # Defined before the resources it releases, and covering the setup as
    # well as the loop: anything that raises while the shared segment, the
    # decoder children, the trigger child or the stream are coming up must
    # still leave receiving.value False, no child alive and the segment
    # closed and unlinked. Otherwise the leak compounds, because rx_start's
    # claim is `_prev.is_alive()` and a dead thread passes it.
    #
    # `_res` is what makes the early definition possible: the closure only
    # names a dict that already exists, and each resource registers itself
    # as it comes up, so a partial setup releases exactly what it built.
    _res = {"shm": None, "procs": [], "trigger": None, "streaming": False}
    _released = [False]

    def _release():
        if _released[0]:
            return
        _released[0] = True
        print("Stopping RX")
        receiving.value = False
        steps = []
        if _res["streaming"]:
            steps.append(("stream stop", lambda: streamer.issue_stream_cmd(
                uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))))
        if _res["procs"]:
            def _stop_decoders():
                # One try per child, so a terminate() that raises does not
                # abandon the children after it. The bounded join handles
                # a child that hangs; this is the other way a child can
                # refuse to go.
                for _p in _res["procs"]:
                    try:
                        _p.terminate()
                        _p.join(CHILD_JOIN_TIMEOUT)
                    except Exception as exc:       # noqa: BLE001
                        print(bcolors.WARNING
                              + "RX cleanup: decoder %s did not stop: %r"
                              % (getattr(_p, "name", "?"), exc)
                              + bcolors.ENDC)
            steps.append(("decoders", _stop_decoders))
        if _res["trigger"] is not None:
            steps.append(("trigger", lambda: (
                _res["trigger"].terminate(),
                _res["trigger"].join(CHILD_JOIN_TIMEOUT))))
        if _res["shm"] is not None:
            steps.append(("shm close", _res["shm"].close))
            steps.append(("shm unlink", _res["shm"].unlink))
        # Each join is bounded and the steps run in order, so one child
        # that ignores terminate() cannot block shm.unlink() behind it --
        # the segment is the resource whose leak costs the most.
        for what, fn in steps:
            try:
                fn()
            except Exception as exc:               # noqa: BLE001
                print(bcolors.WARNING + "RX cleanup: %s failed: %r"
                      % (what, exc) + bcolors.ENDC)

    # Wrapped, so a failure ANYWHERE in setup leaves through the same
    # cleanup the loop does. Re-raised, because a receiver that cannot
    # start is the caller's problem -- but it must not also be a leak.
    try:
        #CREATION OF THE SHARED MEMORY AREA
        shm = mp.shared_memory.SharedMemory(create = True, size = block_size * BUF_SIZE)
        _res["shm"] = shm
        samples_buffer = np.ndarray(complex_data_number * BUF_SIZE, dtype=np.complex64, buffer=shm.buf)
        buffer_size = samples_buffer.size
        index_queue = mp.Queue(0)


        # The start method actually in force (get_start_method() takes
        # `allow_none`, not a method name). The decoder processes attach to
        # the ring buffer by name, not by inheritance; this is printed for
        # diagnosis.
        print("########################")
        print("multiprocessing start method:", mp.get_start_method())
        print("########################")
        #CREATION OF THE DECODER PROCESSES

        for index, sf in enumerate(sf_arr):
            queue_arr[index] = mp.Queue(0)
            processes_arr[index] = mp.Process(name="Decoder SF" + str(sf), target=decoder_process, args=(shm.name, complex_data_number, BUF_SIZE,
                                                                            queue_arr[index], packet_queue[index], sf, bandwidth,
                                                                            sample_rate, SF_sample_th[index], opts_arr[index], max_payload_len))


            # Daemon, like the trigger process below. A decoder that
            # outlived rx() as a non-daemon child would loop forever and
            # block interpreter exit. terminate() in the cleanup is the
            # intended route; this covers the case where that route is
            # missed.
            processes_arr[index].daemon = True
            processes_arr[index].start()
            _res["procs"].append(processes_arr[index])
            # Bounded, and watching the run flag. is_alive() is False both
            # before start() takes effect and after the child has exited,
            # so an unbounded wait on a child that dies on its way up
            # would never reach the loop or release anything, and
            # _rx_claim() would refuse every later rx_start(). Reachable:
            # decoder_process builds a StreamDecoder as its second
            # statement, and rx() is a public entry point that does not
            # validate max_payload_len -- only rx_start does.
            _t0 = time.time()
            while not processes_arr[index].is_alive():
                if not receiving.value:
                    raise RuntimeError(
                        "rx: stopped while waiting for the SF%d decoder"
                        % sf)
                if time.time() - _t0 > CHILD_START_TIMEOUT:
                    raise RuntimeError(
                        # max_payload_len only. rx() resolves and
                        # validates every SF's decode_opts in the
                        # parent, through resolve_decode_opts, before
                        # the first mp.Process exists, so a bad option
                        # raises there and never reaches this timeout.
                        "rx: the SF%d decoder did not come up within "
                        "%.0f s. decode_opts are already validated in "
                        "the parent, so the likely causes are "
                        "max_payload_len (not validated here) or the "
                        "host being unable to start another process."
                        % (sf, CHILD_START_TIMEOUT))
                time.sleep(0.05)

        # The respawn pass, also bounded. Every replacement is registered
        # in _res["procs"] as well as rebound in processes_arr, so
        # _release() reaps the live replacement and not only the dead
        # original.
        for index, sf in enumerate(sf_arr):
            _tries = 0
            while not processes_arr[index].is_alive():
                if not receiving.value:
                    raise RuntimeError(
                        "rx: stopped while restarting the SF%d decoder"
                        % sf)
                _tries += 1
                if _tries > CHILD_RESTART_MAX:
                    raise RuntimeError(
                        "rx: the SF%d decoder died %d times in a row. "
                        "Respawning forever hides the cause; the usual "
                        "one is a decode option the child rejects."
                        % (sf, CHILD_RESTART_MAX))
                print(bcolors.WARNING + "rx: restarting the SF%d "
                      "decoder (attempt %d of %d)"
                      % (sf, _tries, CHILD_RESTART_MAX) + bcolors.ENDC)
                processes_arr[index] = mp.Process(name="Decoder SF" + str(sf), target=decoder_process,
                                                  args=(shm.name, complex_data_number, BUF_SIZE,
                                                        queue_arr[index], packet_queue[index], sf, bandwidth,
                                                        sample_rate, SF_sample_th[index], opts_arr[index], max_payload_len))

                processes_arr[index].daemon = True
                processes_arr[index].start()
                _res["procs"].append(processes_arr[index])
                time.sleep(1)


        #CREATION OF THE THRESHOLD TRIGGER PROCESS


        trigger_process = mp.Process(target=threshold_trigger_process, args=(index_queue, SF_sample_th, tuple(queue_arr)))
        trigger_process.daemon = True
        trigger_process.start()
        _res["trigger"] = trigger_process


        # Off by default: see RX_DEBUG_METADATA.
        debug = RX_DEBUG_METADATA


        # # Start Stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        streamer.issue_stream_cmd(stream_cmd)
        _res["streaming"] = True
    except BaseException as exc:
        # Recorded, then signalled, before the cleanup. rx() runs in a
        # worker thread, so this raise goes nowhere a caller can see it
        # beyond a threading.excepthook traceback. Handing the exception
        # back through rx_failure lets rx_start() refuse instead. Setting
        # rx_started here too means rx_start() does not sit out its whole
        # grace period on a failure it already knows.
        if rx_failure is not None:
            rx_failure.append(exc)
        if rx_started is not None:
            rx_started.set()
        _release()
        raise
    # receiving.value is not set here. rx_start() sets it True before
    # starting this thread, and assigning True again would overwrite an
    # rx_stop() that lands during startup -- and startup is long: one
    # mp.Process start plus an is_alive() poll per spreading factor,
    # seconds for a 6-SF listen.
    if not receiving.value:
        print("Receiver: stopped during startup")

    metadata = uhd.types.RXMetadata()
    print("Starting...")

    #("Channels",streamer.get_num_channels())
    buf_length = complex_data_number
    # (num_channels, num_samps), not 1-D. UHD's Python recv() reads
    # shape(0) for the channel count and shape(1) for the sample count,
    # so a 1-D array raises on the first recv(); the send() paths reshape
    # to (1, N) for the same reason. The view keeps the rest of the loop
    # working on one dimension.
    recv_buffer = np.zeros((1, buf_length), dtype=np.complex64)
    recv_view = recv_buffer[0]
    # Setup is done and everything is up. rx_start() waits on this
    # before returning, because rx() runs in a worker thread, where a
    # setup failure raises with nobody looking. A large window with the
    # automatic ring floor can ask SharedMemory for gigabytes, so such
    # a failure is reachable on ordinary settings.
    if rx_started is not None:
        rx_started.set()
    #START RECEIVING
    # See CHILD_LIVENESS_PERIOD. Reported once per child: repeating every
    # period would bury the heartbeat that is the ongoing signal.
    #
    # The trigger is watched too, and first. It is the single point of
    # failure for every spreading factor at once: it is the only thing
    # that turns index_queue entries into per-SF window notifications, so
    # if it dies every decoder blocks in its queue forever while this loop
    # goes on filling the ring.
    _next_liveness = time.time() + CHILD_LIVENESS_PERIOD
    _reported_dead = set()
    while receiving.value:
        try:

            pause_rec.wait()
            if time.time() >= _next_liveness:
                _next_liveness = time.time() + CHILD_LIVENESS_PERIOD
                if "trigger" not in _reported_dead \
                        and not trigger_process.is_alive():
                    _reported_dead.add("trigger")
                    print(bcolors.FAIL
                          + "RX: the threshold trigger has died. EVERY "
                            "spreading factor is off the air: the decoders "
                            "are waiting on window notifications that can "
                            "no longer be sent, so this receiver will not "
                            "report another packet however long it runs. "
                            "rx_stop() then rx_start()." + bcolors.ENDC)
                    # And tell the readers, not just stdout. Every SF is
                    # affected, and each one has a consumer that may be
                    # parked in get() with nothing left to wake it.
                    for _q in packet_queue:
                        try:
                            _q.put(DECODER_STOPPED)
                        except Exception:
                            pass
                for _i, _sf in enumerate(sf_arr):
                    if _i in _reported_dead:
                        continue
                    if not processes_arr[_i].is_alive():
                        _reported_dead.add(_i)
                        print(bcolors.FAIL
                              + "RX: the SF%d decoder has died. That "
                                "spreading factor is off the air and "
                                "packets on it will simply stop; the "
                                "receiver keeps running for the others. "
                                "rx_stop() then rx_start() brings it back."
                                % int(_sf) + bcolors.ENDC)
                        # Once, on that SF's queue only: the others are
                        # still live and must not be told to stop.
                        try:
                            packet_queue[_i].put(DECODER_STOPPED)
                        except Exception:
                            pass
            # recv() returns how many samples it actually produced. The rest
            # of recv_buffer still holds the previous read's samples, which
            # the decoder must not see as real signal.
            n_rx = streamer.recv(recv_buffer, metadata)
            if(debug):
                print(bcolors.OKGREEN + "DEBUG metadata ", metadata, bcolors.ENDC)

            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                # ERROR_CODE_OVERFLOW here means the host could not keep up and
                # samples were dropped: the stream is no longer contiguous.
                print(bcolors.FAIL + "RX error: " + str(metadata.error_code)
                      + bcolors.ENDC)

            if n_rx <= 0:
                continue
            if n_rx < buf_length:
                # Short read: zero-fill the remainder so no stale samples are
                # decoded, and keep the ring buffer's block alignment intact.
                recv_view[n_rx:] = 0

            samples_buffer[cursor:cursor + buf_length] = recv_view
            cursor = (cursor + buf_length) % buffer_size
            # Unconditional: `cursor` starts at 0 and advances by exactly
            # buf_length, which is complex_data_number, so every read
            # completes one chunk. The stream runs continuously; it is
            # not restarted each time round the ring.
            index_queue.put(cursor)




        except KeyboardInterrupt:
            # Unreachable from a worker thread -- CPython delivers
            # KeyboardInterrupt to the main thread only -- but kept for the
            # case where rx() is called directly on the main thread.
            print("Stopping RX")
            receiving.value = False
            break
        except Exception as exc:                   # noqa: BLE001
            # Leave through the cleanup, not past it. A streamer that goes
            # away mid-stream raises here, and an exception allowed to
            # propagate out of a thread would strand the device, the
            # children and the shared segment -- with receiving.value still
            # True, so every caller would go on believing the receiver is
            # running. Named and traced rather than swallowed.
            print(bcolors.FAIL + "RX: the receive loop failed: %r"
                  % (exc,) + bcolors.ENDC)
            traceback.print_exc()
            receiving.value = False
            break
        except BaseException:                      # noqa: BLE001
            # SystemExit, GeneratorExit and anything else outside
            # Exception, which would otherwise walk straight past
            # _release() -- the setup block above catches BaseException
            # for the same reason. Re-raised, because this is not ours to
            # swallow; _release() is idempotent, so running it here and
            # again below is safe.
            receiving.value = False
            _release()
            raise

    _release()















class _RXClaimed(object):
    """Stands in for the receive thread between the claim and the
    thread's creation, so a concurrent rx_start() sees it taken."""

    def is_alive(self):
        return True

    def join(self, timeout=None):
        return None


_RX_CLAIMED = _RXClaimed()


class lora_transceiver():

    @property
    def signal_amplitude(self):
        return self._signal_amplitude

    @signal_amplitude.setter
    def signal_amplitude(self, value):
        # The transceiver's own output level, multiplied into every frame
        # the workers send. Validated by the same rule as
        # encode_opts["amplitude"], on every assignment and not only in
        # __init__: NaN would put a frame of NaNs on the streamer (the zero
        # lead-in too, nan * 0 being nan), 0.0 a silent frame, and a
        # negative value a frame rotated 180 degrees.
        try:
            v = lora._check_amplitude(value)
        except ValueError as e:
            raise ValueError("signal_amplitude: %s" % e) from None
        # Past UHD's fc32 range: warned about, not refused. tx() and
        # tx_burst() send amplitude * frame as given, and UHD clips past
        # +-1.0 -- or wraps where its converter casts without clamping. A
        # warning keeps a deliberate setting working.
        if v > 1.0:
            print(bcolors.WARNING
                  + "signal_amplitude %g is past UHD's +-1.0 sample range: "
                    "frames will clip on the radio (or wrap, with a "
                    "converter that does not clamp). Use tx_gain for "
                    "level, and an amplitude of at most 1." % v
                  + bcolors.ENDC)
        self._signal_amplitude = v

    def __init__(self,address,rx_gain,tx_gain,bandwidth, rx_freq, tx_freq, sample_rate, rx_channel_ID, tx_channel_ID, signal_amplitude = 1):
        self.address = address
        self.rx_gain = rx_gain # dB
        self.tx_gain = tx_gain # dB
        self.bandwidth = bandwidth  # Hz
        self.rx_freq = rx_freq # Hz
        self.tx_freq = tx_freq  # Hz
        self.sample_rate = sample_rate # Hz
        self.receiving = mp.Value("i",False)
        self.sending = mp.Value("i",False)
        # Guards the test-and-set in _tx_claim(). mp.Value has its own
        # lock for a single access, which is not the same thing as making
        # "check then start a worker" atomic.
        self._tx_lock = threading.Lock()
        # The receive side needs one too: its claim is a check-then-act
        # across several statements.
        self._rx_lock = threading.Lock()
        self._tx_threads = []
        # Validated by the property above, before any hardware is touched.
        self.signal_amplitude = signal_amplitude
        # Without this the line below raises AttributeError on NoneType,
        # which tells the caller nothing about what to install.
        _require_uhd("the lora_transceiver class")
        self.usrp = uhd.usrp.MultiUSRP("address=" + self.address)
        self.rx_channel = rx_channel_ID
        self.tx_channel = tx_channel_ID
        self.usrp.set_rx_rate(sample_rate, self.rx_channel)
        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(rx_freq), self.rx_channel)
        self.usrp.set_rx_gain(rx_gain, self.rx_channel)
        self.usrp.set_rx_bandwidth(bandwidth, self.rx_channel)
        self.report_rx_config(requested_bandwidth=bandwidth)
        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [self.rx_channel]
        self.rx_streamer = self.usrp.get_rx_stream(st_args)
        self.rx_pause_flag = mp.Event()
        self.rx_pause_flag.set()
        #

        self.usrp.set_tx_rate(sample_rate, self.tx_channel)
        self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(tx_freq), self.tx_channel)
        self.usrp.set_tx_gain(tx_gain, self.tx_channel)
        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [self.tx_channel]
        self.tx_streamer = self.usrp.get_tx_stream(st_args)
        self.rx_proc = None

        self.tx_queue = None
        self.rx_queues = None

        #print("Main class", self.rx_streamer)

    def report_rx_config(self, requested_bandwidth=None):
        """Print what UHD coerced the RX settings to, and return them.

        Every set_rx_*() is a request, not a command: UHD rounds it to what the
        hardware supports. The analog bandwidth matters most here -- the
        demodulator decimates by fs/BW, which folds every Nyquist zone of noise
        onto the signal, so any analog bandwidth wider than the LoRa channel
        costs sensitivity. If the coerced value is much larger than the channel,
        decode()'s channel_filter_rx is doing that work in software instead.
        """
        cfg = {}
        for name, getter in (("rate", "get_rx_rate"),
                             ("freq", "get_rx_freq"),
                             ("gain", "get_rx_gain"),
                             ("bandwidth", "get_rx_bandwidth")):
            try:
                cfg[name] = getattr(self.usrp, getter)(self.rx_channel)
            except Exception as exc:          # not all boards expose all four
                cfg[name] = None
                print(bcolors.WARNING + "RX config: " + getter
                      + " unavailable (" + str(exc) + ")" + bcolors.ENDC)

        print("RX actual configuration (UHD-coerced):")
        if cfg["rate"] is not None:
            print("   sample rate : %.6f MS/s" % (cfg["rate"] / 1e6))
        if cfg["freq"] is not None:
            print("   center freq : %.6f MHz" % (cfg["freq"] / 1e6))
        if cfg["gain"] is not None:
            print("   gain        : %.2f dB" % cfg["gain"])

        bw = cfg["bandwidth"]
        if bw is not None:
            line = "   analog BW   : %.3f kHz" % (bw / 1e3)
            if requested_bandwidth:
                line += "  (requested %.3f kHz)" % (requested_bandwidth / 1e3)
            print(line)
            if requested_bandwidth and bw > 1.5 * requested_bandwidth:
                excess = bw / float(requested_bandwidth)
                print(bcolors.WARNING
                      + "   NOTE: analog BW is %.1fx the LoRa channel, so the front"
                        % excess
                      + " end is not filtering it. Out-of-band noise reaching the"
                      + " ADC gets folded by the demodulator's decimation --"
                      + " keep decode()'s channel_filter_rx enabled."
                      + bcolors.ENDC)
        # The demodulator decimates by exactly fs/BW and steps the preamble
        # search in whole symbols. UHD coerces the requested rate to what the
        # board's clock can synthesise, so check the coerced one -- a rate the
        # decoder cannot use is worth one loud line at startup rather than a
        # receiver that runs forever and never reports a packet.
        if cfg["rate"] is not None and requested_bandwidth:
            ratio = cfg["rate"] / float(requested_bandwidth)
            if abs(ratio - round(ratio)) > 1e-6 * max(1.0, ratio):
                print(bcolors.FAIL
                      + "   ERROR: coerced rate / LoRa BW = %.6f is not an"
                        " integer. decode() will reject it. Ask for %.6f MS/s"
                        " instead." % (ratio,
                                       round(ratio) * requested_bandwidth / 1e6)
                      + bcolors.ENDC)
            # Below the tuned ratio, state the cost. There is no detection
            # cliff: from 2x up every frame is found down to about -10 dB.
            # What oversampling buys is sensitivity -- roughly 1.5 dB per
            # doubling, half delivery near -12.5 dB at 2x, -14 dB at 4x and
            # below -16 dB at 8x. lora._MIN_OSF is 2, and ratios 1 and 2
            # interoperate with gr-lora_sdr.
            elif round(ratio) < _RX_TUNED_OSF:
                print(bcolors.WARNING
                      + "   WARNING: coerced rate is %dx the LoRa BW; this"
                        " receiver is tuned for %dx. The cost is sensitivity,"
                        " not detection: at SF7 every ratio from 2x up"
                        " delivers every frame down to -10 dB, and half"
                        " delivery falls near -12.5 dB at 2x, -14 dB at 4x and"
                        " below -16 dB at 8x -- about 1.5 dB per doubling."
                        " The samples are decoded as they are: decode()"
                        " interpolates only below %dx, which this is not."
                      % (round(ratio), _RX_TUNED_OSF, lora._MIN_OSF)
                      + bcolors.ENDC)

        self.rx_config = cfg
        return cfg

    def rx_start(self, sf_list, complex_data_number = 3000000, block_size = 20,
                 decode_opts = None, decode_opts_per_sf = None,
                 max_payload_len = 251):
        """Start one decoder process per spreading factor.

        decode_opts: a fixed set of lora.decode() keywords, applied to every
        SF. The set is DECODE_DEFAULTS; print sorted(DECODE_DEFAULTS) for the
        current list. Anything outside it is refused by name rather than
        ignored, including lora.decode() keywords that make no sense here:
        `with_consumed` is the one a caller reaches for, and StreamDecoder
        already uses it internally to carry the window boundary. Two with
        the most operational consequence: `invert_iq`, which a LoRaWAN
        downlink needs, and `report_crc_errors`, without which corrupted
        frames are dropped rather than counted.

        decode_opts_per_sf: the same, keyed by SF, winning where it overlaps.
        SF5 and SF6 require implicit-header settings here, because LoRa has no
        room for a header below SF7:

            rx_start([5, 7], decode_opts_per_sf={
                5: {'implicit_header': True, 'payload_len': 16, 'cr': 1}})

        Every entry is resolved with decode_opts applied, including one for
        an SF not in sf_list: it is not applied, but a typo or a value its
        own SF refuses is refused rather than ignored. Only the SFs in
        sf_list are also built as decoders below, so the checks that live
        in StreamDecoder's construction (numtaps against the tail, the
        working set) reach those alone.
        """
        # A duplicate SF would make one queue per list entry and one decoder
        # per distinct spreading factor, so the extra queue would never be
        # written and a blocking reader on it would wait forever. Refused.
        # _whole, by name: int() would make 7.9 an SF7 receiver.
        sf_list = [lora._whole(s, "SF") for s in sf_list]
        if len(set(sf_list)) != len(sf_list):
            dup = sorted({s for s in sf_list if sf_list.count(s) > 1})
            raise ValueError(
                "rx_start: sf_list has repeated spreading factors %s -- one "
                "decoder is started per distinct SF, so the extra queues "
                "would never be written and a reader on one would block "
                "forever" % dup)
        if not sf_list:
            raise ValueError("rx_start: sf_list is empty")
        # Here, in the caller's thread, for the same reason the decode
        # options are: checked only inside rx() or the decoder child, a
        # bad SF or max_payload_len would come back as a receiver that
        # reports running and never delivers.
        for sf in sf_list:
            if sf < 5 or sf > 12:
                raise ValueError(
                    "rx_start: sf must be 5..12, got %r -- a decoder "
                    "started on it dies immediately and the receiver "
                    "then reports running while delivering nothing"
                    % sf)
        # Caught, so nan and inf get this message rather than _whole's.
        try:
            _mpl = lora._whole(max_payload_len, "max_payload_len")
        except ValueError:
            _mpl = None
        if _mpl is None or _mpl < 1 or _mpl > 251:
            raise ValueError(
                "rx_start: max_payload_len must be a whole number in "
                "1..251 (the on-air LENGTH field is one byte and the "
                "pseudo-header spends 4 of them), got %r"
                % (max_payload_len,))
        decode_opts = with_rx_carrier(decode_opts,
                                      getattr(self, "rx_freq", None))

        # THE DEFAULT MINIMUM BLOCK OF SAMPLES ROUGHLY CORRESPONDS TO 5 TIMES THE SIZE OF A 251 BYTES SF7 PACKET
        rx_packet_queue = np.empty((len(sf_list)), dtype=mp_queues.Queue)
        #print("SF LIST",sf_list)
        print("RX Start")
        for index in range(len(sf_list)):
            rx_packet_queue[index] = mp.Queue(0)
        # Validate here, in the caller's thread. rx() runs in a worker whose
        # exceptions go nowhere visible, so a rejected option would otherwise
        # look like a receiver that simply never reports a packet.
        # The decoder itself too, which is cheap (it allocates nothing):
        # its working-set limit and its numtaps-vs-tail guard need the SF
        # and max_payload_len as well, and a decoder child that refuses at
        # construction would leave a receiver reporting running and
        # delivering nothing.
        for sf in sf_list:
            _o = resolve_decode_opts(int(sf), decode_opts, decode_opts_per_sf,
                                     BW=self.bandwidth, fs=self.sample_rate)
            StreamDecoder(int(sf), self.bandwidth, self.sample_rate,
                          decode_opts=_o, max_payload_len=_mpl)
        # daemon=True: rx()'s own KeyboardInterrupt handler cannot fire in
        # a worker thread, because CPython delivers KeyboardInterrupt to
        # the main thread only, and a non-daemon receive thread would keep
        # the process from exiting. A daemon thread lets the interpreter
        # go; rx_stop() remains the clean path and is what the examples
        # call.
        # The ring must outrun the decoder's worst lag. MAX_DECODER_BACKLOG
        # is cpu_count-1 and the ring is block_size windows, with no
        # relation between them -- and decoder_process drains 1 + waiting
        # windows without re-checking, so its lag reaches
        # 2*(MAX_DECODER_BACKLOG+1). At the default block_size of 20 that is
        # fine on 8 CPUs (~16) and an overrun on 12 or 16 (~24, ~32): the
        # decoder would read samples the receiver has already overwritten,
        # silently, since the shed warning counts only what it drops.
        #
        # Raised rather than refused, and capped. The cost is
        # complex_data_number * block_size * 8 bytes of shared memory (the
        # ring is block_size windows of complex_data_number samples), and
        # the floor grows as 2*cpu_count, so an uncapped floor on a
        # 64-core host would ask for 3.07 GB at rx_example.py's settings
        # and 12.29 GB at tx_rx_example.py's, with the OSError from
        # SharedMemory(create=True) landing inside rx()'s setup.
        #
        # So the floor is bounded: past RING_WINDOWS_MAX the right answer
        # is to say the host cannot keep up, not to allocate more. Above
        # 16 CPUs the lag and the floor are different numbers -- the lag
        # is 64 on a 32-core host and 128 on a 64-core one, while the
        # floor stops at RING_WINDOWS_MAX = 32 -- so the messages below
        # report each separately.
        _lag = 2 * (MAX_DECODER_BACKLOG + 1)
        _ring_min = min(_lag, RING_WINDOWS_MAX)
        if block_size < _ring_min:
            _bytes = complex_data_number * _ring_min * 8
            print(bcolors.WARNING + "rx_start: block_size %d is smaller "
                  "than the ring floor of %d windows on this host (%d "
                  "CPUs, worst-case decoder lag %d); raising it to %d "
                  "(%.2f GB of shared memory), or the decoder would read "
                  "samples the receiver had already overwritten"
                  % (block_size, _ring_min, os.cpu_count() or 0, _lag,
                     _ring_min, _bytes / 1e9)
                  + bcolors.ENDC)
            block_size = _ring_min
        # And say so when the cap binds, whatever block_size is. Above 16
        # CPUs RING_WINDOWS_MAX holds the ring below the lag the backlog
        # valve permits, so a decoder that falls the full distance behind
        # reads samples the receiver has overwritten, and the shed warning
        # counts only what it drops. The valve is not bounded by the ring
        # instead -- see decoder_process -- because on a small ring that
        # sheds windows a healthy decoder keeps up with.
        if _lag > RING_WINDOWS_MAX and block_size < _lag:
            print(bcolors.WARNING + "rx_start: this host allows %d windows "
                  "of decoder backlog (%d CPUs), so the worst case is %d "
                  "windows behind, but the ring holds %d -- %.0f%% of it. "
                  "A decoder that falls the whole way behind will read "
                  "overwritten samples, and only the shed counter would "
                  "move. Pass block_size=%d (%.2f GB) if the memory is "
                  "there."
                  % (MAX_DECODER_BACKLOG, os.cpu_count() or 0, _lag,
                     block_size, 100.0 * block_size / _lag, _lag,
                     complex_data_number * _lag * 8 / 1e9)
                  + bcolors.ENDC)
        # One receiver at a time. A second receive thread calling recv() on
        # the same rx_streamer would split the sample stream with the first,
        # double the shared segments and decoder children, and overwrite
        # self.rx_proc so the first could no longer be joined or stopped.
        # _tx_claim() is the transmit side's equivalent.
        if not self._rx_claim():
            raise RuntimeError(
                "rx_start: a receiver is already running. Two threads "
                "calling recv() on one streamer split the sample stream "
                "between them and neither gets a decodable buffer. Call "
                "rx_stop(wait=True) first.")
        # Claim -> thread -> start, atomically. _rx_claim() has already
        # taken the claim, and _RX_CLAIMED stands in for the thread until
        # the assignment below. If Thread() or .start() raises in between
        # -- thread exhaustion, or "can't create new thread at interpreter
        # shutdown" from an atexit path -- the except clause releases the
        # claim and clears receiving.value, so a later rx_start() is not
        # refused for the life of the process and rx_stop(wait=True) never
        # meets a thread that was never started. The transmit side
        # recovers from the same shape because _tx_workers() filters on
        # is_alive().
        # A startup handshake. rx() runs in a worker thread, so everything
        # its setup does -- SharedMemory(create=True), mp.Process.start(),
        # issue_stream_cmd -- raises where nobody is looking. Without the
        # handshake rx_start() would return a full set of queues for a
        # receiver whose thread was already dead. A large window with the
        # automatic ring floor can ask SharedMemory for gigabytes
        # (tx_rx_example.py's 12,000,000-sample window at the default
        # block_size of 20 is 1.92 GB), so this is an ordinary failure.
        #
        # rx() sets the event once setup is complete, and sets it on its
        # way out of a failed setup too, so a failure never leaves this
        # waiting. target stays rx() itself rather than a wrapper: the
        # hand-off can be checked by binding inspect.signature(target)
        # against the kwargs, and a wrapper taking **kw would erase that.
        # rx() records its own setup failure instead.
        _rx_started = threading.Event()
        _rx_failure = []

        try:
            self.rx_proc = threading.Thread(
                daemon=True,
                name="Receiver", target=rx,
                kwargs=dict(sample_rate=self.sample_rate, sf_list=sf_list,
                            bandwidth=self.bandwidth,
                            receiving=self.receiving,
                            packet_queue=rx_packet_queue,
                            complex_data_number=complex_data_number,
                            streamer=self.rx_streamer,
                            pause_rec=self.rx_pause_flag,
                            buf_size=block_size,
                            decode_opts=decode_opts,
                            decode_opts_per_sf=decode_opts_per_sf,
                            max_payload_len=max_payload_len,
                            rx_started=_rx_started,
                            rx_failure=_rx_failure))
            self.rx_proc.start()
        except BaseException:
            self.rx_proc = None
            self.receiving.value = False
            raise

        # Wait for the handshake. A setup failure sets the event on its
        # way out, so this returns at once in the case it exists for; a
        # slow but healthy start is bounded by the same timeout rx() gives
        # each child, plus a little. Timing out is not an error here --
        # only a thread that has already failed is -- so a receiver that
        # merely starts slowly is returned as usual.
        _rx_started.wait(RX_START_GRACE)
        if _rx_failure:
            self.rx_proc = None
            self.receiving.value = False
            raise RuntimeError(
                "the receiver failed during setup and never started: %r. "
                "Nothing was listening, so every queue rx_start() would "
                "have returned stays empty for ever."
                % (_rx_failure[0],)) from _rx_failure[0]
        # Returned in the caller's order. rx() binds queue[i] to the i-th
        # entry of np.array([5..12])[isin(...)], which is always ascending,
        # so without this rx_start(sf_list=[12, 7]) would hand back queue[0]
        # carrying SF7. Reordering here keeps rx() simple and makes the
        # returned array mean what the argument said.
        order = sorted(range(len(sf_list)), key=lambda i: sf_list[i])
        self.rx_queues = np.empty((len(sf_list),), dtype=mp_queues.Queue)
        for caller_pos, ascending_pos in enumerate(order):
            self.rx_queues[ascending_pos] = rx_packet_queue[caller_pos]
        self.rx_sf_order = list(sf_list)
        # self.rx_queues, not rx_packet_queue: the return value is what the
        # examples and the README use, so it is the reordered array.
        return self.rx_queues


    def rx_stop(self, wait=False, timeout=10.0):
        """Stop the receiver. Safe to call while paused, and before start."""
        self.receiving.value = False
        # Wake a paused loop. rx() parks in pause_rec.wait() with no
        # timeout, so clearing only the run flag would leave rx_pause()
        # followed by rx_stop() a permanent deadlock, with the stream, the
        # children and the shared segment all still up. Setting the event
        # releases the wait; the loop condition then sees receiving.value
        # False and leaves through the normal cleanup. It does not resume
        # reception -- the flag is already clear -- so rx_pause()'s
        # guarantee is intact.
        self.rx_pause_flag.set()
        if wait:
            # Idempotent, and safe before any rx_start(). A
            # constructed-but-never-started Thread would raise "cannot join
            # thread before it is started" here, but rx_start's except
            # clause sets rx_proc back to None on exactly that path, so this
            # object never holds one.
            if self.rx_proc is not None:
                self.rx_proc.join(timeout)
                if self.rx_proc.is_alive():
                    print(bcolors.WARNING + "RX: the receive thread did "
                          "not stop within %.1f s" % timeout + bcolors.ENDC)

    def rx_pause(self):
        self.rx_pause_flag.clear()

    def rx_resume(self):
        self.rx_pause_flag.set()


    def _rx_claim(self):
        """Atomically take the receiver, or report it already taken.

        Test rx_proc, set receiving.value and install a placeholder under
        one lock, so two concurrent rx_start() calls cannot both pass the
        test and put two receive threads on one streamer.

        The placeholder is what makes it atomic: the real thread does not
        exist yet at the moment the claim must become visible.
        """
        lock = getattr(self, "_rx_lock", None)
        if lock is None:
            lock = self._rx_lock = threading.Lock()
        with lock:
            prev = getattr(self, "rx_proc", None)
            if prev is not None and prev.is_alive():
                return False
            self.rx_proc = _RX_CLAIMED
            # Set here, before the thread exists, so an rx_stop() that lands
            # during rx()'s multi-second startup is seen rather than
            # overwritten.
            self.receiving.value = True
            return True

    def _tx_claim(self):
        """Atomically take the transmitter, or report it already taken.

        The flag is tested and set under one lock, by the caller and before
        the worker exists, so two back-to-back tx_* calls cannot both pass
        and drive one tx_streamer concurrently.
        """
        with self._tx_lock:
            if self.sending.value:
                return False
            # And no worker still running. tx_stop() clears the flag at
            # once, but the worker can be parked in _tx_sleep() for far
            # longer than any join timeout -- an EU868 duty cycle forces
            # roughly 100x airtime between SF12 frames, so a sleep_time
            # above join_timeout is the normal setting. Re-setting the flag
            # while that stale worker is alive would wake it, and two
            # workers would share one streamer.
            if self._tx_workers():
                return False
            self.sending.value = True
            return True

    def _tx_release(self):
        with self._tx_lock:
            self.sending.value = False

    def tx_send_burst(self, pkt_list, sleep_time, verbose = False, encode_opts = None,
                      join_timeout=10.0):
        """Transmit a list of LoRaPackets.

        Each packet's own cr / ih / has_crc are what go on the air. encode_opts
        carries the rest of lora.encode()'s parameters -- preamble_bits,
        pseudo_header, cfo_ppm, sync_word, ldro -- for the whole burst, and may
        also override cr / enable_crc / implicit_header for all of it.
        """
        # Validated here, in the caller's thread, before a worker exists.
        encode_opts = resolve_encode_opts(encode_opts, fs=self.sample_rate,
                                          f0=self.tx_freq)
        # Join a previous worker first, exactly as tx_start does. Otherwise
        # tx_stop() -- whose default is wait=False -- followed by a burst
        # could put two workers on one tx_streamer, and the stale worker's
        # release clause would then clear the new worker's claim.
        self._tx_join_previous(join_timeout)
        if self._tx_claim():
            # name and daemon, like the tx_start worker, and tracked by
            # _tx_track so tx_stop(wait=True) and _tx_join_previous can
            # find it.
            tx_burst_proc = threading.Thread(
                name="Transmitter", daemon=True,
                target=tx_burst,
                kwargs=dict(sample_rate=self.sample_rate,
                            center_freq=self.tx_freq, pkt_list=pkt_list,
                            sleep_time=sleep_time, sending=self.sending,
                            streamer=self.tx_streamer,
                            amplitude=self.signal_amplitude, verbose=verbose,
                            encode_opts=encode_opts))
            self._tx_track(tx_burst_proc)
            tx_burst_proc.start()
            return tx_burst_proc
        else:
            print("The radio is already transmitting!")
            return None


    def tx_send_burst_multi_sf(self, pkt_list, sleep_time, sf_list, encode_opts = None,
                               join_timeout=10.0):
        encode_opts = resolve_encode_opts(encode_opts, fs=self.sample_rate,
                                          f0=self.tx_freq)
        # The SF list, checked here in the caller like every other transmit
        # argument: [7, 13], [9, 5] with explicit-header packets, [] and
        # [7, 7.5] would otherwise return a live thread that dies before
        # its first send().
        # Any iterable of whole SFs, as rx_start takes it (a numpy array or
        # a range included). And the header mode a packet will really go
        # out with: encode_opts' implicit_header, when given, overrides the
        # packet's own ih in encode_packet, so that is what SF5/SF6 is
        # checked on.
        if isinstance(sf_list, (str, bytes)) or not hasattr(sf_list,
                                                            "__iter__"):
            raise ValueError("sf_list must be a list of spreading factors, "
                             "got %r" % (sf_list,))
        try:
            sfs = [lora._whole(s, "SF") for s in sf_list]
        except ValueError as e:
            raise ValueError("sf_list must hold whole spreading factors: %s"
                             % e)
        if not sfs or not all(5 <= s <= 12 for s in sfs):
            raise ValueError("sf_list must be a non-empty list of whole "
                             "spreading factors 5..12, got %r" % (sf_list,))
        ih_opt = encode_opts.get("implicit_header")
        if min(sfs) < 7 and not all(
                (int(ih_opt) if ih_opt is not None else getattr(p, "ih", 0))
                for p in pkt_list):
            raise ValueError("sf_list includes SF%d, which has no room for "
                             "an explicit header: every packet must go out "
                             "implicit-header (ih=1, or encode_opts "
                             "implicit_header=1)" % min(sfs))
        sf_list = sfs
        # Join a previous worker first, exactly as tx_start does. Otherwise
        # tx_stop() -- whose default is wait=False -- followed by a burst
        # could put two workers on one tx_streamer, and the stale worker's
        # release clause would then clear the new worker's claim.
        self._tx_join_previous(join_timeout)
        if self._tx_claim():
            # name and daemon, like the tx_start worker, and tracked by
            # _tx_track so tx_stop(wait=True) and _tx_join_previous can
            # find it.
            tx_burst_proc = threading.Thread(
                name="Transmitter", daemon=True,
                target=tx_burst_multi_sf,
                kwargs=dict(sample_rate=self.sample_rate,
                            center_freq=self.tx_freq, pkt_list=pkt_list,
                            sleep_time=sleep_time, sending=self.sending,
                            streamer=self.tx_streamer, sf_list=sf_list,
                            amplitude=self.signal_amplitude,
                            encode_opts=encode_opts))
            self._tx_track(tx_burst_proc)
            tx_burst_proc.start()
            return tx_burst_proc
        else:
            print("The radio is already transmitting!")
            return None



    def tx_start(self, sleep_time, verbose = False, encode_opts = None,
                 join_timeout=10.0):
        """Start the transmit worker; see tx_send_burst for encode_opts.

        Waits for a previous worker to finish first. tx_stop() defaults to
        wait=False, so without the wait tx_stop() then tx_start() could put
        two workers on one tx_streamer, and the stale worker's finally
        would clear sending.value -- which by then belongs to the new
        worker -- stranding its queued packets. Two threads calling send()
        on one UHD streamer is undefined behaviour in any case.
        """
        # Validated here, before the worker exists, so a bad value cannot
        # kill the worker on the first packet and lose every later put().
        encode_opts = resolve_encode_opts(encode_opts, fs=self.sample_rate,
                                          f0=self.tx_freq)
        self._tx_join_previous(join_timeout)
        if self._tx_claim():
            # One local binding, handed to the worker and returned to the
            # caller. tx_stop() replaces self.tx_queue (that is what stops
            # a late put() reaching a dying worker), so reading it back at
            # the return statement would race a tx_stop() and could hand
            # the caller a queue the worker is not draining.
            q = mp.Queue(0)
            self.tx_queue = q
            # daemon=True, for the same reason the receiver thread is one:
            # it parks in pkt_queue.get() with no timeout, and a non-daemon
            # thread doing that keeps the interpreter from exiting.
            # tx_stop() remains the clean path, and the examples call it.
            tx_proc = threading.Thread(
                daemon=True,
                name="Transmitter", target=tx,
                kwargs=dict(sample_rate=self.sample_rate,
                            center_freq=self.tx_freq, pkt_queue=q,
                            sleep_time=sleep_time, sending=self.sending,
                            streamer=self.tx_streamer,
                            amplitude=self.signal_amplitude, verbose=verbose,
                            encode_opts=encode_opts))
            self._tx_track(tx_proc)
            tx_proc.start()
            return q


        else:
            # Raise, as rx_start does on the same condition. Returning
            # self.tx_queue here would hand the caller a queue that nothing
            # will ever drain: tx_stop() replaces tx_queue, so a stale
            # worker is parked on the old one, and the claim is refused
            # without setting sending.value -- a truthy queue and an idle
            # flag, both saying "fine", while every packet put() vanishes.
            raise RuntimeError(
                "tx_start: a transmitter is already running. Two threads "
                "calling send() on one streamer is undefined behaviour, and "
                "the queue this would return is not the one the live worker "
                "is draining. Call tx_stop(wait=True) first.")

    def _tx_workers(self):
        """This radio's live transmit threads.

        The instance's own, not every thread in the process named
        "Transmitter": the name is process-global, so with two radio
        objects each would otherwise wait on, and warn about, the other's
        worker. The name is still set, because it is what a stack trace
        shows.
        """
        me = threading.current_thread()
        return [th for th in getattr(self, "_tx_threads", [])
                if th is not me and th.is_alive()]

    def _tx_track(self, th):
        self._tx_threads = [x for x in getattr(self, "_tx_threads", [])
                            if x.is_alive()] + [th]

    def _tx_join_previous(self, timeout):
        """Wait for any worker from a previous tx_start() to finish.

        Wakes it first: without the sentinel it is parked in
        pkt_queue.get() with no timeout and the join would be the full
        `timeout` every time.
        """
        stale = self._tx_workers()
        if not stale:
            return
        # Only wake a worker whose run flag is already clear. If
        # sending.value is still set the transmitter is legitimately
        # busy and the claim below is about to be refused; a sentinel
        # would poison the running worker's queue and kill the live
        # stream behind a refusal that says nothing happened.
        if self.sending.value:
            return
        q = getattr(self, "tx_queue", None)
        if q is not None:
            try:
                q.put(None)
            except Exception:
                pass
        for th in stale:
            th.join(timeout)
            if th.is_alive():
                # "tx:", because this is called from tx_start,
                # tx_send_burst and tx_send_burst_multi_sf. The call is
                # refused, not started anyway: _tx_claim() refuses on
                # `if self._tx_workers()` immediately after this returns.
                print(bcolors.WARNING + "tx: a previous transmitter is "
                      "still running after %.1f s. This call will be "
                      "REFUSED rather than share the streamer with it; "
                      "tx_stop(wait=True) first, or wait for the burst "
                      "to finish." % timeout
                      + bcolors.ENDC)

    def tx_stop(self, wait=False, timeout=10.0):
        """Stop the transmit worker and drop anything still queued.

        Clearing the flag is not enough: tx() re-reads sending.value only
        after pkt_queue.get() returns, so a worker parked on an empty queue
        would not notice, and a packet queued after the stop would still be
        transmitted. A sentinel wakes it; the queue is then replaced so a
        late put() cannot reach a worker that is on its way out.
        """
        self.sending.value = False
        q = getattr(self, "tx_queue", None)
        if q is not None:
            try:
                q.put(None)          # wake a worker blocked in get()
            except Exception:
                pass
        # And replace it, so a caller holding the returned queue cannot
        # put() into the one a dying worker is draining.
        self.tx_queue = mp.Queue(0)
        if wait:
            for th in self._tx_workers():
                th.join(timeout)
                if th.is_alive():
                    print("tx_stop: transmitter did not stop within "
                          "%.1f s" % timeout)




        # print("Main class", self.rx_streamer)
















