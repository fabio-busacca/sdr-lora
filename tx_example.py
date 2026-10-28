# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2024 SDR-LoRa contributors.
#
# Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
# See LICENSE for the full text.
import lora_transceiver
import lora
import lora_utils
import numpy as np
import multiprocessing as mp
import multiprocessing.queues as mp_queues
import queue as q
import time





# The script body runs under `if __name__ == "__main__":`. It starts
# multiprocessing.Process children, and every start method except fork
# re-imports __main__ in the child, so without the guard the child would
# re-run this file and start its own children. That covers Windows and
# macOS, and Linux from CPython 3.14, where the default is forkserver.


# Upper bound on the wait for the transmit queue to drain, in seconds.
DRAIN_TIMEOUT = 120.0


def main():
    address = "192.168.40.2"
    rx_gain = 10
    tx_gain = 20
    bandwidth = 125000
    center_freq = 1e9
    sample_rate = 1e6

    tx_freq = 990e6  # Hz
    rx_freq = 1010e6  # Hz
    tx_ch_ID = 1
    rx_ch_ID = 0


    SF = 7



    sleep_time = 1
    loradio = lora_transceiver.lora_transceiver(address, rx_gain, tx_gain, bandwidth, rx_freq, tx_freq, sample_rate,
                                                rx_ch_ID, tx_ch_ID)


    data_array = np.ones(500,dtype=np.uint8)
    packet_size = 250
    srcID = 0
    dstID = 1
    CR = 1

    tx_queue = loradio.tx_start(sleep_time = sleep_time, verbose=True)

    data = lora_utils.pack_lora_data(data_array, SF, bandwidth, packet_size, srcID, dstID, extended_sqn=False, CR = CR)

    for pack in data:
        tx_queue.put(pack)

    # Wait for the queue to drain, then stop the transmitter. The worker
    # blocks in pkt_queue.get() with no timeout, so the script does not
    # exit until tx_stop() is called, and tx_stop() drops whatever is still
    # queued -- hence the wait before it.
    #
    # empty(), not qsize(): multiprocessing.Queue.qsize() raises
    # NotImplementedError on macOS. Both are approximate (the feeder thread
    # can still hold an item that has not reached the pipe); the extra
    # settle below covers that.
    #
    # The wait is bounded, and it stops early if the worker has died (a
    # packet it cannot encode ends it, and then the queue never empties).
    # sending.value is the transmitter's own claim flag, cleared however
    # the worker ends, so it says whether anything is still draining.
    deadline = time.time() + DRAIN_TIMEOUT
    while not tx_queue.empty():
        if not loradio.sending.value:
            print("Transmitter: the worker stopped with packets still "
                  "queued; not waiting for a queue nothing is draining.")
            break
        if time.time() > deadline:
            print("Transmitter: timed out waiting for the queue to drain.")
            break
        time.sleep(0.5)
    time.sleep(0.5)
    loradio.tx_stop(wait=True)


if __name__ == "__main__":
    main()
