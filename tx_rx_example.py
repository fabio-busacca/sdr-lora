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

def pkt_reader(pkt_queue):
    # The queue carries LoRaPacket objects, an int(0) for a window that
    # decoded nothing, and lora_transceiver.DECODER_STOPPED when that
    # spreading factor will never report again -- its decoder died, or
    # the threshold trigger did. The reader exits on the last one instead
    # of blocking in get() for ever; the reason is printed by the receiver.
    while True:
        pkt = pkt_queue.get()
        if pkt == lora_transceiver.DECODER_STOPPED:
            print("this spreading factor has stopped reporting; "
                  "see the RX message above. Reader exiting.")
            return
        if(isinstance(pkt,lora.LoRaPacket)):
            print(pkt)


sf_rx = [7]
sleep_time = 1

rx_listeners = list()

# Number of samples in the base receiving window. The longer it is, the less
# often the decoders run. It does not have to grow with SF: a packet that
# straddles a window boundary is carried into the next window, so every SF
# decodes at any window size. Each packet is reported exactly once, because
# windows do not overlap; the carry lives inside StreamDecoder.
samplesBlockRec = 12000000



# The script body runs under `if __name__ == "__main__":`. It starts
# multiprocessing.Process children, and every start method except fork
# re-imports __main__ in the child, so without the guard the child would
# re-run this file and start its own children. That covers Windows and
# macOS, and Linux from CPython 3.14, where the default is forkserver.


def main():
    address = "192.168.40.2"
    rx_gain = 10
    tx_gain = 20
    bandwidth = 125000
    center_freq = 1e9
    sample_rate = 1e6




    tx_freq = 990e6  # Hz
    rx_freq = 1010e6  # Hz

    # Ask until the answer is 0 or 1. Any other number would fall into the
    # else below and make this a second node 1: both ends would transmit on
    # the same frequency and neither would hear the other.
    while True:
        try:
            raw = input("Insert Node ID (0 or 1):\n")
        except EOFError:
            print("No Node ID given (end of input). Run this from a terminal, "
                  "or pipe in 0 or 1.")
            return
        try:
            ID = int(raw)
        except ValueError:
            print("Node ID must be 0 or 1, not %r." % raw.strip())
            continue
        if ID in (0, 1):
            break
        print("Node ID must be 0 or 1, not %d. The two nodes swap transmit "
              "and receive frequencies, so any other value silently makes "
              "this a second node 1." % ID)

    if(ID == 0):
        srcID = 0
        dstID = 1

    else:
        srcID = 1
        dstID = 0
        temp = tx_freq
        tx_freq = rx_freq
        rx_freq = temp


    tx_ch_ID = 1
    rx_ch_ID = 0


    SF = 7



    sleep_time = 1
    loradio = lora_transceiver.lora_transceiver(address, rx_gain, tx_gain, bandwidth, rx_freq, tx_freq, sample_rate,
                                                rx_ch_ID, tx_ch_ID)


    data_array = np.random.randint(255,size=(500),dtype=np.uint8)


    packet_size = 250

    CR = 1

    tx_queue = loradio.tx_start(sleep_time = sleep_time, verbose=False)
    rx_queues = loradio.rx_start(sf_rx, samplesBlockRec)




    data = lora_utils.pack_lora_data(data_array, SF, bandwidth, packet_size, srcID, dstID, extended_sqn=False, CR = CR)

    for i in range(len(sf_rx)):
        rx_listeners.append(mp.Process(target=pkt_reader, args=(rx_queues[i],)))
        rx_listeners[i].start()




    for _ in range(10):
        for pack in data:
            tx_queue.put(pack)
        time.sleep(3)



    # The readers loop until their spreading factor stops, so joining them
    # would never stop either radio. Wait for a Ctrl-C instead, then shut
    # down in order.
    print("Receiving. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        loradio.tx_stop(wait=True)
        loradio.rx_stop(wait=True)
        for listener in rx_listeners:
            listener.terminate()
            listener.join()


if __name__ == "__main__":
    main()
