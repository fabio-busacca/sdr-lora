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

    rx_freq = 990e6  # Hz
    tx_freq = 1010e6  # Hz
    tx_ch_ID = 1
    rx_ch_ID = 0


    sf = [7,8,9,10,11,12]
    sleep_time = 1

    rx_listeners = list()
    samplesBlockRec = 3000000
    loradio = lora_transceiver.lora_transceiver(address, rx_gain, tx_gain, bandwidth, rx_freq, tx_freq, sample_rate,
                                                rx_ch_ID, tx_ch_ID)

    rx_queues = loradio.rx_start(sf, samplesBlockRec)

    for i in range(len(sf)):
        rx_listeners.append(mp.Process(target=pkt_reader, args=(rx_queues[i],)))
        rx_listeners[i].start()

    # The readers loop until their spreading factor stops, so joining them
    # would leave the RX stream running, the decoder children alive and the
    # shared segment mapped. Wait for a Ctrl-C instead, then shut down in
    # order.
    print("Receiving. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        loradio.rx_stop(wait=True)
        for listener in rx_listeners:
            listener.terminate()
            listener.join()


if __name__ == "__main__":
    main()
