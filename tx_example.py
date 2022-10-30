import lora_transceiver
import lora
import lora_utils
import numpy as np
import multiprocessing as mp
import multiprocessing.queues as mp_queues
import queue as q
import time





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
