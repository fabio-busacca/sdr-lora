

import threading
import numpy as np
#import queue as Queue
import socket
import time
import lora
import uhd
import multiprocessing as mp
import multiprocessing.queues as mp_queues
import sys
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


def decoder_process(shm_name,complex_data_number, buf_size,sf_index_queue,pkt_queue, sf, BW, fs, SF_sample_th):
    # temp_samples = np.zeros(complex_data_number,dtype=np.complex64)



    print("Decoder on SF", str(sf), ", PID:", str(os.getgid()))
    sf_minimum_win_size_arr = np.array((650e3, 1.2e6, 2e6, 3.6e6, 8e6, 14.5e6), dtype= np.uint32)
    sf_minimum_win_size = int(1.1 * sf_minimum_win_size_arr[sf-7])

    buf_length = complex_data_number * buf_size
    temp_samples = np.zeros(sf_minimum_win_size, dtype=np.complex64)
    existing_shm = mp.shared_memory.SharedMemory(name=shm_name)
    rec_buffer = np.ndarray((buf_length,), dtype=np.complex64, buffer=existing_shm.buf)
    sf_windows_len = complex_data_number * SF_sample_th
    old_rec = 0
    rep_count = 0

    win_start_index = 0

    count_debug = 0

    while True:

        count_debug = count_debug + 1
        if(count_debug == 5):
            print(bcolors.OKGREEN + "process on SF",sf, "is alive!!!" + bcolors.ENDC)
            count_debug = 0
        #sys.stdout.flush()
        # start_t = time.time()
        #print("Decoder")
        item = sf_index_queue.get()
        #print("Starting decoding on SF",sf)
        c = rec_buffer[win_start_index:win_start_index + sf_windows_len]
        win_start_index = (win_start_index + sf_windows_len) % buf_length
        # print(c.tolist())
        # c.real = np.array(samples_buffer[cursor:cursor+block_size:2 * (1 * float_size) ], dtype = np.float32)
        # c.imag = np.array(samples_buffer[cursor+(1 * float_size) :cursor+block_size:2 * (1 * float_size) ], dtype = np.float32)
        #DECODE THE SAMPLES THROUGH THE DECODE FUNCTION FROM THE LORA MODULE
        #print("Decoder End")

        #print("Passing the samples from sf", sf,  "to a decoder process")
        (mp.Process(target=thread_decode, args=(np.concatenate((temp_samples,c)), sf, BW, fs, pkt_queue))).start()

        temp_samples = c[-sf_minimum_win_size:]

    return
    #print("esco")

def thread_decode(samples, sf, BW, fs, pkt_queue):
    print(bcolors.OKBLUE + "[DECODER PROCESS] Decoding samples from sf", sf, bcolors.ENDC)

    ans = lora.decode(samples, sf, BW, fs)
    if ans.size > 0:
        print("# DECODED PACKETS", ans.size)

        for pkt in ans:
            pkt_queue.put(pkt)
    else:
        pkt_queue.put(int(0))

        # for pkt in ans:
        #     print(pkt)



def tx_burst(sample_rate, center_freq, pkt_list, sleep_time, sending, streamer, amplitude, verbose):

    #usrp = uhd.usrp.MultiUSRP("address=" + address)

    #print("TX Acquiring")
    metadata = uhd.types.TXMetadata()
    for index,pkt in enumerate(pkt_list):
        print("Transmitter: Processing Packet #", index)

        sending.value = True
        buffer_samps = streamer.get_max_num_samps()

        samples = amplitude * lora.encode(center_freq, pkt.SF, pkt.BW, pkt.payload, sample_rate, pkt.src, pkt.dst, pkt.seqn, 1, 1, 0, 8)
        print("Encoded Packet #", index)
        proto_len = samples.shape[-1]
        send_samps = 0
        samples = samples.reshape(1, samples.size)
        while send_samps < proto_len:
            real_samps = min(proto_len, buffer_samps - send_samps)
            if real_samps < proto_len:
                n_samples = streamer.send(samples[:real_samps], metadata)
            else:
                n_samples = streamer.send(samples, metadata)
            send_samps += n_samples
            if (verbose):
                print("Sent samples", n_samples)
        metadata.end_of_burst = True
        print("Ending Burst for Packet #", index)

        streamer.send(np.zeros((1, 1), dtype=np.complex64), metadata)

        print("Sent Packet #", index)

        # Send EOB to terminate Tx

        #print("Sent packet with seq number", pkt.seqn)
        time.sleep(sleep_time)

    sending.value = False


def tx_burst_multi_sf(sample_rate, center_freq, pkt_list, sleep_time, sending, streamer, sf_list, amplitude):

    #usrp = uhd.usrp.MultiUSRP("address=" + address)


    n_pack = len(pkt_list)
    print("N_pack:", n_pack)
    #remaining_pkt = np.array([len(pkt_list)] * len(sf_list))
    pkt_count = np.zeros((len(sf_list)-1), dtype=np.uint32)
    sf_list.sort(reverse = True)
    #print("TX Acquiring")
    metadata = uhd.types.TXMetadata()
    for index,pkt in enumerate(pkt_list):

        sending.value = True
        #print("TX Acquired")

        buffer_samps = streamer.get_max_num_samps()
        #print("LOL")



        samples = amplitude * lora.encode(center_freq, sf_list[0], pkt.BW, pkt.payload, sample_rate, pkt.src, pkt.dst, pkt.seqn, 1, 1, 0, 8)
        max_len = samples.size

        for i,sf in enumerate(sf_list[1:]):
            add_samples = np.zeros((samples.size), dtype=np.complex64)
            length = 0
            incr = 0
            coded_pkt = 0
            while (pkt_count[i] < n_pack):
                current_pkt = pkt_list[pkt_count[i]]
                current_samples = lora.encode(center_freq, sf, current_pkt.BW, current_pkt.payload, sample_rate, current_pkt.src, current_pkt.dst,
                                      current_pkt.seqn, 1, 1, 0, 8)


                add_samples[length:length + current_samples.size] = current_samples
                length = length + current_samples.size
                coded_pkt = coded_pkt + 1
                pkt_count[i] = pkt_count[i] + 1
                if (length * ((coded_pkt + 1) / coded_pkt)) > max_len:
                    print("Pkt Count for SF", sf,":", pkt_count[i])
                    break




            samples = samples + add_samples



        proto_len = samples.shape[-1]
        send_samps = 0
        samples = samples.reshape(1, samples.size)
        while send_samps < proto_len:

            real_samps = min(proto_len, buffer_samps - send_samps)
            if real_samps < proto_len:
                n_samples = streamer.send(samples[:real_samps], metadata)
            else:
                n_samples = streamer.send(samples, metadata)
            send_samps += n_samples

            metadata.end_of_burst = True
            streamer.send(np.zeros((1, 1), dtype=np.complex64), metadata)


        # Send EOB to terminate Tx

        #print("Sent packet with seq number", pkt.seqn)
        time.sleep(sleep_time)

    sending.value = False






def tx(sample_rate, center_freq, pkt_queue, sleep_time, sending, streamer, amplitude, verbose = False):

    #usrp = uhd.usrp.MultiUSRP("address=" + address)

    sending.value = True
    metadata = uhd.types.TXMetadata()
    while sending.value:

        pkt = pkt_queue.get()
        if(verbose):
            print("Sending pkt", pkt)

        buffer_samps = streamer.get_max_num_samps()

        samples = amplitude * lora.encode(center_freq, pkt.SF, pkt.BW, pkt.payload, sample_rate, pkt.src, pkt.dst, pkt.seqn, 1, 1, 0, 8)
        proto_len = samples.shape[-1]
        send_samps = 0
        samples = samples.reshape(1, samples.size)
        while send_samps < proto_len:



            real_samps = min(proto_len, buffer_samps - send_samps)
            if real_samps < proto_len:
                n_samples = streamer.send(samples[:real_samps], metadata)
            else:
                n_samples = streamer.send(samples, metadata)
            send_samps += n_samples

            if(verbose):
                print("Sent samples", n_samples)
                print("Total Sent Samples", send_samps)

        metadata.end_of_burst = True
        streamer.send(np.zeros((1, 1), dtype=np.complex64), metadata)


        # Send EOB to terminate Tx

        #print("Sent packet with seq number", pkt.seqn)
        time.sleep(sleep_time)

    sending.value = False













def rx(sample_rate, sf_list, bandwidth, receiving, packet_queue, complex_data_number, streamer, pause_rec,buf_size = 120):






    #LORA RECEIVER MAIN THREAD#


    #THE RECEIVER SCRIPT IS STRUCTURED AS A TWO-PROCESS PROGRAM: ONE PROCESS (THE MAIN THREAD OR RECEIVER THREAD) IS RESPONSIBLE FOR READING AND
    #BUFFERING OF RF DATA FROM THE USRP RADIO; THE OTHER PROCESS (THE DECODER THREAD) READS AND PROCESSES DATA FROM THE BUFFER
    #THE PROGRAM RESORTS TO A CIRCULAR BUFFER, AND MAKES USE OF A QUEUE TO EXCHANGDE DATA BETWEEN THE PROCESSES
    #MORE IN DETAIL, THE BUFFER IS LOCATED IN A SHARED MEMORY AREA. EACH TIME A NEW CHUNK OF DATA IS RECEIVED, THE RECEIVER PUTS THE DATA START INDEX IN THE QUEUE.
    #THE PROCESSING THREAD CAN ACCORDINGLY READ AND, POSSIBLY, DECODE THE LORA DATA IN THE CHUNK.


    #MAXIMUM NUMBER OF DATA CHUNKS IN THE SHARED MEMORY BUFFER
    BUF_SIZE = int(buf_size) #MAKE SURE THIS NUMBER IS A MULTIPLE OF THE MAXIMUM SAMPLES THRESHOLD DIVIDED BY COMPLEX_DATA_NUMBER
    #FOR INSTANCE, WE NOW HAVE 72 MS FOR SF 12, AND 3M AS COMPLEX_DATA_NUMBER. 72/3 = 24, AND 120 IS INDEED A MULTIPLE OF 24

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
    SF_sample_th = np.array([1, 1, 1, 1, 1, 1])
    sf_arr = np.array([7, 8, 9, 10, 11, 12])


    indexes = np.isin(sf_arr, sf_list)

    SF_sample_th = SF_sample_th[indexes]
    sf_arr = sf_arr[indexes]


    queue_arr = np.empty(shape=(sf_arr.size,), dtype=mp_queues.Queue)
    processes_arr = np.empty(shape=(sf_arr.size,), dtype=mp.Process)





    #CREATION OF THE SHARED MEMORY AREA
    shm = mp.shared_memory.SharedMemory(create = True, size = block_size * BUF_SIZE)
    samples_buffer = np.ndarray(complex_data_number * BUF_SIZE, dtype=np.complex64, buffer=shm.buf)
    buffer_size = samples_buffer.size
    index_queue = mp.Queue(0)


    print("########################")
    print(mp.get_start_method('spawn'))
    print("########################")
    #CREATION OF THE DECODER PROCESSES

    for index, sf in enumerate(sf_arr):
        queue_arr[index] = mp.Queue(0)
        processes_arr[index] = mp.Process(name="Decoder SF" + str(sf), target=decoder_process, args=(shm.name, complex_data_number, BUF_SIZE,
                                                                        queue_arr[index], packet_queue[index], sf, bandwidth,
                                                                        sample_rate, SF_sample_th[index]))


        processes_arr[index].start()
        while (not (processes_arr[index].is_alive())):
            time.sleep(0.1)

    for index, sf in enumerate(sf_arr):
        while (not (processes_arr[index].is_alive())):
            processes_arr[index] = mp.Process(name="Decoder SF" + str(sf), target=decoder_process,
                                              args=(shm.name, complex_data_number, BUF_SIZE,
                                                    queue_arr[index], packet_queue[index], sf, bandwidth,
                                                    sample_rate, SF_sample_th[index]))

            processes_arr[index].start()
            time.sleep(1)


    #CREATION OF THE THRESHOLD TRIGGER PROCESS


    trigger_process = mp.Process(target=threshold_trigger_process, args=(index_queue, SF_sample_th, tuple(queue_arr)))
    trigger_process.daemon = True
    trigger_process.start()


    debug = True


    # # Start Stream
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)
    receiving.value = True

    metadata = uhd.types.RXMetadata()

    print("Starting...")

    #("Channels",streamer.get_num_channels())
    buf_length = complex_data_number
    recv_buffer = np.zeros((buf_length,), dtype=np.complex64)
    #START RECEIVING
    while receiving.value:
        try:

            pause_rec.wait()
            streamer.recv(recv_buffer, metadata)
            #streamer.recv(recv_buffer, metadata, timeout = 10)
            #streamer.recv(samples_buffer[cursor:cursor + complex_data_number], metadata)
            if(debug):
                print(bcolors.OKGREEN + "DEBUG metadata ", metadata, bcolors.ENDC)

            samples_buffer[cursor:cursor + buf_length] = recv_buffer
            cursor = (cursor + buf_length) % buffer_size
            if(cursor % complex_data_number == 0):
                index_queue.put(cursor)
            # cursor = (cursor + complex_data_number) % buffer_size
            #print("Rec")
            if(cursor == 0):
                # stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                # streamer.issue_stream_cmd(stream_cmd)
                # time.sleep(0.1)
                # stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
                # stream_cmd.stream_now = True
                # streamer.issue_stream_cmd(stream_cmd)
                pass




        except KeyboardInterrupt:
            print("Stopping RX")
            rec = False
            # Stop Stream
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            streamer.issue_stream_cmd(stream_cmd)
            for proc in processes_arr:
                proc.terminate()
                proc.join()
            trigger_process.terminate()
            trigger_process.join()
            shm.close()
            shm.unlink()
            break

    else:
        print("Stopping RX")
        # Stop Stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stream_cmd)
        for proc in processes_arr:
            proc.terminate()
            proc.join()
        trigger_process.terminate()
        trigger_process.join()
        shm.close()
        shm.unlink()















class lora_transceiver():

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
        self.signal_amplitude = signal_amplitude
        self.usrp = uhd.usrp.MultiUSRP("address=" + self.address)
        self.rx_channel = rx_channel_ID
        self.tx_channel = tx_channel_ID
        self.usrp.set_rx_rate(sample_rate, self.rx_channel)
        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(rx_freq), self.rx_channel)
        self.usrp.set_rx_gain(rx_gain, self.rx_channel)
        self.usrp.set_rx_bandwidth(bandwidth, self.rx_channel)
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

    def rx_start(self, sf_list, complex_data_number = 3000000, block_size = 20):
        # THE DEFAULT MINIMUM BLOCK OF SAMPLES ROUGHLY CORRESPONDS TO 5 TIMES THE SIZE OF A 251 BYTES SF7 PACKET
        rx_packet_queue = np.empty((len(sf_list)), dtype=mp_queues.Queue)
        #print("SF LIST",sf_list)
        print("RX Start")
        for index in range(len(sf_list)):
            rx_packet_queue[index] = mp.Queue(0)
        self.rx_proc = threading.Thread(name = "Receiver", target=rx, args=(self.sample_rate, sf_list, self.bandwidth, self.receiving, rx_packet_queue, complex_data_number, self.rx_streamer, self.rx_pause_flag, block_size))
        self.rx_proc.start()
        self.rx_queues = rx_packet_queue
        return rx_packet_queue


    def rx_stop(self, wait = False):
        self.receiving.value = False
        if(wait):
            self.rx_proc.join()

    def rx_pause(self):
        self.rx_pause_flag.clear()

    def rx_resume(self):
        self.rx_pause_flag.set()


    def tx_send_burst(self, pkt_list, sleep_time, verbose = False):
        if(not self.sending.value):
            tx_burst_proc = threading.Thread(target=tx_burst, args=(self.sample_rate, self.tx_freq, pkt_list, sleep_time, self.sending, self.tx_streamer, self.signal_amplitude, verbose))
            tx_burst_proc.start()
            return tx_burst_proc
        else:
            print("The radio is already transmitting!")
            return None


    def tx_send_burst_multi_sf(self, pkt_list, sleep_time, sf_list):
        if(not self.sending.value):
            tx_burst_proc = threading.Thread(target=tx_burst_multi_sf, args=(self.sample_rate, self.tx_freq, pkt_list, sleep_time, self.sending, self.tx_streamer, self.signal_amplitude, sf_list))
            tx_burst_proc.start()
            return tx_burst_proc
        else:
            print("The radio is already transmitting!")
            return None



    def tx_start(self, sleep_time, verbose = False):

        if(not self.sending.value):
            self.tx_queue = mp.Queue(0)
            tx_proc = threading.Thread(name="Transmitter", target=tx, args=(self.sample_rate, self.tx_freq, self.tx_queue, sleep_time, self.sending, self.tx_streamer, self.signal_amplitude, verbose))
            tx_proc.start()
            return self.tx_queue


        else:
            print("TX is already ON!")
            return self.tx_queue

    def tx_stop(self):
        self.sending.value = False




        # print("Main class", self.rx_streamer)
















