

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

np.set_printoptions(threshold=sys.maxsize)



#AUSILIARY PROCESS TO KEEP TRACK OF THE RECEIVED PACKETS, USEFUL FOR DEBUGGING
def process_packets(pkt_queue, sf):
    packets_list = np.zeros(3000, dtype=np.uint8)
    ofs = 0
    last_seqn = 0
    while True:
        ans = pkt_queue.get()
        for el in ans:


            if last_seqn - int(el.seqn) > 0:
                ofs = ofs + 1
                print("New Ofs", ofs)

            last_seqn = int(el.seqn)
            packets_list[256 * ofs + el.seqn] = packets_list[256 * ofs + el.seqn] + 1
            print("Received packet with seq number", el.seqn, "on SF", sf)


            # if rec - old_rec == 0:
            #     rep_count = rep_count + 1
            #     if (rep_count > 20):
            #         print("Doubled Pack")
            #         print(packets_list)
            #
            # else:
            #     old_rec = rec
            #     rep_count = 0
        rec = (np.nonzero(packets_list))[0].size
        print("Rec So Far on SF",sf, ":", rec)
        if (rec % 100) == 0 or (rec % 256 == 0):
                print(packets_list)




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

    sf_minimum_win_size_arr = np.array((650e3, 1.2e6, 2e6, 3.6e6, 8e6, 14.5e6), dtype= np.uint32)
    sf_minimum_win_size = sf_minimum_win_size_arr[sf-7]

    buf_length = complex_data_number * buf_size
    temp_samples = np.zeros(sf_minimum_win_size, dtype=np.complex64)
    existing_shm = mp.shared_memory.SharedMemory(name=shm_name)
    rec_buffer = np.ndarray((buf_length,), dtype=np.complex64, buffer=existing_shm.buf)
    sf_windows_len = complex_data_number * SF_sample_th[sf-7]
    old_rec = 0
    rep_count = 0

    win_start_index = 0



    while True:

        #sys.stdout.flush()
        # start_t = time.time()
        item = sf_index_queue.get()
        #print("Starting decoding on SF",sf)
        c = rec_buffer[win_start_index:win_start_index + sf_windows_len]
        win_start_index = (win_start_index + sf_windows_len) % buf_length
        # print(c.tolist())
        # c.real = np.array(samples_buffer[cursor:cursor+block_size:2 * (1 * float_size) ], dtype = np.float32)
        # c.imag = np.array(samples_buffer[cursor+(1 * float_size) :cursor+block_size:2 * (1 * float_size) ], dtype = np.float32)
        #DECODE THE SAMPLES THROUGH THE DECODE FUNCTION FROM THE LORA MODULE

        (mp.Process(target=thread_decode, args=(np.concatenate((temp_samples,c)), sf, BW, fs, pkt_queue))).start()

        temp_samples = c[-sf_minimum_win_size:]

    return
    print("esco")

def thread_decode(samples, sf, BW, fs, pkt_queue):
    ans = lora.decode(samples, sf, BW, fs)
    if ans.size > 0:
        pkt_queue.put(ans)







#LORA RECEIVER MAIN THREAD#


#THE RECEIVER SCRIPT IS STRUCTURED AS A TWO-PROCESS PROGRAM: ONE PROCESS (THE MAIN THREAD OR RECEIVER THREAD) IS RESPONSIBLE FOR READING AND
#BUFFERING OF RF DATA FROM THE USRP RADIO; THE OTHER PROCESS (THE DECODER THREAD) READS AND PROCESSES DATA FROM THE BUFFER
#THE PROGRAM RESORTS TO A CIRCULAR BUFFER, AND MAKES USE OF A QUEUE TO EXCHANGDE DATA BETWEEN THE PROCESSES
#MORE IN DETAIL, THE BUFFER IS LOCATED IN A SHARED MEMORY AREA. EACH TIME A NEW CHUNK OF DATA IS RECEIVED, THE RECEIVER PUTS THE DATA START INDEX IN THE QUEUE.
#THE PROCESSING THREAD CAN ACCORDINGLY READ AND, POSSIBLY, DECODE THE LORA DATA IN THE CHUNK.


#MAXIMUM NUMBER OF DATA CHUNKS IN THE SHARED MEMORY BUFFER
BUF_SIZE = 120 #MAKE SURE THIS NUMBER IS A MULTIPLE OF THE MAXIMUM SAMPLES THRESHOLD DIVIDED BY COMPLEX_DATA_NUMBER
#FOR INSTANCE, WE NOW HAVE 72 MS FOR SF 12, AND 3M AS COMPLEX_DATA_NUMBER. 72/3 = 24, AND 120 IS INDEED A MULTIPLE OF 24

#BYTES PER COMPLEX SAMPLES
data_size = 8  # bytes

#NUMBER OF COMPLEX SAMPLES IN A DATA CHUNK
#complex_data_number = 500000

#THE MINIMUM BLOCK OF SAMPLES ROUGHLY CORRESPONDS TO 5 TIMES THE SIZE OF A 251 BYTES SF7 PACKET
complex_data_number = 3000000
#SLIDING WINDOW CURSOR FOR THE RECEIVER BUFFER
cursor = 0


sample_rate = 1e6 # Hz
bandwidth = 125000 #Hz



#SIZE, IN BYTES, OF A DATA CHUNK
block_size = complex_data_number * data_size
#samples_buffer = np.zeros(, dtype=np.complex64)

#PROCESS UTILITIES
#complex_data_number
#SF_sample_th = np.array([3e6,6e6,12e6,18e6,42e6,72e6])


#THE BIGGEST THRESHOLD IS THE LEAST COMMON MULTIPLE OF ALL THE NUMBERS IN THE ARRAY, TO IMPROVE WINDOWED DECODING
SF_sample_th = np.array([1, 2, 4, 6, 12, 24])
sf_arr = np.array([7, 8, 9, 10, 11, 12])
queue_arr = np.empty(shape=(sf_arr.size,), dtype=mp_queues.Queue)
pkt_arr = np.empty(shape=(sf_arr.size,), dtype=mp_queues.Queue)
processes_arr = np.empty(shape=(sf_arr.size,), dtype=mp.Process)
pkt_processes_arr = np.empty(shape=(sf_arr.size,), dtype=mp.Process)




#CREATION OF THE SHARED MEMORY AREA
shm = mp.shared_memory.SharedMemory(create = True, size = block_size * BUF_SIZE)
samples_buffer = np.ndarray(complex_data_number * BUF_SIZE, dtype=np.complex64, buffer=shm.buf)
buffer_size = samples_buffer.size
index_queue = mp.Queue(0)


#CREATION OF THE DECODER PROCESSES

for index, sf in enumerate(sf_arr):
    queue_arr[index] = mp.Queue(0)
    pkt_arr[index] = mp.Queue(0)
    processes_arr[index] = mp.Process(target=decoder_process, args=(shm.name, complex_data_number, BUF_SIZE,
                                                                    queue_arr[index], pkt_arr[index], sf, bandwidth,
                                                                    sample_rate, SF_sample_th))
    pkt_processes_arr[index] = mp.Process(target=process_packets, args=(pkt_arr[index], sf))

    #processes_arr[index].daemon = True
    processes_arr[index].start()
    pkt_processes_arr[index].start()
#CREATION OF THE THRESHOLD TRIGGER PROCESS

trigger_process = mp.Process(target=threshold_trigger_process, args=(index_queue, SF_sample_th, tuple(queue_arr)))
trigger_process.daemon = True
trigger_process.start()



# pkt_processing = mp.Process(target=process_packets, args=(pkt_queue,))
# pkt_processing.daemon = True
# pkt_processing.start()

# run_event = threading.Event()
# run_event.set()

#usrp = uhd.usrp.MultiUSRP("serial=322B74F")
usrp = uhd.usrp.MultiUSRP("address=192.168.40.2")


#USRP SETTING AND INITIALIZATION
#num_samps = 100000000 # number of samples received
center_freq = 1e9 # Hz
gain = 10# dB


usrp.set_rx_rate(sample_rate, 0)
usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
usrp.set_rx_gain(gain, 0)
usrp.set_rx_bandwidth(bandwidth,0)
# Set up the stream and receive buffer
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
metadata = uhd.types.RXMetadata()
streamer = usrp.get_rx_stream(st_args)


# Start Stream
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
streamer.issue_stream_cmd(stream_cmd)

rec = True


print("Starting...")


#START RECEIVING
while rec:
    try:
        streamer.recv(samples_buffer[cursor:cursor + complex_data_number], metadata)
        index_queue.put(cursor)
        cursor = (cursor + complex_data_number) % buffer_size
        # print(usrp.get_rx_sensor("rssi"))


    except KeyboardInterrupt:
        print("Exiting")
        rec = False
        # Stop Stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stream_cmd)
        for proc in processes_arr:
            proc.terminate()
        trigger_process.terminate()
        shm.close()
        shm.unlink()
        break




