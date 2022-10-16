import uhd
import lora
import numpy as np
import time


usrp = uhd.usrp.MultiUSRP("serial=322B754")
#usrp = uhd.usrp.MultiUSRP("address=192.168.40.2")


payload = np.array([100,200],dtype=np.uint8)
bw = 125000
sample_rate = 1e6
sf = 7
center_freq = 1e9
gain = 70 # [dB] start low then work your way up

usrp.set_tx_rate(sample_rate, 0)
usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
usrp.set_tx_gain(gain, 0)

# Set up the stream and receive buffer
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
metadata = uhd.types.TXMetadata()
streamer = usrp.get_tx_stream(st_args)
buffer_samps = streamer.get_max_num_samps()




for seqn in range(255):
    samples = lora.encode(center_freq, sf, bw, payload, sample_rate, 0, 1, np.uint8(seqn % 256), 1, 1, 0, 8)
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

    # Send EOB to terminate Tx
    metadata.end_of_burst = True
    streamer.send(np.zeros((1, 1), dtype=np.complex64), metadata)

    print("Sent packet with seq number", seqn)
    time.sleep(2)




