
import numpy as np
import lora






NACK_CODE = 255
POLLING_CODE = 254
ACK_CODE = 253
BROADCAST_ID = 255





def gen_pack_polling(SF, BW, srcID, dstID):
    payload = np.zeros((1,), dtype=np.uint8)
    payload[0] = POLLING_CODE
    return lora.LoRaPacket(payload, srcID, dstID, seqn= 0, hdr_ok=1, has_crc=1, crc_ok=1,
                    cr=1, ih=0, SF=SF, BW=BW)

def pack_lora_data(data, SF, BW, packet_size, srcID, dstID, extended_sqn=True):
    if (extended_sqn):
        act_pkt_size = packet_size - 1
        pkt_group = -1

    else:
        act_pkt_size = packet_size
    data_bytes = data.view(dtype=np.uint8)
    n_packets = int(np.ceil(data_bytes.size / act_pkt_size))
    pack_array = np.empty(shape=(n_packets,), dtype=lora.LoRaPacket)

    start = 0
    for index in range(n_packets):
        if (extended_sqn):
            chunk = data_bytes[start:start + act_pkt_size]
            payload = np.zeros((chunk.size + 1,), dtype=np.uint8)
            if (index % 256 == 0):
                pkt_group = pkt_group + 1
            payload[0] = pkt_group
            payload[1:chunk.size + 1] = chunk
        else:
            payload = data_bytes[start:start + act_pkt_size]

        pack_array[index] = lora.LoRaPacket(payload, srcID, dstID, seqn=(index) % 256, hdr_ok=1, has_crc=1, crc_ok=1,
                                            cr=1, ih=0, SF=SF, BW=BW)
        start = start + act_pkt_size

    return pack_array


def pack_lora_nack(data, SF, BW, packet_size, srcID, dstID):

    act_pkt_size = packet_size - 1

    if(data.size == 0):
        n_packets = 1
    else:
        n_packets = int(np.ceil(data.size / act_pkt_size))
    pack_array = np.empty(shape=(n_packets,), dtype=lora.LoRaPacket)
    data_bytes = data.view(dtype=np.uint8)
    start = 0
    for index in range(n_packets):
        chunk = data_bytes[start:start + act_pkt_size]
        payload = np.zeros((chunk.size + 1,), dtype=np.uint8)

        if (index == n_packets - 1):
            payload[0] = ACK_CODE
        else:
            payload[0] = NACK_CODE

        payload[1:chunk.size + 1] = chunk


        pack_array[index] = lora.LoRaPacket(payload, srcID, dstID, seqn=(index) % 256, hdr_ok=1, has_crc=1, crc_ok=1,
                                            cr=1, ih=0, SF=SF, BW=BW)
        start = start + act_pkt_size

    return pack_array


def pack16bit(high_byte,low_byte):
    high_byte = np.uint8(high_byte)
    low_byte = np.uint8(low_byte)
    temp_arr = np.array(([low_byte, high_byte])).view(dtype = np.uint16)
    return temp_arr[0]



def unpack_lora_data(pkt_array, extended_sqn = True):
    array_size = 0
    array_index = 0
    for pkt in pkt_array:
        array_size = array_size + pkt.payload.size
    if extended_sqn:
        array_size = array_size - pkt_array.size


    #print("Arr size",array_size)
    data_array = np.zeros((array_size,), dtype = np.uint8)

    for pkt in pkt_array:
        if extended_sqn:
            data_array[array_index : array_index + pkt.payload.size - 1] = pkt.payload[1:]
            array_index = array_index + pkt.payload.size - 1
        else:
            data_array[array_index:array_index + pkt.payload.size] = pkt.payload
            array_index = array_index + pkt.payload.size

    return data_array


def unpack_lora_ack(acks_array):
    missing_seqn = np.zeros((250 * len(acks_array),), dtype= np.uint8)
    index = 0
    for pack in (acks_array):
        pld_size = pack.payload.size
        if (pack.payload[0] == 255 and pld_size == 1):
            break
        missing_seqn[index: index + pld_size - 1] = pack.payload[1:]
        index = index + pld_size - 1

    missing_seqn = missing_seqn[:index]
    missing_seqn = missing_seqn.view(dtype = np.uint16)
    return missing_seqn