# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2024 SDR-LoRa contributors.
#
# Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
# See LICENSE for the full text.

import numpy as np
import lora






NACK_CODE = 255
POLLING_CODE = 254
POLLING_CODE_BROADCAST = 252
ACK_CODE = 253
BROADCAST_ID = 255
MAX_PACKET_SIZE = 251






def rate_calculator(sf, bw, cr):
    """Nominal bit rate in bit/s for a spreading factor, bandwidth and CR.

    sf must be 5..12, cr 1..4 and bw finite and positive (Hz); anything
    else raises ValueError naming the argument.

    The power of two uses a float base (`2.0`). Under NEP 50
    `np.power(2, np.uint8(8))` stays uint8 and wraps to zero, and
    lora.decode() stores the SF of every LoRaPacket as np.uint8, so an
    integer base would give `inf` from SF8 up.
    """
    # Range-checked by name, as the PHY's own helpers do.
    sf = lora._whole(sf, "sf")
    cr = lora._whole(cr, "cr")
    bw = lora._real(bw, "bw")
    if not 5 <= sf <= 12:
        raise ValueError("rate_calculator: sf must be 5..12, got %r" % (sf,))
    if not 1 <= cr <= 4:
        raise ValueError("rate_calculator: cr must be 1..4, got %r" % (cr,))
    if not (np.isfinite(bw) and bw > 0):
        raise ValueError("rate_calculator: bw must be finite and positive, "
                         "got %r" % (bw,))
    return sf * (4 / (4 + cr)) * (bw / (1 * np.power(2.0, sf)))


def gen_pack_polling(SF, BW, srcID, dstID, CR = 1, brdcst = False):
    payload = np.zeros((1,), dtype=np.uint8)
    if(brdcst):
        payload[0] = POLLING_CODE_BROADCAST
    else:
        payload[0] = POLLING_CODE
    return lora.LoRaPacket(payload, srcID, dstID, seqn= 0, hdr_ok=1, has_crc=1, crc_ok=1,
                    cr=CR, ih=0, SF=SF, BW=BW)

def _check_packet_size(who, packet_size):
    """Validate packet_size as a whole number in 2..MAX_PACKET_SIZE.

    Returns it as an int; raises ValueError otherwise. Below 2 there is
    no room for payload next to the marker or group byte. The upper bound
    is the longest payload a LoRa frame carries, which lora.encode()
    enforces in _check_payload_length.
    """
    # Whole, not truncated: 10.5 is refused rather than read as 10.
    n = lora._whole(packet_size, "%s: packet_size" % who)
    if n < 2 or n > MAX_PACKET_SIZE:
        raise ValueError(
            "%s: packet_size must be 2..%d, got %r. One byte carries the "
            "ACK/NACK marker, so a size of 1 leaves no room for payload."
            % (who, MAX_PACKET_SIZE, packet_size))
    # Callers use the returned int, so 10.0 works as a slice bound.
    return n


# extended_sqn numbers packets with a one-byte group and the one-byte seqn,
# so it can tell 256 * 256 packets apart and no more.
EXTENDED_SQN_MAX_PACKETS = 256 * 256


def pack_lora_data(data, SF, BW, packet_size, srcID, dstID, extended_sqn=True, CR = 1):
    packet_size = _check_packet_size("pack_lora_data", packet_size)
    if (extended_sqn):
        act_pkt_size = packet_size - 1
        pkt_group = -1

    else:
        act_pkt_size = packet_size
    data_bytes = data.view(dtype=np.uint8)
    n_packets = int(np.ceil(data_bytes.size / act_pkt_size))
    if extended_sqn and n_packets > EXTENDED_SQN_MAX_PACKETS:
        # Refused up front, before any packet is built: the uint8 group
        # byte would overflow at packet 65537.
        raise ValueError(
            "pack_lora_data: %d bytes need %d packets of %d, and extended_sqn "
            "numbers at most %d (a one-byte group and the one-byte seqn). "
            "Split the data or raise packet_size."
            % (data_bytes.size, n_packets, packet_size,
               EXTENDED_SQN_MAX_PACKETS))
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
            # A copy, not a view: a numpy slice aliases the caller's
            # buffer, and reusing that buffer would change payloads
            # already queued for transmission. The extended_sqn branch
            # allocates too, so both branches behave alike.
            payload = data_bytes[start:start + act_pkt_size].copy()

        pack_array[index] = lora.LoRaPacket(payload, srcID, dstID, seqn=(index) % 256, hdr_ok=1, has_crc=1, crc_ok=1,
                                            cr=CR, ih=0, SF=SF, BW=BW)
        start = start + act_pkt_size

    return pack_array


def pack_lora_nack(data, SF, BW, packet_size, srcID, dstID, CR = 1):

    packet_size = _check_packet_size("pack_lora_nack", packet_size)
    act_pkt_size = packet_size - 1

    # Sized in bytes, not elements, as pack_lora_data is: a NACK is
    # normally built from the uint16 array unpack_lora_ack returns.
    _arr = np.asarray(data)
    # The byte sizing only means what the caller intends if the caller
    # chose the width. A python list of ints becomes int64, and
    # unpack_lora_ack always views its result as uint16, so each number
    # would come back as four. Refused rather than coerced: a cast to
    # uint16 is right for sequence numbers and wrong for anything else,
    # and the function cannot tell which it has. Any unsigned width
    # (uint8, uint16, uint32) is accepted.
    if _arr.dtype.kind != "u":
        raise ValueError(
            "pack_lora_nack: data must be an unsigned fixed-width array "
            "(uint8/uint16/uint32), got dtype %s. The packets are sized in "
            "BYTES, so the element width decides how many there are, and a "
            "python list becomes %s -- eight bytes per sequence number, of "
            "which six are zero. unpack_lora_ack views its result as "
            "uint16, so pass the array it returns, or "
            "np.asarray(data, dtype=np.uint16)."
            % (_arr.dtype, np.asarray([1]).dtype))
    data_bytes = _arr.view(dtype=np.uint8)
    if(data_bytes.size == 0):
        n_packets = 1
    else:
        n_packets = int(np.ceil(data_bytes.size / act_pkt_size))
    pack_array = np.empty(shape=(n_packets,), dtype=lora.LoRaPacket)
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
                                            cr=CR, ih=0, SF=SF, BW=BW)
        start = start + act_pkt_size



    return pack_array


def pack16bit(high_byte,low_byte):
    high_byte = np.uint8(high_byte)
    low_byte = np.uint8(low_byte)
    temp_arr = np.array(([low_byte, high_byte])).view(dtype = np.uint16)
    return temp_arr[0]



def unpack_lora_data(pkt_array, arr_type = np.uint8, extended_sqn = True):
    array_size = 0
    array_index = 0
    for pkt in pkt_array:
        array_size = array_size + pkt.payload.size
    if extended_sqn:
        array_size = array_size - pkt_array.size


    data_array = np.zeros((array_size,), dtype = np.uint8)

    for pkt in pkt_array:
        if extended_sqn:
            data_array[array_index : array_index + pkt.payload.size - 1] = pkt.payload[1:]
            array_index = array_index + pkt.payload.size - 1
        else:
            data_array[array_index:array_index + pkt.payload.size] = pkt.payload
            array_index = array_index + pkt.payload.size

    # The dtype view is applied once, after every packet is copied in;
    # viewing inside the loop would shrink data_array under later writes.
    if not (arr_type == np.uint8):
        # Checked, not trimmed. unpack_lora_ack drops a trailing byte to
        # make its view fit, because a NACK list that does not divide is
        # a truncated control message. Here the bytes are user data, so a
        # remainder means a missing packet or a wrong arr_type, and is
        # reported as such instead of discarding payload.
        _w = int(np.dtype(arr_type).itemsize)
        if _w > 1 and data_array.size % _w:
            raise ValueError(
                "unpack_lora_data: %d payload bytes reassembled from %d "
                "packet(s) do not divide into %s (%d bytes each), leaving "
                "%d over. Either a packet is missing from pkt_array or "
                "arr_type does not match what pack_lora_data was given -- "
                "the bytes are not trimmed to fit, because that would "
                "discard payload without saying so."
                % (data_array.size, len(pkt_array),
                   np.dtype(arr_type).name, _w, data_array.size % _w))
        data_array = data_array.view(dtype=arr_type)

    return data_array


def unpack_lora_ack(acks_array):
    # Sized from the payloads themselves, so the result does not depend on
    # the packet_size the NACK was built with.
    _total = sum(max(int(np.asarray(p.payload).size) - 1, 0) for p in acks_array)
    missing_seqn = np.zeros((_total,), dtype= np.uint8)
    index = 0
    # De-duplicated by position, not by seqn. pack_lora_nack assigns
    # seqn = index % 256, so seqn repeats past the 256th packet. The
    # duplicate this guards against is a retransmission, which arrives
    # adjacent to its original, so comparing with the previous packet
    # catches it without a ceiling.
    prev_seqn = None
    for pack in (acks_array):
        if prev_seqn is not None and pack.seqn == prev_seqn:
            continue
        prev_seqn = pack.seqn
        pld_size = pack.payload.size
        # A packet carrying only the marker byte contributes nothing; a
        # NACK with no sequence numbers is not the end of the list.
        missing_seqn[index: index + pld_size - 1] = pack.payload[1:]
        index = index + pld_size - 1
        # The end of the list is the marker, not the length: pack_lora_nack
        # writes ACK_CODE into payload[0] of the last packet and NACK_CODE
        # into the others, whatever packet_size it was given.
        if int(pack.payload[0]) == ACK_CODE:
            break
    missing_seqn = missing_seqn[:index]
    try:
        missing_seqn = missing_seqn.view(dtype = np.uint16)
    except ValueError:
        missing_seqn = missing_seqn[:-1].view(dtype=np.uint16)



    return missing_seqn
