import time

import scipy.io
import numpy as np


#############CONSTANTS#####################
# rising and falling edges duration
Trise = 50e-6  
# Carrier Frequency Offset as declared in the datasheet
Cfo_PPM = 17
# Receiver Noise Figure
NF_dB = 6
#Minimum duration for a LoRa packet
min_time_lora_packet = 20e-3 #20 milliseconds


##########################################################





def lora_packet(BW, OSF, SF, k1, k2, n_pr, IH, CR, MAC_CRC, SRC, DST, SEQNO, MESSAGE, Trise, t0_frac, phi0):
    # in our implementation, the payload includes a 4 byte pseudo-header: (SRC, DST, SEQNO, LENGTH)
    LENGTH = np.uint16(4 + (MESSAGE.size))

    # n_sym_hdr: number of chirps encoding the header (0 if IH: True)
    # n_bits_hdr: [bit DI PAYLOAD presenti nell'header CHIEDERE A STEFANO]
    (n_sym_hdr, n_bits_hdr) = lora_header_init(SF, IH)
    [PAYLOAD, n_sym_payload] = lora_payload_init(SF, LENGTH, MAC_CRC, CR, n_bits_hdr, DST, SRC, SEQNO, MESSAGE)

    # -------------------------------------------------BIT TO SYMBOL MAPPING
    payload_ofs = 0
    if IH:
        k_hdr = []
    else:
        [k_hdr, payload_ofs] = lora_header(SF, LENGTH, CR, MAC_CRC, PAYLOAD, payload_ofs)

    k_payload = lora_payload(SF, CR, n_sym_payload, PAYLOAD, payload_ofs)

    # --------------------------------- CSS MODULATION

    # number of samples per chirp
    K = np.power(2, SF)
    N = int(K * OSF)
    # number of samples in the rising/falling edge
    Nrise = int(np.ceil(Trise * BW * OSF))

    # preamble modulation
    (p, phi) = lora_preamble(n_pr, k1, k2, BW, K, OSF, t0_frac, phi0)

    # samples initialization
    s = np.concatenate((np.zeros(Nrise), p, np.zeros(N * (n_sym_hdr + n_sym_payload) + Nrise)))

    # rising edge samples
    s[0:Nrise] = p[N - 1 + np.arange(-Nrise + 1, 1, dtype=np.int16)] * np.power(
        np.sin(np.pi / 2 * np.arange(1, Nrise + 1) / Nrise), 2)
    s_ofs = p.size + Nrise

    # header modulation, if any
    for sym in range(0, n_sym_hdr):
        k = k_hdr[sym]
        (s[s_ofs + np.arange(0, N)], phi) = lora_chirp(+1, k, BW, K, OSF, t0_frac, phi)
        s_ofs = s_ofs + N

    # payload modulation
    for sym in range(0, n_sym_payload):
        k = k_payload[sym]
        (s[s_ofs + np.arange(0, N)], phi) = lora_chirp(+1, k, BW, K, OSF, t0_frac, phi)
        s_ofs = s_ofs + N

    # falling edge samples
    s[s_ofs + np.arange(0, Nrise)] = s[s_ofs + np.arange(0, Nrise)] * np.power(
        np.cos(np.pi / 2 * np.arange(0, Nrise) / Nrise), 2)

    return s, k_hdr, k_payload



def lora_header_init(SF, IH):
    if (IH):
        n_sym_hdr = np.uint16(0)
        n_bits_hdr = np.uint16(0)
    else:
        # interleaving block size, respectively for header & payload
        CR_hdr = np.uint16(4)
        DE_hdr = np.uint16(1)
        n_sym_hdr = 4 + CR_hdr
        intlv_hdr_size = (SF - 2 * DE_hdr) * (4 + CR_hdr)
        n_bits_hdr = np.uint16(intlv_hdr_size * 4 / (4 + CR_hdr) - 20)

    return n_sym_hdr, n_bits_hdr


# function [k_hdr,payload_ofs] =
def lora_header(SF, LENGTH, CR, MAC_CRC, PAYLOAD, payload_ofs):
    # header parity check matrix (reverse engineered through brute force search)

    header_FCS = np.array(([1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 1, 0, 0],
                           [0, 1, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1],
                           [0, 0, 1, 1, 1],
                           [0, 1, 0, 1, 1]))

    CR_hdr = np.uint8(4)
    Hamming_hdr = np.array(([1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]), dtype=np.uint8)
    n_sym_hdr = 4 + CR_hdr
    DE_hdr = np.uint8(1)
    PPM = SF - 2 * DE_hdr
    gray_hdr = gray_lut(PPM)[0]
    intlv_hdr_size = PPM * n_sym_hdr

    # header (20 bit)
    LENGTH_bits = num2binary(LENGTH, 8)
    CR_bits = num2binary(CR, 3)
    hdr = np.concatenate((LENGTH_bits[np.arange(3, -1, -1, dtype=np.uint8)],
                          LENGTH_bits[np.arange(7, 3, -1, dtype=np.uint8)], np.array([MAC_CRC], dtype=np.uint8),
                          CR_bits[np.arange(2, -1, -1, dtype=np.uint8)], np.zeros(8, dtype=np.uint8)))
    hdr_chk_indexes = np.concatenate((np.arange(3, -1, -1, dtype=np.uint8), np.arange(7, 3, -1, dtype=np.uint8),
                                      np.arange(11, 7, -1, dtype=np.uint8)))

    hdr_chk = np.mod(hdr[hdr_chk_indexes] @ header_FCS, 2)
    hdr[12] = hdr_chk[0]
    hdr[16:20] = hdr_chk[4:-0:-1]

    # parity bit calculation
    C = np.zeros((PPM, 4 + CR_hdr))
    for k in range(0, 5):
        C[k, 0:4] = hdr[k * 4 + np.arange(0, 4)]
        C[k, 3 + np.arange(1, CR_hdr + 1)] = np.mod(C[k, 0:4] @ Hamming_hdr, 2)

    for k in range(5, PPM):
        C[k, 0:4] = PAYLOAD[payload_ofs + np.arange(0, 4, dtype=np.uint8)]
        payload_ofs = payload_ofs + 4
        C[k, 3 + np.arange(1, CR_hdr + 1)] = np.mod(C[k, 0:4] @ Hamming_hdr, 2)

    # rows flip
    C = np.flip(C, 0)

    S = np.zeros((4 + CR_hdr, PPM), dtype=np.uint8)
    for ii in range(0, PPM):
        for jj in range(0, 4 + CR_hdr):
            S[jj, np.mod(ii + jj, PPM)] = C[ii, jj]

    bits_hdr = np.reshape(S.transpose(), intlv_hdr_size, order='F')

    # bit to symbol mapping
    k_hdr = np.zeros(n_sym_hdr)
    K = np.power(2, SF)
    for sym in range(0, n_sym_hdr):
        k_hdr[sym] = K - 1 - np.power(2, (2 * DE_hdr)) * gray_hdr[
            bits_hdr[sym * PPM + np.arange(0, PPM, dtype=np.uint16)] @ np.power(2, np.arange(PPM - 1, -1, -1))]

    return k_hdr, payload_ofs



def lora_payload_init(SF, LENGTH, MAC_CRC, CR, n_bits_hdr, DST, SRC, SEQNO, MESSAGE):
    # bigger spreading factors (11 and 12) use 2 less bits per symbol
    if SF > 10:
        DE = np.uint8(1)
    else:
        DE = np.uint8(0)
    PPM = SF - 2 * DE
    n_bits_blk = PPM * 4
    n_bits_tot = 8 * LENGTH + 16 * MAC_CRC
    n_blk_tot = int(np.ceil((n_bits_tot - n_bits_hdr) / n_bits_blk))
    n_sym_blk = 4 + CR
    n_sym_payload = n_blk_tot * n_sym_blk

    byte_range = np.arange(7, -1, -1, dtype=np.uint8)
    PAYLOAD = np.zeros(int(n_bits_hdr + n_blk_tot * n_bits_blk), dtype=np.uint8)
    PAYLOAD[byte_range] = num2binary(DST, 8)
    PAYLOAD[8 + byte_range] = num2binary(SRC, 8)
    PAYLOAD[8 * 2 + byte_range] = num2binary(SEQNO, 8)
    PAYLOAD[8 * 3 + byte_range] = num2binary(LENGTH, 8)
    for k in range(0, MESSAGE.size):
        PAYLOAD[8 * (4 + k) + byte_range] = num2binary(MESSAGE[k], 8)

    if MAC_CRC:
        PAYLOAD[8 * LENGTH + np.arange(0, 16, dtype=np.uint8)] = CRC16(PAYLOAD[0:8 * LENGTH])

    # ----------------------------------------------------------- WHITENING
    W = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    W_fb = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
    for k in range(1, int(np.floor(len(PAYLOAD) / 8) + 1)):
        PAYLOAD[(k - 1) * 8 + np.arange(0, 8, dtype=np.int32)] = np.mod(
            PAYLOAD[(k - 1) * 8 + np.arange(0, 8, dtype=np.int32)] + W, 2)
        W1 = np.array([np.mod(np.sum(W * W_fb), 2)])
        W = np.concatenate((W1, W[0:-1]))

    return PAYLOAD, n_sym_payload


#CRC-16 calculation for LoRa (reverse engineered from a Libelium board)
def CRC16(bits):
    length = int(len(bits) / 8)
    # initial states for crc16 calculation, valid for lenghts in the range 5,255
    state_vec = np.array([46885, 27367, 35014, 54790, 18706, 15954, \
                          9784, 59350, 12042, 22321, 46211, 20984, 56450, 7998, 62433, 35799, \
                          2946, 47628, 30930, 52144, 59061, 10600, 56648, 10316, 34962, 55618, \
                          57666, 2088, 61160, 25930, 63354, 24012, 29658, 17909, 41022, 17072, \
                          42448, 5722, 10472, 56651, 40183, 19835, 21851, 13020, 35306, 42553, \
                          12394, 57960, 8434, 25101, 63814, 29049, 27264, 213, 13764, 11996, \
                          46026, 6259, 8758, 22513, 43163, 38423, 62727, 60460, 29548, 18211, \
                          6559, 61900, 55362, 46606, 19928, 6028, 35232, 29422, 28379, 55218, \
                          38956, 12132, 49339, 47243, 39300, 53336, 29575, 53957, 5941, 63650, \
                          9502, 28329, 44510, 28068, 19538, 19577, 36943, 59968, 41464, 33923, \
                          54504, 49962, 64357, 12382, 44678, 11234, 58436, 47434, 63636, 51152, \
                          29296, 61176, 33231, 32706, 27862, 11005, 41129, 38527, 32824, 20579, \
                          37742, 22493, 37464, 56698, 29428, 27269, 7035, 27911, 55897, 50485, \
                          10543, 38817, 54183, 52989, 24549, 33562, 8963, 38328, 13330, 24139, \
                          5996, 8270, 49703, 60444, 8277, 43598, 1693, 60789, 32523, 36522, \
                          17339, 33912, 23978, 55777, 34725, 2990, 13722, 60616, 61229, 19060, \
                          58889, 43920, 9043, 10131, 26896, 8918, 64347, 42307, 42863, 7853, \
                          4844, 60762, 21736, 62423, 53096, 19242, 55756, 26615, 53246, 11257, \
                          2844, 47011, 10022, 13541, 18296, 44005, 23544, 18733, 23770, 33147, \
                          5237, 45754, 4432, 22560, 40752, 50620, 32260, 2407, 26470, 2423, \
                          33831, 34260, 1057, 552, 56487, 62909, 4753, 7924, 40021, 7849, \
                          4895, 10401, 32039, 40207, 63952, 10156, 53647, 51938, 16861, 46769, \
                          7703, 9288, 33345, 16184, 56808, 30265, 10696, 4218, 7708, 32139, \
                          34174, 32428, 20665, 3869, 43003, 6609, 60431, 22531, 11704, 63584, \
                          13620, 14292, 37000, 8503, 38414, 38738, 10517, 48783, 30506, 63444, \
                          50520, 34666, 341, 34793, 2623], dtype=np.uint16)
    crc_tmp = num2binary(state_vec[length - 5], 16)
    # crc_poly = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1]
    # 	for j = 1:numel(bits)/8
    # 		for k = 1:8
    # 			add = crc_tmp(1)
    # 			crc_tmp = [crc_tmp(2: ),bits((j-1)*8+9-k)]
    # 			if add
    # 				crc_tmp = mod(crc_tmp+crc_poly(2: ),2)
    #
    #
    #
    # 	CRC=crc_tmp(16:-1:1)
    pos = 0
    pos4 = 4
    pos11 = 11
    for j in range(0, length):
        for k in range(0, 8):
            add = crc_tmp[pos]
            crc_tmp[pos] = bits[j * 8 + 7 - k]
            if add:
                crc_tmp[pos4] = 1 - crc_tmp[pos4]
                crc_tmp[pos11] = 1 - crc_tmp[pos11]
                crc_tmp[pos] = 1 - crc_tmp[pos]

            pos = np.mod(pos + 1, 16)
            pos4 = np.mod(pos4 + 1, 16)
            pos11 = np.mod(pos11 + 1, 16)

    CRC = crc_tmp[np.mod(pos + np.arange(15, -1, -1, dtype=np.int32), 16)]
    return CRC


# gray and reversed-gray mappings

def gray_lut(n):
    pow_n = np.power(2, n)
    g = np.zeros(pow_n, dtype=np.uint16)

    vec = np.atleast_2d(np.arange(0, pow_n, dtype=np.uint16)).transpose()
    vec = np.flip((vec.view(np.uint8)), 1)
    vec = np.unpackbits(vec).reshape(pow_n, 16)
    vec = vec[:, -n:]
    vec = vec.transpose()
    support_x = np.zeros((n, pow_n))
    support_x[1:, :] = vec[:-1, :]

    ig = np.matmul(np.power(2, np.arange(n - 1, -1, -1)), np.mod(vec + support_x, 2))
    ig = ig.astype(int)
    g[ig] = np.arange(0, pow_n, dtype=np.uint16)
    return g, ig


# function k_payload = \
def lora_payload(SF, CR, n_sym_payload, PAYLOAD, payload_ofs):
    # varargout = {DST, SRC, SEQNO, MESSAGE}

    # hamming parity check matrices 
    Hamming_P1 = np.array(([1], [1], [1], [1]), dtype=np.uint8)
    Hamming_P2 = np.array(([1, 0], [1, 1], [1, 1], [0, 1]), dtype=np.uint8)
    Hamming_P3 = np.array(([1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]), dtype=np.uint8)
    Hamming_P4 = np.array(([1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]), dtype=np.uint8)

    if CR == 1:
        Hamming = Hamming_P1
    elif CR == 2:
        Hamming = Hamming_P2
    elif CR == 3:
        Hamming = Hamming_P3
    elif CR == 4:
        Hamming = Hamming_P4

    if SF > 10:
        DE = 1
    else:
        DE = 0

    PPM = SF - 2 * DE
    n_sym_blk = (4 + CR)
    intlv_blk_size = PPM * n_sym_blk
    gray = gray_lut(PPM)[0]
    K = np.power(2, SF)
    n_blk_tot = int(n_sym_payload / n_sym_blk)
    C = np.zeros((PPM, 4 + CR))
    S = np.zeros((4 + CR, PPM))
    k_payload = np.zeros(n_sym_payload)
    for blk in range(0, n_blk_tot):
        for k in range(0, PPM):
            C[k, 0:4] = PAYLOAD[payload_ofs + np.arange(0, 4, dtype=np.uint8)]
            payload_ofs = payload_ofs + 4
            C[k, 4: 4 + CR] = np.mod(C[k, 0:4] @ Hamming, 2)

        # row flip
        C = np.flip(C, 0)

        # interleaving
        for ii in range(0, PPM):
            for jj in range(0, 4 + CR):
                S[jj, np.mod(ii + jj, PPM)] = C[ii, jj]

        bits_blk = np.reshape(S.transpose(), intlv_blk_size, order='F')

        # bit to symbol mapping
        for sym in range(0, n_sym_blk):
            k_payload[(blk) * n_sym_blk + sym] = K - 1 - np.power(2, (2 * DE)) * gray[int(
                bits_blk[(sym * PPM + np.arange(0, PPM, dtype=np.uint16))] @ np.power(2, np.arange(PPM - 1, -1, -1)))]

    return k_payload





def chirp(f_ini,N,Ts,Df,t0_frac = 0,phi_ini = 0):
    t = (t0_frac+np.arange(0,N, dtype = np.int32))*Ts
    T = N*Ts
    s = np.exp(1j*(phi_ini + 2*np.pi*f_ini*t + np.pi*Df*np.power(t,2)))
    phi_fin = phi_ini + 2*np.pi*f_ini*T + np.pi*Df*np.power(T,2)
    return s, phi_fin


def lora_packet_rx(s,SF,BW,OSF,Trise,p_ofs_est,Cfo_est):
    truncated = False
    fs = BW*OSF
    Ts = 1/fs
    N = np.power(2,SF)*OSF
    Nrise = np.ceil(Trise*fs)
    d0 = chirp(BW/2,N,Ts,-BW/(N*Ts))[0]
    DST = -1
    SRC = -1
    SEQNO = -1
    CR = -1
    HAS_CRC = -1
    # demodula i simboli dell'header
    # n_sym_hdr == 8 o 0 in caso di header implicito
    # numero di campioni del preambolo
    ofs = int(Nrise+12.25*N)

    CR_hdr = 4
    n_sym_hdr = 4+CR_hdr
    k_hdr_est = np.zeros((n_sym_hdr))
    for i in range(0,n_sym_hdr):

        try:

            temp = np.exp(-1j * 2 * np.pi * Cfo_est * Ts * (ofs+np.arange(0,N))) * s[p_ofs_est+ofs+np.arange(0,N, dtype = np.int32)] *d0
        except IndexError:
            k_hdr_est = None
            MAC_CRC_OK = False
            HDR_FCS_OK = False
            k_payload_est = None
            MSG = None
            truncated = True
            return k_hdr_est,HDR_FCS_OK,k_payload_est,MAC_CRC_OK,MSG,DST,SRC,SEQNO,CR,HAS_CRC,truncated,ofs


        ofs = ofs+N
        pos = np.argmax(np.abs(np.fft.ifft(temp[0:-1:OSF])))
        k_hdr_est[i]=pos-1


    #in case of header checksum failure, we assume the message to be lost/corrupted [ToDo: implement implicit header mode]
    (HDR_FCS_OK,LENGTH,HAS_CRC,CR,PAYLOAD_bits_hdr) = lora_header_decode(SF,k_hdr_est)
    if not HDR_FCS_OK:
        k_hdr_est = None
        MAC_CRC_OK = False
        k_payload_est = None
        MSG = None

        return k_hdr_est,HDR_FCS_OK,k_payload_est,MAC_CRC_OK,MSG,DST,SRC,SEQNO,CR,HAS_CRC,truncated,ofs
    else:
        n_bits_hdr = (PAYLOAD_bits_hdr.size)
        n_sym_payload = lora_payload_n_sym(SF,LENGTH,HAS_CRC,CR,n_bits_hdr)

        k_payload_est = np.zeros((n_sym_payload))
        for i in range(0, n_sym_payload):
            try:
                temp = np.exp(-1j*2*np.pi*Cfo_est*Ts*(ofs+np.arange(0,N, dtype = np.int32))) * s[p_ofs_est+ofs+np.arange(0,N, dtype = np.int32)] * d0
            except:
                k_hdr_est = None
                MAC_CRC_OK = False
                k_payload_est = None
                MSG = None

                truncated = True
                return k_hdr_est,HDR_FCS_OK,k_payload_est,MAC_CRC_OK,MSG,DST,SRC,SEQNO,CR,HAS_CRC,truncated,ofs

            ofs = ofs+N
            pos = np.argmax(np.abs(np.fft.ifft(temp[1:-1:OSF])))
            k_payload_est[i] = pos-1


        (MAC_CRC_OK,DST,SRC,SEQNO,MSG,HAS_CRC) = lora_payload_decode(SF,k_payload_est,PAYLOAD_bits_hdr,HAS_CRC,CR,LENGTH)
        # if MAC_CRC_OK:
        #
        # 	#fprintf('DST: #d SRC: #d SEQ: #d LEN: #d DATA: "',...
        # 		#DST,SRC,SEQNO,LENGTH)
        # 	for i in range(1,len(MSG)+1):
        # 		if MSG[i-1]>=32 and MSG[i-1]<=127:
        # 			#fprintf('#c',MSG(i))
        # 			pass
        # 		else:
        # 			pass
        # 			#fprintf('\\#d',MSG(i))
        return k_hdr_est, HDR_FCS_OK, k_payload_est, MAC_CRC_OK, MSG, DST,SRC,SEQNO, CR,HAS_CRC, truncated,ofs






def lora_header_decode(SF,k_hdr):

    if SF<7:
        FCS_OK = False
        return






    DE_hdr = 1
    PPM = SF-2*DE_hdr
    (_,degray) = gray_lut(PPM)

    #np.fft(downsample(s_rx.*chirp)) demodulates the signal
    CR_hdr = 4
    n_sym_hdr = 4+CR_hdr
    intlv_hdr_size = PPM*n_sym_hdr
    bits_est = np.zeros((intlv_hdr_size), dtype = np.uint8)
    K = np.power(2,SF)
    for sym in range(0,n_sym_hdr):
        # gray decode
        bin = int(np.round((K-1-k_hdr[sym])/4))
        bits_est[(sym)*PPM+np.arange(0,PPM, dtype = np.int32)] = num2binary(degray[np.mod(bin,np.power(2,PPM))],PPM)


    # interleaver del brevetto
    S = np.reshape(bits_est,(PPM,4+CR_hdr),order = 'F').transpose()
    C = np.zeros((PPM,4+CR_hdr),dtype = np.uint8)
    for ii in range(0,PPM):
        for jj in range (0,4+CR_hdr):
            C[ii,jj] = S[jj,np.mod(ii+jj,PPM)]



    # row flip
    C = np.flip(C,0)



    # header parity check matrix (reverse engineered through brute force search)
    header_FCS = np.array(([1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,0], \
        [0,1,0,1,0],[0,1,0,0,1],[0,0,1,1,0],[0,0,1,0,1],[0,0,0,1,1],[0,0,1,1,1], \
        [0,1,0,1,1]))

    FCS_chk = np.mod(np.concatenate((C[0,np.arange(3,-1,-1, dtype = np.int32)],C[1,np.arange(3,-1,-1)],C[2,np.arange(3,-1,-1, dtype = np.int32)],np.array([C[3,0]],dtype=np.uint8),C[4,np.arange(3,-1,-1, dtype = np.int32)])) @ [np.concatenate((header_FCS,np.eye(5)),0)],2)
    FCS_OK = not np.any(FCS_chk)
    if not FCS_OK:
        LENGTH = -1
        HAS_CRC = False
        CR = -1
        PAYLOAD_bits = -1
      

    else:
        # header decoding
        # Header=char(['len:',C(1,4:-1:1)+'0',C(2,4:-1:1)+'0',...
        # 	' CR:',C(3,4:-1:2)+'0',...
        # 	' MAC-CRC:',C(3,1)+'0',...
        # 	' HDR-FCS:',C(4,4:-1:1)+'0',C(5,4:-1:1)+'0'])
        #fprintf('Header Decode: #s [OK]\n',Header)


        LENGTH = bit2uint8(np.concatenate((C[1,0:4],C[0,0:4])))
        HAS_CRC = C[2,0] # HAS_CRC
        CR = bit2uint8(C[2,1:4])
        FCS_HDR = bit2uint8(np.concatenate((C[4, 0:4], C[3, 0:4])))

        n_bits_hdr = PPM*4-20
        PAYLOAD_bits=np.zeros((1,n_bits_hdr),dtype=np.uint8)
        for i in range(5,PPM):
            #C(i,4+(1:CR_hdr)) = mod(C(i,1:4)*Hamming_hdr,2)
            PAYLOAD_bits[0,(i-5)*4+np.arange(0,4, dtype = np.int32)]=C[i,0:4]


    return FCS_OK,LENGTH,HAS_CRC,CR,PAYLOAD_bits


def num2binary(num,length = 0):

    num = np.array([num],dtype=np.uint16)
    num = np.flip(num.view(np.uint8))
    num = np.unpackbits(num)
    return num[-length:]



def lora_payload_n_sym(SF,LENGTH,MAC_CRC,CR,n_bits_hdr):

    # bigger spreading factors (11 and 12) use 2 less bits per symbol
    if SF > 10:
        DE = 1

    else:
        DE = 0
    PPM = SF-2*DE
    n_bits_blk = PPM*4
    n_bits_tot = 8*LENGTH+16*MAC_CRC
    n_blk_tot = np.ceil((n_bits_tot-n_bits_hdr)/n_bits_blk)
    n_sym_blk = 4+CR
    n_sym_payload = int(n_blk_tot*n_sym_blk)
    return n_sym_payload




def lora_payload_decode(SF,k_payload,PAYLOAD_hdr_bits,HAS_CRC,CR,LENGTH_FROM_HDR):


    # hamming parity check matrices 
    Hamming_P1 = np.array(([1],[1],[1],[1]), dtype=np.uint8)
    Hamming_P2 = np.array(([1,0], [1,1], [1,1], [0,1]), dtype=np.uint8)
    Hamming_P3 = np.array(([1,0,1], [1,1,1], [1,1,0], [0,1,1]), dtype=np.uint8)
    Hamming_P4 = np.array(([1,0,1,1], [1,1,1,0], [1,1,0,1], [0,1,1,1]),dtype = np.uint8)

    if CR == 1:
        Hamming = Hamming_P1
    elif CR == 2:
        Hamming = Hamming_P2
    elif CR == 3:
        Hamming = Hamming_P3
    elif CR == 4:
        Hamming = Hamming_P4

    if SF > 10:
        DE = 1
    else:
        DE = 0
    PPM = SF-2*DE
    n_sym_blk = (4+CR)
    n_bits_blk = PPM*4
    intlv_blk_size = PPM*n_sym_blk
    [_,degray] = gray_lut(PPM)
    K = np.power(2,SF)
    n_sym_payload = len(k_payload)
    n_blk_tot = int(n_sym_payload/n_sym_blk)
    try:
        PAYLOAD = np.concatenate((np.squeeze(PAYLOAD_hdr_bits),np.zeros((int(n_bits_blk*n_blk_tot)), dtype = np.uint8)))
    except ValueError:
        PAYLOAD = np.zeros((int(n_bits_blk*n_blk_tot)), dtype = np.uint8)
    payload_ofs = (PAYLOAD_hdr_bits.size)
    for blk in range (0,n_blk_tot):

        bits_blk = np.zeros((intlv_blk_size))
        for sym in range(0,n_sym_blk):
            bin = round((K-2-k_payload[(blk)*n_sym_blk+sym])/np.power(2,(2*DE)))
            # gray decode
            bits_blk[(sym)*PPM+np.arange(0,PPM, dtype = np.int32)] = num2binary(degray[int(np.mod(bin,np.power(2,PPM)))],PPM)


        # interleaving



        S = np.reshape(bits_blk,(PPM, (4+CR)),order = 'F').transpose()
        C = np.zeros((PPM,4+CR),dtype=np.uint8)
        for ii in range(0,PPM):
            for jj in range(0,4+CR):
                C[ii,jj]= S[jj,np.mod(ii+jj,PPM)]


        # row flip
        C = np.flip(C,0)



        for k in range(0,PPM):
            PAYLOAD[payload_ofs+np.arange(0,4, dtype = np.int32)] = C[k,0:4]
            payload_ofs = payload_ofs+4



    # ----------------------------------------------------------- WHITENING
    W = np.array([1,1,1,1,1,1,1,1], dtype=np.uint8)
    W_fb = np.array([0,0,0,1,1,1,0,1], dtype = np.uint8)
    for k in range (1,int(np.floor(len(PAYLOAD)/8) + 1)):
        PAYLOAD[(k-1)*8+np.arange(0,8, dtype = np.int32)] = np.mod(PAYLOAD[(k-1)*8+np.arange(0,8, dtype = np.int32)]+W,2)
        W1 = np.array([np.mod(np.sum(W*W_fb),2)])
        W = np.concatenate((W1,W[0:-1]))




    #NOTE HOW THE TOTAL LENGTH IS 4 BYTES + THE PAYLOAD LENGTH
    #INDEED, THE FIRST 4 BYTES ENCODE DST, SRC, SEQNO AND LENGTH INFOS
    DST = bit2uint8(PAYLOAD[0:8])
    SRC = bit2uint8(PAYLOAD[8+np.arange(0,8, dtype = np.int32)])
    SEQNO = bit2uint8(PAYLOAD[8*2+np.arange(0,8, dtype = np.int32)])
    LENGTH = bit2uint8(PAYLOAD[8*3+np.arange(0,8, dtype = np.int32)])
    if (LENGTH == 0):
    	LENGTH = LENGTH_FROM_HDR
    	
    
    MSG_LENGTH = LENGTH-4
    if (((LENGTH+2)*8 > len(PAYLOAD) and HAS_CRC) or (LENGTH*8 > len(PAYLOAD) and not (HAS_CRC)) or LENGTH<4):
        MAC_CRC_OK = False

        return MAC_CRC_OK, DST, SRC, SEQNO, None, HAS_CRC

    MSG=np.zeros((int(MSG_LENGTH)), dtype = np.uint8)
    for i in range (0,int(MSG_LENGTH)):
        MSG[i]=bit2uint8(PAYLOAD[8*(4+i)+np.arange(0,8, dtype = np.int32)])

    if not HAS_CRC:
        MAC_CRC_OK = True
    else:
        #fprintf('CRC-16: 0x#02X#02X ',...
            #PAYLOAD(8*LENGTH+(1:8))*2.^(0:7)',...
            #PAYLOAD(8*LENGTH+8+(1:8))*2.^(0:7)')
        temp = CRC16(PAYLOAD[0:LENGTH*8])
        temp = np.power(2,np.arange(0,8, dtype = np.int32)) @ (np.reshape(temp,(8,2),order = 'F'))
        #fprintf('(CRC-16: 0x#02X#02X)',temp(1),temp(2))

        if np.any(PAYLOAD[8*LENGTH+np.arange(0,16, dtype = np.int32)] != CRC16(PAYLOAD[0:8*LENGTH])):
            #fprintf(' [CRC FAIL]\n')
            MAC_CRC_OK = False
        else:
            #fprintf(' [CRC OK]\n')
            MAC_CRC_OK = True



    return MAC_CRC_OK,DST,SRC,SEQNO,MSG,HAS_CRC




def bit2uint8(bits):
    return np.packbits(bits, bitorder='little')[0]




# LoRa preamble (reverse engineered from a real signal, is actually different from the one described in the patent)
def lora_preamble(n_preamble, k1, k2, BW, K, OSF, t0_frac=0, phi0=0):
    (u0, phi) = lora_chirp(+1, 0, BW, K, OSF, t0_frac, phi0)
    (u1, phi) = lora_chirp(+1, k1, BW, K, OSF, t0_frac, phi)
    (u2, phi) = lora_chirp(+1, k2, BW, K, OSF, t0_frac, phi)
    (d0, phi) = lora_chirp(-1, 0, BW, K, OSF, t0_frac, phi)
    (d0_4, phi) = chirp(BW / 2, K * OSF / 4, 1 / (BW * OSF), -np.power(BW, 2) / K, t0_frac, phi)
    s = np.concatenate((np.tile(u0, (n_preamble)), u1, u2, d0, d0, d0_4))
    return s, phi


#generate a lora chirp
def lora_chirp(mu, k, BW, K, OSF, t0_frac=0, phi0=0):
    fs = BW * OSF
    Ts = 1 / fs
    # number of samples in one period T
    N = K * OSF
    T = N * Ts
    # derivative of the instant frequency
    Df = mu * BW / T
    if k > 0:
        (s1, phi) = chirp(mu * BW * (1 / 2 - k / K), k * OSF, Ts, Df, t0_frac, phi0)
        (s2, phi) = chirp(-mu * BW / 2, (K - k) * OSF, Ts, Df, t0_frac, phi)
        s = np.concatenate((s1, s2))
    else:
        (s, phi) = chirp(-mu * BW / 2, K * OSF, Ts, Df, t0_frac, phi0)
    return s, phi

def samples_decoding(s,BW,N,Ts,K,OSF,Nrise,SF,Trise):

    max_packets = int(np.ceil(Ts * s.size / min_time_lora_packet))
    pack_array = np.empty(shape=(max_packets,), dtype=LoRaPacket)



    OSF = int(OSF)
    N = int(N)
    K = int(K)
    SF = int(SF)
    cumulative_index = 0
    last_index = 0
    received = 0




    while True:
        if s.size < N:
            print("size")
            break

        (success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset) = rf_decode(s[cumulative_index:] ,BW,N,Ts,K,OSF,Nrise,SF,Trise)

        if truncated:
            # print("Truncated")
            break

        if(success):
            # print("success")
            # print(payload)
            # print("message","".join([chr(int(item)) for item in payload]))
            # print("FCS Check", HDR_FCS_OK)
            # print("MAC CRC",MAC_CRC_OK)
            # print("PAYLOAD LENGTH", len(payload))
            pack_array[received] = LoRaPacket(payload,SRC,DST,SEQNO,HDR_FCS_OK,HAS_CRC,MAC_CRC_OK,CR,0,SF,BW)
            received = received + 1



        if(last_index == -1):
            break

        else:
            #cumulative_index = cumulative_index + last_index + 10*N
            #cumulative_index = cumulative_index + last_index + 28 * N
            if (offset != -1):
                cumulative_index = cumulative_index + last_index + offset
            else:
                cumulative_index = cumulative_index + last_index + 28 * N
    return pack_array[:received]





def rf_decode(s,BW,N,Ts,K,OSF,Nrise,SF,Trise):
    truncated = False
    payload = None
    last_index = -1
    success = False
    HDR_FCS_OK = None
    MAC_CRC_OK = None
    DST = -1
    SRC = -1
    SEQNO = -1
    CR = -1
    HAS_CRC = -1
    ns = len(s)
    # base upchirp & downchirp
    u0 = chirp(-BW/2,N,Ts,BW/(N*Ts))[0]

    d0 = np.conj(u0)

    # parameters and state variables in the synch block
    m_phases = 2
    m_phase = 0
    m_vec = -1*np.ones((2*m_phases,6)) # peak values positioning in the DFT
    # oversampling factor for fine-grained esteem
    OSF_fine_sync = 4

    missed_sync = True
    sync_metric = np.Inf

    offset = -1
    main_loop = True
    s_ofs = 0

    #         s = s(620000:1000000)
    #         ns = numel(s)
    while main_loop:
      
        # sample window for a full chirp
        try:

            s_win = s[np.arange(s_ofs,s_ofs+N, dtype = np.int32)]
        except IndexError:
            success = False
            payload = None

            return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset

        # multiplication of the signal with a downchirp and a upchirp, respectively.





        Su = np.abs(np.fft.fft(s_win * d0))
        Sd = np.abs(np.fft.fft(s_win * u0))
        m_u=np.argmax(Su)
        m_d=np.argmax(Sd)

        # convert the positioning in values in the range [-N/2,N/2-1] 
        m_u=np.mod(m_u-1+N/2,N)-N/2
        m_d=np.mod(m_d-1+N/2,N)-N/2


        m_vec[np.arange(m_phase * 2, m_phase * 2 + 2), 1:] = m_vec[np.arange(m_phase * 2, m_phase * 2 + 2), :-1]
        m_vec[np.arange(m_phase * 2, m_phase * 2 + 2), 0] = np.array([m_u, m_d]) #Numpy automatically converts the row into a column





        # three upchirpsm followed by two upchirps shifted by
        # 8 e 16 bits, further followed by two downchirps, correspond to
        # a preamble
        if np.abs(m_vec[m_phase*2+1,0]-m_vec[m_phase*2+1,1])<=1 and \
                np.abs(m_vec[m_phase*2,2]-m_vec[m_phase*2,3]-8)<=1 and \
                np.abs(m_vec[m_phase*2,3]-m_vec[m_phase*2,4]-8)<=1 and \
                np.abs(m_vec[m_phase*2,4]-m_vec[m_phase*2,5]) <=1:

            missed_sync = False
            #keyboard
            tmp = np.sum(np.abs(m_vec[m_phase*2+1,1:2])+np.abs(m_vec[m_phase*2,5:6]))
            if tmp < sync_metric:
                sync_metric = tmp

                #fprintf('phase: #g\n',m_phase)
                #display(m_vec(m_phase*2+(1:2),:))
                Nu = 2
                # fine-grained estimation of the positions of the maximums in the DFT 
                m_u0 = 0
                for i in range (1,Nu+1):
                    try:
                        Su = np.abs(np.fft.fft(((s[np.arange(s_ofs - (4+i) * N,s_ofs - (4+i) * N + N, dtype = np.int32)]) * d0),N*OSF_fine_sync))
                    except IndexError:
                        truncated = True
                        return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset

                    m_u = np.argmax(Su)
                    if (m_u > 0 and m_u < N*OSF_fine_sync-1):
                        m_u = m_u + 0.5*(Su[m_u-1]-Su[m_u+1]) / (Su[m_u-1]-2*Su[m_u]+Su[m_u+1])

                    m_u0 = m_u0 + np.mod(m_u-1+N*OSF_fine_sync/2,N*OSF_fine_sync)-N*OSF_fine_sync/2

                m_u0 = m_u0/Nu

                Nd = 2
                m_d0 = 0
                for i in range(1,Nd+1):
                    Sd = np.abs(np.fft.fft(((s[np.arange(s_ofs - (i - 1) * N, s_ofs - (i - 1) * N + N, dtype = np.int32)]) * u0), int(N * OSF_fine_sync)))
                    m_d = np.argmax(Sd)
                    if m_d > 1 and m_d < N*OSF_fine_sync:
                        try:
                            m_d = m_d + 0.5*(Sd[m_d-1]-Sd[m_d+1]) / (Sd[m_d-1]-2*Sd[m_d]+Sd[m_d+1])
                        except IndexError:
                            pass

                    m_d0 = m_d0 + np.mod(m_d-1+N*OSF_fine_sync/2,N*OSF_fine_sync)-N*OSF_fine_sync/2

                m_d0 = m_d0/Nd

                # Cfo_est: frequency error
                #t_est: timing error
                Cfo_est = (m_u0+m_d0)/2*BW/K/OSF_fine_sync
                t_est = (m_d0-m_u0)*OSF/2/OSF_fine_sync + s_ofs-11*N-Nrise # n_pr = 8 + 2 syncword + 1 downchirp
                break





        m_phase = np.mod(m_phase+1,m_phases)
        s_ofs = s_ofs + N/m_phases
        if s_ofs+N > ns:
            main_loop = False
            success = False
            return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset



    if not missed_sync:

        missed_sync = True

       
       #symbol-level receiver
        p_ofs_est = int(np.ceil(t_est))

        last_index = p_ofs_est

        t0_frac_est = np.mod(-t_est,1)
        #keyboard
        (k_hdr_est,HDR_FCS_OK,k_payload_est,MAC_CRC_OK,MSG,DST,SRC,SEQNO,CR,HAS_CRC,truncated,offset) = lora_packet_rx(s,SF,BW,OSF,Trise,p_ofs_est,Cfo_est)


        if (not truncated) and (HDR_FCS_OK and MAC_CRC_OK):
            n_sym_hdr = len(k_hdr_est)
            n_sym_payload = len(k_payload_est)
            rx_success = True
            success = True
            payload = MSG

        return success, payload, last_index, truncated, HDR_FCS_OK, MAC_CRC_OK, DST, SRC, SEQNO, CR, HAS_CRC, offset


#SUPPORT FUNCTION TO ENCODE A LORA PACKET
def complex_lora_packet(K, n_pr, IH, CR, MAC_CRC, SRC, DST, SEQNO, BW, OSF, SF, Trise, N, Ts, fs, Cfo_PPM, f0,  MESSAGE, t0_frac, phi0):
    k1 = K-8
    k2 = K-16
    SF = np.uint8(SF)
    p = lora_packet(BW,OSF,SF,k1,k2,n_pr,IH,CR,MAC_CRC,SRC,DST,SEQNO,MESSAGE,Trise,t0_frac,phi0)[0]
    size_p = p.size
     
    ntail = N
    Ttail = ntail*Ts
    T = np.ceil(size_p*Ts+Ttail)   # WHOLE NUMBER OF SECONDS
    ns = int(T*fs)
    s = np.zeros(ns, dtype= np.complex64)
     #p_ofs=ceil((ns-np-ntail)*rand)
    p_ofs = 10000

    t = np.arange(0,size_p)*Ts
    # Cfo_TX = Cfo_PPM*1e-6*f0*(2*rand-1)
    # Cfo = Cfo_TX+Cfo_PPM*1e-6*f0*(2*rand-1)
    Cfo_TX = Cfo_PPM * 1e-6 * f0 * 1
    Cfo = Cfo_TX + Cfo_PPM * 1e-6 * f0 * 1
    s[p_ofs+np.arange(0,size_p)] = s[p_ofs+np.arange(0,size_p)] + p * np.exp(1j*(2*np.pi*Cfo*t+phi0))
    s = s[np.arange(0,p_ofs+size_p)]
    return s



#ENCODER FUNCTION. GIVEN A PAYLOAD (IN BYTES), AND THE INTENDED TRANSMISSION PARAMETERS, GENERATES THE SAMPLES FOR THE CORRESPONDING LORA PACKET
def encode(f0, SF, BW, payload, fs, src, dst, seqn, cr=1, enable_crc=1, implicit_header=0, preamble_bits=8):

    OSF = fs / BW
    Ts = 1 / fs
    Nrise = np.ceil(Trise * fs)
    K = np.power(2, SF)
    N = K * OSF
    # t0_frac = rand
    t0_frac = 0
    # phi0 = 2*pi*rand
    phi0 = 0
    complex_samples = complex_lora_packet(K, preamble_bits, implicit_header, cr, enable_crc, src, dst, seqn, BW, OSF, SF, Trise, N,
                            Ts, fs, Cfo_PPM, f0, payload, t0_frac, phi0)
    return complex_samples

#DECODER FUNCTION. LOOKS FOR LORA PACKETS IN THE INPUT COMPLEX SAMPLES. RETURNS ALL THE PACKETS FOUND IN THE SAMPLES.

def decode(complex_samples,SF, BW, fs):
    OSF = fs / BW
    Ts = 1 / fs
    Nrise = np.ceil(Trise * fs)

    K = np.power(2, SF)
    N = K * OSF
    return samples_decoding(complex_samples, BW, N, Ts, K, OSF, Nrise, SF, Trise)


#CLASS TO CONVENIENTLY ENCAPSULATE LORA PACKETS
class LoRaPacket:
    def __init__(self,payload,src,dst,seqn,hdr_ok,has_crc,crc_ok,cr,ih,SF,BW):
        self.payload = payload
        self.src = np.uint8(src)
        self.dst = np.uint8(dst)
        self.seqn = np.uint8(seqn)
        self.hdr_ok = np.uint8(hdr_ok)
        self.has_crc = np.uint8(has_crc)
        self.crc_ok = np.uint8(crc_ok)
        self.cr = np.uint8(cr)
        self.ih = np.uint8(ih)
        self.SF = np.uint8(SF)
        self.BW = BW

    def __eq__(self, other):
        payload_eq = np.all(self.payload == other.payload)
        src_eq = (self.src == other.src)
        dst_eq = (self.dst == other.dst)
        seqn_eq = (self.seqn == other.seqn)
        hdr_ok_eq  = (self.hdr_ok == other.hdr_ok)
        has_crc_eq = (self.has_crc == other.has_crc)
        crc_ok_eq = (self.crc_ok == other.crc_ok)
        cr_eq = (self.cr == other.cr)
        ih_eq = (self.ih == other.ih)
        SF_eq = (self.SF == other.SF)
        BW_eq = (self.BW == other.BW)

        return payload_eq and src_eq and dst_eq and seqn_eq and hdr_ok_eq and has_crc_eq and crc_ok_eq and cr_eq and ih_eq and SF_eq and BW_eq

    def __repr__(self):
        desc = "LoRa Packet Info:\n"
        sf_desc = "Spreading Factor: " + str(self.SF) + "\n"
        bw_desc = "Bandwidth: " + str(self.BW) + "\n"
        if not self.hdr_ok:
            hdr_chk = "Header Integrity Check Failed" + "\n"
            return desc + sf_desc + bw_desc + hdr_chk

        else:
            if self.ih:
                ih_desc = "Implicit Header ON" + "\n"
                if self.has_crc:
                    if self.crc_ok:
                        crc_check = "Payload Integrity Check OK" + "\n"
                        pl_str = "Payload: " + str(self.payload) + "\n"
                        pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                        return desc + sf_desc + bw_desc + ih_desc + crc_check + pl_len + pl_str
                    else:
                        crc_check = "Payload Integrity Check Failed" + "\n"
                        return desc + sf_desc + bw_desc + ih_desc + crc_check
                else:
                    crc_check = "CRC Disabled for this packet. Payload may be corrupted" + "\n"
                    pl_str = "Payload: " + str(self.payload) + "\n"
                    pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                    return desc + sf_desc + bw_desc + ih_desc +  pl_len + pl_str
            else:
                ih_desc = "Explicit Header ON" + "\n"
                hdr_chk = "Header Integrity Check OK" + "\n"
                src_desc = "Source: " + str(self.src) + "\n"
                dest_desc = "Destination: " + str(self.dst) + "\n"
                seq_desc = "Sequence number: " + str(self.seqn) + "\n"
                cr_desc = "Coding Rate: " + str(self.cr) + "\n"
                if self.has_crc:
                    if self.crc_ok:
                        crc_check = "Payload Integrity Check OK" + "\n"
                        pl_str = "Payload: " + str(self.payload) + "\n"
                        pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                        return desc + sf_desc + bw_desc + hdr_chk + ih_desc + src_desc + dest_desc + seq_desc + cr_desc + crc_check + pl_len + pl_str
                    else:
                        crc_check = "Payload Integrity Check Failed" + "\n"
                        return desc + sf_desc + bw_desc + hdr_chk + ih_desc + src_desc + dest_desc + seq_desc + cr_desc + crc_check
                else:
                    crc_check = "CRC Check Disabled for this packet. Payload may be corrupted." + "\n"
                    pl_str = "Payload: " + str(self.payload) + "\n"
                    pl_len = "Payload Length: " + str(self.payload.size) + "\n"
                    return desc + sf_desc + bw_desc + hdr_chk + ih_desc + src_desc + dest_desc + seq_desc + cr_desc + crc_check + pl_len + pl_str


if __name__ == "__main__":
    pass


