# SDR-LoRa: dissecting and implementing LoRa on software-defined radios to advance experimental IoT research
## Authors: Fabio Busacca<sup>1</sup>, Stefano Mangione<sup>1</sup>, Ilenia Tinnirello<sup>1</sup>, Sergio Palazzo<sup>2</sup>, Francesco Restuccia<sup>3</sup>
### <sup>1</sup>Department of Engineering, University of Palermo, Italy
### <sup>2</sup>Department of Electrical, Electronics and Computer Engineering (DIEEI), University of Catania, Italy
### <sup>3</sup>Institute for the Wireless Internet of Things, Northeastern University, Boston, USA
### Paper Link: https://dl.acm.org/doi/pdf/10.1145/3556564.3558239
### Plug-and-play Colosseum Container: https://zenodo.org/record/7520494

In this repository, we provide a complete implementation of the LoRa physical (PHY) layer, i.e., both the encoder and the decoder, written in pure NumPy. More in detail, `lora.py` turns a payload into In-phase and Quadrature (IQ) samples and IQ samples back into packets, without GNU Radio, vendor blocks or compiled dependencies. Therefore, every stage of the waveform is open to inspection and modification.

Moreover, [gr-sdrlora/](gr-sdrlora) provides a bit-exact C++ port of both the transmitter and the receiver, together with GNU Radio 3.10 blocks. The two implementations produce identical waveforms and agree packet by packet. Both of them decode captures from commercial LoRa transceivers, interoperate with gr-lora_sdr in both directions, and have been tested over the air against an SX1276 node at every Spreading Factor (SF) it supports.

## What's new in this version

This is a major new release of SDR-LoRa. We have reworked the original encoder and decoder throughout, fixed many defects and heavily extended the suite. In particular:

- **C++ port and GNU Radio 3.10 blocks.** The release includes the LoRa Encoder, LoRa Decoder and LoRa Packet Print blocks, their GNU Radio Companion (GRC) integration, example flowgraphs and Python bindings for the C++ PHY. The waveforms of the C++ encoder are byte-identical to the ones of `lora.py`, and the two receivers agree packet by packet.
- **A much stronger receiver.** The Hamming Forward Error Correction (FEC) is now used to correct errors, and soft-decision maximum-likelihood decoding has been added. Moreover, the receiver now includes a channel filter, accumulation-based preamble detection, fine Carrier Frequency Offset (CFO) estimation and sample-clock tracking, with a drift retry for long frames gated by the Cyclic Redundancy Check (CRC). Finally, Low Data Rate Optimization (LDRO) follows the 16 ms rule, with a retry for transmitters that do not.
- **Interoperability.** Standard LoRa framing (`pseudo_header=False`) works with SX127x nodes and gr-lora_sdr in both directions. In addition, this release supports implicit header in both directions, SF5 and SF6, a configurable sync word, preamble length and LDRO, inverted IQ for LoRaWAN downlinks, frequency hopping, and payloads of up to 255 bytes.
- **Measurement support.** `report_crc_errors` provides Packet Error Rate (PER) counts, and a streaming decoder reports each packet exactly once across buffer boundaries. Moreover, frames without a CRC are checked against noise.
- **USRP transceiver.** The Universal Software Radio Peripheral (USRP) transceiver exposes every PHY parameter, per SF. It decodes in line with overload protection, and its multi-SF bursts now work.
- **Robustness.** Every parameter is validated, with an error message that names it, and NumPy 2 is supported.
- **Clean carrier by default.** `lora.encode` used to add a simulated 17 ppm crystal offset to every frame (29.5 kHz at 868 MHz). It now defaults to `cfo_ppm=0`, as the C++ API and the GNU Radio blocks do; pass `cfo_ppm=lora.Cfo_PPM` to simulate the old offset.

## What's implemented

The table below summarizes the full LoRa chain, which we reverse-engineered from the specification and from real signals.

| Stage | Notes |
|---|---|
| Explicit header | 20-bit header with its own Hamming(8,4) code and 5-bit Frame Check Sequence (FCS) |
| Implicit header | Both directions; the receiver takes length, Coding Rate (CR) and CRC flag as parameters |
| Whitening | Linear-Feedback Shift Register (LFSR) sequence, applied per byte |
| CRC-16 | LoRa's variant, including the length-dependent initial states |
| Hamming FEC | All four rates, with soft-decision maximum-likelihood decoding |
| Diagonal interleaving | As in the patent, including the row flip |
| Gray mapping | With LDRO, enabled when the symbol time `2^SF/BW` exceeds 16 ms: SF11/SF12 at 125 kHz and SF12 at 250 kHz, never at 500 kHz (`lora.ldro_enabled`) |
| Chirp Spread Spectrum (CSS) modulation | Chirp synthesis with continuous phase and raised edges |
| Preamble and sync | Up-chirps, sync words, down-chirps and quarter down-chirp |
| Receiver | FFT dechirping, coarse and fine synchronization, CFO estimation, channel filter, sample-clock tracking and soft-decision FEC |

SF ranges from 5 to 12, and CR from 4/5 to 4/8. Payloads can reach 251 bytes with the 4-byte pseudo-header of SDR-LoRa, or the full **255** bytes with `pseudo_header=False`, i.e., standard framing with no prefix on air. Note that SF5 and SF6 support implicit header only, since the 20-bit header does not fit in the `4*(SF-2)` bits of their header block.

As for sample rates, the Python implementation accepts any integer `fs/BW`, while the C++/GNU Radio port requires a power-of-two multiple, since its synchronization FFTs need power-of-two lengths. Moreover, `fs/BW = 1` is critically sampled, and hence interpolated to 4× internally. Both implementations cap the ratio at `fs/BW ≤ 65536` and `fs ≤ 10¹² Hz` (`lora.OSF_MAX`, `lora.FS_MAX`).

## Install

The PHY only needs NumPy. Therefore, it is enough to clone the repository and install its dependency; `lora.py` is then imported from the repository root:

```bash
git clone https://github.com/fabio-busacca/sdr-lora
cd sdr-lora
pip install -r requirements.txt
```

The project is deliberately **not** published on PyPI. In fact, the module is named `lora`, an import name already owned by an unrelated PyPI project, and we prefer to keep the name used in the paper. Hence, install it from a clone (`pip install .` also works).

Driving a USRP through `lora_transceiver.py` also requires the **Python bindings of the USRP Hardware Driver (UHD)**, which ship with the driver and are not available on PyPI (see the [UHD Python API docs](https://files.ettus.com/manual/page_python.html)). For GNU Radio, build the module in [gr-sdrlora/](gr-sdrlora).

## Quickstart: no radio required

`lora.py` is self-contained, so a packet can be generated and decoded entirely in software:

```python
import numpy as np
import lora

f0, BW, fs = 868e6, 125e3, 1e6      # carrier, bandwidth, sample rate
SF = 7

payload = np.frombuffer(b"hello lora", dtype=np.uint8)

# Payload -> IQ samples
samples = lora.encode(f0, SF, BW, payload, fs, src=3, dst=7, seqn=1,
                      cr=1, enable_crc=1)

# IQ samples -> packets
for pkt in lora.decode(samples, SF, BW, fs):
    print(pkt)
```

```
LoRa Packet Info:
Spreading Factor: 7
Bandwidth: 125000.0
Header Integrity Check OK
Explicit Header ON
Source: 3
Destination: 7
Sequence number: 1
Coding Rate: 1
Payload Integrity Check OK
Payload Length: 10
Payload: [104 101 108 108 111  32 108 111 114  97]
```

Since `encode` returns plain IQ samples, they can go through any channel model (e.g., Additive White Gaussian Noise (AWGN), fading or a collision between two packets) and then straight back to `decode`.

## API

### `lora.encode(f0, SF, BW, payload, fs, src, dst, seqn, cr=1, enable_crc=1, implicit_header=0, preamble_bits=8, pseudo_header=True, cfo_ppm=0.0, sync_word=0x12, ldro=2, lead_in=None, invert_iq=False, trise=None, t0_frac=0.0, phi0=0.0, amplitude=1.0, hop_period=0, hop_freqs=None)`

`encode` builds the complete packet and returns its IQ samples as `np.complex64`. Note that `fs` must be an integer multiple of `BW`, and that `payload` is a `uint8` array of up to 251 bytes with the default pseudo-header, or 255 with `pseudo_header=False`.

**Pseudo-header.** `pseudo_header=False` sends a standard LoRa frame. Conversely, the default prefixes the payload with the 4-byte DST/SRC/SEQNO/LENGTH block of SDR-LoRa, which other LoRa implementations do not expect; hence, use `False` to communicate with any other device. The pseudo-header comes from the original SDR-LoRa release, which put this block at the front of every payload. Since LoRa itself has no addresses or sequence numbers, on air the block is ordinary payload, i.e., a standard receiver sees it as the first four bytes of the message. We keep it as the default for compatibility with the original release. Note that both ends must use the same setting, since nothing on air tells which framing a frame uses.

**Lead-in.** `lead_in` is the number of zero samples ahead of the preamble. Its default, `None`, means `lora.default_lead_in(SF, BW, fs, preamble_bits)`, namely 10000 samples, topped up for short preambles so that the receiver has the history it needs.

**Carrier offset.** `cfo_ppm` adds a simulated crystal offset to the waveform, applied once per end of the link, for a total of 2 × `cfo_ppm` × 1e-6 × `f0`. The default, 0, is a clean carrier, which is the right choice for anything sent over the air, since a real radio brings its own offset. Conversely, `lora.Cfo_PPM` (17 ppm, 29.5 kHz at 868 MHz) is a typical crystal error for simulations.

The table below describes every parameter.

| Parameter | Default | Meaning |
|---|---|---|
| `f0` | | RF carrier, in Hz. The output is at baseband, so `f0` only scales `cfo_ppm`. |
| `SF` | | Spreading factor, from 5 to 12. SF5 and SF6 need `implicit_header=1`. |
| `BW` | | Bandwidth, in Hz. |
| `payload` | | The message, as a `uint8` array. |
| `fs` | | Sample rate, in Hz, an integer multiple of `BW`. |
| `src`, `dst`, `seqn` | | Pseudo-header fields, from 0 to 255. They are ignored, and may be `None`, with `pseudo_header=False`. |
| `cr` | 1 | Coding rate, from 1 to 4, for 4/5 to 4/8. |
| `enable_crc` | 1 | Appends the payload CRC-16. |
| `implicit_header` | 0 | Leaves the PHY header out (see [Implicit header](#implicit-header)). |
| `preamble_bits` | 8 | Preamble length in up-chirps (symbols, despite the name), from 1 to 65535. |
| `pseudo_header` | True | The 4-byte DST/SRC/SEQNO/LENGTH prefix described above. |
| `cfo_ppm` | 0.0 | Simulated crystal offset, as described above. |
| `sync_word` | 0x12 | One byte (0x12 for private networks, 0x34 for LoRaWAN), or two raw sync symbols. |
| `ldro` | 2 | Low data rate optimization: 2 is auto (enabled when a symbol lasts more than 16 ms, see `lora.ldro_enabled`), 1 is on and 0 is off. |
| `lead_in` | None | Zero samples ahead of the preamble, as described above. |
| `invert_iq` | False | Swaps I and Q (see [Inverted IQ](#inverted-iq)). |
| `trise` | None (50 µs) | Duration of the raised edge at each chirp boundary, in seconds. |
| `t0_frac` | 0.0 | Fractional timing offset of the whole frame, in samples, between −1 and 1, used as a test stimulus. |
| `phi0` | 0.0 | Initial carrier phase, in radians. |
| `amplitude` | 1.0 | Output scale. |
| `hop_period`, `hop_freqs` | 0, None | Frequency hopping (see [Frequency hopping](#frequency-hopping)). |

### `lora.decode(complex_samples, SF, BW, fs, channel_filter_rx=True, implicit_header=False, payload_len=None, cr=1, enable_crc=1, fec_correct=True, soft_decision=True, robust_sync=True, sync_phases=None, pseudo_header=True, interpolate=True, sync_word=0x12, ldro=2, report_crc_errors=False, with_consumed=False, invert_iq=False, trise=None, cutoff_ratio=None, numtaps=None, sync_threshold=None, search_tail_symbols=None, min_osf=None, interp_target_osf=None, hop_period=0, hop_freqs=None, center_freq=None)`

`decode` scans a sample buffer and returns an array with every `LoRaPacket` it finds, since a buffer may hold several of them. **Always check `pkt.crc_ok` before trusting `pkt.payload`, as well as `pkt.has_crc` if your transmitters always send a CRC.** In fact, a header corrupted beyond what its 5-bit checksum detects can claim that no CRC is present. Moreover, for a frame without a CRC, `crc_ok` only states that its symbols look like signal rather than noise. This rejects false locks on noise, but not a real frame received with bit errors; therefore, send a CRC whenever the payload matters.

- `sync_word` must match the transmitter, since it gates detection. `ldro=2` (auto) applies the 16 ms rule and, at SF11 and SF12, retries a frame that fails its CRC with LDRO the other way, since the Arduino LoRa library 0.8.0 leaves it off at SF11 / 125 kHz. Conversely, `0` and `1` are taken as given.
- `soft_decision` decodes from the FFT magnitudes, while `fec_correct` uses the Hamming parity; together, they are worth about 2–2.5 dB. Moreover, `channel_filter_rx` band-limits the input to the LoRa channel first, which is worth about 6 dB when the samples carry out-of-channel noise.
- `center_freq` is the RF carrier the capture was tuned to. Given it, the receiver derives the sample-clock error from the carrier offset it measures (since the clock and the Local Oscillator (LO) of a radio share one crystal), and moves each symbol window accordingly; long frames at high SF need this. In any case, a frame that is located but fails its CRC is demodulated again with the drift estimated from its own symbols, and kept only if the CRC then passes. This also covers a carrier offset that is not entirely due to crystal error.
- `sync_phases` sets the granularity of the preamble scan: 4 below SF7 (the default there), and 2 from SF7 on. From SF7, 4 finds more weak frames, at 1.1–1.8× the decode time.

The table below describes every parameter.

| Parameter | Default | Meaning |
|---|---|---|
| `complex_samples` | | The IQ capture, as a complex array. |
| `SF`, `BW`, `fs` | | As for `encode`. |
| `channel_filter_rx` | True | Band-limits the input to the LoRa channel before synchronization, as described above. |
| `implicit_header`, `payload_len`, `cr`, `enable_crc` | False, None, 1, 1 | Implicit-header reception (see [Implicit header](#implicit-header)). With an explicit header, `cr` and `enable_crc` are ignored, since the header carries them. |
| `fec_correct` | True | Corrects errors with the Hamming parity. Note that only CR 4/7 and 4/8 can locate an error, while 4/5 and 4/6 can only detect one. |
| `soft_decision` | True | Maximum-likelihood decoding from the FFT magnitudes. It helps at every coding rate, and supersedes `fec_correct` for the payload. |
| `robust_sync` | True | Detects the preamble from an accumulation of its up-chirps, rather than symbol by symbol. This prevents a co-channel LoRa frame from stealing the lock, but needs about 6 symbols of signal or lead-in before the sync word. Pass False for truncated preambles. |
| `sync_phases` | None | Granularity of the preamble scan, as described above. |
| `pseudo_header` | True | Must match the transmitter. With it off, `src`, `dst` and `seqn` are returned as `None`. |
| `interpolate` | True | At `fs/BW = 1`, interpolates the input up to `interp_target_osf`, instead of decoding at a rate that does not work. |
| `sync_word`, `ldro`, `invert_iq` | 0x12, 2, False | Must match the transmitter, as described above. |
| `report_crc_errors` | False | Also returns the frames that failed their CRC (see [Measuring a packet error rate](#measuring-a-packet-error-rate)). |
| `with_consumed` | False | Returns `(packets, consumed)`, where `consumed` is the number of samples the scan is finished with. It is meant for feeding a stream manually; `StreamDecoder` already does it. |
| `trise` | None (50 µs) | Raised edge of the transmitter. |
| `cutoff_ratio`, `numtaps` | None (1.15, 255) | Cutoff of the channel filter, in multiples of `BW`, and its (odd) length. |
| `sync_threshold` | None (5.5) | Peak-to-mean threshold of the preamble detector. A lower value finds weaker frames, at the cost of more false locks. |
| `search_tail_symbols` | None (24) | Symbols left unscanned at the end of the buffer, since a frame needs its preamble, sync and header before it can be judged. It matters for streaming, and ranges from 13 to 4096. |
| `min_osf`, `interp_target_osf` | None (2, 4) | Below `min_osf`, the input is interpolated to `interp_target_osf`. |
| `hop_period`, `hop_freqs` | 0, None | Must match the transmitter (see [Frequency hopping](#frequency-hopping)). |
| `center_freq` | None | RF carrier the capture was tuned to, for sample-clock tracking, as described above. |

### `lora.channel_filter(s, BW, fs, cutoff_ratio=None, numtaps=None)`

This is the receive filter on its own, e.g., to filter a long capture once before decoding slices of it.

### Helpers

- `lora.ldro_enabled(SF, BW, ldro)` returns whether LDRO is enabled for a configuration (the 16 ms rule under auto).
- `lora.default_lead_in(SF, BW, fs, preamble_bits=8)` returns the lead-in that `encode` uses when `lead_in=None`.
- `lora.default_sync_phases(SF)` returns the `sync_phases` that `decode` uses when given `None`.
- `lora.sync_word_symbols(sync_word)` returns the two sync symbols that a sync word puts on air.
- `lora.Cfo_PPM` is a typical crystal error (17 ppm), to be used as `cfo_ppm`.

### Implicit header

`encode(..., implicit_header=1)` leaves the 20 header bits out, but keeps the shape of the first block (8 symbols at CR 4/8, carrying payload bits only), as LoRa and gr-lora_sdr build it. Therefore, the receiver must be told what the header would have carried:

```python
s = lora.encode(f0, SF, BW, payload, fs, src, dst, seqn, cr=1, implicit_header=1)
pkts = lora.decode(s, SF, BW, fs, implicit_header=True,
                   payload_len=payload.size, cr=1, enable_crc=1)
```

A mismatched `payload_len` makes the CRC fail, rather than returning wrong data. Note that implicit header with `pseudo_header=False` and `enable_crc=0` carries no integrity check at all; hence, enable the CRC, or validate payloads at the application level.

### `lora.LoRaPacket`

Its fields are `payload`, `src`, `dst`, `seqn`, `hdr_ok`, `has_crc`, `crc_ok`, `cr`, `ih`, `SF` and `BW`. Moreover, it provides a readable `__repr__` (shown above) and value equality.

### `lora_utils`

`lora_utils` provides fragmentation and reassembly above the PHY, as well as Acknowledgment (ACK), Negative Acknowledgment (NACK) and polling packets. All its builders return `LoRaPacket`s, ready to be transmitted (e.g., with `tx_send_burst`); `srcID` and `dstID` fill their pseudo-header fields. For instance, an array of 600 `uint16` values travels in 13 packets of at most 100 bytes each:

```python
import numpy as np
import lora_utils

data = np.arange(600, dtype=np.uint16)
pkts = lora_utils.pack_lora_data(data, 7, 125e3, packet_size=100, srcID=1, dstID=2)
back = lora_utils.unpack_lora_data(pkts, np.uint16)     # equal to data
```

The table below describes every function.

| Function | What it does |
|---|---|
| `rate_calculator(sf, bw, cr)` | Nominal bit rate in bit/s, computed as `sf · 4/(4+cr) · bw/2^sf` (e.g., 5.47 kb/s at SF7, 125 kHz and CR 4/5). |
| `pack_lora_data(data, SF, BW, packet_size, srcID, dstID, extended_sqn=True, CR=1)` | Splits any NumPy array, viewed as bytes, into packets with at most `packet_size` payload bytes (from 2 to 251), numbered through `seqn` (from 0 to 255, then wrapping). With `extended_sqn`, the first payload byte of each packet carries a group number, which increments every 256 packets; therefore, up to 65536 packets stay distinct, and each of them carries `packet_size − 1` data bytes. |
| `unpack_lora_data(pkt_array, arr_type=np.uint8, extended_sqn=True)` | The inverse operation: it concatenates the payloads (dropping the group byte with `extended_sqn`) and returns them as an array of `arr_type`. The packets must be complete and in order; moreover, a byte count that does not divide into `arr_type` raises an error, rather than being trimmed. |
| `pack_lora_nack(data, SF, BW, packet_size, srcID, dstID, CR=1)` | Packs a list of missing sequence numbers (an unsigned array, e.g., `uint16`) into packets whose first payload byte is `NACK_CODE` (255), except for the last one, which carries `ACK_CODE` (253). An empty list gives a single ACK. |
| `unpack_lora_ack(acks_array)` | The inverse operation: it returns the missing sequence numbers as `uint16`, skipping repeated packets and stopping at the ACK. |
| `gen_pack_polling(SF, BW, srcID, dstID, CR=1, brdcst=False)` | A one-byte polling packet, carrying `POLLING_CODE` (254), or `POLLING_CODE_BROADCAST` (252) with `brdcst=True`. |
| `pack16bit(high_byte, low_byte)` | Combines two bytes into a `uint16`. |

The module also defines `MAX_PACKET_SIZE` (251), `ACK_CODE`, `NACK_CODE`, `POLLING_CODE`, `POLLING_CODE_BROADCAST` and `BROADCAST_ID` (255).

### Streaming

`lora_transceiver.StreamDecoder(sf, BW, fs, decode_opts=None, max_payload_len=251)` decodes a stream fed in chunks of any size, against a rolling buffer. It needs neither a radio nor UHD, and reports each packet exactly once, including a packet that straddles two chunks:

```python
from lora_transceiver import StreamDecoder

dec = StreamDecoder(7, 125e3, 1e6, decode_opts={"pseudo_header": False})
for chunk in chunks:                  # numpy complex64 arrays, any size
    for pkt in dec.feed(chunk):
        print(pkt)
# At the end of a file or capture, flush: a frame ending in the last
# stretch comes out only here.
for pkt in dec.feed(np.zeros(0, np.complex64), final=True):
    print(pkt)
```

`decode_opts` takes the `lora.decode` options above, except for `with_consumed`, which the stream decoder handles itself. Moreover, `max_payload_len` bounds how much signal the decoder keeps for a frame that is still arriving, so a longer frame is not delivered. Finally, `dec.buffered` is the number of samples currently held, and `dec.cap` the maximum. Note that the live transceiver and the GNU Radio decoder block both work in the same way.

## Measuring a packet error rate

`lora.decode(..., report_crc_errors=True)` also returns the frames that had a valid header but failed their payload CRC, as `LoRaPacket`s with `crc_ok = 0`:

```python
got = lora.decode(samples, 7, BW, fs, pseudo_header=False,
                  report_crc_errors=True)
ok  = sum(1 for p in got if p.crc_ok and p.has_crc)
bad = sum(1 for p in got if not p.crc_ok)
```

Note that the denominator here is the number of frames *detected*. In fact, a frame whose preamble or header was lost is never seen, so a true PER against what was sent needs the count of the transmitter. Moreover, two configurations are refused, rather than mismeasured: **implicit header**, since it has no header FCS to tell a frame from a false lock, and **`pseudo_header=True`**, since it cannot size a damaged frame. Therefore, use standard framing (`pseudo_header=False`). In GNU Radio, the same feature is the **Report CRC Failures** parameter of the decoder block, with `crc_errors()` beside `packets()`.

## GNU Radio blocks

Every PHY parameter is exposed in GRC; [gr-sdrlora/README.md](gr-sdrlora/README.md) describes how to build and install the module. Moreover, [gr-sdrlora/examples](gr-sdrlora/examples) contains ready-to-run flowgraphs: a loopback that needs no radio, and transmitters and receivers for a USRP and a HackRF.

**LoRa Encoder**: `sf`, `cr`, `bw`, `samp_rate`, `f0`, `has_crc`, `implicit_header`, `preamble_symbols`, `src`, `dst`, `pseudo_header`, `seqn` (initial sequence number), `cfo_ppm`, `sync_word`, `ldro`, `lead_in`, `invert_iq`, `trise`, `t0_frac`, `phi0`, `amplitude`, `hop_period`, `hop_freqs` and `burst_tail` (zeros closing each burst: -1, the default, clears a downstream filter, while 0 suits a radio).

**LoRa Decoder**: `sf`, `bw`, `samp_rate`, `window`, `channel_filter_rx`, `soft_decision`, `robust_sync`, `sync_phases`, `implicit_header`, `payload_len`, `cr`, `has_crc`, `pseudo_header`, `sync_word`, `ldro`, `fec_correct`, `interpolate`, `report_crc_errors`, `max_payload_len`, `invert_iq`, `trise`, `cutoff_ratio`, `numtaps`, `sync_threshold`, `search_tail_symbols`, `min_osf`, `interp_target_osf`, `hop_period`, `hop_freqs`, `low_latency` and `center_freq`.

**LoRa Packet Print** prints each decoded packet as a line of text, together with its metadata. Every block parameter is described, with its default, in [gr-sdrlora/README.md](gr-sdrlora/README.md#use-it), as well as in the documentation tab of each block in GRC.

These are the `lora.encode` / `lora.decode` parameters, plus four block-specific ones: `burst_tail`, `window` (the samples handed to the decoder per call, whose default `low_latency` halves), `max_payload_len` (the reassembly bound of the block), and the Protocol Data Unit (PDU) port in place of `payload`. On the decoder, `cr` and `has_crc` only appear in implicit-header mode, since with an explicit header they come from the air.

`sync_threshold` is the peak-to-mean threshold of the preamble detector. Its default of 5.5 works well; however, the false-lock rate that a deployment can tolerate depends on the deployment itself, which is why the threshold is exposed. Note that the GRC files install to `share/gnuradio/grc/blocks`, so GRC shows changes only after the module is reinstalled.

## Running on a USRP

> **The UHD radio I/O has not been run against a USRP.** The engine itself is exercised on files and arrays, but no USRP was available to test the radio path. Hence, treat it as untested.

`lora_transceiver.py` wraps the PHY in a multiprocessing TX/RX engine. More in detail, a receive process fills a shared-memory ring buffer, and one decoder process per SF consumes it, so that several SFs are demodulated concurrently.

```python
import lora_transceiver

radio = lora_transceiver.lora_transceiver(
    "192.168.40.2",          # USRP address
    rx_gain=10, tx_gain=20,
    bandwidth=125000,
    rx_freq=990e6, tx_freq=1010e6,
    sample_rate=1e6,
    rx_channel_ID=0, tx_channel_ID=1)

tx_queue  = radio.tx_start(sleep_time=1)      # put LoRaPackets here to send
rx_queues = radio.rx_start([7, 8], 3_000_000) # one queue per SF
```

[tx_example.py](tx_example.py), [rx_example.py](rx_example.py) and [tx_rx_example.py](tx_rx_example.py) show complete examples. The `cr`, `ih` and `has_crc` of each packet go on air, while `encode_opts` sets the rest for a whole burst, and `decode_opts` / `decode_opts_per_sf` configure the receivers:

```python
tx_queue = radio.tx_start(sleep_time=1, encode_opts={
    "cfo_ppm": 0.0,
    "sync_word": 0x34,       # LoRaWAN public network (0x12 is the private default)
    "preamble_bits": 12})

rx_queues = radio.rx_start([5, 7], 3_000_000,
    decode_opts={"sync_word": 0x34},
    decode_opts_per_sf={  # 0x34's second symbol, 32, is the preamble's bin at SF5
        5: {"implicit_header": True, "payload_len": 16, "cr": 1, "sync_word": 0x12}})
```

`decode_opts` accepts `center_freq` (which defaults to the `rx_freq` of the radio, while 0 turns it off), `channel_filter_rx`, `cr`, `cutoff_ratio`, `enable_crc`, `fec_correct`, `hop_freqs`, `hop_period`, `implicit_header`, `interp_target_osf`, `interpolate`, `invert_iq`, `ldro`, `min_osf`, `numtaps`, `payload_len`, `pseudo_header`, `report_crc_errors`, `robust_sync`, `search_tail_symbols`, `soft_decision`, `sync_phases`, `sync_threshold`, `sync_word` and `trise`. Similarly, `encode_opts` accepts `amplitude`, `cfo_ppm`, `hop_freqs`, `hop_period`, `invert_iq`, `ldro`, `lead_in`, `phi0`, `preamble_bits`, `pseudo_header`, `sync_word`, `t0_frac` and `trise`. An unknown key is rejected, rather than silently ignored.

The table below describes the methods of the radio.

| Call | What it does |
|---|---|
| `lora_transceiver(address, rx_gain, tx_gain, bandwidth, rx_freq, tx_freq, sample_rate, rx_channel_ID, tx_channel_ID, signal_amplitude=1)` | Opens the USRP at `address` and configures both chains. `signal_amplitude` scales every transmitted burst, and can be changed later through `radio.signal_amplitude`. |
| `rx_start(sf_list, complex_data_number=3000000, block_size=20, decode_opts=None, decode_opts_per_sf=None, max_payload_len=251)` | Starts one decoder per SF in `sf_list`, and returns one packet queue per SF. `complex_data_number` is the receive window, in samples, while `block_size` is the length of the shared-memory ring, in windows. |
| `rx_pause()`, `rx_resume()` | Stop and restart reading samples from the radio, without tearing the decoders down. |
| `rx_stop(wait=False, timeout=10.0)` | Stops the receiver. |
| `report_rx_config()` | Prints and returns the settings that UHD actually applied. |
| `tx_start(sleep_time, verbose=False, encode_opts=None)` | Starts the transmit worker and returns its queue: each `LoRaPacket` put there is sent, with `sleep_time` seconds between packets. |
| `tx_send_burst(pkt_list, sleep_time, verbose=False, encode_opts=None)` | Sends a list of packets once, `sleep_time` seconds apart. |
| `tx_send_burst_multi_sf(pkt_list, sleep_time, sf_list, encode_opts=None)` | Sends packets on several SFs, superimposed in one burst. |
| `tx_stop(wait=False, timeout=10.0)` | Stops the transmit worker, and drops anything still queued. |

- **SF5 and SF6** need `payload_len` (as well as `cr` and `enable_crc`, if not the defaults) per SF, since there is no header on air to read them from.
- **`payload_len` is the message length.** With `pseudo_header=True`, the 4-byte prefix is added and stripped automatically, and is not counted.
- **Check the analog bandwidth that UHD actually set.** In fact, `set_rx_bandwidth()` is a request, and many daughterboards round it to a value far wider than a 125 kHz channel. The constructor prints the coerced configuration, and `radio.report_rx_config()` returns it. When the front end does not filter, the software channel filter does.
- **The receive window** (the second argument of `rx_start`) trades latency against CPU, not reliability. Indeed, a frame in flight is carried from one window to the next, so even a window much shorter than a frame delivers it, exactly once.
- **Multi-SF bursts.** `tx_send_burst_multi_sf` superimposes packets on several SFs in one burst, scaled together by `amplitude`.

**Real-time budget.** The table below reports the share of real time used by the decoder, with the channel filter and soft decision on, at a 3,000,000-sample window at 1 MS/s on an idle machine. Expect a variation of about 30% with machine load.

| SF | per buffer | steady state |
|---|---|---|
| 7 | 32% | 38% |
| 8 | 25% | 31% |
| 9 | 25% | 27% |
| 10 | 22% | 23% |
| 11 | 25% | 30% |
| 12 | 27% | 43% |

`decode(..., channel_filter_rx=False, soft_decision=False)` reduces the CPU load, at the cost of about 8 dB of sensitivity. Moreover, windows the decoder cannot keep up with are dropped with a warning (`MAX_DECODER_BACKLOG`), rather than queued.

## Frequency hopping

Frequency-Hopping Spread Spectrum (FHSS) is implemented on both sides: the transmitter stays `hop_period` symbols on each channel, following the `hop_freqs` list (offsets in Hz), which the host supplies and the air does not carry. `hop_period = 0` turns hopping off, as `RegHopPeriod` does.

- The schedule is anchored at the **header**, not at the frame start. In fact, the preamble and sync go out on `hop_freqs[0]`, which **must be 0**, since the receiver acquires there. Hence, a receiver built to interoperate must de-hop from the header.
- `hop_period` must be at least 13 symbols, the length of the sync pattern.
- The receive filter widens automatically to cover the plan (with cutoff `1 + 2·max|f|/BW`, capped at 64), which bounds `max|f| ≤ 31.5·BW`. Pass `channel_filter_rx=False` for a wider plan.
- The phase restarts at each hop, as the one of a real synthesizer does.

## Inverted IQ

Swapping I and Q, as the SX127x `InvertIQ` bits do, is equivalent to conjugating the frame at complex baseband, i.e., every chirp reverses its direction. This is the polarity of a **LoRaWAN downlink**. In fact, gateways transmit inverted and end devices normal, so an inverted frame deliberately does not decode as a normal one.

    lora.encode(..., invert_iq=True)      # and decode(..., invert_iq=True)
    p.invert_iq = True                    # C++ tx_params; "Invert IQ" in GRC

The conjugation applies before the carrier-offset term, since the frequency error of a transmitter does not invert with its modulator. Moreover, the Python and C++ encoders produce identical inverted waveforms, and each decoder reads the waveforms of the other.

## Limitations

- **No Channel Activity Detection (CAD)** and **no LoRaWAN MAC** (join procedure, Message Integrity Code (MIC), AES encryption, frame counters and Adaptive Data Rate (ADR)), since this is a PHY. Note that inverted IQ, described above, is the one LoRaWAN-specific PHY feature, and it is implemented.
- **Bandwidths below 125 kHz** work; however, a simulated carrier offset that is fine at 125 kHz is a larger fraction of a narrower channel. For instance, 29.5 kHz (`cfo_ppm=17` at 868 MHz) exceeds the capture range of the receiver there. Moreover, these bandwidths are less exercised than 125–500 kHz.
- **The USRP radio I/O is untested** on hardware (see above). Conversely, other front ends feeding samples to `lora.decode`, to the `StreamDecoder` or to the GNU Radio block have been used live.
- **Frames without a CRC** cannot be protected against bit errors, since the receiver only rejects frames that look like noise. Therefore, send a CRC.

## Compatibility

SDR-LoRa supports Python 3.9 and later, and NumPy 1.23 and later, including NumPy 2. In particular, the bit-packing arithmetic is written so that the integer promotion rules of NumPy 2 (NEP 50) do not change any result.

## Citing

If you use this code, please cite the journal article:

```bibtex
@article{busacca2024sdrlora,
  title    = {{SDR-LoRa}, an open-source, full-fledged implementation of
              {LoRa} on {Software-Defined-Radios}: Design and potential
              exploitation},
  author   = {Busacca, Fabio and Mangione, Stefano and Palazzo, Sergio and
              Restuccia, Francesco and Tinnirello, Ilenia},
  journal  = {Computer Networks},
  volume   = {241},
  pages    = {110194},
  year     = {2024},
  issn     = {1389-1286},
  doi      = {10.1016/j.comnet.2024.110194},
  publisher = {Elsevier}
}
```

The earlier workshop paper describes the original design:

```bibtex
@inproceedings{busacca2022sdrlora,
  title     = {Dissecting and Implementing {LoRa} on Software-Defined Radios
               to Advance Experimental {IoT} Research},
  author    = {Busacca, Fabio and Mangione, Stefano and Tinnirello, Ilenia and
               Palazzo, Sergio and Restuccia, Francesco},
  booktitle = {Proceedings of the 16th ACM Workshop on Wireless Network
               Testbeds, Experimental Evaluation \& Characterization (WiNTECH)},
  year      = {2022},
  doi       = {10.1145/3556564.3558239}
}
```

`CITATION.cff` carries the same record, so that the **Cite this repository** button of GitHub returns the journal article.

## License

GPL-3.0-only (see [LICENSE](LICENSE)).
