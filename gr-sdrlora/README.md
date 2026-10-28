# gr-sdrlora

gr-sdrlora is a GNU Radio 3.10 out-of-tree module that implements the LoRa physical (PHY) layer in C++. More in detail, it provides three blocks, namely **LoRa Encoder**, **LoRa Decoder** and **LoRa Packet Print**, makes the PHY itself callable from Python, and installs the PHY as a plain C++ library, which can also be built without GNU Radio (see [C++ library without GNU Radio](#c-library-without-gnu-radio)).

The Digital Signal Processing (DSP) code lives in [`lib/lora_phy.cc`](lib/lora_phy.cc) (transmission), [`lib/lora_rx.cc`](lib/lora_rx.cc) (demodulation and decoding) and [`lib/lora_sync.cc`](lib/lora_sync.cc) (channel filter, preamble search, Carrier Frequency Offset (CFO) estimation and clock tracking). Note that this code has **no GNU Radio dependency**, since the blocks are only a wrapper around it.

## Build

On Ubuntu 24.04 with GNU Radio 3.10 installed, the only extra packages needed are `cmake`, `make` and pybind11:

```bash
sudo apt install -y cmake make pybind11-dev
```

In order to install the blocks where GNU Radio Companion (GRC) finds them, use the prefix of GNU Radio itself:

```bash
cmake -S gr-sdrlora -B build -DCMAKE_INSTALL_PREFIX=$(gnuradio-config-info --prefix)
cmake --build build -j$(nproc)
sudo cmake --install build && sudo ldconfig
```

A user prefix, such as `~/.local`, also builds and installs. However, the Python extension then imports only with its own directory on `PYTHONPATH`, and `from gnuradio import sdrlora` (the import that the GRC files emit) does not resolve. In fact, a Python package does not merge across `sys.path` entries, so for GRC the files have to land beside the ones of GNU Radio:

```bash
PYTHONPATH=~/.local/lib/python3/dist-packages/gnuradio/sdrlora \
    python3 -c "import sdrlora_python"
```

**Upgrading.** GRC caches block definitions, and may keep showing the old parameters after a new version is installed over an old one. In this case, close GRC, delete its cache and start it again:

```bash
rm ~/.cache/grc_gnuradio/cache_v2.json
```

### If the build fails inside spdlog

A long list of template errors from `/usr/include/spdlog/common.h`, typically `error: 'basic_format_string' is not a member of 'fmt'`, means that a **second copy of fmt** is shadowing the one the system spdlog was built against. In fact, `find_package` searches prefixes derived from `PATH`, so anything on `PATH` that ships its own fmt takes precedence. An active conda environment is the usual source: Miniconda ships fmt 11, while Ubuntu 24.04 ships fmt 9, and the spdlog of Ubuntu only compiles against the latter. The configuration step detects this case and stops, naming the offending directory. Either of the following fixes works:

```bash
conda deactivate          # then re-run cmake in a fresh build directory
```

```bash
cmake -S gr-sdrlora -B build -DCMAKE_IGNORE_PREFIX_PATH="$CONDA_PREFIX"
```

## Examples

[`examples/`](examples) contains ready-to-run flowgraphs, which `cmake --install` copies to `share/sdrlora/examples` under the install prefix. To open one of them in GRC:

```bash
gnuradio-companion gr-sdrlora/examples/lora_loopback.grc
```

| Flowgraph | Hardware | What it does |
|---|---|---|
| [`lora_loopback.grc`](examples/lora_loopback.grc) | none | Encodes "Hello LoRa!" once a second, adds noise, decodes it and prints each packet as text, together with its metadata. A slider sets the noise level, while a waterfall shows the chirps. This is the best starting point. |
| [`lora_tx_usrp.grc`](examples/lora_tx_usrp.grc) | USRP (UHD) | Transmits "Hello LoRa!" once a second. |
| [`lora_rx_usrp.grc`](examples/lora_rx_usrp.grc) | USRP (UHD) | Receives and prints every packet as text, with a waterfall of the channel. |
| [`lora_tx_hackrf.grc`](examples/lora_tx_hackrf.grc) | HackRF (SoapySDR) | Transmits "Hello LoRa!" once a second. |
| [`lora_rx_hackrf.grc`](examples/lora_rx_hackrf.grc) | HackRF (SoapySDR) | Receives and prints every packet as text, with a waterfall of the channel. |

All five flowgraphs use standard LoRa framing (Pseudo-header off) at 868.1 MHz, with Spreading Factor (SF) 7, 125 kHz, Coding Rate (CR) 4/5 and a Cyclic Redundancy Check (CRC). Therefore, they communicate with an SX127x node configured in the same way. The settings are variables at the top of each flowgraph. The Universal Software Radio Peripheral (USRP) examples run at 1 MS/s, while the HackRF ones run at 2 MS/s, the lowest rate of the HackRF, which is still a power-of-two multiple of 125 kHz. Before transmitting, check that the frequency and the transmit gain are legal in your country.

## Use it

In GRC, the blocks appear under `[sdrlora]` as **LoRa Encoder**, **LoRa Decoder** and **LoRa Packet Print**.

* **LoRa Encoder** takes Protocol Data Units (PDUs) on its `pdus` message port (as `metadata . u8vector payload` pairs), and emits `complex` baseband samples. Each PDU becomes one burst, with `tx_sob`/`tx_eob` tags on its first and last sample.
* **LoRa Decoder** takes `complex` baseband samples, and emits a PDU on its `pdus` message port for every packet it decodes. The metadata carries `crc_ok`, `sf`, `cr`, `src`, `dst` and `seqn`.
* **LoRa Packet Print** prints each of those PDUs as one line of text, e.g., `SF7 CR4/5 CRC ok  dst 7 src 3 seqn 1  11 bytes  "Hello LoRa!"`. Bytes that are not printable text appear as `\xNN`. Moreover, Show Hex adds the payload in hexadecimal, and Line Prefix labels the lines when several decoders print side by side. Note that the Message Debug block of GNU Radio prints the payload in hexadecimal only. From Python, the block is `sdrlora.packet_print(show_hex=False, label="")`.

A decoder instance handles a single SF; hence, several instances side by side listen on several SFs at once. Moreover, `decoder.packets()` and `decoder.crc_errors()` count the packets that passed and, with Report CRC Failures, the ones that failed.

The documentation tab of each block in GRC describes its parameters in detail.

### LoRa Encoder parameters

| Parameter | GRC label | Default | Meaning |
|---|---|---|---|
| `sf` | Spreading Factor | 7 | From 5 to 12. SF5 and SF6 need Implicit Header. |
| `cr` | Coding Rate | 1 | From 1 to 4, for 4/5 to 4/8. |
| `bw` | Bandwidth (Hz) | 125000 | |
| `samp_rate` | Sample Rate (Hz) | | An integer multiple of `bw`. |
| `f0` | Carrier (Hz) | 868e6 | The RF carrier, of which Carrier Offset is a fraction in ppm. |
| `has_crc` | Payload CRC | True | |
| `implicit_header` | Implicit Header | False | No header on air; hence, the receiver must know length, CR and CRC. |
| `preamble_symbols` | Preamble Symbols | 8 | |
| `pseudo_header` | 4-byte Pseudo-header | True | The `DST/SRC/SEQNO/LENGTH` payload prefix of SDR-LoRa. Turn it off for standard LoRa. |
| `src`, `dst`, `seqn` | Source/Destination Address, Initial Sequence Number | 0 | Pseudo-header fields; `seqn` increments at each packet. |
| `sync_word` | Sync Word | `[0x12]` | One byte (0x12 for private networks, 0x34 for LoRaWAN), or two raw symbols. |
| `ldro` | Low Data Rate Optimization | auto | Auto enables it when a symbol lasts more than 16 ms. |
| `invert_iq` | Invert IQ | False | Inverted for LoRaWAN downlinks. |
| `cfo_ppm` | Carrier Offset (ppm of f0) | 0 | A simulated crystal error, applied once per end of the link; 0 is a clean carrier. |
| `lead_in` | Lead-in (zero samples) | auto | Zeros before the preamble. |
| `burst_tail` | Burst Tail (zero samples) | auto | Zeros closing each burst. Auto flushes a downstream filter, while 0 sends the frame alone, for a radio right after the encoder. |
| `amplitude` | Output Amplitude | 1.0 | |
| `hop_period`, `hop_freqs` | FHSS Hop Period, FHSS Channels | off | Frequency hopping: symbols per hop (0 for off, otherwise at least 13) and channel offsets from the carrier, the first one being 0. |
| `trise`, `t0_frac`, `phi0` | Chirp Edge, Timing Offset, Initial Phase | 50 µs, 0, 0 | Waveform shaping and test stimulus. |

### LoRa Decoder parameters

| Parameter | GRC label | Default | Meaning |
|---|---|---|---|
| `sf` | Spreading Factor | 7 | From 5 to 12. SF5 and SF6 need Implicit Header. |
| `bw` | Bandwidth (Hz) | 125000 | |
| `samp_rate` | Sample Rate (Hz) | | A power-of-two multiple of `bw` (1×, 2×, 4×, 8×, …). |
| `implicit_header` | Implicit Header | False | In this case, `payload_len`, `cr` and `has_crc` must match the transmitter. |
| `payload_len`, `cr`, `has_crc` | Payload Length, Coding Rate, Payload CRC (implicit only) | 0, 1, True | Ignored with an explicit header, since the header carries them. |
| `pseudo_header` | 4-byte Pseudo-header | True | Turn it off to receive standard LoRa. |
| `sync_word` | Sync Word | `[0x12]` | Must match the transmitter. |
| `ldro` | Low Data Rate Optimization | auto | Auto also retries a frame with LDRO flipped, for transmitters that set it wrongly. |
| `invert_iq` | Invert IQ | False | Must match the transmitter. |
| `center_freq` | Carrier for Clock Tracking (Hz) | 0 (off) | The RF carrier the samples were tuned to. It lets the decoder follow the sample-clock drift from the carrier offset it measures; set it for long frames on a real radio. |
| `report_crc_errors` | Report CRC Failures | False | Also emits the packets whose CRC failed, flagged with `crc_ok = False`. It needs an explicit header and Pseudo-header off. |
| `max_payload_len` | Longest Payload Expected | 251 | Bounds how much signal the block keeps for a frame in progress. |
| `window` | Decode Window (samples) | auto | New samples between scans, trading latency against CPU. |
| `low_latency` | Low Latency | False | Halves the automatic window, at about 1.45× the CPU. |
| `channel_filter_rx` | Channel Filter | True | Band-limits the input to the LoRa channel before synchronization, which is worth about 6 dB. |
| `soft_decision` | Soft-Decision FEC | True | Maximum-likelihood decoding of the payload, worth about 2 dB. |
| `fec_correct` | Hamming FEC Correction | True | Hard-decision Hamming correction, always on the header and on the payload when soft decision is off. |
| `robust_sync` | Accumulated Preamble Detection | True | Detects the preamble over several symbols, rather than one. |
| `sync_phases` | Sync Scan Granularity | auto | Offsets per symbol that the preamble search tries. |
| `interpolate`, `min_osf`, `interp_target_osf` | Interpolate Below 2x, Minimum fs/BW, Interpolate Up To | True, 2, 4 | Below `min_osf`, the input is interpolated to `interp_target_osf`. |
| `cutoff_ratio`, `numtaps` | RX Filter Cutoff, RX Filter Taps | 1.15, 255 | Cutoff (in multiples of `bw`) and length of the channel filter. |
| `sync_threshold` | Detect Threshold (peak/mean) | 5.5 | Threshold of the preamble detector: a lower value finds weaker frames, at the cost of more false locks. |
| `search_tail_symbols` | Search Tail (symbols) | 24 | Signal kept between scans, so that a frame across a window boundary is found. |
| `hop_period`, `hop_freqs` | FHSS Hop Period, FHSS Channels | off | Must match the transmitter. |
| `trise` | Chirp Edge (s) | 50 µs | Must match the transmitter. |

### LoRa Packet Print parameters

| Parameter | GRC label | Default | Meaning |
|---|---|---|---|
| `show_hex` | Show Hex | False | Adds the payload in hexadecimal at the end of each line. |
| `label` | Line Prefix | empty | Text at the start of each line, e.g., to tell several decoders apart. |

### From Python

```python
from gnuradio import gr, blocks, sdrlora

enc = sdrlora.encoder(sf=7, cr=1, bw=125e3, samp_rate=1e6, f0=868.1e6,
                      pseudo_header=False)
dec = sdrlora.decoder(sf=7, bw=125e3, samp_rate=1e6, pseudo_header=False,
                      center_freq=868.1e6)
```

The PHY core is also exposed without a flowgraph, through `tx_params`, `phy_encode`, `phy_encode_symbols`, `phy_modulate`, `phy_decode`, `phy_decode_consumed`, `phy_packet_rx`, `channel_filter` and the `rx_result` they return:

```python
from gnuradio import sdrlora
import numpy as np

p = sdrlora.tx_params()
p.SF, p.bw, p.fs = 7, 125e3, 1e6
p.pseudo_header = False                    # standard LoRa framing
iq = np.array(sdrlora.phy_encode(p, list(b"Hello LoRa!")), dtype=np.complex64)

for pkt in sdrlora.phy_decode(p, iq):
    print(pkt.crc_ok, bytes(pkt.payload))
```

## C++ library without GNU Radio

The PHY core is also available as a plain C++ library, `sdrlora-phy`, which only depends on the C++ standard library and can be used from C++11 on. To build and install this library alone, without GNU Radio, pass `-DSDRLORA_ENABLE_GNURADIO=OFF`:

```bash
cmake -S gr-sdrlora -B build-phy -DSDRLORA_ENABLE_GNURADIO=OFF
cmake --build build-phy -j$(nproc)
sudo cmake --install build-phy
```

By default, the library is static (`libsdrlora-phy.a`), while `-DBUILD_SHARED_LIBS=ON` builds a shared one instead. Note that the full build, with GNU Radio, installs the same library as well. A CMake project then finds it as follows:

```cmake
find_package(sdrlora-phy 2.0 REQUIRED)
target_link_libraries(my_app PRIVATE sdrlora::sdrlora-phy)
```

Without CMake, it is enough to link `-lsdrlora-phy`, e.g., `g++ -std=c++11 my_app.cc -lsdrlora-phy`.

[`examples/cpp/hello_lora.cc`](examples/cpp/hello_lora.cc) encodes "Hello LoRa!", adds noise and decodes it back. Moreover, it is installed to `share/sdrlora/examples/cpp`, together with a `CMakeLists.txt` that builds it against the installed library. Its core is the following:

```cpp
#include <sdrlora/lora_phy.h>
#include <sdrlora/lora_rx.h>

gr::sdrlora::tx_params p;                  // SF7, CR 4/5, 125 kHz, 1 MS/s
p.pseudo_header = false;                   // standard LoRa framing
const std::vector<uint8_t> msg = {'H', 'i', '!'};

std::vector<std::complex<float> > iq;
gr::sdrlora::encode(p, msg, iq);           // payload -> IQ samples

for (const gr::sdrlora::rx_result& r : gr::sdrlora::decode_buffer(iq, p))
    if (r.crc_ok)
        std::printf("%zu bytes received\n", r.payload.size());
```

Everything lives in the `gr::sdrlora` namespace (the same as the GNU Radio module), and an invalid parameter throws `std::invalid_argument`, with a message that names it. The table below describes the main types and functions.

| Type / function | What it does |
|---|---|
| `tx_params` | All the PHY parameters, for both transmission and reception, with the same meaning and defaults as the block parameters above. The fields are `SF`, `cr`, `bw`, `fs`, `f0`, `has_crc`, `implicit_header`, `preamble_symbols`, `src`, `dst`, `seqn`, `pseudo_header`, `sync_word`, `ldro`, `invert_iq`, `cfo_ppm`, `lead_in`, `amplitude`, `trise`, `t0_frac`, `phi0`, `hop_period` and `hop_freqs`, plus `center_freq`, `cutoff_ratio`, `numtaps`, `sync_threshold`, `search_tail_symbols`, `min_osf` and `interp_target_osf` for reception. Note that the sample rate is `fs`, and the spreading factor `SF`. |
| `encode(p, msg, out)` | Validates `p`, and writes into `out` the IQ samples of the frame carrying `msg`. |
| `decode_buffer(in, p, channel_filter_rx, robust_sync, sync_phases, soft_decision, implicit_header, ih_payload_len, fec_correct, interpolate, report_crc_errors)` | Scans a buffer and returns a `std::vector<rx_result>` with every packet found. All the flags but `p` are optional, with the same meaning and defaults as the decoder block parameters. In implicit-header mode, `ih_payload_len`, `p.cr` and `p.has_crc` must match the transmitter. |
| `rx_result` | One decoded packet: `crc_ok`, `hdr_ok`, `has_crc`, `cr`, `payload`, `src`/`dst`/`seqn` (−1 without the pseudo-header), `length`, `truncated`, and `gap_score` (the noise check of frames without a CRC). |
| `channel_filter(in, bw, fs, out, cutoff_ratio = 1.15, numtaps = 255)` | The receive filter on its own. |
| `ldro_enabled`, `default_lead_in`, `default_sync_phases`, `sync_word_symbols`, `max_packet_symbols` | Whether LDRO is enabled for a configuration, the automatic lead-in and `sync_phases`, the two symbols of a sync word, and the length of the longest frame in symbols. |

**Streaming.** `decode_buffer` also supports a stream fed buffer by buffer. More in detail, through its optional `consumed_out` argument it reports how many samples the caller can drop, while `search_from` and `walk_end` resume a scan where the previous one stopped. The LoRa Decoder block builds on this contract; however, a C++-only application has to keep the rolling buffer itself.

## Interoperability with other LoRa implementations

`tx_params::pseudo_header`, which is the **4-byte Pseudo-header** switch of both blocks, selects the framing:

* `true` (default): the payload carries the 4-byte `DST/SRC/SEQNO/LENGTH` prefix of SDR-LoRa.
* `false`: standard LoRa. The message starts at payload byte 0, the LENGTH field of the PHY header is the message length, and `src`/`dst`/`seqn` are unused (reported as −1).

The pseudo-header comes from the original SDR-LoRa release, which put a 4-byte DST/SRC/SEQNO/LENGTH block at the front of every payload. Since LoRa itself has no addresses or sequence numbers, on air the block is ordinary payload, i.e., a standard receiver sees it as the first four bytes of the message. We keep it as the default for compatibility with the original release.

**Both ends must use the same setting**, since nothing on air tells which framing a frame uses. On the one hand, a receiver with the pseudo-header off delivers a prefixed frame with those four bytes at the start of the message. On the other hand, a receiver with the pseudo-header on drops standard frames, except for the rare one whose fourth byte equals its length or 0, which arrives with its first three bytes taken as dst/src/seqn. Note that, with the pseudo-header off, the decoder reports dst, src and seqn as −1, since they do not exist in standard LoRa.

Therefore, **use `false` to receive from any other device**, such as an SX127x node or gr-lora_sdr, since the prefix is not part of LoRa. Conversely, gr-lora_sdr also decodes the default frames, since for LoRa those four bytes are just payload.

## Constraints

* **`samp_rate` must be a power-of-two multiple of `bw`** for the decoder, since the synchronization FFTs need power-of-two lengths. Moreover, 1× is interpolated to 4× internally, while 2× and above work at every SF. The encoder only needs an integer multiple. Both blocks throw an error on any other value, rather than returning no packets. The upper bound is `fs/bw ≤ 65536`, with `fs ≤ 10¹² Hz` (`OSF_MAX` and `FS_MAX` in `lora_phy.h`).
* **Long frames and clock error.** Given `center_freq`, the decoder follows the sample clock from the carrier offset it measures. Moreover, a frame that is located but fails its CRC is retried with the drift estimated from its own symbols, and kept only if the CRC then passes. This also covers a front end whose carrier and clock disagree by a few ppm.
* **Frames without a CRC** are reported as `crc_ok` only if their symbols look like signal. More in detail, for each dechirped peak, the gap above the largest bin outside its main lobe (in noise-power units, and capped at 8) is averaged over the frame and compared with a per-SF threshold (`nocrc_min_gap`), and `rx_result.gap_score` carries the score. This rejects false locks on noise, but not bit errors in real frames; hence, send a CRC.
* **Payloads** range from 0 to 251 bytes with the pseudo-header, and from 1 to 255 bytes without it. The encoder block drops a longer PDU with a warning, since a flowgraph cannot raise an exception, while `phy_encode` refuses it. Finally, a 1-byte payload with the CRC enabled is refused, since a LoRa CRC needs two payload bytes.

## Implementation notes

Two details decide whether a change keeps the waveform bit-exact, and both are documented at the top of [`lib/lora_phy.cc`](lib/lora_phy.cc):

1. **Bit order.** The payload bit array is LSB-first *within each byte*, i.e., `PAYLOAD[base + i] = bit i`.
2. **Phase accumulation order.** `chirp()` returns `((phi + a) + b)`, not `phi + (a + b)`. In fact, pre-summing the two terms shifts the result by one Unit in the Last Place (ULP), which changes the emitted `float32`.
