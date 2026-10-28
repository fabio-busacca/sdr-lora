/* Encode a LoRa frame, add noise and decode it back, with the C++ PHY
 * library alone (no GNU Radio). SPDX-License-Identifier: GPL-3.0-only */
#include <sdrlora/lora_phy.h>
#include <sdrlora/lora_rx.h>

#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

int main()
{
    gr::sdrlora::tx_params p;     /* SF7, CR 4/5, 125 kHz, 1 MS/s by default */
    p.pseudo_header = false;      /* standard LoRa framing */

    const std::string text = "Hello LoRa!";
    const std::vector<uint8_t> msg(text.begin(), text.end());

    /* Payload -> IQ samples */
    std::vector<std::complex<float> > iq;
    gr::sdrlora::encode(p, msg, iq);

    /* Complex white Gaussian noise, 10 dB below the unit-power signal */
    std::mt19937 rng(1);
    std::normal_distribution<float> noise(0.0f, 0.22f);
    for (size_t i = 0; i < iq.size(); i++)
        iq[i] += std::complex<float>(noise(rng), noise(rng));

    /* IQ samples -> packets */
    const std::vector<gr::sdrlora::rx_result> pkts =
        gr::sdrlora::decode_buffer(iq, p);
    for (size_t i = 0; i < pkts.size(); i++) {
        const gr::sdrlora::rx_result& r = pkts[i];
        std::printf("CRC %s, CR 4/%d, %d bytes: \"%s\"\n",
                    r.crc_ok ? "ok" : "FAIL", 4 + r.cr,
                    (int)r.payload.size(),
                    std::string(r.payload.begin(), r.payload.end()).c_str());
    }
    return (pkts.size() == 1 && pkts[0].crc_ok) ? 0 : 1;
}
