"""SDR-LoRa blocks for GNU Radio."""
_NAMES = ("encoder", "decoder", "tx_params", "rx_result",
          "phy_encode", "phy_encode_symbols", "phy_modulate",
          "phy_decode", "phy_decode_consumed",
          "phy_packet_rx", "max_packet_symbols",
          "live_overlap_samples", "channel_filter",
          "default_lead_in", "resolved_lead_in",
          "hop_cutoff_ratio", "apply_hops")

try:
    from . import sdrlora_python as _ext
except ImportError:
    # Installed layout puts the extension alongside the other gnuradio modules.
    import sdrlora_python as _ext

# ONE list, used for both the import and __all__, so the two cannot drift
# apart. Every symbol the bindings (encoder_python.cc, decoder_python.cc)
# export belongs in _NAMES: a name bound there but missing here is reachable
# only as sdrlora.sdrlora_python.*, not as sdrlora.*.
for _n in _NAMES:
    globals()[_n] = getattr(_ext, _n)
del _n

from .packet_print import packet_print  # noqa: E402

__all__ = list(_NAMES) + ["packet_print"]
