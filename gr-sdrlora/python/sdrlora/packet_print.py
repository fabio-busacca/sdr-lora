"""LoRa Packet Print: one readable line per decoded packet."""
import pmt
from gnuradio import gr


def payload_text(data):
    """The payload as text: UTF-8 when all of it is printable, otherwise
    ASCII with every other byte written as \\xNN."""
    try:
        s = data.decode("utf-8")
        if s.isprintable():
            return s
    except UnicodeDecodeError:
        pass
    return "".join(chr(b) if 32 <= b < 127 else "\\x%02x" % b for b in data)


def format_packet(meta, data, show_hex=False, label=""):
    """One line for a decoded packet: its metadata dict and payload bytes."""
    parts = ["[%s]" % label] if label else []
    if "sf" in meta:
        parts.append("SF%d" % meta["sf"])
    if meta.get("cr", 0) in (1, 2, 3, 4):
        parts.append("CR4/%d" % (4 + meta["cr"]))
    if "crc_ok" in meta:
        parts.append("CRC ok" if meta["crc_ok"] else "CRC FAIL")
    # -1 means the frame carried no pseudo-header, so there is nothing to show.
    if all(meta.get(k, -1) >= 0 for k in ("dst", "src", "seqn")):
        parts.append(" dst %d src %d seqn %d" % (meta["dst"], meta["src"],
                                                 meta["seqn"]))
    parts.append(" %d byte%s" % (len(data), "" if len(data) == 1 else "s"))
    parts.append(' "%s"' % payload_text(data))
    if show_hex:
        parts.append(" | " + " ".join("%02x" % b for b in data))
    return " ".join(parts)


class packet_print(gr.basic_block):
    """Prints each PDU from a LoRa Decoder as one line of text: spreading
    factor, coding rate, CRC verdict, the pseudo-header fields when the frame
    carried them, the length and the payload as text, and optionally in hex.
    """

    def __init__(self, show_hex=False, label=""):
        gr.basic_block.__init__(self, name="sdrlora_packet_print",
                                in_sig=None, out_sig=None)
        self.show_hex = bool(show_hex)
        self.label = str(label)
        self.message_port_register_in(pmt.intern("pdus"))
        self.set_msg_handler(pmt.intern("pdus"), self._handle)

    def _handle(self, msg):
        if not (pmt.is_pair(msg) and pmt.is_u8vector(pmt.cdr(msg))):
            print("%snot a PDU: %s" % ("[%s] " % self.label if self.label else "",
                                       msg), flush=True)
            return
        meta = pmt.to_python(pmt.car(msg))
        if not isinstance(meta, dict):
            meta = {}
        data = bytes(pmt.u8vector_elements(pmt.cdr(msg)))
        print(format_packet(meta, data, self.show_hex, self.label), flush=True)
