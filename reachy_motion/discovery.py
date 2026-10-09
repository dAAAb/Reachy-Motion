"""Find the Taiwanese speech services on the LAN.

A Mac running ``reachy-motion-node`` advertises ``_reachy-taigi._tcp.local.`` over mDNS with the ports of its ASR /
LLM / TTS relays in the TXT record. The robot browses for it, then health-checks each service.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_reachy-taigi._tcp.local."
_ALT_PORTS = {":18001/": ":8001/", ":8001/": ":18001/", ":8883/": ":18883/", ":18883/": ":8883/"}


@dataclass
class TaigiProbe:
    asr_url: str
    llm_url: str
    tts_url: str
    llm_model: str | None = None
    source: str = "configured"  # configured | mdns:<host>
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def _healthy(http: httpx.Client, url: str, kind: str) -> bool:
    base = url.split("/v1/")[0]
    try:
        if kind == "llm":
            http.get(base + "/api/tags", timeout=3).raise_for_status()
        else:
            http.get(base + "/health", timeout=3).raise_for_status()
        return True
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return False


def browse(timeout_s: float = 3.0) -> list[dict]:
    """Return advertised Taiwanese nodes: ``[{host, asr, llm, tts, llm_model, name}]``."""
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        logger.info("zeroconf not installed; skipping LAN discovery")
        return []
    found: dict[str, dict] = {}

    class Listener:
        def add_service(self, zc, type_, name):  # noqa: ANN001
            info = zc.get_service_info(type_, name, timeout=1500)
            if not info or not info.addresses:
                return
            props = {k.decode(): (v.decode() if v else "") for k, v in info.properties.items()}
            host = socket.inet_ntoa(info.addresses[0])
            found[name] = {"name": name, "host": host, **props}

        def update_service(self, *a):  # noqa: ANN002
            pass

        def remove_service(self, *a):  # noqa: ANN002
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, SERVICE_TYPE, Listener())
        time.sleep(timeout_s)
    finally:
        zc.close()
    return list(found.values())


def probe_taigi(asr_url: str, llm_url: str, tts_url: str, llm_model: str | None = None) -> TaigiProbe:
    """Check the configured services (and their alternate local ports), then look for a LAN node."""
    with httpx.Client() as http:
        p = TaigiProbe(asr_url, llm_url, tts_url, llm_model)
        for kind, attr in (("asr", "asr_url"), ("llm", "llm_url"), ("tts", "tts_url")):
            url = getattr(p, attr)
            if _healthy(http, url, kind):
                continue
            alt = next((url.replace(a, b) for a, b in _ALT_PORTS.items() if a in url), None)
            if alt and _healthy(http, alt, kind):
                setattr(p, attr, alt)
                continue
            p.missing.append(kind)
        if p.ok:
            return p

        for node in browse():
            host = node["host"]
            cand = TaigiProbe(
                asr_url=f"http://{host}:{node.get('asr')}/v1/audio/transcriptions",
                llm_url=f"http://{host}:{node.get('llm')}/v1/chat/completions",
                tts_url=f"http://{host}:{node.get('tts')}/v1/audio/speech",
                llm_model=node.get("llm_model") or llm_model,
                source=f"mdns:{host}",
            )
            for kind, attr in (("asr", "asr_url"), ("llm", "llm_url"), ("tts", "tts_url")):
                if not node.get(kind) or not _healthy(http, getattr(cand, attr), kind):  # node doesn't share it
                    cand.missing.append(kind)
            if cand.ok or len(cand.missing) < len(p.missing):
                p = cand
            if p.ok:
                break
        return p
