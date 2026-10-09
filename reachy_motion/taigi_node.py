"""``reachy-motion-node``: share this Mac's local Taiwanese speech services with Reachy on the LAN.

The AIRI lab services (Breeze-ASR-26, Ollama + SARC-Taigi-LLM, KaedeTai TTS) listen on 127.0.0.1 only. This node
starts small HTTP reverse proxies on the LAN for them — without touching their configuration — and advertises
``_reachy-taigi._tcp.local.`` over mDNS, so a robot asked to 「講台語」 can find and health-check them.

An HTTP-aware proxy (not a raw TCP forward) is needed because Ollama rejects requests whose ``Host`` header is not
localhost when it is bound to loopback; the proxy rewrites ``Host`` and streams responses (LLM tokens) through.

    reachy-motion-node                  # auto-detects the local ports, prints what it shares
    reachy-motion-node --bind 0.0.0.0 --asr http://127.0.0.1:8001 --tts http://127.0.0.1:8883
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket

import httpx

logger = logging.getLogger("reachy_motion.node")

DEFAULTS = {
    "asr": ["http://127.0.0.1:18001", "http://127.0.0.1:8001"],
    "llm": ["http://127.0.0.1:11434", "http://127.0.0.1:12434"],
    "tts": ["http://127.0.0.1:8883", "http://127.0.0.1:18883"],
}
PUBLIC_PORTS = {"asr": 18101, "llm": 18102, "tts": 18103}
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}


def _health_path(kind: str) -> str:
    return "/api/tags" if kind == "llm" else "/health"


def detect(kind: str, candidates: list[str]) -> str | None:
    for base in candidates:
        try:
            httpx.get(base + _health_path(kind), timeout=2).raise_for_status()
            return base
        except httpx.HTTPError:
            continue
    return None


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is sent; picks the outbound interface
        return s.getsockname()[0]
    finally:
        s.close()


def make_proxy(upstream: str):
    from aiohttp import ClientSession, ClientTimeout, web

    async def handler(request: web.Request) -> web.StreamResponse:
        url = upstream + request.rel_url.path_qs
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        body = await request.read()
        async with ClientSession(timeout=ClientTimeout(total=300)) as s:
            async with s.request(request.method, url, headers=headers, data=body, allow_redirects=False) as up:
                resp = web.StreamResponse(status=up.status, headers={
                    k: v for k, v in up.headers.items() if k.lower() not in HOP_BY_HOP})
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", handler)
    return app


async def serve(services: dict[str, str], bind: str, llm_model: str) -> None:
    from aiohttp import web
    from zeroconf import ServiceInfo
    from zeroconf.asyncio import AsyncZeroconf

    runners = []
    for kind, upstream in services.items():
        runner = web.AppRunner(make_proxy(upstream))
        await runner.setup()
        await web.TCPSite(runner, bind, PUBLIC_PORTS[kind]).start()
        runners.append(runner)

    ip = lan_ip()
    props = {k: str(PUBLIC_PORTS[k]) for k in services} | {"llm_model": llm_model, "v": "1"}
    info = ServiceInfo(
        "_reachy-taigi._tcp.local.",
        f"Reachy Taigi on {socket.gethostname().split('.')[0]}._reachy-taigi._tcp.local.",
        addresses=[socket.inet_aton(ip)],
        port=PUBLIC_PORTS["asr"],
        properties=props,
        server=f"{socket.gethostname().split('.')[0]}.local.",
    )
    azc = AsyncZeroconf()
    await azc.async_register_service(info)
    print(f"\n  Sharing Taiwanese speech services on {ip} (mDNS _reachy-taigi._tcp):")
    for kind, upstream in services.items():
        print(f"    {kind.upper():3}  http://{ip}:{PUBLIC_PORTS[kind]}  ->  {upstream}")
    print("  Ctrl+C to stop.\n")
    try:
        await asyncio.Event().wait()
    finally:
        await azc.async_unregister_service(info)
        await azc.async_close()
        for r in runners:
            await r.cleanup()


def main() -> None:
    p = argparse.ArgumentParser(prog="reachy-motion-node", description=__doc__.split("\n\n")[0])
    p.add_argument("--bind", default="0.0.0.0", help="interface for the relays (default: all)")
    p.add_argument("--asr", help="ASR base URL (default: auto-detect 18001 / 8001)")
    p.add_argument("--llm", help="Ollama base URL (default: auto-detect 11434 / 12434)")
    p.add_argument("--tts", help="TTS base URL (default: auto-detect 8883 / 18883)")
    p.add_argument("--llm-model", default="hf.co/Speech-AI-Research-Center/SARC-Taigi-LLM-12b-GGUF:Q4_K_M")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    services, missing = {}, []
    for kind in ("asr", "llm", "tts"):
        given = getattr(a, kind)
        base = detect(kind, [given] if given else DEFAULTS[kind])
        if base:
            services[kind] = base
        else:
            missing.append(kind)
    if missing:
        print(f"  ⚠ not running locally: {', '.join(missing)} — sharing the others; start them for the full pipeline.")
    if not services:
        raise SystemExit("nothing to share")
    try:
        asyncio.run(serve(services, a.bind, a.llm_model))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
