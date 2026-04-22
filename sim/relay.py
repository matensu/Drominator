"""
relay.py — Reçoit les frames Three.js via WebSocket et les sert en MJPEG.
OpenCV peut ensuite lire le flux avec cv2.VideoCapture("http://localhost:8766/stream")
"""
import asyncio
import sys
import cv2
import numpy as np
from aiohttp import web
import websockets

latest_jpeg: bytes | None = None

# ── WebSocket : reçoit les frames du navigateur ───────────────────────────────
async def ws_handler(websocket):
    global latest_jpeg
    print(f"[WS]  Navigateur connecté")
    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                latest_jpeg = msg
            elif isinstance(msg, str) and ',' in msg:
                import base64
                latest_jpeg = base64.b64decode(msg.split(',', 1)[1])
    except websockets.exceptions.ConnectionClosed:
        pass
    print(f"[WS]  Navigateur déconnecté")

# ── HTTP : sert le flux MJPEG ─────────────────────────────────────────────────
async def mjpeg_stream(request):
    resp = web.StreamResponse()
    resp.content_type = 'multipart/x-mixed-replace; boundary=frame'
    resp.headers['Cache-Control'] = 'no-cache'
    await resp.prepare(request)
    print(f"[HTTP] Client MJPEG connecté : {request.remote}")
    try:
        while True:
            if latest_jpeg:
                chunk = (b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n'
                         + latest_jpeg + b'\r\n')
                await resp.write(chunk)
            await asyncio.sleep(1 / 30)
    except (ConnectionResetError, Exception):
        pass
    print(f"[HTTP] Client MJPEG déconnecté")
    return resp

async def index(request):
    return web.Response(text="Drominator relay — stream: /stream", content_type="text/plain")

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    # WebSocket server (reçoit les frames du navigateur)
    ws_server = await websockets.serve(ws_handler, "localhost", 8765)
    print("[SIM] WebSocket  → ws://localhost:8765")

    # HTTP MJPEG server (sert le flux à OpenCV)
    app = web.Application()
    app.router.add_get('/',       index)
    app.router.add_get('/stream', mjpeg_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8766)
    await site.start()
    print("[SIM] MJPEG      → http://localhost:8766/stream")
    print()
    print("[SIM] 1. Ouvre sim/scene.html dans ton navigateur")
    print("[SIM] 2. Lance : python test_webcam.py --sim")
    print("[SIM] Ctrl+C pour arrêter\n")

    await asyncio.Event().wait()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\n[SIM] Arrêté.")
