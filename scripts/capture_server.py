#!/usr/bin/env python3
"""Static file server + capture sink for the WebGPU splat denoising dataset.

Serves the whole project root over http://localhost so that both the renderer
(webgpu-splatting-dithering-nrg/capture.html) and the scene files
(data/scenes/*.splat) are reachable from one origin (no CORS, secure context).

It also accepts POST /save?path=<relpath> and writes the request body under
data/renders/<relpath>, which is how capture.js streams out the rendered
noisy / clean / depth files.

Usage:
    python3 scripts/capture_server.py            # port 8000
    python3 scripts/capture_server.py 8080       # custom port

Then open:
    http://localhost:8000/webgpu-splatting-dithering-nrg/capture.html?scene=bonsai-7k
"""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

ROOT = Path(__file__).resolve().parent.parent
RENDERS = ROOT / "data" / "renders"


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        parts = urlsplit(self.path)
        if parts.path != "/save":
            self.send_error(404, "Only POST /save is supported")
            return

        rel = parse_qs(parts.query).get("path", [None])[0]
        if not rel:
            self.send_error(400, "Missing ?path=")
            return

        target = (RENDERS / rel).resolve()
        if not str(target).startswith(str(RENDERS.resolve())):
            self.send_error(403, "Path escapes data/renders")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)

        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt, *args):
        # Quiet the per-GET spam; still surface saves.
        if self.command == "POST":
            super().log_message(fmt, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    RENDERS.mkdir(parents=True, exist_ok=True)
    handler = partial(Handler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving {ROOT} at http://localhost:{port}")
    print("Capture page:")
    print(f"  http://localhost:{port}/webgpu-splatting-dithering-nrg/capture.html?scene=bonsai-7k")
    print(f"Saving uploads under {RENDERS}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
