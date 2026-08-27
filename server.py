"""Tiny http.server app for emojic.

Serves the static page in web/ and a single JSON endpoint:

    GET /predict?text=...  ->  {emoji, feeling, bg1, bg2, text_color}

The trained checkpoint (model.pt, produced by `uv run main.py`) is loaded once
at startup and a forward pass is run per request.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from config import EMBED_SIZE, H_SIZE
from main import Model, predict

WEB_DIR = Path(__file__).parent / "web"
MODEL_PATH = Path(__file__).parent / "model.pt"

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}

model = Model(embed_dim=EMBED_SIZE, hidden_dim=H_SIZE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model.eval()

# ThreadingHTTPServer runs each request on its own thread; serialize the shared
# model's forward pass rather than rely on it being reentrant.
_model_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/predict":
            text = parse_qs(parsed.query).get("text", [""])[0]
            try:
                with _model_lock:
                    result = predict(model, text)
            except Exception:
                self._send(
                    500, b"prediction failed", "text/plain; charset=utf-8"
                )
                return
            self._send(
                200,
                json.dumps(result, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return

        entry = STATIC.get(parsed.path)
        if entry is None:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return

        filename, content_type = entry
        path = WEB_DIR / filename
        if not path.is_file():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        self._send(200, path.read_bytes(), content_type)

    def log_message(self, *args):  # keep the console quiet
        pass


if __name__ == "__main__":
    host, port = "127.0.0.1", 8000
    print(f"emojic serving on http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
