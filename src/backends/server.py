from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from src.engine.simulation import SimulationConfig, SimulationEngine, load_config


ROOT = Path(__file__).resolve().parents[2]
UI_PATH = ROOT / "src" / "ui" / "index.html"
DEFAULT_CONFIG = ROOT / "config" / "default.json"

LAST_RESULT = {
    "status": "idle",
    "result": None,
}


class GenesisHandler(BaseHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ["/", "/index.html"]:
            html = UI_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/api/state":
            self._json(LAST_RESULT)
            return

        self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path == "/api/run":
            content_len = int(self.headers.get("Content-Length", 0))
            incoming = json.loads(self.rfile.read(content_len) or b"{}")

            cfg = load_config(DEFAULT_CONFIG)
            for key in ["seed", "agents", "generations", "tasks_per_generation"]:
                if key in incoming:
                    setattr(cfg, key, int(incoming[key]))

            LAST_RESULT["status"] = "running"
            engine = SimulationEngine(cfg)
            result = engine.run()
            LAST_RESULT["status"] = "done"
            LAST_RESULT["result"] = result
            self._json(LAST_RESULT)
            return

        self._json({"error": "not found"}, status=404)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), GenesisHandler)
    print(f"Genesis2 UI: http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.host, args.port)
