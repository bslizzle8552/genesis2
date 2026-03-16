from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Dict, List

from src.engine.simulation import SimulationEngine, load_config


ROOT = Path(__file__).resolve().parents[2]
UI_PATH = ROOT / "src" / "ui" / "index.html"
CONFIG_PATH = ROOT / "config"
DEFAULT_CONFIG = CONFIG_PATH / "default.json"

LAST_RESULT = {
    "status": "idle",
    "result": None,
}


def discover_presets() -> List[Dict[str, object]]:
    presets: List[Dict[str, object]] = []
    for path in sorted(CONFIG_PATH.glob("*.json")):
        if path.name.startswith("_"):
            continue
        label = path.stem.replace("_", " ").title()
        cfg = json.loads(path.read_text(encoding="utf-8"))
        presets.append({
            "id": path.name,
            "label": label,
            "seed": cfg.get("seed"),
            "agents": cfg.get("agents"),
            "generations": cfg.get("generations"),
            "tasks_per_generation": cfg.get("tasks_per_generation"),
        })

    if not any(p["id"] == "default.json" for p in presets) and DEFAULT_CONFIG.exists():
        presets.insert(0, {"id": "default.json", "label": "Default", "seed": 42, "agents": 10, "generations": 50, "tasks_per_generation": 8})

    return presets


def config_from_request(payload: dict):
    preset = payload.get("preset", "default.json")
    preset_path = (CONFIG_PATH / preset).resolve()

    try:
        preset_path.relative_to(CONFIG_PATH.resolve())
    except ValueError as exc:
        raise ValueError("invalid preset path") from exc

    cfg = load_config(preset_path if preset_path.exists() else DEFAULT_CONFIG)
    for key in ["seed", "agents", "generations", "tasks_per_generation"]:
        if key in payload and payload[key] is not None:
            setattr(cfg, key, int(payload[key]))
    return cfg


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

        if self.path == "/api/presets":
            self._json({"presets": discover_presets()})
            return

        self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path == "/api/run":
            content_len = int(self.headers.get("Content-Length", 0))
            incoming = json.loads(self.rfile.read(content_len) or b"{}")

            try:
                cfg = config_from_request(incoming)
            except ValueError as exc:
                self._json({"error": str(exc)}, status=400)
                return

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
