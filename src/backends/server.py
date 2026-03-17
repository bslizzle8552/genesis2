from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Dict, List

from src.engine.simulation import SimulationEngine, load_config


ROOT = Path(__file__).resolve().parents[2]
UI_PATH = ROOT / "src" / "ui" / "index.html"
CONFIG_PATH = ROOT / "config"
DEFAULT_CONFIG = CONFIG_PATH / "default.json"

LAST_RESULT = {
    "status": "idle",
    "result": None,
    "progress": None,
    "error": None,
}
RUN_LOCK = threading.Lock()


def discover_presets() -> List[Dict[str, object]]:
    presets: List[Dict[str, object]] = []
    for path in sorted(CONFIG_PATH.glob("*.json")):
        if path.name.startswith("_"):
            continue
        label = path.stem.replace("_", " ").title()
        cfg = json.loads(path.read_text(encoding="utf-8"))
        presets.append(
            {
                "id": path.name,
                "label": label,
                "seed": cfg.get("seed"),
                "agents": cfg.get("agents"),
                "generations": cfg.get("generations"),
                "tasks_per_generation": cfg.get("tasks_per_generation"),
                "initial_energy": cfg.get("initial_energy"),
                "upkeep_cost": cfg.get("upkeep_cost"),
                "reproduction_threshold": cfg.get("reproduction_threshold", 130),
                "mutation_rate": cfg.get("mutation_rate", 0.15),
                "diversity_bonus": cfg.get("diversity_bonus", 1.0),
                "diversity_min_lineages": cfg.get("diversity_min_lineages", 4),
                "immigrant_injection_count": cfg.get("immigrant_injection_count", 2),
                "tier_mix": cfg.get("tier_mix", {"1": 0.34, "2": 0.31, "3": 0.21, "4": 0.14}),
            }
        )

    if not any(p["id"] == "default.json" for p in presets) and DEFAULT_CONFIG.exists():
        presets.insert(
            0,
            {
                "id": "default.json",
                "label": "Default",
                "seed": 42,
                "agents": 10,
                "generations": 50,
                "tasks_per_generation": 15,
                "initial_energy": 100,
                "upkeep_cost": 6,
                "reproduction_threshold": 130,
                "mutation_rate": 0.15,
                "diversity_bonus": 1.0,
                "diversity_min_lineages": 4,
                "immigrant_injection_count": 2,
                "tier_mix": {"1": 0.34, "2": 0.31, "3": 0.21, "4": 0.14},
            },
        )

    return presets


def config_from_request(payload: dict):
    preset = payload.get("preset", "default.json")
    preset_path = (CONFIG_PATH / preset).resolve()

    try:
        preset_path.relative_to(CONFIG_PATH.resolve())
    except ValueError as exc:
        raise ValueError("invalid preset path") from exc

    cfg = load_config(preset_path if preset_path.exists() else DEFAULT_CONFIG)
    integer_fields = ["seed", "agents", "generations", "tasks_per_generation", "initial_energy", "upkeep_cost", "diversity_min_lineages", "immigrant_injection_count"]
    float_fields = ["reproduction_threshold", "mutation_rate", "diversity_bonus"]

    for key in integer_fields:
        if key in payload and payload[key] is not None:
            setattr(cfg, key, int(payload[key]))

    for key in float_fields:
        if key in payload and payload[key] is not None:
            setattr(cfg, key, float(payload[key]))

    if "tier_mix" in payload and isinstance(payload["tier_mix"], dict):
        cfg.tier_mix = {str(k): float(v) for k, v in payload["tier_mix"].items()}

    if "run_label" in payload and payload["run_label"] is not None:
        cfg.run_label = str(payload["run_label"])

    if "overwrite" in payload:
        cfg.overwrite = bool(payload["overwrite"])

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
            with RUN_LOCK:
                if LAST_RESULT["status"] == "running":
                    self._json({"error": "A run is already in progress"}, status=409)
                    return

                content_len = int(self.headers.get("Content-Length", 0))
                incoming = json.loads(self.rfile.read(content_len) or b"{}")

                try:
                    cfg = config_from_request(incoming)
                except ValueError as exc:
                    self._json({"error": str(exc)}, status=400)
                    return

                LAST_RESULT["status"] = "running"
                LAST_RESULT["result"] = None
                LAST_RESULT["error"] = None
                LAST_RESULT["progress"] = {
                    "generation": 0,
                    "population": cfg.agents,
                    "births": 0,
                    "deaths": 0,
                }

                def worker() -> None:
                    try:
                        engine = SimulationEngine(cfg)

                        def progress_callback(step: Dict) -> None:
                            LAST_RESULT["progress"] = {
                                "generation": step.get("generation", 0),
                                "population": step.get("population", 0),
                                "births": step.get("births", 0),
                                "deaths": step.get("deaths", 0),
                                "avg_energy": step.get("energy_distribution", {}).get("mean", 0),
                            }

                        result = engine.run(progress_callback=progress_callback)
                        LAST_RESULT["status"] = "done"
                        LAST_RESULT["result"] = result
                    except Exception as exc:  # pragma: no cover
                        LAST_RESULT["status"] = "error"
                        LAST_RESULT["error"] = str(exc)

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()

            self._json({"status": "running"}, status=202)
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
