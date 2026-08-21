"""FastAPI backend for the marker-matching benchmark web app.

What it does
------------
1. Serves the single-page frontend (static/index.html).
2. Accepts an annotator session JSON upload + matching parameters.
3. Decodes the JSON into ce/ new/ points.csv (reusing extract_pairs.py).
4. Runs each selected engine in an isolated subprocess (clean peak-memory /
   CPU measurement) via vis_core.py, which produces:
       - a metrics JSON (err mean/median, <3/<5/<10px success, ms/image,
         CPU sec/%, peak mem MB)
       - one side-by-side overlay image per pair (matches, RANSAC
         inlier/outlier, projected vs. manual marker + error).
5. Returns the metrics + a per-pair index; overlay images are served as static
   files under /runs/<run_id>/vis/<engine>/pair_XXX.jpg.

Engines (device switching)
--------------------------
    opencv   -> OpenCV SIFT (CPU)
    pytorch  -> PyTorch DISK + LightGlue (no OpenVINO)
    ov_cpu   -> PyTorch + OpenVINO (CPU)
    ov_gpu   -> PyTorch + OpenVINO (GPU)      (only if a GPU device exists)
    ov_npu   -> PyTorch + OpenVINO (NPU)      (only if an NPU device exists)

Frozen / PyInstaller note
-------------------------
Engine subprocesses re-invoke *this* program with a hidden ``--worker`` flag so
the same one-folder build works without a separate python.exe. In worker mode
the program dispatches to vis_core.main() instead of starting the server.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

# FastAPI types must live in module globals so pydantic can resolve the
# UploadFile / Form annotations on the route handlers below.
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
ANNOTATOR_DIR = APP_DIR.parent

# Path resolution differs between "python app.py" and the PyInstaller exe:
#   BUNDLE_DIR = read-only bundled data (static/, ov_models/) -> sys._MEIPASS
#   BASE_DIR   = a writable folder next to the exe            -> exe's own folder
FROZEN = getattr(sys, "frozen", False)
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", str(APP_DIR)))
BASE_DIR = Path(sys.executable).resolve().parent if FROZEN else APP_DIR

RUNS_DIR = BASE_DIR / "runs"           # outputs (data / vis / metrics / logs) written here
STATIC_DIR = BUNDLE_DIR / "static"     # frontend, read-only
# dev reuses the existing ../ov_models cache; the exe keeps a writable copy next to itself
OV_MODELS_DIR = (BASE_DIR / "ov_models") if FROZEN else (ANNOTATOR_DIR / "ov_models")
BUNDLED_OV_MODELS = BUNDLE_DIR / "ov_models"  # optional pre-exported ONNX shipped in the build

# make sibling modules importable both as a script and when frozen
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ANNOTATOR_DIR))


# --------------------------------------------------------------------------- #
# Engine catalogue
# --------------------------------------------------------------------------- #
ENGINES = [
    {"id": "opencv",  "label": "OpenCV SIFT (CPU)",              "method": "sift"},
    {"id": "pytorch", "label": "PyTorch LightGlue (no OpenVINO)", "method": "lightglue"},
    {"id": "ov_cpu",  "label": "PyTorch + OpenVINO (CPU)",       "method": "openvino", "ov_device": "CPU"},
    {"id": "ov_gpu",  "label": "PyTorch + OpenVINO (GPU)",       "method": "openvino", "ov_device": "GPU"},
    {"id": "ov_npu",  "label": "PyTorch + OpenVINO (NPU)",       "method": "openvino", "ov_device": "NPU"},
]
ENGINE_BY_ID = {e["id"]: e for e in ENGINES}

_OV_DEVICES_CACHE: list[str] | None = None


def ov_devices() -> list[str]:
    """OpenVINO devices available on this machine (cached)."""
    global _OV_DEVICES_CACHE
    if _OV_DEVICES_CACHE is None:
        try:
            import openvino as ov
            _OV_DEVICES_CACHE = list(ov.Core().available_devices)
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[warn] cannot query OpenVINO devices: {exc}", file=sys.stderr)
            _OV_DEVICES_CACHE = []
    return _OV_DEVICES_CACHE


def available_engines() -> list[dict]:
    """Engine list filtered to what this machine can actually run."""
    devs = ov_devices()
    out = []
    for e in ENGINES:
        if e.get("ov_device"):
            base = e["ov_device"].split(".")[0]
            if not any(d == e["ov_device"] or d.split(".")[0] == base for d in devs):
                continue
        out.append({"id": e["id"], "label": e["label"], "method": e["method"]})
    return out


# --------------------------------------------------------------------------- #
# Extraction (in-process, reuses extract_pairs.py)
# --------------------------------------------------------------------------- #
def extract_session(session_json: dict, data_dir: Path) -> dict:
    import extract_pairs as ep

    ce_dir = data_dir / "ce"
    new_dir = data_dir / "new"
    n_ce = ep.write_items(session_json.get("ceItems", []), ce_dir)
    n_new = ep.write_items(session_json.get("newItems", []), new_dir)
    n_pts = ep.write_points_csv(session_json.get("points", []), data_dir / "points.csv")
    return {"ce": n_ce, "new": n_new, "points": n_pts}


# --------------------------------------------------------------------------- #
# Engine subprocess
# --------------------------------------------------------------------------- #
def _engine_cmd(data_dir: Path, engine: dict, vis_dir: Path, out_json: Path, params: dict) -> list[str]:
    if getattr(sys, "frozen", False):
        base = [sys.executable]            # the packaged exe re-invokes itself
    else:
        base = [sys.executable, str(APP_DIR / "app.py")]
    cmd = base + [
        "--worker",
        "--data", str(data_dir),
        "--method", engine["method"],
        "--vis-dir", str(vis_dir),
        "--out-json", str(out_json),
        "--max-dim", str(params["max_dim"]),
        "--max-features", str(params["max_features"]),
        "--lg-features", str(params["lg_features"]),
        "--min-matches", str(params["min_matches"]),
        "--ratio", str(params["ratio"]),
        "--ransac-thresh", str(params["ransac_thresh"]),
        "--ov-size", str(params["ov_size"]),
        "--onnx-dir", str(OV_MODELS_DIR),
    ]
    if engine["method"] == "openvino":
        cmd += ["--ov-device", engine["ov_device"]]
    return cmd


def run_engine(data_dir: Path, engine: dict, run_dir: Path, params: dict) -> dict:
    vis_dir = run_dir / "vis" / engine["id"]
    out_json = run_dir / "metrics" / f"{engine['id']}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = _engine_cmd(data_dir, engine, vis_dir, out_json, params)

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env.setdefault("HTTPS_PROXY", "http://proxy-png.intel.com:912")
    env.setdefault("HTTP_PROXY", "http://proxy-png.intel.com:912")

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover
        return {"id": engine["id"], "label": engine["label"], "error": str(exc)}

    # persist the engine's full stdout+stderr for later inspection
    log_path = run_dir / "logs" / f"{engine['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout or ''}\n"
        f"--- stderr ---\n{proc.stderr or ''}\n",
        encoding="utf-8",
    )
    log_url = f"/runs/{run_dir.name}/logs/{engine['id']}.log"

    if proc.returncode != 0 or not out_json.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        return {"id": engine["id"], "label": engine["label"],
                "error": "engine failed", "log": "\n".join(tail), "log_url": log_url}

    metrics = json.loads(out_json.read_text(encoding="utf-8"))
    metrics["id"] = engine["id"]
    metrics["label"] = engine["label"]
    metrics["log_url"] = log_url
    return metrics


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
def default_run_id() -> str:
    """Default output folder name: timestamp + short random suffix."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]


_NAME_RE = re.compile(r"[^A-Za-z0-9_.\-]")


def sanitize_run_name(name: str) -> str | None:
    """Keep only safe chars (blocks path traversal). Return None if unusable."""
    name = (name or "").strip().replace(" ", "_")
    name = _NAME_RE.sub("", name)          # drop / \ .. and anything unusual
    if not name or name in (".", ".."):
        return None
    return name[:120]
def build_app():
    app = FastAPI(title="Marker Matching Benchmark")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # seed the writable ONNX cache from any ONNX shipped inside the bundle
    if FROZEN and BUNDLED_OV_MODELS.exists():
        OV_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        # copy the ONNX *and* any companion external-data sidecars (*.onnx.data)
        for f in BUNDLED_OV_MODELS.glob("*.onnx*"):
            dst = OV_MODELS_DIR / f.name
            if not dst.exists():
                shutil.copy2(f, dst)

    @app.get("/api/devices")
    def api_devices():
        return {"engines": available_engines(), "ov_devices": ov_devices()}

    @app.get("/api/new_name")
    def api_new_name():
        return {"name": default_run_id()}

    @app.get("/api/check_name")
    def api_check_name(name: str = ""):
        safe = sanitize_run_name(name)
        if safe is None:
            return {"valid": False, "exists": False, "safe_name": None}
        return {"valid": True, "safe_name": safe, "exists": (RUNS_DIR / safe).exists()}

    @app.get("/api/runs")
    def api_runs():
        """List previous run folders that hold at least one engine result."""
        out = []
        if RUNS_DIR.exists():
            for d in RUNS_DIR.iterdir():
                mdir = d / "metrics"
                if not d.is_dir() or not mdir.is_dir():
                    continue
                engine_ids = sorted(p.stem for p in mdir.glob("*.json"))
                if not engine_ids:
                    continue
                out.append({"run_id": d.name, "engines": engine_ids,
                            "mtime": d.stat().st_mtime})
        out.sort(key=lambda r: r["mtime"], reverse=True)
        return {"runs": out}

    @app.get("/api/result/{run_id}")
    def api_result(run_id: str):
        """Rebuild the /api/run response from a saved run folder (no recompute)."""
        safe = sanitize_run_name(run_id)
        if safe is None:
            raise HTTPException(status_code=400, detail="名稱不合法")
        run_dir = RUNS_DIR / safe
        mdir = run_dir / "metrics"
        if not mdir.is_dir():
            raise HTTPException(status_code=404, detail="找不到此結果資料夾")

        manifest = {}
        mf = run_dir / "run.json"
        if mf.exists():
            try:
                manifest = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        order = manifest.get("engines") or []

        files = sorted(mdir.glob("*.json"),
                       key=lambda p: (order.index(p.stem) if p.stem in order else 999, p.stem))
        engines = []
        for jf in files:
            eid = jf.stem
            try:
                m = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            m["id"] = eid
            eng = ENGINE_BY_ID.get(eid)
            m["label"] = eng["label"] if eng else eid
            if (run_dir / "logs" / f"{eid}.log").exists():
                m["log_url"] = f"/runs/{safe}/logs/{eid}.log"
            engines.append(m)
        if not engines:
            raise HTTPException(status_code=404, detail="此資料夾沒有可讀取的結果")
        return {
            "run_id": safe,
            "counts": manifest.get("counts"),
            "params": manifest.get("params"),
            "vis_base": f"/runs/{safe}/vis",
            "engines": engines,
        }

    @app.post("/api/run")
    async def api_run(
        file: UploadFile = File(...),
        run_name: str = Form(""),
        engines: str = Form("opencv,pytorch,ov_cpu"),
        lg_features: int = Form(1024),
        max_features: int = Form(4000),
        max_dim: int = Form(1280),
        min_matches: int = Form(8),
        ratio: float = Form(0.75),
        ransac_thresh: float = Form(5.0),
        ov_size: int = Form(1024),
    ):
        raw = await file.read()
        try:
            session_json = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}")

        # output folder name: user-provided (sanitized) or auto default
        if run_name.strip():
            run_id = sanitize_run_name(run_name)
            if run_id is None:
                raise HTTPException(status_code=400, detail="資料夾名稱不合法")
            if (RUNS_DIR / run_id).exists():
                raise HTTPException(status_code=409, detail=f"資料夾已存在: {run_id}")
        else:
            run_id = default_run_id()
        run_dir = RUNS_DIR / run_id
        data_dir = run_dir / "data"
        counts = extract_session(session_json, data_dir)
        if counts["points"] == 0:
            raise HTTPException(status_code=400, detail="no correspondence points in JSON")

        params = {
            "lg_features": lg_features, "max_features": max_features, "max_dim": max_dim,
            "min_matches": min_matches, "ratio": ratio, "ransac_thresh": ransac_thresh,
            "ov_size": ov_size,
        }

        want = [ENGINE_BY_ID[e.strip()] for e in engines.split(",")
                if e.strip() in ENGINE_BY_ID]
        avail_ids = {e["id"] for e in available_engines()}
        want = [e for e in want if e["id"] in avail_ids]
        if not want:
            raise HTTPException(status_code=400, detail="no runnable engine selected")

        results = [run_engine(data_dir, e, run_dir, params) for e in want]
        # manifest so this run can be reopened later without recomputing
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": run_id,
            "counts": counts,
            "params": params,
            "engines": [e["id"] for e in want],
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
        return JSONResponse({
            "run_id": run_id,
            "counts": counts,
            "params": params,
            "vis_base": f"/runs/{run_id}/vis",
            "engines": results,
        })

    # static overlays + frontend
    app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


# --------------------------------------------------------------------------- #
# Entry point (also the PyInstaller entry)
# --------------------------------------------------------------------------- #
def main() -> None:
    # worker mode: run one engine, then exit (used by run_engine subprocesses)
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        import vis_core
        vis_core.main()
        return

    import webbrowser
    import threading
    import uvicorn

    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))
    url = f"http://{host}:{port}/"
    print(f"Marker Matching Benchmark -> {url}")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(build_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
