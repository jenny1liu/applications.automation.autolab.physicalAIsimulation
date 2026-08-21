# PyInstaller one-folder spec for the Marker Matching Benchmark web app.
#
# Build (from annotator/webapp/):
#     pip install pyinstaller
#     pyinstaller build_exe.spec --noconfirm
#
# Output: dist/MarkerBenchmark/MarkerBenchmark.exe  (double-click to launch).
#
# The one-folder build re-invokes its own exe with a hidden --worker flag to run
# each matching engine in a clean subprocess (see app.py).
#
# NOTE (offline machines): the DISK/LightGlue weights and the DISK-U-Net ONNX are
# downloaded / exported on first use. To ship a fully offline build, before
# building either (a) run the app once so ../ov_models/disk_unet_1024.onnx and the
# torch hub weights get cached, then uncomment the datas lines below, or (b) copy a
# prepared ov_models/ next to the exe. See README.md.

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

app_dir = Path(SPECPATH)
annotator_dir = app_dir.parent

datas = [
    (str(app_dir / "static"), "static"),
]
binaries = []
hiddenimports = [
    "vis_core",
    "extract_pairs",
    "benchmark_matchers",
    "uvicorn.logging",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
]

# Bundle the heavy native libraries in full.
for pkg in ("openvino", "torch", "torchvision", "kornia", "cv2", "onnx", "onnxscript",
            "numpy", "psutil"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Optional: ship a pre-exported ONNX so no export/torch-onnx is needed at runtime.
_onnx = annotator_dir / "ov_models"
if _onnx.exists():
    datas.append((str(_onnx), "ov_models"))

a = Analysis(
    [str(app_dir / "app.py")],
    pathex=[str(app_dir), str(annotator_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MarkerBenchmark",
    console=True,          # keep the console so users see the local URL / logs
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MarkerBenchmark",
)
