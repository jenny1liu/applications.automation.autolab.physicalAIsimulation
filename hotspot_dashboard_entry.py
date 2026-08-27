"""PyInstaller entry point for the Hotspot Detection Benchmark Dashboard.

Wraps hotspot_detector.ui.main() so the packaged .exe has a single top-level
script to build from (build_hotspot_dashboard.bat / HotspotDashboard.spec).
"""

from __future__ import annotations

import sys
import traceback


def main() -> None:
    try:
        from hotspot_detector.ui import main as run_dashboard
        run_dashboard()
    except Exception:
        # Print full traceback so it is visible in the packaged console window
        # instead of the app just closing silently.
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
