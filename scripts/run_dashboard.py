"""
Run Streamlit Dashboard
=======================

Script to start the Streamlit dashboard.

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8501
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Streamlit dashboard")

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run on"
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open browser automatically"
    )

    return parser.parse_args()


def main():
    """Run the dashboard."""
    args = parse_args()

    dashboard_path = project_root / "src" / "dashboard" / "app.py"

    print(f"""
    ╔═══════════════════════════════════════════════════╗
    ║       Churn Prediction Dashboard                  ║
    ╠═══════════════════════════════════════════════════╣
    ║  URL: http://localhost:{args.port}                      ║
    ╠═══════════════════════════════════════════════════╣
    ║  NOTE: Make sure the API server is running!       ║
    ║        Run: python scripts/run_api.py             ║
    ╚═══════════════════════════════════════════════════╝
    """)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
        "--server.headless", str(not args.browser).lower(),
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
