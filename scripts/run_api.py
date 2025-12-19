"""
Run FastAPI Server
==================

Script to start the FastAPI backend server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8000 --reload
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn

from config import get_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run FastAPI server")

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers"
    )

    return parser.parse_args()


def main():
    """Run the API server."""
    args = parse_args()

    print(f"""
    ╔═══════════════════════════════════════════════════╗
    ║       Churn Prediction API Server                 ║
    ╠═══════════════════════════════════════════════════╣
    ║  Host: {args.host:<15}                          ║
    ║  Port: {args.port:<15}                          ║
    ║  Reload: {str(args.reload):<13}                          ║
    ╠═══════════════════════════════════════════════════╣
    ║  API Docs: http://localhost:{args.port}/docs            ║
    ║  ReDoc:    http://localhost:{args.port}/redoc           ║
    ╚═══════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
