#!/usr/bin/env python3
"""Phase 0 mock 송신기 실행 — iPhone 없이 가상 카메라 프레임을 수신기로 보낸다.

먼저 다른 터미널에서 수신기를 띄운 뒤 이 스크립트를 실행한다::

    # 터미널 1
    .venv/bin/python scripts/run_receiver.py
    # 터미널 2
    .venv/bin/python scripts/run_mock_sender.py

rerun 뷰어에서 카메라가 원을 그리며 도는 것이 보이면 Phase 0 배관이 통한 것이다.
"""

from __future__ import annotations

import argparse
import asyncio

from pysical.io.mock_sender import run_mock_sender
from pysical.utils.config import load_config


def parse_args() -> argparse.Namespace:
    cfg = load_config("phase0")
    p = argparse.ArgumentParser(description="Phase 0 mock 프레임 송신기")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=cfg["server"]["port"])
    p.add_argument("--fps", type=float, default=cfg["mock"]["fps"])
    p.add_argument("--duration", type=float, default=cfg["mock"]["duration_s"])
    p.add_argument("--radius", type=float, default=cfg["mock"]["radius_m"])
    p.add_argument("--height", type=float, default=cfg["mock"]["height_m"])
    p.add_argument("--no-rgb", action="store_true", help="더미 RGB를 보내지 않는다.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    uri = f"ws://{args.host}:{args.port}"
    asyncio.run(
        run_mock_sender(
            uri,
            fps=args.fps,
            duration_s=args.duration,
            radius_m=args.radius,
            height_m=args.height,
            send_rgb=not args.no_rgb,
        )
    )


if __name__ == "__main__":
    main()
