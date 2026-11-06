#!/usr/bin/env python3
"""Phase 0 수신기 실행 — iPhone(또는 mock)의 프레임을 받아 rerun에 시각화한다.

이 스크립트가 Phase 0의 '노트북 쪽 본체'다. 흐름은::

    WebSocket 수신 → 프레임 디코드 → ARKit→ROS 변환 → rerun 3D 표시

실행::

    .venv/bin/python scripts/run_receiver.py
    .venv/bin/python scripts/run_receiver.py --no-viewer --save out.rrd   # 헤드리스

그런 다음 다른 터미널에서 mock 송신기를 띄우면 카메라가 움직이는 걸 볼 수 있다::

    .venv/bin/python scripts/run_mock_sender.py
"""

from __future__ import annotations

import argparse
import asyncio

import rerun as rr

from pysical.io.receiver import FrameReceiver
from pysical.io.types import Frame
from pysical.utils.config import load_config
from pysical.utils.logging import get_logger
from pysical.viz.rerun_viz import RerunVisualizer

log = get_logger("run_receiver")


def parse_args() -> argparse.Namespace:
    cfg = load_config("phase0")
    p = argparse.ArgumentParser(description="Phase 0 프레임 수신기 + rerun 시각화")
    p.add_argument("--host", default=cfg["server"]["host"])
    p.add_argument("--port", type=int, default=cfg["server"]["port"])
    p.add_argument(
        "--no-viewer",
        action="store_true",
        help="rerun 뷰어 창을 띄우지 않는다(헤드리스 서버용).",
    )
    p.add_argument(
        "--save",
        default=None,
        help="지정하면 시각화를 이 .rrd 파일로 저장한다(나중에 rerun으로 열람).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("phase0")["viz"]

    visualizer = RerunVisualizer(
        app_id=cfg["app_id"],
        spawn_viewer=not args.no_viewer,
        trail_length=cfg["camera_trail_length"],
        axis_length_m=cfg["axis_length_m"],
    )
    if args.save:
        rr.save(args.save)
        log.info("시각화를 %s 에 저장합니다.", args.save)

    def on_frame(frame: Frame) -> None:
        visualizer.log_frame(frame)

    receiver = FrameReceiver(on_frame=on_frame)
    try:
        asyncio.run(receiver.serve(args.host, args.port))
    except KeyboardInterrupt:
        log.info("종료 (Ctrl-C). 총 %d 프레임 처리.", receiver.frame_count)


if __name__ == "__main__":
    main()
