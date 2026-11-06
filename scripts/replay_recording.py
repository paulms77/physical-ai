#!/usr/bin/env python3
"""③단계 내부 검증 — 녹화(.pyrec)를 재생하며 전체 파이프라인을 시계열로 돌린다.

iOS 앱으로 실기기 라이브를 하기 전 마지막 검증이다. 녹화된 동영상+포즈+depth를
프레임 단위로 흘려보내며, 각 프레임에서 검출→위치추정→rerun 시각화를 수행한다.
실기기 라이브와 완전히 같은 처리 경로(decode_frame 이후)를 탄다.

사용법::

    # 검출 없이 궤적/depth만 (Phase 0 수준 재생)
    .venv/bin/python scripts/replay_recording.py --rec data/recordings/desk.pyrec

    # 검출 + 위치추정까지
    .venv/bin/python scripts/replay_recording.py --rec desk.pyrec --prompt "사과 컵"

    # 헤드리스(뷰어 없이 .rrd 저장)
    .venv/bin/python scripts/replay_recording.py --rec desk.pyrec --no-viewer --save out.rrd

녹화 파일이 없으면, mock 송신기 대신 --make-mock로 합성 녹화를 만들어 재생기를 검증할 수 있다.
"""

from __future__ import annotations

import argparse

import numpy as np
import rerun as rr

from pysical.io.recording import read_recording
from pysical.io.types import Frame
from pysical.perception.detector import normalize_prompt
from pysical.perception.localize import localize_object
from pysical.utils.config import load_config
from pysical.utils.logging import get_logger
from pysical.viz.rerun_viz import RerunVisualizer

log = get_logger("replay")


def parse_args() -> argparse.Namespace:
    cfg = load_config("phase0")["viz"]
    p = argparse.ArgumentParser(description="녹화 재생 + 전체 파이프라인")
    p.add_argument("--rec", required=True, help=".pyrec 녹화 파일")
    p.add_argument("--prompt", default=None, help='검출할 물체(예: "사과 컵"). 없으면 검출 생략')
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--save", default=None, help=".rrd로 저장")
    p.add_argument("--conf", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("phase0")["viz"]

    viz = RerunVisualizer(
        app_id="pysical_replay",
        spawn_viewer=not args.no_viewer,
        trail_length=cfg["camera_trail_length"],
        axis_length_m=cfg["axis_length_m"],
    )
    if args.save:
        rr.save(args.save)

    # 검출기는 프롬프트가 있을 때만 로드(무거운 import 회피).
    detector = None
    if args.prompt:
        from pysical.perception.yoloe_detector import YoloeDetector

        detector = YoloeDetector(conf=args.conf, device="cpu")
        detector.set_prompt(normalize_prompt(args.prompt))

    n = 0
    for frame in read_recording(args.rec):
        viz.log_frame(frame)                      # 카메라 궤적·RGB·depth 시각화

        if detector is not None and frame.rgb is not None:
            detections = detector.detect(frame)
            for i, det in enumerate(detections):
                locs = localize_object(det, frame)
                # 대표 좌표(있으면 plane, 없으면 아무거나)를 rerun에 3D 포인트로.
                if locs:
                    key = "plane" if "plane" in locs else next(iter(locs))
                    p = locs[key].position_world
                    rr.set_time("iphone_clock", duration=frame.timestamp)
                    rr.log(f"map/objects/{det.label}_{i}",
                           rr.Points3D([p], radii=0.03, labels=[det.label]))
        n += 1

    log.info("재생 완료: %d 프레임 처리 (%s)", n, args.rec)


if __name__ == "__main__":
    main()
