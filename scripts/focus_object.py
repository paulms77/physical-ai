#!/usr/bin/env python3
"""Auto-Focus 데모 — 지정 대상을 검출·정밀화하고 중앙 확대·계측해 보여준다.

이 프로젝트의 차별점(의료 "병변 확대 정밀 뷰"의 기능적 버전)을 한 장으로 보여준다::

    "컵 보여줘" → 검출 → (저신뢰도면) Focus-and-Refine 재검출
               → 대상만 추출·중앙 확대·정렬 → 3D 거리/크기 계측 오버레이

출력: <image>_focus.jpg (지정 대상의 확대 정밀 뷰).

사용법::

    .venv/bin/python scripts/focus_object.py --image data/desk.jpeg --target 컵
    # depth까지 주면 거리·크기 계측
    .venv/bin/python scripts/focus_object.py --image data/desk.jpeg --target 컵 \
        --mono-depth --fx 1000 --fy 1000 --cx 583 --cy 437
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from pysical.io.types import CameraIntrinsics, Frame
from pysical.perception.detector import normalize_prompt
from pysical.perception.localize import localize_object
from pysical.perception.refine import refine_all
from pysical.utils.logging import get_logger
from pysical.viz.focus import estimate_object_size, make_focus_view

log = get_logger("focus")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-Focus: 지정 대상 확대 정밀 뷰")
    p.add_argument("--image", required=True)
    p.add_argument("--target", required=True, help='포커스할 대상(예: 컵)')
    p.add_argument("--conf", type=float, default=0.3, help="1차 검출 임계값(낮춰서 후보 확보)")
    p.add_argument("--no-refine", action="store_true", help="Focus-and-Refine 끄기")
    p.add_argument("--mono-depth", action="store_true")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"이미지를 못 읽었습니다: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    intrinsics = None
    depth_mono = None
    if args.mono_depth:
        if None in (args.fx, args.fy, args.cx, args.cy):
            raise SystemExit("--mono-depth엔 --fx --fy --cx --cy 필요")
        intrinsics = CameraIntrinsics(args.fx, args.fy, args.cx, args.cy)
        from pysical.perception.depth_estimator import MonocularDepthEstimator

        log.info("mono depth 추정 중...")
        depth_mono = MonocularDepthEstimator().estimate(rgb)

    frame = Frame(timestamp=0.0, T_world_camera=np.eye(4), rgb=rgb,
                  intrinsics=intrinsics, depth_mono=depth_mono)

    from pysical.perception.yoloe_detector import YoloeDetector

    detector = YoloeDetector(conf=args.conf, device="cpu")
    labels = normalize_prompt(args.target)
    detector.set_prompt(labels)

    detections = detector.detect(frame)
    if not detections:
        log.info("✗ '%s' → 화면에 없음. 포커스할 대상이 없습니다.", args.target)
        return

    # Focus-and-Refine: 저신뢰도 검출을 확대 재검출로 정밀화.
    if not args.no_refine:
        detections = refine_all(rgb, detections, detector, conf_below=0.6)

    # 가장 신뢰도 높은 대상을 포커스(여러 개면 1등).
    target = max(detections, key=lambda d: d.score)
    log.info("포커스 대상: %s (conf=%.2f)", target.label, target.score)

    # 3D 위치·크기 계측(depth 있을 때).
    location = None
    size_m = None
    if depth_mono is not None and intrinsics is not None:
        locs = localize_object(target, frame)
        if locs:
            location = next(iter(locs.values()))
        size_m = estimate_object_size(target, depth_mono, intrinsics)

    view = make_focus_view(rgb, target, out_size=512, location=location, size_m=size_m)
    out_path = args.image.rsplit(".", 1)[0] + "_focus.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
    log.info("Auto-Focus 뷰 저장: %s", out_path)


if __name__ == "__main__":
    main()
