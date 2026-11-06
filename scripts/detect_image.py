#!/usr/bin/env python3
"""②단계 내부 검증 — 정지 이미지 1장으로 검출(+위치추정)을 확인한다.

iOS 앱으로 실기기 데모하기 **전에** 반드시 거치는 단계다. 실제 이미지에서 YOLOE가
프롬프트 물체를 찾는지, 검출이 위치추정 파이프라인에 제대로 흘러가는지를 사진 한 장으로
확인한다. iPhone도 동영상도 필요 없다.

사용법::

    # 검출만 (depth 없이)
    .venv/bin/python scripts/detect_image.py --image desk.jpg --prompt "컵 마우스"

    # depth npy(미터)와 포즈를 함께 주면 3D 위치까지
    .venv/bin/python scripts/detect_image.py --image desk.jpg --prompt "사과" \
        --depth depth.npy --fx 600 --fy 600 --cx 320 --cy 240

결과는 <image>_result.jpg 로 저장된다(검출 박스/마스크/접지점/3D좌표 오버레이).
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from pysical.io.types import CameraIntrinsics, Frame
from pysical.perception.detector import normalize_prompt
from pysical.perception.localize import localize_object
from pysical.utils.logging import get_logger
from pysical.viz.draw import draw_detections

log = get_logger("detect_image")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="정지 이미지 검출 + 위치추정 검증")
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    p.add_argument("--prompt", required=True, help='찾을 물체(예: "사과" 또는 "컵 마우스")')
    p.add_argument("--conf", type=float, default=0.4,
                   help="신뢰도 임계값(기본 0.4). 낮추면 없는 물체를 억지로 찾는 false positive↑")
    p.add_argument("--device", default="cpu")
    # depth를 주면 3D 위치추정까지 (선택)
    p.add_argument("--depth", default=None, help="LiDAR depth .npy (H,W) 미터. 있으면 3D 좌표 계산")
    p.add_argument("--mono-depth", action="store_true",
                   help="Depth Anything V2로 depth를 추정해 방법 B로 3D 좌표 계산(LiDAR 없이)")
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
    h, w = rgb.shape[:2]
    log.info("이미지 로드: %s (%dx%d)", args.image, w, h)

    # depth/intrinsics가 있으면 Frame에 실어 위치추정까지 간다.
    depth = None
    intrinsics = None
    depth_intrinsics = None
    if args.depth:
        depth = np.load(args.depth).astype(np.float32)
        if None in (args.fx, args.fy, args.cx, args.cy):
            raise SystemExit("--depth를 쓰려면 --fx --fy --cx --cy도 필요합니다.")
        intrinsics = CameraIntrinsics(args.fx, args.fy, args.cx, args.cy)
        depth_intrinsics = intrinsics
        log.info("depth 로드: %s %s", args.depth, depth.shape)

    # mono depth 추정 (선택): LiDAR 없이 방법 B로 3D 좌표를 내고 싶을 때.
    depth_mono = None
    if args.mono_depth:
        if None in (args.fx, args.fy, args.cx, args.cy):
            raise SystemExit("--mono-depth를 쓰려면 --fx --fy --cx --cy도 필요합니다.")
        if intrinsics is None:
            intrinsics = CameraIntrinsics(args.fx, args.fy, args.cx, args.cy)
        from pysical.perception.depth_estimator import MonocularDepthEstimator

        log.info("Depth Anything V2로 mono depth 추정 중...")
        depth_mono = MonocularDepthEstimator().estimate(rgb)
        log.info("mono depth 완료: shape=%s (상대 depth 주의 — metric 보정 필요)", depth_mono.shape)

    frame = Frame(
        timestamp=0.0,
        T_world_camera=np.eye(4),      # 정지 이미지는 카메라를 원점으로 가정
        rgb=rgb,
        intrinsics=intrinsics,
        depth_lidar=depth,
        depth_intrinsics=depth_intrinsics,
        depth_mono=depth_mono,
    )

    # 검출기 로드 (무거운 import라 여기서)
    from pysical.perception.yoloe_detector import YoloeDetector

    detector = YoloeDetector(conf=args.conf, device=args.device)
    labels = normalize_prompt(args.prompt)
    log.info("프롬프트 '%s' → 라벨 %s", args.prompt, labels)
    detector.set_prompt(labels)

    detections = detector.detect(frame)
    log.info("검출 %d개 (conf ≥ %.2f):", len(detections), args.conf)
    for d in detections:
        log.info("  ✓ %s (%.2f) bbox=%s", d.label, d.score, d.bbox_xyxy.astype(int).tolist())

    # "없으면 없다"를 명확히 알린다. 프롬프트에 넣었지만 하나도 안 잡힌 라벨을 표시.
    found_labels = {d.label for d in detections}
    missing = [lbl for lbl in labels if lbl not in found_labels]
    for lbl in missing:
        log.info("  ✗ %s → 화면에 없음 (검출 0개)", lbl)

    # depth(LiDAR 또는 mono)가 있으면 각 검출의 3D 위치를 계산한다.
    locations: dict[int, dict] = {}
    if frame.has_any_depth:
        for i, det in enumerate(detections):
            locs = localize_object(det, frame)
            if locs:
                locations[i] = locs
                for method, loc in locs.items():
                    log.info("  [%d] %s → %s: %s m", i, det.label, method,
                             np.round(loc.position_world, 3).tolist())

    # 결과 오버레이 저장
    out = draw_detections(rgb, detections, locations or None)
    out_path = args.image.rsplit(".", 1)[0] + "_result.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    log.info("결과 저장: %s", out_path)


if __name__ == "__main__":
    main()
