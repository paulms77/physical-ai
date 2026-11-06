#!/usr/bin/env python3
"""실시간 성능 벤치마크 — 각 단계 지연을 측정하고 실시간 가능 여부를 판정한다.

부하 낮은 환경(전용 노트북, 또는 A6000 한가할 때)에서 돌려야 의미 있는 숫자가 나온다.
서버 부하가 높으면 결과가 왜곡되므로, 실행 전 load average를 함께 출력한다.

측정 항목: 검출(YOLOE) · depth(Depth Anything) · SR(ESPCN) · 트래커 · 좌표변환.
그리고 detect-and-track 구조에서 유효 처리량(fps)을 추정한다.

사용법::

    .venv/bin/python scripts/benchmark.py --image data/desk.jpeg --runs 5
    .venv/bin/python scripts/benchmark.py --image data/desk.jpeg --threads 4
"""

from __future__ import annotations

import argparse
import os
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="실시간 성능 벤치마크")
    p.add_argument("--image", default="data/desk.jpeg")
    p.add_argument("--runs", type=int, default=5, help="각 항목 반복 측정 횟수")
    p.add_argument("--threads", type=int, default=0, help="스레드 제한(0=제한 안 함)")
    p.add_argument("--detect-every", type=int, default=5, help="detect-and-track 주기")
    return p.parse_args()


def _load_avg() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return -1.0


def _bench(fn, runs: int, warmup: int = 1) -> float:
    """fn을 warmup 후 runs회 실행해 평균 ms를 돌려준다."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.monotonic()
        fn()
        ts.append(time.monotonic() - t0)
    return 1000 * sum(ts) / len(ts)


def main() -> None:
    args = parse_args()

    if args.threads > 0:
        from pysical.utils.runtime import limit_threads
        limit_threads(args.threads)

    import cv2
    import numpy as np

    from pysical.io.types import CameraIntrinsics, Frame

    n_cpu = os.cpu_count() or 1
    load = _load_avg()
    print("=" * 56)
    print(f"환경: {n_cpu}코어, load average={load:.1f}", end="")
    if load > n_cpu * 0.7:
        print("  ⚠️ 부하 높음 — 결과 왜곡 가능(부하 낮을 때 재측정 권장)")
    else:
        print("  ✅ 부하 양호")
    print(f"스레드 제한: {args.threads or '없음'}")
    print("=" * 56)

    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    K = CameraIntrinsics(w, w, w / 2, h / 2)
    frame = Frame(timestamp=0.0, T_world_camera=np.eye(4), rgb=rgb, intrinsics=K)

    results: dict[str, float] = {}

    # 1) 검출 (YOLOE)
    from pysical.perception.yoloe_detector import YoloeDetector
    det = YoloeDetector(conf=0.4, device="cpu")
    det.set_prompt(["cup"])
    results["검출(YOLOE)"] = _bench(lambda: det.detect(frame), args.runs)

    # 2) depth (Depth Anything ONNX)
    try:
        from pysical.perception.depth_estimator import MonocularDepthEstimator
        depth_est = MonocularDepthEstimator()
        results["depth(Depth Anything)"] = _bench(lambda: depth_est.estimate(rgb), args.runs)
    except Exception as e:
        print(f"depth 스킵: {e}")

    # 3) SR (ESPCN) — 작은 크롭
    from pysical.perception.upscale import get_upscaler
    up = get_upscaler(3)
    small = rgb[:80, :60]
    results["SR(ESPCN x3)"] = _bench(lambda: up.upscale(small), args.runs)

    # 4) 트래커 update
    from pysical.tracking.tracker import BoxTracker
    tr = BoxTracker()
    tr.reinit(rgb, np.array([w * 0.4, h * 0.4, w * 0.5, h * 0.5]))
    results["트래커 update"] = _bench(lambda: tr.update(rgb), args.runs)

    # 결과 출력
    print(f"\n{'항목':28} {'평균 지연':>12}")
    print("-" * 44)
    for k, v in results.items():
        print(f"{k:28} {v:>9.1f} ms")

    # detect-and-track 유효 처리량 추정
    detect_ms = results.get("검출(YOLOE)", 0)
    track_ms = results.get("트래커 update", 0)
    every = args.detect_every
    # every 프레임마다: 1번 검출 + (every-1)번 트래커
    avg_frame_ms = (detect_ms + (every - 1) * track_ms) / every
    est_fps = 1000 / avg_frame_ms if avg_frame_ms > 0 else 0

    print("\n" + "=" * 56)
    print(f"detect-and-track (검출 1/{every} 프레임) 추정:")
    print(f"  평균 프레임 처리: {avg_frame_ms:.1f} ms")
    print(f"  추정 처리량: {est_fps:.1f} fps", end="")
    print("  ✅ 실시간(≥15fps)" if est_fps >= 15 else
          ("  △ 준실시간" if est_fps >= 8 else "  ❌ 최적화 필요"))
    print("=" * 56)


if __name__ == "__main__":
    main()
