#!/usr/bin/env python3
"""실시간 인식 루프 — 녹화(.pyrec) 또는 수신 스트림을 실시간 속도로 처리한다.

docs/REALTIME.md의 detect-and-track 엔진을 구동한다. 검출은 2~5Hz, 그 사이는
트래커 보간, 3D는 object map이 유지(손 가림에 강건). FPS와 검출/트래킹 프레임 수를
계측해 "실시간이 되는가"를 숫자로 보여준다.

사용법::

    # 녹화 재생을 실시간 처리 (iOS 앱 없이 검증)
    .venv/bin/python scripts/run_realtime.py --rec data/recordings/synth.pyrec --prompt "컵"

    # rerun 시각화까지
    .venv/bin/python scripts/run_realtime.py --rec desk.pyrec --prompt "컵" --viz

계측만 하려면 --viz 없이. FPS·검출횟수·인스턴스 수가 출력된다.
"""

from __future__ import annotations

import argparse
import time

# 무거운 import 전에 스레드 제한(경합 완화, docs/REALTIME.md §8).
from pysical.utils.runtime import limit_threads

limit_threads(4)

from pysical.io.recording import read_recording
from pysical.perception.detector import normalize_prompt
from pysical.perception.realtime import RealtimeConfig, RealtimeEngine
from pysical.utils.logging import get_logger

log = get_logger("realtime")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="실시간 인식 루프 (detect-and-track)")
    p.add_argument("--rec", required=True, help=".pyrec 녹화 파일")
    p.add_argument("--prompt", required=True, help='추적 대상(예: "컵")')
    p.add_argument("--detect-every", type=int, default=5, help="N프레임마다 검출(기본 5)")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--viz", action="store_true", help="rerun 3D 시각화")
    p.add_argument("--focus", action="store_true", help="최고 신뢰도 대상의 Auto-Focus 확대 뷰도 표시")
    p.add_argument("--no-detector", action="store_true", help="검출 없이 배관만 계측")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    detector = None
    if not args.no_detector:
        from pysical.perception.yoloe_detector import YoloeDetector

        detector = YoloeDetector(conf=args.conf, device="cpu")
        detector.set_prompt(normalize_prompt(args.prompt))

    engine = RealtimeEngine(detector, RealtimeConfig(detect_every=args.detect_every))

    viz = None
    if args.viz:
        from pysical.viz.realtime_viz import RealtimeVisualizer

        viz = RealtimeVisualizer(app_id="pysical_realtime", spawn_viewer=True)

    # 계측
    n_frames = 0
    n_detect_frames = 0
    t_start = time.monotonic()
    process_times: list[float] = []

    for frame in read_recording(args.rec):
        t0 = time.monotonic()
        boxes, tracked = engine.process(frame)
        dt = time.monotonic() - t0
        process_times.append(dt)

        n_frames += 1
        if engine._frame_idx % args.detect_every == 0 and detector is not None:
            n_detect_frames += 1

        if viz is not None:
            # Auto-Focus 뷰(선택): 가장 신뢰도 높은 대상을 확대 정밀 뷰로.
            focus_view = None
            if args.focus and boxes and frame.rgb is not None:
                from pysical.viz.focus import make_focus_view

                target = max(boxes, key=lambda b: b.score)
                focus_view = make_focus_view(frame.rgb, target, out_size=384)
            viz.log(frame, boxes, tracked, focus_view=focus_view)

    total = time.monotonic() - t_start
    avg_ms = 1000 * sum(process_times) / max(1, len(process_times))
    fps = n_frames / total if total > 0 else 0

    log.info("=" * 50)
    log.info("실시간 처리 결과")
    log.info("  총 프레임: %d", n_frames)
    log.info("  검출 프레임: %d (나머지 %d는 트래커 보간)", n_detect_frames, n_frames - n_detect_frames)
    log.info("  평균 프레임 처리: %.1f ms", avg_ms)
    log.info("  처리 처리량: %.1f fps", fps)
    log.info("  최종 인스턴스: %d개", len(engine.object_map.objects))
    log.info("=" * 50)


if __name__ == "__main__":
    main()
