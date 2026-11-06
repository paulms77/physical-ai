#!/usr/bin/env python3
"""실기기 라이브 — iOS 앱의 실시간 스트림을 받아 인식·추적·시각화한다.

Phase 0의 수신기(FrameReceiver)에 실시간 엔진(RealtimeEngine)과 시각화를 붙인
'실기기 라이브' 모드다. iPhone(눈)이 WebSocket으로 프레임+음성을 보내면, 노트북
(두뇌+화면)이 검출·추적·3D·Auto-Focus를 수행해 rerun에 띄운다.

구조 (지연 누적 방지)
--------------------
수신 콜백은 최신 프레임만 공유 슬롯에 넣고 즉시 반환한다(네트워크를 막지 않음).
별도 처리 루프가 최신 프레임만 꺼내 무겁게 처리한다(밀린 프레임은 드롭 = 항상 최신).
음성 프롬프트(텍스트)가 오면 엔진의 대상 어휘를 즉시 교체한다.

사용법::

    # 노트북에서 실행 → iPhone 앱이 접속(Bonjour 자동 or 수동 IP)
    .venv/bin/python scripts/run_live.py --prompt "컵" --viz

    # iOS 앱 없이 검증: 다른 터미널에서 mock 송신기
    .venv/bin/python scripts/run_live.py --prompt "mouse"     # 터미널1
    .venv/bin/python scripts/run_mock_sender.py               # 터미널2
"""

from __future__ import annotations

import argparse
import asyncio
import threading

from pysical.utils.runtime import limit_threads, pick_device

limit_threads(4)

from pysical.io.receiver import FrameReceiver
from pysical.io.types import Frame
from pysical.perception.llm_intent import OllamaIntentParser, resolve_intent
from pysical.perception.detector import normalize_prompt
from pysical.perception.realtime import RealtimeConfig, RealtimeEngine
from pysical.utils.config import load_config
from pysical.utils.logging import get_logger

log = get_logger("live")


def parse_args() -> argparse.Namespace:
    cfg = load_config("phase0")
    p = argparse.ArgumentParser(description="실기기 라이브 인식")
    p.add_argument("--prompt", default="cup", help='초기 추적 대상(음성으로 교체 가능)')
    p.add_argument("--voice-only", action="store_true",
                   help="초기 대상 없이 시작 — 음성으로 지정할 때까지 검출하지 않는다")
    p.add_argument("--host", default=cfg["server"]["host"])
    p.add_argument("--port", type=int, default=cfg["server"]["port"])
    p.add_argument("--detect-every", type=int, default=5)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--viz", action="store_true", help="rerun GUI 뷰어(화면 있는 노트북용)")
    p.add_argument("--save", default=None, metavar="OUT.rrd",
                   help="헤드리스 서버용: 시각화를 .rrd 파일로 저장(로컬에서 `rerun OUT.rrd`로 열람)")
    p.add_argument("--serve-web", action="store_true",
                   help="헤드리스 서버용: 웹 뷰어 실행(브라우저로 접속)")
    p.add_argument("--focus", action="store_true",
                   help='Auto-Focus 확대 패널 초기 ON (음성 "포커스"/"포커스 꺼"로 전환)')
    p.add_argument("--no-detector", action="store_true")
    p.add_argument("--llm", default="auto", choices=["auto", "ollama", "none"],
                   help="대화형 발화 해석: auto(기본, Ollama 있으면 사용) | ollama | none")
    p.add_argument("--llm-model", default="qwen2.5:3b", help="Ollama 모델 태그")
    p.add_argument("--device", default="auto",
                   help="검출 장치: auto(기본, Apple Silicon이면 mps) | cpu | mps | 0(CUDA)")
    return p.parse_args()


async def main_async(args) -> None:
    # 엔진 준비
    detector = None
    if not args.no_detector:
        from pysical.perception.yoloe_detector import YoloeDetector

        device = pick_device(args.device)
        log.info("검출 장치: %s", device)
        detector = YoloeDetector(conf=args.conf, device=device)
    engine = RealtimeEngine(detector, RealtimeConfig(detect_every=args.detect_every))

    # 대화형 발화 해석기(선택). 없으면 규칙 파서만 쓴다.
    llm = None
    if args.llm != "none":
        candidate = OllamaIntentParser(model=args.llm_model)
        if candidate.available():
            llm = candidate
            log.info("의도 해석: Ollama %s (짧은 명령은 규칙이 0ms에 처리)", args.llm_model)
            # 첫 호출이 느린 것을 피해 미리 올린다. 수신을 막지 않도록 스레드로.
            threading.Thread(target=candidate.warmup, daemon=True).start()
        elif args.llm == "ollama":
            log.warning("Ollama를 쓸 수 없어 규칙 파서로 진행한다")
    if llm is None:
        log.info("의도 해석: 규칙 파서만")

    # 초기 대상. --voice-only면 비워두고 음성 지정을 기다린다.
    initial = [] if args.voice_only else normalize_prompt(args.prompt)
    if initial:
        engine.set_prompt(initial)
        log.info("초기 대상: %s", initial)
    else:
        log.info('초기 대상 없음 — 음성으로 지정하세요 (예: "컵 찾아줘")')

    viz = None
    if args.viz or args.save or args.serve_web:
        from pysical.viz.realtime_viz import RealtimeVisualizer

        viz = RealtimeVisualizer(
            app_id="pysical_live",
            spawn_viewer=args.viz,
            save_path=args.save,
            serve_web=args.serve_web,
        )

    # 최신 프레임 공유 슬롯(지연 누적 방지: 밀린 프레임은 덮어써 드롭).
    latest: dict[str, Frame | None] = {"frame": None}

    def on_frame(frame: Frame) -> None:
        latest["frame"] = frame  # 즉시 반환(네트워크 안 막음)

    # 포커스는 이제 런타임 상태다(--focus는 초기값, 음성으로 켜고 끈다).
    from pysical.viz.focus_select import FocusSelector

    focus_state = {"on": args.focus}
    selector = FocusSelector()

    def apply_prompt(text: str) -> None:
        """프롬프트를 실제로 반영한다. **워커 스레드에서 실행된다.**"""
        intent = resolve_intent(text, llm)
        if intent.is_noop():
            log.info("음성 프롬프트 해석 실패(무시): %r", text)
            return

        if intent.targets:
            already = set(engine.labels)
            if intent.focus and already and set(intent.targets) <= already:
                # "포커스 펜" — 이미 찾고 있는 것 중 하나를 포커스로 지정한다.
                # 검출 어휘는 그대로 두어 나머지 물체도 계속 잡는다.
                selector.request_label(intent.targets[0])
                log.info("음성 → 포커스 대상 지정: %s", intent.targets[0])
            else:
                log.info("음성 → 대상 교체: %r → %s", text, intent.targets)
                engine.set_prompt(intent.targets)
                selector.reset()  # 대상이 바뀌면 포커스 고정도 푼다
        if intent.focus is not None:
            focus_state["on"] = intent.focus
            log.info("음성 → 포커스 %s", "켜기" if intent.focus else "끄기")
            if not intent.focus:
                selector.reset()
        if intent.select is not None:
            selector.request(intent.select)
            log.info("음성 → 포커스 대상 전환: %s", intent.select)

    async def _apply_prompt_async(text: str) -> None:
        try:
            await asyncio.to_thread(apply_prompt, text)
        except Exception:
            # 프롬프트 처리가 실패해도 연결을 끊지 않는다.
            log.exception("프롬프트 처리 실패(무시): %r", text)

    def on_prompt(text: str) -> None:
        """수신 루프에서 즉시 반환한다.

        set_prompt는 YOLOE 텍스트 인코더를 돌려 첫 호출 ~1.5초, 이후 ~250ms가
        걸린다. 이걸 메시지 루프에서 그대로 기다리면 그동안 프레임을 못 읽어
        지연이 쌓이고 연결이 끊긴다. 별도 태스크로 넘긴다.
        """
        asyncio.get_running_loop().create_task(_apply_prompt_async(text))

    receiver = FrameReceiver(on_frame=on_frame, on_prompt=on_prompt)

    async def detect_loop() -> None:
        """검출 전용 루프 — 자기 속도로 돈다(표시 루프를 막지 않는다).

        YOLOE 추론은 MPS에서 ~34ms, CPU에서 ~95ms다. 이걸 표시 루프와 같은
        스레드에서 돌리면 검출 프레임마다 화면이 멎는다. 분리해 두면 표시는
        일정한 주기를 유지하고, 검출은 되는 만큼만 따라온다.
        """
        last_ts = None
        while True:
            frame = latest["frame"]
            if frame is None or frame.timestamp == last_ts or not engine.labels:
                await asyncio.sleep(0.01)
                continue
            last_ts = frame.timestamp
            await asyncio.to_thread(engine.detect_step, frame)

    def render(frame):
        """트래커 보간 + 포커스 + rerun 기록. 워커 스레드에서 실행된다."""
        boxes, tracked = engine.track_step(frame)
        if viz is None:
            return
        focus_view = None
        if focus_state["on"]:
            # 고정된 대상을 이어받는다(신뢰도 역전으로 깜빡이지 않음).
            target = selector.select(boxes, frame)
            if target is not None and frame.rgb is not None:
                from pysical.viz.focus import make_focus_view

                focus_view = make_focus_view(frame.rgb, target, out_size=384)
        viz.log(frame, boxes, tracked, focus_view=focus_view)

    async def display_loop() -> None:
        """표시 루프 — 매 프레임 가볍게 돈다(보간 ~5ms + 기록 ~2ms).

        rerun 기록도 스레드로 넘긴다. 이벤트 루프에서 하면 JPEG 압축이
        WebSocket 수신을 잠깐씩 막는다.
        """
        last_ts = None
        while True:
            frame = latest["frame"]
            if frame is None or frame.timestamp == last_ts:
                await asyncio.sleep(0.005)  # 새 프레임 없으면 잠깐 양보
                continue
            last_ts = frame.timestamp
            await asyncio.to_thread(render, frame)

    # 수신기와 처리 루프를 함께 돌린다. Bonjour로 서비스를 광고해 iOS가 자동 탐색.
    from pysical.io.discovery import ServiceAdvertiser

    log.info("실기기 라이브 시작 — iPhone 앱이 자동 탐색(또는 수동 IP)으로 접속하세요 (포트 %d)", args.port)
    with ServiceAdvertiser(port=args.port):
        await asyncio.gather(
            receiver.serve(args.host, args.port),
            detect_loop(),
            display_loop(),
        )


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log.info("종료 (Ctrl-C).")


if __name__ == "__main__":
    main()
