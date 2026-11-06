"""수신기 프롬프트(텍스트 메시지) 처리 테스트.

iOS 앱은 센서를 바이너리로, 음성 프롬프트를 텍스트로 보낸다. 수신기가 텍스트를
on_prompt 콜백으로 올바로 라우팅하는지, 바이너리는 on_frame으로 가는지 확인한다.

실제 WebSocket 없이, _handle_connection에 가짜 메시지 스트림을 넣어 검증한다.

실행:  .venv/bin/python tests/test_receiver_prompt.py
"""

from __future__ import annotations

import asyncio

import numpy as np

from pysical.geometry.mock_camera import orbit_pose_arkit
from pysical.io.protocol import encode_frame
from pysical.io.receiver import FrameReceiver
from pysical.io.types import Frame


class _FakeWebSocket:
    """async for로 미리 정한 메시지들을 내보내는 가짜 소켓."""

    def __init__(self, messages):
        self._messages = messages
        self.remote_address = ("127.0.0.1", 9999)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for m in self._messages:
            yield m


def _run(messages):
    frames_got = []
    prompts_got = []

    receiver = FrameReceiver(
        on_frame=lambda f: frames_got.append(f),
        on_prompt=lambda p: prompts_got.append(p),
    )
    asyncio.run(receiver._handle_connection(_FakeWebSocket(messages)))
    return frames_got, prompts_got


def test_text_goes_to_prompt():
    """텍스트 메시지는 on_prompt로 간다."""
    frames, prompts = _run(["사과 찾아줘"])
    assert prompts == ["사과 찾아줘"]
    assert frames == []


def test_binary_goes_to_frame():
    """바이너리 메시지는 decode되어 on_frame으로 간다."""
    fr = Frame(timestamp=1.0, T_world_camera=orbit_pose_arkit(1.0))
    frames, prompts = _run([encode_frame(fr)])
    assert len(frames) == 1
    assert np.isclose(frames[0].timestamp, 1.0)
    assert prompts == []


def test_mixed_stream_routes_correctly():
    """섞인 스트림에서 각 타입이 올바른 콜백으로 분리된다."""
    fr = Frame(timestamp=2.0, T_world_camera=orbit_pose_arkit(2.0))
    frames, prompts = _run(["컵", encode_frame(fr), "마우스"])
    assert prompts == ["컵", "마우스"]
    assert len(frames) == 1


def test_no_prompt_handler_ignores_text():
    """on_prompt 없이 텍스트가 와도 크래시하지 않는다(무시)."""
    receiver = FrameReceiver(on_frame=lambda f: None)  # on_prompt 없음
    asyncio.run(receiver._handle_connection(_FakeWebSocket(["hello"])))
    # 예외 없이 통과하면 성공


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 통과")
    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_all() else 1)
