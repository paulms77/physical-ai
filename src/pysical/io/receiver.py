"""WebSocket 프레임 수신기.

iPhone(또는 mock 송신기)이 보내는 프레임을 받아 :func:`decode_frame` 으로 풀고,
등록된 콜백에 넘긴다. '받아서 푸는' 일과 '무엇을 할지(시각화 등)'를 분리해,
수신기는 네트워크에만 집중하고 처리 로직은 바깥에서 갈아끼울 수 있게 한다.

사용 예::

    receiver = FrameReceiver(on_frame=my_handler)
    asyncio.run(receiver.serve(host="0.0.0.0", port=8765))
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import websockets
from websockets.asyncio.server import ServerConnection

from ..utils.logging import get_logger
from .protocol import decode_frame
from .types import Frame

log = get_logger(__name__)

# 콜백은 동기 함수 또는 코루틴 둘 다 허용한다.
FrameHandler = Callable[[Frame], None | Awaitable[None]]
PromptHandler = Callable[[str], None | Awaitable[None]]


class FrameReceiver:
    """프레임(바이너리)과 프롬프트(텍스트)를 받아 콜백으로 넘기는 WebSocket 서버.

    iOS 앱은 센서 프레임을 **바이너리**로, 음성/텍스트 프롬프트를 **텍스트**로 보낸다
    (WebSocketClient.sendBinary / sendPrompt). 이 둘을 각각의 콜백으로 분리한다.
    """

    def __init__(self, on_frame: FrameHandler, on_prompt: PromptHandler | None = None) -> None:
        self._on_frame = on_frame
        self._on_prompt = on_prompt
        self._frame_count = 0

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """한 클라이언트(iPhone)와의 연결을 처리한다."""
        peer = websocket.remote_address
        log.info("클라이언트 연결됨: %s", peer)
        try:
            async for message in websocket:
                # 텍스트 = 프롬프트("사과 찾아줘"), 바이너리 = 센서 프레임.
                if isinstance(message, str):
                    log.info("프롬프트 수신: %r", message)
                    if self._on_prompt is not None:
                        # 프롬프트 처리가 실패해도 연결 전체를 끊지 않는다.
                        # (프레임 디코드 실패를 건너뛰는 것과 같은 원칙)
                        try:
                            result = self._on_prompt(message)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception:
                            log.exception("프롬프트 처리 실패(건너뜀): %r", message)
                    continue

                try:
                    frame = decode_frame(bytes(message))
                except ValueError as e:
                    # 한 프레임이 깨져도 연결 전체를 끊지 않는다.
                    log.warning("프레임 디코드 실패(건너뜀): %s", e)
                    continue

                self._frame_count += 1
                result = self._on_frame(frame)
                if asyncio.iscoroutine(result):
                    await result
        except websockets.ConnectionClosed:
            log.info("클라이언트 연결 종료: %s (총 %d 프레임)", peer, self._frame_count)

    async def serve(self, host: str, port: int) -> None:
        """서버를 띄우고 무한 대기한다 (Ctrl-C로 종료)."""
        log.info("수신기 시작 — ws://%s:%d 에서 대기 중", host, port)
        async with websockets.serve(self._handle_connection, host, port):
            await asyncio.Future()  # 영원히 대기

    @property
    def frame_count(self) -> int:
        return self._frame_count
